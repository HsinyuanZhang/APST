"""Shared training, cache, metric, and provenance helpers for DANDI688."""

from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from collections import Counter
from typing import Iterable
import numpy as np
from sklearn.metrics import r2_score
from apst.legacy_dandi import protocol
from apst.legacy_dandi.carrier import estimate_move_t4, fit_normalizer
from apst.legacy_dandi.data import SessionData, load_cached_session

RECIPE = {
    "segments": 24,
    "updates_per_segment": 3165,
    "batch": 32,
    "optimizer": "AdamW",
    "lr_peak": 3e-4,
    "lr_min_factor": 0.1,
    "warmup_updates": 3165,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "grad_clip": 1.0,
    "ema_decay": 0.9995,
    "whole_unit_dropout": 0.1,
    "context_bins": 50,
    "layers": 4,
    "windows": [13, 12, 12, 12],
    "width": 256,
    "heads": 8,
    "proj_dim": 16,
    "encoder_pretrain": "fresh_source_only_fixed_final_ema",
    "carrier_estimator": "MOVE_T4_OLS",
    "identity_encoder": "B3S_concat",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(p: Path, x: dict):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, sort_keys=True, default=_json) + "\n")
    os.replace(tmp, p)


def _json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value) -> str:
    """Stable JSON/array receipt digest used by the F0 campaign."""
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_records(cache: Path, representation: str, split: str = "train", *, session_ids=None) -> list[SessionData]:
    aliases = {"train": "source", "dev": "development"}
    split = aliases.get(split, split)
    roster = {
        "source": protocol.SOURCE_SESSIONS,
        "development": protocol.DEVELOPMENT_SESSIONS,
        "final": protocol.FINAL_SESSIONS,
    }[split]
    if split == "final":
        raise PermissionError("source/development cache loading cannot open final data")
    ids = tuple(roster if session_ids is None else session_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(roster):
        raise ValueError("cache request is outside its frozen source/development roster")
    out = [
        load_cached_session(Path(cache) / f"{sid}.{representation}.npz")
        for sid in ids
    ]
    if any(x.split != split or x.representation != representation for x in out):
        raise ValueError("cache split mismatch")
    return out


def record_binding(records) -> dict:
    return {
        record.session_id: {
            "representation": record.representation,
            "split": record.split,
            "raw_nwb_sha256": record.metadata["raw_nwb_sha256"],
            "array_sha256": record.metadata["array_sha256"],
        }
        for record in records
    }


def require_full_source(records) -> None:
    if len(records) != len(protocol.SOURCE_SESSIONS) or {record.session_id for record in records} != set(protocol.SOURCE_SESSIONS):
        raise ValueError("formal fitting requires all and only frozen source sessions")
    if any(record.split != "source" for record in records):
        raise ValueError("development/final data cannot enter source fitting")


def fit_source_stats(records: list[SessionData]) -> dict:
    if {record.session_id for record in records} != set(protocol.SOURCE_SESSIONS):
        raise ValueError("fit requires all and only the 18 source sessions")
    total = np.zeros(2, dtype=np.float64)
    square = np.zeros(2, dtype=np.float64)
    count = 0
    profiles = []
    for record in records:
        labels = np.asarray(record.velocity[record.query_indices], dtype=np.float64)
        total += labels.sum(axis=0)
        square += np.square(labels).sum(axis=0)
        count += len(labels)
        profiles.append(estimate_move_t4(record.carrier_counts, record.carrier_angles))
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 0.0))
    std[std < 1e-8] = 1.0
    return {
        # Keep the source receipt at legacy float64 precision. Training explicitly
        # converts these values to float32, while physical-unit denormalization
        # uses the unrounded source statistics.
        "velocity_mean": mean,
        "velocity_std": std,
        "carrier": fit_normalizer(profiles),
    }


class PairedSampler:
    """Exact paired session/endpoint stream used by every neural arm per seed."""

    def __init__(
        self,
        records: list[SessionData],
        seed: int,
        batch: int,
        updates: int = None,
        updates_per_segment: int = None,
    ) -> None:
        self.records = {record.session_id: record for record in records}
        self.names = sorted(self.records)
        self.batch = int(batch)
        self.updates = int(updates if updates is not None else updates_per_segment)
        if not self.names or self.batch < 1 or self.updates < 1:
            raise ValueError("nonempty records and positive batch/updates are required")
        self.session_rng = np.random.default_rng(seed)
        self.endpoint_rng = {
            name: np.random.default_rng(np.random.SeedSequence([seed, index, 68815]))
            for index, name in enumerate(self.names)
        }
        self.orders = {
            name: self.endpoint_rng[name].permutation(self.records[name].query_indices)
            for name in self.names
        }
        self.positions = {name: 0 for name in self.names}
        self.coverage = []

    def segment(self):
        order = np.resize(
            np.roll(np.asarray(self.names), -len(self.coverage)), self.updates
        )
        self.session_rng.shuffle(order)
        counts = Counter()
        for name_value in order:
            name = str(name_value)
            chosen = []
            while len(chosen) < self.batch:
                remaining = self.orders[name][self.positions[name] :]
                take = min(len(remaining), self.batch - len(chosen))
                chosen.extend(remaining[:take].tolist())
                self.positions[name] += take
                if self.positions[name] == len(self.orders[name]):
                    self.orders[name] = self.endpoint_rng[name].permutation(
                        self.records[name].query_indices
                    )
                    self.positions[name] = 0
            counts[name] += 1
            yield self.records[name], np.asarray(chosen, dtype=np.int64)
        self.coverage.append(dict(sorted(counts.items())))

    def receipt(self) -> dict:
        return {
            "batch": self.batch,
            "updates_per_segment": self.updates,
            "segment_session_batch_counts": self.coverage,
        }


def score(record: SessionData, prediction: np.ndarray) -> dict:
    target = np.asarray(record.velocity[record.query_indices], dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("predictions must match finite Q50 targets")
    return {
        "session_id": record.session_id,
        "r2": float(r2_score(target, prediction, multioutput="variance_weighted")),
        "n_queries": len(target),
    }


def score_predictions(record: SessionData, prediction: np.ndarray) -> dict:
    expected = np.asarray(record.velocity[record.query_indices], dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if predicted.shape != expected.shape or not np.isfinite(predicted).all():
        raise ValueError("predictions must match all finite physical Q50 target rows")
    per_output = np.asarray(r2_score(expected, predicted, multioutput="raw_values"))
    return {
        "session_id": record.session_id,
        "n_queries": len(expected),
        "r2": float(r2_score(expected, predicted, multioutput="variance_weighted")),
        "r2_per_output": per_output.tolist(),
        "query_indices_sha256": record.metadata["array_sha256"]["query_indices"],
        "velocity_sha256": record.metadata["array_sha256"]["velocity"],
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate zero sessions")
    return {"mean_r2": float(np.mean([row["r2"] for row in rows])), "sessions": rows}


def aggregate_scores(scores: list[dict]) -> dict:
    if not scores or len({score["session_id"] for score in scores}) != len(scores):
        raise ValueError("metrics require a nonempty unique session roster")
    return {
        "metric": "physical_velocity_sklearn_variance_weighted_equal_session_mean",
        "mean_r2": float(np.mean([score["r2"] for score in scores])),
        "n_sessions": len(scores),
        "sessions": scores,
    }
