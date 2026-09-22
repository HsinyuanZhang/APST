"""Shared receipts, source statistics, metrics and paired sampling for DANDI v2."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import r2_score

from . import protocol
from .carrier import estimate_move_t4, fit_carrier_normalizer
from .data import SessionData, load_cached_session

PACKAGE = Path(__file__).resolve().parent
SCHEMA = "dandi688_v2_2015_b3s_move_t4_learned"
RECIPE = {
    "segments": 24, "updates_per_segment": 3165, "batch": 32,
    "optimizer": "AdamW", "lr_peak": 3e-4, "lr_min_factor": .1,
    "warmup_updates": 3165, "weight_decay": .01, "betas": [.9, .999],
    "eps": 1e-8, "grad_clip": 1., "ema_decay": .9995,
    "whole_unit_dropout": .1, "loss": "mse_source_standardized_velocity",
    "context_bins": 50, "layers": 4, "windows": [13, 12, 12, 12],
    "width": 256, "heads": 8, "proj_dim": 16,
    "identity_interface": "proj_add", "tier": "learned_slope", "ladder": "default",
    "sampler": "session_balanced_batches_then_shuffled_endpoint_cycles",
    "encoder_pretrain": "same_24x3165_recipe_source_only_fixed_final_ema",
    "dropout_seed_domain": "M2 unit_dropout_seed(seed, segment_1based, batch_0based)",
    "precision": "fp32_cpu_bfloat16_autocast_cuda",
    "carrier_estimator": "M2_MOVE_T4_OLS_cosine_counts_per_native_bin",
    "identity_encoder": "B3S",
    "encoder_side_zero_init": True,
    "encoder_line": "concat",
    "encoder_carrier_fusion": "post_pool_input_concat",
    "encoder_film": False,
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes() -> dict[str, str]:
    """Hash the release package and the concrete APST modules imported at runtime."""
    import importlib
    modules = (
        "apst.dandi_subc.common", "apst.dandi_subc.data", "apst.dandi_subc.carrier",
        "apst.dandi_subc.protocol", "apst.models.decoder", "apst.models.recency.temporal",
        "apst.models.film", "apst.models.ema", "apst.models.frontend", "apst.models.schedule",
    )
    paths = {Path(importlib.import_module(name).__file__).resolve() for name in modules}
    return {str(path): sha256(path) for path in sorted(paths)}


def fresh_directory(dest: Path) -> Path:
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"output must be fresh: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def cache_path(cache: Path, session_id: str, representation: str) -> Path:
    return Path(cache) / f"{session_id}.{representation}.npz"


def load_records(cache: Path, representation: str, split: str = "train", *,
                 session_ids: Iterable[str] | None = None) -> list[SessionData]:
    if representation not in {"sua", "pmua"}:
        raise ValueError("representation must be sua or pmua")
    roster = {"train": protocol.TRAIN_SESSIONS, "dev": protocol.DEV_SESSIONS}.get(split)
    if roster is None:
        raise PermissionError("source/development cache loading cannot open final data")
    ids = tuple(roster if session_ids is None else session_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(roster):
        raise ValueError("cache request is outside its 2015 train/development roster")
    records = [load_cached_session(cache_path(cache, name, representation)) for name in ids]
    for name, record in zip(ids, records):
        if (record.session_id, record.representation, record.split) != (name, representation, split):
            raise ValueError("cache payload identity differs from requested session")
    keys = [r.metadata["canonical_electrode_keys"] for r in records]
    if any(key != keys[0] for key in keys):
        raise ValueError("canonical M1 electrode table differs between sessions")
    return records


def record_binding(records: Iterable[SessionData]) -> dict:
    return {r.session_id: {"representation": r.representation, "split": r.split,
                          "raw_nwb_sha256": r.metadata["raw_nwb_sha256"],
                          "array_sha256": r.metadata["array_sha256"]}
            for r in records}


def require_full_source(records: list[SessionData]) -> None:
    if len(records) != 18 or {r.session_id for r in records} != set(protocol.TRAIN_SESSIONS):
        raise ValueError("formal fitting requires all and only the 18 2015 source sessions")
    if any(r.split != "train" for r in records):
        raise ValueError("development/final data cannot enter source fitting")


def fit_source_stats(records: list[SessionData], *, smoke: bool = False) -> dict:
    if not smoke:
        require_full_source(records)
    if not records or any(r.split != "train" or r.session_id not in protocol.TRAIN_SESSIONS for r in records):
        raise ValueError("statistics accept 2015 source sessions only")
    if len({r.representation for r in records}) != 1:
        raise ValueError("SUA and PMUA carrier statistics must be fit independently")
    count, total, square = 0, np.zeros(2), np.zeros(2)
    raw_carriers = []
    for r in records:
        y = np.asarray(r.velocity[r.query_indices], np.float64)
        if y.ndim != 2 or y.shape[1] != 2 or not len(y) or not np.isfinite(y).all():
            raise ValueError("source labels must be nonempty finite physical 2D velocity")
        count += len(y)
        total += y.sum(axis=0)
        square += np.square(y).sum(axis=0)
        raw_carriers.append(estimate_move_t4(r.carrier_counts, r.carrier_angles))
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 0.))
    std[std < 1e-8] = 1.
    state = {"schema": SCHEMA + "_source_stats", "status": "SMOKE" if smoke else "FORMAL",
             "representation": records[0].representation, "source_sessions": [r.session_id for r in records],
             "carrier_estimator": RECIPE["carrier_estimator"],
             "source_binding": record_binding(records), "velocity_mean": mean, "velocity_std": std,
             "velocity_fit_rows": count, "carrier": fit_carrier_normalizer(raw_carriers)}
    state = jsonable(state)
    state["sha256"] = digest(state)
    return state


def verify_stats(state: dict, records: list[SessionData], *, smoke: bool = False) -> None:
    payload = {key: value for key, value in state.items() if key != "sha256"}
    if state.get("sha256") != digest(payload):
        raise ValueError("source-statistics receipt hash mismatch")
    if state.get("source_binding") != record_binding(records):
        raise ValueError("source data differ from the fitted statistics receipt")
    if not smoke and state.get("status") != "FORMAL":
        raise ValueError("SMOKE statistics cannot be used for formal training")


def score_predictions(record: SessionData, prediction: np.ndarray) -> dict:
    expected = np.asarray(record.velocity[record.query_indices], np.float64)
    predicted = np.asarray(prediction, np.float64)
    if predicted.shape != expected.shape or not np.isfinite(predicted).all():
        raise ValueError("predictions must match all finite physical Q50 target rows")
    per_output = np.asarray(r2_score(expected, predicted, multioutput="raw_values"))
    return {"session_id": record.session_id, "n_queries": len(expected),
            "r2": float(r2_score(expected, predicted, multioutput="variance_weighted")),
            "r2_per_output": per_output.tolist(),
            "query_indices_sha256": record.metadata["array_sha256"]["query_indices"],
            "velocity_sha256": record.metadata["array_sha256"]["velocity"]}


def aggregate_scores(scores: list[dict]) -> dict:
    if not scores or len({s["session_id"] for s in scores}) != len(scores):
        raise ValueError("metrics require a nonempty unique session roster")
    return {"metric": "physical_velocity_sklearn_variance_weighted_equal_session_mean",
            "mean_r2": float(np.mean([s["r2"] for s in scores])),
            "n_sessions": len(scores), "sessions": scores}


class PairedSampler:
    """One shared batch/endpoint stream per seed for all six neural cells.

    Session batches are balanced within every fixed training segment.  Each
    session cycles through independently shuffled full query endpoint arrays.
    The order depends only on IDs/endpoints/seed, never neural representation.
    """

    def __init__(self, records: list[SessionData], seed: int, *, batch: int = 32,
                 updates_per_segment: int = 3165) -> None:
        if not records or batch < 1 or updates_per_segment < 1:
            raise ValueError("nonempty records and positive batch/update counts are required")
        self.records = {r.session_id: r for r in records}
        self.names = sorted(self.records)
        if len(self.records) != len(records):
            raise ValueError("sampler sessions must be unique")
        self.batch, self.updates_per_segment, self.seed = batch, updates_per_segment, int(seed)
        self.session_rng = np.random.default_rng(seed)
        self.endpoint_rng = {name: np.random.default_rng(np.random.SeedSequence([seed, index, 68815]))
                             for index, name in enumerate(self.names)}
        self.orders = {name: self.endpoint_rng[name].permutation(self.records[name].query_indices)
                       for name in self.names}
        if any(not len(order) for order in self.orders.values()):
            raise ValueError("sampler cannot use an empty query surface")
        self.positions = {name: 0 for name in self.names}
        self._digest = hashlib.sha256()
        self.coverage: list[dict] = []

    def segment(self):
        # Rotate which sessions get a surplus batch between segments.
        order = np.resize(np.roll(np.asarray(self.names), -len(self.coverage)), self.updates_per_segment)
        self.session_rng.shuffle(order)
        counts = Counter()
        for name_value in order:
            name = str(name_value)
            chosen: list[int] = []
            while len(chosen) < self.batch:
                remaining = self.orders[name][self.positions[name]:]
                take = min(len(remaining), self.batch - len(chosen))
                chosen.extend(remaining[:take].tolist())
                self.positions[name] += take
                if self.positions[name] == len(self.orders[name]):
                    self.orders[name] = self.endpoint_rng[name].permutation(self.records[name].query_indices)
                    self.positions[name] = 0
            indices = np.asarray(chosen, np.int64)
            self._digest.update(name.encode())
            self._digest.update(indices.tobytes())
            counts[name] += 1
            yield self.records[name], indices
        self.coverage.append(dict(sorted(counts.items())))

    def receipt(self) -> dict:
        return {"seed": self.seed, "batch": self.batch, "updates_per_segment": self.updates_per_segment,
                "sha256": self._digest.hexdigest(), "segment_session_batch_counts": self.coverage}
