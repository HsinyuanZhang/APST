"""Shared receipts, digests, selection rules, and batch iteration."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


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
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def array_sha256(value: Any) -> str:
    """Typed digest: dtype, shape, and raw bytes of a contiguous array."""
    array = np.ascontiguousarray(np.asarray(value))
    result = hashlib.sha256()
    result.update(array.dtype.str.encode())
    result.update(str(array.shape).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fresh_directory(dest: Path) -> Path:
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"output must be fresh: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def earliest_finite_maximum(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    """One-based epoch of the earliest maximum of the finite ``key`` values."""
    finite = [(float(row[key]), -int(row["epoch"])) for row in rows
              if np.isfinite(float(row.get(key, float("nan"))))]
    if not finite:
        raise ValueError("no finite epoch scores for earliest-maximum selection")
    best = max(finite)
    return -best[1]


def centered_variance_weighted_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    """Per-output centered R2 aggregated over outputs, in float64."""
    prediction = np.asarray(prediction, np.float64)
    target = np.asarray(target, np.float64)
    if prediction.shape != target.shape or not len(target):
        raise ValueError("prediction/target must be aligned and nonempty")
    sse = float(np.square(prediction - target).sum())
    tss = float(np.square(target - target.mean(axis=0)).sum())
    return 1.0 - sse / tss if tss > 0.0 else float("nan")


def grouped_seven_r2(per_key: Mapping[str, Mapping[str, np.ndarray]],
                     groups: Mapping[str, Sequence[str]]) -> float:
    """Concatenate grouped recordings, score per group, and average groups."""
    values = []
    for group in sorted(groups):
        members = groups[group]
        prediction = np.concatenate([np.asarray(per_key[key]["prediction"], np.float64)
                                     for key in members])
        target = np.concatenate([np.asarray(per_key[key]["target"], np.float64)
                                 for key in members])
        values.append(centered_variance_weighted_r2(prediction, target))
    if not values or not all(np.isfinite(values)):
        raise ValueError("grouped-seven scoring requires finite per-group R2")
    return float(np.mean(values))


def epoch_batches(items: Mapping[str, Any], epoch: int, batch: int,
                  seed: int) -> list[tuple[str, np.ndarray]]:
    """Session-major local shuffle, then global batch shuffle (one full pass)."""

    def _count(item: Any) -> int:
        starts = getattr(item, "starts", None)
        if starts is None:
            starts = item["starts"]
        return len(starts)

    rng = np.random.default_rng(seed + epoch * 1000003)
    out: list[tuple[str, np.ndarray]] = []
    for session in sorted(items):
        order = rng.permutation(_count(items[session]))
        out.extend((session, order[lo:lo + batch]) for lo in range(0, len(order), batch))
    rng.shuffle(out)
    return out


def source_hashes() -> dict[str, str]:
    """Hash the concrete apst.falcon and apst.models modules used at runtime."""
    import importlib
    modules = (
        "apst.falcon.common", "apst.falcon.protocol", "apst.falcon.data",
        "apst.falcon.carrier", "apst.falcon.profile", "apst.falcon.model",
        "apst.falcon.training", "apst.models.decoder", "apst.models.temporal",
        "apst.models.recency.temporal", "apst.models.recency.config",
        "apst.models.film", "apst.models.ema", "apst.models.frontend",
        "apst.models.schedule",
    )
    paths = set()
    for name in modules:
        try:
            paths.add(Path(importlib.import_module(name).__file__).resolve())
        except Exception:  # pragma: no cover - optional module absent
            continue
    return {str(path): sha256(path) for path in sorted(paths)}


def freeze_rng(seed: int) -> None:
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
