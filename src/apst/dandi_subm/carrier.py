"""Literal M2 MOVE--T4 carrier primitives.

The formal M2 route reduces each native 20-ms count window to one row per
channel with ``t4_from_trial_sums``.  It fits ordinary least squares against
``[1, cos(theta), sin(theta)]`` on finite target-angle trials, then emits
``[a, c, hypot(a, c), baseline]``. Counts here are already the 25-bin
``[5, 30)`` trial sums; the fixed window is part of the protocol.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


BIN_SECONDS = 0.02
MOVE_START_BIN = 5
MOVE_STOP_BIN = 30
MOVE_BINS = MOVE_STOP_BIN - MOVE_START_BIN
MOVE_DURATION_SECONDS = MOVE_BINS * BIN_SECONDS
CARRIER_DIM = 4
_STD_FLOOR = 1.0e-6


def estimate_move_t4(
    counts: np.ndarray, angles: np.ndarray, duration_s: float = MOVE_DURATION_SECONDS,
) -> np.ndarray:
    """Fit formal-M2 cosine-tuning rows ``[N, 4]`` from MOVE trial sums.

    This is numerically the same calculation as
    ``streaming_calibration_exp.src.data.falcon_t4_features.t4_from_trial_sums``
    with a fixed ``trial_lengths`` vector of 25 native bins. ``duration_s``
    remains a fixed-window contract guard.
    """
    sums = np.asarray(counts, dtype=np.float64)
    target_angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    if sums.ndim != 2 or sums.shape[0] != target_angles.size:
        raise ValueError(f"T4 shape mismatch: sums={sums.shape}, angles={target_angles.shape}")
    if sums.shape[1] < 1 or np.any(~np.isfinite(sums)):
        raise ValueError("Invalid native-MUA calibration spike sums")
    if not np.isfinite(duration_s) or duration_s != MOVE_DURATION_SECONDS:
        raise ValueError(f"MOVE-T4 requires its fixed {MOVE_DURATION_SECONDS}-second window")
    usable = np.isfinite(target_angles)
    theta = target_angles[usable]
    design = np.stack([np.ones(theta.shape[0]), np.cos(theta), np.sin(theta)], axis=1)
    if design.shape[0] < 3:
        raise ValueError(f"T4 needs >=3 directional trials, got {design.shape[0]}")
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        raise ValueError(f"T4 direction design is rank {rank}, not 3")
    # Formal t4_from_trial_sums uses native counts per 20-ms bin.
    rates = sums[usable] / float(MOVE_BINS)
    coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, rates, rcond=None)
    if int(fitted_rank) != 3:
        raise ValueError(f"T4 least-squares rank {fitted_rank}, not 3")
    baseline, a, c = coefficients
    raw = np.stack([a, c, np.sqrt(a * a + c * c), baseline], axis=-1).astype(np.float32)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Non-finite native-MUA T4 features")
    return raw


def fit_carrier_normalizer(raw_rows: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Fit the formal source-only column normalizer for concatenated T4 rows."""
    if not raw_rows:
        raise ValueError("carrier normalizer needs at least one source row block")
    values = np.concatenate([np.asarray(rows, dtype=np.float32) for rows in raw_rows], axis=0)
    if values.ndim != 2 or values.shape[1] != CARRIER_DIM or not np.all(np.isfinite(values)):
        raise ValueError("carrier rows must be finite [N,4]")
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std <= _STD_FLOOR] = 1.0
    return {"mean": mean, "std": std}


def apply_carrier_normalizer(raw: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    """Apply an already-frozen source-only normalizer without refitting it."""
    if set(state) != {"mean", "std"}:
        raise ValueError("normalizer state must have exactly mean and std")
    values = np.asarray(raw, dtype=np.float32)
    mean, std = np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != CARRIER_DIM or mean.shape != (CARRIER_DIM,) or std.shape != (CARRIER_DIM,):
        raise ValueError("invalid carrier-normalizer shapes")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("carrier normalizer must be finite with positive std")
    return np.ascontiguousarray((values - mean) / std, dtype=np.float32)


__all__ = [
    "BIN_SECONDS", "MOVE_START_BIN", "MOVE_STOP_BIN", "MOVE_BINS", "MOVE_DURATION_SECONDS",
    "CARRIER_DIM", "estimate_move_t4", "fit_carrier_normalizer", "apply_carrier_normalizer",
]
