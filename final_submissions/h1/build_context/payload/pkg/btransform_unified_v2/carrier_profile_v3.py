"""Shared conditional-response carrier template for M2 / DANDI688.

Poisson standardization, occupancy shrinkage, harmonic and signed-state
readouts.  Used by the 2026-09-09 carrier-iteration builders only.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

K_DIR = 8
STD_FLOOR = 1.0e-6
POISSON_COUNT_FLOOR = 1.0


def canonical_directions_rad() -> np.ndarray:
    return np.asarray(
        [-3.0 * math.pi / 4.0 + k * (math.pi / 4.0) for k in range(K_DIR)],
        dtype=np.float64,
    )


def nearest_canonical_index(angle: float) -> int:
    directions = canonical_directions_rad()
    wrapped = (directions - float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return int(np.argmin(np.abs(wrapped)))


def poisson_standardize(
    rates: np.ndarray, duration_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-wise Poisson-floor standardization (H1 `_unit_standardize` formula)."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[0] == 0:
        raise ValueError("rates must be nonempty [rows, units]")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    rate_mean = rates.mean(axis=0)
    mean_count = np.maximum(rate_mean * duration_s, POISSON_COUNT_FLOOR)
    noise_rate = np.sqrt(mean_count) / duration_s
    standardized = (rates - rate_mean[None, :]) / noise_rate[None, :]
    return standardized, rate_mean, noise_rate


def apply_affine(rates: np.ndarray, rate_mean: np.ndarray, noise_rate: np.ndarray) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float64)
    return (rates - np.asarray(rate_mean, dtype=np.float64)[None, :]) / np.asarray(
        noise_rate, dtype=np.float64
    )[None, :]


def conditional_response(z: np.ndarray, weights: np.ndarray, n0: float) -> np.ndarray:
    """R[u,k] = sum_t w[t,k] z[t,u] / (sum_t w[t,k] + n0)."""
    z = np.asarray(z, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if z.ndim != 2 or weights.ndim != 2 or z.shape[0] != weights.shape[0]:
        raise ValueError(f"z/W alignment: z={z.shape} W={weights.shape}")
    if n0 < 0:
        raise ValueError("n0 must be nonnegative")
    numer = weights.T @ z
    denom = weights.sum(axis=0) + float(n0)
    if np.any(denom <= 0):
        raise ValueError("conditional_response denominator is nonpositive")
    return np.ascontiguousarray((numer / denom[:, None]).T, dtype=np.float64)


def one_hot_directions(indices: np.ndarray, n_dirs: int = K_DIR) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= n_dirs):
        raise ValueError(f"direction indices must be in [0, {n_dirs})")
    weights = np.zeros((indices.size, n_dirs), dtype=np.float64)
    weights[np.arange(indices.size), indices] = 1.0
    return weights


def harmonic_readout(
    response_dir: np.ndarray, theta: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-harmonic readout on a fixed 8-direction table; missing dirs stay 0."""
    response_dir = np.asarray(response_dir, dtype=np.float64)
    if response_dir.ndim != 2 or response_dir.shape[1] != K_DIR:
        raise ValueError(f"response_dir must be [units, {K_DIR}]")
    if theta is None:
        theta = canonical_directions_rad()
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (K_DIR,):
        raise ValueError("theta must be length 8")
    a = 2.0 * (response_dir * np.cos(theta)[None, :]).mean(axis=1)
    c = 2.0 * (response_dir * np.sin(theta)[None, :]).mean(axis=1)
    return a, c, np.hypot(a, c)


def softplus(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(value, dtype=np.float64))


def signed_state_weights(velocity: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Per-axis [softplus(+v/rms), softplus(-v/rms)] concatenated."""
    velocity = np.asarray(velocity, dtype=np.float64)
    rms = np.maximum(np.asarray(rms, dtype=np.float64), STD_FLOOR)
    if velocity.ndim != 2 or rms.shape != (velocity.shape[1],):
        raise ValueError(f"velocity/rms alignment: v={velocity.shape} rms={rms.shape}")
    scaled = velocity / rms[None, :]
    parts = []
    for axis in range(velocity.shape[1]):
        parts.append(softplus(scaled[:, axis]))
        parts.append(softplus(-scaled[:, axis]))
    return np.column_stack(parts)


def fit_column_normalizer(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate([np.asarray(row, dtype=np.float64) for row in rows], axis=0)
    if joined.ndim != 2 or joined.shape[0] == 0:
        raise ValueError("normalizer rows must be nonempty [N, D]")
    mean = joined.mean(axis=0).astype(np.float64)
    std = np.maximum(joined.std(axis=0), STD_FLOOR).astype(np.float64)
    return mean, std


def apply_column_normalizer(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), STD_FLOOR)
    out = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("normalized carrier is nonfinite")
    return out


def stack_t4(a: np.ndarray, c: np.ndarray, m: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.column_stack((a, c, m, b)), dtype=np.float32)


def harmonic_t4_from_rows(
    rates: np.ndarray,
    direction_indices: np.ndarray,
    *,
    duration_s: float | None,
    n0: float,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Optional Poisson-std + one-hot harmonic T4. `baseline` is column 4 if given."""
    rates = np.asarray(rates, dtype=np.float64)
    if duration_s is None:
        z = rates
    else:
        z, _, _ = poisson_standardize(rates, duration_s)
    response = conditional_response(z, one_hot_directions(direction_indices), n0)
    a, c, m = harmonic_readout(response)
    if baseline is None:
        baseline = response.mean(axis=1)
    return stack_t4(a, c, m, baseline)


def assert_harmonic_matches_lstsq_complete8(
    rates: np.ndarray, direction_indices: np.ndarray, reference: np.ndarray, *, atol: float = 1e-5
) -> None:
    """n0=0, no standardize, all 8 dirs present → harmonic T4 equals per-dir-mean lstsq."""
    present = sorted(set(int(x) for x in np.asarray(direction_indices).tolist()))
    if present != list(range(K_DIR)):
        raise ValueError("identity gate requires all 8 directions")
    got = harmonic_t4_from_rows(rates, direction_indices, duration_s=None, n0=0.0)
    ref = np.asarray(reference, dtype=np.float32)
    if got.shape != ref.shape or not np.allclose(got, ref, atol=atol, rtol=0.0):
        max_abs = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
        raise AssertionError(f"harmonic vs lstsq identity failed: max_abs={max_abs}")
