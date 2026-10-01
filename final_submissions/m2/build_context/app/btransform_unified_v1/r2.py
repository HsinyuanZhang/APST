"""R^2 metrics with the pooled / session-mean dual-report contract (NOTE P1-11).

The official evaluator aggregates a per-session variance-weighted R^2 by
averaging across sessions; the local formal path pools all points. These are
DIFFERENT surfaces: the two numbers returned by :func:`session_mean_report`
must never be subtracted from each other (nor from historical-best rows) —
report both side by side or convert explicitly.

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import plan


def _flat_float64(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    plan.require(array.size >= 1, f"{name} must be non-empty")
    plan.require(bool(np.isfinite(array).all()), f"{name} contains non-finite values")
    return array


def variance_weighted_r2(target: Any, pred: Any) -> float:
    """Centered coefficient of determination in float64: 1 - SSres/SStot.

    ``SSres = sum((target - pred)^2)``; ``SStot = sum((target - mean(target))^2)``.
    Refuses constant targets (SStot == 0): R^2 is undefined there.
    """
    t = _flat_float64(target, "target")
    p = _flat_float64(pred, "pred")
    plan.require(t.size == p.size, f"target/pred size mismatch: {t.size} vs {p.size}")
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    plan.require(ss_tot > 0.0, "SStot == 0 (constant target): R^2 undefined")
    ss_res = float(np.sum((t - p) ** 2))
    return 1.0 - ss_res / ss_tot


def _session_groups(session_ids: Any, n_points: int) -> list[tuple[str, np.ndarray]]:
    ids = np.asarray(session_ids)
    plan.require(ids.ndim == 1 and ids.size == n_points, "session_ids must be 1-D matching target length")
    groups: list[tuple[str, np.ndarray]] = []
    for sid in dict.fromkeys(ids.tolist()):  # first-appearance order, hashable ids
        groups.append((str(sid), np.flatnonzero(ids == sid)))
    return groups


def equal_session_mean(target: Any, pred: Any, session_ids: Any) -> float:
    """Unweighted mean of per-session :func:`variance_weighted_r2` (official-style
    aggregation 'each session's variance-weighted R^2, then average')."""
    t = _flat_float64(target, "target")
    p = _flat_float64(pred, "pred")
    groups = _session_groups(session_ids, t.size)
    per_session = [variance_weighted_r2(t[idx], p[idx]) for _, idx in groups]
    return float(np.mean(per_session))


def session_mean_report(target: Any, pred: Any, session_ids: Any) -> dict[str, Any]:
    """Both aggregation surfaces for one prediction set.

    Returns ``pooled_r2`` (all points concatenated) and ``session_mean_r2``
    (equal-weight mean of per-session R^2) plus the per-session breakdown.
    The two surfaces are NOT interchangeable: never subtract one from the
    other or mix them across rows (NOTE P1-11).
    """
    t = _flat_float64(target, "target")
    p = _flat_float64(pred, "pred")
    groups = _session_groups(session_ids, t.size)
    per_session = {sid: variance_weighted_r2(t[idx], p[idx]) for sid, idx in groups}
    return {
        "pooled_r2": variance_weighted_r2(t, p),
        "session_mean_r2": float(np.mean(list(per_session.values()))),
        "per_session_r2": per_session,
        "n_points": int(t.size),
        "n_sessions": int(len(per_session)),
        "contract": (
            "pooled and session-mean are different aggregation surfaces; "
            "report both, never subtract across rows (NOTE P1-11)"
        ),
    }


__all__ = ["variance_weighted_r2", "equal_session_mean", "session_mean_report"]
