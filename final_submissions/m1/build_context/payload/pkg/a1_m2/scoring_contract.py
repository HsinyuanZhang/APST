"""M2 A1 development scoring contract; no model selection outside EXT6."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


EXT6 = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
)


def causal_output_ema(prediction: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Causal per-output EMA for one chronologically ordered session.

    The first row is passed through unchanged (``state_0 = pred_0``), then
    ``state_t = alpha * state_(t-1) + (1-alpha) * pred_t``.  Callers must
    invoke this once per session; carrying state between sessions is forbidden.
    """
    values = np.ascontiguousarray(np.asarray(prediction, dtype=np.float64))
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("EMA requires a nonempty finite [Q,outputs] prediction array")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("EMA alpha must be in [0,1]")
    out = np.empty_like(values)
    out[0] = values[0]
    for index in range(1, len(values)):
        out[index] = float(alpha) * out[index - 1] + (1.0 - float(alpha)) * values[index]
    return out


def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(); h.update(a.dtype.str.encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def output_centered_variance_weighted_r2(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """M2 metric: center each output dimension before aggregate weighting."""
    y = np.ascontiguousarray(np.asarray(truth, dtype=np.float64)); p = np.ascontiguousarray(np.asarray(prediction, dtype=np.float64))
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("M2 truth/prediction must be finite matching [Q,2] arrays")
    mean = y.mean(axis=0); sse = ((y - p) ** 2).sum(axis=0); tss = ((y - mean) ** 2).sum(axis=0)
    if np.any(tss <= 0):
        raise ValueError("M2 output has non-positive centered TSS")
    per_output = 1.0 - sse / tss
    return {"r2": float(1.0 - sse.sum() / tss.sum()), "per_output_r2": per_output.tolist(),
            "sse_per_output": sse.tolist(), "tss_per_output": tss.tolist(), "truth_sha256": array_sha256(y),
            "prediction_sha256": array_sha256(p), "window_count": int(len(y)), "centering": "per-output session truth mean"}


def ext6_equal_session_selection(per_session: Mapping[str, Mapping[str, Any]]) -> float:
    if set(per_session) != set(EXT6):
        raise ValueError("selection requires exactly the six EXT6 development sessions")
    return float(np.mean([float(per_session[name]["r2"]) for name in EXT6]))


def earliest_finite_maximum(rows: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> tuple[int, float]:
    """Select only EMA candidate rows already scored on the full EXT6 surface."""
    candidates = [(int(epoch), ext6_equal_session_selection(per)) for epoch, per in rows.items()]
    if not candidates or not all(np.isfinite(score) for _, score in candidates):
        raise ValueError("all candidate epochs require finite full EXT6 scores")
    best = max(score for _, score in candidates)
    return min(epoch for epoch, score in candidates if score == best), float(best)
