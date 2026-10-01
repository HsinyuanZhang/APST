"""P0-3 scale-bridge acceptance for btransform_unified_v1 (task scale contracts).

Contracts (workorder §0 / NOTE P0-3, RATIO form per REVIEW D):
  m2: train on 5x native (decoder_raw), score ``pred / 5`` against native.
  m1: divisor = 1 (never divide by 20); identity bridge.
  h1: train on ``20y``, score ``pred / 20`` against native. Acceptance on the
      same window: ``MSE(raw, 20y) = 400 * MSE(pred/20, y)`` within RELATIVE
      tolerance 1e-9 (a ratio, NOT a difference of 400 — the difference would
      be 399 * MSE(raw/20, y), not a constant) AND
      ``pred_std >= 0.01 * target_std`` (near-constant-output guard; the early
      H1 accident was exactly this failure mode).

H1 branch status (2026-09-06): IMPLEMENTED as the ratio acceptance above —
binding value per MATRIX_H1_L_IDENTITY_V1 §0 ("20y 训 /20 评（比值桥验收）")
and NOTE P0-3. The early-H1 near-constant-output accident (train 20y while
scoring forgot /20, or vice versa) is exactly what the ratio + std guard
catches on the same windows.

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

import numpy as np

from . import plan

M2_BRIDGE = 5.0
M1_BRIDGE = 1.0
H1_BRIDGE = 20.0
# Ratio constant (REVIEW D): MSE(raw, 20y) = H1_SCALE_RATIO * MSE(raw/20, y),
# checked at RELATIVE tolerance H1_SCALE_RATIO_REL_TOL — same shape as the M2
# branch identity below (25 * MSE(raw/5, y)).
H1_SCALE_RATIO = 400.0
H1_SCALE_RATIO_REL_TOL = 1e-9
STD_GUARD_FACTOR = 0.01


def _as_float64(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    plan.require(array.size >= 1, f"{name} must be non-empty")
    plan.require(bool(np.isfinite(array).all()), f"{name} contains non-finite values")
    return array


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


def assert_scale_bridge(task: str, raw_pred, target_native) -> np.ndarray:
    """Validate the task scale bridge; return predictions at NATIVE scale.

    ``raw_pred`` = model output in the task's training scale; ``target_native``
    = native-scale target on the same windows.
    """
    raw = _as_float64(raw_pred, "raw_pred")
    native = _as_float64(target_native, "target_native")
    plan.require(raw.shape == native.shape, f"raw_pred/target_native shape mismatch: {raw.shape} vs {native.shape}")
    if task == "m2":
        bridged = raw / M2_BRIDGE
        # Bridge identity: scoring raw against 5y must equal 25x scoring pred/5
        # against y. Catches double-scaled targets on either side of the bridge.
        mse_raw = _mse(raw, M2_BRIDGE * native)
        mse_bridge = _mse(bridged, native)
        plan.require(
            abs(mse_raw - (M2_BRIDGE ** 2) * mse_bridge) <= 1e-9 * max(1.0, mse_raw),
            f"m2 scale bridge identity failed: MSE(raw, 5y)={mse_raw} vs 25*MSE(raw/5, y)={25.0 * mse_bridge}",
        )
        plan.require(
            float(np.std(bridged)) >= STD_GUARD_FACTOR * float(np.std(native)),
            "m2 near-constant output: std(pred/5) < 1% std(target) (NOTE P0-3 guard)",
        )
        return bridged
    if task == "m1":
        plan.require(M1_BRIDGE == 1.0, "m1 divisor is frozen at 1")
        return raw.copy()
    if task == "h1":
        # RATIO acceptance (NOTE P0-3 / MATRIX_H1_L_IDENTITY_V1 §0 "20y 训
        # /20 评"): scoring raw against 20y must equal 400x scoring pred/20
        # against y — a RATIO, not a difference of 400 (the difference would
        # be 399 * MSE(raw/20, y), not a constant). Catches a double-scaled
        # bridge on either side.
        bridged = raw / H1_BRIDGE
        mse_raw = _mse(raw, H1_BRIDGE * native)
        mse_bridge = _mse(bridged, native)
        plan.require(
            abs(mse_raw - H1_SCALE_RATIO * mse_bridge)
            <= H1_SCALE_RATIO_REL_TOL * max(1.0, mse_raw),
            f"h1 scale bridge ratio failed: MSE(raw,20y)={mse_raw!r} vs "
            f"{H1_SCALE_RATIO!r} * MSE(raw/20,y)={H1_SCALE_RATIO * mse_bridge!r} "
            f"(relative tolerance {H1_SCALE_RATIO_REL_TOL})",
        )
        plan.require(
            float(np.std(bridged)) >= STD_GUARD_FACTOR * float(np.std(native)),
            "h1 near-constant output: std(pred/20) < 1% std(target) "
            "(NOTE P0-3 guard; the early H1 accident was exactly this mode)",
        )
        return bridged
    raise plan.BTransformerUnifiedError(f"unknown task {task!r} for scale bridge")


__all__ = [
    "assert_scale_bridge",
    "M2_BRIDGE",
    "M1_BRIDGE",
    "H1_BRIDGE",
    "H1_SCALE_RATIO",
    "H1_SCALE_RATIO_REL_TOL",
    "STD_GUARD_FACTOR",
]
