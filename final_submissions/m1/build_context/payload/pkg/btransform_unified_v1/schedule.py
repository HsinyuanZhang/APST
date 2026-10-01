"""Warmup + cosine LR schedule (S1-grade recipe, workorder §5).

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

import math

from . import plan


def warmup_cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float = plan.LR_PEAK,
    min_factor: float = plan.LR_MIN_FACTOR,
) -> float:
    """LR at optimizer step ``step`` (0-indexed, ``0 <= step <= total_steps``).

    Warmup: linear ``peak * step / warmup_steps`` for ``step < warmup_steps``,
    so ``step == warmup_steps`` lands exactly on ``peak``. Cosine: from peak at
    ``step == warmup_steps`` down to ``peak * min_factor`` at
    ``step == total_steps``. ``warmup_steps <= 0`` disables warmup (pure
    cosine from peak).
    """
    plan.require(total_steps >= 1, f"total_steps must be >= 1, got {total_steps}")
    plan.require(warmup_steps >= 0, f"warmup_steps must be >= 0, got {warmup_steps}")
    plan.require(0 <= step <= total_steps, f"step {step} outside [0, total_steps={total_steps}]")
    plan.require(peak > 0.0, "peak LR must be positive")
    plan.require(0.0 <= min_factor <= 1.0, "min_factor must be in [0, 1]")
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (float(step) / float(warmup_steps))
    if step == warmup_steps or warmup_steps >= total_steps:
        return peak  # land EXACTLY on peak at the warmup/cosine junction
    floor = peak * min_factor
    progress = (float(step) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * cos_factor


__all__ = ["warmup_cosine_lr"]
