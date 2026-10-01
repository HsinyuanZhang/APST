"""F_nope_learned with a uniform near-flat slope start (user-directed 2026-09-17).

The learned_slope core parameterizes ``effective_slope = base * exp(slope_log)``
with the default ladder in ``base``.  An exact-zero start would zero the
gradient path as well (softplus-style saturation: d(eff)/d(theta) = eff), so
"start at flat" is realized as a *near-flat* uniform start: all eight heads
active at half-life = ``nearflat_context_multiple`` x the task context
duration (decay over the full context = 2**(-1/multiple), 2.7% at 25x).  The
slopes remain trainable and can only grow away from flat if the loss asks
for it, which restores the user's theoretical safety argument in practice.
"""
from __future__ import annotations

import torch

from learnable_recency_v1.config import half_life_to_slope
from pretrained_nope_learned.temporal_v1 import install_f_nope_learned

CONTEXT_SECONDS = {"m1": 100 * 0.02, "m2": 50 * 0.02, "h1": 300 * 0.02}
NEARFLAT_CONTEXT_MULTIPLE = 25


def install_f_nope_nearflat(decoder, task: str, *, context_multiple: int = NEARFLAT_CONTEXT_MULTIPLE):
    """Turn an E decoder into F whose slope start is uniformly near-flat."""
    if task not in CONTEXT_SECONDS:
        raise ValueError(f"unknown task {task!r}")
    install_f_nope_learned(decoder, task)
    core = decoder.temporal.core
    half_life = context_multiple * CONTEXT_SECONDS[task]
    slope = half_life_to_slope(half_life, 0.02)
    if not slope > 0.0:
        raise RuntimeError("near-flat slope must be strictly positive to stay trainable")
    with torch.no_grad():
        core.recency_slopes.copy_(
            torch.full_like(core.recency_slopes.to(dtype=torch.float32), float(slope))
        )
        # All eight heads now carry a strictly positive, trainable base slope.
        core._learn_mask.copy_(torch.ones_like(core._learn_mask, dtype=torch.bool))
        core.slope_log.zero_()
    effective = core.effective_slopes()
    if effective.numel() != 8 * 4 or not bool(torch.all(effective == float(slope))):
        raise RuntimeError("near-flat effective slopes must be uniform")
    if int((core.recency_slopes > 0).sum().item()) != 8:
        raise RuntimeError("near-flat requires all eight active heads")
    return decoder


__all__ = ["install_f_nope_nearflat", "CONTEXT_SECONDS", "NEARFLAT_CONTEXT_MULTIPLE"]
