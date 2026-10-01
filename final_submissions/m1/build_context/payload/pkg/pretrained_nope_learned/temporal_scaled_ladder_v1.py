"""Task-scaled ladder initializer: multi-scale half-lives adjusted per task.

The default ladder (0.08–2.56s) is optimal for M2 (1s context).  For other
tasks, the ladder is stretched by a scale factor to maintain the same
multi-scale coverage relative to the task's context duration:

  M2 (1s):  scale=1  → 0.08–2.56s  (default, proven optimal)
  M1 (2s):  scale=4  → 0.32–10.24s (gentler, avoids too-steep heads)
  H1 (6s):  scale=6  → 0.48–15.36s (stretched for long context)

All 8 heads remain trainable (slope_log), preserving multi-scale diversity
while shifting the absolute time constants to match each task's dynamics.
"""
from __future__ import annotations

import torch

from learnable_recency_v1.config import half_life_to_slope, dataset_config
from pretrained_nope_learned.temporal_v1 import install_f_nope_learned

# Default ladder half-lives (seconds), 6 active + 2 flat heads
DEFAULT_LADDER = (0.08, 0.16, 0.32, 0.64, 1.28, 2.56, None, None)

TASK_CONTEXT_SECONDS = {"m1": 100 * 0.02, "m2": 50 * 0.02, "h1": 300 * 0.02}

# Recommended scale factors per task (user-directed 2026-09-19)
TASK_SCALE = {"m1": 4.0, "m2": 1.0, "h1": 6.0}


def scaled_ladder(scale: float) -> tuple:
    """Return the ladder half-lives multiplied by a scale factor."""
    return tuple(None if hl is None else hl * scale for hl in DEFAULT_LADDER)


def install_f_scaled_ladder(decoder, task: str, scale: float | None = None):
    """Install F with a task-scaled ladder initialization.

    Args:
        decoder: the RiftDecoder to modify
        task: 'm1', 'm2', or 'h1'
        scale: scale factor; if None, uses TASK_SCALE[task]
    """
    if task not in TASK_CONTEXT_SECONDS:
        raise ValueError(f"unknown task {task!r}")
    if scale is None:
        scale = TASK_SCALE[task]
    if scale <= 0:
        raise ValueError("scale must be positive")

    # First install default F (creates the slope_log parameter)
    install_f_nope_learned(decoder, task)

    # Then override the recency_slopes buffer with scaled ladder values
    core = decoder.temporal.core
    ladder = scaled_ladder(scale)
    slopes = [half_life_to_slope(hl, 0.02) if hl is not None else 0.0 for hl in ladder]
    slopes_t = torch.tensor(slopes, dtype=core.recency_slopes.dtype)
    with torch.no_grad():
        if core.recency_slopes.ndim == 1:
            # Shared slopes across layers: shape [8]
            core.recency_slopes.copy_(slopes_t)
        else:
            # Per-layer slopes: shape [4, 8]
            for layer in range(core.recency_slopes.shape[0]):
                core.recency_slopes[layer].copy_(slopes_t)
        core.slope_log.zero_()  # slope_log=0 → effective = base = scaled ladder
    return decoder


__all__ = ["install_f_scaled_ladder", "scaled_ladder", "DEFAULT_LADDER", "TASK_SCALE", "TASK_CONTEXT_SECONDS"]
