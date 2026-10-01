"""Task-aware temporal variants for the isolated unified Falcon baseline.

This module deliberately leaves A1 and legacy training packages untouched.  It
uses A1's public facade/specification constants, but constructs a core with the
requested task's RIFT context before stealing the decoder's initialized blocks.
"""
from __future__ import annotations

from typing import Literal

import torch

from a1_m2.temporal_variants import A1Temporal, ALIBI_SLOPES, SPECS, VariantSpec
from btransform_unified_v2.config import RiftTemporalConfig
from btransform_unified_v2.streaming import RiftStreamDecoder
from btransform_unified_v2.temporal import RiftTemporal
from learnable_recency_v1.config import TASK_CONTEXT_BINS, dataset_config
from learnable_recency_v1.temporal import LearnableRecencyState, LearnableRecencyTemporal


Task = Literal["m1", "m2", "h1"]
Arm = Literal["A_flat", "B_alibi", "C_learned", "D_alibi_plus_pe"]


def task_windows(task: Task) -> tuple[int, ...]:
    """Return the native four-layer RIFT windows for the task's raw context."""
    if task not in TASK_CONTEXT_BINS:
        raise ValueError(f"unknown task {task!r}")
    config = RiftTemporalConfig.for_context(TASK_CONTEXT_BINS[task], layers=4, bias_mode="recency")
    return config.windows


def _new_core(task: Task, arm: Arm) -> tuple[RiftTemporal, VariantSpec]:
    if task not in TASK_CONTEXT_BINS:
        raise ValueError(f"unknown task {task!r}; use m1, m2, or h1")
    if arm not in SPECS:
        raise ValueError(f"unknown temporal arm {arm!r}")
    spec = SPECS[arm]
    core = RiftTemporal(RiftTemporalConfig.for_context(TASK_CONTEXT_BINS[task], layers=4, bias_mode="recency"))
    if spec.bias == "none":
        core.recency_slopes.zero_()
    elif spec.bias == "fixed_alibi":
        core.recency_slopes.copy_(torch.tensor(ALIBI_SLOPES, dtype=core.recency_slopes.dtype))
    elif spec.bias == "learned":
        # Explicit default ladder is a baseline contract, not an inherited M2 setting.
        cfg = dataset_config(task, tier="learned_slope", layers=4, ladder="default")
        core = LearnableRecencyTemporal.from_initialized(core, cfg)
    else:
        raise RuntimeError(f"unhandled bias {spec.bias!r}")
    core.set_attention_backend("local")
    return core, spec


def install_temporal(decoder, task: Task, arm: Arm):
    """Install a task-native A1 facade without redrawing shared transformer blocks.

    ``decoder.temporal.blocks`` is moved into the new task-shaped core.  Thus
    QKV/out/FFN/LayerNorm bytes are retained, while only bias policy/PE facade
    and C's declared learnable parameters are added.
    """
    if task not in TASK_CONTEXT_BINS:
        raise ValueError(f"unknown task {task!r}")
    if getattr(decoder, "task", task) != task:
        raise ValueError(f"decoder task {getattr(decoder, 'task', None)!r} != requested {task!r}")
    old = decoder.temporal
    if not isinstance(old, RiftTemporal):
        raise TypeError("decoder.temporal must be an initialized RiftTemporal")
    expected = task_windows(task)
    if tuple(old.config.windows) != expected:
        raise ValueError(f"decoder temporal windows {tuple(old.config.windows)} != {expected} for {task}")
    core, spec = _new_core(task, arm)
    core = core.to(next(decoder.parameters()).device)
    core.blocks = old.blocks
    temporal = A1Temporal(core, spec).to(next(decoder.parameters()).device)
    decoder.temporal = temporal
    decoder.temporal_config = temporal.config
    decoder.bias_mode = spec.bias
    return decoder


class UnifiedStreamDecoder(RiftStreamDecoder):
    """Task-independent stream packer; preserves C's increment caches per row."""

    def _pack_states(self, states: list):
        packed = super()._pack_states(states)
        core = self.decoder.temporal.core
        if not isinstance(core, LearnableRecencyTemporal):
            return packed
        if not isinstance(packed, LearnableRecencyState):
            raise TypeError("learned temporal did not create LearnableRecencyState")
        if any(not isinstance(state, LearnableRecencyState) for state in states):
            raise TypeError("learned temporal stream state drift")
        packed.increments = [
            [list(state.increments[layer][0]) for state in states]
            for layer in range(self.decoder.temporal_config.layers)
        ]
        return packed


__all__ = ["A1Temporal", "ALIBI_SLOPES", "SPECS", "UnifiedStreamDecoder", "install_temporal", "task_windows"]
