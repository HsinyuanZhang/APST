"""Preallocated CPU inference cache for :class:`LearnableRecencyTemporal`.

Mirrors ``CpuRiftTemporalRuntime``: fixed-B FP32, right-aligned KV, plus a
``[B, window-1, H]`` increment buffer for cumulative tiers. ``learned_slope``
can instead export constant slopes into an unchanged ``CpuRiftTemporalRuntime``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from btransform_unified_v2.cpu_temporal import CpuRiftTemporalRuntime
from btransform_unified_v2.temporal import RiftTemporal

from .temporal import LearnableRecencyTemporal


class CpuLearnableRecencyRuntime:
    """Fixed-B FP32 stepper with bounded KV and increment buffers."""

    def __init__(
        self,
        temporal: LearnableRecencyTemporal,
        batch_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if temporal.training:
            raise ValueError("CpuLearnableRecencyRuntime requires temporal.eval()")
        if batch_size < 1 or dtype != torch.float32:
            raise ValueError("runtime requires positive batch_size and FP32 dtype")
        self.temporal = temporal
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        if next(temporal.parameters()).device != self.device:
            raise ValueError("temporal parameters and runtime device must match")
        config = temporal.config
        heads = config.heads
        head_dim = config.head_dim
        self.keys = [
            torch.zeros(batch_size, window - 1, heads, head_dim, device=self.device, dtype=dtype)
            for window in config.windows
        ]
        self.values = [torch.zeros_like(value) for value in self.keys]
        self.increments = [
            torch.zeros(batch_size, window - 1, heads, device=self.device, dtype=dtype)
            for window in config.windows
        ]
        self.lengths = [torch.zeros(batch_size, device=self.device, dtype=torch.long) for _ in config.windows]
        self._scratch_k = [
            torch.zeros(batch_size, window, heads, head_dim, device=self.device, dtype=dtype)
            for window in config.windows
        ]
        self._scratch_v = [torch.zeros_like(value) for value in self._scratch_k]
        self._scratch_a = [
            torch.zeros(batch_size, window, heads, device=self.device, dtype=dtype) for window in config.windows
        ]
        self._scratch_valid = [
            torch.zeros(batch_size, window, device=self.device, dtype=torch.bool) for window in config.windows
        ]
        self._positions = [torch.arange(window - 1, device=self.device) for window in config.windows]

    @torch.inference_mode()
    def step(self, z: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        config = self.temporal.config
        if z.shape != (self.batch_size, config.width) or z.device != self.device or z.dtype != self.dtype:
            raise ValueError("z must be FP32 [fixed batch_size, configured width] on runtime device")
        if valid_mask is None:
            valid_mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        if valid_mask.shape != (self.batch_size,) or valid_mask.dtype != torch.bool or valid_mask.device != self.device:
            raise ValueError("valid_mask must be bool [fixed batch_size] on runtime device")
        hidden = z
        tokens = z.unsqueeze(1)
        for layer, (block, window) in enumerate(zip(self.temporal.blocks, config.windows)):
            increments, gates = self.temporal._increments_and_gates(tokens, layer)
            a_now = increments[:, 0]
            g_now = gates[:, 0]
            qkv = block.qkv(block.norm1(hidden)).view(self.batch_size, 3, config.heads, config.head_dim)
            q, k, v = qkv.unbind(1)
            cap = window - 1
            scratch_k, scratch_v, scratch_a, scratch_valid = (
                self._scratch_k[layer],
                self._scratch_v[layer],
                self._scratch_a[layer],
                self._scratch_valid[layer],
            )
            scratch_k[:, :cap].copy_(self.keys[layer])
            scratch_v[:, :cap].copy_(self.values[layer])
            scratch_a[:, :cap].copy_(self.increments[layer])
            scratch_valid[:, :cap] = self._positions[layer].unsqueeze(0) >= (cap - self.lengths[layer]).unsqueeze(1)
            scratch_k[:, cap].copy_(k)
            scratch_v[:, cap].copy_(v)
            scratch_a[:, cap].copy_(a_now)
            scratch_valid[:, cap] = valid_mask
            logits = torch.einsum("bhd,bkhd->bhk", q, scratch_k) * (config.head_dim ** -0.5)
            ages = torch.arange(window - 1, -1, -1, device=self.device, dtype=self.dtype)
            increment_sums = self.temporal._age_plus_delta_sums(scratch_a.permute(0, 2, 1), ages)
            logits = logits - g_now.unsqueeze(-1) * increment_sums
            logits.masked_fill_(~scratch_valid[:, None], float("-inf"))
            logits = torch.where(valid_mask[:, None, None], logits, torch.zeros_like(logits))
            attended = torch.einsum("bhk,bkhd->bhd", torch.softmax(logits, -1), scratch_v).reshape(
                self.batch_size, config.width
            )
            residual = hidden + block.out(attended)
            hidden = residual + block.ffn(block.norm2(residual))
            tokens = hidden.unsqueeze(1)
            if cap:
                self.keys[layer].copy_(torch.where(valid_mask[:, None, None, None], scratch_k[:, 1:], self.keys[layer]))
                self.values[layer].copy_(
                    torch.where(valid_mask[:, None, None, None], scratch_v[:, 1:], self.values[layer])
                )
                self.increments[layer].copy_(
                    torch.where(valid_mask[:, None, None], scratch_a[:, 1:], self.increments[layer])
                )
                self.lengths[layer].copy_(
                    torch.where(valid_mask, (self.lengths[layer] + 1).clamp_max(cap), self.lengths[layer])
                )
        return torch.where(valid_mask[:, None], hidden, torch.zeros_like(hidden))

    @torch.inference_mode()
    def reset_rows(self, rows: Sequence[int] | Tensor) -> None:
        idx = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        if idx.ndim != 1 or (idx < 0).any() or (idx >= self.batch_size).any():
            raise IndexError("rows outside fixed batch")
        for layer in range(len(self.keys)):
            self.keys[layer][idx] = 0
            self.values[layer][idx] = 0
            self.increments[layer][idx] = 0
            self.lengths[layer][idx] = 0

    @torch.inference_mode()
    def reorder(self, rows: Sequence[int] | Tensor) -> None:
        idx = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        if idx.shape != (self.batch_size,) or (idx < 0).any() or (idx >= self.batch_size).any():
            raise ValueError("reorder needs [fixed batch] valid row indices")
        for layer in range(len(self.keys)):
            self.keys[layer] = self.keys[layer].index_select(0, idx)
            self.values[layer] = self.values[layer].index_select(0, idx)
            self.increments[layer] = self.increments[layer].index_select(0, idx)
            self.lengths[layer] = self.lengths[layer].index_select(0, idx)


def exported_cpu_runtime(
    temporal: LearnableRecencyTemporal, batch_size: int, device: str | torch.device = "cpu"
) -> tuple[RiftTemporal, CpuRiftTemporalRuntime]:
    """Build an unchanged ``CpuRiftTemporalRuntime`` when every layer agrees.

    Per-layer default training diverges; score-time then uses
    ``CpuLearnableRecencyRuntime`` on the learnable module, not this export.
    """
    if temporal.recency_cfg.tier != "learned_slope":
        raise ValueError("export path is defined only for learned_slope")
    exported = RiftTemporal(temporal.config).to(device=torch.device(device))
    exported.load_state_dict(
        {name: value for name, value in temporal.state_dict().items() if name in exported.state_dict()},
        strict=True,
    )
    temporal.copy_exported_slopes_(exported)
    exported.eval()
    return exported, CpuRiftTemporalRuntime(exported, batch_size, device=device)


__all__ = ["CpuLearnableRecencyRuntime", "exported_cpu_runtime"]
