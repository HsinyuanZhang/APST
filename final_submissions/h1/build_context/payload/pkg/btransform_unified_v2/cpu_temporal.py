"""Preallocated CPU-only inference cache for :class:`RiftTemporal`.

The caller owns model lifecycle: construct only from an eval-mode temporal
module and discard this runtime whenever its weights change.
"""
from __future__ import annotations

from typing import Sequence
import torch
from torch import Tensor
from .temporal import RiftTemporal


class CpuRiftTemporalRuntime:
    """Fixed-B FP32 RIFT stepper with bounded contiguous per-layer KV buffers."""
    def __init__(self, temporal: RiftTemporal, batch_size: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> None:
        if temporal.training:
            raise ValueError("CpuRiftTemporalRuntime requires temporal.eval()")
        if batch_size < 1 or dtype != torch.float32:
            raise ValueError("runtime requires positive batch_size and FP32 dtype")
        self.temporal, self.batch_size, self.device, self.dtype = temporal, batch_size, torch.device(device), dtype
        if next(temporal.parameters()).device != self.device:
            raise ValueError("temporal parameters and runtime device must match")
        c, h, dh = temporal.config, temporal.config.heads, temporal.config.head_dim
        self.keys = [torch.zeros(batch_size, window - 1, h, dh, device=self.device, dtype=dtype) for window in c.windows]
        self.values = [torch.zeros_like(value) for value in self.keys]
        self.lengths = [torch.zeros(batch_size, device=self.device, dtype=torch.long) for _ in c.windows]
        self._scratch_k = [torch.zeros(batch_size, window, h, dh, device=self.device, dtype=dtype) for window in c.windows]
        self._scratch_v = [torch.zeros_like(value) for value in self._scratch_k]
        self._scratch_valid = [torch.zeros(batch_size, window, device=self.device, dtype=torch.bool) for window in c.windows]
        self._ages = [torch.arange(window - 1, -1, -1, device=self.device, dtype=dtype) for window in c.windows]
        self._positions = [torch.arange(window - 1, device=self.device) for window in c.windows]

    @torch.inference_mode()
    def step(self, z: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        c = self.temporal.config
        if z.shape != (self.batch_size, c.width) or z.device != self.device or z.dtype != self.dtype:
            raise ValueError("z must be FP32 [fixed batch_size, configured width] on runtime device")
        if valid_mask is None:
            valid_mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        if valid_mask.shape != (self.batch_size,) or valid_mask.dtype != torch.bool or valid_mask.device != self.device:
            raise ValueError("valid_mask must be bool [fixed batch_size] on runtime device")
        h = z
        for layer, (block, window) in enumerate(zip(self.temporal.blocks, c.windows)):
            qkv = block.qkv(block.norm1(h)).view(self.batch_size, 3, c.heads, c.head_dim)
            q, k, v = qkv.unbind(1)
            cap = window - 1; sk, sv, sm = self._scratch_k[layer], self._scratch_v[layer], self._scratch_valid[layer]
            # Permanent right alignment makes the rolling update a whole-buffer
            # shift. No per-row stack, clone, or cache copy is needed.
            sk[:, :cap].copy_(self.keys[layer]); sv[:, :cap].copy_(self.values[layer])
            sm[:, :cap] = self._positions[layer].unsqueeze(0) >= (cap - self.lengths[layer]).unsqueeze(1)
            sk[:, cap].copy_(k); sv[:, cap].copy_(v); sm[:, cap] = valid_mask
            logits = torch.einsum("bhd,bkhd->bhk", q, sk) * (c.head_dim ** -0.5)
            logits -= self.temporal.recency_slopes.to(dtype=self.dtype).view(1, c.heads, 1) * self._ages[layer].view(1, 1, -1)
            logits.masked_fill_(~sm[:, None], float("-inf"))
            logits = torch.where(valid_mask[:, None, None], logits, torch.zeros_like(logits))
            attended = torch.einsum("bhk,bkhd->bhd", torch.softmax(logits, -1), sv).reshape(self.batch_size, c.width)
            residual = h + block.out(attended); h = residual + block.ffn(block.norm2(residual))
            if cap:
                self.keys[layer].copy_(torch.where(valid_mask[:, None, None, None], sk[:, 1:], self.keys[layer]))
                self.values[layer].copy_(torch.where(valid_mask[:, None, None, None], sv[:, 1:], self.values[layer]))
                self.lengths[layer].copy_(torch.where(valid_mask, (self.lengths[layer] + 1).clamp_max(cap), self.lengths[layer]))
        return torch.where(valid_mask[:, None], h, torch.zeros_like(h))

    @torch.inference_mode()
    def reset_rows(self, rows: Sequence[int] | Tensor) -> None:
        idx = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        if idx.ndim != 1 or (idx < 0).any() or (idx >= self.batch_size).any(): raise IndexError("rows outside fixed batch")
        for layer in range(len(self.keys)):
            self.keys[layer][idx] = 0; self.values[layer][idx] = 0; self.lengths[layer][idx] = 0

    @torch.inference_mode()
    def reorder(self, rows: Sequence[int] | Tensor) -> None:
        idx = torch.as_tensor(rows, device=self.device, dtype=torch.long)
        if idx.shape != (self.batch_size,) or (idx < 0).any() or (idx >= self.batch_size).any(): raise ValueError("reorder needs [fixed batch] valid row indices")
        for layer in range(len(self.keys)):
            self.keys[layer] = self.keys[layer].index_select(0, idx); self.values[layer] = self.values[layer].index_select(0, idx); self.lengths[layer] = self.lengths[layer].index_select(0, idx)
