"""Incremental finite-context Transformer temporal core used by RIFT."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import RiftTemporalConfig


@dataclass
class RiftTemporalState:
    """Per-batch-row streaming cache.

    Cache lists intentionally retain autograd history.  They must be discarded
    at an optimizer/weight update and are not a cross-update training cache.
    """

    keys: list[list[list[Tensor]]]
    values: list[list[list[Tensor]]]
    positions: list[int]
    device: torch.device
    dtype: torch.dtype

    @property
    def batch_size(self) -> int:
        return len(self.positions)

    @property
    def valid_lengths(self) -> list[list[int]]:
        """Current prior-KV count for every ``[layer][batch_row]``."""
        return [[len(row) for row in layer] for layer in self.keys]

    def select_rows(self, row_indices: Tensor | Sequence[int]) -> "RiftTemporalState":
        indices = _normalise_rows(row_indices, self.batch_size, self.device)
        selected = indices.tolist()
        return RiftTemporalState(
            keys=[[[entry for entry in layer[index]] for index in selected] for layer in self.keys],
            values=[[[entry for entry in layer[index]] for index in selected] for layer in self.values],
            positions=[self.positions[index] for index in selected],
            device=self.device,
            dtype=self.dtype,
        )

    def reset_rows(self, row_indices: Tensor | Sequence[int]) -> "RiftTemporalState":
        indices = _normalise_rows(row_indices, self.batch_size, self.device)
        for index in indices.tolist():
            for layer in range(len(self.keys)):
                self.keys[layer][index] = []
                self.values[layer][index] = []
            self.positions[index] = 0
        return self


def _normalise_rows(rows: Tensor | Sequence[int], batch_size: int, device: torch.device) -> Tensor:
    indices = torch.as_tensor(rows, device=device)
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("row indices must be a rank-1 integer tensor or sequence")
    indices = indices.to(dtype=torch.long)
    if (indices < 0).any() or (indices >= batch_size).any():
        raise IndexError("row index is outside state batch size")
    return indices


class _RiftBlock(nn.Module):
    def __init__(self, config: RiftTemporalConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, config.width),
        )


class RiftTemporal(nn.Module):
    """D-layer pre-LN RIFT temporal stack with exact streaming KV reuse."""

    def __init__(self, config: RiftTemporalConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(_RiftBlock(config) for _ in config.windows)
        slopes = [0.0 if half_life is None else log(2.0) * config.bin_seconds / half_life
                  for half_life in config.half_life_seconds]
        if config.bias_mode == "flat":
            slopes = [0.0] * config.heads
        self.register_buffer("recency_slopes", torch.tensor(slopes, dtype=torch.float32), persistent=True)
        self._attention_backend = "local"

    def set_attention_backend(self, backend: str) -> None:
        """Select the default batch-attention backend without changing weights."""
        if backend not in ("local", "dense"):
            raise ValueError("backend must be 'local' or 'dense'")
        self._attention_backend = backend

    def init_state(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> RiftTemporalState:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        state_device = torch.device(device)
        return RiftTemporalState(
            keys=[[[] for _ in range(batch_size)] for _ in self.config.windows],
            values=[[[] for _ in range(batch_size)] for _ in self.config.windows],
            positions=[0 for _ in range(batch_size)],
            device=state_device,
            dtype=dtype,
        )

    def select_rows(self, state: RiftTemporalState, row_indices: Tensor | Sequence[int]) -> RiftTemporalState:
        """Return state in an arbitrary new row order (duplicates are allowed)."""
        self._validate_state(state)
        return state.select_rows(row_indices)

    def reset_rows(self, state: RiftTemporalState, row_indices: Tensor | Sequence[int]) -> RiftTemporalState:
        """Clear only the selected streams and restart their relative time at zero."""
        self._validate_state(state)
        return state.reset_rows(row_indices)

    def forward(self, z: Tensor, valid_mask: Tensor | None = None, *, backend: str | None = None) -> Tensor:
        """Evaluate a full sequence with true-local attention or the dense oracle.

        ``backend='local'`` is the training default: its QK and AV products
        have only ``window`` keys per query.  ``'dense'`` exists solely as an
        independent correctness oracle and deliberately constructs ``T x T``.
        """
        if z.ndim != 3:
            raise ValueError("z must have shape [batch, time, width]")
        if z.shape[-1] != self.config.width:
            raise ValueError(f"z final dimension must equal configured width {self.config.width}")
        if z.shape[0] < 1:
            raise ValueError("z batch dimension must be non-empty")
        if valid_mask is not None:
            if valid_mask.shape != z.shape[:2] or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be a bool tensor with shape [batch, time]")
            if valid_mask.device != z.device:
                raise ValueError("valid_mask must be on the same device as z")
        if valid_mask is None:
            valid_mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        elif (valid_mask[:, :-1] & ~valid_mask[:, 1:]).any():
            raise ValueError("valid_mask may only contain left padding, not internal or right holes")
        backend = self._attention_backend if backend is None else backend
        if backend == "local":
            return self._forward_local(z, valid_mask)
        if backend == "dense":
            return self._forward_dense(z, valid_mask)
        raise ValueError("backend must be 'local' or 'dense'")

    def _forward_dense(self, z: Tensor, valid_mask: Tensor) -> Tensor:
        """Reference implementation that intentionally materializes T x T logits."""
        h = z
        time = z.shape[1]
        coordinates = torch.arange(time, device=z.device)
        ages = coordinates[:, None] - coordinates[None, :]
        slopes = self.recency_slopes.to(dtype=z.dtype).view(1, self.config.heads, 1, 1)
        for block, window in zip(self.blocks, self.config.windows):
            qkv = block.qkv(block.norm1(h)).view(z.shape[0], time, 3, self.config.heads, self.config.head_dim)
            q, k, v = qkv.unbind(dim=2)
            logits = torch.einsum("bthd,bshd->bhts", q, k) * (self.config.head_dim ** -0.5)
            logits = logits - slopes * ages.to(dtype=z.dtype).view(1, 1, time, time)
            allowed = ((ages >= 0) & (ages < window)).view(1, 1, time, time)
            allowed = allowed & valid_mask[:, None, None, :]
            logits = logits.masked_fill(~allowed, float("-inf"))
            logits = torch.where(valid_mask[:, None, :, None], logits, torch.zeros_like(logits))
            weights = torch.softmax(logits, dim=-1)
            attention = torch.einsum("bhts,bshd->bthd", weights, v).reshape(z.shape[0], time, self.config.width)
            residual = h + block.out(attention)
            h = residual + block.ffn(block.norm2(residual))
            h = torch.where(valid_mask.unsqueeze(-1), h, torch.zeros_like(h))
        return h

    def _local_attention(self, q: Tensor, k: Tensor, v: Tensor, valid_mask: Tensor, window: int) -> Tensor:
        """Causal fixed-width attention without constructing a T x T score tensor.

        Windows are ordered oldest-to-current.  Left padding is made invalid
        before softmax; a padding query receives an all-zero score row to keep
        softmax finite and is zeroed by the caller.
        """
        batch, time, heads, head_dim = q.shape
        # Padding occurs along time only.  ``unfold`` returns a local view with
        # shape [B,H,D,T,W], so QK/AV operate on [B,H,T,W], never [B,H,T,T].
        k_windows = functional.pad(k.permute(0, 2, 3, 1), (window - 1, 0)).unfold(-1, window, 1)
        v_windows = functional.pad(v.permute(0, 2, 3, 1), (window - 1, 0)).unfold(-1, window, 1)
        k_windows = k_windows.permute(0, 3, 1, 4, 2)
        v_windows = v_windows.permute(0, 3, 1, 4, 2)
        key_valid = functional.pad(valid_mask, (window - 1, 0), value=False).unfold(1, window, 1)
        logits = torch.einsum("bthd,bthkd->bhtk", q, k_windows) * (head_dim ** -0.5)
        ages = torch.arange(window - 1, -1, -1, device=q.device, dtype=q.dtype)
        slopes = self.recency_slopes.to(dtype=q.dtype).view(1, heads, 1, 1)
        logits = logits - slopes * ages.view(1, 1, 1, window)
        logits = logits.masked_fill(~key_valid[:, None, :, :], float("-inf"))
        logits = torch.where(valid_mask[:, None, :, None], logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bhtk,bthkd->bthd", weights, v_windows).reshape(batch, time, self.config.width)

    def _forward_local(self, z: Tensor, valid_mask: Tensor) -> Tensor:
        """Vectorized true-local batch implementation used for training."""
        h = z
        for block, window in zip(self.blocks, self.config.windows):
            qkv = block.qkv(block.norm1(h)).view(
                z.shape[0], z.shape[1], 3, self.config.heads, self.config.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            attention = self._local_attention(q, k, v, valid_mask, window)
            residual = h + block.out(attention)
            h = residual + block.ffn(block.norm2(residual))
            h = torch.where(valid_mask.unsqueeze(-1), h, torch.zeros_like(h))
        return h

    def step(
        self, z: Tensor, state: RiftTemporalState, valid_mask: Tensor | None = None
    ) -> tuple[Tensor, RiftTemporalState]:
        """Advance every state row exactly once with one current frontend token."""
        if z.ndim != 2:
            raise ValueError("z must have shape [batch, width]")
        if z.shape[-1] != self.config.width:
            raise ValueError(f"z final dimension must equal configured width {self.config.width}")
        self._validate_state(state)
        if z.shape[0] != state.batch_size:
            raise ValueError("z batch dimension must equal state batch size")
        if z.device != state.device or z.dtype != state.dtype:
            raise ValueError("z device and dtype must match state")
        if valid_mask is None:
            valid_mask = torch.ones(state.batch_size, dtype=torch.bool, device=z.device)
        elif valid_mask.shape != (state.batch_size,) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a bool tensor with shape [batch]")
        elif valid_mask.device != z.device:
            raise ValueError("valid_mask must be on the same device as z")

        h = z
        valid_rows = valid_mask.tolist()
        head_dim = self.config.head_dim
        scale = head_dim ** -0.5
        for layer_index, (block, window) in enumerate(zip(self.blocks, self.config.windows)):
            qkv = block.qkv(block.norm1(h)).view(state.batch_size, 3, self.config.heads, head_dim)
            q, k, v = qkv.unbind(dim=1)
            cache_capacity = window - 1
            keys = z.new_zeros(state.batch_size, cache_capacity + 1, self.config.heads, head_dim)
            values = z.new_zeros(state.batch_size, cache_capacity + 1, self.config.heads, head_dim)
            key_valid = torch.zeros(state.batch_size, cache_capacity + 1, dtype=torch.bool, device=z.device)
            for row in range(state.batch_size):
                old_keys = state.keys[layer_index][row]
                old_values = state.values[layer_index][row]
                if old_keys:
                    old_count = len(old_keys)
                    keys[row, cache_capacity - old_count:cache_capacity] = torch.stack(old_keys)
                    values[row, cache_capacity - old_count:cache_capacity] = torch.stack(old_values)
                    key_valid[row, cache_capacity - old_count:cache_capacity] = True
                if valid_rows[row]:
                    keys[row, cache_capacity] = k[row]
                    values[row, cache_capacity] = v[row]
                    key_valid[row, cache_capacity] = True
                if valid_rows[row] and window > 1:
                    state.keys[layer_index][row] = [*old_keys, k[row]][-(window - 1):]
                    state.values[layer_index][row] = [*old_values, v[row]][-(window - 1):]
            ages = torch.arange(cache_capacity, -1, -1, device=z.device, dtype=z.dtype)
            logits = torch.einsum("bhd,bkhd->bhk", q, keys) * scale
            logits = logits - self.recency_slopes.to(dtype=z.dtype).view(1, self.config.heads, 1) * ages.view(1, 1, -1)
            logits = logits.masked_fill(~key_valid[:, None, :], float("-inf"))
            logits = torch.where(valid_mask[:, None, None], logits, torch.zeros_like(logits))
            weights = torch.softmax(logits, dim=-1)
            attention = torch.einsum("bhk,bkhd->bhd", weights, values).reshape(state.batch_size, self.config.width)
            residual = h + block.out(attention)
            h = residual + block.ffn(block.norm2(residual))
        state.positions = [position + int(is_valid) for position, is_valid in zip(state.positions, valid_rows)]
        return torch.where(valid_mask.unsqueeze(1), h, torch.zeros_like(h)), state

    def _validate_state(self, state: RiftTemporalState) -> None:
        if not isinstance(state, RiftTemporalState):
            raise TypeError("state must be a RiftTemporalState created by init_state")
        if len(state.keys) != self.config.layers or len(state.values) != self.config.layers:
            raise ValueError("state layer count does not match this model")
        if state.batch_size < 1:
            raise ValueError("state batch size must be non-empty")
        for layer_index, window in enumerate(self.config.windows):
            if len(state.keys[layer_index]) != state.batch_size or len(state.values[layer_index]) != state.batch_size:
                raise ValueError("state cache rows do not match state batch size")
            for keys, values in zip(state.keys[layer_index], state.values[layer_index]):
                if len(keys) != len(values) or len(keys) > window - 1:
                    raise ValueError("state cache is malformed or exceeds configured capacity")
