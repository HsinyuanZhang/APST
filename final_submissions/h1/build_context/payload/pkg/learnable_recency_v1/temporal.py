"""Learnable cumulative-increment recency bias on top of ``RiftTemporal``.

Bias (all tiers):

    bias_h(t, s) = - g_h(t) * (C_h(t) - C_h(s))
    C_h(t) = Σ_{l ≤ t} a_h(l),   a_h(l) ≥ 0  (default; see learn_flat_heads)

``C(t) - C(s) = Σ_{l=s+1..t} a(l)`` is a reverse cumsum over the ≤75-entry
window, so the streaming cache never stores an unbounded prefix sum.

Increment source is the pre-LN residual ``h`` of the current layer (documented
in DESIGN.md). New parameters are constant-initialized and do not consume the
torch RNG, so shared ``named_parameters`` stay byte-equal to a same-seed
fixed-recency ``RiftTemporal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1, log
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from btransform_unified_v2.config import RiftTemporalConfig
from btransform_unified_v2.temporal import RiftTemporal, RiftTemporalState, _normalise_rows

from .config import LearnableRecencyConfig


def _inv_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus inverse requires a positive target")
    return log(expm1(value))


def _logit_exp_neg(slope: float) -> float:
    """``logit(exp(-s))`` so ``-log σ(b) = s`` when the FoX weight is zero."""
    if slope <= 0:
        raise ValueError("FoX init bias is undefined for a non-positive slope")
    return -slope - log(-expm1(-slope))


@dataclass
class LearnableRecencyState(RiftTemporalState):
    """Streaming cache plus per-token increments ``a_h(l)`` for cumulative tiers.

    ``increments[layer][row]`` is a python list of ``[heads]`` tensors, one per
    cached prior token (same length bound as the KV lists: ``window - 1``).
    """

    increments: list[list[list[Tensor]]]

    def select_rows(self, row_indices: Tensor | Sequence[int]) -> "LearnableRecencyState":
        indices = _normalise_rows(row_indices, self.batch_size, self.device)
        selected = indices.tolist()
        return LearnableRecencyState(
            keys=[[[entry for entry in layer[index]] for index in selected] for layer in self.keys],
            values=[[[entry for entry in layer[index]] for index in selected] for layer in self.values],
            positions=[self.positions[index] for index in selected],
            increments=[[[entry for entry in layer[index]] for index in selected] for layer in self.increments],
            device=self.device,
            dtype=self.dtype,
        )

    def reset_rows(self, row_indices: Tensor | Sequence[int]) -> "LearnableRecencyState":
        indices = _normalise_rows(row_indices, self.batch_size, self.device)
        for index in indices.tolist():
            for layer in range(len(self.keys)):
                self.keys[layer][index] = []
                self.values[layer][index] = []
                self.increments[layer][index] = []
            self.positions[index] = 0
        return self


def window_increment_sums(increments: Tensor) -> Tensor:
    """``sum(a[k+1:])`` along the last dim: ``C(t)-C(s)`` for each window key."""
    reverse = increments.flip(-1).cumsum(-1).flip(-1)
    return reverse - increments


class LearnableRecencyTemporal(RiftTemporal):
    """``RiftTemporal`` whose recency bias is a learned cumulative increment."""

    def __init__(
        self,
        config: RiftTemporalConfig,
        recency_cfg: LearnableRecencyConfig,
        *,
        steal_from: RiftTemporal | None = None,
    ) -> None:
        rng = torch.get_rng_state()
        try:
            if steal_from is None:
                super().__init__(config)
            else:
                nn.Module.__init__(self)
                self.config = steal_from.config
                self.add_module("blocks", steal_from.blocks)
                self.register_buffer(
                    "recency_slopes", steal_from.recency_slopes.detach().clone(), persistent=True
                )
                self._attention_backend = steal_from._attention_backend
            if recency_cfg.half_life_seconds != self.config.half_life_seconds:
                if len(recency_cfg.half_life_seconds) != self.config.heads:
                    raise ValueError("learnable half_life_seconds must match temporal heads")
            self.recency_cfg = recency_cfg
            self._install_learnable_parameters()
        finally:
            torch.set_rng_state(rng)

    @classmethod
    def from_initialized(cls, temporal: RiftTemporal, recency_cfg: LearnableRecencyConfig) -> "LearnableRecencyTemporal":
        """Steal an already-initialized stack and add new params without touching RNG."""
        return cls(temporal.config, recency_cfg, steal_from=temporal)

    def _install_learnable_parameters(self) -> None:
        cfg = self.recency_cfg
        heads = self.config.heads
        layers = self.config.layers
        width = self.config.width
        # New params must follow the module (a stolen cuda temporal must not
        # register cpu params; register_parameter never moves tensors).
        device = self.recency_slopes.device
        slopes = self.recency_slopes.detach().to(dtype=torch.float32)
        active = slopes > 0
        learn_mask = torch.ones(heads, dtype=torch.bool, device=device) if cfg.learn_flat_heads else active
        self.register_buffer("_learn_mask", learn_mask, persistent=True)
        self._new_param_names: list[str] = []

        def _add(name: str, tensor: Tensor) -> None:
            self.register_parameter(name, nn.Parameter(tensor))
            self._new_param_names.append(name)

        def _zeros(*shape: int) -> Tensor:
            return torch.zeros(shape, dtype=torch.float32, device=device)

        if cfg.tier == "learned_slope":
            shape = (layers, heads) if cfg.per_layer else (heads,)
            _add("slope_log", torch.zeros(shape, dtype=torch.float32, device=device))
            return
        if cfg.tier == "fox_gate":
            if cfg.per_layer:
                _add("fox_weight", _zeros(layers, heads, width))
                bias = torch.zeros(layers, heads, dtype=torch.float32, device=device)
                for head, slope in enumerate(slopes.tolist()):
                    if bool(learn_mask[head]) and slope > 0:
                        bias[:, head] = _logit_exp_neg(slope)
                    elif bool(learn_mask[head]) and slope == 0:
                        bias[:, head] = 20.0
                _add("fox_bias", bias)
            else:
                _add("fox_weight", _zeros(heads, width))
                bias = torch.zeros(heads, dtype=torch.float32, device=device)
                for head, slope in enumerate(slopes.tolist()):
                    if bool(learn_mask[head]) and slope > 0:
                        bias[head] = _logit_exp_neg(slope)
                    elif bool(learn_mask[head]) and slope == 0:
                        bias[head] = 20.0
                _add("fox_bias", bias)
            return
        if cfg.tier != "cable":
            raise ValueError(f"unknown tier {cfg.tier!r}")
        hidden = cfg.cable_hidden
        inv = torch.zeros(heads, dtype=torch.float32, device=device)
        for head, slope in enumerate(slopes.tolist()):
            if slope > 0:
                inv[head] = _inv_softplus(slope)
        if cfg.per_layer:
            _add("cable_in_weight", _zeros(layers, hidden, width))
            _add("cable_in_bias", _zeros(layers, hidden))
            _add("cable_out_weight", _zeros(layers, heads, hidden))
            _add("cable_out_bias", inv.unsqueeze(0).expand(layers, heads).contiguous().clone())
            if not cfg.cable_nw:
                _add("cable_gate_weight", _zeros(layers, heads, width))
                _add(
                    "cable_gate_bias",
                    torch.full((layers, heads), _inv_softplus(1.0), dtype=torch.float32, device=device),
                )
        else:
            _add("cable_in_weight", _zeros(hidden, width))
            _add("cable_in_bias", _zeros(hidden))
            _add("cable_out_weight", _zeros(heads, hidden))
            _add("cable_out_bias", inv.clone())
            if not cfg.cable_nw:
                _add("cable_gate_weight", _zeros(heads, width))
                _add("cable_gate_bias", torch.full((heads,), _inv_softplus(1.0), dtype=torch.float32, device=device))

    def new_parameter_names(self) -> list[str]:
        return list(self._new_param_names)

    def _layer_param(self, name: str, layer_index: int) -> Tensor:
        value = getattr(self, name)
        if not self.recency_cfg.per_layer:
            return value
        return value[layer_index]

    def effective_slopes(self) -> Tensor:
        """Learned constant slopes. Shape ``[H]`` or ``[L, H]`` for ``learned_slope``."""
        if self.recency_cfg.tier != "learned_slope":
            raise RuntimeError("effective_slopes is defined only for learned_slope")
        base = self.recency_slopes.to(dtype=torch.float32)
        theta = self.slope_log
        if self.recency_cfg.learn_flat_heads:
            active = (base > 0).to(dtype=theta.dtype)
            eps = float(base[base > 0].min().item()) if bool((base > 0).any()) else 0.0
            learned = base * torch.exp(theta)
            flat = eps * (torch.exp(theta) - 1.0)
            mask = active if theta.ndim == 1 else active.unsqueeze(0)
            return torch.where(mask > 0, learned, flat)
        if theta.ndim == 2:
            return base.unsqueeze(0) * torch.exp(theta)
        return base * torch.exp(theta)

    def export_constant_slopes(self) -> Tensor:
        """Constant slopes: ``[H]`` when shared, ``[L, H]`` when ``per_layer``."""
        slopes = self.effective_slopes().detach().to(dtype=torch.float32)
        if self.recency_cfg.per_layer and slopes.ndim == 1:
            slopes = slopes.unsqueeze(0).expand(self.config.layers, -1).contiguous()
        return slopes

    def export_shared_slope_vector(self) -> Tensor:
        """Single ``[H]`` vector for unchanged ``CpuRiftTemporalRuntime``.

        Succeeds only when every layer agrees. Per-layer default training
        diverges across layers; score-time then uses ``CpuLearnableRecencyRuntime``.
        """
        table = self.export_constant_slopes()
        if table.ndim == 1:
            return table
        if not torch.allclose(table, table[:1], atol=0.0, rtol=0.0):
            raise RuntimeError(
                "per_layer learned_slope layers disagree; use CpuLearnableRecencyRuntime"
            )
        return table[0].contiguous()

    def copy_exported_slopes_(self, dest: RiftTemporal) -> Tensor:
        exported = self.export_shared_slope_vector().to(device=dest.recency_slopes.device)
        dest.recency_slopes.copy_(exported)
        return exported

    def _increments_and_gates(self, tokens: Tensor, layer_index: int) -> tuple[Tensor, Tensor]:
        """Return ``a`` and ``g`` with shape ``[B, T, H]`` from pre-LN residual ``tokens``."""
        cfg = self.recency_cfg
        batch, time, _width = tokens.shape
        heads = self.config.heads
        base = self.recency_slopes.to(device=tokens.device, dtype=tokens.dtype)
        if cfg.tier == "learned_slope":
            slopes = self.effective_slopes().to(device=tokens.device, dtype=tokens.dtype)
            if slopes.ndim == 2:
                slopes = slopes[layer_index]
            increments = slopes.view(1, 1, heads).expand(batch, time, heads)
            gates = tokens.new_ones(batch, time, heads)
            return increments, gates
        if cfg.tier == "fox_gate":
            weight = self._layer_param("fox_weight", layer_index).to(dtype=tokens.dtype)
            bias = self._layer_param("fox_bias", layer_index).to(dtype=tokens.dtype)
            logits = functional.linear(tokens, weight, bias)
            # Same-rank softplus so w=0 ⇒ residual is exact 0 and a ≡ s0 in float32.
            init_logits = bias.to(dtype=tokens.dtype).view(1, 1, heads).expand_as(logits)
            increments = base.view(1, 1, heads) + (functional.softplus(-logits) - functional.softplus(-init_logits))
            originally_flat = base <= 0
            if not cfg.learn_flat_heads:
                increments = increments * (~originally_flat).to(dtype=increments.dtype).view(1, 1, heads)
            gates = tokens.new_ones(batch, time, heads)
            return increments, gates
        weight_in = self._layer_param("cable_in_weight", layer_index).to(dtype=tokens.dtype)
        bias_in = self._layer_param("cable_in_bias", layer_index).to(dtype=tokens.dtype)
        weight_out = self._layer_param("cable_out_weight", layer_index).to(dtype=tokens.dtype)
        bias_out = self._layer_param("cable_out_bias", layer_index).to(dtype=tokens.dtype)
        hidden = functional.relu(functional.linear(tokens, weight_in, bias_in))
        preact = functional.linear(hidden, weight_out, bias_out)
        init_preact = bias_out.to(dtype=tokens.dtype).view(1, 1, heads).expand_as(preact)
        residual = functional.softplus(preact) - functional.softplus(init_preact)
        # softplus(linear==bias) vs softplus(expanded bias) can differ by 1 ULP;
        # cancel the value at exact init while keeping residual autograd.
        match = torch.eq(preact, init_preact).to(dtype=residual.dtype)
        increments = base.view(1, 1, heads) + residual - residual.detach() * match
        originally_flat = base <= 0
        if not cfg.learn_flat_heads:
            increments = increments * (~originally_flat).to(dtype=increments.dtype).view(1, 1, heads)
        if cfg.cable_nw:
            gates = tokens.new_ones(batch, time, heads)
        else:
            gate_w = self._layer_param("cable_gate_weight", layer_index).to(dtype=tokens.dtype)
            gate_b = self._layer_param("cable_gate_bias", layer_index).to(dtype=tokens.dtype)
            pre_g = functional.linear(tokens, gate_w, gate_b)
            init_g = gate_b.to(dtype=tokens.dtype).view(1, 1, heads).expand_as(pre_g)
            gates = functional.softplus(pre_g) / functional.softplus(init_g)
        return increments, gates

    def _masked_increments(self, increments: Tensor, valid_mask: Tensor) -> Tensor:
        return torch.where(valid_mask.unsqueeze(-1), increments, torch.zeros_like(increments))

    def _apply_bias(self, logits: Tensor, increment_sums: Tensor, gates: Tensor) -> Tensor:
        return logits - gates * increment_sums

    def _age_plus_delta_sums(self, increments: Tensor, ages: Tensor) -> Tensor:
        """``s0 * age + Σ(a-s0)`` so init matches ``recency_slopes * age`` in float32.

        ``increments`` layout is ``[B, H, T, W]`` (local) or ``[B, H, W]`` (step).
        """
        base = self.recency_slopes.to(device=increments.device, dtype=increments.dtype)
        shape = [1] * increments.ndim
        shape[1] = base.shape[0]
        delta = increments - base.view(*shape)
        age_shape = [1] * increments.ndim
        age_shape[-1] = -1
        return base.view(*shape) * ages.view(*age_shape) + window_increment_sums(delta)

    def _forward_dense(self, z: Tensor, valid_mask: Tensor) -> Tensor:
        h = z
        time = z.shape[1]
        coordinates = torch.arange(time, device=z.device)
        ages = coordinates[:, None] - coordinates[None, :]
        for layer_index, (block, window) in enumerate(zip(self.blocks, self.config.windows)):
            increments, gates = self._increments_and_gates(h, layer_index)
            increments = self._masked_increments(increments, valid_mask)
            qkv = block.qkv(block.norm1(h)).view(z.shape[0], time, 3, self.config.heads, self.config.head_dim)
            q, k, v = qkv.unbind(dim=2)
            logits = torch.einsum("bthd,bshd->bhts", q, k) * (self.config.head_dim ** -0.5)
            base = self.recency_slopes.to(device=z.device, dtype=z.dtype).view(1, self.config.heads, 1, 1)
            da = increments - self.recency_slopes.to(device=z.device, dtype=z.dtype).view(1, 1, self.config.heads)
            prefix = da.cumsum(dim=1)
            delta = prefix.permute(0, 2, 1).unsqueeze(-1) - prefix.permute(0, 2, 1).unsqueeze(-2)
            increment_sums = base * ages.to(dtype=z.dtype).view(1, 1, time, time) + delta
            gate = gates.permute(0, 2, 1).unsqueeze(-1)
            logits = self._apply_bias(logits, increment_sums, gate)
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

    def _local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        valid_mask: Tensor,
        window: int,
        increments: Tensor | None = None,
        gates: Tensor | None = None,
    ) -> Tensor:
        batch, time, heads, head_dim = q.shape
        k_windows = functional.pad(k.permute(0, 2, 3, 1), (window - 1, 0)).unfold(-1, window, 1)
        v_windows = functional.pad(v.permute(0, 2, 3, 1), (window - 1, 0)).unfold(-1, window, 1)
        k_windows = k_windows.permute(0, 3, 1, 4, 2)
        v_windows = v_windows.permute(0, 3, 1, 4, 2)
        key_valid = functional.pad(valid_mask, (window - 1, 0), value=False).unfold(1, window, 1)
        logits = torch.einsum("bthd,bthkd->bhtk", q, k_windows) * (head_dim ** -0.5)
        if increments is None:
            ages = torch.arange(window - 1, -1, -1, device=q.device, dtype=q.dtype)
            slopes = self.recency_slopes.to(dtype=q.dtype).view(1, heads, 1, 1)
            logits = logits - slopes * ages.view(1, 1, 1, window)
        else:
            a_windows = functional.pad(increments.permute(0, 2, 1), (window - 1, 0)).unfold(-1, window, 1)
            ages = torch.arange(window - 1, -1, -1, device=q.device, dtype=q.dtype)
            increment_sums = self._age_plus_delta_sums(a_windows, ages)
            gate = gates.permute(0, 2, 1).unsqueeze(-1) if gates is not None else q.new_ones(1)
            logits = self._apply_bias(logits, increment_sums, gate)
        logits = logits.masked_fill(~key_valid[:, None, :, :], float("-inf"))
        logits = torch.where(valid_mask[:, None, :, None], logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bhtk,bthkd->bthd", weights, v_windows).reshape(batch, time, self.config.width)

    def _forward_local(self, z: Tensor, valid_mask: Tensor) -> Tensor:
        h = z
        for layer_index, (block, window) in enumerate(zip(self.blocks, self.config.windows)):
            increments, gates = self._increments_and_gates(h, layer_index)
            increments = self._masked_increments(increments, valid_mask)
            qkv = block.qkv(block.norm1(h)).view(
                z.shape[0], z.shape[1], 3, self.config.heads, self.config.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            attention = self._local_attention(q, k, v, valid_mask, window, increments, gates)
            residual = h + block.out(attention)
            h = residual + block.ffn(block.norm2(residual))
            h = torch.where(valid_mask.unsqueeze(-1), h, torch.zeros_like(h))
        return h

    def init_state(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> LearnableRecencyState:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        state_device = torch.device(device)
        return LearnableRecencyState(
            keys=[[[] for _ in range(batch_size)] for _ in self.config.windows],
            values=[[[] for _ in range(batch_size)] for _ in self.config.windows],
            positions=[0 for _ in range(batch_size)],
            increments=[[[] for _ in range(batch_size)] for _ in self.config.windows],
            device=state_device,
            dtype=dtype,
        )

    def step(
        self, z: Tensor, state: RiftTemporalState, valid_mask: Tensor | None = None
    ) -> tuple[Tensor, RiftTemporalState]:
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
        if self.recency_cfg.tier != "learned_slope" and not isinstance(state, LearnableRecencyState):
            raise TypeError("fox_gate/cable step() requires LearnableRecencyState increment cache")

        h = z
        tokens = z.unsqueeze(1)
        valid_rows = valid_mask.tolist()
        head_dim = self.config.head_dim
        scale = head_dim ** -0.5
        for layer_index, (block, window) in enumerate(zip(self.blocks, self.config.windows)):
            increments, gates = self._increments_and_gates(tokens, layer_index)
            a_now = increments[:, 0]
            g_now = gates[:, 0]
            qkv = block.qkv(block.norm1(h)).view(state.batch_size, 3, self.config.heads, head_dim)
            q, k, v = qkv.unbind(dim=1)
            cache_capacity = window - 1
            keys = z.new_zeros(state.batch_size, cache_capacity + 1, self.config.heads, head_dim)
            values = z.new_zeros(state.batch_size, cache_capacity + 1, self.config.heads, head_dim)
            cached_a = z.new_zeros(state.batch_size, cache_capacity + 1, self.config.heads)
            key_valid = torch.zeros(state.batch_size, cache_capacity + 1, dtype=torch.bool, device=z.device)
            increment_lists = state.increments if isinstance(state, LearnableRecencyState) else None
            for row in range(state.batch_size):
                old_keys = state.keys[layer_index][row]
                old_values = state.values[layer_index][row]
                old_increments = increment_lists[layer_index][row] if increment_lists is not None else []
                if old_keys:
                    old_count = len(old_keys)
                    keys[row, cache_capacity - old_count:cache_capacity] = torch.stack(old_keys)
                    values[row, cache_capacity - old_count:cache_capacity] = torch.stack(old_values)
                    key_valid[row, cache_capacity - old_count:cache_capacity] = True
                    if old_increments:
                        cached_a[row, cache_capacity - old_count:cache_capacity] = torch.stack(old_increments)
                if valid_rows[row]:
                    keys[row, cache_capacity] = k[row]
                    values[row, cache_capacity] = v[row]
                    cached_a[row, cache_capacity] = a_now[row]
                    key_valid[row, cache_capacity] = True
                if valid_rows[row] and window > 1:
                    state.keys[layer_index][row] = [*old_keys, k[row]][-(window - 1):]
                    state.values[layer_index][row] = [*old_values, v[row]][-(window - 1):]
                    if increment_lists is not None:
                        increment_lists[layer_index][row] = [*old_increments, a_now[row]][-(window - 1):]
            logits = torch.einsum("bhd,bkhd->bhk", q, keys) * scale
            if self.recency_cfg.tier == "learned_slope" and increment_lists is None:
                ages = torch.arange(cache_capacity, -1, -1, device=z.device, dtype=z.dtype)
                slopes = self.effective_slopes().to(dtype=z.dtype)
                if slopes.ndim == 2:
                    slopes = slopes[layer_index]
                logits = logits - slopes.view(1, self.config.heads, 1) * ages.view(1, 1, -1)
            else:
                ages = torch.arange(cache_capacity, -1, -1, device=z.device, dtype=z.dtype)
                increment_sums = self._age_plus_delta_sums(cached_a.permute(0, 2, 1), ages)
                logits = self._apply_bias(logits, increment_sums, g_now.unsqueeze(-1))
            logits = logits.masked_fill(~key_valid[:, None, :], float("-inf"))
            logits = torch.where(valid_mask[:, None, None], logits, torch.zeros_like(logits))
            weights = torch.softmax(logits, dim=-1)
            attention = torch.einsum("bhk,bkhd->bhd", weights, values).reshape(state.batch_size, self.config.width)
            residual = h + block.out(attention)
            h = residual + block.ffn(block.norm2(residual))
            tokens = h.unsqueeze(1)
        state.positions = [position + int(is_valid) for position, is_valid in zip(state.positions, valid_rows)]
        return torch.where(valid_mask.unsqueeze(1), h, torch.zeros_like(h)), state

    def _validate_state(self, state: RiftTemporalState) -> None:
        super()._validate_state(state)
        if not isinstance(state, LearnableRecencyState):
            return
        if len(state.increments) != self.config.layers:
            raise ValueError("state increment layer count does not match this model")
        for layer_index, window in enumerate(self.config.windows):
            if len(state.increments[layer_index]) != state.batch_size:
                raise ValueError("state increment rows do not match state batch size")
            for keys, increments in zip(state.keys[layer_index], state.increments[layer_index]):
                if self.recency_cfg.tier != "learned_slope" and (len(increments) != len(keys) or len(increments) > window - 1):
                    raise ValueError("state increment cache is malformed or exceeds configured capacity")

    @torch.no_grad()
    def increment_statistics(self, z: Tensor, valid_mask: Tensor | None = None) -> dict[str, object]:
        """Per-head increment summary for epoch logs (minival / smoke)."""
        if valid_mask is None:
            valid_mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        if self.recency_cfg.tier == "learned_slope":
            slopes = self.effective_slopes().detach().cpu()
            return {
                "tier": "learned_slope",
                "slopes": slopes.tolist(),
                "shape": list(slopes.shape),
            }
        h = z
        collected: list[Tensor] = []
        for layer_index, (block, _window) in enumerate(zip(self.blocks, self.config.windows)):
            increments, _gates = self._increments_and_gates(h, layer_index)
            increments = self._masked_increments(increments, valid_mask)
            collected.append(increments.reshape(-1, self.config.heads))
            qkv = block.qkv(block.norm1(h)).view(z.shape[0], z.shape[1], 3, self.config.heads, self.config.head_dim)
            q, k, v = qkv.unbind(dim=2)
            attention = self._local_attention(q, k, v, valid_mask, self.config.windows[layer_index], increments, _gates)
            residual = h + block.out(attention)
            h = residual + block.ffn(block.norm2(residual))
            h = torch.where(valid_mask.unsqueeze(-1), h, torch.zeros_like(h))
        stacked = torch.cat(collected, dim=0)
        keep = valid_mask.reshape(-1).repeat(len(self.blocks))
        values = stacked[keep]
        if values.numel() == 0:
            values = stacked
        quantiles = torch.quantile(values, torch.tensor([0.10, 0.50, 0.90], device=values.device), dim=0)
        return {
            "tier": self.recency_cfg.tier,
            "mean": values.mean(dim=0).cpu().tolist(),
            "p10": quantiles[0].cpu().tolist(),
            "p50": quantiles[1].cpu().tolist(),
            "p90": quantiles[2].cpu().tolist(),
            "n_tokens": int(values.shape[0]),
        }


__all__ = ["LearnableRecencyState", "LearnableRecencyTemporal", "window_increment_sums"]
