"""P16 frontend, legacy-compatible parameter initialization."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _plan as plan
from .bank import TaskBank

# P16 structural constants.
CONV_CHANNELS = 16
CONV_KERNEL = 5
SET_DIM = 256
N_SLOTS = 8
N_HEADS = 8
N_LAYERS = 4
TEMPORAL_WIDTH = 256
FFN_DIM = 512
SLOT_FFN_DIM = 1024
READOUT_HIDDEN = 128

# Local deterministic initialization for the P16 module layout.

_FRONTEND_RNG_DOMAIN = 0
_TEMPORAL_RNG_DOMAIN = 1_000_003
_FRONTEND_PREFIXES = ("frontend.", "final_norm.", "readout.")
_TEMPORAL_PREFIXES = ("temporal.",)


def _kaiming_uniform_(
    tensor: torch.Tensor, generator: torch.Generator, a: float = math.sqrt(5)
) -> None:
    if tensor.ndim < 2:
        bound = 1.0 / math.sqrt(max(tensor.numel(), 1))
    else:
        fan_in = tensor.size(1)
        for s in tensor.shape[2:]:
            fan_in *= s
        gain = math.sqrt(2.0 / (1.0 + a * a))
        std = gain / math.sqrt(max(fan_in, 1))
        bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound, generator=generator)


def _reinit_param(name: str, param: torch.Tensor, generator: torch.Generator) -> None:
    if param.ndim >= 2:
        _kaiming_uniform_(param, generator)
        return
    if name.endswith("bias") or param.ndim == 1:
        fan = param.numel()
        bound = 1.0 / math.sqrt(max(fan, 1))
        with torch.no_grad():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1.0)
            elif "norm" in name and name.endswith("bias"):
                param.zero_()
            else:
                param.uniform_(-bound, bound, generator=generator)
        return


def initialize_decoder(module: nn.Module, seed: int) -> None:
    """Initialize the P16 owner with deterministic frontend and temporal streams."""
    front_g = torch.Generator(device="cpu").manual_seed(
        int(seed) + _FRONTEND_RNG_DOMAIN
    )
    temp_g = torch.Generator(device="cpu").manual_seed(int(seed) + _TEMPORAL_RNG_DOMAIN)
    for name, param in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        if name.endswith("A_log") or name.endswith(".D") or name.endswith("dt_bias"):
            continue
        if param.device.type != "cpu":
            cpu = param.detach().cpu().clone()
            if name.startswith(_FRONTEND_PREFIXES):
                _reinit_param(name, cpu, front_g)
            elif name.startswith(_TEMPORAL_PREFIXES):
                _reinit_param(name, cpu, temp_g)
            else:
                _reinit_param(name, cpu, front_g)
            with torch.no_grad():
                param.copy_(cpu.to(device=param.device, dtype=param.dtype))
            continue
        if name.startswith(_FRONTEND_PREFIXES):
            _reinit_param(name, param, front_g)
        elif name.startswith(_TEMPORAL_PREFIXES):
            _reinit_param(name, param, temp_g)
        else:
            _reinit_param(name, param, front_g)
    if hasattr(module, "frontend") and hasattr(module.frontend, "slots"):
        slot_g = torch.Generator(device="cpu").manual_seed(
            int(seed) + _FRONTEND_RNG_DOMAIN + 17
        )
        slots = module.frontend.slots
        cpu = torch.randn(slots.shape, generator=slot_g, dtype=torch.float32)
        with torch.no_grad():
            slots.copy_((cpu * 0.02).to(device=slots.device, dtype=slots.dtype))


def unit_dropout_seed(seed: int, epoch: int, batch_id: int) -> int:
    """Return the deterministic per-batch unit-dropout seed."""
    payload = f"m2_small_unit_dropout|{int(seed)}|{int(epoch)}|{int(batch_id)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def whole_unit_dropout(
    unit_mask: torch.Tensor,
    p: float = plan.UNIT_DROPOUT,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop whole units for one window; the mask has no time axis.

    Equivalent to ``m2_dual_track_v1.decoders.whole_unit_dropout``:
    ``unit_mask`` is ``[N]`` or ``[B, N]`` (True = eligible). Each window drops
    units independently with probability ``p``. If every eligible unit of a
    window is dropped, restore its lowest originally-valid index (index 0 when
    the incoming mask is empty).
    """
    if not isinstance(unit_mask, torch.Tensor):
        unit_mask = torch.as_tensor(unit_mask)
    if unit_mask.dtype != torch.bool:
        unit_mask = unit_mask.bool()
    orig = unit_mask if unit_mask.dim() == 2 else unit_mask.unsqueeze(0)
    if p <= 0.0:
        return orig.clone()
    if p >= 1.0:
        drop = torch.ones(orig.shape, dtype=torch.bool, device=orig.device)
    else:
        if generator is None:
            rnd = torch.rand(orig.shape, device=orig.device)
        else:
            rnd = torch.rand(orig.shape, generator=generator).to(device=orig.device)
        drop = rnd < p
    keep = orig & ~drop
    empty = keep.sum(dim=-1) == 0
    if bool(empty.any()):
        restore = orig.to(torch.uint8).argmax(dim=-1)  # first True, else 0
        keep[torch.arange(keep.size(0), device=keep.device), restore] = True
    return keep


def _sinusoidal_pe(max_len: int, width: int) -> torch.Tensor:
    pe = torch.zeros(max_len, width)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class SharedCausalConv(nn.Module):
    """Per-unit causal Conv1d(1->16, k=5, left-pad 4) + SiLU. [B,L,N] -> [B,L,N,16].

    P1a-v2 route B: the DEFAULT forward is structurally identical to
    ``m2_b_small_stability_v1.decoder.SharedCausalConv.forward`` —
    ``F.pad(left_pad)`` then the ``nn.Conv1d`` module call (the ``F.conv1d``
    primitive) — so bf16 autocast evaluates the conv exactly like S1. The
    former fixed-order per-tap accumulation survives as
    :meth:`forward_local_reference` (reference/test path): it computes the
    identical linear causal operator with position-local, order-fixed
    arithmetic. Empirical record (torch 2.5.1 CPU, Phase 0): the
    ``F.conv1d`` primitive on shapes like [B*N, 1, L+4] bleeds ~1e-7 float
    roundoff across OUTPUT positions (im2col/GEMM path), so bit-level
    causality self-certification is impossible on the primitive path;
    ``causal_check`` therefore gates on ``CAUSAL_CHECK_TOLERANCE`` (1e-5).
    """

    def __init__(
        self, channels: int = CONV_CHANNELS, kernel: int = CONV_KERNEL
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size=kernel, padding=0, bias=True)
        self.act = nn.SiLU()
        self.left_pad = kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, n_units = x.shape
        h = x.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        h = F.pad(h, (self.left_pad, 0))  # [B*N, 1, W + K - 1]
        h = self.act(self.conv(h))  # F.conv1d primitive (bf16 under autocast, S1 path)
        return h.reshape(batch, n_units, self.conv.out_channels, width).permute(
            0, 3, 1, 2
        )

    def forward_local_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Explicit per-tap accumulation with a fixed summation order.

        Reference implementation of the same causal operator (kept for tests
        and diagnostics); not used by the default forward since route B.
        """
        batch, width, n_units = x.shape
        h = x.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        h = F.pad(h, (self.left_pad, 0))  # [B*N, 1, W + K - 1]
        weight = self.conv.weight  # [C_out, 1, K]
        bias = self.conv.bias  # [C_out]
        acc = (
            bias.view(1, -1, 1)
            .expand(batch * n_units, self.conv.out_channels, width)
            .clone()
        )
        for k in range(self.conv.kernel_size[0]):
            tap = h[:, 0, k : k + width].unsqueeze(1)  # x[t + k - (K-1)]
            acc = acc + weight[:, 0, k].view(1, -1, 1) * tap
        h = self.act(acc)
        return h.reshape(batch, n_units, self.conv.out_channels, width).permute(
            0, 3, 1, 2
        )


class SharedSetFrontend(nn.Module):
    """Per-bin set encoder: local conv + E0/carrier fusion + 8-slot MHA.

    Module attribute names are PINNED to the S1 blueprint
    (``m2_b_small_stability_v1.decoder.SharedSetFrontend``) so the state-dict
    key set matches exactly: local_conv / token_mlp / slots / slot_norm /
    token_norm / mha / slot_ffn_norm / slot_ffn / slot_proj.
    """

    def __init__(self, token_in: int) -> None:
        super().__init__()
        self.token_in = int(token_in)
        plan.require(
            self.token_in > CONV_CHANNELS + 4,
            "token_in must exceed local + carrier widths",
        )
        self.local_conv = SharedCausalConv()
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_in, SET_DIM),
            nn.GELU(),
            nn.Linear(SET_DIM, SET_DIM),
        )
        self.slots = nn.Parameter(torch.zeros(N_SLOTS, SET_DIM))
        self.slot_norm = nn.LayerNorm(SET_DIM)
        self.token_norm = nn.LayerNorm(SET_DIM)
        self.mha = nn.MultiheadAttention(
            embed_dim=SET_DIM, num_heads=N_HEADS, dropout=0.0, batch_first=True
        )
        self.slot_ffn_norm = nn.LayerNorm(SET_DIM)
        self.slot_ffn = nn.Sequential(
            nn.Linear(SET_DIM, SLOT_FFN_DIM),
            nn.GELU(),
            nn.Linear(SLOT_FFN_DIM, SET_DIM),
        )
        self.slot_proj = nn.Linear(N_SLOTS * SET_DIM, TEMPORAL_WIDTH)

    def first_token_weight(self) -> torch.Tensor:
        """[SET_DIM, token_in] weight of the first token layer (fold source)."""
        return self.token_mlp[0].weight

    def first_token_bias(self) -> torch.Tensor:
        return self.token_mlp[0].bias

    def forward(
        self,
        x: torch.Tensor,
        e0: torch.Tensor,
        t4: torch.Tensor,
        unit_keep: torch.Tensor,
    ) -> torch.Tensor:
        batch, width, n_units = x.shape
        local = self.local_conv(x)  # [B, L, N, 16]
        if e0.dim() == 2:
            e0 = e0.view(1, 1, n_units, -1).expand(batch, width, n_units, -1)
        else:
            e0 = e0.unsqueeze(1).expand(batch, width, n_units, -1)
        if t4.dim() == 2:
            t4 = t4.view(1, 1, n_units, -1).expand(batch, width, n_units, -1)
        else:
            t4 = t4.unsqueeze(1).expand(batch, width, n_units, -1)
        tokens = self.token_mlp(torch.cat([local, e0, t4], dim=-1))
        tokens = self.token_norm(tokens)
        slots = (
            self.slot_norm(self.slots)
            .view(1, 1, N_SLOTS, SET_DIM)
            .expand(batch, width, N_SLOTS, SET_DIM)
        )
        q = slots.reshape(batch * width, N_SLOTS, SET_DIM)
        k = tokens.reshape(batch * width, n_units, SET_DIM)
        pad = (
            (~unit_keep)
            .unsqueeze(1)
            .expand(batch, width, n_units)
            .reshape(batch * width, n_units)
        )
        attn_out, _ = self.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attn_out
        slots_out = slots_out + self.slot_ffn(self.slot_ffn_norm(slots_out))
        fused = slots_out.reshape(batch, width, N_SLOTS * SET_DIM)
        return self.slot_proj(fused)


class CausalSelfAttention(nn.Module):
    """8-head causal self-attention over the fused window (no cross-window KV)."""

    def __init__(self, width: int = TEMPORAL_WIDTH, heads: int = N_HEADS) -> None:
        super().__init__()
        plan.require(width % heads == 0, "temporal width must divide by heads")
        self.n_heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, dim = x.shape
        qkv = self.qkv(x).view(batch, width, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, hd]
        q, k, v = qkv.unbind(dim=0)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, width, dim)
        return self.proj(out)


class CausalTransformerBlock(nn.Module):
    """Pre-LN causal block: x + attn(LN(x)); x + FFN(LN(x)). FFN 256->512->256."""

    def __init__(self, width: int = TEMPORAL_WIDTH, ffn: int = FFN_DIM) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformerStack(nn.Module):
    """In-window sinusoidal PE + N_LAYERS pre-LN causal blocks (CausalPE4)."""

    def __init__(self, max_len: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(CausalTransformerBlock() for _ in range(N_LAYERS))
        self.register_buffer(
            "pe", _sinusoidal_pe(max_len, TEMPORAL_WIDTH), persistent=False
        )
        self.max_len = int(max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan.require(x.size(1) <= self.pe.size(0), "positional encoding overflow")
        h = x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype)
        for block in self.blocks:
            h = block(h)
        return h


__all__ = [
    "CausalSelfAttention",
    "CausalTransformerBlock",
    "CausalTransformerStack",
    "SharedCausalConv",
    "SharedSetFrontend",
    "whole_unit_dropout",
    "unit_dropout_seed",
    "UNIT_DROPOUT_DOMAIN_META",
    "S1_UNIT_DROPOUT_PAYLOAD_PREFIX",
    "CAUSAL_CHECK_TOLERANCE",
    "S1_M2_PARAM_COUNT",
    "CONV_CHANNELS",
    "CONV_KERNEL",
    "SET_DIM",
    "N_SLOTS",
    "N_HEADS",
    "N_LAYERS",
    "TEMPORAL_WIDTH",
    "FFN_DIM",
    "SLOT_FFN_DIM",
    "READOUT_HIDDEN",
]
