"""BTransformerUnifiedDecoder: one structure, tasks differ only in geometry.

Identity: B-transformer unified series (8 learned slots + causal temporal core,
dual_track B arm -> S1-SMALL-COS -> this package). **NOT SPINT**. Comparisons
against the SPINT family are "same scoring surface, different system" only.

Structure (workorder §2/§3, blueprint =
``tfpd_exploration/src/m2_b_small_stability_v1/decoder.py::SmallTransformerDecoder``):

  frontend.local_conv   SharedCausalConv(1->16, k=5, left-pad 4, SiLU)
  frontend.token_mlp    Linear(16+d_e+4 -> 256) GELU Linear(256->256)
  frontend.token_norm   LayerNorm(256)
  frontend.slots/slot_norm/mha/slot_ffn_norm/slot_ffn/slot_proj
                        8-slot MHA (slot=Q, token=K=V) + slot FFN 256->1024->256
                        + slot_proj(8*256 -> 256)
  final_norm            LayerNorm(256) on the temporal output
  readout               Linear(256->128) GELU Linear(128->out_dim)
  temporal.blocks       in-window sinusoidal PE + 4 pre-LN causal blocks
                        (FFN 512)  [CausalPE4]

**S1 state-dict parity (P1a prerequisite)**: module names are pinned so the
``state_dict`` KEY SET (and per-key shapes) of the m2 build is EXACTLY the
sorted key set of ``SmallTransformerDecoder(seed=42)`` — 77 keys. This matters
because ``initialize_decoder`` visits parameters in sorted-name order with
fixed RNG domains, so an identical (name, shape) sequence consumes the RNG
identically and reproduces the S1 initialization bit for bit. The single
``frontend.token_mlp.0`` Linear(70 -> 256) replaces the old split
(w_local/w_e0/w_carrier/token_bias) representation; static folding (SPD-A1)
is now a VIEW of that one weight's columns
(:meth:`BTransformerUnifiedDecoder.static_term`), so folding no longer changes
the parameter layout.

Base forward arithmetic mirrors S1 op-for-op (cat([local, E0, T4]) -> single
token GEMM -> LayerNorm). P1a-v2 route B (ADDENDUM-P1a, preregistered)
removed the two P1a float-domain deviations:
  1. the causal conv forward now runs the S1 pad + ``F.conv1d`` primitive
     path (:meth:`SharedCausalConv.forward`, structurally identical to
     ``m2_b_small_stability_v1.decoder.SharedCausalConv``), so bf16 autocast
     evaluates it exactly like S1; the former fixed-order per-tap
     accumulation survives as
     :meth:`SharedCausalConv.forward_local_reference` (reference/test path),
     and :meth:`BTransformerUnifiedDecoder.causal_check` gates on a 1e-5
     tolerance because the conv primitive couples output positions at the
     ~1e-7 float level (Phase 0 record);
  2. whole-unit dropout seeds now come from the S1 source domain — payload
     ``m2_small_unit_dropout|{seed}|{epoch}|{batch_id}`` — via
     :func:`unit_dropout_seed` (imports the S1 function; a byte-identical
     fallback runs when tfpd_exploration is unavailable).
The remaining documented non-bitwise construct is ``forward_static_folded``
(SPD-A1 fast path: first token layer as split GEMMs; parity with
``forward`` is ``max|Δ| <= 1e-6`` in FP32, the workorder P3 gate caliber).
Folding is not used in training or scoring.

Whole-unit dropout p=0.10 applies ONLY in training mode (equivalent of
``tfpd_exploration/src/m2_dual_track_v1/decoders.py::whole_unit_dropout``)
and is generator-reproducible (:func:`unit_dropout_seed`, code item M); a
precomputed mask can be supplied as ``dropout_keep`` (S1 ``_fuse`` semantics:
shared across the batch when one row, never re-dropped).
Parameter initialization reuses the S1 source
``tfpd_exploration/src/m2_dual_track_v1/decoders.py::initialize_decoder``
so P1a is a true S1 replication (code item L); a local bit-equivalent
fallback runs only when that import is unavailable (``init_meta`` records
which source was used).
No cross-window KV: exact-E causality is preserved
(:meth:`BTransformerUnifiedDecoder.causal_check` self-certifies).
Input length is PINNED to ``l_in = window + prefix`` in training and
inference alike (code item N); the only variable-length constructs live
inside :meth:`BTransformerUnifiedDecoder.causal_check`.
Observations entering this module must already satisfy NOTE P0-5
(C-contiguous float32, finite); the module re-validates finiteness of X.
"""

from __future__ import annotations

import hashlib
import math
import sys
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import plan
from .bank import TaskBank

# Frozen structural constants (workorder §3): shared by every task.
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

# S1 blueprint invariant: m2 decoder parameter count (m2_b_small_stability_v1
# config.DECODER_PARAMS). The P1a build must reproduce it exactly.
S1_M2_PARAM_COUNT = 3_543_010

# Route-B causality gate caliber: the F.conv1d primitive couples output
# positions at the ~1e-7 float level (Phase 0 record), so causal_check gates
# on max|delta| <= this tolerance instead of bit equality.
CAUSAL_CHECK_TOLERANCE = 1e-5

# ---------------------------------------------------------------------------
# Initialization source (code item L): P1a replication requires init to be
# drawn by the SAME code as S1. Primary source is imported read-only from
# tfpd_exploration (workspace root = parent of this series root, put on
# sys.path); if that import is unavailable for any reason, a local
# bit-equivalent reimplementation below runs instead and ``init_meta``
# records fallback=True.
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = str(plan.REPO_ROOT.parent)


def _load_tfpd_initialize_decoder():
    """Import m2_dual_track_v1.decoders.initialize_decoder or return None."""
    if _WORKSPACE_ROOT not in sys.path:
        sys.path.insert(0, _WORKSPACE_ROOT)
    try:
        from tfpd_exploration.src.m2_dual_track_v1.decoders import initialize_decoder
    except Exception:  # unavailable in this environment -> local fallback
        return None
    return initialize_decoder


# Fallback constants/paths mirroring tfpd_exploration/src/m2_dual_track_v1/
# decoders.py exactly (FRONTEND_RNG_DOMAIN, TEMPORAL_RNG_DOMAIN, prefixes,
# kaiming(a=sqrt(5)) => bound 1/sqrt(fan_in), norm weight=1/bias=0, sorted
# named_parameters, dedicated slot sub-stream for `frontend.slots`).
_FALLBACK_FRONTEND_RNG_DOMAIN = 0
_FALLBACK_TEMPORAL_RNG_DOMAIN = 1_000_003
_FALLBACK_FRONTEND_PREFIXES = ("frontend.", "final_norm.", "readout.")
_FALLBACK_TEMPORAL_PREFIXES = ("temporal.",)


def _fallback_kaiming_uniform_(tensor: torch.Tensor, generator: torch.Generator, a: float = math.sqrt(5)) -> None:
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


def _fallback_reinit_param(name: str, param: torch.Tensor, generator: torch.Generator) -> None:
    if param.ndim >= 2:
        _fallback_kaiming_uniform_(param, generator)
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


def _initialize_decoder_fallback(module: nn.Module, seed: int) -> None:
    """Bit-equivalent local reimplementation of ``initialize_decoder``.

    Used only when :func:`_load_tfpd_initialize_decoder` returns None; the
    parameter count and shapes are untouched (re-initialization only).
    """
    front_g = torch.Generator(device="cpu").manual_seed(int(seed) + _FALLBACK_FRONTEND_RNG_DOMAIN)
    temp_g = torch.Generator(device="cpu").manual_seed(int(seed) + _FALLBACK_TEMPORAL_RNG_DOMAIN)
    for name, param in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        if name.endswith("A_log") or name.endswith(".D") or name.endswith("dt_bias"):
            continue
        if param.device.type != "cpu":
            cpu = param.detach().cpu().clone()
            if name.startswith(_FALLBACK_FRONTEND_PREFIXES):
                _fallback_reinit_param(name, cpu, front_g)
            elif name.startswith(_FALLBACK_TEMPORAL_PREFIXES):
                _fallback_reinit_param(name, cpu, temp_g)
            else:
                _fallback_reinit_param(name, cpu, front_g)
            with torch.no_grad():
                param.copy_(cpu.to(device=param.device, dtype=param.dtype))
            continue
        if name.startswith(_FALLBACK_FRONTEND_PREFIXES):
            _fallback_reinit_param(name, param, front_g)
        elif name.startswith(_FALLBACK_TEMPORAL_PREFIXES):
            _fallback_reinit_param(name, param, temp_g)
        else:
            _fallback_reinit_param(name, param, front_g)
    if hasattr(module, "frontend") and hasattr(module.frontend, "slots"):
        slot_g = torch.Generator(device="cpu").manual_seed(int(seed) + _FALLBACK_FRONTEND_RNG_DOMAIN + 17)
        slots = module.frontend.slots
        cpu = torch.randn(slots.shape, generator=slot_g, dtype=torch.float32)
        with torch.no_grad():
            slots.copy_((cpu * 0.02).to(device=slots.device, dtype=slots.dtype))


def _load_s1_unit_dropout_seed():
    """Import m2_b_small_stability_v1.training.unit_dropout_seed or return None."""
    if _WORKSPACE_ROOT not in sys.path:
        sys.path.insert(0, _WORKSPACE_ROOT)
    try:
        from tfpd_exploration.src.m2_b_small_stability_v1.training import unit_dropout_seed
    except Exception:  # unavailable in this environment -> byte-identical fallback
        return None
    return unit_dropout_seed


_S1_UNIT_DROPOUT_SEED = _load_s1_unit_dropout_seed()

# Route-B alignment (ADDENDUM-P1a): the dropout payload domain IS the S1
# domain ``m2_small_unit_dropout|{seed}|{epoch}|{batch_id}`` — same source
# function when importable, byte-identical replication otherwise.
S1_UNIT_DROPOUT_PAYLOAD_PREFIX = "m2_small_unit_dropout"

UNIT_DROPOUT_DOMAIN_META: dict[str, Any] = {
    "payload_prefix": S1_UNIT_DROPOUT_PAYLOAD_PREFIX,
    "source": "import" if _S1_UNIT_DROPOUT_SEED is not None else "fallback",
    "origin": "tfpd_exploration.src.m2_b_small_stability_v1.training.unit_dropout_seed",
    "construction": "sha256(payload).digest()[:8] little-endian % 2**63",
}


def unit_dropout_seed(seed: int, epoch: int, batch_id: int) -> int:
    """Deterministic per-(seed, epoch, batch) dropout seed (code item M).

    P1a-v2 route B: bit-identical to
    ``tfpd_exploration/src/m2_b_small_stability_v1/training.py::
    unit_dropout_seed`` — the S1 function is imported and called directly;
    when tfpd_exploration is unavailable a byte-identical replication of the
    payload ``"m2_small_unit_dropout|{seed}|{epoch}|{batch_id}"`` runs
    instead (``UNIT_DROPOUT_DOMAIN_META`` records which source was used).
    Training loops derive the generator per batch as
    ``torch.Generator(device="cpu").manual_seed(unit_dropout_seed(seed,
    epoch, batch))`` — exactly S1 ``unit_dropout_mask``'s construction — and
    pass the drawn mask (``dropout_keep``) so whole-unit dropout reproduces
    S1's mask stream bit for bit.
    """
    if _S1_UNIT_DROPOUT_SEED is not None:
        return int(_S1_UNIT_DROPOUT_SEED(int(seed), int(epoch), int(batch_id)))
    payload = (
        f"{S1_UNIT_DROPOUT_PAYLOAD_PREFIX}|{int(seed)}|{int(epoch)}|{int(batch_id)}".encode()
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


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
    div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
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

    def __init__(self, channels: int = CONV_CHANNELS, kernel: int = CONV_KERNEL) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size=kernel, padding=0, bias=True)
        self.act = nn.SiLU()
        self.left_pad = kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, n_units = x.shape
        h = x.permute(0, 2, 1).reshape(batch * n_units, 1, width)
        h = F.pad(h, (self.left_pad, 0))  # [B*N, 1, W + K - 1]
        h = self.act(self.conv(h))  # F.conv1d primitive (bf16 under autocast, S1 path)
        return h.reshape(batch, n_units, self.conv.out_channels, width).permute(0, 3, 1, 2)

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
        acc = bias.view(1, -1, 1).expand(batch * n_units, self.conv.out_channels, width).clone()
        for k in range(self.conv.kernel_size[0]):
            tap = h[:, 0, k : k + width].unsqueeze(1)  # x[t + k - (K-1)]
            acc = acc + weight[:, 0, k].view(1, -1, 1) * tap
        h = self.act(acc)
        return h.reshape(batch, n_units, self.conv.out_channels, width).permute(0, 3, 1, 2)


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
        plan.require(self.token_in > CONV_CHANNELS + 4, "token_in must exceed local + carrier widths")
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

    def forward(self, x: torch.Tensor, e0: torch.Tensor, t4: torch.Tensor, unit_keep: torch.Tensor) -> torch.Tensor:
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
        slots = self.slot_norm(self.slots).view(1, 1, N_SLOTS, SET_DIM).expand(
            batch, width, N_SLOTS, SET_DIM
        )
        q = slots.reshape(batch * width, N_SLOTS, SET_DIM)
        k = tokens.reshape(batch * width, n_units, SET_DIM)
        pad = (~unit_keep).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
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
        self.ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformerStack(nn.Module):
    """In-window sinusoidal PE + N_LAYERS pre-LN causal blocks (CausalPE4)."""

    def __init__(self, max_len: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(CausalTransformerBlock() for _ in range(N_LAYERS))
        self.register_buffer("pe", _sinusoidal_pe(max_len, TEMPORAL_WIDTH), persistent=False)
        self.max_len = int(max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan.require(x.size(1) <= self.pe.size(0), "positional encoding overflow")
        h = x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype)
        for block in self.blocks:
            h = block(h)
        return h


class BTransformerUnifiedDecoder(nn.Module):
    """Unified 8-slot + causal-temporal-core decoder (NOT SPINT).

    ``task`` is a task name (``"m2"`` / ``"m1"`` / ``"h1"``) resolved through
    :data:`btransform_unified_v1.plan.TASK_GEOMETRY`, or an explicit geometry
    mapping with the same keys. ``"h1"`` cannot be built by name while its
    ``e0_dim`` or ``prefix`` carries a PENDING sentinel — temporary dev
    geometry must be passed as a mapping (e0_dim) plus an explicit
    ``override_prefix`` (prefix sentinel, code item P) and may not back any
    alignment claim (NOTE §6). P1a passes an explicit m2 mapping with
    ``prefix=0`` (S1 parity, L_in=window=50). An integer geometry prefix is
    frozen: ``override_prefix`` only ever resolves the sentinel, never
    overrides a settled value.

    Top-level module names (frontend / final_norm / readout / temporal) are
    pinned to the S1 blueprint so ``initialize_decoder``'s sorted-name RNG
    walk is identical (see module docstring).
    """

    name = "B-TRANSFORMER-UNIFIED"
    schema = plan.SCHEMA

    def __init__(
        self,
        task: str | Mapping[str, Any],
        seed: int = plan.SEED,
        override_prefix: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(task, str):
            self.task = task
            geometry = plan.task_geometry(task)
        else:
            self.task = str(task.get("task", "adhoc-geometry"))
            geometry = dict(task)
            for key in ("window", "prefix", "units", "e0_dim", "carrier_dim", "out_dim"):
                plan.require(key in geometry, f"ad-hoc geometry missing key {key!r}")
        self.geometry = dict(geometry)
        self.window = int(geometry["window"])
        self.units = int(geometry["units"])
        self.e0_dim = plan.resolved_e0_dim(geometry)
        raw_prefix = geometry["prefix"]
        self.prefix = plan.resolved_prefix(geometry, override_prefix)
        self.pending_prefix = raw_prefix if isinstance(raw_prefix, str) else None
        self.carrier_dim = int(geometry["carrier_dim"])
        self.out_dim = int(geometry["out_dim"])
        self.l_in = self.window + self.prefix
        self.token_in = CONV_CHANNELS + self.e0_dim + self.carrier_dim
        plan.require(self.window >= 1, "window must be >= 1")
        plan.require(self.prefix >= 0, "prefix must be >= 0")
        plan.require(self.units >= 1, "units must be >= 1")
        plan.require(self.carrier_dim == 4, "carrier_dim is 4 for every task in this series")
        self.unit_dropout_p = float(plan.UNIT_DROPOUT)

        # --- frontend (names pinned to S1 SharedSetFrontend) ----------------
        self.frontend = SharedSetFrontend(self.token_in)
        # --- readout head (names pinned: final_norm / readout) ---------------
        self.final_norm = nn.LayerNorm(TEMPORAL_WIDTH)
        self.readout = nn.Sequential(
            nn.Linear(TEMPORAL_WIDTH, READOUT_HIDDEN),
            nn.GELU(),
            nn.Linear(READOUT_HIDDEN, self.out_dim),
        )
        # --- temporal core (names pinned: temporal.blocks.*, CausalPE4) ------
        self.temporal = CausalTransformerStack(self.l_in)
        self.init_meta: dict[str, Any] = self._init_parameters(seed)

    # ------------------------------------------------------------------ init
    def _init_parameters(self, seed: int) -> dict[str, Any]:
        """Initialize parameters with the S1-source semantics (code item L).

        Primary source: ``tfpd_exploration/src/m2_dual_track_v1/decoders.py::
        initialize_decoder`` — the exact code S1 ran, so P1a replication keeps
        init same-source. It re-initializes parameters by module name
        (sorted named_parameters, frontend/temporal RNG domains, kaiming-style
        uniform with bound 1/sqrt(fan_in), norm weight=1/bias=0); parameter
        COUNT and SHAPES are untouched, so the blueprint invariant
        3,543,010 params for m2 is preserved. Because the state-dict key set
        matches SmallTransformerDecoder exactly, the RNG walk (sorted names x
        numel) is identical and the drawn init values are bit-identical to S1.
        If the tfpd_exploration import is unavailable, a local bit-equivalent
        implementation runs instead and the returned init_meta records
        ``fallback=True``.
        """
        initializer = _load_tfpd_initialize_decoder()
        if initializer is not None:
            initializer(self, seed)
            meta: dict[str, Any] = {
                "source": "initialize_decoder",
                "origin": "tfpd_exploration.src.m2_dual_track_v1.decoders",
                "fallback": False,
            }
        else:
            _initialize_decoder_fallback(self, seed)
            meta = {
                "source": "fallback",
                "origin": "btransform_unified_v1.model._initialize_decoder_fallback",
                "fallback": True,
            }
        meta["seed"] = int(seed)
        return meta

    # ------------------------------------------------------------ bank utils
    def _bank_arrays(self, bank: TaskBank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = torch.from_numpy(np.ascontiguousarray(bank.E0, dtype=np.float32))
        carrier = torch.from_numpy(np.ascontiguousarray(bank.carrier, dtype=np.float32))
        mask = torch.from_numpy(np.ascontiguousarray(bank.unit_mask, dtype=np.bool_))
        plan.require(e0.shape == (self.units, self.e0_dim), f"bank E0 {tuple(e0.shape)} != model {(self.units, self.e0_dim)}")
        plan.require(carrier.shape == (self.units, self.carrier_dim), "bank carrier shape mismatch")
        plan.require(mask.shape == (self.units,), "bank unit_mask shape mismatch")
        return e0, carrier, mask

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        plan.require(x.dim() == 3, f"X must be [B, L, N], got {tuple(x.shape)}")
        plan.require(x.size(2) == self.units, f"X units {x.size(2)} != {self.units}")
        # Code item N: training and inference pin the SAME input length
        # l_in = window + prefix (PE is counted from the window head; a
        # variable L would shift the trailing-bin PE position). Variable-length
        # constructs are only allowed inside causal_check.
        plan.require(
            x.size(1) == self.l_in,
            f"X length {x.size(1)} != pinned l_in {self.l_in} (window {self.window} "
            f"+ prefix {self.prefix}); training and inference must use the same length",
        )
        plan.require(torch.isfinite(x).all(), "X contains non-finite values (NOTE P0-5)")
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return x

    def _expand_keep(
        self,
        bank: TaskBank,
        unit_mask: torch.Tensor | None,
        batch: int,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resolve the [B, N] keep mask (S1 ``_fuse`` semantics).

        ``dropout_keep`` (if given) is used verbatim — expanded across the
        batch when it carries a single row — and is NEVER re-dropped, exactly
        like ``SmallTransformerDecoder._fuse(dropout_keep=...)``. Otherwise the
        bank mask (or ``unit_mask``) is expanded and, in training mode with
        p>0, whole-unit dropout is applied (drawn on the CPU ``generator``
        when supplied so CUDA tensors stay valid).
        """
        if dropout_keep is not None:
            keep = dropout_keep.to(device=self.frontend.slots.device, dtype=torch.bool)
            if keep.dim() == 1:
                keep = keep.unsqueeze(0)
            plan.require(keep.dim() == 2 and keep.size(1) == self.units, "dropout_keep must be [N] or [B, N]")
            if keep.size(0) == 1 and batch > 1:
                keep = keep.expand(batch, -1)
            plan.require(keep.size(0) in (1, batch), "dropout_keep batch mismatch")
            return keep.contiguous()
        source = bank.unit_mask if unit_mask is None else _as_numpy(unit_mask)
        keep = torch.from_numpy(np.ascontiguousarray(source, dtype=np.bool_))
        if keep.dim() == 1:
            keep = keep.unsqueeze(0).expand(batch, -1)
        keep = keep.contiguous().to(device=self.frontend.slots.device)
        if self.training and self.unit_dropout_p > 0.0:
            keep = whole_unit_dropout(keep, p=self.unit_dropout_p, generator=dropout_generator)
        plan.require(bool(keep.any(dim=1).all()), "unit mask became empty for some window")
        return keep

    # ------------------------------------------------------------- static fold
    def static_term(
        self,
        E0: torch.Tensor | np.ndarray,
        carrier: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Precomputed static part of the first token layer (SPD-A1 view).

        Returns ``[N, SET_DIM]`` float32: ``E0 @ W_e0^T + carrier @ W_carrier^T
        + b`` where the slices are column views of the single
        ``frontend.token_mlp.0`` Linear(70 -> 256) weight (the split no longer
        changes the parameter layout). ``forward_static_folded`` fed with this
        tensor matches ``forward`` within the P3 parity gate (FP32
        ``max|Δ| <= 1e-6``); the base forward keeps S1's single-GEMM
        arithmetic, so folding is a fast path, not a bitwise twin.
        """
        e0 = E0 if isinstance(E0, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(E0, dtype=np.float32))
        t4 = carrier if isinstance(carrier, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(carrier, dtype=np.float32))
        e0 = e0.to(device=self.frontend.first_token_weight().device, dtype=torch.float32).reshape(-1, self.e0_dim)
        t4 = t4.to(device=self.frontend.first_token_weight().device, dtype=torch.float32).reshape(-1, self.carrier_dim)
        weight = self.frontend.first_token_weight()
        w_e0 = weight[:, CONV_CHANNELS : CONV_CHANNELS + self.e0_dim]
        w_carrier = weight[:, CONV_CHANNELS + self.e0_dim :]
        return e0 @ w_e0.t() + t4 @ w_carrier.t() + self.frontend.first_token_bias()

    def bank_static_term(self, bank: TaskBank) -> torch.Tensor:
        """``static_term`` on a bank's frozen arrays (what ``forward`` uses)."""
        e0, carrier, _ = self._bank_arrays(bank)
        return self.static_term(e0, carrier)

    # ------------------------------------------------------------------ core
    def _frontend(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        keep: torch.Tensor,
        static: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, width, n_units = x.shape
        if static is None:
            e0, carrier, _ = self._bank_arrays(bank)
            e0 = e0.to(x.device)
            carrier = carrier.to(x.device)
            fused = self.frontend(x, e0, carrier, keep)
        else:
            local = self.frontend.local_conv(x)  # [B, L, N, 16]
            weight = self.frontend.first_token_weight()
            w_local = weight[:, :CONV_CHANNELS]
            pre = F.linear(local, w_local) + static.to(local.dtype)  # broadcast [N, SET_DIM]
            tokens = self.frontend.token_norm(self.frontend.token_mlp[2](F.gelu(pre)))
            slots = self.frontend.slot_norm(self.frontend.slots).view(1, 1, N_SLOTS, SET_DIM)
            slots = slots.expand(batch, width, N_SLOTS, SET_DIM)
            q = slots.reshape(batch * width, N_SLOTS, SET_DIM)
            k = tokens.reshape(batch * width, n_units, SET_DIM)
            pad = (~keep).unsqueeze(1).expand(batch, width, n_units).reshape(batch * width, n_units)
            attn_out, _ = self.frontend.mha(q, k, k, key_padding_mask=pad, need_weights=False)
            slots_out = q + attn_out
            slots_out = slots_out + self.frontend.slot_ffn(self.frontend.slot_ffn_norm(slots_out))
            fused_pre = slots_out.reshape(batch, width, N_SLOTS * SET_DIM)
            fused = self.frontend.slot_proj(fused_pre)
        return fused

    def _temporal(self, fused: torch.Tensor) -> torch.Tensor:
        pe = self.temporal.pe
        h = fused + pe[: fused.size(1)].unsqueeze(0).to(fused.dtype)
        for block in self.temporal.blocks:
            h = block(h)
        return h

    def _scores_from_static(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        unit_mask: torch.Tensor | None,
        static: torch.Tensor | None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._check_input(x)
        keep = self._expand_keep(bank, unit_mask, x.size(0), dropout_generator, dropout_keep)
        fused = self._frontend(x, bank, keep, static)
        hidden = self._temporal(fused)
        return self.readout(self.final_norm(hidden))  # [B, L, out]

    # ----------------------------------------------------------------- API
    def forward_scores(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        unit_mask: torch.Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-bin readout ``[B, L, out_dim]`` (used by causal_check).

        ``dropout_generator`` / ``dropout_keep`` (code item M) make
        training-mode whole-unit dropout reproducible; derive the generator
        per batch via :func:`unit_dropout_seed` or pass the drawn mask.
        """
        return self._scores_from_static(x, bank, unit_mask, None, dropout_generator, dropout_keep)

    def forward(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        unit_mask: torch.Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Last-bin readout ``[B, out_dim]``; the model never sees future bins.

        ``dropout_generator`` / ``dropout_keep`` (code item M) make
        training-mode whole-unit dropout reproducible; derive the generator
        per batch via :func:`unit_dropout_seed`. Ignored in eval mode (no
        dropout there).
        """
        return self._scores_from_static(x, bank, unit_mask, None, dropout_generator, dropout_keep)[:, -1, :]

    def forward_static_folded(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        static_term: torch.Tensor,
    ) -> torch.Tensor:
        """Last-bin readout with the static first-layer term supplied ahead.

        Parity caliber vs ``forward``: FP32 ``max|Δ| <= 1e-6`` (workorder P3
        gate). The base ``forward`` keeps S1's single token GEMM, so the
        split-GEMM folded path is a documented ~1e-7-class float neighbor,
        not a bitwise twin.
        """
        plan.require(
            static_term.shape == (self.units, SET_DIM),
            f"static_term must be [{self.units}, {SET_DIM}], got {tuple(static_term.shape)}",
        )
        return self._scores_from_static(x, bank, None, static_term.to(torch.float32))[:, -1, :]

    def causal_check(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        unit_mask: torch.Tensor | None = None,
        n_bins: int = 5,
        tolerance: float = CAUSAL_CHECK_TOLERANCE,
    ) -> dict[str, Any]:
        """Causality self-certification by future scrambling (runs in eval).

        Scrambles the trailing ``n_bins`` input bins (time-reversal) and
        requires every per-bin output strictly before the scrambled tail to
        match within ``tolerance`` (default
        :data:`CAUSAL_CHECK_TOLERANCE` = 1e-5) — in particular the readout
        at the last causal bin (index ``L - n_bins - 1``), the "last bin" of
        the causal region that scoring reads. A causal model gives future
        keys exactly-zero softmax weight, so the untouched bins carry no
        functional dependence on the tail; only float coupling remains: the
        ``F.conv1d`` primitive (route-B default conv path) bleeds ~1e-7
        roundoff across output positions (Phase 0 record), so the gate is a
        max-|delta| tolerance, not bit equality. Also requires the scrambled
        tail to actually change the tail readout (non-vacuous check).
        Raises BTransformerUnifiedError on violation; returns the report
        dict (bit-exactness flags are reported informationally).
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                x = self._check_input(x)
                length = x.size(1)
                plan.require(
                    n_bins >= 1 and length >= n_bins + 2,
                    f"need length >= n_bins + 2 for causal_check, got L={length}, n_bins={n_bins}",
                )
                y_full = self.forward_scores(x, bank, unit_mask)
                cut = length - n_bins
                scrambled = x.clone()
                scrambled[:, cut:] = torch.flip(x[:, cut:], dims=[1])
                y_scram = self.forward_scores(scrambled, bank, unit_mask)
                tail_bitexact = torch.equal(y_full[:, :cut, :], y_scram[:, :cut, :])
                last_causal_bitexact = torch.equal(y_full[:, cut - 1, :], y_scram[:, cut - 1, :])
                tail_max_delta = float((y_full[:, :cut, :] - y_scram[:, :cut, :]).abs().max())
                last_causal_max_delta = float(
                    (y_full[:, cut - 1, :] - y_scram[:, cut - 1, :]).abs().max()
                )
                tail_within = tail_max_delta <= float(tolerance)
                last_causal_within = last_causal_max_delta <= float(tolerance)
                perturb_effective = not torch.equal(y_full[:, -1, :], y_scram[:, -1, :])
            report: dict[str, Any] = {
                "passed": bool(tail_within and last_causal_within and perturb_effective),
                "length": int(length),
                "scrambled_bins": int(n_bins),
                "causal_region_end_index": int(cut - 1),
                "tolerance": float(tolerance),
                "tail_max_abs_delta": tail_max_delta,
                "tail_within_tolerance": bool(tail_within),
                "last_causal_bin_max_abs_delta": last_causal_max_delta,
                "last_causal_bin_within_tolerance": bool(last_causal_within),
                "tail_bitexact": bool(tail_bitexact),
                "last_causal_bin_bitexact": bool(last_causal_bitexact),
                "perturbation_effective": bool(perturb_effective),
            }
            plan.require(report["passed"], f"causality violation: {report}")
            return report
        finally:
            self.train(was_training)

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        return {name: param for name, param in self.named_parameters() if param.requires_grad}

    def assert_s1_state_dict_parity(self, reference_keys: list[str] | None = None) -> dict[str, Any]:
        """Assert the state-dict key set equals S1 SmallTransformerDecoder's.

        P1a prerequisite: ``initialize_decoder`` walks parameters in sorted
        name order with fixed RNG domains, so an identical (name, shape)
        sequence consumes the RNG identically. When the tfpd_exploration S1
        module is importable, the reference is built live and compared
        bitwise (values too); otherwise ``reference_keys`` (sorted key list)
        is compared by name only. Returns a report dict for receipts.
        """
        own = {k: tuple(v.shape) for k, v in self.state_dict().items()}
        report: dict[str, Any] = {
            "own_key_count": len(own),
            "own_param_count": int(sum(p.numel() for p in self.parameters())),
        }
        plan.require(
            report["own_param_count"] == S1_M2_PARAM_COUNT,
            f"param count {report['own_param_count']} != S1 blueprint {S1_M2_PARAM_COUNT}",
        )
        try:
            if _WORKSPACE_ROOT not in sys.path:
                sys.path.insert(0, _WORKSPACE_ROOT)
            from tfpd_exploration.src.m2_b_small_stability_v1.decoder import SmallTransformerDecoder

            s1 = SmallTransformerDecoder(seed=self.init_meta["seed"])
            s1_shapes = {k: tuple(v.shape) for k, v in s1.state_dict().items()}
            report["reference"] = "live SmallTransformerDecoder(seed=42)"
            missing = sorted(set(s1_shapes) - set(own))
            extra = sorted(set(own) - set(s1_shapes))
            plan.require(not missing and not extra, f"state-dict key drift missing={missing} extra={extra}")
            for key, shape in s1_shapes.items():
                plan.require(own[key] == shape, f"shape drift at {key}: {own[key]} != {shape}")
            with torch.no_grad():
                max_delta = max(
                    float((self.state_dict()[key].cpu().float() - s1.state_dict()[key].cpu().float()).abs().max())
                    for key in s1_shapes
                )
            report["value_parity"] = "bitwise" if max_delta == 0.0 else f"max|delta|={max_delta:.3e}"
            plan.require(max_delta == 0.0, f"init values differ from S1 (max|delta|={max_delta:.3e})")
        except ImportError:
            plan.require(reference_keys is not None, "S1 reference unavailable and no key list given")
            plan.require(sorted(own) == sorted(reference_keys), "state-dict key set differs from provided reference")
            report["reference"] = "provided key list"
        report["keys_match"] = True
        return report


__all__ = [
    "BTransformerUnifiedDecoder",
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
