"""Identity-usage variants of BTransformerUnifiedDecoder (H1 matrix, NOT SPINT).

Matrix doc: ``docs/MATRIX_H1_L_IDENTITY_V1_20260906.md`` (workorder §6 five-arm
control generalized to L in {250, 350}). ``model.py`` stays byte-stable
(S1-parity discipline); the variants live here as a subclass so the m2/P1a
build is untouched.

Six ``identity_mode`` values (matrix letters; workorder §6 + M-F250):

  concat (a)  status quo: bank E0 [N, d_e] statically concatenated into every
              token (token_in = 16 + d_e + 4; d_e = 700 for the H1 matrix).
  joined (b)  bank E0 becomes the TRUE 36-d C2 pre-pool joined representation
              (carrier_pre_pool 32 + H-C 4; d_e = 36, token_in = 16+36+4 = 56
              — dynamic per mode, unlike m2's 70 or m1's 120).
  add_tail(c) SPINT-style: bank E0 [N, 700] is a TIME axis end-aligned with the
              input; the last min(L, 700) = L columns of E0 are ADDED onto the
              last L bins of X (X then flows through the original causal conv);
              no E0 enters the token concat (d_e = 0, token_in = 20).
  zero   (d)  bank E0 zeroed at forward time (carrier only) — identity total-
              contribution control. Same parameterization as concat.
  permute(e)  bank E0 columns reversed at forward time — structure-break
              control. Same parameterization as concat.
  proj_add(f) M-F250 USER PROPOSAL (2026-09-06): learnable projection
              P = Linear(d_e -> R, bias=False) applied to the bank E0; P(E0)
              is ADDED channel-wise onto the local conv channels before the
              token MLP: tokens = token_mlp([local16 + P(E0)] | carrier4).
              R (``proj_dim``) defaults to 16 (the frozen
              ``h1_config.PROJ_ADD_OUT_DIM``); R must be a multiple of the 16
              local channels and the bracket generalizes by BROADCAST: local16
              is duplicated once per 16-d group of P(E0) and each group is
              added onto its copy — bracket width R, token_in = R + 4
              (R=16 -> token_in 20, bitwise the original single-group add).
              Mathematically a RANK-R BOTTLENECK version of the concat first
              layer (E0's influence on token_mlp.0 is rank-<=R by
              construction); L-independent (no 700-bin template alignment
              question). P(E0) is session-static, so the SPD-A1 static fold
              applies (:meth:`static_term` folds sum_g P_g @ W_g^T + carrier
              @ W_carrier^T + b). With P zeroed the mode degenerates exactly
              to the no-E0 local+carrier pathway (zero-mode semantics at the
              R+4-wide parameterization).

Shared invariants per mode: forward shapes are unchanged ([B, out_dim] /
forward_scores [B, L, out_dim]); causality self-certification
(:meth:`BTransformerUnifiedDecoder.causal_check`) keeps working (the add_tail
modification is applied to the input bins themselves before the causal conv,
so future-scrambling still cannot leak into earlier readouts); init semantics
are the same S1-source ``initialize_decoder`` walk (parameter COUNT varies
with the mode's token width and is recorded in ``init_meta``); input length
stays pinned to ``l_in`` (code item N).

``override_window`` (new here) is the matrix L-axis control: it may only
SHORTEN the settled H1 full window 700 (matrix L in {250, 350}) and is refused
for any other task window — the same "explicit opt-in, never quietly settled"
discipline as ``override_prefix`` resolving the ``H1_PREFIX_PENDING`` sentinel.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import h1_config, plan
from .bank import TaskBank
from .model import (
    CONV_CHANNELS,
    N_HEADS,
    N_LAYERS,
    N_SLOTS,
    READOUT_HIDDEN,
    SET_DIM,
    SLOT_FFN_DIM,
    TEMPORAL_WIDTH,
    BTransformerUnifiedDecoder,
    CausalTransformerBlock,
    CausalTransformerStack,
    SharedCausalConv,
    SharedSetFrontend,
)


class CausalTransformerStackDepth(CausalTransformerStack):
    """Depth-axis control: ``n_layers`` temporal blocks instead of N_LAYERS=4.

    Workorder ADDENDUM-DEPTH2-PROMOTED (2026-09-07, M1 proj_add runtime/quality
    §7 priority plan promoted to a formal front-row cell): temporal depth
    4 -> 2 keeps P16, W=100, 8 slots, the frontend, and the in-window
    sinusoidal PE verbatim; ONLY the block count changes. The public default
    stays ``N_LAYERS`` everywhere (``model.py`` is byte-stable; a plain
    ``CausalTransformerStack`` build never sees this class).
    """

    def __init__(self, max_len: int, n_layers: int) -> None:
        super().__init__(max_len)  # PE buffer + default N_LAYERS blocks
        plan.require(
            isinstance(n_layers, int) and not isinstance(n_layers, bool) and n_layers >= 1,
            f"temporal depth must be an int >= 1, got {n_layers!r}",
        )
        self.n_layers = int(n_layers)
        if self.n_layers != N_LAYERS:
            # rebuilt BEFORE _init_parameters runs, so the S1-source sorted-name
            # RNG walk only ever sees the final module tree; the discarded
            # default blocks never receive init values.
            self.blocks = nn.ModuleList(CausalTransformerBlock() for _ in range(self.n_layers))


def token_e0_dim(base_e0_dim: int, identity_mode: str, proj_dim: int | None = None) -> int:
    """Token-side identity width for a mode given the geometry's bank width.

    proj_add generalization (M1 proj_add series + M2 P32 retry, 2026-09-06):
    the projection width may be raised above the settled
    ``PROJ_ADD_OUT_DIM`` (M1 runs P in {16, 32}; M2 keeps the default 16).
    The identity rides the LOCAL block additively, so the token-side width is
    the part of the first token layer beyond the 16 local channels:
    ``max(0, proj_dim - 16)`` — 0 for the settled P=16 builds, 16 for P=32
    (the bracket ``[local+P_1 ; ... ; local+P_G]`` spans R = proj_dim
    columns; group semantics in :meth:`SharedSetFrontendIdentity.forward`).
    The constructor passes no ``proj_dim`` here (it recomputes token_in
    directly as ``proj_out_dim + carrier_dim``), so this parameter exists for
    width arithmetic callers and checks only.
    """
    mode = h1_config.normalize_identity_mode(identity_mode)
    if mode == "joined":
        plan.require(
            int(base_e0_dim) == h1_config.JOINED_DIM,
            f"joined mode requires the true {h1_config.JOINED_DIM}-d C2 pre-pool "
            f"joined bank (geometry e0_dim {base_e0_dim} != {h1_config.JOINED_DIM}); "
            "the five-arm PROXY (first 32 dims of fused E0) is NOT the real object",
        )
        return h1_config.JOINED_DIM
    if mode == "proj_add":
        width = h1_config.PROJ_ADD_OUT_DIM if proj_dim is None else int(proj_dim)
        plan.require(
            width >= CONV_CHANNELS,
            f"proj_add requires proj_dim >= CONV_CHANNELS ({CONV_CHANNELS}) so "
            f"the local conv block is never truncated, got {width}",
        )
        return width - CONV_CHANNELS
    if mode == "add_tail":
        return 0  # identity enters the input bins, not the concat
    return int(base_e0_dim)


def transform_token_e0(e0: torch.Tensor, identity_mode: str) -> torch.Tensor:
    """Apply the mode's token-side transform to a bank E0 tensor [N, d_e]."""
    mode = h1_config.normalize_identity_mode(identity_mode)
    if mode in ("concat", "joined", "proj_add"):
        return e0  # proj_add: the frontend/static paths project P(e0) themselves
    if mode == "zero":
        return torch.zeros_like(e0)
    if mode == "permute":
        return torch.flip(e0, dims=[-1]).contiguous()
    # add_tail: identity never enters the token concat — return an empty
    # [N, 0] tensor so the concatenation width arithmetic stays uniform.
    return e0[:, :0].contiguous()


def apply_add_tail(x: torch.Tensor, e0: torch.Tensor) -> torch.Tensor:
    """SPINT-style end-aligned additive identity (matrix mode (c)).

    ``x`` [B, L, N] observation bins; ``e0`` [N, E] identity TIME axis. The
    last ``min(L, E)`` columns of E0 (E = 700, L = 250/350 -> exactly L) are
    aligned to the input END and added bin-wise onto X:

        X'[:, t, n] = X[:, t, n] + E0[n, E - L + t]   for t in [0, L)

    Same rule as ``scripts/h1_sec6_fivearm_l100.py::_prepare_x``
    (``x + e0[:, -WINDOW:].T[None]``), generalized to any L.
    """
    plan.require(x.dim() == 3 and e0.dim() == 2, "add_tail needs x [B,L,N], e0 [N,E]")
    plan.require(x.size(2) == e0.size(0), "add_tail unit mismatch between x and e0")
    length = x.size(1)
    tail = e0[:, -length:].to(dtype=x.dtype, device=x.device)  # [N, min(L, E)]
    return x + tail.t().unsqueeze(0)  # broadcast [1, L, N]


class SharedSetFrontendIdentity(SharedSetFrontend):
    """SharedSetFrontend with the token-width floor relaxed for e0-free modes.

    Identical module attribute names/layout to the pinned S1 blueprint (so the
    ``initialize_decoder`` sorted-name walk is unchanged); the ONLY structural
    differences are:

      - the constructor gate is ``token_in >= CONV_CHANNELS + 4`` instead of a
        strict ``>`` — add_tail/proj_add have d_e = 0, so their token_in is
        exactly 16 + 0 + 4 = 20;
      - an optional ``e0_proj`` submodule (proj_add mode, M-F250): a learnable
        P = Linear(d_e -> R, bias=False). When present, :meth:`forward` fuses
        the identity into the LOCAL channels additively — each 16-d group of
        P(E0) is added onto the local conv channels (broadcast over groups) —
        instead of concatenating E0: ``[local+P_1 ; ... ; local+P_G] | carrier``.
    """

    def __init__(self, token_in: int) -> None:
        nn.Module.__init__(self)  # skip the parent's strict-width gate only
        self.token_in = int(token_in)
        plan.require(
            self.token_in >= CONV_CHANNELS + 4,
            "token_in must cover the local conv (16) + carrier (4) widths",
        )
        self.e0_proj: nn.Linear | None = None  # set by the proj_add build
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

    def forward(self, x: torch.Tensor, e0: torch.Tensor, t4: torch.Tensor, unit_keep: torch.Tensor) -> torch.Tensor:
        if self.e0_proj is None:
            return super().forward(x, e0, t4, unit_keep)
        # proj_add (M-F250): tokens = token_mlp([local16 + P(E0)] | carrier4).
        batch, width, n_units = x.shape
        plan.require(
            e0.dim() == 2 and e0.size(0) == n_units,
            f"proj_add needs the static bank E0 [N, d_e], got {tuple(e0.shape)}",
        )
        local = self.local_conv(x)  # [B, L, N, 16]
        proj = self.e0_proj(e0.to(device=x.device, dtype=local.dtype))  # [N, R]
        n_groups = proj.shape[-1] // CONV_CHANNELS
        plan.require(
            proj.shape[-1] == n_groups * CONV_CHANNELS,
            f"proj_add projection width {proj.shape[-1]} must be a multiple "
            f"of the {CONV_CHANNELS} local conv channels",
        )
        # each 16-d group of P(E0) is ADDED channel-wise onto the local conv
        # channels (broadcast over the group axis); R=16 collapses to the
        # original single-group add with identical arithmetic.
        fused_local = (
            local.unsqueeze(-2) + proj.view(1, 1, n_units, n_groups, CONV_CHANNELS)
        ).reshape(batch, width, n_units, n_groups * CONV_CHANNELS)
        if t4.dim() == 2:
            t4 = t4.view(1, 1, n_units, -1).expand(batch, width, n_units, -1)
        else:
            t4 = t4.unsqueeze(1).expand(batch, width, n_units, -1)
        tokens = self.token_mlp(torch.cat([fused_local, t4], dim=-1))
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


class BTransformerUnifiedDecoderIdentity(BTransformerUnifiedDecoder):
    """BTransformerUnifiedDecoder + ``identity_mode`` + ``override_window``.

    ``identity_mode`` defaults to ``"concat"`` (backward compatible: with a
    settled geometry the state-dict key set, shapes, and bit-values of the
    init are IDENTICAL to the parent build — ``token_e0_width`` equals the
    geometry d_e and the frontend module tree matches name-for-name).

    ``override_window`` is the matrix L axis: it may only shorten the settled
    H1 full window (700 -> 250/350) and is refused otherwise; the resolved
    window lands in ``self.window`` / ``self.l_in`` / the PE length, and the
    original settled value stays visible through ``self.window_override`` /
    ``self.geometry``.

    ``temporal_layers`` is the depth axis (workorder ADDENDUM-DEPTH2-PROMOTED,
    2026-09-07): ``None`` (default) keeps the frozen ``N_LAYERS`` = 4 build
    bit-identical (same class, same module tree, same init walk); an explicit
    int != N_LAYERS swaps in :class:`CausalTransformerStackDepth`. It is NOT a
    weight-equivalent transform — a different depth is a new architecture that
    must be retrained from scratch (§7: never truncate e24 layers and call it
    a formal quality number). Recorded in ``init_meta["temporal_layers"]``.
    """

    def __init__(
        self,
        task: str | Mapping[str, Any],
        seed: int = plan.SEED,
        override_prefix: int | None = None,
        identity_mode: str = h1_config.IDENTITY_DEFAULT,
        override_window: int | None = None,
        proj_dim: int | None = None,
        temporal_layers: int | None = None,
    ) -> None:
        nn.Module.__init__(self)  # parent __init__ is intentionally not run
        if isinstance(task, str):
            self.task = task
            geometry = plan.task_geometry(task)
        else:
            self.task = str(task.get("task", "adhoc-geometry"))
            geometry = dict(task)
            for key in ("window", "prefix", "units", "e0_dim", "carrier_dim", "out_dim"):
                plan.require(key in geometry, f"ad-hoc geometry missing key {key!r}")
        self.identity_mode = h1_config.normalize_identity_mode(identity_mode)

        settled_window = int(geometry["window"])
        if override_window is not None:
            plan.require(
                isinstance(override_window, int)
                and not isinstance(override_window, bool)
                and override_window >= 1,
                "override_window must be a positive int (matrix L axis)",
            )
            plan.require(
                settled_window == h1_config.FULL_WINDOW,
                "override_window only shortens the settled H1 full window "
                f"{h1_config.FULL_WINDOW} (matrix L-sweep {h1_config.MATRIX_L}); "
                f"task {self.task!r} window {settled_window} is frozen",
            )
            plan.require(
                override_window < settled_window,
                f"override_window must SHORTEN the window ({settled_window}); "
                "build with the settled window itself instead",
            )
            window = int(override_window)
        else:
            window = settled_window
        self.window_override: int | None = None if override_window is None else window

        self.geometry = dict(geometry)
        self.geometry["window"] = window
        self.window = window
        self.units = int(geometry["units"])
        self.base_e0_dim = plan.resolved_e0_dim(geometry)  # bank-side width
        raw_prefix = geometry["prefix"]
        self.prefix = plan.resolved_prefix(geometry, override_prefix)
        self.pending_prefix = raw_prefix if isinstance(raw_prefix, str) else None
        self.carrier_dim = int(geometry["carrier_dim"])
        self.out_dim = int(geometry["out_dim"])
        self.l_in = self.window + self.prefix
        self.token_e0_width = token_e0_dim(self.base_e0_dim, self.identity_mode)
        # proj_add width axis (P16 -> P32 retry, user directive 2026-09-06):
        # R defaults to the frozen h1_config.PROJ_ADD_OUT_DIM (16) so every
        # pre-existing build is unchanged; R must be a multiple of the 16
        # local conv channels (P(E0) groups are ADDED onto local copies) and
        # token_in = R + carrier4.
        self.proj_out_dim: int | None = None
        self.proj_groups = 0
        self.proj_dim_override: int | None = None if proj_dim is None else int(proj_dim)
        if self.identity_mode == "proj_add":
            self.proj_out_dim = h1_config.PROJ_ADD_OUT_DIM if proj_dim is None else int(proj_dim)
            plan.require(
                self.proj_out_dim % CONV_CHANNELS == 0,
                f"proj_dim {self.proj_out_dim} must be a multiple of the "
                f"{CONV_CHANNELS} local conv channels (grouped channel-wise add)",
            )
            self.proj_groups = self.proj_out_dim // CONV_CHANNELS
        # inherited utilities (static_term / _check_input) read self.e0_dim as
        # the TOKEN-side width; the bank-side width stays on self.base_e0_dim.
        self.e0_dim = self.token_e0_width
        self.token_in = CONV_CHANNELS + self.token_e0_width + self.carrier_dim
        if self.identity_mode == "proj_add":
            self.token_in = int(self.proj_out_dim) + self.carrier_dim
        plan.require(self.window >= 1, "window must be >= 1")
        plan.require(self.prefix >= 0, "prefix must be >= 0")
        plan.require(self.units >= 1, "units must be >= 1")
        plan.require(self.carrier_dim == 4, "carrier_dim is 4 for every task in this series")
        self.unit_dropout_p = float(plan.UNIT_DROPOUT)

        # --- frontend (names pinned; width gate relaxed for e0-free modes) ----
        self.frontend = SharedSetFrontendIdentity(self.token_in)
        if self.identity_mode == "proj_add":
            # M-F250 user proposal: P = Linear(d_e -> R, bias=False). The
            # bias choice is frozen here and recorded in init_meta.
            self.frontend.e0_proj = nn.Linear(
                self.base_e0_dim, int(self.proj_out_dim), bias=False
            )
        # --- readout head (names pinned: final_norm / readout) ----------------
        self.final_norm = nn.LayerNorm(TEMPORAL_WIDTH)
        self.readout = nn.Sequential(
            nn.Linear(TEMPORAL_WIDTH, READOUT_HIDDEN),
            nn.GELU(),
            nn.Linear(READOUT_HIDDEN, self.out_dim),
        )
        # --- temporal core (names pinned: temporal.blocks.*, CausalPE4) -------
        # depth axis: None -> the frozen N_LAYERS build, byte-identical path.
        if temporal_layers is not None:
            plan.require(
                isinstance(temporal_layers, int)
                and not isinstance(temporal_layers, bool)
                and temporal_layers >= 1,
                f"temporal_layers must be an int >= 1, got {temporal_layers!r}",
            )
        self.temporal_layers: int = N_LAYERS if temporal_layers is None else int(temporal_layers)
        if self.temporal_layers == N_LAYERS:
            self.temporal = CausalTransformerStack(self.l_in)
        else:
            self.temporal = CausalTransformerStackDepth(self.l_in, self.temporal_layers)
        meta: dict[str, Any] = self._init_parameters(seed)
        meta.update(
            {
                "identity_mode": self.identity_mode,
                "matrix_letter": h1_config.IDENTITY_MODE_TO_MATRIX_LETTER[self.identity_mode],
                "base_e0_dim": int(self.base_e0_dim),
                "token_e0_width": int(self.token_e0_width),
                "token_in": int(self.token_in),
                "window_override": self.window_override,
                "temporal_layers": int(self.temporal_layers),
                "param_count": int(sum(p.numel() for p in self.parameters())),
            }
        )
        if self.identity_mode == "proj_add":
            meta.update(
                {
                    "proj_bias": False,
                    "proj_in_dim": int(self.base_e0_dim),
                    "proj_out_dim": int(self.proj_out_dim),
                    "proj_groups": int(self.proj_groups),
                    "fused_local_width": int(self.proj_out_dim),
                    "proj_dim_override": self.proj_dim_override,
                    "proj_param_count": int(self.frontend.e0_proj.weight.numel()),
                }
            )
        self.init_meta = meta

    # ------------------------------------------------------------ bank utils
    def _bank_arrays(self, bank: TaskBank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = torch.from_numpy(np.ascontiguousarray(bank.E0, dtype=np.float32))
        carrier = torch.from_numpy(np.ascontiguousarray(bank.carrier, dtype=np.float32))
        mask = torch.from_numpy(np.ascontiguousarray(bank.unit_mask, dtype=np.bool_))
        expected = (
            h1_config.JOINED_DIM
            if self.identity_mode == "joined"
            else self.base_e0_dim
        )
        plan.require(
            e0.shape == (self.units, expected),
            f"bank E0 {tuple(e0.shape)} != model {(self.units, expected)} "
            f"(identity_mode={self.identity_mode!r})",
        )
        plan.require(carrier.shape == (self.units, self.carrier_dim), "bank carrier shape mismatch")
        plan.require(mask.shape == (self.units,), "bank unit_mask shape mismatch")
        return e0, carrier, mask

    def _token_e0(self, e0: torch.Tensor) -> torch.Tensor:
        return transform_token_e0(e0, self.identity_mode)

    # ------------------------------------------------------------- frontend
    def _frontend(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        keep: torch.Tensor,
        static: torch.Tensor | None,
    ) -> torch.Tensor:
        if static is not None:
            if self.identity_mode == "proj_add" and self.proj_groups > 1:
                # the parent fold hardcodes the single 16-d local block; the
                # multi-group bracket needs the summed per-group local fold.
                return self._frontend_folded_multigroup(x, keep, static)
            # the folded path never reads the bank E0 (the static term was
            # precomputed from the mode-transformed E0 by bank_static_term).
            return super()._frontend(x, bank, keep, static)
        e0, carrier, _ = self._bank_arrays(bank)
        carrier = carrier.to(x.device)
        if self.identity_mode == "proj_add":
            # the frontend projects P(E0) itself and fuses it into the local
            # channels; pass the RAW bank E0.
            return self.frontend(x, e0.to(x.device), carrier, keep)
        e0 = self._token_e0(e0).to(x.device)
        return self.frontend(x, e0, carrier, keep)

    # ------------------------------------------------------------------ core
    def _frontend_folded_multigroup(
        self,
        x: torch.Tensor,
        keep: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        """SPD-A1 folded fast path for the multi-group proj_add bracket.

        bracket = ``[local+P_1 ; ... ; local+P_G]`` -> first-layer
        preactivation ``sum_g local @ W_g^T + static`` where ``static``
        already carries ``sum_g P_g @ W_g^T + carrier @ W_c^T + b``.
        FP32 fast path, parity-gated at 1e-6 exactly like the parent fold
        (folding is a fast path, never a bitwise twin).
        """
        batch, width, n_units = x.shape
        local = self.frontend.local_conv(x)  # [B, L, N, 16]
        weight = self.frontend.first_token_weight()
        w_groups = weight[:, : int(self.proj_out_dim)].reshape(
            SET_DIM, self.proj_groups, CONV_CHANNELS
        )
        w_eff = w_groups.sum(dim=1)  # [SET_DIM, 16]
        pre = F.linear(local, w_eff) + static.to(local.dtype)  # broadcast [N, SET_DIM]
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
        return self.frontend.slot_proj(fused_pre)

    def _scores_from_static(
        self,
        x: torch.Tensor,
        bank: TaskBank,
        unit_mask: torch.Tensor | None,
        static: torch.Tensor | None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.identity_mode == "add_tail":
            e0, _, _ = self._bank_arrays(bank)
            x = apply_add_tail(x, e0.to(dtype=x.dtype))
        return super()._scores_from_static(
            x, bank, unit_mask, static, dropout_generator, dropout_keep
        )

    # --------------------------------------------------------- static fold
    def static_term(
        self,
        E0: torch.Tensor | np.ndarray,
        carrier: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Precomputed static part of the first token layer for the mode.

        concat/joined/zero/permute/add_tail: parent semantics (the mode's
        token-side E0 @ W_e0^T + carrier @ W_carrier^T + b). proj_add
        (M-F250): the first layer input is ``[local16 + P(E0)] | carrier`` —
        the E0 influence rides the LOCAL columns, so the static term is the
        rank-R-bottleneck fold ``sum_g P_g @ W_g^T + carrier @ W_carrier^T +
        b`` [N, SET_DIM] (session-static; same SPD-A1 style as the parent),
        where P_g / W_g are the 16-d group slices of P(E0) and of the first
        token layer's local columns.
        """
        if self.identity_mode != "proj_add":
            return super().static_term(E0, carrier)
        plan.require(
            self.frontend.e0_proj is not None, "proj_add requires frontend.e0_proj"
        )
        e0 = E0 if isinstance(E0, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(E0, dtype=np.float32))
        t4 = carrier if isinstance(carrier, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(carrier, dtype=np.float32))
        device = self.frontend.first_token_weight().device
        e0 = e0.to(device=device, dtype=torch.float32).reshape(-1, self.base_e0_dim)
        t4 = t4.to(device=device, dtype=torch.float32).reshape(-1, self.carrier_dim)
        weight = self.frontend.first_token_weight()
        # first-layer input layout is [fused_local(R) | carrier(4)]: the
        # carrier columns are the LAST 4 for every R (R=16 -> cols 16:20).
        w_carrier = weight[:, int(self.proj_out_dim):]
        proj = self.frontend.e0_proj(e0)  # [N, R]
        if self.proj_groups == 1:
            w_local = weight[:, :CONV_CHANNELS]
            return proj @ w_local.t() + t4 @ w_carrier.t() + self.frontend.first_token_bias()
        # multi-group bracket: the E0-dependent static part sums the per-group
        # projected slices against their matching local-column slices.
        w_groups = weight[:, : int(self.proj_out_dim)].reshape(
            SET_DIM, self.proj_groups, CONV_CHANNELS
        )
        proj_groups = proj.reshape(proj.shape[0], self.proj_groups, CONV_CHANNELS)
        static_e0 = torch.einsum("ngc,ogc->no", proj_groups, w_groups)
        return static_e0 + t4 @ w_carrier.t() + self.frontend.first_token_bias()

    def bank_static_term(self, bank: TaskBank) -> torch.Tensor:
        """``static_term`` over the mode-TRANSFORMED bank arrays."""
        e0, carrier, _ = self._bank_arrays(bank)
        return self.static_term(self._token_e0(e0), carrier)


__all__ = [
    "BTransformerUnifiedDecoderIdentity",
    "CausalTransformerStackDepth",
    "SharedSetFrontendIdentity",
    "token_e0_dim",
    "transform_token_e0",
    "apply_add_tail",
]
