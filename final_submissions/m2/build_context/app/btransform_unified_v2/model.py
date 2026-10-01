"""RIFT decoder: stable k=5 frontend tokens followed by :mod:`temporal`.

The frontend is deliberately constructed from the accepted v1 ``proj_add``
implementation.  Only its local/set computation is reused: RIFT does *not*
reuse v1's positional temporal stack or represent an old checkpoint as an
equivalent RIFT model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from btransform_unified_v1.bank import TaskBank
from btransform_unified_v1.model import whole_unit_dropout
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity

from .config import RiftTemporalConfig
from .temporal import RiftTemporal


TASK_GEOMETRIES: dict[str, dict[str, int]] = {
    "h1": {"window": 300, "prefix": 0, "units": 176, "e0_dim": 700, "carrier_dim": 4, "out_dim": 7},
    "m1": {"window": 100, "prefix": 0, "units": 64, "e0_dim": 100, "carrier_dim": 4, "out_dim": 16},
    "m2": {"window": 50, "prefix": 0, "units": 96, "e0_dim": 50, "carrier_dim": 4, "out_dim": 2},
}


def _geometry(task: str | Mapping[str, Any], context_bins: int) -> tuple[str, dict[str, int]]:
    if isinstance(task, str):
        if task not in TASK_GEOMETRIES:
            raise ValueError(f"unknown RIFT task {task!r}; use h1, m1, m2, or an explicit geometry")
        return task, {**TASK_GEOMETRIES[task], "window": context_bins}
    required = ("units", "e0_dim", "carrier_dim", "out_dim")
    missing = [key for key in required if key not in task]
    if missing:
        raise ValueError(f"RIFT geometry missing {missing}")
    geometry = {key: int(value) for key, value in task.items() if key in (*required, "window", "prefix")}
    geometry["window"] = context_bins
    geometry.setdefault("prefix", 0)
    return str(task.get("task", "adhoc")), geometry


class RiftDecoder(nn.Module):
    """D4/256 RIFT decoder with stable per-event frontend tokens.

    ``input_valid_mask`` identifies real raw observations, not scoring labels.
    It may only be left padding: false entries produce no temporal KV evidence.
    A real all-zero spike bin must therefore be marked true.
    """

    def __init__(
        self,
        task: str | Mapping[str, Any],
        context_bins: int | None = None,
        bias_mode: str = "recency",
        seed: int = 42,
        proj_dim: int = 16,
    ) -> None:
        super().__init__()
        default_context = TASK_GEOMETRIES[task]["window"] if isinstance(task, str) and task in TASK_GEOMETRIES else 300
        context_bins = default_context if context_bins is None else context_bins
        if not isinstance(context_bins, int) or context_bins < 5:
            raise ValueError("context_bins must be an integer >= frontend kernel size 5")
        self.task, self.geometry = _geometry(task, context_bins)
        self.context_bins = context_bins
        self.units = self.geometry["units"]
        self.out_dim = self.geometry["out_dim"]
        # This initializes the v1-defined frontend/readout reproducibly.  Its
        # old temporal module is discarded immediately; it is never executed.
        base = BTransformerUnifiedDecoderIdentity(
            {"task": self.task, **self.geometry}, identity_mode="proj_add", proj_dim=proj_dim, seed=seed
        )
        self.frontend = base.frontend
        self.final_norm = base.final_norm
        self.readout = base.readout
        self._frontend_owner = base
        # Do not register the obsolete temporal parameters in this model.
        self._frontend_owner.temporal = nn.Identity()
        temporal_config = RiftTemporalConfig.for_context(context_bins, bias_mode=bias_mode)  # type: ignore[arg-type]
        self.temporal = RiftTemporal(temporal_config)
        self.temporal_config = temporal_config
        self.bias_mode = bias_mode
        self.seed = int(seed)
        self.proj_dim = int(proj_dim)
        # Reinitialize RIFT-only weights under a local seed.  bias mode is
        # intentionally absent from this RNG domain, yielding byte-identical
        # shared parameters for recency and flat initializations.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 0x52494654)
            for module in self.temporal.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def _check_input(self, x: Tensor, input_valid_mask: Tensor | None) -> Tensor:
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != self.units:
            raise ValueError(f"x must be nonempty [B,T,{self.units}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values")
        if input_valid_mask is not None:
            if input_valid_mask.dtype != torch.bool or input_valid_mask.shape != x.shape[:2]:
                raise ValueError("input_valid_mask must be bool [B,T]")
            # Internal holes have no defined raw-history meaning; consumers
            # must split streams rather than silently turn missing bins to zero.
            if (input_valid_mask[:, :-1] & ~input_valid_mask[:, 1:]).any():
                raise ValueError("input_valid_mask may only contain left padding")
        return x.to(dtype=torch.float32) if x.dtype != torch.float32 else x

    def frontend_tokens(
        self, x: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None, dropout_keep: Tensor | None = None,
    ) -> Tensor:
        """Compute stable tokens for an entire chronological raw sequence.

        The v1 local convolution pads only before the real sequence.  Thus
        each returned token is exactly ``F_last(x[t-4:t])`` and is never
        redrawn when a later query arrives.
        """
        x = self._check_input(x, None)
        banks = self._normalise_banks(bank, x.shape[0])
        keep = self._resolve_keep(banks, unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        e0 = torch.stack([torch.from_numpy(value.E0) for value in banks]).to(x.device, dtype=torch.float32)
        carrier = torch.stack([torch.from_numpy(value.carrier) for value in banks]).to(x.device, dtype=torch.float32)
        return self._batched_proj_add_frontend(x, e0, carrier, keep)

    def _batched_proj_add_frontend(self, x: Tensor, e0: Tensor, carrier: Tensor, keep: Tensor) -> Tensor:
        """The v1 proj_add frontend generalized from static [N,*] to [B,N,*]."""
        local = self.frontend.local_conv(x)
        return self._fuse_batched_local(local, e0, carrier, keep)

    def _fuse_batched_local(self, local: Tensor, e0: Tensor, carrier: Tensor, keep: Tensor) -> Tensor:
        """Apply proj_add/token/set fusion to ``local [B,T,N,16]`` exactly once."""
        batch, width, units, _ = local.shape
        projection = self.frontend.e0_proj
        if projection is None:
            raise RuntimeError("RIFT requires the v1 proj_add frontend")
        proj = projection(e0.to(dtype=local.dtype))
        groups = proj.shape[-1] // 16
        fused_local = (local.unsqueeze(-2) + proj.view(batch, 1, units, groups, 16)).reshape(batch, width, units, -1)
        tokens = self.frontend.token_norm(self.frontend.token_mlp(torch.cat([
            fused_local, carrier.to(dtype=local.dtype).unsqueeze(1).expand(batch, width, units, -1)
        ], dim=-1)))
        slots = self.frontend.slot_norm(self.frontend.slots).view(1, 1, 8, 256).expand(batch, width, 8, 256)
        q = slots.reshape(batch * width, 8, 256)
        k = tokens.reshape(batch * width, units, 256)
        pad = (~keep).unsqueeze(1).expand(batch, width, units).reshape(batch * width, units)
        attended, _ = self.frontend.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attended
        slots_out = slots_out + self.frontend.slot_ffn(self.frontend.slot_ffn_norm(slots_out))
        return self.frontend.slot_proj(slots_out.reshape(batch, width, 8 * 256))

    def frontend_last(
        self, raw5: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask: Tensor | None = None,
    ) -> Tensor:
        """Vectorized ``F_last`` for exactly five raw bins, one token per row."""
        if raw5.ndim != 3 or raw5.shape[1:] != (5, self.units):
            raise ValueError(f"raw5 must be [B,5,{self.units}]")
        raw5 = self._check_input(raw5, None)
        batch = raw5.shape[0]
        banks = self._normalise_banks(bank, batch)
        keep = self._resolve_keep(banks, unit_mask, batch, raw5.device, None, None)
        e0 = torch.stack([torch.from_numpy(value.E0) for value in banks]).to(raw5.device, dtype=torch.float32)
        carrier = torch.stack([torch.from_numpy(value.carrier) for value in banks]).to(raw5.device, dtype=torch.float32)
        # Unlike SharedCausalConv.forward this uses no left padding: exactly
        # five real/reset-filled raw points yield one final causal convolution
        # output.  All downstream set work therefore has T=1, not T=5.
        flat = raw5.permute(0, 2, 1).reshape(batch * self.units, 1, 5)
        conv = self.frontend.local_conv
        local = conv.act(conv.conv(flat)).reshape(batch, self.units, 16, 1).permute(0, 3, 1, 2)
        return self._fuse_batched_local(local, e0, carrier, keep)[:, 0]

    def _normalise_banks(self, bank: TaskBank | Sequence[TaskBank], batch: int) -> list[TaskBank]:
        banks = [bank] * batch if isinstance(bank, TaskBank) else list(bank)
        if len(banks) != batch or not all(isinstance(value, TaskBank) for value in banks):
            raise ValueError("bank must be a TaskBank or a sequence of one TaskBank per batch row")
        for value in banks:
            if value.E0.shape != (self.units, self.geometry["e0_dim"]) or value.carrier.shape != (self.units, 4):
                raise ValueError("bank geometry does not match this decoder")
        return banks

    def _resolve_keep(self, banks: list[TaskBank], unit_mask: Tensor | None, batch: int, device: torch.device,
                      dropout_generator: torch.Generator | None, dropout_keep: Tensor | None) -> Tensor:
        if dropout_keep is not None:
            keep = dropout_keep.to(device=device, dtype=torch.bool)
            if keep.ndim == 1:
                keep = keep.unsqueeze(0)
            if keep.shape[1] != self.units or keep.shape[0] not in (1, batch):
                raise ValueError("dropout_keep must be [N] or [B,N]")
            return keep.expand(batch, -1).contiguous() if keep.shape[0] == 1 else keep.contiguous()
        if unit_mask is None:
            keep = torch.stack([torch.from_numpy(value.unit_mask) for value in banks])
        else:
            keep = unit_mask.to(dtype=torch.bool)
            if keep.ndim == 1:
                keep = keep.unsqueeze(0).expand(batch, -1)
            if keep.shape != (batch, self.units):
                raise ValueError("unit_mask must be [N] or [B,N]")
        keep = keep.to(device=device).contiguous()
        if self.training and self._frontend_owner.unit_dropout_p > 0:
            keep = whole_unit_dropout(keep, self._frontend_owner.unit_dropout_p, dropout_generator)
        if not bool(keep.any(dim=1).all()):
            raise ValueError("unit mask became empty for some batch row")
        return keep

    def forward_scores(
        self, x: Tensor, bank: TaskBank, unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None, dropout_keep: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
    ) -> Tensor:
        x = self._check_input(x, input_valid_mask)
        # Padding payload is not an observation.  It must be neutralized before
        # the causal k5 convolution as well as masked out of temporal KV.
        # This preserves the true-reset raw-zero boundary without treating
        # padding itself as an event/token.
        if input_valid_mask is not None:
            x = torch.where(input_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        z = self.frontend_tokens(x, bank, unit_mask, dropout_generator, dropout_keep)
        hidden = self.temporal(z, input_valid_mask)
        return self.readout(self.final_norm(hidden))

    def forward(
        self, x: Tensor, bank: TaskBank, unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None, dropout_keep: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
    ) -> Tensor:
        return self.forward_scores(x, bank, unit_mask, dropout_generator, dropout_keep, input_valid_mask)[:, -1]


__all__ = ["RiftDecoder", "TASK_GEOMETRIES"]
