"""Fixed-batch CPU inference runtime for trained RIFT models.

This module is deliberately separate from the correctness/reference adapter.
It caches immutable bank tensors at registration and owns preallocated raw
history; it never changes decoder weights or temporal semantics.
"""
from __future__ import annotations

from typing import Hashable, Sequence
import torch
from torch import Tensor
from btransform_unified_v1.bank import TaskBank
from .model import RiftDecoder
from .temporal import RiftTemporalState
from .cpu_temporal import CpuRiftTemporalRuntime


class CpuRiftRuntime:
    """One real observed bin per row, with fixed logical stream slots."""
    def __init__(
        self, decoder: RiftDecoder, banks: Sequence[TaskBank], stream_ids: Sequence[Hashable], *,
        temporal_backend: str = "state",
    ) -> None:
        if decoder.training: raise ValueError("CpuRiftRuntime requires decoder.eval()")
        if len(banks) < 1 or len(banks) != len(stream_ids) or len(set(stream_ids)) != len(stream_ids):
            raise ValueError("one unique stream_id and TaskBank per fixed row required")
        if temporal_backend not in ("state", "cached"):
            raise ValueError("temporal_backend must be 'state' or 'cached'")
        self.decoder, self.ids, self.banks = decoder, list(stream_ids), list(banks)
        self.temporal_backend = temporal_backend
        self.row = {key: index for index, key in enumerate(self.ids)}
        b = len(banks); d = next(decoder.parameters()).device
        decoder._normalise_banks(self.banks, b)
        self.e0 = torch.stack([torch.from_numpy(bank.E0) for bank in banks]).to(d, torch.float32)
        self.carrier = torch.stack([torch.from_numpy(bank.carrier) for bank in banks]).to(d, torch.float32)
        self.keep = torch.stack([torch.from_numpy(bank.unit_mask) for bank in banks]).to(d, torch.bool)
        # This runtime intentionally caches bank-derived tensors and temporal
        # state.  A parameter update would otherwise combine stale cached KV
        # with new projections/readout.  Match RiftStreamDecoder's explicit
        # invalidation contract rather than silently returning mixed results.
        self.weight_versions = tuple(parameter._version for parameter in decoder.parameters())
        if not bool(self.keep.any(dim=1).all()):
            raise ValueError("unit mask became empty for some batch row")
        self.raw4 = torch.zeros(b, 4, decoder.units, device=d)
        self.state: RiftTemporalState | None = (
            decoder.temporal.init_state(b, d, torch.float32) if temporal_backend == "state" else None
        )
        self.cached_temporal = (
            CpuRiftTemporalRuntime(decoder.temporal, b, d) if temporal_backend == "cached" else None
        )
        self.last: Tensor | None = None

    @torch.inference_mode()
    def advance(self, observed: Tensor, stream_ids: Sequence[Hashable] | None = None, *, valid_mask: Tensor | None = None) -> Tensor:
        if self.decoder.training: raise RuntimeError("decoder switched to train mode")
        if tuple(parameter._version for parameter in self.decoder.parameters()) != self.weight_versions:
            raise RuntimeError("decoder parameters changed after CpuRiftRuntime registration; create a new runtime")
        if observed.shape != (len(self.ids), self.decoder.units): raise ValueError("observed must match fixed [B,N]")
        if valid_mask is None:
            valid_mask = torch.ones(len(self.ids), device=observed.device, dtype=torch.bool)
        if valid_mask.dtype != torch.bool or valid_mask.shape != (len(self.ids),):
            raise ValueError("valid_mask must be bool [B]")
        if stream_ids is not None:
            if set(stream_ids) != set(self.ids) or len(stream_ids) != len(self.ids): raise ValueError("stream_ids must be a permutation of registered ids")
            take = torch.tensor([self.row[key] for key in stream_ids], device=observed.device)
            self.raw4 = self.raw4.index_select(0, take); self.e0=self.e0.index_select(0,take); self.carrier=self.carrier.index_select(0,take); self.keep=self.keep.index_select(0,take)
            if self.state is not None:
                self.state = self.decoder.temporal.select_rows(self.state, take)
            if self.cached_temporal is not None:
                self.cached_temporal.reorder(take)
            if self.last is not None: self.last = self.last.index_select(0, take)
            self.ids=list(stream_ids); self.row={k:i for i,k in enumerate(self.ids)}
        x = observed.to(self.raw4.device, torch.float32)
        valid = valid_mask.to(self.raw4.device)
        raw5 = torch.cat((self.raw4, x.unsqueeze(1)), dim=1)
        conv = self.decoder.frontend.local_conv
        flat=raw5.permute(0,2,1).reshape(x.shape[0]*self.decoder.units,1,5)
        local=conv.act(conv.conv(flat)).reshape(x.shape[0],self.decoder.units,16,1).permute(0,3,1,2)
        z=self.decoder._fuse_batched_local(local,self.e0,self.carrier,self.keep)[:,0]
        if self.cached_temporal is None:
            assert self.state is not None
            h, self.state = self.decoder.temporal.step(z, self.state, valid)
        else:
            h = self.cached_temporal.step(z, valid)
        self.raw4.copy_(torch.where(valid[:, None, None], raw5[:, 1:], self.raw4))
        score=self.decoder.readout(self.decoder.final_norm(h))
        self.last = score if self.last is None else torch.where(valid[:, None], score, self.last)
        return score

    @torch.inference_mode()
    def reset_rows(self, stream_ids: Sequence[Hashable]) -> None:
        rows=[self.row[key] for key in stream_ids]
        self.raw4[rows] = 0
        if self.state is not None:
            self.decoder.temporal.reset_rows(self.state,rows)
        if self.cached_temporal is not None:
            self.cached_temporal.reset_rows(rows)
        if self.last is not None:
            self.last[rows] = 0
