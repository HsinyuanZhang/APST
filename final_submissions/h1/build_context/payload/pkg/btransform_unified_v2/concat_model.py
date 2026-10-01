"""Matched full-E0 concat interface for the frozen RIFT implementation."""
from __future__ import annotations
import torch
from torch import Tensor
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity
from .model import RiftDecoder

class RiftConcatDecoder(RiftDecoder):
    """RIFT with `[local16 | full E0_50 | carrier4]` tokens.

    It inherits all input checks, per-row banks, dropout, masks, and streaming
    geometry from RiftDecoder.  Only its frontend identity fusion changes.
    """
    def __init__(self, task: str = "m2", *, context_bins: int = 50, bias_mode: str = "recency", seed: int = 42) -> None:
        super().__init__(task, context_bins=context_bins, bias_mode=bias_mode, seed=seed, proj_dim=16)
        reference_frontend = self.frontend
        base = BTransformerUnifiedDecoderIdentity({"task": self.task, **self.geometry}, identity_mode="concat", seed=seed)
        if base.frontend.token_in != 16 + self.geometry["e0_dim"] + self.geometry["carrier_dim"]:
            raise RuntimeError("concat token width drift")
        base.temporal = torch.nn.Identity()
        concat = base.frontend
        ref = dict(reference_frontend.named_parameters())
        with torch.no_grad():
            for name, value in concat.named_parameters():
                if name in ref and value.shape == ref[name].shape:
                    value.copy_(ref[name])
            # Exact function-preserving expansion at init:
            # y = Wl*local + Wp*P(E0) + Wc*carrier + b
            # becomes y = [Wl | Wp@P | Wc] * [local | E0 | carrier] + b.
            old = reference_frontend.token_mlp[0]
            projection = reference_frontend.e0_proj
            if projection is None: raise RuntimeError("matched proj_add reference lacks e0_proj")
            folded = torch.cat((old.weight[:, :16], old.weight[:, :16] @ projection.weight, old.weight[:, 16:]), dim=1)
            if tuple(folded.shape) != tuple(concat.token_mlp[0].weight.shape): raise RuntimeError("concat fold shape drift")
            concat.token_mlp[0].weight.copy_(folded); concat.token_mlp[0].bias.copy_(old.bias)
        self.frontend = concat
        # Retain RiftDecoder's owner: it owns the active final norm/readout.
        # Replacing it with `base` would register a second, unused readout path.
        self._frontend_owner.frontend = concat

    def _fuse_batched_local(self, local: Tensor, e0: Tensor, carrier: Tensor, keep: Tensor) -> Tensor:
        batch, width, units, _ = local.shape
        if e0.shape != (batch, units, self.geometry["e0_dim"]) or carrier.shape != (batch, units, 4):
            raise ValueError("concat bank batch geometry mismatch")
        tokens = self.frontend.token_norm(self.frontend.token_mlp(torch.cat((
            local, e0.to(dtype=local.dtype).unsqueeze(1).expand(batch, width, units, -1),
            carrier.to(dtype=local.dtype).unsqueeze(1).expand(batch, width, units, -1)), dim=-1)))
        slots = self.frontend.slot_norm(self.frontend.slots).view(1, 1, 8, 256).expand(batch, width, 8, 256)
        q = slots.reshape(batch * width, 8, 256); k = tokens.reshape(batch * width, units, 256)
        pad = (~keep).unsqueeze(1).expand(batch, width, units).reshape(batch * width, units)
        attended, _ = self.frontend.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attended; slots_out = slots_out + self.frontend.slot_ffn(self.frontend.slot_ffn_norm(slots_out))
        return self.frontend.slot_proj(slots_out.reshape(batch, width, 8 * 256))
