"""Calibration-free static frontend for the learnable-recency RIFT decoder.

This variant keeps the accepted P16 local/set frontend and RIFT temporal
stack, but learns one identity vector per recorded unit.  It deliberately has
no :class:`TaskBank`, E0 projection, or source carrier dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

import torch
from torch import Tensor, nn

from btransform_unified_v1.model import whole_unit_dropout

from .config import LearnableRecencyConfig
from .temporal import LearnableRecencyTemporal
from .wrap import LearnableRiftDecoder


STATIC_IDENTITY_SEED_OFFSET = 0x53544154


class StaticLearnableRiftDecoder(LearnableRiftDecoder):
    """P16 M1/M2/H1 decoder trained entirely from raw observations.

    ``static_identity[N,16]`` replaces ``frontend.e0_proj(E0)`` and the
    carrier is a persistent literal-zero ``[N,4]`` buffer.  All other
    frontend, temporal, norm, and readout parameters use the full decoder's
    initialization path for the same seed.
    """

    def __init__(
        self,
        task: str,
        recency_cfg: LearnableRecencyConfig,
        *,
        context_bins: int | None = None,
        seed: int = 42,
    ) -> None:
        if task not in ("m1", "m2", "h1"):
            raise ValueError("StaticLearnableRiftDecoder supports only M1, M2, and H1")
        super().__init__(
            task,
            recency_cfg,
            context_bins=context_bins if context_bins is not None else recency_cfg.context_bins,
            seed=seed,
            proj_dim=16,
        )
        # `frontend` is also owned by `_frontend_owner`; replace the module in
        # place so no aliased owner state retains a trainable E0 projection.
        self.frontend.e0_proj = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + STATIC_IDENTITY_SEED_OFFSET)
            identity = torch.empty(self.units, 16, dtype=torch.float32)
            nn.init.normal_(identity, mean=0.0, std=0.02)
        self.static_identity = nn.Parameter(identity)
        self.register_buffer("static_carrier", torch.zeros(self.units, 4), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        carrier_key = prefix + "static_carrier"
        carrier = state_dict.get(carrier_key)
        if carrier is not None and bool(torch.count_nonzero(carrier)):
            error_msgs.append(f"{carrier_key} must remain the literal zero static carrier")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _resolve_keep(
        self,
        unit_mask: Tensor | None,
        batch: int,
        device: torch.device,
        dropout_generator: torch.Generator | None,
        dropout_keep: Tensor | None,
    ) -> Tensor:
        if unit_mask is None:
            base = torch.ones(batch, self.units, dtype=torch.bool, device=device)
        else:
            keep = unit_mask.to(device=device, dtype=torch.bool)
            if keep.ndim == 1:
                keep = keep.unsqueeze(0).expand(batch, -1)
            if keep.shape != (batch, self.units):
                raise ValueError("unit_mask must be [N] or [B,N]")
            base = keep
        base = base.contiguous()
        if not bool(base.any(dim=1).all()):
            raise ValueError("unit mask is empty for some batch row")
        if dropout_keep is not None:
            keep = dropout_keep.to(device=device, dtype=torch.bool)
            if keep.ndim == 1:
                keep = keep.unsqueeze(0)
            if keep.ndim != 2 or keep.shape[1] != self.units or keep.shape[0] not in (1, batch):
                raise ValueError("dropout_keep must be [N] or [B,N]")
            keep = keep.expand(batch, -1) if keep.shape[0] == 1 else keep
            # Callers may sample whole-unit dropout outside the model.  It is
            # final here: never redraw it, and never let it revive invalid
            # channels from the base unit mask.
            keep = base & keep
        elif self.training and self._frontend_owner.unit_dropout_p > 0:
            keep = whole_unit_dropout(base, self._frontend_owner.unit_dropout_p, dropout_generator)
        else:
            keep = base
        keep = keep.contiguous()
        if not bool(keep.any(dim=1).all()):
            raise ValueError("unit mask became empty for some batch row")
        return keep

    def _fuse_static_local(self, local: Tensor, keep: Tensor) -> Tensor:
        batch, width, units, channels = local.shape
        if units != self.units or channels != 16:
            raise RuntimeError("static P16 frontend received incompatible local features")
        identity = self.static_identity.to(device=local.device, dtype=local.dtype)
        fused = local + identity.view(1, 1, units, 16)
        carrier = self.static_carrier.to(device=local.device, dtype=local.dtype)
        tokens = self.frontend.token_norm(self.frontend.token_mlp(torch.cat([
            fused, carrier.view(1, 1, units, 4).expand(batch, width, units, 4)
        ], dim=-1)))
        slots = self.frontend.slot_norm(self.frontend.slots).view(1, 1, 8, 256).expand(batch, width, 8, 256)
        q = slots.reshape(batch * width, 8, 256)
        k = tokens.reshape(batch * width, units, 256)
        pad = (~keep).unsqueeze(1).expand(batch, width, units).reshape(batch * width, units)
        attended, _ = self.frontend.mha(q, k, k, key_padding_mask=pad, need_weights=False)
        slots_out = q + attended
        slots_out = slots_out + self.frontend.slot_ffn(self.frontend.slot_ffn_norm(slots_out))
        return self.frontend.slot_proj(slots_out.reshape(batch, width, 8 * 256))

    def frontend_tokens(
        self,
        x: Tensor,
        unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: Tensor | None = None,
    ) -> Tensor:
        x = self._check_input(x, None)
        keep = self._resolve_keep(unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        return self._fuse_static_local(self.frontend.local_conv(x), keep)

    def frontend_last(self, raw5: Tensor, unit_mask: Tensor | None = None) -> Tensor:
        if raw5.ndim != 3 or raw5.shape[1:] != (5, self.units):
            raise ValueError(f"raw5 must be [B,5,{self.units}]")
        raw5 = self._check_input(raw5, None)
        batch = raw5.shape[0]
        keep = self._resolve_keep(unit_mask, batch, raw5.device, None, None)
        flat = raw5.permute(0, 2, 1).reshape(batch * self.units, 1, 5)
        conv = self.frontend.local_conv
        local = conv.act(conv.conv(flat)).reshape(batch, self.units, 16, 1).permute(0, 3, 1, 2)
        return self._fuse_static_local(local, keep)[:, 0]

    def forward_scores(
        self,
        x: Tensor,
        unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
    ) -> Tensor:
        x = self._check_input(x, input_valid_mask)
        if input_valid_mask is not None:
            x = torch.where(input_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        z = self.frontend_tokens(x, unit_mask, dropout_generator, dropout_keep)
        hidden = self.temporal(z, input_valid_mask)
        return self.readout(self.final_norm(hidden))

    def forward(
        self,
        x: Tensor,
        unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
    ) -> Tensor:
        return self.forward_scores(x, unit_mask, dropout_generator, dropout_keep, input_valid_mask)[:, -1]


@dataclass
class _StaticStream:
    raw: list[Tensor] = field(default_factory=list)
    state: object | None = None
    unit_mask: Tensor | None = None
    weight_versions: tuple[int, ...] = ()
    last_prediction: Tensor | None = None


class StaticRiftStreamDecoder:
    """Inference-only, calibration-free stream adapter for static decoders."""

    def __init__(self, model: StaticLearnableRiftDecoder) -> None:
        if model.training:
            raise ValueError("StaticRiftStreamDecoder is inference-only; call model.eval()")
        self.decoder = model
        self._streams: dict[Hashable, _StaticStream] = {}

    def reset(self, stream_ids: Hashable | Sequence[Hashable] | None = None) -> None:
        ids = list(self._streams) if stream_ids is None else ([stream_ids] if not isinstance(stream_ids, (list, tuple)) else stream_ids)
        for stream_id in ids:
            self._streams.pop(stream_id, None)

    invalidate = reset

    def predict(self, stream_id: Hashable) -> Tensor:
        stream = self._streams.get(stream_id)
        if stream is None or stream.last_prediction is None:
            raise KeyError(f"stream {stream_id!r} has no consumed observation")
        return stream.last_prediction

    def _row_masks(self, unit_mask: Tensor | None, batch: int) -> list[Tensor | None]:
        if unit_mask is None:
            return [None] * batch
        if unit_mask.ndim == 1:
            return [unit_mask] * batch
        if unit_mask.ndim != 2 or unit_mask.shape != (batch, self.decoder.units):
            raise ValueError("unit_mask must be [N] or [B,N]")
        return [unit_mask[row] for row in range(batch)]

    def _pack_states(self, states: list[object]) -> object:
        temporal = self.decoder.temporal
        rows = states
        packed = temporal.init_state(len(rows), rows[0].device, rows[0].dtype)
        packed.keys = [[list(state.keys[layer][0]) for state in rows] for layer in range(self.decoder.temporal_config.layers)]
        packed.values = [[list(state.values[layer][0]) for state in rows] for layer in range(self.decoder.temporal_config.layers)]
        packed.positions = [state.positions[0] for state in rows]
        if isinstance(temporal, LearnableRecencyTemporal):
            packed.increments = [[list(state.increments[layer][0]) for state in rows] for layer in range(self.decoder.temporal_config.layers)]
        return packed

    def stream_step(
        self,
        new_x: Tensor,
        stream_ids: Sequence[Hashable],
        unit_mask: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.decoder.training:
            raise RuntimeError("decoder was switched to train mode; streaming cache requires eval mode")
        if new_x.ndim != 2 or new_x.shape != (len(stream_ids), self.decoder.units):
            raise ValueError(f"new_x must be [B,{self.decoder.units}] and match stream_ids")
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream_ids must be unique within one stream_step")
        if valid_mask is None:
            valid_mask = torch.ones(new_x.shape[0], dtype=torch.bool, device=new_x.device)
        if valid_mask.dtype != torch.bool or valid_mask.shape != (new_x.shape[0],):
            raise ValueError("valid_mask must be bool [B]")
        versions = tuple(parameter._version for parameter in self.decoder.parameters())
        masks = self._row_masks(unit_mask, new_x.shape[0])
        streams: list[_StaticStream] = []
        for row, stream_id in enumerate(stream_ids):
            stream = self._streams.setdefault(stream_id, _StaticStream())
            mask = None if masks[row] is None else masks[row].detach().to(dtype=torch.bool, device="cpu").clone()
            if stream.state is not None and (stream.weight_versions != versions or not _same_mask(stream.unit_mask, mask)):
                raise RuntimeError("unit mask or weights changed; call invalidate/reset before advancing this stream")
            if stream.state is None:
                stream.unit_mask, stream.weight_versions = mask, versions
            if bool(valid_mask[row]):
                # A caller commonly reuses a staging tensor for the next bin;
                # the causal five-bin history must own its old sample.
                stream.raw = (stream.raw + [new_x[row].detach().clone()])[-5:]
            streams.append(stream)
        histories = torch.stack([torch.stack([torch.zeros_like(new_x[row]) for _ in range(5 - len(stream.raw))] + stream.raw) for row, stream in enumerate(streams)])
        with torch.inference_mode():
            frontend_mask = None if all(mask is None for mask in masks) else torch.stack([
                torch.ones(self.decoder.units, dtype=torch.bool, device=new_x.device) if mask is None else mask.to(new_x.device)
                for mask in masks
            ])
            z = self.decoder.frontend_last(histories, frontend_mask)
            for stream in streams:
                if stream.state is None:
                    stream.state = self.decoder.temporal.init_state(1, z.device, z.dtype)
            packed = self._pack_states([stream.state for stream in streams])
            hidden, packed = self.decoder.temporal.step(z, packed, valid_mask)
            unpacked = [self.decoder.temporal.select_rows(packed, [row]) for row in range(len(streams))]
            score = self.decoder.readout(self.decoder.final_norm(hidden))
        for row, stream in enumerate(streams):
            stream.state = unpacked[row]
            if bool(valid_mask[row]):
                stream.last_prediction = score[row]
        return score


def _same_mask(left: Tensor | None, right: Tensor | None) -> bool:
    return left is None and right is None or left is not None and right is not None and torch.equal(left, right)


__all__ = ["StaticLearnableRiftDecoder", "StaticRiftStreamDecoder"]
