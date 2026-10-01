"""Stream-id keyed RIFT adapter with explicit cache invalidation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np
import torch
from torch import Tensor

from btransform_unified_v1.bank import TaskBank

from .model import RiftDecoder
from .temporal import RiftTemporalState


def _bank_key(bank: TaskBank, unit_mask: Tensor | None) -> str:
    digest = hashlib.sha256()
    for value in (bank.E0, bank.carrier, bank.unit_mask if unit_mask is None else unit_mask.detach().cpu().numpy()):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class _Stream:
    raw: list[Tensor] = field(default_factory=list)
    state: RiftTemporalState | None = None
    bank_key: str = ""
    weight_versions: tuple[int, ...] = ()
    last_prediction: Tensor | None = None


class RiftStreamDecoder:
    """Incremental decoder state is keyed solely by logical ``stream_id``.

    ``stream_step`` consumes one event once. ``predict`` is a pure read of the
    cached result, making repeated prediction calls idempotent.  This adapter
    deliberately keeps state per stream so batch-row reorder and independent
    reset cannot mix histories; bulk offline B8 inference belongs to
    :meth:`RiftDecoder.forward_scores`, which evaluates B8 as a true batch.
    """

    def __init__(self, decoder: RiftDecoder) -> None:
        if decoder.training:
            raise ValueError(
                "RiftStreamDecoder is inference-only; call decoder.eval() so whole-unit dropout "
                "cannot change within a cache lifecycle"
            )
        self.decoder = decoder
        self._streams: dict[Hashable, _Stream] = {}

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

    def stream_step(
        self, new_x: Tensor, bank: TaskBank | Sequence[TaskBank], stream_ids: Sequence[Hashable], unit_mask: Tensor | None = None,
        *, valid_mask: Tensor | None = None,
    ) -> Tensor:
        if self.decoder.training:
            raise RuntimeError("decoder was switched to train mode; streaming cache requires eval mode")
        if new_x.ndim != 2 or new_x.shape[1] != self.decoder.units or new_x.shape[0] != len(stream_ids):
            raise ValueError(f"new_x must be [B,{self.decoder.units}] and match stream_ids")
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream_ids must be unique within one stream_step")
        if valid_mask is None:
            valid_mask = torch.ones(new_x.shape[0], dtype=torch.bool, device=new_x.device)
        if valid_mask.dtype != torch.bool or valid_mask.shape != (new_x.shape[0],):
            raise ValueError("valid_mask must be bool [B]")
        versions = tuple(parameter._version for parameter in self.decoder.parameters())
        banks = self.decoder._normalise_banks(bank, new_x.shape[0])
        masks = self._row_masks(unit_mask, new_x.shape[0])
        streams: list[_Stream] = []
        for row, stream_id in enumerate(stream_ids):
            stream = self._streams.setdefault(stream_id, _Stream())
            key = _bank_key(banks[row], masks[row])
            if stream.state is not None and (stream.bank_key != key or stream.weight_versions != versions):
                raise RuntimeError("bank, unit mask, or weights changed; call invalidate/reset before advancing this stream")
            if stream.state is None:
                stream.bank_key, stream.weight_versions = key, versions
            raw = new_x[row]
            if bool(valid_mask[row]):
                stream.raw.append(raw)
                stream.raw = stream.raw[-5:]
            streams.append(stream)
        # One frontend call: each row's local k5 history produces precisely one
        # new stable token.  Per-row bank tensors are stacked by frontend_last.
        histories = torch.stack([
            torch.stack([torch.zeros_like(new_x[row]) for _ in range(5 - len(stream.raw))] + stream.raw)
            for row, stream in enumerate(streams)
        ])
        with torch.inference_mode():
            frontend_mask = None if all(mask is None for mask in masks) else torch.stack([
                torch.from_numpy(bank_row.unit_mask) if mask is None else mask
                for bank_row, mask in zip(banks, masks)
            ])
            z = self.decoder.frontend_last(histories, banks, frontend_mask)
            for stream in streams:
                if stream.state is None:
                    stream.state = self.decoder.temporal.init_state(1, z.device, z.dtype)
            packed = self._pack_states([stream.state for stream in streams])
            hidden, packed = self.decoder.temporal.step(z, packed, valid_mask)
            unpacked = [self.decoder.temporal.select_rows(packed, [row]) for row in range(len(streams))]
            score = self.decoder.readout(self.decoder.final_norm(hidden))
        results: list[Tensor] = []
        for row, stream in enumerate(streams):
            stream.state = unpacked[row]
            if bool(valid_mask[row]):
                stream.last_prediction = score[row]
            results.append(score[row])
        return torch.stack(results)

    def _row_masks(self, unit_mask: Tensor | None, batch: int) -> list[Tensor | None]:
        if unit_mask is None:
            return [None] * batch
        if unit_mask.ndim == 1:
            return [unit_mask] * batch
        if unit_mask.ndim != 2 or unit_mask.shape != (batch, self.decoder.units):
            raise ValueError("unit_mask must be [N] or [B,N]")
        return [unit_mask[row] for row in range(batch)]

    def _pack_states(self, states: list[RiftTemporalState | None]) -> RiftTemporalState:
        if any(state is None for state in states):
            raise RuntimeError("stream state initialization failed")
        rows = [state for state in states if state is not None]
        packed = self.decoder.temporal.init_state(len(rows), rows[0].device, rows[0].dtype)
        packed.keys = [[list(state.keys[layer][0]) for state in rows] for layer in range(self.decoder.temporal_config.layers)]
        packed.values = [[list(state.values[layer][0]) for state in rows] for layer in range(self.decoder.temporal_config.layers)]
        packed.positions = [state.positions[0] for state in rows]
        return packed


__all__ = ["RiftStreamDecoder"]
