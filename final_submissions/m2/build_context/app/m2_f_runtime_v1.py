#!/usr/bin/env python3
"""CPU runtime for M2 F+FiLM with reset-time banks and a contiguous KV ring.

Same payload contract as the admitted F+FiLM package. Streaming no longer
re-hashes banks or repacks Python K/V lists on every bin; E0/carrier/keep are
stacked once at reset, and learned-slope attention uses CpuLearnableRecencyRuntime.
Output EMA remains inference-only float64 after the native /5 unscale.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder
from a1_m2.nope_temporal_v1 import build_decoder
from btransform_unified_v1.bank import TaskBank, array_sha256
from learnable_recency_v1.cpu_temporal import CpuLearnableRecencyRuntime
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from pretrained_nope_learned.temporal_v1 import install_f_nope_learned

ALPHA = 1.0 / 3.0
WEIGHT_EMA_DECAY = 0.9995
SESSION_COUNT = 13
PAYLOAD_SCHEMA = "apst_m2_f_export_v1_payload"
DECODER_STATE_SCHEMA = "apst_m2_f_selected_ema_decoder_v1"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def typed(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.dtype.str.encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def doc(p: Path) -> dict[str, Any]:
    x = json.loads(p.read_text())
    if not isinstance(x, dict):
        raise ValueError("payload manifest is not an object")
    return x


class CachedFRingRuntime:
    """Fixed-row ring cache. Public method name matches A1M2StreamDecoder.stream_step."""

    def __init__(self, decoder, banks: Sequence[TaskBank]):
        if decoder.training:
            raise ValueError("CachedFRingRuntime requires decoder.eval()")
        core = decoder.temporal.core
        if decoder.temporal.spec.use_sinusoidal_pe or not isinstance(core, LearnableRecencyTemporal):
            raise TypeError("cached F runtime requires noPE LearnableRecencyTemporal core")
        if not banks:
            raise ValueError("cached F runtime requires at least one bank")
        self.decoder = decoder
        self.banks = list(banks)
        self._bank_ids = tuple(id(bank) for bank in self.banks)
        device = next(decoder.parameters()).device
        batch = len(self.banks)
        self.e0 = torch.stack([torch.from_numpy(np.ascontiguousarray(bank.E0)) for bank in self.banks]).to(
            device, torch.float32
        )
        self.carrier = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(bank.carrier)) for bank in self.banks]
        ).to(device, torch.float32)
        self.keep = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(bank.unit_mask)) for bank in self.banks]
        ).to(device, torch.bool)
        self.raw4 = torch.zeros(batch, 4, decoder.units, device=device, dtype=torch.float32)
        self.cached_temporal = CpuLearnableRecencyRuntime(core, batch, device)
        self.versions = tuple(parameter._version for parameter in decoder.parameters())

    @torch.inference_mode()
    def stream_step(self, new_x, bank, stream_ids, unit_mask=None, *, valid_mask=None):
        if self.decoder.training:
            raise RuntimeError("decoder was switched to train mode")
        if tuple(parameter._version for parameter in self.decoder.parameters()) != self.versions:
            raise RuntimeError("decoder parameters changed after runtime registration")
        banks = list(bank) if not isinstance(bank, TaskBank) else [bank] * new_x.shape[0]
        if tuple(id(item) for item in banks) != self._bank_ids:
            raise RuntimeError("bank objects changed; call reset before advancing this stream")
        if new_x.ndim != 2 or new_x.shape != (len(self.banks), self.decoder.units):
            raise ValueError("observed must match fixed [B,96]")
        if valid_mask is None:
            valid_mask = torch.ones(len(self.banks), dtype=torch.bool, device=new_x.device)
        if valid_mask.dtype != torch.bool or valid_mask.shape != (len(self.banks),):
            raise ValueError("valid_mask must be bool [B]")
        x = new_x.to(self.raw4.device, torch.float32)
        valid = valid_mask.to(self.raw4.device)
        raw5 = torch.cat((self.raw4, x.unsqueeze(1)), dim=1)
        conv = self.decoder.frontend.local_conv
        flat = raw5.permute(0, 2, 1).reshape(x.shape[0] * self.decoder.units, 1, 5)
        local = conv.act(conv.conv(flat)).reshape(x.shape[0], self.decoder.units, 16, 1).permute(0, 3, 1, 2)
        z = self.decoder._fuse_batched_local(local, self.e0, self.carrier, self.keep)[:, 0]
        hidden = self.cached_temporal.step(z, valid)
        self.raw4.copy_(torch.where(valid[:, None, None], raw5[:, 1:], self.raw4))
        return self.decoder.readout(self.decoder.final_norm(hidden))


class M2FNoPELearnedDecoder(BCIDecoder):
    alpha = ALPHA

    def __init__(self, task_config, model_path, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.root = Path(model_path).resolve()
        self.doc = doc(self.root / "payload_manifest.json")
        self._check_payload()
        status = self.doc.get("status")
        formal_status = "FORMAL_SELECTED_PREPARATION_NO_POST"
        if (
            self.doc.get("schema") != PAYLOAD_SCHEMA
            or self.doc.get("arm") != "F_nope_learned"
            or status not in ("PREPARATION_NOT_GLOBAL_SELECTION", "NONFORMAL_CPU2_SMOKE", formal_status)
        ):
            raise ValueError("F payload identity/status drift")
        self.formal_payload = status == formal_status
        final = self.doc.get("final_decoder", {})
        if self.formal_payload:
            selection = self.doc.get("selection")
            if (
                not isinstance(selection, dict)
                or selection.get("formal") is not True
                or not isinstance(selection.get("epoch"), int)
                or selection["epoch"] < 1
                or selection["epoch"] != final.get("checkpoint_epoch")
                or not isinstance(final.get("checkpoint_sha256"), str)
                or len(final["checkpoint_sha256"]) != 64
            ):
                raise ValueError("formal F selection/epoch/checkpoint metadata drift")
        state = torch.load(self.root / "selected_ema_decoder.pt", map_location="cpu", weights_only=False)
        if (
            state.get("schema") != DECODER_STATE_SCHEMA
            or state.get("weight_ema_decay") != WEIGHT_EMA_DECAY
            or state.get("output_ema_alpha") != ALPHA
        ):
            raise ValueError("selected F EMA metadata drift")
        if self.formal_payload and (
            state.get("decoder_checkpoint_sha256") != final["checkpoint_sha256"]
            or state.get("decoder_checkpoint_epoch") != final["checkpoint_epoch"]
        ):
            raise ValueError("formal F checkpoint state/manifest binding drift")
        self.model = build_decoder("E_nope_flat", seed=42, proj_dim=16)
        install_f_nope_learned(self.model, "m2")
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.eval()
        core = self.model.temporal.core
        if (
            self.model.temporal.spec.use_sinusoidal_pe
            or tuple(core.slope_log.shape) != (4, 8)
            or int((core.recency_slopes > 0).sum()) != 6
        ):
            raise ValueError("loaded model is not F noPE learned-slope [4,8]/six-active-heads")
        rows = self.doc.get("sessions", {})
        if not isinstance(rows, dict) or len(rows) != SESSION_COUNT:
            raise ValueError("expected 13 legal M33 banks")
        self.banks = {}
        self.ids = []
        self.runtime = None
        self.ema = None
        self.initialized = None
        for sid, row in sorted(rows.items()):
            self._bank(sid, row)

    def _check_payload(self):
        expected = self.doc.get("payload_files_sha256")
        actual = {
            str(p.relative_to(self.root)): sha(p)
            for p in self.root.rglob("*")
            if p.is_file() and p.name != "payload_manifest.json"
        }
        if not isinstance(expected, dict) or actual != expected:
            raise ValueError("payload closure/hash ledger drift")

    def _bank(self, sid: str, row: dict[str, Any]):
        e = self.root / "calibration" / sid / "E0.npy"
        c = self.root / "calibration" / sid / "carrier.npy"
        if sha(e) != row.get("payload_e0_file_sha256") or sha(c) != row.get("payload_carrier_file_sha256"):
            raise ValueError(sid + ": bank hash drift")
        e0 = np.load(e, mmap_mode="r", allow_pickle=False)
        carrier = np.load(c, mmap_mode="r", allow_pickle=False)
        if (
            e0.shape != (96, 50)
            or carrier.shape != (96, 4)
            or e0.dtype != np.float32
            or carrier.dtype != np.float32
            or typed(e0) != row.get("payload_e0_typed_sha256")
            or typed(carrier) != row.get("payload_carrier_typed_sha256")
        ):
            raise ValueError(sid + ": bank geometry/typed hash drift")
        tag = self.task_config.hash_dataset(Path(str(row["official_tag_stem"])).stem)
        if tag != row.get("official_dataset_tag_hash") or tag in self.banks:
            raise ValueError(sid + ": dataset tag drift")
        self.banks[tag] = TaskBank(
            sid,
            np.asarray(e0),
            np.asarray(carrier),
            np.ones(96, dtype=np.bool_),
            np.zeros((0, 1, 96), np.float32),
            np.zeros((0, 2), np.float32),
            np.zeros(0, np.int64),
            {
                "shape": [96, 50],
                "trial_count": 33,
                "estimator": "frozen_pretrained_A_E0_carrier",
                "array_sha256": array_sha256(np.asarray(e0)),
                "budget": 33,
            },
        )

    def reset(self, dataset_tags=(Path(""),)):
        tags = [self.task_config.hash_dataset(Path(x).stem) for x in dataset_tags]
        if not tags or len(set(tags)) != len(tags) or any(x not in self.banks for x in tags) or len(tags) > self.batch_size:
            raise ValueError("unknown/duplicate/oversize reset roster")
        self.ids = tags
        self.runtime = CachedFRingRuntime(self.model, [self.banks[k] for k in tags])
        self.ema = None
        self.initialized = None

    def observe(self, neural_observations):
        return None

    def on_done(self, dones):
        return None

    def predict(self, neural_observations):
        if self.runtime is None:
            raise RuntimeError("reset must precede predict")
        x = np.asarray(neural_observations, dtype=np.float32)
        n = x.shape[0] if x.ndim == 2 else 0
        if x.ndim != 2 or x.shape[1:] != (96,) or not 1 <= n <= len(self.ids) or not np.isfinite(x).all():
            raise ValueError("observations must be finite [1..batch,96]")
        padded = np.zeros((len(self.ids), 96), np.float32)
        padded[:n] = x
        valid = torch.zeros(len(self.ids), dtype=torch.bool)
        valid[:n] = True
        with torch.no_grad():
            score = self.runtime.stream_step(
                torch.from_numpy(padded), [self.banks[k] for k in self.ids], self.ids, valid_mask=valid
            )
        native = score.detach().cpu().numpy().astype(np.float64, copy=False) / 5.0
        if self.ema is None:
            self.ema = np.zeros_like(native, dtype=np.float64)
            self.initialized = np.zeros(len(self.ids), dtype=np.bool_)
        active = valid.cpu().numpy() & self.initialized
        fresh = valid.cpu().numpy() & ~self.initialized
        self.ema[active] = ALPHA * self.ema[active] + (1 - ALPHA) * native[active]
        self.ema[fresh] = native[fresh]
        self.initialized |= valid.cpu().numpy()
        return self.ema[:n].copy()


M2FSharedFiLMDecoder = M2FNoPELearnedDecoder
__all__ = ["M2FNoPELearnedDecoder", "M2FSharedFiLMDecoder", "ALPHA"]
