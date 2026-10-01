"""Three-arm M2 model for the sealed cross-session calibration ablation.

The arms deliberately share the complete query frontend, RIFT decoder, seed,
and source sampling protocol.  They differ only in static session information:
``Z_NONE`` receives neither calibration-derived identity nor carrier; ``B``
receives a live M33 activity identity with all carrier values zero; ``D``
receives the same live identity plus the real T4 carrier through both existing
production paths (FiLM side features and decoder-token carrier features).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from btransform_unified_v1.bank import TaskBank
from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder

from .model import RiftDecoder

ARM_Z = "Z_NONE"
ARM_B = "B_ACTIVITY_ONLY"
ARM_D = "D_JOINT"
ARMS = (ARM_Z, ARM_B, ARM_D)


class CrossSessionM2Decoder(RiftDecoder):
    """M2 RIFT decoder whose calibration inputs are explicit arm switches."""

    def __init__(self, arm: str, *, seed: int, encoder_init: str = "random") -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown cross-session arm {arm!r}")
        super().__init__("m2", context_bins=50, bias_mode="recency", seed=seed, proj_dim=16)
        self.arm = arm
        self.identity_enabled = arm in (ARM_B, ARM_D)
        self.carrier_to_film = arm == ARM_D
        self.carrier_to_decoder = arm == ARM_D
        if encoder_init != "random":
            raise ValueError("this sealed protocol permits only encoder_init=random")
        self.encoder_init = encoder_init
        # Z must not instantiate an unused calibration encoder merely to appear
        # parameter-matched.  A/B/D decoder initialization is already identical
        # because RiftDecoder initializes its shared decoder under local RNG.
        if self.identity_enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed) + 0x4D324353)
                self.encoder = HoldContrastFiLMEarlyPoolEncoder(
                    100, 50, 64, side_dim=8, film_rank=8, num_post_layers=3,
                    film_input="t4_plus_contrast",
                )
        self._calib: dict[str, Tensor] = {}
        self._carrier: dict[str, Tensor] = {}

    def install_session_memory(self, banks: Mapping[str, TaskBank], cache_root: Path) -> None:
        """Install only the static source/target-session inputs required by this arm."""
        for session, bank in banks.items():
            key = session.replace("-", "_")
            if self.identity_enabled:
                raw = np.load(cache_root / session / "calib_activity.npy").astype(np.float32, copy=False)
                if raw.shape != (33, 100, 96):
                    raise ValueError(f"raw M33 geometry drift {session}: {raw.shape}")
                if not hasattr(self, f"calib_{key}"):
                    self.register_buffer(f"calib_{key}", torch.from_numpy(np.ascontiguousarray(raw)), persistent=False)
                elif not np.array_equal(getattr(self, f"calib_{key}").cpu().numpy(), raw):
                    raise ValueError(f"cross-surface calibration drift {session}")
                self._calib[session] = getattr(self, f"calib_{key}")
            if self.carrier_to_film or self.carrier_to_decoder:
                # The shared query-only TaskBank deliberately contains zeros.
                # D alone opens the actual source/target M33 carrier artifact.
                carrier = np.ascontiguousarray(
                    np.load(cache_root / session / "T.npy"), dtype=np.float32
                )
                if carrier.shape != (96, 4):
                    raise ValueError(f"carrier geometry drift {session}: {carrier.shape}")
                if not hasattr(self, f"carrier_{key}"):
                    self.register_buffer(f"carrier_{key}", torch.from_numpy(carrier), persistent=False)
                elif not np.array_equal(getattr(self, f"carrier_{key}").cpu().numpy(), carrier):
                    raise ValueError(f"cross-surface carrier drift {session}")
                self._carrier[session] = getattr(self, f"carrier_{key}")

    def _identity(self, sessions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        encoded: dict[str, Tensor] = {}
        carriers: dict[str, Tensor] = {}
        with torch.autocast(device_type=device.type, enabled=False):
            for session in dict.fromkeys(sessions):
                carrier = (self._carrier[session].to(device=device, dtype=torch.float32)
                           if (self.carrier_to_film or self.carrier_to_decoder) else
                           torch.zeros((96, 4), device=device, dtype=torch.float32))
                if not self.identity_enabled:
                    encoded[session] = torch.zeros((96, 50), device=device, dtype=torch.float32)
                else:
                    raw = self._calib[session].to(device=device, dtype=torch.float32)
                    side = torch.zeros((96, 8), device=device, dtype=torch.float32)
                    if self.carrier_to_film:
                        side[:, :4] = carrier
                    state = self.encoder.reset_stream(1, 96, device, torch.float32)
                    state["side_features"] = side.unsqueeze(0)
                    for trial in raw:
                        state = self.encoder.push_trial(state, trial)
                    encoded[session] = self.encoder.finalize_identity(state).squeeze(0)
                carriers[session] = carrier if self.carrier_to_decoder else torch.zeros_like(carrier)
        return torch.stack([encoded[s] for s in sessions]), torch.stack([carriers[s] for s in sessions])

    def frontend_tokens(self, x: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask=None,
                        dropout_generator=None, dropout_keep=None) -> Tensor:
        x = self._check_input(x, None)
        banks = self._normalise_banks(bank, x.shape[0])
        keep = self._resolve_keep(banks, unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        e0, carrier = self._identity([b.session_id for b in banks], x.device)
        return self._batched_proj_add_frontend(x, e0, carrier, keep)

    def frontend_last(self, raw5: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask=None) -> Tensor:
        if raw5.ndim != 3 or raw5.shape[1:] != (5, self.units):
            raise ValueError(f"raw5 must be [B,5,{self.units}]")
        raw5 = self._check_input(raw5, None)
        banks = self._normalise_banks(bank, raw5.shape[0])
        keep = self._resolve_keep(banks, unit_mask, raw5.shape[0], raw5.device, None, None)
        e0, carrier = self._identity([b.session_id for b in banks], raw5.device)
        flat = raw5.permute(0, 2, 1).reshape(raw5.shape[0] * self.units, 1, 5)
        conv = self.frontend.local_conv
        local = conv.act(conv.conv(flat)).reshape(raw5.shape[0], self.units, 16, 1).permute(0, 3, 1, 2)
        return self._fuse_batched_local(local, e0, carrier, keep)[:, 0]


__all__ = ["CrossSessionM2Decoder", "ARM_Z", "ARM_B", "ARM_D", "ARMS"]
