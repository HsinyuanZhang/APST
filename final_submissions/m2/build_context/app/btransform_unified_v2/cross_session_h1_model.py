"""Fresh H1 C2-shaped three-arm RIFT decoder for date-LODO ablation.

Z has no calibration encoder or calibration tensors. B and D each instantiate
the same fresh C2 encoder under the same local RNG domain; B consumes activity
only, while D consumes activity and the source-fitted H-C carrier.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import hashlib
import torch
from torch import Tensor, nn
from btransform_unified_v1.bank import TaskBank
from .model import RiftDecoder

ARM_Z = "Z_NONE"
ARM_B = "B_ACTIVITY_ONLY"
ARM_D = "D_JOINT"
ARMS = (ARM_Z, ARM_B, ARM_D)


class RandomC2Identity(nn.Module):
    """Fresh random C2-shaped M3 activity/carrier identity encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(1024, 32), nn.ReLU())
        self.post = nn.Sequential(
            nn.Linear(36, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 700),
        )

    def forward(self, activity: Tensor, carrier: Tensor) -> Tensor:
        if activity.ndim != 4 or activity.shape[1:] != (3, 1024, 176):
            raise ValueError(f"activity must be [B,3,1024,176], got {tuple(activity.shape)}")
        if carrier.shape != (activity.shape[0], 176, 4):
            raise ValueError(f"carrier must be [B,176,4], got {tuple(carrier.shape)}")
        # C2 is deliberately FP32, including under the decoder's bf16 trunk.
        device_type = activity.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pooled = self.pre(activity.float().permute(0, 1, 3, 2)).mean(1)
            return self.post(torch.cat((pooled, carrier.float()), dim=-1))


class CrossSessionH1Decoder(RiftDecoder):
    """Matched Z/B/D H1 decoder.  Calibration is fixed memory, never labels."""
    def __init__(self, arm: str, *, seed: int) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        super().__init__("h1", context_bins=300, bias_mode="recency", seed=seed, proj_dim=16)
        self.arm = arm
        self.identity_enabled = arm in (ARM_B, ARM_D)
        self.carrier_enabled = arm == ARM_D
        # Z physically has no encoder and cannot accidentally consume M3.
        # B/D create their matching fresh C2 encoder in an isolated RNG domain.
        if self.identity_enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 0x48314332)
                self.encoder = RandomC2Identity()
        self._activity: dict[str, Tensor] = {}
        self._carrier: dict[str, Tensor] = {}

    @staticmethod
    def _buffer_name(prefix: str, session: str) -> str:
        digest = hashlib.sha256(session.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def install_memory(
        self, banks: Mapping[str, TaskBank], activity: Mapping[str, Tensor] | None = None,
    ) -> None:
        """Install arm-permitted frozen memory once; reject replacement drift."""
        activity = {} if activity is None else activity
        for session, bank in banks.items():
            if self.identity_enabled:
                if session not in activity:
                    raise ValueError(f"{self.arm} requires M3 activity for {session}")
                value = activity[session].detach().to(dtype=torch.float32, device="cpu").contiguous()
                if tuple(value.shape) != (3, 1024, 176) or not torch.isfinite(value).all():
                    raise ValueError(f"{session}: invalid M3 activity {tuple(value.shape)}")
                name = self._buffer_name("activity", session)
                if hasattr(self, name):
                    if not torch.equal(getattr(self, name).cpu(), value):
                        raise RuntimeError(f"attempted to replace frozen activity for {session}")
                else:
                    self.register_buffer(name, value, persistent=False)
                self._activity[session] = getattr(self, name)
            if self.carrier_enabled:
                value = torch.from_numpy(bank.carrier).to(dtype=torch.float32, device="cpu").contiguous()
                if tuple(value.shape) != (176, 4) or not torch.isfinite(value).all():
                    raise ValueError(f"{session}: invalid H-C carrier {tuple(value.shape)}")
                name = self._buffer_name("carrier", session)
                if hasattr(self, name):
                    if not torch.equal(getattr(self, name).cpu(), value):
                        raise RuntimeError(f"attempted to replace frozen carrier for {session}")
                else:
                    self.register_buffer(name, value, persistent=False)
                self._carrier[session] = getattr(self, name)

    def _mem(self, sessions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        """Compute each session memory once per forward, then restore row order.

        This is a forward-local map. It removes duplicate deterministic C2
        work without caching an autograd graph across optimizer steps.
        """
        memories: dict[str, tuple[Tensor, Tensor]] = {}
        for session in dict.fromkeys(sessions):
            carrier = (self._carrier[session].to(device=device, dtype=torch.float32)
                       if self.carrier_enabled else torch.zeros(176, 4, device=device))
            if self.identity_enabled:
                identity = self.encoder(self._activity[session].to(device=device).unsqueeze(0), carrier.unsqueeze(0)).squeeze(0)
            else:
                identity = torch.zeros(176, 700, device=device, dtype=torch.float32)
            memories[session] = (identity, carrier)
        return (torch.stack([memories[session][0] for session in sessions]),
                torch.stack([memories[session][1] for session in sessions]))

    def frontend_tokens(
        self, x: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None, dropout_keep: Tensor | None = None,
    ) -> Tensor:
        x = self._check_input(x, None)
        banks = self._normalise_banks(bank, x.shape[0])
        keep = self._resolve_keep(banks, unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        identities, carriers = self._mem([item.session_id for item in banks], x.device)
        return self._batched_proj_add_frontend(x, identities, carriers, keep)
