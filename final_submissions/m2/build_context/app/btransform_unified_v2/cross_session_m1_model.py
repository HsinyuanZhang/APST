"""Three-arm M1 cross-session decoder: no calibration, activity, and joint."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import numpy as np
import torch
from torch import Tensor, nn

from btransform_unified_v1.bank import TaskBank
from .concat_model import RiftConcatDecoder

Z_NONE = "Z_NONE"
B_ACTIVITY_ONLY = "B_ACTIVITY_ONLY"
D_JOINT = "D_JOINT"
ARMS = (Z_NONE, B_ACTIVITY_ONLY, D_JOINT)


class CrossSessionM1Decoder(RiftConcatDecoder):
    """R100/D4 concat decoder with the preregistered Z/B/D information arms.

    Z never stores, reads, or differentiates through calibration tensors.  B
    and D use an identically initialized random activity encoder; D alone
    admits the source/target-specific four-column carrier in both legal paths.
    """
    def __init__(self, arm: str, *, seed: int) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown cross-session M1 arm {arm!r}")
        super().__init__("m1", context_bins=100, bias_mode="recency", seed=seed)
        self.arm = arm
        self.encoder: nn.Module | None = None
        self._calib: dict[str, Tensor] = {}
        self._carrier: dict[str, Tensor] = {}
        if arm != Z_NONE:
            # Deliberately random: loading Sfix/all-source B3 would leak folds.
            streaming = Path(__file__).resolve().parents[3] / "streaming_calibration_exp"
            if str(streaming) not in sys.path:
                sys.path.insert(0, str(streaming))
            from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 0x423353)
                self.encoder = SideFeatureEarlyPoolEncoder(
                    trial_length=1024, window_size=100, hidden_dim=64,
                    side_dim=4, num_post_layers=3,
                )

    @property
    def uses_calibration(self) -> bool:
        return self.arm != Z_NONE

    def install_calibration(self, banks: Mapping[str, TaskBank], calib: Mapping[str, np.ndarray | Tensor]) -> None:
        """Attach M10 only for B/D; reject even accidental Z calibration IO."""
        if self.arm == Z_NONE:
            if calib:
                raise RuntimeError("Z_NONE must not receive calibration tensors")
            return
        for name, bank in banks.items():
            raw = calib.get(name, calib.get(bank.session_id))
            if raw is None:
                raise ValueError(f"missing M10 calibration for {bank.session_id}")
            raw_t = torch.as_tensor(np.ascontiguousarray(raw, dtype=np.float32))
            carrier = torch.as_tensor(np.ascontiguousarray(bank.carrier, dtype=np.float32))
            if tuple(raw_t.shape) != (10, 1024, 64) or tuple(carrier.shape) != (64, 4):
                raise ValueError("M1 calibration/carrier geometry drift")
            self._calib[bank.session_id] = raw_t
            self._carrier[bank.session_id] = carrier

    def _identity(self, sessions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        b = len(sessions)
        if self.arm == Z_NONE:
            return (torch.zeros((b, 64, 100), device=device), torch.zeros((b, 64, 4), device=device))
        if self.encoder is None:
            raise RuntimeError("activity arm lacks its encoder")
        from btransform_unified_v1.m1_b3s_joint import encode_b3s
        e0, direct = {}, {}
        for session in dict.fromkeys(sessions):
            raw = self._calib[session].to(device=device, dtype=torch.float32)
            carrier = self._carrier[session].to(device=device, dtype=torch.float32)
            side = carrier if self.arm == D_JOINT else torch.zeros_like(carrier)
            # The live calibration branch is intentionally FP32 even when the
            # decoder runs under BF16 autocast; its trial pooling is a small
            # but numerically sensitive source-side identity calculation.
            with torch.autocast(device_type=device.type, enabled=False):
                e0[session] = encode_b3s(self.encoder, raw, side)
            direct[session] = carrier if self.arm == D_JOINT else torch.zeros_like(carrier)
        return torch.stack([e0[s] for s in sessions]), torch.stack([direct[s] for s in sessions])

    def frontend_tokens(self, x: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask=None,
                        dropout_generator=None, dropout_keep=None) -> Tensor:
        x = self._check_input(x, None); banks = self._normalise_banks(bank, x.shape[0])
        keep = self._resolve_keep(banks, unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        e0, carrier = self._identity([item.session_id for item in banks], x.device)
        return self._batched_proj_add_frontend(x, e0, carrier, keep)


def parameter_accounting(seed: int = 42) -> dict[str, dict[str, int]]:
    """Count actual registered and optimizer-active parameters; no fake padding."""
    result = {}
    for arm in ARMS:
        model = CrossSessionM1Decoder(arm, seed=seed)
        result[arm] = {"total": sum(p.numel() for p in model.parameters()),
                       "active": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return result
