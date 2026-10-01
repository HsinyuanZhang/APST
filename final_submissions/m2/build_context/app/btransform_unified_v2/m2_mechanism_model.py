"""Independently trainable M2 information and architecture mechanism arms.

This module deliberately has its own surface: it does not alter the frozen
proj-add, concat, or joint-FiLM implementations.  The information arms are
``B`` (activity), ``C`` (carrier), ``D`` (joint), and ``D_SHUFFLE`` (joint
with one fixed carrier/unit correspondence permutation per session).  The two
architecture controls are selected with ``aggregation`` and ``temporal``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from btransform_unified_v1.bank import TaskBank
from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder
from tfpd_exploration.src.m2_dual_track_v1 import champion
from .model import RiftDecoder

ARM_B = "B_ACTIVITY_ONLY"
ARM_C = "C_CARRIER_ONLY"
ARM_D = "D_JOINT"
ARM_D_SHUFFLE = "D_SHUFFLE"
ARMS = (ARM_B, ARM_C, ARM_D, ARM_D_SHUFFLE)
AGGREGATIONS = ("slots", "mean")
TEMPORALS = ("attention", "nonattention")


class CausalDepthwiseTemporal(nn.Module):
    """Four pre-LN causal depthwise-convolution blocks with raw RF 50.

    The windows include the current token.  With local frontend kernel five,
    ``1 + (13-1) + 11 + 11 + 11 + 4`` is exactly 50 raw bins.
    """
    windows = (13, 12, 12, 12)
    width = 256
    receptive_field = 50

    def __init__(self) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(self.width) for _ in self.windows)
        self.depthwise = nn.ModuleList(nn.Conv1d(self.width, self.width, w, groups=self.width, bias=True)
                                       for w in self.windows)
        self.pointwise = nn.ModuleList(nn.Linear(self.width, self.width) for _ in self.windows)
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(self.width) for _ in self.windows)
        self.ffns = nn.ModuleList(nn.Sequential(nn.Linear(self.width, 512), nn.GELU(), nn.Linear(512, self.width))
                                  for _ in self.windows)

    def forward(self, z: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if z.ndim != 3 or z.shape[-1] != self.width:
            raise ValueError("z must be [B,T,256]")
        if valid_mask is None:
            valid_mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        if valid_mask.dtype != torch.bool or valid_mask.shape != z.shape[:2]:
            raise ValueError("valid_mask must be bool [B,T]")
        if (valid_mask[:, :-1] & ~valid_mask[:, 1:]).any():
            raise ValueError("valid_mask may only contain left padding")
        h = torch.where(valid_mask.unsqueeze(-1), z, torch.zeros_like(z))
        for norm, depthwise, pointwise, ffn_norm, ffn in zip(
            self.norms, self.depthwise, self.pointwise, self.ffn_norms, self.ffns
        ):
            # LayerNorm's learned bias would otherwise turn a padded zero
            # token into causal convolution evidence after training.
            y = torch.where(valid_mask.unsqueeze(-1), norm(h), torch.zeros_like(h)).transpose(1, 2)
            y = depthwise(F.pad(y, (depthwise.kernel_size[0] - 1, 0))).transpose(1, 2)
            h = h + pointwise(y)
            h = h + ffn(ffn_norm(h))
            h = torch.where(valid_mask.unsqueeze(-1), h, torch.zeros_like(h))
        return h

    def init_state(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> dict[str, Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return {"tokens": torch.empty((batch_size, 0, self.width), device=device, dtype=dtype),
                "valid": torch.empty((batch_size, 0), device=device, dtype=torch.bool)}

    def step(self, z: Tensor, state: dict[str, Tensor], valid_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if z.ndim != 2 or z.shape[-1] != self.width:
            raise ValueError("z must be [B,256]")
        if valid_mask is None:
            valid_mask = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
        tokens = torch.cat((state["tokens"], z.unsqueeze(1)), dim=1)
        valid = torch.cat((state["valid"], valid_mask.unsqueeze(1)), dim=1)
        # RF samples are sufficient for the last output and retain causal state.
        tokens, valid = tokens[:, -self.receptive_field:], valid[:, -self.receptive_field:]
        out = self(tokens, valid)[:, -1]
        return out, {"tokens": tokens, "valid": valid}

    def select_rows(self, state: dict[str, Tensor], row_indices: Tensor | Sequence[int]) -> dict[str, Tensor]:
        rows = torch.as_tensor(row_indices, device=state["tokens"].device, dtype=torch.long)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= state["tokens"].shape[0]).any():
            raise ValueError("row indices are outside temporal state")
        return {"tokens": state["tokens"][rows], "valid": state["valid"][rows]}

    def reset_rows(self, state: dict[str, Tensor], row_indices: Tensor | Sequence[int]) -> dict[str, Tensor]:
        rows = torch.as_tensor(row_indices, device=state["tokens"].device, dtype=torch.long)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= state["tokens"].shape[0]).any():
            raise ValueError("row indices are outside temporal state")
        state["tokens"][rows] = 0; state["valid"][rows] = False
        return state


class M2MechanismDecoder(RiftDecoder):
    """M2 RIFT mechanism arm with static information explicitly controlled."""
    def __init__(self, arm: str, *, aggregation: str = "slots", temporal: str = "attention", seed: int = 42) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown mechanism arm {arm!r}")
        if aggregation not in AGGREGATIONS or temporal not in TEMPORALS:
            raise ValueError("aggregation must be slots/mean and temporal attention/nonattention")
        super().__init__("m2", context_bins=50, bias_mode="recency", seed=seed, proj_dim=16)
        self.arm, self.aggregation, self.temporal_kind = arm, aggregation, temporal
        if arm != ARM_C:
            self.encoder = HoldContrastFiLMEarlyPoolEncoder(100, 50, 64, side_dim=8, film_rank=8,
                                                              num_post_layers=3, film_input="t4_plus_contrast")
            champion.overlay_canonical_p0_and_empty_head(self.encoder)
        if aggregation == "mean":
            # Remove every learned-slot component: the only set aggregation is
            # masked mean(tokens), followed by this explicit 256 -> 256 map.
            for name in ("slots", "slot_norm", "mha", "slot_ffn", "slot_ffn_norm", "slot_proj"):
                delattr(self.frontend, name)
            self.mean_token_proj = nn.Linear(256, 256)
        if temporal == "nonattention":
            self.temporal = CausalDepthwiseTemporal()
            self.temporal_config = None
        self._calib: dict[str, Tensor] = {}
        self._carrier: dict[str, Tensor] = {}

    def install_session_memory(self, banks: Mapping[str, TaskBank], cache_root: Path) -> None:
        for session, bank in banks.items():
            key = session.replace("-", "_")
            carrier = np.ascontiguousarray(bank.carrier.astype(np.float32, copy=False))
            if carrier.shape != (96, 4):
                raise ValueError(f"carrier geometry drift {session}: {carrier.shape}")
            if self.arm != ARM_C:
                raw = np.load(cache_root / session / "calib_activity.npy").astype(np.float32, copy=False)
                if raw.shape != (33, 100, 96):
                    raise ValueError(f"raw M33 geometry drift {session}: {raw.shape}")
                if hasattr(self, f"calib_{key}"):
                    if not np.array_equal(getattr(self, f"calib_{key}").cpu().numpy(), raw):
                        raise ValueError(f"cross-surface calibration drift {session}")
                else:
                    self.register_buffer(f"calib_{key}", torch.from_numpy(np.ascontiguousarray(raw)), persistent=False)
                self._calib[session] = getattr(self, f"calib_{key}")
            if hasattr(self, f"carrier_{key}"):
                if not np.array_equal(getattr(self, f"carrier_{key}").cpu().numpy(), carrier):
                    raise ValueError(f"cross-surface carrier drift {session}")
            else:
                self.register_buffer(f"carrier_{key}", torch.from_numpy(carrier), persistent=False)
            self._carrier[session] = getattr(self, f"carrier_{key}")
            if not hasattr(self, f"carrier_perm_{key}"):
                generator = torch.Generator(device="cpu"); generator.manual_seed(101)
                perm = torch.randperm(96, generator=generator)
                self.register_buffer(f"carrier_perm_{key}", perm, persistent=True)

    def _identity(self, sessions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        unique = list(dict.fromkeys(sessions)); encoded: dict[str, Tensor] = {}; carriers: dict[str, Tensor] = {}
        with torch.autocast(device_type=device.type, enabled=False):
            for session in unique:
                carrier = getattr(self, f"carrier_{session.replace('-', '_')}").to(device=device, dtype=torch.float32)
                if self.arm == ARM_D_SHUFFLE:
                    carrier = carrier[getattr(self, f"carrier_perm_{session.replace('-', '_')}")]
                if self.arm == ARM_C:
                    encoded[session] = torch.zeros((96, 50), device=device, dtype=torch.float32)
                else:
                    raw = getattr(self, f"calib_{session.replace('-', '_')}").to(device=device, dtype=torch.float32)
                    side = torch.zeros((96, 8), device=device, dtype=torch.float32)
                    if self.arm in (ARM_D, ARM_D_SHUFFLE): side[:, :4] = carrier
                    state = self.encoder.reset_stream(1, 96, device, torch.float32); state["side_features"] = side.unsqueeze(0)
                    for trial in raw: state = self.encoder.push_trial(state, trial)
                    encoded[session] = self.encoder.finalize_identity(state).squeeze(0)
                carriers[session] = carrier if self.arm in (ARM_C, ARM_D, ARM_D_SHUFFLE) else torch.zeros_like(carrier)
        return torch.stack([encoded[s] for s in sessions]), torch.stack([carriers[s] for s in sessions])

    def _fuse_batched_local(self, local: Tensor, e0: Tensor, carrier: Tensor, keep: Tensor) -> Tensor:
        if self.aggregation == "slots":
            return super()._fuse_batched_local(local, e0, carrier, keep)
        batch, width, units, _ = local.shape
        projection = self.frontend.e0_proj
        assert projection is not None
        proj = projection(e0.to(dtype=local.dtype)); groups = proj.shape[-1] // 16
        fused = (local.unsqueeze(-2) + proj.view(batch, 1, units, groups, 16)).reshape(batch, width, units, -1)
        tokens = self.frontend.token_norm(self.frontend.token_mlp(torch.cat((fused, carrier.to(local.dtype).unsqueeze(1).expand(batch, width, units, -1)), dim=-1)))
        masked = tokens * keep[:, None, :, None].to(tokens.dtype)
        return self.mean_token_proj(masked.sum(2) / keep.sum(1).view(batch, 1, 1).to(tokens.dtype))

    def frontend_tokens(self, x: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask=None, dropout_generator=None, dropout_keep=None) -> Tensor:
        x = self._check_input(x, None); banks = self._normalise_banks(bank, x.shape[0])
        keep = self._resolve_keep(banks, unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        e0, carrier = self._identity([b.session_id for b in banks], x.device)
        return self._batched_proj_add_frontend(x, e0, carrier, keep)

    def frontend_last(self, raw5: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask=None) -> Tensor:
        if raw5.ndim != 3 or raw5.shape[1:] != (5, self.units): raise ValueError("raw5 must be [B,5,96]")
        raw5 = self._check_input(raw5, None); banks = self._normalise_banks(bank, raw5.shape[0]); keep = self._resolve_keep(banks, unit_mask, raw5.shape[0], raw5.device, None, None)
        e0, carrier = self._identity([b.session_id for b in banks], raw5.device)
        flat = raw5.permute(0, 2, 1).reshape(raw5.shape[0] * self.units, 1, 5); conv = self.frontend.local_conv
        local = conv.act(conv.conv(flat)).reshape(raw5.shape[0], self.units, 16, 1).permute(0, 3, 1, 2)
        return self._fuse_batched_local(local, e0, carrier, keep)[:, 0]


__all__ = ["M2MechanismDecoder", "CausalDepthwiseTemporal", "ARM_B", "ARM_C", "ARM_D", "ARM_D_SHUFFLE", "ARMS"]
