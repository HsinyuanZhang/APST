"""Trainable B3S identity for M1 B3S+BT joint cells.

B3S is SideFeatureEarlyPoolEncoder (hidden 64, side_dim 4, window 100)
initialized from the frozen B3 Sfix e11 ``student.id_encoder``. Side columns
of the first post-pool Linear start at zero, so the untrained encoder matches
``compute_identity(side_features=None)`` when the rSyn3 side is concatenated.
The encoder is then jointly trained with the B-transformer; rSyn3 stays a
closed-form carrier and is also the B3S side input.

Session calib / side tensors are not part of the state_dict (re-attached at
train / score / pick). Live E0 is FP32 and in the decoder graph.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn

from . import m1_projadd as mp
from . import plan
from .identity_variant import BTransformerUnifiedDecoderIdentity
from .bank import TaskBank

B3S_TRIAL_LENGTH = 1024
B3S_HIDDEN = 64
B3S_SIDE_DIM = 4
B3S_POST_LAYERS = 3


def load_trainable_b3s_from_sfix() -> nn.Module:
    """B3S initialized from Sfix B3; side columns zero; all params trainable."""
    mp._workspace_path()
    mp._streaming_path()
    from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder

    b3 = mp.load_b3_id_encoder()
    b3s = SideFeatureEarlyPoolEncoder(
        trial_length=B3S_TRIAL_LENGTH,
        window_size=mp.M1_E0_DIM,
        hidden_dim=B3S_HIDDEN,
        side_dim=B3S_SIDE_DIM,
        num_post_layers=B3S_POST_LAYERS,
    )
    b3s.pre_pool.load_state_dict(b3.pre_pool.state_dict())
    b3_post = list(b3.post_pool)
    b3s_post = list(b3s.post_pool)
    first = b3s_post[0]
    src = b3_post[0]
    if not isinstance(first, nn.Linear) or not isinstance(src, nn.Linear):
        raise RuntimeError("B3/B3S post_pool[0] is not Linear")
    if tuple(src.weight.shape) != (B3S_HIDDEN, B3S_HIDDEN):
        raise RuntimeError(f"B3 post_pool[0] shape drift: {tuple(src.weight.shape)}")
    if tuple(first.weight.shape) != (B3S_HIDDEN, B3S_HIDDEN + B3S_SIDE_DIM):
        raise RuntimeError(f"B3S post_pool[0] shape drift: {tuple(first.weight.shape)}")
    with torch.no_grad():
        first.weight[:, :B3S_HIDDEN].copy_(src.weight)
        first.weight[:, B3S_HIDDEN:].zero_()
        first.bias.copy_(src.bias)
        for src_layer, dst_layer in zip(b3_post[1:], b3s_post[1:]):
            if isinstance(src_layer, nn.Linear) and isinstance(dst_layer, nn.Linear):
                dst_layer.load_state_dict(src_layer.state_dict())
    b3s.train()
    for param in b3s.parameters():
        param.requires_grad_(True)
    return b3s


def encode_b3s(encoder: nn.Module, trials: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Differentiable mean-pool identity.

    ``trials`` is the datamodule layout ``[T, L, N]`` (L=1024, N=64), same as
    ``b3_identity`` / ``push_trial``. ``side`` is ``[N, 4]``.
    """
    if trials.dim() != 3:
        raise RuntimeError(f"B3S trials must be [T, L, N], got {tuple(trials.shape)}")
    if trials.shape[1] != B3S_TRIAL_LENGTH or trials.shape[2] != mp.M1_UNITS:
        raise RuntimeError(f"B3S trials shape {tuple(trials.shape)} != (T, {B3S_TRIAL_LENGTH}, {mp.M1_UNITS})")
    if side.shape != (mp.M1_UNITS, B3S_SIDE_DIM):
        raise RuntimeError(f"B3S side shape {tuple(side.shape)} != ({mp.M1_UNITS}, {B3S_SIDE_DIM})")
    feats = encoder.pre_pool(trials.permute(0, 2, 1))  # [T, N, H]
    mean = feats.mean(dim=0)
    if int(getattr(encoder, "side_dim", 0)) > 0:
        mean = torch.cat([mean, side], dim=-1)
    identity = encoder.post_pool(mean)
    if tuple(identity.shape) != (mp.M1_UNITS, mp.M1_E0_DIM):
        raise RuntimeError(f"B3S identity shape drift: {tuple(identity.shape)}")
    return identity


def b3s_param_count() -> int:
    encoder = load_trainable_b3s_from_sfix()
    return int(sum(p.numel() for p in encoder.parameters()))


def init_e0_matches_b3(calib: np.ndarray, side: np.ndarray, atol: float = 1e-6) -> dict[str, float | bool]:
    """Untrained B3S + any finite side must match frozen B3 (side columns are 0)."""
    b3 = mp.load_b3_id_encoder()
    ref = mp.b3_identity(b3, calib)
    b3s = load_trainable_b3s_from_sfix()
    trials = torch.as_tensor(np.ascontiguousarray(calib, dtype=np.float32))
    side_t = torch.as_tensor(np.ascontiguousarray(side, dtype=np.float32))
    with torch.inference_mode():
        got = encode_b3s(b3s, trials, side_t).cpu().numpy()
    delta = float(np.max(np.abs(got - ref)))
    return {"max_abs_delta": delta, "matched": bool(delta <= atol)}


class B3SJointDecoder(BTransformerUnifiedDecoderIdentity):
    """P16 (or any identity_mode) decoder with a jointly trained B3S encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b3s = load_trainable_b3s_from_sfix()
        self.require_live_e0 = False
        self._calib: dict[str, torch.Tensor] = {}
        self._side: dict[str, torch.Tensor] = {}

    def register_session(self, session: str, calib: np.ndarray | torch.Tensor, side: np.ndarray | torch.Tensor) -> None:
        trials = torch.as_tensor(np.ascontiguousarray(calib, dtype=np.float32))
        if trials.dim() != 3 or trials.shape[1:] != (B3S_TRIAL_LENGTH, mp.M1_UNITS):
            raise RuntimeError(f"{session}: calib must be [T, 1024, 64], got {tuple(trials.shape)}")
        side_t = torch.as_tensor(np.ascontiguousarray(side, dtype=np.float32))
        self._calib[session] = trials
        self._side[session] = side_t

    def attach_session_memories(
        self,
        banks: Mapping[str, TaskBank],
        calib_by_session: Mapping[str, np.ndarray],
        device: torch.device,
    ) -> None:
        for key, bank in banks.items():
            calib = calib_by_session.get(key)
            if calib is None:
                calib = calib_by_session.get(bank.session_id)
            if calib is None:
                raise RuntimeError(f"no calib trials for bank {key!r} / {bank.session_id!r}")
            self.register_session(bank.session_id, calib, bank.carrier)
            if key != bank.session_id:
                self.register_session(key, calib, bank.carrier)
        self.move_session_memories(device)
        self.require_live_e0 = True

    def move_session_memories(self, device: torch.device) -> None:
        self._calib = {k: v.to(device=device, dtype=torch.float32) for k, v in self._calib.items()}
        self._side = {k: v.to(device=device, dtype=torch.float32) for k, v in self._side.items()}

    def live_e0(self, session: str) -> torch.Tensor:
        if session not in self._calib:
            raise RuntimeError(f"B3S session {session!r} was not attached")
        device = next(self.b3s.parameters()).device
        trials = self._calib[session].to(device=device, dtype=torch.float32)
        side = self._side[session].to(device=device, dtype=torch.float32)
        if trials.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return encode_b3s(self.b3s, trials, side)
        return encode_b3s(self.b3s, trials, side)

    def _bank_arrays(self, bank: TaskBank):
        e0, carrier, mask = super()._bank_arrays(bank)
        if not self.require_live_e0:
            return e0, carrier, mask
        live = self.live_e0(bank.session_id)
        carrier = carrier.to(device=live.device, dtype=torch.float32)
        mask = mask.to(device=live.device)
        return live, carrier, mask


def expected_joint_params(proj_dim: int, depth: int) -> int:
    temporal_block = 527104
    decoder = mp.m1_projadd_param_count(proj_dim)
    if int(depth) != 4:
        decoder = decoder - (4 - int(depth)) * temporal_block
    return decoder + b3s_param_count()
