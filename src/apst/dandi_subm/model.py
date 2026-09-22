"""DANDI688 F0/Flat profile-site models for the 2026-09-18 campaign.

The approved sites are ACT (no profile), identity-only (E0 profile),
token-only, and dual-site. Explicit gates prevent silent profile injection.
"""
from __future__ import annotations

from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from apst.models.decoder import RiftDecoder
from apst.models.recency.config import dataset_config, half_life_to_slope
from apst.models.recency.temporal import LearnableRecencyTemporal
from apst.models.film import PooledCarrierFiLM

Method = Literal["act_only", "identity_only", "token_only", "dual_site"]
Temporal = Literal["F0", "flat"]
Stage = Literal["pretrain", "train"]

UNITS, E0_DIM, PROFILE_DIM, HIDDEN, CONTEXT = 100, 50, 4, 64, 50
METHODS = frozenset(("act_only", "identity_only", "token_only", "dual_site"))
TEMPORALS = frozenset(("F0", "flat"))


def _affine_stack() -> nn.Sequential:
    # Fixed 68-wide post input removes a representation-width confound.  The
    # final four inputs are zeros in the activity family and T4 in profile arms.
    return nn.Sequential(
        nn.Linear(HIDDEN + PROFILE_DIM, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, E0_DIM),
    )


class ProfileEncoder(nn.Module):
    """M33 activity encoder with an optional E0 association-profile route."""
    def __init__(self, method: Method, *, enable_film: bool, seed: int) -> None:
        super().__init__()
        self.method = method
        self.profile_gate = method in {"identity_only", "dual_site"}
        # The profile family shares source pretraining with concat side input;
        # FiLM is added only in its frozen stage-2 model.
        self.uses_concat = self.profile_gate
        self.uses_film = bool(enable_film and self.profile_gate)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 0x44414E44)
            self.pre_pool = nn.Sequential(nn.Linear(100, HIDDEN), nn.ReLU())
            self.post_pool = _affine_stack()
        # Concat arms use profile after source pretraining, yet begin with the
        # same activity path as a zero-side encoder.
        with torch.no_grad():
            self.post_pool[0].weight[:, HIDDEN:].zero_()
        self.film = (PooledCarrierFiLM(HIDDEN, film_rank=8, init_seed=int(seed) + 0x46494C4D)
                     if self.uses_film else None)

    def forward(self, activity: Tensor, carrier: Tensor | None = None) -> Tensor:
        if activity.ndim != 3 or activity.shape[1] != 100 or activity.shape[2] > UNITS:
            raise ValueError("activity must be [M33,100,N], with 1 <= N <= 100")
        if not torch.is_floating_point(activity) or not bool(torch.isfinite(activity).all()):
            raise ValueError("activity must be finite floating point")
        units = activity.shape[2]
        if not self.profile_gate:
            if carrier is not None:
                raise ValueError(f"{self.method} encoder rejects carrier: activity-only E0 has no profile input")
            profile = None
        else:
            if carrier is None or tuple(carrier.shape) != (units, PROFILE_DIM):
                raise ValueError(f"{self.method} requires finite carrier [N,{PROFILE_DIM}]")
            if not torch.is_floating_point(carrier) or not bool(torch.isfinite(carrier).all()):
                raise ValueError("carrier must be finite floating point")
            profile = carrier.to(device=activity.device, dtype=activity.dtype)
        pooled = self.pre_pool(activity.permute(0, 2, 1)).mean(dim=0)
        if self.film is not None:
            assert profile is not None
            pooled = self.film(pooled, profile)
        side = profile if self.uses_concat else torch.zeros(units, PROFILE_DIM, device=activity.device, dtype=activity.dtype)
        return self.post_pool(torch.cat((pooled, side), dim=-1))


def _install_f0(decoder: RiftDecoder) -> None:
    """Install real trainable near-flat learned recency without a PE facade."""
    cfg = dataset_config("m2", tier="learned_slope", ladder="default", learn_flat_heads=True)
    if tuple(cfg.temporal_config.windows) != (13, 12, 12, 12) or cfg.temporal_config.width != 256 or cfg.temporal_config.heads != 8:
        raise RuntimeError("F0 requires M2 R50/D4/256/8 geometry")
    learned = LearnableRecencyTemporal.from_initialized(decoder.temporal, cfg)
    slope = half_life_to_slope(25.0, 0.02)  # 25 x M2's one-second context.
    with torch.no_grad():
        learned.recency_slopes.fill_(float(slope))
        learned._learn_mask.fill_(True)
        learned.slope_log.zero_()
    if tuple(learned.slope_log.shape) != (4, 8) or not bool(torch.all(learned.effective_slopes() == float(slope))):
        raise RuntimeError("F0 uniform all-head near-flat initialization failed")
    decoder.temporal = learned
    decoder.temporal_config = learned.config
    decoder.bias_mode = "F0_nearflat_nope"


class DandiF0FiLMModel(nn.Module):
    """Frozen-or-trainable E0 encoder paired with a gated RIFT token route."""
    def __init__(self, method: Method, temporal: Temporal, *, seed: int, stage: Stage,
                 encoder_state: Mapping[str, Tensor] | None) -> None:
        super().__init__()
        if method not in METHODS or temporal not in TEMPORALS or stage not in {"pretrain", "train"}:
            raise ValueError("unsupported method, temporal, or stage")
        self.method, self.temporal_kind, self.stage = method, temporal, stage
        self.profile_gate = method in {"identity_only", "dual_site"}
        self.token_gate = method in {"token_only", "dual_site"}
        # Pretraining intentionally learns the shared encoder without FiLM;
        # only the frozen stage=train profile family creates rank-8 FiLM.
        self.encoder = ProfileEncoder(method, enable_film=(stage == "train"), seed=seed)
        if encoder_state is not None:
            base_keys = {k for k in self.encoder.state_dict() if not k.startswith("film.")}
            if set(encoder_state) != base_keys:
                raise ValueError("encoder_state must exactly match the no-FiLM source-pretrain encoder state")
            missing, unexpected = self.encoder.load_state_dict(dict(encoder_state), strict=False)
            allowed_missing = {k for k in self.encoder.state_dict() if k.startswith("film.")}
            if set(missing) != allowed_missing or unexpected:
                raise ValueError(f"encoder_state incompatible: missing={missing}, unexpected={unexpected}")
        if stage == "train":
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            # A frozen source encoder is deliberately followed by a fresh
            # FiLM adapter. Its zero init preserves source encoder E0 at step0.
            if self.encoder.film is not None:
                for parameter in self.encoder.film.parameters():
                    parameter.requires_grad_(True)
            self.encoder.pre_pool.eval(); self.encoder.post_pool.eval()
        geometry = {"task": "dandi688_f0_film", "units": UNITS, "e0_dim": E0_DIM,
                    "carrier_dim": PROFILE_DIM, "out_dim": 2}
        self.decoder = RiftDecoder(geometry, context_bins=CONTEXT, bias_mode="flat", seed=seed, proj_dim=16)
        if temporal == "F0":
            _install_f0(self.decoder)
        if tuple(self.decoder.temporal.config.windows) != (13, 12, 12, 12):
            raise RuntimeError("R50 temporal window drift")
        self.decoder.temporal.set_attention_backend("local")
        self.register_buffer("token_carrier", torch.zeros(UNITS, PROFILE_DIM), persistent=True)

    def export_encoder_state(self) -> dict[str, Tensor]:
        """CPU clone suitable for a source-pretrain to frozen-train hand-off."""
        return {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()
                if not k.startswith("film.")}

    def encode(self, activity: Tensor, carrier: Tensor | None = None) -> Tensor:
        # Encoder/E0 and FiLM follow the established FALCON precision contract:
        # retain FP32 under the trainer's surrounding bf16 decoder autocast.
        device_type = activity.device.type
        profile = None if carrier is None else carrier.float()
        with torch.autocast(device_type=device_type, enabled=False):
            return self.encoder(activity.float(), profile)

    def _batch_e0(self, e0: Tensor, batch: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        if e0.ndim == 2:
            if tuple(e0.shape) != (UNITS, E0_DIM):
                raise ValueError("e0 must be [100,50] or [B,100,50]")
            e0 = e0.unsqueeze(0).expand(batch, -1, -1)
        if tuple(e0.shape) != (batch, UNITS, E0_DIM):
            raise ValueError("e0 must be [100,50] or [B,100,50]")
        return e0.to(device=device, dtype=dtype)

    def _keep(self, value: Tensor | None, batch: int, device: torch.device, name: str) -> Tensor:
        if value is None:
            return torch.ones(batch, UNITS, dtype=torch.bool, device=device)
        value = value.to(device=device, dtype=torch.bool)
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(batch, -1)
        if tuple(value.shape) != (batch, UNITS) or not bool(value.any(dim=1).all()):
            raise ValueError(f"{name} must be a nonempty [100] or [B,100] mask")
        return value

    def _token_carrier(self, carrier: Tensor | None, batch: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        if not self.token_gate:
            if carrier is not None and bool(torch.count_nonzero(carrier)):
                raise ValueError("this method fixes decoder token carrier to literal zero")
            return self.token_carrier.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
        if carrier is None:
            raise ValueError("token-profile method requires carrier [100,4]")
        if carrier.ndim == 2:
            if tuple(carrier.shape) != (UNITS, PROFILE_DIM):
                raise ValueError("token carrier must be [100,4] or [B,100,4]")
            carrier = carrier.unsqueeze(0).expand(batch, -1, -1)
        if tuple(carrier.shape) != (batch, UNITS, PROFILE_DIM) or not bool(torch.isfinite(carrier).all()):
            raise ValueError("token carrier must be finite [100,4] or [B,100,4]")
        return carrier.to(device=device, dtype=dtype)

    def forward_identity(self, x: Tensor, e0: Tensor, carrier: Tensor | None = None, *,
                         unit_mask: Tensor | None = None, dropout_keep: Tensor | None = None,
                         input_valid_mask: Tensor | None = None) -> Tensor:
        x = self.decoder._check_input(x, input_valid_mask)
        if input_valid_mask is not None:
            x = torch.where(input_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        batch = x.shape[0]
        keep = self._keep(unit_mask, batch, x.device, "unit_mask") & self._keep(dropout_keep, batch, x.device, "dropout_keep")
        if not bool(keep.any(dim=1).all()):
            raise ValueError("combined unit mask is empty")
        e0_batch = self._batch_e0(e0, batch, x.dtype, x.device)
        token_carrier = self._token_carrier(carrier, batch, x.dtype, x.device)
        tokens = self.decoder._batched_proj_add_frontend(x, e0_batch, token_carrier, keep)
        return self.decoder.readout(self.decoder.final_norm(self.decoder.temporal(tokens, input_valid_mask)))[:, -1]

    def forward(self, x: Tensor, activity: Tensor, carrier: Tensor | None = None, *,
                unit_mask: Tensor | None = None, dropout_keep: Tensor | None = None,
                input_valid_mask: Tensor | None = None) -> Tensor:
        if self.method == "act_only" and carrier is not None:
            raise ValueError("act_only rejects carrier at both E0 and token sites")
        encoder_carrier = carrier if self.profile_gate else None
        e0 = self.encode(activity, encoder_carrier)
        return self.forward_identity(x, e0, carrier=carrier if self.token_gate else None,
                                     unit_mask=unit_mask, dropout_keep=dropout_keep,
                                     input_valid_mask=input_valid_mask)


def build_model(method: Method, temporal: Temporal, seed: int = 42, stage: Stage = "train",
                encoder_state: Mapping[str, Tensor] | None = None) -> DandiF0FiLMModel:
    """Build one approved profile-site/temporal cell."""
    return DandiF0FiLMModel(method, temporal, seed=int(seed), stage=stage, encoder_state=encoder_state)


__all__ = ["DandiF0FiLMModel", "ProfileEncoder", "build_model", "METHODS", "TEMPORALS"]
