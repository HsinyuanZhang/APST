"""Label-free activity identity for the P16 learnable-recency RIFT frontend.

The support encoder follows the SPINT identity order exactly: a shared
nonlinear map is applied to each ``[T]`` unit trial, valid trials are averaged,
and a second nonlinear map produces the E0 vector consumed by the existing
proj-add frontend.  It deliberately owns neither a TaskBank nor a per-unit
static parameter.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from btransform_unified_v1.model import whole_unit_dropout

from .config import LearnableRecencyConfig
from .wrap import LearnableRiftDecoder


ACTIVITY_IDENTITY_SEED_OFFSET = 0x41435449
DEFAULT_SUPPORT_BINS = {"m1": 1024, "m2": 100, "h1": 1024}
DEFAULT_IDENTITY_HIDDEN = {"m1": 64, "m2": 64, "h1": 32}


def _three_linear_post_pool(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    """The original B3/B3S three-affine post-pooling stack."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ActivityIdentityEncoder(nn.Module):
    """Map neural-only support ``[M,T,N]`` to an E0 matrix ``[N,E]``."""

    def __init__(self, input_bins: int, output_dim: int, hidden_dim: int = 64, seed: int = 42) -> None:
        super().__init__()
        if not isinstance(input_bins, int) or input_bins < 1:
            raise ValueError("input_bins must be a positive integer")
        if not isinstance(output_dim, int) or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        self.input_bins = input_bins
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # Construct in an isolated RNG domain so adding the activity branch
        # cannot change any inherited P16/RIFT initialization bytes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + ACTIVITY_IDENTITY_SEED_OFFSET)
            # Exact B3/B3S activity trunk: a single affine+ReLU before trial
            # pooling and three affine layers after pooling.  There are no T4,
            # contrast, FiLM, electrode, or descriptor inputs in this branch.
            self.pre_pool = nn.Sequential(nn.Linear(input_bins, hidden_dim), nn.ReLU())
            self.post_pool = _three_linear_post_pool(hidden_dim, output_dim, hidden_dim)

    @property
    def phi(self) -> nn.Sequential:
        """Compatibility alias; parameters remain named ``pre_pool``."""
        return self.pre_pool

    @property
    def psi(self) -> nn.Sequential:
        """Compatibility alias; parameters remain named ``post_pool``."""
        return self.post_pool

    def forward(self, activity: Tensor, trial_mask: Tensor | None = None) -> Tensor:
        if activity.ndim != 3 or activity.shape[0] < 1 or activity.shape[1] != self.input_bins or activity.shape[2] < 1:
            raise ValueError(f"activity must be nonempty [M,{self.input_bins},N], got {tuple(activity.shape)}")
        if not torch.is_floating_point(activity):
            activity = activity.to(dtype=torch.float32)
        if not torch.isfinite(activity).all():
            raise ValueError("activity contains non-finite values")
        trials, _bins, units = activity.shape
        if trial_mask is None:
            mask = torch.ones(trials, dtype=torch.bool, device=activity.device)
        else:
            if trial_mask.dtype != torch.bool or trial_mask.shape != (trials,):
                raise ValueError("trial_mask must be bool [M]")
            mask = trial_mask.to(device=activity.device)
        if not bool(mask.any()):
            raise ValueError("trial_mask selects no support trials")

        # [M,T,N] -> [M,N,T]; pre_pool is shared across both M and N axes.
        # The nonlinear per-trial activity features are averaged before post_pool.
        features = self.pre_pool(activity.permute(0, 2, 1))
        weights = mask.to(dtype=features.dtype).view(trials, 1, 1)
        pooled = (features * weights).sum(dim=0) / weights.sum(dim=0)
        identity = self.post_pool(pooled)
        if identity.shape != (units, self.output_dim):
            raise RuntimeError("activity identity output geometry drift")
        return identity


class ActivityLearnableRiftDecoder(LearnableRiftDecoder):
    """P16/D4 RIFT decoder whose E0 is produced from neural support activity."""

    def __init__(
        self,
        task: str,
        cfg: LearnableRecencyConfig,
        *,
        context_bins: int | None = None,
        seed: int = 42,
        support_bins: int | None = None,
        identity_hidden: int | None = None,
        proj_dim: int = 16,
    ) -> None:
        if task not in DEFAULT_SUPPORT_BINS:
            raise ValueError("ActivityLearnableRiftDecoder supports only m1, m2, and h1")
        super().__init__(task, cfg, context_bins=context_bins, seed=seed, proj_dim=int(proj_dim))
        self.support_bins = DEFAULT_SUPPORT_BINS[task] if support_bins is None else int(support_bins)
        if self.support_bins < 1:
            raise ValueError("support_bins must be a positive integer")
        self.identity_hidden = DEFAULT_IDENTITY_HIDDEN[task] if identity_hidden is None else int(identity_hidden)
        if self.identity_hidden < 1:
            raise ValueError("identity_hidden must be a positive integer")
        self.identity_encoder = ActivityIdentityEncoder(
            self.support_bins,
            self.geometry["e0_dim"],
            hidden_dim=self.identity_hidden,
            seed=seed,
        )
        self.register_buffer("zero_carrier", torch.zeros(self.units, 4), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        carrier_key = prefix + "zero_carrier"
        carrier = state_dict.get(carrier_key)
        if carrier is not None and bool(torch.count_nonzero(carrier)):
            error_msgs.append(f"{carrier_key} must remain the literal zero activity carrier")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def encode_activity(self, activity: Tensor, trial_mask: Tensor | None = None) -> Tensor:
        """Encode support with gradients intact for source-joint training."""
        return self.identity_encoder(activity, trial_mask)

    def calibrate(self, activity: Tensor, trial_mask: Tensor | None = None) -> Tensor:
        """Produce a detached, label-free identity only in evaluation mode."""
        if self.training:
            raise RuntimeError("calibrate is eval-only; call model.eval()")
        with torch.no_grad():
            return self.encode_activity(activity, trial_mask).detach()

    def _identity_batch(self, identity: Tensor, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if identity.ndim == 2:
            if identity.shape != (self.units, self.geometry["e0_dim"]):
                raise ValueError("identity must be [N,E0dim] or [B,N,E0dim]")
            identity = identity.unsqueeze(0).expand(batch, -1, -1)
        elif identity.ndim == 3:
            if identity.shape != (batch, self.units, self.geometry["e0_dim"]):
                raise ValueError("identity must be [N,E0dim] or [B,N,E0dim]")
        else:
            raise ValueError("identity must be [N,E0dim] or [B,N,E0dim]")
        if not torch.is_floating_point(identity) or not torch.isfinite(identity).all():
            raise ValueError("identity must be finite floating-point")
        return identity.to(device=device, dtype=dtype)

    def _resolve_activity_keep(
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
            base = unit_mask.to(device=device, dtype=torch.bool)
            if base.ndim == 1:
                base = base.unsqueeze(0).expand(batch, -1)
            if base.shape != (batch, self.units):
                raise ValueError("unit_mask must be [N] or [B,N]")
        if not bool(base.any(dim=1).all()):
            raise ValueError("unit mask is empty for some batch row")
        if dropout_keep is not None:
            keep = dropout_keep.to(device=device, dtype=torch.bool)
            if keep.ndim == 1:
                keep = keep.unsqueeze(0)
            if keep.ndim != 2 or keep.shape[1] != self.units or keep.shape[0] not in (1, batch):
                raise ValueError("dropout_keep must be [N] or [B,N]")
            keep = keep.expand(batch, -1) if keep.shape[0] == 1 else keep
            keep = base & keep
        elif self.training and self._frontend_owner.unit_dropout_p > 0:
            keep = whole_unit_dropout(base, self._frontend_owner.unit_dropout_p, dropout_generator)
        else:
            keep = base
        if not bool(keep.any(dim=1).all()):
            raise ValueError("unit mask became empty for some batch row")
        return keep.contiguous()

    def _fuse_activity_local(self, local: Tensor, identity: Tensor, keep: Tensor) -> Tensor:
        carrier = self.zero_carrier.to(device=local.device, dtype=local.dtype)
        carrier = carrier.unsqueeze(0).expand(local.shape[0], -1, -1)
        return self._fuse_batched_local(local, identity, carrier, keep)

    def frontend_tokens(
        self,
        x: Tensor,
        identity: Tensor,
        unit_mask: Tensor | None = None,
        dropout_generator: torch.Generator | None = None,
        dropout_keep: Tensor | None = None,
    ) -> Tensor:
        x = self._check_input(x, None)
        identity = self._identity_batch(identity, x.shape[0], x.device, x.dtype)
        keep = self._resolve_activity_keep(unit_mask, x.shape[0], x.device, dropout_generator, dropout_keep)
        return self._fuse_activity_local(self.frontend.local_conv(x), identity, keep)

    def frontend_last(self, raw5: Tensor, identity: Tensor, unit_mask: Tensor | None = None) -> Tensor:
        if raw5.ndim != 3 or raw5.shape[1:] != (5, self.units):
            raise ValueError(f"raw5 must be [B,5,{self.units}]")
        raw5 = self._check_input(raw5, None)
        identity = self._identity_batch(identity, raw5.shape[0], raw5.device, raw5.dtype)
        keep = self._resolve_activity_keep(unit_mask, raw5.shape[0], raw5.device, None, None)
        batch = raw5.shape[0]
        flat = raw5.permute(0, 2, 1).reshape(batch * self.units, 1, 5)
        conv = self.frontend.local_conv
        local = conv.act(conv.conv(flat)).reshape(batch, self.units, 16, 1).permute(0, 3, 1, 2)
        return self._fuse_activity_local(local, identity, keep)[:, 0]

    def forward_scores(
        self,
        x: Tensor,
        activity: Tensor | None = None,
        *,
        identity: Tensor | None = None,
        unit_mask: Tensor | None = None,
        dropout_keep: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
        trial_mask: Tensor | None = None,
    ) -> Tensor:
        if (activity is None) == (identity is None):
            raise ValueError("provide exactly one of activity or identity")
        if identity is not None and trial_mask is not None:
            raise ValueError("trial_mask is only valid with activity")
        x = self._check_input(x, input_valid_mask)
        e0 = self.encode_activity(activity, trial_mask) if activity is not None else identity
        assert e0 is not None
        if input_valid_mask is not None:
            x = torch.where(input_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        z = self.frontend_tokens(x, e0, unit_mask=unit_mask, dropout_keep=dropout_keep)
        hidden = self.temporal(z, input_valid_mask)
        return self.readout(self.final_norm(hidden))

    def forward(
        self,
        x: Tensor,
        activity: Tensor | None = None,
        *,
        identity: Tensor | None = None,
        unit_mask: Tensor | None = None,
        dropout_keep: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
        trial_mask: Tensor | None = None,
    ) -> Tensor:
        return self.forward_scores(
            x,
            activity,
            identity=identity,
            unit_mask=unit_mask,
            dropout_keep=dropout_keep,
            input_valid_mask=input_valid_mask,
            trial_mask=trial_mask,
        )[:, -1]


__all__ = ["ActivityIdentityEncoder", "ActivityLearnableRiftDecoder", "DEFAULT_SUPPORT_BINS", "DEFAULT_IDENTITY_HIDDEN"]
