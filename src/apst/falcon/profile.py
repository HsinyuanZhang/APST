"""The association-profile encoder (component A) and closed-form calibration.

Architecture per task (frozen protocol): per-trial ``pre_pool``
``Linear(support_bins -> hidden) + ReLU`` applied unit-major, equal-weight
mean over the calibration trials, an optional rank-8 carrier FiLM (exactly
identity at zero initialization), carrier concat, and a three-layer
``post_pool`` MLP producing the E0 conditioning ``[units, e0_dim]``.

Target-session calibration is closed form: the (frozen) trunk runs one
``no_grad`` forward pass over support activity; no target parameter is ever
updated.  When FiLM is enabled and trained in stage 2, the trained FiLM is
baked offline into exported banks, ``E0 = post(cat(FiLM(pooled, c), c))``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from . import protocol
from .common import array_sha256, sha256

TRUNK_SEED_OFFSET = 0x434F4E43   # "CONC"
FILM_SEED_OFFSET = 0x46494C4D    # "FILM"


class AssociationProfile(nn.Module):
    """Per-task A trunk: pre_pool -> equal-weight trial mean -> FiLM -> post_pool."""

    def __init__(self, task: str, *, film: bool, seed: int) -> None:
        super().__init__()
        spec = protocol.task_protocol(task)
        self.task = task
        self.units = int(spec["units"])
        self.hidden = int(spec["profile_hidden"])
        self.support_bins = int(spec["support_bins"])
        self.e0_dim = int(spec["e0_dim"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + TRUNK_SEED_OFFSET)
            self.pre_pool = nn.Sequential(nn.Linear(self.support_bins, self.hidden), nn.ReLU())
            self.post_pool = nn.Sequential(
                nn.Linear(self.hidden + 4, self.hidden), nn.ReLU(),
                nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                nn.Linear(self.hidden, self.e0_dim),
            )
        from apst.models.film import PooledCarrierFiLM
        self.film = PooledCarrierFiLM(self.hidden, film_rank=8, enabled=bool(film),
                                      init_seed=int(seed) + FILM_SEED_OFFSET)

    def pooled(self, activity: torch.Tensor, trial_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Equal-weight mean of per-trial features over valid support trials."""
        if activity.ndim != 3 or activity.shape[1:] != (self.support_bins, self.units):
            raise ValueError(f"{self.task} support must be [M,{self.support_bins},{self.units}]")
        features = self.pre_pool(activity.permute(0, 2, 1))
        if trial_mask is None:
            return features.mean(dim=0)
        if trial_mask.shape != (activity.shape[0],) or not bool(trial_mask.any()):
            raise ValueError("trial_mask must select at least one support trial")
        weight = trial_mask.to(device=features.device, dtype=features.dtype)
        return (features * weight[:, None, None]).sum(dim=0) / weight.sum()

    def e0(self, pooled: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        """FiLM-modulated pooled features through the frozen-form post pool."""
        return self.post_pool(torch.cat((self.film(pooled.float(), carrier.float()),
                                         carrier.detach().to(pooled.dtype)), dim=-1))

    def forward(self, activity: torch.Tensor, carrier: torch.Tensor,
                trial_mask: torch.Tensor | None = None) -> torch.Tensor:
        if carrier.shape != (self.units, 4):
            raise ValueError(f"{self.task} carrier must be [{self.units},4]")
        return self.e0(self.pooled(activity, trial_mask), carrier)

    def freeze_trunk(self) -> None:
        """Freeze the stage-1 trained pooling trunk; FiLM stays trainable."""
        for module in (self.pre_pool, self.post_pool):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def trunk_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in
                list(self.pre_pool.state_dict().items()) + list(self.post_pool.state_dict().items())}


def decode(decoder: nn.Module, x: torch.Tensor, e0: torch.Tensor, carrier: torch.Tensor,
           valid: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Decoder forward shared by supervision and stage-2 (noPE temporal)."""
    z = decoder._batched_proj_add_frontend(x, e0, carrier, keep)
    hidden = decoder.temporal(z, valid)
    return decoder.readout(decoder.final_norm(hidden))[:, -1]


class Supervision(nn.Module):
    """Stage-1 wrapper: A trunk plus a fresh flat-temporal supervision decoder."""

    def __init__(self, task: str, *, seed: int) -> None:
        super().__init__()
        spec = protocol.task_protocol(task)
        from apst.models import RiftDecoder
        self.a = AssociationProfile(task, film=False, seed=seed)
        self.decoder = RiftDecoder(task, context_bins=int(spec["context_bins"]),
                                   bias_mode="flat", seed=seed, proj_dim=16)

    def forward(self, x, activity, carrier, valid, keep, trial_mask=None):
        e0 = self.a(activity, carrier, trial_mask).unsqueeze(0).expand(x.shape[0], -1, -1)
        c = carrier.unsqueeze(0).expand(x.shape[0], -1, -1)
        return decode(self.decoder, x, e0, c, valid, keep)


class Stage2Joint(nn.Module):
    """Stage-2 wrapper: frozen A trunk, trainable FiLM (when enabled), fresh decoder."""

    def __init__(self, task: str, *, seed: int, film: bool, ladder_scale: float,
                 encoder_state: Mapping[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.task = task
        self.film_enabled = bool(film)
        self.a = AssociationProfile(task, film=film, seed=seed)
        if encoder_state is not None:
            # A stage-1 checkpoint carries the pooling trunk (``a.*`` keys);
            # FiLM parameters are absent there and keep their zero init.
            state = _strip_prefix(encoder_state, "a.")
            self.a.load_state_dict(
                {key: value for key, value in state.items() if not key.startswith("film.")},
                strict=False)
        self.a.freeze_trunk()
        self.decoder = build_stage2_decoder(task, seed=seed, ladder_scale=float(ladder_scale))

    def e0(self, pooled: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        # FiLM-modulated E0 stays FP32 even under decoder autocast.
        with torch.autocast(device_type="cuda", enabled=False):
            return self.a.e0(pooled.float(), carrier.float())

    def forward(self, x, pooled, carrier, valid, keep):
        e0 = self.e0(pooled, carrier).unsqueeze(0).expand(x.shape[0], -1, -1)
        c = carrier.unsqueeze(0).expand(x.shape[0], -1, -1)
        return decode(self.decoder, x, e0, c, valid, keep)


def _strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if any(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    return dict(state)


def build_stage2_decoder(task: str, *, seed: int, ladder_scale: float):
    """Fresh F0 decoder: noPE temporal with a ladder-scaled learned-slope init."""
    from apst.models import RiftDecoder
    from apst.models.recency.config import DEFAULT_HALF_LIVES, dataset_config
    from apst.models.recency.utils import install_temporal
    spec = protocol.task_protocol(task)
    decoder = RiftDecoder(task, context_bins=int(spec["context_bins"]),
                          bias_mode="flat", seed=seed, proj_dim=16)
    half_lives = tuple(None if half is None else half * float(ladder_scale)
                       for half in DEFAULT_HALF_LIVES)
    config = dataset_config(task, tier="learned_slope", half_life_seconds=half_lives)
    install_temporal(decoder, config, seed)
    return decoder


@torch.no_grad()
def calibrate(encoder: nn.Module, activity: np.ndarray, carrier: np.ndarray,
              *, trial_mask: np.ndarray | None = None,
              device: str | torch.device = "cpu") -> np.ndarray:
    """Closed-form target calibration: one frozen forward pass, no gradients."""
    encoder.eval()
    activity_t = torch.as_tensor(np.asarray(activity, np.float32), device=device)
    carrier_t = torch.as_tensor(np.asarray(carrier, np.float32), device=device)
    mask = None if trial_mask is None else torch.as_tensor(np.asarray(trial_mask, bool), device=device)
    e0 = encoder(activity_t, carrier_t, mask)
    return e0.detach().cpu().numpy().astype(np.float32)


def export_bank(dest: Path, session_id: str, e0: np.ndarray, carrier: np.ndarray,
                unit_mask: np.ndarray, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Write one ``{E0,carrier,unit_mask}.npy`` bank with a manifest row."""
    out = Path(dest) / session_id
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"E0.npy": np.asarray(e0, np.float32), "carrier.npy": np.asarray(carrier, np.float32),
              "unit_mask.npy": np.asarray(unit_mask, bool)}
    for name, value in arrays.items():
        np.save(out / name, value)
    return {
        "session_id": session_id,
        "arrays": {name: {"path": str((out / name).relative_to(dest)), "sha256": sha256(out / name),
                          "typed_sha256": array_sha256(value), "shape": list(value.shape)}
                   for name, value in arrays.items()},
        "meta": dict(meta),
    }


def unit_mask_for(task: str) -> np.ndarray:
    return np.ones(protocol.task_protocol(task)["units"], dtype=bool)


def budget_slice(activity: np.ndarray, trials: Sequence[int]) -> np.ndarray:
    """Support-budget view: the first ``len(trials)`` calibration trials."""
    indices = list(trials)
    if not indices or indices != list(range(len(indices))):
        raise ValueError("budget trials must be the prefix 0..M-1")
    return np.ascontiguousarray(np.asarray(activity, np.float32)[:len(indices)])
