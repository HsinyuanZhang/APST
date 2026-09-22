"""Decoder / stream wrappers that attach learnable recency without editing mainline."""

from __future__ import annotations

from math import log
from typing import Sequence

import torch
from torch import Tensor, nn

from ..config import RiftTemporalConfig
from ..decoder import RiftDecoder
from ..temporal import RiftTemporal

from .config import LearnableRecencyConfig
from .temporal import LearnableRecencyTemporal

RIFT_TEMPORAL_SEED_OFFSET = 0x52494654


def _architecture_key(config: RiftTemporalConfig) -> tuple:
    return (config.layers, config.windows, config.width, config.heads, config.ffn_width)


def _slopes_from_config(
    config: RiftTemporalConfig, device: torch.device | None = None
) -> Tensor:
    slopes = [
        0.0 if half is None else log(2.0) * config.bin_seconds / half
        for half in config.half_life_seconds
    ]
    tensor = torch.tensor(slopes, dtype=torch.float32)
    return tensor if device is None else tensor.to(device=device)


def reinit_rift_linears(temporal: RiftTemporal, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + RIFT_TEMPORAL_SEED_OFFSET)
        for module in temporal.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def seeded_rift_temporal(
    config: RiftTemporalConfig,
    seed: int,
    *,
    device: torch.device | str | None = None,
) -> RiftTemporal:
    """Build a ``RiftTemporal`` and re-init Linears under the decoder temporal seed."""
    rng = torch.get_rng_state()
    try:
        temporal = RiftTemporal(config)
        reinit_rift_linears(temporal, seed)
        if device is not None:
            temporal = temporal.to(device=torch.device(device))
        return temporal
    finally:
        torch.set_rng_state(rng)


def _shell_with_blocks(
    existing: RiftTemporal, config: RiftTemporalConfig
) -> RiftTemporal:
    """Keep already-seeded blocks; swap frozen config and recency slope buffer."""
    temporal = RiftTemporal.__new__(RiftTemporal)
    nn.Module.__init__(temporal)
    temporal.config = config
    temporal.add_module("blocks", existing.blocks)
    temporal.register_buffer(
        "recency_slopes",
        _slopes_from_config(config, existing.recency_slopes.device),
        persistent=True,
    )
    temporal._attention_backend = existing._attention_backend
    return temporal


def adopt_temporal_config(
    existing: RiftTemporal, config: RiftTemporalConfig, seed: int
) -> RiftTemporal:
    if _architecture_key(existing.config) != _architecture_key(config):
        device = next(existing.parameters()).device
        return seeded_rift_temporal(config, seed, device=device)
    if (
        existing.config.half_life_seconds == config.half_life_seconds
        and existing.config.bin_seconds == config.bin_seconds
    ):
        return existing
    return _shell_with_blocks(existing, config)


def install_temporal(
    decoder: nn.Module, recency_cfg: LearnableRecencyConfig, seed: int
) -> nn.Module:
    """Rebuild the temporal stack to the requested depth/ladder, then wrap or leave fixed."""
    existing = decoder.temporal
    if not isinstance(existing, RiftTemporal):
        raise TypeError("decoder.temporal must be a RiftTemporal")
    rebuilt = adopt_temporal_config(existing, recency_cfg.temporal_config, seed)
    if recency_cfg.tier == "fixed":
        decoder.temporal = rebuilt
    else:
        decoder.temporal = attach_learnable_recency(rebuilt, recency_cfg)
    decoder.temporal_config = recency_cfg.temporal_config
    decoder.learnable_cfg = recency_cfg
    decoder.bias_mode = (
        "recency" if recency_cfg.tier == "fixed" else f"learnable_{recency_cfg.tier}"
    )
    return decoder


def attach_learnable_recency(
    temporal: RiftTemporal, recency_cfg: LearnableRecencyConfig
) -> LearnableRecencyTemporal:
    """Replace a finished ``RiftTemporal`` with a learnable stack (no RNG)."""
    if recency_cfg.tier == "fixed":
        raise ValueError("fixed tier must stay a plain RiftTemporal")
    if isinstance(temporal, LearnableRecencyTemporal):
        raise TypeError("temporal is already a LearnableRecencyTemporal")
    return LearnableRecencyTemporal.from_initialized(temporal, recency_cfg)


def new_parameter_names(module: torch.nn.Module) -> list[str]:
    temporal = getattr(module, "temporal", module)
    if not isinstance(temporal, LearnableRecencyTemporal):
        return []
    prefix = "temporal." if getattr(module, "temporal", None) is temporal else ""
    return [f"{prefix}{name}" for name in temporal.new_parameter_names()]


def split_optimizer_parameters(
    module: torch.nn.Module,
    *,
    peak_lr: float,
    weight_decay: float,
    lr_multiplier: float,
) -> list[dict]:
    """Two AdamW groups: shared weights keep recipe WD; new recency params have WD=0."""
    new_names = set(new_parameter_names(module))
    shared = [
        param
        for name, param in module.named_parameters()
        if name not in new_names and param.requires_grad
    ]
    extra = [
        param
        for name, param in module.named_parameters()
        if name in new_names and param.requires_grad
    ]
    groups = [
        {
            "params": shared,
            "lr": peak_lr,
            "weight_decay": weight_decay,
            "lr_multiplier": 1.0,
        }
    ]
    if extra:
        groups.append(
            {
                "params": extra,
                "lr": peak_lr * lr_multiplier,
                "weight_decay": 0.0,
                "lr_multiplier": lr_multiplier,
            }
        )
    return groups


def apply_group_lrs(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_multiplier", 1.0))
