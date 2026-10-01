"""Decoder-only EMA shadow (decay 0.9995), after m2_b_small_stability_v1.ema.

The shadow never trains and never aliases RAW parameter storage. First
successful update copies RAW; later updates use ``decay * EMA + (1-decay) * RAW``.
Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from . import plan


def _trainable_named(module: nn.Module) -> dict[str, nn.Parameter]:
    if hasattr(module, "trainable_parameters"):
        return dict(module.trainable_parameters())
    return {name: param for name, param in module.named_parameters() if param.requires_grad}


class DecoderEMA:
    """EMA shadow of a decoder's trainable parameters (default decay 0.9995)."""

    def __init__(self, module: nn.Module, decay: float = plan.EMA_DECAY) -> None:
        plan.require(0.0 < float(decay) < 1.0, f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.n_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: torch.empty_like(param.detach(), dtype=torch.float32)
            for name, param in _trainable_named(module).items()
        }
        for tensor in self.shadow.values():
            tensor.requires_grad_(False)

    def update_after_step(self, module: nn.Module) -> None:
        """Fold RAW into the shadow after one optimizer step (no grad)."""
        named = _trainable_named(module)
        plan.require(set(named) == set(self.shadow), "trainable parameter set changed since EMA init")
        with torch.no_grad():
            if self.n_updates == 0:
                for name, param in named.items():
                    self.shadow[name] = param.detach().to(dtype=torch.float32).clone()
                    self.shadow[name].requires_grad_(False)
            else:
                one_minus = 1.0 - self.decay
                for name, param in named.items():
                    raw = param.detach().to(dtype=torch.float32)
                    self.shadow[name].mul_(self.decay).add_(raw, alpha=one_minus)
        self.n_updates += 1

    def apply_to(self, module: nn.Module) -> nn.Module:
        """Copy the shadow into RAW parameters in place (caller restores/keeps RAW)."""
        named = _trainable_named(module)
        plan.require(set(named) == set(self.shadow), "trainable parameter set changed since EMA init")
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        return module

    def state_dict(self) -> dict[str, Any]:
        """Serializable EMA state (cloned tensors; no aliasing with the shadow)."""
        return {
            "decay": self.decay,
            "n_updates": self.n_updates,
            "shadow": {name: tensor.detach().clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        plan.require("decay" in state and "n_updates" in state and "shadow" in state, "EMA state missing keys")
        plan.require(float(state["decay"]) == self.decay, "EMA decay mismatch on load")
        shadow = dict(state["shadow"])
        plan.require(set(shadow) == set(self.shadow), "EMA shadow keys mismatch on load")
        self.n_updates = int(state["n_updates"])
        for name, tensor in shadow.items():
            self.shadow[name].copy_(tensor.detach().to(dtype=torch.float32))


__all__ = ["DecoderEMA"]
