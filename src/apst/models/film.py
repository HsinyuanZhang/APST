"""A RNG-isolated FiLM adapter for a pooled activity representation.

The module intentionally owns only the post-mean modulation.  Callers retain
their existing activity trunk, pooling procedure, and decoder/post-pool.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class PooledCarrierFiLM(nn.Module):
    """Apply carrier-conditioned FiLM to ``h[..., H]`` after pooling.

    ``carrier`` has four coordinates (the task carrier such as T4).  Its
    leading dimensions may broadcast to those of ``h``.  The output is
    ``(1 + gamma) * h + beta``, where
    ``[gamma, beta] = out(ReLU(context(carrier)))``.  The final projection is
    exactly zero-initialized, hence a newly constructed enabled module is an
    exact identity map.

    Parameter construction runs inside ``torch.random.fork_rng`` with an
    explicit seed so it cannot perturb the caller's global CPU RNG state.
    """

    carrier_dim = 4

    def __init__(
        self,
        hidden_dim: int,
        *,
        film_rank: int = 8,
        enabled: bool = True,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if film_rank <= 0:
            raise ValueError("film_rank must be positive")
        self.hidden_dim = int(hidden_dim)
        self.film_rank = int(film_rank)
        self.enabled = bool(enabled)
        self.init_seed = int(init_seed)

        # Linear constructors draw random weights.  Keep that draw local so a
        # wrapper can be inserted after its base model without changing its
        # initialization or a subsequently constructed decoder.
        with torch.random.fork_rng(devices=[]):
            # Do not use torch.manual_seed here: it also seeds every CUDA
            # generator, while this fork saves only CPU state.
            torch.random.default_generator.manual_seed(self.init_seed)
            self.context = nn.Sequential(
                nn.Linear(self.carrier_dim, self.film_rank),
                nn.ReLU(),
            )
            self.modulation = nn.Linear(self.film_rank, 2 * self.hidden_dim)
        with torch.no_grad():
            self.modulation.weight.zero_()
            self.modulation.bias.zero_()

    def forward(self, h: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        """Return FiLM-modulated pooled vectors, preserving broadcast shape."""
        if h.ndim < 1 or h.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"h must end in hidden_dim={self.hidden_dim}, got {tuple(h.shape)}"
            )
        if carrier.ndim < 1 or carrier.shape[-1] != self.carrier_dim:
            raise ValueError(
                f"carrier must end in {self.carrier_dim}, got {tuple(carrier.shape)}"
            )
        try:
            leading = torch.broadcast_shapes(h.shape[:-1], carrier.shape[:-1])
        except RuntimeError as exc:
            raise ValueError(
                f"h and carrier leading shapes cannot broadcast: {tuple(h.shape)} vs {tuple(carrier.shape)}"
            ) from exc
        h_broadcast = h.expand(*leading, self.hidden_dim)
        carrier_broadcast = carrier.to(device=h.device, dtype=h.dtype).expand(
            *leading, self.carrier_dim
        )
        if not self.enabled:
            return h_broadcast
        gamma, beta = self.modulation(self.context(carrier_broadcast)).chunk(2, dim=-1)
        return (1.0 + gamma) * h_broadcast + beta

    @classmethod
    def from_legacy_t4_plus_zeros(
        cls,
        *,
        hidden_dim: int,
        legacy_context_weight: torch.Tensor,
        legacy_context_bias: torch.Tensor,
        legacy_modulation_weight: torch.Tensor,
        legacy_modulation_bias: torch.Tensor,
        enabled: bool = True,
        init_seed: int = 0,
    ) -> "PooledCarrierFiLM":
        """Map an old ``[T4, 0_4]`` FiLM head to four carrier inputs.

        For every T4 vector ``c``, this produces the same modulation as the
        legacy head evaluated on ``torch.cat([c, zeros_like(c)], -1)``.  The
        old fourth-to-eighth input columns are intentionally discarded because
        their input is identically zero for the EMPTY/native path.
        """
        if legacy_context_weight.ndim != 2 or legacy_context_weight.shape[1] != 8:
            raise ValueError("legacy_context_weight must be [rank, 8]")
        rank = int(legacy_context_weight.shape[0])
        if tuple(legacy_context_bias.shape) != (rank,):
            raise ValueError("legacy_context_bias shape mismatch")
        if tuple(legacy_modulation_weight.shape) != (2 * hidden_dim, rank):
            raise ValueError("legacy_modulation_weight shape mismatch")
        if tuple(legacy_modulation_bias.shape) != (2 * hidden_dim,):
            raise ValueError("legacy_modulation_bias shape mismatch")
        result = cls(hidden_dim, film_rank=rank, enabled=enabled, init_seed=init_seed)
        with torch.no_grad():
            result.context[0].weight.copy_(legacy_context_weight[:, : cls.carrier_dim])
            result.context[0].bias.copy_(legacy_context_bias)
            result.modulation.weight.copy_(legacy_modulation_weight)
            result.modulation.bias.copy_(legacy_modulation_bias)
        return result

    @classmethod
    def from_legacy_state_dict(
        cls,
        *,
        hidden_dim: int,
        legacy_state: Mapping[str, torch.Tensor],
        enabled: bool = True,
        init_seed: int = 0,
    ) -> "PooledCarrierFiLM":
        """Convenience mapping for HoldContrast's four FiLM state keys."""
        required = {
            "contrast_context.0.weight",
            "contrast_context.0.bias",
            "contrast_film.weight",
            "contrast_film.bias",
        }
        missing = required - set(legacy_state)
        if missing:
            raise ValueError(f"legacy state lacks keys: {sorted(missing)}")
        return cls.from_legacy_t4_plus_zeros(
            hidden_dim=hidden_dim,
            legacy_context_weight=legacy_state["contrast_context.0.weight"],
            legacy_context_bias=legacy_state["contrast_context.0.bias"],
            legacy_modulation_weight=legacy_state["contrast_film.weight"],
            legacy_modulation_bias=legacy_state["contrast_film.bias"],
            enabled=enabled,
            init_seed=init_seed,
        )
