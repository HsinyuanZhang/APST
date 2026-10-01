"""F_nope_learned temporal policy: E's no-PE facade plus learned slopes.

This module deliberately operates on the *facade* installed by the actual E
factories. Replacing that facade would lose its no-positional-encoding
contract; only ``facade.core`` is exchanged after the E initializer has run.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

W = Path(__file__).resolve().parents[2]
for path in (
    W / "src",
    W.parent / "btransform_unified_v2" / "learnable_recency_v1" / "src",
    W.parent / "btransform_unified_v2" / "learnable_recency_v1" / "scripts",
    W.parent / "btransform_unified_v2" / "src",
    W.parent / "btransform_unified_v1" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from a1_m2.nope_temporal_v1 import install_nope
from a1_m2.temporal_variants import A1Temporal, VariantSpec
from learnable_recency_v1.config import dataset_config, half_life_to_slope
from learnable_recency_v1.temporal import LearnableRecencyTemporal


# This is deliberately independent of E_SPEC: F keeps E's noPE behavior but
# is a learned-bias arm, so its serialized architecture metadata must not say
# ``E_nope_flat`` / ``bias='none'``.
F_SPEC = VariantSpec("F_nope_learned", False, "learned")


class FNoPELearnedTemporal(A1Temporal):
    """NoPE learned-slope facade whose eval path remains vectorized.

    ``A1Temporal`` routes all learned variants through sequential ``step`` at
    eval time for C's PE/cache contract. F has no positional encoding and its
    learned-slope core has an independently verified local-vectorized parity,
    so retain the native vectorized core path for F inference as well.
    """

    def forward(
        self,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        backend: str | None = None,
        absolute_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Validate caller-provided coordinates consistently with the common
        # facade, even though noPE means the values do not enter the core.
        if absolute_positions is not None:
            self._positions(z, absolute_positions, valid_mask)
        return self.core(z, valid_mask, backend=backend)


def _default_slope_ladder(cfg, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Materialize the declared default ladder without consuming the RNG."""
    return torch.tensor(
        [half_life_to_slope(value, cfg.bin_seconds) for value in cfg.half_life_seconds],
        device=device,
        dtype=dtype,
    )


def install_f_nope_learned(decoder, task: str):
    """Turn an actual E decoder into F while retaining all E trainable bytes.

    F adds exactly ``temporal.core.slope_log`` with shape ``[4, 8]``. The two
    flat default heads remain in that state-dict tensor, but their effective
    slopes and gradients are masked, leaving 24 active scalar directions.
    """
    if task not in ("m1", "m2", "h1"):
        raise ValueError(f"unknown task {task!r}")

    # Establish the real E control first. This changes the non-trainable
    # zero-slope buffer and facade spec only; shared trainable tensors stay
    # byte-identical to the same-seed E factory output.
    install_nope(decoder, task)
    facade = decoder.temporal
    old_core = facade.core
    cfg = dataset_config(task, tier="learned_slope", ladder="default")
    if old_core.config.layers != 4 or old_core.config.heads != 8:
        raise RuntimeError("F requires the native four-layer/eight-head E temporal core")
    if tuple(old_core.config.windows) != tuple(cfg.temporal_config.windows):
        raise RuntimeError("E core windows do not match F's native task configuration")

    # ``from_initialized`` steals the initialized blocks. Restore the declared
    # ladder after E zeroed only its fixed recency buffer. This construction
    # and constant parameter initialization do not advance torch's RNG.
    learned = LearnableRecencyTemporal.from_initialized(old_core, cfg)
    with torch.no_grad():
        learned.recency_slopes.copy_(
            _default_slope_ladder(
                cfg, device=learned.recency_slopes.device, dtype=learned.recency_slopes.dtype
            )
        )
    # Rebuild only the non-trainable facade around the stolen core. Its public
    # parameter paths remain ``temporal.core.*`` and the forward override
    # avoids C's PE-oriented sequential eval branch.
    f_facade = FNoPELearnedTemporal(learned, F_SPEC).to(learned.recency_slopes.device)
    decoder.temporal = f_facade
    decoder.temporal_config = learned.config
    decoder.bias_mode = "learned_slope"

    if f_facade.spec.use_sinusoidal_pe or f_facade.spec.arm != "F_nope_learned" or f_facade.spec.bias != "learned":
        raise RuntimeError("F must retain E's no-PE facade")
    if tuple(learned.slope_log.shape) != (4, 8):
        raise RuntimeError("F learned-slope shape drift")
    if int((learned.recency_slopes > 0).sum().item()) != 6:
        raise RuntimeError("F default ladder must have exactly two flat heads")
    return decoder


__all__ = ["install_f_nope_learned"]
