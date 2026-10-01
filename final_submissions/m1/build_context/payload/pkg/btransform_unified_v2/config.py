"""Configuration for the RIFT streaming temporal stack."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


BiasMode = Literal["recency", "flat"]

DEFAULT_HALF_LIVES = (0.08, 0.16, 0.32, 0.64, 1.28, 2.56, None, None)


@dataclass(frozen=True)
class RiftTemporalConfig:
    """Static shape and local-attention policy for a RIFT temporal stack.

    ``windows[l]`` includes the current token.  Consequently layer ``l``
    persists exactly ``windows[l] - 1`` prior KV entries per stream row.
    """

    width: int = 256
    heads: int = 8
    ffn_width: int = 512
    windows: tuple[int, ...] = (75, 75, 75, 74)
    half_life_seconds: tuple[float | None, ...] = DEFAULT_HALF_LIVES
    bin_seconds: float = 0.02
    bias_mode: BiasMode = "recency"

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(self.heads, int) or self.heads <= 0:
            raise ValueError("heads must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(self.ffn_width, int) or self.ffn_width <= 0:
            raise ValueError("ffn_width must be a positive integer")
        if not self.windows:
            raise ValueError("windows must contain at least one layer")
        if any(not isinstance(window, int) or window < 1 for window in self.windows):
            raise ValueError("every window must be a positive integer (including current token)")
        if len(self.half_life_seconds) != self.heads:
            raise ValueError("half_life_seconds must have one entry per attention head")
        for half_life in self.half_life_seconds:
            if half_life is not None and (not isfinite(half_life) or half_life <= 0):
                raise ValueError("half lives must be positive finite seconds or None")
        if not isfinite(self.bin_seconds) or self.bin_seconds <= 0:
            raise ValueError("bin_seconds must be a positive finite number")
        if self.bias_mode not in ("recency", "flat"):
            raise ValueError("bias_mode must be 'recency' or 'flat'")

    @property
    def layers(self) -> int:
        return len(self.windows)

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def token_receptive_field(self) -> int:
        """Maximum frontend-token span; add frontend kernel extent externally."""
        return 1 + sum(window - 1 for window in self.windows)

    @classmethod
    def for_context(
        cls,
        context_bins: int,
        kernel: int = 5,
        bias_mode: BiasMode = "recency",
        *,
        width: int = 256,
        heads: int = 8,
        ffn_width: int = 512,
        layers: int = 4,
        half_life_seconds: tuple[float | None, ...] = DEFAULT_HALF_LIVES,
        bin_seconds: float = 0.02,
    ) -> "RiftTemporalConfig":
        """Build approximately equal D-layer windows for a raw receptive field.

        Given frontend kernel ``k``, distributes ``context_bins - k`` temporal
        extensions left-to-right.  E.g. R300/k5/D4 is ``(75,75,75,74)``.
        """
        if not isinstance(kernel, int) or kernel < 1:
            raise ValueError("kernel must be a positive integer")
        if not isinstance(context_bins, int) or context_bins < kernel:
            raise ValueError("context_bins must be an integer greater than or equal to kernel")
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("layers must be a positive integer")
        extensions, remainder = divmod(context_bins - kernel, layers)
        windows = tuple(1 + extensions + (index < remainder) for index in range(layers))
        return cls(
            width=width,
            heads=heads,
            ffn_width=ffn_width,
            windows=windows,
            half_life_seconds=half_life_seconds,
            bin_seconds=bin_seconds,
            bias_mode=bias_mode,
        )
