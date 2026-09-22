"""Per-dataset config for the learnable recency-bias family."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import log
from typing import Any, Literal, Mapping

from ..config import DEFAULT_HALF_LIVES, RiftTemporalConfig

Tier = Literal["learned_slope", "fixed"]
IncrementFrom = Literal["pre_ln"]
Ladder = Literal["scaled", "default"]

TASK_CONTEXT_BINS = {"m1": 100, "m2": 50, "h1": 300}
ANCHOR_WINDOW = 75


def half_life_to_slope(half_life: float | None, bin_seconds: float = 0.02) -> float:
    if half_life is None:
        return 0.0
    return log(2.0) * bin_seconds / half_life


def default_slopes(bin_seconds: float = 0.02) -> tuple[float, ...]:
    return tuple(half_life_to_slope(value, bin_seconds) for value in DEFAULT_HALF_LIVES)


def first_layer_window(context_bins: int, layers: int = 4) -> int:
    return int(RiftTemporalConfig.for_context(context_bins, layers=layers).windows[0])


def scaled_half_lives(
    context_bins: int,
    layers: int = 4,
    bin_seconds: float = 0.02,
) -> tuple[float | None, ...]:
    """``DEFAULT_HALF_LIVES * (W0 / 75)``. Flat heads stay ``None``. Exact fractions."""
    del bin_seconds
    window = first_layer_window(context_bins, layers)
    factor = window / ANCHOR_WINDOW
    return tuple(None if half is None else half * factor for half in DEFAULT_HALF_LIVES)


def half_life_bins(
    half_lives: tuple[float | None, ...], bin_seconds: float = 0.02
) -> tuple[float | None, ...]:
    return tuple(None if half is None else half / bin_seconds for half in half_lives)


def parse_half_lives(text: str) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            raise ValueError("empty half-life token")
        if token.lower() == "none":
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError("half-lives must be nonempty")
    return tuple(values)


def resolve_half_lives(
    *,
    context_bins: int,
    layers: int,
    ladder: Ladder = "scaled",
    half_lives: str | None = None,
    bin_seconds: float = 0.02,
) -> tuple[float | None, ...]:
    if half_lives is not None:
        return parse_half_lives(half_lives)
    if ladder == "default":
        return DEFAULT_HALF_LIVES
    if ladder != "scaled":
        raise ValueError("ladder must be 'scaled' or 'default'")
    return scaled_half_lives(context_bins, layers, bin_seconds)


@dataclass(frozen=True)
class LearnableRecencyConfig:
    """Static policy for one learnable-recency temporal stack.

    Defaults: ``per_layer=True``, task-scaled half-life ladder, 4 layers.
    ``learned_slope`` then has 4x6=24 trainable scalars (flat heads masked).
    """

    tier: Tier = "learned_slope"
    per_layer: bool = True
    learn_flat_heads: bool = False
    lr_multiplier: float = 1.0
    increment_from: IncrementFrom = "pre_ln"
    half_life_seconds: tuple[float | None, ...] = DEFAULT_HALF_LIVES
    bin_seconds: float = 0.02
    context_bins: int = 50
    layers: int = 4
    task: str = "m2"
    ladder: Ladder = "scaled"

    def __post_init__(self) -> None:
        if self.tier not in ("learned_slope", "fixed"):
            raise ValueError("tier must be learned_slope or fixed")
        if self.increment_from != "pre_ln":
            raise ValueError(
                "increment_from must be 'pre_ln' (layer residual before norm1)"
            )
        if self.lr_multiplier <= 0:
            raise ValueError("lr_multiplier must be positive")
        if self.layers < 1:
            raise ValueError("layers must be a positive integer")
        if len(self.half_life_seconds) < 1:
            raise ValueError("half_life_seconds must be nonempty")
        if self.ladder not in ("scaled", "default"):
            raise ValueError("ladder must be 'scaled' or 'default'")

    @property
    def temporal_config(self) -> RiftTemporalConfig:
        return RiftTemporalConfig.for_context(
            self.context_bins,
            layers=self.layers,
            bias_mode="recency",
            half_life_seconds=self.half_life_seconds,
            bin_seconds=self.bin_seconds,
        )

    @property
    def half_life_bins(self) -> tuple[float | None, ...]:
        return half_life_bins(self.half_life_seconds, self.bin_seconds)

    def ladder_metadata(self) -> dict[str, object]:
        window = first_layer_window(self.context_bins, self.layers)
        return {
            "ladder": self.ladder,
            "first_layer_window": window,
            "scale_vs_h1": window / ANCHOR_WINDOW,
            "half_life_seconds": [
                None if value is None else float(value)
                for value in self.half_life_seconds
            ],
            "half_life_bins": [
                None if value is None else float(value) for value in self.half_life_bins
            ],
            "windows": list(self.temporal_config.windows),
            "layers": self.layers,
        }


def dataset_config(
    task: str, *, tier: Tier = "learned_slope", **overrides: object
) -> LearnableRecencyConfig:
    """Table defaults: M1 R100, M2 R50, H1 R300. Scaled ladder; per_layer True."""
    if task not in TASK_CONTEXT_BINS:
        raise ValueError(f"unknown task {task!r}; use m1, m2, or h1")
    context = TASK_CONTEXT_BINS[task]
    layers = int(overrides.get("layers", 4))  # type: ignore[arg-type]
    ladder: Ladder = overrides.get("ladder", "scaled")  # type: ignore[assignment]
    if "half_life_seconds" in overrides:
        half = overrides["half_life_seconds"]
    else:
        half = resolve_half_lives(context_bins=context, layers=layers, ladder=ladder)
    values: dict[str, object] = dict(
        task=task,
        context_bins=context,
        layers=layers,
        ladder=ladder,
        tier=tier,
        per_layer=True,
        learn_flat_heads=False,
        lr_multiplier=1.0,
        increment_from="pre_ln",
        half_life_seconds=half,
        bin_seconds=0.02,
    )
    values.update(overrides)
    return LearnableRecencyConfig(**values)  # type: ignore[arg-type]


def config_from_run_meta(meta: Mapping[str, Any], task: str) -> LearnableRecencyConfig:
    raw = meta.get("learnable_config")
    if not isinstance(raw, Mapping):
        return dataset_config(task, tier=str(meta.get("tier", "learned_slope")))  # type: ignore[arg-type]
    allowed = {item.name for item in fields(LearnableRecencyConfig)}
    values: dict[str, Any] = {key: raw[key] for key in raw if key in allowed}
    half = values.get("half_life_seconds")
    if isinstance(half, list):
        values["half_life_seconds"] = tuple(
            None if item is None else float(item) for item in half
        )
    return LearnableRecencyConfig(**values)


DATASET_TABLE = {
    "m1": dataset_config("m1"),
    "m2": dataset_config("m2"),
    "h1": dataset_config("h1"),
}


__all__ = [
    "ANCHOR_WINDOW",
    "DATASET_TABLE",
    "DEFAULT_HALF_LIVES",
    "LearnableRecencyConfig",
    "TASK_CONTEXT_BINS",
    "config_from_run_meta",
    "dataset_config",
    "default_slopes",
    "first_layer_window",
    "half_life_bins",
    "half_life_to_slope",
    "parse_half_lives",
    "resolve_half_lives",
    "scaled_half_lives",
]
