"""Small shared validation and training defaults."""

from __future__ import annotations
from typing import Any

LR_PEAK = 3e-4
LR_MIN_FACTOR = 0.1
EMA_DECAY = 0.9995
UNIT_DROPOUT = 0.1
TASK_GEOMETRY = {
    "m2": {
        "window": 50,
        "prefix": 0,
        "units": 96,
        "e0_dim": 50,
        "carrier_dim": 4,
        "out_dim": 2,
    },
    "m1": {
        "window": 100,
        "prefix": 0,
        "units": 64,
        "e0_dim": 100,
        "carrier_dim": 4,
        "out_dim": 16,
    },
    "h1": {
        "window": 300,
        "prefix": 0,
        "units": 176,
        "e0_dim": 700,
        "carrier_dim": 4,
        "out_dim": 7,
    },
}


class APSTError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise APSTError(message)


def task_geometry(task: str) -> dict[str, Any]:
    require(task in TASK_GEOMETRY, f"unknown task {task!r}")
    return dict(TASK_GEOMETRY[task])


def resolved_e0_dim(g: dict[str, Any]) -> int:
    v = g["e0_dim"]
    require(isinstance(v, int) and v > 0, "e0_dim must be positive")
    return v


def resolved_prefix(g: dict[str, Any], override: int | None = None) -> int:
    v = g["prefix"] if override is None else override
    require(isinstance(v, int) and v >= 0, "prefix must be nonnegative")
    return v


SCHEMA = "apst"
