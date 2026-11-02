"""Config loading and validation for the unified FALCON recipe.

A config is a JSON document with schema ``apst_falcon_v1``.  Like the DANDI
Sub-C module, configs are only accepted with an adjacent ``.json.sha256``
sidecar written by ``init-config``, and the embedded protocol snapshot must
match :mod:`apst.falcon.protocol` exactly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import protocol

SCHEMA = "apst_falcon_v1"
DEFAULT_RECIPE = {
    "epochs": 24,
    "batch": 32,
    "lr_peak": 3e-4,
    "weight_decay": 0.01,
    "slope_weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "grad_clip": 1.0,
    "ema_decay": 0.9995,
    "unit_dropout": 0.1,
    "warmup_updates": None,
    "cosine_min_factor": 0.1,
    "cosine_schedule_epochs": None,
}
RECIPE_KEYS = tuple(sorted(DEFAULT_RECIPE))


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"config is missing its sha256 sidecar: {sidecar}")
    expected = sidecar.read_text().split()[0]
    if sha(path) != expected:
        raise ValueError(f"config file hash does not match sidecar: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    validate(config)
    return config


def validate(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    task = config.get("task")
    if task not in protocol.TASKS:
        raise ValueError(f"config task must be one of {protocol.TASKS}, got {task!r}")
    if config.get("stage") not in ("pretrain", "stage2"):
        raise ValueError("config stage must be 'pretrain' or 'stage2'")
    if not isinstance(config.get("film"), bool):
        raise ValueError("config film must be a boolean")
    scale = config.get("ladder_scale")
    if not isinstance(scale, (int, float)) or not float(scale) > 0.0:
        raise ValueError("config ladder_scale must be a positive number")
    if not isinstance(config.get("seed"), int):
        raise ValueError("config seed must be an integer")
    recipe = config.get("recipe")
    if not isinstance(recipe, dict) or set(recipe) != set(RECIPE_KEYS):
        raise ValueError(f"config recipe keys must be exactly {list(RECIPE_KEYS)}")
    snapshot = config.get("protocol")
    if not isinstance(snapshot, dict):
        raise ValueError("config must embed a protocol snapshot (run init-config)")
    protocol.validate(snapshot)
    cache = config.get("cache")
    if not isinstance(cache, str) or not Path(cache).is_absolute():
        raise ValueError("config cache must be an absolute prepared-cache path")
    receipt = Path(cache) / "prepared_receipt.json"
    if not receipt.is_file():
        raise FileNotFoundError(f"prepared cache receipt missing: {receipt}")
    if config.get("prepared_receipt_sha256") != sha(receipt):
        raise ValueError("config prepared_receipt_sha256 does not match the cache receipt")
    output_ema = config.get("output_ema")
    if not isinstance(output_ema, dict) or abs(float(output_ema.get("alpha", 0.0)) - 1.0 / 3.0) > 1e-12 \
            or output_ema.get("training") is not False:
        raise ValueError("config output_ema must be the frozen inference-only alpha=1/3 contract")
    if config.get("updates_per_epoch") is not None:
        if not isinstance(config["updates_per_epoch"], int) or config["updates_per_epoch"] < 1:
            raise ValueError("config updates_per_epoch must be a positive integer when present")
