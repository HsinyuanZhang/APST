#!/usr/bin/env python3
"""M2 A-flat diagnostic recipe: old AdamW decay partition over frozen v2 flow.

Base runner SHA-256 at recipe creation:
8db2d84683c611796220c379ba4f6badda3b395f1cb839dbbb13fa88aa8d2d3b.
The sole training change is the optimizer parameter partition below.  Data
admission, decoder construction, RNG, bf16, schedule, EMA, scoring and replay
are delegated unchanged to ``a1_m2_train_v2``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

WEEK = Path(__file__).resolve().parents[2]
if str(WEEK / "src") not in sys.path:
    sys.path.insert(0, str(WEEK / "src"))
from a1_m2 import a1_m2_train_v2 as base

BASE_SHA256 = "8db2d84683c611796220c379ba4f6badda3b395f1cb839dbbb13fa88aa8d2d3b"
RECIPE_SCHEMA = "apst_m2_stage2_legacy_decay_bridge_v1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def old_no_decay(name: str, parameter: Any) -> bool:
    """Exact predicate from m2_dual_track_v1.training.is_no_weight_decay."""
    lowered = name.lower()
    return (parameter.ndim <= 1 or name.endswith(".bias") or lowered.endswith("bias")
            or "norm" in lowered or "ln" in lowered.split(".") or "bn" in lowered.split(".")
            or "dynamics" in lowered)


def coverage(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, groups = [], {"decay": [], "no_decay": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"shared trainable parameter appears twice: {name}")
        seen.add(id(parameter))
        group = "no_decay" if old_no_decay(name, parameter) else "decay"
        row = {"name": name, "numel": int(parameter.numel()), "shape": list(parameter.shape), "group": group}
        rows.append(row); groups[group].append(parameter)
    if not rows or len(seen) != len(rows):
        raise RuntimeError("optimizer coverage is empty or non-unique")
    names = {row["name"] for row in rows}
    if names != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise RuntimeError("optimizer coverage does not include every trainable parameter exactly once")
    # Execute, rather than merely paraphrase, the old helper against this exact
    # decoder.  Its groups must classify every parameter identically.
    from tfpd_exploration.src.m2_dual_track_v1.training import adamw_param_groups
    old_groups = adamw_param_groups(model.named_parameters(), weight_decay=0.01)
    old_by_id = {id(p): float(group["weight_decay"]) for group in old_groups for p in group["params"]}
    expected_by_id = {id(p): (0.0 if old_no_decay(n, p) else 0.01) for n, p in model.named_parameters() if p.requires_grad}
    if old_by_id != expected_by_id:
        raise RuntimeError("legacy decay predicate diverges from executed old adamw_param_groups")
    meta = []
    for group in ("decay", "no_decay"):
        selected = [row for row in rows if row["group"] == group]
        meta.append({"group": group, "weight_decay": 0.01 if group == "decay" else 0.0,
                     "parameter_count": len(selected), "numel": sum(row["numel"] for row in selected),
                     "names_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    return rows, meta


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--bank-manifest", required=True)
    p.add_argument("--attestation", required=True); p.add_argument("--attestation-sha256", required=True)
    p.add_argument("--arm", choices=("A_flat",), required=True); p.add_argument("--dest", required=True)
    p.add_argument("--stage", choices=("train", "score"), default="train"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume"); p.add_argument("--allow-smoke", action="store_true"); p.add_argument("--max-updates", type=int)
    p.add_argument("--cpu-threads", type=int, default=4)
    return p.parse_args()


def decorate(dest: Path, rows: list[dict[str, Any]], groups: list[dict[str, Any]], cfg: Path) -> None:
    meta_path = dest / "run_meta.json"; meta = json.loads(meta_path.read_text())
    payload = {"schema": RECIPE_SCHEMA, "label": "diagnostic_recipe_not_A1_arm",
               "base_runner": str(Path(base.__file__).resolve()), "base_runner_sha256": BASE_SHA256,
               "recipe_runner_sha256": file_sha256(Path(__file__)), "recipe_config_sha256": file_sha256(cfg),
               "old_rule_source": "/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_dual_track_v1/training.py:49-86",
               "coverage": rows, "groups": groups,
               "coverage_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "coverage_verified_exactly_once": True, "old_helper_group_parity_verified": True}
    meta.update(diagnostic_recipe_not_A1_arm=True, legacy_decay_bridge=payload)
    base.atom(meta_path, meta)
    for name in ("train_receipt.json", "smoke_receipt.json", "score_receipt.json"):
        path = dest / name
        if path.is_file():
            receipt = json.loads(path.read_text()); receipt["diagnostic_recipe_not_A1_arm"] = True
            receipt["legacy_decay_bridge"] = payload; base.atom(path, receipt)


def main() -> None:
    args = parse(); cfg = Path(args.config).resolve(); config = json.loads(cfg.read_text())
    if config.get("diagnostic_recipe") != RECIPE_SCHEMA:
        raise RuntimeError("requires the frozen legacy-decay bridge config")
    frozen = Path(config["source_frozen_config"]).resolve()
    if file_sha256(frozen) != config.get("source_frozen_config_sha256"):
        raise RuntimeError("rev5 frozen source config SHA drift")
    if file_sha256(Path(base.__file__)) != BASE_SHA256:
        raise RuntimeError("base v2 runner SHA changed; recipe must be reviewed, not silently reused")
    dest = Path(args.dest).resolve()
    if args.stage == "score":
        result = base.score_stage(args)
        meta = json.loads((dest / "run_meta.json").read_text()); bridge = meta.get("legacy_decay_bridge")
        if not isinstance(bridge, dict): raise RuntimeError("score requires bridge-trained run metadata")
        decorate(dest, bridge["coverage"], bridge["groups"], cfg)
        print(json.dumps(result, indent=2, sort_keys=True)); return
    captured: dict[str, Any] = {}
    original_build, original_adamw = base.build_decoder, base.torch.optim.AdamW
    def build(*a: Any, **kw: Any):
        model = original_build(*a, **kw); rows, groups = coverage(model); captured.update(model=model, rows=rows, groups=groups); return model
    def adamw(parameters: Any, *a: Any, **kw: Any):
        model = captured.get("model")
        if model is None: raise RuntimeError("bridge AdamW constructed before decoder coverage")
        supplied = list(parameters); expected = [p for _, p in model.named_parameters() if p.requires_grad]
        if {id(p) for p in supplied} != {id(p) for p in expected} or len(supplied) != len(expected):
            raise RuntimeError("bridge AdamW parameter iterable differs from all decoder trainable parameters")
        groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not old_no_decay(n, p)], "weight_decay": float(kw.pop("weight_decay"))},
                  {"params": [p for n, p in model.named_parameters() if p.requires_grad and old_no_decay(n, p)], "weight_decay": 0.0}]
        if not all(group["params"] for group in groups): raise RuntimeError("unexpected empty legacy optimizer group")
        return original_adamw(groups, *a, **kw)
    base.build_decoder, base.torch.optim.AdamW = build, adamw
    try:
        result = base.run(args)
    finally:
        base.build_decoder, base.torch.optim.AdamW = original_build, original_adamw
    decorate(dest, captured["rows"], captured["groups"], cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
