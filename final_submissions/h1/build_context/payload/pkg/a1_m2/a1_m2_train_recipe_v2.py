#!/usr/bin/env python3
"""Immutable-metadata v2 of the M2 legacy-decay diagnostic recipe.

Base runner SHA-256: 8db2d84683c611796220c379ba4f6badda3b395f1cb839dbbb13fa88aa8d2d3b.
Only AdamW decay partition differs from the base.  v1 remains a historical
smoke artifact and must not be used for a formal launch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

WEEK = Path(__file__).resolve().parents[2]
if str(WEEK / "src") not in sys.path: sys.path.insert(0, str(WEEK / "src"))
from a1_m2 import a1_m2_train_recipe_v1 as rule
from a1_m2 import a1_m2_train_v2 as base

SCHEMA = "apst_m2_stage2_legacy_decay_bridge_v2"
BASE_SHA = "8db2d84683c611796220c379ba4f6badda3b395f1cb839dbbb13fa88aa8d2d3b"


def sha(path: Path) -> str: return rule.file_sha256(path)


def order_hash(items: Any, epoch: int, batch: int, seed: int) -> str:
    """Independent implementation of the frozen source-order contract."""
    import numpy as np
    rng = np.random.default_rng(seed + epoch * 1_000_003); rows = []
    for session in sorted(items):
        order = rng.permutation(len(items[session]["starts"]))
        rows += [(session, order[i:i + batch]) for i in range(0, len(order), batch)]
    rng.shuffle(rows); h = hashlib.sha256()
    for session, ids in rows: h.update(session.encode()); h.update(np.asarray(ids, np.int64).tobytes())
    return h.hexdigest()


def recipe_payload(rows: list[dict[str, Any]], groups: list[dict[str, Any]], cfg: Path, init_sha: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "label": "diagnostic_recipe_not_A1_arm",
            "base_runner": str(Path(base.__file__).resolve()), "base_runner_sha256": BASE_SHA,
            "recipe_runner": str(Path(__file__).resolve()), "recipe_runner_sha256": sha(Path(__file__)),
            "recipe_config_sha256": sha(cfg), "initialization_state_sha256": init_sha,
            "old_rule_source": "/home/xinyuan/Work_host/SPINT/tfpd_exploration/src/m2_dual_track_v1/training.py",
            "old_rule_source_sha256": sha(WEEK.parent / "tfpd_exploration/src/m2_dual_track_v1/training.py"),
            "coverage": rows, "groups": groups,
            "coverage_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "coverage_verified_exactly_once": True, "old_helper_group_parity_verified": True,
            "optimizer_parameter_policy": {"name": "AdamW", "groups": "old adamw_param_groups exact predicate",
              "decay": ".01 for remaining rank>=2 non-bias/non-norm/non-dynamics parameters",
              "no_decay": "0 for rank1, bias, norm/LN/BN, dynamics"}}


def parse():
    args = rule.parse()
    if args.resume: raise RuntimeError("legacy-decay bridge v2 rejects --resume; no inherited resume provenance")
    return args


def main() -> None:
    args = parse(); cfg = Path(args.config).resolve(); config = json.loads(cfg.read_text())
    if config.get("diagnostic_recipe") != SCHEMA: raise RuntimeError("requires v2 bridge config")
    frozen = Path(config["source_frozen_config"]).resolve()
    if sha(frozen) != config["source_frozen_config_sha256"]: raise RuntimeError("rev5 frozen config SHA drift")
    if sha(Path(base.__file__)) != BASE_SHA: raise RuntimeError("base v2 runner SHA drift")
    if args.stage == "score":
        dest = Path(args.dest).resolve(); meta = json.loads((dest / "run_meta.json").read_text())
        bridge = meta.get("legacy_decay_bridge")
        if not isinstance(bridge, dict) or bridge.get("schema") != SCHEMA or bridge.get("recipe_config_sha256") != sha(cfg):
            raise RuntimeError("score refuses an unbound or mismatched bridge train run")
        result = base.score_stage(args)
        receipt = json.loads((dest / "score_receipt.json").read_text()); receipt["diagnostic_recipe_not_A1_arm"] = True
        receipt["legacy_decay_bridge"] = bridge; base.atom(dest / "score_receipt.json", receipt)
        print(json.dumps(result, indent=2, sort_keys=True)); return

    # This construction is solely a preflight fingerprint. fork_rng guarantees
    # it cannot perturb the seed/reset used by base.run for the actual decoder.
    import torch
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        probe = base.build_decoder("A_flat", seed=int(config["seed"]), proj_dim=16).cpu()
        rows, groups = rule.coverage(probe); init_sha = base.state_sha(probe)
    payload = recipe_payload(rows, groups, cfg, init_sha)
    captured: dict[str, Any] = {}; original_build = base.build_decoder; original_adamw = base.torch.optim.AdamW; original_atom = base.atom; original_batches = base.batches; original_load_data = base.load_data

    def load_data(*a: Any, **kw: Any):
        result = original_load_data(*a, **kw)
        planes = result[1]
        captured["batch_order_sha256"] = order_hash(planes["source_train"], 1, base.BATCH, int(config["seed"]))
        return result

    def build(*a: Any, **kw: Any):
        model = original_build(*a, **kw)
        actual = base.state_sha(model)
        if actual != init_sha: raise RuntimeError("actual decoder initialization differs from RNG-isolated bridge preflight")
        captured["model"] = model; return model

    def adamw(parameters: Any, *a: Any, **kw: Any):
        model = captured.get("model")
        if model is None: raise RuntimeError("AdamW before decoder initialization")
        supplied = list(parameters); named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if len(supplied) != len(named) or {id(p) for p in supplied} != {id(p) for _, p in named}: raise RuntimeError("AdamW coverage drift")
        wd = float(kw.pop("weight_decay"));
        if wd != .01: raise RuntimeError("frozen bridge weight decay drift")
        gs = [{"params": [p for n, p in named if not rule.old_no_decay(n, p)], "weight_decay": wd},
              {"params": [p for n, p in named if rule.old_no_decay(n, p)], "weight_decay": 0.0}]
        return original_adamw(gs, *a, **kw)

    def batches(items: Any, epoch: int, batch_size: int, seed: int):
        result = original_batches(items, epoch, batch_size, seed)
        if epoch == 1:
            h = hashlib.sha256()
            for session, ids in result: h.update(session.encode()); h.update(__import__("numpy").asarray(ids, __import__("numpy").int64).tobytes())
            actual, expected = h.hexdigest(), captured["batch_order_sha256"]
            if actual != expected: raise RuntimeError("base batch order drift against independent frozen-order implementation")
        return result

    def atom(path: Path, value: Any):
        bound = dict(payload)
        if "batch_order_sha256" in captured: bound["first_epoch_batch_order_sha256"] = captured["batch_order_sha256"]
        if path.name in {"run_meta.json", "run_meta_start.json"}:
            value = dict(value)
            value["diagnostic_recipe_not_A1_arm"] = True; value["legacy_decay_bridge"] = bound
            value["optimizer_parameter_policy"] = bound["optimizer_parameter_policy"]
        elif path.name in {"smoke_receipt.json", "train_receipt.json"}:
            value = dict(value); value["diagnostic_recipe_not_A1_arm"] = True; value["legacy_decay_bridge"] = bound
        return original_atom(path, value)

    base.build_decoder, base.torch.optim.AdamW, base.batches, base.atom, base.load_data = build, adamw, batches, atom, load_data
    try:
        result = base.run(args)
    finally:
        base.build_decoder, base.torch.optim.AdamW, base.batches, base.atom, base.load_data = original_build, original_adamw, original_batches, original_atom, original_load_data
    # The receipt hashes the metadata written through the hook.  Add no fields
    # afterward; verify this invariant before returning.
    dest = Path(args.dest).resolve(); meta_sha = sha(dest / "run_meta.json")
    for name in ("smoke_receipt.json", "train_receipt.json"):
        p = dest / name
        if p.is_file():
            receipt = json.loads(p.read_text())
            if receipt.get("run_meta_sha256") != meta_sha: raise RuntimeError("receipt final-meta SHA mismatch")
    ck = dest / "epoch_001.pt"
    if ck.is_file():
        state = torch.load(ck, map_location="cpu", weights_only=False)
        if state.get("run_meta_start_sha256") != sha(dest / "run_meta_start.json"): raise RuntimeError("checkpoint start-meta SHA mismatch")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
