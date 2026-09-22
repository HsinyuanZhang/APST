"""Source/dev-only trainer for the DANDI F0 FiLM study.

This module intentionally has no final-data loader or final-score entry point.
The final gate is a separate, sealed concern.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent

from apst.models.ema import DecoderEMA
from apst.models.frontend import unit_dropout_seed, whole_unit_dropout
from apst.models.schedule import warmup_cosine_lr
from apst.dandi_subm import protocol
from apst.dandi_subm.carrier import apply_carrier_normalizer, estimate_move_t4
from apst.dandi_subm.subm_common import (PairedSampler, aggregate_scores, atomic_json, digest,
                                         fit_source_stats, load_records, record_binding,
                                         require_full_source, score_predictions, sha256)
from apst.dandi_subm.subm_data import SessionData, padded_windows

SCHEMA = "dandi688_subm_f0_film_v1"
METHODS = frozenset(("act_only", "identity_only", "token_only", "dual_site"))
TEMPORALS = frozenset(("F0", "flat"))
CONDITIONS = frozenset(("intact", "e0_shuffle", "profile_shuffle", "joint_permutation"))


def _json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_json(path, dict(value))


def _hash_source() -> dict[str, str]:
    paths = [_HERE / name for name in ("training.py", "model.py", "contract.py", "subm_common.py", "subm_data.py")]
    return {str(p.relative_to(_HERE.parent.parent)): sha256(p) for p in paths if p.is_file()}


def _runtime_hashes() -> dict[str, str]:
    """Hash every release-package Python module actually imported by this run."""
    paths = set()
    for module in list(sys.modules.values()):
        path = Path(getattr(module, "__file__", "") or "")
        if path.is_file() and path.suffix == ".py" and "apst" in path.resolve().parts:
            paths.add(path.resolve())
    paths.update((_HERE / name for name in ("contract.py", "model.py", "training.py", "subm_common.py", "subm_data.py")))
    return {str(path): sha256(path) for path in sorted(paths) if path.is_file()}


def _recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    recipe = dict(config.get("recipe", {}))
    required = {"segments": 24, "updates_per_segment": 3165, "batch": 32,
                "lr_peak": 3e-4, "lr_min_factor": .1, "warmup_updates": 3165,
                "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8,
                "grad_clip": 1., "ema_decay": .9995, "whole_unit_dropout": .1}
    if recipe != required:
        raise ValueError("F0 FiLM recipe differs from the frozen 24x3165 contract")
    return recipe


def _validate_config(config: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    c = dict(config)
    if c.get("schema") != SCHEMA or c.get("stage") not in {"pretrain", "train"}:
        raise ValueError("F0 FiLM schema/stage mismatch")
    if c.get("method") not in METHODS or c.get("temporal") not in TEMPORALS:
        raise ValueError("unknown F0 FiLM method or temporal condition")
    if c.get("representation") != "sua" or c.get("seed") != 42:
        raise ValueError("invalid representation or preregistered seed")
    if c.get("token_profile") is not (c["method"] in {"token_only", "dual_site"}) or c.get("target_parameter_updates") is not False:
        raise ValueError("method/token-profile or target-update contract mismatch")
    if c.get("encoder_family") not in {"activity", "concat"}:
        raise ValueError("encoder_family must be activity or concat")
    expected_family = "activity" if c["method"] in {"act_only", "token_only"} else "concat"
    if c["encoder_family"] != expected_family:
        raise ValueError("method/encoder_family binding mismatch")
    if c["stage"] == "pretrain":
        if c["method"] not in {"act_only", "identity_only"} or c["temporal"] != "F0" or c["seed"] != 42:
            raise ValueError("pretraining is seed42 F0 and only activity/concat family")
    elif c["method"] != "dual_site" and c["temporal"] != "F0":
        raise ValueError("only dual_site is compared across F0 versus flat; all ablations are F0")
    try:
        from apst.dandi_subm.contract import validate
        c = dict(validate(c))
    except ImportError:
        pass
    if c.get("output_ema") != {"alpha": 1 / 3, "training": False, "clock": "full_recording", "reset": "session", "state_0": "pred_0"}:
        raise ValueError("output EMA contract mismatch")
    requested = c.get("evaluation_conditions", ["intact"])
    if not isinstance(requested, list) or (c["stage"] == "train" and not requested) or not set(requested) <= CONDITIONS:
        raise ValueError("invalid evaluation conditions")
    _recipe(c)
    cache = Path(c.get("cache", ""))
    if not cache.is_absolute() or not (cache / "prepared_receipt.json").is_file():
        raise ValueError("cache must be an absolute prepared source/dev cache")
    if not smoke and (c.get("source_ids") is not None or c.get("dev_ids") is not None):
        raise ValueError("formal runs require complete frozen source/dev rosters")
    return c


def _build(method: str, temporal: str, seed: int, stage: str, encoder_state: Mapping[str, Any] | None):
    from apst.dandi_subm.model import build_model
    return build_model(method, temporal, seed=seed, stage=stage, encoder_state=encoder_state)


def _units(model: nn.Module) -> int:
    return int(getattr(model, "units", 100))


def build_optimizer(model: nn.Module, recipe: Mapping[str, Any]) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Build the frozen two-group AdamW contract and expose its exact coverage."""
    required = {"lr_peak", "weight_decay", "betas", "eps"}
    if not required <= set(recipe):
        raise ValueError("optimizer recipe lacks a required field")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    slope = [(name, parameter) for name, parameter in named if name.endswith("slope_log")]
    regular = [(name, parameter) for name, parameter in named if not name.endswith("slope_log")]
    temporal = getattr(model, "temporal_kind", None)
    if temporal == "F0" and not slope:
        raise RuntimeError("F0 model lacks trainable slope_log")
    if temporal == "flat" and slope:
        raise RuntimeError("flat model must not advertise a trainable slope_log")
    all_ids = {id(parameter) for _, parameter in named}
    if not regular or len(all_ids) != len(named) or {id(p) for _, p in regular + slope} != all_ids:
        raise RuntimeError("optimizer trainable coverage is incomplete or overlapping")
    optimizer = torch.optim.AdamW(
        [{"params": [p for _, p in regular], "weight_decay": float(recipe["weight_decay"]), "lr_multiplier": 1.0,
          "group_name": "trunk_decay"},
         {"params": [p for _, p in slope], "weight_decay": 0., "lr_multiplier": 1.0,
          "group_name": "recency_no_decay"}],
        lr=float(recipe["lr_peak"]), betas=tuple(recipe["betas"]), eps=float(recipe["eps"]),
    )
    receipt = {"groups": [{"name": "trunk_decay", "members": [n for n, _ in regular],
                            "weight_decay": float(recipe["weight_decay"]), "lr_multiplier": 1.0},
                           {"name": "recency_no_decay", "members": [n for n, _ in slope],
                            "weight_decay": 0., "lr_multiplier": 1.0}],
               "trainable": [n for n, _ in named], "frozen": frozen,
               "coverage": {"trainable_count": len(named), "frozen_count": len(frozen),
                            "complete_nonoverlapping": True}, "temporal": temporal}
    return optimizer, receipt


def _carrier(record: SessionData, stats: Mapping[str, Any], device: torch.device, n_pad: int) -> torch.Tensor:
    count = record.neural.shape[1]
    if count > n_pad:
        raise ValueError("DANDI channel count exceeds model padding")
    raw = estimate_move_t4(record.carrier_counts, record.carrier_angles)
    normalized = apply_carrier_normalizer(raw, stats["carrier"])
    result = np.zeros((n_pad, 4), np.float32); result[:count] = normalized
    return torch.from_numpy(result).to(device)


def _activity(record: SessionData, device: torch.device, n_pad: int) -> torch.Tensor:
    value = np.zeros((protocol.ACTIVITY_TRIALS, 100, n_pad), np.float32)
    value[:, :, :record.neural.shape[1]] = record.activity
    return torch.from_numpy(value).to(device)


def _zero_carrier(model: nn.Module, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros((_units(model), 4), device=device, dtype=dtype)


def _forward(model: nn.Module, x: torch.Tensor, activity: torch.Tensor, carrier: torch.Tensor | None,
             *, unit_mask: torch.Tensor, dropout_keep: torch.Tensor | None, valid: torch.Tensor) -> torch.Tensor:
    # The model owns the site routing. ACT rejects a carrier; each other method
    # gets the one normalized profile it may route to E0, token, both, or neither.
    supplied = None if getattr(model, "method", None) == "act_only" else carrier
    return model(x, activity, supplied, unit_mask=unit_mask, dropout_keep=dropout_keep,
                 input_valid_mask=valid)


def _derangement(count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("shuffle condition requires at least two observed units")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    # Deterministic cyclic repair turns any fixed points into a derangement.
    fixed = np.flatnonzero(order == np.arange(count))
    if len(fixed) == 1:
        other = 0 if fixed[0] != 0 else 1
        order[fixed[0]], order[other] = order[other], order[fixed[0]]
    elif len(fixed) > 1:
        order[fixed] = np.roll(order[fixed], 1)
    if np.any(order == np.arange(count)):
        raise RuntimeError("deterministic derangement construction failed")
    return order


def _ema_full_clock(values: np.ndarray, alpha: float = 1 / 3) -> np.ndarray:
    values = np.asarray(values, np.float64)
    result = np.empty_like(values)
    previous = None
    for index, row in enumerate(values):
        previous = row if previous is None else alpha * previous + (1. - alpha) * row
        result[index] = previous
    return result.astype(np.float32)


def _full_clock_windows(record: SessionData, endpoints: np.ndarray, n_pad: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal 50-bin windows for every native recording bin, with left padding."""
    endpoints = np.asarray(endpoints, np.int64)
    if endpoints.ndim != 1 or np.any(endpoints < 0) or np.any(endpoints >= record.neural.shape[0]):
        raise ValueError("full-clock endpoint is outside record")
    x = np.zeros((len(endpoints), protocol.WINDOW_BINS, n_pad), np.float32)
    valid = np.zeros((len(endpoints), protocol.WINDOW_BINS), bool)
    for row, end in enumerate(endpoints):
        start = max(0, int(end) - protocol.WINDOW_BINS + 1); width = int(end) - start + 1
        x[row, -width:, :record.neural.shape[1]] = record.neural[start:int(end) + 1]
        valid[row, -width:] = True
    mask = np.zeros(n_pad, bool); mask[:record.neural.shape[1]] = True
    return x, valid, mask


def predict_record(model: nn.Module, record: SessionData, stats: Mapping[str, Any], *, condition: str = "intact",
                   shuffle_seed: int = 42, batch: int = 32, smoke_bins: int | None = None) -> tuple[np.ndarray, dict]:
    """Predict every valid causal bin, apply reset-per-recording EMA, then gather Q50.

    This deliberately never uses final loaders: callers must pass an already
    authorized source/dev ``SessionData`` record.
    """
    if condition not in CONDITIONS or batch < 1:
        raise ValueError("invalid prediction condition or batch")
    device = next(model.parameters()).device
    model.eval()
    all_endpoints = np.arange(record.neural.shape[0], dtype=np.int64)
    if smoke_bins is not None:
        # A smoke may only score a prefix whose Q50 members are explicitly retained.
        all_endpoints = all_endpoints[:int(smoke_bins)]
        if not len(all_endpoints): raise ValueError("smoke_bins yields no causal bins")
    activity = _activity(record, device, _units(model))
    method = getattr(model, "method", None)
    profile = None if method == "act_only" else _carrier(record, stats, device, _units(model))
    n_real = record.neural.shape[1]
    order = _derangement(n_real, shuffle_seed)
    diagnostics: dict[str, Any] = {"condition": condition, "clock": "full_recording_causal_bins",
                                    "ema_alpha": 1 / 3, "session_reset": True,
                                    "real_units": n_real, "endpoint_count": int(len(all_endpoints))}
    if condition == "profile_shuffle":
        if profile is None: raise ValueError("ACT has no profile-shuffle condition")
        profile = profile.clone(); profile[:n_real] = profile[:n_real][torch.from_numpy(order).to(device)]
    if condition == "joint_permutation":
        # Permute neural/activity/profile together.  This is an equivariance check;
        # the same output coordinates are expected after the set frontend.
        activity = activity.clone(); activity[:, :, :n_real] = activity[:, :, :n_real][:, :, torch.from_numpy(order).to(device)]
        if profile is not None:
            profile = profile.clone(); profile[:n_real] = profile[:n_real][torch.from_numpy(order).to(device)]
    # E0 is session calibration, never a query-batch computation.  Materialize
    # it once after any legal profile/permutation transformation.
    with torch.inference_mode():
        base_e0 = model.encode(activity, None if method in {"act_only", "token_only"} else profile)
    if condition == "e0_shuffle":
        base_e0 = base_e0.clone(); base_e0[:n_real] = base_e0[:n_real][torch.from_numpy(order).to(device)]
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(all_endpoints), batch):
            endpoints = all_endpoints[start:start + batch]
            x, valid, unitmask = _full_clock_windows(record, endpoints, _units(model))
            if condition == "joint_permutation":
                x = x.copy(); x[:, :, :n_real] = x[:, :, :n_real][:, :, order]
            tx = torch.from_numpy(x).to(device); tv = torch.from_numpy(valid).to(device)
            um = torch.from_numpy(unitmask).to(device)
            token = profile if method in {"token_only", "dual_site"} else None
            value = model.forward_identity(tx, base_e0, token, unit_mask=um, dropout_keep=None, input_valid_mask=tv)
            predictions.append(value.float().cpu().numpy())
    raw = np.concatenate(predictions).astype(np.float32)
    physical = raw * np.asarray(stats["velocity_std"], np.float32) + np.asarray(stats["velocity_mean"], np.float32)
    filtered = _ema_full_clock(physical)
    position = {int(endpoint): index for index, endpoint in enumerate(all_endpoints)}
    if any(int(q) not in position for q in record.query_indices):
        raise ValueError("full-clock smoke prefix does not cover all Q50 query endpoints")
    query = filtered[np.asarray([position[int(q)] for q in record.query_indices], np.int64)]
    diagnostics.update({"raw_full_clock_sha256": digest(raw), "ema_full_clock_sha256": digest(filtered),
                        "query_indices_sha256": record.metadata["array_sha256"]["query_indices"],
                        "prediction_sha256": digest(query), "derangement": order.tolist()})
    return query, diagnostics


def _save(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temp); os.replace(temp, path)


def _state_hash(module: nn.Module) -> str:
    digestor = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digestor.update(name.encode()); digestor.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digestor.hexdigest()


def _frozen_trunk_hash(module: nn.Module) -> str:
    digestor = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name.startswith("film."):
            continue
        digestor.update(name.encode()); digestor.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digestor.hexdigest()


def _encoder_payload(model: nn.Module, metadata: Mapping[str, Any]) -> dict[str, Any]:
    state = {name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()}
    return {"schema": SCHEMA + "_encoder", "encoder_state": state, "receipt_binding": digest(metadata)}


def _load_encoder(path: Path, config: Mapping[str, Any], stats: Mapping[str, Any], *, allow_smoke: bool) -> Mapping[str, Any]:
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file(): raise ValueError("encoder checkpoint/receipt missing")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("checkpoint_sha256") != sha256(path) or receipt.get("schema") != SCHEMA + "_encoder":
        raise ValueError("encoder checkpoint hash/schema mismatch")
    if receipt.get("representation") != config["representation"] or receipt.get("encoder_family") != config["encoder_family"]:
        raise ValueError("encoder representation/family mismatch")
    if receipt.get("source_stats_sha256") != stats["sha256"] or receipt.get("source_sessions") != list(protocol.TRAIN_SESSIONS):
        raise ValueError("encoder source/stat binding mismatch")
    if not allow_smoke and (receipt.get("status") != "FORMAL" or receipt.get("actual_updates") != 24 * 3165):
        raise ValueError("formal training requires a completed formal 18-source encoder")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    core = {key: value for key, value in receipt.items() if key != "checkpoint_sha256"}
    if payload.get("receipt_binding") != digest(core) or not isinstance(payload.get("encoder_state"), dict):
        raise ValueError("encoder payload/receipt binding mismatch")
    return payload["encoder_state"]


def run_training(config: dict, dest: Path, *, encoder_path: Path | None = None, device: str = "cpu",
                 smoke_updates: int | None = None, smoke_dev_replay: bool = False) -> dict:
    smoke = smoke_updates is not None
    if smoke and not 1 <= int(smoke_updates) <= 8: raise ValueError("smoke_updates must be 1..8")
    c, recipe = _validate_config(config, smoke=smoke), _recipe(config)
    dest = Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()): raise FileExistsError("destination must be new/empty")
    dest.mkdir(parents=True)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    records = load_records(Path(c["cache"]), c["representation"], "train", session_ids=c.get("source_ids"))
    if not smoke: require_full_source(records)
    stats = fit_source_stats(records, smoke=smoke); _json(dest / "source_stats.json", stats)
    encoder_state = None
    if c["stage"] == "train":
        if encoder_path is None: raise ValueError("stage=train requires encoder_path")
        encoder_state = _load_encoder(Path(encoder_path), c, stats, allow_smoke=smoke)
    model = _build(c["method"], c["temporal"], int(c["seed"]), c["stage"], encoder_state).to(torch_device)
    runtime_start = _runtime_hashes()
    source_before = _frozen_trunk_hash(model.encoder) if c["stage"] == "train" else None
    frozen_names = {name for name, p in model.encoder.named_parameters() if not p.requires_grad}
    if c["stage"] == "train" and not {name for name, _ in model.encoder.named_parameters() if not name.startswith("film.")} <= frozen_names:
        raise RuntimeError("train stage must freeze the supplied source encoder trunk")
    opt, optimizer_contract = build_optimizer(model, recipe)
    params = [p for p in model.parameters() if p.requires_grad]
    ema = DecoderEMA(model, decay=recipe["ema_decay"])
    if any(f"encoder.{name}" in ema.shadow for name in frozen_names):
        raise RuntimeError("frozen source encoder entered EMA")
    segments, updates, batch = ((1, int(smoke_updates), 2) if smoke else (recipe["segments"], recipe["updates_per_segment"], recipe["batch"]))
    metadata = {"schema": SCHEMA + "_training", "status": "SMOKE" if smoke else "FORMAL", "config": c,
                "config_sha256": digest(c), "stage": c["stage"], "method": c["method"], "temporal": c["temporal"],
                "representation": c["representation"], "seed": c["seed"], "source_sessions": [r.session_id for r in records],
                "source_binding": record_binding(records), "source_stats_sha256": stats["sha256"], "code_hashes": runtime_start,
                "encoder_checkpoint_sha256": sha256(encoder_path) if encoder_path else None,
                "optimizer_contract": optimizer_contract,
                "final_sessions_opened": 0, "actual_budget": {"segments": segments, "updates_per_segment": updates, "batch": batch},
                "frozen_config_file_sha256": sha256(dest / "frozen_config.json") if (dest / "frozen_config.json").is_file() else None}
    _json(dest / "frozen_config.json", c)
    metadata["frozen_config_file_sha256"] = sha256(dest / "frozen_config.json")
    _json(dest / "protocol.json", metadata)
    _json(dest / "optimizer_receipt.json", {"schema": SCHEMA + "_optimizer", "status": metadata["status"],
                                               "config_sha256": metadata["config_sha256"],
                                               "source_stats_sha256": stats["sha256"],
                                               "source_binding": metadata["source_binding"],
                                               "optimizer": optimizer_contract,
                                               "frozen_config_file_sha256": metadata["frozen_config_file_sha256"],
                                               "final_sessions_opened": 0})
    _json(dest / "run_meta_start.json", {"schema": SCHEMA, "status": "SMOKE_RUNNING" if smoke else "FORMAL_RUNNING",
                                           "machine": socket.gethostname(), "device": str(torch_device), "started_utc": datetime.now(timezone.utc).isoformat(),
                                           "config_sha256": digest(c), "code_hashes": runtime_start, "final_sessions_opened": 0})
    calibration = {r.session_id: (_activity(r, torch_device, _units(model)),
                                  None if c["method"] == "act_only" else _carrier(r, stats, torch_device, _units(model)))
                   for r in records}
    ymean = torch.tensor(stats["velocity_mean"], device=torch_device); ystd = torch.tensor(stats["velocity_std"], device=torch_device)
    sampler, curve, step, started = PairedSampler(records, int(c["seed"]), batch=batch, updates_per_segment=updates), [], 0, time.monotonic()
    dev = load_records(Path(c["cache"]), c["representation"], "dev") if (not smoke or smoke_dev_replay) and c["stage"] == "train" else []
    best: tuple[float, Path] | None = None
    for segment in range(segments):
        model.train(); losses = []
        for bi, (record, endpoints) in enumerate(sampler.segment()):
            x, valid, mask = padded_windows(record, endpoints, _units(model)); tx = torch.from_numpy(x).to(torch_device)
            tv, um = torch.from_numpy(valid).to(torch_device), torch.from_numpy(mask).to(torch_device).expand(len(endpoints), -1)
            keep = whole_unit_dropout(um, p=recipe["whole_unit_dropout"], generator=torch.Generator(device="cpu").manual_seed(unit_dropout_seed(int(c["seed"]), segment + 1, bi)))
            activity, carrier = calibration[record.session_id]; target = (torch.from_numpy(record.velocity[endpoints]).to(torch_device) - ymean) / ystd
            step += 1; lr = warmup_cosine_lr(step, total_steps=segments * updates, warmup_steps=min(recipe["warmup_updates"], max(1, segments * updates // 2)) if smoke else recipe["warmup_updates"], peak=recipe["lr_peak"], min_factor=recipe["lr_min_factor"])
            for group in opt.param_groups: group["lr"] = lr * float(group.get("lr_multiplier", 1.0))
            opt.zero_grad(set_to_none=True); amp = torch.autocast("cuda", dtype=torch.bfloat16) if torch_device.type == "cuda" else nullcontext()
            with amp: loss = nn.functional.mse_loss(_forward(model, tx, activity, carrier, unit_mask=um, dropout_keep=keep, valid=tv).float(), target.float())
            if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite source loss")
            loss.backward(); nn.utils.clip_grad_norm_(params, recipe["grad_clip"], error_if_nonfinite=True); opt.step(); ema.update_after_step(model); losses.append(float(loss.detach().cpu()))
        checkpoint = dest / f"segment_{segment + 1:02d}.pt"
        raw = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}; ema.apply_to(model)
        try:
            _save(checkpoint, {"schema": SCHEMA + "_checkpoint", "status": metadata["status"], "metadata": metadata, "global_step": step, "model_state": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}, "source_stats": stats})
            development = None
            if dev:
                rows = [score_predictions(r, predict_record(model, r, stats)[0]) for r in dev]
                development = aggregate_scores(rows)
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in raw: p.copy_(raw[n])
        row = {"segment": segment + 1, "global_step": step, "mean_loss": float(np.mean(losses)), "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint), "development": development}; curve.append(row)
        if source_before is not None and _frozen_trunk_hash(model.encoder) != source_before:
            raise RuntimeError("frozen stage-2 source encoder changed after optimizer step")
        if development is not None and np.isfinite(development["mean_r2"]) and (best is None or development["mean_r2"] > best[0]): best = (development["mean_r2"], checkpoint)
        _json(dest / "progress.json", {"global_step": step, "segments": curve, "elapsed_seconds": time.monotonic() - started})
    encoder_checkpoint = None
    if c["stage"] == "pretrain":
        receipt = {"schema": SCHEMA + "_encoder", "status": metadata["status"], "representation": c["representation"], "encoder_family": c["encoder_family"], "source_sessions": [r.session_id for r in records], "source_stats_sha256": stats["sha256"], "actual_updates": step, "config_sha256": digest(c), "code_hashes": _hash_source(), "final_sessions_opened": 0}
        # The loop restores raw parameters after each EMA checkpoint. Export the
        # encoder from the final serialized EMA state, never from restored raw.
        final_payload = torch.load(dest / f"segment_{segments:02d}.pt", map_location="cpu", weights_only=False)
        ema_encoder = {name.removeprefix("encoder."): value for name, value in final_payload["model_state"].items() if name.startswith("encoder.")}
        encoder_checkpoint = dest / "encoder.pt"
        _save(encoder_checkpoint, {"schema": SCHEMA + "_encoder", "encoder_state": ema_encoder, "receipt_binding": digest(receipt)})
        receipt["checkpoint_sha256"] = sha256(encoder_checkpoint); _json(encoder_checkpoint.with_suffix(".json"), receipt)
    if best is not None:
        selection = {"schema": SCHEMA + "_selection", "status": metadata["status"], "rule": "earliest_max_equal_session_dev_r2", "checkpoint": str(best[1]), "checkpoint_sha256": sha256(best[1]), "mean_dev_r2": best[0], "dev_sessions": list(protocol.DEV_SESSIONS), "final_sessions_opened": 0}; _json(dest / "selection.json", selection)
    runtime_end = _runtime_hashes()
    if any(runtime_end.get(path) != value for path, value in runtime_start.items()):
        raise RuntimeError("an execution source changed during training")
    receipt = {**metadata, "completed": True, "global_step": step, "actual_updates": step, "segments": curve, "sampler": sampler.receipt(), "elapsed_seconds": time.monotonic() - started, "ended_utc": datetime.now(timezone.utc).isoformat(), "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
               "frozen_encoder_before_after_equal": source_before is None or _frozen_trunk_hash(model.encoder) == source_before}
    _json(dest / "train_receipt.json", receipt)
    _json(dest / "run_meta.json", {"schema": SCHEMA, "status": "SMOKE_COMPLETED" if smoke else "FORMAL_COMPLETED",
                                     "machine": socket.gethostname(), "device": str(torch_device), "actual_updates": step,
                                     "config_sha256": digest(c), "code_hashes": runtime_end, "source_hashes_unchanged": True, "ended_utc": datetime.now(timezone.utc).isoformat(), "final_sessions_opened": 0})
    if best is not None:
        # Re-open the exact selected EMA payload and record an independently
        # materialized full-clock dev cache.  This is replay evidence, never a
        # second selection pass.
        selected_model, selected_stats, _ = load_checkpoint(best[1], device=device, allow_smoke=smoke)
        replay: dict[str, Any] = {}
        replay_dir = dest / "selected_replay_cache"; replay_dir.mkdir()
        for record in dev:
            prediction, diag = predict_record(selected_model, record, selected_stats)
            np.savez_compressed(replay_dir / f"{record.session_id}.npz", prediction=prediction,
                                query_indices=record.query_indices,
                                truth=record.velocity[record.query_indices])
            replay[record.session_id] = {**score_predictions(record, prediction), **diag,
                                         "cache_sha256": sha256(replay_dir / f"{record.session_id}.npz")}
        replay_metrics = aggregate_scores([{"session_id": sid, "n_queries": row["n_queries"], "r2": row["r2"], "r2_per_output": row["r2_per_output"], "query_indices_sha256": row["query_indices_sha256"], "velocity_sha256": row["velocity_sha256"]} for sid, row in replay.items()])
        if replay_metrics["mean_r2"] != best[0]: raise RuntimeError("selected full-clock replay differs from selection")
        _json(dest / "score_receipt.json", {"schema": SCHEMA + "_development_score", "status": metadata["status"], "selection": json.loads((dest / "selection.json").read_text()), "replay": replay, "replay_metrics": replay_metrics, "final_sessions_opened": 0})
    return receipt


def load_checkpoint(path: Path, *, device: str = "cpu", allow_smoke: bool = False):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA + "_checkpoint": raise ValueError("not an F0 FiLM checkpoint")
    metadata = payload.get("metadata", {})
    if not allow_smoke and metadata.get("status") != "FORMAL": raise ValueError("SMOKE checkpoint cannot enter formal use")
    c = metadata.get("config")
    if not isinstance(c, dict): raise ValueError("checkpoint lacks frozen config")
    model = _build(c["method"], c["temporal"], int(c["seed"]), c["stage"], None).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), payload["source_stats"], metadata
