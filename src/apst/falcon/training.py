"""Two-stage APST trainer for FALCON M1/M2/H1.

Stage 1 (``pretrain``) trains the association-profile trunk ``A`` with a
fresh flat-temporal supervision decoder on held-in source recordings;
selection uses development held-out sessions (equal-session mean of the raw
decode).

Stage 2 (``stage2``) freezes ``A``, optionally trains the rank-8 carrier
FiLM between pooling and ``post_pool``, and trains a fresh F0 decoder
(noPE temporal, ladder-scaled learned slopes) over the frozen banks.
Selection uses the inference contract: causal output EMA with
``alpha = 1/3`` over the full recording clock, earliest finite maximum of
the task's development metric.  Target-session parameters are never
updated.
"""
from __future__ import annotations

import contextlib
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import contract, data, protocol
from .common import (atomic_json, centered_variance_weighted_r2,
                     earliest_finite_maximum, epoch_batches, fresh_directory,
                     grouped_seven_r2, jsonable, sha256, source_hashes)
from .profile import Stage2Joint, Supervision, export_bank, unit_mask_for

SCHEMA = contract.SCHEMA


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _trunk_state(joint) -> dict[str, torch.Tensor]:
    """Detached copy of the (frozen) pooling trunk tensors."""
    state = {f"pre_pool.{k}": v.detach().clone()
             for k, v in joint.a.pre_pool.state_dict().items()}
    state.update({f"post_pool.{k}": v.detach().clone()
                  for k, v in joint.a.post_pool.state_dict().items()})
    return state


def _keep_mask(units: int, batch: int, dropout: float, seed: int, device) -> torch.Tensor:
    from apst.models.frontend import whole_unit_dropout
    generator = torch.Generator()
    generator.manual_seed(seed)
    keep = whole_unit_dropout(torch.ones(units, dtype=torch.bool), p=dropout, generator=generator)
    return keep.to(device).expand(batch, -1)


def _pretrain_score(model: Supervision, sessions, device, max_windows=None) -> dict:
    model.eval()
    per_session = {}
    with torch.no_grad():
        for name in sorted(sessions):
            session = sessions[name]
            activity = torch.as_tensor(session.support, device=device)
            carrier = torch.as_tensor(session.carrier, device=device)
            preds, truths = [], []
            limit = len(session.starts) if max_windows is None else min(max_windows, len(session.starts))
            for lo in range(0, limit, 32):
                ids = np.arange(lo, min(lo + 32, limit), dtype=np.int64)
                x, y, valid = session.windows(ids, 1.0)
                keep = torch.ones(len(ids), session.carrier.shape[0], dtype=torch.bool, device=device)
                pred = model(torch.as_tensor(x, device=device), activity, carrier,
                             torch.as_tensor(valid, device=device), keep)
                preds.append(pred.float().cpu().numpy())
                truths.append(y)
            prediction = np.concatenate(preds).astype(np.float64)
            target = np.concatenate(truths).astype(np.float64)
            scale = float(protocol.task_protocol(session.task)["behavior_scale"])
            per_session[name] = centered_variance_weighted_r2(prediction, target * scale)
    values = list(per_session.values())
    if not values or not all(np.isfinite(values)):
        return {"equal_session_r2": float("nan"), "per_session": per_session}
    return {"equal_session_r2": float(np.mean(values)), "per_session": per_session}


def _full_clock_score(joint: Stage2Joint, session, pooled, device,
                      max_windows=None) -> dict[str, np.ndarray]:
    """Decode every bin in order, EMA(alpha=1/3) the outputs, gather queries."""
    spec = protocol.task_protocol(session.task)
    scale = float(spec["behavior_scale"])
    context = int(spec["context_bins"])
    carrier = torch.as_tensor(session.carrier, device=device)
    total = session.x.shape[0] - context
    limit = total if max_windows is None else min(max_windows, total)
    query_positions = {int(v): index for index, v in enumerate(session.starts)}
    predictions = np.zeros((len(session.starts), int(spec["outputs"])), np.float64)
    filled = np.zeros(len(session.starts), bool)
    targets = session.y.astype(np.float64)
    state: np.ndarray | None = None
    with torch.no_grad():
        for lo in range(0, limit, 32):
            ends = np.arange(lo, min(lo + 32, limit), dtype=np.int64)
            starts = ends  # padded-timeline window starts equal raw endpoints
            x = np.ascontiguousarray(session.x[starts[:, None] + np.arange(context)], np.float32)
            valid = torch.as_tensor(starts[:, None] + np.arange(context) >= session.pad, device=device)
            keep = torch.ones(len(ends), session.carrier.shape[0], dtype=torch.bool, device=device)
            pred = joint(torch.as_tensor(x, device=device), pooled, carrier, valid, keep)
            raw = pred.float().cpu().numpy().astype(np.float64)
            for row, end in zip(raw, ends):
                state = row if state is None else (1.0 / 3.0) * state + (2.0 / 3.0) * row
                if int(end) in query_positions:
                    index = query_positions[int(end)]
                    predictions[index] = state / scale
                    filled[index] = True
    if max_windows is None and not filled.all():
        raise RuntimeError("full-clock scoring left query predictions unset")
    return {"prediction": predictions, "target": targets}


def _dev_metric(task: str, per_key: dict[str, dict[str, np.ndarray]]) -> float:
    if task == "h1":
        return grouped_seven_r2(per_key, protocol.h1_groups())
    values = [centered_variance_weighted_r2(row["prediction"], row["target"])
              for row in per_key.values()]
    if not values or not all(np.isfinite(values)):
        raise ValueError("development scoring requires finite per-session R2")
    return float(np.mean(values))


def _swap_ema(model: nn.Module, ema) -> dict:
    raw = {name: value.detach().clone() for name, value in model.named_parameters()}
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in ema.shadow:
                value.copy_(ema.shadow[name].to(value.device, value.dtype))
    return raw


def _restore(model: nn.Module, raw: dict) -> None:
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(raw[name])


def _write_meta(dest: Path, config: dict, status: str, stage: str) -> dict:
    meta = {
        "schema": SCHEMA, "stage": stage, "status": status, "task": config["task"],
        "seed": config["seed"], "film": config["film"],
        "ladder_scale": config["ladder_scale"], "started_utc": _now(),
        "machine": socket.gethostname(),
        "config_sha256": sha256(dest / "frozen_config.json"),
        "code_sha256": source_hashes(),
        "target_parameter_updates": False,
        "output_ema": {"alpha": 1.0 / 3.0, "training": False, "clock": "full_recording",
                       "reset": "session", "state_0": "pred_0"},
    }
    atomic_json(dest / "run_meta_start.json", meta)
    return meta


def run_pretrain(config: dict, dest: Path, *, device: str = "cpu",
                 smoke_updates: int | None = None) -> dict:
    """Stage 1: train the association-profile trunk with supervision."""
    if config["stage"] != "pretrain":
        raise ValueError("run_pretrain requires config stage=pretrain")
    smoke = smoke_updates is not None
    if smoke and not 1 <= smoke_updates <= 8:
        raise ValueError("smoke_updates must be in 1..8")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    out = fresh_directory(dest)
    shutil.copy2(config["_path"], out / "frozen_config.json")
    task = config["task"]
    spec = protocol.task_protocol(task)
    source = data.load_sessions(config["cache"], task, "source")
    dev = data.load_sessions(config["cache"], task, "dev")
    meta = _write_meta(out, config, "RUNNING_SMOKE" if smoke else "RUNNING", "pretrain")

    model = Supervision(task, seed=config["seed"]).to(torch_device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["recipe"]["lr_peak"],
                                  weight_decay=config["recipe"]["weight_decay"],
                                  betas=tuple(config["recipe"]["betas"]),
                                  eps=config["recipe"]["eps"])
    from apst.models.ema import DecoderEMA
    ema = DecoderEMA(model, decay=config["recipe"]["ema_decay"])
    total_batches = data.updates_per_epoch(config["cache"], task)
    epochs = 1 if smoke else int(config["recipe"]["epochs"])
    schedule_epochs = config["recipe"].get("cosine_schedule_epochs") or epochs
    warmup = config["recipe"]["warmup_updates"] or total_batches
    rows = []
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for name, ids in epoch_batches(source, epoch, config["recipe"]["batch"], config["seed"]):
            session = source[name]
            x, y, valid = session.windows(ids, float(spec["behavior_scale"]))
            activity = torch.as_tensor(session.support, device=torch_device)
            carrier = torch.as_tensor(session.carrier, device=torch_device)
            keep = _keep_mask(session.carrier.shape[0], len(ids),
                              config["recipe"]["unit_dropout"], config["seed"] + epoch * 100000 + step,
                              torch_device)
            step += 1
            from apst.models.schedule import warmup_cosine_lr
            lr = warmup_cosine_lr(step, total_steps=int(schedule_epochs) * total_batches,
                                  warmup_steps=int(warmup), peak=config["recipe"]["lr_peak"],
                                  min_factor=config["recipe"]["cosine_min_factor"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch_device):
                prediction = model(torch.as_tensor(x, device=torch_device), activity, carrier,
                                   torch.as_tensor(valid, device=torch_device), keep)
                loss = nn.functional.mse_loss(prediction.float(), torch.as_tensor(y, device=torch_device))
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config["recipe"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            ema.update_after_step(model)
            losses.append(float(loss.detach()))
            if smoke and step >= smoke_updates:
                break
        raw = _swap_ema(model, ema)
        score = _pretrain_score(model, dev, torch_device, max_windows=32 if smoke else None)
        _restore(model, raw)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev": score}
        rows.append(row)
        torch.save({"schema": SCHEMA, "stage": "pretrain", "epoch": epoch,
                    "raw_state_dict": model.state_dict(), "ema": ema.state_dict(), "row": row},
                   out / f"epoch_{epoch:03d}.pt")
        if smoke:
            break
    selected = earliest_finite_maximum(rows, "equal_session_r2") if not smoke else rows[-1]["epoch"]
    checkpoint = out / f"epoch_{selected:03d}.pt"
    state = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    model.load_state_dict(state["raw_state_dict"])
    raw = _swap_ema(model, ema)
    ema_state = state["ema"]
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name in ema_state["shadow"]:
                value.copy_(ema_state["shadow"][name].to(torch_device))
    replay = _pretrain_score(model, dev, torch_device, max_windows=32 if smoke else None)
    _restore(model, raw)
    if not smoke and jsonable(replay) != jsonable(rows[selected - 1]["dev"]):
        raise RuntimeError("selected EMA replay drifted from its epoch score")
    receipt = {
        "schema": SCHEMA, "stage": "pretrain", "status": "COMPLETED_SMOKE" if smoke else "COMPLETED",
        "task": task, "seed": config["seed"], "selected_epoch": int(selected),
        "selection_rule": "earliest finite maximum of dev equal-session raw-decode R2",
        "selected_checkpoint_sha256": sha256(checkpoint),
        "selected_replay_equal_session_r2": replay["equal_session_r2"],
        "epochs": epochs, "updates_per_epoch": total_batches, "ended_utc": _now(),
    }
    meta.update(status=receipt["status"], selected_epoch=int(selected), ended_utc=_now())
    atomic_json(out / "run_meta.json", meta)
    atomic_json(out / ("smoke_receipt.json" if smoke else "train_receipt.json"), receipt)
    return receipt


def run_stage2(config: dict, dest: Path, *, encoder_path: Path, device: str = "cpu",
               smoke_updates: int | None = None) -> dict:
    """Stage 2: frozen trunk, optional FiLM, fresh F0 decoder."""
    if config["stage"] != "stage2":
        raise ValueError("run_stage2 requires config stage=stage2")
    smoke = smoke_updates is not None
    if smoke and not 1 <= smoke_updates <= 8:
        raise ValueError("smoke_updates must be in 1..8")
    torch_device = torch.device(device)
    out = fresh_directory(dest)
    shutil.copy2(config["_path"], out / "frozen_config.json")
    encoder_path = Path(encoder_path)
    encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=False)
    if encoder_state.get("schema") != SCHEMA or encoder_state.get("stage") != "pretrain":
        raise ValueError(f"encoder checkpoint is not an APST pretrain artifact: {encoder_path}")
    task = config["task"]
    spec = protocol.task_protocol(task)
    source = data.load_sessions(config["cache"], task, "source")
    dev = data.load_sessions(config["cache"], task, "dev")
    meta = _write_meta(out, config, "RUNNING_SMOKE" if smoke else "RUNNING", "stage2")
    meta["encoder_checkpoint"] = str(encoder_path)
    meta["encoder_checkpoint_sha256"] = sha256(encoder_path)

    joint = Stage2Joint(task, seed=config["seed"], film=config["film"],
                        ladder_scale=float(config["ladder_scale"]),
                        encoder_state=encoder_state["raw_state_dict"]).to(torch_device)
    pooled = {name: joint.a.pooled(torch.as_tensor(s.support, device=torch_device)).detach()
              for name, s in source.items()}
    dev_pooled = {name: joint.a.pooled(torch.as_tensor(s.support, device=torch_device)).detach()
                  for name, s in dev.items()}
    trunk_before = _trunk_state(joint)

    regular = [p for n, p in joint.named_parameters()
               if p.requires_grad and not n.endswith("slope_log")]
    slopes = [p for n, p in joint.named_parameters() if p.requires_grad and n.endswith("slope_log")]
    if joint.film_enabled and not any(n.startswith("a.film") for n, _ in joint.named_parameters()):
        raise RuntimeError("film enabled but no FiLM parameters are trainable")
    optimizer = torch.optim.AdamW(
        [{"name": "regular", "params": regular, "weight_decay": config["recipe"]["weight_decay"]},
         {"name": "recency_no_decay", "params": slopes,
          "weight_decay": config["recipe"]["slope_weight_decay"]}],
        lr=config["recipe"]["lr_peak"], betas=tuple(config["recipe"]["betas"]),
        eps=config["recipe"]["eps"])
    from apst.models.ema import DecoderEMA
    ema = DecoderEMA(joint, decay=config["recipe"]["ema_decay"])
    total_batches = data.updates_per_epoch(config["cache"], task)
    epochs = 1 if smoke else int(config["recipe"]["epochs"])
    schedule_epochs = config["recipe"].get("cosine_schedule_epochs") or epochs
    warmup = config["recipe"]["warmup_updates"] or total_batches
    from apst.models.schedule import warmup_cosine_lr
    rows = []
    step = 0
    for epoch in range(1, epochs + 1):
        joint.train()
        losses = []
        for name, ids in epoch_batches(source, epoch, config["recipe"]["batch"], config["seed"]):
            session = source[name]
            x, y, valid = session.windows(ids, float(spec["behavior_scale"]))
            carrier = torch.as_tensor(session.carrier, device=torch_device)
            keep = _keep_mask(session.carrier.shape[0], len(ids),
                              config["recipe"]["unit_dropout"], config["seed"] + epoch * 100000 + step,
                              torch_device)
            step += 1
            lr = warmup_cosine_lr(step, total_steps=int(schedule_epochs) * total_batches,
                                  warmup_steps=int(warmup), peak=config["recipe"]["lr_peak"],
                                  min_factor=config["recipe"]["cosine_min_factor"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch_device):
                prediction = joint(torch.as_tensor(x, device=torch_device), pooled[name], carrier,
                                   torch.as_tensor(valid, device=torch_device), keep)
                loss = nn.functional.mse_loss(prediction.float(), torch.as_tensor(y, device=torch_device))
            loss.backward()
            nn.utils.clip_grad_norm_(regular + slopes, config["recipe"]["grad_clip"],
                                     error_if_nonfinite=True)
            optimizer.step()
            ema.update_after_step(joint)
            losses.append(float(loss.detach()))
            if smoke and step >= smoke_updates:
                break
        raw = _swap_ema(joint, ema)
        per_key = {name: _full_clock_score(joint, dev[name], dev_pooled[name], torch_device,
                                           max_windows=256 if smoke else None)
                   for name in sorted(dev)}
        metric = _dev_metric(task, per_key)
        _restore(joint, raw)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_metric": metric}
        rows.append(row)
        torch.save({"schema": SCHEMA, "stage": "stage2", "epoch": epoch,
                    "raw_state_dict": joint.state_dict(), "ema": ema.state_dict(), "row": row},
                   out / f"epoch_{epoch:03d}.pt")
        atomic_json(out / "epoch_scores.json", {str(r["epoch"]): r["dev_metric"] for r in rows})
        if smoke:
            break
    after = _trunk_state(joint)
    for key, value in after.items():
        if not torch.equal(value, trunk_before[key]):
            raise RuntimeError(f"frozen association-profile trunk changed during stage 2: {key}")
    selected = rows[-1]["epoch"] if smoke else earliest_finite_maximum(rows, "dev_metric")
    checkpoint = out / f"epoch_{selected:03d}.pt"
    state = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    joint.load_state_dict(state["raw_state_dict"])
    raw = _swap_ema(joint, ema)
    with torch.no_grad():
        for name, value in joint.named_parameters():
            if name in state["ema"]["shadow"]:
                value.copy_(state["ema"]["shadow"][name].to(value.device, value.dtype))
    per_key = {name: _full_clock_score(joint, dev[name], dev_pooled[name], torch_device,
                                       max_windows=256 if smoke else None)
               for name in sorted(dev)}
    replay_metric = _dev_metric(task, per_key)
    _restore(joint, raw)
    if not smoke and replay_metric != rows[selected - 1]["dev_metric"]:
        raise RuntimeError("selected EMA replay drifted from its epoch score")
    receipt = {
        "schema": SCHEMA, "stage": "stage2", "status": "COMPLETED_SMOKE" if smoke else "COMPLETED",
        "task": task, "seed": config["seed"], "film": config["film"],
        "ladder_scale": config["ladder_scale"], "selected_epoch": int(selected),
        "selection_rule": "earliest finite maximum of the task development metric "
                          "under full-clock output EMA alpha=1/3",
        "selected_checkpoint_sha256": sha256(checkpoint),
        "selected_dev_metric": float(replay_metric),
        "frozen_trunk_unchanged": True, "epochs": epochs,
        "updates_per_epoch": total_batches, "ended_utc": _now(),
    }
    meta.update(status=receipt["status"], selected_epoch=int(selected), ended_utc=_now())
    atomic_json(out / "run_meta.json", meta)
    atomic_json(out / ("smoke_receipt.json" if smoke else "train_receipt.json"), receipt)
    return receipt


def calibrate_banks(config: dict, dest: Path, checkpoint: Path, *, split: str = "dev",
                    device: str = "cpu") -> dict:
    """Export closed-form banks for every session of a split from a stage-2 run.

    The trained FiLM is baked offline: ``E0 = post(cat(FiLM(pooled, c), c))``
    computed once under the selected EMA weights, so exported banks feed the
    frozen-bank streaming runtime directly.
    """
    task = config["task"]
    out = fresh_directory(dest)
    state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if state.get("schema") != SCHEMA or state.get("stage") != "stage2":
        raise ValueError(f"not a stage-2 checkpoint: {checkpoint}")
    joint = Stage2Joint(task, seed=config["seed"], film=config["film"],
                        ladder_scale=float(config["ladder_scale"]),
                        encoder_state=None).to(device)
    joint.load_state_dict(state["raw_state_dict"])
    with torch.no_grad():
        for name, value in joint.named_parameters():
            if name in state["ema"]["shadow"]:
                value.copy_(state["ema"]["shadow"][name].to(value.device))
    sessions = data.load_sessions(config["cache"], task, split)
    manifest = {"schema": "apst_falcon_bank_export_v1", "task": task, "split": split,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "selected_epoch": int(state["epoch"]), "sessions": {}}
    for name in sorted(sessions):
        session = sessions[name]
        with torch.no_grad():
            pooled = joint.a.pooled(torch.as_tensor(session.support, device=device))
            e0 = joint.e0(pooled, torch.as_tensor(session.carrier, device=device)).cpu().numpy()
        manifest["sessions"][name] = export_bank(
            out, name, e0, session.carrier, unit_mask_for(task),
            {"estimator": "apst association profile (FiLM baked)" if config["film"]
                          else "apst association profile",
             "budget": int(session.support.shape[0]),
             "support_trials": session.provenance.get("trial_ids"),
             "raw_nwb_sha256": session.provenance.get("raw_nwb_sha256")})
    atomic_json(out / "bank_manifest.json", manifest)
    return manifest
