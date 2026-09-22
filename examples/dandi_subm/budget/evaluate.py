#!/usr/bin/env python3
"""Fixed-checkpoint sub-M calibration-budget evaluation.

This is deliberately separate from ``src/training.py``: it never trains or
selects a checkpoint.  M only changes session calibration input E0 (and the
dual-site association profile), while the selected decoder keeps its normal
full-clock, alpha=1/3 EMA prediction path.
"""
from __future__ import annotations

import argparse, contextlib, csv, hashlib, json, os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from apst.dandi_subm import protocol
from apst.dandi_subm import training as T
from apst.dandi_subm.carrier import apply_carrier_normalizer, estimate_move_t4
from apst.dandi_subm.subm_common import digest, load_records, score_predictions, sha256
from apst.dandi_subm.subm_data import SessionData, load_cached_session
ROOT = protocol.output_root()

BUDGETS = (4, 8, 16, 32)
METHODS = (("APST-FULL", "dual_site"), ("ACT-only", "act_only"))
CONFIG = HERE / "config.json"
CONFIG_SHA = HERE / "config.json.sha256"
SCHEMA = "dandi688_subm_budget_v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return digest(np.ascontiguousarray(value))


def config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text())
    if value.get("schema") != SCHEMA or tuple(value.get("budgets", ())) != BUDGETS:
        raise ValueError("budget config/schema mismatch")
    if value.get("methods") != dict(METHODS):
        raise ValueError("budget method binding mismatch")
    if not CONFIG_SHA.is_file() or CONFIG_SHA.read_text().strip() != file_sha(CONFIG):
        raise ValueError("frozen budget config SHA sidecar mismatch")
    return value


def _rank(angles: np.ndarray) -> int:
    finite = np.asarray(angles)[np.isfinite(angles)]
    if not len(finite):
        return 0
    design = np.stack((np.ones(len(finite)), np.cos(finite), np.sin(finite)), axis=1)
    return int(np.linalg.matrix_rank(design))


def budget_inputs(record: SessionData, method: str, budget: int, stats: dict[str, Any], n_pad: int) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Historic ext_4_8_16_32 input semantics, without refitting statistics."""
    if method not in dict(METHODS).values() or budget not in BUDGETS and budget != 33:
        raise ValueError("unsupported method or calibration budget")
    count = int(record.neural.shape[1])
    if count > n_pad or record.activity.shape[0] < budget:
        raise ValueError("invalid session geometry/calibration availability")
    activity = np.zeros((budget, 100, n_pad), np.float32)
    activity[:, :, :count] = record.activity[:budget]
    if method == "act_only":
        return activity, None, 3
    rank = _rank(record.carrier_angles[:budget])
    if rank < 3:
        return activity, None, rank
    # estimate_move_t4 itself converts the 0.5 s MOVE sum to its native 20 ms
    # count scale.  Do not apply a second Hz or seconds conversion here.
    raw = estimate_move_t4(record.carrier_counts[:budget], record.carrier_angles[:budget])
    normalized = apply_carrier_normalizer(raw, stats["carrier"])
    carrier = np.zeros((n_pad, 4), np.float32)
    carrier[:count] = normalized
    return activity, carrier, rank


@contextlib.contextmanager
def patched_calibration(activity: np.ndarray, carrier: np.ndarray | None) -> Iterator[None]:
    """Inject only local budget inputs into the unchanged prediction routine."""
    old_activity, old_carrier = T._activity, T._carrier
    T._activity = lambda record, device, units: torch.from_numpy(activity).to(device)
    T._carrier = lambda record, stats, device, units: None if carrier is None else torch.from_numpy(carrier).to(device)
    try:
        yield
    finally:
        T._activity, T._carrier = old_activity, old_carrier


def _runs() -> dict[str, dict[str, Any]]:
    cfg = config()
    result: dict[str, dict[str, Any]] = {}
    for label, method in METHODS:
        run = (ROOT / cfg["selection_runs"][method]).resolve()
        selection, receipt, stats_path = run / "selection.json", run / "train_receipt.json", run / "source_stats.json"
        if not all(p.is_file() for p in (selection, receipt, stats_path)):
            raise FileNotFoundError(f"formal selected run missing for {label}: {run}")
        sel, rec, stats = json.loads(selection.read_text()), json.loads(receipt.read_text()), json.loads(stats_path.read_text())
        checkpoint = Path(sel.get("checkpoint", ""))
        if (sel.get("status") != "FORMAL" or sel.get("rule") != "earliest_max_equal_session_dev_r2"
                or sel.get("dev_sessions") != list(protocol.DEV_SESSIONS) or not checkpoint.is_file()
                or file_sha(checkpoint) != sel.get("checkpoint_sha256") or rec.get("status") != "FORMAL"
                or rec.get("completed") is not True or rec.get("final_sessions_opened") != 0
                or stats.get("sha256") != rec.get("source_stats_sha256")):
            raise ValueError(f"selected checkpoint/source-stat binding invalid for {label}")
        model, checkpoint_stats, metadata = T.load_checkpoint(checkpoint, device="cpu")
        if metadata.get("method") != method or checkpoint_stats.get("sha256") != stats.get("sha256"):
            raise ValueError(f"checkpoint method/stat drift for {label}")
        result[method] = {"label": label, "run": run, "selection": selection, "selection_sha256": file_sha(selection),
                          "checkpoint": checkpoint, "checkpoint_sha256": file_sha(checkpoint), "stats": checkpoint_stats,
                          "stats_path": stats_path, "stats_sha256": file_sha(stats_path)}
    return result


def _cache_records(cap: Any) -> list[SessionData]:
    receipt_path = ROOT / "final" / "final_cache_receipt.json"
    cache = ROOT / "final" / "final_cache"
    if not receipt_path.is_file():
        raise FileNotFoundError("sealed final cache receipt is missing")
    receipt = json.loads(receipt_path.read_text())
    seal_sha = file_sha(cap.path)
    if receipt.get("schema") != "dandi688_subm_final_cache_v1" or receipt.get("seal_sha256") != seal_sha:
        raise PermissionError("final cache receipt/seal mismatch")
    if receipt.get("protocol") != protocol.protocol_dict() or receipt.get("final_sessions_opened") != 3:
        raise PermissionError("final cache protocol/roster mismatch")
    rows = []
    for sid in protocol.FINAL_SESSIONS:
        entry = receipt.get("sessions", {}).get(sid, {})
        path = Path(entry.get("file", cache / f"{sid}.sua.npz"))
        if not path.is_file() or file_sha(path) != entry.get("sha256"):
            raise PermissionError(f"final cache file hash mismatch: {sid}")
        rows.append(load_cached_session(path, final_access=cap))
    return rows


def _predict(model: torch.nn.Module, record: SessionData, stats: dict[str, Any], method: str, budget: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    activity, carrier, rank = budget_inputs(record, method, budget, stats, T._units(model))
    base = {"rank_D": rank, "activity_sha256": array_sha(activity), "carrier_sha256": None if carrier is None else array_sha(carrier),
            "act_profile_is_none": carrier is None}
    if method == "dual_site" and rank < 3:
        return None, {**base, "status": "UNAVAILABLE_RANK_LT3"}
    with patched_calibration(activity, carrier):
        prediction, diagnostics = T.predict_record(model, record, stats, condition="intact")
    return prediction, {**base, "status": "SCORED", "diagnostics": diagnostics}


def smoke(device: str) -> dict[str, Any]:
    """Dev-only admission smoke; no model checkpoint/final cache is opened."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    records = load_records(ROOT / "prepared_sua", "sua", "dev", session_ids=[protocol.DEV_SESSIONS[0]])
    record = records[0]
    # Source-only stats are available from the formal encoder restart and are
    # enough to test the exact calibration construction, not scoring/selection.
    stats_path = ROOT / "runs" / "sua_concat_pretrain_s42" / "source_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError("source-only stats unavailable for dev smoke")
    stats = json.loads(stats_path.read_text())
    n = 100
    old_a = T._activity(record, torch.device("cpu"), n).cpu().numpy()
    old_c = T._carrier(record, stats, torch.device("cpu"), n).cpu().numpy()
    a33, c33, r33 = budget_inputs(record, "dual_site", 33, stats, n)
    if not np.array_equal(old_a, a33) or c33 is None or not np.array_equal(old_c, c33):
        raise RuntimeError("M33 calibration parity with training helper failed")
    rows = []
    qhash = array_sha(record.query_indices)
    for label, method in METHODS:
        for budget in BUDGETS:
            activity, carrier, rank = budget_inputs(record, method, budget, stats, n)
            if array_sha(record.query_indices) != qhash:
                raise RuntimeError("budget changed query indices")
            if method == "act_only" and carrier is not None:
                raise RuntimeError("ACT-only constructed a profile")
            rows.append({"method": label, "budget": budget, "rank_D": rank,
                         "activity_sha256": array_sha(activity), "carrier_sha256": None if carrier is None else array_sha(carrier),
                         "query_indices_sha256": qhash, "act_profile_is_none": carrier is None})
    # Exercise the actual local injection/prediction path on a causal dev
    # prefix. Random approved geometry is interface evidence only; it is not a
    # score, checkpoint selection, or formal model evaluation.
    # This is expressly a nonprotocol causal prefix.  The separate admission
    # checks above establish that the real Q50 array is unchanged by budget.
    # Keeping it to 128 real bins makes this a mechanism smoke, not a hidden
    # partial development evaluation.
    prefix_bins = 128
    short = replace(record, neural=record.neural[:prefix_bins], velocity=record.velocity[:prefix_bins],
                    bin_edges=record.bin_edges[:prefix_bins + 1], query_indices=np.asarray([prefix_bins - 1], np.int64))
    predictor_smoke = []
    for label, method in METHODS:
        stage = "train" if method == "dual_site" else "pretrain"
        model = T._build(method, "F0", 42, stage, None).to(device).eval()
        prediction, detail = _predict(model, short, stats, method, 4)
        if prediction is None or prediction.shape != (1, 2) or detail["status"] != "SCORED":
            raise RuntimeError("budget prediction injection smoke failed")
        if method == "act_only" and not detail["act_profile_is_none"]:
            raise RuntimeError("ACT prediction smoke received a profile")
        predictor_smoke.append({"method": label, "budget": 4,
                                "prefix_end_inclusive": int(short.query_indices[0]), "surface": "SMOKE_PREFIX_NONPROTOCOL",
                                "prediction_shape": list(prediction.shape),
                                "act_profile_is_none": detail["act_profile_is_none"],
                                "clock": detail["diagnostics"]["clock"],
                                "ema_alpha": detail["diagnostics"]["ema_alpha"]})
    dest = ROOT / "budget" / "smoke_dev_predict_v1"
    if dest.exists():
        raise FileExistsError(f"smoke destination already exists: {dest}")
    dest.mkdir()
    receipt = {"schema": SCHEMA + "_smoke_v1", "status": "SMOKE_COMPLETED", "split": "dev", "session": record.session_id,
               "source_stats": str(stats_path), "source_stats_sha256": file_sha(stats_path), "M33_exact_activity_parity": True,
               "M33_exact_profile_parity": True, "M33_rank_D": r33, "budgets": list(BUDGETS), "rows": rows,
               "query_indices_unchanged": True, "ACT_profile_never_constructed": True,
               "predictor_smoke": predictor_smoke, "final_sessions_opened": 0,
               "checkpoint_opened": False, "selection": False}
    atomic_json(dest / "receipt.json", receipt)
    return {"receipt": str(dest / "receipt.json"), "rows": len(rows), "final_sessions_opened": 0}


def evaluate(split: str, device: str) -> dict[str, Any]:
    if split not in {"dev", "final"}:
        raise ValueError("split must be dev or final")
    cfg, bindings = config(), _runs()
    cap = None
    if split == "dev":
        records = load_records(ROOT / "prepared_sua", "sua", "dev")
        dest = ROOT / "budget" / "dev_results"
    else:
        from apst.dandi_subm.final_access import FinalAccess
        seal = ROOT / "final" / "selection_seal.json"
        cap = FinalAccess.from_manifest(seal)
        records = _cache_records(cap)
        dest = ROOT / "budget" / "final_results"
    if dest.exists():
        raise FileExistsError(f"budget {split} result destination already exists")
    dest.mkdir()
    rows: list[dict[str, Any]] = []
    try:
        for label, method in METHODS:
            bind = bindings[method]
            model, stats, _ = T.load_checkpoint(bind["checkpoint"], device=device)
            for record in records:
                for budget in BUDGETS:
                    if cap is not None:
                        # Current capability validates the sealed artifact set on
                        # construction; re-open it per final cell to catch drift.
                        cap = __import__("final_access").FinalAccess.from_manifest(cap.path)
                    prediction, detail = _predict(model, record, stats, method, budget)
                    row = {"method": label, "model_method": method, "budget": budget, "session_id": record.session_id,
                           "split": split, "checkpoint": str(bind["checkpoint"]), "checkpoint_sha256": bind["checkpoint_sha256"],
                           "source_path": str(bind["checkpoint"]), "sha256": bind["checkpoint_sha256"], "seed": 42,
                           "source_stats": str(bind["stats_path"]), "source_stats_sha256": bind["stats_sha256"],
                           "query_indices_sha256": record.metadata["array_sha256"]["query_indices"],
                           "truth_sha256": array_sha(record.velocity[record.query_indices]), **detail}
                    if prediction is not None:
                        metric = score_predictions(record, prediction)
                        npz = dest / f"{method}.M{budget}.{record.session_id}.npz"
                        np.savez_compressed(npz, prediction=prediction, query_indices=record.query_indices,
                                            truth=record.velocity[record.query_indices])
                        row.update({**metric, "prediction_path": str(npz), "prediction_sha256": file_sha(npz),
                                    "prediction_array_sha256": array_sha(prediction)})
                    rows.append(row)
        fields = sorted({key for row in rows for key in row if key != "diagnostics"})
        with (dest / "rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows([{key: row.get(key) for key in fields} for row in rows])
        summary_rows = []
        for label, method in METHODS:
            for budget in BUDGETS:
                cell = [r for r in rows if r["method"] == label and r["budget"] == budget]
                scored = [r for r in cell if r["status"] == "SCORED"]
                complete = len(cell) == len(records) and len(scored) == len(records)
                mean = float(np.mean([r["r2"] for r in scored])) if scored else ""
                summary_rows.append({"method": label, "model_method": method, "budget": budget, "split": split, "seed": 42,
                                     "source_path": str(bindings[method]["checkpoint"]), "sha256": bindings[method]["checkpoint_sha256"],
                                     "n_expected": len(records), "n_scored": len(scored),
                                     "status": "COMPLETE" if complete else "INCOMPLETE_UNAVAILABLE",
                                     "equal_session_mean_r2": mean if complete else "", "available_mean_r2": mean})
        with (dest / "summary.csv").open("w", newline="") as handle:
            fieldnames = ["method", "model_method", "budget", "split", "seed", "source_path", "sha256", "n_expected", "n_scored", "status", "equal_session_mean_r2", "available_mean_r2"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(summary_rows)
        receipt = {"schema": SCHEMA + "_results_v1", "status": "FINAL_SCORED" if split == "final" else "DEV_SCORED",
                   "split": split, "config": str(CONFIG), "config_sha256": file_sha(CONFIG), "config_sha256_sidecar": str(CONFIG_SHA), "budget_grid": list(BUDGETS),
                   "methods": dict(METHODS), "prediction_contract": cfg["prediction"],
                   "source_statistics_refit": False, "selected_checkpoint_reselection": False,
                   "rows": len(rows), "expected_rows": len(METHODS) * len(BUDGETS) * len(records),
                   "unavailable_rows": sum(r["status"] != "SCORED" for r in rows), "csv": str(dest / "rows.csv"),
                   "csv_sha256": file_sha(dest / "rows.csv"), "summary_csv": str(dest / "summary.csv"), "summary_csv_sha256": file_sha(dest / "summary.csv"), "selection_bindings": {m: {k: v for k, v in b.items() if k in {"selection_sha256", "checkpoint_sha256", "stats_sha256"}} for m, b in bindings.items()},
                   "seal": None if cap is None else str(ROOT / "final" / "selection_seal.json"),
                   "seal_sha256": None if cap is None else file_sha(cap.path),
                   "final_cache_receipt": None if split == "dev" else str(ROOT / "final" / "final_cache_receipt.json"),
                   "final_sessions_opened": 0 if split == "dev" else len(records)}
        atomic_json(dest / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        atomic_json(dest / "failure.json", {"schema": SCHEMA, "status": "FAILED", "split": split, "error": repr(exc),
                                              "rows_completed": len(rows), "final_sessions_opened": 0 if split == "dev" else len(records)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "final"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="dev-only M33/input-contract smoke; opens no checkpoint or final data")
    args = parser.parse_args()
    if args.smoke:
        if args.split is not None:
            raise ValueError("--smoke does not accept --split")
        print(json.dumps(smoke(args.device), sort_keys=True))
    elif args.split is None:
        raise ValueError("--split dev|final is required unless --smoke")
    else:
        print(json.dumps(evaluate(args.split, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
