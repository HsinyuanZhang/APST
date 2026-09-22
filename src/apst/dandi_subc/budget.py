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
from . import protocol
from . import training as T
from .carrier import apply_carrier_normalizer, estimate_move_t4
from .common import digest, load_records, score_predictions, sha256, atomic_json
from .data import SessionData, load_cached_session
from .final_access import FinalAccess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


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


def _runs(seal: Path, full_cell: str | None = None, act_cell: str | None = None) -> dict[str, dict[str, Any]]:
    doc=json.loads(Path(seal).read_text()); cells=doc.get("cell_selections", doc.get("cells", {})); out={}
    for label, method in METHODS:
        wanted = full_cell if method == "dual_site" else act_cell
        item = cells.get(wanted) if wanted else None
        matches=[v for v in cells.values() if isinstance(v,dict) and v.get("method")==method and v.get("temporal")=="F0" and int(v.get("seed",42))==42]
        item = item or (matches[0] if len(matches)==1 else None)
        if wanted is None and len(matches)!=1: raise ValueError(f"ambiguous or missing F0 seed42 selection for {method}; pass explicit cell")
        if not isinstance(item,dict): raise ValueError(f"seal lacks selected {method}")
        checkpoint=Path(item["checkpoint"]); stats_path=Path(item["source_stats"])
        if not checkpoint.is_file() or not stats_path.is_file() or sha256(checkpoint)!=item.get("checkpoint_sha256"): raise ValueError("selected checkpoint binding mismatch")
        model,stats,meta=T.load_checkpoint(checkpoint,device="cpu")
        if meta.get("method")!=method or stats.get("sha256")!=json.loads(stats_path.read_text()).get("sha256"): raise ValueError("selected method/stats mismatch")
        out[method]={"label":label,"checkpoint":checkpoint,"checkpoint_sha256":sha256(checkpoint),"stats":stats,"stats_path":stats_path,"stats_sha256":sha256(stats_path)}
    return out

def _cache_records(cap: Any, cache: Path, receipt_path: Path) -> list[SessionData]:
    receipt=json.loads(receipt_path.read_text());
    if receipt.get("seal_sha256") != sha256(cap.manifest_path): raise PermissionError("final cache receipt/seal mismatch")
    rows=[]
    for sid in protocol.FINAL_SESSIONS:
        item=receipt.get("sessions",{}).get(sid,{ }); path=Path(item.get("file",cache/f"{sid}.sua.npz"))
        if not path.is_file() or sha256(path)!=item.get("sha256"): raise PermissionError(f"cache hash mismatch: {sid}")
        rows.append(load_cached_session(path,final_access=cap))
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


def smoke(device: str="cpu") -> dict[str,Any]:
    rng=np.random.default_rng(42); n,u=128,8
    record=SessionData("smoke","dev","sua",rng.poisson(1,(n,u)).astype(np.float32),rng.normal(size=(n,2)).astype(np.float32),np.arange(n+1),np.array([127]),np.arange(64),np.arange(8,41),rng.poisson(1,(33,100,u)).astype(np.float32),rng.poisson(1,(33,u)).astype(np.float32),np.linspace(0,6,33),np.arange(u),np.arange(u),np.ones(u,bool),{"array_sha256":{"query_indices":"smoke"}})
    stats={"carrier":{"mean":np.zeros(4),"std":np.ones(4)},"velocity_mean":[0,0],"velocity_std":[1,1]}; rows=[]
    for label,method in METHODS:
      model=T._build(method,"F0",42,"train" if method=="dual_site" else "pretrain",None).to(device).eval()
      for budget in BUDGETS:
       pred,detail=_predict(model,record,stats,method,budget)
       if pred is None or pred.shape!=(1,2): raise RuntimeError("budget smoke prediction failed")
       rows.append({"method":label,"budget":budget,"profile_none":detail["act_profile_is_none"]})
    return {"status":"SMOKE_COMPLETED","rows":rows,"final_sessions_opened":0}

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--seal",type=Path);p.add_argument("--final-cache",type=Path);p.add_argument("--final-receipt",type=Path);p.add_argument("--dest",type=Path);p.add_argument("--device",default="cpu");p.add_argument("--smoke",action="store_true");p.add_argument("--full-cell");p.add_argument("--act-cell");a=p.parse_args(argv)
 if a.smoke: print(json.dumps(smoke(a.device)));return
 if not all((a.seal,a.final_cache,a.final_receipt,a.dest)): raise ValueError("--seal --final-cache --final-receipt --dest required")
 if a.dest.exists(): raise FileExistsError("budget destination must be fresh")
 cap=FinalAccess.from_manifest(a.seal);cap.validate(); records=_cache_records(cap,a.final_cache,a.final_receipt); bindings=_runs(a.seal,a.full_cell,a.act_cell); rows=[]
 for label,method in METHODS:
  model,stats,_=T.load_checkpoint(bindings[method]["checkpoint"],device=a.device)
  for r in records:
   for budget in BUDGETS:
    cap.validate(); pred,detail=_predict(model,r,stats,method,budget); row={"method":label,"model_method":method,"budget":budget,"session_id":r.session_id,"source_path":str(bindings[method]["checkpoint"]),"sha256":bindings[method]["checkpoint_sha256"],"split":"final","seed":42,"query_indices_sha256":r.metadata["array_sha256"]["query_indices"],"truth_sha256":digest(np.ascontiguousarray(r.velocity[r.query_indices])),**detail}
    if pred is not None:
     m=score_predictions(r,pred); row.update({"r2":m["r2"],"r2_x":m["r2_per_output"][0],"r2_y":m["r2_per_output"][1],"n_queries":m["n_queries"]})
    rows.append(row)
 a.dest.mkdir(parents=True,exist_ok=False); fields=sorted({k for x in rows for k in x if k!="diagnostics"})
 with (a.dest/"final_rows.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows([{k:v for k,v in x.items() if k in fields} for x in rows])
 summary=[]
 for label,method in METHODS:
  for budget in BUDGETS:
   group=[x for x in rows if x["method"]==label and x["budget"]==budget]; scored=[x for x in group if x["status"]=="SCORED"]
   summary.append({"method":label,"budget":budget,"n_expected":6,"n_scored":len(scored),"status":"SCORED" if len(scored)==6 else "UNAVAILABLE","mean_r2":float(np.mean([x["r2"] for x in scored])) if len(scored)==6 else None})
 with (a.dest/"summary.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 atomic_json(a.dest/"receipt.json",{"schema":"dandi_subc_budget_v1","status":"FINAL_SCORED","rows":len(rows),"summary_rows":len(summary),"seal_sha256":sha256(a.seal),"csv_sha256":sha256(a.dest/"final_rows.csv"),"summary_sha256":sha256(a.dest/"summary.csv"),"final_sessions_opened":len(records)})
if __name__=="__main__": main()
