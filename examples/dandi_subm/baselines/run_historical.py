#!/usr/bin/env python3
"""Historical SUA WF-FSS/PCA-WF/FA-WF on the fixed sub-M 2015 split.

CPU-only; selection never reads final arrays.  This is an independent copy of
the sealed Fig. 3 local-WF semantics, adapted only to the new prepared cache.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from apst.legacy_dandi.numerics import FactorModel, causal_lags, fit_ridge_grid, smooth_raw
from apst.dandi_subm import protocol

CAMPAIGN = protocol.output_root()
PREPARED = CAMPAIGN / "prepared_sua"
OUT = CAMPAIGN / "baselines" / "results"
SOURCE = tuple(f"sub-M_ses-CO-{x}" for x in ("20150511", "20150512", "20150610", "20150611", "20150612", "20150615"))
DEV = tuple(f"sub-M_ses-CO-{x}" for x in ("20150616", "20150617"))
FINAL = tuple(f"sub-M_ses-CO-{x}" for x in ("20150623", "20150625", "20150626"))
ALPHAS = (1e2, 1e3, 1e4, 1e5)
HISTORIES = (1, 10, 50)
DIMS = (4, 8, 16)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

@dataclass
class Record:
    session_id: str
    neural: np.ndarray
    velocity: np.ndarray
    support_indices: np.ndarray
    carrier_indices: np.ndarray
    query_indices: np.ndarray
    source_path: str
    source_sha256: str

def load(sid: str) -> Record:
    path = PREPARED / f"{sid}.sua.npz"
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        need = ("neural", "velocity", "support_indices", "carrier_indices", "query_indices")
        missing = [x for x in need if x not in z]
        if missing: raise ValueError(f"{path}: missing {missing}")
        r = Record(sid, *(np.asarray(z[x]) for x in need), str(path.resolve()), digest(path))
    if r.neural.ndim != 2 or r.velocity.ndim != 2 or len(r.neural) != len(r.velocity): raise ValueError(f"{sid}: neural/velocity shape mismatch")
    for name in ("support_indices", "carrier_indices", "query_indices"):
        idx = getattr(r, name)
        if idx.ndim != 1 or not len(idx) or np.any(idx < 0) or np.any(idx >= len(r.neural)): raise ValueError(f"{sid}: invalid {name}")
    return r

def score(record: Record, pred: np.ndarray) -> float:
    truth = record.velocity[record.query_indices].astype(np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != truth.shape: raise ValueError(f"{record.session_id}: prediction shape mismatch")
    sse = ((truth - pred) ** 2).sum(0); sst = ((truth - truth.mean(0)) ** 2).sum(0)
    return float(1.0 - sse.sum() / sst.sum())

def aggregate(records: list[Record], pred: dict[str, np.ndarray]) -> dict:
    sessions = [{"session_id": r.session_id, "r2": score(r, pred[r.session_id])} for r in records]
    return {"mean_r2": float(np.mean([x["r2"] for x in sessions])), "sessions": sessions}

def grid() -> list[dict]:
    out = [{"method": "wf_fss", "history_bins": h, "smooth": s, "alpha": a}
           for h in HISTORIES for s in (False, True) for a in ALPHAS]
    out += [{"method": m, "components": k, "history_bins": h, "smooth": s, "alpha": a}
            for m in ("pca_wf", "fa_wf") for k in DIMS for h in HISTORIES for s in (False, True) for a in ALPHAS]
    return out

def masked_stream(r: Record, smooth: bool) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(len(r.neural), bool); mask[r.support_indices] = True
    stream = np.zeros_like(r.neural, dtype=np.float64); stream[mask] = r.neural[mask]
    if smooth: stream = smooth_raw(stream)
    stream[~mask] = 0.0
    return stream, mask

def fit_factor(x: np.ndarray, k: int) -> FactorModel:
    first = FactorModel.fit(x, k, seed=42, max_iter=10_000, tol=1e-6, n_init=3, noise_floor=1e-6)
    if first.diagnostics["converged"]: return first
    retry = FactorModel.fit(x, k, seed=42, max_iter=50_000, tol=1e-6, n_init=3, noise_floor=1e-6)
    retry.diagnostics["initial_attempt"] = first.diagnostics
    if not retry.diagnostics["converged"]: raise RuntimeError("FA did not converge within sealed retry budget 50000")
    return retry

@dataclass
class Model:
    config: dict; mean: np.ndarray; scale: np.ndarray; transform: Any; readout: Any; latent: np.ndarray | None; labels: np.ndarray; carrier: np.ndarray

def fit(r: Record, cfg: dict) -> Model:
    stream, mask = masked_stream(r, bool(cfg["smooth"]))
    support = stream[r.support_indices]; mean = support.mean(0); scale = support.std(0); scale[scale < 1e-6] = 1.0
    z = (stream - mean) / scale
    if cfg["method"] == "wf_fss":
        z[~mask] = 0.0
        x = causal_lags(z, r.carrier_indices, history_bins=int(cfg["history_bins"]))
        return Model(dict(cfg), mean, scale, None, fit_ridge_grid(x, r.velocity[r.carrier_indices], [cfg["alpha"]])[cfg["alpha"]], z, r.velocity[r.carrier_indices], r.carrier_indices)
    k = int(cfg["components"])
    if not 0 < k < min(len(r.support_indices), r.neural.shape[1]): raise ValueError(f"rank {k} invalid: requires k < min(M33 rows={len(r.support_indices)}, units={r.neural.shape[1]})")
    from sklearn.decomposition import PCA
    trans = PCA(n_components=k, svd_solver="full").fit(z[r.support_indices]) if cfg["method"] == "pca_wf" else fit_factor(z[r.support_indices], k)
    latent = trans.transform(z); latent[~mask] = 0.0
    x = causal_lags(latent, r.carrier_indices, history_bins=int(cfg["history_bins"]))
    return Model(dict(cfg), mean, scale, trans, fit_ridge_grid(x, r.velocity[r.carrier_indices], [cfg["alpha"]])[cfg["alpha"]], latent, r.velocity[r.carrier_indices], r.carrier_indices)

def predict(model: Model, r: Record) -> np.ndarray:
    stream = smooth_raw(r.neural.astype(np.float64)) if model.config["smooth"] else r.neural.astype(np.float64)
    z = (stream - model.mean) / model.scale
    values = z if model.transform is None else model.transform.transform(z)
    return np.asarray(model.readout.predict(causal_lags(values, r.query_indices, history_bins=int(model.config["history_bins"]))), np.float32)

def binding(records: list[Record]) -> dict:
    return {r.session_id: {"path": r.source_path, "sha256": r.source_sha256, "units": int(r.neural.shape[1]), "support_rows": int(len(r.support_indices))} for r in records}

def select() -> None:
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    receipt = PREPARED / "prepared_receipt.json"
    if not receipt.is_file(): raise FileNotFoundError(receipt)
    source = [load(x) for x in SOURCE]; dev = [load(x) for x in DEV]
    rows = {m: [] for m in ("wf_fss", "pca_wf", "fa_wf")}; winners: dict[str, dict] = {}
    for cfg in grid():
        try:
            models = {r.session_id: fit(r, cfg) for r in dev}
            pred = {r.session_id: predict(models[r.session_id], r) for r in dev}
            metrics = aggregate(dev, pred); row = {"config": cfg, "eligible": True, "score": metrics["mean_r2"], "metrics": metrics}
            if cfg["method"] not in winners or row["score"] > winners[cfg["method"]]["score"]:
                winners[cfg["method"]] = {**row, "models": models, "pred": pred}
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e: row = {"config": cfg, "eligible": False, "reason": str(e)}
        rows[cfg["method"]].append(row)
    OUT.mkdir(); selected = {}; artifacts = {}
    for method in rows:
        if method not in winners:
            selected[method] = {"status": "UNAVAILABLE", "reason": "no eligible candidate"}; continue
        win = winners[method]; selected[method] = {k: v for k, v in win.items() if k not in {"models", "pred"}}
        with (OUT / f"{method}.dev_models.pkl").open("wb") as f: pickle.dump(win["models"], f)
        np.savez_compressed(OUT / f"{method}.dev_predictions.npz", **win["pred"])
        artifacts[method] = {"models_sha256": digest(OUT / f"{method}.dev_models.pkl"), "predictions_sha256": digest(OUT / f"{method}.dev_predictions.npz")}
    status = "DEV_SELECTED" if all(m in winners for m in rows) else "DEV_SELECTED_WITH_UNAVAILABLE"
    payload = {"schema": "dandi688_subm2015_historical_sua_wf_v1", "status": status, "final_loaded": False, "source_sessions": list(SOURCE), "dev_sessions": list(DEV), "final_sessions": list(FINAL), "prepared_receipt": str(receipt.resolve()), "prepared_receipt_sha256": digest(receipt), "source_binding": binding(source), "dev_binding": binding(dev), "candidate_grid": rows, "selected": selected, "artifacts": artifacts, "supervision": "target-local dense physical velocity at M33 carrier_indices for every method; PCA/FA transform uses target M33 neural support", "code_sha256": digest(Path(__file__).resolve())}
    atomic(OUT / "selection.json", payload)
    print(json.dumps({"status": status, "selected": selected}, indent=2))

def final() -> None:
    selection_path = OUT / "selection.json"; claim = OUT / ".final_claim"
    if not selection_path.is_file(): raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text())
    if selection.get("status") not in {"DEV_SELECTED", "DEV_SELECTED_WITH_UNAVAILABLE"} or selection.get("final_loaded") is not False: raise ValueError("selection is not an untouched no-final development selection")
    if selection.get("code_sha256") != digest(Path(__file__).resolve()): raise ValueError("baseline code drift since selection")
    claim.open("x").write(digest(selection_path) + "\n")
    final_records = [load(x) for x in FINAL]
    results = {}
    for method, choice in selection["selected"].items():
        if choice.get("status") == "UNAVAILABLE": results[method] = choice; continue
        cfg = choice["config"]; models = {r.session_id: fit(r, cfg) for r in final_records}; pred = {r.session_id: predict(models[r.session_id], r) for r in final_records}
        np.savez_compressed(OUT / f"{method}.final_predictions.npz", **pred)
        results[method] = {"status": "SCORED", "config": cfg, "metrics": aggregate(final_records, pred), "predictions_sha256": digest(OUT / f"{method}.final_predictions.npz")}
    receipt = {"schema": "dandi688_subm2015_historical_sua_wf_final_v1", "status": "FINAL_SCORED", "selection_sha256": digest(selection_path), "final_binding": binding(final_records), "results": results, "supervision": selection["supervision"]}
    atomic(OUT / "final_score_receipt.json", receipt)
    print(json.dumps({k: v.get("metrics", {}).get("mean_r2") for k, v in results.items()}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage", choices=("select", "final")); args = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
    {"select": select, "final": final}[args.stage]()
