"""SUA-only local PCA/FA transforms and dense-label Wiener filters.

M33 calibration is always masked before optional causal smoothing and masked
again afterwards.  Latent rows are masked again before carrier lags are made,
so neither a carrier feature nor any of its history can contain activity from
outside the allowed M33 support.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any
import numpy as np
from sklearn.decomposition import PCA
from apst.legacy_dandi.numerics import FactorModel, RidgeReadout, causal_lags, fit_ridge_grid, smooth_raw

ALPHAS = (1e2, 1e3, 1e4, 1e5)
HISTORIES = (1, 10, 50)
DIMS = (4, 8, 16)
FROZEN_WF_FSS = {"method": "wf_fss", "history_bins": 10, "smooth": False, "alpha": 1e4}


def grid():
    return [
        {"method": m, "components": k, "history_bins": h, "smooth": s, "alpha": a}
        for m in ("pca_wf", "fa_wf")
        for k in DIMS
        for h in HISTORIES
        for s in (False, True)
        for a in ALPHAS
    ]


def _field(r, n):
    return getattr(r, n) if hasattr(r, n) else r[n]


def _array(x, n):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError(f"{n} must be finite nonempty [time,channel]")
    return x


def _indices(x, n, name):
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 1 or not len(x) or np.any(x < 0) or np.any(x >= n):
        raise ValueError(f"invalid {name}")
    return x


def _factor(x, k):
    first = FactorModel.fit(
        x, k, seed=42, max_iter=10_000, tol=1e-6, n_init=3, noise_floor=1e-6
    )
    if first.diagnostics["converged"]:
        return first
    retry = FactorModel.fit(
        x, k, seed=42, max_iter=50_000, tol=1e-6, n_init=3, noise_floor=1e-6
    )
    retry.diagnostics["initial_attempt"] = first.diagnostics
    if not retry.diagnostics["converged"]:
        raise RuntimeError("FA did not converge within sealed retry budget 50000")
    return retry


def _transform(method, x, k):
    if not 0 < k < min(x.shape):
        raise ValueError("components must be less than M33 rows and SUA units")
    return (
        PCA(n_components=k, svd_solver="full").fit(x)
        if method == "pca_wf"
        else (
            _factor(x, k)
            if method == "fa_wf"
            else (_ for _ in ()).throw(ValueError("method must be pca_wf or fa_wf"))
        )
    )


def _calibration(record, smooth):
    raw = _array(_field(record, "neural"), "neural")
    support = _indices(_field(record, "support_indices"), len(raw), "support_indices")
    mask = np.zeros(len(raw), bool)
    mask[support] = True
    masked = np.zeros_like(raw)
    masked[mask] = raw[mask]
    stream = smooth_raw(masked) if smooth else masked
    stream[~mask] = 0.0
    return raw, stream, mask, support


@dataclass
class LocalSUAWF:
    config: dict
    mean_: np.ndarray
    scale_: np.ndarray
    transform: Any
    readout: Any
    latent_calibration: np.ndarray
    carrier_indices: np.ndarray
    fit_features: np.ndarray
    fit_labels: np.ndarray


def fit_local_wf(record: Any, config: dict) -> LocalSUAWF:
    if str(_field(record, "representation")) != "sua":
        raise ValueError("SUA extension accepts SUA records only")
    if (
        set(config) != {"method", "components", "history_bins", "smooth", "alpha"}
        or config not in grid()
    ):
        raise ValueError("configuration is outside frozen PCA/FA-WF grid")
    raw, stream, mask, support = _calibration(record, bool(config["smooth"]))
    carrier = _indices(_field(record, "carrier_indices"), len(raw), "carrier_indices")
    mean = stream[support].mean(0)
    scale = stream[support].std(0)
    scale[scale < 1e-6] = 1.0
    transformer = _transform(
        config["method"], (stream[support] - mean) / scale, int(config["components"])
    )
    latent = transformer.transform((stream - mean) / scale)
    latent[~mask] = 0.0
    feat = causal_lags(latent, carrier, history_bins=int(config["history_bins"]))
    labels = _array(_field(record, "velocity"), "velocity")[carrier]
    return LocalSUAWF(
        dict(config),
        mean,
        scale,
        transformer,
        fit_ridge_grid(feat, labels, [config["alpha"]])[config["alpha"]],
        latent,
        carrier,
        feat,
        labels,
    )


def refit_local(model: LocalSUAWF, config: dict) -> LocalSUAWF:
    if {k: v for k, v in config.items() if k not in {"history_bins", "alpha"}} != {
        k: v for k, v in model.config.items() if k not in {"history_bins", "alpha"}
    } or config not in grid():
        raise ValueError("refit changes frozen transform configuration")
    f = causal_lags(
        model.latent_calibration,
        model.carrier_indices,
        history_bins=int(config["history_bins"]),
    )
    r = fit_ridge_grid(f, model.fit_labels, [config["alpha"]])[config["alpha"]]
    return replace(model, config=dict(config), readout=r, fit_features=f)


def predict_local_wf(model: LocalSUAWF, record: Any) -> np.ndarray:
    if str(_field(record, "representation")) != "sua":
        raise ValueError("SUA extension accepts SUA records only")
    raw = _array(_field(record, "neural"), "neural")
    q = _indices(_field(record, "query_indices"), len(raw), "query_indices")
    stream = smooth_raw(raw) if model.config["smooth"] else raw
    latent = model.transform.transform((stream - model.mean_) / model.scale_)
    return np.asarray(
        model.readout.predict(
            causal_lags(latent, q, history_bins=int(model.config["history_bins"]))
        ),
        np.float32,
    )


def wf_grid():
    return [
        {"method": "wf_fss", "history_bins": h, "smooth": sm, "alpha": a}
        for h in HISTORIES
        for sm in (False, True)
        for a in ALPHAS
    ]


def _fit_wf_ridge(
    features: np.ndarray, labels: np.ndarray, alpha: float
) -> RidgeReadout:
    """Fit the legacy WF-FSS intercept ridge, including its wide-design dual solve."""
    x = _array(features, "features")
    y = _array(labels, "labels")
    if len(x) != len(y):
        raise ValueError("feature and label rows must have equal length")
    if len(x) >= x.shape[1]:
        return fit_ridge_grid(x, y, [alpha])[float(alpha)]
    mean_x, mean_y = x.mean(0), y.mean(0)
    centered_x, centered_y = x - mean_x, y - mean_y
    coefficient = centered_x.T @ np.linalg.solve(
        centered_x @ centered_x.T + float(alpha) * np.eye(len(x)), centered_y
    )
    return RidgeReadout(coefficient, mean_y - mean_x @ coefficient, float(alpha))


def fit_wf_fss(record: Any, config: dict):
    if (
        set(config) != {"method", "history_bins", "smooth", "alpha"}
        or config not in wf_grid()
    ):
        raise ValueError("configuration is outside frozen WF-FSS grid")
    raw, stream, mask, _ = _calibration(record, bool(config["smooth"]))
    idx = _indices(_field(record, "carrier_indices"), len(raw), "carrier_indices")
    mean = stream[mask].mean(0)
    scale = stream[mask].std(0)
    scale[scale < 1e-6] = 1.0
    z = (stream - mean) / scale
    z[~mask] = 0.0
    features = causal_lags(z, idx, history_bins=int(config["history_bins"]))
    labels = _array(_field(record, "velocity"), "velocity")[idx]
    readout = _fit_wf_ridge(features, labels, float(config["alpha"]))
    return {
        "config": dict(config),
        "mean": mean,
        "scale": scale,
        "readout": readout,
        "representation": str(_field(record, "representation")),
    }


def predict_wf_fss(model: dict, record: Any) -> np.ndarray:
    if str(_field(record, "representation")) != model["representation"]:
        raise ValueError("representation mismatch")
    raw = _array(_field(record, "neural"), "neural")
    q = _indices(_field(record, "query_indices"), len(raw), "query_indices")
    stream = smooth_raw(raw) if model["config"]["smooth"] else raw
    features = causal_lags(
        (stream - model["mean"]) / model["scale"],
        q,
        history_bins=int(model["config"]["history_bins"]),
    )
    return np.asarray(model["readout"].predict(features), np.float32)


def _score(record: Any, prediction: np.ndarray) -> float:
    neural = _field(record, "neural")
    query = _indices(_field(record, "query_indices"), len(neural), "query_indices")
    target = _array(_field(record, "velocity"), "velocity")[query]
    return float(
        1 - ((target - prediction) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()
    )


def _best_candidate(records: list[Any], method: str) -> dict:
    if method == "wf_fss":
        # This dense-label comparator was frozen before the current campaign;
        # it is evaluated here but never re-selected on development data.
        return {"config": dict(FROZEN_WF_FSS), "selection": "historical_frozen"}

    candidates = [config for config in grid() if config["method"] == method]
    # The original protocol fits each PCA/FA transform once per
    # (method, components, smooth, session), then only refits ridge/lags.
    transforms: dict[tuple, dict[str, LocalSUAWF]] = {}
    lag_models: dict[tuple, dict[str, LocalSUAWF]] = {}
    best = None
    for config in candidates:
        transform_key = (method, config["components"], config["smooth"])
        lag_key = (*transform_key, config["history_bins"])
        try:
            if transform_key not in transforms:
                base = {**config, "history_bins": 1, "alpha": 1e2}
                transforms[transform_key] = {
                    record.session_id: fit_local_wf(record, base) for record in records
                }
            if lag_key not in lag_models:
                base = {**config, "alpha": 1e2}
                lag_models[lag_key] = {
                    session_id: refit_local(model, base)
                    for session_id, model in transforms[transform_key].items()
                }
            models = {
                session_id: refit_local(model, config)
                for session_id, model in lag_models[lag_key].items()
            }
            value = float(
                np.mean(
                    [
                        _score(
                            record, predict_local_wf(models[record.session_id], record)
                        )
                        for record in records
                    ]
                )
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if best is None or value > best["score"]:
            best = {"config": config, "score": value, "selection": "development_grid"}
    if best is None:
        raise RuntimeError(f"no eligible {method} candidate")
    return best


def select_sua_baselines(development_records: list[Any]) -> dict:
    """Use frozen WF-FSS and select PCA/FA configurations on development only."""
    return {
        method: _best_candidate(development_records, method)
        for method in ("wf_fss", "pca_wf", "fa_wf")
    }


def evaluate_sua_baselines(records: list[Any], selection: dict) -> dict:
    fitters = {
        "wf_fss": (fit_wf_fss, predict_wf_fss),
        "pca_wf": (fit_local_wf, predict_local_wf),
        "fa_wf": (fit_local_wf, predict_local_wf),
    }
    result = {}
    for method, selected in selection.items():
        fit, predict = fitters[method]
        models = [fit(record, selected["config"]) for record in records]
        values = [
            _score(record, predict(model, record))
            for record, model in zip(records, models)
        ]
        result[method] = {
            "selection": selected,
            "mean_r2": float(np.mean(values)),
            "session_r2": values,
        }
    return result


def _rows(records, selection):
    results=evaluate_sua_baselines(records, selection)
    return {name: {**value, "target_local_dense_velocity_ridge": True} for name,value in results.items()}

def main(argv=None):
    import argparse,json
    from pathlib import Path
    from .common import load_records, atomic_json, sha256
    parser=argparse.ArgumentParser(description="Sub-C target-local WF-FSS/PCA-WF/FA-WF")
    sub=parser.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("select-dev");p.add_argument("--cache",type=Path,required=True);p.add_argument("--dest",type=Path,required=True)
    p=sub.add_parser("score-final");p.add_argument("--final-cache",type=Path,required=True);p.add_argument("--final-receipt",type=Path,required=True);p.add_argument("--seal",type=Path,required=True);p.add_argument("--selection",type=Path,required=True);p.add_argument("--dest",type=Path,required=True)
    a=parser.parse_args(argv)
    if a.cmd=="select-dev":
        dev=load_records(a.cache,"sua","dev"); selection=select_sua_baselines(dev)
        out={"schema":"dandi_subc_wf_selection_v1","status":"DEV_SELECTED","protocol":__import__("apst.dandi_subc.protocol",fromlist=["protocol_dict"]).protocol_dict(),"selection":selection,"dev_scores":_rows(dev,selection),"dense_velocity_target_local_ridge":True,"final_sessions_opened":0}
        atomic_json(a.dest,out); return
    from .final_access import FinalAccess
    from .data import load_cached_session
    cap=FinalAccess.from_manifest(a.seal);cap.validate(); receipt=json.loads(a.final_receipt.read_text())
    if receipt.get("seal_sha256") != sha256(a.seal): raise PermissionError("final cache receipt/seal mismatch")
    selection=json.loads(a.selection.read_text())
    if selection.get("status") != "DEV_SELECTED" or selection.get("protocol") != __import__("apst.dandi_subc.protocol",fromlist=["protocol_dict"]).protocol_dict(): raise PermissionError("invalid WF development selection")
    if a.dest.exists(): raise FileExistsError("WF final destination exists")
    sessions=receipt.get("sessions", {})
    if set(sessions) != set(__import__("apst.dandi_subc.protocol",fromlist=["FINAL_SESSIONS"]).FINAL_SESSIONS): raise PermissionError("incomplete final cache receipt")
    records=[]
    for sid in __import__("apst.dandi_subc.protocol",fromlist=["FINAL_SESSIONS"]).FINAL_SESSIONS:
        item=sessions[sid]; path=Path(item.get("file",a.final_cache/f"{sid}.sua.npz"))
        if not path.is_file() or sha256(path)!=item.get("sha256"): raise PermissionError("final cache hash mismatch: "+sid)
        records.append(load_cached_session(path, final_access=cap))
    result={"schema":"dandi_subc_wf_final_v1","status":"FINAL_SCORED","selection_sha256":sha256(a.selection),"seal_sha256":sha256(a.seal),"results":_rows(records,selection["selection"]),"target_local_dense_velocity_ridge":True,"final_sessions_opened":len(records)}
    atomic_json(a.dest,result)

if __name__ == "__main__": main()
