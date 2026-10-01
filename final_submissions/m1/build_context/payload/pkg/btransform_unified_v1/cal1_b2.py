"""C2-CAL-1 B2 banks: V1 M7 starts + M4 / deploy-M3 carrier switch.

Plan arrays are rebuilt with q=12, lambda=10. Missing V1 npz SHA is not a gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from btransform_unified_v1.c2_protocol import starts_for_budget

V1_AUTH = Path(
    "/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/h1_series_20260830/"
    "results/h1_cal_aug_all_source_m3_deployment_v1/source_authority"
)
SELECTED = {"q": 12, "lambda": 10.0}
S_SRC_SEALED = 6.8260113140959355e-06
NORMALIZER_FLOOR = 1e-12
SUPPORT_TRIALS = 4
EXPECTED_NEURONS = 176


def load_v1_inventory() -> dict[str, Any]:
    cache = json.loads((V1_AUTH / "carrier_cache.json").read_text(encoding="utf-8"))
    normalizer = json.loads((V1_AUTH / "normalizer.json").read_text(encoding="utf-8"))
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in cache["entries"]:
        by_session.setdefault(str(row["session"]), []).append(row)
    for session, rows in by_session.items():
        rows.sort(key=lambda item: int(item["start_index"]))
    s_src = float(normalizer.get("s_src", S_SRC_SEALED))
    return {
        "by_session": by_session,
        "s_src": s_src,
        "denominator": max(s_src, NORMALIZER_FLOOR),
        "starts": {session: tuple(int(row["start_index"]) for row in rows) for session, rows in by_session.items()},
    }


def _support_rates(record: Any) -> np.ndarray:
    trials = [record.blocks_for(value) for value in record.trial_values[:SUPPORT_TRIALS]]
    return np.concatenate([item.rates for item in trials], axis=0).astype(np.float64)


def _ridge_fit(record: Any, mean: np.ndarray, scale: np.ndarray, pcs: np.ndarray, q: int, lam: float) -> np.ndarray:
    trials = [record.blocks_for(value) for value in record.trial_values[:SUPPORT_TRIALS]]
    rates = np.concatenate([item.rates for item in trials], axis=0).astype(np.float64)
    labels = np.concatenate([item.velocity for item in trials], axis=0).astype(np.float64)
    projected = ((rates - mean[None, :]) / scale[None, :]) @ pcs[:q].T
    design = np.column_stack((np.ones(len(projected), dtype=np.float64), projected))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(lam)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ labels)


def rebuild_plan(records: dict[str, Any]) -> Any:
    rates = np.concatenate([_support_rates(record) for record in records.values()], axis=0)
    mean = rates.mean(axis=0, dtype=np.float64)
    scale = np.maximum(rates.std(axis=0, dtype=np.float64), 1.0e-6)
    _, _, right = np.linalg.svd((rates - mean[None, :]) / scale[None, :], full_matrices=False)
    pcs = np.asarray(right[:16], dtype=np.float64)
    if pcs.shape != (16, EXPECTED_NEURONS):
        raise ValueError(f"source PCA shape drift {pcs.shape}")
    q = int(SELECTED["q"])
    lam = float(SELECTED["lambda"])
    pooled = np.concatenate(
        [
            (pcs[:q].T @ _ridge_fit(record, mean, scale, pcs, q, lam)[1:]) / scale[:, None]
            for record in records.values()
        ],
        axis=0,
    )
    _, _, right_u = np.linalg.svd(pooled, full_matrices=False)
    u = np.asarray(right_u[:4].T, dtype=np.float64)
    carriers = pooled @ u
    mu = carriers.mean(axis=0, dtype=np.float64)
    tau2 = float(np.square(carriers - mu[None, :], dtype=np.float64).sum(dtype=np.float64) / (carriers.shape[0] * 4))
    return SimpleNamespace(
        mean=mean,
        scale=scale,
        pcs=pcs,
        q=q,
        ridge_lambda=lam,
        U=u,
        mu=mu,
        tau2=tau2,
    )


def normalize_carrier(raw: np.ndarray, denominator: float) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(raw, np.float64) / float(denominator), dtype=np.float32)


def legal_starts(starts: tuple[int, ...], n_trials: int, budget: int) -> tuple[int, ...]:
    return starts_for_budget(starts, n_trials=n_trials, budget=int(budget))
