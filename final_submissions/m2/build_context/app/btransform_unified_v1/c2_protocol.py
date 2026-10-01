"""C2-identical H1 selection and prefix-cycle primitives.

Selection surface = visible held-out-calib (14 recordings, S6-S12 grouped).
Tie-break and metric names match the sealed C2 HO-M3 contract.
This module does not train and does not open hidden-test query labels.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import r2_score

C2_CYCLE: tuple[int, ...] = (7, 5, 4, 3)
C2_PREFIX_SCHEMA = "h1_cal_aug_m3_aware_dual_selection_v2"
B2_START_SCHEMA = "btransform_unified_v1_h1_c2_cal1_b2"
HO_SELECTION_METRIC = "val_ho_m3_grouped/r2_mean"
C2_TIE_BREAK: tuple[str, ...] = (
    "higher mean",
    "higher worst-session",
    "lower population std",
    "earlier epoch",
)

HELDOUT_SESSION_TO_FALCON_KEY: tuple[tuple[str, str], ...] = (
    ("ses-19250126T113454", "S6_set_1"),
    ("ses-19250126T114029", "S6_set_2"),
    ("ses-19250127T120333", "S7_set_1"),
    ("ses-19250127T120826", "S7_set_2"),
    ("ses-19250129T112555", "S8_set_1"),
    ("ses-19250129T113059", "S8_set_2"),
    ("ses-19250202T113958", "S9_set_1"),
    ("ses-19250202T114452", "S9_set_2"),
    ("ses-19250203T113515", "S10_set_1"),
    ("ses-19250203T114018", "S10_set_2"),
    ("ses-19250206T112219", "S11_set_1"),
    ("ses-19250206T112712", "S11_set_2"),
    ("ses-19250209T111826", "S12_set_1"),
    ("ses-19250209T112327", "S12_set_2"),
)


def prefix_schedule(epoch_zero_based: int, n_batches: int) -> tuple[int, ...]:
    """C2 per-batch M7/M5/M4/M3 cycle. Same hash as the sealed C2 contract."""
    if n_batches < len(C2_CYCLE):
        raise ValueError("prefix schedule needs at least 4 batches")
    token = hashlib.sha256(
        f"{C2_PREFIX_SCHEMA}|prefix|h1_all_source_13|{int(epoch_zero_based)}".encode()
    ).digest()
    offset = int.from_bytes(token[:8], "big") % len(C2_CYCLE)
    row = tuple(C2_CYCLE[(index + offset) % len(C2_CYCLE)] for index in range(int(n_batches)))
    counts = {m: row.count(m) for m in C2_CYCLE}
    if set(row) != set(C2_CYCLE):
        raise ValueError("C2 cycle coverage drift")
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("C2 cycle balance drift")
    return row


def starts_for_budget(starts: Sequence[int], *, n_trials: int, budget: int) -> tuple[int, ...]:
    """Keep V1 starts that still have ``budget`` trials from that index."""
    need = int(budget)
    n = int(n_trials)
    out = []
    for start in starts:
        start_i = int(start)
        if start_i >= 0 and start_i + need <= n:
            out.append(start_i)
    return tuple(out)


def pick_m7_start(session: str, *, epoch0: int, step: int, starts: Sequence[int]) -> int:
    """Hash-pick one V1 M7/M4 block start for this (session, epoch, step)."""
    roster = tuple(int(s) for s in starts)
    if not roster:
        raise ValueError(f"no M7 starts for {session}")
    token = hashlib.sha256(
        f"{B2_START_SCHEMA}|m7_start|{session}|{int(epoch0)}|{int(step)}".encode()
    ).digest()
    return roster[int.from_bytes(token[:8], "big") % len(roster)]


def select_epoch(rows: Sequence[Mapping[str, Any]], metric: str = HO_SELECTION_METRIC) -> dict[str, Any]:
    """C2 tie-break: higher mean, higher worst-session, lower std, earlier epoch."""
    if not rows:
        raise ValueError("selection requires at least one epoch row")
    for row in rows:
        if metric not in row or any(key not in row for key in ("worst_session_r2", "session_std_population", "epoch_zero_based")):
            raise ValueError("selection metric incomplete")
    picked = max(
        rows,
        key=lambda row: (
            float(row[metric]),
            float(row["worst_session_r2"]),
            -float(row["session_std_population"]),
            -int(row["epoch_zero_based"]),
        ),
    )
    return dict(picked)


def grouped_session_metrics(
    predictions: Mapping[str, Any],
    targets: Mapping[str, Any],
    masks: Mapping[str, Any],
    mapping: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """C2 HO-M3 aggregation: group set_1/set_2, variance-weighted R², pop std."""
    grouped: dict[str, list[str]] = {}
    per_recording: dict[str, float] = {}
    for _session, key in mapping:
        grouped.setdefault(key.split("_set_", 1)[0], []).append(key)
        pred = np.asarray(predictions[key], np.float64)
        target = np.asarray(targets[key], np.float64)
        mask = np.asarray(masks[key], bool).reshape(-1)
        if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 7 or len(mask) != len(pred):
            raise ValueError(f"surface shape drift: {key}")
        per_recording[key] = float(r2_score(target[mask], pred[mask], multioutput="variance_weighted"))
    per_session: dict[str, float] = {}
    for group in sorted(grouped):
        pred = np.concatenate(
            [np.asarray(predictions[key], np.float64)[np.asarray(masks[key], bool).reshape(-1)] for key in grouped[group]]
        )
        target = np.concatenate(
            [np.asarray(targets[key], np.float64)[np.asarray(masks[key], bool).reshape(-1)] for key in grouped[group]]
        )
        per_session[group] = float(r2_score(target, pred, multioutput="variance_weighted"))
    values = np.asarray(list(per_session.values()), np.float64)
    return {
        "r2_mean": float(np.mean(values)),
        "r2_std_population": float(np.std(values, ddof=0)),
        "worst_session_r2": float(np.min(values)),
        "per_session_r2": per_session,
        "per_recording_r2": per_recording,
        "spint_recording_mean_r2": float(np.mean(list(per_recording.values()), dtype=np.float64)),
    }
