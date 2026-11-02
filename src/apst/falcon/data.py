"""Prepared-cache construction for FALCON M1/M2/H1 from raw NWB.

``prepare`` turns public DANDI NWB recordings into the frozen tensors the
APST trainer consumes: per-trial support activity, closed-form carrier rows,
and padded query planes.  Support builders never read a query target; query
planes always store native (unscaled) targets.

Window convention (shared by all tasks): ``X`` is the raw neural timeline
left-padded with ``context - 1`` zero bins and a query window is
``X[start : start + context]``, so ``starts`` are raw-recording endpoint
indices and a window cell is padding exactly when its absolute index is
below the pad.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import interp1d

from . import carrier as carrier_mod
from . import protocol
from .common import array_sha256, atomic_json, sha256


@dataclass
class SessionData:
    task: str
    split: str
    session: str
    x: np.ndarray            # padded [T + pad, units] float32
    starts: np.ndarray       # int64 padded-window starts (raw endpoint indices)
    y: np.ndarray            # native targets at window endpoints [Q, outputs]
    pad: int
    support: np.ndarray      # [M, support_bins, units] float32
    carrier: np.ndarray      # [units, 4] float32
    provenance: dict[str, Any]

    def windows(self, indices, scale: float):
        rows = np.asarray(indices, np.int64).reshape(-1)
        starts = self.starts[rows]
        offsets = np.arange(self.pad + 1, dtype=np.int64)
        x = np.ascontiguousarray(self.x[starts[:, None] + offsets], np.float32)
        y = np.ascontiguousarray(self.y[rows] * np.float32(scale))
        valid = (starts[:, None] + offsets) >= self.pad
        return x, y, valid


def _cubic_to_bins(values: np.ndarray, bins: int) -> np.ndarray:
    original = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, bins)
    interpolator = interp1d(original, values, axis=0, kind="cubic", fill_value="extrapolate")
    return interpolator(target).astype(np.float32)


def _load_nwb(path: Path, task: str):
    from apst.data.load import load_nwb_file
    neural, covariates, trial_change, eval_mask = load_nwb_file(path, task)
    return (np.asarray(neural, np.float32), np.asarray(covariates, np.float32),
            np.asarray(trial_change, bool), np.asarray(eval_mask, bool))


def _padded(neural: np.ndarray, pad: int) -> np.ndarray:
    return np.ascontiguousarray(np.pad(neural, ((pad, 0), (0, 0))), np.float32)


def _session_path(task: str, split: str, session: str, root: Path) -> Path:
    from apst.data.load import list_sessions
    falcon_split = "held_in" if split == "source" else "held_out"
    rows = list_sessions(task, falcon_split, root=root)
    key = session[4:] if session.startswith("ses-") else session
    for row in rows:
        if row["session"] == key or row["session"] == session:
            return row["path"]
    available = sorted(row["session"] for row in rows)
    raise KeyError(f"{task}/{split} session {session!r} not found; available: {available}")


# --------------------------------------------------------------------------
# M1: 64 units, first-10-trial cubic support, muscle SVD4 carrier.
# --------------------------------------------------------------------------

def _m1_trials_mask(path: Path, neural: np.ndarray, eval_mask: np.ndarray):
    """Trial starts on the raw neural timeline, then the eval-valid view."""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        timestamps = np.asarray(nwb.acquisition["eval_mask"].timestamps[:], np.float64)
        starts_time = np.asarray(nwb.trials.to_dataframe().reset_index()["start_time"], np.float64)
    if len(timestamps) != len(neural) or len(eval_mask) != len(neural):
        raise RuntimeError(f"{path.name}: M1 neural/timestamp topology drift")
    changes = np.zeros(len(neural), bool)
    changes[np.searchsorted(timestamps, starts_time, side="left").clip(0, len(neural) - 1)] = True
    raw_trial_starts = np.flatnonzero(changes).astype(np.int64)
    neural_valid = neural[eval_mask]
    changes_valid = changes[eval_mask]
    starts_valid = np.flatnonzero(changes_valid).astype(np.int64)
    return raw_trial_starts, neural_valid, starts_valid


def _m1_session(task: str, split: str, session: str, root: Path, spec: Mapping[str, Any],
                muscle_fit) -> SessionData:
    path = _session_path(task, split, session, root)
    neural, covariates, _trial_change, eval_mask = _load_nwb(path, task)
    trial_starts, neural_valid, starts_valid = _m1_trials_mask(path, neural, eval_mask)
    n_support = int(spec["support_trials"])
    if len(starts_valid) < n_support + (1 if split == "source" else 0):
        raise RuntimeError(f"{session}: M1 lacks {n_support} support trials plus query")
    pieces = []
    for index in range(n_support):
        low = int(starts_valid[index])
        high = int(starts_valid[index + 1]) if index + 1 < len(starts_valid) else len(neural_valid)
        if high - low < 4:
            raise RuntimeError(f"{session}: short M1 support trial {index}")
        pieces.append(_cubic_to_bins(neural_valid[low:high], int(spec["support_bins"])))
    support = np.ascontiguousarray(np.stack(pieces), np.float32)
    context = int(spec["context_bins"])
    pad = context - 1
    x = _padded(neural, pad)
    starts = np.flatnonzero(eval_mask).astype(np.int64)
    y = np.ascontiguousarray(covariates[starts], np.float32)
    if split == "source":
        # Source queries must not reach into the support prefix.
        boundary = int(trial_starts[n_support]) + pad
        keep = np.flatnonzero(starts >= boundary)
        starts = np.ascontiguousarray(starts[keep])
        y = np.ascontiguousarray(y[keep])
    carrier_row, carrier_meta = carrier_mod.m1_project(muscle_fit, path)
    provenance = {
        "raw_nwb": str(path), "raw_nwb_sha256": sha256(path),
        "trial_ids": list(range(n_support)),
        "trialization": "trials.start_time on eval_mask timestamps; eval-valid cubic to 1024 bins",
        "query_rule": ("whole padded W100 window starts at/after Falcon trial_start_indices[10]"
                       if split == "source" else "all eval-mask-valid endpoints"),
        "carrier": carrier_meta,
    }
    return SessionData(task, split, session, x, starts, y, pad, support, carrier_row, provenance)


# --------------------------------------------------------------------------
# M2: 96 units, first-33-trial active-timeline cubic support, MOVE-T4 carrier.
# --------------------------------------------------------------------------

def _m2_calib_timeline(neural: np.ndarray, covariates: np.ndarray,
                       trial_change: np.ndarray, eval_mask: np.ndarray,
                       horizon: int = 33):
    """Bins of the chronological first-``horizon``-trial calibration prefix.

    In FALCON M2 the leading trials of every recording are the calibration
    surface; the query phase follows them.  The eval mask is True on this
    prefix (it does not separate calibration from query), so the boundary is
    the trial-change start of the first post-support trial.
    """
    del eval_mask, covariates
    starts = np.flatnonzero(np.asarray(trial_change, bool))
    if len(starts) < horizon:
        raise ValueError("M2 recording lacks a full calibration prefix")
    end = int(starts[horizon])
    return neural[:end], trial_change[:end]


def _m2_session(task: str, split: str, session: str, root: Path, spec: Mapping[str, Any],
                normalizer) -> SessionData:
    from pynwb import NWBHDF5IO
    path = _session_path(task, split, session, root)
    neural, covariates, trial_change, eval_mask = _load_nwb(path, task)
    calib_neural, calib_change = _m2_calib_timeline(neural, covariates, trial_change, eval_mask,
                                                    horizon=int(spec["support_trials"]))
    starts = np.flatnonzero(calib_change).astype(np.int64)
    horizon = int(spec["support_trials"])
    if len(starts) < horizon:
        raise RuntimeError(f"{session}: M2 calibration has fewer than {horizon} trials")
    pieces, move_counts, lengths = [], [], []
    lo_bin, hi_bin = spec["move_t4_window_bins"]
    for index in range(horizon):
        low = int(starts[index])
        high = int(starts[index + 1]) if index + 1 < len(starts) else len(calib_neural)
        trial = calib_neural[low:high]
        if trial.shape[0] < hi_bin:
            raise RuntimeError(f"{session}: M2 trial {index} shorter than MOVE window")
        pieces.append(_cubic_to_bins(trial, int(spec["support_bins"])))
        move_counts.append(trial[lo_bin:hi_bin].sum(axis=0, dtype=np.float64))
        lengths.append(hi_bin - lo_bin)
    support = np.ascontiguousarray(np.stack(pieces), np.float32)
    carrier_row = carrier_mod.m2_move_t4(np.stack(move_counts), path, lengths, normalizer)
    context = int(spec["context_bins"])
    pad = context - 1
    x = _padded(neural, pad)
    endpoints = np.arange(context - 1, len(neural), dtype=np.int64)
    endpoints = endpoints[eval_mask[endpoints]]
    y = np.ascontiguousarray(covariates[endpoints], np.float32)
    if split == "source":
        query_change = np.pad(trial_change, (pad, 0), constant_values=False)
        padded_trial_starts = np.flatnonzero(query_change)
        if len(padded_trial_starts) <= horizon:
            raise RuntimeError(f"{session}: M2 lacks a post-support trial")
        boundary = int(padded_trial_starts[horizon])
        keep = np.flatnonzero(endpoints >= boundary)
        endpoints = np.ascontiguousarray(endpoints[keep])
        y = np.ascontiguousarray(y[keep])
    provenance = {
        "raw_nwb": str(path), "raw_nwb_sha256": sha256(path),
        "trial_ids": list(range(horizon)),
        "trialization": "active in-trial calibration bins; cubic to 100 bins per trial",
        "move_t4_window_bins": [int(lo_bin), int(hi_bin)],
        "query_rule": ("whole padded W50 window starts at/after the padded start of trial 33"
                       if split == "source" else "all eval-mask-valid endpoints"),
    }
    return SessionData(task, split, session, x, endpoints, y, pad, support, carrier_row, provenance)


# --------------------------------------------------------------------------
# H1: 176 units, first-3 eval-valid trial cubic support, signed-state carrier.
# --------------------------------------------------------------------------

def _h1_session(task: str, split: str, session: str, root: Path, spec: Mapping[str, Any],
                plan) -> SessionData:
    path = _session_path(task, split, session, root)
    neural, velocity, _trial_change, eval_mask = _load_nwb(path, task)
    record = carrier_mod.h1_record(path, neural, velocity, eval_mask)
    n_support = int(spec["support_trials"])
    values = list(record["trial_values"][:n_support])
    if len(values) != n_support:
        raise RuntimeError(f"{session}: H1 lacks {n_support} eval-valid support trials")
    support = np.ascontiguousarray(
        np.stack([_cubic_to_bins(record["eval_trial_neural"](value), int(spec["support_bins"]))
                  for value in values]), np.float32)
    carrier_row = carrier_mod.h1_deploy(plan, record, values)
    context = int(spec["context_bins"])
    pad = context - 1
    x = _padded(neural, pad)
    endpoints = np.flatnonzero(eval_mask).astype(np.int64)
    y = np.ascontiguousarray(velocity[endpoints], np.float32)
    provenance = {
        "raw_nwb": str(path), "raw_nwb_sha256": sha256(path),
        "trial_ids": values,
        "trialization": "TrialNum/eval-mask selected trials; cubic to 1024 bins",
        "query_rule": "all eval-mask-valid endpoints",
    }
    return SessionData(task, split, session, x, endpoints, y, pad, support, carrier_row, provenance)


# --------------------------------------------------------------------------
# prepare / load
# --------------------------------------------------------------------------

def _sessions_for(task: str, split: str, spec: Mapping[str, Any]) -> list[str]:
    roster = spec["source_sessions"] if split == "source" else spec["dev_sessions"]
    if isinstance(roster, str):
        if task == "m2" and split == "source" and roster == "held_in_enumerated":
            from apst.data.load import list_sessions
            return sorted(row["session"] for row in list_sessions(task, "held_in"))
        if task == "h1" and split == "dev" and roster == "held_out_grouped_seven":
            return [s for members in protocol.h1_groups().values() for s in members]
    return list(roster)


def prepare(cache: Path, raw_root: Path | None = None, tasks: tuple[str, ...] = protocol.TASKS) -> dict:
    """Materialize the prepared cache for the requested tasks."""
    from .bootstrap import data_root
    root = Path(raw_root).resolve() if raw_root else data_root()
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {"schema": "apst_falcon_prepared_v1", "tasks": {}}
    for task in tasks:
        if task not in protocol.TASKS:
            raise ValueError(f"unknown task {task!r}")
        spec = protocol.task_protocol(task)
        rows: dict[str, Any] = {"source": {}, "dev": {}}
        task_dir = cache / task
        task_dir.mkdir(parents=True, exist_ok=True)
        shared: dict[str, Any] = {}
        if task == "m1":
            fit = carrier_mod.m1_fit_source(
                [_session_path(task, "source", s, root) for s in spec["carrier_fit_sessions"]],
                emg_stop=int(spec["muscle_emg_source_stop_trial"]))
            carrier_mod.m1_save_fit(fit, task_dir / "muscle_carrier_fit.npz")
            shared["muscle_carrier_fit"] = "muscle_carrier_fit.npz"
            builders = {"source": lambda s: _m1_session(task, "source", s, root, spec, fit),
                        "dev": lambda s: _m1_session(task, "dev", s, root, spec, fit)}
        elif task == "m2":
            source_paths = [_session_path(task, "source", s, root)
                            for s in _sessions_for(task, "source", spec)]
            normalizer = carrier_mod.m2_fit_normalizer(source_paths, spec)
            carrier_mod.m2_save_normalizer(normalizer, task_dir / "move_t4_normalizer.npz")
            shared["move_t4_normalizer"] = "move_t4_normalizer.npz"
            builders = {"source": lambda s: _m2_session(task, "source", s, root, spec, normalizer),
                        "dev": lambda s: _m2_session(task, "dev", s, root, spec, normalizer)}
        else:
            plan = carrier_mod.h1_fit_plan(
                [_session_path(task, "source", s, root) for s in spec["source_sessions"]])
            carrier_mod.h1_save_plan(plan, task_dir / "signed_state_plan.npz")
            shared["signed_state_plan"] = "signed_state_plan.npz"
            builders = {"source": lambda s: _h1_session(task, "source", s, root, spec, plan),
                        "dev": lambda s: _h1_session(task, "dev", s, root, spec, plan)}
        for split in ("source", "dev"):
            for session in _sessions_for(task, split, spec):
                data = builders[split](session)
                out = task_dir / split / session
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / "x.npy", data.x)
                np.savez_compressed(
                    out / "session.npz",
                    starts=data.starts, y=data.y, support=data.support, carrier=data.carrier)
                rows[split][session] = {
                    "x": "x.npy", "session": "session.npz", "pad": data.pad,
                    "windows": int(len(data.starts)),
                    "support_shape": list(data.support.shape),
                    "x_sha256": sha256(out / "x.npy"),
                    "starts_sha256": array_sha256(data.starts),
                    "y_sha256": array_sha256(data.y),
                    "support_sha256": array_sha256(data.support),
                    "carrier_sha256": array_sha256(data.carrier),
                    "provenance": data.provenance,
                }
        receipt["tasks"][task] = {"spec": spec, "shared": shared, "sessions": rows,
                                  "raw_root": str(root)}
    receipt["sha256"] = None
    atomic_json(cache / "prepared_receipt.json", receipt)
    return receipt


def load_sessions(cache: Path, task: str, split: str) -> dict[str, SessionData]:
    """Load prepared sessions for one task split."""
    cache = Path(cache).resolve()
    spec = protocol.task_protocol(task)
    rows = _read_receipt(cache)["tasks"][task]["sessions"][split]
    out: dict[str, SessionData] = {}
    for session, row in rows.items():
        base = cache / task / split / session
        x = np.load(base / "x.npy", mmap_mode="r")
        with np.load(base / "session.npz") as z:
            starts = np.asarray(z["starts"], np.int64)
            y = np.asarray(z["y"], np.float32)
            support = np.asarray(z["support"], np.float32)
            carrier = np.asarray(z["carrier"], np.float32)
        out[session] = SessionData(task, split, session, x, starts, y, row["pad"],
                                   support, carrier, row["provenance"])
    if not out:
        raise ValueError(f"prepared cache has no {task}/{split} sessions")
    return out


def updates_per_epoch(cache: Path, task: str) -> int:
    sessions = load_sessions(cache, task, "source")
    return int(sum((len(s.starts) + 31) // 32 for s in sessions.values()))


def _read_receipt(cache: Path) -> dict:
    import json
    path = Path(cache) / "prepared_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "apst_falcon_prepared_v1":
        raise ValueError(f"unexpected prepared receipt schema in {path}")
    return receipt
