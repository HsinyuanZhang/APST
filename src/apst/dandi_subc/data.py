"""Raw, paired SUA/PMUA DANDI session materialization for the 2015-only protocol."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pynwb import NWBHDF5IO
from scipy.interpolate import interp1d

from . import protocol

Representation = Literal["sua", "pmua"]


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(repr(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SessionData:
    session_id: str
    split: str
    representation: str
    neural: np.ndarray
    velocity: np.ndarray
    bin_edges: np.ndarray
    query_indices: np.ndarray
    support_indices: np.ndarray
    carrier_indices: np.ndarray
    activity: np.ndarray
    carrier_counts: np.ndarray
    carrier_angles: np.ndarray
    channel_indices: np.ndarray
    unit_to_electrode: np.ndarray
    observed_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def velocity_physical(self) -> np.ndarray:
        return self.velocity


def _canonical_metadata(nwb: Any, units_df: Any) -> tuple[list[tuple[str, str, int, str]], np.ndarray, np.ndarray, np.ndarray]:
    electrodes = nwb.electrodes.to_dataframe()
    required = {"group_name", "location", "bank", "pin", "label"}
    if not required <= set(electrodes.columns):
        raise ValueError(f"electrode table lacks required columns: {sorted(required - set(electrodes.columns))}")
    key_by_id: dict[int, tuple[str, str, int, str]] = {}
    for electrode_id, row in electrodes.iterrows():
        if str(row.group_name) == "electrode_group_M1" and str(row.location) == "Primary Motor Cortex":
            key_by_id[int(electrode_id)] = (str(row.group_name), str(row.bank), int(row.pin), str(row.label))
    canonical = sorted(key_by_id.values())
    if len(canonical) != protocol.CANONICAL_ELECTRODES or len(set(canonical)) != len(canonical):
        raise ValueError("expected 96 unique canonical hardware electrode keys")
    canonical_index = {key: idx for idx, key in enumerate(canonical)}
    unit_to_electrode: list[int] = []
    selected_unit_indices: list[int] = []
    for unit_row, region in enumerate(units_df["electrodes"]):
        indices = getattr(region, "index", None)
        if indices is None or len(indices) != 1:
            raise ValueError(f"unit {unit_row} must map to exactly one electrode")
        electrode_id = int(indices[0])
        if electrode_id not in key_by_id:
            continue
        key = key_by_id[electrode_id]
        # This is the explicit M1 filter; non-M1 sorted units are absent downstream.
        selected_unit_indices.append(unit_row)
        unit_to_electrode.append(canonical_index[key])
    unit_map = np.asarray(unit_to_electrode, dtype=np.int64)
    observed = np.zeros(protocol.CANONICAL_ELECTRODES, dtype=bool)
    observed[unit_map] = True
    if not selected_unit_indices:
        raise ValueError("NWB has no M1 sorted units")
    return canonical, unit_map, observed, np.asarray(selected_unit_indices, dtype=np.int64)


def _rewarded_trials(trials_df: Any, bin_edges: np.ndarray) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    n_bins = len(bin_edges) - 1
    for raw_row, row in trials_df.iterrows():
        if row.get("result") != "R":
            continue
        start, stop = float(row.start_time), float(row.stop_time)
        start_bin = max(0, int(np.searchsorted(bin_edges, start)))
        stop_bin = min(n_bins, int(np.searchsorted(bin_edges, stop)))
        if stop_bin - start_bin < protocol.WINDOW_BINS:
            continue
        selected.append({"raw_trial_row": int(raw_row), "start": start_bin, "stop": stop_bin,
                         "start_time": start, "stop_time": stop,
                         "go_cue_time": _finite_or_none(row, "go_cue_time"),
                         "target_dir": _finite_or_none(row, "target_dir")})
    if len(selected) < protocol.Q50_TRIAL_INDEX + 1:
        raise ValueError("fewer than 51 legal rewarded trials")
    return selected


def _resample_trial(values: np.ndarray) -> np.ndarray:
    length, n_channels = values.shape
    if length < 2:
        raise ValueError("activity trial has fewer than two bins")
    x_old = np.linspace(0.0, 1.0, length)
    x_new = np.linspace(0.0, 1.0, 100)
    kind = "cubic" if length >= 4 else "linear"
    return interp1d(x_old, values, axis=0, kind=kind, bounds_error=True)(x_new).astype(np.float32)


def _record_access(access_log: Any, payload: dict[str, Any]) -> None:
    if access_log is None:
        return
    if hasattr(access_log, "append"):
        access_log.append(payload)
    elif hasattr(access_log, "record"):
        access_log.record(payload)
    else:
        raise TypeError("access_log must implement append or record")


def _finite_or_none(row: Any, name: str) -> float | None:
    value = row.get(name) if name in row else None
    return float(value) if value is not None and np.isfinite(value) else None


def validate_carrier_angles(angles: np.ndarray) -> dict[str, int]:
    """M2 OLS law: keep every M33 trial; regression ignores only NaN direction rows."""
    angles = np.asarray(angles, dtype=np.float64)
    if angles.shape != (protocol.CARRIER_TRIALS,):
        raise ValueError("carrier angle geometry mismatch")
    finite = np.isfinite(angles)
    if int(finite.sum()) < 3:
        raise ValueError("OLS carrier requires at least three finite direction labels")
    theta = angles[finite]
    design = np.stack((np.ones(theta.size), np.cos(theta), np.sin(theta)), axis=1)
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        raise ValueError(f"OLS carrier direction design rank {rank}, expected 3")
    return {"finite_direction_count": int(finite.sum()), "direction_design_rank": rank}


def load_pair(session_id: str, *, raw_root: Path = protocol.DEFAULT_RAW_ROOT,
              purpose: Literal["source", "development", "final"], access_log: Any = None,
              final_access: Any = None) -> dict[str, SessionData]:
    """Open one authorized NWB once and return identical-timebase SUA and PMUA views."""
    if purpose == "final":
        # Import here to keep ordinary source/dev usage independent from the seal module.
        from .final_access import FinalAccess
        if not isinstance(final_access, FinalAccess):
            raise PermissionError("final raw access requires a validated FinalAccess capability")
        if protocol.split_for(session_id) != "final":
            raise PermissionError("final purpose only accepts a frozen final-roster session")
        final_access.authorize(session_id)
        split = "final"
    else:
        split = protocol.assert_authorized(session_id, purpose)
    raw_root = Path(raw_root).resolve()
    nwb_path = (raw_root / f"{session_id}_behavior+ecephys.nwb").resolve()
    if nwb_path.parent != raw_root or not nwb_path.is_file():
        raise FileNotFoundError(nwb_path)
    # Authorization above deliberately precedes existence/read checks for final sessions.
    _record_access(access_log, {"session_id": session_id, "split": split, "purpose": purpose, "path": str(nwb_path)})
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        canonical_keys, unit_map, observed, selected_unit_indices = _canonical_metadata(nwb, units)
        if not (0 < len(selected_unit_indices) <= protocol.MAX_UNITS):
            raise ValueError(f"{session_id}: M1 unit count {len(selected_unit_indices)} is outside 1..{protocol.MAX_UNITS}")
        spikes = [np.asarray(units.iloc[index]["spike_times"], dtype=np.float64) for index in selected_unit_indices]
        all_spikes = np.concatenate(spikes)
        edges = np.arange(float(all_spikes.min()), float(all_spikes.max()) + protocol.BIN_SECONDS, protocol.BIN_SECONDS)
        neural_sua = np.stack([np.histogram(s, bins=edges)[0] for s in spikes], axis=1).astype(np.float32)
        velocity_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        vel_times, vel_raw = np.asarray(velocity_series.timestamps[:]), np.asarray(velocity_series.data[:])
        centers = (edges[:-1] + edges[1:]) / 2.0
        velocity = np.stack([np.interp(centers, vel_times, vel_raw[:, axis], left=0.0, right=0.0)
                             for axis in range(vel_raw.shape[1])], axis=1).astype(np.float32)
        trials = _rewarded_trials(nwb.intervals["trials"].to_dataframe(), edges)

    n_bins = neural_sua.shape[0]
    # Query indices are the inclusive *end* bin of each 50-bin causal history.
    # Thus all consumers may use velocity[query_indices] directly.
    query_indices = np.asarray([start + protocol.WINDOW_BINS - 1
                                for trial in trials[protocol.Q50_TRIAL_INDEX:]
                                for start in range(trial["start"], trial["stop"] - protocol.WINDOW_BINS + 1)], dtype=np.int64)
    support_indices = np.concatenate([np.arange(t["start"], t["stop"], dtype=np.int64)
                                      for t in trials[:protocol.ACTIVITY_TRIALS]])
    carrier_native: list[np.ndarray] = []
    carrier_counts_sua: list[np.ndarray] = []
    for trial in trials[:protocol.CARRIER_TRIALS]:
        if trial["go_cue_time"] is None:
            raise ValueError(f"{session_id}: M33 carrier trial lacks finite go_cue_time")
        left = trial["go_cue_time"] + protocol.MOVE_START_AFTER_GO_SECONDS
        right = trial["go_cue_time"] + protocol.MOVE_STOP_AFTER_GO_SECONDS
        if not (trial["start_time"] <= left < right <= trial["stop_time"] and edges[0] <= left < right <= edges[-1]):
            raise ValueError(f"{session_id}: MOVE window is not contained in trial/raw support")
        # Exact physical-time, half-open counts; no arbitrary 20 ms-grid rounding.
        carrier_counts_sua.append(np.asarray([np.searchsorted(s, right, side="left") - np.searchsorted(s, left, side="left") for s in spikes], dtype=np.float32))
        inside = np.flatnonzero((centers >= left) & (centers < right)).astype(np.int64)
        if inside.size == 0:
            raise ValueError(f"{session_id}: MOVE window has no native bin labels")
        carrier_native.append(inside)
    carrier_indices = np.concatenate(carrier_native)
    if not set(carrier_indices.tolist()) <= set(support_indices.tolist()):
        raise RuntimeError("carrier native labels must be inside M33 support")
    if set(carrier_indices.tolist()) & set(query_indices.tolist()):
        raise RuntimeError("carrier native labels must be disjoint from Q50")
    carrier_angles = np.asarray([t["target_dir"] if t["target_dir"] is not None else np.nan
                                 for t in trials[:protocol.CARRIER_TRIALS]], dtype=np.float64)
    carrier_angle_audit = validate_carrier_angles(carrier_angles)
    activity_sua = np.stack([_resample_trial(neural_sua[t["start"]:t["stop"]]) for t in trials[:protocol.ACTIVITY_TRIALS]])
    pmua_channels = np.unique(unit_map)
    neural_pmua = np.zeros((n_bins, pmua_channels.size), dtype=np.float32)
    inverse = np.searchsorted(pmua_channels, unit_map)
    np.add.at(neural_pmua.T, inverse, neural_sua.T)
    activity_pmua = np.zeros((protocol.ACTIVITY_TRIALS, 100, pmua_channels.size), dtype=np.float32)
    np.add.at(activity_pmua.transpose(2, 0, 1), inverse, activity_sua.transpose(2, 0, 1))
    activity_pmua_direct = np.stack([_resample_trial(neural_pmua[t["start"]:t["stop"]]) for t in trials[:protocol.ACTIVITY_TRIALS]])
    if not np.allclose(activity_pmua, activity_pmua_direct, rtol=1e-5, atol=1e-5):
        raise RuntimeError("PMUA activity must equal resample(pooled raw counts)")
    carrier_counts_sua_array = np.asarray(carrier_counts_sua, dtype=np.float32)
    carrier_counts_pmua = np.zeros((protocol.CARRIER_TRIALS, pmua_channels.size), dtype=np.float32)
    np.add.at(carrier_counts_pmua.T, inverse, carrier_counts_sua_array.T)
    if not np.array_equal(neural_pmua.sum(axis=1), neural_sua.sum(axis=1)):
        raise RuntimeError("PMUA pooling violated per-bin spike conservation")
    base_metadata = {
        "schema": "dandi688_bench_v2_session_v1", "protocol": protocol.protocol_dict(),
        "raw_nwb_sha256": _file_sha256(nwb_path), "raw_trial_rows": [t["raw_trial_row"] for t in trials],
        "selected_original_unit_indices": selected_unit_indices.tolist(),
        "carrier_trial_positions": list(range(protocol.CARRIER_TRIALS)),
        **carrier_angle_audit,
        "canonical_electrode_keys": [list(key) for key in canonical_keys],
        "move_window": "[go_cue_time + 0.1, go_cue_time + 0.6) DANDI event-relative 500 ms anchor",
        "query_index_semantics": "inclusive end bin of a 50-bin causal window; target is velocity[query_indices]",
        "activity_resampling": "np.linspace(0,1,length)->np.linspace(0,1,100), scipy interp1d cubic",
        "window_receipts": [{"raw_trial_row": t["raw_trial_row"], "go_cue_time": t["go_cue_time"],
                              "move_start": t["go_cue_time"] + .1, "move_stop": t["go_cue_time"] + .6}
                             for t in trials[:protocol.CARRIER_TRIALS]],
    }

    def make(representation: Representation, neural: np.ndarray, activity: np.ndarray, channels: np.ndarray,
             carrier_counts: np.ndarray) -> SessionData:
        metadata = dict(base_metadata)
        arrays = {"neural": neural, "velocity": velocity, "bin_edges": edges, "query_indices": query_indices,
                  "support_indices": support_indices, "carrier_indices": carrier_indices, "activity": activity,
                  "carrier_counts": carrier_counts, "carrier_angles": carrier_angles,
                  "channel_indices": channels, "unit_to_electrode": unit_map, "observed_mask": observed}
        metadata["array_sha256"] = {name: _array_sha256(value) for name, value in arrays.items()}
        return SessionData(session_id, split, representation, neural, velocity, edges.astype(np.float64), query_indices,
                           support_indices, carrier_indices, activity, arrays["carrier_counts"], arrays["carrier_angles"].astype(np.float64),
                           channels, unit_map, observed, metadata)
    return {"sua": make("sua", neural_sua, activity_sua, unit_map.copy(), carrier_counts_sua_array),
            "pmua": make("pmua", neural_pmua, activity_pmua, pmua_channels, carrier_counts_pmua)}


def padded_windows(record: SessionData, indices: np.ndarray, n_pad: int = protocol.MAX_UNITS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (record.neural.shape[1] <= n_pad <= protocol.MAX_UNITS):
        raise ValueError("n_pad must cover channels and remain <=100")
    indices = np.asarray(indices, dtype=np.int64)
    if (indices.ndim != 1 or np.any(indices < protocol.WINDOW_BINS - 1)
            or np.any(indices >= record.neural.shape[0])):
        raise ValueError("window end index out of record bounds")
    X = np.zeros((len(indices), protocol.WINDOW_BINS, n_pad), dtype=np.float32)
    X[:, :, :record.neural.shape[1]] = np.stack(
        [record.neural[i - protocol.WINDOW_BINS + 1:i + 1] for i in indices]
    )
    timevalid = np.ones((len(indices), protocol.WINDOW_BINS), dtype=bool)
    unitmask = np.zeros(n_pad, dtype=bool); unitmask[:record.neural.shape[1]] = True
    return X, timevalid, unitmask


def save_session(data: SessionData, path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(data, name) for name in ("neural", "velocity", "bin_edges", "query_indices", "support_indices", "carrier_indices", "activity", "carrier_counts", "carrier_angles", "channel_indices", "unit_to_electrode", "observed_mask")}
    metadata = dict(data.metadata); metadata.update({"session_id": data.session_id, "split": data.split, "representation": data.representation})
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, **payload, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_cached_session(path: Path, final_access: Any = None) -> SessionData:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        session_id, split = str(metadata["session_id"]), str(metadata["split"])
        if protocol.split_for(session_id) != split:
            raise PermissionError("cached forged or wrong-split session is forbidden")
        if split == "final":
            from .final_access import FinalAccess
            if not isinstance(final_access, FinalAccess):
                raise PermissionError("final cache access requires a validated FinalAccess capability")
            final_access.authorize(session_id)
        elif split not in {"train", "dev"}:
            raise PermissionError("cached session split is forbidden")
        if metadata.get("schema") != "dandi688_bench_v2_session_v1":
            raise ValueError("cache schema provenance mismatch")
        if metadata.get("protocol") != protocol.protocol_dict():
            raise ValueError("cache protocol provenance mismatch")
        if metadata.get("query_index_semantics") != "inclusive end bin of a 50-bin causal window; target is velocity[query_indices]":
            raise ValueError("cache query-index semantics mismatch")
        values = {name: np.asarray(archive[name]) for name in ("neural", "velocity", "bin_edges", "query_indices", "support_indices", "carrier_indices", "activity", "carrier_counts", "carrier_angles", "channel_indices", "unit_to_electrode", "observed_mask")}
    expected = metadata.get("array_sha256", {})
    for name, value in values.items():
        if expected.get(name) != _array_sha256(value):
            raise ValueError(f"cache array hash mismatch: {name}")
    representation = str(metadata["representation"])
    neural = values["neural"]
    if representation not in {"sua", "pmua"} or neural.ndim != 2 or not (0 < neural.shape[1] <= protocol.MAX_UNITS):
        raise ValueError("cache representation/neural geometry mismatch")
    if (values["velocity"].shape != (neural.shape[0], 2) or values["bin_edges"].shape != (neural.shape[0] + 1,)
            or values["activity"].shape != (protocol.ACTIVITY_TRIALS, 100, neural.shape[1])
            or values["carrier_counts"].shape != (protocol.CARRIER_TRIALS, neural.shape[1])
            or values["carrier_angles"].shape != (protocol.CARRIER_TRIALS,)
            or values["channel_indices"].shape != (neural.shape[1],)
            or values["observed_mask"].shape != (protocol.CANONICAL_ELECTRODES,)):
        raise ValueError("cache array dimensions mismatch")
    if (values["query_indices"].ndim != 1 or not len(values["query_indices"])
            or np.any(values["query_indices"] < protocol.WINDOW_BINS - 1)
            or np.any(values["query_indices"] >= neural.shape[0])):
        raise ValueError("cache query endpoint geometry mismatch")
    if not np.all(np.isfinite(values["neural"])) or not np.all(np.isfinite(values["velocity"])) or not np.all(np.isfinite(values["activity"])) or not np.all(np.isfinite(values["carrier_counts"])):
        raise ValueError("cache non-angle arrays must be finite")
    carrier_audit = validate_carrier_angles(values["carrier_angles"])
    if (metadata.get("finite_direction_count") != carrier_audit["finite_direction_count"]
            or metadata.get("direction_design_rank") != carrier_audit["direction_design_rank"]):
        raise ValueError("cache carrier direction audit mismatch")
    selected = metadata.get("selected_original_unit_indices")
    if (not isinstance(selected, list) or values["unit_to_electrode"].shape != (len(selected),)
            or np.any(values["unit_to_electrode"] < 0) or np.any(values["unit_to_electrode"] >= protocol.CANONICAL_ELECTRODES)):
        raise ValueError("cache unit-to-electrode provenance mismatch")
    if (values["support_indices"].ndim != 1 or not len(values["support_indices"])
            or np.any(values["support_indices"] < 0) or np.any(values["support_indices"] >= neural.shape[0])
            or not set(values["carrier_indices"].tolist()) <= set(values["support_indices"].tolist())
            or set(values["carrier_indices"].tolist()) & set(values["query_indices"].tolist())):
        raise ValueError("cache support/carrier/query geometry mismatch")
    return SessionData(session_id, split, representation, **values, metadata=metadata)
