"""Load DANDI 000688 SUA NWB files.

Binned M1 spike counts and cursor velocity are read with pynwb. This is the
raw session surface; it does not apply the APST calibration protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from apst.data.catalog import DANDI688_SUBJECTS, DANDISETS, default_data_root

BIN_SECONDS = 0.020


def dandi688_root(root: str | Path | None = None) -> Path:
    base = Path(root).expanduser().resolve() if root else default_data_root()
    for candidate in (base / "000688", base / "dandi_000688", base):
        if any((candidate / subject).is_dir() for subject in DANDI688_SUBJECTS):
            return candidate
    raise FileNotFoundError(
        f"DANDI 000688 not found under {base / '000688'}. "
        "Run `python -m apst.data.download --tasks dandi688` or set APST_DATA_ROOT."
    )


def _session_id(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_behavior+ecephys")


def _m1_unit_indices(nwb: Any, units_df: Any) -> np.ndarray:
    electrodes = nwb.electrodes.to_dataframe()
    required = {"group_name", "location"}
    if not required <= set(electrodes.columns):
        return np.arange(len(units_df), dtype=np.int64)
    keep_ids = set()
    for electrode_id, row in electrodes.iterrows():
        if str(row.group_name) == "electrode_group_M1" and str(row.location) == "Primary Motor Cortex":
            keep_ids.add(int(electrode_id))
    selected: list[int] = []
    for unit_row, region in enumerate(units_df["electrodes"]):
        indices = getattr(region, "index", None)
        if indices is None or len(indices) != 1:
            continue
        if int(indices[0]) in keep_ids:
            selected.append(unit_row)
    if not selected:
        raise ValueError("NWB has no M1 sorted units")
    return np.asarray(selected, dtype=np.int64)


def list_dandi688_sessions(
    subject: str | None = None,
    *,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List downloaded DANDI 000688 NWB files."""
    subjects = DANDI688_SUBJECTS if subject is None else (subject,)
    unknown = [s for s in subjects if s not in DANDI688_SUBJECTS]
    if unknown:
        raise ValueError(f"unknown subject(s): {unknown}; expected one of {DANDI688_SUBJECTS}")
    base = dandi688_root(root)
    rows = []
    for sub in subjects:
        folder = base / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.nwb")):
            sid = _session_id(path)
            rows.append(
                {
                    "dataset": "dandi688",
                    "subject": sub,
                    "session": sid,
                    "path": path,
                }
            )
    if not rows:
        raise FileNotFoundError(f"no DANDI 000688 NWB files under {base}")
    return rows


def load_dandi688_nwb(path: str | Path) -> dict[str, Any]:
    """Load one DANDI 000688 NWB: 20 ms M1 counts and cursor velocity."""
    from pynwb import NWBHDF5IO

    path = Path(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
        selected = _m1_unit_indices(nwb, units)
        spikes = [np.asarray(units.iloc[int(i)]["spike_times"], dtype=np.float64) for i in selected]
        all_spikes = np.concatenate(spikes) if spikes else np.zeros(0, dtype=np.float64)
        if all_spikes.size == 0:
            raise ValueError(f"{path.name}: no spikes in selected M1 units")
        edges = np.arange(float(all_spikes.min()), float(all_spikes.max()) + BIN_SECONDS, BIN_SECONDS)
        neural = np.stack([np.histogram(s, bins=edges)[0] for s in spikes], axis=1).astype(np.float32)
        velocity_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        vel_times = np.asarray(velocity_series.timestamps[:], dtype=np.float64)
        vel_raw = np.asarray(velocity_series.data[:], dtype=np.float64)
        centers = (edges[:-1] + edges[1:]) / 2.0
        velocity = np.stack(
            [np.interp(centers, vel_times, vel_raw[:, axis], left=0.0, right=0.0) for axis in range(vel_raw.shape[1])],
            axis=1,
        ).astype(np.float32)
        trials_df = nwb.intervals["trials"].to_dataframe()
        trials = []
        for raw_row, row in trials_df.iterrows():
            trials.append(
                {
                    "raw_trial_row": int(raw_row),
                    "start_time": float(row.start_time),
                    "stop_time": float(row.stop_time),
                    "result": None if row.get("result") is None else str(row.get("result")),
                    "go_cue_time": None if row.get("go_cue_time") is None or not np.isfinite(row.get("go_cue_time")) else float(row.go_cue_time),
                    "target_dir": None if row.get("target_dir") is None or not np.isfinite(row.get("target_dir")) else float(row.target_dir),
                }
            )
    session = _session_id(path)
    subject = session.split("_", 1)[0]
    meta = DANDISETS["dandi688"]
    return {
        "dataset": "dandi688",
        "subject": subject,
        "session": session,
        "path": path,
        "neural": neural,
        "velocity": velocity,
        "bin_edges": edges.astype(np.float64),
        "trials": trials,
        "n_units": int(neural.shape[1]),
        "n_outputs": int(meta["n_outputs"]),
        "bin_size_ms": int(meta["bin_size_ms"]),
        "unit_indices": selected,
    }


def load_dandi688_session(
    session: str,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Load one session id such as ``sub-C_ses-CO-20150716``."""
    matches = [row for row in list_dandi688_sessions(root=root) if row["session"] == session]
    if not matches:
        available = [row["session"] for row in list_dandi688_sessions(root=root)]
        raise KeyError(f"DANDI 000688 session {session!r} not found; available: {available[:12]}...")
    return load_dandi688_nwb(matches[0]["path"])
