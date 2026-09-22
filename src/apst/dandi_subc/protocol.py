"""Frozen, 2015-only DANDI 000688 SUA/PMUA data-access protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
from typing import Final

PACKAGE_ROOT: Final = Path(__file__).resolve().parent
MANIFEST_PATH: Final = PACKAGE_ROOT / "protocol_manifest.json"
DEFAULT_RAW_ROOT: Final = Path(os.environ.get("APST_DATA_ROOT", ".")).expanduser().resolve()
MANIFEST_SHA256: Final = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
BIN_SECONDS: Final = 0.020
WINDOW_BINS: Final = 50
ACTIVITY_TRIALS: Final = 33
CARRIER_TRIALS: Final = 33
Q50_TRIAL_INDEX: Final = 50
MAX_UNITS: Final = 100
CANONICAL_ELECTRODES: Final = 96
MOVE_START_AFTER_GO_SECONDS: Final = 0.100
MOVE_STOP_AFTER_GO_SECONDS: Final = 0.600


def _load_manifest() -> dict:
    blob = MANIFEST_PATH.read_bytes()
    if hashlib.sha256(blob).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("strict DANDI manifest SHA-256 drift")
    payload = json.loads(blob)
    if payload.get("task") != "CO" or payload.get("split_counts") != [27, 6, 6]:
        raise RuntimeError("unexpected strict DANDI manifest shape")
    return payload


_MANIFEST = _load_manifest()
_SPLITS = _MANIFEST["session_splits"]
TRAIN_SESSIONS: Final = tuple(s for s in _SPLITS["train"] if "-2015" in s)
DEV_SESSIONS: Final = tuple(_SPLITS["val"])
FINAL_SESSIONS: Final = tuple(_SPLITS["test"])

if len(TRAIN_SESSIONS) != 18 or any("-2013" in s for s in TRAIN_SESSIONS):
    raise RuntimeError("2015-only source roster is not exactly 18 sessions")
if len(DEV_SESSIONS) != 6 or len(FINAL_SESSIONS) != 6:
    raise RuntimeError("development/final roster drift")


def split_for(session_id: str) -> str:
    if session_id in TRAIN_SESSIONS:
        return "train"
    if session_id in DEV_SESSIONS:
        return "dev"
    if session_id in FINAL_SESSIONS:
        return "final"
    raise ValueError(f"session is not in the strict DANDI manifest: {session_id}")


def assert_authorized(session_id: str, purpose: str) -> str:
    if purpose not in {"source", "development"}:
        raise ValueError("purpose must be 'source' or 'development'")
    split = split_for(session_id)
    if split == "final":
        raise PermissionError("final-test DANDI NWB is forbidden by the v2 data loader")
    if purpose == "source" and split != "train":
        raise PermissionError(f"source purpose cannot open {split} session {session_id}")
    if purpose == "development" and split != "dev":
        raise PermissionError(f"development purpose cannot open {split} session {session_id}")
    return split


def protocol_dict() -> dict:
    return {
        "schema": "dandi688_bench_v2_2015_only", "manifest_sha256": MANIFEST_SHA256,
        "train_sessions": list(TRAIN_SESSIONS), "dev_sessions": list(DEV_SESSIONS),
        "final_sessions": list(FINAL_SESSIONS), "bin_seconds": BIN_SECONDS, "window_bins": WINDOW_BINS,
        "activity_trials": ACTIVITY_TRIALS, "carrier_trials": CARRIER_TRIALS, "q50_trial_index": Q50_TRIAL_INDEX,
        "move_window_after_go_seconds": [MOVE_START_AFTER_GO_SECONDS, MOVE_STOP_AFTER_GO_SECONDS],
        "move_window_note": "DANDI event-relative [go+0.1,go+0.6) is a 500 ms task anchor; it is not an M2 absolute timestamp claim",
        "max_units": MAX_UNITS, "canonical_electrodes": CANONICAL_ELECTRODES,
    }
