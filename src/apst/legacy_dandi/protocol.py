"""Fixed, public DANDI688 2015 center-out protocol."""

from __future__ import annotations
import json
from pathlib import Path

_PROTOCOL = json.loads((Path(__file__).with_suffix(".json")).read_text())
SOURCE_SESSIONS = tuple(_PROTOCOL["source_sessions"])
DEVELOPMENT_SESSIONS = tuple(_PROTOCOL["development_sessions"])
FINAL_SESSIONS = tuple(_PROTOCOL["final_sessions"])
BIN_SECONDS = float(_PROTOCOL["bin_seconds"])
WINDOW_BINS = int(_PROTOCOL["window_bins"])
SUPPORT_TRIALS = int(_PROTOCOL["support_trials"])
QUERY_TRIAL_INDEX = int(_PROTOCOL["query_trial_index"])
Q50_TRIAL_INDEX = QUERY_TRIAL_INDEX
MAX_UNITS = int(_PROTOCOL["max_units"])
CANONICAL_ELECTRODES = int(_PROTOCOL["canonical_electrodes"])
MOVE_START_AFTER_GO_SECONDS, MOVE_STOP_AFTER_GO_SECONDS = _PROTOCOL[
    "move_window_after_go_seconds"
]


def split_for(session_id: str) -> str:
    if session_id in SOURCE_SESSIONS:
        return "source"
    if session_id in DEVELOPMENT_SESSIONS:
        return "development"
    if session_id in FINAL_SESSIONS:
        return "final"
    raise ValueError(f"unknown protocol session: {session_id}")


def protocol_dict() -> dict:
    return dict(_PROTOCOL)


def nwb_filename(session_id: str) -> str:
    return f"{session_id}_behavior+ecephys.nwb"


ACTIVITY_TRIALS = SUPPORT_TRIALS
CARRIER_TRIALS = SUPPORT_TRIALS
