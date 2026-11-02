"""Frozen FALCON training protocol: task geometry, rosters, and budgets.

The manifest next to this module is the single source of truth for the
per-task constants used by :mod:`apst.falcon`.  Configs embed a snapshot of
:func:`protocol_dict` and are re-validated against the manifest at load and
run time, mirroring the DANDI Sub-C module contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

PACKAGE = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE / "protocol_manifest.json"

TASKS = ("m1", "m2", "h1")


def protocol_dict() -> dict[str, Any]:
    """Return the frozen protocol snapshot embedded in training configs."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def task_protocol(task: str) -> dict[str, Any]:
    table = protocol_dict()
    if task not in TASKS:
        raise ValueError(f"unknown FALCON task {task!r}; expected one of {TASKS}")
    row = dict(table["tasks"][task])
    row["windows"] = tuple(row["windows"])
    return row


def dev_sessions(task: str) -> tuple[str, ...]:
    return tuple(task_protocol(task)["dev_sessions"])


def source_sessions(task: str) -> tuple[str, ...]:
    return tuple(task_protocol(task)["source_sessions"])


def h1_groups() -> dict[str, tuple[str, ...]]:
    """Held-out recordings grouped S6..S12 for the grouped-seven metric."""
    rows = protocol_dict()["h1_grouped_seven"]
    return {group: tuple(members) for group, members in rows.items()}


def validate(snapshot: Mapping[str, Any]) -> None:
    """Refuse a config whose protocol snapshot drifted from the manifest."""
    expected = protocol_dict()
    if json.dumps(snapshot, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(
            "config protocol snapshot differs from apst.falcon protocol_manifest.json; "
            "regenerate configs with `python -m apst.falcon init-config`"
        )
