"""Portable paths for the DANDI688 Sub-C release."""
from __future__ import annotations
import os
from pathlib import Path
from apst.legacy_dandi.data import resolve_data_root

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[2]

def output_root() -> Path:
    return Path(os.environ.get("APST_SUBC_ROOT", PROJECT / "runs" / "dandi_subc")).expanduser().resolve()

def data_root() -> Path:
    return Path(resolve_data_root()).resolve()

def resolve_path(value: str | Path | None, *, default: Path) -> Path:
    if value is None:
        return default
    return Path(os.path.expandvars(str(value))).expanduser().resolve()
