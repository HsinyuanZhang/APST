"""Portable paths for the DANDI688 Sub-C release."""
from __future__ import annotations
import os
from pathlib import Path
from apst.legacy_dandi.data import resolve_data_root

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[2]

def output_root() -> Path:
    """Return a user-writable artifact root in both checkout and wheel installs.

    A package-installed ``__file__`` lives below ``site-packages``.  Falling
    back to a path derived from it made an omitted environment variable point
    at an installation directory, which is commonly read-only and is not a
    campaign workspace.  The documented ``APST_SUBC_ROOT`` remains the
    reproducible explicit setting; the fallback is only a portable local
    convenience.
    """
    return Path(
        os.environ.get("APST_SUBC_ROOT", Path.cwd() / "runs" / "dandi_subc")
    ).expanduser().resolve()

def data_root() -> Path:
    return Path(resolve_data_root()).resolve()

def resolve_path(value: str | Path | None, *, default: Path) -> Path:
    if value is None:
        return default
    return Path(os.path.expandvars(str(value))).expanduser().resolve()
