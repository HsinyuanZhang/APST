"""Portable paths for the FALCON training release."""
from __future__ import annotations

import os
from pathlib import Path

from apst.data.catalog import default_data_root


def data_root() -> Path:
    """FALCON NWB root: ``APST_DATA_ROOT``/``EVAL_DATA_PATH`` or ``<cwd>/data``."""
    return Path(default_data_root()).resolve()


def output_root() -> Path:
    """User-writable run-artifact root (``APST_FALCON_ROOT``)."""
    return Path(os.environ.get("APST_FALCON_ROOT", Path.cwd() / "runs" / "falcon")
                ).expanduser().resolve()
