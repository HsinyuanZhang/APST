"""Official FALCON dandisets used by APST.

IDs match ``falcon_challenge.evaluator.DANDI_FOLDER_MAP`` for M1, M2, and H1.
"""
from __future__ import annotations

import os
from pathlib import Path

TASKS = ("m1", "m2", "h1")

DANDISETS = {
    "m1": {
        "dandiset": "000941",
        "subject": "sub-MonkeyL",
        "url": "https://dandiarchive.org/dandiset/000941",
        "citation": (
            "Rouse, A. G. & Schieber, M. H. (2024). FALCON Benchmark M1-A: "
            "primary motor cortex recordings in primate during reach-to-grasp task. "
            "DANDI archive. https://dandiarchive.org/dandiset/000941"
        ),
        "falcon_task": "m1",
        "n_channels": 64,
        "n_outputs": 16,
        "bin_size_ms": 20,
    },
    "m2": {
        "dandiset": "000953",
        "subject": "sub-MonkeyN",
        "url": "https://dandiarchive.org/dandiset/000953",
        "citation": (
            "Nason-Tomaszewski, S. R., Mender, M. J., & Chestek, C. (2024). "
            "FALCON Benchmark M2: primary motor cortex recordings in primate during finger movement. "
            "DANDI archive. https://dandiarchive.org/dandiset/000953"
        ),
        "falcon_task": "m2",
        "n_channels": 96,
        "n_outputs": 2,
        "bin_size_ms": 20,
    },
    "h1": {
        "dandiset": "000954",
        "subject": "sub-HumanPitt",
        "url": "https://dandiarchive.org/dandiset/000954",
        "citation": (
            "Ye, J., Collinger, J. L., & Gaunt, R. (2024). FALCON Benchmark H1: "
            "Human 7DoF Reach and Grasp Motor BCI. DANDI archive. "
            "https://dandiarchive.org/dandiset/000954"
        ),
        "falcon_task": "h1",
        "n_channels": 176,
        "n_outputs": 7,
        "bin_size_ms": 20,
    },
}

SPLITS = {
    "held_in": "held-in-calib",
    "minival": "held-in-minival",
    "held_out": "held-out-calib",
}


def default_data_root() -> Path:
    env = os.environ.get("APST_DATA_ROOT") or os.environ.get("EVAL_DATA_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / "data"


DATA_ROOT = default_data_root()
