"""Load official FALCON NWB files.

Neural, kinematics, trial-change, and eval-mask arrays come from
``falcon_challenge.dataloaders.load_nwb``, the same reader used by the
EvalAI evaluator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from apst.data.catalog import DANDISETS, FALCON_TASKS, SPLITS, default_data_root

_SES_TOKEN = "_ses-"


def _task_root(task: str, root: Path) -> Path:
    dandiset = DANDISETS[task]["dandiset"]
    candidate = root / dandiset
    if candidate.is_dir():
        return candidate
    named = root / task
    if named.is_dir():
        return named
    raise FileNotFoundError(
        f"FALCON {task.upper()} data not found under {root / dandiset} or {named}. "
        "Run `python -m apst.data.download --tasks "
        f"{task}` or set APST_DATA_ROOT."
    )


def _session_id(path: Path) -> str:
    stem = path.stem
    if _SES_TOKEN not in stem:
        return stem
    rest = stem.split(_SES_TOKEN, 1)[1]
    return rest.split("_", 1)[0]


def list_sessions(
    task: str,
    split: str = "held_in",
    *,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return NWB paths for one FALCON split, sorted by session id."""
    task = task.lower()
    if task not in FALCON_TASKS:
        raise ValueError(f"unknown FALCON task {task!r}; expected one of {FALCON_TASKS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {tuple(SPLITS)}")
    data_root = Path(root).expanduser().resolve() if root else default_data_root()
    subject = DANDISETS[task]["subject"]
    folder = _task_root(task, data_root) / f"{subject}-{SPLITS[split]}"
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    rows = []
    for path in sorted(folder.glob("*.nwb")):
        rows.append(
            {
                "task": task,
                "split": split,
                "session": _session_id(path),
                "path": path,
            }
        )
    if not rows:
        raise FileNotFoundError(f"no NWB files in {folder}")
    return rows


def load_nwb_file(path: str | Path, task: str):
    """Load one NWB file with the official FALCON reader.

    Returns ``(neural, covariates, trial_change, eval_mask)``.
    """
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    task = task.lower()
    if task not in FALCON_TASKS:
        raise ValueError(f"unknown FALCON task {task!r}")
    falcon_task = getattr(FalconTask, DANDISETS[task]["falcon_task"])
    return load_nwb(Path(path), falcon_task)


def load_session(
    task: str,
    session: str,
    split: str = "held_in",
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Load one session from a FALCON split."""
    matches = [row for row in list_sessions(task, split, root=root) if row["session"] == session]
    if not matches:
        available = [row["session"] for row in list_sessions(task, split, root=root)]
        raise KeyError(f"{task}/{split} session {session!r} not found; available: {available}")
    row = matches[0]
    neural, covariates, trial_change, eval_mask = load_nwb_file(row["path"], task)
    meta = DANDISETS[task]
    return {
        "task": task,
        "split": split,
        "session": session,
        "path": row["path"],
        "neural": neural,
        "covariates": covariates,
        "trial_change": trial_change,
        "eval_mask": eval_mask,
        "n_channels": int(meta["n_channels"]),
        "n_outputs": int(meta["n_outputs"]),
        "bin_size_ms": int(meta["bin_size_ms"]),
    }
