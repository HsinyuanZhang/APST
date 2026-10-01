"""H1 training-matrix constants (MATRIX_H1_L_IDENTITY_V1_20260906.md).

Frozen constants for the H1 L x identity-usage matrix:

  - MATRIX_L      the two preregistered input lengths (user ruling 2026-09-06:
                  L in {250, 350}; 150 withdrawn as too aggressive).
  - IDENTITY_MODES the six identity feeding modes, matrix letters
                  (a) concat / (b) joined / (c) add_tail / (d) zero /
                  (e) permute / (f) proj_add (M-F250 user proposal).
  - CAL1_BUDGETS  CAL-1 prefix-cycle budgets {7, 5, 4, 3} (workorder §3; the
                  only officially confirmed gain component, +0.043).
  - DEPLOY_BUDGET deployment budget M3.
  - LODO_*        method-selection holdout: ONE DATE held out of the 6 H1
                  held-in dates (SUBMISSION-PROTOCOL stage 1 — same-session
                  minival faces are proven broken selectors, so cells are
                  compared on the held-out DATE's sessions via (session, end)
                  coordinates; stage-2 full-data retraining is NOT part of the
                  matrix).

Session/date inventory (verified against the frozen source cache
``tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt``
and ``h1_temporal_decoder_quick_product_v1.config.HELDIN_SESSIONS``; 13 held-in
sessions on 6 dates):

  1925-01-01  ses-19250101T111740, ses-19250101T112404                (2)
  1925-01-08  ses-19250108T110520, ses-19250108T111022,
              ses-19250108T111455                                     (3)
  1925-01-13  ses-19250113T120811, ses-19250113T121303                (2)
  1925-01-15  ses-19250115T110633, ses-19250115T111328                (2)
  1925-01-19  ses-19250119T113543, ses-19250119T114045                (2)
  1925-01-20  ses-19250120T115044, ses-19250120T115537                (2)

LODO rule (matrix doc §0/§3): hold out the LAST date; require >= 2 sessions on
it (single-session exams are too noisy). 1925-01-20 is the last date and
carries exactly 2 sessions, so it is held out as-is; had it carried fewer than
2, the rule would fall back to the latest date with the most sessions.

Identity declaration: B-transformer unified series, NOT SPINT. Phase-CPU-only;
no CUDA initialization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import plan

# ---------------------------------------------------------------------------
# Matrix axes (MATRIX_H1_L_IDENTITY_V1 §1, user revision 2026-09-06).
# ---------------------------------------------------------------------------

MATRIX_L: tuple[int, ...] = (250, 350)
FULL_WINDOW = 700  # settled TASK_GEOMETRY["h1"]["window"]; L-sweep only shortens
FULL_E0_DIM = 700  # C2 fused identity width [176, 700]
JOINED_DIM = 36    # true C2 pre-pool joined: carrier_pre_pool 32 + H-C 4
UNITS = 176
OUT_DIM = 7
CARRIER_DIM = 4
H1_COLUMNS = ("tx", "ty", "tz", "rx", "g1", "g2", "g3")  # frozen semantics (P2-15)

IDENTITY_MODES: tuple[str, ...] = (
    "concat",
    "joined",
    "add_tail",
    "zero",
    "permute",
    "proj_add",
)
IDENTITY_MODE_TO_MATRIX_LETTER: dict[str, str] = {
    "concat": "a",
    "joined": "b",
    "add_tail": "c",
    "zero": "d",
    "permute": "e",
    "proj_add": "f",
}
IDENTITY_DEFAULT = "concat"

# proj_add (M-F250, user proposal 2026-09-06): learnable P: Linear(d_e -> 16,
# bias=False) fuses the identity into the LOCAL channels additively —
# mathematically a rank-16 bottleneck version of the concat first layer;
# L-independent (no 700-bin template alignment). See identity_variant.
PROJ_ADD_OUT_DIM = 16

# ---------------------------------------------------------------------------
# CAL-1 calibration budgets (workorder §3; matrix doc §0 "CAL-1 全局启用").
# ---------------------------------------------------------------------------

CAL1_BUDGETS: tuple[int, ...] = (7, 5, 4, 3)  # train prefix-cycle, descending
DEPLOY_BUDGET = 3  # inference budget (H1 = M3)

# Scale bridge (NOTE P0-3 ratio form; matrix doc §0 "20y 训 /20 评").
TARGET_MULTIPLIER = 20.0
SCALE_RATIO = 400.0  # MSE(raw, 20y) == 400 * MSE(raw/20, y)
SCALE_RATIO_REL_TOL = 1e-9

# ---------------------------------------------------------------------------
# Frozen real-data authorities (read-only; never rewritten by this series).
# ---------------------------------------------------------------------------

REPO_ROOT = plan.REPO_ROOT
WORKSPACE_ROOT = REPO_ROOT.parent

# Five-arm H1 source cache (scripts/h1_sec6_fivearm_l100.py CACHE; the same
# cache backs the 20,325 full-flow / 2,908 selection faces, NOTE §3).
SOURCE_CACHE_PATH = (
    WORKSPACE_ROOT
    / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/source_cache.pt"
)
SOURCE_CACHE_SHA256 = (
    "51ff9ebfcd10a032f9c173ec426bfb4c421b751502271577582239c51bcc91b4"
)

# C2 e15 checkpoint (workorder §4 H1 row; SHA frozen in
# two_mainlines_long_v1/decoder/h1_config.py::C2_CKPT_SHA256).
C2_CKPT_PATH = Path(
    "/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/"
    "h1_series_20260830/artifacts/c2_references/c2_epoch_015.ckpt"
)
C2_CKPT_SHA256 = (
    "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"
)

# C2 EvalAI payload carrying the per-session M3 trialized activity
# [3, 1024, 176] + H-C carrier [176, 4] — the ONLY located source of trial
# activity (this is exactly what h1_temporal_decoder_quick_product_v1 used to
# materialize the cache banks). Budgets > 3 are NOT reconstructible from it.
C2_M3_PAYLOAD_PATH = (
    WORKSPACE_ROOT
    / "tfpd_exploration/submissions/evalai_h1_c2_ho_epoch15_v1/artifacts/decoder.pt"
)
C2_M3_PAYLOAD_SHA256 = (
    "91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a"
)

SURFACES: tuple[str, ...] = ("train", "minival")

# ---------------------------------------------------------------------------
# LODO date holdout (matrix doc §0 method-selection exam; fixed before run).
# ---------------------------------------------------------------------------

H1_SESSION_DATES: tuple[str, ...] = (
    "1925-01-01",
    "1925-01-08",
    "1925-01-13",
    "1925-01-15",
    "1925-01-19",
    "1925-01-20",
)

H1_SESSIONS_BY_DATE: dict[str, tuple[str, ...]] = {
    "1925-01-01": ("ses-19250101T111740", "ses-19250101T112404"),
    "1925-01-08": (
        "ses-19250108T110520",
        "ses-19250108T111022",
        "ses-19250108T111455",
    ),
    "1925-01-13": ("ses-19250113T120811", "ses-19250113T121303"),
    "1925-01-15": ("ses-19250115T110633", "ses-19250115T111328"),
    "1925-01-19": ("ses-19250119T113543", "ses-19250119T114045"),
    "1925-01-20": ("ses-19250120T115044", "ses-19250120T115537"),
}

H1_ALL_SESSIONS: tuple[str, ...] = tuple(
    session for date in H1_SESSION_DATES for session in H1_SESSIONS_BY_DATE[date]
)


def _select_lodo_holdout() -> tuple[str, tuple[str, ...]]:
    """Last date with >= 2 sessions; else the latest date with the most."""
    for date in reversed(H1_SESSION_DATES):
        if len(H1_SESSIONS_BY_DATE[date]) >= 2:
            return date, H1_SESSIONS_BY_DATE[date]
    # fallback (currently unreachable: every date carries >= 2 sessions)
    best = max(H1_SESSION_DATES, key=lambda d: (len(H1_SESSIONS_BY_DATE[d]), d))
    return best, H1_SESSIONS_BY_DATE[best]


LODO_HOLDOUT_DATE, LODO_HOLDOUT_SESSIONS = _select_lodo_holdout()
LODO_TRAIN_SESSIONS: tuple[str, ...] = tuple(
    s for s in H1_ALL_SESSIONS if s not in LODO_HOLDOUT_SESSIONS
)
LODO_EVAL_COORD_RULE = (
    "stage-1 method selection scores ONLY the holdout-date sessions, at their "
    "minival eval_mask (session, end) coordinates (five-arm minival rule; "
    "join the Original archive by (session, end), P0-4). Stage-2 full-data "
    "retraining is outside the matrix; LOSO on a retrained submission "
    "candidate is forbidden (SUBMISSION-PROTOCOL)."
)


def lodo_split() -> dict[str, Any]:
    """Validated LODO split (train sessions / holdout date + sessions)."""
    plan.require(
        LODO_HOLDOUT_DATE in H1_SESSION_DATES,
        "LODO holdout date must be one of the frozen H1 dates",
    )
    plan.require(
        len(LODO_HOLDOUT_SESSIONS) >= 2,
        f"LODO holdout date {LODO_HOLDOUT_DATE} must keep >= 2 sessions "
        "(matrix doc §3: single-session exams are too noisy)",
    )
    train, holdout = set(LODO_TRAIN_SESSIONS), set(LODO_HOLDOUT_SESSIONS)
    plan.require(
        not (train & holdout),
        "LODO train/holdout session sets must be disjoint",
    )
    plan.require(
        train | holdout == set(H1_ALL_SESSIONS),
        "LODO split must cover all 13 held-in sessions exactly once",
    )
    plan.require(len(train) >= 1, "LODO train side must be non-empty")
    return {
        "holdout_date": LODO_HOLDOUT_DATE,
        "holdout_sessions": list(LODO_HOLDOUT_SESSIONS),
        "train_sessions": list(LODO_TRAIN_SESSIONS),
        "n_holdout_sessions": len(LODO_HOLDOUT_SESSIONS),
        "n_train_sessions": len(LODO_TRAIN_SESSIONS),
        "eval_coord_rule": LODO_EVAL_COORD_RULE,
        "protocol": "SUBMISSION-PROTOCOL stage 1 (local method selection only)",
    }


# ---------------------------------------------------------------------------
# CAL-1 budget rotation: deterministic (session, epoch) -> budget mapping.
# ---------------------------------------------------------------------------


def cal1_budget(session: str, epoch: int) -> int:
    """Deterministic CAL-1 budget for one (session, epoch) training visit.

    Rule (workorder §3 "训练时每 (session, epoch) 从集合轮换取用"):
    ``idx = (stable_session_hash + epoch) % len(CAL1_BUDGETS)`` with
    ``stable_session_hash`` = sha256 of the session id (PYTHONHASHSEED-free,
    machine-stable). Any 4 consecutive epochs of a session visit all budgets
    {7, 5, 4, 3} exactly once.
    """
    digest = hashlib.sha256(str(session).encode("utf-8")).digest()
    base = int.from_bytes(digest[:8], "little")
    idx = (base + int(epoch)) % len(CAL1_BUDGETS)
    return int(CAL1_BUDGETS[idx])


# ---------------------------------------------------------------------------
# Matrix geometry helper: explicit-mapping geometry for one (L, mode) cell.
# ---------------------------------------------------------------------------


def normalize_identity_mode(mode: str) -> str:
    plan.require(
        isinstance(mode, str) and mode in IDENTITY_MODES,
        f"unknown identity mode {mode!r}; expected one of {list(IDENTITY_MODES)}",
    )
    return mode


def matrix_e0_dim(identity_mode: str) -> int:
    """Bank-side E0 width for a mode (joined carries the true 36-d object)."""
    mode = normalize_identity_mode(identity_mode)
    return JOINED_DIM if mode == "joined" else FULL_E0_DIM


def h1_matrix_geometry(window: int, identity_mode: str) -> dict[str, Any]:
    """Explicit-mapping geometry for a matrix cell (five-arm style).

    Mirrors ``scripts/h1_sec6_fivearm_l100.py::_model``'s explicit mapping but
    keeps the two PENDING sentinels of ``TASK_GEOMETRY['h1']`` so nothing is
    quietly settled: the caller must resolve the prefix sentinel with
    ``override_prefix=0`` and set the L axis with ``override_window=window``
    (only shortening the settled 700 window is allowed). The mapping may not
    back any alignment claim by itself (NOTE §6).
    """
    mode = normalize_identity_mode(identity_mode)
    plan.require(
        isinstance(window, int) and not isinstance(window, bool) and window >= 1,
        "matrix window must be a positive int",
    )
    plan.require(
        window in MATRIX_L or window == FULL_WINDOW,
        f"matrix window {window} is not one of MATRIX_L {MATRIX_L} (or the full "
        f"{FULL_WINDOW} reference window)",
    )
    geometry = dict(plan.TASK_GEOMETRY["h1"])  # window 700, sentinels kept
    geometry["e0_dim"] = matrix_e0_dim(mode)
    geometry["task"] = f"h1-matrix-L{window}-{mode}"
    return geometry


__all__ = [
    "MATRIX_L",
    "FULL_WINDOW",
    "FULL_E0_DIM",
    "JOINED_DIM",
    "UNITS",
    "OUT_DIM",
    "CARRIER_DIM",
    "H1_COLUMNS",
    "IDENTITY_MODES",
    "IDENTITY_MODE_TO_MATRIX_LETTER",
    "IDENTITY_DEFAULT",
    "PROJ_ADD_OUT_DIM",
    "CAL1_BUDGETS",
    "DEPLOY_BUDGET",
    "TARGET_MULTIPLIER",
    "SCALE_RATIO",
    "SCALE_RATIO_REL_TOL",
    "SOURCE_CACHE_PATH",
    "SOURCE_CACHE_SHA256",
    "C2_CKPT_PATH",
    "C2_CKPT_SHA256",
    "C2_M3_PAYLOAD_PATH",
    "C2_M3_PAYLOAD_SHA256",
    "SURFACES",
    "H1_SESSION_DATES",
    "H1_SESSIONS_BY_DATE",
    "H1_ALL_SESSIONS",
    "LODO_HOLDOUT_DATE",
    "LODO_HOLDOUT_SESSIONS",
    "LODO_TRAIN_SESSIONS",
    "LODO_EVAL_COORD_RULE",
    "lodo_split",
    "cal1_budget",
    "normalize_identity_mode",
    "matrix_e0_dim",
    "h1_matrix_geometry",
]
