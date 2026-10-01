"""Frozen constants, task geometry, and alignment-table contract for btransform_unified_v1.

Identity declaration (workorder, mandatory): this package is the **B-transformer
unified series** — 8 learned slots + causal temporal core (dual_track B arm ->
S1-SMALL-COS -> this package). It is **NOT SPINT**. Every comparison against the
SPINT family (Original / C2 / 581727) may only be written as "same scoring
surface, different system"; writing it as "same decoder with a swapped core" or
inheriting SPINT-family numbers is forbidden. See
``docs/WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md`` and
``tfpd_exploration/docs/NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN_20260906.md``.

Phase 0 (this skeleton) is CPU-only: no CUDA initialization anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

SCHEMA = "btransform_unified_v1"

# ---------------------------------------------------------------------------
# Roots. REPO_ROOT is the btransform_unified_v1/ series root (workspace top
# level, sibling of SPINT-main/ and tfpd_exploration/). Never write outside it.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results"
WORKORDER_PATH = REPO_ROOT / "docs/WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md"
MISALIGN_NOTE_PATH = (
    REPO_ROOT.parent
    / "tfpd_exploration/docs/NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN_20260906.md"
)

# ---------------------------------------------------------------------------
# Task geometry (workorder §3 table). One structure; tasks differ only here.
#   window     = W_task, scoring-segment bins (readout at the last bin)
#   prefix     = P_task, prefix bins supplied by the real upstream stream.
#                H1 is PENDING (workorder §7 latency-budget gate: P_H1 must
#                stay disabled until a C-grade speed route is selected and
#                written into the contract), so the field carries the frozen
#                sentinel below, resolved ONLY through an explicit
#                ``override_prefix`` at model construction (same style as the
#                e0_dim sentinel; never replaced by a quiet integer here).
#   e0_dim     = identity dimension; H1 is PENDING until the NOTE §6 control
#                experiment closes (frozen sentinel below must not be replaced
#                by a quiet integer)
#   target_scale = P0-3 scale-bridge contract (m2: decoder_raw x5; m1: 1;
#                h1: train 20y / score /20 with the RATIO acceptance
#                ``MSE(raw,20y) = 400 * MSE(raw/20,y)`` at relative tolerance
#                1e-9; binding value PENDING with the §6 control)
# ---------------------------------------------------------------------------

H1_E0_DIM_PENDING = "PENDING_SEC6_CONTROL"
H1_TARGET_SCALE_PENDING = "PENDING"
H1_PREFIX_PENDING = "H1_PREFIX_PENDING"

TASK_GEOMETRY: dict[str, dict[str, Any]] = {
    "m2": {
        "window": 50,
        "prefix": 50,
        "units": 96,
        "e0_dim": 50,
        "carrier_dim": 4,
        "out_dim": 2,
        "target_scale": 5.0,
    },
    "m1": {
        "window": 100,
        "prefix": 100,
        "units": 64,
        "e0_dim": 100,
        "carrier_dim": 4,
        "out_dim": 16,
        "target_scale": 1.0,
    },
    "h1": {
        "window": 700,
        "prefix": H1_PREFIX_PENDING,
        "units": 176,
        "e0_dim": H1_E0_DIM_PENDING,
        "carrier_dim": 4,
        "out_dim": 7,
        "target_scale": H1_TARGET_SCALE_PENDING,
    },
}

# ---------------------------------------------------------------------------
# Training constants (S1-grade recipe, workorder §5). Frozen; Phase 1/2 runs
# must consume these names, not re-type the numbers.
# ---------------------------------------------------------------------------

LR_PEAK = 3.0e-4            # warmup target, cosine peak
LR_MIN_FACTOR = 0.1         # cosine floor = LR_MIN_FACTOR * LR_PEAK = 3e-5
EPOCHS = 24
WARMUP_EPOCHS = 1
EMA_DECAY = 0.9995
UNIT_DROPOUT = 0.10
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42

# ---------------------------------------------------------------------------
# Update-caliber recipe constants (REVIEW H; workorder §5, code item Q).
# The recipe is transferred by UPDATE COUNT, not by epoch count: warmup /
# total / EMA horizon are frozen in updates because an epoch means a
# different number of optimizer steps per task. Phase 1/2 training loops
# must consume ``recipe_updates`` and these names, not re-type the numbers.
#   m2 3165 upd/ep  : S1 parity (m2_b_small_stability_v1/config.py
#                     UPDATES_PER_EPOCH); 24 ep = 75,960 updates.
#   h1 731 upd/ep   : formal12 receipt updates_per_epoch; 24 ep = 17,544
#                     updates; EMA horizon ~2.7 ep; warmup 1 ep = 731 steps.
#   m1 None         : PENDING — the M1 update caliber is fixed in Phase 2a
#                     when the M1 loader lands; ``recipe_updates`` refuses.
# ---------------------------------------------------------------------------

UPDATES_PER_EPOCH: dict[str, int | None] = {
    "m2": 3165,
    "m1": None,
    "h1": 731,
}
EMA_HORIZON_UPDATES = 2000  # EMA 0.9995 horizon ~ 2000 updates (REF TRN-1)

# Workorder §8 resource discipline: two-GPU machine shared with another agent.
# Before ANY GPU phase: preflight (external pid + UUID), pin
# CUDA_VISIBLE_DEVICES to exactly one UUID below, FP32. Phase 0 touches none.
GPU_UUIDS: tuple[str, ...] = (
    "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
    "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
)

# ---------------------------------------------------------------------------
# Alignment-table contract (NOTE §6 six-row table). Every scoring receipt for
# H1/M1 must fill all six fields by these exact names; a missing row forbids
# claiming alignment (or claiming "the core does not work").
# ---------------------------------------------------------------------------

ALIGNMENT_TABLE_FIELDS: list[str] = [
    "system",
    "consumer",
    "calibration_object",
    "scoring_surface",
    "scale",
    "single_difference_vs_historical_best",
]


class BTransformerUnifiedError(RuntimeError):
    """Contract violation inside btransform_unified_v1."""


def require(condition: Any, message: str) -> None:
    """Raise BTransformerUnifiedError unless ``condition`` holds."""
    if not condition:
        raise BTransformerUnifiedError(message)


def task_geometry(task: str) -> dict[str, Any]:
    """Return a copy of the frozen geometry for ``task`` ('m2' | 'm1' | 'h1')."""
    require(
        task in TASK_GEOMETRY,
        f"unknown task {task!r}; expected one of {sorted(TASK_GEOMETRY)}",
    )
    return dict(TASK_GEOMETRY[task])


def resolved_e0_dim(geometry: Mapping[str, Any]) -> int:
    """Return the integer identity dimension or refuse (H1 PENDING sentinel)."""
    e0_dim = geometry["e0_dim"]
    require(
        isinstance(e0_dim, int) and not isinstance(e0_dim, bool) and e0_dim >= 1,
        "e0_dim not resolved: "
        f"{e0_dim!r}. H1 d_e is PENDING until the NOTE §6 control experiment "
        "closes; build temporary dev geometry by passing an explicit mapping, "
        "never by editing TASK_GEOMETRY.",
    )
    return int(e0_dim)


def resolved_prefix(geometry: Mapping[str, Any], override_prefix: int | None = None) -> int:
    """Return the integer prefix or refuse (H1 PENDING sentinel, code item P).

    An integer geometry prefix is frozen: ``override_prefix`` may not replace
    it. The ``H1_PREFIX_PENDING`` sentinel (workorder §7 latency-budget gate:
    P_H1 stays disabled until a C-grade speed route is written into the
    contract) resolves ONLY through an explicit ``override_prefix`` — a
    temporary dev value that backs no alignment claim.
    """
    prefix = geometry["prefix"]
    if isinstance(prefix, int) and not isinstance(prefix, bool):
        require(
            override_prefix is None,
            f"override_prefix may only resolve the {H1_PREFIX_PENDING!r} "
            f"sentinel, not replace the frozen integer prefix {prefix}",
        )
        return int(prefix)
    require(
        isinstance(prefix, str) and prefix == H1_PREFIX_PENDING,
        f"prefix must be an int or the {H1_PREFIX_PENDING!r} sentinel, got {prefix!r}",
    )
    require(
        isinstance(override_prefix, int)
        and not isinstance(override_prefix, bool)
        and override_prefix >= 0,
        "prefix not resolved: "
        f"{prefix!r}. The H1 prefix is PENDING until the workorder §7 "
        "latency-budget gate selects a C-grade speed route; dev geometry must "
        "pass override_prefix explicitly, never edit TASK_GEOMETRY.",
    )
    return int(override_prefix)


def recipe_updates(task: str, epochs: int = EPOCHS) -> dict[str, int]:
    """Return the S1 recipe for ``task`` in update caliber (code item Q).

    warmup_updates = UPDATES_PER_EPOCH[task] * WARMUP_EPOCHS;
    total_updates  = UPDATES_PER_EPOCH[task] * epochs;
    ema_horizon_updates = EMA_HORIZON_UPDATES (EMA 0.9995 ~ 2000 updates).
    Raises NotImplementedError while ``UPDATES_PER_EPOCH[task]`` is None
    (m1 is fixed in Phase 2a).
    """
    require(
        task in UPDATES_PER_EPOCH,
        f"unknown task {task!r}; expected one of {sorted(UPDATES_PER_EPOCH)}",
    )
    per_epoch = UPDATES_PER_EPOCH[task]
    if per_epoch is None:
        raise NotImplementedError(
            f"UPDATES_PER_EPOCH[{task!r}] is PENDING: the M1 update caliber is "
            "fixed in Phase 2a when the M1 loader lands; refusing to convert "
            "epochs to updates before that."
        )
    return {
        "warmup_updates": int(per_epoch) * int(WARMUP_EPOCHS),
        "total_updates": int(per_epoch) * int(epochs),
        "ema_horizon_updates": int(EMA_HORIZON_UPDATES),
    }


def alignment_table(task: str) -> dict[str, str]:
    """Six-row alignment-table template for a scoring receipt (NOTE §6).

    Returns one string per field of ALIGNMENT_TABLE_FIELDS. Template rows carry
    the contract text that a receipt must replace with concrete values (or the
    PENDING marker where the §6 control has not closed). The returned mapping
    is checked to contain exactly the six frozen field names.
    """
    geometry = task_geometry(task)
    scale = geometry["target_scale"]
    if scale == 5.0:
        scale_text = (
            "x5 decoder_raw contract: train on 5x native target, score pred/5 "
            "against native; scale_bridge.assert_scale_bridge must pass"
        )
    elif scale == 1.0:
        scale_text = (
            "divisor=1: train and score native target (M1 never divides by 20)"
        )
    else:
        scale_text = (
            "PENDING: H1 trains on 20y and scores pred/20 against native; "
            "ratio acceptance MSE(raw,20y) == 400 * MSE(raw/20,y) within "
            "relative tolerance 1e-9 (NOT a difference of 400) and "
            "pred_std>=0.01*target_std (P0-3, H1_SCALE_RATIO); binding only "
            "after the NOTE §6 control closes"
        )
    if task == "m2":
        calib = (
            "REQUIRED per receipt: shape=[96,50] E0 + MOVE-T4 [96,4]; fill "
            "trial_count / estimator / array_sha256 (P0-2)"
        )
    elif task == "m1":
        calib = (
            "REQUIRED per receipt: B3 post_pool [64,100] drawn ONLY from "
            "student.id_encoder (never B3/Sfix decoder weights) + rSyn3 [64,4]; "
            "fill trial_count / estimator / array_sha256 (P0-2, P1-10)"
        )
    else:
        calib = (
            "PENDING §6 control: concat [176,d_e] E0 vs C2/Original identity "
            "usage not yet shown to be the same scientific quantity; fill "
            "shape / trial_count / estimator / array_sha256 once bound (P0-2)"
        )
    table = {
        "system": (
            "new B-transformer (btransform_unified_v1; 8-slot + causal temporal "
            "core CausalPE4) — NOT SPINT"
        ),
        "consumer": (
            "8-slot Transformer (this series). Historical best consumer = SPINT; "
            "comparisons are 'same scoring surface, different system' only (P0-1)"
        ),
        "calibration_object": calib,
        "scoring_surface": (
            "REQUIRED per receipt: n_points / endpoint law / cache sha256 / "
            "pooled vs session-mean (report both, never subtract across rows, "
            "P1-11)"
        ),
        "scale": scale_text,
        "single_difference_vs_historical_best": (
            "REQUIRED per receipt: exactly ONE item differing from the "
            "historical-best row; more than one item is not an ablation (NOTE §6)"
        ),
    }
    require(
        list(table.keys()) == ALIGNMENT_TABLE_FIELDS,
        "alignment table must carry exactly ALIGNMENT_TABLE_FIELDS in order",
    )
    require(
        all(isinstance(v, str) and v for v in table.values()),
        "alignment table rows must be non-empty strings",
    )
    return table


__all__ = [
    "SCHEMA",
    "REPO_ROOT",
    "RESULT_ROOT",
    "WORKORDER_PATH",
    "MISALIGN_NOTE_PATH",
    "H1_E0_DIM_PENDING",
    "H1_TARGET_SCALE_PENDING",
    "H1_PREFIX_PENDING",
    "TASK_GEOMETRY",
    "LR_PEAK",
    "LR_MIN_FACTOR",
    "EPOCHS",
    "WARMUP_EPOCHS",
    "UPDATES_PER_EPOCH",
    "EMA_HORIZON_UPDATES",
    "EMA_DECAY",
    "UNIT_DROPOUT",
    "BATCH_SIZE",
    "WEIGHT_DECAY",
    "GRAD_CLIP",
    "SEED",
    "GPU_UUIDS",
    "ALIGNMENT_TABLE_FIELDS",
    "BTransformerUnifiedError",
    "require",
    "task_geometry",
    "resolved_e0_dim",
    "resolved_prefix",
    "recipe_updates",
    "alignment_table",
]
