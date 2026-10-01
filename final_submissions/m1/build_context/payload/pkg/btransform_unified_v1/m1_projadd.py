"""M1 proj_add series core (ADDENDUM-UNIFIED-ADD, M1 leg; workorder 2026-09-06).

User directive 2026-09-06: restart the M1 line directly on the unified
``identity_mode="proj_add"`` interface with a P-dimension series
``SERIES_PROJ_DIMS = (16, 32)``, the M2-verified noise-aligned TRN-1 recipe,
and peak LR 1e-4 (H1 lesson: 3e-4 was never validated on M1). This module is
the CPU-safe core the training/eval script
``btransform_unified_v1/scripts/m1_projadd_series.py`` consumes; NOTHING here
touches CUDA.

Geometry (DEFERRED-receipt locked explicit mapping; the frozen plan table's
``m1 prefix=100`` is NOT used):

  W=100 (L_in=100, P=0), N=64, d_e=100, carrier=rSyn3(4), out=16,
  divisor=1 (TRN-8: M1 never divides; train and score the native target).

proj_add mode (matrix letter (f), generalized in ``identity_variant.py`` with
``proj_dim``): bank E0 [64,100] -> learnable P = Linear(100 -> R, bias=False),
R in {16, 32}; the R-wide P(E0) bracket is ADDED onto the local conv channels
by group broadcast -> tokens = token_mlp([local+P_1 ; ... ; local+P_G] |
carrier4), token_in = R + 4 (20 / 36). The identity is session-static, so the
SPD-A1 fold applies per session (:meth:`BTransformerUnifiedDecoderIdentity.
bank_static_term` — P(E0) @ W_local^T + carrier @ W_carrier^T + b, summed
over groups).

Banks (DEFERRED receipt ``mechanism_decisions_locked``): the canonical
``load_frozen_b3`` is unusable (Lightning teacher ckpt dir absent), so the
identity comes from the ``m1_family_loso_outer20120924.py::_family_bank``
mechanism — EarlyPoolEncoder STRICT-load of ``student.id_encoder`` from the
frozen B3 Sfix e11 checkpoint (sha 7976e0b0...), push the 10 chronological
calibration trials, ``finalize_identity`` == ``compute_identity
(side_features=None)``; rSyn3 does NOT enter the identity (NOTE P1-10).
Carrier = rSyn3 (sealed ``rSyn3-refit-v1.source-only.npz`` for
ses-20120926/27/28; ses-20120924 re-encoded with the SAME sealed basis +
source normalizer — the LOSO script's ``_encode_outer_carrier`` mechanism).
Per-session banks are frozen (CAL-2, M10) and every ``calibration_meta``
carries ``budget=10`` (code item S).

Data plane (two-stage protocol, stage-2 framing prepared FIRST per user
instruction; the stage-1 LOSO method-selection face stays available as a
DIAGNOSTIC side report only):
  - training = ALL 4 held-in sessions ses-20120924/26/27/28 (20120924 is a
    legal stage-2 member; fold-local train-split law, batch 32,
    SessionBatchSampler seed 42, no reshuffle; ~183k windows / ~5.7k
    updates/epoch expected — the exact count is sealed by the probe stage).
  - eval (DIAGNOSTIC) = the 31,252-window source-minival face
    (``m1_optimized_v2.source_dev`` chron-80 dev tail; Original 0.809 exposed
    reference) + the LOSO ses-20120924 face (26,496 windows; coordinates join
    per the family_flat.json precedent). After a stage-2 all-session retrain
    both faces are POLLUTED for selection — they are recorded as auxiliary
    readouts only (ADDENDUM-SUBMISSION-PROTOCOL); the official score is the
    only selector-bearing number.

Acceptance law (ADDENDUM-UNIFIED-ADD): after picking the cell, proj_add must
SIGNIFICANTLY beat SPINT Original on the governed face — operationalized as
>= ORIGINAL_MINIVAL_POOLED + ACCEPT_MARGIN = 0.839 on the minival face (or
the same +0.03 rule on the LOSO face against its Original reference).

Identity: B-transformer unified series, NOT SPINT. Historical roots are
read-only; the five-arm/LOSO scripts are read-only mechanism sources.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from . import h1_config, plan
from .bank import TaskBank, array_sha256
from .identity_variant import BTransformerUnifiedDecoderIdentity
from .model import CONV_CHANNELS, SET_DIM, S1_M2_PARAM_COUNT, READOUT_HIDDEN

# ---------------------------------------------------------------------------
# Series constants (frozen before any number is read).
# ---------------------------------------------------------------------------

SERIES_PROJ_DIMS: tuple[int, ...] = (16, 32)
M10_BUDGET = 10  # CAL budget (fold_plan.SUPPORT_TRIALS); every bank budget=10

M1_SESSIONS: tuple[str, ...] = (
    "ses-20120924",
    "ses-20120926",
    "ses-20120927",
    "ses-20120928",
)
M1_OUTER_SESSION = "ses-20120924"  # fold-0 LOSO target; legal stage-2 train member
M1_SOURCE_SESSIONS: tuple[str, ...] = ("ses-20120926", "ses-20120927", "ses-20120928")

M1_WINDOW = 100
M1_PREFIX = 0  # DEFERRED locked: L_in = W = 100 (frozen table prefix=100 NOT used)
M1_UNITS = 64
M1_E0_DIM = 100
M1_CARRIER_DIM = 4
M1_OUT_DIM = 16
M1_DIVISOR = 1.0  # TRN-8: divisor=1, native target direct

# Peak LR starts at 1e-4 (workorder ADDENDUM-UNIFIED-ADD M1 leg; H1 lesson:
# "3e-4 需按任务验证" — 3e-4 has never been validated on M1, so it is NOT the
# default; a 3e-4 arm would be a separately labeled cell).
DEFAULT_PEAK_LR = 1.0e-4
PEAK_LR_NOTE = (
    "peak LR 1e-4 default (workorder ADDENDUM-UNIFIED-ADD M1 leg); the TRN-1 "
    "3e-4 peak is unverified on M1 (H1 lesson) — any 3e-4 arm must be a "
    "separately labeled cell, never the series default"
)

# Comparison targets (family_flat.json + workorder ADDENDUM; same-face,
# annotation-exposed readings — 'same scoring surface, different system').
ORIGINAL_MINIVAL_POOLED = 0.809289  # Original on the 31,252 source-minival face
OFFICIAL_ORIGINAL = 0.649  # official October held-out Original
ORIGINAL_LOSO_OUTER_POOLED = 0.7983276120971466  # family_flat.json original
ORIGINAL_LOSO_LEAKAGE_NOTE = (
    "Original's 0.7983 on the LOSO ses-20120924 face is leakage-disclosed: its "
    "training INCLUDES ses-20120924, so it is not a true LOSO number "
    "(family_flat.json note; ADDENDUM-THREE-CHEAP)"
)
ACCEPT_MARGIN = 0.03  # ADDENDUM-UNIFIED-ADD 'significantly above Original'
ACCEPT_THRESHOLD_MINIVAL = ORIGINAL_MINIVAL_POOLED + ACCEPT_MARGIN  # 0.839289

# Diagnostic face cardinalities (frozen; probe/seal stages verify against them).
SOURCE_MINIVAL_WINDOWS = 31252  # m1_optimized_v2 chron-80 dev tail (3 source sessions)
LOSO_OUTER_WINDOWS = 26496  # family_flat.json n_points (ses-20120924 query)

SERIES_SEED = plan.SEED  # 42
SERIES_EPOCHS = plan.EPOCHS  # 24

# Read-only mechanism sources (never modified by this series).
REPO_ROOT = plan.REPO_ROOT
WORKSPACE_ROOT = REPO_ROOT.parent
DEFERRED_RECEIPT_PATH = (
    REPO_ROOT / "results/m1_fullsession_submission_v1/DEFERRED_receipt.json"
)
FAMILY_FLAT_PATH = REPO_ROOT / "results/m1_loso_outer20120924_v1/family_flat.json"
LOSO_SCRIPT_REFERENCE = REPO_ROOT / "scripts/m1_family_loso_outer20120924.py"
FULLSESSION_SCRIPT_REFERENCE = REPO_ROOT / "scripts/m1_fullsession_submission_v1.py"


def m1_projadd_geometry(proj_dim: int) -> dict[str, Any]:
    """Explicit DEFERRED-locked geometry mapping for one series cell.

    ``proj_dim`` is the identity-interface width, not a geometry field: the
    mapping is identical for both cells (the P axis lives on the model's
    ``proj_dim``). Validated against the frozen M1 constants; the plan table's
    ``m1`` prefix sentinel stays untouched (explicit mapping, prefix=0).
    """
    plan.require(
        proj_dim in SERIES_PROJ_DIMS,
        f"proj_dim {proj_dim!r} is not a series cell {SERIES_PROJ_DIMS}",
    )
    return {
        "task": "m1",
        "window": M1_WINDOW,
        "prefix": M1_PREFIX,
        "units": M1_UNITS,
        "e0_dim": M1_E0_DIM,
        "carrier_dim": M1_CARRIER_DIM,
        "out_dim": M1_OUT_DIM,
        "target_scale": M1_DIVISOR,
    }


# ---------------------------------------------------------------------------
# Parameter arithmetic (recorded, asserted by tests and the probe stage).
# ---------------------------------------------------------------------------

# Concat m1 token width: local16 + d_e(100) + carrier4.
M1_TOKEN_IN_CONCAT = CONV_CHANNELS + M1_E0_DIM + M1_CARRIER_DIM  # 120
# The m2 S1 blueprint carries token_in 70 and out_dim 2; everything else
# (frontend hidden widths, temporal core) is identical, so the concat-m1
# parameter count follows arithmetically (verified by instantiation in tests).
M1_CONCAT_PARAM_COUNT = (
    S1_M2_PARAM_COUNT
    + SET_DIM * (M1_TOKEN_IN_CONCAT - 70)
    + (M1_OUT_DIM - 2) * (READOUT_HIDDEN + 1)
)


def m1_projadd_param_count(proj_dim: int) -> int:
    """Expected decoder parameter count for a series cell.

    proj_add narrows token_mlp.0 from 120 inputs to R+4 and adds the rank-R
    projection P = Linear(100 -> R, bias=False).
    """
    plan.require(
        proj_dim in SERIES_PROJ_DIMS,
        f"proj_dim {proj_dim!r} is not a series cell {SERIES_PROJ_DIMS}",
    )
    token_in = proj_dim + M1_CARRIER_DIM
    return (
        M1_CONCAT_PARAM_COUNT
        - SET_DIM * (M1_TOKEN_IN_CONCAT - token_in)
        + proj_dim * M1_E0_DIM
    )


def build_m1_projadd_model(proj_dim: int, seed: int = SERIES_SEED) -> BTransformerUnifiedDecoderIdentity:
    """Build one series cell (CPU-safe; no CUDA initialization anywhere).

    Asserts the full geometry/interface contract before returning: l_in=100,
    d_e=100 bank-side, token_in = R + 4, e0_proj shape [R, 100] bias-free,
    identity_mode proj_add (matrix letter (f)).
    """
    model = BTransformerUnifiedDecoderIdentity(
        m1_projadd_geometry(proj_dim), seed=seed, identity_mode="proj_add", proj_dim=proj_dim
    )
    plan.require(model.identity_mode == "proj_add", "identity_mode drift")
    plan.require(model.init_meta["matrix_letter"] == "f", "matrix letter drift")
    plan.require(model.l_in == M1_WINDOW and model.prefix == M1_PREFIX, "l_in drift")
    plan.require(model.base_e0_dim == M1_E0_DIM, "bank-side d_e drift")
    plan.require(model.units == M1_UNITS and model.out_dim == M1_OUT_DIM, "units/out drift")
    plan.require(
        model.token_in == proj_dim + M1_CARRIER_DIM,
        f"token_in {model.token_in} != R+4 ({proj_dim + M1_CARRIER_DIM})",
    )
    plan.require(
        model.frontend.e0_proj is not None
        and tuple(model.frontend.e0_proj.weight.shape) == (proj_dim, M1_E0_DIM)
        and model.frontend.e0_proj.bias is None,
        "e0_proj P drift",
    )
    n_params = sum(p.numel() for p in model.parameters())
    plan.require(
        n_params == m1_projadd_param_count(proj_dim),
        f"param count {n_params} != arithmetic {m1_projadd_param_count(proj_dim)}",
    )
    return model


def build_m1_concat_model(seed: int = SERIES_SEED) -> BTransformerUnifiedDecoderIdentity:
    """Concat-identity M1 cell: token_in = 16+100+4 = 120 (matrix letter a).

    Same bank-side E0 [64,100] as proj_add; identity is concatenated into every
    token instead of a rank-R bottleneck add. Geometry mapping is shared with
    the P16 cell (e0_dim stays 100).
    """
    model = BTransformerUnifiedDecoderIdentity(
        m1_projadd_geometry(SERIES_PROJ_DIMS[0]), seed=seed, identity_mode="concat"
    )
    plan.require(model.identity_mode == "concat", "identity_mode drift")
    plan.require(model.init_meta["matrix_letter"] == "a", "matrix letter drift")
    plan.require(model.l_in == M1_WINDOW and model.prefix == M1_PREFIX, "l_in drift")
    plan.require(model.base_e0_dim == M1_E0_DIM, "bank-side d_e drift")
    plan.require(model.units == M1_UNITS and model.out_dim == M1_OUT_DIM, "units/out drift")
    plan.require(
        model.token_in == M1_TOKEN_IN_CONCAT,
        f"token_in {model.token_in} != concat 120 ({M1_TOKEN_IN_CONCAT})",
    )
    plan.require(model.frontend.e0_proj is None, "concat must not allocate e0_proj")
    n_params = sum(p.numel() for p in model.parameters())
    plan.require(
        n_params == M1_CONCAT_PARAM_COUNT,
        f"param count {n_params} != concat arithmetic {M1_CONCAT_PARAM_COUNT}",
    )
    return model


def series_cells() -> tuple[dict[str, Any], ...]:
    """The preregistered cell manifest (P16 / P32), frozen before any run."""
    return tuple(
        {
            "cell_id": f"M1-PROJADD-P{proj_dim}",
            "proj_dim": proj_dim,
            "geometry": m1_projadd_geometry(proj_dim),
            "expected_param_count": m1_projadd_param_count(proj_dim),
            "peak_lr": DEFAULT_PEAK_LR,
            "epochs": SERIES_EPOCHS,
            "seed": SERIES_SEED,
            "identity_mode": "proj_add",
            "matrix_letter": "f",
        }
        for proj_dim in SERIES_PROJ_DIMS
    )


# ---------------------------------------------------------------------------
# Scale bridge: divisor=1 identity (TRN-8).
# ---------------------------------------------------------------------------


def assert_divisor_identity(model: BTransformerUnifiedDecoderIdentity | None = None) -> dict[str, Any]:
    """Assert the M1 scale contract: divisor 1 — prediction IS the target scale.

    Geometry ``target_scale`` must be 1.0 and the numerical bridge must be the
    identity map (dividing by 1 is bit-exact; MSE(raw, y) == MSE(raw/1, y)).
    """
    geometry = m1_projadd_geometry(SERIES_PROJ_DIMS[0])
    plan.require(
        float(geometry["target_scale"]) == M1_DIVISOR,
        "M1 geometry target_scale must be 1.0 (divisor=1, TRN-8)",
    )
    if model is not None:
        plan.require(
            float(model.geometry["target_scale"]) == M1_DIVISOR,
            "model geometry target_scale drift",
        )
    y = np.array([[-1.25, 0.0, 3.5]], dtype=np.float32)
    divided = y / M1_DIVISOR
    plan.require(
        np.array_equal(y, divided),
        "divisor=1 bridge is not the identity map (bit-exactness violated)",
    )
    mse_raw = float(np.mean(np.square(y - np.array([[-1.0, 0.5, 3.0]], dtype=np.float32))))
    mse_div = float(np.mean(np.square(divided - np.array([[-1.0, 0.5, 3.0]], dtype=np.float32))))
    plan.require(
        mse_raw == mse_div,
        "MSE(raw, y) != MSE(raw/1, y) under divisor=1",
    )
    return {
        "divisor": M1_DIVISOR,
        "contract": "divisor=1: model output IS the EMG prediction in native units; never divide by 20",
        "identity_map_bitexact": True,
        "mse_identity": mse_raw,
    }


# ---------------------------------------------------------------------------
# Banks: the DEFERRED-locked _family_bank mechanism + rSyn3 carrier, CAL-2
# M10, per-session frozen. Heavy lineage imports are LAZY so tests and CPU
# tooling can import this module without the NWB/lineage stack; tests stub
# ``identity_provider`` / pass synthetic carriers.
# ---------------------------------------------------------------------------


def _streaming_path() -> None:
    p = str(WORKSPACE_ROOT / "streaming_calibration_exp")
    if p not in sys.path:
        sys.path.insert(0, p)


def _workspace_path() -> None:
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))


def load_b3_id_encoder():
    """Strict-load ``student.id_encoder`` from the frozen B3 Sfix e11 ckpt.

    The canonical ``m1_optimized_v2/calibration.py::load_frozen_b3`` is
    unusable (its Lightning teacher checkpoint directory is absent), so this
    is the DEFERRED receipt's locked alternative — byte-for-byte the
    ``m1_family_loso_outer20120924.py::_family_bank`` loading mechanism:
    SHA-gate the checkpoint, strip ``student.id_encoder.*`` keys ONLY (no
    B3/Sfix decoder weight may enter the network, NOTE P1-10), strict-load
    into EarlyPoolEncoder(1024 -> 64 pre-pool, 3-layer 64->100 post-pool),
    eval + freeze. No lineage module is modified.
    """
    _workspace_path()
    _streaming_path()
    import hashlib

    from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import (
        S_FIX_PATH,
        S_FIX_SHA256,
    )
    from src.models.components.streaming_encoders import EarlyPoolEncoder

    digest = hashlib.sha256(Path(S_FIX_PATH).read_bytes()).hexdigest()
    plan.require(
        digest == S_FIX_SHA256,
        "frozen B3 Sfix e11 checkpoint checksum drift "
        f"({digest} != {S_FIX_SHA256}); expected 7976e0b0... per DEFERRED receipt",
    )
    payload = torch.load(S_FIX_PATH, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    prefix = "student.id_encoder."
    incoming = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not any(k.startswith("student.decoder.") for k in state):
        raise RuntimeError("B3 Sfix checkpoint lacks decoder provenance keys")
    encoder = EarlyPoolEncoder(trial_length=1024, window_size=100, hidden_dim=64, num_post_layers=3)
    expected = set(encoder.state_dict())
    missing = sorted(expected - set(incoming))
    unexpected = sorted(set(incoming) - expected)
    if not incoming or missing or unexpected:
        raise RuntimeError(
            f"strict B3 id_encoder keys failed: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    encoder.load_state_dict(incoming, strict=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


class _PinnedThreads:
    """Pin torch intraop threads for a determinism-critical CPU forward.

    The EarlyPoolEncoder identity forward reduces GEMMs in a thread-count
    dependent order (empirical: bitwise-stable at <=3 torch threads, ~7e-7
    drift at 4 vs the sealed runtime cache). Banks must be BYTE-stable across
    environments (probe/train/score digest assertions + runtime-cache
    crosscheck), so the identity pass runs pinned at 1 thread and the ambient
    setting is restored afterwards. The forward is tiny (10 trials); the cost
    is negligible.
    """

    def __init__(self, n: int = 1) -> None:
        self.n = int(n)
        self.saved: int | None = None

    def __enter__(self) -> "_PinnedThreads":
        self.saved = torch.get_num_threads()
        if self.saved != self.n:
            torch.set_num_threads(self.n)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.saved is not None and self.saved != self.n:
            torch.set_num_threads(self.saved)
        self.saved = None


@torch.no_grad()
def b3_identity(encoder, calib_first10: np.ndarray) -> np.ndarray:
    """``compute_identity(side_features=None)`` semantics on 10 calib trials.

    ``encoder`` is any duck-typed EarlyPoolEncoder exposing
    ``reset_stream / push_trial / finalize_identity`` (the real frozen encoder
    or a test stub). Returns the [64, 100] identity; rSyn3 never enters.
    Computed under :class:`_PinnedThreads` so the bank bytes are independent
    of the ambient thread environment (byte-identical to the sealed
    ``m1_optimized_v2`` runtime cache, verified for ses-20120926/27/28).
    """
    from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan

    calib = torch.as_tensor(np.ascontiguousarray(calib_first10, dtype=np.float32)[None, ...])
    trials = calib[0] if calib.dim() == 4 else calib
    with _PinnedThreads(1):
        stream = encoder.reset_stream(1, m1_plan.N_UNITS, trials.device, trials.dtype)
        for trial in trials:
            encoder.push_trial(stream, trial.unsqueeze(0))
        identity = encoder.finalize_identity(stream)
    if identity.dim() == 3:
        identity = identity[0]
    if tuple(identity.shape) != (m1_plan.N_UNITS, M1_E0_DIM):
        raise RuntimeError(f"B3 identity shape drift: {tuple(identity.shape)}")
    return np.ascontiguousarray(identity.detach().cpu().numpy(), dtype=np.float32)


def default_identity_provider() -> Callable[[np.ndarray], np.ndarray]:
    """Real provider: frozen Sfix e11 encoder bound to :func:`b3_identity`."""
    encoder = load_b3_id_encoder()
    return lambda calib: b3_identity(encoder, calib)


def load_source_carriers() -> dict[str, np.ndarray]:
    """Normalized rSyn3 carriers [64, 4] for the 3 source sessions (sealed NPZ)."""
    _workspace_path()
    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank

    loaded = src_bank.load()
    return {
        name: np.ascontiguousarray(loaded["normalized"][name], dtype=np.float32)
        for name in M1_SOURCE_SESSIONS
    }


def encode_outer_carrier() -> tuple[np.ndarray, dict[str, Any]]:
    """ses-20120924 carrier via the sealed basis + source normalizer.

    Read-only reuse of ``m1_family_loso_outer20120924.py::_encode_outer_carrier``
    (the DEFERRED receipt's locked mechanism for the outer session; narrow
    role=target support read).
    """
    _workspace_path()
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.carrier_bank import _encode_session
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan, syn3
    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank
    from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan

    loaded = src_bank.load()
    path = parent_data.require_source_path(WORKSPACE_ROOT / parent_plan.SOURCE_RELATIVE[M1_OUTER_SESSION])
    if parent_data.file_sha256(path) != parent_plan.SOURCE_FILE_SHA256[M1_OUTER_SESSION]:
        raise RuntimeError("outer session source hash drift")
    record = fold_data.load_fold_session(path, role="target")
    blob = np.load(m1_plan.BANK_NPZ, allow_pickle=False)
    basis = syn3.SourceBasis(
        kind="nnmf",
        scale=np.asarray(blob["scale"]),
        dictionary=np.asarray(blob["d0"]),
        activations=np.asarray(blob["activations"]),
        order=tuple(int(v) for v in blob["nmf_order"]),
        reconstruction_digest=str(blob["reconstruction_digest"][0]),
        library={"source": "rSyn3-refit-v1.sealed"},
        extra={},
    )
    raw = _encode_session(record, basis)
    normalized = np.ascontiguousarray(
        syn3.normalize_carriers(raw, loaded["normalizer_mean"], loaded["normalizer_scale"]),
        dtype=np.float32,
    )
    meta = {
        "raw_carrier_array_digest": syn3.array_digest(np.ascontiguousarray(raw)),
        "normalized_carrier_sha256": array_sha256(normalized),
        "mechanism": "sealed-basis encode (m1_family_loso_outer20120924._encode_outer_carrier); role=target narrow support read",
    }
    return normalized, meta


def make_m1_bank(
    session: str,
    E0: np.ndarray,
    carrier: np.ndarray,
    *,
    unit_mask: np.ndarray | None = None,
    X_store: np.ndarray | None = None,
    target_store: np.ndarray | None = None,
    window_ids: np.ndarray | None = None,
) -> TaskBank:
    """Frozen per-session M10 bank with the full calibration_meta contract.

    Enforces the M1 geometry (E0 [64, 100], carrier [64, 4]), the all-true
    unit mask (units DataFrame row order, NOTE P2-13), ``budget=10`` (code
    item S), and carrier/session provenance in the meta. The window stores
    stay EMPTY by default: training/scoring windows are served by the
    fold-local FalconDataset, the bank payload is E0/carrier/unit_mask
    (fullsession precedent); tests may pass synthetic stores.
    """
    plan.require(session in M1_SESSIONS, f"unknown M1 session {session!r}")
    e0 = np.ascontiguousarray(E0, dtype=np.float32)
    t4 = np.ascontiguousarray(carrier, dtype=np.float32)
    if e0.shape != (M1_UNITS, M1_E0_DIM):
        raise RuntimeError(f"E0 shape {e0.shape} != ({M1_UNITS}, {M1_E0_DIM})")
    if t4.shape != (M1_UNITS, M1_CARRIER_DIM):
        raise RuntimeError(f"carrier shape {t4.shape} != ({M1_UNITS}, {M1_CARRIER_DIM})")
    mask = (
        np.ones(M1_UNITS, dtype=np.bool_)
        if unit_mask is None
        else np.ascontiguousarray(unit_mask, dtype=np.bool_)
    )
    meta = {
        "shape": tuple(e0.shape),
        "trial_count": M10_BUDGET,
        "estimator": (
            "B3 Sfix e11 student.id_encoder compute_identity(side_features=None) "
            "(EarlyPoolEncoder 1024->64 pre-pool, trial-mean, 3-layer affine "
            "64->100 post-pool) + rSyn3 carrier (sealed rSyn3-refit-v1 NPZ"
            + ("; sealed-basis encode for ses-20120924" if session == M1_OUTER_SESSION else "")
            + ")"
        ),
        "array_sha256": array_sha256(e0),
        "budget": M10_BUDGET,
        "surface": "m1-projadd-series",
        "session": session,
        "carrier_sha256": array_sha256(t4),
        "carrier_source": (
            "sealed-basis-encode" if session == M1_OUTER_SESSION else "rSyn3-refit-v1.source-only.npz normalized"
        ),
        "unit_mask_all_true": bool(mask.all()),
        "store_note": (
            "window store intentionally empty; training/scoring windows are served "
            "by the fold-local FalconDataset; the bank payload is E0/carrier/unit_mask"
        ),
    }
    return TaskBank(
        session_id=session,
        E0=e0,
        carrier=t4,
        unit_mask=mask,
        X_store=(
            np.zeros((0, M1_WINDOW, M1_UNITS), dtype=np.float32)
            if X_store is None
            else np.ascontiguousarray(X_store, dtype=np.float32)
        ),
        target_store=(
            np.zeros((0, M1_OUT_DIM), dtype=np.float32)
            if target_store is None
            else np.ascontiguousarray(target_store, dtype=np.float32)
        ),
        window_ids=(
            np.zeros(0, dtype=np.int64) if window_ids is None else np.ascontiguousarray(window_ids, dtype=np.int64)
        ),
        calibration_meta=meta,
    )


def build_banks(
    calib_by_session: Mapping[str, np.ndarray],
    carriers: Mapping[str, np.ndarray],
    *,
    identity_provider: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[dict[str, TaskBank], dict[str, Any]]:
    """All-session frozen banks via the _family_bank mechanism.

    ``identity_provider`` defaults to the real frozen-Sfix provider; tests
    inject synthetic providers (the duck-typed contract is
    ``calib_first10 [10, 1024, 64] -> identity [64, 100]``). Carriers are
    supplied per session (sealed NPZ for 26/27/28, ``encode_outer_carrier``
    for 24) so the rSyn3 lineage stays outside this function's heavy imports.
    """
    provider = default_identity_provider() if identity_provider is None else identity_provider
    plan.require(
        set(calib_by_session) == set(M1_SESSIONS),
        f"calib sessions {sorted(calib_by_session)} != {sorted(M1_SESSIONS)}",
    )
    plan.require(
        set(carriers) == set(M1_SESSIONS),
        f"carrier sessions {sorted(carriers)} != {sorted(M1_SESSIONS)}",
    )
    banks: dict[str, TaskBank] = {}
    report: dict[str, Any] = {}
    for session in M1_SESSIONS:
        calib = np.asarray(calib_by_session[session])
        if calib.shape[0] < M10_BUDGET:
            raise RuntimeError(f"{session}: fewer than {M10_BUDGET} calib trials")
        e0 = provider(calib[:M10_BUDGET])
        banks[session] = make_m1_bank(session, e0, np.asarray(carriers[session]))
        report[session] = {
            "e0_sha256": banks[session].calibration_meta["array_sha256"],
            "carrier_sha256": banks[session].calibration_meta["carrier_sha256"],
            "trial_count": M10_BUDGET,
            "budget": M10_BUDGET,
        }
    return banks, report


# ---------------------------------------------------------------------------
# Static fold (SPD-A1): P(E0) precomputed per session.
# ---------------------------------------------------------------------------


def bank_static_terms(
    model: BTransformerUnifiedDecoderIdentity, banks: Mapping[str, TaskBank]
) -> dict[str, torch.Tensor]:
    """Per-session folded static terms [N, SET_DIM] (proj_add group-fold)."""
    return {session: model.bank_static_term(bank) for session, bank in banks.items()}


def assert_fold_parity(
    model: BTransformerUnifiedDecoderIdentity,
    bank: TaskBank,
    x: torch.Tensor,
    static: torch.Tensor,
    tolerance: float = 1e-6,
) -> float:
    """Folded vs unfolded last-bin parity (workorder P3 caliber, FP32 1e-6)."""
    with torch.no_grad():
        unfolded = model.forward_scores(x, bank)[:, -1, :]
        folded = model.forward_static_folded(x, bank, static)
    delta = float((unfolded - folded).abs().max())
    plan.require(
        delta <= tolerance,
        f"proj_add static-fold parity {delta:.3e} exceeded {tolerance:.0e}",
    )
    return delta


# ---------------------------------------------------------------------------
# Data-plane mechanism (lazy; read-only reuse of the fold-local loaders).
# ---------------------------------------------------------------------------


def build_loso_datamodule():
    """Fold-0 LOSO datamodule exactly as the read-only mechanism scripts build it.

    Source: ``m1_family_loso_outer20120924.py::_loso_module`` /
    ``m1_fullsession_submission_v1.py::_loso_module`` (identical kwargs;
    reproduced here so the series owns its copy without importing scripts).
    """
    _workspace_path()
    _streaming_path()
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceLOSODataModule
    from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan

    dm = M1VersionBSourceLOSODataModule(
        task="m1",
        data_dir=str(m1_plan.DATA_DIR),
        source_session_names=list(m1_plan.SOURCE_SESSIONS),
        heldin_session_names=list(m1_plan.SOURCE_SESSIONS),
        batch_size=32,
        window_size=m1_plan.WINDOW,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="loso",
        loso_fold=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=10,
        heldin_query_end_trial=210,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=m1_plan.SEED,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        afc4_arm="none",
    )
    dm.setup("test")
    if dm.outer_left_out != M1_OUTER_SESSION:
        raise RuntimeError("fold-0 outer session drift")
    return dm


def assemble_training_universe(dm):
    """Combined 4-session FalconDataset + SessionBatchSampler (stage-2 law).

    Mechanism reproduced read-only from
    ``m1_fullsession_submission_v1.py::assemble_training_universe`` (DEFERRED
    receipt ``training universe`` lock): the three source sessions enter as
    the fold's train split; ses-20120924 enters under the SAME law via the
    datamodule's own ``prepare_session_data``; full-timeline eligible windows
    (query_start_trial=0, query_end_trial=None), eval_mask at the last bin,
    99-bin zero pre-history pad; sampler = SessionBatchSampler(batch=32,
    shuffle=True, seed=42, balance=False, no reshuffle).
    """
    from src.data.falcon_datamodule import FalconDataset, SessionBatchSampler

    h = dm.hparams
    task = dm._falcon_task_m1()
    source_records = dm.train_calib_heldin_sessions
    cov_mean = source_records[M1_SOURCE_SESSIONS[0]]["covariates_mean"]
    cov_std = source_records[M1_SOURCE_SESSIONS[0]]["covariates_std"]
    outer_record = dm.prepare_session_data(
        dm.target_path,
        task,
        standardize_covariates=False,
        covariates_mean=cov_mean,
        covariates_std=cov_std,
        use_intertrials=True,
    )
    combined = {M1_OUTER_SESSION: outer_record}
    for name in M1_SOURCE_SESSIONS:
        combined[name] = source_records[name]
    records = {name: combined[name] for name in M1_SESSIONS}  # fold-plan order
    dataset = FalconDataset(
        sessions_dict=records,
        calib_sessions_dict=records,
        window_size=h.window_size,
        split="train",
        calibration_n_trials=h.calibration_n_trials,
        random_calibration=False,
        smooth_calibration=h.smooth_calibration,
        max_trial_length=h.max_trial_length,
        use_calib_intertrials=h.use_calib_intertrials,
        trial_feature_type=h.trial_feature_type,
        remove_still_times=h.remove_still_times,
        remove_calib_still_times=h.remove_calib_still_times,
        use_calib_active_segments=h.use_calib_active_segments,
        calib_n_active_segments=h.calib_n_active_segments,
        interpolate_trials=h.interpolate_trials,
        interpolate_trials_kind=h.interpolate_trials_kind,
        pad_value=h.pad_value,
        query_start_trial=0,  # train law: full timeline per session
        query_end_trial=None,
        allow_empty_query_sessions=False,
    )
    sampler = SessionBatchSampler(
        dataset,
        32,
        shuffle=True,
        seed=SERIES_SEED,
        balance_sessions=False,
        reshuffle_each_epoch=False,
    )
    return dataset, sampler


def calib_trials_from_dataset(dataset) -> dict[str, np.ndarray]:
    """First M10 chronological trialized calib spikes per session (bank input)."""
    return {
        name: np.asarray(dataset.calib_trialized_neural_features[name][:M10_BUDGET])
        for name in M1_SESSIONS
    }


def sampler_digest(sampler) -> str:
    """Hash the frozen batch order (fold-datamodule digest law)."""
    import hashlib

    digest = hashlib.sha256()
    for batch_index, batch in enumerate(sampler.batched_indices):
        indices = np.asarray(batch, dtype=np.int64).reshape(-1)
        digest.update(np.asarray([batch_index, indices.size], dtype=np.int64).tobytes())
        digest.update(indices.tobytes())
    return digest.hexdigest()


def source_minival_dev_rows() -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """The 31,252-window diagnostic source-minival face (chron-80 dev tail).

    Mechanism (read-only): ``m1_optimized_v2.source_dev._split`` over the
    source-only datamodule — per source session, decoder-legal windows end
    before the 80% trial cut; dev = full-W windows starting at/after the cut.
    Returns (dev_rows as (session, start_bin), split report). This face is
    DIAGNOSTIC ONLY for this series (stage-2 training covers it).
    """
    _workspace_path()
    from tfpd_exploration.src.m1_optimized_v2 import bank as src_bank
    from tfpd_exploration.src.m1_optimized_v2 import plan as m1_plan
    from tfpd_exploration.src.m1_optimized_v2 import source_dev

    loaded = src_bank.load()
    dm = source_dev.build_source_only_datamodule(loaded)
    train_rows, dev_rows, rows_report = source_dev._split(dm)
    plan.require(
        len(dev_rows) == SOURCE_MINIVAL_WINDOWS,
        f"source-minival cardinality drift: {len(dev_rows)} != {SOURCE_MINIVAL_WINDOWS}",
    )
    report = {
        "schema": "m1_optimized_v2_known_source_chron80_split_v1 (read-only reuse)",
        "face": "source-minival (DIAGNOSTIC; polluted by stage-2 all-session training)",
        "n_windows": len(dev_rows),
        "n_train_windows": len(train_rows),
        "rows": rows_report,
        "outer_session": m1_plan.OUTER_SESSION,
        "outer_opened": False,
    }
    return list(dev_rows), report


__all__ = [
    "SERIES_PROJ_DIMS",
    "M10_BUDGET",
    "M1_SESSIONS",
    "M1_OUTER_SESSION",
    "M1_SOURCE_SESSIONS",
    "M1_WINDOW",
    "M1_PREFIX",
    "M1_UNITS",
    "M1_E0_DIM",
    "M1_CARRIER_DIM",
    "M1_OUT_DIM",
    "M1_DIVISOR",
    "DEFAULT_PEAK_LR",
    "PEAK_LR_NOTE",
    "ORIGINAL_MINIVAL_POOLED",
    "OFFICIAL_ORIGINAL",
    "ORIGINAL_LOSO_OUTER_POOLED",
    "ORIGINAL_LOSO_LEAKAGE_NOTE",
    "ACCEPT_MARGIN",
    "ACCEPT_THRESHOLD_MINIVAL",
    "SOURCE_MINIVAL_WINDOWS",
    "LOSO_OUTER_WINDOWS",
    "SERIES_SEED",
    "SERIES_EPOCHS",
    "DEFERRED_RECEIPT_PATH",
    "FAMILY_FLAT_PATH",
    "M1_TOKEN_IN_CONCAT",
    "M1_CONCAT_PARAM_COUNT",
    "m1_projadd_geometry",
    "m1_projadd_param_count",
    "build_m1_projadd_model",
    "build_m1_concat_model",
    "series_cells",
    "assert_divisor_identity",
    "load_b3_id_encoder",
    "b3_identity",
    "default_identity_provider",
    "load_source_carriers",
    "encode_outer_carrier",
    "make_m1_bank",
    "build_banks",
    "bank_static_terms",
    "assert_fold_parity",
    "build_loso_datamodule",
    "assemble_training_universe",
    "calib_trials_from_dataset",
    "sampler_digest",
    "source_minival_dev_rows",
]
