"""Per-task bank builders with frozen contracts (m2 live; h1 matrix live at M3; m1 gated).

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import h1_config as _H1
from . import plan
from .bank import TaskBank, array_sha256

# P1a (workorder §7): m2 reads the frozen read-only dual_track cache built by
# stage0 on 2026-09-05 (surface session banks + E0/T provenance). This series
# NEVER rewrites that cache and never opens hidden/test records.
_M2_CACHE_ROOT = (
    plan.REPO_ROOT.parent
    / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
)
_M2_SURFACES = ("source_train", "source_minival", "ext4")
_M2_BUDGET = 33  # CAL-2 M33, S1 parity (SUPPORT_HORIZON = 33)


# ---------------------------------------------------------------------------
# H1 matrix loaders (Phase 2b skeleton; frozen read-only sources, SHA-pinned).
# ---------------------------------------------------------------------------

_H1_MEMO: dict[str, Any] = {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _h1_missing(missing: list[Path], purpose: str) -> NotImplementedError:
    return NotImplementedError(
        f"Phase 2b H1 bank: {purpose} — missing frozen file(s): "
        + "; ".join(str(p) for p in missing)
        + ". This package does not fabricate paths; see h1_config for the "
        "full frozen-source inventory."
    )


def _h1_source_cache() -> dict[str, Any]:
    """Five-arm source cache (SHA-pinned, memoized, read-only)."""
    if "source_cache" not in _H1_MEMO:
        path = _H1.SOURCE_CACHE_PATH
        if not path.is_file():
            raise _h1_missing([path], "window/neural/target access needs the frozen source cache")
        digest = _file_sha256(path)
        plan.require(
            digest == _H1.SOURCE_CACHE_SHA256,
            f"H1 source cache SHA drift: {digest} != {_H1.SOURCE_CACHE_SHA256}",
        )
        _H1_MEMO["source_cache"] = torch.load(path, map_location="cpu", weights_only=False)
    return _H1_MEMO["source_cache"]


def _h1_payload() -> dict[str, Any]:
    """C2 EvalAI payload (M3 trialized activity + H-C carrier; SHA-pinned)."""
    if "payload" not in _H1_MEMO:
        path = _H1.C2_M3_PAYLOAD_PATH
        if not path.is_file():
            raise _h1_missing([path], "C2 trialized activity needs the M3 payload")
        digest = _file_sha256(path)
        plan.require(
            digest == _H1.C2_M3_PAYLOAD_SHA256,
            f"C2 M3 payload SHA drift: {digest} != {_H1.C2_M3_PAYLOAD_SHA256}",
        )
        _H1_MEMO["payload"] = torch.load(path, map_location="cpu", weights_only=False)
    return _H1_MEMO["payload"]


def _h1_payload_arrays(session: str) -> tuple[np.ndarray, np.ndarray]:
    """(activity [M,1024,176], carrier [176,4]) for a session from the payload."""
    payload = _h1_payload()
    rows = payload.get("sessions", {})
    by_token: dict[str, Any] = {}
    for key, row in rows.items():
        token = str(row["session"])
        by_token[token] = row
        by_token.setdefault(token.replace("ses-", ""), row)
    row = by_token.get(session) or by_token.get(session.replace("ses-", ""))
    if row is None:
        raise plan.BTransformerUnifiedError(
            f"session {session!r} has no M3 payload row (payload carries "
            f"{len(rows)} rows)"
        )
    activity = np.ascontiguousarray(row["identity"], dtype=np.float32)
    carrier = np.ascontiguousarray(row["carrier"], dtype=np.float32)
    plan.require(
        activity.ndim == 3
        and activity.shape[1:] == (1024, _H1.UNITS)
        and carrier.shape == (_H1.UNITS, _H1.CARRIER_DIM),
        f"payload array drift for {session}: activity {activity.shape}, carrier {carrier.shape}",
    )
    return activity, carrier


def _h1_materializer():
    """Frozen C2 materializer (two_mainlines_long_v1, SHA self-verified)."""
    if "materializer" not in _H1_MEMO:
        if not _H1.C2_CKPT_PATH.is_file():
            raise _h1_missing([_H1.C2_CKPT_PATH], "C2 materializer needs the frozen e15 checkpoint")
        root = str(plan.REPO_ROOT.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import (
                load_frozen_c2_materializer,
            )
        except Exception as exc:  # unavailable -> explicit stub, no fallback math
            raise NotImplementedError(
                "Phase 2b H1 bank: cannot import the frozen C2 materializer "
                "(tfpd_exploration.src.two_mainlines_long_v1.decoder."
                f"h1_calibration.load_frozen_c2_materializer): {exc!r}"
            ) from exc
        # load_frozen_c2_materializer verifies the ckpt SHA (ce46267e...) itself.
        _H1_MEMO["materializer"] = load_frozen_c2_materializer(_H1.C2_CKPT_PATH)
    return _H1_MEMO["materializer"]


def _h1_windows(neural: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    """Five-arm windowing generalized to any L (end-anchored, left zero-pad).

    Exactly ``scripts/h1_sec6_fivearm_l100.py::_windows`` with WINDOW replaced
    by the matrix L: window ``i`` covers bins ``[end-L+1, end]``; when
    ``start < 0`` the head is zero-padded (full-flow early-segment semantics).
    """
    hist = np.zeros((len(ends), int(window), neural.shape[1]), dtype=np.float32)
    for i, end in enumerate(ends):
        start = int(end) - int(window) + 1
        if start >= 0:
            hist[i] = neural[start : int(end) + 1]
        else:
            hist[i, -int(end) - 1 :] = neural[: int(end) + 1]
    return hist


def _m2_dual_track():
    root = str(plan.REPO_ROOT.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
    from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

    return old_data, old_plan


def build_m2_bank(surface: str, session: str, *, budget: int = _M2_BUDGET) -> TaskBank:
    """Phase 1a — M2 real-data TaskBank from the frozen dual_track cache.

    Calibration-object source table (workorder §4; REVIEW C; NOTE P0-2):
      - encoder: B3S (hidden 64, side_dim 4) from the M2 T4 mainline SPINT
        champion checkpoint
        ``streaming_calibration_exp/outputs/streaming_calibration/
        m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/
        best.ckpt`` with sha256
        ``25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e``
        (as recorded in ``sua_exploration/evalai_t4_m2/export_t4_payload.py``).
      - extraction: ``m2_dual_track_v1/champion.py::native_e0_and_u``
        (push_trial/finalize; NOT batched-mean) -> E0 [96, 50] + MOVE-T4
        carrier [96, 4] (``fit_move_t4`` -> ``t4_from_trial_sums``, least
        squares on [1, cos, sin]).
      - budget: **M33** (S1 parity, ``SUPPORT_HORIZON = 33``); Phase 1a is
        CAL-2 (single frozen budget), so ``budget`` must be 33 here.

    Contract (disjoint discipline inherited from
    ``tfpd_exploration/src/m2_dual_track_v1/data.py``):
      - Source-only training surface + ext-4 external evaluation surface;
        the frozen cache contains no hidden/test records and this builder
        never opens NWB files, only the compact cache arrays.
      - Calibration arrays are mask-filtered 100-bin interpolated views;
        query arrays are a DIFFERENT timeline with 49-bin leading padding;
        boundaries are mapped by original trial IDs / raw intervals — never
        by comparing the two array offsets directly.
      - Geometry m2 {window 50, prefix 0 for P1a, units 96, e0_dim 50,
        carrier 4 (MOVE-T4), out 2}; targets are stored at native scale and
        training multiplies by 5 (decoder_raw contract, validated by
        ``scale_bridge.assert_scale_bridge``).
      - ``calibration_meta`` carries shape / trial_count / estimator /
        array_sha256 / budget (NOTE P0-2 + code item S) plus the frozen
        dual_track provenance dict (e0/t4/eligible_start digests).
    """
    plan.require(surface in _M2_SURFACES, f"unknown m2 surface {surface!r} (expected one of {_M2_SURFACES})")
    plan.require(int(budget) == _M2_BUDGET, f"P1a is CAL-2 M33: budget must be {_M2_BUDGET}, got {budget}")
    plan.require(_M2_CACHE_ROOT.is_dir(), f"frozen m2 cache root missing: {_M2_CACHE_ROOT}")
    old_data, old_plan = _m2_dual_track()
    sessions = list(old_plan.EXT4_SESSIONS if surface == "ext4" else old_plan.HELDIN_SESSIONS)
    plan.require(session in sessions, f"session {session!r} is not on surface {surface!r}")
    sb = old_data.load_session_bank(surface, session, device="cpu")
    e0 = np.ascontiguousarray(sb.E0.detach().cpu().numpy(), dtype=np.float32)
    carrier = np.ascontiguousarray(sb.T.detach().cpu().numpy(), dtype=np.float32)
    unit_mask = np.ascontiguousarray(sb.unit_mask.detach().cpu().numpy(), dtype=np.bool_)
    store = np.asarray(sb.X_store)
    starts = np.ascontiguousarray(np.asarray(sb.eligible_starts), dtype=np.int64)
    window = int(old_plan.WINDOW)
    if store.ndim == 3:  # already-windowed store (defensive; cache is 2-D today)
        x_store = np.ascontiguousarray(store, dtype=np.float32)
    else:
        # 2-D padded-timeline store [T_total, N]: materialize each window as
        # store[start : start + WINDOW] (exactly dual_track's gather rule).
        x_store = np.ascontiguousarray(
            np.stack([store[int(s) : int(s) + window] for s in starts]), dtype=np.float32
        )
    target_store = np.ascontiguousarray(np.asarray(sb.target_store), dtype=np.float32)
    window_ids = starts
    provenance = dict(getattr(sb, "provenance", {}) or {})
    meta = {
        "shape": tuple(e0.shape),
        "trial_count": int(budget),
        "estimator": (
            "B3S frozen-champion native_e0_and_u (push_trial/finalize, not "
            "batched-mean) + MOVE-T4 fit_move_t4/t4_from_trial_sums; frozen "
            "dual_track cache 20260905_101500"
        ),
        "array_sha256": array_sha256(e0),
        "budget": int(budget),
        "surface": surface,
        "session": session,
        "carrier_sha256": array_sha256(carrier),
        "x_store_sha256": array_sha256(x_store),
        "target_store_sha256": array_sha256(target_store),
        "cache_root": str(_M2_CACHE_ROOT),
        "dual_track_provenance": provenance,
    }
    return TaskBank(
        session_id=str(session),
        E0=e0,
        carrier=carrier,
        unit_mask=unit_mask,
        X_store=x_store,
        target_store=target_store,
        window_ids=window_ids,
        calibration_meta=meta,
    )


def build_m1_bank(*args: Any, budget: int, **kwargs: Any) -> TaskBank:
    """Phase 2a — M1 EMG bank (contract below; NOT authorized yet).

    Calibration-object source table (workorder §4; REVIEW C; NOTE P0-2):
      - encoder: B3 Sfix e11 ``student.id_encoder`` -> identity [64, 100]
        drawn via ``compute_identity(side_features=None)`` — rSyn3 does NOT
        enter the identity (``m1_optimized_v2/calibration.py``).
      - extraction: ``m1_optimized_v2/calibration.py::load_frozen_b3``
        (id_encoder ONLY; bringing any B3/Sfix decoder weight into the new
        network is forbidden, NOTE P1-10).
      - carrier: rSyn3 4-dim, NNMF ``RANK=3``, ``RIDGE_LAMBDA=1.0``,
        ``SUPPORT_TRIALS=10``.
      - budget: **M10** (``budget`` must be 10 per bank; CAL-1 precomputes
        per (session, M) when rotation is enabled).
      - scale: ``prediction_divisor = 1`` — M1 never divides by 20.

    Contract:
      - Identity source: ``m1_emg_rsyn3_fold_local_v1``; the activity identity
        may ONLY be drawn from ``student.id_encoder`` (B3 post_pool, [64, 100]).
        Bringing any B3/Sfix decoder weight into the new network is forbidden
        (NOTE P1-10) — such a run must not be compared against Original 0.809.
      - Carrier: rSyn3 4-dim (M10 / ``rSyn3-refit`` lineage); unit column
        order = ``units`` DataFrame row order (NOTE P2-13).
      - Geometry m1 {window 100, prefix 100, units 64, e0_dim 100, carrier 4,
        out 16}; ``prediction_divisor = 1`` — M1 never divides by 20.
      - Faces differ: local 31,252 source-minival vs official LOSO back-half;
        report pooled AND session-mean, never subtract (NOTE P1-11).
      - ``calibration_meta`` must carry shape / trial_count / estimator /
        array_sha256 / budget (NOTE P0-2 + code item S).
    """
    raise NotImplementedError("Phase 2a")


def build_h1_bank(
    surface: str | None = None,
    session: str | None = None,
    *,
    budget: int = _H1.DEPLOY_BUDGET,
    window: int | None = None,
    identity_mode: str = _H1.IDENTITY_DEFAULT,
    limit_windows: int | None = None,
) -> TaskBank:
    """Phase 2b — H1 matrix TaskBank from frozen read-only sources.

    Calibration-object source table (workorder §4, REVIEW C; NOTE P0-2):
      - encoder/materializer: C2 e15 checkpoint ``c2_epoch_015.ckpt`` with
        sha256
        ``ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215``
        (C2 ``carrier_pre_pool(1024->32)`` / ``carrier_post_pool(36->32->32->
        700)``, H-C concatenated in).
      - extraction: ``two_mainlines_long_v1/decoder/h1_calibration.py::
        load_frozen_c2_materializer`` — calibration base ONLY, never an
        initialization of the new network's decoder weights.
      - budget: CAL-1 prefix-cycle **{7, 5, 4, 3}** during training -> deploy
        **M3**; ``budget`` = the M this bank was recomputed at (first
        ``budget`` trials). CAL-2 (fixed frozen budget) is forbidden on the
        H1 mainline (workorder §3).
      - scale: x20 RATIO bridge — ``MSE(raw,20y) == 400 * MSE(raw/20,y)``
        within relative tolerance 1e-9 (``H1_SCALE_RATIO``), plus
        ``pred_std >= 0.01 * target_std``.
      - unit column order: ``units`` DataFrame row order; the 7 output columns
        keep the frozen semantics ``tx, ty, tz, rx, g1, g2, g3``.
      - coordinates: every window joins the Original archive by
        ``(session, end)`` (P0-4).

    Implemented here (H1 matrix skeleton, MATRIX_H1_L_IDENTITY_V1):
      - ``window`` = matrix L (250/350; default full 700). Windowing reuses
        the five-arm mechanism (``scripts/h1_sec6_fivearm_l100.py``): train
        ends = ``query_starts + 699`` filtered by ``eval_mask`` (target
        coordinates FROZEN — L only changes how far back the input reaches),
        minival ends = all ``eval_mask`` bins; windows left-zero-pad when
        ``start < 0`` exactly like the five-arm ``_windows``.
      - ``E0`` [176, 700] fused + ``joined36`` [176, 36] TRUE C2 pre-pool
        joined (carrier_pre_pool 32 + H-C 4 — NOT the five-arm PROXY of the
        first 32 fused dims): BOTH stored (bank ``E0`` = the
        ``identity_mode``'s object; the other array + both sha256s travel in
        ``calibration_meta``). At ``budget == 3`` the materialized E0/carrier
        must reproduce the cache bank arrays BITWISE (provenance loop closed
        against the same source the five-arm consumed).
      - ``target_store`` is NATIVE scale; the training loop applies x20 and
        scores pred/20 via ``scale_bridge.assert_scale_bridge``.

    STILL STUBBED (explicit NotImplementedError, no fabricated paths):
      - ``budget in {7, 5, 4}``: the only located trialized activity is the
        M3 deploy payload (exactly 3 trials/session). The list of searched
        files ships in the error. Budget 3 and deploy banks are live.
      - Calling without ``(surface, session)`` keeps the Phase 2b gate.
    """
    if surface is None or session is None:
        raise NotImplementedError(
            "Phase 2b: build_h1_bank requires (surface, session) — surface in "
            f"{list(_H1.SURFACES)}, session = one of the 13 held-in sessions "
            "in h1_config.H1_ALL_SESSIONS"
        )
    mode = _H1.normalize_identity_mode(identity_mode)
    if window is None:
        length = _H1.FULL_WINDOW
    else:
        plan.require(
            isinstance(window, int) and not isinstance(window, bool),
            "window must be an int (matrix L)",
        )
        length = int(window)
        plan.require(
            length in _H1.MATRIX_L or length == _H1.FULL_WINDOW,
            f"window {length} is not a matrix L {_H1.MATRIX_L} (or the full {_H1.FULL_WINDOW})",
        )
    plan.require(
        isinstance(budget, int) and not isinstance(budget, bool) and budget >= 1,
        "budget must be a positive int (CAL calibration-trial count)",
    )
    plan.require(
        budget in _H1.CAL1_BUDGETS,
        f"H1 mainline is CAL-1 only: budget must be one of {_H1.CAL1_BUDGETS} "
        f"(deploy {_H1.DEPLOY_BUDGET}); CAL-2 fixed frozen budgets are "
        "forbidden (workorder §3)",
    )
    if limit_windows is not None:
        plan.require(
            isinstance(limit_windows, int) and limit_windows >= 1,
            "limit_windows must be a positive int (test/diagnostic cap)",
        )

    cache = _h1_source_cache()
    plan.require(surface in _H1.SURFACES, f"unknown h1 surface {surface!r} (expected one of {_H1.SURFACES})")
    plan.require(surface in cache, f"cache split {surface!r} missing")
    plan.require(session in cache[surface], f"session {session!r} is not on surface {surface!r}")
    row = cache[surface][session]

    # --- calibration objects: C2 materializer over the payload M3 activity ---
    activity, payload_carrier = _h1_payload_arrays(session)
    n_trials = int(activity.shape[0])
    if int(budget) > n_trials:
        raise NotImplementedError(
            f"Phase 2b budget {budget}: trialized calibration activity beyond "
            f"the deploy M3 payload is not materialized in any located file. "
            f"Searched (both located, both hold only {n_trials} trials/session): "
            f"{_H1.C2_M3_PAYLOAD_PATH} (payload['sessions'][*]['identity'] "
            f"[3, 1024, 176]), {_H1.SOURCE_CACHE_PATH} (bank arrays only, no "
            "trials). Recomputing CAL-1 budgets {7, 5, 4} requires re-trializing "
            "the held-in-calib NWB files "
            "(SPINT-main/data/000954/sub-HumanPitt-held-in-calib/*.nwb) at 1024 "
            "bins with the C2 trialization rule, which is not wired into this "
            "package; refusing to fabricate a path."
        )
    materializer = _h1_materializer()
    activity_t = torch.from_numpy(np.ascontiguousarray(activity[: int(budget)])).unsqueeze(0)
    carrier_t = torch.from_numpy(np.ascontiguousarray(payload_carrier)).unsqueeze(0)
    e0_fused, hc = materializer.materialize_bank(activity_t, carrier_t)  # [176,700] / [176,4]
    pooled = materializer.activity_signature(activity_t)  # [1, 176, 32]
    joined36 = torch.cat((pooled[0], hc), dim=1).contiguous()  # [176, 36] TRUE joined

    # provenance loop: at the deploy budget the materialized arrays must
    # reproduce the cache bank the five-arm consumed, bit for bit.
    if int(budget) == n_trials:
        cached_e0 = row["bank"]["E0"]
        cached_e0 = cached_e0.cpu() if hasattr(cached_e0, "cpu") else torch.as_tensor(np.asarray(cached_e0))
        cached_t = row["bank"]["T"]
        cached_t = cached_t.cpu() if hasattr(cached_t, "cpu") else torch.as_tensor(np.asarray(cached_t))
        plan.require(
            torch.equal(e0_fused, cached_e0),
            "materialized E0 does not reproduce the cache bank E0 (provenance drift)",
        )
        plan.require(torch.equal(hc, cached_t), "materialized H-C carrier drift vs cache bank T")

    # --- windows + (session, end) coordinates (five-arm mechanism, L<=700) ---
    neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
    velocity = np.ascontiguousarray(row["velocity"], dtype=np.float32)
    if surface == "train":
        starts = np.asarray(row["query_starts"], dtype=np.int64)
        ends = starts + _H1.FULL_WINDOW - 1  # P0-4: end = start + 699, frozen
        eval_mask = np.asarray(row["eval_mask"], dtype=np.bool_)
        ends = ends[(ends >= 0) & (ends < eval_mask.shape[0])]
        ends = ends[eval_mask[ends]] if eval_mask.size else ends
    else:
        ends = np.flatnonzero(np.asarray(row["eval_mask"], dtype=np.bool_)).astype(np.int64)
    if limit_windows is not None:
        ends = ends[: int(limit_windows)]
    x_store = _h1_windows(neural, ends, length)
    target_store = np.ascontiguousarray(velocity[ends], dtype=np.float32)
    unit_mask = row["bank"]["unit_mask"]
    unit_mask = (
        unit_mask.cpu().numpy() if hasattr(unit_mask, "cpu") else np.asarray(unit_mask)
    )
    unit_mask = np.ascontiguousarray(unit_mask, dtype=np.bool_)

    bank_e0 = (joined36 if mode == "joined" else e0_fused).cpu().numpy()
    bank_e0 = np.ascontiguousarray(bank_e0, dtype=np.float32)
    carrier_np = np.ascontiguousarray(hc.cpu().numpy(), dtype=np.float32)
    meta = {
        "shape": tuple(bank_e0.shape),
        "trial_count": int(budget),
        "estimator": (
            "C2 e15 frozen materializer carrier_pre_pool(1024->32) / "
            "carrier_post_pool(36->32->32->700) over the M3 payload trialized "
            f"activity [M={int(budget)}, 1024, 176] (first {int(budget)} "
            "trials); H-C 4-dim carrier; ckpt sha "
            f"{materializer.checkpoint_sha256[:16]}..."
        ),
        "array_sha256": array_sha256(bank_e0),
        "budget": int(budget),
        "identity_mode": mode,
        "matrix_letter": _H1.IDENTITY_MODE_TO_MATRIX_LETTER[mode],
        "window": int(length),
        "surface": surface,
        "session": session,
        "joined36": np.ascontiguousarray(joined36.cpu().numpy(), dtype=np.float32),
        "joined36_sha256": array_sha256(joined36.cpu().numpy().astype(np.float32)),
        "joined36_note": (
            "TRUE C2 pre-pool joined (carrier_pre_pool 32 + H-C 4), NOT the "
            "five-arm b_joined36 PROXY (first 32 dims of fused E0)"
        ),
        "e0_fused_sha256": array_sha256(e0_fused.cpu().numpy().astype(np.float32)),
        "carrier_sha256": array_sha256(carrier_np),
        "x_store_sha256": array_sha256(x_store),
        "target_store_sha256": array_sha256(target_store),
        "target_scale": "native; training multiplies by 20 and scores pred/20 "
        "(ratio bridge MSE(raw,20y) == 400 * MSE(raw/20,y), rel tol 1e-9)",
        "target_multiplier": float(_H1.TARGET_MULTIPLIER),
        "out_columns": list(_H1.H1_COLUMNS),
        "coord_rule": "window_ids are END bins; join the Original archive by (session, end) (P0-4)",
        "cache_sha256": _H1.SOURCE_CACHE_SHA256,
        "payload_sha256": _H1.C2_M3_PAYLOAD_SHA256,
        "ckpt_sha256": materializer.checkpoint_sha256,
        "lodo_holdout": session in _H1.LODO_HOLDOUT_SESSIONS,
        "lodo_holdout_date": _H1.LODO_HOLDOUT_DATE,
    }
    return TaskBank(
        session_id=str(session),
        E0=bank_e0,
        carrier=carrier_np,
        unit_mask=unit_mask,
        X_store=x_store,
        target_store=target_store,
        window_ids=np.ascontiguousarray(ends, dtype=np.int64),
        calibration_meta=meta,
    )


__all__ = ["build_m2_bank", "build_m1_bank", "build_h1_bank"]
