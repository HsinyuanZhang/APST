"""Frozen per-session calibration bank for btransform_unified_v1.

A :class:`TaskBank` is the session-frozen calibration memory consumed by
:class:`~btransform_unified_v1.model.BTransformerUnifiedDecoder`:
``E0 [N, d_e]`` activity identity + ``carrier [N, 4]`` functional carrier +
``unit_mask [N]``. NOTE P0-2: same names do not count as alignment — every
bank must carry ``calibration_meta`` with at least ``shape`` / ``trial_count``
/ ``estimator`` / ``array_sha256`` / ``budget`` describing the *object* that
was estimated (``budget`` = the CAL calibration-trial budget that produced the
arrays, code item S: CAL-1 precomputes one bank per (session, M) so the
budget a bank was built at must be inseparable from the bank itself).

Observation contract (NOTE P0-5): stored arrays are C-contiguous float32 (bool
for the mask), finite, no NaN. This series is the B-transformer unified series,
NOT SPINT.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import plan

CALIBRATION_META_REQUIRED_KEYS: tuple[str, ...] = (
    "shape",
    "trial_count",
    "estimator",
    "array_sha256",
    "budget",
)

# Per-task default synthetic budgets (code item S): M33 (S1 parity), M10, and
# H1's DEPLOY budget M3 (training uses the CAL-1 prefix-cycle {7,5,4,3}).
DEFAULT_SYNTHETIC_BUDGET: dict[str, int] = {"m2": 33, "m1": 10, "h1": 3}


def array_sha256(array: np.ndarray) -> str:
    """SHA-256 over the C-contiguous bytes of ``array`` (dtype-preserving)."""
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(data.dtype.str).encode("utf-8"))
    digest.update(str(data.shape).encode("utf-8"))
    digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskBank:
    """Immutable per-session frozen calibration memory plus window stores.

    Fields:
      session_id:    session identifier (P0-4 coordinate family)
      E0:            [N, d_e] float32 activity identity (static, frozen)
      carrier:       [N, carrier_dim=4] float32 functional carrier
      unit_mask:     [N] bool, True = eligible unit (units DataFrame row order,
                     NOTE P2-13)
      X_store:       [S, L_in, N] float32 compact observation store
      target_store:  [S, out_dim] float32 NATIVE-scale targets (scale bridges
                     such as M2 x5 / H1 x20 are applied by the training loop,
                     validated by scale_bridge.assert_scale_bridge)
      window_ids:    [S] int64 window identifiers (e.g. start/end coordinates)
      calibration_meta: dict that MUST contain shape / trial_count / estimator /
                     array_sha256 / budget (NOTE P0-2 + code item S; budget is
                     the positive int CAL calibration-trial count this bank
                     was estimated from — CAL-1 stores one bank per (session,
                     M), so the budget travels with the bank)
    """

    session_id: str
    E0: np.ndarray
    carrier: np.ndarray
    unit_mask: np.ndarray
    X_store: np.ndarray
    target_store: np.ndarray
    window_ids: np.ndarray
    calibration_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plan.require(bool(self.session_id), "TaskBank.session_id required")
        e0 = np.ascontiguousarray(self.E0, dtype=np.float32)
        carrier = np.ascontiguousarray(self.carrier, dtype=np.float32)
        mask = np.ascontiguousarray(self.unit_mask, dtype=np.bool_)
        x_store = np.ascontiguousarray(self.X_store, dtype=np.float32)
        t_store = np.ascontiguousarray(self.target_store, dtype=np.float32)
        wids = np.ascontiguousarray(self.window_ids, dtype=np.int64)

        n_units = e0.shape[0] if e0.ndim == 2 else -1
        plan.require(e0.ndim == 2 and n_units >= 1, f"E0 must be [N, d_e], got {self.E0.shape}")
        plan.require(
            carrier.ndim == 2 and carrier.shape == (n_units, 4),
            f"carrier must be [N, 4] with N={n_units}, got {self.carrier.shape}",
        )
        plan.require(
            mask.ndim == 1 and mask.shape[0] == n_units,
            f"unit_mask must be [N] with N={n_units}, got {self.unit_mask.shape}",
        )
        plan.require(bool(mask.any()), "unit_mask must keep at least one unit")
        plan.require(
            np.isfinite(e0).all() and np.isfinite(carrier).all(),
            "E0/carrier must be finite (NOTE P0-5: reject NaN)",
        )
        plan.require(
            x_store.ndim == 3 and x_store.shape[2] == n_units,
            f"X_store must be [S, L_in, N={n_units}], got {self.X_store.shape}",
        )
        plan.require(
            t_store.ndim == 2 and t_store.shape[0] == x_store.shape[0] and t_store.shape[1] >= 1,
            f"target_store must be [S, out_dim] with S={x_store.shape[0]}, got {self.target_store.shape}",
        )
        plan.require(
            wids.ndim == 1 and wids.shape[0] == x_store.shape[0],
            f"window_ids must be [S] with S={x_store.shape[0]}, got {self.window_ids.shape}",
        )
        plan.require(
            np.isfinite(x_store).all() and np.isfinite(t_store).all(),
            "X_store/target_store must be finite (NOTE P0-5: reject NaN)",
        )
        meta = dict(self.calibration_meta)
        missing = [key for key in CALIBRATION_META_REQUIRED_KEYS if key not in meta]
        plan.require(
            not missing,
            f"calibration_meta missing required keys {missing} (NOTE P0-2 + "
            "code item S: shape / trial_count / estimator / array_sha256 / budget)",
        )
        budget = meta.get("budget")
        plan.require(
            isinstance(budget, int) and not isinstance(budget, bool) and budget >= 1,
            "calibration_meta['budget'] must be a positive int (CAL-1 "
            "calibration-trial budget, code item S)",
        )
        declared = meta.get("shape")
        if isinstance(declared, (tuple, list)):
            plan.require(
                tuple(int(v) for v in declared) == tuple(e0.shape),
                f"calibration_meta['shape'] {declared} != E0.shape {e0.shape}",
            )

        # frozen dataclass: normalize dtype/layout via object.__setattr__
        object.__setattr__(self, "E0", e0)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "unit_mask", mask)
        object.__setattr__(self, "X_store", x_store)
        object.__setattr__(self, "target_store", t_store)
        object.__setattr__(self, "window_ids", wids)
        object.__setattr__(self, "calibration_meta", meta)


def make_synthetic_bank(
    task: str,
    n_windows: int = 16,
    seed: int = 0,
    e0_dim: int | None = None,
    prefix: int | None = None,
    budget: int | None = None,
) -> TaskBank:
    """Deterministic synthetic bank for CPU smoke tests (no real data touched).

    The target is a fixed random linear readout of the mean of the last 5
    observation bins — exactly the receptive field of the k=5 causal conv — so
    the synthetic-convergence smoke can drop loss without any real-data
    dependency. Targets are stored at NATIVE scale.

    ``e0_dim`` overrides the geometry identity dimension and ``prefix``
    resolves the H1 ``H1_PREFIX_PENDING`` sentinel; both exist only so
    H1-shaped smoke can run with temporary dev values while the NOTE §6
    control / §7 latency gate are open — they are dev-surface constructs and
    may not back any alignment claim. Integer geometry values cannot be
    overridden (frozen geometry).

    ``budget`` defaults per task to 33 (m2, M33 S1 parity) / 10 (m1, M10) /
    3 (h1 deploy M3) and lands in ``calibration_meta['budget']`` (code item
    S); an explicit value mimics a CAL-1 multi-budget bank.
    """
    geometry = plan.task_geometry(task)
    if e0_dim is None:
        e0_dim = plan.resolved_e0_dim(geometry)
    plan.require(isinstance(e0_dim, int) and e0_dim >= 1, "e0_dim override must be a positive int")
    if budget is None:
        budget = DEFAULT_SYNTHETIC_BUDGET[task]
    plan.require(isinstance(budget, int) and budget >= 1, "budget must be a positive int")
    units = int(geometry["units"])
    prefix_bins = plan.resolved_prefix(geometry, prefix)
    l_in = int(geometry["window"]) + prefix_bins
    out_dim = int(geometry["out_dim"])
    carrier_dim = int(geometry["carrier_dim"])

    rng = np.random.default_rng(seed)
    e0 = rng.standard_normal((units, e0_dim)).astype(np.float32)
    carrier = rng.standard_normal((units, carrier_dim)).astype(np.float32)
    keep = rng.random(units) >= 0.1
    keep[0] = True  # at least one eligible unit
    x_store = rng.standard_normal((n_windows, l_in, units)).astype(np.float32)
    readout = (rng.standard_normal((out_dim, units)) / np.sqrt(units)).astype(np.float32)
    tail_mean = x_store[:, -5:, :].mean(axis=1)  # [S, N]
    target = (tail_mean @ readout.T).astype(np.float32)
    window_ids = np.arange(n_windows, dtype=np.int64)
    meta = {
        "shape": tuple(e0.shape),
        "trial_count": int(0),
        "estimator": "synthetic_standard_normal",
        "array_sha256": array_sha256(e0),
        "budget": int(budget),
        "synthetic": True,
        "task": task,
        "seed": int(seed),
    }
    return TaskBank(
        session_id=f"synthetic-{task}-{seed}",
        E0=e0,
        carrier=carrier,
        unit_mask=keep,
        X_store=x_store,
        target_store=target,
        window_ids=window_ids,
        calibration_meta=meta,
    )


__all__ = [
    "CALIBRATION_META_REQUIRED_KEYS",
    "DEFAULT_SYNTHETIC_BUDGET",
    "TaskBank",
    "array_sha256",
    "make_synthetic_bank",
]
