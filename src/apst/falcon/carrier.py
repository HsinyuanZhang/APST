"""Closed-form association-profile carriers for FALCON M1/M2/H1.

Every estimator here is source-frozen: basis/normalizer quantities are fit
only on declared held-in source recordings and are then applied unchanged to
target-session support.  No carrier uses a query target.

* M1 -- descriptive muscle-recruitment response: rectified-EMG-weighted,
  Poisson-floor standardized rates, pooled SVD to 4 dims, global-RMS scale.
* M2 -- MOVE-window direction OLS (``apst.legacy_dandi.carrier``), the same
  estimator as the DANDI release, with a source-only column normalizer.
* H1 -- signed-state behavioral profile over 100-ms support blocks, source
  pooled SVD basis, source-only mean/scale normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BIN_SECONDS = 0.02
MUSCLES = 16
M1_UNITS = 64
OCCUPANCY_PSEUDOBINS = 10.0
H1_BLOCK_BINS = 5
H1_BLOCK_SECONDS = 0.1
H1_VELOCITY_DIM = 7
H1_PROFILE_DIM = 4
H1_OCCUPANCY_PSEUDO_BLOCKS = 10.0
H1_POISSON_COUNT_FLOOR = 1.0


class CarrierError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierError(message)


# --------------------------------------------------------------------------
# M1 muscle SVD4 carrier
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MuscleFit:
    muscle_rms: np.ndarray          # [16]
    svd_components: np.ndarray      # [4, 16]
    normalizer_mean: np.ndarray     # [4]
    normalizer_scale: np.ndarray    # [4]
    source_sessions: tuple[str, ...]
    metadata: dict[str, Any]


def _m1_trials(path: Path):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        frame = nwb.trials.to_dataframe().reset_index()
        trials = [dict(row) for _, row in frame.iterrows()]
        raw_emg = nwb.acquisition["preprocessed_emg"]
        channels = tuple(raw_emg.time_series.keys())
        _require(channels, f"{path.name}: M1 preprocessed_emg has no channels")
        first = raw_emg.get_timeseries(channels[0])
        timestamps = np.asarray(first.timestamps[:], np.float64)
        emg = np.column_stack([
            np.asarray(raw_emg.get_timeseries(c).data[:], np.float64) for c in channels])
        units = nwb.units.to_dataframe()
        unit_count = len(units)
    _require(emg.shape == (len(timestamps), len(channels)), f"{path.name}: M1 EMG shape drift")
    _require(unit_count == M1_UNITS, f"{path.name}: expected 64 M1 units")
    return trials, timestamps, emg, channels


def _movement_bounds(row) -> tuple[float, float]:
    onset = float(row["move_onset_time"]) if "move_onset_time" in row else float("nan")
    if not np.isfinite(onset):
        onset = float(row["start_time"])
    stop = float(row["stop_time"])
    _require(np.isfinite(onset) and np.isfinite(stop) and stop > onset,
             f"M1 trial movement bounds invalid: {onset}..{stop}")
    return onset, stop


def m1_aligned_support(path: Path, *, emg_trial_stop: int, neural_trial_stop: int):
    """Aligned EMG values and binned rates on the official M1 EMG clock.

    Both outputs cover movement rows of trials ``[0, emg_trial_stop)`` /
    ``[0, neural_trial_stop)`` on the shared EMG-timestamp timeline; rates
    use ``falcon_challenge.dataloaders.bin_units`` exactly like the official
    M1 reader.
    """
    from falcon_challenge.dataloaders import bin_units
    trials, timestamps, emg, channels = _m1_trials(path)

    def rows(stop: int) -> tuple[np.ndarray, np.ndarray]:
        selected, ids = [], []
        for trial_id, row in enumerate(trials[:stop]):
            onset, stop_time = _movement_bounds(row)
            left = int(np.searchsorted(timestamps, onset, side="left"))
            right = int(np.searchsorted(timestamps, stop_time, side="right"))
            _require(0 <= left < right <= len(timestamps),
                     f"{path.name}: trial {trial_id} has no EMG-clock movement rows")
            selected.append(np.arange(left, right, dtype=np.int64))
            ids.append(np.full(right - left, trial_id, dtype=np.int64))
        return np.concatenate(selected), np.concatenate(ids)

    emg_indices, emg_ids = rows(emg_trial_stop)
    rate_indices, rate_ids = rows(neural_trial_stop)
    emg_values = emg[emg_indices]
    _, neural_stop_time = _movement_bounds(trials[neural_trial_stop - 1])
    prefix_stop = int(np.searchsorted(timestamps, neural_stop_time, side="right"))
    _require(prefix_stop > int(rate_indices[-1]), f"{path.name}: M1 clock prefix misses support rows")
    with _m1_units(path) as units:
        counts = np.asarray(
            bin_units(units, bin_size_s=BIN_SECONDS, bin_timestamps=timestamps[:prefix_stop],
                      is_timestamp_bin_start=False), np.float64)
    _require(counts.shape == (prefix_stop, M1_UNITS), f"{path.name}: M1 bin_units contract drift")
    return {
        "emg": emg_values,
        "emg_trial_ids": emg_ids,
        "rates": counts[rate_indices] / BIN_SECONDS,
        "rate_trial_ids": rate_ids,
        "channels": channels,
    }


class _m1_units:
    """Context manager exposing the NWB Units frame without holding the file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def __enter__(self):
        from pynwb import NWBHDF5IO
        self._io = NWBHDF5IO(str(self._path), "r", load_namespaces=True)
        return self._io.read().units.to_dataframe()

    def __exit__(self, *_exc) -> None:
        self._io.close()


def _m1_raw16(fit: MuscleFit, emg: np.ndarray, rates: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(emg, np.float64), 0.0) / fit.muscle_rms[None, :]
    mean_rate = rates.mean(axis=0)
    mean_count = mean_rate * BIN_SECONDS
    poisson_scale = np.sqrt(np.maximum(mean_count, 1.0)) / BIN_SECONDS
    zrate = (rates - mean_rate[None, :]) / poisson_scale[None, :]
    return ((weights.T @ zrate) / (weights.sum(axis=0)[:, None] + OCCUPANCY_PSEUDOBINS)).T


def _orient(components: np.ndarray) -> np.ndarray:
    out = components.copy()
    for k in range(out.shape[0]):
        anchor = int(np.argmax(np.abs(out[k])))
        if out[k, anchor] < 0:
            out[k] *= -1
    return out


def m1_fit_source(paths: Sequence[Path], *, emg_stop: int = 310) -> MuscleFit:
    """Fit muscle RMS, SVD basis, and global-RMS normalizer on source only."""
    names = tuple(Path(p).stem.split("_ses-")[1].split("_")[0] for p in paths)
    prefixes = tuple("ses-" + n for n in names)
    supports = [m1_aligned_support(Path(p), emg_trial_stop=emg_stop, neural_trial_stop=10)
                for p in paths]
    source_emg = np.concatenate([np.maximum(s["emg"][s["emg_trial_ids"] < emg_stop], 0.0)
                                 for s in supports])
    _require(source_emg.shape[1] == MUSCLES, "M1 EMG must have 16 channels")
    rms = np.maximum(np.sqrt(np.mean(source_emg ** 2, axis=0)), 1e-8)
    provisional = MuscleFit(rms, np.zeros((4, MUSCLES)), np.zeros(4), np.ones(4), prefixes, {})
    raw_rows = []
    for s in supports:
        emg_rows = s["emg"][s["emg_trial_ids"] < 10]
        rate_rows = s["rates"][s["rate_trial_ids"] < 10]
        _require(emg_rows.shape[0] == rate_rows.shape[0],
                 "M1 aligned EMG/rate M10 row mismatch")
        raw_rows.append(_m1_raw16(provisional, emg_rows, rate_rows))
    pooled = np.concatenate(raw_rows, axis=0)
    _require(pooled.shape[0] >= 4, "M1 pooled raw profile needs at least four unit rows")
    components = _orient(np.linalg.svd(pooled, full_matrices=False)[2][:4])
    projected = np.concatenate([raw @ components.T for raw in raw_rows], axis=0)
    mean = projected.mean(axis=0)
    variance = np.square(np.maximum(projected.std(axis=0), 1e-6))
    scale = np.full(4, max(float(np.sqrt(np.mean(variance))), 1e-6))
    return MuscleFit(rms, components, mean, scale, prefixes, {
        "candidate": "muscle_response16_svd4",
        "normalization": "global_rms",
        "source_emg_trials": [0, int(emg_stop)],
        "support_trials": 10,
        "response": "sum(rectified_EMG/source_RMS * centered_poisson_scaled_rate)/(weight_sum+10)",
    })


def m1_project(fit: MuscleFit, path: Path, *, lo: int = 0, hi: int = 10) -> tuple[np.ndarray, dict]:
    support = m1_aligned_support(Path(path), emg_trial_stop=hi, neural_trial_stop=hi)
    keep = (support["rate_trial_ids"] >= lo) & (support["rate_trial_ids"] < hi)
    raw16 = _m1_raw16(fit, support["emg"][(support["emg_trial_ids"] >= lo)
                                          & (support["emg_trial_ids"] < hi)], support["rates"][keep])
    raw4 = raw16 @ fit.svd_components.T
    profile = ((raw4 - fit.normalizer_mean) / fit.normalizer_scale).astype(np.float32)
    _require(profile.shape == (M1_UNITS, 4) and np.isfinite(profile).all(),
             f"{path.name}: invalid M1 muscle profile")
    return profile, {"estimator": "muscle_response16_svd4_global_rms",
                     "trial_interval": [lo, hi], "source_sessions": list(fit.source_sessions)}


def m1_save_fit(fit: MuscleFit, path: Path) -> None:
    import json as _json
    np.savez_compressed(path, muscle_rms=fit.muscle_rms, svd_components=fit.svd_components,
                        normalizer_mean=fit.normalizer_mean, normalizer_scale=fit.normalizer_scale,
                        source_sessions=np.array(fit.source_sessions),
                        metadata=np.array(_json.dumps(fit.metadata, sort_keys=True)))


def m1_load_fit(path: Path) -> MuscleFit:
    import json as _json
    with np.load(path, allow_pickle=False) as z:
        sessions = tuple(str(s) for s in z["source_sessions"]) if "source_sessions" in z.files else ()
        metadata: dict[str, Any] = {}
        if "metadata" in z.files:
            metadata = _json.loads(str(z["metadata"].item()))
            sessions = sessions or tuple(metadata.get("source_sessions", ()))
        return MuscleFit(z["muscle_rms"], z["svd_components"], z["normalizer_mean"],
                         z["normalizer_scale"], sessions, metadata)


# --------------------------------------------------------------------------
# M2 MOVE-T4 direction OLS carrier
# --------------------------------------------------------------------------

def _m2_angles(path: Path, horizon: int) -> np.ndarray:
    """Per-trial target angles from ``tgt_loc``; centre/rest trials are NaN."""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        frame = io.read().trials.to_dataframe()
    if "tgt_loc" not in frame:
        raise CarrierError(f"{path.name}: M2 trials have no tgt_loc")
    values = np.asarray(frame["tgt_loc"].tolist(), np.float64)
    _require(values.shape[0] >= horizon, f"{path.name}: fewer than {horizon} M2 trials")
    values = values[:horizon]
    angles = np.full(len(values), np.nan)
    centre = np.asarray([0.5, 0.5])
    for index, point in enumerate(values):
        point = np.asarray(point, np.float64).reshape(-1)
        if point.size != 2 or not np.all(np.isfinite(point)):
            continue
        delta = point - centre
        if np.linalg.norm(delta) <= 1e-8:
            continue
        angles[index] = np.arctan2(delta[1], delta[0])
    return angles.astype(np.float32)


def m2_raw_t4(move_counts: np.ndarray, path: Path, lengths: Sequence[int]) -> np.ndarray:
    from apst.legacy_dandi.carrier import estimate_move_t4
    horizon = int(move_counts.shape[0])
    angles = _m2_angles(Path(path), horizon)
    usable = np.isfinite(angles)
    _require(usable.sum() >= 3, f"{path.name}: M2 MOVE-T4 needs three labelled directions")
    raw = estimate_move_t4(np.asarray(move_counts, np.float64)[usable], angles[usable])
    return np.ascontiguousarray(raw, np.float32)


def m2_fit_normalizer(paths: Sequence[Path], spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Source-only per-column T4 normalizer over the first support trials."""
    from apst.legacy_dandi.carrier import fit_carrier_normalizer
    from .data import _m2_calib_timeline, _load_nwb
    lo_bin, hi_bin = spec["move_t4_window_bins"]
    rows = []
    for path in paths:
        neural, covariates, trial_change, eval_mask = _load_nwb(Path(path), "m2")
        calib_neural, calib_change = _m2_calib_timeline(neural, covariates, trial_change, eval_mask,
                                                        horizon=int(spec["support_trials"]))
        starts = np.flatnonzero(calib_change)
        horizon = int(spec["support_trials"])
        counts = np.stack([calib_neural[int(s) + lo_bin:int(s) + hi_bin].sum(axis=0, dtype=np.float64)
                           for s in starts[:horizon]])
        rows.append(m2_raw_t4(counts, path, [hi_bin - lo_bin] * horizon))
    return fit_carrier_normalizer(rows)


def m2_move_t4(move_counts: np.ndarray, path: Path, lengths: Sequence[int],
               normalizer: dict[str, np.ndarray] | None) -> np.ndarray:
    raw = m2_raw_t4(move_counts, path, lengths)
    if normalizer is None:
        raise CarrierError("M2 carrier requires the frozen source normalizer")
    from apst.legacy_dandi.carrier import apply_carrier_normalizer
    return apply_carrier_normalizer(raw, normalizer)


def m2_save_normalizer(state: dict[str, np.ndarray], path: Path) -> None:
    np.savez_compressed(path, **state)


def m2_load_normalizer(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {"mean": z["mean"], "std": z["std"]}


# --------------------------------------------------------------------------
# H1 signed-state behavioral carrier
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SignedStatePlan:
    behavior_rms: np.ndarray       # [7]
    raw_basis: np.ndarray          # [14, 4]
    profile_mean: np.ndarray       # [4]
    profile_scale: np.ndarray      # [4]
    source_sessions: tuple[str, ...]


def _h1_trial_num(path: Path) -> np.ndarray:
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "TrialNum" not in nwb.acquisition:
            raise CarrierError(f"{path.name}: TrialNum acquisition missing")
        return np.asarray(nwb.acquisition["TrialNum"].data[:], np.float64)


def h1_record(path: Path, neural: np.ndarray, velocity: np.ndarray, eval_mask: np.ndarray):
    """Trial metadata, eval-valid trial neural, and 100-ms support blocks.

    The 100-ms block aggregation promotes the loader outputs to float64
    before summing/averaging.
    """
    path = Path(path)
    neural64 = np.asarray(neural, np.float64)
    velocity64 = np.asarray(velocity, np.float64)
    trial_num = _h1_trial_num(path)
    length = neural.shape[0]
    _require(trial_num.shape == (length,) and eval_mask.shape == (length,),
             f"{path.name}: H1 TrialNum/eval_mask length mismatch")
    ordered = trial_num[eval_mask & np.isfinite(trial_num)]
    values: list[float] = []
    for value in ordered.tolist():
        if not values or value != values[-1]:
            values.append(float(value))
    _require(values, f"{path.name}: H1 recording has no eval-valid TrialNum trials")
    # Public held-out recordings expose exactly the three support trials (M3);
    # held-in recordings continue with query trials after them.

    def eval_trial_neural(trial: float) -> np.ndarray:
        legal = eval_mask & np.isfinite(trial_num) & (trial_num == float(trial)) \
            & np.isfinite(neural).all(axis=1)
        rows = neural[legal]
        _require(rows.shape[0] >= 4,
                 f"{path.name}: H1 cubic identity needs four eval-valid bins for trial {trial}")
        return np.asarray(rows, np.float32)

    def blocks(trial: float):
        legal = (trial_num == float(trial)) & eval_mask \
            & np.isfinite(neural64).all(axis=1) & np.isfinite(velocity64).all(axis=1)
        rate_rows, velocity_rows = [], []
        run_start = None
        for index in range(length + 1):
            active = bool(legal[index]) if index < length else False
            if active and run_start is None:
                run_start = index
            elif not active and run_start is not None:
                usable = ((index - run_start) // H1_BLOCK_BINS) * H1_BLOCK_BINS
                for offset in range(0, usable, H1_BLOCK_BINS):
                    block = np.arange(run_start + offset, run_start + offset + H1_BLOCK_BINS)
                    rate_rows.append(neural64[block].sum(axis=0) / H1_BLOCK_SECONDS)
                    velocity_rows.append(velocity64[block].mean(axis=0))
                run_start = None
        _require(rate_rows, f"{path.name}: H1 trial {trial} has no 100-ms support block")
        return np.stack(rate_rows).astype(np.float64), np.stack(velocity_rows).astype(np.float64)

    return {"session": path.stem.split("_ses-")[1].split("_")[0],
            "trial_values": tuple(values), "eval_trial_neural": eval_trial_neural, "blocks": blocks}


def _h1_standardize(rates: np.ndarray):
    rate_mean = np.mean(rates, axis=0)
    mean_count = np.maximum(rate_mean * H1_BLOCK_SECONDS, H1_POISSON_COUNT_FLOOR)
    noise_rate = np.sqrt(mean_count) / H1_BLOCK_SECONDS
    return (rates - rate_mean[None, :]) / noise_rate[None, :]


def _h1_raw_signed_state(rates: np.ndarray, velocity: np.ndarray, behavior_rms: np.ndarray):
    z = _h1_standardize(rates)
    x = velocity / behavior_rms[None, :]
    softplus = np.logaddexp(0.0, x)
    weights = np.concatenate((softplus, np.logaddexp(0.0, -x)), axis=1)
    occupancy = np.sum(weights, axis=0)
    raw = (weights.T @ z / (occupancy[:, None] + H1_OCCUPANCY_PSEUDO_BLOCKS)).T
    _require(raw.shape == (rates.shape[1], 14) and np.isfinite(raw).all(),
             "H1 signed-state raw profile invalid")
    return raw


def h1_fit_plan(paths: Sequence[Path]) -> SignedStatePlan:
    from .data import _load_nwb
    records = {}
    for path in paths:
        neural, velocity, _change, mask = _load_nwb(Path(path), "h1")
        records[Path(path).stem] = h1_record(Path(path), neural, velocity, mask)
    names = tuple(sorted(records))
    rates = np.concatenate([records[n]["blocks"](records[n]["trial_values"][0])[0] for n in names])
    velocities = np.concatenate([records[n]["blocks"](records[n]["trial_values"][0])[1]
                                 for n in names])
    behavior_rms = np.maximum(np.sqrt(np.mean(np.square(velocities), axis=0)), 1e-8)
    raw_rows = []
    for name in names:
        record = records[name]
        block_rates, block_velocity = record["blocks"](record["trial_values"][0])
        for trial in record["trial_values"][1:3]:
            r, v = record["blocks"](trial)
            block_rates = np.concatenate([block_rates, r])
            block_velocity = np.concatenate([block_velocity, v])
        raw_rows.append(_h1_raw_signed_state(block_rates, block_velocity, behavior_rms))
    pooled = np.concatenate(raw_rows, axis=0)
    right = np.linalg.svd(pooled, full_matrices=False)[2]
    basis = np.asarray(right[:H1_PROFILE_DIM].T, np.float64)
    projected = pooled @ basis
    mean = np.mean(projected, axis=0)
    scale = np.maximum(np.sqrt(np.mean(np.square(projected - mean[None, :]), axis=0)), 1e-8)
    sessions = tuple("ses-" + records[n]["session"] for n in names)
    return SignedStatePlan(behavior_rms, basis, mean, scale, sessions)


def h1_deploy(plan: SignedStatePlan, record: dict, values: Sequence[float]) -> np.ndarray:
    rates = np.concatenate([record["blocks"](float(v))[0] for v in values])
    velocities = np.concatenate([record["blocks"](float(v))[1] for v in values])
    raw = _h1_raw_signed_state(rates, velocities, plan.behavior_rms)
    projected = raw @ plan.raw_basis
    profile = ((projected - plan.profile_mean[None, :]) / plan.profile_scale[None, :]).astype(np.float32)
    _require(profile.shape == (raw.shape[0], H1_PROFILE_DIM) and np.isfinite(profile).all(),
             "H1 deployed profile must be finite [176,4]")
    return profile


def h1_save_plan(plan: SignedStatePlan, path: Path) -> None:
    np.savez_compressed(path, behavior_rms=plan.behavior_rms, raw_basis=plan.raw_basis,
                        profile_mean=plan.profile_mean, profile_scale=plan.profile_scale,
                        source_sessions=np.array(plan.source_sessions))


def h1_load_plan(path: Path) -> SignedStatePlan:
    with np.load(path, allow_pickle=False) as z:
        return SignedStatePlan(z["behavior_rms"], z["raw_basis"], z["profile_mean"],
                               z["profile_scale"], tuple(str(s) for s in z["source_sessions"]))
