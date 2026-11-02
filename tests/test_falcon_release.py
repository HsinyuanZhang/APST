"""CPU contract tests for the FALCON training release (no data download)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

from apst.falcon import carrier as fc
from apst.falcon import data as fdata
from apst.falcon import protocol
from apst.falcon.common import (centered_variance_weighted_r2, epoch_batches,
                                grouped_seven_r2)
from apst.falcon.profile import AssociationProfile, Stage2Joint, calibrate
from apst.falcon.training import _full_clock_score


@pytest.fixture(scope="module")
def spec():
    return protocol.task_protocol("m1")


# ---------------------------------------------------------------- carriers

def test_m1_poisson_r_unit_scaling():
    """M1 rate standardization is sqrt(max(mean_count, 1)) / 0.02 exactly."""
    rates = np.array([[0.0, 0.0], [10.0, 0.0]], np.float64)  # 2 bins, 2 units
    mean_rate = rates.mean(axis=0)
    mean_count = mean_rate * 0.02
    scale = np.sqrt(np.maximum(mean_count, 1.0)) / 0.02
    # both units: mean count <= 1 -> Poisson floor 1 count -> scale 50 Hz
    assert np.allclose(scale, [50.0, 50.0])
    zrate = (rates - mean_rate) / scale
    assert np.allclose(zrate[:, 0], [-0.1, 0.1]) and np.allclose(zrate[:, 1], [0.0, 0.0])


def test_m1_muscle_raw16_occupancy_floor():
    """Response denominator is weight_sum + 10 pseudo-bins (direct formula)."""
    fit = fc.MuscleFit(muscle_rms=np.full(16, 1.0), svd_components=np.eye(4, 16),
                       normalizer_mean=np.zeros(4), normalizer_scale=np.ones(4),
                       source_sessions=("s",), metadata={})
    t = 4
    emg = np.zeros((t, 16)); emg[:, 0] = np.arange(1, t + 1) * 2.0  # varying weights
    rates = np.tile(np.arange(t, dtype=np.float64)[:, None], (1, 2))  # 2 units
    raw = fc._m1_raw16(fit, emg, rates)
    mean_rate = rates.mean(axis=0)
    scale = np.sqrt(np.maximum(mean_rate * 0.02, 1.0)) / 0.02
    z = (rates - mean_rate) / scale
    w = np.maximum(emg, 0.0) / fit.muscle_rms[None, :]
    expect = ((w.T @ z) / (w.sum(axis=0)[:, None] + 10.0)).T
    assert raw.shape == (2, 16) and np.allclose(raw, expect)
    # the +10 floor shrinks: dropping it would multiply column 0 by 8/18
    unfloored = (w.T @ z).T[:, 0] / w[:, 0].sum()
    assert not np.allclose(raw[:, 0], unfloored)


def test_m2_move_t4_rank_fail_closed():
    """The direction OLS refuses rank-deficient designs."""
    from apst.legacy_dandi.carrier import estimate_move_t4
    counts = np.zeros((3, 4), np.float64)
    same_angle = np.zeros(3)  # rank 1 design
    with pytest.raises(Exception):
        estimate_move_t4(counts, same_angle)


def test_m2_angles_from_tgt_loc():
    """The tgt_loc conversion needs a trials table; the rank fail-closed
    behaviour above guards the estimator contract without one."""
    pytest.skip("requires an M2 NWB fixture")


# ---------------------------------------------------------------- profile

def test_profile_shapes_and_masked_mean():
    for task in protocol.TASKS:
        spec = protocol.task_protocol(task)
        torch.manual_seed(0)
        encoder = AssociationProfile(task, film=False, seed=42)
        activity = torch.zeros(spec["support_trials"], spec["support_bins"], spec["units"])
        carrier = torch.zeros(spec["units"], 4)
        e0 = encoder(activity, carrier)
        assert tuple(e0.shape) == (spec["units"], spec["e0_dim"])
        # equal-weight mean over a subset of trials equals masked mean
        encoder.pre_pool[0].weight.data.normal_()
        a2 = torch.zeros_like(activity); a2[:] = torch.randn_like(a2)
        full = encoder.pooled(a2)
        masked = encoder.pooled(a2, torch.tensor([True] + [False] * (a2.shape[0] - 1)))
        assert torch.allclose(full, encoder.pooled(a2, None))
        assert torch.equal(masked, encoder.pre_pool(a2.permute(0, 2, 1))[0])


def test_film_zero_init_is_identity():
    encoder = AssociationProfile("m2", film=True, seed=7)
    pooled = torch.randn(96, 64)
    carrier = torch.randn(96, 4)
    assert torch.equal(encoder.film(pooled, carrier), pooled)


def test_stage2_freezes_trunk_and_adds_film_and_slope():
    joint = Stage2Joint("m2", seed=42, film=True, ladder_scale=1.0)
    trainable = {n for n, p in joint.named_parameters() if p.requires_grad}
    assert any(n.startswith("a.film") for n in trainable)
    assert not any(n.startswith("a.pre_pool") or n.startswith("a.post_pool") for n in trainable)
    slopes = [n for n in trainable if n.endswith("slope_log")]
    assert len(slopes) == 1
    core = joint.decoder.temporal
    assert tuple(core.slope_log.shape)[-1] == 8
    # ladder_scale multiplies the default half-life ladder
    joint16 = Stage2Joint("m2", seed=42, film=False, ladder_scale=16.0)
    from apst.models.recency.config import half_life_to_slope
    scaled = (0.08 * 16, 0.16 * 16, 0.32 * 16, 0.64 * 16, 1.28 * 16, 2.56 * 16, None, None)
    expect = torch.tensor([half_life_to_slope(h, 0.02) if h else 0.0 for h in scaled])
    assert torch.allclose(joint16.decoder.temporal.recency_slopes.view(-1)[-8:], expect, atol=1e-7)


def test_calibrate_is_closed_form_and_grad_free():
    spec = protocol.task_protocol("m1")
    encoder = AssociationProfile("m1", film=False, seed=1)
    activity = np.random.default_rng(0).random((10, 1024, 64)).astype(np.float32)
    carrier = np.random.default_rng(1).standard_normal((64, 4)).astype(np.float32)
    e0 = calibrate(encoder, activity, carrier)
    assert e0.shape == (64, spec["e0_dim"]) and np.isfinite(e0).all()
    assert not any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in encoder.parameters())


# ---------------------------------------------------------------- scoring

def test_full_clock_output_ema_convention():
    """state_0 = pred_0; then state = state/3 + 2*p/3 (previous coefficient 1/3)."""

    class FakeJoint:
        def __call__(self, x, pooled, carrier, valid, keep):
            column = x[:, -1, 0]
            return torch.stack((column, column), dim=-1)

    class FakeSession:
        task = "m2"

        def __init__(self):
            self.x = (np.arange(56, dtype=np.float32)[:, None, None]
                      * np.ones((1, 1, 1), np.float32))
            self.starts = np.array([3, 4, 5], np.int64)
            self.y = np.zeros((3, 2), np.float32)
            self.pad = 49
            self.carrier = np.zeros((96, 4), np.float32)

    # context = 50 (task m2): window at endpoint e ends at x[e + 49]; the six
    # chronological endpoints emit predictions 49..54 and queries are 3/4/5.
    result = _full_clock_score(FakeJoint(), FakeSession(), torch.zeros(96),
                               torch.device("cpu"), max_windows=None)
    states = [49.0]
    for value in (50.0, 51.0, 52.0, 53.0, 54.0):
        states.append(states[-1] / 3.0 + 2.0 * value / 3.0)
    expect = np.array([[states[i] / 5.0, states[i] / 5.0] for i in (3, 4, 5)])  # M2 scale bridge
    assert np.allclose(result["prediction"], expect)


def test_centered_r2_and_grouped_seven():
    target = np.zeros((10, 2)); target[5:, 0] = 4.0
    assert centered_variance_weighted_r2(target, target) == pytest.approx(1.0)
    mean_prediction = np.tile(target.mean(axis=0), (10, 1))
    assert centered_variance_weighted_r2(mean_prediction, target) == pytest.approx(0.0)
    groups = {"S6": ["a", "b"]}
    per_key = {k: {"prediction": target, "target": target} for k in ("a", "b")}
    assert grouped_seven_r2(per_key, groups) == pytest.approx(1.0)


def test_epoch_batches_full_pass_and_balance():
    from types import SimpleNamespace
    items = {"a": SimpleNamespace(starts=np.arange(7)), "b": SimpleNamespace(starts=np.arange(3))}
    batches = epoch_batches(items, epoch=1, batch=4, seed=42)
    counts = {name: sum(len(ids) for s, ids in batches if s == name) for name in items}
    # one full pass per session; the final partial batch is kept as-is
    assert counts == {"a": 7, "b": 3} and len(batches) == 3


# ---------------------------------------------------------------- data helpers

def test_budget_slice_prefix_only():
    from apst.falcon.profile import budget_slice
    activity = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    assert np.array_equal(budget_slice(activity, [0]), activity[:1])
    with pytest.raises(ValueError):
        budget_slice(activity, [1])


def test_m2_calib_timeline_cuts_at_trial_boundary():
    neural = np.arange(200, dtype=np.float32)[:, None] * np.ones((1, 3))
    change = np.zeros(200, bool); change[[0, 40, 80, 120]] = True
    calib, calib_change = fdata._m2_calib_timeline(neural, None, change, np.ones(200, bool), horizon=3)
    assert len(calib) == 120 and calib_change.sum() == 3
