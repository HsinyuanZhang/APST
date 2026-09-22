"""Small CPU checks for the released model and calibration contracts."""
import importlib

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('subject', ['subc', 'subm'])
def test_activity_only_rejects_profile(subject):
    model_module = importlib.import_module(f'apst.dandi_{subject}.model')
    model = model_module.build_model('act_only', 'F0', 42, 'train')
    with pytest.raises(ValueError, match='rejects carrier'):
        model(torch.zeros(1, 50, 100), torch.zeros(33, 100, 100), torch.ones(100, 4))


@pytest.mark.parametrize('subject', ['subc', 'subm'])
def test_stage_two_preserves_encoder_and_trains_film(subject):
    module = importlib.import_module(f'apst.dandi_{subject}.model')
    pretrained = module.build_model('identity_only', 'F0', 42, 'pretrain')
    model = module.build_model('dual_site', 'F0', 42, 'train', pretrained.export_encoder_state())
    assert all(not p.requires_grad for n, p in model.encoder.named_parameters() if not n.startswith('film.'))
    assert all(p.requires_grad for p in model.encoder.film.parameters())
    torch.manual_seed(91)
    activity, profile = torch.rand(33, 100, 100), torch.rand(100, 4)
    with torch.no_grad():
        torch.testing.assert_close(model.encode(activity, profile), pretrained.encode(activity, profile), rtol=0, atol=0)
    model(torch.rand(1, 50, 100), activity, profile).square().mean().backward()
    assert all(p.grad is None for n, p in model.encoder.named_parameters() if not n.startswith('film.'))
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.encoder.film.parameters())


@pytest.mark.parametrize('subject', ['subc', 'subm'])
def test_output_ema_uses_historical_state_coefficient(subject):
    training = importlib.import_module(f'apst.dandi_{subject}.training')
    values = np.asarray([[0.0], [3.0], [0.0]], dtype=np.float32)
    expected = np.asarray([[0.0], [2.0], [2.0 / 3.0]])
    np.testing.assert_allclose(training._ema_full_clock(values), expected, rtol=1e-6)
    np.testing.assert_array_equal(training._ema_full_clock(values[:1]), values[:1])


@pytest.mark.parametrize('subject', ['subc', 'subm'])
def test_optimizer_excludes_frozen_trunk_and_recency_decay(subject):
    module = importlib.import_module(f'apst.dandi_{subject}.model')
    training = importlib.import_module(f'apst.dandi_{subject}.training')
    model = module.build_model('dual_site', 'F0', 42, 'train')
    optimizer, _ = training.build_optimizer(model, {'weight_decay': .01, 'lr_peak': 3e-4, 'betas': [.9, .999], 'eps': 1e-8})
    groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
    for name, parameter in model.named_parameters():
        assert (id(parameter) in groups) == parameter.requires_grad
        if name.endswith('slope_log'):
            assert groups[id(parameter)]['weight_decay'] == 0


def test_subc_seal_roundtrip_and_artifact_drift(tmp_path):
    import hashlib
    import json
    from apst.dandi_subc import finalize, protocol
    from apst.dandi_subc.final_access import FinalAccess

    run = tmp_path / 'run'
    run.mkdir()
    checkpoint = run / 'selected.pt'
    checkpoint.write_bytes(b'capability-test-only-not-a-model')
    stats = {'sha256': 'test-source-stat-binding'}
    (run / 'source_stats.json').write_text(json.dumps(stats))
    selection = {
        'status': 'FORMAL', 'rule': 'earliest_max_equal_session_dev_r2',
        'dev_sessions': list(protocol.DEV_SESSIONS),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'final_sessions_opened': 0,
    }
    (run / 'selection.json').write_text(json.dumps(selection))
    (run / 'train_receipt.json').write_text(json.dumps({
        'status': 'FORMAL', 'completed': True, 'final_sessions_opened': 0,
        'method': 'dual_site', 'representation': 'sua', 'seed': 42,
        'source_sessions': list(protocol.TRAIN_SESSIONS),
        'source_stats_sha256': stats['sha256'],
    }))
    seal = tmp_path / 'selection_seal.json'
    finalize.seal(seal, [f'arbitrary_name={run}'])
    cap = FinalAccess.from_manifest(seal)
    cap.authorize(protocol.FINAL_SESSIONS[0])
    with pytest.raises(PermissionError):
        cap.authorize(protocol.DEV_SESSIONS[0])
    checkpoint.write_bytes(b'changed-after-selection')
    with pytest.raises(PermissionError):
        cap.validate()
