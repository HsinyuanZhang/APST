"""M1 scaled-ladder x16 + offline FiLM runtime with a contiguous KV ring.

Same payload contract as the M1 frozen-EMA adapters. Streaming no longer
recomputes the R=100 window per bin: E0/carrier/keep are stacked at reset,
and the learned-slope temporal path uses CpuLearnableRecencyRuntime.
No output scaling. Batch clamp 4. Output EMA α=1/3.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np, torch
from falcon_challenge.interface import BCIDecoder

HERE = Path(__file__).resolve().parent
for p in (HERE / 'pkg', HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from btransform_unified_v1.bank import TaskBank
from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.cpu_temporal import CpuLearnableRecencyRuntime
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from a1_m2.temporal_variants import A1Temporal, VariantSpec, SPECS
from pretrained_nope_learned.temporal_scaled_ladder_v1 import install_f_scaled_ladder

SCHEMA = 'apst_m1_ladder_x16_film_runtime_payload_v1'
N, O, R, MAXB = 64, 16, 100, 4
ALPHA = np.float32(1 / 3)
LADDER_SCALE = 16.0


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_decoder():
    d = RiftDecoder('m1', context_bins=R, bias_mode='recency', seed=42, proj_dim=16)
    d.temporal = A1Temporal(d.temporal, SPECS['A_flat'])
    d.temporal.spec = VariantSpec('E_nope_flat', False, 'none')
    install_f_scaled_ladder(d, 'm1', scale=LADDER_SCALE)
    core = d.temporal.core
    if d.temporal.spec.use_sinusoidal_pe or d.temporal.spec.arm != 'F_nope_learned' or tuple(core.slope_log.shape) != (4, 8):
        raise RuntimeError('F temporal construction drift')
    if int((core.recency_slopes > 0).sum()) != 6 or int((core.recency_slopes == 0).sum()) != 2:
        raise RuntimeError('F scaled-ladder six-active two-flat head drift')
    return d


class CachedFRingRuntime:
    """Fixed-row ring cache. Public method name matches A1M2StreamDecoder.stream_step."""

    def __init__(self, decoder, banks: Sequence[TaskBank]):
        if decoder.training:
            raise ValueError('CachedFRingRuntime requires decoder.eval()')
        core = decoder.temporal.core
        if decoder.temporal.spec.use_sinusoidal_pe or not isinstance(core, LearnableRecencyTemporal):
            raise TypeError('cached F runtime requires noPE LearnableRecencyTemporal core')
        if not banks:
            raise ValueError('cached F runtime requires at least one bank')
        self.decoder = decoder
        self.banks = list(banks)
        self._bank_ids = tuple(id(bank) for bank in self.banks)
        device = next(decoder.parameters()).device
        batch = len(self.banks)
        self.e0 = torch.stack([torch.from_numpy(np.ascontiguousarray(bank.E0)) for bank in self.banks]).to(device, torch.float32)
        self.carrier = torch.stack([torch.from_numpy(np.ascontiguousarray(bank.carrier)) for bank in self.banks]).to(device, torch.float32)
        self.keep = torch.stack([torch.from_numpy(np.ascontiguousarray(bank.unit_mask)) for bank in self.banks]).to(device, torch.bool)
        self.raw4 = torch.zeros(batch, 4, decoder.units, device=device, dtype=torch.float32)
        self.cached_temporal = CpuLearnableRecencyRuntime(core, batch, device)
        self.versions = tuple(parameter._version for parameter in decoder.parameters())

    @torch.inference_mode()
    def stream_step(self, new_x, bank, stream_ids, unit_mask=None, *, valid_mask=None):
        if self.decoder.training:
            raise RuntimeError('decoder was switched to train mode')
        if tuple(parameter._version for parameter in self.decoder.parameters()) != self.versions:
            raise RuntimeError('decoder parameters changed after runtime registration')
        banks = list(bank) if not isinstance(bank, TaskBank) else [bank] * new_x.shape[0]
        if tuple(id(item) for item in banks) != self._bank_ids:
            raise RuntimeError('bank objects changed; call reset before advancing this stream')
        if new_x.ndim != 2 or new_x.shape != (len(self.banks), self.decoder.units):
            raise ValueError('observed must match fixed [B,64]')
        if valid_mask is None:
            valid_mask = torch.ones(len(self.banks), dtype=torch.bool, device=new_x.device)
        if valid_mask.dtype != torch.bool or valid_mask.shape != (len(self.banks),):
            raise ValueError('valid_mask must be bool [B]')
        x = new_x.to(self.raw4.device, torch.float32)
        valid = valid_mask.to(self.raw4.device)
        raw5 = torch.cat((self.raw4, x.unsqueeze(1)), dim=1)
        conv = self.decoder.frontend.local_conv
        flat = raw5.permute(0, 2, 1).reshape(x.shape[0] * self.decoder.units, 1, 5)
        local = conv.act(conv.conv(flat)).reshape(x.shape[0], self.decoder.units, 16, 1).permute(0, 3, 1, 2)
        z = self.decoder._fuse_batched_local(local, self.e0, self.carrier, self.keep)[:, 0]
        hidden = self.cached_temporal.step(z, valid)
        self.raw4.copy_(torch.where(valid[:, None, None], raw5[:, 1:], self.raw4))
        return self.decoder.readout(self.decoder.final_norm(hidden))


class M1FrozenEMAOfficialDecoder(BCIDecoder):
    def __init__(self, task_config, model_path, batch_size=1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.root = Path(model_path)
        self.set_batch_size(batch_size)
        self.doc = json.loads((self.root / 'payload_manifest.json').read_text())
        if self.doc.get('schema') != SCHEMA or self.doc.get('arm') != 'F_scaled_ladder_x16':
            raise ValueError('payload contract')
        if self.doc.get('runtime_film_module') is not False:
            raise ValueError('runtime must not carry a FiLM module')
        if self.doc.get('output_ema', {}).get('alpha') != 1 / 3 or self.doc.get('readout_adaptation') is not False:
            raise ValueError('inference-only EMA/readout contract')
        files = self.doc.get('payload_files_sha256', {})
        if not files:
            raise ValueError('missing payload ledger')
        for rel, d in files.items():
            if not (self.root / rel).is_file() or sha(self.root / rel) != d:
                raise ValueError('payload hash drift ' + rel)
        rows = self.doc.get('sessions', {})
        if len(rows) != 7:
            raise ValueError('need exactly official M1 seven tags')
        self.banks = {}
        for tag, row in rows.items():
            if self.task_config.hash_dataset(Path(row['official_tag_stem']).stem) != tag:
                raise ValueError('official tag drift')
            e = np.load(self.root / row['e0_file'])
            c = np.load(self.root / row['carrier_file'])
            if e.shape != (N, R) or c.shape != (N, 4) or e.dtype != np.float32 or c.dtype != np.float32:
                raise ValueError('bank geometry')
            self.banks[tag] = TaskBank(
                session_id=row['session'],
                E0=np.ascontiguousarray(e),
                carrier=np.ascontiguousarray(c),
                unit_mask=np.ones(N, np.bool_),
                X_store=np.zeros((0, R, N), np.float32),
                target_store=np.zeros((0, O), np.float32),
                window_ids=np.zeros(0, np.int64),
                calibration_meta={
                    'shape': (N, R),
                    'trial_count': int(row['support_trials']),
                    'estimator': 'frozen B3S FiLM-baked E0 (ladder x16 rank8); carrier from admitted frozen_bank_v2_replay',
                    'array_sha256': sha(self.root / row['e0_file']),
                    'budget': int(row['support_trials']),
                },
            )
        state = torch.load(self.root / 'selected_ema_decoder.pt', map_location='cpu', weights_only=False)
        self.decoder = build_decoder().cpu()
        self.decoder.load_state_dict(state['state_dict'], strict=True)
        self.decoder.eval()
        if tuple(self.decoder.temporal.core.slope_log.shape) != (4, 8):
            raise ValueError('loaded F slopes absent')
        if int((self.decoder.temporal.core.recency_slopes > 0).sum()) != 6:
            raise ValueError('loaded F ladder geometry drift')
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self._versions = tuple(p._version for p in self.decoder.parameters())
        self.ids = []
        self.stream = None
        self.ema = self.ready = None
        self.last_raw = None

    def set_batch_size(self, batch_size):
        self.batch_size = int(batch_size)
        if not 1 <= self.batch_size <= MAXB:
            raise ValueError('M1 batch size out of range')
        self.ids = []
        self.stream = None
        self.ema = self.ready = None
        self.last_raw = None

    def reset(self, dataset_tags: Iterable[Path] = (Path(''),)):
        ids = [self.task_config.hash_dataset(Path(x).stem) for x in dataset_tags]
        if not ids or len(ids) > self.batch_size or len(ids) != len(set(ids)) or any(i not in self.banks for i in ids):
            raise ValueError('unknown/invalid official M1 tag roster')
        self.ids = ids
        self.stream = CachedFRingRuntime(self.decoder, [self.banks[k] for k in self.ids])
        self.ema = self.ready = None
        self.last_raw = None

    def observe(self, neural_observations):
        return None

    def on_done(self, dones):
        return None

    @torch.inference_mode()
    def predict(self, neural_observations):
        if self.stream is None:
            raise RuntimeError('reset first')
        if tuple(p._version for p in self.decoder.parameters()) != self._versions:
            raise RuntimeError('frozen weights mutated')
        x = np.asarray(neural_observations, np.float32)
        active = len(x)
        if x.ndim != 2 or x.shape[1] != N or not 1 <= active <= len(self.ids):
            raise ValueError('expected active [B,64]')
        valid = torch.zeros(len(self.ids), dtype=torch.bool)
        valid[:active] = True
        z = np.zeros((len(self.ids), N), np.float32)
        z[:active] = x
        raw = self.stream.stream_step(torch.from_numpy(z), [self.banks[k] for k in self.ids], self.ids, valid_mask=valid).float()
        self.last_raw = raw.detach().clone()
        if self.ema is None:
            self.ema = raw.clone()
            self.ready = valid.clone()
        else:
            self.ema = torch.where((valid & self.ready)[:, None], ALPHA * self.ema + (1.0 - ALPHA) * raw, self.ema)
            self.ema = torch.where((valid & ~self.ready)[:, None], raw, self.ema)
            self.ready |= valid
        ans = self.ema[:active].cpu().numpy().astype(np.float32)
        if ans.shape != (active, O) or not np.isfinite(ans).all():
            raise RuntimeError('invalid output')
        return ans
