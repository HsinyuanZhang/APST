#!/usr/bin/env python3
"""SUA static ablation: learned global [units, e0_dim] table, no encoder, no token.

Mirrors the sealed sites recipe exactly (24x3165, warmup 3165, cosine .1,
AdamW two-group, EMA .9995, unit dropout .1, per-segment dev selection on the
2015 dev_2015 split).  The model is the act_only/F0 cell with its ProfileEncoder
replaced by a static table: act_only routing already supplies carrier=None and
token=None, so the static semantics fall out of the existing site dispatch.
"""
from __future__ import annotations
import argparse, contextlib, json, socket, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn

from . import training as T
from .bootstrap import output_root
from apst.models.ema import DecoderEMA
from apst.models.frontend import whole_unit_dropout, unit_dropout_seed
from apst.models.schedule import warmup_cosine_lr


SCHEMA = 'dandi688_sua_static_v1'
CACHE = None
RECIPE = T._recipe({'recipe': {'segments':24,'updates_per_segment':3165,'batch':32,'lr_peak':3e-4,'lr_min_factor':.1,'warmup_updates':3165,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'grad_clip':1.,'ema_decay':.9995,'whole_unit_dropout':.1}})


class StaticEncoder(nn.Module):
    def __init__(self, units: int, e0_dim: int, seed: int = 42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 0x53544154)
            self.static_e0 = nn.Parameter(torch.randn(units, e0_dim) * 0.02)

    def forward(self, activity, profile):
        return self.static_e0

    def encode(self, activity, profile):
        return self.static_e0


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def atom(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp'); q.write_text(json.dumps(d, indent=2, default=str) + '\n'); q.replace(p)


def run(a):
    dest = Path(a.dest).resolve()
    if dest.exists(): raise RuntimeError('fresh dest required')
    dev_device = torch.device(a.device)
    torch.manual_seed(42); np.random.seed(42)
    dest.mkdir(parents=True)

    cache=Path(a.cache).resolve(); records = T.load_records(cache, 'sua', 'train')
    dev = T.load_records(cache, 'sua', 'dev')
    stats = T.fit_source_stats(records, smoke=False)
    atom(dest / 'source_stats.json', stats)

    # act_only/F0 trunk with the ACT pretrain encoder (immediately replaced)
    model = T._build('act_only', 'F0', 42, 'pretrain', None).to(dev_device)
    units = T._units(model)
    model.encoder = StaticEncoder(units, 50, 42).to(dev_device)
    model.method = 'act_only'  # site routing: carrier=None, token=None

    opt, _contract = T.build_optimizer(model, RECIPE)
    params = [p for p in model.parameters() if p.requires_grad]
    ema = DecoderEMA(model, decay=RECIPE['ema_decay'])

    meta = {'schema': SCHEMA, 'status': 'FORMAL_RUNNING', 'method': 'static', 'temporal': 'F0',
            'representation': 'sua', 'seed': 42, 'device': str(dev_device),
            'model_type': 'static_identity_global_table', 'no_encoder': True, 'no_token': True,
            'static_e0_shape': [units, 50], 'recipe': RECIPE,
            'started_utc': datetime.now(timezone.utc).isoformat(), 'machine': socket.gethostname()}
    atom(dest / 'run_meta_start.json', meta)

    ymean = torch.tensor(stats['velocity_mean'], device=dev_device)
    ystd = torch.tensor(stats['velocity_std'], device=dev_device)
    calibration = {r.session_id: (T._activity(r, dev_device, units), None) for r in records}
    sampler = T.PairedSampler(records, 42, batch=RECIPE['batch'], updates_per_segment=RECIPE['updates_per_segment'])
    curve, step, best = [], 0, None
    for segment in range(RECIPE['segments']):
        model.train(); losses = []
        for bi, (record, endpoints) in enumerate(sampler.segment()):
            x, valid, mask = T.padded_windows(record, endpoints, units)
            tx = torch.from_numpy(x).to(dev_device)
            tv = torch.from_numpy(valid).to(dev_device)
            um = torch.from_numpy(mask).to(dev_device).expand(len(endpoints), -1)
            keep = whole_unit_dropout(um, p=RECIPE['whole_unit_dropout'],
                                      generator=torch.Generator(device='cpu').manual_seed(unit_dropout_seed(42, segment + 1, bi)))
            activity, carrier = calibration[record.session_id]
            target = (torch.from_numpy(record.velocity[endpoints]).to(dev_device) - ymean) / ystd
            step += 1
            lr = warmup_cosine_lr(step, total_steps=RECIPE['segments'] * RECIPE['updates_per_segment'],
                                  warmup_steps=RECIPE['warmup_updates'], peak=RECIPE['lr_peak'],
                                  min_factor=RECIPE['lr_min_factor'])
            for g in opt.param_groups: g['lr'] = lr * float(g.get('lr_multiplier', 1.0))
            opt.zero_grad(set_to_none=True)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if dev_device.type == 'cuda' else contextlib.nullcontext()
            with amp:
                loss = nn.functional.mse_loss(T._forward(model, tx, activity, carrier, unit_mask=um, dropout_keep=keep, valid=tv).float(), target.float())
            if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite source loss')
            loss.backward()
            nn.utils.clip_grad_norm_(params, RECIPE['grad_clip'], error_if_nonfinite=True)
            opt.step(); ema.update_after_step(model); losses.append(float(loss.detach()))
        ck = dest / f'segment_{segment + 1:02d}.pt'
        raw = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        ema.apply_to(model)
        try:
            torch.save({'schema': SCHEMA + '_checkpoint', 'segment': segment + 1, 'global_step': step,
                        'model_state': {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                        'source_stats': stats}, ck)
            rows = [T.score_predictions(r, T.predict_record(model, r, stats)[0]) for r in dev]
            development = T.aggregate_scores(rows)
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in raw: p.copy_(raw[n])
        curve.append({'segment': segment + 1, 'global_step': step, 'mean_loss': float(np.mean(losses)),
                      'checkpoint': ck.name, 'checkpoint_sha256': sha256(ck), 'development': development})
        if np.isfinite(development['mean_r2']) and (best is None or development['mean_r2'] > best[0]):
            best = (development['mean_r2'], ck)
        atom(dest / 'progress.json', {'segments': curve})
        print(json.dumps({'segment': segment + 1, 'dev_mean_r2': development['mean_r2']}), flush=True)

    meta.update(status='COMPLETED', selected_segment=best[1].stem.split('_')[1],
                selected_dev=best[0], selected_checkpoint=best[1].name,
                selected_checkpoint_sha256=sha256(best[1]))
    atom(dest / 'run_meta.json', meta)
    atom(dest / 'score_receipt.json', {'schema': SCHEMA, 'status': 'COMPLETED', 'method': 'static',
                                       'temporal': 'F0', 'representation': 'sua', 'seed': 42,
                                       'split': 'dev_2015', 'selected_segment': int(best[1].stem.split('_')[1]),
                                       'selected_dev_mean_r2': best[0],
                                       'selection': 'best dev segment, EMA weights',
                                       'per_segment_curve': [{'segment': c['segment'], 'dev_mean_r2': c['development']['mean_r2']} for c in curve]})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dest', required=True); p.add_argument('--cache', required=True); p.add_argument('--device', default='cuda:0')
    run(p.parse_args())
