"""Dedicated source/dev trainer for Static-F0 global E0.

Unlike the shared profile trainer, Static-F0 has no frozen source encoder: its
single global E0 table is deliberately source-trainable.  This module keeps
the shared sampler, optimizer, EMA, full-clock scorer and recipe, while making
that distinction explicit rather than bypassing the shared frozen-encoder
guard.
"""
from __future__ import annotations
import json, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
from torch import nn

from apst.dandi_subm import training as T
from apst.dandi_subm import protocol
from apst.dandi_subm.subm_common import (PairedSampler, aggregate_scores, atomic_json, digest,
                         fit_source_stats, load_records, record_binding, require_full_source, sha256)
from apst.dandi_subm.subm_data import padded_windows
from apst.models.ema import DecoderEMA
from apst.models.frontend import unit_dropout_seed, whole_unit_dropout
from apst.models.schedule import warmup_cosine_lr

SCHEMA='dandi688_subm_static_f0_v1'
RECIPE={"segments":24,"updates_per_segment":3165,"batch":32,"lr_peak":3e-4,"lr_min_factor":.1,"warmup_updates":3165,"weight_decay":.01,"betas":[.9,.999],"eps":1e-8,"grad_clip":1.,"ema_decay":.9995,"whole_unit_dropout":.1}

class StaticEncoder(nn.Module):
 def __init__(self,units=100,e0_dim=50,seed=42):
  super().__init__()
  with torch.random.fork_rng(devices=[]):torch.manual_seed(seed+0x53544154);self.static_e0=nn.Parameter(torch.randn(units,e0_dim)*.02)
 def forward(self,activity,profile=None):return self.static_e0
 def encode(self,activity,profile=None):return self.static_e0

def build(seed=42):
 model=T._build('act_only','F0',seed,'pretrain',None)
 model.encoder=StaticEncoder(T._units(model),50,seed);model.method='act_only'
 return model

def smoke_dev_score(model, records, stats, device):
 """Eight Q50 endpoints/session, direct causal forward, no full-recording EMA.

 This is solely an adapter smoke surface; formal training below retains the
 exact shared full-clock development scorer.
 """
 rows=[]
 for record in records:
  endpoints=np.asarray(record.query_indices[:8],np.int64);x,valid,mask=padded_windows(record,endpoints,T._units(model))
  with torch.inference_mode():
   output=T._forward(model,torch.from_numpy(x).to(device),T._activity(record,device,T._units(model)),None,unit_mask=torch.from_numpy(mask).to(device).expand(len(endpoints),-1),dropout_keep=None,valid=torch.from_numpy(valid).to(device))
  physical=output.float().cpu().numpy()*np.asarray(stats['velocity_std'],np.float32)+np.asarray(stats['velocity_mean'],np.float32)
  rows.append(T.score_predictions(replace(record,query_indices=endpoints),physical))
 return aggregate_scores(rows)

def train(cache:Path,dest:Path,*,device='cpu',smoke_updates:int|None=None):
 cache,dest=Path(cache).resolve(),Path(dest).resolve();smoke=smoke_updates is not None
 if dest.exists() and any(dest.iterdir()):raise FileExistsError('static destination must be new')
 if not (cache/'prepared_receipt.json').is_file():raise FileNotFoundError(cache/'prepared_receipt.json')
 if smoke and not 1<=int(smoke_updates)<=8:raise ValueError('smoke updates must be 1..8')
 dest.mkdir(parents=True);dev_device=torch.device(device)
 if dev_device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.manual_seed(42);np.random.seed(42)
 source=load_records(cache,'sua','train');require_full_source(source);stats=fit_source_stats(source);atomic_json(dest/'source_stats.json',stats)
 model=build().to(dev_device);opt,optimizer_receipt=T.build_optimizer(model,RECIPE);params=[p for p in model.parameters() if p.requires_grad]
 if 'encoder.static_e0' not in optimizer_receipt['trainable']:raise RuntimeError('static E0 missing from optimizer')
 ema=DecoderEMA(model,decay=RECIPE['ema_decay'])
 # It is intentionally in EMA because it is not a frozen source encoder.
 if 'encoder.static_e0' not in ema.shadow:raise RuntimeError('static E0 missing from EMA')
 dev=load_records(cache,'sua','dev');segments,updates,batch=((1,int(smoke_updates),2) if smoke else (24,3165,32))
 meta={'schema':SCHEMA,'status':'SMOKE' if smoke else 'FORMAL','method':'Static-F0 (source-trained global E0)','seed':42,'model_type':'static_identity_global_table','no_session_encoder':True,'no_profile':True,'no_token':True,'static_e0_shape':[100,50],'recipe':RECIPE,'source_binding':record_binding(source),'source_stats_sha256':stats['sha256'],'final_sessions_opened':0}
 atomic_json(dest/'protocol.json',meta);atomic_json(dest/'optimizer_receipt.json',{'schema':SCHEMA+'_optimizer','optimizer':optimizer_receipt,'final_sessions_opened':0})
 activity={r.session_id:T._activity(r,dev_device,T._units(model)) for r in source};ymean=torch.tensor(stats['velocity_mean'],device=dev_device);ystd=torch.tensor(stats['velocity_std'],device=dev_device)
 sampler=PairedSampler(source,42,batch=batch,updates_per_segment=updates);curve=[];best=None;step=0;started=time.monotonic()
 for segment in range(segments):
  model.train();losses=[]
  for bi,(record,endpoints) in enumerate(sampler.segment()):
   x,valid,mask=padded_windows(record,endpoints,T._units(model));tx=torch.from_numpy(x).to(dev_device);tv=torch.from_numpy(valid).to(dev_device);um=torch.from_numpy(mask).to(dev_device).expand(len(endpoints),-1)
   keep=whole_unit_dropout(um,p=RECIPE['whole_unit_dropout'],generator=torch.Generator(device='cpu').manual_seed(unit_dropout_seed(42,segment+1,bi)))
   target=(torch.from_numpy(record.velocity[endpoints]).to(dev_device)-ymean)/ystd;step+=1
   lr=warmup_cosine_lr(step,total_steps=segments*updates,warmup_steps=min(RECIPE['warmup_updates'],max(1,segments*updates//2)) if smoke else RECIPE['warmup_updates'],peak=RECIPE['lr_peak'],min_factor=RECIPE['lr_min_factor'])
   for group in opt.param_groups:group['lr']=lr*float(group.get('lr_multiplier',1.))
   opt.zero_grad(set_to_none=True);ctx=torch.autocast('cuda',dtype=torch.bfloat16) if dev_device.type=='cuda' else __import__('contextlib').nullcontext()
   with ctx:loss=nn.functional.mse_loss(T._forward(model,tx,activity[record.session_id],None,unit_mask=um,dropout_keep=keep,valid=tv).float(),target.float())
   if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite static loss')
   loss.backward();nn.utils.clip_grad_norm_(params,RECIPE['grad_clip'],error_if_nonfinite=True);opt.step();ema.update_after_step(model);losses.append(float(loss.detach().cpu()))
  ck=dest/f'segment_{segment+1:02d}.pt';raw={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};ema.apply_to(model)
  try:
   torch.save({'schema':SCHEMA+'_checkpoint','model_state':{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},'source_stats':stats,'metadata':meta,'global_step':step},ck)
   scores=smoke_dev_score(model,dev,stats,dev_device) if smoke else aggregate_scores([T.score_predictions(r,T.predict_record(model,r,stats)[0]) for r in dev])
  finally:
   with torch.no_grad():
    for n,p in model.named_parameters():
     if n in raw:p.copy_(raw[n])
  row={'segment':segment+1,'global_step':step,'mean_loss':float(np.mean(losses)),'checkpoint':str(ck),'checkpoint_sha256':sha256(ck),'development':scores,'development_surface':'SMOKE_8Q50_direct_no_EMA' if smoke else 'FORMAL_full_clock_EMA'};curve.append(row)
  if best is None or scores['mean_r2']>best['mean_r2']:best={'mean_r2':scores['mean_r2'],'checkpoint':str(ck),'checkpoint_sha256':sha256(ck),'segment':segment+1}
  atomic_json(dest/'progress.json',{'segments':curve,'final_sessions_opened':0})
 if best is None:raise RuntimeError('no static development selection')
 selection={'schema':SCHEMA+'_selection','status':'SMOKE' if smoke else 'FORMAL','rule':'earliest_max_equal_session_dev_r2','checkpoint':best['checkpoint'],'checkpoint_sha256':best['checkpoint_sha256'],'mean_dev_r2':best['mean_r2'],'selected_segment':best['segment'],'dev_sessions':list(protocol.DEV_SESSIONS),'final_sessions_opened':0};atomic_json(dest/'selection.json',selection)
 receipt={**meta,'completed':True,'actual_updates':step,'segments':curve,'selection':selection,'sampler':sampler.receipt(),'elapsed_seconds':time.monotonic()-started};atomic_json(dest/'train_receipt.json',receipt)
 return receipt
