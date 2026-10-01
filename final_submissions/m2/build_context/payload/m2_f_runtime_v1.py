#!/usr/bin/env python3
"""Portable CPU runtime for M2 F: no positional encoding plus learned recency.

The payload is deliberately self-contained: its build context vendors the F
temporal installer, and this runner only accepts the selected weight-EMA state.
Scores are in the producer's scale during streaming and are divided by five at
the public SDK boundary; output EMA is inference-only.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder
from a1_m2.nope_temporal_v1 import build_decoder
from a1_m2.temporal_variants import A1M2StreamDecoder
from btransform_unified_v1.bank import TaskBank, array_sha256
from pretrained_nope_learned.temporal_v1 import install_f_nope_learned

ALPHA=1.0/3.0; WEIGHT_EMA_DECAY=.9995; SESSION_COUNT=13
PAYLOAD_SCHEMA='apst_m2_f_export_v1_payload'; DECODER_STATE_SCHEMA='apst_m2_f_selected_ema_decoder_v1'
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def typed(x:np.ndarray)->str:
 x=np.ascontiguousarray(x); return hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def doc(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text());
 if not isinstance(x,dict): raise ValueError('payload manifest is not an object')
 return x

class M2FNoPELearnedDecoder(BCIDecoder):
 alpha=ALPHA
 def __init__(self,task_config,model_path,batch_size:int=1):
  super().__init__(task_config=task_config,batch_size=batch_size); self.task_config=task_config; self.root=Path(model_path).resolve()
  self.doc=doc(self.root/'payload_manifest.json'); self._check_payload()
  status=self.doc.get('status'); formal_status='FORMAL_SELECTED_PREPARATION_NO_POST'
  if self.doc.get('schema')!=PAYLOAD_SCHEMA or self.doc.get('arm')!='F_nope_learned' or status not in ('PREPARATION_NOT_GLOBAL_SELECTION','NONFORMAL_CPU2_SMOKE',formal_status):
   raise ValueError('F payload identity/status drift')
  self.formal_payload=(status==formal_status); final=self.doc.get('final_decoder',{})
  if self.formal_payload:
   selection=self.doc.get('selection')
   if not isinstance(selection,dict) or selection.get('formal') is not True or not isinstance(selection.get('epoch'),int) or selection['epoch']<1 or selection['epoch']!=final.get('checkpoint_epoch') or not isinstance(final.get('checkpoint_sha256'),str) or len(final['checkpoint_sha256'])!=64:
    raise ValueError('formal F selection/epoch/checkpoint metadata drift')
  state=torch.load(self.root/'selected_ema_decoder.pt',map_location='cpu',weights_only=False)
  if state.get('schema')!=DECODER_STATE_SCHEMA or state.get('weight_ema_decay')!=WEIGHT_EMA_DECAY or state.get('output_ema_alpha')!=ALPHA:
   raise ValueError('selected F EMA metadata drift')
  if self.formal_payload and (state.get('decoder_checkpoint_sha256')!=final['checkpoint_sha256'] or state.get('decoder_checkpoint_epoch')!=final['checkpoint_epoch']):
   raise ValueError('formal F checkpoint state/manifest binding drift')
  self.model=build_decoder('E_nope_flat',seed=42,proj_dim=16); install_f_nope_learned(self.model,'m2'); self.model.load_state_dict(state['state_dict'],strict=True); self.model.eval()
  core=self.model.temporal.core
  if self.model.temporal.spec.use_sinusoidal_pe or tuple(core.slope_log.shape)!=(4,8) or int((core.recency_slopes>0).sum())!=6:
   raise ValueError('loaded model is not F noPE learned-slope [4,8]/six-active-heads')
  rows=self.doc.get('sessions',{})
  if not isinstance(rows,dict) or len(rows)!=SESSION_COUNT: raise ValueError('expected 13 legal M33 banks')
  self.banks={}; self.ids=[]; self.runtime=None; self.ema=None; self.initialized=None
  for sid,row in sorted(rows.items()): self._bank(sid,row)
 def _check_payload(self):
  expected=self.doc.get('payload_files_sha256');
  actual={str(p.relative_to(self.root)):sha(p) for p in self.root.rglob('*') if p.is_file() and p.name!='payload_manifest.json'}
  if not isinstance(expected,dict) or actual!=expected: raise ValueError('payload closure/hash ledger drift')
 def _bank(self,sid:str,row:dict[str,Any]):
  e=self.root/'calibration'/sid/'E0.npy'; c=self.root/'calibration'/sid/'carrier.npy'
  if sha(e)!=row.get('payload_e0_file_sha256') or sha(c)!=row.get('payload_carrier_file_sha256'): raise ValueError(sid+': bank hash drift')
  e0=np.load(e,mmap_mode='r',allow_pickle=False); carrier=np.load(c,mmap_mode='r',allow_pickle=False)
  if e0.shape!=(96,50) or carrier.shape!=(96,4) or e0.dtype!=np.float32 or carrier.dtype!=np.float32 or typed(e0)!=row.get('payload_e0_typed_sha256') or typed(carrier)!=row.get('payload_carrier_typed_sha256'): raise ValueError(sid+': bank geometry/typed hash drift')
  tag=self.task_config.hash_dataset(Path(str(row['official_tag_stem'])).stem)
  if tag!=row.get('official_dataset_tag_hash') or tag in self.banks: raise ValueError(sid+': dataset tag drift')
  self.banks[tag]=TaskBank(sid,np.asarray(e0),np.asarray(carrier),np.ones(96,dtype=np.bool_),np.zeros((0,1,96),np.float32),np.zeros((0,2),np.float32),np.zeros(0,np.int64),{'shape':[96,50],'trial_count':33,'estimator':'frozen_pretrained_A_E0_carrier','array_sha256':array_sha256(np.asarray(e0)),'budget':33})
 def reset(self,dataset_tags=(Path(''),)):
  tags=[self.task_config.hash_dataset(Path(x).stem) for x in dataset_tags]
  if not tags or len(set(tags))!=len(tags) or any(x not in self.banks for x in tags) or len(tags)>self.batch_size: raise ValueError('unknown/duplicate/oversize reset roster')
  self.ids=tags; self.runtime=A1M2StreamDecoder(self.model); self.ema=None; self.initialized=None
 def observe(self,neural_observations): return None
 def on_done(self,dones): return None
 def predict(self,neural_observations):
  if self.runtime is None: raise RuntimeError('reset must precede predict')
  x=np.asarray(neural_observations,dtype=np.float32); n=x.shape[0] if x.ndim==2 else 0
  if x.ndim!=2 or x.shape[1:]!=(96,) or not 1<=n<=len(self.ids) or not np.isfinite(x).all(): raise ValueError('observations must be finite [1..batch,96]')
  padded=np.zeros((len(self.ids),96),np.float32); padded[:n]=x; valid=torch.zeros(len(self.ids),dtype=torch.bool); valid[:n]=True
  # Match the producer scorer: unscale first, then retain causal EMA in
  # native float64.  Inference never builds an autograd graph.
  with torch.no_grad(): score=self.runtime.stream_step(torch.from_numpy(padded),[self.banks[k] for k in self.ids],self.ids,valid_mask=valid)
  native=score.detach().cpu().numpy().astype(np.float64,copy=False)/5.0
  if self.ema is None: self.ema=np.zeros_like(native,dtype=np.float64); self.initialized=np.zeros(len(self.ids),dtype=np.bool_)
  active=valid.cpu().numpy() & self.initialized; fresh=valid.cpu().numpy() & ~self.initialized; self.ema[active]=ALPHA*self.ema[active]+(1-ALPHA)*native[active]; self.ema[fresh]=native[fresh]; self.initialized|=valid.cpu().numpy()
  return self.ema[:n].copy()
__all__=['M2FNoPELearnedDecoder','ALPHA']
