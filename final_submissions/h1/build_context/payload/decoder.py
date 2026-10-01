"""H1 F scaled-ladder x2 runtime with a contiguous KV ring.

Same payload contract as v1. Streaming no longer re-hashes banks or packs
Python K/V lists per bin: E0/carrier/keep are stacked at reset, and the
learned-slope temporal path uses CpuLearnableRecencyRuntime.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np, torch
from falcon_challenge.interface import BCIDecoder

HERE=Path(__file__).resolve().parent
for p in (HERE/'pkg',HERE):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from btransform_unified_v1.bank import TaskBank
from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.cpu_temporal import CpuLearnableRecencyRuntime
from learnable_recency_v1.temporal import LearnableRecencyTemporal
from unified_falcon.temporal import install_temporal
from a1_m2.nope_temporal_v1 import install_nope
from pretrained_nope_learned.temporal_scaled_ladder_v1 import install_f_scaled_ladder

SCHEMA='apst_pretrained_h1_frozen_c2_A_official_payload_v2'; PAYLOAD_SCHEMA=SCHEMA; CHANNELS=176; OUTPUTS=7; CONTEXT=300; MAX_BATCH=8
ALPHA=np.float32(1/3); SCALE=np.float32(20); LADDER_SCALE=2.0
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def build_decoder():
 d=install_temporal(RiftDecoder('h1',context_bins=CONTEXT,bias_mode='recency',seed=42),'h1','A_flat');install_nope(d,'h1')
 install_f_scaled_ladder(d,'h1',scale=LADDER_SCALE)
 if d.temporal.spec.use_sinusoidal_pe or d.temporal.spec.arm!='F_nope_learned' or tuple(d.temporal.core.slope_log.shape)!=(4,8): raise RuntimeError('F temporal construction drift')
 if int((d.temporal.core.recency_slopes>0).sum())!=6 or int((d.temporal.core.recency_slopes==0).sum())!=2:raise RuntimeError('F scaled-ladder six-active two-flat head drift')
 if not bool((d.temporal.core.effective_slopes()==d.temporal.core.recency_slopes.to(dtype=torch.float32)).all()):raise RuntimeError('F scaled-ladder uniform base drift')
 return d

class CachedFRingRuntime:
 """Fixed-row ring cache. Public method name matches A1M2StreamDecoder.stream_step."""
 def __init__(self, decoder, banks: Sequence[TaskBank]):
  if decoder.training: raise ValueError('CachedFRingRuntime requires decoder.eval()')
  core=decoder.temporal.core
  if decoder.temporal.spec.use_sinusoidal_pe or not isinstance(core, LearnableRecencyTemporal):
   raise TypeError('cached F runtime requires noPE LearnableRecencyTemporal core')
  if not banks: raise ValueError('cached F runtime requires at least one bank')
  self.decoder=decoder; self.banks=list(banks); self._bank_ids=tuple(id(bank) for bank in self.banks)
  device=next(decoder.parameters()).device; batch=len(self.banks)
  self.e0=torch.stack([torch.from_numpy(np.ascontiguousarray(bank.E0)) for bank in self.banks]).to(device, torch.float32)
  self.carrier=torch.stack([torch.from_numpy(np.ascontiguousarray(bank.carrier)) for bank in self.banks]).to(device, torch.float32)
  self.keep=torch.stack([torch.from_numpy(np.ascontiguousarray(bank.unit_mask)) for bank in self.banks]).to(device, torch.bool)
  self.raw4=torch.zeros(batch, 4, decoder.units, device=device, dtype=torch.float32)
  self.cached_temporal=CpuLearnableRecencyRuntime(core, batch, device)
  self.versions=tuple(parameter._version for parameter in decoder.parameters())
 @torch.inference_mode()
 def stream_step(self, new_x, bank, stream_ids, unit_mask=None, *, valid_mask=None):
  if self.decoder.training: raise RuntimeError('decoder was switched to train mode')
  if tuple(parameter._version for parameter in self.decoder.parameters())!=self.versions:
   raise RuntimeError('decoder parameters changed after runtime registration')
  banks=list(bank) if not isinstance(bank, TaskBank) else [bank]*new_x.shape[0]
  if tuple(id(item) for item in banks)!=self._bank_ids:
   raise RuntimeError('bank objects changed; call reset before advancing this stream')
  if new_x.ndim!=2 or new_x.shape!=(len(self.banks), self.decoder.units):
   raise ValueError('observed must match fixed [B,176]')
  if valid_mask is None:
   valid_mask=torch.ones(len(self.banks), dtype=torch.bool, device=new_x.device)
  if valid_mask.dtype!=torch.bool or valid_mask.shape!=(len(self.banks),):
   raise ValueError('valid_mask must be bool [B]')
  x=new_x.to(self.raw4.device, torch.float32); valid=valid_mask.to(self.raw4.device)
  raw5=torch.cat((self.raw4, x.unsqueeze(1)), dim=1)
  conv=self.decoder.frontend.local_conv
  flat=raw5.permute(0,2,1).reshape(x.shape[0]*self.decoder.units, 1, 5)
  local=conv.act(conv.conv(flat)).reshape(x.shape[0], self.decoder.units, 16, 1).permute(0,3,1,2)
  z=self.decoder._fuse_batched_local(local, self.e0, self.carrier, self.keep)[:,0]
  hidden=self.cached_temporal.step(z, valid)
  self.raw4.copy_(torch.where(valid[:,None,None], raw5[:,1:], self.raw4))
  return self.decoder.readout(self.decoder.final_norm(hidden))

class H1FrozenC2FOfficialDecoder(BCIDecoder):
 def __init__(self,task_config,model_path:str,batch_size:int=1):
  super().__init__(task_config=task_config,batch_size=batch_size);self.task_config=task_config;self.batch_size=int(batch_size)
  if not 1<=self.batch_size<=MAX_BATCH:raise ValueError('batch_size must be 1..8')
  self.root=Path(model_path).resolve();self.doc=json.loads((self.root/'payload_manifest.json').read_text())
  if self.doc.get('schema')!=PAYLOAD_SCHEMA or self.doc.get('status') not in ('SMOKE_ONLY_NOT_SELECTION','FORMAL_SELECTED'):raise ValueError('payload identity drift')
  if self.doc.get('arm')!='F_scaled_ladder_x2' or self.doc.get('runtime_film_module') is not False:raise ValueError('ladder-x2/noFiLM contract drift')
  files=self.doc.get('payload_files_sha256',{})
  if not files or any(not (self.root/k).is_file() or sha(self.root/k)!=v for k,v in files.items()):raise ValueError('payload hash drift')
  rows=self.doc.get('sessions',{})
  if len(rows)!=27:raise ValueError('requires 27 legal M3 bank rows')
  self.banks={}
  for tag,row in rows.items():
   if self.task_config.hash_dataset(Path(str(row['official_tag_stem'])).stem)!=tag:raise ValueError('official tag drift '+tag)
   e=np.load(self.root/row['e0_file'],allow_pickle=False);c=np.load(self.root/row['carrier_file'],allow_pickle=False);m=np.load(self.root/row['unit_mask_file'],allow_pickle=False)
   if e.shape!=(176,700) or c.shape!=(176,4) or m.shape!=(176,) or e.dtype!=np.float32 or c.dtype!=np.float32 or m.dtype!=np.bool_:raise ValueError('bank geometry drift '+tag)
   self.banks[tag]=TaskBank(session_id=str(row['session']),E0=np.ascontiguousarray(e),carrier=np.ascontiguousarray(c),unit_mask=np.ascontiguousarray(m),X_store=np.zeros((0,CONTEXT,CHANNELS),np.float32),target_store=np.zeros((0,OUTPUTS),np.float32),window_ids=np.zeros(0,np.int64),calibration_meta={'shape':(176,700),'trial_count':3,'budget':3,'estimator':'frozen C2 authority bank','array_sha256':str(row['e0_typed_sha256'])})
  state=torch.load(self.root/'selected_ema_decoder.pt',map_location='cpu',weights_only=False)
  if state.get('schema')!='apst_pretrained_h1_stage2_ema_state_v2':raise ValueError('EMA state schema drift')
  self.decoder=build_decoder().cpu();self.decoder.load_state_dict(state['state_dict'],strict=True);self.decoder.eval()
  if tuple(self.decoder.temporal.core.slope_log.shape)!=(4,8):raise ValueError('loaded F slopes absent')
  if int((self.decoder.temporal.core.recency_slopes>0).sum())!=6:raise ValueError('loaded F ladder geometry drift')
  for p in self.decoder.parameters():p.requires_grad_(False)
  self._versions=tuple(p._version for p in self.decoder.parameters());self.ids=[];self.stream=None;self.ema=None;self.init=None
 def reset(self,dataset_tags:Iterable[Path]=(Path(''),)):
  tags=[self.task_config.hash_dataset(Path(x).stem) for x in dataset_tags]
  if not tags or len(tags)>self.batch_size or len(tags)!=len(set(tags)) or any(x not in self.banks for x in tags):raise ValueError('invalid official H1 reset tags')
  self.ids=tags;self.stream=CachedFRingRuntime(self.decoder,[self.banks[t] for t in self.ids]);self.ema=None;self.init=None
 def on_done(self,dones):return None
 def observe(self,neural_observations):return None
 def set_batch_size(self,batch_size):
  self.batch_size=int(batch_size);self.ids=[];self.stream=None;self.ema=self.init=None
 @torch.inference_mode()
 def predict(self,neural_observations):
  if self.stream is None:raise RuntimeError('reset required')
  if tuple(p._version for p in self.decoder.parameters())!=self._versions:raise RuntimeError('decoder mutated')
  x=np.asarray(neural_observations,np.float32);n=len(x)
  if x.ndim!=2 or x.shape[1]!=CHANNELS or not 1<=n<=len(self.ids):raise ValueError('predict expects [active,176]')
  z=np.zeros((len(self.ids),CHANNELS),np.float32);z[:n]=x;valid=torch.zeros(len(self.ids),dtype=torch.bool);valid[:n]=True
  raw=self.stream.stream_step(torch.from_numpy(z),[self.banks[t] for t in self.ids],self.ids,valid_mask=valid)
  if self.ema is None:self.ema,self.init=raw.clone(),valid.clone()
  else:
   self.ema=torch.where((valid&self.init)[:,None],ALPHA*self.ema+(1-ALPHA)*raw,self.ema);self.ema=torch.where((valid&~self.init)[:,None],raw,self.ema);self.init|=valid
  out=(self.ema[:n].cpu().numpy()/SCALE).astype(np.float32)
  if out.shape!=(n,7) or not np.isfinite(out).all():raise RuntimeError('invalid prediction')
  return out

H1FrozenC2AOfficialDecoder=H1FrozenC2FOfficialDecoder
