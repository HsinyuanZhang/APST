"""Faithful portable extraction of the historical DANDI one-layer LSTM campaign."""
from __future__ import annotations
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from .core import BUDGETS,HPS,ProtocolError,RNN,choose_development,evaluate,label_stats,load_manifest,seed_everything,sha256,train,write_json
EVAL=(1,2,5,10,20,40,80,120,160,200); LRS=(1e-5,3e-5,1e-4,3e-4); STEPS=(0,16,32,64,128,256,512)
def dev(name):
 d=torch.device(name)
 if d.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
 return d
def fresh(h,d):seed_everything();return RNN(h).to(d)
def source(data,out,d,c):
 ids=[r.query for r in data['train']];mean,std=label_stats(data['train'],ids);cells=[]
 for h,lr in c['source_hps']:
  m=fresh(h,d);curve=[];best=None;state=None
  for epoch in range(1,c['source_epochs']+1):
   state=train(m,data['train'],ids,mean,std,lr=lr,updates=c['source_updates_per_epoch'],device=d,state=state);score,rows=evaluate(m,data['dev'],mean,std,d);entry={'epoch':epoch,'mean_r2':score,'sessions':rows};curve.append(entry)
   if best is None or score>best['mean_r2']:
    p=out/f'zero_h{h}_lr{lr:g}.pt';torch.save({'state_dict':m.state_dict(),'mean':mean,'std':std,'hidden':h,'epoch':epoch},p);best=entry|{'checkpoint':str(p.resolve()),'checkpoint_sha256':sha256(p)}
  cells.append({'config':{'hidden':h,'lr':lr,'epochs':c['source_epochs'],'updates_per_epoch':c['source_updates_per_epoch']},'dev_r2':best['mean_r2'],'selected_epoch':best['epoch'],'checkpoint':best['checkpoint'],'checkpoint_sha256':best['checkpoint_sha256'],'curve':curve})
 ans=choose_development(cells);write_json(out/'source_development.json',{'status':'SELECTED','selection_rule':'max equal-session dev R2; configuration order then earliest epoch break ties','candidates':cells,'selected':ans});return ans
def scratch(data,d,c):
 cells=[]
 for h,lr in c['scratch_hps']:
  curve=[]
  for epoch in c['scratch_eval_epochs']:
   rows=[]
   for r in data['dev']:
    m=fresh(h,d);mean,std=label_stats([r],[r.support[32]]);train(m,[r],[r.support[32]],mean,std,lr=lr,updates=epoch*c['scratch_updates_per_epoch'],device=d);rows+=evaluate(m,[r],mean,std,d)[1]
   curve.append({'epoch':epoch,'mean_r2':float(np.mean([x['r2'] for x in rows])),'sessions':rows})
  b=max(curve,key=lambda x:x['mean_r2']);cells.append({'config':{'hidden':h,'lr':lr,'updates_per_epoch':c['scratch_updates_per_epoch']},'selected_epoch':b['epoch'],'dev_r2':b['mean_r2'],'curve':curve})
 return choose_development(cells),cells
def fine(data,d,src,c):
 z=torch.load(src['checkpoint'],map_location='cpu',weights_only=False);cells=[]
 for lr in c['finetune_lrs']:
  for steps in c['finetune_steps']:
   rows=[]
   for r in data['dev']:
    m=fresh(int(z['hidden']),d);m.load_state_dict(z['state_dict']);mean=np.asarray(z['mean'],np.float32);std=np.asarray(z['std'],np.float32);train(m,[r],[r.support[32]],mean,std,lr=lr,updates=steps,device=d);rows+=evaluate(m,[r],mean,std,d)[1]
   cells.append({'lr':lr,'updates':steps,'dev_r2':float(np.mean([x['r2'] for x in rows])),'sessions':rows})
 return choose_development(cells),cells
def final(records,d,recipe,kind,src):
 rows=[];z=torch.load(src['checkpoint'],map_location='cpu',weights_only=False)
 for b in BUDGETS:
  for r in records:
   if kind=='scratch':m=fresh(recipe['config']['hidden'],d);mean,std=label_stats([r],[r.support[b]]);lr=recipe['config']['lr'];updates=recipe['selected_epoch']*recipe['config']['updates_per_epoch']
   else:m=fresh(int(z['hidden']),d);m.load_state_dict(z['state_dict']);mean=np.asarray(z['mean'],np.float32);std=np.asarray(z['std'],np.float32);lr=recipe['lr'];updates=recipe['updates']
   train(m,[r],[r.support[b]],mean,std,lr=lr,updates=updates,device=d);_,receipt=evaluate(m,[r],mean,std,d);rows.append({'arm':kind,'budget':b,'lr':lr,'updates':updates,**receipt[0]})
 return rows
def run(a):
 c=json.loads(Path(a.config).read_text());c.setdefault('source_hps',[list(x) for x in HPS]);c.setdefault('source_epochs',20);c.setdefault('source_updates_per_epoch',256);c.setdefault('scratch_hps',[list(x) for x in HPS]);c.setdefault('scratch_eval_epochs',list(EVAL));c.setdefault('scratch_updates_per_epoch',16);c.setdefault('finetune_lrs',list(LRS));c.setdefault('finetune_steps',list(STEPS));out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);d=dev(a.device);data=load_manifest(a.manifest,('train','dev'));src=source(data,out,d,c);scr,sgrid=scratch(data,d,c);fin,fgrid=fine(data,d,src,c)
 manifest=json.loads(Path(a.manifest).read_text());subject=manifest.get('subject')
 if subject not in ('sub-C','sub-M'):raise ValueError('manifest subject must be sub-C or sub-M')
 if subject=='sub-C':from apst.dandi_subc import protocol
 else:from apst.dandi_subm import protocol
 config_snapshot=out/'frozen_config.json';manifest_snapshot=out/'frozen_source_dev_manifest.json';shutil.copyfile(a.config,config_snapshot);shutil.copyfile(a.manifest,manifest_snapshot)
 artifacts={src['checkpoint']:src['checkpoint_sha256'],str(config_snapshot.resolve()):sha256(config_snapshot),str(manifest_snapshot.resolve()):sha256(manifest_snapshot)}
 seal={'schema':'dandi_rnn_final_selection_v1','status':'FROZEN_BEFORE_FINAL_LOAD','subject':subject,'protocol':protocol.protocol_dict(),'authorized_sessions':list(protocol.FINAL_SESSIONS),'source':src,'scratch_m32':scr,'finetune_m32':fin,'budgets':list(BUDGETS),'artifacts':artifacts,'config_sha256':sha256(config_snapshot),'source_dev_manifest_sha256':sha256(manifest_snapshot),'selection_rule':'M32-only target selection, frozen recipe reused at M4/M8/M16/M32'};seal['sha256']=hashlib.sha256(json.dumps(seal,sort_keys=True,separators=(',',':')).encode()).hexdigest();write_json(out/'development_seal.json',seal)
 try:
  # Validate the immutable RNN seal before *every* cached-final load too.
  from .adapter import _rnn_final_access
  _rnn_final_access(subject[-1],out/'development_seal.json');records=load_manifest(a.manifest,('final',))['final']
 except ProtocolError:
  if not a.raw_root: raise
  from .adapter import prepare
  prepare(subject[-1],Path(a.raw_root),Path(a.manifest).resolve().parent,out/'development_seal.json');records=load_manifest(a.manifest,('final',))['final']
 z=torch.load(src['checkpoint'],map_location='cpu',weights_only=False);m=fresh(int(z['hidden']),d);m.load_state_dict(z['state_dict']);zero=evaluate(m,records,np.asarray(z['mean']),np.asarray(z['std']),d);write_json(out/'results.json',{'schema':'dandi_rnn_release_v1','development_seal':str((out/'development_seal.json').resolve()),'source_pretraining':src,'zero_shot':{'mean_r2':zero[0],'sessions':zero[1]},'target_from_scratch':{'m32_grid':sgrid,'final':final(records,d,scr,'scratch',src)},'source_pretrained_target_finetuning':{'m32_grid':fgrid,'final':final(records,d,fin,'finetune',src)}})
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--raw-root',help='lazy sealed final-NWB materialization if manifest has no final split');run(p.parse_args(argv))
