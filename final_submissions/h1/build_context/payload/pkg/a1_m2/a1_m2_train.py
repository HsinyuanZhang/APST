#!/usr/bin/env python3
"""Formal M2 A1 temporal-arm runner over a frozen pure-concat A bank.

This owns no association-profile parameters: the admission boundary validates
and freezes the producer bank before a decoder is initialized.  Every arm
therefore sees the identical E0/carrier arrays, source batches, optimizer,
seed, decoder non-temporal initialization, and output EMA selection rule.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, random, shutil, socket, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent; WEEK = HERE.parents[1]; WS = WEEK.parent
V2 = WS / "btransform_unified_v2"; LR = V2 / "learnable_recency_v1"; V1 = WS / "btransform_unified_v1"
for item in (WEEK / "src", V2 / "src", LR / "src", LR / "scripts", V2 / "src", V1 / "src", WS):
    if str(item) not in sys.path: sys.path.insert(0, str(item))
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
import activity_data
from a1_m2.pure_concat_bank import load_and_validate_manifest, sha256
from a1_m2.scoring_contract import causal_output_ema, earliest_finite_maximum, ext6_equal_session_selection, output_centered_variance_weighted_r2
from a1_m2.temporal_variants import SPECS, build_decoder, contract

SCHEMA = "apst_final_week_a1_m2_temporal_runner_v1"; CONTEXT=50; BATCH=32

def now() -> str: return datetime.now(timezone.utc).isoformat()
def atom(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8"); tmp.replace(path)
def append(path: Path, value: Any) -> None:
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(value,sort_keys=True)+"\n")
def state_sha(model: nn.Module) -> str:
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()): h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def rng_state(device:torch.device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
def restore_rng(state:Mapping[str,Any],device:torch.device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type=='cuda' and state.get('torch_cuda') is not None: torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])

def absolute(starts: np.ndarray, pad: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(starts,np.int64)[:,None]-int(pad)+np.arange(CONTEXT)[None,:],device=device)

def item_batch(item: Mapping[str,Any], ids: np.ndarray, device: torch.device):
    x,y,valid=activity_data.windows(item,ids,CONTEXT,device,5.0)
    return x,y,valid,absolute(np.asarray(item['starts'])[ids],int(item['pad']),device)

def _bank(manifest: Mapping[str,Any], item: Mapping[str,Any], plane: str, session: str) -> TaskBank:
    row=manifest['sessions'][f'{plane}::{session}']; root=Path(manifest['_path']).parent
    e0=np.load(root/row['e0_path']); carrier=np.load(root/row['carrier_path'])
    # The decoder consumes only E0/carrier/mask.  TaskBank additionally
    # validates observational stores, so provide a local shape-valid ledger
    # rather than accidentally reinterpret raw [bins,units] as windows.
    count=len(item['starts']); targets=np.zeros((count,2),np.float32)
    return TaskBank(session,np.asarray(e0,np.float32),np.asarray(carrier,np.float32),np.ones(96,np.bool_),
        np.zeros((count,1,96),np.float32),targets,np.asarray(item['starts'],np.int64),
        {'shape':list(e0.shape),'trial_count':33,'estimator':'new_pure_concat_association_profile','array_sha256':array_sha256(e0),'budget':33,
         'bank_manifest_sha256':manifest['_sha'],'carrier_sha256':array_sha256(carrier)})

def load_data(manifest_path: Path, *, allow_smoke: bool):
    admitted=load_and_validate_manifest(manifest_path,allow_smoke=allow_smoke)
    doc=json.loads(manifest_path.read_text()); doc['_path']=str(manifest_path); doc['_sha']=admitted['manifest_sha256']
    data=activity_data.load_m2_data(include_eval=True)
    planes={'source_train':data['train'],'source_minival':data['validation'],'ext6_development':data['evaluation']}
    banks={plane:{s:_bank(doc,item,plane,s) for s,item in items.items()} for plane,items in planes.items()}
    # Required query coordinate artifacts must match the actual items consumed.
    root=manifest_path.parent
    for plane,items in planes.items():
        for s,item in items.items():
            row=doc['sessions'][f'{plane}::{s}']; starts=np.load(root/row['query_padded_window_start_path']); valid=np.load(root/row['query_valid_mask_path'])
            expected=np.asarray(item['starts'],np.int64); expected_valid=expected[:,None]+np.arange(CONTEXT)[None,:]>=int(item['pad'])
            if not np.array_equal(starts,expected) or not np.array_equal(valid,expected_valid): raise RuntimeError(f'{plane}/{s}: bank query coordinate binding drift')
    return doc,planes,banks,admitted

def batches(items: Mapping[str,Any], epoch:int, batch_size:int, seed:int):
    rng=np.random.default_rng(seed+epoch*1_000_003); rows=[]
    for s in sorted(items):
        order=rng.permutation(len(items[s]['starts'])); rows += [(s,order[x:x+batch_size]) for x in range(0,len(order),batch_size)]
    rng.shuffle(rows); return rows

def with_ema(model:nn.Module,ema:DecoderEMA,fn):
    raw={n:p.detach().clone() for n,p in model.named_parameters()}; was=model.training
    try:
        with torch.no_grad():
            for n,p in model.named_parameters(): p.copy_(ema.shadow[n].to(p.device,p.dtype))
        return fn()
    finally:
        with torch.no_grad():
            for n,p in model.named_parameters(): p.copy_(raw[n])
        model.train(was)

def score(model:nn.Module, items:Mapping[str,Any], banks:Mapping[str,TaskBank], device:torch.device, *, output_alpha:float, max_batches:int|None=None):
    model.eval(); per={}; by_date={}
    with torch.no_grad():
        for session in sorted(items):
            item=items[session]; ps=[]; ts=[]
            for offset in range(0,len(item['starts']),BATCH):
                ids=np.arange(offset,min(offset+BATCH,len(item['starts'])),dtype=np.int64); x,y,valid,pos=item_batch(item,ids,device)
                ps.append(model(x,banks[session],input_valid_mask=valid,absolute_positions=pos).float().cpu().numpy()/5.0); ts.append(y.float().cpu().numpy()/5.0)
                if max_batches is not None and len(ps)>=max_batches: break
            pred=np.concatenate(ps); truth=np.concatenate(ts); filtered=causal_output_ema(pred,output_alpha)
            per[session]={**output_centered_variance_weighted_r2(truth,filtered),'raw_prediction_sha256':hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(),
                         'output_ema':{'alpha':output_alpha,'state_0':'pred_0','session_reset':True}}
            day='-'.join(session.split('-')[1:4]);by_date.setdefault(day,[]).append((truth,filtered))
    diagnostic={day:output_centered_variance_weighted_r2(np.concatenate([x[0] for x in rows]),np.concatenate([x[1] for x in rows]))['r2'] for day,rows in sorted(by_date.items())}
    return {'per_session':per,'equal_session_r2':ext6_equal_session_selection(per) if set(per)==set(__import__('a1_m2.scoring_contract',fromlist=['EXT6']).EXT6) else float(np.mean([x['r2'] for x in per.values()])),
            'four_date_concatenated_diagnostic_r2':diagnostic,'four_date_diagnostic_not_used_for_selection':True}

def checkpoint(dest:Path, epoch:int, step:int, model:nn.Module,opt,ema, meta:Mapping[str,Any], *, smoke:bool):
    p=dest/f'epoch_{epoch:03d}.pt'
    torch.save({'schema':SCHEMA,'epoch':epoch,'step':step,'raw_state_dict':model.state_dict(),'optimizer':opt.state_dict(),'ema':ema.state_dict(),'rng':rng_state(next(model.parameters()).device),'run_meta_sha256':sha256(dest/'run_meta.json'),'smoke':smoke},p)
    return p

def fail(dest:Path, exc:Exception):
    atom(dest/'failure_receipt.json',{'schema':SCHEMA,'status':'FAILED','utc':now(),'error_type':type(exc).__name__,'error':str(exc)})

def run(args):
    cfgp=Path(args.config).resolve(); cfg=json.loads(cfgp.read_text()); arm=args.arm
    if arm not in SPECS: raise ValueError(f'unknown arm {arm}')
    if args.allow_smoke and args.max_updates is None: raise ValueError('--allow-smoke is only valid together with --max-updates')
    dest=Path(args.dest).resolve(); smoke=args.max_updates is not None
    if dest.exists() and any(dest.iterdir()) and args.resume is None: raise FileExistsError('destination must be new')
    dest.mkdir(parents=True,exist_ok=True)
    try:
        doc,planes,banks,admitted=load_data(Path(args.bank_manifest).resolve(),allow_smoke=args.allow_smoke)
        device=torch.device(args.device); torch.set_num_threads(args.cpu_threads); seed=int(cfg['seed']); torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        if not args.resume:
            shutil.copytree(HERE,dest/'source_snapshot/a1_m2')
            meta={'schema':SCHEMA,'status':'RUNNING','task':'m2','arm':arm,'seed':seed,'split':'FALCON M2 source_train / EXT6 development only','started_utc':now(),'machine':socket.gethostname(),
                  'config_path':str(cfgp),'config_sha256':sha256(cfgp),'bank_manifest_path':str(Path(args.bank_manifest).resolve()),'bank_manifest_sha256':admitted['manifest_sha256'],
                  'bank_admission':admitted,'temporal_contract':contract(arm),'output_ema':cfg['output_ema'],'official_test_used':False,'dandi_used':False,
                  'optimizer_parameter_policy':{'name':'AdamW','all_trainable_parameters':True,'weight_decay':cfg['optimization']['weight_decay'],'learned_slope_weight_decay':'same AdamW group as every temporal parameter','peak_lr':cfg['optimization']['peak_lr']},
                  'code_sha256':{str(x):sha256(x) for x in (Path(__file__),HERE/'temporal_variants.py',HERE/'scoring_contract.py',HERE/'pure_concat_bank.py')}}
            atom(dest/'run_meta.json',meta); start_epoch=1; step=0
        else:
            meta=json.loads((dest/'run_meta.json').read_text()); state=torch.load(args.resume,map_location=device,weights_only=False)
            expected_code={str(x):sha256(x) for x in (Path(__file__),HERE/'temporal_variants.py',HERE/'scoring_contract.py',HERE/'pure_concat_bank.py')}
            if state.get('schema')!=SCHEMA or state.get('smoke') or meta.get('arm')!=arm or meta.get('config_sha256')!=sha256(cfgp) or meta.get('code_sha256')!=expected_code or meta.get('bank_manifest_sha256')!=admitted['manifest_sha256'] or state.get('run_meta_sha256')!=sha256(dest/'run_meta.json'): raise RuntimeError('resume contract mismatch')
            for live in (Path(__file__),HERE/'temporal_variants.py',HERE/'scoring_contract.py',HERE/'pure_concat_bank.py'):
                snap=dest/'source_snapshot/a1_m2'/live.name
                if not snap.is_file() or sha256(snap)!=meta['code_sha256'][str(live)]: raise RuntimeError('run source snapshot drift')
            start_epoch=int(state['epoch'])+1;step=int(state['step'])
        model=build_decoder(arm,seed=seed,proj_dim=16).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimization']['peak_lr'],weight_decay=cfg['optimization']['weight_decay'],betas=tuple(cfg['optimization']['betas']),eps=cfg['optimization']['eps']); ema=DecoderEMA(model,decay=cfg['optimization']['ema_decay'])
        if args.resume: model.load_state_dict(state['raw_state_dict']);opt.load_state_dict(state['optimizer']);ema.load_state_dict(state['ema']);restore_rng(state['rng'],device)
        updates_per_epoch=len(batches(planes['source_train'],1,BATCH,seed));
        if not smoke and (updates_per_epoch!=int(cfg['optimization']['updates_per_epoch']) or int(cfg['optimization']['epochs'])!=24): raise RuntimeError('frozen source full-pass or epoch contract drift')
        epochs=1 if smoke else int(cfg['optimization']['epochs']); total=int(cfg['optimization']['epochs'])*updates_per_epoch; curve_path=dest/'ext6_epoch_scores.json'; curve=json.loads(curve_path.read_text()).get('per_epoch',{}) if curve_path.is_file() else {}
        for epoch in range(start_epoch,epochs+1):
            model.train(); losses=[]
            for bi,(session,ids) in enumerate(batches(planes['source_train'],epoch,BATCH,seed)):
                x,y,valid,pos=item_batch(planes['source_train'][session],ids,device); keep=whole_unit_dropout(torch.ones(96,dtype=torch.bool),p=float(cfg['optimization']['unit_dropout']),generator=torch.Generator(device='cpu').manual_seed(seed+epoch*100000+bi)).to(device).expand(x.shape[0],-1)
                step+=1; lr=warmup_cosine_lr(step,total_steps=total,warmup_steps=int(cfg['optimization']['warmup_updates']),peak=float(cfg['optimization']['peak_lr']),min_factor=float(cfg['optimization']['cosine_min_factor']))
                for group in opt.param_groups: group['lr']=lr
                opt.zero_grad(set_to_none=True)
                with (torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()): loss=nn.functional.mse_loss(model(x,banks['source_train'][session],dropout_keep=keep,input_valid_mask=valid,absolute_positions=pos).float(),y)
                loss.backward(); grad=float(nn.utils.clip_grad_norm_(model.parameters(),float(cfg['optimization']['grad_clip_norm']),error_if_nonfinite=True));opt.step();ema.update_after_step(model);losses.append(float(loss.detach().cpu()))
                if step % 100 == 0: atom(dest/'heartbeat.json',{'status':'TRAINING','utc':now(),'epoch':epoch,'step':step,'train_mse_recent':float(np.mean(losses[-100:])),'lr':lr,'grad_norm':grad})
                if args.max_updates and step>=args.max_updates: break
            ck=checkpoint(dest,epoch,step,model,opt,ema,meta,smoke=smoke)
            report=with_ema(model,ema,lambda:score(model,planes['ext6_development'],banks['ext6_development'],device,output_alpha=float(cfg['output_ema']['alpha']),max_batches=1 if smoke else None))
            row={'epoch':epoch,'step':step,'train_mse':float(np.mean(losses)),'grad_norm':grad,'checkpoint_sha256':sha256(ck),'ext6_ema_output':report,'smoke':smoke};append(dest/'metrics.jsonl',row);atom(dest/'heartbeat.json',{'status':'SMOKE' if smoke else 'TRAINING','utc':now(),**row});curve[str(epoch)]={'checkpoint_sha256':sha256(ck),**report};atom(curve_path,{'schema':SCHEMA,'split':'EXT6 development only','per_epoch':curve,'four_date_diagnostic':'not used for selection'})
            if smoke: break
        if smoke:
            atom(dest/'smoke_receipt.json',{'schema':SCHEMA,'status':'COMPLETED_SMOKE','run_meta_sha256':sha256(dest/'run_meta.json'),'bank_manifest_sha256':admitted['manifest_sha256'],'official_test_used':False,'dandi_used':False}); return {'status':'COMPLETED_SMOKE'}
        # score every saved epoch on complete EXT6 in a separate resumable stage.
        meta.update(status='TRAIN_COMPLETED',ended_utc=now());atom(dest/'run_meta.json',meta);atom(dest/'train_receipt.json',{'schema':SCHEMA,'status':'COMPLETED','epochs':epochs,'updates_per_epoch':updates_per_epoch,'run_meta_sha256':sha256(dest/'run_meta.json'),'bank_manifest_sha256':admitted['manifest_sha256'],'official_test_used':False,'dandi_used':False});return {'status':'TRAIN_COMPLETED'}
    except Exception as exc:
        fail(dest,exc); raise

def score_stage(args):
    dest=Path(args.dest).resolve(); meta=json.loads((dest/'run_meta.json').read_text()); cfg=json.loads(Path(args.config).read_text());
    if meta.get('status')!='TRAIN_COMPLETED' or not (dest/'train_receipt.json').is_file() or meta.get('arm')!=args.arm or meta.get('config_sha256')!=sha256(Path(args.config)): raise RuntimeError('score requires matching completed formal train/config/arm')
    doc,planes,banks,admitted=load_data(Path(args.bank_manifest).resolve(),allow_smoke=False)
    if meta.get('bank_manifest_sha256')!=admitted['manifest_sha256']: raise RuntimeError('bank manifest drift before scoring')
    curve=json.loads((dest/'ext6_epoch_scores.json').read_text()); rows={int(k):v['per_session'] for k,v in curve.get('per_epoch',{}).items()}
    if set(rows)!=set(range(1,int(cfg['optimization']['epochs'])+1)): raise RuntimeError('incomplete full EXT6 epoch score curve')
    selected,score_value=earliest_finite_maximum(rows); device=torch.device(args.device);torch.set_num_threads(args.cpu_threads)
    state=torch.load(dest/f'epoch_{selected:03d}.pt',map_location=device,weights_only=False); model=build_decoder(args.arm,seed=int(cfg['seed'])).to(device);model.load_state_dict(state['raw_state_dict']);ema=DecoderEMA(model,decay=float(cfg['optimization']['ema_decay']));ema.load_state_dict(state['ema'])
    replay=with_ema(model,ema,lambda:score(model,planes['ext6_development'],banks['ext6_development'],device,output_alpha=float(cfg['output_ema']['alpha']))); replay_rows=replay['per_session']
    if any(replay_rows[s]['prediction_sha256']!=rows[selected][s]['prediction_sha256'] or replay_rows[s]['truth_sha256']!=rows[selected][s]['truth_sha256'] for s in replay_rows): raise RuntimeError('selected checkpoint replay does not reproduce full epoch score receipt')
    continuity={s:{'first':int(np.asarray(planes['ext6_development'][s]['starts'])[0]),'last':int(np.asarray(planes['ext6_development'][s]['starts'])[-1]),'count':int(len(planes['ext6_development'][s]['starts'])),'gaps':int(np.count_nonzero(np.diff(np.asarray(planes['ext6_development'][s]['starts']))!=1))} for s in sorted(planes['ext6_development'])}
    receipt={'schema':SCHEMA,'status':'COMPLETED','split':'FALCON M2 EXT6 development only','selection':{'rule':'output EMA .45; unweighted six-session centered variance-weighted R2; earliest finite maximum','selected_epoch':selected,'selected_score':score_value},'per_epoch':rows,'selected_checkpoint_replay':{'checkpoint_sha256':sha256(dest/f'epoch_{selected:03d}.pt'),'matches_epoch_receipt':True},'query_continuity_for_output_ema':continuity,'query_truth_binding':'each per-session row contains prediction/truth hashes and SSE/TSS','bank_manifest_sha256':admitted['manifest_sha256'],'official_test_used':False,'dandi_used':False};atom(dest/'score_receipt.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--bank-manifest',required=True);p.add_argument('--arm',choices=tuple(SPECS),required=True);p.add_argument('--dest',required=True);p.add_argument('--stage',choices=('train','score'),default='train');p.add_argument('--device',default='cuda:0');p.add_argument('--resume');p.add_argument('--allow-smoke',action='store_true');p.add_argument('--max-updates',type=int);p.add_argument('--cpu-threads',type=int,default=4);a=p.parse_args()
    print(json.dumps(score_stage(a) if a.stage=='score' else run(a),indent=2,sort_keys=True))
if __name__=='__main__': main()
