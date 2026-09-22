#!/usr/bin/env python3
import csv,hashlib,json
from pathlib import Path
from apst.dandi_subm import protocol
from apst.dandi_subm.final_access import FinalAccess
from apst.dandi_subm.subm_data import save_session
from apst.dandi_subm.final_data import load_final_pair,load_final_cached
from apst.dandi_subm.subm_common import sha256,score_predictions,aggregate_scores
from apst.dandi_subm.training import load_checkpoint,predict_record
ROOT=protocol.output_root()
def seal():
 artifacts={}; cells={}
 for name in ('sua_dual_site_F0_s42','sua_act_only_F0_s42'):
  run=ROOT/json.loads((ROOT/'configs'/f'{name}.json').read_text())['run_dir']; sel=run/'selection.json'; rec=run/'train_receipt.json'; stats=run/'source_stats.json'
  if not all(x.is_file() for x in (sel,rec,stats)):raise ValueError('incomplete training: '+name)
  s=json.loads(sel.read_text()); t=json.loads(rec.read_text()); st=json.loads(stats.read_text()); ck=Path(s['checkpoint'])
  if t.get('status')!='FORMAL' or t.get('completed') is not True or t.get('final_sessions_opened')!=0 or s.get('status')!='FORMAL' or s.get('rule')!='earliest_max_equal_session_dev_r2' or s.get('dev_sessions')!=list(protocol.DEV_SESSIONS):raise ValueError('nonformal selection '+name)
  if not ck.is_file() or sha256(ck)!=s['checkpoint_sha256'] or t.get('source_stats_sha256')!=st.get('sha256') or t.get('source_sessions')!=list(protocol.TRAIN_SESSIONS):raise ValueError('bad selection/stat binding '+name)
  cells[name]={'checkpoint':str(ck),'checkpoint_sha256':sha256(ck),'conditions':['intact','profile_shuffle'] if 'dual' in name else ['intact']}
  for p in (sel,rec,stats,ck):artifacts[str(p.resolve())]=sha256(p)
 for p in (ROOT/'baselines/results/selection.json',ROOT/'static/runs/static_f0_s42/selection.json',ROOT/'static/runs/static_f0_s42/static_receipt.json',ROOT/'static/runs/static_f0_s42/source_stats.json'):
  if not p.is_file():raise ValueError('missing locked baseline/static selection')
  artifacts[str(p.resolve())]=sha256(p)
 static_sel=json.loads((ROOT/'static/runs/static_f0_s42/selection.json').read_text()); static_ck=Path(static_sel.get('checkpoint',''))
 if not static_ck.is_file() or sha256(static_ck)!=static_sel.get('checkpoint_sha256'):raise ValueError('bad static checkpoint binding')
 artifacts[str(static_ck.resolve())]=sha256(static_ck)
 d={'schema':'dandi688_subm_final_seal_v1','status':'SEALED','protocol':protocol.protocol_dict(),'authorized_sessions':list(protocol.FINAL_SESSIONS),'cells':cells,'artifacts':artifacts,'final_sessions_opened':0}; out=ROOT/'final';out.mkdir(exist_ok=True); path=out/'selection_seal.json'
 if path.exists():raise FileExistsError('final seal already exists')
 path.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n');return path
def score(sealpath,device='cuda:0'):
 cap=FinalAccess.from_manifest(sealpath); d=cap.doc; out=ROOT/'final/apst_act_scores';out.mkdir(exist_ok=False); cache=ROOT/'final/final_cache';cache.mkdir(exist_ok=True); rows=[]; cache_sessions={}
 for sid in protocol.FINAL_SESSIONS:
  cp=cache/f'{sid}.sua.npz'
  if cp.is_file(): rec=load_final_cached(cp,cap); reused=True
  else: rec=load_final_pair(sid,cap)['sua'];save_session(rec,cp);reused=False
  if rec.session_id!=sid or rec.representation!='sua':raise ValueError('final cache identity mismatch')
  cache_sessions[sid]={'file':str(cp.resolve()),'sha256':sha256(cp),'array_sha256':rec.metadata['array_sha256'],'reused_after_failure':reused}
  for name,e in d['cells'].items():
   model,stats,_=load_checkpoint(Path(e['checkpoint']),device=device)
   for cond in e['conditions']:
    pred,diag=predict_record(model,rec,stats,condition=cond,shuffle_seed=42); m=score_predictions(rec,pred); npz=out/f'{name}.{cond}.{sid}.npz'; __import__('numpy').savez_compressed(npz,prediction=pred,query_indices=rec.query_indices,truth=rec.velocity[rec.query_indices]); rows.append({**m,'cell':name,'session_id':sid,'condition':cond,'split':'final','seed':42,'status':'SCORED','source_path':e['checkpoint'],'source_sha256':e['checkpoint_sha256'],'prediction_path':str(npz),'prediction_sha256':sha256(npz),'diagnostics':diag})
 for cell in d['cells']:
  for cond in d['cells'][cell]['conditions']:
   subset=[x for x in rows if x['cell']==cell and x['condition']==cond]
   mean=aggregate_scores(subset)['mean_r2']
   for x in subset:x['equal_session_r2']=mean
 (out/'rows.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n'); fields=sorted({k for r in rows for k in r if k!='diagnostics'})
 with (out/'final_rows.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])
 cache_receipt={'schema':'dandi688_subm_final_cache_v1','protocol':protocol.protocol_dict(),'sessions':cache_sessions,'final_sessions_opened':3,'seal_sha256':sha256(sealpath)}; (ROOT/'final/final_cache_receipt.json').write_text(json.dumps(cache_receipt,indent=2,sort_keys=True)+'\n')
 (out/'receipt.json').write_text(json.dumps({'schema':'dandi688_subm_apst_act_final_v1','status':'FINAL_SCORED','rows':len(rows),'final_sessions_opened':3,'seal':str(sealpath),'seal_sha256':sha256(sealpath),'csv':str(out/'final_rows.csv'),'csv_sha256':sha256(out/'final_rows.csv'),'final_cache_receipt':str(ROOT/'final/final_cache_receipt.json'),'final_cache_receipt_sha256':sha256(ROOT/'final/final_cache_receipt.json'),'shuffle_contract':{'dual_site':['intact','profile_shuffle'],'seed':42,'joint_permutation':'internal_only'}},indent=2)+'\n')
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('cmd',choices=['seal','score']);p.add_argument('--seal');p.add_argument('--device',default='cuda:0');a=p.parse_args();print(seal() if a.cmd=='seal' else score(Path(a.seal),a.device))
