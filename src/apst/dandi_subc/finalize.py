"""Seal formal Sub-C selections and score the one capability-gated final pass."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from . import data,protocol,training as T
from .common import aggregate_scores,score_predictions,sha256
from .final_access import FinalAccess,SEAL_SCHEMA,SEAL_STATUS,canonical_payload_sha256

def _read(p): return json.loads(Path(p).read_text())
def _atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def _runs(values):
 out={}
 for value in values:
  name,raw=value.split('=',1); path=Path(raw).resolve()
  if not name or name in out: raise ValueError('runs must be unique NAME=PATH')
  out[name]=path
 return out
def seal(dest,runs):
 dest=Path(dest).resolve()
 if dest.exists(): raise FileExistsError(dest)
 arts={}; cells={}
 for name,run in _runs(runs).items():
  selection,receipt,stats=run/'selection.json',run/'train_receipt.json',run/'source_stats.json'
  if not all(p.is_file() for p in (selection,receipt,stats)): raise ValueError(f'{name}: incomplete formal run')
  s,t=_read(selection),_read(receipt); ck=Path(s.get('checkpoint','')).resolve()
  if s.get('status')!='FORMAL' or s.get('rule')!='earliest_max_equal_session_dev_r2' or s.get('dev_sessions')!=list(protocol.DEV_SESSIONS) or t.get('status')!='FORMAL' or t.get('completed') is not True or t.get('final_sessions_opened')!=0 or not ck.is_file(): raise ValueError(f'{name}: non-formal selection')
  if s.get('checkpoint_sha256')!=sha256(ck) or t.get('source_stats_sha256')!=_read(stats).get('sha256'): raise ValueError(f'{name}: artifact binding')
  paths={'checkpoint':ck,'source_stats':stats.resolve(),'selection':selection.resolve(),'train_receipt':receipt.resolve()}
  for p in paths.values(): arts[str(p)]=sha256(p)
  cells[name]={'status':'SELECTED','representation':'sua','method':t.get('method'),'temporal':t.get('temporal'),'seed':t.get('seed'),'checkpoint_sha256':sha256(ck),'condition':['intact','profile_shuffle'] if t.get('method')=='dual_site' else ['intact'],**{k:str(v) for k,v in paths.items()}}
 payload={'schema':SEAL_SCHEMA,'status':SEAL_STATUS,'model_schema':T.SCHEMA,'protocol':protocol.protocol_dict(),'authorized_sessions':list(protocol.FINAL_SESSIONS),'artifacts':arts,'cell_selections':cells}
 payload['sha256']=canonical_payload_sha256(payload);_atomic(dest,payload);return dest
def score(seal_path,dest,device='cpu'):
 cap=FinalAccess.from_manifest(seal_path); doc=_read(seal_path);dest=Path(dest).resolve()
 if dest.exists(): raise FileExistsError(dest)
 dest.mkdir(parents=True); cache=dest/'final_cache';cache.mkdir(); records=[];sessions={}
 for sid in protocol.FINAL_SESSIONS:
  rec=data.load_pair(sid,purpose='final',final_access=cap)['sua']; p=cache/f'{sid}.sua.npz';data.save_session(rec,p);records.append(rec);sessions[sid]={'file':str(p),'sha256':sha256(p)}
 _atomic(dest/'final_cache_receipt.json',{'schema':'apst_dandi_subc_final_cache_v1','sessions':sessions,'seal_sha256':sha256(seal_path)})
 rows=[]
 for name,c in doc['cell_selections'].items():
  model,stats,_=T.load_checkpoint(c['checkpoint'],device=device)
  for condition in c['condition']:
   for rec in records:
    pred,diag=T.predict_record(model,rec,stats,condition=condition,shuffle_seed=42);rows.append({**score_predictions(rec,pred),'cell':name,'condition':condition,'seed':c['seed'],'split':'final','status':'SCORED','source_path':c['checkpoint'],'source_sha256':c['checkpoint_sha256'],'diagnostics':diag})
 _atomic(dest/'rows.json',rows)
 with (dest/'final_rows.csv').open('w',newline='') as f:
  keys=['cell','condition','session_id','n_queries','r2','r2_per_output','query_indices_sha256','velocity_sha256','seed','split','status','source_path','source_sha256'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r.get(k) for k in keys} for r in rows])
 _atomic(dest/'receipt.json',{'schema':'apst_dandi_subc_final_score_v1','status':'FINAL_SCORED','seal_sha256':sha256(seal_path),'csv_sha256':sha256(dest/'final_rows.csv'),'metrics':{f'{n}:{c}':aggregate_scores([r for r in rows if r['cell']==n and r['condition']==c]) for n in doc['cell_selections'] for c in doc['cell_selections'][n]['condition']}})
def main(argv=None):
 p=argparse.ArgumentParser();q=p.add_subparsers(dest='cmd',required=True);a=q.add_parser('seal');a.add_argument('--dest',type=Path,required=True);a.add_argument('--run',action='append',required=True);b=q.add_parser('score');b.add_argument('--seal',type=Path,required=True);b.add_argument('--dest',type=Path,required=True);b.add_argument('--device',default='cpu');x=p.parse_args(argv);print(seal(x.dest,x.run) if x.cmd=='seal' else score(x.seal,x.dest,x.device))
if __name__=='__main__':main()
