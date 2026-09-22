#!/usr/bin/env python3
"""One-use final evaluator for Static-F0 after the shared campaign seal.

The shared final gate is owned by the training workflow.  This evaluator only
consumes its validated capability; it cannot construct or bypass final access.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
from apst.dandi_subm import protocol, training as T
from apst.dandi_subm.subm_common import aggregate_scores, score_predictions, sha256
from apst.dandi_subm.subm_data import load_cached_session
CAMPAIGN=protocol.output_root()

class StaticEncoder(nn.Module):
 def __init__(self, units:int=100, e0_dim:int=50, seed:int=42):
  super().__init__()
  with torch.random.fork_rng(devices=[]): torch.manual_seed(seed+0x53544154); self.static_e0=nn.Parameter(torch.randn(units,e0_dim)*.02)
 def forward(self,activity,profile=None): return self.static_e0
 def encode(self,activity,profile=None): return self.static_e0

def digest(path:Path)->str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path:Path,value:dict)->None:
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seal',type=Path,required=True);p.add_argument('--final-cache',type=Path,required=True);p.add_argument('--final-receipt',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 run=CAMPAIGN/'static'/'runs'/'static_f0_s42';receipt_path=run/'static_receipt.json';selection_path=run/'selection.json';claim=run/'.final_claim'
 if not receipt_path.is_file() or not selection_path.is_file():raise FileNotFoundError('formal Static-F0 receipt/selection required')
 static=json.loads(receipt_path.read_text());sel=json.loads(selection_path.read_text())
 if static.get('status')!='DEV_SELECTED' or static.get('method')!='Static-F0 (source-trained global E0)' or static.get('final_sessions_opened')!=0:raise ValueError('Static-F0 is not a formal no-final selection')
 if sel.get('rule')!='earliest_max_equal_session_dev_r2' or sel.get('final_sessions_opened')!=0:raise ValueError('Static-F0 selection contract mismatch')
 ck=Path(sel['checkpoint']);
 if not ck.is_file() or sha256(ck)!=sel.get('checkpoint_sha256'):raise ValueError('selected checkpoint binding mismatch')
 # This import must be supplied by the shared finalization owner.  It validates
 # the manifest, selected neural checkpoints, fixed source stats, and creates
 # the only capability accepted by final cached-session loading.
 from apst.dandi_subm.final_access import FinalAccess
 cap=FinalAccess.from_manifest(Path(a.seal).resolve())
 cache_receipt=json.loads(a.final_receipt.read_text())
 if cache_receipt.get('schema')!='dandi688_subm_final_cache_v1' or cache_receipt.get('protocol')!=cap.doc.get('protocol') or cache_receipt.get('seal_sha256')!=digest(a.seal):raise ValueError('final-cache receipt/seal mismatch')
 if set(cache_receipt.get('sessions',{}))!=set(protocol.FINAL_SESSIONS):raise ValueError('final-cache receipt roster mismatch')
 for sid in protocol.FINAL_SESSIONS:
  entry=cache_receipt['sessions'][sid];path=a.final_cache/f'{sid}.sua.npz'
  if Path(str(entry.get('file',''))).resolve()!=path.resolve() or entry.get('sha256')!=sha256(path):raise ValueError(f'final-cache binding mismatch: {sid}')
 if claim.exists():raise PermissionError('Static-F0 final scorer already claimed')
 claim.open('x').write(digest(a.seal)+'\n')
 stats_path=run/'source_stats.json';payload=torch.load(ck,map_location='cpu',weights_only=False)
 if not stats_path.is_file():raise FileNotFoundError(stats_path)
 frozen_stats=json.loads(stats_path.read_text())
 if (payload.get('source_stats') is None or payload['source_stats'].get('sha256')!=frozen_stats.get('sha256')
     or payload.get('metadata',{}).get('source_stats_sha256')!=frozen_stats.get('sha256')):raise ValueError('checkpoint/source-stat binding mismatch')
 original=T._build
 try:
  def build(method,temporal,seed,stage,encoder_state):
   model=original(method,temporal,seed,stage,encoder_state);model.encoder=StaticEncoder(T._units(model),50,seed);model.method='act_only';return model
  T._build=build
  model=T._build('act_only','F0',42,'pretrain',None).to(a.device);model.load_state_dict(payload['model_state'],strict=True);model.eval()
 finally:T._build=original
 results=[];pred_dir=run/'final_predictions';pred_dir.mkdir()
 for sid in protocol.FINAL_SESSIONS:
  record=load_cached_session(a.final_cache/f'{sid}.sua.npz',final_access=cap)
  if (record.session_id,record.split,record.representation)!=(sid,'final','sua'):raise ValueError('final cache identity mismatch')
  pred,diag=T.predict_record(model,record,payload['source_stats']);np.savez_compressed(pred_dir/f'{sid}.npz',prediction=pred,query_indices=record.query_indices,truth=record.velocity[record.query_indices])
  results.append({**score_predictions(record,pred),**diag,'prediction_file':str((pred_dir/f'{sid}.npz').resolve()),'prediction_sha256':sha256(pred_dir/f'{sid}.npz')})
 out={'schema':'dandi688_subm2015_static_f0_final_v2','status':'FINAL_SCORED','method':'Static-F0 (source-trained global E0)','seal_sha256':digest(a.seal),'final_cache_receipt_sha256':digest(a.final_receipt),'static_receipt_sha256':digest(receipt_path),'selection_sha256':digest(selection_path),'checkpoint_sha256':sha256(ck),'source_stats_sha256':frozen_stats['sha256'],'metrics':aggregate_scores(results),'final_sessions_opened':len(results)}
 atomic(run/'final_score_receipt.json',out);print(json.dumps({'mean_r2':out['metrics']['mean_r2'],'sessions':len(results)},indent=2))
if __name__=='__main__':main()
