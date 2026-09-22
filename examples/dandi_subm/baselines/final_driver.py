#!/usr/bin/env python3
"""Capability-gated final scorer for the already sealed historical selection.

This separate adapter leaves `run_historical.py`'s development-selection
snapshot intact.  It only adds final-cache/capability routing and records both
the frozen selection code hash and this adapter's hash in its receipt.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
import run_historical as H
from apst.dandi_subm.final_access import FinalAccess
from apst.dandi_subm.subm_data import load_cached_session

def verify_receipt(path, cache, cap):
 receipt=json.loads(path.read_text())
 if receipt.get('schema')!='dandi688_subm_final_cache_v1' or receipt.get('protocol')!=cap.doc.get('protocol') or receipt.get('seal_sha256')!=H.digest(cap.path):raise ValueError('final-cache receipt/seal mismatch')
 if set(receipt.get('sessions',{}))!=set(H.FINAL):raise ValueError('final-cache receipt roster mismatch')
 for sid in H.FINAL:
  target=cache/f'{sid}.sua.npz';entry=receipt['sessions'][sid]
  if Path(str(entry.get('file',''))).resolve()!=target.resolve() or entry.get('sha256')!=H.digest(target):raise ValueError(f'final-cache binding mismatch: {sid}')

def load_final(cache,sid,cap):
 item=load_cached_session(cache/f'{sid}.sua.npz',final_access=cap)
 if (item.session_id,item.split,item.representation)!=(sid,'final','sua'):raise ValueError('final cache identity mismatch')
 return H.Record(sid,item.neural,item.velocity,item.support_indices,item.carrier_indices,item.query_indices,str((cache/f'{sid}.sua.npz').resolve()),H.digest(cache/f'{sid}.sua.npz')),item.metadata['array_sha256']

def score_detail(record,pred,array_hashes):
 truth=np.asarray(record.velocity[record.query_indices],np.float64);pred=np.asarray(pred,np.float64)
 if pred.shape!=truth.shape or not np.isfinite(pred).all():raise ValueError('invalid final prediction')
 sse=((truth-pred)**2).sum(0);sst=((truth-truth.mean(0))**2).sum(0)
 if np.any(sst<=0):raise ValueError('degenerate final truth')
 return {'session_id':record.session_id,'n_queries':len(truth),'r2':float(1-sse.sum()/sst.sum()),'r2_per_output':(1-sse/sst).tolist(),'query_indices_sha256':array_hashes['query_indices'],'velocity_sha256':array_hashes['velocity']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--seal',type=Path,required=True);p.add_argument('--final-cache',type=Path,required=True);p.add_argument('--final-receipt',type=Path,required=True);a=p.parse_args()
 selection_path=H.OUT/'selection.json';claim=H.OUT/'.final_claim';selection=json.loads(selection_path.read_text())
 if selection.get('status') not in {'DEV_SELECTED','DEV_SELECTED_WITH_UNAVAILABLE'} or selection.get('final_loaded') is not False:raise ValueError('not a sealed no-final baseline selection')
 # Every selected dev artifact remains byte-bound; this detects a replaced
 # selection/model while allowing this separate final adapter to evolve.
 for method,artifact in selection.get('artifacts',{}).items():
  for name,key in ((f'{method}.dev_models.pkl','models_sha256'),(f'{method}.dev_predictions.npz','predictions_sha256')):
   path=H.OUT/name
   if not path.is_file() or H.digest(path)!=artifact.get(key):raise ValueError(f'development artifact drift: {method}/{name}')
 cap=FinalAccess.from_manifest(a.seal.resolve());verify_receipt(a.final_receipt.resolve(),a.final_cache.resolve(),cap)
 if claim.exists():raise PermissionError('baseline final scorer already claimed')
 claim.open('x').write(json.dumps({'selection_sha256':H.digest(selection_path),'selection_code_sha256':selection.get('code_sha256'),'adapter_code_sha256':H.digest(Path(__file__).resolve()),'seal_sha256':H.digest(a.seal),'final_cache_receipt_sha256':H.digest(a.final_receipt)},sort_keys=True)+'\n')
 loaded=[load_final(a.final_cache.resolve(),sid,cap) for sid in H.FINAL];records=[x[0] for x in loaded];hashes={x[0].session_id:x[1] for x in loaded};results={}
 for method,choice in selection['selected'].items():
  if choice.get('status')=='UNAVAILABLE':results[method]=choice;continue
  cfg=choice['config'];models={r.session_id:H.fit(r,cfg) for r in records};pred={r.session_id:H.predict(models[r.session_id],r) for r in records}
  path=H.OUT/f'{method}.final_predictions.npz';np.savez_compressed(path,**pred)
  sessions=[score_detail(r,pred[r.session_id],hashes[r.session_id]) for r in records]
  results[method]={'status':'SCORED','config':cfg,'metrics':{'metric':'physical_velocity_variance_weighted_equal_session_mean','mean_r2':float(np.mean([x['r2'] for x in sessions])),'n_sessions':len(sessions),'sessions':sessions},'predictions_sha256':H.digest(path)}
 out={'schema':'dandi688_subm2015_historical_sua_wf_final_v3','status':'FINAL_SCORED','selection_sha256':H.digest(selection_path),'selection_code_sha256':selection.get('code_sha256'),'final_adapter_code_sha256':H.digest(Path(__file__).resolve()),'seal_sha256':H.digest(a.seal),'final_cache_receipt_sha256':H.digest(a.final_receipt),'final_binding':H.binding(records),'results':results,'supervision':selection['supervision'],'final_sessions_opened':len(records)}
 H.atomic(H.OUT/'final_score_receipt.json',out);print(json.dumps({k:v.get('metrics',{}).get('mean_r2') for k,v in results.items()},indent=2))
if __name__=='__main__':main()
