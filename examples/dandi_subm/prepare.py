#!/usr/bin/env python3
import json, hashlib
from pathlib import Path
from apst.dandi_subm import protocol
from apst.dandi_subm.subm_data import load_pair, save_session
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 dest=protocol.output_root()/'prepared_sua'; dest.mkdir(parents=True,exist_ok=True)
 rows={}
 for split,ids,purpose in [('train',protocol.TRAIN_SESSIONS,'source'),('dev',protocol.DEV_SESSIONS,'development')]:
  for sid in ids:
   rec=load_pair(sid,purpose=purpose)['sua']; path=dest/f'{sid}.sua.npz'; save_session(rec,path)
   rows[sid]={'split':split,'file':path.name,'sha256':sha(path),'channels':int(rec.neural.shape[1]),'legal_trials':len(rec.metadata['raw_trial_rows']),'direction_design_rank':rec.metadata['direction_design_rank'],'finite_direction_count':rec.metadata['finite_direction_count'],'m1_only':True}
 # Final is metadata/schema audited only; raw neural/query/velocity are never materialized here.
 final=[]
 for sid in protocol.FINAL_SESSIONS:
  from pynwb import NWBHDF5IO
  import numpy as np
  path=protocol.raw_root()/f'{sid}_behavior+ecephys.nwb'
  with NWBHDF5IO(str(path),'r',load_namespaces=True) as io:
   nwb=io.read(); trials=nwb.intervals['trials'].to_dataframe(); units=nwb.units.to_dataframe()
   valid=trials[trials['result']=='R']; angles=np.asarray(valid.iloc[:33]['target_dir'],dtype=float); finite=angles[np.isfinite(angles)]; rank=int(np.linalg.matrix_rank(np.stack([np.ones(len(finite)),np.cos(finite),np.sin(finite)],axis=1)))
   m1=sum(str(getattr(nwb.electrodes.to_dataframe().loc[int(getattr(r,'electrodes').index[0])],'location'))=='Primary Motor Cortex' for _,r in units.iterrows() if getattr(r,'electrodes',None) is not None and len(getattr(r,'electrodes').index)==1)
  final.append({'session_id':sid,'m1_units':m1,'rewarded_trials_unfiltered':len(valid),'first33_rewarded_metadata_direction_design_rank':rank,'first33_rewarded_metadata_finite_direction_count':len(finite),'cached':False,'scored':False})
 out={'schema':'dandi688_subm_prepared_sua_v1','protocol':protocol.protocol_dict(),'sessions':rows,'final_schema_rank_audit':final,'final_sessions_opened_for_scoring':0,'sua_only':True}
 (dest/'prepared_receipt.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'status':'PASS','source_dev':len(rows),'final_audited':len(final)}))
if __name__=='__main__':main()
