"""Score a preselected Static-F0 checkpoint on a sealed Sub-C final cache."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from . import data,protocol,training as T
from .common import aggregate_scores,score_predictions,sha256
from .final_access import FinalAccess
from .static import SCHEMA,StaticEncoder

def _read(p): return json.loads(Path(p).read_text())
def _write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def score(seal,cache_receipt,static_run,dest,device='cpu'):
 seal=Path(seal).resolve();cap=FinalAccess.from_manifest(seal);receipt=_read(cache_receipt);cache=Path(cache_receipt).resolve().parent/'final_cache'
 if receipt.get('schema')!='apst_dandi_subc_final_cache_v1' or receipt.get('seal_sha256')!=sha256(seal) or set(receipt.get('sessions',{}))!=set(protocol.FINAL_SESSIONS):raise ValueError('core final cache receipt mismatch')
 run=Path(static_run).resolve();meta=_read(run/'run_meta.json');ck=run/meta.get('selected_checkpoint','');stats=run/'source_stats.json'
 if meta.get('status')!='COMPLETED' or meta.get('representation')!='sua' or meta.get('selected_checkpoint_sha256')!=sha256(ck) or not stats.is_file():raise ValueError('static run is not a selected formal SUA run')
 dest=Path(dest).resolve()
 if dest.exists():raise FileExistsError(dest)
 binding={'sealsha':sha256(seal),'static_run_meta':sha256(run/'run_meta.json'),'checkpoint':sha256(ck),'source_stats':sha256(stats)};_write(dest/'selection_binding.json',binding)
 payload=torch.load(ck,map_location='cpu',weights_only=False)
 if payload.get('schema')!=SCHEMA+'_checkpoint' or sha256(stats)!=binding['source_stats'] or payload.get('source_stats')!=_read(stats):raise ValueError('static checkpoint/source-stat binding mismatch')
 model=T._build('act_only','F0',42,'pretrain',None);model.encoder=StaticEncoder(T._units(model),50,42);model.method='act_only';model.load_state_dict(payload['model_state'],strict=True);model.to(device).eval();state=payload['source_stats'];rows=[]
 for sid in protocol.FINAL_SESSIONS:
  entry=receipt['sessions'][sid];path=cache/f'{sid}.sua.npz'
  if Path(entry['file']).resolve()!=path.resolve() or entry['sha256']!=sha256(path):raise ValueError('cache binding mismatch')
  rec=data.load_cached_session(path,final_access=cap);pred,diag=T.predict_record(model,rec,state);rows.append({**score_predictions(rec,pred),'seed':42,'split':'final','status':'SCORED','source_path':str(ck),'source_sha256':sha256(ck),'diagnostics':diag})
 _write(dest/'receipt.json',{'schema':'apst_dandi_subc_static_final_v1','status':'FINAL_SCORED','binding':binding,'metrics':aggregate_scores(rows),'rows':rows})
 with (dest/'final_rows.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['session_id','n_queries','r2','r2_per_output','query_indices_sha256','velocity_sha256','seed','split','status','source_path','source_sha256']);w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in rows])
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--seal',required=True);p.add_argument('--final-cache-receipt',required=True);p.add_argument('--static-run',required=True);p.add_argument('--dest',required=True);p.add_argument('--device',default='cpu');a=p.parse_args(argv);score(a.seal,a.final_cache_receipt,a.static_run,a.dest,a.device)
if __name__=='__main__':main()
