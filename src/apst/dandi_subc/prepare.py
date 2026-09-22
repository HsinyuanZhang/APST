"""Materialize source/development SUA cache without opening final recordings."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from . import protocol
from .data import load_pair
from .data import save_session

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(dest, raw_root=None):
 dest=Path(dest).resolve()
 if dest.exists() and any(dest.iterdir()): raise FileExistsError('prepare destination must be fresh')
 dest.mkdir(parents=True,exist_ok=False); rows={}
 for split, ids in [('train',protocol.TRAIN_SESSIONS),('dev',protocol.DEV_SESSIONS)]:
  for sid in ids:
   purpose = {'train': 'source', 'dev': 'development'}[split]
   rec=load_pair(sid,raw_root=(protocol.DEFAULT_RAW_ROOT if raw_root is None else raw_root),purpose=purpose)['sua']; path=dest/f'{sid}.sua.npz';save_session(rec,path)
   rows[sid]={'split':split,'file':path.name,'sha256':sha(path),'channels':int(rec.neural.shape[1]),'m1_only':True}
 out={'schema':'dandi688_subc_prepared_sua_v1','protocol':protocol.protocol_dict(),'sessions':rows,'final_sessions_opened_for_scoring':0,'sua_only':True}
 (dest/'prepared_receipt.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');return out
