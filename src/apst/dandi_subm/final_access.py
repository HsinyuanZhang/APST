"""Validated capability for one sealed sub-M final pass."""
import hashlib,json
from pathlib import Path
from apst.dandi_subm import protocol
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class FinalAccess:
 def __init__(self,path,doc):self.path=Path(path);self.doc=doc
 @classmethod
 def from_manifest(cls,path):
  p=Path(path); d=json.loads(p.read_text())
  if d.get('schema')!='dandi688_subm_final_seal_v1' or d.get('status')!='SEALED' or d.get('protocol')!=protocol.protocol_dict() or d.get('authorized_sessions')!=list(protocol.FINAL_SESSIONS) or not isinstance(d.get('artifacts'),dict) or not d['artifacts'] or not isinstance(d.get('cells'),dict) or set(d['cells'])!={'sua_dual_site_F0_s42','sua_act_only_F0_s42'}:raise PermissionError('invalid final seal')
  for raw,h in d.get('artifacts',{}).items():
   if not Path(raw).is_file() or sha(raw)!=h:raise PermissionError('sealed artifact drift')
  return cls(p,d)
 def authorize(self,sid):
  if sid not in protocol.FINAL_SESSIONS:raise PermissionError('outside final roster')
