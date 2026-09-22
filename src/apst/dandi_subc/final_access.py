"""Release-specific capability for one sealed Sub-C final-data phase."""
from __future__ import annotations
import hashlib,json
from dataclasses import dataclass
from pathlib import Path
from . import protocol
SEAL_SCHEMA="apst_dandi_subc_release_final_selection_v1"; SEAL_STATUS="FROZEN_FORMAL_SELECTION"
def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def canonical_payload_sha256(p):
 x=dict(p);x.pop("sha256",None);return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def _load(path):
 p=Path(path).resolve();x=json.loads(p.read_text())
 if x.get("schema")!=SEAL_SCHEMA or x.get("status")!=SEAL_STATUS or x.get("model_schema")!="dandi688_f0_film_sites_v1" or x.get("protocol")!=protocol.protocol_dict() or x.get("authorized_sessions")!=list(protocol.FINAL_SESSIONS) or x.get("sha256")!=canonical_payload_sha256(x):raise PermissionError("invalid release final seal")
 arts=x.get("artifacts"); cells=x.get("cell_selections")
 if not isinstance(arts,dict) or not arts or not isinstance(cells,dict) or not cells:raise PermissionError("seal lacks artifacts/cells")
 for raw,d in arts.items():
  q=Path(raw)
  if not q.is_absolute() or not q.is_file() or not isinstance(d,str) or _sha(q)!=d:raise PermissionError("artifact drift")
 for c in cells.values():
  if not isinstance(c,dict) or c.get("status")!="SELECTED" or c.get("representation")!="sua" or any(not isinstance(c.get(k),str) or c[k] not in arts for k in ("checkpoint","source_stats","selection","train_receipt")):raise PermissionError("invalid SUA selection")
 return p,x,arts
@dataclass(frozen=True)
class FinalAccess:
 manifest_path:Path; manifest_sha256:str; artifact_hashes:tuple
 @classmethod
 def from_manifest(cls,path):
  p,_,a=_load(path);return cls(p,_sha(p),tuple(sorted(a.items())))
 def validate(self):
  p,_,a=_load(self.manifest_path)
  if _sha(p)!=self.manifest_sha256 or tuple(sorted(a.items()))!=self.artifact_hashes:raise PermissionError("seal changed")
 def authorize(self,sid):
  self.validate()
  if sid not in protocol.FINAL_SESSIONS:raise PermissionError("outside final roster")
