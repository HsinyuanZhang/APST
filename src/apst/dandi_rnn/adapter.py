"""Materialize portable RNN records directly from public DANDI SUA NWB."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO
from .core import BUDGETS, WINDOW, sha256, write_json


def roster(subject: str):
    if subject == "C":
        from apst.dandi_subc import protocol, data
    elif subject == "M":
        from apst.dandi_subm import protocol, subm_data as data
    else: raise ValueError("subject must be C or M")
    return protocol, data


def endpoints_from_raw(cache: Path, raw: Path, budget: int) -> np.ndarray:
    """First M legal rewarded trials, keeping only complete 50-bin histories."""
    with np.load(cache, allow_pickle=False) as z: edges=np.asarray(z["bin_edges"],np.float64)
    with NWBHDF5IO(str(raw), "r", load_namespaces=True) as io:
        trials=io.read().intervals["trials"].to_dataframe(); legal=[]
        for _, row in trials.iterrows():
            if row.get("result") != "R": continue
            left=max(0,int(np.searchsorted(edges,float(row.start_time))))
            right=min(len(edges)-1,int(np.searchsorted(edges,float(row.stop_time))))
            if right-left >= WINDOW: legal.append((left,right))
    if len(legal) < budget: raise ValueError(f"{raw}: fewer than {budget} legal rewarded trials")
    bins=np.unique(np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in legal[:budget]]))
    available=np.zeros(len(edges)-1,bool);available[bins]=True
    return np.flatnonzero(np.convolve(available.astype(np.int8),np.ones(WINDOW,np.int8),mode="valid") == WINDOW)+WINDOW-1


def convert(subject: str, cache_root: Path, raw_root: Path, out: Path) -> None:
    protocol, loader = roster(subject); out.mkdir(parents=True,exist_ok=True); splits={}
    for split, ids in (("train",protocol.TRAIN_SESSIONS),("dev",protocol.DEV_SESSIONS),("final",protocol.FINAL_SESSIONS)):
        rows=[]
        for sid in ids:
            source=cache_root/f"{sid}.sua.npz"
            # No APST final capability is bypassed: this adapter consumes a cache
            # materialized by the caller's separately sealed final phase.
            with np.load(source,allow_pickle=False) as z:
                neural=np.asarray(z["neural"],np.float32);velocity=np.asarray(z["velocity"],np.float32);query=np.asarray(z["query_indices"],np.int64)
            supports={f"support_{b}":endpoints_from_raw(source,raw_root/f"{sid}_behavior+ecephys.nwb",b) for b in BUDGETS}
            target=out/f"{sid}.rnn.npz";np.savez_compressed(target,session_id=np.asarray(sid),neural=neural,velocity=velocity,query=query,**supports)
            rows.append({"file":target.name,"sha256":sha256(target),"session_id":sid})
        splits[split]=rows
    write_json(out/"manifest.json",{"schema":"dandi_rnn_records_v1","subject":f"sub-{subject}","protocol":protocol.protocol_dict(),"splits":splits})


def _rnn_final_access(subject: str, seal_path: Path):
    """A narrow subclass accepted by the existing public raw-NWB loaders.

    It is intentionally a new RNN capability: it validates this release's
    development seal and source checkpoint instead of accepting APST model seals.
    """
    protocol, _ = roster(subject); seal_path=seal_path.resolve(); doc=json.loads(seal_path.read_text())
    original=dict(doc); claimed=original.pop('sha256',None)
    if claimed!=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest(): raise PermissionError('RNN final seal self-digest mismatch')
    if (doc.get('schema')!='dandi_rnn_final_selection_v1' or doc.get('status')!='FROZEN_BEFORE_FINAL_LOAD'
        or doc.get('subject')!=f'sub-{subject}' or doc.get('protocol')!=protocol.protocol_dict()
        or doc.get('authorized_sessions')!=list(protocol.FINAL_SESSIONS) or not doc.get('config_sha256') or not doc.get('source_dev_manifest_sha256')):
        raise PermissionError('invalid DANDI RNN final seal')
    artifacts=doc.get('artifacts',{})
    if not artifacts: raise PermissionError('RNN seal has no immutable artifacts')
    for raw,digest in artifacts.items():
        if not Path(raw).is_file() or sha256(raw)!=digest: raise PermissionError('RNN sealed artifact drift')
    if subject=='C':
        from apst.dandi_subc.final_access import FinalAccess
        class Cap(FinalAccess):
            def validate(self):
                for raw,digest in artifacts.items():
                    if not Path(raw).is_file() or sha256(raw)!=digest: raise PermissionError('RNN sealed artifact drift')
            def authorize(self,sid):
                self.validate()
                if sid not in protocol.FINAL_SESSIONS: raise PermissionError('outside RNN final roster')
        return Cap(seal_path,sha256(seal_path),tuple(sorted(artifacts.items())))
    from apst.dandi_subm.final_access import FinalAccess
    class Cap(FinalAccess):
        def __init__(self): pass
        def authorize(self,sid):
            for raw,digest in artifacts.items():
                if not Path(raw).is_file() or sha256(raw)!=digest: raise PermissionError('RNN sealed artifact drift')
            if sid not in protocol.FINAL_SESSIONS: raise PermissionError('outside RNN final roster')
    return Cap()


def prepare(subject: str, raw_root: Path, out: Path, seal: Path | None = None) -> None:
    """Standalone public-NWB materialization for source/dev, and sealed final only."""
    protocol, data = roster(subject); out.mkdir(parents=True,exist_ok=True); splits={}
    plan=[('train',protocol.TRAIN_SESSIONS,'source'),('dev',protocol.DEV_SESSIONS,'development')]
    if seal is not None: plan.append(('final',protocol.FINAL_SESSIONS,'final'))
    for split,ids,purpose in plan:
        rows=[]; cap=_rnn_final_access(subject,seal) if purpose=='final' else None
        for sid in ids:
            if purpose=='final': record=data.load_pair(sid,raw_root=raw_root,purpose='final',final_access=cap)['sua']
            else: record=data.load_pair(sid,raw_root=raw_root,purpose=purpose)['sua']
            supports={f'support_{b}': endpoints_from_record(record,raw_root/f'{sid}_behavior+ecephys.nwb',b) for b in BUDGETS}
            target=out/f'{sid}.rnn.npz';np.savez_compressed(target,session_id=np.asarray(sid),neural=record.neural.astype('float32'),velocity=record.velocity.astype('float32'),query=record.query_indices.astype('int64'),**supports)
            rows.append({'file':target.name,'sha256':sha256(target),'session_id':sid})
        splits[split]=rows
    manifest=out/'manifest.json'; old=json.loads(manifest.read_text()) if manifest.exists() else {'schema':'dandi_rnn_records_v1','subject':f'sub-{subject}','protocol':protocol.protocol_dict(),'splits':{}}
    old['splits'].update(splits);write_json(manifest,old)


def endpoints_from_record(record, raw: Path, budget: int) -> np.ndarray:
    edges=np.asarray(record.bin_edges,np.float64)
    with NWBHDF5IO(str(raw),'r',load_namespaces=True) as io:
        trials=io.read().intervals['trials'].to_dataframe(); legal=[]
        for _,row in trials.iterrows():
            if row.get('result')!='R': continue
            a=max(0,int(np.searchsorted(edges,float(row.start_time))));b=min(len(record.neural),int(np.searchsorted(edges,float(row.stop_time))))
            if b-a>=WINDOW: legal.append((a,b))
    if len(legal)<budget: raise ValueError(f'{raw}: fewer than {budget} legal rewarded trials')
    available=np.zeros(len(record.neural),bool);available[np.concatenate([np.arange(a,b) for a,b in legal[:budget]])]=True
    return np.flatnonzero(np.convolve(available.astype(np.int8),np.ones(WINDOW,np.int8),mode='valid')==WINDOW)+WINDOW-1


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--subject',choices=('C','M'),required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-root',type=Path);p.add_argument('--seal',type=Path);a=p.parse_args(argv)
    if a.cache_root is not None: convert(a.subject,a.cache_root,a.raw_root,a.out)
    else: prepare(a.subject,a.raw_root,a.out,a.seal)

if __name__ == "__main__": main()
