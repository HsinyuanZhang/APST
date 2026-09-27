import json,hashlib
from pathlib import Path
import numpy as np

from apst.dandi_rnn.core import BUDGETS, Record, causal_windows, full_clock_prediction, RNN
from apst.dandi_rnn.runner import main
from apst.dandi_rnn.adapter import _rnn_final_access


def _seal(tmp_path, subject):
    if subject=='C': from apst.dandi_subc import protocol
    else: from apst.dandi_subm import protocol
    checkpoint=tmp_path/f'{subject}.pt';checkpoint.write_bytes(b'checkpoint')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    value={'schema':'dandi_rnn_final_selection_v1','status':'FROZEN_BEFORE_FINAL_LOAD','subject':f'sub-{subject}','protocol':protocol.protocol_dict(),'authorized_sessions':list(protocol.FINAL_SESSIONS),'artifacts':{str(checkpoint):digest},'config_sha256':'config','source_dev_manifest_sha256':'manifest'}
    value['sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest();path=tmp_path/f'{subject}.json';path.write_text(json.dumps(value));return path


def test_rnn_final_capabilities_are_loader_compatible_and_reject_tampering(tmp_path):
    c=_rnn_final_access('C',_seal(tmp_path,'C'));m=_rnn_final_access('M',_seal(tmp_path,'M'))
    from apst.dandi_subc.final_access import FinalAccess as CAccess
    from apst.dandi_subm.final_access import FinalAccess as MAccess
    assert isinstance(c,CAccess) and isinstance(m,MAccess)
    bad=_seal(tmp_path,'C');data=json.loads(bad.read_text());data['config_sha256']='tampered';bad.write_text(json.dumps(data))
    import pytest
    with pytest.raises(PermissionError): _rnn_final_access('C',bad)


def _record(path: Path, name: str, shift: int):
    rng=np.random.default_rng(shift); x=rng.normal(size=(80,3)).astype('float32'); y=np.c_[x[:,0],x[:,1]].astype('float32')
    query=np.arange(65,80,dtype='int64'); supports={f'support_{b}':np.arange(49,55,dtype='int64') for b in BUDGETS}
    np.savez_compressed(path,session_id=np.asarray(name),neural=x,velocity=y,query=query,**supports)


def test_windows_and_reduced_end_to_end_cpu(tmp_path):
    rows={}
    for split, n in [('train',1),('dev',1),('final',1)]:
        p=tmp_path/f'{split}.npz';_record(p,split,hash(split)%100);rows[split]=[{'file':p.name}]
    manifest=tmp_path/'records.json';manifest.write_text(json.dumps({'schema':'dandi_rnn_records_v1','subject':'sub-C','status':'SMOKE','splits':rows}))
    config=tmp_path/'config.json';config.write_text(json.dumps({'source_hps':[[2,0.001]],'source_epochs':1,'source_updates_per_epoch':1,'scratch_hps':[[2,0.001]],'scratch_eval_epochs':[1],'scratch_updates_per_epoch':1,'finetune_lrs':[0.001],'finetune_steps':[0,1]}))
    main(['--manifest',str(manifest),'--config',str(config),'--out',str(tmp_path/'out'),'--device','cpu'])
    result=json.loads((tmp_path/'out/results.json').read_text());assert result['development_seal'];assert len(result['target_from_scratch']['final'])==4


def test_runner_lazily_materializes_final_only_after_seal(tmp_path, monkeypatch):
    rows={}
    for split in ('train','dev'):
        p=tmp_path/f'{split}.npz';_record(p,split,10);rows[split]=[{'file':p.name}]
    manifest=tmp_path/'records.json';manifest.write_text(json.dumps({'schema':'dandi_rnn_records_v1','subject':'sub-C','status':'SMOKE','splits':rows}))
    config=tmp_path/'config.json';config.write_text(json.dumps({'source_hps':[[2,0.001]],'source_epochs':1,'source_updates_per_epoch':1,'scratch_hps':[[2,0.001]],'scratch_eval_epochs':[1],'scratch_updates_per_epoch':1,'finetune_lrs':[0.001],'finetune_steps':[0]}))
    calls=[]
    def fake_prepare(subject, raw_root, record_root, seal):
        calls.append((subject,Path(seal).exists()))
        assert Path(seal).exists()
        p=tmp_path/'final.npz';_record(p,'final',11)
        doc=json.loads(manifest.read_text());doc['splits']['final']=[{'file':p.name}];manifest.write_text(json.dumps(doc))
    monkeypatch.setattr('apst.dandi_rnn.adapter.prepare',fake_prepare)
    main(['--manifest',str(manifest),'--config',str(config),'--out',str(tmp_path/'out'),'--raw-root',str(tmp_path),'--device','cpu'])
    assert calls==[('C',True)]
