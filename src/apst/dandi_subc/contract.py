"""Frozen Sub-C F0/flat site contract, parameterized only by portable roots."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from . import protocol
from .bootstrap import output_root, resolve_path

SCHEMA = 'dandi688_f0_film_sites_v1'
SITES = {
 'dual_site': {'encoder_family':'concat','identity_profile':True,'token_profile':True,'film':True},
 'identity_only': {'encoder_family':'concat','identity_profile':True,'token_profile':False,'film':True},
 'token_only': {'encoder_family':'activity','identity_profile':False,'token_profile':True,'film':False},
 'act_only': {'encoder_family':'activity','identity_profile':False,'token_profile':False,'film':False},
}
RECIPE=dict(segments=24,updates_per_segment=3165,batch=32,lr_peak=3e-4,lr_min_factor=.1,warmup_updates=3165,weight_decay=.01,betas=[.9,.999],eps=1e-8,grad_clip=1.,ema_decay=.9995,whole_unit_dropout=.1)
OUTPUT_EMA=dict(alpha=1/3,training=False,clock='full_recording',reset='session',state_0='pred_0')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def cache_root(config):
 return resolve_path(config.get('cache'), default=output_root()/'prepared_sua')
def validate(config):
 config=dict(config)
 if config['schema'] != SCHEMA or config['method'] not in SITES: raise ValueError('unknown campaign/method')
 if config['stage'] not in ('pretrain','train') or config['temporal'] not in ('F0','flat'): raise ValueError('unknown training stage or temporal policy')
 if config['seed'] not in (42,43,44) or config['representation'] != 'sua': raise ValueError('Sub-C release is SUA only')
 if config['protocol'] != protocol.protocol_dict() or config['recipe'] != RECIPE: raise ValueError('2015 data split or matched training recipe drift')
 if config['target_parameter_updates'] is not False or config['output_ema'] != OUTPUT_EMA: raise ValueError('target adaptation/EMA contract drift')
 expected=dict(SITES[config['method']])
 if config['stage']=='pretrain':
  if config['method'] not in ('act_only','identity_only') or config['seed']!=42 or config['temporal']!='F0': raise ValueError('pretraining is source-only family seed42 F0')
  expected['film']=False
 elif config['temporal']=='flat' and config['method']!='dual_site': raise ValueError('F0/flat comparison is restricted to FULL dual-site')
 for k,v in expected.items():
  if config[k] != v: raise ValueError('site routing drift: '+k)
 cache=cache_root(config); receipt=cache/'prepared_receipt.json'
 if not receipt.is_file(): raise ValueError('prepared cache receipt missing: '+str(receipt))
 if config.get('prepared_receipt_sha256') != sha(receipt): raise ValueError('prepared receipt binding changed')
 config['cache']=str(cache)
 return config
def load(path):
 path=Path(path); sidecar=path.with_suffix('.json.sha256')
 if not sidecar.is_file(): raise ValueError('frozen config SHA sidecar missing')
 if sha(path)!=sidecar.read_text().split()[0]: raise ValueError('frozen config SHA mismatch')
 return validate(json.loads(path.read_text()))
