from pathlib import Path
import hashlib,json
from apst.dandi_subm import protocol
SCHEMA='dandi688_subm_f0_film_v1'

def cache_root(root=None): return protocol.output_root(root) / 'prepared_sua'
RECIPE={'segments':24,'updates_per_segment':3165,'batch':32,'lr_peak':3e-4,'lr_min_factor':.1,'warmup_updates':3165,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'grad_clip':1.,'ema_decay':.9995,'whole_unit_dropout':.1}
SITES={'dual_site':dict(encoder_family='concat',identity_profile=True,token_profile=True,film=True),'identity_only':dict(encoder_family='concat',identity_profile=True,token_profile=False,film=False),'act_only':dict(encoder_family='activity',identity_profile=False,token_profile=False,film=False)}
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def validate(c):
 if c.get('schema')!=SCHEMA or c.get('method') not in SITES or c.get('stage') not in ('pretrain','train') or c.get('seed')!=42 or c.get('representation')!='sua' or c.get('temporal')!='F0':raise ValueError('sub-M frozen matrix drift')
 cache=cache_root(c.get('output_root'))
 if c.get('protocol')!=protocol.protocol_dict() or c.get('recipe')!=RECIPE or Path(c.get('cache','')).resolve()!=cache.resolve():raise ValueError('protocol/recipe/cache drift')
 if c.get('prepared_receipt_sha256')!=sha(cache/'prepared_receipt.json') or c.get('target_parameter_updates') is not False:raise ValueError('prepared binding or target-gradient drift')
 if any(c.get(k)!=v for k,v in SITES[c['method']].items()):raise ValueError('site routing drift')
 if c['stage']=='pretrain' and c['method'] not in ('act_only','identity_only'):raise ValueError('pretraining family drift')
 return c
def load(p):
 p=Path(p); expected=p.with_suffix('.json.sha256').read_text().split()[0]
 if sha(p)!=expected:raise ValueError('config hash drift')
 return validate(json.loads(p.read_text()))
