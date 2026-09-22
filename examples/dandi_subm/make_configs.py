import json,hashlib
from pathlib import Path
from apst.dandi_subm import protocol
from apst.dandi_subm.contract import RECIPE,SCHEMA,sha
def write(name,method,stage,enc=None):
 root=protocol.output_root()
 c={'schema':SCHEMA,'output_root':str(root),'cache':str(root/'prepared_sua'),'prepared_receipt_sha256':sha(root/'prepared_sua/prepared_receipt.json'),'protocol':protocol.protocol_dict(),'recipe':RECIPE,'representation':'sua','seed':42,'method':method,'stage':stage,'temporal':'F0','target_parameter_updates':False,'output_ema':{'alpha':1/3,'training':False,'clock':'full_recording','reset':'session','state_0':'pred_0'},'encoder_family':'concat' if method in ('identity_only','dual_site') else 'activity','identity_profile':method in ('identity_only','dual_site'),'token_profile':method=='dual_site','film':method=='dual_site','evaluation_conditions':(['intact','profile_shuffle'] if method=='dual_site' and stage=='train' else ['intact'] if stage=='train' else []),'shuffle_seed':42,'run_dir':str(root/'runs'/name)}
 if enc:c['encoder_source']=str(root/'runs'/enc)
 p=root/'configs'/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(c,sort_keys=True,indent=2)+'\n');p.with_suffix('.json.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
for v in [('sua_concat_pretrain_s42','identity_only','pretrain',None),('sua_dual_site_F0_s42','dual_site','train','sua_concat_pretrain_s42'),('sua_activity_pretrain_s42','act_only','pretrain',None),('sua_act_only_F0_s42','act_only','train','sua_activity_pretrain_s42')]:write(*v)
