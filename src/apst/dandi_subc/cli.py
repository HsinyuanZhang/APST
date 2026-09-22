"""Executable Sub-C campaign entry points."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from .bootstrap import output_root
from .contract import load,sha
from .training import run_training

def main(argv=None):
 p=argparse.ArgumentParser(prog='apst-dandi-subc'); q=p.add_subparsers(dest='cmd',required=True)
 def args(name):
  x=q.add_parser(name);x.add_argument('--config',type=Path,required=True);x.add_argument('--dest',type=Path,required=True);x.add_argument('--device',default='cpu');x.add_argument('--smoke-updates',type=int)
 args('pretrain');x=q.add_parser('stage2');x.add_argument('--config',type=Path,required=True);x.add_argument('--dest',type=Path,required=True);x.add_argument('--encoder',type=Path,required=True);x.add_argument('--device',default='cpu');x.add_argument('--smoke-updates',type=int)
 x=q.add_parser('prepare');x.add_argument('--cache',type=Path,required=True);x.add_argument('--raw-root',type=Path)
 x=q.add_parser('init-config');x.add_argument('--template',type=Path,required=True);x.add_argument('--cache',type=Path,required=True);x.add_argument('--dest',type=Path,required=True)
 x=q.add_parser('static');x.add_argument('--cache',type=Path,required=True);x.add_argument('--dest',type=Path,required=True);x.add_argument('--device',default='cpu')
 a=p.parse_args(argv)
 if a.cmd=='init-config':
  receipt=a.cache/'prepared_receipt.json'
  if not receipt.is_file(): raise SystemExit('missing '+str(receipt))
  c=json.loads(a.template.read_text()); from . import protocol; c['protocol']=protocol.protocol_dict(); c['cache']=str(a.cache.resolve()); c['prepared_receipt_sha256']=sha(receipt); a.dest.parent.mkdir(parents=True,exist_ok=True); a.dest.write_text(json.dumps(c,indent=2)+'\n'); a.dest.with_suffix('.json.sha256').write_text(sha(a.dest)+'  '+a.dest.name+'\n'); print(str(a.dest)); return
 if a.cmd=='prepare':
  from .prepare import prepare
  print(json.dumps(prepare(a.cache,a.raw_root),default=str));return
 if a.cmd in ('pretrain','stage2'):
  c=load(a.config)
  expected='pretrain' if a.cmd=='pretrain' else 'train'
  if c['stage'] != expected: raise SystemExit(f'{a.cmd} requires config stage={expected}; config declares {c["stage"]}')
  print(json.dumps(run_training(c,a.dest,encoder_path=getattr(a,'encoder',None),device=a.device,smoke_updates=a.smoke_updates),default=str));return
 if a.cmd=='static':
  from .static import run
  result=run(a)
  print(json.dumps(result if result is not None else {'status':'COMPLETED'},default=str));return
if __name__=='__main__': main()
