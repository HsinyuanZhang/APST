#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'pkg'),str(ROOT)]
os.environ.setdefault("OMP_NUM_THREADS","2"); os.environ.setdefault("MKL_NUM_THREADS","2")
import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from decoder import M1FrozenEMAOfficialDecoder
p=argparse.ArgumentParser(description="M1 frozen-EMA official Falcon decoder")
p.add_argument("--evaluation",choices=("local","remote"),required=True);p.add_argument("--model-path",default=str(ROOT));p.add_argument("--split",choices=("m1",),default="m1");p.add_argument("--phase",choices=("minival","test"),default="test");p.add_argument("--batch-size",type=int,default=4);a=p.parse_args()
torch.set_num_threads(2)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
FalconEvaluator(eval_remote=a.evaluation=="remote",split="m1",dataloader_workers=0).evaluate(M1FrozenEMAOfficialDecoder(FalconConfig(task=FalconTask.m1),a.model_path,a.batch_size),phase=a.phase)
