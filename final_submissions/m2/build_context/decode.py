#!/usr/bin/env python3
import argparse
import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from m2_f_runtime_v1 import M2FNoPELearnedDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--evaluation", choices=("local", "remote"), required=True)
parser.add_argument("--model-path", default="/payload")
parser.add_argument("--split", choices=("m2",), default="m2")
parser.add_argument("--phase", choices=("minival", "test"), default="test")
parser.add_argument("--batch-size", type=int, default=7)
args = parser.parse_args()
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
decoder = M2FNoPELearnedDecoder(FalconConfig(task=FalconTask.m2), args.model_path, args.batch_size)
FalconEvaluator(eval_remote=args.evaluation == "remote", split=args.split, dataloader_workers=0).evaluate(decoder, phase=args.phase)
