#!/usr/bin/env python3
"""Queueable source/dev trainer for Static-F0; never opens final data."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
from apst.dandi_subm import training as T
from apst.dandi_subm import protocol
from static_training import train as train_static
CAMPAIGN = protocol.output_root()

OUT = CAMPAIGN / "static" / "runs" / "static_f0_s42"
CACHE = CAMPAIGN / "prepared_sua"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class StaticEncoder(nn.Module):
    def __init__(self, units: int, e0_dim: int = 50, seed: int = 42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 0x53544154)
            self.static_e0 = nn.Parameter(torch.randn(units, e0_dim) * .02)
    def forward(self, activity, profile=None): return self.static_e0
    def encode(self, activity, profile=None): return self.static_e0

def config() -> dict:
    # The shared train stage is required for its sealed per-segment development
    # selection.  The activity pretrain checkpoint satisfies that interface,
    # then the activity encoder is replaced before source optimization.
    return {"schema": T.SCHEMA, "stage": "train", "method": "act_only", "temporal": "F0",
            "representation": "sua", "seed": 42, "token_profile": False,
            "target_parameter_updates": False, "encoder_family": "activity", "film": False, "identity_profile": False,
            "evaluation_conditions": ["intact"], "output_ema": {"alpha": 1/3, "training": False, "clock": "full_recording", "reset": "session", "state_0": "pred_0"},
            "cache": str(CACHE.resolve()), "protocol": protocol.protocol_dict(), "prepared_receipt_sha256": sha(CACHE / "prepared_receipt.json"), "encoder_source": "runs/sua_activity_pretrain_s42",
            "recipe": {"segments":24,"updates_per_segment":3165,"batch":32,"lr_peak":3e-4,"lr_min_factor":.1,"warmup_updates":3165,"weight_decay":.01,"betas":[.9,.999],"eps":1e-8,"grad_clip":1.,"ema_decay":.9995,"whole_unit_dropout":.1}}

def write_receipt(dest: Path, smoke_updates: int | None) -> None:
    selected = json.loads((dest / "selection.json").read_text()) if (dest / "selection.json").is_file() else None
    static = {"schema": "dandi688_subm2015_static_f0_v1", "status": "DEV_SELECTED", "method": "Static-F0 (source-trained global E0)", "seed": 42,
              "model_type": "static_identity_global_table", "static_e0_shape": [100,50], "no_session_encoder": True, "no_profile": True, "no_token": True,
              "recipe": config()["recipe"], "selection": selected, "common_training_receipt_sha256": sha(dest / "train_receipt.json"),
              "final_sessions_opened": 0}
    static["status"] = "SMOKE_COMPLETED" if smoke_updates is not None else static["status"]
    (dest / "static_receipt.json").write_text(json.dumps(static, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": static["status"], "method": static["method"], "selected": None if selected is None else selected["checkpoint"]}, indent=2))

def train(device: str, smoke_updates: int | None = None) -> None:
    dest = CAMPAIGN / "static" / "runs" / ("static_f0_trainstage_smoke_direct" if smoke_updates is not None else "static_f0_s42")
    if dest.exists(): raise FileExistsError(f"refusing to overwrite {dest}")
    if not (CACHE / "prepared_receipt.json").is_file(): raise FileNotFoundError(CACHE / "prepared_receipt.json")
    receipt = train_static(CACHE, dest, device=device, smoke_updates=smoke_updates)
    write_receipt(dest, smoke_updates)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("stage", choices=("train", "write-smoke-receipt")); p.add_argument("--device", default="cuda:0"); p.add_argument("--smoke-updates", type=int); a=p.parse_args()
    if a.stage == "train": train(a.device, a.smoke_updates)
    else: write_receipt(CAMPAIGN / "static" / "runs" / "static_f0_trainstage_smoke_direct", 2)
