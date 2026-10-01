"""Differentiable M2 HoldContrast-FiLM encoder coupled to the RIFT decoder."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import torch
from torch import Tensor
from btransform_unified_v1.bank import TaskBank
from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder
from tfpd_exploration.src.m2_dual_track_v1 import champion
from .model import RiftDecoder

ARM_B = "B_ACTIVITY_ONLY"; ARM_D = "D_JOINT"

class JointM2RiftDecoder(RiftDecoder):
    """RIFT with a fully differentiable chronological M33 FiLM identity provider."""
    def __init__(self, arm: str, *, seed: int = 42) -> None:
        if arm not in (ARM_B, ARM_D): raise ValueError("unknown joint arm")
        super().__init__('m2', context_bins=50, bias_mode='recency', seed=seed, proj_dim=16)
        self.arm=arm
        self.encoder=HoldContrastFiLMEarlyPoolEncoder(100,50,64,side_dim=8,film_rank=8,num_post_layers=3,film_input='t4_plus_contrast')
        champion.overlay_canonical_p0_and_empty_head(self.encoder)
        self._calib: dict[str, Tensor] = {}; self._carrier: dict[str, Tensor] = {}

    def install_session_memory(self, banks: Mapping[str, TaskBank], cache_root: Path) -> None:
        """Register raw calibration and MOVE-T4 as nontrainable buffers once."""
        for session, bank in banks.items():
            key=session.replace('-','_')
            raw=np.load(cache_root / session / 'calib_activity.npy').astype(np.float32, copy=False)
            if raw.shape != (33,100,96): raise ValueError(f'raw M33 geometry drift {session}: {raw.shape}')
            if hasattr(self, f'calib_{key}'):
                prior=getattr(self,f'calib_{key}').detach().cpu().numpy()
                if not np.array_equal(prior, raw): raise ValueError(f'cross-surface calibration drift {session}')
            else:
                self.register_buffer(f'calib_{key}', torch.from_numpy(np.ascontiguousarray(raw)), persistent=False)
                self.register_buffer(f'carrier_{key}', torch.from_numpy(np.ascontiguousarray(bank.carrier)), persistent=False)
            self._calib[session]=getattr(self,f'calib_{key}'); self._carrier[session]=getattr(self,f'carrier_{key}')

    def _identity(self, sessions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        # Encode each distinct session only once per batch. Reusing the graph
        # for repeated rows makes their decoder gradients aggregate naturally.
        unique=list(dict.fromkeys(sessions)); encoded={}; carriers={}
        with torch.autocast(device_type=device.type, enabled=False):
            for session in unique:
                trials=getattr(self, f'calib_{session.replace("-", "_")}').to(device=device,dtype=torch.float32)
                carrier=getattr(self, f'carrier_{session.replace("-", "_")}').to(device=device,dtype=torch.float32)
                side=torch.zeros((96,8),device=device,dtype=torch.float32); direct=torch.zeros_like(carrier)
                if self.arm == ARM_D: side[:,:4]=carrier; direct=carrier
                state=self.encoder.reset_stream(1,96,device,torch.float32); state['side_features']=side.unsqueeze(0)
                for trial in trials: state=self.encoder.push_trial(state,trial)
                encoded[session]=self.encoder.finalize_identity(state).squeeze(0); carriers[session]=direct
        return torch.stack([encoded[s] for s in sessions]),torch.stack([carriers[s] for s in sessions])

    def frontend_last(self, raw5: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask: Tensor | None = None) -> Tensor:
        """Live one-token frontend for streaming; never reads frozen bank E0/T."""
        if raw5.ndim != 3 or raw5.shape[1:] != (5, self.units):
            raise ValueError(f"raw5 must be [B,5,{self.units}]")
        raw5=self._check_input(raw5,None); banks=self._normalise_banks(bank,raw5.shape[0]); keep=self._resolve_keep(banks,unit_mask,raw5.shape[0],raw5.device,None,None)
        e0,carrier=self._identity([b.session_id for b in banks],raw5.device)
        flat=raw5.permute(0,2,1).reshape(raw5.shape[0]*self.units,1,5); conv=self.frontend.local_conv
        local=conv.act(conv.conv(flat)).reshape(raw5.shape[0],self.units,16,1).permute(0,3,1,2)
        return self._fuse_batched_local(local,e0,carrier,keep)[:,0]

    def frontend_tokens(self, x: Tensor, bank: TaskBank | Sequence[TaskBank], unit_mask=None, dropout_generator=None, dropout_keep=None) -> Tensor:
        x=self._check_input(x,None); banks=self._normalise_banks(bank,x.shape[0]); keep=self._resolve_keep(banks,unit_mask,x.shape[0],x.device,dropout_generator,dropout_keep)
        e0,carrier=self._identity([b.session_id for b in banks],x.device)
        return self._batched_proj_add_frontend(x,e0,carrier,keep)
