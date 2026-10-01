"""Live B3S identity M1 concat-RIFT information arms (B and D)."""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import torch
from torch import Tensor
from btransform_unified_v1.bank import TaskBank
from btransform_unified_v1.m1_b3s_joint import load_trainable_b3s_from_sfix, encode_b3s
from .concat_model import RiftConcatDecoder

ARM_B = "B_ACTIVITY_ONLY"
ARM_D = "D_JOINT"
ARMS = (ARM_B, ARM_D)

class JointM1ConcatDecoder(RiftConcatDecoder):
    """R100 full-concat decoder with a differentiable, session-live B3S E0."""
    def __init__(self, arm: str, *, seed: int = 42) -> None:
        if arm not in ARMS: raise ValueError("unknown M1 joint arm")
        super().__init__("m1", context_bins=100, bias_mode="recency", seed=seed)
        self.arm = arm
        self.encoder = load_trainable_b3s_from_sfix()
        self._calib: dict[str, Tensor] = {}; self._carrier: dict[str, Tensor] = {}

    def install_session_memory(self, banks: Mapping[str, TaskBank], calib_by_session: Mapping[str, np.ndarray | Tensor]) -> None:
        for name, bank in banks.items():
            session=bank.session_id; key=session.replace('-','_')
            raw=calib_by_session.get(name,calib_by_session.get(session))
            if raw is None: raise ValueError(f"missing M10 raw calibration {session}")
            raw=np.ascontiguousarray(np.asarray(raw,dtype=np.float32))
            if raw.shape != (10,1024,64): raise ValueError(f"{session}: M10 must be [10,1024,64], got {raw.shape}")
            carrier=np.ascontiguousarray(np.asarray(bank.carrier,dtype=np.float32))
            if carrier.shape != (64,4): raise ValueError(f"{session}: carrier geometry drift")
            if hasattr(self,f'calib_{key}'):
                if not np.array_equal(getattr(self,f'calib_{key}').cpu().numpy(),raw): raise ValueError(f'cross-surface calib drift {session}')
            else:self.register_buffer(f'calib_{key}',torch.from_numpy(raw),persistent=False)
            if hasattr(self,f'carrier_{key}'):
                if not np.array_equal(getattr(self,f'carrier_{key}').cpu().numpy(),carrier): raise ValueError(f'cross-surface carrier drift {session}')
            else:self.register_buffer(f'carrier_{key}',torch.from_numpy(carrier),persistent=False)
            self._calib[session]=getattr(self,f'calib_{key}');self._carrier[session]=getattr(self,f'carrier_{key}')

    def _identity(self, sessions: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        result={}; direct={}
        with torch.autocast(device_type=device.type,enabled=False):
            for s in dict.fromkeys(sessions):
                raw=getattr(self,f'calib_{s.replace("-","_")}').to(device=device,dtype=torch.float32)
                c=getattr(self,f'carrier_{s.replace("-","_")}').to(device=device,dtype=torch.float32)
                side=c if self.arm==ARM_D else torch.zeros_like(c)
                result[s]=encode_b3s(self.encoder,raw,side)
                direct[s]=c if self.arm==ARM_D else torch.zeros_like(c)
        return torch.stack([result[s] for s in sessions]),torch.stack([direct[s] for s in sessions])

    def frontend_tokens(self,x:Tensor,bank:TaskBank|Sequence[TaskBank],unit_mask=None,dropout_generator=None,dropout_keep=None)->Tensor:
        x=self._check_input(x,None);banks=self._normalise_banks(bank,x.shape[0]);keep=self._resolve_keep(banks,unit_mask,x.shape[0],x.device,dropout_generator,dropout_keep);e,c=self._identity([b.session_id for b in banks],x.device);return self._batched_proj_add_frontend(x,e,c,keep)
    def frontend_last(self,raw5:Tensor,bank:TaskBank|Sequence[TaskBank],unit_mask=None)->Tensor:
        if raw5.ndim!=3 or raw5.shape[1:]!=(5,self.units):raise ValueError('raw5 must be [B,5,64]')
        raw5=self._check_input(raw5,None);banks=self._normalise_banks(bank,raw5.shape[0]);keep=self._resolve_keep(banks,unit_mask,raw5.shape[0],raw5.device,None,None);e,c=self._identity([b.session_id for b in banks],raw5.device);flat=raw5.permute(0,2,1).reshape(raw5.shape[0]*self.units,1,5);conv=self.frontend.local_conv;local=conv.act(conv.conv(flat)).reshape(raw5.shape[0],self.units,16,1).permute(0,3,1,2);return self._fuse_batched_local(local,e,c,keep)[:,0]

__all__=['JointM1ConcatDecoder','ARM_B','ARM_D','ARMS']
