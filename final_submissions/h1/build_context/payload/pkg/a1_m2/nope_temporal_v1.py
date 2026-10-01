"""Local E_nope_flat temporal policy; never mutates a1_m2.temporal_variants.SPECS."""
from __future__ import annotations
import torch
from a1_m2.temporal_variants import VariantSpec, build_decoder as _build_a, contract as _a_contract
E_SPEC=VariantSpec('E_nope_flat',False,'none')
SPECS={'E_nope_flat':E_SPEC}
def install_nope(decoder,task):
 """A temporal is already installed; replace only its immutable facade policy."""
 if task not in ('m1','m2','h1'):raise ValueError('unknown task')
 decoder.temporal.spec=E_SPEC
 with torch.no_grad():decoder.temporal.core.recency_slopes.zero_()
 if not torch.count_nonzero(decoder.temporal.core.recency_slopes).item()==0:raise RuntimeError('E slopes not zero')
 return decoder
def build_decoder(arm='E_nope_flat',*,seed=42,proj_dim=16):
 if arm!='E_nope_flat':raise ValueError('E runner admits only E_nope_flat')
 # Build A first: identical initializer/RNG sequence and identical local core blocks.
 d=_build_a('A_flat',seed=seed,proj_dim=proj_dim);return install_nope(d,'m2')
def contract(arm='E_nope_flat'):
 if arm!='E_nope_flat':raise ValueError(arm)
 x=_a_contract('A_flat');x.update(arm='E_nope_flat',sinusoidal_pe=False,bias='none',fixed_slopes=[0.0]*8,changed_factor='absolute sinusoidal PE disabled only');return x
