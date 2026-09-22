"""P16 projection-conditioned frontend owner used to preserve initialization."""

from __future__ import annotations
from typing import Any, Mapping
import torch
from torch import nn
from . import _h1_config as h1_config
from .frontend import (
    CONV_CHANNELS,
    N_HEADS,
    N_SLOTS,
    READOUT_HIDDEN,
    SET_DIM,
    SLOT_FFN_DIM,
    TEMPORAL_WIDTH,
    CausalTransformerStack,
    SharedCausalConv,
    initialize_decoder,
)


class SharedSetFrontendIdentity(nn.Module):
    """P16 local convolution, per-unit projection, and eight-slot set pooling."""

    def __init__(self, token_in: int) -> None:
        super().__init__()
        if token_in != CONV_CHANNELS + 4:
            raise ValueError("P16 token input must be local channels plus carrier")
        self.token_in = token_in
        self.e0_proj: nn.Linear | None = None
        self.local_conv = SharedCausalConv()
        self.token_mlp = nn.Sequential(
            nn.Linear(token_in, SET_DIM), nn.GELU(), nn.Linear(SET_DIM, SET_DIM)
        )
        self.slots = nn.Parameter(torch.zeros(N_SLOTS, SET_DIM))
        self.slot_norm = nn.LayerNorm(SET_DIM)
        self.token_norm = nn.LayerNorm(SET_DIM)
        self.mha = nn.MultiheadAttention(
            SET_DIM, N_HEADS, dropout=0.0, batch_first=True
        )
        self.slot_ffn_norm = nn.LayerNorm(SET_DIM)
        self.slot_ffn = nn.Sequential(
            nn.Linear(SET_DIM, SLOT_FFN_DIM),
            nn.GELU(),
            nn.Linear(SLOT_FFN_DIM, SET_DIM),
        )
        self.slot_proj = nn.Linear(N_SLOTS * SET_DIM, TEMPORAL_WIDTH)


class BTransformerUnifiedDecoderIdentity(nn.Module):
    """Temporary owner whose construction matches the released P16 RNG layout."""

    def __init__(
        self,
        task: str | Mapping[str, Any],
        seed: int = 42,
        identity_mode: str = "proj_add",
        proj_dim: int = 16,
        **_: Any,
    ) -> None:
        super().__init__()
        if (
            identity_mode != h1_config.IDENTITY_DEFAULT
            or proj_dim != h1_config.PROJ_ADD_OUT_DIM
        ):
            raise ValueError("APST supports only P16 proj_add")
        if isinstance(task, str):
            geometry = {
                "h1": (300, 176, 700, 7),
                "m1": (100, 64, 100, 16),
                "m2": (50, 96, 50, 2),
            }[task]
        else:
            geometry = tuple(
                int(task[name]) for name in ("window", "units", "e0_dim", "out_dim")
            )
        self.window, self.units, self.base_e0_dim, self.out_dim = geometry
        self.frontend = SharedSetFrontendIdentity(CONV_CHANNELS + 4)
        self.frontend.e0_proj = nn.Linear(self.base_e0_dim, proj_dim, bias=False)
        self.final_norm = nn.LayerNorm(TEMPORAL_WIDTH)
        self.readout = nn.Sequential(
            nn.Linear(TEMPORAL_WIDTH, READOUT_HIDDEN),
            nn.GELU(),
            nn.Linear(READOUT_HIDDEN, self.out_dim),
        )
        self.temporal = CausalTransformerStack(self.window)
        initialize_decoder(self, seed)
