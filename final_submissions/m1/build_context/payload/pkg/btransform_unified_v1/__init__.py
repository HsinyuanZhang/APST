"""btransform_unified_v1 — the B-transformer unified series (NOT SPINT).

Identity declaration (workorder, mandatory): this package is the B-transformer
series — 8 learned slots + causal temporal core (lineage: dual_track B arm ->
S1-SMALL-COS -> this package). It is **NOT SPINT** and shares no decoder with
the SPINT family (Original / C2 / 581727). Every comparison against SPINT
systems may only be written as "same scoring surface, different system";
writing "same decoder with a swapped core" or inheriting SPINT-family numbers
is forbidden. Every H1/M1 number must carry the NOTE §6 six-row alignment
table (:data:`btransform_unified_v1.plan.ALIGNMENT_TABLE_FIELDS`).

One network structure, tasks differ only in geometry (M2 / M1 / H1).
Phase 0 = CPU-only skeleton; no training, no official/EvalAI surface, no
historical result roots touched. Authoritative docs:
  - btransform_unified_v1/docs/WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md
  - tfpd_exploration/docs/NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN_20260906.md
"""

from __future__ import annotations

from . import (
    adapters,
    bank,
    ema,
    h1_config,
    identity_variant,
    matrix_cells,
    plan,
    r2,
    receipts,
    scale_bridge,
    schedule,
)
from .bank import TaskBank, make_synthetic_bank
from .ema import DecoderEMA
from .identity_variant import BTransformerUnifiedDecoderIdentity
from .model import BTransformerUnifiedDecoder, whole_unit_dropout
from .plan import (
    ALIGNMENT_TABLE_FIELDS,
    SCHEMA,
    BTransformerUnifiedError,
    alignment_table,
    require,
)
from .r2 import equal_session_mean, session_mean_report, variance_weighted_r2
from .receipts import read_sealed, seal_json
from .scale_bridge import assert_scale_bridge
from .schedule import warmup_cosine_lr

__version__ = "0.1.0-phase0"

__all__ = [
    "SCHEMA",
    "ALIGNMENT_TABLE_FIELDS",
    "BTransformerUnifiedError",
    "require",
    "alignment_table",
    "TaskBank",
    "make_synthetic_bank",
    "BTransformerUnifiedDecoder",
    "BTransformerUnifiedDecoderIdentity",
    "whole_unit_dropout",
    "DecoderEMA",
    "warmup_cosine_lr",
    "variance_weighted_r2",
    "equal_session_mean",
    "session_mean_report",
    "assert_scale_bridge",
    "seal_json",
    "read_sealed",
    "adapters",
    "bank",
    "ema",
    "h1_config",
    "identity_variant",
    "matrix_cells",
    "plan",
    "r2",
    "receipts",
    "scale_bridge",
    "schedule",
]
