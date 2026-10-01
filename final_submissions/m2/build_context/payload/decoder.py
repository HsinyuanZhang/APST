"""Portable F/noPE learned-slope runtime used by the M2 shared-FiLM payload.

The selected shared-FiLM E0 values are materialized from the legal M33
supports by the exporter.  Once packaged, streaming is exactly the established
F decoder/runtime path; keeping that implementation in one place avoids a
second, subtly different ``A1M2StreamDecoder`` cache implementation.
"""
from m2_f_runtime_v1 import ALPHA, M2FNoPELearnedDecoder

M2FSharedFiLMDecoder = M2FNoPELearnedDecoder

__all__ = ["ALPHA", "M2FSharedFiLMDecoder"]
