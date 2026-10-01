#!/usr/bin/env python3
"""Real-tensor A1 M2 temporal contract smoke; no training or scoring selection."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = ROOT / "btransform_unified_v2" / "learnable_recency_v1"
V2 = ROOT / "btransform_unified_v2"
V1 = ROOT / "btransform_unified_v1"
for item in (HERE.parent, LEGACY / "scripts", LEGACY / "src", V2 / "src", V2, V1 / "src", ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.rift_v1 import m2_train as frozen
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
from tfpd_exploration.src.m2_dual_track_v1 import sampler as old_sampler

from a1_m2.temporal_variants import A1M2StreamDecoder, SPECS, build_decoder, contract


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    torch.set_num_threads(4)
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cpu")
    manifest = old_sampler.load_manifest(frozen.MANIFEST)
    if manifest["digest"] != frozen.MANIFEST_DIGEST:
        raise RuntimeError("frozen M2 manifest digest drift")
    source, banks = frozen._load_surface("source_train", device)
    padding = frozen._surface_padding("source_train", banks)
    batch = next(old_sampler.iter_manifest_batches(source, manifest, 1, device=device, target_space=old_plan.TRAINING_TARGET_SPACE))
    valid = frozen._batch_valid(batch, surface="source_train", padding=padding, device=device)
    rows: dict[str, object] = {}
    base = build_decoder("A_flat")
    base_front = {name: value.detach().clone() for name, value in base.named_parameters() if not name.startswith("temporal.")}
    for arm in SPECS:
        model = build_decoder(arm).to(device).eval()
        same = all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in base_front.items())
        if not same:
            raise RuntimeError(f"{arm}: non-temporal initialization drift")
        with torch.inference_mode():
            absolute = torch.arange(2192, 2242, dtype=torch.long).view(1, -1)
            # Generic legacy stream uses its zero-origin fallback; absolute
            # cache equivalence is checked separately below through the
            # explicit A1 temporal API.
            full = model(batch.X[:1], banks[batch.session_id], input_valid_mask=valid[:1])
            stream = A1M2StreamDecoder(model)
            last = None
            for offset in range(50):
                last = stream.stream_step(batch.X[:1, offset], banks[batch.session_id], ["a1-m2-smoke"], valid_mask=valid[:1, offset])
            # Explicit absolute-position contract: complete sequence versus a
            # split cache path must agree even when the first session bin is
            # not zero.  This is the contract used by the forthcoming scorer.
            tokens = model.frontend_tokens(batch.X[:1], banks[batch.session_id])
            hidden_full = model.temporal(tokens, valid[:1], absolute_positions=absolute)
            state = model.temporal.init_state(1, tokens.device, tokens.dtype)
            pieces = []
            for offset in range(50):
                item, state = model.temporal.step(tokens[:, offset], state, valid[:1, offset], absolute_positions=absolute[:, offset])
                pieces.append(item)
            hidden_step = torch.stack(pieces, dim=1)
            # Raw-session coordinates for a left-padded window: negative
            # coordinates are legal only where the validity mask is false and
            # must neither receive PE nor advance cached state.
            left_valid = torch.ones((1, 50), dtype=torch.bool); left_valid[:, :3] = False
            left_pos = torch.arange(-3, 47, dtype=torch.long).view(1, -1)
            left_tokens = model.frontend_tokens(torch.where(left_valid.unsqueeze(-1), batch.X[:1], torch.zeros_like(batch.X[:1])), banks[batch.session_id])
            left_full = model.temporal(left_tokens, left_valid, absolute_positions=left_pos)
            left_state = model.temporal.init_state(1, left_tokens.device, left_tokens.dtype)
            left_items = []
            for offset in range(50):
                item, left_state = model.temporal.step(left_tokens[:, offset], left_state, left_valid[:, offset], absolute_positions=left_pos[:, offset])
                left_items.append(item)
            left_delta = float((left_full - torch.stack(left_items, dim=1)).abs().max())
            # Sliding-R50 versus a cache that began one bin earlier.  The
            # final token's finite RIFT receptive field must make these
            # identical when both sides use absolute session coordinates.
            start = int(batch.window_ids[0])
            timeline = np.array(source[batch.session_id].X_store[start:start + 51], dtype=np.float32, copy=True)
            if timeline.shape != (51, 96):
                raise RuntimeError("source timeline cannot provide the R50 sliding smoke")
            raw51 = torch.from_numpy(timeline).unsqueeze(0)
            pos51 = torch.arange(start, start + 51, dtype=torch.long).view(1, -1)
            window_score = model(raw51[:, 1:], banks[batch.session_id],
                                 absolute_positions=pos51[:, 1:])
            stream_tokens = model.frontend_tokens(raw51, banks[batch.session_id])
            slide_state = model.temporal.init_state(1, stream_tokens.device, stream_tokens.dtype)
            for offset in range(51):
                slide_hidden, slide_state = model.temporal.step(
                    stream_tokens[:, offset], slide_state,
                    torch.ones(1, dtype=torch.bool), absolute_positions=pos51[:, offset]
                )
            cache_score = model.readout(model.final_norm(slide_hidden))
        if last is None or not torch.isfinite(full).all() or not torch.isfinite(last).all():
            raise RuntimeError(f"{arm}: non-finite real-tensor prediction")
        delta = float((full - last).abs().max())
        position_delta = float((hidden_full - hidden_step).abs().max())
        sliding_delta = float((window_score - cache_score).abs().max())
        if position_delta > 2e-5:
            raise RuntimeError(f"{arm}: absolute-position full/chunk mismatch {position_delta}")
        if sliding_delta > 2e-5:
            raise RuntimeError(f"{arm}: sliding R50/cache mismatch {sliding_delta}")
        if left_delta > 2e-5:
            raise RuntimeError(f"{arm}: left-padding raw-coordinate mismatch {left_delta}")
        rows[arm] = {"contract": contract(arm), "finite": True, "full_stream_max_abs": delta,
                     "full_stream_within_2e-5": bool(delta <= 2e-5),
                     "absolute_position_origin": 2192, "absolute_full_chunk_max_abs": position_delta,
                     "sliding_r50_cache_max_abs": sliding_delta,
                     "left_padding_negative_position_max_abs": left_delta,
                     "prediction_sha256": hashlib.sha256(np.ascontiguousarray(full.numpy(), dtype=np.float32).tobytes()).hexdigest()}
    out = ROOT / "apst_final_week_20260916" / "receipts" / "a1_m2" / "temporal_tensor_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "apst_a1_m2_temporal_tensor_smoke_v1", "status": "COMPLETED",
               "utc": datetime.now(timezone.utc).isoformat(), "seed": 42, "device": str(device),
               "source_surface": "source_train only", "official_test_used": False, "dandi_used": False,
               "manifest": str(frozen.MANIFEST), "manifest_sha256": sha(frozen.MANIFEST),
               "source_files": {str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                                str(HERE / "temporal_variants.py"): sha(HERE / "temporal_variants.py")},
               "batch_session": batch.session_id, "input_shape": list(batch.X[:1].shape), "rows": rows}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
