#!/usr/bin/env python3
"""CPU functional smoke for isolated task-aware unified temporal variants."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
WORKSPACE = WEEK.parent
for path in (
    WEEK / "src",
    WORKSPACE / "btransform_unified_v2" / "learnable_recency_v1" / "src",
    WORKSPACE / "btransform_unified_v2" / "src",
    WORKSPACE / "btransform_unified_v1" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v2.concat_model import RiftConcatDecoder
from unified_falcon.temporal import SPECS, UnifiedStreamDecoder, install_temporal, task_windows

TOL = 2e-5


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_sha(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_summary(state) -> dict[str, object]:
    return {"positions": list(state.positions), "kv_lengths": state.valid_lengths,
            "increment_lengths": ([[len(row) for row in layer] for layer in state.increments]
                                  if hasattr(state, "increments") else None)}


def check(task: str, arm: str) -> dict[str, object]:
    context = {"m1": 100, "m2": 50, "h1": 300}[task]
    torch.manual_seed(10_000 + context + sum(map(ord, arm)))
    decoder = RiftConcatDecoder(task, context_bins=context, bias_mode="recency", seed=42).eval()
    before = block_sha(decoder.temporal.blocks)
    install_temporal(decoder, task, arm)
    temporal = decoder.temporal.eval()
    after = block_sha(temporal.core.blocks)
    if before != after:
        raise RuntimeError(f"{task}/{arm}: shared attention/FFN/LN byte drift")
    if tuple(temporal.config.windows) != task_windows(task):
        raise RuntimeError(f"{task}/{arm}: window contract drift")
    if task == "m1" and tuple(temporal.config.windows) != (25, 25, 25, 24):
        raise RuntimeError("M1 R100 window contract drift")
    if task == "h1" and tuple(temporal.config.windows) != (75, 75, 75, 74):
        raise RuntimeError("H1 R300 window contract drift")

    # Run beyond twice the native R.  Unequal left pads prove that cached
    # positions are row-local, while the long suffix proves every layer trims
    # its KV (and C increment) history to its finite window bound.
    length = 2 * context + 7
    padding = (3, 5)
    z = torch.randn(2, length, temporal.config.width)
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[0, :padding[0]] = False
    valid[1, :padding[1]] = False
    origin = 8192
    positions = torch.arange(origin, origin + length, dtype=torch.long).view(1, -1).expand(2, -1)
    with torch.inference_mode():
        full = temporal(z, valid, absolute_positions=positions)
        state = temporal.init_state(2, z.device, z.dtype)
        stepped = []
        for index in range(length):
            item, state = temporal.step(z[:, index], state, valid[:, index], absolute_positions=positions[:, index])
            stepped.append(item)
        stream = torch.stack(stepped, dim=1)
    max_abs = float((full - stream).abs().max())
    if max_abs > TOL:
        raise RuntimeError(f"{task}/{arm}: full/step mismatch {max_abs} > {TOL}")
    expected_positions = [length - padding[0], length - padding[1]]
    if state.positions != expected_positions:
        raise RuntimeError(f"{task}/{arm}: left padding advanced state {state.positions}")
    max_kv = [window - 1 for window in temporal.config.windows]
    if any(lengths > max_kv[layer] for layer, lengths in enumerate(state.valid_lengths) for lengths in lengths):
        raise RuntimeError(f"{task}/{arm}: cache exceeded finite window bound {state.valid_lengths}")

    # Row selection must preserve each source position; reset only clears the
    # requested row, including C's increment cache.
    selected = temporal.select_rows(state, [1, 0])
    if selected.positions != [expected_positions[1], expected_positions[0]]:
        raise RuntimeError(f"{task}/{arm}: select_rows position drift")
    temporal.reset_rows(selected, [0])
    if selected.positions[0] != 0 or selected.positions[1] != expected_positions[0]:
        raise RuntimeError(f"{task}/{arm}: reset_rows scope drift")
    if any(selected.valid_lengths[layer][0] for layer in range(temporal.config.layers)):
        raise RuntimeError(f"{task}/{arm}: reset retained KV")
    if hasattr(selected, "increments") and any(len(selected.increments[layer][0]) for layer in range(temporal.config.layers)):
        raise RuntimeError(f"{task}/{arm}: reset retained C increments")

    pack = {"applicable": arm == "C_learned", "preserved": None}
    if arm == "C_learned":
        with torch.inference_mode():
            states = []
            for row in range(2):
                one = temporal.init_state(1, z.device, z.dtype)
                _, one = temporal.step(z[row:row + 1, 3], one, torch.ones(1, dtype=torch.bool))
                states.append(one)
            packed = UnifiedStreamDecoder(decoder)._pack_states(states)
        lengths = [[len(packed.increments[layer][row]) for row in range(2)] for layer in range(temporal.config.layers)]
        if lengths != [[1, 1]] * temporal.config.layers:
            raise RuntimeError(f"{task}/{arm}: stream pack lost increment cache {lengths}")
        pack = {"applicable": True, "preserved": True, "increment_lengths": lengths}

    return {
        "context": context,
        "sequence_length": length,
        "windows": list(temporal.config.windows),
        "origin": origin,
        "left_padding_per_row": list(padding),
        "expected_positions_before_select": expected_positions,
        "cache_max_prior_kv_per_layer": max_kv,
        "full_step_max_abs": max_abs,
        "within_2e-5": max_abs <= TOL,
        "shared_blocks_sha256_before": before,
        "shared_blocks_sha256_after": after,
        "shared_blocks_byte_equal": before == after,
        "post_stream_state": state_summary(state),
        "select_reset_state": state_summary(selected),
        "stream_pack": pack,
    }


def main() -> int:
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rows = {task: {arm: check(task, arm) for arm in SPECS} for task in ("m1", "m2", "h1")}
    output = WEEK / "receipts" / "architecture" / "unified_temporal_smoke_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "unified_falcon_task_aware_temporal_smoke_v2",
        "status": "COMPLETED",
        "utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "cpu_threads": 4,
        "tolerance_max_abs": TOL,
        "training_started": False,
        "official_test_used": False,
        "dandi_used": False,
        "a1_reuse": "A1Temporal facade, SPECS, and ALIBI_SLOPES only; UnifiedStreamDecoder adds only task-independent C increment pack adaptation.",
        "source_files": {str(HERE / "temporal.py"): file_sha(HERE / "temporal.py"), str(Path(__file__).resolve()): file_sha(Path(__file__).resolve())},
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
