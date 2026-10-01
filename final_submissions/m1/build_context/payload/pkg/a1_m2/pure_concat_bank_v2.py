"""Validation boundary for the incoming pure-concat A M2 bank.

The identity-pretraining worker owns bank materialization.  This module owns
the temporal experiment's read-only admission checks so an encoder regression
cannot silently become an A1 temporal result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_MANIFEST = {
    "schema", "pure_concat_checkpoint_sha256", "encoder_source_sha256", "sessions",
    "source_trunk_sha256", "scale", "calibration_trial_ids", "position_origin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _v1_load_and_validate_manifest(path: Path, *, allow_smoke: bool = False) -> dict[str, Any]:
    """Validate the pure-concat A handoff without opening any hidden data."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_MANIFEST - set(doc))
    if missing:
        raise ValueError(f"pure-concat bank manifest missing {missing}")
    # Architecture is a field-level admission rule.  A prose note such as
    # ``no FiLM`` must not make an otherwise valid manifest fail admission.
    architecture = doc.get("association_profile", doc.get("architecture", {}))
    if isinstance(architecture, Mapping) and architecture.get("film") not in (None, False):
        raise ValueError("pure-concat A manifest declares a FiLM architecture")
    producer_meta = path.parent / "run_meta.json"
    producer_receipt = path.parent / "train_receipt.json"
    if not allow_smoke:
        if not producer_meta.is_file() or not producer_receipt.is_file():
            raise ValueError("formal A1 requires producer run_meta.json and completed train_receipt.json")
        meta = json.loads(producer_meta.read_text(encoding="utf-8"))
        receipt = json.loads(producer_receipt.read_text(encoding="utf-8"))
        if meta.get("status") != "COMPLETED" or receipt.get("status") != "COMPLETED":
            raise ValueError("formal A1 rejects a non-completed or smoke association-profile producer")
        if meta.get("bank_manifest_sha256") != sha256(path):
            raise ValueError("producer run_meta does not bind this exact bank manifest")
        if int(meta.get("selected_epoch", -1)) != int(doc.get("selected_epoch", -2)):
            raise ValueError("producer selected epoch does not bind bank manifest")
    sessions = doc["sessions"]
    if not isinstance(sessions, Mapping) or not sessions:
        raise ValueError("pure-concat bank manifest has no sessions")
    root = path.parent
    quality: dict[str, Any] = {}
    for session, row in sorted(sessions.items()):
        for key in ("e0_path", "carrier_path", "e0_sha256", "carrier_sha256", "query_padded_window_start_path", "query_pad_bins", "query_valid_mask_path"):
            if key not in row:
                raise ValueError(f"{session}: missing {key}")
        e0_path, carrier_path, starts_path, valid_path = (root / row[name] for name in ("e0_path", "carrier_path", "query_padded_window_start_path", "query_valid_mask_path"))
        for file_key, hash_key in (("e0_path", "e0_sha256"), ("carrier_path", "carrier_sha256"),
                                   ("query_padded_window_start_path", "query_padded_window_start_sha256"),
                                   ("query_valid_mask_path", "query_valid_mask_sha256")):
            if hash_key not in row:
                if allow_smoke and file_key in ("query_padded_window_start_path", "query_valid_mask_path"):
                    continue
                raise ValueError(f"{session}: missing {hash_key}")
            if sha256(root / row[file_key]) != row[hash_key]:
                raise ValueError(f"{session}: {file_key} SHA mismatch")
        e0 = np.load(e0_path, mmap_mode="r"); carrier = np.load(carrier_path, mmap_mode="r"); starts = np.load(starts_path, mmap_mode="r"); valid = np.load(valid_path, mmap_mode="r")
        if e0.shape != (96, 50) or carrier.shape != (96, 4) or starts.ndim != 1 or valid.shape != (len(starts), 50) or valid.dtype != np.bool_:
            raise ValueError(f"{session}: expected E0[96,50], carrier[96,4], padded starts[Q], valid[Q,50] bool")
        if e0.dtype != np.float32 or carrier.dtype != np.float32 or not np.isfinite(e0).all() or not np.isfinite(carrier).all():
            raise ValueError(f"{session}: E0/carrier must be finite float32")
        pad = int(row["query_pad_bins"])
        raw = np.asarray(starts, dtype=np.int64)[:, None] - pad + np.arange(50, dtype=np.int64)[None, :]
        if len(starts) == 0 or not np.issubdtype(starts.dtype, np.integer) or np.any((raw < 0) & valid):
            raise ValueError(f"{session}: invalid raw coordinate or left-padding mask")
        norms = np.linalg.norm(np.asarray(e0), axis=1)
        quality[session] = {"e0_shape": list(e0.shape), "carrier_shape": list(carrier.shape),
                            "e0_norm_min": float(norms.min()), "e0_norm_mean": float(norms.mean()),
                            "e0_norm_max": float(norms.max()), "query_count": int(len(starts)),
                            "padded_query_start_min": int(starts.min()), "padded_query_start_max": int(starts.max()),
                            "raw_query_position_min": int(raw.min()), "raw_query_position_max": int(raw.max()), "query_pad_bins": pad}
    return {"manifest": str(path), "manifest_sha256": sha256(path), "allow_smoke": bool(allow_smoke), "pure_concat_checkpoint_sha256": doc["pure_concat_checkpoint_sha256"],
            "encoder_source_sha256": doc["encoder_source_sha256"], "source_trunk_sha256": doc["source_trunk_sha256"],
            "scale": doc["scale"], "quality": quality}

def load_and_validate_manifest(path: Path, *, allow_smoke: bool = False, attestation_path: Path | None = None, attestation_sha256: str | None = None) -> dict[str, Any]:
    if attestation_path is None: return _v1_load_and_validate_manifest(path,allow_smoke=allow_smoke)
    if attestation_sha256 is None or sha256(attestation_path)!=attestation_sha256: raise ValueError('coordinate attestation SHA mismatch')
    att=json.loads(attestation_path.read_text()); doc=json.loads(path.read_text()); meta=json.loads((path.parent/'run_meta.json').read_text()); receipt=json.loads((path.parent/'train_receipt.json').read_text())
    if att.get('schema')!='apst_m2_coordinate_attestation_v1' or att.get('status')!='COMPLETED' or att.get('original_manifest_sha256')!=sha256(path) or att.get('original_run_meta_sha256')!=sha256(path.parent/'run_meta.json') or att.get('original_train_receipt_sha256')!=sha256(path.parent/'train_receipt.json') or att.get('selected_checkpoint_sha256')!=doc.get('pure_concat_checkpoint_sha256'): raise ValueError('coordinate attestation binding mismatch')
    if meta.get('status')!='COMPLETED' or receipt.get('status')!='COMPLETED' or meta.get('bank_manifest_sha256')!=sha256(path) or receipt.get('bank_manifest_sha256')!=sha256(path): raise ValueError('formal producer binding mismatch')
    for key,row in doc['sessions'].items():
      side=att['sessions'].get(key)
      if not side or side['starts_path']!=row['query_padded_window_start_path'] or side['valid_path']!=row['query_valid_mask_path'] or sha256(path.parent/side['starts_path'])!=side['starts_sha256'] or sha256(path.parent/side['valid_path'])!=side['valid_sha256']: raise ValueError(f'{key}: coordinate attestation drift')
    return _v1_load_and_validate_manifest(path,allow_smoke=True)
