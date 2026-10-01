"""Portable CPU EvalAI runtime for the M2 E_nope_flat frozen-A candidate.

The payload contains a frozen M33 E0/carrier bank and a fresh stage-2 decoder.
E_nope_flat disables the sinusoidal position encoding only; it has no recency
bias.  The alpha=1/3 recurrence is applied to decoder scores during inference;
the resulting state is divided by five for native M2 units.  It is not part of
training.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve()
# A copied runner lives in /app; a local checkout lives below WEEK/src.
WEEK = HERE.parents[2] if len(HERE.parents) >= 3 else None
if WEEK is not None:
    WORKSPACE = WEEK.parent
    for path in (
        WEEK / "src",
        WORKSPACE / "btransform_unified_v1" / "src",
        WORKSPACE / "btransform_unified_v2" / "src",
        WORKSPACE / "btransform_unified_v2" / "learnable_recency_v1" / "src",
        WORKSPACE / "btransform_unified_v2" / "learnable_recency_v1" / "scripts",
    ):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

from falcon_challenge.interface import BCIDecoder
from a1_m2.nope_temporal_v1 import build_decoder, contract
from a1_m2.temporal_variants import A1M2StreamDecoder
from btransform_unified_v1.bank import TaskBank, array_sha256

ALPHA = 1.0 / 3.0
WEIGHT_EMA_DECAY = 0.9995
PAYLOAD_SCHEMA = "apst_m2_pretrained_E_nope_frozen_A_payload_v1"
DECODER_STATE_SCHEMA = "apst_m2_pretrained_E_nope_frozen_A_selected_ema_decoder_v1"
SOURCE_CHECKPOINT_SHA256 = "b01853a4b68eec80d19cd038a08452062d504fed3df224032cebcc88a83c91b4"
PARENT_ADMISSION_SHA256 = "0ff12e372a5fd9b9c695ce09473ed2d18bdea351f52aaab1e7d44e4e71153acb"
BANK_MANIFEST_SHA256 = "3e3071b1683ef3a4500cfd50e10d0aa1be4970a817ee0e13901a5464eb776209"
SESSION_COUNT = 13


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typed_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("utf-8"))
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class M2PretrainedENoPEFrozenADecoder(BCIDecoder):
    """Streaming M2 decoder bound to E_nope_flat and the frozen-A bank."""

    alpha = ALPHA

    def __init__(self, task_config, model_path, batch_size: int = 1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.task_config = task_config
        self.root = Path(model_path).resolve()
        if not self.root.is_dir():
            raise ValueError(f"payload directory is unavailable: {self.root}")
        self.doc = _json(self.root / "payload_manifest.json")
        self._validate_payload_files()
        self._validate_document()
        selected_path = self.root / "selected_ema_decoder.pt"
        selected = torch.load(selected_path, map_location="cpu", weights_only=False)
        if not isinstance(selected, dict) or selected.get("schema") != DECODER_STATE_SCHEMA:
            raise ValueError("selected decoder state schema drift")
        if selected.get("temporal_contract") != contract("E_nope_flat"):
            raise ValueError("selected E temporal contract drift")
        if float(selected.get("output_ema_alpha", -1.0)) != ALPHA or float(selected.get("weight_ema_decay", -1.0)) != WEIGHT_EMA_DECAY:
            raise ValueError("selected decoder EMA metadata drift")
        final = self.doc["final_decoder"]
        if selected.get("decoder_checkpoint_sha256") != final.get("checkpoint_sha256"):
            raise ValueError("selected checkpoint binding drift")
        state_dict = selected.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("selected decoder state_dict missing")
        self.model = build_decoder("E_nope_flat", seed=42, proj_dim=16)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        if self.model.temporal.spec.use_sinusoidal_pe is not False or self.model.temporal.spec.bias != "none":
            raise ValueError("loaded decoder is not E_nope_flat/no-recency")
        if int(torch.count_nonzero(self.model.temporal.recency_slopes).item()) != 0:
            raise ValueError("E_nope_flat decoder contains nonzero recency slopes")

        sessions = self.doc.get("sessions")
        if not isinstance(sessions, dict) or len(sessions) != SESSION_COUNT:
            raise ValueError("payload must contain exactly 13 calibration sessions")
        self.banks: dict[str, TaskBank] = {}
        self.session_rows: dict[str, dict[str, Any]] = {}
        for session_id, row in sorted(sessions.items()):
            self._load_bank(session_id, row)
        if len(self.banks) != SESSION_COUNT:
            raise ValueError("payload dataset tag hashes are not unique")
        self.runtime: A1M2StreamDecoder | None = None
        self.ids: list[str] = []
        self.ema: torch.Tensor | None = None
        self.initialized: torch.Tensor | None = None

    def _validate_payload_files(self) -> None:
        expected = self.doc.get("payload_files_sha256")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("payload file hash ledger missing")
        actual = {
            str(path.relative_to(self.root)): _sha256_file(path)
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "payload_manifest.json"
        }
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual)); extra = sorted(set(actual) - set(expected))
            raise ValueError(f"payload file set drift: missing={missing}, extra={extra}")
        for relative, digest in expected.items():
            if actual.get(relative) != digest:
                raise ValueError(f"payload file hash drift: {relative}")

    def _validate_document(self) -> None:
        doc = self.doc
        if doc.get("schema") != PAYLOAD_SCHEMA or doc.get("status") != "PREPARATION_NOT_GLOBAL_SELECTION":
            raise ValueError("payload schema/status contract drift")
        if doc.get("candidate") != "E-noPE-frozen-pretrained-A" or doc.get("task") != "M2" or doc.get("seed") != 42:
            raise ValueError("payload candidate contract drift")
        if doc.get("arm") != "E_nope_flat":
            raise ValueError("payload arm must be E_nope_flat")
        final = doc.get("final_decoder")
        if not isinstance(final, dict) or final.get("temporal_contract") != contract("E_nope_flat"):
            raise ValueError("payload E temporal contract drift")
        if (
            final.get("position_encoding") != "none"
            or final.get("recency_bias") != "none"
            or final.get("training_objective") != "raw decoder MSE; output smoothing is inference-only"
            or final.get("selected_checkpoint_replay_verified") is not True
        ):
            raise ValueError("payload final decoder provenance drift")
        output = doc.get("output_ema", {})
        if (
            float(output.get("alpha", -1.0)) != ALPHA
            or output.get("state_0") != "pred_0"
            or output.get("session_reset") is not True
            or output.get("inference_only") is not True
            or output.get("training_included") is not False
            or float(output.get("weight_ema_decay", -1.0)) != WEIGHT_EMA_DECAY
            or output.get("weight_ema_is_distinct") is not True
        ):
            raise ValueError("payload output EMA contract drift")
        source = doc.get("source_pretrained_encoder", {})
        if (
            source.get("checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or source.get("tensor_count") != 8
            or source.get("parent_admission_sha256") != PARENT_ADMISSION_SHA256
            or source.get("available_b3s_not_20260730_champion") is not True
            or source.get("historical_teacher_fit_scope") != "UNKNOWN"
            or source.get("historical_clean_source_only_chain_claim") is not False
            or source.get("no_query_truth_fitting") is not True
            or source.get("producer_source_updates") != 0
            or source.get("export_or_target_parameter_updates") != 0
        ):
            raise ValueError("frozen-A source provenance drift")
        bank = doc.get("source_bank", {})
        if (
            bank.get("parent_admission_sha256") != PARENT_ADMISSION_SHA256
            or bank.get("bank_manifest_sha256") != BANK_MANIFEST_SHA256
            or bank.get("all20_parent_admitted") is not True
            or bank.get("selected_13_E0_carrier_checked") is not True
        ):
            raise ValueError("frozen-A bank provenance drift")
        legal = doc.get("legal_m33_allowlist", {})
        if legal.get("session_count") != SESSION_COUNT or legal.get("trial_ids") != list(range(33)) or legal.get("no_query_arrays_or_labels_read") is not True:
            raise ValueError("legal calibration contract drift")

    def _load_bank(self, session_id: str, row: dict[str, Any]) -> None:
        required = (
            "official_tag_stem", "official_dataset_tag_hash", "payload_e0_file_sha256",
            "payload_carrier_file_sha256", "payload_e0_typed_sha256", "payload_carrier_typed_sha256",
        )
        if any(key not in row for key in required):
            raise ValueError(f"{session_id}: payload bank row incomplete")
        e0_path = self.root / "calibration" / session_id / "E0.npy"
        carrier_path = self.root / "calibration" / session_id / "carrier.npy"
        if _sha256_file(e0_path) != row["payload_e0_file_sha256"] or _sha256_file(carrier_path) != row["payload_carrier_file_sha256"]:
            raise ValueError(f"{session_id}: bank file hash drift")
        e0 = np.load(e0_path, mmap_mode="r", allow_pickle=False)
        carrier = np.load(carrier_path, mmap_mode="r", allow_pickle=False)
        if e0.shape != (96, 50) or e0.dtype != np.float32 or carrier.shape != (96, 4) or carrier.dtype != np.float32:
            raise ValueError(f"{session_id}: bank geometry/dtype drift")
        if _typed_array(e0) != row["payload_e0_typed_sha256"] or _typed_array(carrier) != row["payload_carrier_typed_sha256"]:
            raise ValueError(f"{session_id}: bank typed hash drift")
        tag = str(row["official_dataset_tag_hash"])
        if tag != self.task_config.hash_dataset(Path(str(row["official_tag_stem"])).stem):
            raise ValueError(f"{session_id}: official dataset tag hash drift")
        if tag in self.banks:
            raise ValueError(f"duplicate official dataset tag: {tag}")
        bank = TaskBank(
            session_id,
            np.asarray(e0, dtype=np.float32),
            np.asarray(carrier, dtype=np.float32),
            np.ones(96, dtype=np.bool_),
            np.zeros((0, 1, 96), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            {
                "shape": [96, 50], "trial_count": 33,
                "estimator": "frozen_pretrained_A_E0_carrier",
                "array_sha256": array_sha256(np.asarray(e0)), "budget": 33,
                "source_manifest_sha256": BANK_MANIFEST_SHA256,
                "carrier_sha256": array_sha256(np.asarray(carrier)),
            },
        )
        self.banks[tag] = bank
        self.session_rows[session_id] = row

    def reset(self, dataset_tags=(Path(""),)):
        tags = [self.task_config.hash_dataset(Path(value).stem) for value in dataset_tags]
        if not tags or len(set(tags)) != len(tags) or any(tag not in self.banks for tag in tags):
            raise ValueError("unknown or duplicate dataset tag")
        if len(tags) > int(self.batch_size):
            raise ValueError("reset roster exceeds configured batch size")
        self.ids = tags
        self.runtime = A1M2StreamDecoder(self.model)
        self.ema = None
        self.initialized = None

    def observe(self, neural_observations):
        return None

    def on_done(self, dones):
        # The SDK's done notification does not delimit an independent reset;
        # reset() is the only session boundary, matching the training scorer.
        return None

    def predict(self, neural_observations):
        if self.runtime is None:
            raise RuntimeError("reset must be called before predict")
        x = np.asarray(neural_observations, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 96 or x.shape[0] < 1 or x.shape[0] > len(self.ids):
            raise ValueError("neural observations must be [1..batch,96]")
        if not np.isfinite(x).all():
            raise ValueError("neural observations contain non-finite values")
        n = int(x.shape[0])
        padded = np.zeros((len(self.ids), 96), dtype=np.float32)
        padded[:n] = x
        valid = torch.zeros(len(self.ids), dtype=torch.bool)
        valid[:n] = True
        score = self.runtime.stream_step(torch.from_numpy(padded), [self.banks[tag] for tag in self.ids], self.ids, valid_mask=valid)
        if self.ema is None:
            self.ema = torch.zeros_like(score)
            self.initialized = torch.zeros(len(self.ids), dtype=torch.bool)
        assert self.initialized is not None and self.ema is not None
        active = valid & self.initialized
        fresh = valid & ~self.initialized
        self.ema = torch.where(active[:, None], self.alpha * self.ema + (1.0 - self.alpha) * score, self.ema)
        self.ema = torch.where(fresh[:, None], score, self.ema)
        self.initialized |= valid
        return (self.ema[:n].detach().cpu().numpy() / 5.0).astype(np.float32)


__all__ = ["M2PretrainedENoPEFrozenADecoder", "ALPHA"]
