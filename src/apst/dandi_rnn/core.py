"""One-layer LSTM baselines with an explicit development-to-final boundary.

The module accepts a small, documented NPZ manifest so that fitting is independent
of a checkout path. It uses APST's frozen protocol modules only to validate
formal roster admission; it does not alter APST experiment modules.
``support_<budget>`` arrays contain causal-window *endpoints* whose
complete 50-bin histories are legal labelled calibration support.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

SEED = 42
WINDOW = 50
SLOTS = 100
BATCH = 128
HPS = ((128, 1e-4), (128, 3e-4), (128, 5e-4), (256, 1e-4), (256, 3e-4), (256, 5e-4))
BUDGETS = (4, 8, 16, 32)


class ProtocolError(ValueError):
    """Raised when a portable cache violates the released RNN data contract."""


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Record:
    session_id: str
    neural: np.ndarray
    velocity: np.ndarray
    query: np.ndarray
    support: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        x, y = self.neural, self.velocity
        if x.ndim != 2 or not (0 < x.shape[1] <= SLOTS) or y.shape != (len(x), 2):
            raise ProtocolError(f"{self.session_id}: expected [T,1..100] neural and [T,2] velocity")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ProtocolError(f"{self.session_id}: non-finite data")
        for budget in BUDGETS:
            ends = self.support.get(budget)
            if ends is None or ends.ndim != 1 or not len(ends):
                raise ProtocolError(f"{self.session_id}: missing support_{budget}")
            if np.any(ends < WINDOW - 1) or np.any(ends >= len(x)) or np.intersect1d(ends, self.query).size:
                raise ProtocolError(f"{self.session_id}: invalid/disclosed support_{budget}")
        if self.query.ndim != 1 or not len(self.query) or np.any(self.query < WINDOW - 1) or np.any(self.query >= len(x)):
            raise ProtocolError(f"{self.session_id}: invalid query endpoints")


def load_manifest(path: str | Path, splits: Iterable[str] = ("train", "dev", "final")) -> dict[str, list[Record]]:
    """Load ``.npz`` paths listed under train/dev/final in a JSON manifest."""
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema") != "dandi_rnn_records_v1":
        raise ProtocolError("manifest schema must be dandi_rnn_records_v1")
    result: dict[str, list[Record]] = {}
    subject=raw.get('subject')
    expected=None
    if raw.get('status') != 'SMOKE' and subject in {'sub-C','sub-M'}:
        if subject=='sub-C': from apst.dandi_subc import protocol
        else: from apst.dandi_subm import protocol
        expected={'train':list(protocol.TRAIN_SESSIONS),'dev':list(protocol.DEV_SESSIONS),'final':list(protocol.FINAL_SESSIONS)}
    for split in splits:
        rows = raw.get("splits", {}).get(split)
        if not isinstance(rows, list) or not rows:
            raise ProtocolError(f"manifest has no {split} rows")
        result[split] = []
        for item in rows:
            file = (manifest_path.parent / item["file"]).resolve()
            if item.get("sha256") and item["sha256"] != sha256(file):
                raise ProtocolError(f"cache digest drift: {file}")
            with np.load(file, allow_pickle=False) as z:
                sid = str(z["session_id"].item())
                support = {b: np.asarray(z[f"support_{b}"], np.int64) for b in BUDGETS}
                record = Record(sid, np.asarray(z["neural"], np.float32), np.asarray(z["velocity"], np.float32), np.asarray(z["query"], np.int64), support)
            result[split].append(record)
        if expected is not None and [r.session_id for r in result[split]] != expected[split]:
            raise ProtocolError(f'{split} roster differs from frozen {subject} protocol')
    return result


def causal_windows(record: Record, ends: np.ndarray, *, pad_left: bool = False) -> np.ndarray:
    ends = np.asarray(ends, np.int64)
    if not pad_left and np.any(ends < WINDOW - 1):
        raise ProtocolError("training/scoring endpoint lacks a complete causal history")
    offsets = np.arange(-WINDOW + 1, 1, dtype=np.int64)
    index = ends[:, None] + offsets[None, :]
    if np.any(index >= len(record.neural)):
        raise ProtocolError("window endpoint beyond recording")
    x = np.zeros((len(ends), WINDOW, SLOTS), np.float32)
    valid = index >= 0
    # Advanced indexing flattens the selected time positions; assign only the
    # observed session slots and leave the remaining canonical slots at zero.
    x[:, :, :record.neural.shape[1]][valid] = record.neural[index[valid]]
    return x


class RNN(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(SLOTS, hidden, 1, batch_first=True)
        self.linear = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.rnn(x)[0][:, -1])


def label_stats(records: Iterable[Record], endpoints: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    y = np.concatenate([r.velocity[np.asarray(e, np.int64)] for r, e in zip(records, endpoints)])
    mean = y.mean(0, dtype=np.float64).astype(np.float32)
    std = y.std(0, dtype=np.float64).astype(np.float32); std[std < 1e-8] = 1
    return mean, std


def full_clock_prediction(model: RNN, record: Record, device: torch.device, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    model.eval(); ends = np.arange(len(record.neural), dtype=np.int64); out = np.empty((len(ends), 2), np.float32)
    with torch.no_grad():
        for left in range(0, len(ends), 1024):
            x = torch.from_numpy(causal_windows(record, ends[left:left + 1024], pad_left=True)).to(device)
            out[left:left + len(x)] = model(x).cpu().numpy()
    out = out * std + mean
    # Published local DANDI protocol: causal full-clock EMA, reset per record.
    ema = np.empty(out.shape, np.float64)
    for i, row in enumerate(out.astype(np.float64, copy=False)):
        ema[i] = row if i == 0 else ema[i - 1] / 3 + row * 2 / 3
    return ema


def _array_sha(value: np.ndarray) -> str:
    value=np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def r2(record: Record, prediction: np.ndarray) -> tuple[float, dict[str, object]]:
    y = record.velocity[record.query].astype(np.float64); p = prediction[record.query].astype(np.float64)
    per=r2_score(y,p,multioutput='raw_values')
    score=float(r2_score(y,p,multioutput='variance_weighted'))
    return score,{'r2_per_output':np.asarray(per).tolist(),'query_indices_sha256':_array_sha(record.query),'truth_sha256':_array_sha(y),'prediction_sha256':_array_sha(p),'ema_alpha':1/3,'ema_previous_weight':1/3}


def train(model: RNN, records: list[Record], endpoints: list[np.ndarray], mean: np.ndarray, std: np.ndarray, *, lr: float, updates: int, device: torch.device, seed: int = SEED, state=None):
    """Run updates; pass back state to preserve Adam and sampling RNG across epochs."""
    optimizer, rng = state if state is not None else (torch.optim.Adam(model.parameters(), lr=lr), np.random.default_rng(seed))
    model.train()
    for _ in range(updates):
        i = int(rng.integers(len(records))); ids = endpoints[i]; chosen = rng.choice(ids, BATCH, replace=len(ids) < BATCH)
        x = torch.from_numpy(causal_windows(records[i], chosen)).to(device)
        y = torch.from_numpy((records[i].velocity[chosen] - mean) / std).to(device)
        optimizer.zero_grad(set_to_none=True); loss = nn.functional.mse_loss(model(x), y)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite RNN loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return optimizer, rng


def evaluate(model: RNN, records: list[Record], mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[float, list[dict[str, object]]]:
    rows=[]
    for r in records:
        score,receipt=r2(r,full_clock_prediction(model,r,device,mean,std));rows.append({'session_id':r.session_id,'r2':score,**receipt})
    return float(np.mean([x["r2"] for x in rows])), rows


def choose_development(candidates: list[dict[str, object]]) -> dict[str, object]:
    # Input order is config order; strict comparison preserves earliest exact tie.
    best = candidates[0]
    for candidate in candidates[1:]:
        if float(candidate["dev_r2"]) > float(best["dev_r2"]): best = candidate
    return best
