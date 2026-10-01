"""Sealed JSON receipts (RECEIPT_LAW: atomic replace + chmod 0444 + .sha256 sidecar).

A sealed receipt is written atomically (temp file in the same directory, fsync,
``os.replace``), made read-only (0444), and accompanied by a
``<name>.sha256`` sidecar holding the SHA-256 of the sealed bytes.
:func:`read_sealed` re-hashes the file and refuses to parse on any mismatch
with the sidecar (tamper-evidence). Only ever write under
``btransform_unified_v1/results/``; never touch historical roots.

Identity: B-transformer unified series, NOT SPINT.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import plan

SEALED_MODE = 0o444


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sidecar_path(path: Path | str) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".sha256")


def seal_json(path: Path | str, payload: Any) -> str:
    """Atomically seal ``payload`` as JSON at ``path``; return the sha256 hex.

    The file is rendered deterministically (sorted keys), replaced atomically,
    chmod 0444, and pinned by a ``.sha256`` sidecar.
    """
    p = Path(path)
    plan.require(p.parent != Path(""), "seal target needs a directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    data = text.encode("utf-8")
    digest = _sha256_bytes(data)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, SEALED_MODE)
    os.replace(tmp, p)  # atomic; replaces even a previous 0444 seal
    sidecar = sidecar_path(p)
    sidecar.write_text(f"{digest}  {p.name}\n", encoding="utf-8")
    return digest


def read_sealed(path: Path | str) -> Any:
    """Verify the sidecar hash, then parse and return the sealed payload.

    Raises BTransformerUnifiedError when the file or sidecar is missing, or
    when the file hash does not match the sidecar (tampered).
    """
    p = Path(path)
    plan.require(p.is_file(), f"sealed receipt missing: {p}")
    sidecar = sidecar_path(p)
    plan.require(sidecar.is_file(), f"sealed receipt sidecar missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
    actual = _sha256_bytes(p.read_bytes())
    plan.require(
        expected == actual,
        f"sealed receipt hash mismatch for {p}: sidecar {expected} != file {actual} (tampered?)",
    )
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["seal_json", "read_sealed", "sidecar_path", "SEALED_MODE"]
