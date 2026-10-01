#!/usr/bin/env python3
"""Verify release checksums and the file hashes declared by each payload."""
import hashlib
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ("m1", "m2", "h1")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sums():
    sums = ROOT / "SHA256SUMS"
    seen = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*")
        path = ROOT / relative
        check(relative not in seen, f"duplicate checksum entry: {relative}")
        seen.add(relative)
        check(path.is_file(), f"missing checksum target: {relative}")
        check(sha256(path) == expected, f"checksum mismatch: {relative}")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    check(seen == actual, "SHA256SUMS does not cover every release file")


def verify_payload(task):
    submission = json.loads((ROOT / task / "submission.json").read_text(encoding="utf-8"))
    payload = ROOT / task / "build_context" / "payload"
    manifest_path = payload / "payload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(sha256(manifest_path) == submission["payload_manifest_sha256"],
          f"{task}: payload manifest hash differs from submission.json")
    files = manifest.get("payload_files_sha256")
    check(isinstance(files, dict) and files, f"{task}: no payload_files_sha256 map")
    for relative, expected in files.items():
        path = payload / relative
        check(path.is_file(), f"{task}: missing payload file: {relative}")
        check(sha256(path) == expected, f"{task}: payload hash mismatch: {relative}")
    check(sha256(ROOT / task / "checkpoints" / "source_checkpoint.pt") ==
          submission["source_checkpoint_sha256"], f"{task}: source checkpoint hash mismatch")
    check(sha256(payload / "selected_ema_decoder.pt") ==
          submission["packaged_selected_ema_decoder_sha256"],
          f"{task}: packaged decoder hash mismatch")


def main():
    verify_sums()
    for task in TASKS:
        verify_payload(task)
    print("PASS: checksums and all declared payload file hashes verified")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
