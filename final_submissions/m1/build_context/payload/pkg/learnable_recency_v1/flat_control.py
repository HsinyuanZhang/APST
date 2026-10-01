"""Prepared, no-execution manifests for the FULL learned-recency flat control."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

from learnable_recency_v1.config import LearnableRecencyConfig, dataset_config
from learnable_recency_v1.wrap import install_temporal


PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE.parent
WORKSPACE = ROOT.parent
RESULTS = PACKAGE / "results"
TASKS = ("m1", "m2", "h1")
HALF_LIVES = (None,) * 8
SEED = 42
PROJ_DIM = 16
BATCH = 32

_REFERENCE = {
    "m1": {
        "run": RESULTS / "m1_projadd_learned_slope_default_s42",
        "score": RESULTS / "m1_projadd_learned_slope_default_s42" / "score_receipt.json",
        "runner": PACKAGE / "scripts" / "m1_projadd_learnable_train.py",
        "selection": "all24 visible HO3; earliest maximum equal-session mean EMA",
        "results_dir": RESULTS / "m1_projadd_flat_p16_s42",
        "selection_dir": None,
    },
    "m2": {
        "run": RESULTS / "m2_projadd_learned_slope_default_s42",
        "score": RESULTS / "selection_m2_projadd_learned_slope_default_ext6_s42" / "score_receipt.json",
        "runner": PACKAGE / "scripts" / "m2_projadd_learnable_train.py",
        "score_runner": PACKAGE / "scripts" / "m2_projadd_learnable_score.py",
        "selection": "all24 EXT6; earliest maximum finite unweighted equal-session mean",
        "results_dir": RESULTS / "m2_projadd_flat_p16_s42",
        "selection_dir": RESULTS / "selection_m2_projadd_flat_p16_s42_ext6",
    },
    "h1": {
        "run": RESULTS / "h1_learned_slope_default_s42",
        "score": RESULTS / "h1_learned_slope_default_s42" / "ho_m3_selection.json",
        "runner": PACKAGE / "scripts" / "h1_learnable_train.py",
        "selection": "all32 HO-M3 grouped-seven; earliest maximum",
        "results_dir": RESULTS / "h1_flat_p16_s42",
        "selection_dir": None,
    },
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flat_config(task: str) -> LearnableRecencyConfig:
    """Return the existing fixed-tier configuration with all eight slopes zero."""
    if task not in TASKS:
        raise ValueError(f"unknown flat-control task: {task!r}")
    return dataset_config(
        task,
        tier="fixed",
        half_life_seconds=HALF_LIVES,
        ladder="default",
        per_layer=True,
        layers=4,
    )


def verify_zero_temporal(task: str) -> dict[str, Any]:
    """Build only the temporal module and prove the requested bias is exact zero."""
    from btransform_unified_v2.temporal import RiftTemporal

    config = flat_config(task)
    temporal = RiftTemporal(config.temporal_config)
    rebuilt = install_temporal(
        type("DecoderShell", (), {"temporal": temporal})(), config, SEED
    ).temporal
    slopes = rebuilt.recency_slopes.detach().cpu()
    if tuple(slopes.shape) != (8,) or int(slopes.count_nonzero()) != 0:
        raise RuntimeError(f"{task}: flat-control slopes are not exact zeros")
    return {
        "shape": list(slopes.shape),
        "dtype": str(slopes.dtype).replace("torch.", ""),
        "sha256": hashlib.sha256(slopes.numpy().tobytes()).hexdigest(),
        "zero_count": int(slopes.numel()),
    }


def _source_hashes(task: str) -> dict[str, str]:
    reference = _REFERENCE[task]
    paths = [
        Path(reference["runner"]),
        PACKAGE / "src" / "learnable_recency_v1" / "config.py",
        PACKAGE / "src" / "learnable_recency_v1" / "wrap.py",
        PACKAGE / "src" / "learnable_recency_v1" / "temporal.py",
        ROOT / "src" / "btransform_unified_v2" / "config.py",
        ROOT / "src" / "btransform_unified_v2" / "model.py",
        ROOT / "src" / "btransform_unified_v2" / "temporal.py",
        Path(__file__).resolve(),
    ]
    if "score_runner" in reference:
        paths.append(Path(reference["score_runner"]))
    return {str(path): _sha_file(path) for path in paths}


def _reference_contract(task: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    if task == "m1":
        return {
            "identity_interface": meta["identity_interface"],
            "source_contract": meta["source_contract"],
            "source_hashes": meta["source_hashes"],
        }
    if task == "m2":
        return {
            "identity_interface": meta["identity_interface"],
            "frozen_cache_hashes": meta["frozen_cache_hashes"],
            "paired_recency_reference": meta["paired_recency_reference"],
        }
    return {
        "banks_receipt_sha256": meta["banks_receipt_sha256"],
        "pairing_digests": meta["pairing_digests"],
        "reference": meta["reference"],
    }


def _common_flags() -> list[str]:
    return [
        "--tier",
        "fixed",
        "--half-lives",
        ",".join("none" for _ in HALF_LIVES),
        "--ladder",
        "default",
        "--per-layer",
        "--layers",
        "4",
        "--seed",
        str(SEED),
        "--proj-dim",
        str(PROJ_DIM),
        "--device",
        "cuda:0",
    ]


def _argv(task: str, python_executable: str, device: str) -> tuple[list[str], list[str]]:
    reference = _REFERENCE[task]
    runner = str(reference["runner"])
    destination = str(reference["results_dir"])
    flags = _common_flags()
    flags[-1] = device
    train = [python_executable, runner, *flags, "--dest", destination]
    if task == "m2":
        score = [
            python_executable,
            str(reference["score_runner"]),
            "--run-dir",
            destination,
            "--dest",
            str(reference["selection_dir"]),
            "--tier",
            "fixed",
            "--device",
            device,
        ]
    else:
        score = [
            python_executable,
            runner,
            *flags,
            "--stage",
            "score",
            "--dest",
            destination,
        ]
    return train, score


def _parse_runner(path: Path, argv: list[str]) -> Any:
    original_path = list(sys.path)
    saved_src = {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for candidate in (
        PACKAGE / "src",
        ROOT / "src",
        WORKSPACE / "btransform_unified_v1" / "src",
        WORKSPACE / "btransform_unified_v1" / "scripts",
        WORKSPACE / "SPINT-main",
        WORKSPACE,
    ):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spint_main = str(WORKSPACE / "SPINT-main")
    if spint_main in sys.path:
        sys.path.remove(spint_main)
    # The H1 parser's transitive imports use ``src.data`` from SPINT-main.
    sys.path.insert(0, spint_main)
    if path.name == "h1_learnable_train.py":
        # Earlier M1/M2 parser imports may have cached the unrelated
        # btransform_unified_v1 ``src`` namespace.  H1's frozen parser needs
        # SPINT-main's namespace for ``src.data.h1_m4_eb_pilot``.
        for name in tuple(sys.modules):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
    try:
        spec = importlib.util.spec_from_file_location(f"_flat_control_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load parser from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "build_parser"):
            try:
                return module.build_parser().parse_args(argv[2:])
            except SystemExit as error:
                raise ValueError(f"flat-control train argv drift: parser rejected {path.name}") from error

        # The frozen M2 selector defines its parser in ``main``.  Execute only
        # to that parser call: an intercepted parse exits before ``run(args)``.
        import argparse

        class _ParsedArgs(Exception):
            pass

        captured = None
        original_argv = sys.argv
        original_parse = argparse.ArgumentParser.parse_args

        def _parse_and_stop(parser: argparse.ArgumentParser, args=None, namespace=None):
            nonlocal captured
            captured = original_parse(parser, argv[2:] if args is None else args, namespace)
            raise _ParsedArgs()

        sys.argv = list(argv)
        argparse.ArgumentParser.parse_args = _parse_and_stop
        try:
            try:
                runpy.run_path(str(path), run_name="__main__")
            except _ParsedArgs:
                pass
        finally:
            argparse.ArgumentParser.parse_args = original_parse
            sys.argv = original_argv
        if captured is None:
            raise RuntimeError(f"{path}: parser interception did not capture arguments")
        return captured
    finally:
        sys.path[:] = original_path
        for name in tuple(sys.modules):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
        sys.modules.update(saved_src)


def _validate_reference(task: str, meta: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
    config = meta.get("learnable_config", {})
    expected_windows = {"m1": [25, 25, 25, 24], "m2": [13, 12, 12, 12], "h1": [75, 75, 75, 74]}
    expected_epochs = {"m1": 24, "m2": 24, "h1": 32}
    expected_updates = {"m1": 6665, "m2": 3165, "h1": 731}
    if meta.get("status") != "FORMAL" or meta.get("tier") != "learned_slope":
        raise ValueError(f"{task}: reference is not the current formal learned-slope arm")
    if meta.get("seed") != SEED or meta.get("batch") != BATCH:
        raise ValueError(f"{task}: reference seed or batch drift")
    if meta.get("epochs") != expected_epochs[task] or meta.get("updates_per_epoch") != expected_updates[task]:
        raise ValueError(f"{task}: reference update budget drift")
    if meta.get("context_bins") != flat_config(task).context_bins or meta.get("layer_windows") != expected_windows[task]:
        raise ValueError(f"{task}: reference context or D4 windows drift")
    if not isinstance(config, Mapping) or config.get("tier") != "learned_slope":
        raise ValueError(f"{task}: reference learnable config drift")
    if config.get("ladder") != "default" or config.get("per_layer") is not True or config.get("layers") != 4:
        raise ValueError(f"{task}: reference ladder or layer policy drift")
    if config.get("half_life_seconds") != [0.08, 0.16, 0.32, 0.64, 1.28, 2.56, None, None]:
        raise ValueError(f"{task}: reference half-life ladder drift")
    if task != "h1" and (meta.get("task") != task or meta.get("proj_dim") != PROJ_DIM):
        raise ValueError(f"{task}: reference task or P16 identity drift")
    if task != "h1" and meta.get("identity_interface") != "proj_add":
        raise ValueError(f"{task}: reference identity interface drift")
    if task == "h1":
        if selection.get("status") != "HO_M3_DEVELOPMENT_SELECTION" or "selected" not in selection:
            raise ValueError("h1: reference HO-M3 selection is incomplete")
    elif selection.get("status") != "COMPLETED" or "selection" not in selection:
        raise ValueError(f"{task}: reference selection is incomplete")


def _validate_no_partial_or_resume(parsed: Any) -> None:
    for name in ("max_updates_smoke", "smoke_steps", "resume"):
        value = getattr(parsed, name, None)
        if value not in (None, False):
            raise ValueError(f"flat-control partial or resume argv drift: {name}")


def _validate_train_argv(
    manifest: Mapping[str, Any],
    parsed: Any,
    *,
    expected_stage: str | None = None,
) -> None:
    config = manifest["flat_config"]
    expected_half_lives = ",".join("none" for _ in HALF_LIVES)
    required = {
        "tier": "fixed",
        "half_lives": expected_half_lives,
        "ladder": "default",
        "per_layer": True,
        "layers": 4,
        "seed": SEED,
        "proj_dim": PROJ_DIM,
        "dest": Path(manifest["results_dir"]),
    }
    if manifest["task"] in ("m1", "m2"):
        required["epochs"] = manifest["epochs"]
    if manifest["task"] in ("m2", "h1"):
        required["config"] = manifest["task"]
    for key, value in required.items():
        if getattr(parsed, key, None) != value:
            raise ValueError(f"flat-control train argv drift: {key}")
    if config["half_life_seconds"] != [None] * 8:
        raise ValueError("flat-control manifest config drift")
    _validate_no_partial_or_resume(parsed)
    if expected_stage is not None and getattr(parsed, "stage", None) != expected_stage:
        raise ValueError("flat-control train argv drift: stage")


def _validate_score_argv(manifest: Mapping[str, Any], parsed: Any) -> None:
    _validate_no_partial_or_resume(parsed)
    if manifest["task"] == "m2":
        if getattr(parsed, "tier", None) != "fixed":
            raise ValueError("flat-control M2 score tier drift")
        if getattr(parsed, "run_dir", None) != Path(manifest["results_dir"]):
            raise ValueError("flat-control M2 score run directory drift")
        if getattr(parsed, "dest", None) != Path(manifest["selection_dir"]):
            raise ValueError("flat-control M2 score destination drift")
        return
    _validate_train_argv(manifest, parsed)
    if getattr(parsed, "stage", None) != "score":
        raise ValueError("flat-control score stage drift")


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    """Offline parser and manifest invariant check; never calls a runner action."""
    if manifest.get("schema") != "learnable_recency_flat_control_v1":
        raise ValueError("flat-control schema drift")
    if manifest.get("status") != "PREPARED_NOT_TRAINED" or manifest.get("training_started") is not False:
        raise ValueError("flat-control must remain prepared and untrained")
    task = str(manifest.get("task"))
    if task not in TASKS or manifest.get("effective_bias") != "allzero":
        raise ValueError("flat-control task or effective bias drift")
    if manifest.get("flat_config", {}).get("half_life_seconds") != [None] * 8:
        raise ValueError("flat-control half-life contract drift")
    train = _parse_runner(Path(manifest["runner"]), list(manifest["train_argv"]))
    _validate_train_argv(manifest, train, expected_stage="train" if manifest["task"] != "m2" else None)
    score_runner = Path(manifest.get("score_runner", manifest["runner"]))
    score = _parse_runner(score_runner, list(manifest["score_argv"]))
    _validate_score_argv(manifest, score)


def build_manifest(
    task: str,
    *,
    python_executable: str,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Build a JSON-ready plan without creating a destination or invoking a runner."""
    config = flat_config(task)
    reference = _REFERENCE[task]
    run_dir = Path(reference["run"])
    meta_path = run_dir / "run_meta.json"
    score_path = Path(reference["score"])
    meta = _read_json(meta_path)
    selection = _read_json(score_path)
    _validate_reference(task, meta, selection)
    train_argv, score_argv = _argv(task, python_executable, device)
    manifest: dict[str, Any] = {
        "schema": "learnable_recency_flat_control_v1",
        "status": "PREPARED_NOT_TRAINED",
        "training_started": False,
        "task": task,
        "arm": "FULL_FLAT",
        "effective_bias": "allzero",
        "change_from_reference": "only temporal recency slopes: all eight fixed zero",
        "flat_config": {
            "tier": config.tier,
            "half_life_seconds": list(config.half_life_seconds),
            "ladder": config.ladder,
            "per_layer": config.per_layer,
            "layers": config.layers,
            "context_bins": config.context_bins,
            "windows": list(config.temporal_config.windows),
        },
        "zero_temporal": verify_zero_temporal(task),
        "seed": SEED,
        "proj_dim": PROJ_DIM,
        "batch": BATCH,
        "epochs": meta["epochs"],
        "updates_per_epoch": meta["updates_per_epoch"],
        "total_updates": int(meta["epochs"]) * int(meta["updates_per_epoch"]),
        "reference": {
            "run_dir": str(run_dir),
            "run_meta": str(meta_path),
            "run_meta_sha256": _sha_file(meta_path),
            "selection": str(score_path),
            "selection_sha256": _sha_file(score_path),
            "selection_rule": reference["selection"],
        },
        "reference_contract": _reference_contract(task, meta),
        "runner": str(reference["runner"]),
        "score_runner": str(reference.get("score_runner", reference["runner"])),
        "source_hashes": _source_hashes(task),
        "results_dir": str(reference["results_dir"]),
        "selection_dir": (
            None if reference["selection_dir"] is None else str(reference["selection_dir"])
        ),
        "train_argv": train_argv,
        "score_argv": score_argv,
        "execution": "manifest construction does not execute subprocesses or create results directories",
    }
    verify_manifest(manifest)
    return manifest


__all__ = [
    "TASKS",
    "build_manifest",
    "flat_config",
    "verify_manifest",
    "verify_zero_temporal",
]
