"""Executable FALCON training entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="apst-falcon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare", help="build the prepared cache from raw NWB")
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--raw-root", type=Path)
    prepare.add_argument("--tasks", nargs="+", default=["m1", "m2", "h1"])

    init = sub.add_parser("init-config", help="bind a template to a prepared cache")
    init.add_argument("--template", type=Path, required=True)
    init.add_argument("--cache", type=Path, required=True)
    init.add_argument("--dest", type=Path, required=True)

    pretrain = sub.add_parser("pretrain", help="stage 1: association-profile trunk")
    pretrain.add_argument("--config", type=Path, required=True)
    pretrain.add_argument("--dest", type=Path, required=True)
    pretrain.add_argument("--device", default="cpu")
    pretrain.add_argument("--smoke-updates", type=int)

    stage2 = sub.add_parser("stage2", help="stage 2: FiLM + F0 decoder over frozen banks")
    stage2.add_argument("--config", type=Path, required=True)
    stage2.add_argument("--dest", type=Path, required=True)
    stage2.add_argument("--encoder", type=Path, required=True,
                        help="selected epoch_*.pt from a pretrain run")
    stage2.add_argument("--device", default="cpu")
    stage2.add_argument("--smoke-updates", type=int)

    calibrate = sub.add_parser("calibrate", help="export closed-form banks (FiLM baked)")
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--checkpoint", type=Path, required=True)
    calibrate.add_argument("--dest", type=Path, required=True)
    calibrate.add_argument("--split", default="dev", choices=["source", "dev"])
    calibrate.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        from .data import prepare
        print(json.dumps(prepare(args.cache, args.raw_root, tuple(args.tasks)), default=str))
        return

    if args.cmd == "init-config":
        from . import data, protocol
        from .contract import sha
        receipt = args.cache / "prepared_receipt.json"
        if not receipt.is_file():
            raise SystemExit(f"missing {receipt}")
        config = json.loads(args.template.read_text(encoding="utf-8"))
        config["protocol"] = protocol.protocol_dict()
        config["cache"] = str(args.cache.resolve())
        config["prepared_receipt_sha256"] = sha(receipt)
        config["updates_per_epoch"] = data.updates_per_epoch(args.cache, config["task"])
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        args.dest.with_suffix(args.dest.suffix + ".sha256").write_text(
            f"{sha(args.dest)}  {args.dest.name}\n")
        print(args.dest)
        return

    from .contract import load
    from .training import run_pretrain, run_stage2

    if args.cmd == "pretrain":
        config = load(args.config)
        if config["stage"] != "pretrain":
            raise SystemExit(f"pretrain requires config stage=pretrain; got {config['stage']}")
        config["_path"] = args.config
        result = run_pretrain(config, args.dest, device=args.device,
                              smoke_updates=args.smoke_updates)
        print(json.dumps(result, default=str))
        return

    if args.cmd == "stage2":
        config = load(args.config)
        if config["stage"] != "stage2":
            raise SystemExit(f"stage2 requires config stage=stage2; got {config['stage']}")
        config["_path"] = args.config
        result = run_stage2(config, args.dest, encoder_path=args.encoder,
                            device=args.device, smoke_updates=args.smoke_updates)
        print(json.dumps(result, default=str))
        return

    if args.cmd == "calibrate":
        from .training import calibrate_banks
        config = load(args.config)
        config["_path"] = args.config
        manifest = calibrate_banks(config, args.dest, args.checkpoint,
                                   split=args.split, device=args.device)
        print(json.dumps({"status": "COMPLETED", "bank_manifest":
                          str(Path(args.dest) / "bank_manifest.json"),
                          "sessions": sorted(manifest["sessions"])}, default=str))


if __name__ == "__main__":
    main()
