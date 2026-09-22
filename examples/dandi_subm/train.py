#!/usr/bin/env python3
"""Run one frozen Sub-M training config without scheduler side effects."""

from __future__ import annotations

import argparse
from pathlib import Path

from apst.dandi_subm.contract import load
from apst.dandi_subm import protocol
from apst.dandi_subm.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="run two source updates")
    parser.add_argument("--smoke-dev-replay", action="store_true", help="also exercise the stage-2 development replay")
    args = parser.parse_args()
    config = load(args.config)
    root = protocol.output_root(config.get("output_root"))
    for key in ("run_dir", "encoder_source"):
        if key in config:
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else root / value)
    encoder = (
        Path(config["encoder_source"]) / "encoder.pt"
        if config["stage"] == "train"
        else None
    )
    run_training(
        config,
        Path(config["run_dir"]),
        encoder_path=encoder,
        device=args.device,
        smoke_updates=2 if args.smoke else None,
        smoke_dev_replay=args.smoke_dev_replay,
    )


if __name__ == "__main__":
    main()
