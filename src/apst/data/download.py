"""Download FALCON NWB files from DANDI."""
from __future__ import annotations

import argparse
from pathlib import Path

from apst.data.catalog import DANDISETS, TASKS, default_data_root


def download_falcon(
    tasks: list[str] | tuple[str, ...] | None = None,
    *,
    root: str | Path | None = None,
    existing: str = "skip",
    jobs: int = 4,
) -> Path:
    """Download one or more FALCON dandisets into ``<root>/<dandiset_id>/``.

    Uses the official DANDI archive. Files already present are skipped by default.
    """
    from dandi.download import download, DownloadExisting

    selected = tuple(tasks) if tasks else TASKS
    unknown = [t for t in selected if t not in DANDISETS]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; expected one of {TASKS}")
    dest = Path(root).expanduser().resolve() if root else default_data_root()
    dest.mkdir(parents=True, exist_ok=True)
    existing_enum = DownloadExisting(existing)
    for task in selected:
        dandiset = DANDISETS[task]["dandiset"]
        url = f"https://dandiarchive.org/dandiset/{dandiset}"
        print(f"Downloading FALCON {task.upper()} ({dandiset}) -> {dest / dandiset}")
        download(url, dest, existing=existing_enum, jobs=jobs)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download FALCON dandisets from DANDI.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS),
        default=list(TASKS),
        help="Tasks to download (default: m1 m2 h1).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Output directory (default: $APST_DATA_ROOT or ./data).",
    )
    parser.add_argument(
        "--existing",
        default="skip",
        choices=("skip", "error", "overwrite", "overwrite-different", "refresh"),
        help="What to do if a file already exists.",
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    download_falcon(args.tasks, root=args.root, existing=args.existing, jobs=args.jobs)


if __name__ == "__main__":
    main()
