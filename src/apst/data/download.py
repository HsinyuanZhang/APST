"""Download FALCON and DANDI 000688 NWB files from DANDI."""
from __future__ import annotations

import argparse
from pathlib import Path

from apst.data.catalog import DANDI688_SUBJECTS, DANDISETS, FALCON_TASKS, TASKS, default_data_root


def download_falcon(
    tasks: list[str] | tuple[str, ...] | None = None,
    *,
    root: str | Path | None = None,
    existing: str = "skip",
    jobs: int = 4,
) -> Path:
    """Download one or more FALCON dandisets into ``<root>/<dandiset_id>/``."""
    from dandi.download import download, DownloadExisting

    selected = tuple(tasks) if tasks else FALCON_TASKS
    unknown = [t for t in selected if t not in FALCON_TASKS]
    if unknown:
        raise ValueError(f"unknown FALCON task(s): {unknown}; expected one of {FALCON_TASKS}")
    dest = Path(root).expanduser().resolve() if root else default_data_root()
    dest.mkdir(parents=True, exist_ok=True)
    existing_enum = DownloadExisting(existing)
    for task in selected:
        dandiset = DANDISETS[task]["dandiset"]
        url = f"https://dandiarchive.org/dandiset/{dandiset}"
        print(f"Downloading FALCON {task.upper()} ({dandiset}) -> {dest / dandiset}")
        download(url, dest, existing=existing_enum, jobs=jobs)
    return dest


def download_dandi688(
    subjects: list[str] | tuple[str, ...] | None = None,
    *,
    root: str | Path | None = None,
    existing: str = "skip",
    jobs: int = 4,
) -> Path:
    """Download DANDI 000688 SUA subjects into ``<root>/000688/<subject>/``.

    Only ``sub-C``, ``sub-M``, and ``sub-J`` are fetched. ``sub-T`` is
    threshold crossings and is excluded.
    """
    from dandi.download import download, DownloadExisting

    meta = DANDISETS["dandi688"]
    selected = tuple(subjects) if subjects else DANDI688_SUBJECTS
    unknown = [s for s in selected if s not in DANDI688_SUBJECTS]
    if unknown:
        raise ValueError(f"unknown DANDI 000688 subject(s): {unknown}; expected one of {DANDI688_SUBJECTS}")
    dest = Path(root).expanduser().resolve() if root else default_data_root()
    out = dest / meta["dandiset"]
    out.mkdir(parents=True, exist_ok=True)
    existing_enum = DownloadExisting(existing)
    version = meta["version"]
    dandiset = meta["dandiset"]
    for subject in selected:
        url = (
            f"https://api.dandiarchive.org/api/dandisets/{dandiset}/"
            f"versions/{version}/assets/?path={subject}"
        )
        print(f"Downloading DANDI 000688 {subject} -> {out / subject}")
        download(url, out, existing=existing_enum, jobs=jobs)
    return out


def download_datasets(
    names: list[str] | tuple[str, ...] | None = None,
    *,
    root: str | Path | None = None,
    existing: str = "skip",
    jobs: int = 4,
    subjects: list[str] | tuple[str, ...] | None = None,
) -> Path:
    selected = tuple(names) if names else TASKS
    unknown = [n for n in selected if n not in TASKS]
    if unknown:
        raise ValueError(f"unknown dataset(s): {unknown}; expected one of {TASKS}")
    dest = Path(root).expanduser().resolve() if root else default_data_root()
    falcon = [n for n in selected if n in FALCON_TASKS]
    if falcon:
        download_falcon(falcon, root=dest, existing=existing, jobs=jobs)
    if "dandi688" in selected:
        download_dandi688(subjects, root=dest, existing=existing, jobs=jobs)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download APST datasets from DANDI.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS),
        default=list(TASKS),
        help="Datasets to download (default: m1 m2 h1 dandi688).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        choices=list(DANDI688_SUBJECTS),
        default=None,
        help="DANDI 000688 subjects (default: sub-C sub-M sub-J).",
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
    download_datasets(
        args.tasks,
        root=args.root,
        existing=args.existing,
        jobs=args.jobs,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
