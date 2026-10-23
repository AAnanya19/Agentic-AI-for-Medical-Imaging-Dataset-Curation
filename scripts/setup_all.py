#!/usr/bin/env python3
"""Create the dataset layout and optionally run selected downloaders."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from scripts.common import DATASETS, configure_logging, ensure_structure
except ModuleNotFoundError:
    from common import DATASETS, configure_logging, ensure_structure


METADATA_COLUMNS = {
    "chestxray14": "image_id,path,patient_id,view_position,split,source_labels,finding_stratum\n",
    "chexpert": "image_id,path,patient_id,view_position,split,source_labels\n",
    "isic": "image_id,path,split,diagnosis,artifact_flags\n",
    "brats": "patient_id,path,split,sequence_type\n",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets",
        help=(
            "Comma-separated downloaders to run "
            "(chestxray14,chexpert,isic,brats); omit to only scaffold"
        ),
    )
    parser.add_argument("--chexpert-source", type=Path, help="Local registered CheXpert archive")
    parser.add_argument("--brats-source", type=Path, help="Local approved BraTS archive")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Pass fresh-manifest mode to selected downloaders",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def selected_datasets(value: str | None) -> list[str]:
    if not value:
        return []
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    return selected


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(args.data_dir, "setup_all", args.dry_run)
    selected = selected_datasets(args.datasets)

    if args.dry_run:
        logger.info(
            "DRY RUN: would create the complete directory and metadata scaffold at %s",
            args.data_dir,
        )
    else:
        ensure_structure(args.data_dir)
        for dataset, header in METADATA_COLUMNS.items():
            metadata = args.data_dir / dataset / "metadata.csv"
            if not metadata.exists():
                metadata.write_text(header, encoding="utf-8")
        logger.info("Created complete dataset scaffold at %s", args.data_dir)

    script_dir = Path(__file__).resolve().parent
    for dataset in selected:
        command = [
            sys.executable,
            str(script_dir / f"download_{dataset}.py"),
            "--data-dir",
            str(args.data_dir),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.dry_run:
            command.append("--dry-run")
        if args.fresh:
            command.append("--fresh")
        if dataset == "chexpert":
            if args.chexpert_source is None:
                raise ValueError("--chexpert-source is required when selecting chexpert")
            command.extend(["--source-path", str(args.chexpert_source)])
        if dataset == "brats":
            if args.brats_source is None:
                raise ValueError("--brats-source is required when selecting brats")
            command.extend(["--source-path", str(args.brats_source)])
        logger.info("Running %s downloader", dataset)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
