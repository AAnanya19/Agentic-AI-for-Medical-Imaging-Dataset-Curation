#!/usr/bin/env python3
"""Extract an approved local BraTS archive and index MRI sequences."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from scripts.common import (
        configure_logging,
        ensure_structure,
        load_existing_metadata,
        merge_metadata_frames,
        relative_metadata_path,
        safe_extract,
        validate_nonempty_file,
        write_metadata_atomic,
    )
except ModuleNotFoundError:
    from common import (
        configure_logging,
        ensure_structure,
        load_existing_metadata,
        merge_metadata_frames,
        relative_metadata_path,
        safe_extract,
        validate_nonempty_file,
        write_metadata_atomic,
    )


OUTPUT_COLUMNS = ["patient_id", "path", "split", "sequence_type"]
SEQUENCE_PATTERNS = (
    ("T1ce", re.compile(r"(?:^|[_-])(t1ce|t1c|t1gd)(?:[_.-]|$)", re.I)),
    ("FLAIR", re.compile(r"(?:^|[_-])(flair|t2f)(?:[_.-]|$)", re.I)),
    ("T2", re.compile(r"(?:^|[_-])(t2|t2w)(?:[_.-]|$)", re.I)),
    ("T1", re.compile(r"(?:^|[_-])(t1|t1n|t1w)(?:[_.-]|$)", re.I)),
)


def sequence_type(filename: str) -> str | None:
    for label, pattern in SEQUENCE_PATTERNS:
        if pattern.search(filename):
            return label
    return None


def infer_split(path: Path) -> str:
    for part in path.parts:
        lowered = part.lower()
        if "train" in lowered:
            return "train"
        if "valid" in lowered or lowered == "val":
            return "val"
        if "test" in lowered:
            return "test"
    return "unassigned"


def build_metadata(raw_dir: Path, dataset_dir: Path, limit: int | None) -> pd.DataFrame:
    rows = []
    candidates = sorted(
        path
        for path in raw_dir.rglob("*")
        if (path.name.endswith(".nii.gz") or path.suffix.lower() in {".nii", ".mha"})
        and validate_nonempty_file(path)[0]
    )
    for path in candidates:
        sequence = sequence_type(path.name)
        if sequence is None:
            continue  # Excludes segmentation masks and unrelated volumes.
        rows.append(
            {
                "patient_id": path.parent.name,
                "path": relative_metadata_path(path, dataset_dir),
                "split": infer_split(path.relative_to(raw_dir)),
                "sequence_type": sequence,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise FileNotFoundError("No T1/T1ce/T2/FLAIR volume files were found in the archive")
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--source-path",
        type=Path,
        required=True,
        help="Approved local BraTS .zip/.tar archive",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard existing metadata and rebuild the manifest from this run",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(args.data_dir, "brats", args.dry_run)
    if not args.source_path.is_file():
        raise FileNotFoundError(f"BraTS archive not found: {args.source_path}")
    logger.info("Validated local approved archive: %s", args.source_path)
    dataset_dir = args.data_dir / "brats"
    metadata_path = dataset_dir / "metadata.csv"
    existing_metadata = load_existing_metadata(
        metadata_path,
        OUTPUT_COLUMNS,
        ("path",),
        args.fresh,
        logger,
    )
    if args.dry_run:
        logger.info(
            "DRY RUN: would extract and merge T1/T1ce/T2/FLAIR volume metadata"
        )
        return
    ensure_structure(args.data_dir)
    safe_extract(args.source_path, dataset_dir / "raw")
    new_metadata = build_metadata(dataset_dir / "raw", dataset_dir, args.limit)
    merged_metadata = merge_metadata_frames(
        existing_metadata,
        new_metadata,
        OUTPUT_COLUMNS,
        ("path",),
    )
    write_metadata_atomic(merged_metadata, metadata_path)
    logger.info(
        "Metadata now contains %d rows (%d row(s) indexed in this run)",
        len(merged_metadata),
        len(new_metadata),
    )


if __name__ == "__main__":
    main()
