#!/usr/bin/env python3
"""Extract a registered CheXpert Small archive and normalize its metadata."""

from __future__ import annotations

import argparse
import json
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
        validate_raster_image,
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
        validate_raster_image,
        write_metadata_atomic,
    )


OUTPUT_COLUMNS = ["image_id", "path", "patient_id", "view_position", "split", "source_labels"]
BASE_COLUMNS = {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"}


def find_image(raw_dir: Path, source_path: str, lookup: dict[str, Path]) -> Path:
    normalized = source_path.replace("\\", "/")
    candidates = [normalized, normalized.removeprefix("CheXpert-v1.0-small/")]
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    suffix = "/".join(Path(normalized).parts[-3:])
    matches = [path for key, path in lookup.items() if key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely locate CheXpert image: {source_path}")


def build_metadata(raw_dir: Path, dataset_dir: Path, limit: int | None) -> pd.DataFrame:
    csvs = list(raw_dir.rglob("*.csv"))
    train_files = [path for path in csvs if path.name.lower() == "train.csv"]
    valid_files = [path for path in csvs if path.name.lower() in {"valid.csv", "validation.csv"}]
    if not train_files or not valid_files:
        raise FileNotFoundError("Expected train.csv and valid.csv in the CheXpert archive")
    lookup = {
        path.relative_to(raw_dir).as_posix(): path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    frames: list[pd.DataFrame] = []
    for csv_path, split in ((train_files[0], "train"), (valid_files[0], "val")):
        frame = pd.read_csv(csv_path)
        if "Path" not in frame:
            raise ValueError(f"Path column missing from {csv_path}")
        frame["_split"] = split
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    if limit is not None:
        source = source.head(limit).copy()

    rows = []
    label_columns = [column for column in source.columns if column not in BASE_COLUMNS | {"_split"}]
    for _, row in source.iterrows():
        image_path = find_image(raw_dir, str(row["Path"]), lookup)
        image_is_valid, invalid_reason = validate_raster_image(image_path)
        if not image_is_valid:
            raise ValueError(f"Invalid CheXpert image {image_path}: {invalid_reason}")
        parts = Path(str(row["Path"]).replace("\\", "/")).parts
        patient = next(
            (part for part in parts if part.lower().startswith("patient")),
            image_path.parent.parent.name,
        )
        labels = {column: row[column] for column in label_columns if pd.notna(row[column])}
        rows.append(
            {
                # Basenames such as view1_frontal.jpg repeat across patients.
                # The source-relative path is stable and unique within CheXpert.
                "image_id": Path(str(row["Path"]).replace("\\", "/")).with_suffix("").as_posix(),
                "path": relative_metadata_path(image_path, dataset_dir),
                "patient_id": patient,
                "view_position": (
                    row.get("AP/PA")
                    if pd.notna(row.get("AP/PA"))
                    else row.get("Frontal/Lateral", "")
                ),
                "split": row["_split"],
                "source_labels": json.dumps(labels, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--source-path",
        type=Path,
        required=True,
        help="Registered CheXpert Small zip archive",
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
    logger = configure_logging(args.data_dir, "chexpert", args.dry_run)
    if not args.source_path.is_file():
        raise FileNotFoundError(f"CheXpert archive not found: {args.source_path}")
    logger.info("Validated local archive: %s", args.source_path)
    dataset_dir = args.data_dir / "chexpert"
    metadata_path = dataset_dir / "metadata.csv"
    existing_metadata = load_existing_metadata(
        metadata_path,
        OUTPUT_COLUMNS,
        ("image_id",),
        args.fresh,
        logger,
    )
    if args.dry_run:
        logger.info(
            "DRY RUN: would extract archive and merge train/valid assignments"
        )
        return
    ensure_structure(args.data_dir)
    safe_extract(args.source_path, dataset_dir / "raw")
    new_metadata = build_metadata(dataset_dir / "raw", dataset_dir, args.limit)
    merged_metadata = merge_metadata_frames(
        existing_metadata,
        new_metadata,
        OUTPUT_COLUMNS,
        ("image_id",),
    )
    write_metadata_atomic(merged_metadata, metadata_path)
    logger.info(
        "Metadata now contains %d rows (%d row(s) indexed in this run)",
        len(merged_metadata),
        len(new_metadata),
    )


if __name__ == "__main__":
    main()
