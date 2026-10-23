#!/usr/bin/env python3
"""Archive legacy unassigned ISIC rows and their raw images."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import date
from pathlib import Path

import pandas as pd

try:
    from scripts.common import (
        configure_logging,
        validate_raster_image,
        write_metadata_atomic,
    )
except ModuleNotFoundError:
    from common import configure_logging, validate_raster_image, write_metadata_atomic


REQUIRED_COLUMNS = ("image_id", "path", "split", "diagnosis", "artifact_flags")


def archive_legacy_rows(
    data_dir: Path,
    expected_count: int | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Move unassigned ISIC rows/files into a recoverable legacy archive."""
    logger = configure_logging(
        data_dir,
        "isic_legacy_archive",
        dry_run,
        log_filename="isic_legacy_archive.log",
    )
    dataset_dir = data_dir / "isic"
    raw_dir = dataset_dir / "raw"
    metadata_path = dataset_dir / "metadata.csv"
    archive_dir = dataset_dir / "legacy_unstratified"
    archive_raw_dir = archive_dir / "raw"
    archive_metadata_path = archive_dir / "metadata.csv"
    archive_readme_path = archive_dir / "README.md"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"ISIC metadata does not exist: {metadata_path}")
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    missing_columns = set(REQUIRED_COLUMNS).difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"ISIC metadata is missing columns: {sorted(missing_columns)}")
    if metadata["image_id"].duplicated().any():
        raise ValueError("ISIC metadata contains duplicate image IDs")

    legacy = metadata.loc[
        metadata["split"].eq("unassigned"), list(REQUIRED_COLUMNS)
    ].copy()
    active = metadata.loc[
        ~metadata["split"].eq("unassigned"), list(REQUIRED_COLUMNS)
    ].copy()
    if expected_count is not None and len(legacy) != expected_count:
        raise ValueError(
            f"Expected {expected_count} legacy rows but found {len(legacy)}"
        )
    if legacy.empty:
        raise ValueError("No split=unassigned ISIC rows were found to archive")
    if archive_dir.exists() and any(archive_dir.iterdir()):
        raise FileExistsError(f"Legacy archive is not empty: {archive_dir}")

    raw_root = raw_dir.resolve()
    moves: list[tuple[Path, Path]] = []
    for row in legacy.itertuples(index=False):
        relative_path = Path(str(row.path))
        if relative_path.is_absolute():
            raise ValueError(f"Legacy path is absolute: {relative_path}")
        source = (dataset_dir / relative_path).resolve()
        if not source.is_relative_to(raw_root):
            raise ValueError(f"Legacy path is outside active raw/: {relative_path}")
        is_valid, invalid_reason = validate_raster_image(source)
        if not is_valid:
            raise ValueError(f"Invalid legacy image {source}: {invalid_reason}")
        target = (archive_dir / relative_path).resolve()
        if not target.is_relative_to(archive_dir.resolve()):
            raise ValueError(f"Unsafe archive target: {relative_path}")
        if target.exists():
            raise FileExistsError(f"Archive target already exists: {target}")
        moves.append((source, target))

    logger.info(
        "Validated %d legacy row(s) and %d retained active row(s)",
        len(legacy),
        len(active),
    )
    if dry_run:
        logger.info("DRY RUN: no metadata or image files were changed")
        return len(legacy), len(active)

    archive_raw_dir.mkdir(parents=True, exist_ok=True)
    backup_path = metadata_path.with_name(
        f"metadata.before_legacy_archive_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    shutil.copy2(metadata_path, backup_path)
    moved: list[tuple[Path, Path]] = []
    created_outputs: list[Path] = []
    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            moved.append((source, target))

        write_metadata_atomic(legacy, archive_metadata_path)
        created_outputs.append(archive_metadata_path)
        readme = (
            "# Legacy unstratified ISIC cohort\n\n"
            f"Archived on {date.today().isoformat()}.\n\n"
            f"This directory contains {len(legacy):,} images downloaded by the early "
            "sequential, unstratified ISIC downloader. API upload/ID ordering produced "
            "a diagnosis-imbalanced cohort, so it was superseded by the seeded, "
            "diagnosis-stratified balanced sampler. The images and their original "
            "metadata are retained here for provenance and reproducibility, but they "
            "are excluded from the active ISIC pipeline.\n"
        )
        readme_temporary = archive_readme_path.with_suffix(".md.tmp")
        readme_temporary.write_text(readme, encoding="utf-8")
        readme_temporary.replace(archive_readme_path)
        created_outputs.append(archive_readme_path)
        write_metadata_atomic(active, metadata_path)
    except Exception:
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                target.replace(source)
        for output in created_outputs:
            output.unlink(missing_ok=True)
        shutil.copy2(backup_path, metadata_path)
        raise

    logger.info("Archived %d legacy image(s) to %s", len(legacy), archive_raw_dir)
    logger.info("Active metadata now contains %d row(s)", len(active))
    logger.info("Pre-archive manifest backup: %s", backup_path)
    return len(legacy), len(active)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    archive_legacy_rows(args.data_dir, args.expected_count, args.dry_run)


if __name__ == "__main__":
    main()
