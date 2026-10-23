#!/usr/bin/env python3
"""Download NIH ChestX-ray14 from Kaggle and build a stratified canonical index.

ChestX-ray14 is multi-label: one image can carry several findings (for example
``Cardiomegaly|Effusion``), so there is no single "the diagnosis" to stratify on
the way there is for ISIC. This script derives one stratum per image using a
rarest-label-wins rule (see ``finding_stratum``) and uses it for both sampling
and split assignment. The rule is recorded in the manifest and in the run's
distribution report so the choice is auditable rather than implicit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from scripts.common import (
        assign_grouped_stratified_splits,
        configure_logging,
        ensure_structure,
        load_existing_metadata,
        merge_metadata_frames,
        relative_metadata_path,
        select_stratified_ids,
        validate_raster_image,
        write_metadata_atomic,
    )
except ModuleNotFoundError:  # Allow `python scripts/download_chestxray14.py`.
    from common import (  # type: ignore[no-redef]
        assign_grouped_stratified_splits,
        configure_logging,
        ensure_structure,
        load_existing_metadata,
        merge_metadata_frames,
        relative_metadata_path,
        select_stratified_ids,
        validate_raster_image,
        write_metadata_atomic,
    )


# The full release is ~45 GB / 112,120 images. The official random sample is
# ~5,606 images at original resolution, which is ample for IQA work and avoids
# the ID-ordering bias that a naive "first N images" subset would introduce.
KAGGLE_HANDLES = {
    "sample": "nih-chest-xrays/sample",
    "full": "nih-chest-xrays/data",
}
METADATA_FILENAMES = ("Data_Entry_2017.csv", "sample_labels.csv")
NO_FINDING = "No Finding"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

OUTPUT_COLUMNS = [
    "image_id",
    "path",
    "patient_id",
    "view_position",
    "split",
    "source_labels",
    "finding_stratum",
]
KEY_COLUMNS = ("image_id",)


def find_source_csv(search_root: Path) -> Path:
    """Locate the label CSV shipped with either the full or sample release."""
    for filename in METADATA_FILENAMES:
        candidates = sorted(search_root.rglob(filename))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        f"None of {METADATA_FILENAMES} were found under {search_root}. "
        "Check that the Kaggle download completed and extracted."
    )


def read_source_labels(csv_path: Path) -> pd.DataFrame:
    """Read the NIH label CSV and normalize the columns this script relies on."""
    source = pd.read_csv(csv_path, dtype={"Patient ID": str})
    required = {"Image Index", "Patient ID", "View Position", "Finding Labels"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing expected columns: {sorted(missing)}"
        )
    normalized = pd.DataFrame(
        {
            "image_id": source["Image Index"].astype(str),
            "patient_id": source["Patient ID"].astype(str),
            "view_position": source["View Position"].fillna("").astype(str),
            "source_labels": source["Finding Labels"].fillna("").astype(str),
        }
    )
    return normalized.drop_duplicates(subset="image_id", keep="first")


def split_findings(label_field: str) -> list[str]:
    """Split a pipe-delimited Finding Labels value into individual findings."""
    findings = [part.strip() for part in str(label_field).split("|")]
    return [finding for finding in findings if finding]


def finding_frequencies(label_fields: Iterable[str]) -> dict[str, int]:
    """Count how often each individual finding occurs across the cohort."""
    counter: Counter = Counter()
    for label_field in label_fields:
        counter.update(split_findings(label_field))
    return dict(counter)


def finding_stratum(label_field: str, frequencies: dict[str, int]) -> str:
    """Reduce a multi-label value to one stratum using rarest-label-wins.

    Among the findings present on an image, the globally rarest one becomes the
    stratum, with the label name breaking frequency ties so the result is
    deterministic. Images with no findings form their own ``No Finding`` stratum.

    Rarest-label-wins is preferred over "first listed label" (biased by the
    alphabetical order NIH writes labels in) and over "full label-set string"
    (which fragments the cohort into hundreds of near-empty strata). It also
    maximizes representation of sparse findings such as Hernia, which is
    precisely what balanced sampling is for.
    """
    findings = split_findings(label_field)
    real_findings = [finding for finding in findings if finding != NO_FINDING]
    if not real_findings:
        return NO_FINDING
    return min(real_findings, key=lambda finding: (frequencies.get(finding, 0), finding))


def index_available_images(search_root: Path) -> dict[str, Path]:
    """Map each image filename on disk to its path, ignoring directory layout."""
    return {
        path.name: path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def build_inventory(
    labels: pd.DataFrame,
    available: dict[str, Path],
    frequencies: dict[str, int],
    logger: logging.Logger,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Group the on-disk cohort into strata, reporting anything unavailable."""
    inventory: dict[str, list[str]] = defaultdict(list)
    stratum_by_id: dict[str, str] = {}
    missing = 0
    for row in labels.itertuples(index=False):
        if row.image_id not in available:
            missing += 1
            continue
        stratum = finding_stratum(row.source_labels, frequencies)
        stratum_by_id[row.image_id] = stratum
        inventory[stratum].append(row.image_id)
    if missing:
        logger.warning(
            "%d labelled image(s) are absent from the download and were skipped; "
            "this is expected when the label CSV covers the full release but only "
            "a subset of images is present",
            missing,
        )
    if not stratum_by_id:
        raise RuntimeError(
            "No labelled images were found on disk. Check that the download "
            "completed and that the label CSV matches the downloaded images."
        )
    return dict(inventory), stratum_by_id


def verify_selected_images(
    image_ids: Iterable[str],
    available: dict[str, Path],
    logger: logging.Logger,
) -> list[str]:
    """Pillow-verify only the selected images and drop any that fail.

    Verification is deliberately scoped to the selection rather than the whole
    archive: validating all 112,120 images of the full release would dominate
    runtime for no benefit when only a few thousand are being indexed.
    """
    verified: list[str] = []
    rejected: list[tuple[str, str]] = []
    for image_id in image_ids:
        valid, reason = validate_raster_image(available[image_id])
        if valid:
            verified.append(image_id)
        else:
            rejected.append((image_id, reason))
    if rejected:
        logger.warning(
            "%d selected image(s) failed verification and were excluded, e.g. %s: %s",
            len(rejected),
            rejected[0][0],
            rejected[0][1],
        )
    return verified


def stage_selected_images(
    image_ids: Iterable[str],
    available: dict[str, Path],
    raw_dir: Path,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Copy only the selected images into the dataset's own ``raw/`` directory.

    kagglehub downloads and caches the whole archive regardless, but copying
    only the selection keeps ``data/`` small - the difference between a couple
    of gigabytes and ~45 GB when the full release is used. Copies go to a
    temporary name first and are renamed into place, so an interrupted run
    cannot leave a truncated file that a later run would treat as complete.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    copied = 0
    for image_id in image_ids:
        source = available[image_id]
        destination = raw_dir / image_id
        if destination.is_file() and destination.stat().st_size > 0:
            staged[image_id] = destination
            continue
        if source.resolve() == destination.resolve():
            staged[image_id] = destination
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        copied += 1
        staged[image_id] = destination
    logger.info(
        "Staged %d image(s) into %s (%d newly copied)", len(staged), raw_dir, copied
    )
    return staged


def build_metadata(
    labels: pd.DataFrame,
    staged: dict[str, Path],
    stratum_by_id: dict[str, str],
    splits: dict[str, str],
    dataset_dir: Path,
) -> pd.DataFrame:
    """Assemble the manifest rows for the verified, staged selection."""
    indexed = labels[labels["image_id"].isin(staged)].copy()
    indexed["path"] = [
        relative_metadata_path(staged[image_id], dataset_dir)
        for image_id in indexed["image_id"]
    ]
    indexed["finding_stratum"] = indexed["image_id"].map(stratum_by_id)
    indexed["split"] = indexed["image_id"].map(splits)
    return indexed.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def audit_metadata_files(dataset_dir: Path) -> dict[str, Any]:
    """Report disk/metadata divergence without modifying either."""
    metadata_path = dataset_dir / "metadata.csv"
    raw_dir = dataset_dir / "raw"
    report: dict[str, Any] = {
        "metadata_rows": 0,
        "raw_files": 0,
        "orphaned_rows": [],
        "unindexed_files": [],
        "extensions": {},
    }
    indexed_paths: set[Path] = set()
    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
        report["metadata_rows"] = len(metadata)
        for row in metadata.itertuples(index=False):
            target = (dataset_dir / str(row.path)).resolve()
            try:
                target.relative_to(dataset_dir.resolve())
            except ValueError:
                report["orphaned_rows"].append(
                    {"image_id": row.image_id, "reason": "path escapes dataset dir"}
                )
                continue
            indexed_paths.add(target)
            valid, reason = validate_raster_image(target)
            if not valid:
                report["orphaned_rows"].append(
                    {"image_id": row.image_id, "reason": reason}
                )

    if raw_dir.is_dir():
        extensions: Counter = Counter()
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            report["raw_files"] += 1
            extensions[path.suffix.lower()] += 1
            if path.resolve() not in indexed_paths:
                report["unindexed_files"].append(path.name)
        report["extensions"] = dict(sorted(extensions.items()))
    return report


def run_repair_audit(data_dir: Path, logger: logging.Logger) -> int:
    """Report-only consistency audit; never deletes rows or files."""
    dataset_dir = data_dir / "chestxray14"
    report = audit_metadata_files(dataset_dir)
    logger.info(
        "Repair audit checked %d metadata row(s): %d valid, %d orphaned",
        report["metadata_rows"],
        report["metadata_rows"] - len(report["orphaned_rows"]),
        len(report["orphaned_rows"]),
    )
    logger.info(
        "Raw images: %d (%d unindexed). Extensions: %s",
        report["raw_files"],
        len(report["unindexed_files"]),
        report["extensions"] or "none",
    )
    for entry in report["orphaned_rows"][:10]:
        logger.warning("Orphaned row %s: %s", entry["image_id"], entry["reason"])
    logger.info("Repair audit is report-only; nothing was changed")
    audit_path = data_dir / "logs" / "chestxray14_active_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote audit to %s", audit_path)
    return len(report["orphaned_rows"])


def log_distribution(logger: logging.Logger, title: str, counts: dict[str, int]) -> None:
    logger.info("%s:", title)
    for category in sorted(counts):
        logger.info("  %s: %d", category, counts[category])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--source",
        choices=sorted(KAGGLE_HANDLES),
        default="sample",
        help=(
            "'sample' is the official ~5,606-image random subset (default); "
            "'full' is the complete ~112,120-image, ~45 GB release"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Index at most N images, chosen by stratified sampling (never first-N)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Sample an approximately equal count per finding stratum instead of "
        "following the cohort's natural proportions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling and split seed")
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=15,
        help="Strata with fewer available images than this are reported as rare "
        "and retained at full capacity (never silently excluded)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard existing metadata and rebuild the manifest from this run",
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument("--dry-run", action="store_true")
    execution_mode.add_argument(
        "--repair",
        action="store_true",
        help="Report-only audit of metadata against files on disk",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(args.data_dir, "chestxray14", args.dry_run)
    dataset_dir = args.data_dir / "chestxray14"
    raw_dir = dataset_dir / "raw"
    metadata_path = dataset_dir / "metadata.csv"

    if args.repair:
        ensure_structure(args.data_dir)
        run_repair_audit(args.data_dir, logger)
        return

    # kagglehub authenticates via ~/.kaggle/kaggle.json, the KAGGLE_USERNAME /
    # KAGGLE_KEY environment variables, or anonymously for public datasets, so a
    # missing credentials file is only worth a warning. Let kagglehub itself
    # decide whether the download can actually proceed.
    credentials = Path.home() / ".kaggle" / "kaggle.json"
    has_env_credentials = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    if not credentials.is_file() and not has_env_credentials:
        logger.warning(
            "No Kaggle credentials at %s and KAGGLE_USERNAME/KAGGLE_KEY are unset. "
            "Public datasets usually download anonymously, so continuing. If this "
            "fails with 401/403, create an API token at kaggle.com/settings and "
            "open the dataset page once while logged in.",
            credentials,
        )
    handle = KAGGLE_HANDLES[args.source]
    logger.info("Source=%s handle=%s", args.source, handle)

    existing_metadata = load_existing_metadata(
        metadata_path, OUTPUT_COLUMNS, KEY_COLUMNS, args.fresh, logger
    )

    if args.dry_run:
        logger.info(
            "DRY RUN: would download %s, stage the selected images into %s, and "
            "merge into %s (limit=%s, balanced=%s, seed=%d)",
            handle,
            raw_dir,
            metadata_path,
            args.limit,
            args.balanced,
            args.seed,
        )
        if raw_dir.is_dir() and any(raw_dir.iterdir()):
            logger.info("DRY RUN: %s already contains files", raw_dir)
        return

    ensure_structure(args.data_dir)
    import kagglehub

    cached_path = Path(kagglehub.dataset_download(handle))
    logger.info("Kaggle cache resolved to %s", cached_path)

    source_csv = find_source_csv(cached_path)
    logger.info("Reading labels from %s", source_csv)
    labels = read_source_labels(source_csv)
    logger.info("Label CSV describes %d unique image(s)", len(labels))

    available = index_available_images(cached_path)
    logger.info("Found %d image file(s) in the Kaggle cache", len(available))

    frequencies = finding_frequencies(labels["source_labels"])
    inventory, stratum_by_id = build_inventory(labels, available, frequencies, logger)

    available_counts = {
        stratum: len(image_ids) for stratum, image_ids in inventory.items()
    }
    log_distribution(logger, "Available images per finding stratum", available_counts)

    rare = {
        stratum: count
        for stratum, count in available_counts.items()
        if count < args.rare_threshold
    }
    if rare:
        logger.warning(
            "%d stratum/strata have fewer than %d available images; they are "
            "retained at full capacity, not excluded: %s",
            len(rare),
            args.rare_threshold,
            ", ".join(f"{name} ({count})" for name, count in sorted(rare.items())),
        )

    selected_by_stratum, allocation = select_stratified_ids(
        inventory, args.limit, args.balanced, args.seed
    )
    log_distribution(logger, "Planned selection per stratum", allocation)

    selected_ids = sorted(
        image_id for image_ids in selected_by_stratum.values() for image_id in image_ids
    )
    verified_ids = verify_selected_images(selected_ids, available, logger)
    staged = stage_selected_images(verified_ids, available, raw_dir, logger)

    patient_by_id = labels.set_index("image_id")["patient_id"].astype(str).to_dict()
    splits = assign_grouped_stratified_splits(
        sorted(staged),
        {image_id: stratum_by_id[image_id] for image_id in staged},
        {image_id: patient_by_id[image_id] for image_id in staged},
        args.seed,
    )

    new_metadata = build_metadata(labels, staged, stratum_by_id, splits, dataset_dir)

    if not existing_metadata.empty:
        split_counts = existing_metadata.groupby("patient_id")["split"].nunique()
        conflicting = split_counts[split_counts > 1]
        if not conflicting.empty:
            raise ValueError(
                "Existing ChestX-ray14 metadata already places some patients in "
                "multiple splits; resolve that before merging"
            )
        # A patient already assigned in a previous run keeps that split, so
        # growing the cohort later can never leak a patient across splits.
        previous = (
            existing_metadata.drop_duplicates("patient_id")
            .set_index("patient_id")["split"]
            .to_dict()
        )
        remapped = new_metadata["patient_id"].map(previous)
        if remapped.notna().any():
            logger.info(
                "Preserved the previous split assignment for %d image(s) belonging "
                "to already-known patients",
                int(remapped.notna().sum()),
            )
            new_metadata.loc[remapped.notna(), "split"] = remapped.dropna()

    merged = merge_metadata_frames(
        existing_metadata, new_metadata, OUTPUT_COLUMNS, KEY_COLUMNS
    )
    write_metadata_atomic(merged, metadata_path)

    achieved = dict(Counter(new_metadata["finding_stratum"]))
    log_distribution(logger, "Achieved selection per stratum", achieved)
    logger.info("Split sizes: %s", dict(Counter(new_metadata["split"])))
    logger.info(
        "Metadata now contains %d row(s) (%d indexed in this run)",
        len(merged),
        len(new_metadata),
    )

    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chestxray14_sample_distribution.json").write_text(
        json.dumps(
            {
                "source": args.source,
                "seed": args.seed,
                "balanced": args.balanced,
                "limit": args.limit,
                "stratification_rule": "rarest_present_finding",
                "available": available_counts,
                "planned": allocation,
                "achieved": achieved,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (logs_dir / "chestxray14_rare_strata.json").write_text(
        json.dumps(
            {
                "threshold": args.rare_threshold,
                "exclusion_applied": False,
                "stratum_count": len(rare),
                "strata": [
                    {
                        "finding_stratum": name,
                        "archive_available": count,
                        "selected": int(achieved.get(name, 0)),
                    }
                    for name, count in sorted(rare.items(), key=lambda i: (i[1], i[0]))
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    pd.crosstab(new_metadata["finding_stratum"], new_metadata["split"]).to_csv(
        logs_dir / "chestxray14_split_crosstab.csv"
    )
    logger.info(
        "Wrote distribution, rare-strata, and split-crosstab reports to %s", logs_dir
    )


if __name__ == "__main__":
    main()