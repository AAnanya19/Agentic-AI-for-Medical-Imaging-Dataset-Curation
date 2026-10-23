"""Shared helpers for dataset download scripts."""

from __future__ import annotations

import hashlib
import logging
import random
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from PIL import Image


DATASETS = ("chestxray14", "chexpert", "isic", "brats")


def ensure_structure(data_dir: Path) -> None:
    """Create the common directory layout for every supported dataset."""
    for dataset in DATASETS:
        (data_dir / dataset / "raw").mkdir(parents=True, exist_ok=True)
        (data_dir / dataset / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)


def configure_logging(
    data_dir: Path,
    dataset: str,
    dry_run: bool,
    log_filename: str | None = None,
) -> logging.Logger:
    """Log to the terminal and, outside dry-run mode, a named log file.

    Download scripts retain the original ``<dataset>_download.log`` default,
    while other pipeline stages can supply a stage-specific filename.
    """
    logger = logging.getLogger(f"dataset_download.{dataset}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if not dry_run:
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            data_dir / "logs" / (log_filename or f"{dataset}_download.log"),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a zip/tar archive while rejecting path traversal entries."""
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(resolved_destination):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            handle.extractall(destination)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(resolved_destination):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if member.issym() or member.islnk():
                    link_target = (target.parent / member.linkname).resolve()
                    if not link_target.is_relative_to(resolved_destination):
                        raise ValueError(f"Unsafe archive link: {member.name}")
            # Members and link targets were checked above. Avoid tarfile's newer
            # `filter` argument so the project remains compatible with Python 3.9.
            handle.extractall(destination)
        return

    raise ValueError(f"Unsupported archive format: {archive}")


def copy_download_tree(source: Path, destination: Path) -> None:
    """Copy an already-extracted download tree without deleting existing data."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def relative_metadata_path(path: Path, dataset_dir: Path) -> str:
    """Return a portable POSIX path relative to the dataset directory."""
    return path.resolve().relative_to(dataset_dir.resolve()).as_posix()


def validate_nonempty_file(path: Path) -> tuple[bool, str]:
    """Check that a path is a regular, non-empty file."""
    if not path.is_file():
        return False, "file is missing or is not a regular file"
    try:
        if path.stat().st_size <= 0:
            return False, "file is empty"
    except OSError as error:
        return False, f"file could not be inspected: {error}"
    return True, ""


def validate_raster_image(path: Path) -> tuple[bool, str]:
    """Quickly verify a non-empty image without decoding its full pixel array."""
    valid, reason = validate_nonempty_file(path)
    if not valid:
        return valid, reason
    previous_pixel_limit = Image.MAX_IMAGE_PIXELS
    try:
        # ISIC contains legitimate very-large microscopy mosaics. Verification
        # should detect truncation/corruption without rejecting them by size.
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError) as error:
        return False, f"image verification failed: {error}"
    finally:
        Image.MAX_IMAGE_PIXELS = previous_pixel_limit
    return True, ""


def load_existing_metadata(
    metadata_path: Path,
    columns: Sequence[str],
    key_columns: Sequence[str],
    fresh: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load an existing manifest, or return an empty one for a fresh build."""
    if fresh:
        logger.warning(
            "FRESH MODE: existing metadata at %s will be discarded; raw files "
            "will not be deleted",
            metadata_path,
        )
        return pd.DataFrame(columns=columns)
    if not metadata_path.is_file():
        return pd.DataFrame(columns=columns)

    try:
        existing = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame(columns=columns)
    required = set(columns) | set(key_columns)
    missing = required.difference(existing.columns)
    if missing:
        raise ValueError(
            f"Existing metadata is missing required columns: {sorted(missing)}"
        )
    logger.info("Loaded %d existing metadata row(s) from %s", len(existing), metadata_path)
    return existing.loc[:, list(columns)].copy()


def _normalized_key(values: Iterable[Any]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        normalized.append("" if pd.isna(value) else str(value))
    return tuple(normalized)


def merge_metadata_frames(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    columns: Sequence[str],
    key_columns: Sequence[str],
    update_keys: set[tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Preserve existing keyed rows, append new keys, and update explicit keys.

    Existing row order is retained. A duplicate key in ``new_rows`` only replaces
    an existing row when that normalized key is present in ``update_keys``.
    """
    ordered_columns = list(columns)
    keys = list(key_columns)
    for frame_name, frame in (("existing", existing), ("new", new_rows)):
        missing = set(ordered_columns).difference(frame.columns)
        if missing:
            raise ValueError(
                f"{frame_name} metadata is missing columns: {sorted(missing)}"
            )
        missing_keys = set(keys).difference(frame.columns)
        if missing_keys:
            raise ValueError(
                f"{frame_name} metadata is missing key columns: {sorted(missing_keys)}"
            )

    existing_keys = [
        _normalized_key(row)
        for row in existing.loc[:, keys].itertuples(index=False, name=None)
    ]
    keep_existing = []
    seen_existing: set[tuple[str, ...]] = set()
    for key in existing_keys:
        keep_existing.append(key not in seen_existing)
        seen_existing.add(key)
    result = existing.loc[keep_existing, ordered_columns].reset_index(drop=True)
    result = result.astype(object)
    positions = {
        _normalized_key(row): index
        for index, row in enumerate(
            result.loc[:, keys].itertuples(index=False, name=None)
        )
    }

    deduplicated_new = new_rows.drop_duplicates(subset=keys, keep="last")
    updates = update_keys or set()
    append_indices: list[Any] = []
    for row_index, key_values in zip(
        deduplicated_new.index,
        deduplicated_new.loc[:, keys].itertuples(index=False, name=None),
    ):
        key = _normalized_key(key_values)
        if key not in positions:
            append_indices.append(row_index)
            positions[key] = len(result) + len(append_indices) - 1
        elif key in updates:
            result_position = positions[key]
            for column in ordered_columns:
                result.at[result_position, column] = deduplicated_new.at[
                    row_index, column
                ]

    if append_indices:
        appended = deduplicated_new.loc[append_indices, ordered_columns]
        result = pd.concat([result, appended], ignore_index=True)
    return result.loc[:, ordered_columns]


def write_metadata_atomic(metadata: pd.DataFrame, metadata_path: Path) -> None:
    """Replace a CSV manifest atomically after its complete content is ready."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata.to_csv(temporary, index=False)
    temporary.replace(metadata_path)


# ---------------------------------------------------------------------------
# Stratified sampling and group-safe splitting
#
# These helpers were first written for ISIC and are reproduced here verbatim so
# other datasets can reuse them. `download_isic.py` still carries its own
# copies; deduplicating it against this module is safe but should be done as a
# separate change with the test suite run afterwards.
# ---------------------------------------------------------------------------

SPLIT_RATIOS = {"train": 70, "val": 15, "test": 15}


def scoped_seed(seed: int, *parts: str) -> int:
    """Derive a stable per-stratum seed so ordering never affects selection."""
    digest = hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def largest_remainder_allocation(weights: dict[str, int], target: int) -> dict[str, int]:
    """Hamilton apportionment with lexical tie-breaking and an exact total."""
    positive = {key: int(value) for key, value in weights.items() if int(value) > 0}
    if target < 0:
        raise ValueError("target allocation must be non-negative")
    if not positive or target == 0:
        return {key: 0 for key in sorted(positive)}
    denominator = sum(positive.values())
    allocation = {
        key: (target * weight) // denominator for key, weight in positive.items()
    }
    remaining = target - sum(allocation.values())
    remainders = sorted(
        positive,
        key=lambda key: (-((target * positive[key]) % denominator), key),
    )
    for key in remainders[:remaining]:
        allocation[key] += 1
    return {key: allocation[key] for key in sorted(allocation)}


def allocate_sample_counts(
    category_counts: dict[str, int],
    limit: int | None,
    balanced: bool,
) -> dict[str, int]:
    """Allocate a bounded target either proportionally or equally by category."""
    counts = {key: int(value) for key, value in category_counts.items() if int(value) > 0}
    total_available = sum(counts.values())
    target = total_available if limit is None else min(limit, total_available)
    if not balanced:
        return largest_remainder_allocation(counts, target)

    allocation = {key: 0 for key in sorted(counts)}
    remaining = target
    while remaining:
        eligible = [key for key in allocation if allocation[key] < counts[key]]
        if not eligible:
            break
        minimum = min(allocation[key] for key in eligible)
        for key in (key for key in eligible if allocation[key] == minimum):
            allocation[key] += 1
            remaining -= 1
            if remaining == 0:
                break
    return allocation


def select_stratified_ids(
    inventory: dict[str, list[str]],
    limit: int | None,
    balanced: bool,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Seed-shuffle each category and select exactly its allocated quota."""
    counts = {category: len(ids) for category, ids in inventory.items()}
    allocation = allocate_sample_counts(counts, limit, balanced)
    selected: dict[str, list[str]] = {}
    for category in sorted(inventory):
        identifiers = sorted(set(inventory[category]))
        random.Random(scoped_seed(seed, "sample", category)).shuffle(identifiers)
        selected[category] = identifiers[: allocation.get(category, 0)]
    return selected, allocation


def assign_grouped_stratified_splits(
    item_ids: Sequence[str],
    category_by_id: dict[str, str],
    group_by_id: dict[str, str],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits while minimizing per-category target error.

    Every item sharing a ``group_by_id`` value (a lesion for ISIC, a patient for
    ChestX-ray14) lands in exactly one split, which prevents leakage. Within
    that hard constraint, groups are placed greedily so each category's
    train/val/test counts stay as close as possible to ``SPLIT_RATIOS``.
    """
    category_counts = Counter(category_by_id[item_id] for item_id in item_ids)
    targets = {
        category: largest_remainder_allocation(SPLIT_RATIOS, count)
        for category, count in category_counts.items()
    }

    groups: dict[str, list[str]] = defaultdict(list)
    for item_id in item_ids:
        groups[group_by_id[item_id]].append(item_id)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), scoped_seed(seed, "split", item[0]), item[0]),
    )

    current = {
        category: {split: 0 for split in SPLIT_RATIOS} for category in category_counts
    }
    assignment: dict[str, str] = {}
    split_order = tuple(SPLIT_RATIOS)
    for _group_id, group_items in ordered_groups:
        group_categories = Counter(category_by_id[item_id] for item_id in group_items)

        def placement_cost(candidate_split: str) -> float:
            cost = 0.0
            for category, increment in group_categories.items():
                for split in split_order:
                    actual = current[category][split]
                    if split == candidate_split:
                        actual += increment
                    target = targets[category][split]
                    cost += ((actual - target) ** 2) / max(target, 1)
            return cost

        chosen_split = min(
            split_order,
            key=lambda split: (placement_cost(split), split_order.index(split)),
        )
        for category, increment in group_categories.items():
            current[category][chosen_split] += increment
        for item_id in group_items:
            assignment[item_id] = chosen_split

    for group_id, group_items in groups.items():
        if len({assignment[item_id] for item_id in group_items}) != 1:
            raise RuntimeError(f"group leaked across splits: {group_id}")
    return assignment