#!/usr/bin/env python3
"""Build a stratified ISIC Archive cohort through API v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode, urljoin, urlparse

import pandas as pd
import requests
from tqdm import tqdm

try:
    from scripts.common import (
        configure_logging,
        ensure_structure,
        load_existing_metadata,
        merge_metadata_frames,
        relative_metadata_path,
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
        validate_raster_image,
        write_metadata_atomic,
    )


API_URL = "https://api.isic-archive.com/api/v2/images/"
SEARCH_API_URL = "https://api.isic-archive.com/api/v2/images/search/"
OUTPUT_COLUMNS = ["image_id", "path", "split", "diagnosis", "artifact_flags"]
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
SPLIT_RATIOS = {"train": 70, "val": 15, "test": 15}
UNKNOWN_CATEGORY = "Unknown"
SEARCHABLE_FIELDS = frozenset(
    {
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_4",
        "diagnosis_5",
        "age_approx",
        "sex",
        "diagnosis_confirm_type",
        "personal_hx_mm",
        "family_hx_mm",
        "clin_size_long_diam_mm",
        "melanocytic",
        "acquisition_day",
        "image_type",
        "dermoscopic_type",
        "mel_mitotic_index",
        "mel_thick_mm",
        "mel_ulcer",
    }
)
DIAGNOSIS_PARENT_RULES = (
    ("melanoma", "Melanoma"),
    ("basal cell carcinoma", "Basal cell carcinoma"),
    ("squamous cell carcinoma", "Squamous cell carcinoma"),
    ("seborrheic keratosis", "Seborrheic keratosis"),
    ("actinic keratosis", "Actinic keratosis"),
    ("dermatofibroma", "Dermatofibroma"),
    ("nevus", "Nevus"),
    ("naevus", "Nevus"),
    ("vascular", "Vascular lesion"),
)


def request_with_retry(
    session: requests.Session,
    url: str,
    logger: logging.Logger,
    retries: int,
    timeout: float,
    stream: bool = False,
) -> requests.Response:
    """GET a URL, retrying only transport and transient HTTP failures."""
    for attempt in range(retries + 1):
        response: requests.Response | None = None
        failure: Exception | None = None
        try:
            response = session.get(url, timeout=timeout, stream=stream)
        except (requests.ConnectionError, requests.Timeout, OSError) as error:
            failure = error
        except requests.RequestException:
            # Invalid URLs and other local request errors will not improve on retry.
            raise
        else:
            if response.status_code < 400:
                return response
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                if response.status_code not in TRANSIENT_STATUS_CODES:
                    raise
                failure = error

        if attempt == retries:
            assert failure is not None
            raise failure
        retry_after = response.headers.get("Retry-After") if response is not None else None
        delay = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        logger.warning("Request failed (%s); retrying in %ds", failure, delay)
        time.sleep(delay)
    raise RuntimeError("Unreachable retry state")


def iter_images(
    session: requests.Session,
    logger: logging.Logger,
    retries: int,
    timeout: float,
    page_size: int,
    query: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield lightweight image records from listing or filtered search pages."""
    endpoint = SEARCH_API_URL if query else API_URL
    parameters: dict[str, Any] = {"limit": page_size}
    if query:
        parameters["query"] = query
    url: str | None = f"{endpoint}?{urlencode(parameters)}"
    while url:
        payload = request_with_retry(session, url, logger, retries, timeout).json()
        if isinstance(payload, list):
            yield from payload
            return
        records = payload.get("results") or payload.get("data") or []
        yield from records
        next_url = payload.get("next")
        url = urljoin(url, next_url) if next_url else None


def nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def full_image_url(record: dict[str, Any]) -> str:
    """Read the public original-image URL supplied by the API listing."""
    url = nested(record, "files", "full", "url")
    if not isinstance(url, str) or not url.startswith("https://"):
        image_id = record.get("isic_id") or record.get("_id") or record.get("id") or "unknown"
        raise ValueError(f"ISIC API record {image_id} has no files.full.url")
    return url


def image_id_for(record: dict[str, Any]) -> str:
    """Return a normalized image identifier, or an empty string if absent."""
    return str(record.get("isic_id") or record.get("_id") or record.get("id") or "")


def detailed_diagnosis(record: dict[str, Any]) -> str:
    """Return the most specific diagnosis while retaining the archive wording."""
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    clinical = metadata.get("clinical") if isinstance(metadata.get("clinical"), dict) else {}
    levels = [
        str(clinical[f"diagnosis_{level}"]).strip()
        for level in range(1, 6)
        if clinical.get(f"diagnosis_{level}") not in (None, "")
    ]
    if levels:
        return levels[-1]
    return str(
        clinical.get("diagnosis")
        or metadata.get("diagnosis")
        or record.get("diagnosis")
        or ""
    ).strip()


def diagnosis_category(record: dict[str, Any]) -> str:
    """Collapse detailed diagnoses to a stable entity-level parent category."""
    clinical = nested(record, "metadata", "clinical")
    entity = clinical.get("diagnosis_3") if isinstance(clinical, dict) else None
    value = str(entity or detailed_diagnosis(record)).strip()
    if not value:
        return UNKNOWN_CATEGORY
    lowered = value.casefold()
    for fragment, parent in DIAGNOSIS_PARENT_RULES:
        if fragment in lowered:
            return parent
    parent = value.split(",", 1)[0].strip()
    return parent or UNKNOWN_CATEGORY


def _stable_scalar(value: Any) -> str:
    if value is None or value == "":
        return UNKNOWN_CATEGORY
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).strip() or UNKNOWN_CATEGORY


def stratification_category(record: dict[str, Any], stratify_by: str) -> str:
    """Resolve a reusable stratification field from common API containers."""
    if stratify_by == "diagnosis":
        return diagnosis_category(record)
    if "." in stratify_by:
        value: Any = record
        for key in stratify_by.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        return _stable_scalar(value)

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    clinical = metadata.get("clinical") if isinstance(metadata.get("clinical"), dict) else {}
    acquisition = (
        metadata.get("acquisition")
        if isinstance(metadata.get("acquisition"), dict)
        else {}
    )
    for container in (record, metadata, clinical, acquisition):
        if stratify_by in container:
            return _stable_scalar(container.get(stratify_by))
    return UNKNOWN_CATEGORY


def lesion_group_id(record: dict[str, Any]) -> str:
    """Resolve a lesion group, giving missing IDs a unique per-image fallback."""
    candidates = (
        record.get("lesion_id"),
        nested(record, "metadata", "lesion_id"),
        nested(record, "metadata", "clinical", "lesion_id"),
    )
    lesion_id = next((value for value in candidates if value not in (None, "")), None)
    return f"lesion:{lesion_id}" if lesion_id is not None else f"image:{image_id_for(record)}"


def build_candidate_inventory(
    records: Iterator[dict[str, Any]] | list[dict[str, Any]],
    stratify_by: str,
) -> dict[str, list[str]]:
    """Index usable image IDs by category without retaining full image records."""
    inventory: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        image_id = image_id_for(record)
        if not image_id or image_id in seen:
            continue
        try:
            full_image_url(record)
        except ValueError:
            continue
        seen.add(image_id)
        inventory[stratification_category(record, stratify_by)].append(image_id)
    return {category: sorted(ids) for category, ids in sorted(inventory.items())}


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


def _scoped_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


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
        random.Random(_scoped_seed(seed, "sample", category)).shuffle(identifiers)
        selected[category] = identifiers[: allocation.get(category, 0)]
    return selected, allocation


def category_search_query(stratify_by: str, category: str) -> str | None:
    """Build a conservative API search query; callers still post-filter results."""
    if category == UNKNOWN_CATEGORY:
        return None
    if stratify_by == "diagnosis":
        first_word = re.search(r"[A-Za-z0-9]+", category)
        return f"diagnosis_3:{first_word.group(0)}*" if first_word else None
    if stratify_by not in SEARCHABLE_FIELDS:
        return None
    return f"{stratify_by}:{json.dumps(category, ensure_ascii=False)}"


def fetch_selected_records(
    session: requests.Session,
    logger: logging.Logger,
    selected_ids: dict[str, list[str]],
    stratify_by: str,
    retries: int,
    timeout: float,
    page_size: int,
) -> list[dict[str, Any]]:
    """Materialize selected metadata through category-filtered scans and detail fallback."""
    selected_records: list[dict[str, Any]] = []
    for category in sorted(selected_ids):
        ordered_ids = selected_ids[category]
        wanted = set(ordered_ids)
        found: dict[str, dict[str, Any]] = {}
        query = category_search_query(stratify_by, category)
        if query and wanted:
            logger.info("Querying category %s with %s", category, query)
            for record in iter_images(
                session,
                logger,
                retries,
                timeout,
                page_size,
                query=query,
            ):
                image_id = image_id_for(record)
                if (
                    image_id in wanted
                    and stratification_category(record, stratify_by) == category
                ):
                    found[image_id] = record
                    if len(found) == len(wanted):
                        break

        missing = [image_id for image_id in ordered_ids if image_id not in found]
        if missing:
            logger.info(
                "Retrieving %d selected %s record(s) through the detail endpoint",
                len(missing),
                category,
            )
        for image_id in missing:
            detail_url = urljoin(API_URL, f"{image_id}/")
            record = request_with_retry(
                session, detail_url, logger, retries, timeout
            ).json()
            found[image_id] = record
        selected_records.extend(found[image_id] for image_id in ordered_ids)
    return selected_records


def assign_stratified_splits(
    records: list[dict[str, Any]],
    stratify_by: str,
    seed: int,
) -> dict[str, str]:
    """Assign lesion-safe splits while minimizing per-category target error."""
    category_by_id = {
        image_id_for(record): stratification_category(record, stratify_by)
        for record in records
    }
    category_counts = Counter(category_by_id.values())
    targets = {
        category: largest_remainder_allocation(SPLIT_RATIOS, count)
        for category, count in category_counts.items()
    }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[lesion_group_id(record)].append(record)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            _scoped_seed(seed, "split", item[0]),
            item[0],
        ),
    )

    current = {
        category: {split: 0 for split in SPLIT_RATIOS}
        for category in category_counts
    }
    assignment: dict[str, str] = {}
    split_order = tuple(SPLIT_RATIOS)
    for group_id, group_records in ordered_groups:
        group_categories = Counter(
            category_by_id[image_id_for(record)] for record in group_records
        )

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
        for record in group_records:
            assignment[image_id_for(record)] = chosen_split

    for group_id, group_records in groups.items():
        group_splits = {assignment[image_id_for(record)] for record in group_records}
        if len(group_splits) != 1:
            raise RuntimeError(f"lesion group leaked across splits: {group_id}")
    return assignment


def metadata_row(
    record: dict[str, Any],
    image_path: Path,
    dataset_dir: Path,
    split: str,
) -> dict[str, str]:
    image_id = image_id_for(record)
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    acquisition = (
        metadata.get("acquisition")
        if isinstance(metadata.get("acquisition"), dict)
        else {}
    )
    diagnosis = detailed_diagnosis(record)

    # Preserve documented acquisition/QC fields without inventing binary labels.
    artifacts: dict[str, Any] = {}
    image_type = (
        acquisition.get("image_type")
        or metadata.get("image_type")
        or record.get("image_type")
    )
    if image_type is not None:
        artifacts["image_type"] = image_type
    artifact_containers = (
        ("qc", metadata.get("qc")),
        ("quality", metadata.get("quality")),
        ("qc", record.get("qc")),
    )
    for container_name, container in artifact_containers:
        if isinstance(container, dict):
            artifacts[container_name] = container
    for key, value in metadata.items():
        is_quality_field = any(
            token in key.lower() for token in ("artifact", "quality", "qc")
        )
        if is_quality_field and value is not None:
            artifacts[key] = value
    return {
        "image_id": image_id,
        "path": relative_metadata_path(image_path, dataset_dir),
        "split": split,
        "diagnosis": str(diagnosis),
        "artifact_flags": json.dumps(artifacts, separators=(",", ":"), sort_keys=True),
    }


def extension_for(record: dict[str, Any], download_url: str) -> str:
    """Choose an extension from the authoritative file URL, then legacy names."""
    name = urlparse(download_url).path or str(record.get("name") or record.get("filename") or "")
    suffix = Path(name).suffix.lower()
    return suffix if suffix in SUPPORTED_IMAGE_EXTENSIONS else ".jpg"


def download_image_record(
    session: requests.Session,
    record: dict[str, Any],
    raw_dir: Path,
    logger: logging.Logger,
    retries: int,
    timeout: float,
) -> Path | None:
    """Download one image through its API-supplied URL, or reuse its real file.

    This is the only function that writes ISIC image bytes. Metadata is never
    consulted for resume decisions: a record is complete only when its expected
    final path is an existing regular file.
    """
    image_id = image_id_for(record)
    if not image_id:
        logger.error("Skipping API record without an image ID")
        return None
    try:
        download_url = full_image_url(record)
    except ValueError as error:
        logger.error("Skipping %s: %s", image_id, error)
        return None

    image_path = raw_dir / f"{image_id}{extension_for(record, download_url)}"
    existing_is_valid, existing_reason = validate_raster_image(image_path)
    if existing_is_valid:
        logger.info("Skipping already-downloaded image %s", image_id)
        return image_path
    if image_path.exists():
        logger.warning(
            "Existing image %s is invalid (%s); downloading a verified replacement",
            image_id,
            existing_reason,
        )

    temporary = image_path.with_suffix(image_path.suffix + ".part")
    try:
        response = request_with_retry(
            session,
            download_url,
            logger,
            retries,
            timeout,
            stream=True,
        )
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        finally:
            response.close()
        downloaded_is_valid, invalid_reason = validate_raster_image(temporary)
        if not downloaded_is_valid:
            logger.error(
                "Failed to download %s: downloaded content is invalid (%s)",
                image_id,
                invalid_reason,
            )
            temporary.unlink(missing_ok=True)
            return None
        temporary.replace(image_path)
    except (requests.RequestException, OSError) as error:
        temporary.unlink(missing_ok=True)
        logger.error("Failed to download %s: %s", image_id, error)
        return None

    final_is_valid, final_reason = validate_raster_image(image_path)
    if not final_is_valid:
        logger.error(
            "Failed to download %s: final file is invalid (%s)",
            image_id,
            final_reason,
        )
        return None
    return image_path


def audit_metadata_files(
    dataset_dir: Path,
    metadata_path: Path,
) -> tuple[int, list[dict[str, Any]]]:
    """Return metadata row count and references without a safe regular file."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ISIC metadata file does not exist: {metadata_path}")
    frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if "path" not in frame.columns:
        raise ValueError(f"ISIC metadata has no path column: {metadata_path}")

    dataset_root = dataset_dir.resolve()
    orphaned: list[dict[str, Any]] = []
    for row_offset, (_, row) in enumerate(frame.iterrows(), start=2):
        image_id = str(row.get("image_id", "")).strip()
        stored_path = str(row.get("path", "")).strip()
        resolved_path, reason = validate_metadata_reference(
            dataset_dir,
            dataset_root,
            stored_path,
        )

        if reason:
            orphaned.append(
                {
                    "row_number": row_offset,
                    "image_id": image_id,
                    "path": stored_path,
                    "resolved_path": str(resolved_path) if resolved_path else "",
                    "reason": reason,
                }
            )
    return len(frame), orphaned


def validate_metadata_reference(
    dataset_dir: Path,
    dataset_root: Path,
    stored_path: str,
) -> tuple[Path | None, str]:
    """Resolve and validate one portable metadata path inside a dataset root."""
    if not stored_path:
        return None, "blank path"
    candidate = Path(stored_path)
    if candidate.is_absolute():
        return None, "absolute path is outside the portable metadata convention"
    try:
        resolved_path = (dataset_dir / candidate).resolve()
        if not resolved_path.is_relative_to(dataset_root):
            return resolved_path, "path resolves outside the ISIC dataset directory"
        is_valid, validation_reason = validate_raster_image(resolved_path)
        return resolved_path, "" if is_valid else validation_reason
    except (OSError, RuntimeError, ValueError) as error:
        return None, f"path could not be resolved: {error}"


def run_repair_audit(data_dir: Path, logger: logging.Logger) -> int:
    """Report manifest/disk consistency without changing local data."""
    dataset_dir = data_dir / "isic"
    raw_dir = dataset_dir / "raw"
    metadata_path = dataset_dir / "metadata.csv"
    total_rows, orphaned = audit_metadata_files(dataset_dir, metadata_path)
    frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    dataset_root = dataset_dir.resolve()
    indexed_paths: set[Path] = set()
    for stored_path in frame["path"].astype(str):
        candidate = Path(stored_path.strip())
        if candidate.is_absolute():
            continue
        try:
            resolved = (dataset_dir / candidate).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_relative_to(dataset_root):
            indexed_paths.add(resolved)
    raw_files = (
        {
            path.resolve()
            for path in raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
        if raw_dir.is_dir()
        else set()
    )
    suffix_counts = Counter(
        Path(stored_path).suffix.lower()
        for stored_path in frame["path"].astype(str)
    )
    non_jpg_count = sum(
        count for suffix, count in suffix_counts.items() if suffix != ".jpg"
    )
    logger.info(
        "Repair audit checked %d metadata row(s): %d valid, %d orphaned",
        total_rows,
        total_rows - len(orphaned),
        len(orphaned),
    )
    logger.info("Metadata rows: %d", total_rows)
    logger.info("Files on disk: %d", len(raw_files))
    logger.info("On disk but NOT in metadata: %d", len(raw_files - indexed_paths))
    logger.info("In metadata but file missing/invalid: %d", len(orphaned))
    logger.info("Non-.jpg active images: %d", non_jpg_count)
    for suffix, count in sorted(suffix_counts.items()):
        logger.info("  extension %s: %d", suffix or "<none>", count)
    for orphan in orphaned:
        logger.warning(
            "Orphaned metadata row %d: image_id=%s path=%s (%s)",
            orphan["row_number"],
            orphan["image_id"] or "<missing>",
            orphan["path"] or "<blank>",
            orphan["reason"],
        )
    logger.info("Repair audit is report-only; metadata and raw files were not changed")
    return len(orphaned)


def rebuild_metadata_from_csv(
    data_dir: Path,
    source: str,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Merge valid cached metadata into the live manifest, keeping a backup."""
    dataset_dir = data_dir / "isic"
    metadata_path = dataset_dir / "metadata.csv"
    existing = load_existing_metadata(
        metadata_path,
        OUTPUT_COLUMNS,
        ("image_id",),
        fresh=False,
        logger=logger,
    )
    if source == "-":
        recovered = pd.read_csv(sys.stdin, dtype=str, keep_default_na=False)
        source_description = "standard input"
    else:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Recovery metadata CSV does not exist: {source_path}")
        recovered = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        source_description = str(source_path)
    missing = set(OUTPUT_COLUMNS).difference(recovered.columns)
    if missing:
        raise ValueError(f"Recovery metadata is missing columns: {sorted(missing)}")
    recovered = recovered.loc[:, OUTPUT_COLUMNS].copy()

    dataset_root = dataset_dir.resolve()

    def valid_row_mask(frame: pd.DataFrame, label: str) -> list[bool]:
        mask: list[bool] = []
        rejected = 0
        for stored_path in frame["path"].astype(str):
            _, reason = validate_metadata_reference(
                dataset_dir,
                dataset_root,
                stored_path.strip(),
            )
            is_valid = not reason
            mask.append(is_valid)
            rejected += int(not is_valid)
        if rejected:
            logger.warning(
                "Excluded %d %s metadata row(s) whose image is missing or invalid",
                rejected,
                label,
            )
        return mask

    existing = existing.loc[valid_row_mask(existing, "existing")].copy()
    recovered = recovered.loc[valid_row_mask(recovered, "recovery")].copy()
    merged = merge_metadata_frames(
        existing,
        recovered,
        OUTPUT_COLUMNS,
        ("image_id",),
    )

    if metadata_path.is_file():
        backup_path = metadata_path.with_name(
            f"metadata.before_rebuild_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        shutil.copy2(metadata_path, backup_path)
        logger.info("Backed up the pre-rebuild manifest to %s", backup_path)
    write_metadata_atomic(merged, metadata_path)
    added = len(merged) - len(existing)
    logger.info(
        "Rebuilt metadata from %s: %d total row(s), %d recovered row(s) added",
        source_description,
        len(merged),
        added,
    )
    return len(merged), added


def category_counts_for(
    records: list[dict[str, Any]],
    stratify_by: str,
) -> dict[str, int]:
    counts = Counter(stratification_category(record, stratify_by) for record in records)
    return {category: counts[category] for category in sorted(counts)}


def split_counts_for(
    records: list[dict[str, Any]],
    split_assignment: dict[str, str],
    stratify_by: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        image_id = image_id_for(record)
        counts[stratification_category(record, stratify_by)][split_assignment[image_id]] += 1
    return {
        category: {split: counts[category].get(split, 0) for split in SPLIT_RATIOS}
        for category in sorted(counts)
    }


def log_distribution(
    logger: logging.Logger,
    title: str,
    counts: dict[str, int],
) -> None:
    logger.info("%s", title)
    for category in sorted(counts):
        logger.info("  %s: %d", category, counts[category])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of images in the selected cohort",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--stratify-by",
        default="diagnosis",
        help="Metadata field used to build sampling strata",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Allocate equal samples per category instead of archive proportions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling and split seed")
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
        help="Report metadata rows whose referenced image file is missing, then exit",
    )
    execution_mode.add_argument(
        "--rebuild-metadata-from",
        metavar="CSV",
        help="Merge valid rows from a cached CSV ('-' reads stdin), back up, then exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.stratify_by = args.stratify_by.strip()
    logger = configure_logging(args.data_dir, "isic", args.dry_run)
    if args.fresh and (args.repair or args.rebuild_metadata_from):
        raise ValueError("--fresh cannot be combined with repair/rebuild modes")
    if args.repair:
        run_repair_audit(args.data_dir, logger)
        return
    if args.rebuild_metadata_from:
        rebuild_metadata_from_csv(
            args.data_dir,
            args.rebuild_metadata_from,
            logger,
        )
        return
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.page_size < 1:
        raise ValueError("--page-size must be positive")
    if not args.stratify_by:
        raise ValueError("--stratify-by must not be empty")

    dataset_dir = args.data_dir / "isic"
    raw_dir = dataset_dir / "raw"
    metadata_path = dataset_dir / "metadata.csv"
    existing_metadata = load_existing_metadata(
        metadata_path,
        OUTPUT_COLUMNS,
        ("image_id",),
        args.fresh,
        logger,
    )

    session = requests.Session()
    session.headers["User-Agent"] = "medical-image-quality-dissertation/1.0"
    logger.info(
        "Indexing ISIC metadata for stratification by %s (mode=%s, seed=%d)",
        args.stratify_by,
        "balanced" if args.balanced else "proportional",
        args.seed,
    )
    listing = iter_images(
        session,
        logger,
        args.retries,
        args.timeout,
        args.page_size,
    )
    inventory = build_candidate_inventory(
        tqdm(listing, desc="Indexing ISIC metadata", unit="image"),
        args.stratify_by,
    )
    if not inventory:
        raise RuntimeError("ISIC metadata inventory contained no usable public images")
    archive_counts = {category: len(ids) for category, ids in inventory.items()}
    selected_ids, planned_counts = select_stratified_ids(
        inventory,
        args.limit,
        args.balanced,
        args.seed,
    )
    selected_records = fetch_selected_records(
        session,
        logger,
        selected_ids,
        args.stratify_by,
        args.retries,
        args.timeout,
        args.page_size,
    )
    split_assignment = assign_stratified_splits(
        selected_records,
        args.stratify_by,
        args.seed,
    )
    planned_split_counts = split_counts_for(
        selected_records,
        split_assignment,
        args.stratify_by,
    )
    log_distribution(logger, "Planned ISIC sample distribution:", planned_counts)

    if args.dry_run:
        if selected_records:
            response = request_with_retry(
                session,
                full_image_url(selected_records[0]),
                logger,
                args.retries,
                args.timeout,
                stream=True,
            )
            response.close()
        logger.info(
            "DRY RUN: selected and split %d records; no images, metadata, logs, "
            "or distribution files were written",
            len(selected_records),
        )
        return

    ensure_structure(args.data_dir)
    rows: list[dict[str, str]] = []
    achieved_records: list[dict[str, Any]] = []
    reprocessed_ids: set[str] = set()
    failed_ids: set[str] = set()
    progress = tqdm(
        selected_records,
        total=len(selected_records),
        desc="ISIC images",
        unit="image",
    )
    for record in progress:
        image_id = image_id_for(record)
        try:
            expected_url = full_image_url(record)
            expected_path = raw_dir / f"{image_id}{extension_for(record, expected_url)}"
            was_valid_before = validate_raster_image(expected_path)[0]
        except ValueError:
            was_valid_before = False
        image_path = download_image_record(
            session,
            record,
            raw_dir,
            logger,
            args.retries,
            args.timeout,
        )
        if image_path is None:
            failed_ids.add(image_id)
            continue
        if not was_valid_before:
            reprocessed_ids.add(image_id)
        rows.append(
            metadata_row(
                record,
                image_path,
                dataset_dir,
                split=split_assignment[image_id],
            )
        )
        achieved_records.append(record)

    if failed_ids and not existing_metadata.empty:
        existing_metadata = existing_metadata.loc[
            ~existing_metadata["image_id"].astype(str).isin(failed_ids)
        ].copy()
    new_metadata = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    merged_metadata = merge_metadata_frames(
        existing_metadata,
        new_metadata,
        OUTPUT_COLUMNS,
        ("image_id",),
        update_keys={(image_id,) for image_id in reprocessed_ids},
    )
    write_metadata_atomic(merged_metadata, metadata_path)

    achieved_counts = category_counts_for(achieved_records, args.stratify_by)
    achieved_split_counts = split_counts_for(
        achieved_records,
        split_assignment,
        args.stratify_by,
    )
    distribution = {
        "stratify_by": args.stratify_by,
        "mode": "balanced" if args.balanced else "proportional",
        "seed": args.seed,
        "requested_limit": args.limit,
        "archive_counts": archive_counts,
        "planned_sample_counts": planned_counts,
        "achieved_sample_counts": achieved_counts,
        "planned_split_counts": planned_split_counts,
        "achieved_split_counts": achieved_split_counts,
    }
    distribution_path = args.data_dir / "logs" / "isic_sample_distribution.json"
    distribution_temporary = distribution_path.with_suffix(".json.tmp")
    distribution_temporary.write_text(
        json.dumps(distribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    distribution_temporary.replace(distribution_path)

    log_distribution(logger, "Achieved ISIC sample distribution:", achieved_counts)
    logger.info(
        "Metadata now contains %d rows (%d successful row(s) in this run)",
        len(merged_metadata),
        len(rows),
    )
    logger.info("Wrote sample distribution to %s", distribution_path)


if __name__ == "__main__":
    main()
