"""Tests for pipeline.resolution, pipeline.duplicates, and scripts.detect_duplicates.

Three invariants worth protecting:

1. **Hash resolution must be adequate, and saturation must be detectable.** The
   standard 64-bit pHash cannot discriminate stereotyped medical images: measured
   on synthetic radiographs it produced 17 distinct values across 35 different
   subjects, collapsing precision against known patient groups to 0.007. The
   default hash size is therefore 16 (256-bit), and hash_saturation() must flag
   the failure mode rather than letting it pass silently.

2. **Rows that cannot be checked must not be reported as passing.** A processed
   row with missing canonical_resize provenance has unknown resolution, which is
   a different verdict from "resolution is fine".

3. **Precision/recall must be skipped, not estimated, when no grouping exists.**
   ISIC has no persisted lesion_id, so fabricating a proxy would produce a
   number that looks like validation but is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import pipeline.duplicates as duplicates
import pipeline.resolution as resolution


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _processed_row(image_id: str, width: int, height: int) -> dict:
    return {
        "image_id": image_id,
        "defect_params_json": json.dumps({
            "baseline_preprocessing": {
                "canonical_resize": {
                    "canonical_size": 256,
                    "original_width": width,
                    "original_height": height,
                }
            }
        }),
    }


def test_extract_source_dimensions_reads_recorded_provenance() -> None:
    frame = pd.DataFrame([
        _processed_row("a", 1024, 1024),
        _processed_row("b", 512, 256),
    ])
    extracted = resolution.extract_source_dimensions(frame)
    assert extracted.loc[0, "original_width"] == 1024
    assert extracted.loc[1, "aspect_ratio"] == pytest.approx(2.0)
    assert extracted.loc[0, "megapixels"] == pytest.approx(1024 * 1024 / 1e6)


def test_aspect_ratio_is_orientation_independent() -> None:
    """A portrait and a landscape frame of the same elongation lose the same
    amount to a square centre-crop, so both must report the same ratio."""
    frame = pd.DataFrame([
        _processed_row("landscape", 900, 300),
        _processed_row("portrait", 300, 900),
    ])
    extracted = resolution.extract_source_dimensions(frame)
    assert extracted.loc[0, "aspect_ratio"] == pytest.approx(extracted.loc[1, "aspect_ratio"])
    assert extracted.loc[0, "aspect_ratio"] == pytest.approx(3.0)


def test_missing_provenance_is_not_checkable_rather_than_passing() -> None:
    frame = pd.DataFrame([
        _processed_row("good", 1024, 1024),
        {"image_id": "broken", "defect_params_json": "not json at all"},
        {"image_id": "empty", "defect_params_json": "{}"},
    ])
    flagged = resolution.check_resolution(frame)
    assert bool(flagged.loc[1, "provenance_missing"]) is True
    assert bool(flagged.loc[2, "provenance_missing"]) is True
    # Crucially: not flagged, but also not silently counted as fine.
    assert bool(flagged.loc[1, "flagged"]) is False
    summary = resolution.summarise_resolution(flagged)
    assert summary["n_provenance_missing"] == 2
    assert summary["n_measurable"] == 1


def test_below_min_side_is_flagged() -> None:
    frame = pd.DataFrame([_processed_row(f"big{i}", 1024, 1024) for i in range(20)]
                         + [_processed_row("small", 180, 180)])
    flagged = resolution.check_resolution(frame)
    small = flagged[flagged.image_id == "small"].iloc[0]
    assert bool(small["below_min_side"]) is True
    assert bool(small["flagged"]) is True


def test_extreme_aspect_ratio_is_flagged() -> None:
    frame = pd.DataFrame([_processed_row(f"square{i}", 1024, 1024) for i in range(20)]
                         + [_processed_row("strip", 3000, 500)])
    flagged = resolution.check_resolution(frame)
    strip = flagged[flagged.image_id == "strip"].iloc[0]
    assert bool(strip["extreme_aspect_ratio"]) is True


def test_small_for_dataset_adapts_to_the_datasets_own_distribution() -> None:
    """A 600px image is fine in a thumbnail corpus and an outlier in a
    microscopy corpus, so the threshold must come from the data."""
    microscopy = pd.DataFrame(
        [_processed_row(f"huge{i}", 6000, 4000) for i in range(99)]
        + [_processed_row("modest", 600, 600)]
    )
    flagged = resolution.check_resolution(microscopy)
    modest = flagged[flagged.image_id == "modest"].iloc[0]
    assert bool(modest["small_for_dataset"]) is True


def test_resolution_thresholds_are_recorded_in_the_summary() -> None:
    frame = pd.DataFrame([_processed_row("a", 1024, 1024)])
    flagged = resolution.check_resolution(
        frame, resolution.ResolutionThresholds(min_side=512, max_aspect_ratio=1.2)
    )
    summary = resolution.summarise_resolution(flagged)
    assert summary["thresholds"]["min_side"] == 512
    assert summary["thresholds"]["max_aspect_ratio"] == 1.2


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _write_images(tmp_path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """base, an exact copy, a near-duplicate, and an unrelated image."""
    rng = np.random.default_rng(0)
    base = (rng.random((128, 128)) * 255).astype(np.uint8)
    near = base.copy()
    near[:6, :6] = 0
    other = (rng.random((128, 128)) * 255).astype(np.uint8)

    paths: dict[str, Path] = {}
    for name, array in (
        ("a", base), ("a_copy", base), ("a_near", near), ("b", other)
    ):
        path = tmp_path / f"{name}.png"
        Image.fromarray(array, "L").save(path)
        paths[name] = path
    groups = {"a": "g1", "a_copy": "g1", "a_near": "g1", "b": "g2"}
    return paths, groups


def test_exact_copies_are_found_at_distance_zero(tmp_path: Path) -> None:
    paths, _ = _write_images(tmp_path)
    hashes, failures = duplicates.compute_hashes(paths)
    assert not failures
    exact = duplicates.find_hash_duplicates(hashes, near_distance=0)
    found = {frozenset((pair.image_id_a, pair.image_id_b)) for pair in exact}
    assert frozenset(("a", "a_copy")) in found
    assert all(pair.is_exact for pair in exact)


def test_unrelated_images_are_not_paired(tmp_path: Path) -> None:
    paths, _ = _write_images(tmp_path)
    hashes, _ = duplicates.compute_hashes(paths)
    pairs = duplicates.find_hash_duplicates(hashes, near_distance=10)
    assert not any("b" in (pair.image_id_a, pair.image_id_b) for pair in pairs)


def test_hash_failures_are_collected_not_raised(tmp_path: Path) -> None:
    """One unreadable file must not abort a whole dataset run."""
    paths, _ = _write_images(tmp_path)
    broken = tmp_path / "broken.png"
    broken.write_text("this is not a png")
    paths["broken"] = broken
    hashes, failures = duplicates.compute_hashes(paths)
    assert "broken" in failures
    assert len(hashes) == 4


def test_hash_saturation_detects_an_undersized_hash() -> None:
    """The failure mode that made 64-bit pHash useless on radiographs: many
    distinct images collapsing onto the same hash value."""
    class FakeHash:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:
            return self.value

    # 10 images, only 2 distinct hashes
    hashes = {
        f"img{i}": {"phash": FakeHash("AAAA" if i % 2 else "BBBB")}
        for i in range(10)
    }
    diagnostic = duplicates.hash_saturation(hashes, "phash")
    assert diagnostic["n_distinct_hashes"] == 2
    assert diagnostic["distinct_ratio"] == pytest.approx(0.2)
    assert diagnostic["saturated"] is True


def test_hash_saturation_passes_a_well_resolved_hash() -> None:
    class FakeHash:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:
            return self.value

    hashes = {f"img{i}": {"phash": FakeHash(f"H{i}")} for i in range(10)}
    diagnostic = duplicates.hash_saturation(hashes, "phash")
    assert diagnostic["distinct_ratio"] == pytest.approx(1.0)
    assert diagnostic["saturated"] is False


def test_default_hash_size_exceeds_imagehash_default() -> None:
    """Guards the finding: imagehash defaults to 8 (64-bit), which saturates on
    stereotyped medical images. This project must default higher."""
    assert duplicates.DEFAULT_HASH_SIZE > 8


def test_larger_hash_size_yields_more_distinct_values(tmp_path: Path) -> None:
    """Direct evidence for the fix: increasing hash resolution must increase
    discriminative power on stereotyped images."""
    rng = np.random.default_rng(3)
    paths: dict[str, Path] = {}
    # Highly stereotyped: same smooth template, differing only in faint noise.
    y, x = np.mgrid[0:256, 0:256]
    template = 120 + 40 * np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / 8000)
    for index in range(12):
        array = np.clip(
            template + rng.normal(0, 1.5, template.shape), 0, 255
        ).astype(np.uint8)
        path = tmp_path / f"s{index}.png"
        Image.fromarray(array, "L").save(path)
        paths[f"s{index}"] = path

    coarse, _ = duplicates.compute_hashes(paths, hash_size=8)
    fine, _ = duplicates.compute_hashes(paths, hash_size=16)
    coarse_ratio = duplicates.hash_saturation(coarse)["distinct_ratio"]
    fine_ratio = duplicates.hash_saturation(fine)["distinct_ratio"]
    assert fine_ratio >= coarse_ratio


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------

def test_embedding_duplicates_recover_planted_groups() -> None:
    rng = np.random.default_rng(1)
    image_ids: list[str] = []
    vectors: list[np.ndarray] = []
    groups: dict[str, str] = {}
    for subject in ("p1", "p2"):
        base = rng.normal(0, 1, 64)
        for index in range(3):
            image_id = f"{subject}_img{index}"
            image_ids.append(image_id)
            vectors.append(base + rng.normal(0, 0.02, 64))
            groups[image_id] = subject
    for index in range(4):
        image_id = f"solo{index}"
        image_ids.append(image_id)
        vectors.append(rng.normal(0, 1, 64))
        groups[image_id] = f"solo_{index}"

    pairs = duplicates.find_embedding_duplicates(
        image_ids, np.asarray(vectors), cosine_threshold=0.95
    )
    metrics = duplicates.evaluate_against_groups(pairs, groups, image_ids)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_embedding_duplicates_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        duplicates.find_embedding_duplicates(["a", "b"], np.zeros((3, 8)))


def test_zero_norm_embeddings_do_not_produce_nan() -> None:
    vectors = np.zeros((3, 8))
    pairs = duplicates.find_embedding_duplicates(["a", "b", "c"], vectors, 0.95)
    assert all(np.isfinite(pair.score) for pair in pairs)


def test_each_unordered_pair_is_reported_once() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 32)
    vectors = np.stack([base, base + 1e-6, base - 1e-6])
    pairs = duplicates.find_embedding_duplicates(["a", "b", "c"], vectors, 0.99)
    keys = [frozenset((pair.image_id_a, pair.image_id_b)) for pair in pairs]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Validation and leakage
# ---------------------------------------------------------------------------

def test_evaluate_against_groups_counts_only_same_group_pairs_as_correct() -> None:
    pairs = [
        duplicates.DuplicatePair("a", "b", "phash_near", 2.0, False),  # same group
        duplicates.DuplicatePair("a", "z", "phash_near", 3.0, False),  # different
    ]
    groups = {"a": "g1", "b": "g1", "z": "g2"}
    metrics = duplicates.evaluate_against_groups(pairs, groups, ["a", "b", "z"])
    assert metrics["n_true_positives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)


def test_evaluate_against_groups_returns_nan_without_any_grouping() -> None:
    """ISIC has no persisted lesion_id; validation must decline rather than
    invent a number."""
    pairs = [duplicates.DuplicatePair("a", "b", "phash_near", 2.0, False)]
    metrics = duplicates.evaluate_against_groups(pairs, {}, ["a", "b"])
    assert np.isnan(metrics["recall"])
    assert metrics["n_known_same_group_pairs"] == 0


def test_split_leakage_separates_crossing_from_within_split_pairs() -> None:
    pairs = [
        duplicates.DuplicatePair("a", "b", "phash_near", 1.0, False),
        duplicates.DuplicatePair("c", "d", "phash_near", 1.0, False),
    ]
    splits = {"a": "train", "b": "test", "c": "train", "d": "train"}
    leakage = duplicates.check_split_leakage(pairs, splits)
    assert leakage["n_pairs_crossing_splits"] == 1
    assert leakage["n_pairs_within_one_split"] == 1
    assert leakage["crossing_examples"][0]["image_id_a"] == "a"


def test_sweep_reports_a_precision_recall_tradeoff(tmp_path: Path) -> None:
    paths, groups = _write_images(tmp_path)
    hashes, _ = duplicates.compute_hashes(paths)
    sweep = duplicates.sweep_thresholds(
        hashes, groups, list(paths), distances=(0, 4, 8, 16)
    )
    assert len(sweep) == 4
    assert {"hash_distance", "precision", "recall"} <= set(sweep.columns)
    # More permissive distances cannot detect fewer pairs.
    assert sweep["n_detected_pairs"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def _build_dataset(tmp_path: Path, with_group_column: bool = True) -> Path:
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    (dataset_dir / "processed").mkdir(parents=True)
    (data_dir / "logs").mkdir()

    rng = np.random.default_rng(7)
    source_rows = []
    processed_rows = []
    base = (rng.random((256, 256)) * 255).astype(np.uint8)

    for index in range(14):
        if index < 2:
            array, patient = base, "shared"      # an exact duplicate pair
        else:
            array = (rng.random((256, 256)) * 255).astype(np.uint8)
            patient = f"p{index}"
        name = f"img{index}.png"
        Image.fromarray(array, "L").save(dataset_dir / "processed" / name)
        split = ("train", "val", "test")[index % 3]
        row = {"image_id": name, "path": f"raw/{name}", "split": split}
        if with_group_column:
            row["patient_id"] = patient
        source_rows.append(row)
        processed_rows.append({
            "image_id": name,
            "original_image_id": name,
            "path": f"processed/{name}",
            "split": split,
            "defect_type": "clean",
            "severity_level": "0",
            "canonical_size": "256",
            **_processed_row(name, 1024 if index else 180, 1024),
        })

    pd.DataFrame(source_rows).to_csv(dataset_dir / "metadata.csv", index=False)
    pd.DataFrame(processed_rows).to_csv(
        dataset_dir / "processed_metadata.csv", index=False
    )
    return data_dir


def test_end_to_end_with_group_column(tmp_path: Path) -> None:
    from scripts.detect_duplicates import main as detect_main

    data_dir = _build_dataset(tmp_path, with_group_column=True)
    # exact + reidentification only: the transformation task needs the separate
    # benchmark set built by build_duplicate_benchmark.py.
    detect_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--tasks", "exact,reidentification",
        "--group-column", "patient_id", "--skip-embeddings",
    ])

    logs = data_dir / "logs"
    assert (logs / "demo_resolution.csv").is_file()
    assert (logs / "demo_duplicates_summary.json").is_file()

    summary = json.loads((logs / "demo_duplicates_summary.json").read_text())
    exact = summary["tasks"]["exact"]
    assert exact["hash_phash"]["n_exact_pairs"] >= 1   # the planted exact pair
    assert "hash_saturation" in exact
    reid = summary["tasks"]["reidentification"]
    assert reid["ground_truth_available"] is True


def test_end_to_end_without_group_column_skips_validation(tmp_path: Path) -> None:
    """The ISIC case: exact duplicates still detected, validation declined."""
    from scripts.detect_duplicates import main as detect_main

    data_dir = _build_dataset(tmp_path, with_group_column=False)
    detect_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--tasks", "exact,reidentification", "--skip-embeddings",
    ])
    summary = json.loads(
        (data_dir / "logs" / "demo_duplicates_summary.json").read_text()
    )
    # Exact detection still works without any ground truth...
    assert summary["tasks"]["exact"]["hash_phash"]["n_exact_pairs"] >= 1
    # ...but reidentification declines rather than fabricating metrics.
    assert "skipped" in summary["tasks"]["reidentification"]


def test_unknown_group_column_is_rejected(tmp_path: Path) -> None:
    from scripts.detect_duplicates import main as detect_main

    data_dir = _build_dataset(tmp_path, with_group_column=True)
    with pytest.raises(ValueError):
        detect_main([
            "--dataset", "demo", "--data-dir", str(data_dir),
            "--tasks", "reidentification",
            "--group-column", "not_a_column", "--skip-embeddings",
        ])


def test_degraded_rows_are_excluded_by_default(tmp_path: Path) -> None:
    """Degraded variants of one source image are duplicates by construction, so
    including them would measure this project's own defect injection."""
    from scripts.detect_duplicates import main as detect_main

    data_dir = _build_dataset(tmp_path, with_group_column=True)
    processed_path = data_dir / "demo" / "processed_metadata.csv"
    processed = pd.read_csv(processed_path)
    processed.loc[processed.index[:4], "defect_type"] = "gaussian_noise"
    processed.to_csv(processed_path, index=False)

    detect_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--tasks", "exact", "--group-column", "patient_id", "--skip-embeddings",
    ])
    summary = json.loads(
        (data_dir / "logs" / "demo_duplicates_summary.json").read_text()
    )
    # The exact task uses clean rows only; the 4 rows marked degraded above
    # must be excluded, since defect variants duplicate by construction.
    assert summary["tasks"]["exact"]["n_candidate_rows"] == len(processed) - 4


def test_uniform_resolution_dataset_flags_nothing_as_an_outlier() -> None:
    """Regression: ChestX-ray14 is entirely 1024x1024, so its 1st percentile IS
    1024 and a "<= percentile" outlier test flagged all 5,606 rows as unusually
    small. A dataset of uniform resolution has no outliers to find."""
    frame = pd.DataFrame([_processed_row(f"i{i}", 1024, 1024) for i in range(100)])
    flagged = resolution.check_resolution(frame)
    summary = resolution.summarise_resolution(flagged)
    assert summary["n_flagged"] == 0
    assert summary["n_small_for_dataset"] == 0
    assert summary["outlier_check_skipped_uniform_resolution"] is True


def test_outliers_are_still_caught_when_the_distribution_has_spread() -> None:
    """The fix must not disable the check on datasets where it is meaningful."""
    frame = pd.DataFrame(
        [_processed_row(f"big{i}", 1024, 1024) for i in range(97)]
        + [_processed_row("tiny1", 100, 100),
           _processed_row("tiny2", 120, 120),
           _processed_row("tiny3", 90, 90)]
    )
    flagged = resolution.check_resolution(frame)
    summary = resolution.summarise_resolution(flagged)
    assert summary["outlier_check_skipped_uniform_resolution"] is False
    assert summary["n_flagged"] == 3


def test_an_image_exactly_at_the_percentile_is_not_its_own_outlier() -> None:
    """Strict less-than: a boundary image must not be flagged against itself."""
    frame = pd.DataFrame(
        [_processed_row(f"i{i}", 1024, 1024) for i in range(50)]
        + [_processed_row(f"j{i}", 400, 400) for i in range(50)]
    )
    flagged = resolution.check_resolution(frame)
    boundary = flagged[flagged.image_id == "j0"].iloc[0]
    assert bool(boundary["small_for_dataset"]) is False


# ---------------------------------------------------------------------------
# All four classical hashes
# ---------------------------------------------------------------------------

def test_all_four_classical_hashes_are_computed(tmp_path: Path) -> None:
    """The standard comparative evaluations use AHash, DHash, PHash and WHash."""
    paths, _ = _write_images(tmp_path)
    hashes, _ = duplicates.compute_hashes(paths)
    assert set(duplicates.HASH_TYPES) == {"ahash", "phash", "dhash", "whash"}
    for entry in hashes.values():
        assert set(entry) == set(duplicates.HASH_TYPES)


def test_whash_hash_size_is_coerced_to_a_power_of_two() -> None:
    """whash rejects non-power-of-two sizes; the row must not fail because of it."""
    assert duplicates._power_of_two_at_most(16) == 16
    assert duplicates._power_of_two_at_most(20) == 16
    assert duplicates._power_of_two_at_most(24) == 16
    assert duplicates._power_of_two_at_most(32) == 32


def test_non_power_of_two_hash_size_still_produces_all_hashes(tmp_path: Path) -> None:
    paths, _ = _write_images(tmp_path)
    hashes, failures = duplicates.compute_hashes(paths, hash_size=20)
    assert not failures
    for entry in hashes.values():
        assert set(entry) == set(duplicates.HASH_TYPES)


def test_every_hash_type_can_find_an_exact_copy(tmp_path: Path) -> None:
    paths, _ = _write_images(tmp_path)
    hashes, _ = duplicates.compute_hashes(paths)
    for hash_type in duplicates.HASH_TYPES:
        exact = duplicates.find_hash_duplicates(hashes, 0, hash_type)
        found = {frozenset((p.image_id_a, p.image_id_b)) for p in exact}
        assert frozenset(("a", "a_copy")) in found, hash_type


# ---------------------------------------------------------------------------
# Task separation
# ---------------------------------------------------------------------------

def test_the_three_tasks_are_registered_with_definitions() -> None:
    from scripts.detect_duplicates import TASKS, TASK_NOTES

    assert set(TASKS) == {"exact", "transformation", "reidentification"}
    assert set(TASK_NOTES) == set(TASKS)


def test_resolve_tasks_accepts_all_and_subsets_and_rejects_unknown() -> None:
    from scripts.detect_duplicates import TASKS, resolve_tasks

    assert resolve_tasks("all") == TASKS
    assert resolve_tasks("exact,reidentification") == ("exact", "reidentification")
    with pytest.raises(ValueError):
        resolve_tasks("exact,not_a_task")


def test_transformation_task_requires_the_dedicated_benchmark_set(
    tmp_path: Path,
) -> None:
    """The main corpus has one treatment per source image, so it contains no
    near-duplicate pairs. The task must say so rather than silently returning
    a meaningless zero."""
    import logging

    from scripts.detect_duplicates import rows_for_task

    data_dir = _build_dataset(tmp_path, with_group_column=True)
    processed = pd.read_csv(data_dir / "demo" / "processed_metadata.csv", dtype=str)
    with pytest.raises(FileNotFoundError, match="build_duplicate_benchmark"):
        rows_for_task(
            "transformation", processed, data_dir / "demo", 100, 42,
            logging.getLogger("test"),
        )


def test_reidentification_is_skipped_without_a_subject_column(tmp_path: Path) -> None:
    from scripts.detect_duplicates import main as detect_main

    data_dir = _build_dataset(tmp_path, with_group_column=False)
    detect_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--tasks", "reidentification", "--skip-embeddings",
    ])
    summary = json.loads(
        (data_dir / "logs" / "demo_duplicates_summary.json").read_text()
    )
    entry = summary["tasks"]["reidentification"]
    assert "skipped" in entry
    assert "subject column" in entry["skipped"]


def test_exact_task_needs_no_ground_truth(tmp_path: Path) -> None:
    from scripts.detect_duplicates import main as detect_main

    data_dir = _build_dataset(tmp_path, with_group_column=False)
    detect_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--tasks", "exact", "--skip-embeddings",
    ])
    summary = json.loads(
        (data_dir / "logs" / "demo_duplicates_summary.json").read_text()
    )
    entry = summary["tasks"]["exact"]
    assert "skipped" not in entry
    assert entry["hash_phash"]["n_exact_pairs"] >= 1


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------

def _write_mixed_mode_dataset(tmp_path: Path) -> Path:
    """A dataset spanning every source mode the real corpora contain.

    ChestX-ray14 has 23 fully-opaque RGBA images and ISIC is RGB; 16-bit
    grayscale also occurs. The benchmark builder must normalise all of them.
    """
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    (dataset_dir / "raw").mkdir(parents=True)
    (data_dir / "logs").mkdir()

    rng = np.random.default_rng(4)
    rows = []
    for index in range(24):
        array = (rng.random((128, 128)) * 200 + 30).astype(np.uint8)
        if index % 4 == 0:                       # opaque RGBA
            image = Image.fromarray(
                np.dstack([array, array, array, np.full_like(array, 255)]), "RGBA"
            )
        elif index % 4 == 1:                     # genuine colour
            image = Image.fromarray(
                np.dstack([array, np.roll(array, 7, 0), np.roll(array, 13, 1)]), "RGB"
            )
        elif index % 4 == 2:                     # 16-bit grayscale
            image = Image.fromarray(array.astype(np.uint16) * 257)
        else:
            image = Image.fromarray(array, "L")
        name = f"img{index:02d}.png"
        image.save(dataset_dir / "raw" / name)
        rows.append({
            "image_id": name, "path": f"raw/{name}",
            "patient_id": f"p{index:02d}",
            "split": ("train", "val", "test")[index % 3],
        })
    pd.DataFrame(rows).to_csv(dataset_dir / "metadata.csv", index=False)
    return data_dir


def test_benchmark_builder_handles_rgba_and_16bit_sources(tmp_path: Path) -> None:
    """Regression: calling canonical_resize directly bypassed alpha flattening
    and 8-bit normalisation, so ChestX-ray14's RGBA images crashed on save.
    The builder must use prepare_cli_source, which handles both."""
    from scripts.build_duplicate_benchmark import main as build_main

    data_dir = _write_mixed_mode_dataset(tmp_path)
    build_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--n-sources", "6", "--n-distractors", "6",
    ])

    benchmark_dir = data_dir / "demo" / "duplicate_benchmark"
    frame = pd.read_csv(benchmark_dir / "benchmark_metadata.csv")
    manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text())
    assert manifest["n_failed_sources"] == 0, "no source mode should fail"
    # 6 sources x 6 transformations + 6 single-copy distractors
    assert len(frame) == 6 * 6 + 6
    # Every written file must be loadable.
    for relative in frame["path"]:
        with Image.open(data_dir / "demo" / relative) as handle:
            assert handle.size == (256, 256)


def test_benchmark_ground_truth_yields_the_expected_pair_count(tmp_path: Path) -> None:
    """Groups of 6 give 15 pairs each; distractors contribute none."""
    from scripts.build_duplicate_benchmark import main as build_main

    data_dir = _write_mixed_mode_dataset(tmp_path)
    build_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--n-sources", "5", "--n-distractors", "5",
    ])
    manifest = json.loads(
        (data_dir / "demo" / "duplicate_benchmark" / "benchmark_manifest.json").read_text()
    )
    assert manifest["n_true_duplicate_pairs"] == 5 * (6 * 5 // 2)
    assert manifest["n_distractors"] == 5


def test_benchmark_distractors_have_unique_group_ids(tmp_path: Path) -> None:
    """A distractor must have no duplicate partner anywhere, or precision is
    measured against a population that contains hidden true pairs."""
    from scripts.build_duplicate_benchmark import main as build_main

    data_dir = _write_mixed_mode_dataset(tmp_path)
    build_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--n-sources", "5", "--n-distractors", "6",
    ])
    frame = pd.read_csv(
        data_dir / "demo" / "duplicate_benchmark" / "benchmark_metadata.csv"
    )
    solo = frame[~frame["is_duplicate_group"]]
    assert len(solo) == solo["group_id"].nunique()


def test_benchmark_rejects_a_request_larger_than_the_dataset(tmp_path: Path) -> None:
    from scripts.build_duplicate_benchmark import main as build_main

    data_dir = _write_mixed_mode_dataset(tmp_path)
    with pytest.raises(ValueError, match="metadata.csv has"):
        build_main([
            "--dataset", "demo", "--data-dir", str(data_dir),
            "--n-sources", "500", "--n-distractors", "500",
        ])


def test_transformation_task_looks_for_the_benchmark_embedding_cache(
    tmp_path: Path,
) -> None:
    """Regression: the benchmark set's image_ids do not exist in the main corpus
    cache, so the embedding method silently reported 0 matched embeddings and
    dropped out of the comparison entirely. The task must look for a
    _benchmark-suffixed cache, and say so loudly when it is absent."""
    import logging

    from scripts.build_duplicate_benchmark import main as build_main
    from scripts.detect_duplicates import main as detect_main

    data_dir = _write_mixed_mode_dataset(tmp_path)
    build_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--n-sources", "4", "--n-distractors", "4",
    ])
    # A main-corpus cache exists but a benchmark one does not.
    (data_dir / "embeddings").mkdir(exist_ok=True)
    import pipeline.embeddings as embeddings_module

    embeddings_module.save_embeddings(
        data_dir / "embeddings" / "demo_resnet18.npz",
        ["unrelated"], np.zeros((1, 8), dtype=np.float32), {},
    )

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())

    processed = pd.DataFrame([
        {**_processed_row("x", 1024, 1024), "defect_type": "clean",
         "split": "train", "path": "raw/img00.png", "original_image_id": "img00.png"}
    ])
    logger = logging.getLogger("benchmark_cache_test")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    from scripts.detect_duplicates import run_task

    from argparse import Namespace

    args = Namespace(
        dataset="demo", data_dir=data_dir, backbone="resnet18",
        hash_size=16, hash_distance=10, cosine_threshold=0.95,
        skip_hashing=True, skip_embeddings=False,
        max_transformation_sources=100, seed=42, group_column=None,
    )
    run_task(
        "transformation", args, data_dir / "demo", processed,
        data_dir / "logs", logger,
    )
    joined = " ".join(records)
    assert "demo_benchmark_resnet18.npz" in joined, (
        "must look for the benchmark-suffixed cache, not the main corpus one"
    )
    assert "--cache-suffix _benchmark" in joined, (
        "must tell the user exactly how to build the missing cache"
    )


def test_transformation_task_looks_for_the_benchmark_embedding_cache(
    tmp_path: Path, caplog
) -> None:
    """Regression: the benchmark images have new image_ids that do not exist in
    the main corpus cache, so the transformation task matched 0 embeddings and
    silently dropped the embedding half of the comparison. It must look for the
    separate _benchmark cache and say so when it is absent."""
    import logging

    from scripts.build_duplicate_benchmark import main as build_main
    from scripts.detect_duplicates import main as detect_main

    data_dir = _write_mixed_mode_dataset(tmp_path)
    # detect_duplicates needs processed_metadata.csv for the resolution check.
    source = pd.read_csv(data_dir / "demo" / "metadata.csv")
    processed = source.copy()
    processed["original_image_id"] = processed["image_id"]
    processed["defect_type"] = "clean"
    processed["severity_level"] = "0"
    processed["defect_params_json"] = json.dumps(
        {"baseline_preprocessing": {"canonical_resize": {
            "original_width": 128, "original_height": 128}}}
    )
    processed.to_csv(data_dir / "demo" / "processed_metadata.csv", index=False)

    build_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--n-sources", "5", "--n-distractors", "5",
    ])
    with caplog.at_level(logging.WARNING):
        detect_main([
            "--dataset", "demo", "--data-dir", str(data_dir),
            "--tasks", "transformation", "--skip-hashing",
        ])
    # The warning must name the benchmark cache, not the corpus cache.
    assert "_benchmark_" in caplog.text, caplog.text
    assert "--subset benchmark" in caplog.text, caplog.text


def test_extract_embeddings_benchmark_subset_writes_a_separate_cache(
    tmp_path: Path,
) -> None:
    """The benchmark cache must never overwrite the main corpus cache."""
    from scripts.build_duplicate_benchmark import main as build_main
    from scripts.extract_embeddings import main as extract_main

    data_dir = _write_mixed_mode_dataset(tmp_path)
    build_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--n-sources", "3", "--n-distractors", "3",
    ])
    extract_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--subset", "benchmark", "--no-pretrained", "--backbones", "resnet18",
    ])
    embeddings_dir = data_dir / "embeddings"
    assert (embeddings_dir / "demo_benchmark_resnet18.npz").is_file()
    assert not (embeddings_dir / "demo_resnet18.npz").is_file()