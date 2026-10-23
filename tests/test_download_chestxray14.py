"""Tests for the NIH ChestX-ray14 downloader.

These exercise the behaviours that were learned the hard way on ISIC:
stratified (never first-N) sampling, group-safe splits, merge-not-overwrite
metadata, and refusing to index images that are missing or corrupt.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import scripts.download_chestxray14 as cxr


FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def build_fake_cache(
    root: Path,
    images_per_patient: int = 2,
    patients: int = 120,
    corrupt: int = 0,
    label_rows_without_images: int = 0,
) -> Path:
    """Create a ChestX-ray14-shaped download cache with deterministic content."""
    cache = root / "cache"
    (cache / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for patient in range(1, patients + 1):
        for _ in range(images_per_patient):
            index = len(rows) + 1
            image_id = f"{index:08d}_000.png"
            # Deterministic label spread; Hernia is deliberately ultra-rare.
            if index % 3 == 0:
                labels = "No Finding"
            elif index % 37 == 0:
                labels = "Hernia"
            else:
                first = FINDINGS[index % len(FINDINGS)]
                second = FINDINGS[(index * 7) % len(FINDINGS)]
                labels = "|".join(sorted({first, second}))
            rows.append(
                {
                    "Image Index": image_id,
                    "Finding Labels": labels,
                    "Patient ID": str(patient),
                    "View Position": "PA" if index % 2 else "AP",
                }
            )
            pixels = rng.integers(0, 255, size=(16, 16), dtype=np.uint8)
            Image.fromarray(pixels, mode="L").save(cache / "images" / image_id)

    for offset in range(corrupt):
        (cache / "images" / rows[offset]["Image Index"]).write_bytes(b"not an image")

    frame = pd.DataFrame(rows)
    if label_rows_without_images:
        phantom = frame.iloc[:label_rows_without_images].copy()
        phantom["Image Index"] = [
            f"9999{i:04d}_000.png" for i in range(label_rows_without_images)
        ]
        frame = pd.concat([frame, phantom], ignore_index=True)
    frame.to_csv(cache / "sample_labels.csv", index=False)
    return cache


@pytest.fixture()
def kaggle_stub(monkeypatch, tmp_path):
    """Install a fake kagglehub and Kaggle credentials for the duration of a test."""
    cache = build_fake_cache(tmp_path)

    module = types.ModuleType("kagglehub")
    module.dataset_download = lambda handle: str(cache)  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "kagglehub", module)

    credentials = tmp_path / "home" / ".kaggle"
    credentials.mkdir(parents=True)
    (credentials / "kaggle.json").write_text('{"username":"t","key":"t"}')
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    return cache


def test_rarest_present_finding_wins_the_stratum() -> None:
    frequencies = {"Infiltration": 800, "Mass": 120, "Hernia": 4}
    assert cxr.finding_stratum("Infiltration|Mass", frequencies) == "Mass"
    assert cxr.finding_stratum("Infiltration|Mass|Hernia", frequencies) == "Hernia"
    assert cxr.finding_stratum("Infiltration", frequencies) == "Infiltration"


def test_no_finding_is_its_own_stratum() -> None:
    frequencies = {"Effusion": 10}
    assert cxr.finding_stratum("No Finding", frequencies) == "No Finding"
    assert cxr.finding_stratum("", frequencies) == "No Finding"
    # A stray "No Finding" alongside a real finding must not win.
    assert cxr.finding_stratum("No Finding|Effusion", frequencies) == "Effusion"


def test_equal_frequencies_break_ties_deterministically() -> None:
    frequencies = {"Edema": 10, "Mass": 10}
    assert cxr.finding_stratum("Mass|Edema", frequencies) == "Edema"
    assert cxr.finding_stratum("Edema|Mass", frequencies) == "Edema"


def test_selection_is_stratified_not_first_n(kaggle_stub, tmp_path) -> None:
    """A limited run must span strata rather than taking the head of the CSV."""
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--limit", "90", "--seed", "42"])
    metadata = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)

    assert len(metadata) == 90
    assert metadata["finding_stratum"].nunique() > 5

    # First-N selection would take a contiguous block of low image indices.
    indices = sorted(int(name.split("_")[0]) for name in metadata["image_id"])
    assert max(indices) - min(indices) > 90


def test_no_patient_appears_in_more_than_one_split(kaggle_stub, tmp_path) -> None:
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--seed", "42"])
    metadata = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)
    per_patient = metadata.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all()


def test_split_ratios_are_approximately_seventy_fifteen_fifteen(
    kaggle_stub, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--seed", "42"])
    metadata = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)
    shares = metadata["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.06)
    assert shares["val"] == pytest.approx(0.15, abs=0.06)
    assert shares["test"] == pytest.approx(0.15, abs=0.06)


def test_balanced_mode_equalizes_strata(kaggle_stub, tmp_path) -> None:
    data_dir = tmp_path / "data"
    cxr.main(
        ["--data-dir", str(data_dir), "--limit", "40", "--balanced", "--seed", "42"]
    )
    metadata = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)
    counts = metadata["finding_stratum"].value_counts()
    # Capacity-limited strata may fall short, but nothing may run away.
    assert counts.max() - counts.min() <= 1


def test_same_seed_reproduces_selection_and_splits(kaggle_stub, tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    cxr.main(["--data-dir", str(first), "--limit", "60", "--seed", "42"])
    cxr.main(["--data-dir", str(second), "--limit", "60", "--seed", "42"])
    left = pd.read_csv(first / "chestxray14" / "metadata.csv", dtype=str)
    right = pd.read_csv(second / "chestxray14" / "metadata.csv", dtype=str)
    pd.testing.assert_frame_equal(left, right)


def test_different_seed_changes_selection(kaggle_stub, tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    cxr.main(["--data-dir", str(first), "--limit", "60", "--seed", "42"])
    cxr.main(["--data-dir", str(second), "--limit", "60", "--seed", "99"])
    left = pd.read_csv(first / "chestxray14" / "metadata.csv", dtype=str)
    right = pd.read_csv(second / "chestxray14" / "metadata.csv", dtype=str)
    assert set(left["image_id"]) != set(right["image_id"])


def test_rerun_merges_and_preserves_existing_patient_splits(
    kaggle_stub, tmp_path
) -> None:
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--limit", "60", "--seed", "42"])
    before = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)

    cxr.main(["--data-dir", str(data_dir), "--seed", "42"])
    after = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)

    assert len(after) > len(before)
    # Every originally indexed image keeps its original split.
    merged = before.merge(after, on="image_id", suffixes=("_before", "_after"))
    assert len(merged) == len(before)
    assert (merged["split_before"] == merged["split_after"]).all()
    # Growing the cohort must still not leak a patient across splits.
    assert (after.groupby("patient_id")["split"].nunique() == 1).all()


def test_fresh_rebuilds_the_manifest(kaggle_stub, tmp_path) -> None:
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--seed", "42"])
    full = len(pd.read_csv(data_dir / "chestxray14" / "metadata.csv"))

    cxr.main(["--data-dir", str(data_dir), "--limit", "30", "--fresh", "--seed", "42"])
    rebuilt = pd.read_csv(data_dir / "chestxray14" / "metadata.csv")
    assert len(rebuilt) == 30 < full


def test_dry_run_writes_no_metadata(kaggle_stub, tmp_path) -> None:
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--dry-run", "--limit", "10"])
    assert not (data_dir / "chestxray14" / "metadata.csv").is_file()


def test_corrupt_images_are_never_indexed(monkeypatch, tmp_path) -> None:
    cache = build_fake_cache(tmp_path, corrupt=5)
    module = types.ModuleType("kagglehub")
    module.dataset_download = lambda handle: str(cache)  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "kagglehub", module)
    credentials = tmp_path / "home" / ".kaggle"
    credentials.mkdir(parents=True)
    (credentials / "kaggle.json").write_text("{}")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--seed", "42"])
    metadata = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)

    corrupt_ids = {f"{index:08d}_000.png" for index in range(1, 6)}
    assert not corrupt_ids & set(metadata["image_id"])
    for row in metadata.itertuples(index=False):
        assert (data_dir / "chestxray14" / row.path).is_file()


def test_label_rows_without_images_are_skipped(monkeypatch, tmp_path) -> None:
    cache = build_fake_cache(tmp_path, label_rows_without_images=7)
    module = types.ModuleType("kagglehub")
    module.dataset_download = lambda handle: str(cache)  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "kagglehub", module)
    credentials = tmp_path / "home" / ".kaggle"
    credentials.mkdir(parents=True)
    (credentials / "kaggle.json").write_text("{}")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--seed", "42"])
    metadata = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)
    assert not any(name.startswith("9999") for name in metadata["image_id"])


def test_repair_reports_orphaned_rows_without_deleting(kaggle_stub, tmp_path) -> None:
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--limit", "40", "--seed", "42"])
    metadata_path = data_dir / "chestxray14" / "metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype=str)

    victim = data_dir / "chestxray14" / metadata.iloc[0]["path"]
    victim.unlink()

    orphaned = cxr.run_repair_audit(data_dir, cxr.configure_logging(data_dir, "t", True))
    assert orphaned == 1
    # Report-only: the manifest itself is untouched.
    assert len(pd.read_csv(metadata_path)) == len(metadata)


def test_reports_are_written(kaggle_stub, tmp_path) -> None:
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--limit", "60", "--seed", "42"])
    logs = data_dir / "logs"
    assert (logs / "chestxray14_sample_distribution.json").is_file()
    assert (logs / "chestxray14_rare_strata.json").is_file()
    assert (logs / "chestxray14_split_crosstab.csv").is_file()


def test_only_selected_images_are_staged(kaggle_stub, tmp_path) -> None:
    """raw/ must hold the selection, not a copy of the whole archive."""
    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--limit", "50", "--seed", "42"])
    staged = list((data_dir / "chestxray14" / "raw").glob("*.png"))
    cached = list((kaggle_stub / "images").glob("*.png"))
    assert len(staged) == 50
    assert len(cached) > 50


def test_runs_without_a_credentials_file(monkeypatch, tmp_path) -> None:
    """Public datasets download anonymously; a missing token must not abort."""
    cache = build_fake_cache(tmp_path)
    module = types.ModuleType("kagglehub")
    module.dataset_download = lambda handle: str(cache)  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "kagglehub", module)

    # Deliberately no ~/.kaggle/kaggle.json and no env credentials.
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: empty_home))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    data_dir = tmp_path / "data"
    cxr.main(["--data-dir", str(data_dir), "--limit", "30", "--seed", "42"])
    metadata = pd.read_csv(data_dir / "chestxray14" / "metadata.csv", dtype=str)
    assert len(metadata) == 30