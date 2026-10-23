"""Regression tests for archiving the superseded ISIC cohort."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.archive_isic_legacy import archive_legacy_rows


METADATA_COLUMNS = (
    "image_id",
    "path",
    "split",
    "diagnosis",
    "artifact_flags",
)


def _write_image(path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(24, 96, 160)).save(path, format=image_format)


def _build_isic_tree(data_dir: Path) -> pd.DataFrame:
    """Create active and legacy rows whose paths cannot be inferred from IDs."""
    dataset_dir = data_dir / "isic"
    rows = pd.DataFrame(
        [
            {
                "image_id": "legacy-id-one",
                "path": "raw/sequential-original.jpeg",
                "split": "unassigned",
                "diagnosis": "Nevus, Clark",
                "artifact_flags": "{}",
            },
            {
                "image_id": "active-id",
                "path": "raw/current-balanced-image.png",
                "split": "train",
                "diagnosis": "Melanoma",
                "artifact_flags": '{"image_type":"dermoscopic"}',
            },
            {
                "image_id": "legacy-id-two",
                "path": "raw/nested/second-source.png",
                "split": "unassigned",
                "diagnosis": "Basal cell carcinoma",
                "artifact_flags": "{}",
            },
        ],
        columns=METADATA_COLUMNS,
    )
    _write_image(dataset_dir / rows.loc[0, "path"], "JPEG")
    _write_image(dataset_dir / rows.loc[1, "path"], "PNG")
    _write_image(dataset_dir / rows.loc[2, "path"], "PNG")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(dataset_dir / "metadata.csv", index=False)
    return rows


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_archive_dry_run_does_not_change_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _build_isic_tree(data_dir)
    before = _tree_snapshot(data_dir)

    counts = archive_legacy_rows(data_dir, expected_count=2, dry_run=True)

    assert counts == (2, 1)
    assert _tree_snapshot(data_dir) == before
    assert not (data_dir / "isic" / "legacy_unstratified").exists()
    assert not (data_dir / "logs").exists()


def test_archive_moves_only_unassigned_paths_and_preserves_active_rows(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    original = _build_isic_tree(data_dir)
    dataset_dir = data_dir / "isic"

    counts = archive_legacy_rows(data_dir, expected_count=2, dry_run=False)

    assert counts == (2, 1)
    archive_dir = dataset_dir / "legacy_unstratified"

    # Sources are resolved from metadata paths, including nested and mixed-extension
    # names that intentionally differ from their image IDs.
    assert not (dataset_dir / "raw" / "sequential-original.jpeg").exists()
    assert not (dataset_dir / "raw" / "nested" / "second-source.png").exists()
    assert (archive_dir / "raw" / "sequential-original.jpeg").is_file()
    assert (archive_dir / "raw" / "nested" / "second-source.png").is_file()

    active_image = dataset_dir / "raw" / "current-balanced-image.png"
    assert active_image.is_file()
    active = pd.read_csv(dataset_dir / "metadata.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(
        active.reset_index(drop=True),
        original.loc[[1], list(METADATA_COLUMNS)].reset_index(drop=True),
    )

    legacy = pd.read_csv(
        archive_dir / "metadata.csv", dtype=str, keep_default_na=False
    )
    pd.testing.assert_frame_equal(
        legacy.reset_index(drop=True),
        original.loc[[0, 2], list(METADATA_COLUMNS)].reset_index(drop=True),
    )
    readme = (archive_dir / "README.md").read_text(encoding="utf-8")
    assert "sequential, unstratified ISIC downloader" in readme
    assert "diagnosis-imbalanced cohort" in readme
    assert "2 images" in readme
