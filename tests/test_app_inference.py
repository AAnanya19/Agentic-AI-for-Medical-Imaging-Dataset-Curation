"""Tests for single-image inspection behind the demonstration interface.

Four invariants:

1. **Every source mode the corpora contain must load.** ChestX-ray14 has 23
   fully-opaque RGBA images; ISIC sources span 0.01 to 302 megapixels; 16-bit
   greyscale, CMYK and palette images all occur in the wild. Each of these has
   broken some part of this project at least once.

2. **A bad upload must produce a message, never a traceback.** The interface
   receives untrusted input, and "could not read the image" is actionable where a
   PIL exception is not.

3. **The interface must agree with the batch pipeline.** It reuses
   ``DataInspectorAgent.inspect_image`` rather than reimplementing the decision, so
   a demo that disagreed with the pipeline - which would be worse than no demo -
   is impossible by construction.

4. **Scores and thresholds must be displayed in the same space.** Stored
   thresholds are oriented (higher = worse); showing a raw score against an
   oriented threshold made a clean image with Laplacian variance 610 appear to
   fail a threshold of -614.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.inference import (
    MAX_UPLOAD_MEGAPIXELS, InspectionContext, UnsupportedImageError,
    available_datasets, check_resolution_only, inspect_single_image,
    load_upload, per_check_table,
)


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------

def as_upload(image: Image.Image) -> io.BytesIO:
    buffer = io.BytesIO()
    fmt = "TIFF" if image.mode in ("CMYK", "I;16", "I") else "PNG"
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


def noise_array(height: int = 512, width: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) * 255).astype(np.uint8)


@pytest.mark.parametrize(
    "mode_name,build",
    [
        ("L", lambda a: Image.fromarray(a, "L")),
        ("RGB", lambda a: Image.fromarray(np.dstack([a, a, a]), "RGB")),
        # The ChestX-ray14 case: 23 images carry a fully opaque alpha channel.
        ("RGBA opaque", lambda a: Image.fromarray(
            np.dstack([a, a, a, np.full_like(a, 255)]), "RGBA")),
        ("RGBA translucent", lambda a: Image.fromarray(
            np.dstack([a, a, a, np.full_like(a, 128)]), "RGBA")),
        ("16-bit", lambda a: Image.fromarray(a.astype(np.uint16) * 257)),
        ("CMYK", lambda a: Image.fromarray(np.dstack([a, a, a, a]), "CMYK")),
        ("palette", lambda a: Image.fromarray(a, "L").convert("P")),
    ],
)
def test_every_source_mode_canonicalises(mode_name, build) -> None:
    canonical, source = load_upload(as_upload(build(noise_array())))
    assert canonical.size == (256, 256)
    assert canonical.mode in ("L", "RGB"), mode_name
    assert source["original_mode"] is not None


def test_a_large_source_is_accepted() -> None:
    """ISIC sources reach 302 megapixels, so large inputs are normal here."""
    canonical, source = load_upload(as_upload(Image.fromarray(noise_array(2000, 3000))))
    assert canonical.size == (256, 256)
    assert source["megapixels"] == pytest.approx(6.0, abs=0.1)


def test_an_absurdly_large_source_is_refused_before_scoring() -> None:
    """Pillow's decompression-bomb guard is relaxed inside the pipeline for
    legitimate medical sources, so an explicit ceiling is needed for untrusted
    input."""
    class HugeStub:
        size = (30000, 30000)
        mode = "L"

        def load(self) -> None:
            pass

    import app.inference as inference

    original = Image.open
    Image.open = lambda *args, **kwargs: HugeStub()   # type: ignore[assignment]
    try:
        with pytest.raises(UnsupportedImageError, match="megapixels"):
            inference.load_upload(io.BytesIO(b"x"))
    finally:
        Image.open = original   # type: ignore[assignment]


def test_an_upscaled_source_is_flagged_as_such() -> None:
    """A 32x32 source upscaled to 256x256 has interpolated detail, so every
    sharpness metric is then measuring fabrication."""
    _, source = load_upload(as_upload(Image.fromarray(noise_array(32, 32))))
    assert source["upscaled_to_canonical"] is True


def test_a_large_source_is_not_flagged_as_upscaled() -> None:
    _, source = load_upload(as_upload(Image.fromarray(noise_array(512, 512))))
    assert source["upscaled_to_canonical"] is False


def test_aspect_ratio_is_recorded_orientation_independently() -> None:
    _, wide = load_upload(as_upload(Image.fromarray(noise_array(200, 800))))
    _, tall = load_upload(as_upload(Image.fromarray(noise_array(800, 200))))
    assert wide["aspect_ratio"] == pytest.approx(tall["aspect_ratio"])
    assert wide["aspect_ratio"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "payload,description",
    [
        (b"this is not an image at all", "text file"),
        (b"", "empty file"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "truncated png"),
    ],
)
def test_a_bad_upload_raises_an_actionable_message(payload, description) -> None:
    with pytest.raises(UnsupportedImageError) as error:
        load_upload(io.BytesIO(payload))
    # A message the user can act on, not a PIL internal.
    assert len(str(error.value)) > 20, description


def test_a_tiny_image_is_refused() -> None:
    with pytest.raises(UnsupportedImageError, match="too small"):
        load_upload(as_upload(Image.fromarray(noise_array(4, 4))))


# ---------------------------------------------------------------------------
# Resolution for a single image
# ---------------------------------------------------------------------------

def test_single_image_resolution_uses_absolute_criteria_only() -> None:
    """The distributional check needs a dataset, which one upload cannot supply -
    so the limitation is stated rather than the check being faked."""
    result = check_resolution_only({
        "original_width": 1024, "original_height": 1024, "aspect_ratio": 1.0,
    })
    assert result["flagged"] is False
    assert "single image" in result["note"]


def test_an_undersized_upload_is_flagged() -> None:
    result = check_resolution_only({
        "original_width": 100, "original_height": 100, "aspect_ratio": 1.0,
    })
    assert result["below_min_side"] is True
    assert result["flagged"] is True


def test_an_elongated_upload_is_flagged() -> None:
    result = check_resolution_only({
        "original_width": 3000, "original_height": 500, "aspect_ratio": 6.0,
    })
    assert result["extreme_aspect_ratio"] is True


# ---------------------------------------------------------------------------
# Context loading
# ---------------------------------------------------------------------------

def _build_scored_dataset(tmp_path: Path) -> Path:
    """A dataset with the artefacts a completed batch run would leave behind."""
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    dataset_dir.mkdir(parents=True)
    logs_dir = data_dir / "logs"
    logs_dir.mkdir()

    rng = np.random.default_rng(3)
    metrics = [
        "laplacian_variance", "tenengrad", "noise_estimate", "rms_contrast",
        "michelson_contrast", "shannon_entropy", "brisque", "clip_iqa",
        "brightness_deviation",
    ]
    rows = []
    for split, n_clean, n_degraded in (("val", 120, 60), ("test", 40, 20)):
        for index in range(n_clean):
            row = {
                "image_id": f"{split}_clean_{index}", "split": split,
                "defect_type": "clean", "severity_level": "0",
            }
            for name in metrics:
                row[name] = rng.normal(0, 1)
            rows.append(row)
        for index in range(n_degraded):
            row = {
                "image_id": f"{split}_blur_{index}", "split": split,
                "defect_type": "gaussian_blur", "severity_level": "2",
            }
            for name in metrics:
                row[name] = rng.normal(3, 1)
            rows.append(row)
    pd.DataFrame(rows).to_csv(dataset_dir / "iqa_scores.csv", index=False)

    (dataset_dir / "brightness_reference.json").write_text(
        json.dumps({"reference_mean": 128.0, "n_images": 120})
    )
    (logs_dir / "demo_inspection.json").write_text(json.dumps({
        "dataset": "demo",
        "thresholds": {
            "overall_quality": {
                "flag_threshold": 0.5, "reject_threshold": 0.8,
                "flag_fpr_target": 0.05, "reject_fpr_target": 0.01,
                "n_clean_validation_rows": 120, "score_column": "fused_logistic",
                "higher_is_better": False, "can_decide": True,
            },
        },
        "images": [],
        "provenance": {"run_id": "batch123", "schema_version": "1.0"},
    }))
    return data_dir


def test_available_datasets_lists_only_those_with_thresholds(tmp_path: Path) -> None:
    data_dir = _build_scored_dataset(tmp_path)
    assert available_datasets(data_dir) == ["demo"]


def test_a_dataset_without_a_batch_run_is_not_offered(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    assert available_datasets(data_dir) == []


def test_loading_without_a_batch_run_explains_what_to_do(tmp_path: Path) -> None:
    """Thresholds are a property of a dataset; a single image cannot supply the
    clean validation distribution they are derived from."""
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        InspectionContext.load("demo", data_dir)


def test_the_context_records_which_batch_run_supplied_the_thresholds(
    tmp_path: Path,
) -> None:
    """Applying one modality's thresholds to another inflated false positives
    eighteen-fold, so the interface must be able to state the source."""
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    assert context.inspection_run_id == "batch123"
    assert context.dataset == "demo"


def test_the_composite_scorer_is_reconstructed_from_validation_rows(
    tmp_path: Path,
) -> None:
    """Without it the overall verdict is unavailable and the interface degrades to
    per-check attribution with no decision."""
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    assert context.fusion_model is not None
    assert context.fusion_feature_names


def test_a_missing_scores_file_degrades_rather_than_breaking(tmp_path: Path) -> None:
    data_dir = _build_scored_dataset(tmp_path)
    (data_dir / "demo" / "iqa_scores.csv").unlink()
    context = InspectionContext.load("demo", data_dir)
    assert context.fusion_model is None      # degraded, not raised


# ---------------------------------------------------------------------------
# End-to-end inspection
# ---------------------------------------------------------------------------

def test_inspecting_an_upload_produces_a_decision(tmp_path: Path) -> None:
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    result, canonical = inspect_single_image(
        as_upload(Image.fromarray(noise_array())), context
    )
    assert result["decision"] in ("accept", "flag", "reject")
    assert canonical.size == (256, 256)
    assert result["thresholds_from"]["dataset"] == "demo"
    assert result["thresholds_from"]["inspection_run_id"] == "batch123"


def test_the_result_carries_raw_scores_and_source_properties(tmp_path: Path) -> None:
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    result, _ = inspect_single_image(
        as_upload(Image.fromarray(noise_array(300, 900))), context
    )
    assert "laplacian_variance" in result["raw_scores"]
    assert result["source"]["original_width"] == 900
    assert result["source"]["aspect_ratio"] == pytest.approx(3.0)


def test_the_per_check_table_shows_score_and_threshold_in_the_same_space(
    tmp_path: Path,
) -> None:
    """Regression on a real display bug. Stored thresholds are oriented, so a raw
    Laplacian variance of 610 shown against a threshold of -614 appeared to fail
    when it had in fact passed."""
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    result, _ = inspect_single_image(
        as_upload(Image.fromarray(noise_array())), context
    )
    table = per_check_table(result)
    if table.empty:
        pytest.skip("no assessable checks in this configuration")
    assert "score (oriented)" in table.columns
    assert "raw score" in table.columns
    # Where both are present, a flag must be consistent with the comparison.
    for row in table.itertuples():
        oriented = getattr(row, "_2")           # "score (oriented)"
        threshold = getattr(row, "_3")          # "flag threshold"
        if pd.notna(oriented) and pd.notna(threshold):
            assert row.flagged == bool(oriented > threshold), row.check


def test_flagged_checks_are_listed_first(tmp_path: Path) -> None:
    """The explanation for a verdict belongs at the top of the table."""
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    result, _ = inspect_single_image(
        as_upload(Image.fromarray(noise_array())), context
    )
    table = per_check_table(result)
    if table.empty or not table["flagged"].any():
        pytest.skip("no checks fired for this image")
    assert bool(table.iloc[0]["flagged"]) is True


def test_a_bad_upload_does_not_reach_the_scorer(tmp_path: Path) -> None:
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    with pytest.raises(UnsupportedImageError):
        inspect_single_image(io.BytesIO(b"not an image"), context)


def test_inspection_is_reproducible(tmp_path: Path) -> None:
    """The same upload must always give the same verdict."""
    data_dir = _build_scored_dataset(tmp_path)
    context = InspectionContext.load("demo", data_dir)
    upload = Image.fromarray(noise_array())
    first, _ = inspect_single_image(as_upload(upload), context)
    second, _ = inspect_single_image(as_upload(upload), context)
    assert first["decision"] == second["decision"]
    assert first["raw_scores"] == second["raw_scores"]


def test_the_interface_reuses_the_pipeline_decision_logic() -> None:
    """A demo that disagreed with the batch pipeline would be worse than no demo,
    so the decision must come from the agent rather than a reimplementation."""
    import inspect as inspect_module

    import app.inference as inference

    source = inspect_module.getsource(inference.inspect_single_image)
    assert "agent.inspect_image" in source