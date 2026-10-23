"""Tests for pipeline.iqa_metrics, scripts.compute_iqa, and scripts.evaluate_iqa.

Two deliberate design points these tests protect:

1. Direction validation for BRISQUE and CLIP-IQA uses a real photograph, not a
   synthetic pattern. During development, both metrics gave a *better* score to
   a degraded synthetic sinusoidal test image than to the clean one - because
   their natural-scene-statistics/CLIP-prior assumptions are calibrated on real
   photographic content and behave unpredictably outside it. A synthetic
   pattern is therefore not a valid way to check these two metrics' direction;
   classical metrics (Laplacian variance, noise estimate, ...) are fine to
   check on synthetic images since they are direct pixel-statistics, not
   learned/calibrated models.

2. brightness_shift's offset is randomly signed (see pipeline.defects), so
   brightness_deviation must be tested against both a darker and a lighter
   image and shown to score both as comparably "deviant" - a naive mean-
   intensity check would only catch one sign.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import ndimage as ndi

import pipeline.iqa_metrics as iqa
import scripts.compute_iqa as compute_iqa_module
import scripts.evaluate_iqa as evaluate_iqa_module
from scripts.compute_iqa import main as compute_main
from scripts.evaluate_iqa import main as evaluate_main


# ---------------------------------------------------------------------------
# Fixtures: real photo (for learned-metric direction) and structured synthetic
# images (for classical-metric direction and the end-to-end pipeline tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_photo() -> np.ndarray:
    """A real natural photograph, needed to validate BRISQUE/CLIP-IQA direction."""
    from skimage import data as skdata

    return skdata.camera()


def structured_image(size: int = 128, seed: int = 0) -> np.ndarray:
    """A synthetic image with real spatial structure (not pure noise)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 40 * np.sin(x / 9.0) + 30 * np.cos(y / 13.0)
    base += rng.normal(0, 3, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Classical metrics: correctness and direction on synthetic degradations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "metric_name,degrade,expect_worse_is",
    [
        ("laplacian_variance", lambda img: ndi.gaussian_filter(img, sigma=4), "lower"),
        ("tenengrad", lambda img: ndi.gaussian_filter(img, sigma=4), "lower"),
        (
            "noise_estimate",
            lambda img: np.clip(
                img.astype(np.float64) + np.random.default_rng(1).normal(0, 30, img.shape),
                0, 255,
            ).astype(np.uint8),
            "higher",
        ),
    ],
)
def test_classical_metric_direction_on_synthetic_degradation(
    metric_name, degrade, expect_worse_is
) -> None:
    clean = structured_image()
    degraded = degrade(clean)
    clean_score = iqa.METRICS[metric_name].function(clean)
    degraded_score = iqa.METRICS[metric_name].function(degraded)
    if expect_worse_is == "lower":
        assert degraded_score < clean_score
    else:
        assert degraded_score > clean_score


def test_laplacian_and_tenengrad_are_confounded_by_noise() -> None:
    """Documented limitation: derivative-based sharpness metrics rise under
    noise, since noise itself is high-frequency content. This is expected
    behaviour to be aware of when reading the AUROC tables, not a bug."""
    clean = structured_image()
    noisy = np.clip(
        clean.astype(np.float64) + np.random.default_rng(2).normal(0, 30, clean.shape),
        0, 255,
    ).astype(np.uint8)
    assert iqa.laplacian_variance(noisy) > iqa.laplacian_variance(clean)
    assert iqa.tenengrad(noisy) > iqa.tenengrad(clean)


def test_michelson_contrast_handles_uniform_image_without_division_error() -> None:
    uniform = np.full((64, 64), 128, dtype=np.uint8)
    value = iqa.michelson_contrast(uniform)
    assert np.isfinite(value)


def test_rgb_and_grayscale_inputs_are_both_supported() -> None:
    gray = structured_image()
    rgb = np.stack([gray, gray, gray], axis=-1)
    for metric_name in iqa.available_metrics(include_torch=False):
        gray_score = iqa.METRICS[metric_name].function(gray)
        rgb_score = iqa.METRICS[metric_name].function(rgb)
        assert np.isfinite(gray_score)
        assert np.isfinite(rgb_score)


# ---------------------------------------------------------------------------
# Learned metrics: direction validated on a REAL photo only, per the module
# docstring above
# ---------------------------------------------------------------------------

def test_brisque_and_clip_iqa_direction_on_real_photo(real_photo) -> None:
    noisy = np.clip(
        real_photo.astype(np.float64)
        + np.random.default_rng(0).normal(0, 25, real_photo.shape),
        0, 255,
    ).astype(np.uint8)
    blurred = ndi.gaussian_filter(real_photo, sigma=4).astype(np.uint8)

    clean_brisque = iqa.brisque(real_photo)
    noisy_brisque = iqa.brisque(noisy)
    blurred_brisque = iqa.brisque(blurred)
    # brisque: higher = worse
    assert clean_brisque < noisy_brisque
    assert clean_brisque < blurred_brisque

    clean_clip = iqa.clip_iqa(real_photo)
    noisy_clip = iqa.clip_iqa(noisy)
    blurred_clip = iqa.clip_iqa(blurred)
    # clip_iqa: higher = better
    assert clean_clip > noisy_clip
    assert clean_clip > blurred_clip


def test_clip_iqa_replicates_grayscale_to_three_channels() -> None:
    """Verifies the explicit replicate_to_rgb path is actually taken, rather
    than relying on undocumented single-channel tolerance inside piq."""
    gray = structured_image()
    tensor = iqa.to_torch_tensor(gray, replicate_to_rgb=True)
    assert tensor.shape[1] == 3
    assert np.isfinite(iqa.clip_iqa(gray))


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

def test_every_registered_metric_has_an_explicit_polarity_or_is_marked_descriptive() -> None:
    for name, spec in iqa.METRICS.items():
        assert spec.higher_is_better in (True, False, None), (
            f"{name} has an invalid polarity {spec.higher_is_better!r}"
        )


def test_compute_all_metrics_returns_finite_values_for_every_registered_metric() -> None:
    image = structured_image()
    scores = iqa.compute_all_metrics(image)
    assert set(scores) == set(iqa.available_metrics())
    for name, value in scores.items():
        assert np.isfinite(value), f"{name} produced a non-finite value: {value}"


# ---------------------------------------------------------------------------
# BrightnessReference / brightness_deviation: the randomly-signed-offset case
# ---------------------------------------------------------------------------

def test_brightness_deviation_treats_darker_and_lighter_as_comparably_deviant() -> None:
    """brightness_shift applies a randomly signed offset (see pipeline.defects),
    so only absolute deviation from a reference is monotonic with severity."""
    base = structured_image()
    reference = iqa.BrightnessReference.fit([base])
    darker = np.clip(base.astype(np.int16) - 50, 0, 255).astype(np.uint8)
    lighter = np.clip(base.astype(np.int16) + 50, 0, 255).astype(np.uint8)

    darker_deviation = iqa.brightness_deviation(darker, reference)
    lighter_deviation = iqa.brightness_deviation(lighter, reference)
    clean_deviation = iqa.brightness_deviation(base, reference)

    assert darker_deviation > clean_deviation
    assert lighter_deviation > clean_deviation
    # A naive signed-mean check would show these as opposite; deviation must not.
    assert darker_deviation == pytest.approx(lighter_deviation, rel=0.15)


def test_brightness_reference_uses_median_not_mean_for_robustness() -> None:
    dim = np.full((32, 32), 50, dtype=np.uint8)
    bright_outlier = np.full((32, 32), 240, dtype=np.uint8)
    reference = iqa.BrightnessReference.fit([dim, dim, dim, dim, bright_outlier])
    # Median of {50,50,50,50,240} is 50, not the mean (~78).
    assert reference.reference_mean == pytest.approx(50, abs=1)


def test_brightness_reference_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        iqa.BrightnessReference.fit([])


def test_brightness_reference_round_trips_through_json() -> None:
    reference = iqa.BrightnessReference.fit([structured_image()])
    restored = iqa.BrightnessReference.from_dict(reference.to_dict())
    assert restored.reference_mean == pytest.approx(reference.reference_mean)
    assert restored.n_images == reference.n_images


# ---------------------------------------------------------------------------
# End-to-end: compute_iqa.py + evaluate_iqa.py against a real injected dataset
# ---------------------------------------------------------------------------

def _build_processed_dataset(tmp_path: Path) -> Path:
    """Build a small but real dataset via the actual defect-injection module,
    not a mock, so these tests exercise real integration."""
    import pipeline.defects as defects

    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True)

    rows = []
    payload = []
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    for index, split in enumerate(splits):
        image = structured_image(size=64, seed=index)
        name = f"img{index:03d}.png"
        Image.fromarray(image, mode="L").save(raw_dir / name)
        rows.append({"image_id": name, "path": f"raw/{name}", "split": split})

        # Half clean, half degraded across a spread of defects/severities so
        # severity correlation has more than one point per defect.
        if index % 2 == 0:
            defect_type, severity, out_array = "clean", "0", image
        else:
            defect_name = ["gaussian_noise", "gaussian_blur", "jpeg_compression"][index % 3]
            severity = 1 + (index % 3)
            degraded, _ = defects.apply_defect(
                Image.fromarray(image, mode="L"), defect_name, severity, rng=index
            )
            defect_type, out_array = defect_name, np.asarray(degraded)
            severity = str(severity)
        out_name = f"{name}_{defect_type}_{severity}.jpg"
        Image.fromarray(out_array, mode="L").save(processed_dir / out_name, quality=100)
        payload.append(
            {
                "image_id": Path(out_name).stem,
                "original_image_id": name,
                "path": f"processed/{out_name}",
                "split": split,
                "defect_type": defect_type,
                "severity_level": severity,
                "canonical_size": 64,
                "defect_params_json": "{}",
            }
        )

    pd.DataFrame(rows).to_csv(dataset_dir / "metadata.csv", index=False)
    pd.DataFrame(payload).to_csv(dataset_dir / "processed_metadata.csv", index=False)

    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True)
    import json

    (logs_dir / "demo_ground_truth_validation.json").write_text(
        json.dumps(
            {
                "binary_detection_only": [],
                "severity_correlation_eligible": [
                    "gaussian_noise", "gaussian_blur", "jpeg_compression",
                ],
            }
        )
    )
    return data_dir


def test_compute_iqa_end_to_end_scores_every_row(tmp_path: Path) -> None:
    data_dir = _build_processed_dataset(tmp_path)
    compute_main(["--dataset", "demo", "--data-dir", str(data_dir)])

    scores = pd.read_csv(data_dir / "demo" / "iqa_scores.csv")
    processed = pd.read_csv(data_dir / "demo" / "processed_metadata.csv")
    assert len(scores) == len(processed)
    for metric_name in iqa.available_metrics():
        assert scores[metric_name].notna().all()
    assert scores["brightness_deviation"].notna().all()

    reference_path = data_dir / "demo" / "brightness_reference.json"
    assert reference_path.is_file()


def test_compute_iqa_fitted_brightness_reference_is_from_clean_train_only(
    tmp_path: Path,
) -> None:
    data_dir = _build_processed_dataset(tmp_path)
    compute_main(["--dataset", "demo", "--data-dir", str(data_dir)])

    import json

    reference = json.loads((data_dir / "demo" / "brightness_reference.json").read_text())
    processed = pd.read_csv(data_dir / "demo" / "processed_metadata.csv", dtype=str)
    expected_n = len(
        processed[(processed.defect_type == "clean") & (processed.split == "train")]
    )
    assert reference["n_images"] == expected_n


def test_evaluate_iqa_end_to_end_produces_matrices_with_correct_shape(
    tmp_path: Path,
) -> None:
    data_dir = _build_processed_dataset(tmp_path)
    compute_main(["--dataset", "demo", "--data-dir", str(data_dir)])
    evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir)])

    logs_dir = data_dir / "logs"
    auroc = pd.read_csv(logs_dir / "demo_iqa_auroc.csv", index_col=0)
    rho = pd.read_csv(logs_dir / "demo_iqa_severity_rho.csv", index_col=0)

    assert set(auroc.index) == set(iqa.available_metrics()) | {"brightness_deviation"}
    assert set(auroc.columns) == {"gaussian_noise", "gaussian_blur", "jpeg_compression"}
    # Only eligible defects should appear as columns at all in the rho matrix.
    assert set(rho.columns) <= {"gaussian_noise", "gaussian_blur", "jpeg_compression"}

    summary_path = logs_dir / "demo_iqa_summary.json"
    assert summary_path.is_file()


def test_evaluate_iqa_requires_ground_truth_validation_report(tmp_path: Path) -> None:
    data_dir = _build_processed_dataset(tmp_path)
    compute_main(["--dataset", "demo", "--data-dir", str(data_dir)])
    (data_dir / "logs" / "demo_ground_truth_validation.json").unlink()

    with pytest.raises(FileNotFoundError):
        evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir)])


def test_polarity_is_respected_when_building_auroc() -> None:
    """A synthetic sanity check that a 'higher is better' metric scoring
    degraded images higher than clean produces a LOW auroc after orientation,
    not a high one - i.e. the sign flip in oriented_scores is actually wired
    correctly end to end, not just present in isolation."""
    clean = pd.DataFrame({"defect_type": ["clean"] * 5, "fake_metric": [10, 11, 9, 10, 12]})
    degraded = pd.DataFrame(
        {"defect_type": ["fake_defect"] * 5, "fake_metric": [20, 21, 19, 22, 18]}
    )
    scored = pd.concat([clean, degraded], ignore_index=True)

    original_polarity = evaluate_iqa_module.metric_polarity
    evaluate_iqa_module.metric_polarity = lambda name: True if name == "fake_metric" else None
    try:
        result = evaluate_iqa_module.binary_detection(scored, "fake_metric", "fake_defect")
    finally:
        evaluate_iqa_module.metric_polarity = original_polarity

    # Degraded images scored HIGHER on a "higher is better" metric: this metric
    # is being actively fooled, so AUROC should be near 0, not near 1.
    assert result["auroc"] < 0.2


def test_bootstrap_ci_returns_nan_gracefully_on_tiny_samples() -> None:
    """bootstrap_auroc_ci now returns a BootstrapCI dataclass rather than a
    tuple - it moved to pipeline.significance during Stage 4 so the paired
    bootstrap and the single-scorer CI share one implementation."""
    from pipeline.significance import bootstrap_auroc_ci

    labels = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    ci = bootstrap_auroc_ci(labels, scores, n_bootstrap=20)
    assert np.isfinite(ci.point)
    # With only 2 points, most resamples will be single-class; nan is the
    # honest answer, not a crash.
    assert np.isnan(ci.ci_low) or np.isfinite(ci.ci_low)