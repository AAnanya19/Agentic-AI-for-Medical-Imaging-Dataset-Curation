"""Tests for pipeline.embeddings, pipeline.anomaly, and scripts.evaluate_anomaly.

The methodological guarantees these tests exist to protect:

1. **No fitting-data leakage.** Detectors are fitted on clean *train*-split
   embeddings, so the evaluation must never include train rows. A silent leak
   here would inflate every Stage 3 and Stage 4 number.
2. **Never fitted on degraded data.** The premise of the whole stage is modelling
   normality; fitting on defects would make it a supervised classifier instead.
3. **Ledoit-Wolf shrinkage is actually applied.** With ISIC's 383 clean training
   images against 512-d ResNet features (n/d = 0.75), the unshrunk covariance
   has a condition number around 8.6e18. NumPy still inverts it without error
   and returns finite garbage, which is more dangerous than a crash - so the
   shrinkage must be verified present, not assumed.
4. **Uniform score direction.** Every detector returns higher = more anomalous,
   so downstream code needs no per-detector sign handling.

Backbone tests use ``pretrained=False``. Random weights produce meaningless
*features*, but the architecture, input sizing, patch alignment, channel
handling and output dimensions are identical - which is exactly what these
tests check. Weight downloading is not exercised here.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import pipeline.anomaly as anomaly
import pipeline.embeddings as embeddings


# ---------------------------------------------------------------------------
# Detector behaviour
# ---------------------------------------------------------------------------

def make_clean_and_anomalous(
    n_clean: int = 300, n_anomalous: int = 60, dimensions: int = 64, shift: float = 4.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    clean_fit = rng.normal(0, 1, (n_clean, dimensions))
    clean_held_out = rng.normal(0, 1, (n_anomalous, dimensions))
    anomalous = rng.normal(shift, 1, (n_anomalous, dimensions))
    return clean_fit, clean_held_out, anomalous


@pytest.mark.parametrize(
    "detector_factory",
    [
        lambda: anomaly.KNNDistance(n_neighbours=5),
        lambda: anomaly.MahalanobisDistance(),
        lambda: anomaly.IsolationForestDetector(n_estimators=50),
    ],
)
def test_every_detector_scores_anomalies_higher_than_clean(detector_factory) -> None:
    clean_fit, clean_held_out, anomalous = make_clean_and_anomalous()
    detector = detector_factory().fit(clean_fit)
    assert detector.score(anomalous).mean() > detector.score(clean_held_out).mean()


@pytest.mark.parametrize(
    "detector_factory",
    [
        lambda: anomaly.KNNDistance(n_neighbours=5),
        lambda: anomaly.MahalanobisDistance(),
        lambda: anomaly.IsolationForestDetector(n_estimators=50),
    ],
)
def test_detectors_refuse_to_score_before_being_fitted(detector_factory) -> None:
    detector = detector_factory()
    with pytest.raises(RuntimeError):
        detector.score(np.zeros((3, 64)))


def test_knn_rejects_k_larger_than_the_clean_training_set() -> None:
    with pytest.raises(ValueError):
        anomaly.KNNDistance(n_neighbours=50).fit(np.zeros((10, 32)))


def test_mahalanobis_applies_ledoit_wolf_shrinkage() -> None:
    clean_fit, _, _ = make_clean_and_anomalous()
    detector = anomaly.MahalanobisDistance().fit(clean_fit)
    description = detector.describe()
    assert description["covariance_estimator"] == "LedoitWolf"
    assert 0.0 <= description["shrinkage_coefficient"] <= 1.0
    assert description["n_dimensions"] == clean_fit.shape[1]


def _low_sample_warnings(caught) -> list[str]:
    """Only this module's own samples-per-dimension warning.

    Matching on RuntimeWarning alone is too broad: scipy and sklearn emit their
    own RuntimeWarnings for unrelated internal reasons, and which ones appear
    varies by library version. The test must assert on our message, not on any
    warning that happens to share its category.
    """
    return [
        str(entry.message)
        for entry in caught
        if issubclass(entry.category, RuntimeWarning)
        and "samples in" in str(entry.message)
    ]


def test_mahalanobis_warns_when_samples_are_fewer_than_dimensions() -> None:
    """The ISIC + ResNet-18 case (383 clean train vs 512-d) must be flagged,
    because the unshrunk covariance there is rank-deficient yet still
    'invertible' enough to return plausible-looking garbage."""
    rng = np.random.default_rng(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anomaly.MahalanobisDistance().fit(rng.normal(0, 1, (383, 512)))
    assert _low_sample_warnings(caught)


def test_mahalanobis_stays_silent_when_well_conditioned() -> None:
    """ChestX-ray14 (1178 clean train vs 512-d) is comfortable; no warning."""
    rng = np.random.default_rng(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anomaly.MahalanobisDistance().fit(rng.normal(0, 1, (1178, 512)))
    assert not _low_sample_warnings(caught)


def test_mahalanobis_uses_cholesky_rather_than_an_explicit_inverse() -> None:
    """Forming C^-1 is both less stable and, on some scipy versions, emits
    divide-by-zero/overflow warnings even for well-conditioned inputs."""
    rng = np.random.default_rng(8)
    detector = anomaly.MahalanobisDistance().fit(rng.normal(0, 1, (300, 64)))
    assert detector.describe()["solver"] == "cholesky_triangular_solve"


def test_mahalanobis_matches_the_textbook_definition() -> None:
    """Cholesky solve must give the same answer as the explicit-inverse formula
    on a well-conditioned problem, so the optimisation is provably equivalent."""
    rng = np.random.default_rng(9)
    clean = rng.normal(0, 1, (500, 16))
    detector = anomaly.MahalanobisDistance().fit(clean)
    queries = rng.normal(1.0, 1, (10, 16))

    from sklearn.covariance import LedoitWolf

    reference_estimator = LedoitWolf(store_precision=True).fit(clean)
    centred = queries - reference_estimator.location_
    expected = np.sqrt(
        np.einsum("ij,jk,ik->i", centred, reference_estimator.precision_, centred)
    )
    assert np.allclose(detector.score(queries), expected, rtol=1e-6)


def test_mahalanobis_remains_finite_when_samples_are_fewer_than_dimensions() -> None:
    """Shrinkage must keep distances usable in the rank-deficient regime."""
    rng = np.random.default_rng(3)
    clean = rng.normal(0, 1, (100, 384))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        detector = anomaly.MahalanobisDistance().fit(clean)
    scores = detector.score(rng.normal(5, 1, (20, 384)))
    assert np.isfinite(scores).all()
    assert (scores > 0).all()


def test_detector_suite_fits_all_detectors_and_shares_one_scaler() -> None:
    clean_fit, clean_held_out, anomalous = make_clean_and_anomalous()
    suite = anomaly.DetectorSuite(knn_neighbours=(1, 5)).fit(clean_fit)

    assert set(suite.detectors) == {"knn_k1", "knn_k5", "mahalanobis", "isolation_forest"}
    assert suite.scaler is not None

    clean_scores = suite.score_all(clean_held_out)
    anomalous_scores = suite.score_all(anomalous)
    for name in suite.detectors:
        assert anomalous_scores[name].mean() > clean_scores[name].mean(), name
        assert np.isfinite(clean_scores[name]).all()


def test_detector_suite_skips_knn_values_exceeding_the_training_size() -> None:
    clean_fit = np.random.default_rng(4).normal(0, 1, (8, 16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        suite = anomaly.DetectorSuite(knn_neighbours=(1, 5, 20)).fit(clean_fit)
    assert "knn_k20" not in suite.detectors  # 20 > 8 available samples
    assert "knn_k5" in suite.detectors


def test_detector_suite_refuses_to_score_before_fitting() -> None:
    with pytest.raises(RuntimeError):
        anomaly.DetectorSuite().score_all(np.zeros((3, 16)))


def test_isolation_forest_is_reproducible_for_a_fixed_seed() -> None:
    clean_fit, _, anomalous = make_clean_and_anomalous()
    first = anomaly.IsolationForestDetector(n_estimators=50, random_state=7).fit(clean_fit)
    second = anomaly.IsolationForestDetector(n_estimators=50, random_state=7).fit(clean_fit)
    assert np.allclose(first.score(anomalous), second.score(anomalous))


def test_detector_names_match_a_fitted_suite() -> None:
    clean_fit, _, _ = make_clean_and_anomalous()
    suite = anomaly.DetectorSuite().fit(clean_fit)
    assert set(anomaly.detector_names()) == set(suite.detectors)


# ---------------------------------------------------------------------------
# Backbones (architecture/plumbing only - random weights, see module docstring)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", sorted(embeddings.BACKBONES))
def test_backbone_produces_its_declared_feature_dimension(backbone) -> None:
    extractor = embeddings.FeatureExtractor(backbone, pretrained=False)
    gray = Image.fromarray(np.full((256, 256), 120, dtype=np.uint8), mode="L")
    features = extractor.embed_batch([gray, gray])
    assert features.shape == (2, embeddings.BACKBONES[backbone].expected_dim)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


@pytest.mark.parametrize("backbone", sorted(embeddings.BACKBONES))
def test_backbone_accepts_both_grayscale_and_rgb_sources(backbone) -> None:
    """ChestX-ray14 processed images are mode L, ISIC's are RGB; one extractor
    must handle both without the caller special-casing modality."""
    extractor = embeddings.FeatureExtractor(backbone, pretrained=False)
    gray = Image.fromarray(np.full((256, 256), 90, dtype=np.uint8), mode="L")
    rgb = Image.fromarray(np.full((256, 256, 3), 90, dtype=np.uint8), mode="RGB")
    features = extractor.embed_batch([gray, rgb])
    assert features.shape[0] == 2
    assert np.isfinite(features).all()


def test_dinov2_input_size_is_patch_aligned() -> None:
    """252 = 14 x 18. A ViT-S/14 bakes resolution into its position embeddings,
    so a non-patch-multiple size is a real correctness risk, not a style nit."""
    spec = embeddings.BACKBONES["dinov2_vits14"]
    assert spec.input_size % 14 == 0
    assert spec.input_size <= 256  # at or below our canonical size


def test_preprocess_resizes_rather_than_crops() -> None:
    """A centre crop would clip occlusion_patch defects near the frame edge and
    silently change their effective severity, so resizing is required."""
    extractor = embeddings.FeatureExtractor("resnet18", pretrained=False)
    # A marker in the extreme corner must survive preprocessing.
    array = np.zeros((256, 256), dtype=np.uint8)
    array[0:8, 0:8] = 255
    processed = extractor.preprocess(Image.fromarray(array, mode="L"))
    size = embeddings.BACKBONES["resnet18"].input_size
    assert processed.shape == (3, size, size)
    # The corner content is still present (bright relative to the dark centre).
    assert processed[0, 0, 0] > processed[0, size // 2, size // 2]


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError):
        embeddings.FeatureExtractor("not_a_real_backbone", pretrained=False)


def test_embeddings_round_trip_through_the_npz_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    image_ids = ["a", "b", "c"]
    values = np.random.default_rng(5).normal(0, 1, (3, 16)).astype(np.float32)
    metadata = {"backbone": "resnet18", "input_size": 224}

    embeddings.save_embeddings(path, image_ids, values, metadata)
    loaded_ids, loaded_values, loaded_metadata = embeddings.load_embeddings(path)

    assert loaded_ids == image_ids
    assert np.allclose(loaded_values, values)
    assert loaded_metadata == metadata


def test_empty_batch_returns_correctly_shaped_empty_array() -> None:
    extractor = embeddings.FeatureExtractor("resnet18", pretrained=False)
    features = extractor.embed_batch([])
    assert features.shape == (0, 512)


# ---------------------------------------------------------------------------
# End-to-end evaluation: the leakage guarantees
# ---------------------------------------------------------------------------

def _build_dataset_with_embeddings(tmp_path: Path) -> Path:
    """A processed dataset plus a matching embedding cache, no images needed."""
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    dataset_dir.mkdir(parents=True)
    (data_dir / "logs").mkdir()
    (data_dir / "embeddings").mkdir()

    rng = np.random.default_rng(6)
    rows = []
    vectors = []
    # 120 clean train (the fitting pool), then clean/degraded in val and test.
    for index in range(120):
        rows.append({
            "image_id": f"train_clean_{index}", "original_image_id": f"src{index}",
            "path": f"processed/train_clean_{index}.jpg", "split": "train",
            "defect_type": "clean", "severity_level": "0",
            "canonical_size": "256", "defect_params_json": "{}",
        })
        vectors.append(rng.normal(0, 1, 32))
    for split in ("val", "test"):
        for index in range(30):
            rows.append({
                "image_id": f"{split}_clean_{index}", "original_image_id": f"s{index}",
                "path": f"processed/{split}_clean_{index}.jpg", "split": split,
                "defect_type": "clean", "severity_level": "0",
                "canonical_size": "256", "defect_params_json": "{}",
            })
            vectors.append(rng.normal(0, 1, 32))
        for defect in ("gaussian_noise", "occlusion_patch"):
            for severity in (1, 2, 3):
                for repeat in range(4):
                    rows.append({
                        "image_id": f"{split}_{defect}_{severity}_{repeat}",
                        "original_image_id": f"d{repeat}",
                        "path": f"processed/{split}_{defect}_{severity}_{repeat}.jpg",
                        "split": split, "defect_type": defect,
                        "severity_level": str(severity),
                        "canonical_size": "256", "defect_params_json": "{}",
                    })
                    # Severity-proportional shift so rho is meaningful.
                    vectors.append(rng.normal(1.5 * severity, 1, 32))

    processed = pd.DataFrame(rows)
    processed.to_csv(dataset_dir / "processed_metadata.csv", index=False)

    embeddings.save_embeddings(
        data_dir / "embeddings" / "demo_resnet18.npz",
        processed["image_id"].tolist(),
        np.asarray(vectors, dtype=np.float32),
        {"backbone": "resnet18", "expected_dim": 32, "input_size": 224},
    )
    (data_dir / "logs" / "demo_ground_truth_validation.json").write_text(
        json.dumps({
            "binary_detection_only": [],
            "severity_correlation_eligible": ["gaussian_noise", "occlusion_patch"],
        })
    )
    return data_dir


def test_evaluation_never_includes_train_rows(tmp_path: Path) -> None:
    """The core leakage guarantee: detectors are fitted on clean train rows, so
    train must not appear in the evaluated set."""
    from scripts.evaluate_anomaly import main as evaluate_main

    data_dir = _build_dataset_with_embeddings(tmp_path)
    evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir),
                   "--backbone", "resnet18"])

    scored = pd.read_csv(data_dir / "demo" / "anomaly_scores_resnet18.csv")
    assert "train" not in set(scored["split"])
    assert set(scored["split"]) == {"val", "test"}


def test_detectors_are_fitted_only_on_clean_train_rows(tmp_path: Path) -> None:
    from scripts.evaluate_anomaly import main as evaluate_main

    data_dir = _build_dataset_with_embeddings(tmp_path)
    evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir),
                   "--backbone", "resnet18"])

    summary = json.loads(
        (data_dir / "logs" / "demo_anomaly_summary.json").read_text()
    )
    processed = pd.read_csv(data_dir / "demo" / "processed_metadata.csv", dtype=str)
    expected = len(
        processed[(processed.defect_type == "clean") & (processed.split == "train")]
    )
    assert summary["backbones"]["resnet18"]["n_clean_train_fitted"] == expected


def test_evaluation_writes_all_four_matrices(tmp_path: Path) -> None:
    from scripts.evaluate_anomaly import main as evaluate_main

    data_dir = _build_dataset_with_embeddings(tmp_path)
    evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir),
                   "--backbone", "resnet18"])

    logs_dir = data_dir / "logs"
    for matrix_name in ("auroc", "auprc", "recall_at_5pct_fpr", "severity_rho"):
        path = logs_dir / f"demo_anomaly_resnet18_{matrix_name}.csv"
        assert path.is_file(), matrix_name
        matrix = pd.read_csv(path, index_col=0)
        assert set(matrix.index) == set(anomaly.detector_names())


def test_evaluation_detects_the_planted_severity_signal(tmp_path: Path) -> None:
    """The fixture shifts embeddings proportionally to severity, so a working
    pipeline must recover high AUROC and positive severity correlation."""
    from scripts.evaluate_anomaly import main as evaluate_main

    data_dir = _build_dataset_with_embeddings(tmp_path)
    evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir),
                   "--backbone", "resnet18"])

    auroc = pd.read_csv(
        data_dir / "logs" / "demo_anomaly_resnet18_auroc.csv", index_col=0
    )
    rho = pd.read_csv(
        data_dir / "logs" / "demo_anomaly_resnet18_severity_rho.csv", index_col=0
    )
    assert auroc.loc["mahalanobis", "gaussian_noise"] > 0.9
    assert rho.loc["mahalanobis", "gaussian_noise"] > 0.5


def test_evaluation_requires_the_ground_truth_validation_report(tmp_path: Path) -> None:
    from scripts.evaluate_anomaly import main as evaluate_main

    data_dir = _build_dataset_with_embeddings(tmp_path)
    (data_dir / "logs" / "demo_ground_truth_validation.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir),
                       "--backbone", "resnet18"])


def test_evaluation_records_eval_splits_for_stage_four_alignment(tmp_path: Path) -> None:
    """Stage 4 must restrict the IQA scores to these same rows before fusing,
    so which rows were used has to be recorded rather than inferred."""
    from scripts.evaluate_anomaly import main as evaluate_main

    data_dir = _build_dataset_with_embeddings(tmp_path)
    evaluate_main(["--dataset", "demo", "--data-dir", str(data_dir),
                   "--backbone", "resnet18"])
    summary = json.loads((data_dir / "logs" / "demo_anomaly_summary.json").read_text())
    assert summary["eval_splits"] == ["val", "test"]