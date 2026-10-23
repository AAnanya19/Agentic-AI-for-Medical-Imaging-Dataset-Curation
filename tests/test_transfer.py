"""Tests for Stage 4b (feature-set ablation) and Stage 5 (cross-modality transfer).

Two invariants:

1. **The feature-set ablation must isolate the embeddings' marginal
   contribution.** Stage 4 showed that fusing everything beats the best single
   scorer, but could not say whether the gain came from combining IQA metrics
   with one another or from adding embeddings. If `all` and `iqa_only` fit the
   same features, or the comparison is not paired on identical rows, the
   ablation answers nothing.

2. **Transfer must never silently zero-fill a missing feature.** A model fitted
   on one modality can only be applied to another if the feature vector means
   the same thing in both. Filling an absent column with zeros would feed the
   model a value it never saw while fitting, producing a plausible-looking but
   meaningless score - so unshared columns are dropped instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_fusion import FEATURE_SETS, oriented_matrix
from scripts.evaluate_transfer import (
    realised_rates,
    shared_feature_columns,
    threshold_at_target_fpr,
)


# ---------------------------------------------------------------------------
# Stage 4b: feature-set selection
# ---------------------------------------------------------------------------

def _score_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)],
        "defect_type": ["clean"] * (n // 2) + ["gaussian_noise"] * (n - n // 2),
        "severity_level": ["0"] * (n // 2) + ["2"] * (n - n // 2),
        "split": ["val"] * (n // 2) + ["test"] * (n - n // 2),
        "noise_estimate": rng.normal(0, 1, n),
        "laplacian_variance": rng.normal(0, 1, n),
        "clip_iqa": rng.normal(0, 1, n),
        "brightness_deviation": rng.normal(0, 1, n),
        "mahalanobis": rng.normal(0, 1, n),
        "knn_k5": rng.normal(0, 1, n),
    })


def test_every_feature_set_is_registered_with_a_family_pair() -> None:
    assert set(FEATURE_SETS) == {"iqa_only", "embeddings_only", "all"}
    for name, (use_iqa, use_embeddings) in FEATURE_SETS.items():
        assert isinstance(use_iqa, bool) and isinstance(use_embeddings, bool), name
        assert use_iqa or use_embeddings, f"{name} selects no features at all"


def test_iqa_only_excludes_every_detector_column() -> None:
    frame = _score_frame()
    iqa = ["noise_estimate", "laplacian_variance", "clip_iqa", "brightness_deviation"]
    detectors = ["mahalanobis", "knn_k5"]
    _, names = oriented_matrix(frame, iqa, detectors, "iqa_only")
    assert names and all(name.startswith("iqa:") for name in names)
    assert not any(name.startswith("emb:") for name in names)


def test_embeddings_only_excludes_every_iqa_column() -> None:
    frame = _score_frame()
    iqa = ["noise_estimate", "laplacian_variance"]
    detectors = ["mahalanobis", "knn_k5"]
    _, names = oriented_matrix(frame, iqa, detectors, "embeddings_only")
    assert names and all(name.startswith("emb:") for name in names)


def test_all_is_the_union_of_the_two_single_family_sets() -> None:
    """The ablation is only meaningful if 'all' genuinely contains both, so a
    regression that quietly dropped one family would be caught here."""
    frame = _score_frame()
    iqa = ["noise_estimate", "laplacian_variance", "clip_iqa"]
    detectors = ["mahalanobis", "knn_k5"]
    _, only_iqa = oriented_matrix(frame, iqa, detectors, "iqa_only")
    _, only_emb = oriented_matrix(frame, iqa, detectors, "embeddings_only")
    _, everything = oriented_matrix(frame, iqa, detectors, "all")
    assert set(everything) == set(only_iqa) | set(only_emb)
    assert len(everything) == len(only_iqa) + len(only_emb)


def test_oriented_matrix_rejects_an_unknown_feature_set() -> None:
    frame = _score_frame()
    with pytest.raises(ValueError):
        oriented_matrix(frame, ["noise_estimate"], ["mahalanobis"], "not_a_set")


def test_oriented_matrix_raises_when_a_feature_set_yields_no_columns() -> None:
    """embeddings_only with no detectors available must fail loudly rather than
    return an empty matrix that would produce a meaningless fit."""
    frame = _score_frame()
    with pytest.raises(RuntimeError):
        oriented_matrix(frame, ["noise_estimate"], [], "embeddings_only")


# ---------------------------------------------------------------------------
# Stage 5: shared feature layout
# ---------------------------------------------------------------------------

def test_shared_columns_are_the_intersection_across_datasets() -> None:
    frame_a = _score_frame()
    frame_b = _score_frame().drop(columns=["clip_iqa", "knn_k5"])
    shared_iqa, shared_detectors = shared_feature_columns({"a": frame_a, "b": frame_b})
    assert "clip_iqa" not in shared_iqa
    assert "knn_k5" not in shared_detectors
    assert "noise_estimate" in shared_iqa
    assert "mahalanobis" in shared_detectors


def test_shared_columns_never_zero_fill_a_missing_feature() -> None:
    """The dangerous alternative to dropping is zero-filling, which would feed
    the model a value it never saw during fitting."""
    frame_a = _score_frame()
    frame_b = _score_frame().drop(columns=["mahalanobis"])
    _, shared_detectors = shared_feature_columns({"a": frame_a, "b": frame_b})
    assert "mahalanobis" not in shared_detectors
    # And the surviving layout must be usable on both frames without KeyError.
    for frame in (frame_a, frame_b):
        matrix, _ = oriented_matrix(frame, [], shared_detectors, "embeddings_only")
        assert matrix.shape[0] == len(frame)


def test_shared_columns_excludes_non_directional_metrics() -> None:
    """shannon_entropy has no quality polarity, so it cannot be oriented and
    must not enter a transferred feature vector."""
    frame = _score_frame()
    frame["shannon_entropy"] = np.random.default_rng(0).normal(0, 1, len(frame))
    shared_iqa, _ = shared_feature_columns({"a": frame, "b": frame})
    assert "shannon_entropy" not in shared_iqa


# ---------------------------------------------------------------------------
# Stage 5: threshold transfer arithmetic
# ---------------------------------------------------------------------------

def test_threshold_at_target_fpr_flags_the_requested_fraction() -> None:
    clean = np.linspace(0.0, 1.0, 1000)
    threshold = threshold_at_target_fpr(clean, 0.05)
    fpr, _ = realised_rates(clean, np.ones(10), threshold)
    assert fpr == pytest.approx(0.05, abs=0.01)


def test_realised_rates_reports_both_fpr_and_recall() -> None:
    clean = np.array([0.0, 0.1, 0.2, 0.3])
    degraded = np.array([0.8, 0.9, 0.05])
    fpr, recall = realised_rates(clean, degraded, threshold=0.5)
    assert fpr == 0.0
    assert recall == pytest.approx(2 / 3)


def test_a_threshold_from_a_shifted_distribution_inflates_fpr() -> None:
    """The core Stage 5 concern: if the two modalities' score distributions do
    not overlap, a transferred threshold cannot hold its target FPR."""
    rng = np.random.default_rng(0)
    source_clean = rng.normal(0.0, 1.0, 2000)
    target_clean = rng.normal(2.0, 1.0, 2000)  # shifted well upward
    threshold = threshold_at_target_fpr(source_clean, 0.05)
    source_fpr, _ = realised_rates(source_clean, np.ones(5), threshold)
    target_fpr, _ = realised_rates(target_clean, np.ones(5), threshold)
    assert source_fpr == pytest.approx(0.05, abs=0.01)
    assert target_fpr > 0.5  # badly inflated, as intended


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def _build_two_datasets(tmp_path: Path) -> Path:
    """Two datasets sharing a schema but with deliberately different scales, so
    a transfer penalty genuinely exists and must be detected."""
    data_dir = tmp_path / "data"
    (data_dir / "logs").mkdir(parents=True)
    rng = np.random.default_rng(5)

    for name, shift, scale in (("alpha", 0.0, 1.0), ("beta", 2.5, 2.0)):
        dataset_dir = data_dir / name
        dataset_dir.mkdir()
        rows = []
        for split in ("val", "test"):
            for index in range(70):
                rows.append({
                    "image_id": f"{name}_{split}_clean_{index}",
                    "defect_type": "clean", "severity_level": "0", "split": split,
                })
            for index in range(50):
                rows.append({
                    "image_id": f"{name}_{split}_noise_{index}",
                    "defect_type": "gaussian_noise",
                    "severity_level": str(1 + index % 3), "split": split,
                })
        frame = pd.DataFrame(rows)
        degraded = (frame["defect_type"] != "clean").to_numpy()
        frame["original_image_id"] = frame["image_id"]
        for column in ("noise_estimate", "laplacian_variance", "clip_iqa",
                       "brightness_deviation"):
            frame[column] = (
                rng.normal(shift, scale, len(frame)) + 2.0 * degraded * scale
            )
        frame.to_csv(dataset_dir / "iqa_scores.csv", index=False)

        anomaly = frame[["image_id", "defect_type", "severity_level", "split"]].copy()
        for detector in ("mahalanobis", "knn_k1", "knn_k5", "knn_k10", "knn_k20",
                         "isolation_forest"):
            anomaly[detector] = rng.normal(shift, scale, len(frame))
        anomaly.to_csv(dataset_dir / "anomaly_scores_resnet18.csv", index=False)
    return data_dir


def test_transfer_end_to_end_writes_all_three_experiments(tmp_path: Path) -> None:
    from scripts.evaluate_transfer import main as transfer_main

    data_dir = _build_two_datasets(tmp_path)
    transfer_main([
        "--datasets", "alpha,beta", "--data-dir", str(data_dir),
        "--backbone", "resnet18", "--n-bootstrap", "100",
    ])

    path = data_dir / "logs" / "transfer_alpha_beta_resnet18.json"
    assert path.is_file()
    payload = json.loads(path.read_text())
    results = payload["results"]

    # Experiment 1: all four (source, target) cells present.
    for source in ("alpha", "beta"):
        for target in ("alpha", "beta"):
            assert f"fit_{source}__test_{target}" in results
    # Experiment 2 and 3.
    assert "threshold_transfer" in results
    assert "clean_score_distributions" in results
    assert set(payload["shared_detectors"]) >= {"mahalanobis", "knn_k5"}


def test_transfer_detects_the_penalty_when_distributions_differ(
    tmp_path: Path,
) -> None:
    """Positive control: the fixture shifts and rescales beta's scores, so a
    threshold calibrated on one dataset must not hold on the other."""
    from scripts.evaluate_transfer import main as transfer_main

    data_dir = _build_two_datasets(tmp_path)
    transfer_main([
        "--datasets", "alpha,beta", "--data-dir", str(data_dir),
        "--backbone", "resnet18", "--n-bootstrap", "200", "--target-fpr", "0.05",
    ])
    payload = json.loads(
        (data_dir / "logs" / "transfer_alpha_beta_resnet18.json").read_text()
    )
    thresholds = payload["results"]["threshold_transfer"]
    # At least one direction must show the target FPR being badly missed.
    assert any(entry["recalibration_required"] for entry in thresholds.values())


def test_transfer_requires_exactly_two_datasets(tmp_path: Path) -> None:
    from scripts.evaluate_transfer import main as transfer_main

    data_dir = _build_two_datasets(tmp_path)
    with pytest.raises(ValueError):
        transfer_main(["--datasets", "alpha", "--data-dir", str(data_dir)])