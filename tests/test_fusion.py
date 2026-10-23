"""Tests for pipeline.fusion and scripts.evaluate_fusion.

The invariant that matters most here is the **positive control**: on synthetic
data where two scorers genuinely are complementary (each detecting a disjoint
half of the anomalies), all three fusion strategies must beat either scorer
alone. Without that check, Stage 4's headline negative result on real data
would be uninterpretable - "fusion didn't help" could just mean "the combiners
are broken". These tests establish that the combiners work, so the null result
is attributable to the data.

The second invariant is **no test-set leakage**: fusion weights are fitted on
val and reported on test. If that separation broke, any fusion benefit would be
unfalsifiable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import pipeline.fusion as fusion


# ---------------------------------------------------------------------------
# Positive control: fusion MUST help when complementarity genuinely exists
# ---------------------------------------------------------------------------

def _complementary_problem(
    n: int = 400, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two scorers, each detecting a disjoint half of the degraded rows.

    Neither alone can exceed roughly 0.75 AUROC; a working combiner should
    reach far higher. This is the shape of genuine complementarity - which is
    exactly what Stage 3 looked for on real data and did not find.
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    half = n // 2
    scorer_a = np.concatenate([
        rng.normal(0, 1, n),
        np.concatenate([rng.normal(3, 1, half), rng.normal(0, 1, n - half)]),
    ])
    scorer_b = np.concatenate([
        rng.normal(0, 1, n),
        np.concatenate([rng.normal(0, 1, half), rng.normal(3, 1, n - half)]),
    ])
    scores = np.column_stack([scorer_a, scorer_b])
    indices = rng.permutation(len(labels))
    split = len(indices) // 2
    return scores, labels, indices[:split], indices[split:]


@pytest.mark.parametrize("strategy_name", sorted(fusion.FUSION_STRATEGIES))
def test_fusion_beats_both_single_scorers_when_they_are_complementary(
    strategy_name,
) -> None:
    scores, labels, val, test = _complementary_problem()
    factory = fusion.FUSION_STRATEGIES[strategy_name]
    strategy = factory() if strategy_name != "logistic" else factory(random_state=0)
    if strategy_name == "logistic":
        strategy.fit(scores[val], labels[val], feature_names=["a", "b"])
    else:
        strategy.fit(scores[val], labels[val])

    fused_auroc = roc_auc_score(labels[test], strategy.score(scores[test]))
    a_auroc = roc_auc_score(labels[test], scores[test, 0])
    b_auroc = roc_auc_score(labels[test], scores[test, 1])
    assert fused_auroc > a_auroc, f"{strategy_name} failed to beat scorer a"
    assert fused_auroc > b_auroc, f"{strategy_name} failed to beat scorer b"


def test_fusion_suite_fits_every_registered_strategy() -> None:
    scores, labels, val, _ = _complementary_problem()
    suite = fusion.FusionSuite(random_state=0).fit(
        scores[val], labels[val], feature_names=["a", "b"]
    )
    assert set(suite.strategies) == set(fusion.FUSION_STRATEGIES)
    assert set(suite.score_all(scores)) == set(fusion.FUSION_STRATEGIES)


def test_all_fused_scores_are_finite_and_oriented_higher_is_worse() -> None:
    scores, labels, val, test = _complementary_problem()
    suite = fusion.FusionSuite(random_state=0).fit(scores[val], labels[val])
    for name, fused in suite.score_all(scores[test]).items():
        assert np.isfinite(fused).all(), name
        # Degraded rows must score higher on average than clean rows.
        assert fused[labels[test] == 1].mean() > fused[labels[test] == 0].mean(), name


# ---------------------------------------------------------------------------
# Fitting discipline
# ---------------------------------------------------------------------------

def test_strategies_refuse_to_score_before_being_fitted() -> None:
    for name, factory in fusion.FUSION_STRATEGIES.items():
        strategy = factory() if name != "logistic" else factory(random_state=0)
        with pytest.raises(RuntimeError):
            strategy.score(np.zeros((3, 2)))


def test_fusion_suite_refuses_to_score_before_being_fitted() -> None:
    with pytest.raises(RuntimeError):
        fusion.FusionSuite().score_all(np.zeros((3, 2)))


def test_zscore_standardises_against_clean_rows_only() -> None:
    """Standardising against all rows would let degraded images inflate the
    variance of whichever score detects them best, down-weighting the most
    informative signal."""
    rng = np.random.default_rng(1)
    clean = rng.normal(0, 1, (200, 2))
    degraded = rng.normal(0, 1, (200, 2)) + np.array([50.0, 0.0])  # huge shift on col 0
    scores = np.vstack([clean, degraded])
    labels = np.concatenate([np.zeros(200), np.ones(200)])

    strategy = fusion.ZScoreWeightedSum().fit(scores, labels)
    # The fitted scale should reflect clean rows (std ~1), not the combined
    # distribution (std ~25 on column 0).
    assert strategy._scaler.scale_[0] < 5.0
    assert strategy.describe()["n_clean_validation_rows"] == 200


def test_zscore_rejects_too_few_clean_rows() -> None:
    scores = np.random.default_rng(0).normal(0, 1, (5, 2))
    labels = np.ones(5)  # no clean rows at all
    with pytest.raises(ValueError):
        fusion.ZScoreWeightedSum().fit(scores, labels)


def test_rank_fusion_rejects_too_few_clean_rows() -> None:
    scores = np.random.default_rng(0).normal(0, 1, (5, 2))
    labels = np.ones(5)
    with pytest.raises(ValueError):
        fusion.RankFusion().fit(scores, labels)


def test_logistic_requires_both_classes() -> None:
    scores = np.random.default_rng(0).normal(0, 1, (20, 2))
    with pytest.raises(ValueError):
        fusion.LogisticFusion().fit(scores, np.zeros(20))


def test_logistic_reports_named_coefficients_and_convergence() -> None:
    """The learned weights are themselves a reportable result - they show which
    score families the combiner actually relies on."""
    scores, labels, val, _ = _complementary_problem()
    strategy = fusion.LogisticFusion(random_state=0).fit(
        scores[val], labels[val], feature_names=["alpha", "beta"]
    )
    description = strategy.describe()
    assert description["fitted"] is True
    assert set(description["coefficients"]) == {"alpha", "beta"}
    assert description["converged"] is True
    assert description["class_weight"] == "balanced"


def test_logistic_falls_back_to_positional_coefficients_without_names() -> None:
    scores, labels, val, _ = _complementary_problem()
    strategy = fusion.LogisticFusion(random_state=0).fit(scores[val], labels[val])
    assert isinstance(strategy.describe()["coefficients"], list)


def test_rank_fusion_is_invariant_to_monotone_rescaling() -> None:
    """Rank fusion's whole point: one score having a skewed distribution or
    wildly different units must not change its contribution."""
    scores, labels, val, test = _complementary_problem()
    baseline = fusion.RankFusion().fit(scores[val], labels[val])
    plain = baseline.score(scores[test])

    rescaled = scores.copy()
    rescaled[:, 0] = np.exp(rescaled[:, 0] / 2.0) * 1000.0  # monotone, extreme
    transformed = fusion.RankFusion().fit(rescaled[val], labels[val])
    assert np.allclose(transformed.score(rescaled[test]), plain, atol=1e-9)


def test_strategies_are_reproducible_for_a_fixed_seed() -> None:
    scores, labels, val, test = _complementary_problem()
    first = fusion.FusionSuite(random_state=7).fit(scores[val], labels[val])
    second = fusion.FusionSuite(random_state=7).fit(scores[val], labels[val])
    for name in first.strategies:
        assert np.allclose(
            first.score_all(scores[test])[name], second.score_all(scores[test])[name]
        )


def test_available_strategies_matches_the_registry() -> None:
    assert set(fusion.available_strategies()) == set(fusion.FUSION_STRATEGIES)


# ---------------------------------------------------------------------------
# End-to-end evaluate_fusion.py
# ---------------------------------------------------------------------------

def _build_fusion_dataset(tmp_path: Path) -> Path:
    """A dataset with IQA and anomaly score files covering val and test."""
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    dataset_dir.mkdir(parents=True)
    (data_dir / "logs").mkdir()

    rng = np.random.default_rng(3)
    rows = []
    for split in ("val", "test"):
        for index in range(60):
            rows.append({
                "image_id": f"{split}_clean_{index}", "defect_type": "clean",
                "severity_level": "0", "split": split,
            })
        for defect in ("gaussian_noise", "occlusion_patch"):
            for index in range(30):
                rows.append({
                    "image_id": f"{split}_{defect}_{index}", "defect_type": defect,
                    "severity_level": str(1 + index % 3), "split": split,
                })
    frame = pd.DataFrame(rows)
    degraded = (frame["defect_type"] != "clean").to_numpy()

    # noise_estimate detects gaussian_noise; mahalanobis detects occlusion.
    is_noise = (frame["defect_type"] == "gaussian_noise").to_numpy()
    is_occlusion = (frame["defect_type"] == "occlusion_patch").to_numpy()
    frame["noise_estimate"] = rng.normal(0, 1, len(frame)) + 3.0 * is_noise
    frame["laplacian_variance"] = rng.normal(0, 1, len(frame))
    frame["clip_iqa"] = rng.normal(0, 1, len(frame))
    frame["brightness_deviation"] = rng.normal(0, 1, len(frame)) + 0.2 * degraded

    iqa_columns = [
        "image_id", "defect_type", "severity_level", "split",
        "noise_estimate", "laplacian_variance", "clip_iqa", "brightness_deviation",
    ]
    iqa = frame[iqa_columns].copy()
    iqa["original_image_id"] = iqa["image_id"]
    iqa.to_csv(dataset_dir / "iqa_scores.csv", index=False)

    anomaly = frame[["image_id", "defect_type", "severity_level", "split"]].copy()
    anomaly["mahalanobis"] = rng.normal(0, 1, len(frame)) + 3.0 * is_occlusion
    anomaly["knn_k5"] = rng.normal(0, 1, len(frame)) + 1.0 * is_occlusion
    anomaly["isolation_forest"] = rng.normal(0, 1, len(frame))
    anomaly.to_csv(dataset_dir / "anomaly_scores_resnet18.csv", index=False)

    (data_dir / "logs" / "demo_ground_truth_validation.json").write_text(
        json.dumps({
            "binary_detection_only": [],
            "severity_correlation_eligible": ["gaussian_noise", "occlusion_patch"],
        })
    )
    return data_dir


def test_evaluate_fusion_end_to_end_writes_the_ablation_table(tmp_path: Path) -> None:
    from scripts.evaluate_fusion import main as fusion_main

    data_dir = _build_fusion_dataset(tmp_path)
    fusion_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--backbone", "resnet18", "--n-bootstrap", "100",
    ])

    table_path = data_dir / "logs" / "demo_fusion_ablation_resnet18.csv"
    assert table_path.is_file()
    table = pd.read_csv(table_path)
    assert set(table["defect"]) == {"gaussian_noise", "occlusion_patch"}
    for strategy in fusion.available_strategies():
        for suffix in ("auroc", "diff", "ci_low", "ci_high", "p", "significant"):
            assert f"{strategy}_{suffix}" in table.columns


def test_evaluate_fusion_records_the_val_fit_test_report_protocol(
    tmp_path: Path,
) -> None:
    """Fusion must be fitted on val and reported on test; if that separation
    broke, any observed fusion benefit would be unfalsifiable."""
    from scripts.evaluate_fusion import main as fusion_main

    data_dir = _build_fusion_dataset(tmp_path)
    fusion_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--backbone", "resnet18", "--n-bootstrap", "100",
    ])

    summary = json.loads(
        (data_dir / "logs" / "demo_fusion_summary_resnet18.json").read_text()
    )
    assert summary["n_val_rows_fitted"] == 120  # 60 clean + 2 x 30 degraded
    assert summary["n_test_rows_reported"] == 120
    assert "val" in summary["protocol"] and "test" in summary["protocol"]


def test_evaluate_fusion_recovers_complementarity_on_the_aggregate_task(
    tmp_path: Path,
) -> None:
    """The positive control for the whole stage.

    The fixture plants genuine complementarity: noise_estimate detects noise,
    mahalanobis detects occlusion, and neither detects the other. Fusion must
    therefore win on the AGGREGATE task (clean vs any degraded), because no
    single scorer covers both defects.

    Note it must be the aggregate task, not the per-defect one. Per defect, the
    specialist baseline is re-selected and already near-perfect on its own
    defect, so fusion has nothing to add there - which is exactly why the
    aggregate comparison had to be added to the evaluation.

    Without this control passing, Stage 4's negative result on real data would
    be uninterpretable: "fusion didn't help" could not be distinguished from
    "the combiners or the harness are broken".
    """
    from scripts.evaluate_fusion import main as fusion_main

    data_dir = _build_fusion_dataset(tmp_path)
    fusion_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--backbone", "resnet18", "--n-bootstrap", "300",
    ])
    summary = json.loads(
        (data_dir / "logs" / "demo_fusion_summary_resnet18.json").read_text()
    )
    aggregate = summary["aggregate_clean_vs_any_degraded"]
    best_single = aggregate["best_single_auroc"]
    # At least one strategy must significantly beat the best single scorer.
    beat_it = [
        name for name in fusion.available_strategies()
        if aggregate[f"{name}_significant"] and aggregate[f"{name}_diff"] > 0
    ]
    assert beat_it, (
        f"no strategy beat the best single scorer ({best_single:.3f}) despite "
        "complementarity being planted by construction"
    )


def test_evaluate_fusion_requires_matching_anomaly_scores(tmp_path: Path) -> None:
    from scripts.evaluate_fusion import main as fusion_main

    data_dir = _build_fusion_dataset(tmp_path)
    (data_dir / "demo" / "anomaly_scores_resnet18.csv").unlink()
    with pytest.raises(FileNotFoundError):
        fusion_main([
            "--dataset", "demo", "--data-dir", str(data_dir), "--backbone", "resnet18",
        ])