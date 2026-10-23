"""Tests for pipeline.significance.

The two invariants these tests exist to protect:

1. **Paired bootstrap correctness.** Independent bootstraps on each scorer's
   AUROC produce a wider (more conservative) CI on the difference than a
   paired bootstrap that resamples the same rows for both scorers. If the
   paired CI here is NOT tighter than the independent one on correlated data,
   the pairing is not actually happening.

2. **Reproducibility given a seed.** Every run with the same seed must produce
   identical numbers - CI endpoints, p-values, everything - because Stage 4's
   published table and the accompanying dissertation text will cite specific
   values, and those values must not drift between runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.significance import (
    BootstrapCI,
    PairedBootstrapResult,
    bootstrap_auroc_ci,
    orient_scores,
    paired_bootstrap_diff,
)


def _synthetic_scores(
    n_clean: int = 200, n_degraded: int = 200, offset: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A clean vs degraded classification problem with two correlated scorers.

    Both scorers are noisy views of the same underlying signal - which is what
    two IQA metrics (or two anomaly detectors) applied to the same images
    actually look like. Independent noise per scorer would understate the
    pairing benefit.
    """
    rng = np.random.default_rng(seed)
    signal = np.concatenate([rng.normal(0, 1, n_clean), rng.normal(offset, 1, n_degraded)])
    scorer_a = signal + rng.normal(0, 0.3, len(signal))
    scorer_b = signal + rng.normal(0, 0.3, len(signal))
    labels = np.concatenate([np.zeros(n_clean), np.ones(n_degraded)])
    return labels, scorer_a, scorer_b


# ---------------------------------------------------------------------------
# bootstrap_auroc_ci
# ---------------------------------------------------------------------------

def test_bootstrap_auroc_ci_returns_finite_ci_that_brackets_the_point() -> None:
    labels, scores, _ = _synthetic_scores(offset=1.5)
    ci = bootstrap_auroc_ci(labels, scores, n_bootstrap=500, seed=1)
    assert isinstance(ci, BootstrapCI)
    assert np.isfinite(ci.point)
    assert np.isfinite(ci.ci_low)
    assert np.isfinite(ci.ci_high)
    assert ci.ci_low <= ci.point <= ci.ci_high
    assert ci.n_bootstrap > 0


def test_bootstrap_auroc_ci_is_reproducible_for_a_fixed_seed() -> None:
    labels, scores, _ = _synthetic_scores(offset=1.0)
    first = bootstrap_auroc_ci(labels, scores, n_bootstrap=200, seed=42)
    second = bootstrap_auroc_ci(labels, scores, n_bootstrap=200, seed=42)
    assert first == second


def test_bootstrap_auroc_ci_returns_nan_when_only_one_class_is_present() -> None:
    labels = np.zeros(50)
    scores = np.random.default_rng(0).normal(0, 1, 50)
    ci = bootstrap_auroc_ci(labels, scores, n_bootstrap=50)
    assert np.isnan(ci.point)
    assert ci.n_bootstrap == 0


def test_bootstrap_auroc_ci_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        bootstrap_auroc_ci(np.array([0, 1, 0]), np.array([0.1, 0.9]))


# ---------------------------------------------------------------------------
# paired_bootstrap_diff
# ---------------------------------------------------------------------------

def test_paired_bootstrap_reproduces_the_real_cxr_occlusion_finding() -> None:
    """Regression test against actual data: the ChestX-ray14 occlusion result
    computed before this module existed was diff = +0.032, CI [-0.032, +0.099],
    p = 0.170. This module must reproduce those numbers, or the values quoted
    in the dissertation and the code would drift.

    Skips when the real score files are absent (e.g. a fresh clone before the
    pipeline has been run), so the suite stays green without them.
    """
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[1]
    iqa_path = repo_root / "data" / "chestxray14" / "iqa_scores.csv"
    emb_path = repo_root / "data" / "chestxray14" / "anomaly_scores_resnet18.csv"
    if not (iqa_path.exists() and emb_path.exists()):
        pytest.skip(
            "real ChestX-ray14 score files not present; run compute_iqa.py and "
            "evaluate_anomaly.py to enable this regression test"
        )

    iqa = pd.read_csv(iqa_path)
    emb = pd.read_csv(emb_path)
    merged = iqa.merge(emb[["image_id", "mahalanobis"]], on="image_id", how="inner")
    clean = merged[merged.defect_type == "clean"]
    occlusion = merged[merged.defect_type == "occlusion_patch"]
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(occlusion))])
    emb_scores = np.concatenate([clean.mahalanobis, occlusion.mahalanobis])
    iqa_scores = orient_scores(
        np.concatenate([clean.clip_iqa, occlusion.clip_iqa]),
        higher_is_better=True,
    )

    result = paired_bootstrap_diff(labels, emb_scores, iqa_scores, n_bootstrap=2000, seed=42)
    assert result.auroc_a == pytest.approx(0.811, abs=0.005)
    assert result.auroc_b == pytest.approx(0.779, abs=0.005)
    assert result.diff == pytest.approx(0.032, abs=0.010)
    assert result.ci_low == pytest.approx(-0.032, abs=0.010)
    assert result.ci_high == pytest.approx(0.099, abs=0.010)
    assert result.p_diff_le_zero == pytest.approx(0.170, abs=0.020)
    assert not result.significant_at_95


def test_paired_bootstrap_ci_is_tighter_than_independent_bootstraps() -> None:
    """The whole reason for pairing: when two scorers see the same images and
    their errors correlate, the paired CI on the difference must be narrower
    than what independent bootstraps would give. If it is not, the pairing is
    not actually happening.
    """
    labels, scorer_a, scorer_b = _synthetic_scores(offset=1.5)
    paired = paired_bootstrap_diff(labels, scorer_a, scorer_b, n_bootstrap=500, seed=1)
    ci_a = bootstrap_auroc_ci(labels, scorer_a, n_bootstrap=500, seed=1)
    ci_b = bootstrap_auroc_ci(labels, scorer_b, n_bootstrap=500, seed=1)
    paired_width = paired.ci_high - paired.ci_low
    independent_width = (ci_a.ci_high - ci_a.ci_low) + (ci_b.ci_high - ci_b.ci_low)
    assert paired_width < independent_width


def test_paired_bootstrap_finds_significance_when_scorer_a_dominates() -> None:
    """If scorer_a is genuinely much better than scorer_b, the CI on the
    difference should exclude zero and p(diff <= 0) should be near zero."""
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    strong = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
    weak = rng.normal(0, 1, 400)  # pure noise
    result = paired_bootstrap_diff(labels, strong, weak, n_bootstrap=500, seed=0)
    assert result.diff > 0.2
    assert result.significant_at_95
    assert result.p_diff_le_zero < 0.05


def test_paired_bootstrap_reports_no_significance_when_scorers_are_equal() -> None:
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    signal = np.concatenate([rng.normal(0, 1, 200), rng.normal(1, 1, 200)])
    # Two nearly-identical noisy views of the same signal
    scorer_a = signal + rng.normal(0, 0.1, 400)
    scorer_b = signal + rng.normal(0, 0.1, 400)
    result = paired_bootstrap_diff(labels, scorer_a, scorer_b, n_bootstrap=500, seed=0)
    assert abs(result.diff) < 0.05
    assert not result.significant_at_95


def test_paired_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    labels, scorer_a, scorer_b = _synthetic_scores()
    first = paired_bootstrap_diff(labels, scorer_a, scorer_b, n_bootstrap=200, seed=7)
    second = paired_bootstrap_diff(labels, scorer_a, scorer_b, n_bootstrap=200, seed=7)
    assert first == second


def test_paired_bootstrap_significance_flag_flips_only_when_ci_excludes_zero() -> None:
    labels, scorer_a, scorer_b = _synthetic_scores(offset=0.05)  # a near-tie
    result = paired_bootstrap_diff(labels, scorer_a, scorer_b, n_bootstrap=500, seed=0)
    assert result.significant_at_95 == (result.ci_low > 0 or result.ci_high < 0)


def test_paired_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_diff(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.2])
        )


# ---------------------------------------------------------------------------
# orient_scores
# ---------------------------------------------------------------------------

def test_orient_scores_negates_iff_higher_is_better() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(orient_scores(values, higher_is_better=True), -values)
    assert np.allclose(orient_scores(values, higher_is_better=False), values)


def test_orient_scores_accepts_lists_not_only_arrays() -> None:
    """The function is called with pandas .to_numpy() output which is already
    an ndarray, but it should also cope with a plain list without error."""
    assert np.allclose(orient_scores([1.0, 2.0], higher_is_better=True), [-1.0, -2.0])