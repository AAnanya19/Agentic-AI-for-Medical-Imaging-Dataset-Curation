"""Tests for the operating-point sweep.

The sweep exists to answer a specific question honestly: **how much detection do
you get for a given amount of discarded clean data?** Four invariants:

1. **False rejects and false flags must be reported separately.** A false flag
   costs a reviewer seconds; a false reject destroys an image. Measured on
   ChestX-ray14, the combined rate was 5.9% while the false *reject* rate was
   1.2% - so a combined figure overstates the harm by roughly fivefold.

2. **The curve must be monotonic.** A looser target cannot produce fewer false
   rejects or less recall. If it does, the thresholds are not being derived
   consistently and the curve means nothing.

3. **A budget query must return the best achievable recall, not just any point
   within budget.** "Given I can lose 1% of clean data, what do I catch?" is the
   deployment question.

4. **Targets too strict for the validation sample must be flagged.** A 0.1%
   threshold from 77 clean images is the single most extreme observation, not an
   estimated quantile, and presenting it unqualified would overstate precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.operating_points import (
    DEFAULT_REJECT_TARGETS, OperatingPoint, points_to_frame,
    smallest_sufficient_point, sweep_operating_points,
    threshold_stability_warning,
)


def separable_problem(
    n_validation: int = 400, n_clean: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Clean scores near 0.2, degraded spread by severity so mild cases overlap.

    Mirrors the real scorer's behaviour, where severity-1 defects sit close to the
    clean distribution and severity-3 defects are far from it.
    """
    rng = np.random.default_rng(seed)
    validation_clean = rng.normal(0.2, 0.08, n_validation)
    clean = rng.normal(0.2, 0.08, n_clean)
    mild = rng.normal(0.35, 0.10, 100)
    moderate = rng.normal(0.60, 0.10, 100)
    severe = rng.normal(0.85, 0.08, 100)
    scores = np.concatenate([clean, mild, moderate, severe])
    is_degraded = np.concatenate([
        np.zeros(n_clean, dtype=bool), np.ones(300, dtype=bool)
    ])
    severity = ["0"] * n_clean + ["1"] * 100 + ["2"] * 100 + ["3"] * 100
    return validation_clean, scores, is_degraded, severity


# ---------------------------------------------------------------------------
# Shape of the curve
# ---------------------------------------------------------------------------

def test_false_reject_rate_rises_monotonically_with_the_target() -> None:
    validation, scores, degraded, severity = separable_problem()
    points = sweep_operating_points(validation, scores, degraded, severity)
    rates = [point.false_reject_rate for point in points]
    assert rates == sorted(rates)


def test_recall_rises_monotonically_with_the_target() -> None:
    validation, scores, degraded, severity = separable_problem()
    points = sweep_operating_points(validation, scores, degraded, severity)
    recalls = [point.recall_any_action for point in points]
    assert recalls == sorted(recalls)


def test_the_realised_false_reject_rate_tracks_its_target() -> None:
    """Not exactly - the evaluation split is a different sample - but closely
    enough that the target is meaningful."""
    validation, scores, degraded, severity = separable_problem(n_validation=4000)
    points = sweep_operating_points(
        validation, scores, degraded, severity, reject_targets=(0.02, 0.05, 0.10)
    )
    for point in points:
        assert point.false_reject_rate == pytest.approx(
            point.target_reject_fpr, abs=0.03
        )


def test_false_flags_and_false_rejects_are_reported_separately() -> None:
    """The core reporting requirement: their costs differ by orders of
    magnitude, so a combined figure alone overstates the harm."""
    validation, scores, degraded, severity = separable_problem()
    points = sweep_operating_points(validation, scores, degraded, severity)
    point = points[-1]        # the loosest target, where both are non-zero
    assert point.false_flag_rate > 0
    assert point.false_reject_rate > 0
    assert point.combined_false_positive_rate == pytest.approx(
        point.false_reject_rate + point.false_flag_rate, abs=1e-9
    )


def test_a_rejected_image_is_not_double_counted_as_flagged() -> None:
    """Rejection is a strictly stronger verdict, so the two sets are disjoint."""
    validation, scores, degraded, severity = separable_problem()
    for point in sweep_operating_points(validation, scores, degraded, severity):
        assert point.combined_false_positive_rate <= 1.0
        assert point.recall_rejected <= point.recall_any_action


def test_severe_defects_are_caught_even_at_the_strictest_target() -> None:
    """The substantive answer to "we cannot tolerate false positives": at a zero
    false-reject operating point, severity-3 defects are still detected."""
    validation, scores, degraded, severity = separable_problem()
    points = sweep_operating_points(validation, scores, degraded, severity)
    strictest = points[0]
    assert strictest.recall_by_severity["3"] > 0.9
    # And mild defects are the ones sacrificed, which is the trade-off to report.
    assert strictest.recall_by_severity["1"] < strictest.recall_by_severity["3"]


def test_recall_by_severity_is_ordered_by_severity() -> None:
    validation, scores, degraded, severity = separable_problem()
    point = sweep_operating_points(validation, scores, degraded, severity)[3]
    by_severity = point.recall_by_severity
    assert by_severity["1"] <= by_severity["2"] <= by_severity["3"]


# ---------------------------------------------------------------------------
# Budget queries
# ---------------------------------------------------------------------------

def test_a_budget_query_returns_the_highest_recall_within_budget() -> None:
    validation, scores, degraded, severity = separable_problem()
    points = sweep_operating_points(validation, scores, degraded, severity)
    best = smallest_sufficient_point(points, 0.05)
    assert best is not None
    assert best.false_reject_rate <= 0.05
    # No eligible point may beat the one returned.
    eligible = [p for p in points if p.false_reject_rate <= 0.05]
    assert best.recall_any_action == max(p.recall_any_action for p in eligible)


def test_an_unachievable_budget_returns_nothing_rather_than_a_violation() -> None:
    validation, scores, degraded, severity = separable_problem()
    points = sweep_operating_points(
        validation, scores, degraded, severity, reject_targets=(0.20, 0.40)
    )
    # Both points exceed a 1e-9 budget, so none is eligible.
    assert smallest_sufficient_point(points, 1e-9) is None


def test_a_tighter_budget_never_yields_more_recall() -> None:
    validation, scores, degraded, severity = separable_problem()
    points = sweep_operating_points(validation, scores, degraded, severity)
    tight = smallest_sufficient_point(points, 0.005)
    loose = smallest_sufficient_point(points, 0.10)
    assert tight.recall_any_action <= loose.recall_any_action


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_strict_targets_are_flagged_when_validation_is_small() -> None:
    """ISIC has 77 clean validation rows, so a 0.1% target is one image."""
    validation, scores, degraded, severity = separable_problem(n_validation=77)
    points = sweep_operating_points(validation, scores, degraded, severity)
    warning = threshold_stability_warning(points)
    assert warning is not None
    assert "77" in warning


def test_no_warning_when_validation_is_large_enough() -> None:
    validation, scores, degraded, severity = separable_problem(n_validation=20000)
    points = sweep_operating_points(
        validation, scores, degraded, severity, reject_targets=(0.05, 0.10)
    )
    assert threshold_stability_warning(points) is None


def test_the_validation_sample_size_is_recorded_on_every_point() -> None:
    validation, scores, degraded, severity = separable_problem(n_validation=250)
    for point in sweep_operating_points(validation, scores, degraded, severity):
        assert point.n_clean_validation_rows == 250


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_non_finite_scores_are_excluded_rather_than_crashing() -> None:
    validation, scores, degraded, severity = separable_problem()
    scores = scores.copy()
    scores[0] = np.nan
    scores[1] = np.inf
    points = sweep_operating_points(validation, scores, degraded, severity)
    assert points[0].n_clean == 298      # two clean rows dropped


def test_a_single_class_evaluation_split_is_rejected() -> None:
    validation, scores, _, _ = separable_problem()
    with pytest.raises(ValueError, match="both classes"):
        sweep_operating_points(
            validation, scores[:300], np.zeros(300, dtype=bool)
        )


def test_empty_validation_scores_are_rejected() -> None:
    _, scores, degraded, _ = separable_problem()
    with pytest.raises(ValueError, match="no finite clean validation scores"):
        sweep_operating_points(np.array([np.nan, np.inf]), scores, degraded)


def test_severity_is_optional() -> None:
    validation, scores, degraded, _ = separable_problem()
    points = sweep_operating_points(validation, scores, degraded)
    assert points[0].recall_by_severity == {}
    assert points[0].recall_any_action > 0


def test_the_flag_multiplier_widens_the_review_band() -> None:
    validation, scores, degraded, severity = separable_problem()
    narrow = sweep_operating_points(
        validation, scores, degraded, severity,
        reject_targets=(0.01,), flag_multiplier=2.0,
    )[0]
    wide = sweep_operating_points(
        validation, scores, degraded, severity,
        reject_targets=(0.01,), flag_multiplier=10.0,
    )[0]
    assert wide.false_flag_rate > narrow.false_flag_rate
    # Rejection is unaffected: only the flag threshold moved.
    assert wide.false_reject_rate == narrow.false_reject_rate


def test_default_targets_span_three_orders_of_magnitude() -> None:
    assert min(DEFAULT_REJECT_TARGETS) <= 0.001
    assert max(DEFAULT_REJECT_TARGETS) >= 0.10


# ---------------------------------------------------------------------------
# Tabulation
# ---------------------------------------------------------------------------

def test_the_frame_expands_severity_into_columns() -> None:
    validation, scores, degraded, severity = separable_problem()
    frame = points_to_frame(
        sweep_operating_points(validation, scores, degraded, severity)
    )
    assert "recall_severity_1" in frame.columns
    assert "recall_severity_3" in frame.columns
    assert "recall_by_severity" not in frame.columns


def test_the_frame_has_one_row_per_target() -> None:
    validation, scores, degraded, severity = separable_problem()
    targets = (0.005, 0.01, 0.05)
    frame = points_to_frame(
        sweep_operating_points(
            validation, scores, degraded, severity, reject_targets=targets
        )
    )
    assert len(frame) == len(targets)


def test_a_point_serialises_completely() -> None:
    validation, scores, degraded, severity = separable_problem()
    payload = sweep_operating_points(validation, scores, degraded, severity)[0].to_dict()
    for key in (
        "false_reject_rate", "false_flag_rate", "combined_false_positive_rate",
        "recall_any_action", "recall_rejected", "n_clean_validation_rows",
    ):
        assert key in payload


def test_reproducible_for_identical_input() -> None:
    validation, scores, degraded, severity = separable_problem()
    first = sweep_operating_points(validation, scores, degraded, severity)
    second = sweep_operating_points(validation, scores, degraded, severity)
    assert [p.to_dict() for p in first] == [p.to_dict() for p in second]