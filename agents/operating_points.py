"""Operating-point sweep: the false-reject / recall trade-off, made explicit.

**Why this exists.** "We cannot have false positives on medical images" is the
right instinct but the wrong question, because a false positive is not one thing.
The Inspector's three-way decision separates two errors with very different
costs:

  * A **false flag** sends a clean image to human review. A reviewer glances at
    it and passes it on; the cost is seconds of attention.
  * A **false reject** discards a clean image. That is the expensive error, and
    it is what a clinical reader means when they say they cannot tolerate false
    positives.

Reporting a single combined "false positive rate" conflates the two and overstates
the harm. Measured on ChestX-ray14, the combined rate was 5.9% while the false
*reject* rate was 1.2% - so 12 of the 15 apparent errors cost a reviewer a glance
rather than a discarded image.

The other half of the trade-off is unavoidable: driving either rate toward zero
costs recall, and a missed defect silently corrupts whatever trains on it. This
module measures both across a range of operating points so the trade-off can be
presented rather than argued, and a deployment can choose its own point.

**Method.** Thresholds are re-derived from the same clean validation rows at each
target rate and re-applied to the evaluation split, exactly as the Inspector does
in production. Nothing is refitted, so the sweep is a property of the scorer, not
of a new model per point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# Spanning three orders of magnitude, because the interesting region for a
# clinical deployment is at the strict end where a single-point measurement is
# least informative.
DEFAULT_REJECT_TARGETS: tuple[float, ...] = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10,
)


@dataclass(frozen=True)
class OperatingPoint:
    """Measured behaviour at one target false-reject rate."""

    target_reject_fpr: float
    target_flag_fpr: float
    reject_threshold: float
    flag_threshold: float
    n_clean: int
    n_degraded: int
    false_reject_rate: float
    false_flag_rate: float
    combined_false_positive_rate: float
    recall_any_action: float
    recall_rejected: float
    recall_by_severity: dict[str, float]
    n_clean_validation_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_reject_fpr": self.target_reject_fpr,
            "target_flag_fpr": self.target_flag_fpr,
            "reject_threshold": self.reject_threshold,
            "flag_threshold": self.flag_threshold,
            "n_clean": self.n_clean,
            "n_degraded": self.n_degraded,
            "false_reject_rate": self.false_reject_rate,
            "false_flag_rate": self.false_flag_rate,
            "combined_false_positive_rate": self.combined_false_positive_rate,
            "recall_any_action": self.recall_any_action,
            "recall_rejected": self.recall_rejected,
            "recall_by_severity": self.recall_by_severity,
            "n_clean_validation_rows": self.n_clean_validation_rows,
        }


def sweep_operating_points(
    validation_clean_scores: np.ndarray,
    evaluation_scores: np.ndarray,
    evaluation_is_degraded: np.ndarray,
    evaluation_severity: Sequence[str] | None = None,
    reject_targets: Iterable[float] = DEFAULT_REJECT_TARGETS,
    flag_multiplier: float = 5.0,
) -> list[OperatingPoint]:
    """Measure false-reject, false-flag and recall across target reject rates.

    All score arrays must be pre-oriented so that higher means more degraded.

    ``flag_multiplier`` keeps the flag threshold looser than the reject threshold
    by a fixed factor, mirroring the Inspector's default relationship (a 1%
    reject target alongside a 5% flag target). Holding the ratio fixed means the
    sweep varies one thing, so the resulting curve is interpretable.
    """
    validation_clean_scores = np.asarray(validation_clean_scores, dtype=float)
    validation_clean_scores = validation_clean_scores[
        np.isfinite(validation_clean_scores)
    ]
    if len(validation_clean_scores) == 0:
        raise ValueError("no finite clean validation scores to derive thresholds from")

    evaluation_scores = np.asarray(evaluation_scores, dtype=float)
    evaluation_is_degraded = np.asarray(evaluation_is_degraded, dtype=bool)
    finite = np.isfinite(evaluation_scores)
    evaluation_scores = evaluation_scores[finite]
    evaluation_is_degraded = evaluation_is_degraded[finite]
    severity = (
        np.asarray(evaluation_severity, dtype=object)[finite]
        if evaluation_severity is not None else None
    )

    clean_mask = ~evaluation_is_degraded
    degraded_mask = evaluation_is_degraded
    n_clean = int(clean_mask.sum())
    n_degraded = int(degraded_mask.sum())
    if n_clean == 0 or n_degraded == 0:
        raise ValueError(
            f"the evaluation split needs both classes; got {n_clean} clean and "
            f"{n_degraded} degraded"
        )

    points: list[OperatingPoint] = []
    for target in sorted(reject_targets):
        flag_target = min(0.999, target * flag_multiplier)
        reject_threshold = float(
            np.percentile(validation_clean_scores, 100.0 * (1.0 - target))
        )
        flag_threshold = float(
            np.percentile(validation_clean_scores, 100.0 * (1.0 - flag_target))
        )

        rejected = evaluation_scores > reject_threshold
        # An image past the flag threshold but not the reject threshold is
        # flagged; past both, it is rejected. This mirrors the Inspector, where
        # rejection is a strictly stronger verdict than flagging.
        flagged = (evaluation_scores > flag_threshold) & ~rejected
        actioned = rejected | flagged

        by_severity: dict[str, float] = {}
        if severity is not None:
            for level in sorted({str(value) for value in severity[degraded_mask]}):
                level_mask = degraded_mask & (severity == level)
                if level_mask.sum():
                    by_severity[level] = float(actioned[level_mask].mean())

        points.append(OperatingPoint(
            target_reject_fpr=float(target),
            target_flag_fpr=float(flag_target),
            reject_threshold=reject_threshold,
            flag_threshold=flag_threshold,
            n_clean=n_clean,
            n_degraded=n_degraded,
            false_reject_rate=float(rejected[clean_mask].mean()),
            false_flag_rate=float(flagged[clean_mask].mean()),
            combined_false_positive_rate=float(actioned[clean_mask].mean()),
            recall_any_action=float(actioned[degraded_mask].mean()),
            recall_rejected=float(rejected[degraded_mask].mean()),
            recall_by_severity=by_severity,
            n_clean_validation_rows=int(len(validation_clean_scores)),
        ))
    return points


def points_to_frame(points: Iterable[OperatingPoint]) -> pd.DataFrame:
    """Tabulate a sweep, with severity columns expanded."""
    rows = []
    for point in points:
        row = point.to_dict()
        for level, recall in row.pop("recall_by_severity").items():
            row[f"recall_severity_{level}"] = recall
        rows.append(row)
    return pd.DataFrame(rows)


def smallest_sufficient_point(
    points: Sequence[OperatingPoint], max_false_reject_rate: float
) -> OperatingPoint | None:
    """The highest-recall point whose false-reject rate stays within a budget.

    This is the question a deployment actually asks - "given that I can tolerate
    at most this much data loss, how much detection do I get?" - rather than
    "what threshold should I use".
    """
    eligible = [
        point for point in points
        if point.false_reject_rate <= max_false_reject_rate
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda point: point.recall_any_action)


def threshold_stability_warning(points: Sequence[OperatingPoint]) -> str | None:
    """Whether the strictest targets rest on too few validation images.

    A 0.1% threshold derived from 77 clean images is not an estimate of a 0.1%
    quantile - it is the single most extreme observation. Reporting such a point
    without qualification overstates how well the rate is known.
    """
    if not points:
        return None
    n_validation = points[0].n_clean_validation_rows
    unreliable = [
        point.target_reject_fpr for point in points
        if point.target_reject_fpr * n_validation < 5.0
    ]
    if not unreliable:
        return None
    return (
        f"Targets {unreliable} correspond to fewer than five clean validation "
        f"images each out of {n_validation}, so their thresholds are set by a "
        "handful of extreme observations rather than an estimated quantile. "
        "Treat those rows as indicative only."
    )