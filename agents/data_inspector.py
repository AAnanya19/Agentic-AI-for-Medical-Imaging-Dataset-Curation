"""The Data Inspector Agent: the primary quality-control agent.

**What distinguishes this from the scoring scripts in Stages 2-6.** Those
produce scores; this produces *decisions*, with the evidence attached. Three
properties make it an agent rather than a metric pipeline:

  1. **Thresholds are derived, not chosen.** Each check's threshold is the
     operating point that yields a stated false-positive rate on clean
     *validation* images. Nothing is hand-tuned, and the derivation is recorded.
  2. **Decisions are three-way and attributed.** An image is accepted, flagged
     for human review, or rejected; and the output names which checks fired,
     with their scores and their thresholds side by side. A reviewer can see why
     without re-running anything.
  3. **Output is structured for a downstream consumer.** The Summary Agent reads
     this directly, so the shape is a contract (see agents.base) rather than a
     convenience.

**Thresholds are derived per dataset and never shared across modalities.**
Stage 5 measured what happens otherwise: a threshold set at 5% false-positive
rate on ChestX-ray14 produced 91.2% false positives on ISIC, an eighteen-fold
inflation, because the two modalities' clean-score distributions barely overlap.
``InspectorConfig.allow_cross_modality_thresholds`` defaults to False for that
reason, and the agent refuses to reuse a foreign threshold set unless explicitly
told to.

**Thresholds come from validation, decisions are reported on test.** Deriving a
threshold on the same rows it is then evaluated on would make the reported
false-positive rate meaningless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from agents.base import Agent, write_agent_output
    from agents.inspector_config import CheckConfig, InspectorConfig
    from pipeline.resolution import ResolutionThresholds, check_resolution
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.base import Agent, write_agent_output  # type: ignore[no-redef]
    from agents.inspector_config import CheckConfig, InspectorConfig  # type: ignore[no-redef]
    from pipeline.resolution import ResolutionThresholds, check_resolution  # type: ignore[no-redef]


def orient(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Flip sign so that, uniformly, a higher value means worse quality."""
    return -np.asarray(scores, dtype=float) if higher_is_better else np.asarray(
        scores, dtype=float
    )


def derive_threshold(clean_oriented: np.ndarray, target_fpr: float) -> float:
    """The oriented-score threshold that flags target_fpr of clean images.

    Uses the empirical percentile of the clean distribution rather than assuming
    any parametric form, because Stage 5 showed the score distributions differ
    substantially between modalities and are not reliably Gaussian.
    """
    if len(clean_oriented) == 0:
        return float("nan")
    return float(np.percentile(clean_oriented, 100.0 * (1.0 - target_fpr)))


class DataInspectorAgent(Agent):
    """Scores images, derives thresholds from clean data, and decides."""

    name = "data_inspector"

    def __init__(self, config: InspectorConfig | None = None) -> None:
        self.inspector_config = config or InspectorConfig()
        self.inspector_config.validate()
        super().__init__(self.inspector_config.to_dict())
        self.thresholds: dict[str, dict[str, float]] = {}
        self._threshold_source_dataset: str | None = None

    # ------------------------------------------------------------------ fit
    def derive_thresholds(
        self, validation: pd.DataFrame, dataset: str
    ) -> dict[str, dict[str, float]]:
        """Derive every check's flag and reject threshold from clean val rows.

        Recorded per check together with the clean-row count behind it, so a
        threshold backed by very few images is visible as such rather than
        appearing equally trustworthy.
        """
        clean = validation[validation["defect_type"] == "clean"]
        if clean.empty:
            raise ValueError(
                f"cannot derive thresholds for {dataset}: no clean validation rows"
            )

        derived: dict[str, dict[str, float]] = {}
        checks = [*self.inspector_config.enabled_checks(), self.inspector_config.overall_check]
        for check in checks:
            if check.score_column not in clean.columns:
                continue  # reported as unavailable by run(); not silently passed
            oriented = orient(
                clean[check.score_column].to_numpy(), check.higher_is_better
            )
            finite = oriented[np.isfinite(oriented)]
            flag_fpr, reject_fpr = self.inspector_config.effective_fprs(check)
            derived[check.name] = {
                "flag_threshold": derive_threshold(finite, flag_fpr),
                "reject_threshold": derive_threshold(finite, reject_fpr),
                "flag_fpr_target": flag_fpr,
                "reject_fpr_target": reject_fpr,
                "flag_fpr_configured": check.flag_fpr,
                "fpr_policy": self.inspector_config.fpr_policy,
                "can_decide": self.inspector_config.check_can_decide(check),
                "n_clean_validation_rows": int(len(finite)),
                "score_column": check.score_column,
                "higher_is_better": check.higher_is_better,
            }
            if len(finite) < 50:
                # A percentile estimated from very few points is unstable, and a
                # 1% threshold from 20 rows is barely an estimate at all.
                derived[check.name]["low_sample_warning"] = (
                    f"threshold derived from only {len(finite)} clean row(s); "
                    "percentile estimates at this sample size are unstable"
                )
        self.thresholds = derived
        self._threshold_source_dataset = dataset
        return derived

    # ---------------------------------------------------------------- decide
    def _check_one_image(
        self, row: pd.Series, check: CheckConfig
    ) -> dict[str, Any] | None:
        entry = self.thresholds.get(check.name)
        if entry is None or check.score_column not in row.index:
            return None
        raw = row[check.score_column]
        if not np.isfinite(raw):
            # An unscoreable image is not a passing image; say so explicitly.
            return {
                "score": None,
                "oriented_score": None,
                "flag_threshold": entry["flag_threshold"],
                "reject_threshold": entry["reject_threshold"],
                "flagged": False,
                "rejected": False,
                "scoreable": False,
            }
        oriented = float(orient(np.asarray([raw]), check.higher_is_better)[0])
        return {
            "score": float(raw),
            "oriented_score": oriented,
            "flag_threshold": entry["flag_threshold"],
            "reject_threshold": entry["reject_threshold"],
            "flagged": bool(oriented > entry["flag_threshold"]),
            "rejected": bool(oriented > entry["reject_threshold"]),
            "scoreable": True,
        }

    def _severity(self, oriented: float, entry: dict[str, float]) -> float:
        """How far past the flag threshold a score sits, in threshold units.

        Reported so the Summary Agent can rank flagged images by how badly they
        failed, rather than treating every flag as equivalent. Zero means at the
        threshold; larger means worse.
        """
        span = entry["reject_threshold"] - entry["flag_threshold"]
        if not np.isfinite(span) or span <= 0:
            return 0.0
        return float(max(0.0, (oriented - entry["flag_threshold"]) / span))

    def _confidence(
        self, oriented: float | None, entry: dict[str, float] | None, decision: str
    ) -> dict[str, Any]:
        """How clear-cut a verdict was, in units of its own threshold span.

        Three bands, because "flagged" alone tells a reviewer nothing about
        whether to look:

          * ``clear``      - comfortably past the threshold that produced the
                             verdict, or comfortably below it for an acceptance.
          * ``borderline`` - within borderline_margin_fraction of the decision
                             boundary. The verdict stands, but it was marginal
                             and is the first place a reviewer should look.
          * ``unscored``   - no usable composite score, so no confidence can be
                             stated. Distinct from a confident acceptance.

        The margin is measured against the flag-to-reject span rather than an
        absolute value, so it adapts to whatever scale the composite scorer
        produces.
        """
        if oriented is None or entry is None:
            return {"band": "unscored", "distance_from_boundary": None}

        flag_threshold = entry["flag_threshold"]
        reject_threshold = entry["reject_threshold"]
        span = abs(reject_threshold - flag_threshold)
        if not np.isfinite(span) or span <= 0:
            # A degenerate span means the thresholds coincide, so no meaningful
            # distance can be reported - say so rather than inventing one.
            return {"band": "unscored", "distance_from_boundary": None}

        boundary = reject_threshold if decision == "reject" else flag_threshold
        distance = (oriented - boundary) / span
        margin = self.inspector_config.borderline_margin_fraction
        band = "borderline" if abs(distance) <= margin else "clear"
        return {
            "band": band,
            "distance_from_boundary": float(distance),
            "boundary_used": "reject_threshold" if decision == "reject" else "flag_threshold",
            "borderline_margin_fraction": margin,
        }

    def inspect_image(
        self,
        row: pd.Series,
        resolution_row: pd.Series | None,
        duplicate_of: str | None,
    ) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        for check in self.inspector_config.enabled_checks():
            result = self._check_one_image(row, check)
            if result is not None:
                checks[check.name] = result

        overall_config = self.inspector_config.overall_check
        overall = self._check_one_image(row, overall_config)
        if overall is not None:
            checks[overall_config.name] = overall

        # Structural checks: boolean by nature, so no FPR applies to them.
        if self.inspector_config.enable_resolution_check and resolution_row is not None:
            checks["resolution"] = {
                "flagged": bool(resolution_row.get("flagged", False)),
                "rejected": False,
                "scoreable": not bool(resolution_row.get("provenance_missing", False)),
                "original_width": _as_optional_int(resolution_row.get("original_width")),
                "original_height": _as_optional_int(resolution_row.get("original_height")),
                "below_min_side": bool(resolution_row.get("below_min_side", False)),
                "extreme_aspect_ratio": bool(
                    resolution_row.get("extreme_aspect_ratio", False)
                ),
                "note": "assessed pre-canonicalisation from recorded source dimensions",
            }
        if self.inspector_config.enable_duplicate_check:
            checks["duplicate"] = {
                "flagged": duplicate_of is not None,
                "rejected": False,
                "scoreable": True,
                "duplicate_of": duplicate_of,
            }

        # ---- resolve the three-way decision -----------------------------
        # Rejection is driven by the composite score only. A single specialist
        # check firing is grounds for human review, not for discarding an image:
        # Stage 2 documented that individual metrics have known confounds
        # (laplacian_variance rises under noise), so acting unilaterally on one
        # would reject images for the wrong reason.
        overall_entry = checks.get(overall_config.name)
        rejected = bool(overall_entry and overall_entry.get("rejected"))

        # Under composite_only, per-defect checks are computed and reported for
        # attribution but do not change the decision. This is what keeps the
        # image-level false-positive rate equal to the configured target rather
        # than the union of nine independent tests.
        deciding = {
            check.name
            for check in self.inspector_config.enabled_checks()
            if self.inspector_config.check_can_decide(check)
        }
        deciding.add(overall_config.name)
        flagged_checks = [
            name for name, result in checks.items()
            if result.get("flagged") and name in deciding
        ]
        attribution_only = [
            name for name, result in checks.items()
            if result.get("flagged") and name not in deciding
        ]
        # Structural findings are genuine curation issues but are not quality
        # defects, so by default they are reported without changing the quality
        # decision - see structural_findings_affect_decision in the config for
        # the measurement that motivated this.
        structural = []
        if checks.get("resolution", {}).get("flagged"):
            structural.append(("resolution", self.inspector_config.resolution_decision))
        if checks.get("duplicate", {}).get("flagged"):
            structural.append(("duplicate", self.inspector_config.duplicate_decision))
        structural_affects = self.inspector_config.structural_findings_affect_decision
        if structural_affects and any(
            decision == "reject" for _, decision in structural
        ):
            rejected = True

        structural_flags = [
            name for name, decision_kind in structural if decision_kind != "accept"
        ]
        if rejected:
            decision = "reject"
        elif flagged_checks or (structural_affects and structural_flags):
            decision = "flag"
        else:
            decision = "accept"

        # Primary issue: the check that failed by the widest margin relative to
        # its own threshold, so magnitudes measured in different units remain
        # comparable. Structural flags win when nothing scoreable fired, since
        # they are certain rather than probabilistic.
        primary_issue = None
        widest = -np.inf
        for name, result in checks.items():
            if not result.get("flagged") or not result.get("scoreable"):
                continue
            oriented = result.get("oriented_score")
            if oriented is None:
                continue
            entry = self.thresholds.get(name)
            if entry is None:
                continue
            margin = self._severity(oriented, entry)
            if margin > widest:
                widest, primary_issue = margin, name
        if primary_issue is None and structural and structural_affects:
            primary_issue = structural[0][0]

        overall_oriented = (
            overall_entry.get("oriented_score") if overall_entry else None
        )
        confidence = self._confidence(
            overall_oriented, self.thresholds.get(overall_config.name), decision
        )

        return {
            "image_id": row["image_id"],
            "split": row.get("split"),
            "decision": decision,
            "confidence": confidence,
            "primary_issue": primary_issue,
            "n_checks_flagged": len(flagged_checks) + len(attribution_only),
            "flagged_checks": flagged_checks,
            # Checks that fired but, under the active policy, do not change the
            # decision. Reported because they are still the explanation.
            "attribution_only_checks": attribution_only,
            # Curation issues rather than quality defects. Always reported; only
            # affect the decision when the config says so.
            "structural_findings": structural_flags,
            "structural_findings_affect_decision": structural_affects,
            "overall_quality_score": (
                overall_entry.get("score") if overall_entry else None
            ),
            "checks": checks,
            # Ground truth is carried through so the Summary Agent can build a
            # confusion matrix. A deployed system would not have it; it is
            # explicitly labelled as evaluation-only.
            "ground_truth_defect_type": row.get("defect_type"),
            "ground_truth_severity": row.get("severity_level"),
        }

    # ------------------------------------------------------------------- run
    def run(
        self,
        dataset: str,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        resolution: pd.DataFrame | None = None,
        duplicate_map: dict[str, str] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if (
            self._threshold_source_dataset is not None
            and self._threshold_source_dataset != dataset
            and not self.inspector_config.allow_cross_modality_thresholds
        ):
            raise ValueError(
                f"thresholds were derived on {self._threshold_source_dataset!r} but "
                f"are being applied to {dataset!r}. Stage 5 measured an "
                "eighteen-fold false-positive inflation when a ChestX-ray14 "
                "threshold was applied to ISIC. Derive thresholds for this "
                "dataset, or set allow_cross_modality_thresholds=True and state "
                "the consequence."
            )
        if not self.thresholds:
            self.derive_thresholds(validation, dataset)

        resolution_by_id: dict[str, pd.Series] = {}
        if resolution is not None:
            resolution_by_id = {
                row.image_id: row for row in resolution.itertuples(index=False)
            }
            # itertuples gives namedtuples; normalise to attribute access via a
            # dict so the caller can pass either shape.
            resolution_by_id = {
                key: pd.Series(value._asdict()) for key, value in resolution_by_id.items()
            }
        duplicates = dict(duplicate_map or {})

        images = [
            self.inspect_image(
                row,
                resolution_by_id.get(row["image_id"]),
                duplicates.get(row["image_id"]),
            )
            for _, row in test.iterrows()
        ]

        counts = {decision: 0 for decision in ("accept", "flag", "reject")}
        band_counts: dict[str, int] = {"clear": 0, "borderline": 0, "unscored": 0}
        for image in images:
            counts[image["decision"]] += 1
            band_counts[image["confidence"]["band"]] += 1

        unavailable = [
            check.name
            for check in self.inspector_config.enabled_checks()
            if check.name not in self.thresholds
        ]

        provenance = self.provenance(
            {
                "dataset": dataset,
                "n_validation_rows": int(len(validation)),
                "n_test_rows": int(len(test)),
                "threshold_source_dataset": self._threshold_source_dataset,
                **(inputs or {}),
            }
        )
        return {
            "provenance": provenance.to_dict(),
            "dataset": dataset,
            "thresholds": self.thresholds,
            "checks_unavailable": unavailable,
            "decision_counts": counts,
            "confidence_counts": band_counts,
            "images": images,
        }


def _as_optional_int(value: Any) -> int | None:
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_resolution_frame(processed: pd.DataFrame) -> pd.DataFrame:
    """Resolution flags for every processed row, ready to pass to run()."""
    return check_resolution(processed, ResolutionThresholds())


def save_inspection(payload: dict[str, Any], path: Path) -> None:
    write_agent_output(payload, path)