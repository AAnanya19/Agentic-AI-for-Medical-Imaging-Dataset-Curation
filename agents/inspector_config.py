"""Externalised configuration for the Data Inspector Agent.

**Why this is a separate module rather than constants in the agent.** Everything
here is policy, not implementation: which checks run, what false-positive rate
the thresholds target, how flag and reject are separated, and which scorer backs
each check. These are exactly the decisions a supervisor or examiner is most
likely to question, so they live in a JSON file that can be shown, edited and
re-run without touching code.

The defaults below encode findings established in earlier stages, and each is
annotated with its justification so a reader can see why the number is what it
is rather than having to trust it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# The three decisions the Inspector can reach. Ordered by severity so a
# comparison is meaningful.
DECISIONS = ("accept", "flag", "reject")


@dataclass
class CheckConfig:
    """One quality check: which scorer backs it and how it is thresholded."""

    name: str
    # Column in the scored frame this check reads. For the composite overall
    # score this is the fusion output; for per-defect checks it is the single
    # metric that Stage 2 found best for that defect.
    score_column: str
    # True when a HIGHER raw score means better quality, so the sign must be
    # flipped before thresholding.
    higher_is_better: bool
    # False-positive rate targeted on clean images when deriving the flag
    # threshold from ground truth.
    flag_fpr: float = 0.05
    # Reject is a stricter threshold on the same score: an image must look worse
    # than this to be rejected outright rather than sent for review.
    reject_fpr: float = 0.01
    enabled: bool = True
    # Free-text note carried into the output, so a reader of the audit report can
    # see the basis for the check without consulting the code.
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score_column": self.score_column,
            "higher_is_better": self.higher_is_better,
            "flag_fpr": self.flag_fpr,
            "reject_fpr": self.reject_fpr,
            "enabled": self.enabled,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckConfig":
        return cls(
            name=payload["name"],
            score_column=payload["score_column"],
            higher_is_better=bool(payload["higher_is_better"]),
            flag_fpr=float(payload.get("flag_fpr", 0.05)),
            reject_fpr=float(payload.get("reject_fpr", 0.01)),
            enabled=bool(payload.get("enabled", True)),
            rationale=payload.get("rationale", ""),
        )


# Per-defect checks, each backed by the metric Stage 2 found strongest for that
# defect on ChestX-ray14. The polarity of every metric comes from its
# registration in pipeline.iqa_metrics and is restated here so the config is
# readable standalone.
DEFAULT_CHECKS: tuple[CheckConfig, ...] = (
    CheckConfig(
        name="blur",
        score_column="laplacian_variance",
        higher_is_better=True,
        rationale=(
            "Laplacian variance reached AUROC 1.000 on gaussian_blur in Stage 2. "
            "Note it is confounded upward by noise, which is why the composite "
            "score below does not rely on it alone."
        ),
    ),
    CheckConfig(
        name="noise",
        score_column="noise_estimate",
        higher_is_better=False,
        rationale="Noise estimate reached AUROC 0.999 on gaussian_noise in Stage 2.",
    ),
    CheckConfig(
        name="contrast",
        score_column="michelson_contrast",
        higher_is_better=True,
        rationale="Michelson contrast reached AUROC 0.971 on contrast_reduction.",
    ),
    CheckConfig(
        name="brightness",
        score_column="brightness_deviation",
        higher_is_better=False,
        rationale=(
            "Absolute deviation from a clean-image reference brightness. Raw mean "
            "intensity cannot be used because the injected offset is randomly "
            "signed."
        ),
    ),
    CheckConfig(
        name="compression",
        score_column="brisque",
        higher_is_better=False,
        rationale=(
            "BRISQUE reached AUROC 0.976 on jpeg_compression, transferring to "
            "radiography better than its natural-scene-statistics basis predicts."
        ),
    ),
    CheckConfig(
        name="motion_blur",
        score_column="brisque",
        higher_is_better=False,
        rationale=(
            "BRISQUE reached AUROC 0.982 on motion_blur in Stage 2. NOTE: this "
            "check shares its score column with 'compression', so the two are "
            "the same test under two names and always fire together (the Summary "
            "Agent reports Jaccard 1.00 for the pair). Their attributions should "
            "be read as one finding: BRISQUE detects that something is wrong "
            "without distinguishing motion blur from compression."
        ),
    ),
    CheckConfig(
        name="occlusion",
        score_column="clip_iqa",
        higher_is_better=True,
        rationale=(
            "CLIP-IQA was the only scorer above chance on occlusion_patch "
            "(0.779). Every pixel-statistics metric was at or below chance, so "
            "this check is expected to be the weakest of the set."
        ),
    ),
    CheckConfig(
        name="rotation",
        score_column="laplacian_variance",
        higher_is_better=True,
        rationale=(
            "Binary detection only. Rotation severity is not ordinal under SSIM "
            "(Sampat et al. 2009), so no severity is inferred for this check. "
            "NOTE: shares its score column with 'blur', so the two always fire "
            "together and their attributions should be read as one finding."
        ),
    ),
)

# The composite check, which is what the accept/flag/reject decision is actually
# based on. Stage 4 established that logistic fusion significantly outperforms
# the best single scorer on the aggregate task, so it backs the overall decision
# while the per-defect checks above supply attribution.
DEFAULT_OVERALL_CHECK = CheckConfig(
    name="overall_quality",
    score_column="fused_logistic",
    higher_is_better=False,
    flag_fpr=0.05,
    reject_fpr=0.01,
    rationale=(
        "Logistic fusion of all IQA metrics and embedding detectors. On "
        "ChestX-ray14 this reached AUROC 0.915 against 0.817 for the best single "
        "scorer (paired bootstrap difference +0.098, 95% CI [+0.072, +0.122])."
    ),
)


# How the per-check false-positive rates are reconciled with the overall rate.
#
# **Why this is necessary.** Nine checks each independently targeting a 5%
# false-positive rate compound: the probability that at least one fires on a
# clean image is 1 - 0.95^9, roughly 37%. Measured on a demo dataset the
# realised rate was 28.6% against a nominal 5% - so a per-check target is
# meaningless as a statement about the agent unless the compounding is handled.
#
#   "per_check"   - each check independently targets its own flag_fpr. Honest
#                   about what it does, but the image-level rate is much higher
#                   than any single number in the config suggests.
#   "bonferroni"  - divide each per-defect check's flag_fpr by the number of
#                   enabled checks, so the image-level rate approximates the
#                   nominal figure. Conservative, and reduces sensitivity.
#   "composite_only" - only the fused composite score can flag or reject; the
#                   per-defect checks still run and are reported for
#                   attribution, but do not themselves change the decision.
#                   This gives an exactly controlled image-level rate and is the
#                   default, because Stage 4 established that the composite
#                   outperforms every single scorer anyway.
FPR_POLICIES = ("per_check", "bonferroni", "composite_only")


@dataclass
class InspectorConfig:
    """Complete Inspector policy. Serialisable, so it can be shown and edited."""

    # See FPR_POLICIES above. Defaults to composite_only so that the reported
    # false-positive rate is the rate actually delivered.
    fpr_policy: str = "composite_only"

    # Thresholds are derived per dataset, never shared. Stage 5 found that a
    # ChestX-ray14 threshold at 5% FPR produced 91.2% FPR on ISIC - an
    # eighteen-fold inflation - because the clean-score distributions barely
    # overlap. Cross-modality threshold reuse is therefore prohibited by default.
    allow_cross_modality_thresholds: bool = False
    checks: list[CheckConfig] = field(
        default_factory=lambda: [CheckConfig.from_dict(c.to_dict()) for c in DEFAULT_CHECKS]
    )
    overall_check: CheckConfig = field(
        default_factory=lambda: CheckConfig.from_dict(DEFAULT_OVERALL_CHECK.to_dict())
    )
    # Structural checks are boolean rather than score-thresholded: an image
    # either is below the resolution floor or is not, so no false-positive rate
    # applies to them.
    enable_resolution_check: bool = True
    enable_duplicate_check: bool = True
    # A duplicate alone is not a quality defect - it is a curation issue - so by
    # default it flags for review rather than rejecting outright.
    duplicate_decision: str = "flag"
    resolution_decision: str = "flag"
    # **Whether structural findings feed the quality decision at all.**
    #
    # A resolution or duplicate finding is not a quality defect: an image with an
    # extreme source aspect ratio genuinely has one, so counting it against a
    # quality false-positive rate is a category error. Measured on ISIC, where
    # 350 of 1,800 rows carry a resolution finding (mostly extreme aspect ratio
    # from the 0.01-302 megapixel source range), letting them flag drove the
    # realised FPR to 0.350 against a 0.050 target while the composite quality
    # score alone delivered 0.059 on ChestX-ray14.
    #
    # With this False (the default) structural findings are still computed and
    # reported in full - they are genuine curation issues the Summary Agent must
    # act on - but they are recorded in a separate ``structural_findings`` field
    # rather than changing the accept/flag/reject decision, so the quality
    # decision keeps its calibrated error rate.
    structural_findings_affect_decision: bool = False
    # **Confidence banding.** A score one point past its threshold and a score far
    # beyond it currently produce the same verdict, which throws away exactly the
    # information a reviewer needs to triage. An image whose composite score lies
    # within this fraction of the gap between the flag and reject thresholds is
    # reported as "borderline": its verdict stands, but the report can say the
    # call was marginal rather than clear-cut.
    #
    # Expressed as a fraction of the flag-to-reject span so it adapts to whatever
    # scale the composite scorer produces, rather than assuming a fixed range.
    borderline_margin_fraction: float = 0.25
    seed: int = 42

    def enabled_checks(self) -> list[CheckConfig]:
        return [check for check in self.checks if check.enabled]

    def effective_fprs(self, check: CheckConfig) -> tuple[float, float]:
        """The flag and reject FPR actually used for a check, after the policy.

        Under "bonferroni" the per-defect targets are divided by the number of
        enabled checks so their union approximates the nominal rate. The
        composite check is never adjusted: it is a single test, and it is the
        one the image-level rate is defined against.
        """
        if check.name == self.overall_check.name or self.fpr_policy != "bonferroni":
            return check.flag_fpr, check.reject_fpr
        divisor = max(1, len(self.enabled_checks()))
        return check.flag_fpr / divisor, check.reject_fpr / divisor

    def check_can_decide(self, check: CheckConfig) -> bool:
        """Whether a check's verdict changes the decision, or is attribution only.

        Under "composite_only" the per-defect checks are still computed and
        reported - they are what makes a decision explainable - but only the
        composite can flag or reject.
        """
        if self.fpr_policy != "composite_only":
            return True
        return check.name == self.overall_check.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "fpr_policy": self.fpr_policy,
            "allow_cross_modality_thresholds": self.allow_cross_modality_thresholds,
            "checks": [check.to_dict() for check in self.checks],
            "overall_check": self.overall_check.to_dict(),
            "enable_resolution_check": self.enable_resolution_check,
            "enable_duplicate_check": self.enable_duplicate_check,
            "duplicate_decision": self.duplicate_decision,
            "resolution_decision": self.resolution_decision,
            "structural_findings_affect_decision": self.structural_findings_affect_decision,
            "borderline_margin_fraction": self.borderline_margin_fraction,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InspectorConfig":
        instance = cls()
        if "checks" in payload:
            instance.checks = [
                CheckConfig.from_dict(entry) for entry in payload["checks"]
            ]
        if "overall_check" in payload:
            instance.overall_check = CheckConfig.from_dict(payload["overall_check"])
        for key in (
            "allow_cross_modality_thresholds",
            "enable_resolution_check",
            "enable_duplicate_check",
            "structural_findings_affect_decision",
        ):
            if key in payload:
                setattr(instance, key, bool(payload[key]))
        for key in ("duplicate_decision", "resolution_decision"):
            if key in payload:
                value = payload[key]
                if value not in DECISIONS:
                    raise ValueError(
                        f"{key} must be one of {DECISIONS}, got {value!r}"
                    )
                setattr(instance, key, value)
        if "fpr_policy" in payload:
            if payload["fpr_policy"] not in FPR_POLICIES:
                raise ValueError(
                    f"fpr_policy must be one of {FPR_POLICIES}, got "
                    f"{payload['fpr_policy']!r}"
                )
            instance.fpr_policy = payload["fpr_policy"]
        if "borderline_margin_fraction" in payload:
            instance.borderline_margin_fraction = float(
                payload["borderline_margin_fraction"]
            )
        if "seed" in payload:
            instance.seed = int(payload["seed"])
        instance.validate()
        return instance

    def validate(self) -> None:
        """Reject a configuration that cannot produce meaningful decisions.

        Catching these at load time rather than mid-run matters because the
        config is the part a non-author is most likely to edit.
        """
        names = [check.name for check in self.checks]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"duplicate check name(s) in config: {sorted(duplicates)}")
        for check in [*self.checks, self.overall_check]:
            if not 0.0 < check.flag_fpr < 1.0:
                raise ValueError(
                    f"{check.name}: flag_fpr must be in (0, 1), got {check.flag_fpr}"
                )
            if not 0.0 < check.reject_fpr < 1.0:
                raise ValueError(
                    f"{check.name}: reject_fpr must be in (0, 1), got {check.reject_fpr}"
                )
            if check.reject_fpr > check.flag_fpr:
                raise ValueError(
                    f"{check.name}: reject_fpr ({check.reject_fpr}) must be no "
                    f"greater than flag_fpr ({check.flag_fpr}) - rejection is a "
                    "stricter standard than flagging, so it tolerates fewer "
                    "false positives"
                )
        if self.fpr_policy not in FPR_POLICIES:
            raise ValueError(
                f"fpr_policy must be one of {FPR_POLICIES}, got {self.fpr_policy!r}"
            )
        if not 0.0 <= self.borderline_margin_fraction < 1.0:
            raise ValueError(
                "borderline_margin_fraction must be in [0, 1), got "
                f"{self.borderline_margin_fraction}"
            )
        if not self.enabled_checks():
            raise ValueError("at least one per-defect check must be enabled")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "InspectorConfig":
        return cls.from_dict(json.loads(path.read_text()))


def default_config() -> InspectorConfig:
    config = InspectorConfig()
    config.validate()
    return config