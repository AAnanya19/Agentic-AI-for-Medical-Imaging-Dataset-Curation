"""The Dataset Summary Agent: aggregates decisions and analyses their consequences.

Counting accepted, flagged and rejected images is a ``groupby``. What makes this
an agent is that it answers a question the Inspector cannot: **what happens to
the dataset if you act on these decisions?**

Four analyses, in increasing order of usefulness:

  1. **Decision aggregation** - counts by decision and by primary issue, plus the
     realised false-positive rate and recall where ground truth is available.
  2. **Issue co-occurrence** - which checks fire together. Two checks that always
     fire together are measuring the same thing, which matters when interpreting
     the per-defect attribution.
  3. **Class-balance impact** - the substantive analysis. Recomputes the class
     distribution after removing rejected images and reports which strata become
     non-viable. The literature review found this specific interaction between
     quality rejection and rare-class survival to be under-articulated: the
     imbalance literature addresses rebalancing, not rejection-induced stratum
     collapse.
  4. **Recommendations** - specific and quantified ("oversample Hernia by 3x in
     train"), never generic. The agent recommends; it never modifies the dataset.

**Multi-label handling.** ChestX-ray14 labels are pipe-separated and an image may
carry several findings, so rejecting one image can reduce several strata at once.
Class counts are therefore computed per label occurrence rather than per image,
and an image contributes to every stratum it belongs to. Treating the label
string as a single categorical value would badly understate the impact on rare
findings, which are the ones at risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from agents.base import Agent, write_agent_output
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.base import Agent, write_agent_output  # type: ignore[no-redef]


# Below this many images a stratum cannot support a stratified train/val/test
# split with at least one image per split and any meaningful validation. The
# ChestX-ray14 Hernia stratum has 13 images in total, which is why this threshold
# is not hypothetical.
DEFAULT_MIN_VIABLE_STRATUM = 10
# A stratum losing more than this fraction of its images to rejection is
# reported as materially affected even if it remains above the viability floor.
DEFAULT_MATERIAL_LOSS_FRACTION = 0.20


@dataclass
class SummaryThresholds:
    min_viable_stratum: int = DEFAULT_MIN_VIABLE_STRATUM
    material_loss_fraction: float = DEFAULT_MATERIAL_LOSS_FRACTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_viable_stratum": self.min_viable_stratum,
            "material_loss_fraction": self.material_loss_fraction,
        }


def explode_labels(labels: pd.Series, separator: str = "|") -> pd.Series:
    """One row per (image, label) pair, so multi-label images count everywhere.

    ChestX-ray14 findings are pipe-separated and frequently co-occur. Counting
    the raw label string as one category would treat "Effusion|Infiltration" as
    a class distinct from either, hiding the true size of both.
    """
    split = labels.fillna("").astype(str).str.split(separator)
    exploded = split.explode().str.strip()
    return exploded[exploded != ""]


class DatasetSummaryAgent(Agent):
    """Aggregates Inspector decisions and analyses their dataset-level impact."""

    name = "dataset_summary"

    def __init__(self, thresholds: SummaryThresholds | None = None) -> None:
        self.thresholds = thresholds or SummaryThresholds()
        super().__init__(self.thresholds.to_dict())

    # ------------------------------------------------------- aggregation
    def aggregate_decisions(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        frame = pd.DataFrame([
            {
                "image_id": image["image_id"],
                "split": image.get("split"),
                "decision": image["decision"],
                "primary_issue": image.get("primary_issue"),
                "n_checks_flagged": image.get("n_checks_flagged", 0),
                "ground_truth": image.get("ground_truth_defect_type"),
                "severity": image.get("ground_truth_severity"),
                "has_structural": bool(image.get("structural_findings")),
                "confidence_band": image.get("confidence", {}).get(
                    "band", "unscored"
                ),
            }
            for image in images
        ])

        by_decision = frame["decision"].value_counts().to_dict()
        summary: dict[str, Any] = {
            "n_images": int(len(frame)),
            "by_decision": {key: int(value) for key, value in by_decision.items()},
            "by_primary_issue": {
                str(key): int(value)
                for key, value in frame["primary_issue"].value_counts().items()
            },
            "n_with_structural_findings": int(frame["has_structural"].sum()),
            "by_confidence": {
                str(band): int(count)
                for band, count in frame["confidence_band"].value_counts().items()
            },
        }

        # Ground truth is present only in evaluation runs; a deployed inspector
        # would not have it, so its absence is normal rather than an error.
        labelled = frame[frame["ground_truth"].notna()]
        if not labelled.empty:
            clean = labelled[labelled["ground_truth"] == "clean"]
            degraded = labelled[labelled["ground_truth"] != "clean"]
            summary["evaluation"] = {
                "n_clean": int(len(clean)),
                "n_degraded": int(len(degraded)),
                "realised_false_positive_rate": (
                    float((clean["decision"] != "accept").mean())
                    if not clean.empty else None
                ),
                "recall_on_degraded": (
                    float((degraded["decision"] != "accept").mean())
                    if not degraded.empty else None
                ),
                "recall_by_severity": {
                    str(severity): float((group["decision"] != "accept").mean())
                    for severity, group in degraded.groupby("severity")
                    if str(severity) not in ("nan", "0")
                },
                "recall_by_defect_type": {
                    str(defect): float((group["decision"] != "accept").mean())
                    for defect, group in degraded.groupby("ground_truth")
                },
            }
        return summary

    def decision_correctness(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        """Which decisions were right, wrong, or marginal - not just how many.

        Aggregate FPR and recall say 6% of clean images were wrongly flagged, but
        not *which* ones, and not whether those calls were marginal or confident.
        A curator needs both: a confident false positive indicates a systematic
        problem with a threshold, whereas a marginal one is simply the operating
        point doing what it was configured to do.

        Requires injected ground truth, so this is an evaluation-time analysis. A
        deployed inspector would report only the confidence bands.
        """
        labelled = [
            image for image in images
            if image.get("ground_truth_defect_type") is not None
        ]
        if not labelled:
            return {
                "skipped": (
                    "no ground truth available, so decision correctness cannot be "
                    "assessed; confidence bands are still reported"
                )
            }

        matrix = {
            "true_positive": [], "false_positive": [],
            "true_negative": [], "false_negative": [],
        }
        for image in labelled:
            actually_degraded = image["ground_truth_defect_type"] != "clean"
            was_actioned = image["decision"] != "accept"
            if actually_degraded and was_actioned:
                key = "true_positive"
            elif not actually_degraded and was_actioned:
                key = "false_positive"
            elif not actually_degraded and not was_actioned:
                key = "true_negative"
            else:
                key = "false_negative"
            matrix[key].append(image)

        def band_breakdown(entries: list[dict[str, Any]]) -> dict[str, int]:
            counts: dict[str, int] = {"clear": 0, "borderline": 0, "unscored": 0}
            for entry in entries:
                band = entry.get("confidence", {}).get("band", "unscored")
                counts[band] = counts.get(band, 0) + 1
            return counts

        summary: dict[str, Any] = {
            "counts": {key: len(value) for key, value in matrix.items()},
            "by_confidence": {
                key: band_breakdown(value) for key, value in matrix.items()
            },
        }

        # A confident mistake is the interesting kind: it means the threshold is
        # wrong for those images, not that they sat on the boundary.
        summary["confident_mistakes"] = {
            "false_positive": band_breakdown(matrix["false_positive"])["clear"],
            "false_negative": band_breakdown(matrix["false_negative"])["clear"],
        }

        # Per-defect, so a defect the inspector systematically misses is visible
        # rather than averaged away.
        per_defect: dict[str, dict[str, int]] = {}
        for image in labelled:
            defect = str(image["ground_truth_defect_type"])
            if defect == "clean":
                continue
            entry = per_defect.setdefault(defect, {"detected": 0, "missed": 0})
            if image["decision"] != "accept":
                entry["detected"] += 1
            else:
                entry["missed"] += 1
        for defect, entry in per_defect.items():
            total = entry["detected"] + entry["missed"]
            entry["recall"] = round(entry["detected"] / total, 4) if total else 0.0
        summary["per_defect"] = per_defect

        worst = sorted(
            per_defect.items(), key=lambda item: item[1]["recall"]
        )[:3]
        summary["least_detected_defects"] = [
            {"defect": defect, **entry} for defect, entry in worst
        ]
        return summary

    def worked_examples(
        self, images: list[dict[str, Any]], per_category: int = 2
    ) -> dict[str, Any]:
        """A few specific images with their scores and thresholds side by side.

        Statistics describe a dataset; a worked example shows how a single
        decision was reached and lets a reviewer audit it. Selected
        deterministically - the most extreme cases in each category rather than a
        random sample - so the examples do not change between runs of the same
        data.
        """
        def distance(image: dict[str, Any]) -> float:
            value = image.get("confidence", {}).get("distance_from_boundary")
            return abs(value) if value is not None else -1.0

        def render(image: dict[str, Any]) -> dict[str, Any]:
            # Only the checks that actually fired, with their thresholds, so the
            # example is auditable without the full inspection file.
            fired = []
            for name, result in image.get("checks", {}).items():
                if not result.get("flagged") or result.get("score") is None:
                    continue
                fired.append({
                    "check": name,
                    "score": round(float(result["score"]), 4),
                    "flag_threshold": (
                        round(float(result["flag_threshold"]), 4)
                        if result.get("flag_threshold") is not None else None
                    ),
                })
            return {
                "image_id": image["image_id"],
                "decision": image["decision"],
                "confidence_band": image.get("confidence", {}).get("band"),
                "distance_from_boundary": (
                    round(image["confidence"]["distance_from_boundary"], 3)
                    if image.get("confidence", {}).get("distance_from_boundary")
                    is not None else None
                ),
                "primary_issue": image.get("primary_issue"),
                "checks_fired": fired[:4],
                "ground_truth_defect_type": image.get("ground_truth_defect_type"),
                "was_correct": (
                    None if image.get("ground_truth_defect_type") is None
                    else (image["decision"] != "accept")
                    == (image["ground_truth_defect_type"] != "clean")
                ),
            }

        labelled = [
            image for image in images
            if image.get("ground_truth_defect_type") is not None
        ]
        pool = labelled or images
        categories: dict[str, list[dict[str, Any]]] = {
            "confident_rejection": [
                image for image in pool
                if image["decision"] == "reject"
                and image.get("confidence", {}).get("band") == "clear"
            ],
            "borderline_decision": [
                image for image in pool
                if image.get("confidence", {}).get("band") == "borderline"
            ],
            "false_positive": [
                image for image in labelled
                if image["decision"] != "accept"
                and image["ground_truth_defect_type"] == "clean"
            ],
            "missed_defect": [
                image for image in labelled
                if image["decision"] == "accept"
                and image["ground_truth_defect_type"] != "clean"
            ],
        }
        examples: dict[str, list[dict[str, Any]]] = {}
        for name, entries in categories.items():
            if not entries:
                continue
            # Most extreme first for confident cases, closest to the boundary for
            # borderline ones.
            reverse = name != "borderline_decision"
            ordered = sorted(entries, key=distance, reverse=reverse)
            examples[name] = [render(image) for image in ordered[:per_category]]
        return examples

    def issue_cooccurrence(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        """Which checks fire together, and how often.

        Two checks that almost always fire together are measuring the same
        underlying defect, which is worth knowing when reading the per-defect
        attribution - it means the attribution is less specific than it appears.
        """
        pair_counts: dict[tuple[str, str], int] = {}
        single_counts: dict[str, int] = {}
        for image in images:
            fired = sorted(
                set(image.get("flagged_checks", []))
                | set(image.get("attribution_only_checks", []))
            )
            for name in fired:
                single_counts[name] = single_counts.get(name, 0) + 1
            for first, second in combinations(fired, 2):
                pair_counts[(first, second)] = pair_counts.get((first, second), 0) + 1

        pairs = []
        for (first, second), count in sorted(
            pair_counts.items(), key=lambda item: -item[1]
        ):
            # Jaccard rather than raw count, so a pair of rare checks that always
            # co-occur is not buried beneath a pair of common ones.
            union = single_counts[first] + single_counts[second] - count
            pairs.append({
                "check_a": first,
                "check_b": second,
                "n_together": int(count),
                "jaccard": float(count / union) if union else 0.0,
            })
        return {
            "checks_fired": {key: int(value) for key, value in single_counts.items()},
            "top_cooccurring_pairs": pairs[:12],
        }

    # -------------------------------------------------- class balance
    def class_balance_impact(
        self,
        images: list[dict[str, Any]],
        source_metadata: pd.DataFrame,
        label_column: str,
        processed_metadata: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Recompute class distribution after rejection and report what breaks.

        This is the analysis the Inspector cannot perform, because it requires
        knowing what the dataset looks like as a whole rather than image by
        image.
        """
        decisions = pd.DataFrame([
            {"image_id": image["image_id"], "decision": image["decision"]}
            for image in images
        ])

        # Inspector image_ids are processed rows; labels live on source rows, so
        # the two are joined through original_image_id when available.
        if processed_metadata is not None and "original_image_id" in processed_metadata:
            link = processed_metadata[["image_id", "original_image_id"]]
            decisions = decisions.merge(link, on="image_id", how="left")
        else:
            decisions["original_image_id"] = decisions["image_id"]

        if label_column not in source_metadata.columns:
            return {
                "skipped": (
                    f"label column {label_column!r} not present in source metadata; "
                    f"available: {sorted(source_metadata.columns)}"
                )
            }

        labels = source_metadata[["image_id", label_column]].rename(
            columns={"image_id": "original_image_id"}
        )
        joined = decisions.merge(labels, on="original_image_id", how="inner")
        if joined.empty:
            return {
                "skipped": (
                    "no inspected image could be joined to a source label; check "
                    "that processed_metadata.csv links to metadata.csv"
                )
            }

        # Explode so a multi-label image contributes to every stratum it is in.
        # Split into lists first and let pandas explode handle the index, rather
        # than assigning a pre-exploded Series back - the latter has duplicate
        # index labels and cannot be reindexed onto the original frame.
        exploded = joined.copy()
        exploded["label"] = (
            exploded[label_column].fillna("").astype(str).str.split("|")
        )
        exploded = exploded.explode("label")
        exploded["label"] = exploded["label"].str.strip()
        exploded = exploded[exploded["label"] != ""]
        if exploded.empty:
            return {
                "skipped": (
                    f"every value in {label_column!r} was empty, so no strata "
                    "could be formed"
                )
            }

        before = exploded.groupby("label").size()
        surviving = exploded[exploded["decision"] != "reject"]
        after = surviving.groupby("label").size().reindex(before.index, fill_value=0)

        strata = []
        for label in before.index:
            n_before = int(before[label])
            n_after = int(after[label])
            lost = n_before - n_after
            fraction_lost = lost / n_before if n_before else 0.0
            strata.append({
                "label": str(label),
                "n_before": n_before,
                "n_after": n_after,
                "n_rejected": lost,
                "fraction_lost": float(fraction_lost),
                "viable_before": n_before >= self.thresholds.min_viable_stratum,
                "viable_after": n_after >= self.thresholds.min_viable_stratum,
                "materially_affected": bool(
                    fraction_lost > self.thresholds.material_loss_fraction
                ),
            })
        strata.sort(key=lambda entry: entry["n_before"])

        became_non_viable = [
            entry for entry in strata
            if entry["viable_before"] and not entry["viable_after"]
        ]
        already_non_viable = [
            entry for entry in strata if not entry["viable_before"]
        ]
        materially_affected = [
            entry for entry in strata
            if entry["materially_affected"] and entry["viable_before"]
        ]

        return {
            "label_column": label_column,
            "n_strata": len(strata),
            "n_images_joined": int(len(joined)),
            "strata": strata,
            "became_non_viable": became_non_viable,
            "already_non_viable_before_rejection": already_non_viable,
            "materially_affected": materially_affected,
            "thresholds": self.thresholds.to_dict(),
        }

    def split_integrity(
        self, images: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Whether rejection falls unevenly across train, validation and test.

        A rejection rate that differs sharply by split changes the effective
        train/test proportions, which silently alters what any downstream model
        is trained and evaluated on.
        """
        frame = pd.DataFrame([
            {"split": image.get("split"), "decision": image["decision"]}
            for image in images
        ])
        frame = frame[frame["split"].notna()]
        if frame.empty:
            return {"skipped": "no split information in the inspection payload"}

        rows = []
        for split, group in frame.groupby("split"):
            surviving = int((group["decision"] != "reject").sum())
            rows.append({
                "split": str(split),
                "n_before": int(len(group)),
                "n_after": surviving,
                "rejection_rate": float((group["decision"] == "reject").mean()),
            })
        rates = [entry["rejection_rate"] for entry in rows]
        return {
            "by_split": rows,
            "rejection_rate_spread": float(max(rates) - min(rates)) if rates else 0.0,
            "uneven": bool(rates and (max(rates) - min(rates)) > 0.10),
        }

    # ------------------------------------------------- recommendations
    def recommendations(
        self,
        aggregation: dict[str, Any],
        balance: dict[str, Any],
        integrity: dict[str, Any],
        cooccurrence: dict[str, Any],
        correctness: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Specific, quantified actions. The agent recommends; it never acts."""
        actions: list[dict[str, Any]] = []

        for entry in balance.get("became_non_viable", []):
            # The central tension: quality rejection can destroy a rare stratum.
            # Both options are stated because the trade-off is a judgement the
            # curator must make, not one the agent should make silently.
            actions.append({
                "priority": "high",
                "type": "stratum_collapse",
                "label": entry["label"],
                "finding": (
                    f"Rejecting {entry['n_rejected']} image(s) reduces the "
                    f"{entry['label']} stratum from {entry['n_before']} to "
                    f"{entry['n_after']}, below the viability floor of "
                    f"{self.thresholds.min_viable_stratum}."
                ),
                "options": [
                    (
                        f"Retain the {entry['n_rejected']} rejected "
                        f"{entry['label']} image(s) with a quality caveat, "
                        "accepting known degradation to preserve the stratum."
                    ),
                    (
                        f"Accept the loss and exclude {entry['label']} from "
                        "stratified evaluation, reporting it as unsupported."
                    ),
                    (
                        f"Source additional {entry['label']} images before "
                        "applying quality rejection."
                    ),
                ],
            })

        for entry in balance.get("materially_affected", []):
            if entry["n_after"] <= 0:
                continue
            factor = entry["n_before"] / entry["n_after"]
            actions.append({
                "priority": "medium",
                "type": "resample",
                "label": entry["label"],
                "finding": (
                    f"{entry['label']} loses {entry['n_rejected']} of "
                    f"{entry['n_before']} image(s) "
                    f"({100 * entry['fraction_lost']:.0f}%) to rejection."
                ),
                "options": [
                    (
                        f"Oversample {entry['label']} by approximately "
                        f"{factor:.1f}x in train to restore its original "
                        "representation."
                    ),
                    (
                        "Apply class-weighted loss instead of resampling, "
                        f"weighting {entry['label']} by {factor:.1f}x."
                    ),
                ],
            })

        if integrity.get("uneven"):
            spread = integrity["rejection_rate_spread"]
            actions.append({
                "priority": "medium",
                "type": "split_imbalance",
                "finding": (
                    f"Rejection rates differ across splits by {100 * spread:.1f} "
                    "percentage points, which changes the effective "
                    "train/validation/test proportions."
                ),
                "options": [
                    "Re-derive splits after rejection rather than before.",
                    "Report the post-rejection split sizes alongside any result.",
                ],
            })

        structural = aggregation.get("n_with_structural_findings", 0)
        if structural:
            actions.append({
                "priority": "low",
                "type": "structural_review",
                "finding": (
                    f"{structural} image(s) carry a structural finding "
                    "(resolution or duplicate). These are curation issues rather "
                    "than quality defects and do not affect the quality decision."
                ),
                "options": [
                    "Review duplicates for split leakage before training.",
                    "Consider excluding undersized sources, whose canonical "
                    "resize upsampled rather than downsampled them.",
                ],
            })

        # A pair of checks that nearly always co-occur is not giving independent
        # evidence, which is worth flagging where attribution is being relied on.
        for pair in cooccurrence.get("top_cooccurring_pairs", []):
            if pair["jaccard"] > 0.8 and pair["n_together"] >= 10:
                actions.append({
                    "priority": "low",
                    "type": "redundant_checks",
                    "finding": (
                        f"{pair['check_a']} and {pair['check_b']} fire together on "
                        f"{pair['n_together']} image(s) (Jaccard "
                        f"{pair['jaccard']:.2f}), so they are largely measuring "
                        "the same defect."
                    ),
                    "options": [
                        "Treat their attributions as one finding rather than two.",
                    ],
                })

        correctness = correctness or {}
        confident = correctness.get("confident_mistakes", {})
        if confident.get("false_positive", 0) > 0:
            actions.append({
                "priority": "medium",
                "type": "confident_false_positives",
                "finding": (
                    f"{confident['false_positive']} clean image(s) were flagged or "
                    "rejected with high confidence, not marginally. A confident "
                    "mistake indicates the threshold is wrong for those images "
                    "rather than the operating point behaving as configured."
                ),
                "options": [
                    "Inspect the worked examples for a shared characteristic.",
                    "Consider whether the composite scorer is reacting to a "
                    "property of the source rather than a quality defect.",
                ],
            })
        least = correctness.get("least_detected_defects", [])
        if least and least[0].get("recall", 1.0) < 0.5:
            worst = least[0]
            actions.append({
                "priority": "medium",
                "type": "systematically_missed_defect",
                "finding": (
                    f"{worst['defect']} was detected in only "
                    f"{worst['detected']} of {worst['detected'] + worst['missed']} "
                    f"case(s) (recall {worst['recall']:.3f}), the weakest of any "
                    "defect type."
                ),
                "options": [
                    f"Report {worst['defect']} detection as a known limitation.",
                    "Add or reweight a check targeting this defect specifically.",
                ],
            })

        evaluation = aggregation.get("evaluation", {})
        recall = evaluation.get("recall_on_degraded")
        if recall is not None and recall < 0.6:
            actions.append({
                "priority": "high",
                "type": "low_recall",
                "finding": (
                    f"Only {100 * recall:.0f}% of known-degraded images were "
                    "flagged or rejected, so a majority of injected defects pass "
                    "inspection at the current operating point."
                ),
                "options": [
                    "Raise the target false-positive rate to buy recall, and "
                    "report the trade-off explicitly.",
                    "Report this recall as a limitation of the configuration.",
                ],
            })
        return actions

    # ------------------------------------------------------------ run
    def run(
        self,
        inspection: dict[str, Any],
        source_metadata: pd.DataFrame | None = None,
        processed_metadata: pd.DataFrame | None = None,
        label_column: str | None = None,
    ) -> dict[str, Any]:
        images = inspection["images"]
        dataset = inspection.get("dataset", "unknown")

        aggregation = self.aggregate_decisions(images)
        correctness = self.decision_correctness(images)
        examples = self.worked_examples(images)
        cooccurrence = self.issue_cooccurrence(images)
        integrity = self.split_integrity(images)

        if source_metadata is not None and label_column:
            balance = self.class_balance_impact(
                images, source_metadata, label_column, processed_metadata
            )
        else:
            balance = {
                "skipped": (
                    "no source metadata or label column supplied, so "
                    "class-balance impact could not be computed"
                )
            }

        actions = self.recommendations(
            aggregation, balance, integrity, cooccurrence, correctness
        )

        provenance = self.provenance({
            "dataset": dataset,
            "inspection_run_id": inspection.get("provenance", {}).get("run_id"),
            "inspection_agent": inspection.get("provenance", {}).get("agent"),
            "n_images": len(images),
            "label_column": label_column,
        })
        return {
            "provenance": provenance.to_dict(),
            "dataset": dataset,
            "decisions": aggregation,
            "decision_correctness": correctness,
            "worked_examples": examples,
            "issue_cooccurrence": cooccurrence,
            "class_balance_impact": balance,
            "split_integrity": integrity,
            "recommendations": actions,
        }


def save_summary(payload: dict[str, Any], path: Path) -> None:
    write_agent_output(payload, path)