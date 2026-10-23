"""Tests for the Dataset Summary Agent.

Four invariants:

1. **Multi-label images must count in every stratum they belong to.**
   ChestX-ray14 findings are pipe-separated and co-occur frequently. Treating
   "Effusion|Infiltration" as a single category would report two 200-image
   strata as one 200-image stratum of a fabricated class, badly understating the
   size - and therefore the risk - of each real finding.

2. **Rejection-induced stratum collapse must be detected.** This is the
   substantive analysis: the ChestX-ray14 Hernia stratum has 13 images, so
   rejecting a handful destroys it. The prior-art review found this specific
   interaction under-articulated in the class-imbalance literature.

3. **Recommendations must be quantified, never generic.** "Oversample Hernia by
   3.0x" is actionable; "consider rebalancing" is not.

4. **The agent recommends but never modifies.** Nothing here writes to the
   dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.base import SCHEMA_VERSION, SchemaMismatchError, write_agent_output
from agents.dataset_summary import (
    DatasetSummaryAgent, SummaryThresholds, explode_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_image(
    image_id: str,
    decision: str,
    split: str = "test",
    primary_issue: str | None = None,
    flagged: list[str] | None = None,
    attribution: list[str] | None = None,
    structural: list[str] | None = None,
    ground_truth: str = "clean",
    severity: str = "0",
) -> dict:
    return {
        "image_id": image_id,
        "split": split,
        "decision": decision,
        "primary_issue": primary_issue,
        "n_checks_flagged": len(flagged or []) + len(attribution or []),
        "flagged_checks": flagged or [],
        "attribution_only_checks": attribution or [],
        "structural_findings": structural or [],
        "ground_truth_defect_type": ground_truth,
        "ground_truth_severity": severity,
    }


def make_inspection(images: list[dict], dataset: str = "demo") -> dict:
    return {
        "dataset": dataset,
        "images": images,
        "provenance": {
            "agent": "data_inspector",
            "run_id": "abc123",
            "schema_version": SCHEMA_VERSION,
        },
    }


# ---------------------------------------------------------------------------
# Multi-label handling
# ---------------------------------------------------------------------------

def test_explode_labels_splits_pipe_separated_multi_labels() -> None:
    labels = pd.Series(["Effusion|Infiltration", "Hernia", "", "Effusion", None])
    exploded = list(explode_labels(labels))
    assert exploded.count("Effusion") == 2
    assert "Infiltration" in exploded and "Hernia" in exploded
    assert "" not in exploded


def test_a_multi_label_image_counts_in_every_stratum() -> None:
    """The core multi-label guarantee. 200 images labelled
    "Effusion|Infiltration" must give TWO strata of 200, not one."""
    images = [make_image(f"i{index}", "accept") for index in range(200)]
    source = pd.DataFrame([
        {"image_id": f"i{index}", "source_labels": "Effusion|Infiltration"}
        for index in range(200)
    ])
    agent = DatasetSummaryAgent()
    balance = agent.class_balance_impact(images, source, "source_labels")
    by_label = {entry["label"]: entry["n_before"] for entry in balance["strata"]}
    assert by_label == {"Effusion": 200, "Infiltration": 200}


def test_empty_labels_do_not_create_a_stratum() -> None:
    images = [make_image(f"i{index}", "accept") for index in range(10)]
    source = pd.DataFrame([
        {"image_id": f"i{index}", "source_labels": "" if index < 5 else "Hernia"}
        for index in range(10)
    ])
    balance = DatasetSummaryAgent().class_balance_impact(
        images, source, "source_labels"
    )
    assert [entry["label"] for entry in balance["strata"]] == ["Hernia"]


def test_single_label_datasets_work_unchanged() -> None:
    """ISIC carries one diagnosis per lesion, with no separator."""
    images = [make_image(f"i{index}", "accept") for index in range(30)]
    source = pd.DataFrame([
        {"image_id": f"i{index}", "diagnosis": "melanoma" if index < 10 else "nevus"}
        for index in range(30)
    ])
    balance = DatasetSummaryAgent().class_balance_impact(images, source, "diagnosis")
    by_label = {entry["label"]: entry["n_before"] for entry in balance["strata"]}
    assert by_label == {"melanoma": 10, "nevus": 20}


# ---------------------------------------------------------------------------
# Stratum collapse
# ---------------------------------------------------------------------------

def _rare_stratum_case() -> tuple[list[dict], pd.DataFrame]:
    """A 12-image Hernia stratum losing 4 to rejection, and a large safe one.

    Mirrors ChestX-ray14, where Hernia has 13 images in the sampled corpus.
    """
    images, source = [], []
    for index in range(12):
        decision = "reject" if index < 4 else "accept"
        images.append(make_image(
            f"h{index}", decision,
            ground_truth="gaussian_blur" if decision == "reject" else "clean",
            severity="2" if decision == "reject" else "0",
        ))
        source.append({"image_id": f"h{index}", "source_labels": "Hernia"})
    for index in range(200):
        decision = "reject" if index < 5 else "accept"
        images.append(make_image(f"e{index}", decision))
        source.append({"image_id": f"e{index}", "source_labels": "Effusion"})
    return images, pd.DataFrame(source)


def test_rejection_induced_stratum_collapse_is_detected() -> None:
    images, source = _rare_stratum_case()
    agent = DatasetSummaryAgent(SummaryThresholds(min_viable_stratum=10))
    balance = agent.class_balance_impact(images, source, "source_labels")
    assert [entry["label"] for entry in balance["became_non_viable"]] == ["Hernia"]
    hernia = next(e for e in balance["strata"] if e["label"] == "Hernia")
    assert hernia["n_before"] == 12 and hernia["n_after"] == 8


def test_a_large_stratum_is_not_reported_as_collapsing() -> None:
    images, source = _rare_stratum_case()
    balance = DatasetSummaryAgent().class_balance_impact(
        images, source, "source_labels"
    )
    effusion = next(e for e in balance["strata"] if e["label"] == "Effusion")
    assert effusion["viable_after"] is True
    assert effusion["materially_affected"] is False


def test_a_stratum_already_below_the_floor_is_reported_separately() -> None:
    """A stratum that was never viable was not destroyed by rejection, and
    conflating the two would misattribute the cause."""
    images = [make_image(f"r{index}", "accept") for index in range(5)]
    source = pd.DataFrame([
        {"image_id": f"r{index}", "source_labels": "VeryRare"} for index in range(5)
    ])
    agent = DatasetSummaryAgent(SummaryThresholds(min_viable_stratum=10))
    balance = agent.class_balance_impact(images, source, "source_labels")
    assert not balance["became_non_viable"]
    assert [e["label"] for e in balance["already_non_viable_before_rejection"]] == [
        "VeryRare"
    ]


def test_only_rejected_images_are_removed_not_flagged_ones() -> None:
    """Flagging routes an image to human review; it does not remove it."""
    images = [make_image(f"f{index}", "flag") for index in range(20)]
    source = pd.DataFrame([
        {"image_id": f"f{index}", "source_labels": "Cardiomegaly"}
        for index in range(20)
    ])
    balance = DatasetSummaryAgent().class_balance_impact(
        images, source, "source_labels"
    )
    entry = balance["strata"][0]
    assert entry["n_before"] == entry["n_after"] == 20
    assert entry["n_rejected"] == 0


def test_class_balance_joins_through_processed_metadata() -> None:
    """Inspector image_ids are processed rows; labels live on source rows."""
    images = [make_image("img0.png_clean", "reject")]
    processed = pd.DataFrame([
        {"image_id": "img0.png_clean", "original_image_id": "img0.png"}
    ])
    source = pd.DataFrame([{"image_id": "img0.png", "source_labels": "Hernia"}])
    balance = DatasetSummaryAgent().class_balance_impact(
        images, source, "source_labels", processed_metadata=processed
    )
    assert balance["n_images_joined"] == 1
    assert balance["strata"][0]["n_rejected"] == 1


def test_missing_label_column_is_reported_not_guessed() -> None:
    images = [make_image("i0", "accept")]
    source = pd.DataFrame([{"image_id": "i0", "some_other_column": "x"}])
    balance = DatasetSummaryAgent().class_balance_impact(
        images, source, "source_labels"
    )
    assert "skipped" in balance
    assert "not present" in balance["skipped"]


def test_unjoinable_ids_are_reported_rather_than_silently_empty() -> None:
    images = [make_image("nonexistent", "accept")]
    source = pd.DataFrame([{"image_id": "different", "source_labels": "Hernia"}])
    balance = DatasetSummaryAgent().class_balance_impact(
        images, source, "source_labels"
    )
    assert "skipped" in balance


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_decision_counts_are_accurate() -> None:
    images = (
        [make_image(f"a{i}", "accept") for i in range(10)]
        + [make_image(f"f{i}", "flag") for i in range(5)]
        + [make_image(f"r{i}", "reject") for i in range(3)]
    )
    aggregation = DatasetSummaryAgent().aggregate_decisions(images)
    assert aggregation["by_decision"] == {"accept": 10, "flag": 5, "reject": 3}
    assert aggregation["n_images"] == 18


def test_realised_fpr_and_recall_are_computed_from_ground_truth() -> None:
    images = (
        [make_image(f"c{i}", "accept", ground_truth="clean") for i in range(90)]
        + [make_image(f"c{i}", "flag", ground_truth="clean") for i in range(90, 100)]
        + [
            make_image(f"d{i}", "reject", ground_truth="gaussian_blur", severity="3")
            for i in range(40)
        ]
        + [
            make_image(f"d{i}", "accept", ground_truth="gaussian_blur", severity="1")
            for i in range(40, 50)
        ]
    )
    evaluation = DatasetSummaryAgent().aggregate_decisions(images)["evaluation"]
    assert evaluation["realised_false_positive_rate"] == pytest.approx(0.10)
    assert evaluation["recall_on_degraded"] == pytest.approx(0.80)
    assert evaluation["recall_by_severity"]["3"] == pytest.approx(1.0)
    assert evaluation["recall_by_severity"]["1"] == pytest.approx(0.0)


def test_evaluation_section_is_absent_without_ground_truth() -> None:
    """A deployed inspector has no ground truth; that is normal, not an error."""
    images = [
        {
            "image_id": f"i{index}", "split": "test", "decision": "accept",
            "primary_issue": None, "flagged_checks": [],
            "attribution_only_checks": [], "structural_findings": [],
        }
        for index in range(10)
    ]
    aggregation = DatasetSummaryAgent().aggregate_decisions(images)
    assert "evaluation" not in aggregation


def test_structural_findings_are_counted_separately() -> None:
    images = (
        [make_image(f"s{i}", "accept", structural=["resolution"]) for i in range(7)]
        + [make_image(f"n{i}", "accept") for i in range(20)]
    )
    aggregation = DatasetSummaryAgent().aggregate_decisions(images)
    assert aggregation["n_with_structural_findings"] == 7


# ---------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------

def test_cooccurrence_identifies_checks_that_always_fire_together() -> None:
    """Two checks reading the same score column are not independent evidence.
    This was found in the real Stage 7 config: compression and motion_blur both
    read brisque, giving Jaccard 1.00."""
    images = [
        make_image(f"i{index}", "flag", flagged=["compression", "motion_blur"])
        for index in range(20
        )
    ]
    cooccurrence = DatasetSummaryAgent().issue_cooccurrence(images)
    pair = cooccurrence["top_cooccurring_pairs"][0]
    assert {pair["check_a"], pair["check_b"]} == {"compression", "motion_blur"}
    assert pair["jaccard"] == pytest.approx(1.0)


def test_cooccurrence_uses_jaccard_not_raw_count() -> None:
    """Otherwise a pair of rare checks that always co-occur is buried beneath a
    pair of common ones that rarely do."""
    images = (
        [make_image(f"r{i}", "flag", flagged=["rare_a", "rare_b"]) for i in range(5)]
        + [make_image(f"c{i}", "flag", flagged=["common_a"]) for i in range(50)]
        + [make_image(f"d{i}", "flag", flagged=["common_b"]) for i in range(50)]
        + [
            make_image(f"e{i}", "flag", flagged=["common_a", "common_b"])
            for i in range(8)
        ]
    )
    pairs = DatasetSummaryAgent().issue_cooccurrence(images)["top_cooccurring_pairs"]
    by_pair = {
        frozenset((entry["check_a"], entry["check_b"])): entry for entry in pairs
    }
    assert by_pair[frozenset(("rare_a", "rare_b"))]["jaccard"] == pytest.approx(1.0)
    assert by_pair[frozenset(("common_a", "common_b"))]["jaccard"] < 0.2


def test_attribution_only_checks_are_included_in_cooccurrence() -> None:
    """Under composite_only these do not decide, but they are still evidence."""
    images = [
        make_image(f"i{index}", "reject", flagged=["overall_quality"],
                   attribution=["blur", "noise"])
        for index in range(15)
    ]
    cooccurrence = DatasetSummaryAgent().issue_cooccurrence(images)
    assert cooccurrence["checks_fired"]["blur"] == 15
    assert any(
        {entry["check_a"], entry["check_b"]} == {"blur", "noise"}
        for entry in cooccurrence["top_cooccurring_pairs"]
    )


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------

def test_uneven_rejection_across_splits_is_flagged() -> None:
    images = (
        [make_image(f"t{i}", "reject", split="train") for i in range(40)]
        + [make_image(f"t{i}", "accept", split="train") for i in range(40, 100)]
        + [make_image(f"v{i}", "accept", split="test") for i in range(100)]
    )
    integrity = DatasetSummaryAgent().split_integrity(images)
    assert integrity["uneven"] is True
    assert integrity["rejection_rate_spread"] == pytest.approx(0.40)


def test_even_rejection_across_splits_is_not_flagged() -> None:
    images = []
    for split in ("train", "val", "test"):
        images += [make_image(f"{split}{i}", "reject", split=split) for i in range(5)]
        images += [
            make_image(f"{split}{i}", "accept", split=split) for i in range(5, 100)
        ]
    integrity = DatasetSummaryAgent().split_integrity(images)
    assert integrity["uneven"] is False


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def test_stratum_collapse_yields_a_high_priority_recommendation() -> None:
    images, source = _rare_stratum_case()
    agent = DatasetSummaryAgent(SummaryThresholds(min_viable_stratum=10))
    payload = agent.run(
        make_inspection(images), source_metadata=source, label_column="source_labels"
    )
    collapse = [
        action for action in payload["recommendations"]
        if action["type"] == "stratum_collapse"
    ]
    assert len(collapse) == 1
    assert collapse[0]["priority"] == "high"
    assert collapse[0]["label"] == "Hernia"
    # The trade-off must be surfaced, not resolved silently by the agent.
    assert len(collapse[0]["options"]) >= 2


def test_recommendations_are_quantified_not_generic() -> None:
    images, source = _rare_stratum_case()
    payload = DatasetSummaryAgent(
        SummaryThresholds(min_viable_stratum=10, material_loss_fraction=0.10)
    ).run(make_inspection(images), source_metadata=source, label_column="source_labels")
    resample = [
        action for action in payload["recommendations"] if action["type"] == "resample"
    ]
    assert resample, "a materially affected stratum should yield a resample action"
    # A specific multiplier, not "consider rebalancing".
    assert any("x in train" in option for option in resample[0]["options"])


def test_low_recall_produces_a_high_priority_recommendation() -> None:
    images = (
        [make_image(f"c{i}", "accept", ground_truth="clean") for i in range(100)]
        + [
            make_image(f"d{i}", "accept", ground_truth="gaussian_blur", severity="1")
            for i in range(60)
        ]
        + [
            make_image(f"d{i}", "reject", ground_truth="gaussian_blur", severity="3")
            for i in range(60, 100)
        ]
    )
    payload = DatasetSummaryAgent().run(make_inspection(images))
    low_recall = [
        action for action in payload["recommendations"]
        if action["type"] == "low_recall"
    ]
    assert low_recall and low_recall[0]["priority"] == "high"


def test_no_recommendations_when_nothing_is_wrong() -> None:
    images = [make_image(f"a{i}", "accept") for i in range(100)]
    payload = DatasetSummaryAgent().run(make_inspection(images))
    assert payload["recommendations"] == []


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_output_carries_provenance_linking_to_the_inspection_run() -> None:
    images, source = _rare_stratum_case()
    payload = DatasetSummaryAgent().run(
        make_inspection(images), source_metadata=source, label_column="source_labels"
    )
    provenance = payload["provenance"]
    assert provenance["agent"] == "dataset_summary"
    assert provenance["schema_version"] == SCHEMA_VERSION
    # Traceability: which inspection produced this summary.
    assert provenance["inputs"]["inspection_run_id"] == "abc123"


def test_agent_never_modifies_the_source_metadata() -> None:
    images, source = _rare_stratum_case()
    before = source.copy(deep=True)
    DatasetSummaryAgent().run(
        make_inspection(images), source_metadata=source, label_column="source_labels"
    )
    pd.testing.assert_frame_equal(source, before)


def test_summary_is_json_serialisable(tmp_path: Path) -> None:
    """The Explainer reads this from disk, so every value must survive JSON."""
    images, source = _rare_stratum_case()
    payload = DatasetSummaryAgent().run(
        make_inspection(images), source_metadata=source, label_column="source_labels"
    )
    path = tmp_path / "summary.json"
    write_agent_output(payload, path)
    restored = json.loads(path.read_text())
    assert restored["dataset"] == payload["dataset"]
    assert len(restored["recommendations"]) == len(payload["recommendations"])


def test_reproducible_for_identical_input() -> None:
    images, source = _rare_stratum_case()
    inspection = make_inspection(images)
    first = DatasetSummaryAgent().run(
        inspection, source_metadata=source, label_column="source_labels"
    )
    second = DatasetSummaryAgent().run(
        inspection, source_metadata=source, label_column="source_labels"
    )
    assert first["decisions"] == second["decisions"]
    assert first["class_balance_impact"] == second["class_balance_impact"]
    assert first["recommendations"] == second["recommendations"]