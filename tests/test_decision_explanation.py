"""Tests for decision correctness, confidence banding, and worked examples.

These close three gaps in what the audit report could previously say. It reported
*how many* images were flagged, but not whether a given flag was right, and not
whether the call was clear-cut or marginal - which is precisely what a curator
needs to triage a review queue.

Four invariants:

1. **The confusion matrix must be exact.** It is the basis for every statement
   about whether decisions were correct.

2. **A confident mistake must be distinguishable from a marginal one.** A clean
   image flagged far past the threshold indicates the threshold is wrong for that
   image; one flagged just past it is the operating point behaving as configured.
   Conflating them hides the actionable case.

3. **Worked examples must be deterministic.** They will be quoted in a
   dissertation, so the same data must always produce the same examples.

4. **The template must still score zero hallucination with per-image numbers in
   play.** Adding worked examples adds numbers the report could misquote, so the
   control has to be re-established rather than assumed.
"""

from __future__ import annotations

import pytest

from agents.data_inspector import DataInspectorAgent
from agents.dataset_summary import DatasetSummaryAgent
from agents.inspector_config import InspectorConfig
from agents.llm_explainer import ExplainerConfig, LLMExplainerAgent, build_fact_sheet


def make_image(
    image_id: str,
    decision: str,
    band: str,
    distance: float,
    ground_truth: str,
    primary_issue: str | None = None,
    checks: dict | None = None,
) -> dict:
    return {
        "image_id": image_id, "split": "test", "decision": decision,
        "confidence": {"band": band, "distance_from_boundary": distance},
        "primary_issue": primary_issue,
        "flagged_checks": [primary_issue] if primary_issue else [],
        "attribution_only_checks": [], "structural_findings": [],
        "ground_truth_defect_type": ground_truth,
        "ground_truth_severity": "2" if ground_truth != "clean" else "0",
        "checks": checks or {},
    }


def population() -> list[dict]:
    """40 true positives, 6 false positives (2 confident), 10 missed, 90 correct
    acceptances - a known confusion matrix."""
    images = []
    for index in range(40):
        images.append(make_image(
            f"tp{index}", "reject", "clear", 3.0, "gaussian_blur", "blur",
            {"blur": {"score": 210.3, "flag_threshold": 615.2, "flagged": True},
             "overall_quality": {"score": 0.913, "flag_threshold": 0.245,
                                 "flagged": True}},
        ))
    for index in range(4):
        images.append(make_image(
            f"fp{index}", "flag", "borderline", 0.1, "clean", "brightness"
        ))
    for index in range(2):
        images.append(make_image(
            f"fpc{index}", "reject", "clear", 2.5, "clean", "occlusion"
        ))
    for index in range(10):
        images.append(make_image(
            f"fn{index}", "accept", "clear", -2.0, "occlusion_patch"
        ))
    for index in range(90):
        images.append(make_image(f"tn{index}", "accept", "clear", -3.0, "clean"))
    return images


def make_summary(images: list[dict] | None = None) -> dict:
    return DatasetSummaryAgent().run({
        "dataset": "chestxray14",
        "images": images if images is not None else population(),
        "provenance": {"run_id": "r1", "agent": "data_inspector"},
    })


# ---------------------------------------------------------------------------
# Confidence banding in the Inspector
# ---------------------------------------------------------------------------

def _score_frame(n_clean: int, n_degraded: int, seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_clean):
        rows.append({
            "image_id": f"c{index}", "split": "x", "defect_type": "clean",
            "severity_level": "0", "m1": rng.normal(0, 1),
            "fused_logistic": rng.normal(0.2, 0.08),
        })
    for index in range(n_degraded):
        rows.append({
            "image_id": f"b{index}", "split": "x", "defect_type": "gaussian_blur",
            "severity_level": "2", "m1": rng.normal(2.5, 1),
            "fused_logistic": rng.normal(0.85, 0.08),
        })
    return pd.DataFrame(rows)


def _banding_config(margin: float = 0.25) -> InspectorConfig:
    from agents.inspector_config import CheckConfig

    config = InspectorConfig()
    config.borderline_margin_fraction = margin
    config.checks = [
        CheckConfig(name="chk1", score_column="m1", higher_is_better=False)
    ]
    config.overall_check = CheckConfig(
        name="overall_quality", score_column="fused_logistic",
        higher_is_better=False,
    )
    config.enable_resolution_check = False
    config.enable_duplicate_check = False
    config.validate()
    return config


def test_every_image_receives_a_confidence_band() -> None:
    agent = DataInspectorAgent(_banding_config())
    payload = agent.run("demo", _score_frame(300, 150, 1), _score_frame(300, 150, 2))
    bands = {image["confidence"]["band"] for image in payload["images"]}
    assert bands <= {"clear", "borderline", "unscored"}
    assert payload["confidence_counts"]["clear"] > 0


def test_borderline_images_sit_closer_to_the_boundary_than_clear_ones() -> None:
    """The band must reflect actual distance, not be arbitrary."""
    import numpy as np

    agent = DataInspectorAgent(_banding_config())
    payload = agent.run("demo", _score_frame(600, 300, 1), _score_frame(600, 300, 2))
    borderline = [
        abs(i["confidence"]["distance_from_boundary"])
        for i in payload["images"] if i["confidence"]["band"] == "borderline"
    ]
    clear = [
        abs(i["confidence"]["distance_from_boundary"])
        for i in payload["images"] if i["confidence"]["band"] == "clear"
    ]
    assert borderline and clear
    assert max(borderline) < np.mean(clear)


def test_a_wider_margin_produces_more_borderline_cases() -> None:
    validation, test = _score_frame(600, 300, 1), _score_frame(600, 300, 2)
    narrow = DataInspectorAgent(_banding_config(0.1)).run("demo", validation, test)
    wide = DataInspectorAgent(_banding_config(0.6)).run("demo", validation, test)
    assert (
        wide["confidence_counts"]["borderline"]
        > narrow["confidence_counts"]["borderline"]
    )


def test_an_unscoreable_image_is_unscored_not_confidently_accepted() -> None:
    import numpy as np

    validation = _score_frame(300, 150, 1)
    test = _score_frame(20, 10, 2)
    test.loc[test.index[0], "fused_logistic"] = np.nan
    payload = DataInspectorAgent(_banding_config()).run("demo", validation, test)
    assert payload["images"][0]["confidence"]["band"] == "unscored"
    assert payload["images"][0]["confidence"]["distance_from_boundary"] is None


def test_config_rejects_an_out_of_range_borderline_margin() -> None:
    config = InspectorConfig()
    config.borderline_margin_fraction = 1.5
    with pytest.raises(ValueError, match="borderline_margin_fraction"):
        config.validate()


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def test_the_confusion_matrix_is_exact() -> None:
    correctness = DatasetSummaryAgent().decision_correctness(population())
    assert correctness["counts"] == {
        "true_positive": 40, "false_positive": 6,
        "true_negative": 90, "false_negative": 10,
    }


def test_confident_mistakes_are_separated_from_marginal_ones() -> None:
    """Of six false positives, four were borderline and two confident. The two
    are the actionable ones, and conflating them would hide that."""
    correctness = DatasetSummaryAgent().decision_correctness(population())
    assert correctness["confident_mistakes"]["false_positive"] == 2
    assert correctness["by_confidence"]["false_positive"] == {
        "clear": 2, "borderline": 4, "unscored": 0,
    }


def test_the_least_detected_defect_is_identified() -> None:
    correctness = DatasetSummaryAgent().decision_correctness(population())
    worst = correctness["least_detected_defects"][0]
    assert worst["defect"] == "occlusion_patch"
    assert worst["recall"] == 0.0


def test_per_defect_recall_is_computed() -> None:
    correctness = DatasetSummaryAgent().decision_correctness(population())
    assert correctness["per_defect"]["gaussian_blur"]["recall"] == 1.0
    assert correctness["per_defect"]["occlusion_patch"]["recall"] == 0.0


def test_correctness_is_skipped_without_ground_truth() -> None:
    """A deployed inspector has none; confidence bands still apply."""
    images = [
        {
            "image_id": f"i{index}", "split": "test", "decision": "accept",
            "confidence": {"band": "clear", "distance_from_boundary": -2.0},
            "primary_issue": None, "flagged_checks": [],
            "attribution_only_checks": [], "structural_findings": [], "checks": {},
        }
        for index in range(10)
    ]
    correctness = DatasetSummaryAgent().decision_correctness(images)
    assert "skipped" in correctness


def test_confident_false_positives_produce_a_recommendation() -> None:
    summary = make_summary()
    types = {action["type"] for action in summary["recommendations"]}
    assert "confident_false_positives" in types


def test_a_systematically_missed_defect_produces_a_recommendation() -> None:
    summary = make_summary()
    types = {action["type"] for action in summary["recommendations"]}
    assert "systematically_missed_defect" in types


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------

def test_worked_examples_cover_all_four_categories() -> None:
    examples = DatasetSummaryAgent().worked_examples(population())
    assert set(examples) == {
        "confident_rejection", "borderline_decision",
        "false_positive", "missed_defect",
    }


def test_worked_examples_record_whether_the_decision_was_correct() -> None:
    examples = DatasetSummaryAgent().worked_examples(population())
    assert examples["confident_rejection"][0]["was_correct"] is True
    assert examples["false_positive"][0]["was_correct"] is False
    assert examples["missed_defect"][0]["was_correct"] is False


def test_worked_examples_carry_scores_beside_their_thresholds() -> None:
    """The point of an example is that it can be audited without the full
    inspection file."""
    examples = DatasetSummaryAgent().worked_examples(population())
    fired = examples["confident_rejection"][0]["checks_fired"]
    assert fired
    assert fired[0]["score"] is not None
    assert fired[0]["flag_threshold"] is not None


def test_worked_examples_are_deterministic() -> None:
    """They will be quoted in a dissertation, so they must not vary between runs."""
    images = population()
    first = DatasetSummaryAgent().worked_examples(images)
    second = DatasetSummaryAgent().worked_examples(images)
    assert first == second


def test_the_borderline_example_is_the_closest_to_the_boundary() -> None:
    images = population()
    images.append(make_image("far", "flag", "borderline", 0.24, "clean", "noise"))
    images.append(make_image("near", "flag", "borderline", 0.01, "clean", "noise"))
    examples = DatasetSummaryAgent().worked_examples(images)
    assert examples["borderline_decision"][0]["image_id"] == "near"


def test_worked_examples_omit_categories_with_no_members() -> None:
    """A dataset with no false positives must not produce a fabricated example."""
    clean_only = [
        make_image(f"tn{index}", "accept", "clear", -3.0, "clean")
        for index in range(20)
    ]
    examples = DatasetSummaryAgent().worked_examples(clean_only)
    assert "false_positive" not in examples
    assert "confident_rejection" not in examples


# ---------------------------------------------------------------------------
# The Explainer
# ---------------------------------------------------------------------------

def test_the_fact_sheet_carries_correctness_confidence_and_examples() -> None:
    sheet = build_fact_sheet(make_summary())
    assert sheet["correctness"]["n_wrongly_actioned"] == 6
    assert sheet["correctness"]["n_confident_wrong_flags"] == 2
    assert sheet["confidence"]["n_borderline"] == 4
    assert "confident_rejection" in sheet["examples"]


def test_the_template_still_scores_zero_with_per_image_numbers() -> None:
    """Adding worked examples adds numbers the report could misquote, so the
    zero-hallucination control must be re-established, not assumed."""
    payload = LLMExplainerAgent(ExplainerConfig(backend="template")).run(make_summary())
    factual = payload["evaluation"]["factual_consistency"]
    assert factual["hallucination_rate"] == 0.0, factual["unverifiable_examples"]
    # The examples really are in the report, so this is a meaningful test.
    assert factual["n_numbers"] > 30


def test_the_report_explains_whether_decisions_were_right() -> None:
    payload = LLMExplainerAgent(ExplainerConfig(backend="template")).run(make_summary())
    report = payload["report"]
    assert "Were the decisions right?" in report
    assert "wrongly actioned" in report
    assert "passed inspection" in report


def test_the_report_distinguishes_confident_from_borderline_mistakes() -> None:
    report = LLMExplainerAgent(ExplainerConfig(backend="template")).run(
        make_summary()
    )["report"]
    assert "high confidence rather than marginally" in report
    assert "threshold is wrong" in report


def test_the_report_shows_a_worked_example_with_its_threshold() -> None:
    report = LLMExplainerAgent(ExplainerConfig(backend="template")).run(
        make_summary()
    )["report"]
    assert "Worked examples" in report
    assert "against a threshold of" in report


def test_the_report_recommends_review_of_borderline_cases() -> None:
    report = LLMExplainerAgent(ExplainerConfig(backend="template")).run(
        make_summary()
    )["report"]
    assert "borderline" in report.lower()
    assert "human review" in report.lower()


def test_sections_are_omitted_when_the_data_is_absent() -> None:
    """A report about a dataset with no ground truth must not claim correctness."""
    images = [
        {
            "image_id": f"i{index}", "split": "test", "decision": "accept",
            "confidence": {"band": "clear", "distance_from_boundary": -2.0},
            "primary_issue": None, "flagged_checks": [],
            "attribution_only_checks": [], "structural_findings": [], "checks": {},
        }
        for index in range(10)
    ]
    report = LLMExplainerAgent(ExplainerConfig(backend="template")).run(
        make_summary(images)
    )["report"]
    assert "Were the decisions right?" not in report


def test_the_prompt_instructs_the_model_on_the_new_sections() -> None:
    from agents.llm_explainer import PROMPT_TEMPLATE

    prompt = PROMPT_TEMPLATE.format(facts="{}")
    assert "wrongly flagged" in prompt.lower()
    assert "borderline" in prompt.lower()
    assert "worked examples" in prompt.lower()
