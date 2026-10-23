"""Tests for the Data Inspector Agent, its configuration, and the agent contract.

Four invariants these tests protect:

1. **A derived threshold must deliver its stated false-positive rate on unseen
   data.** If it does not, every number the agent reports is misleading.

2. **Per-check false-positive rates compound.** Nine checks each targeting 5%
   give an image-level rate of 1 - 0.95^9, roughly 37%; this was measured at
   28.6% on a demo dataset before the policy was added. The default
   composite_only policy must deliver the configured rate, and the per_check
   policy must visibly not.

3. **Thresholds must never cross modalities silently.** Stage 5 measured an
   eighteen-fold false-positive inflation when a ChestX-ray14 threshold was
   applied to ISIC.

4. **The output schema is a contract.** The Summary Agent reads it directly, so
   a major-version mismatch must fail loudly rather than be misread.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.base import (
    SCHEMA_VERSION, SchemaMismatchError, read_agent_output,
    require_compatible_schema, write_agent_output,
)
from agents.data_inspector import DataInspectorAgent, derive_threshold, orient
from agents.inspector_config import (
    DECISIONS, FPR_POLICIES, CheckConfig, InspectorConfig, default_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_frame(n_clean: int, n_degraded: int, seed: int, n_metrics: int = 8,
               split: str = "x") -> pd.DataFrame:
    """Clean and degraded rows, separable on every metric and on the composite."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_clean):
        rows.append({
            "image_id": f"clean{index}", "split": split, "defect_type": "clean",
            "severity_level": "0",
            **{f"m{i}": rng.normal(0, 1) for i in range(1, n_metrics + 1)},
            "fused_logistic": rng.normal(0.2, 0.08),
        })
    for index in range(n_degraded):
        rows.append({
            "image_id": f"bad{index}", "split": split,
            "defect_type": "gaussian_blur", "severity_level": "2",
            **{f"m{i}": rng.normal(2.5, 1) for i in range(1, n_metrics + 1)},
            "fused_logistic": rng.normal(0.85, 0.08),
        })
    return pd.DataFrame(rows)


def make_config(policy: str = "composite_only", n_checks: int = 8) -> InspectorConfig:
    config = InspectorConfig()
    config.fpr_policy = policy
    config.checks = [
        CheckConfig(name=f"chk{i}", score_column=f"m{i}", higher_is_better=False,
                    flag_fpr=0.05, reject_fpr=0.01)
        for i in range(1, n_checks + 1)
    ]
    config.overall_check = CheckConfig(
        name="overall_quality", score_column="fused_logistic",
        higher_is_better=False, flag_fpr=0.05, reject_fpr=0.01,
    )
    config.enable_resolution_check = False
    config.enable_duplicate_check = False
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Threshold derivation
# ---------------------------------------------------------------------------

def test_derive_threshold_flags_the_requested_fraction() -> None:
    clean = np.linspace(0.0, 1.0, 2000)
    threshold = derive_threshold(clean, 0.05)
    assert float(np.mean(clean > threshold)) == pytest.approx(0.05, abs=0.01)


def test_derive_threshold_returns_nan_on_empty_input() -> None:
    assert np.isnan(derive_threshold(np.array([]), 0.05))


def test_orientation_flips_only_when_higher_is_better() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(orient(values, True), -values)
    assert np.allclose(orient(values, False), values)


def test_derived_threshold_delivers_its_target_fpr_on_unseen_rows() -> None:
    """The core guarantee: a threshold fitted on validation must hold on test."""
    agent = DataInspectorAgent(make_config())
    validation = make_frame(600, 300, 1)
    test = make_frame(600, 300, 2)
    payload = agent.run("demo", validation, test)

    clean = [i for i in payload["images"] if i["ground_truth_defect_type"] == "clean"]
    realised = sum(1 for i in clean if i["decision"] != "accept") / len(clean)
    assert realised == pytest.approx(0.05, abs=0.03), (
        f"realised FPR {realised:.3f} is far from the 0.05 target"
    )


def test_thresholds_record_the_clean_sample_size_behind_them() -> None:
    agent = DataInspectorAgent(make_config())
    validation = make_frame(120, 60, 3)
    agent.derive_thresholds(validation, "demo")
    for entry in agent.thresholds.values():
        assert entry["n_clean_validation_rows"] == 120


def test_low_clean_sample_size_is_flagged_in_the_threshold_record() -> None:
    """A 1% threshold from 20 rows is barely an estimate; say so."""
    agent = DataInspectorAgent(make_config())
    agent.derive_thresholds(make_frame(20, 10, 4), "demo")
    assert all(
        "low_sample_warning" in entry for entry in agent.thresholds.values()
    )


def test_deriving_thresholds_without_clean_rows_is_an_error() -> None:
    agent = DataInspectorAgent(make_config())
    degraded_only = make_frame(0, 50, 5)
    with pytest.raises(ValueError, match="no clean validation rows"):
        agent.derive_thresholds(degraded_only, "demo")


# ---------------------------------------------------------------------------
# The multiple-comparison problem
# ---------------------------------------------------------------------------

def test_per_check_policy_visibly_inflates_the_image_level_fpr() -> None:
    """Regression on a real design flaw. Nine checks each at 5% compound to
    1 - 0.95^9 = 0.37. This policy is honest about doing that; the test exists
    so the inflation cannot be mistaken for a bug later."""
    agent = DataInspectorAgent(make_config("per_check"))
    payload = agent.run("demo", make_frame(600, 300, 1), make_frame(600, 300, 2))
    clean = [i for i in payload["images"] if i["ground_truth_defect_type"] == "clean"]
    realised = sum(1 for i in clean if i["decision"] != "accept") / len(clean)
    assert realised > 0.20, "per_check should inflate well above the nominal rate"


def test_composite_only_policy_delivers_the_configured_rate() -> None:
    agent = DataInspectorAgent(make_config("composite_only"))
    payload = agent.run("demo", make_frame(600, 300, 1), make_frame(600, 300, 2))
    clean = [i for i in payload["images"] if i["ground_truth_defect_type"] == "clean"]
    realised = sum(1 for i in clean if i["decision"] != "accept") / len(clean)
    assert realised == pytest.approx(0.05, abs=0.03)


def test_bonferroni_policy_controls_the_union_rate() -> None:
    agent = DataInspectorAgent(make_config("bonferroni"))
    payload = agent.run("demo", make_frame(600, 300, 1), make_frame(600, 300, 2))
    clean = [i for i in payload["images"] if i["ground_truth_defect_type"] == "clean"]
    realised = sum(1 for i in clean if i["decision"] != "accept") / len(clean)
    assert realised < 0.15, "bonferroni should keep the union near the nominal rate"


def test_bonferroni_divides_per_check_targets_but_not_the_composite() -> None:
    config = make_config("bonferroni")
    composite_fprs = config.effective_fprs(config.overall_check)
    assert composite_fprs == (0.05, 0.01), "the composite is a single test"
    check_fprs = config.effective_fprs(config.checks[0])
    assert check_fprs[0] == pytest.approx(0.05 / len(config.enabled_checks()))


def test_composite_only_still_reports_per_check_attribution() -> None:
    """Suppressing a check's vote must not suppress its explanation - the
    attribution is what makes a decision reviewable."""
    agent = DataInspectorAgent(make_config("composite_only"))
    payload = agent.run("demo", make_frame(400, 200, 1), make_frame(400, 200, 2))
    degraded = [
        i for i in payload["images"] if i["ground_truth_defect_type"] != "clean"
    ]
    assert any(i["attribution_only_checks"] for i in degraded), (
        "per-defect checks should still fire and be reported"
    )
    assert any(i["primary_issue"] for i in degraded)


def test_every_check_records_whether_it_can_decide() -> None:
    agent = DataInspectorAgent(make_config("composite_only"))
    agent.derive_thresholds(make_frame(200, 100, 1), "demo")
    assert agent.thresholds["overall_quality"]["can_decide"] is True
    assert agent.thresholds["chk1"]["can_decide"] is False


# ---------------------------------------------------------------------------
# Cross-modality guard
# ---------------------------------------------------------------------------

def test_thresholds_refuse_to_cross_modalities_by_default() -> None:
    """Stage 5: a ChestX-ray14 threshold produced 91.2% FPR on ISIC."""
    agent = DataInspectorAgent(make_config())
    frame = make_frame(200, 100, 1)
    agent.run("chestxray14", frame, frame)
    with pytest.raises(ValueError, match="eighteen-fold"):
        agent.run("isic", frame, frame)


def test_cross_modality_reuse_is_possible_when_explicitly_allowed() -> None:
    config = make_config()
    config.allow_cross_modality_thresholds = True
    agent = DataInspectorAgent(config)
    frame = make_frame(200, 100, 1)
    agent.run("chestxray14", frame, frame)
    payload = agent.run("isic", frame, frame)   # must not raise
    assert payload["dataset"] == "isic"
    assert payload["provenance"]["inputs"]["threshold_source_dataset"] == "chestxray14"


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def test_decision_is_always_one_of_the_three_defined_values() -> None:
    agent = DataInspectorAgent(make_config())
    payload = agent.run("demo", make_frame(200, 100, 1), make_frame(200, 100, 2))
    assert {i["decision"] for i in payload["images"]} <= set(DECISIONS)


def test_decision_counts_match_the_per_image_decisions() -> None:
    agent = DataInspectorAgent(make_config())
    payload = agent.run("demo", make_frame(200, 100, 1), make_frame(200, 100, 2))
    recounted = {d: 0 for d in DECISIONS}
    for image in payload["images"]:
        recounted[image["decision"]] += 1
    assert payload["decision_counts"] == recounted


def test_every_check_reports_its_threshold_beside_its_score() -> None:
    """A decision must be auditable without re-running the agent."""
    agent = DataInspectorAgent(make_config("per_check"))
    payload = agent.run("demo", make_frame(200, 100, 1), make_frame(200, 100, 2))
    for image in payload["images"]:
        for name, result in image["checks"].items():
            if result.get("scoreable") and result.get("score") is not None:
                assert "flag_threshold" in result, name
                assert "reject_threshold" in result, name


def test_a_non_finite_score_is_not_scoreable_rather_than_passing() -> None:
    agent = DataInspectorAgent(make_config())
    validation = make_frame(200, 100, 1)
    test = make_frame(20, 10, 2)
    test.loc[test.index[0], "fused_logistic"] = np.nan
    payload = agent.run("demo", validation, test)
    first = payload["images"][0]["checks"]["overall_quality"]
    assert first["scoreable"] is False
    assert first["flagged"] is False
    assert first["score"] is None


def test_rejection_requires_the_composite_not_a_single_specialist() -> None:
    """Stage 2 documented that individual metrics have confounds - laplacian
    variance rises under noise - so one specialist firing must not discard an
    image on its own."""
    config = make_config("per_check")
    agent = DataInspectorAgent(config)
    validation = make_frame(400, 200, 1)
    agent.derive_thresholds(validation, "demo")

    row = pd.Series({
        "image_id": "x", "split": "test", "defect_type": "clean",
        "severity_level": "0",
        **{f"m{i}": 0.0 for i in range(1, 9)},
        "fused_logistic": 0.2,                    # composite says fine
    })
    row["m1"] = 99.0                              # one specialist screams
    result = agent.inspect_image(row, None, None)
    assert result["decision"] == "flag", "a lone specialist should flag, not reject"


def test_ground_truth_is_carried_through_for_evaluation_only() -> None:
    agent = DataInspectorAgent(make_config())
    payload = agent.run("demo", make_frame(100, 50, 1), make_frame(100, 50, 2))
    assert all("ground_truth_defect_type" in i for i in payload["images"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_default_config_is_valid_and_covers_the_expected_checks() -> None:
    config = default_config()
    names = {check.name for check in config.checks}
    assert {"blur", "noise", "contrast", "brightness", "occlusion"} <= names
    assert config.overall_check.score_column == "fused_logistic"


def test_config_round_trips_through_json(tmp_path: Path) -> None:
    config = default_config()
    path = tmp_path / "config.json"
    config.save(path)
    assert InspectorConfig.load(path).to_dict() == config.to_dict()


def test_config_rejects_reject_fpr_looser_than_flag_fpr() -> None:
    with pytest.raises(ValueError, match="stricter standard"):
        InspectorConfig.from_dict({
            "checks": [{"name": "a", "score_column": "x",
                        "higher_is_better": True, "flag_fpr": 0.01,
                        "reject_fpr": 0.05}],
        })


def test_config_rejects_duplicate_check_names() -> None:
    with pytest.raises(ValueError, match="duplicate check name"):
        InspectorConfig.from_dict({
            "checks": [
                {"name": "a", "score_column": "x", "higher_is_better": True},
                {"name": "a", "score_column": "y", "higher_is_better": True},
            ],
        })


def test_config_rejects_an_unknown_fpr_policy() -> None:
    with pytest.raises(ValueError, match="fpr_policy"):
        InspectorConfig.from_dict({"fpr_policy": "wishful_thinking"})


def test_config_rejects_an_unknown_structural_decision() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        InspectorConfig.from_dict({"duplicate_decision": "incinerate"})


def test_disabling_a_check_removes_it_from_the_decision() -> None:
    config = make_config("per_check")
    config.checks[0].enabled = False
    agent = DataInspectorAgent(config)
    agent.derive_thresholds(make_frame(200, 100, 1), "demo")
    assert "chk1" not in agent.thresholds


def test_all_fpr_policies_are_accepted_by_validation() -> None:
    for policy in FPR_POLICIES:
        config = make_config(policy)
        config.validate()   # must not raise


# ---------------------------------------------------------------------------
# The agent output contract
# ---------------------------------------------------------------------------

def test_output_carries_full_provenance() -> None:
    agent = DataInspectorAgent(make_config())
    payload = agent.run("demo", make_frame(100, 50, 1), make_frame(100, 50, 2))
    provenance = payload["provenance"]
    assert provenance["agent"] == "data_inspector"
    assert provenance["schema_version"] == SCHEMA_VERSION
    assert provenance["run_id"] and provenance["timestamp"]
    # The full policy must be recoverable from the output alone.
    assert provenance["config"]["fpr_policy"] == "composite_only"
    assert "numpy" in provenance["libraries"]


def test_schema_check_rejects_a_foreign_major_version() -> None:
    with pytest.raises(SchemaMismatchError, match="major version"):
        require_compatible_schema(
            {"provenance": {"schema_version": "99.0"}}, "summary"
        )


def test_schema_check_rejects_a_payload_with_no_provenance() -> None:
    with pytest.raises(SchemaMismatchError, match="no provenance"):
        require_compatible_schema({"images": []}, "summary")


def test_schema_check_accepts_a_differing_minor_version() -> None:
    major = SCHEMA_VERSION.split(".")[0]
    require_compatible_schema(
        {"provenance": {"schema_version": f"{major}.99"}}, "summary"
    )   # must not raise


def test_output_round_trips_to_disk_and_back(tmp_path: Path) -> None:
    agent = DataInspectorAgent(make_config())
    payload = agent.run("demo", make_frame(100, 50, 1), make_frame(100, 50, 2))
    path = tmp_path / "inspection.json"
    write_agent_output(payload, path)
    restored = read_agent_output(path, "summary")
    assert restored["decision_counts"] == payload["decision_counts"]


def test_reading_a_missing_agent_output_names_the_upstream_agent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="upstream agent"):
        read_agent_output(tmp_path / "absent.json", "summary")


def test_reproducible_for_a_fixed_seed() -> None:
    validation, test = make_frame(200, 100, 1), make_frame(200, 100, 2)
    first = DataInspectorAgent(make_config()).run("demo", validation, test)
    second = DataInspectorAgent(make_config()).run("demo", validation, test)
    assert first["thresholds"] == second["thresholds"]
    assert [i["decision"] for i in first["images"]] == [
        i["decision"] for i in second["images"]
    ]


# ---------------------------------------------------------------------------
# Structural findings versus the quality decision
# ---------------------------------------------------------------------------

def _resolution_frame(image_ids: list[str], flagged_fraction: float) -> pd.DataFrame:
    """Resolution findings on a fixed fraction of images."""
    n_flagged = int(len(image_ids) * flagged_fraction)
    flags = [index < n_flagged for index in range(len(image_ids))]
    return pd.DataFrame({
        "image_id": image_ids,
        "flagged": flags,
        "provenance_missing": [False] * len(image_ids),
        "below_min_side": [False] * len(image_ids),
        "extreme_aspect_ratio": flags,
        "original_width": [1024] * len(image_ids),
        "original_height": [1024] * len(image_ids),
    })


def test_structural_findings_do_not_corrupt_the_quality_fpr() -> None:
    """Regression on a real bug. ISIC carries a resolution finding on 350 of
    1,800 rows - mostly extreme source aspect ratio - and letting those flag
    drove the realised FPR to 0.350 against a 0.050 target. A source property is
    not a quality defect, so it must not count against a quality error rate."""
    config = make_config()
    assert config.structural_findings_affect_decision is False, "must be the default"
    config.enable_resolution_check = True
    agent = DataInspectorAgent(config)
    validation = make_frame(400, 200, 1)
    test = make_frame(400, 0, 2)          # clean only, so any flag is a false positive
    resolution = _resolution_frame(list(test["image_id"]), flagged_fraction=0.30)
    payload = agent.run("demo", validation, test, resolution=resolution)

    realised = sum(
        1 for image in payload["images"] if image["decision"] != "accept"
    ) / len(payload["images"])
    assert realised == pytest.approx(0.05, abs=0.03), (
        f"structural findings leaked into the quality FPR: {realised:.3f}"
    )


def test_structural_findings_are_still_reported_when_they_do_not_decide() -> None:
    """Suppressing their vote must not suppress their visibility - the Summary
    Agent has to act on them."""
    config = make_config()
    config.enable_resolution_check = True
    agent = DataInspectorAgent(config)
    validation = make_frame(400, 200, 1)
    test = make_frame(300, 0, 2)
    resolution = _resolution_frame(list(test["image_id"]), flagged_fraction=0.30)
    payload = agent.run("demo", validation, test, resolution=resolution)

    reported = [i for i in payload["images"] if i["structural_findings"]]
    assert len(reported) == pytest.approx(90, abs=2)
    assert all("resolution" in i["structural_findings"] for i in reported)
    assert all(
        i["structural_findings_affect_decision"] is False for i in payload["images"]
    )


def test_structural_findings_can_be_made_to_decide_when_configured() -> None:
    config = make_config()
    config.structural_findings_affect_decision = True
    config.enable_resolution_check = True
    agent = DataInspectorAgent(config)
    validation = make_frame(400, 200, 1)
    test = make_frame(300, 0, 2)
    resolution = _resolution_frame(list(test["image_id"]), flagged_fraction=0.30)
    payload = agent.run("demo", validation, test, resolution=resolution)

    realised = sum(
        1 for image in payload["images"] if image["decision"] != "accept"
    ) / len(payload["images"])
    assert realised > 0.25, "with structural findings deciding, the rate must rise"


def test_a_duplicate_is_recorded_without_rejecting_the_image() -> None:
    """A duplicate is a curation issue, not a damaged image."""
    config = make_config()
    config.enable_duplicate_check = True
    agent = DataInspectorAgent(config)
    validation = make_frame(400, 200, 1)
    test = make_frame(50, 0, 2)
    duplicate_map = {test["image_id"].iloc[0]: test["image_id"].iloc[1]}
    payload = agent.run("demo", validation, test, duplicate_map=duplicate_map)

    first = payload["images"][0]
    assert first["checks"]["duplicate"]["flagged"] is True
    assert first["checks"]["duplicate"]["duplicate_of"] == test["image_id"].iloc[1]
    assert first["decision"] != "reject"