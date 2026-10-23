"""Tests for the LLM Explainer Agent and its evaluation harness.

The invariants that make this stage a measurable result rather than an assertion:

1. **The template backend must have a hallucination rate of exactly zero.** It is
   the control arm: every number it emits is read directly from the fact sheet.
   If it ever scores above zero, either the renderer is computing something it
   should not, or the harness is broken - and in both cases every LLM figure
   compared against it becomes meaningless.

2. **The harness must catch fabricated numbers and name them.** A rate with no
   examples is not actionable, and a harness that cannot detect invention cannot
   support any claim about a model.

3. **Legitimate arithmetic must not be counted as hallucination.** A report
   writing "34.8%" for a stored 0.348, or correctly summing two counts, is not
   fabricating. Conflating these with invention would penalise correct reasoning
   and make the template-versus-model comparison uninterpretable.

4. **Factuality alone must not be gameable.** A report that says almost nothing
   scores a perfect hallucination rate, so coverage is measured alongside it and
   the two are always reported together.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from agents.base import SCHEMA_VERSION
from agents.llm_explainer import (
    BACKENDS, ExplainerConfig, LLMExplainerAgent, LLMUnavailableError,
    build_fact_sheet, render_template_report,
)
from agents.report_evaluation import (
    check_completeness, check_factual_consistency, collect_source_values,
    extract_numbers, required_topics_for_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_summary(with_collapse: bool = True, with_evaluation: bool = True) -> dict:
    summary: dict[str, Any] = {
        "dataset": "chestxray14",
        "provenance": {
            "run_id": "sum123", "agent": "dataset_summary",
            "schema_version": SCHEMA_VERSION,
        },
        "decisions": {
            "n_images": 844,
            "by_decision": {"accept": 600, "flag": 100, "reject": 144},
            "by_primary_issue": {"brightness": 95, "blur": 60, "noise": 31},
            "n_with_structural_findings": 12,
        },
        "class_balance_impact": {
            "n_strata": 15, "n_images_joined": 844,
            "became_non_viable": [],
            "already_non_viable_before_rejection": [{"label": "Pneumonia"}],
            "thresholds": {"min_viable_stratum": 10},
        },
        "split_integrity": {
            "by_split": [
                {"split": "test", "n_before": 844, "n_after": 700,
                 "rejection_rate": 0.171}
            ],
            "uneven": False,
        },
        "recommendations": [],
    }
    if with_evaluation:
        summary["decisions"]["evaluation"] = {
            "n_clean": 251, "n_degraded": 593,
            "realised_false_positive_rate": 0.059,
            "recall_on_degraded": 0.753,
            "recall_by_severity": {"1": 0.348, "2": 0.849, "3": 1.0},
        }
    if with_collapse:
        summary["class_balance_impact"]["became_non_viable"] = [
            {"label": "Hernia", "n_before": 3, "n_after": 0,
             "n_rejected": 3, "fraction_lost": 1.0}
        ]
        summary["recommendations"] = [{
            "priority": "high", "type": "stratum_collapse", "label": "Hernia",
            "finding": "Rejecting 3 image(s) reduces the Hernia stratum from 3 to 0.",
            "options": ["Retain the 3 rejected Hernia image(s) with a caveat."],
        }]
    return summary


class _StubHandler(BaseHTTPRequestHandler):
    """Serves a canned response keyed by the requested model name."""

    responses: dict[str, str] = {}

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        text = self.responses.get(body["model"], "")
        payload = json.dumps({
            "response": text, "model": body["model"],
            "eval_count": 200, "total_duration": 3_000_000_000,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def stub_server():
    """A local stand-in for Ollama, so the LLM code path is tested without one."""
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}/api/generate"
    server.shutdown()


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

def test_extract_numbers_finds_decimals_percentages_and_thousands() -> None:
    numbers = extract_numbers(
        "Of 1,234 images, 71.1% were accepted with a rate of 0.059."
    )
    values = [entry.value for entry in numbers]
    assert 1234.0 in values and 71.1 in values and 0.059 in values
    assert any(entry.is_percentage for entry in numbers)


def test_extract_numbers_ignores_digits_inside_identifiers() -> None:
    """resnet18 and ChestX-ray14 are names, not numeric claims."""
    numbers = extract_numbers("We used resnet18 features on ChestX-ray14 data.")
    assert numbers == []


def test_extract_numbers_captures_context_for_diagnosis() -> None:
    numbers = extract_numbers("The recall on degraded images was 0.753 overall.")
    assert "recall" in numbers[0].context.lower()


def test_collect_source_values_walks_nested_structures() -> None:
    values = collect_source_values({"a": {"b": [1.5, {"c": 2.5}]}, "d": 3.5})
    assert {1.5, 2.5, 3.5} <= set(values)


def test_collect_source_values_ignores_booleans() -> None:
    """A True must not make the number 1 verifiable."""
    values = collect_source_values({"uneven": True, "flag": False})
    assert values == {}


# ---------------------------------------------------------------------------
# Factual consistency
# ---------------------------------------------------------------------------

def test_a_fully_grounded_report_scores_zero() -> None:
    source = {"n_images": 844, "n_accepted": 600, "rate": 0.059}
    report = "Of 844 images, 600 were accepted at a rate of 0.059."
    assert check_factual_consistency(report, source).hallucination_rate == 0.0


def test_fabricated_numbers_are_detected_and_named() -> None:
    source = {"n_images": 844, "n_accepted": 600}
    report = "Of 844 images, 512 were accepted with precision 0.887."
    result = check_factual_consistency(report, source)
    assert result.n_unverifiable == 2
    flagged = {entry["value"] for entry in result.unverifiable_examples}
    assert flagged == {512.0, 0.887}


def test_a_percentage_of_a_stored_fraction_is_verified_not_flagged() -> None:
    """0.059 stored, "5.9%" written - the same fact in another form."""
    source = {"rate": 0.059, "recall": 0.753}
    report = "The false-positive rate was 5.9% and recall reached 75.3%."
    assert check_factual_consistency(report, source).n_unverifiable == 0


def test_a_rounded_value_is_verified_not_flagged() -> None:
    source = {"recall": 0.7531234}
    result = check_factual_consistency("Recall was 0.753.", source)
    assert result.n_unverifiable == 0


def test_correct_arithmetic_is_classified_as_derived_not_hallucinated() -> None:
    """100 flagged + 144 rejected = 244 not accepted. True, but not a stored
    field, so it must be credited as derived rather than punished."""
    source = {"n_flagged": 100, "n_rejected": 144}
    result = check_factual_consistency("In total 244 images were not accepted.", source)
    assert result.n_derived == 1
    assert result.n_unverifiable == 0
    assert result.hallucination_rate == 0.0


def test_trivial_values_are_skipped_rather_than_credited() -> None:
    """Finding "1" in the source proves nothing about grounding."""
    source = {"n_images": 844}
    result = check_factual_consistency("There was 1 dataset and 2 splits.", source)
    assert result.n_trivial_skipped == 2
    assert result.n_verified == 0


def test_a_report_with_no_numbers_scores_zero_without_dividing_by_zero() -> None:
    result = check_factual_consistency("Quality was assessed.", {"n": 844})
    assert result.n_numbers == 0
    assert result.hallucination_rate == 0.0


def test_judgements_are_itemised_when_requested() -> None:
    source = {"n_images": 844}
    result = check_factual_consistency(
        "Of 844 images, 512 failed.", source, include_judgements=True
    )
    verdicts = {entry["verdict"] for entry in result.judgements}
    assert verdicts == {"verified", "unverifiable"}


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def test_completeness_detects_a_missing_topic() -> None:
    result = check_completeness(
        "The dataset was assessed and images were accepted.",
        {"decisions": ["accept"], "recall": ["recall", "false positive"]},
    )
    assert result["missing"] == ["recall"]
    assert result["coverage"] == pytest.approx(0.5)


def test_required_topics_adapt_to_the_summary_content() -> None:
    """A report must not be penalised for omitting a collapse that never
    happened."""
    with_collapse = required_topics_for_summary(make_summary(with_collapse=True))
    without = required_topics_for_summary(make_summary(with_collapse=False))
    assert "stratum_collapse" in with_collapse
    assert "stratum_collapse" not in without


def test_required_topics_omit_performance_when_no_ground_truth_exists() -> None:
    topics = required_topics_for_summary(make_summary(with_evaluation=False))
    assert "detection_performance" not in topics


def test_a_terse_report_scores_well_on_factuality_but_badly_on_coverage() -> None:
    """Why both metrics must always be read together: saying nothing is a
    perfect factuality score."""
    summary = make_summary()
    sheet = build_fact_sheet(summary)
    terse = "The dataset chestxray14 contained 844 images."
    factual = check_factual_consistency(terse, sheet)
    completeness = check_completeness(terse, required_topics_for_summary(summary))
    assert factual.hallucination_rate == 0.0
    assert completeness["coverage"] < 0.6


# ---------------------------------------------------------------------------
# Fact sheet
# ---------------------------------------------------------------------------

def test_fact_sheet_precomputes_percentages_so_the_model_never_calculates() -> None:
    sheet = build_fact_sheet(make_summary())
    assert sheet["pct_accepted"] == pytest.approx(71.1, abs=0.1)
    assert sheet["pct_rejected"] == pytest.approx(17.1, abs=0.1)


def test_fact_sheet_omits_sections_absent_from_the_summary() -> None:
    sheet = build_fact_sheet(make_summary(with_evaluation=False))
    assert "evaluation" not in sheet


def test_fact_sheet_handles_a_skipped_class_balance_analysis() -> None:
    summary = make_summary()
    summary["class_balance_impact"] = {"skipped": "no label column"}
    sheet = build_fact_sheet(summary)
    assert "class_balance" not in sheet


def test_fact_sheet_survives_zero_images_without_dividing_by_zero() -> None:
    summary = make_summary()
    summary["decisions"] = {"n_images": 0, "by_decision": {}, "by_primary_issue": {}}
    sheet = build_fact_sheet(summary)
    assert sheet["pct_accepted"] == 0.0


# ---------------------------------------------------------------------------
# Template backend: the control
# ---------------------------------------------------------------------------

def test_template_backend_has_a_hallucination_rate_of_exactly_zero() -> None:
    """The claim the whole comparison rests on."""
    payload = LLMExplainerAgent(ExplainerConfig(backend="template")).run(make_summary())
    factual = payload["evaluation"]["factual_consistency"]
    assert factual["hallucination_rate"] == 0.0, factual["unverifiable_examples"]


def test_template_backend_covers_every_required_topic() -> None:
    payload = LLMExplainerAgent(ExplainerConfig(backend="template")).run(make_summary())
    completeness = payload["evaluation"]["completeness"]
    assert completeness["coverage"] == 1.0, completeness["missing"]


def test_template_report_is_deterministic() -> None:
    summary = make_summary()
    first = render_template_report(build_fact_sheet(summary))
    second = render_template_report(build_fact_sheet(summary))
    assert first == second


def test_template_report_states_the_curation_tension_explicitly() -> None:
    """The Hernia collapse is the substantive finding; a report that lists it
    without explaining why it matters is not doing the job."""
    report = render_template_report(build_fact_sheet(make_summary()))
    assert "Hernia" in report
    assert "rare class" in report.lower() or "judgement" in report.lower()


def test_template_report_always_carries_caveats() -> None:
    report = render_template_report(build_fact_sheet(make_summary()))
    lowered = report.lower()
    assert "synthetic" in lowered
    assert "modalit" in lowered      # thresholds do not transfer
    assert "does not modify" in lowered


def test_template_report_omits_sections_with_no_content() -> None:
    summary = make_summary(with_collapse=False, with_evaluation=False)
    report = render_template_report(build_fact_sheet(summary))
    assert "Detection performance" not in report
    assert "Recommended actions" not in report


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

def test_llm_backend_evaluates_a_grounded_model_as_clean(stub_server) -> None:
    server, url = stub_server
    _StubHandler.responses = {
        "grounded": (
            "# Data quality audit: chestxray14\n\n"
            "Of 844 images, 600 were accepted, 100 flagged and 144 rejected. "
            "The false-positive rate was 0.059 and recall reached 0.753. "
            "The Hernia stratum fell from 3 images to 0. We recommend retaining "
            "them.\n"
        )
    }
    payload = LLMExplainerAgent(
        ExplainerConfig(backend="ollama", model="grounded", url=url)
    ).run(make_summary())
    assert payload["evaluation"]["factual_consistency"]["hallucination_rate"] == 0.0
    assert payload["model"] == "grounded"


def test_llm_backend_catches_a_hallucinating_model(stub_server) -> None:
    server, url = stub_server
    _StubHandler.responses = {
        "loose": (
            "# Data quality audit: chestxray14\n\n"
            "Of 844 images, 512 were accepted. Recall was 0.912, precision 0.887 "
            "and F1 0.899.\n"
        )
    }
    payload = LLMExplainerAgent(
        ExplainerConfig(backend="ollama", model="loose", url=url)
    ).run(make_summary())
    factual = payload["evaluation"]["factual_consistency"]
    assert factual["hallucination_rate"] > 0.7
    flagged = {entry["value"] for entry in factual["unverifiable_examples"]}
    assert {512.0, 0.912, 0.887, 0.899} <= flagged


def test_llm_backend_records_generation_metadata(stub_server) -> None:
    server, url = stub_server
    _StubHandler.responses = {"m": "# Audit\n\nOf 844 images, 600 accepted.\n"}
    payload = LLMExplainerAgent(
        ExplainerConfig(backend="ollama", model="m", url=url)
    ).run(make_summary())
    metadata = payload["generation_metadata"]
    assert metadata["eval_count"] == 200
    assert metadata["total_duration_ns"] == 3_000_000_000
    assert metadata["prompt_chars"] > 0


def test_an_unreachable_model_raises_with_actionable_guidance() -> None:
    agent = LLMExplainerAgent(
        ExplainerConfig(backend="ollama", url="http://127.0.0.1:1/api/generate")
    )
    with pytest.raises(LLMUnavailableError, match="template"):
        agent.run(make_summary())


def test_an_empty_model_response_raises_rather_than_scoring_zero(stub_server) -> None:
    """An empty report would otherwise post a perfect hallucination rate."""
    server, url = stub_server
    _StubHandler.responses = {"silent": "   "}
    agent = LLMExplainerAgent(
        ExplainerConfig(backend="ollama", model="silent", url=url)
    )
    with pytest.raises(LLMUnavailableError, match="empty"):
        agent.run(make_summary())


def test_the_prompt_carries_the_fact_sheet_and_forbids_invention() -> None:
    from agents.llm_explainer import PROMPT_TEMPLATE

    prompt = PROMPT_TEMPLATE.format(facts=json.dumps(build_fact_sheet(make_summary())))
    assert "844" in prompt
    assert "never invent" in prompt.lower()
    assert "do not calculate" in prompt.lower()


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        LLMExplainerAgent(ExplainerConfig(backend="telepathy"))


def test_backends_are_registered() -> None:
    assert set(BACKENDS) == {"template", "ollama"}


# ---------------------------------------------------------------------------
# Contract and output
# ---------------------------------------------------------------------------

def test_output_carries_provenance_linking_to_the_summary_run() -> None:
    payload = LLMExplainerAgent(ExplainerConfig(backend="template")).run(make_summary())
    provenance = payload["provenance"]
    assert provenance["agent"] == "llm_explainer"
    assert provenance["inputs"]["summary_run_id"] == "sum123"
    assert provenance["config"]["backend"] == "template"


def test_provenance_records_the_model_for_an_llm_report(stub_server) -> None:
    """A reported hallucination rate is meaningless without the model tag."""
    server, url = stub_server
    _StubHandler.responses = {"tagged": "# Audit\n\nOf 844 images, 600 accepted.\n"}
    payload = LLMExplainerAgent(
        ExplainerConfig(backend="ollama", model="tagged", url=url)
    ).run(make_summary())
    assert payload["provenance"]["config"]["model"] == "tagged"
    assert payload["provenance"]["inputs"]["generator_model"] == "tagged"


def test_save_report_writes_both_json_and_markdown(tmp_path: Path) -> None:
    from agents.llm_explainer import save_report

    payload = LLMExplainerAgent(ExplainerConfig(backend="template")).run(make_summary())
    json_path = tmp_path / "explainer.json"
    markdown_path = tmp_path / "report.md"
    save_report(payload, json_path, markdown_path)
    assert json.loads(json_path.read_text())["dataset"] == "chestxray14"
    assert markdown_path.read_text().startswith("# Data quality audit")


def test_backend_specs_parse_model_tags_containing_colons() -> None:
    """'ollama:llama3.2:3b' must keep the whole tag, not split at every colon."""
    from scripts.run_explainer import parse_spec

    config = parse_spec("ollama:llama3.2:3b")
    assert config.backend == "ollama"
    assert config.model == "llama3.2:3b"


def test_backend_spec_slug_is_filesystem_safe() -> None:
    from scripts.run_explainer import slug_for

    assert ":" not in slug_for(parse_spec_helper())


def parse_spec_helper() -> ExplainerConfig:
    from scripts.run_explainer import parse_spec

    return parse_spec("ollama:qwen3:8b")


# ---------------------------------------------------------------------------
# Harness calibration: two ways it was measured to be too lenient
# ---------------------------------------------------------------------------

def test_a_fabricated_rate_is_not_verified_against_a_stored_one() -> None:
    """Regression on a real leniency bug. Rounding to zero decimal places
    collapsed distinct rates onto the same integer - round(0.912, 0) and
    round(0.887, 0) are both 1.0 - so fabricated recalls verified against a
    stored severity-3 recall of 1.0. Sub-10 values must be matched at one decimal
    place or finer."""
    source = {"recall_by_severity": {"3": 1.0}}
    result = check_factual_consistency("Recall was 0.912.", source)
    assert result.n_unverifiable == 1


def test_a_fabricated_rate_is_not_credited_as_derived_arithmetic() -> None:
    """The second leniency bug. Pairwise sums of a few dozen values in [0, 1]
    densely cover that interval, so almost any fabricated rate appeared
    derivable. Arithmetic credit is therefore restricted to counts."""
    source = {"a": 0.348, "b": 0.564, "c": 0.059, "d": 0.753}
    result = check_factual_consistency("Precision was 0.907.", source)
    assert result.n_derived == 0
    assert result.n_unverifiable == 1


def test_arithmetic_credit_still_applies_to_counts() -> None:
    """The restriction must not remove legitimate credit: 100 + 144 = 244 is a
    real claim about real counts."""
    source = {"n_flagged": 100, "n_rejected": 144}
    result = check_factual_consistency("In total 244 were not accepted.", source)
    assert result.n_derived == 1
    assert result.n_unverifiable == 0


def test_a_rate_two_decimal_places_short_is_still_verified() -> None:
    """The tightened matching must not reject honest rounding."""
    source = {"recall": 0.753}
    assert check_factual_consistency("Recall was 0.75.", source).n_unverifiable == 0


def test_a_wholly_fabricated_report_scores_near_one() -> None:
    """End-to-end calibration check on the full fact sheet."""
    sheet = build_fact_sheet(make_summary())
    report = (
        "Of 844 images, 512 were accepted. Recall was 0.912, precision 0.887 "
        "and F1 0.899."
    )
    result = check_factual_consistency(report, sheet)
    assert result.hallucination_rate > 0.7


def test_numbers_inside_source_strings_count_as_grounded() -> None:
    """Regression on a bug found on real data. The Summary Agent emits
    pre-formatted findings such as "Edema loses 5 of 19 image(s) (26%)". The
    template copies those strings verbatim, but the value collector walked only
    numeric fields - so the template scored a 0.390 hallucination rate on
    ChestX-ray14, which is impossible by construction."""
    sheet = {
        "n_images": 844,
        "recommendations": [
            {"finding": "Edema loses 5 of 19 image(s) (26%) to rejection."}
        ],
    }
    report = "Of 844 images. Edema loses 5 of 19 image(s) (26%) to rejection."
    assert check_factual_consistency(report, sheet).hallucination_rate == 0.0


def test_string_extraction_does_not_become_a_loophole() -> None:
    """The fix must not credit numbers that appear nowhere, in text or otherwise."""
    sheet = {
        "n_images": 844,
        "recommendations": [{"finding": "Edema loses 5 of 19 image(s)."}],
    }
    result = check_factual_consistency(
        "Recall was 0.912 and precision 0.887.", sheet
    )
    assert result.n_unverifiable == 2
    assert result.hallucination_rate == 1.0


def test_a_percentage_in_source_prose_grounds_a_stored_fraction() -> None:
    sheet = {"recommendations": [{"finding": "26% of Edema was lost."}]}
    assert check_factual_consistency("The fraction lost was 0.26.", sheet).n_unverifiable == 0


def test_template_scores_zero_on_a_summary_with_prose_recommendations() -> None:
    """The exact configuration that exposed the bug: recommendations whose
    findings embed numbers in text."""
    summary = make_summary()
    summary["recommendations"] = [
        {
            "priority": "medium", "type": "resample", "label": "Edema",
            "finding": "Edema loses 5 of 19 image(s) (26%) to rejection.",
            "options": ["Oversample Edema by approximately 1.4x in train."],
        },
        {
            "priority": "medium", "type": "resample", "label": "Emphysema",
            "finding": "Emphysema loses 6 of 19 image(s) (32%) to rejection.",
            "options": ["Oversample Emphysema by approximately 1.5x in train."],
        },
    ]
    payload = LLMExplainerAgent(ExplainerConfig(backend="template")).run(summary)
    factual = payload["evaluation"]["factual_consistency"]
    assert factual["hallucination_rate"] == 0.0, factual["unverifiable_examples"]