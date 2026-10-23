"""The LLM Explainer Agent: turns the Summary Agent's JSON into prose.

**Two backends, and the second is not a fallback.** The ``template`` backend
renders the report deterministically from the summary with no model involved; the
``ollama`` backend asks a local LLM to phrase the same facts. Running both
through the same factual-consistency harness is the ablation that makes this
stage measurable: the template has a hallucination rate of zero by construction,
so it is the control against which any LLM report is judged. A model that scores
worse than the template on factuality and no better on coverage is not adding
value, and that is a finding rather than a failure.

**Slot-filling rather than free generation.** The LLM is given a pre-computed
fact sheet - every number it may use, already extracted and labelled - and asked
to phrase it. It is never asked to compute, compare, or infer. This is the single
most effective control on hallucination available: a model that is handed
"rejected: 144" and told to write a sentence has little reason to invent 512,
whereas one handed raw JSON and asked to summarise has every opportunity.

**Local model, temperature zero.** Reproducibility matters more than fluency for
a dissertation result, and an offline model is the appropriate choice for a
medical-data context. The exact model tag is recorded in the output so a report
can be traced to the model that produced it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from agents.base import Agent, write_agent_output
    from agents.report_evaluation import (
        check_completeness,
        check_factual_consistency,
        required_topics_for_summary,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.base import Agent, write_agent_output  # type: ignore[no-redef]
    from agents.report_evaluation import (  # type: ignore[no-redef]
        check_completeness,
        check_factual_consistency,
        required_topics_for_summary,
    )


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
BACKENDS = ("template", "ollama")


class LLMUnavailableError(RuntimeError):
    """Raised when the local model cannot be reached or does not respond."""


# ---------------------------------------------------------------------------
# Fact sheet: what the model is allowed to say
# ---------------------------------------------------------------------------

def build_fact_sheet(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract exactly the facts a report may assert, pre-computed and labelled.

    Everything the report needs is resolved here, in Python, where it can be
    tested. The model never performs arithmetic - it is handed finished figures.
    Percentages are computed here for the same reason.
    """
    decisions = summary.get("decisions", {})
    by_decision = decisions.get("by_decision", {})
    total = decisions.get("n_images", 0) or 0

    def percentage(count: int) -> float:
        return round(100.0 * count / total, 1) if total else 0.0

    accepted = by_decision.get("accept", 0)
    flagged = by_decision.get("flag", 0)
    rejected = by_decision.get("reject", 0)

    sheet: dict[str, Any] = {
        "dataset": summary.get("dataset", "unknown"),
        "n_images": total,
        "n_accepted": accepted,
        "n_flagged": flagged,
        "n_rejected": rejected,
        "pct_accepted": percentage(accepted),
        "pct_flagged": percentage(flagged),
        "pct_rejected": percentage(rejected),
        "n_structural_findings": decisions.get("n_with_structural_findings", 0),
    }

    issues = decisions.get("by_primary_issue", {})
    sheet["top_issues"] = [
        {"issue": name.replace("_", " "), "count": count}
        for name, count in list(issues.items())[:5]
    ]

    evaluation = decisions.get("evaluation")
    if evaluation:
        sheet["evaluation"] = {
            "n_clean": evaluation.get("n_clean", 0),
            "n_degraded": evaluation.get("n_degraded", 0),
            "false_positive_rate": evaluation.get("realised_false_positive_rate"),
            "recall": evaluation.get("recall_on_degraded"),
            "recall_by_severity": evaluation.get("recall_by_severity", {}),
        }

    correctness = summary.get("decision_correctness", {})
    if "skipped" not in correctness:
        counts = correctness.get("counts", {})
        sheet["correctness"] = {
            "n_correctly_actioned": counts.get("true_positive", 0),
            "n_wrongly_actioned": counts.get("false_positive", 0),
            "n_correctly_accepted": counts.get("true_negative", 0),
            "n_defects_missed": counts.get("false_negative", 0),
            "n_confident_wrong_flags": correctness.get(
                "confident_mistakes", {}
            ).get("false_positive", 0),
            "least_detected": correctness.get("least_detected_defects", [])[:2],
        }

    bands = decisions.get("by_confidence", {})
    if bands:
        sheet["confidence"] = {
            "n_clear": bands.get("clear", 0),
            "n_borderline": bands.get("borderline", 0),
            "n_unscored": bands.get("unscored", 0),
        }

    # A couple of specific images, so the report can show how a decision was
    # reached rather than only how many were reached. Kept small: every number
    # here is one the model could misquote, so the fact sheet stays tight.
    examples = summary.get("worked_examples", {})
    if examples:
        sheet["examples"] = {
            category: [
                {
                    "image_id": entry["image_id"],
                    "decision": entry["decision"],
                    "confidence": entry["confidence_band"],
                    "primary_issue": entry.get("primary_issue"),
                    "was_correct": entry.get("was_correct"),
                    "checks_fired": entry.get("checks_fired", [])[:2],
                }
                for entry in entries[:1]
            ]
            for category, entries in examples.items()
        }

    balance = summary.get("class_balance_impact", {})
    if "skipped" not in balance:
        sheet["class_balance"] = {
            "n_strata": balance.get("n_strata", 0),
            "collapsed": [
                {
                    "label": entry["label"],
                    "n_before": entry["n_before"],
                    "n_after": entry["n_after"],
                }
                for entry in balance.get("became_non_viable", [])
            ],
            "already_below_floor": [
                entry["label"]
                for entry in balance.get("already_non_viable_before_rejection", [])
            ][:8],
            "viability_floor": balance.get("thresholds", {}).get(
                "min_viable_stratum"
            ),
        }

    integrity = summary.get("split_integrity", {})
    if "skipped" not in integrity:
        sheet["split_integrity"] = {
            "uneven": integrity.get("uneven", False),
            "by_split": integrity.get("by_split", []),
        }

    sheet["recommendations"] = [
        {
            "priority": action["priority"],
            "finding": action["finding"],
            "options": action.get("options", []),
        }
        for action in summary.get("recommendations", [])
    ]
    return sheet


# ---------------------------------------------------------------------------
# Template backend: the zero-hallucination control
# ---------------------------------------------------------------------------

def render_template_report(sheet: dict[str, Any]) -> str:
    """A deterministic report built only from fact-sheet values.

    Its hallucination rate is zero by construction, since every number it emits
    is read directly from the sheet. That makes it the control arm of the
    evaluation, not a consolation prize.
    """
    lines: list[str] = []
    dataset = sheet["dataset"]
    lines.append(f"# Data quality audit: {dataset}")
    lines.append("")

    lines.append("## Summary")
    lines.append(
        f"The Data Inspector assessed {sheet['n_images']} images from {dataset}. "
        f"{sheet['n_accepted']} were accepted ({sheet['pct_accepted']}%), "
        f"{sheet['n_flagged']} flagged for review ({sheet['pct_flagged']}%), and "
        f"{sheet['n_rejected']} rejected ({sheet['pct_rejected']}%)."
    )
    if sheet["n_structural_findings"]:
        lines.append(
            f"A further {sheet['n_structural_findings']} images carry structural "
            "findings - resolution or duplicate issues - which are recorded as "
            "curation concerns and do not affect the quality decision."
        )
    lines.append("")

    if sheet["top_issues"]:
        lines.append("## Dominant quality issues")
        for entry in sheet["top_issues"]:
            lines.append(f"- {entry['issue']}: {entry['count']} images")
        lines.append("")

    evaluation = sheet.get("evaluation")
    if evaluation:
        lines.append("## Detection performance")
        if evaluation["false_positive_rate"] is not None:
            lines.append(
                f"On {evaluation['n_clean']} images known to be clean, the "
                f"realised false-positive rate was "
                f"{evaluation['false_positive_rate']:.3f}."
            )
        if evaluation["recall"] is not None:
            lines.append(
                f"On {evaluation['n_degraded']} images with known injected "
                f"defects, recall was {evaluation['recall']:.3f}."
            )
        if evaluation["recall_by_severity"]:
            lines.append("")
            lines.append("Recall by injected severity:")
            for severity, recall in sorted(evaluation["recall_by_severity"].items()):
                lines.append(f"- severity {severity}: {recall:.3f}")
            lines.append("")
            lines.append(
                "Recall rising with severity is expected; the figure for the "
                "mildest level indicates how many subtle defects pass inspection "
                "at this operating point."
            )
        lines.append("")

    correctness = sheet.get("correctness")
    if correctness:
        lines.append("## Were the decisions right?")
        lines.append(
            f"Against known injected defects, {correctness['n_correctly_actioned']} "
            "images were correctly flagged or rejected and "
            f"{correctness['n_correctly_accepted']} correctly accepted. "
            f"{correctness['n_wrongly_actioned']} clean images were wrongly "
            f"actioned, and {correctness['n_defects_missed']} genuinely degraded "
            "images passed inspection."
        )
        if correctness["n_confident_wrong_flags"]:
            lines.append("")
            lines.append(
                f"Of the wrongly actioned images, "
                f"{correctness['n_confident_wrong_flags']} were flagged with high "
                "confidence rather than marginally. A confident mistake is the "
                "more serious kind: it suggests the threshold is wrong for those "
                "images rather than the operating point behaving as configured."
            )
        for entry in correctness["least_detected"]:
            lines.append("")
            lines.append(
                f"The weakest detection was {entry['defect']}, caught in "
                f"{entry['detected']} of {entry['detected'] + entry['missed']} "
                f"cases (recall {entry['recall']:.3f})."
            )
        lines.append("")

    confidence = sheet.get("confidence")
    if confidence:
        lines.append("## How clear-cut were the decisions?")
        lines.append(
            f"{confidence['n_clear']} decisions were clear-cut and "
            f"{confidence['n_borderline']} were borderline, meaning the score sat "
            "close to the threshold that produced the verdict. Borderline cases "
            "are where human review is most worthwhile, since the verdict could "
            "reasonably have gone either way."
        )
        if confidence["n_unscored"]:
            lines.append(
                f"A further {confidence['n_unscored']} images could not be scored "
                "and carry no confidence assessment; these are not the same as "
                "confident acceptances."
            )
        lines.append("")

    examples = sheet.get("examples")
    if examples:
        lines.append("## Worked examples")
        lines.append(
            "The following individual decisions illustrate how the thresholds "
            "behaved in practice."
        )
        titles = {
            "confident_rejection": "A clear rejection",
            "borderline_decision": "A borderline call",
            "false_positive": "A clean image wrongly actioned",
            "missed_defect": "A degraded image that passed",
        }
        for category, entries in examples.items():
            for entry in entries:
                lines.append("")
                lines.append(f"**{titles.get(category, category)}** - "
                             f"`{entry['image_id']}`")
                detail = (
                    f"decision {entry['decision']}, {entry['confidence']} "
                    f"confidence"
                )
                if entry.get("primary_issue"):
                    detail += f", primary issue {entry['primary_issue']}"
                if entry.get("was_correct") is not None:
                    detail += (
                        ", which was correct" if entry["was_correct"]
                        else ", which was incorrect"
                    )
                lines.append(detail + ".")
                for check in entry.get("checks_fired", []):
                    threshold = check.get("flag_threshold")
                    if threshold is None:
                        continue
                    lines.append(
                        f"- {check['check']}: scored {check['score']} against a "
                        f"threshold of {threshold}"
                    )
        lines.append("")

    balance = sheet.get("class_balance")
    if balance:
        lines.append("## Class-balance impact")
        lines.append(
            f"Rejection was assessed against {balance['n_strata']} class strata, "
            f"using a viability floor of {balance['viability_floor']} images."
        )
        if balance["collapsed"]:
            lines.append("")
            lines.append(
                "The following strata fall below the viability floor once "
                "rejected images are removed:"
            )
            for entry in balance["collapsed"]:
                lines.append(
                    f"- {entry['label']}: {entry['n_before']} images before "
                    f"rejection, {entry['n_after']} after"
                )
            lines.append("")
            lines.append(
                "This is the central tension in quality-based curation: removing "
                "degraded images can render a rare class unusable. The choice "
                "between retaining known-degraded images and losing the class "
                "requires a curator's judgement."
            )
        if balance["already_below_floor"]:
            lines.append("")
            lines.append(
                "Already below the floor before any rejection: "
                + ", ".join(balance["already_below_floor"])
                + "."
            )
        lines.append("")

    integrity = sheet.get("split_integrity")
    if integrity and integrity["uneven"]:
        lines.append("## Split integrity")
        lines.append(
            "Rejection rates differ materially across splits, which changes the "
            "effective train, validation and test proportions:"
        )
        for entry in integrity["by_split"]:
            lines.append(
                f"- {entry['split']}: {entry['n_before']} before, "
                f"{entry['n_after']} after (rejection rate "
                f"{entry['rejection_rate']:.3f})"
            )
        lines.append("")

    if sheet["recommendations"]:
        lines.append("## Recommended actions")
        for action in sheet["recommendations"]:
            lines.append(f"**{action['priority'].upper()}** - {action['finding']}")
            for option in action["options"]:
                lines.append(f"  - {option}")
            lines.append("")

    lines.append("## Caveats")
    lines.append(
        "Decision thresholds were derived from clean validation images at a "
        "stated false-positive rate and apply to this dataset only; thresholds "
        "do not transfer across imaging modalities. Ground-truth figures reflect "
        "synthetically injected defects, which may not represent the "
        "distribution of naturally occurring artefacts. This agent recommends "
        "actions but does not modify the dataset."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are writing a data quality audit report for a medical \
imaging dataset. Write clear, professional prose for a technical reader.

CRITICAL RULES:
1. Use ONLY the numbers given in the FACTS below. Never invent a number.
2. Do not calculate, estimate, or infer any new number.
3. If a fact is absent, do not discuss that topic at all.
4. Write in plain prose with short markdown headings. No bullet lists of raw \
numbers.
5. Be concise: aim for 300-450 words.

Cover these sections, in order:
- Summary: how many images were assessed and what was decided
- Dominant quality issues
- Detection performance (if evaluation facts are present)
- Whether the decisions were right (if correctness facts are present): how many were correct, how many clean images were wrongly flagged, how many defects were missed, and note that a CONFIDENT wrong flag is more serious than a borderline one
- How clear-cut the decisions were (if confidence facts are present): explain that borderline cases are where human review is most worthwhile
- Worked examples (if present): describe one or two individual decisions and the check scores behind them
- Class-balance impact (if present) - explain why losing a rare class matters
- Recommended actions

FACTS (the only numbers you may use):
{facts}

Write the report now. Begin with a markdown H1 heading naming the dataset.
"""


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    url: str = OLLAMA_URL,
    timeout: int = 180,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call a local Ollama model, returning its response and metadata.

    ``temperature=0`` for reproducibility: a dissertation reporting a
    hallucination rate needs that rate to be stable across runs.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise LLMUnavailableError(
            f"could not reach the local model at {url}: {error}. Start it with "
            "'ollama serve', or use --backend template."
        ) from error
    except json.JSONDecodeError as error:
        raise LLMUnavailableError(
            f"the model at {url} returned a response that is not JSON: {error}"
        ) from error

    text = payload.get("response", "")
    if not text.strip():
        raise LLMUnavailableError(
            f"the model {model!r} returned an empty response; check that the "
            "model is pulled ('ollama list')"
        )
    return {
        "text": text,
        "model": payload.get("model", model),
        "eval_count": payload.get("eval_count"),
        "total_duration_ns": payload.get("total_duration"),
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class ExplainerConfig:
    backend: str = "template"
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    url: str = OLLAMA_URL
    timeout: int = 180

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "temperature": self.temperature,
        }
        if self.backend == "ollama":
            payload.update({"model": self.model, "url": self.url})
        return payload


class LLMExplainerAgent(Agent):
    """Generates a human-readable audit report and evaluates it."""

    name = "llm_explainer"

    def __init__(self, config: ExplainerConfig | None = None) -> None:
        self.explainer_config = config or ExplainerConfig()
        if self.explainer_config.backend not in BACKENDS:
            raise ValueError(
                f"backend must be one of {BACKENDS}, got "
                f"{self.explainer_config.backend!r}"
            )
        super().__init__(self.explainer_config.to_dict())

    def generate(self, sheet: dict[str, Any]) -> dict[str, Any]:
        if self.explainer_config.backend == "template":
            return {"text": render_template_report(sheet), "model": "template"}
        prompt = PROMPT_TEMPLATE.format(facts=json.dumps(sheet, indent=2))
        result = call_ollama(
            prompt,
            model=self.explainer_config.model,
            url=self.explainer_config.url,
            timeout=self.explainer_config.timeout,
            temperature=self.explainer_config.temperature,
        )
        result["prompt_chars"] = len(prompt)
        return result

    def run(self, summary: dict[str, Any]) -> dict[str, Any]:
        sheet = build_fact_sheet(summary)
        generation = self.generate(sheet)
        report = generation["text"]

        # The report is checked against the FACT SHEET rather than the whole
        # summary: the sheet is what the generator was given, so checking against
        # anything else would credit a lucky guess at a figure the model never
        # saw.
        factual = check_factual_consistency(report, sheet, include_judgements=True)
        completeness = check_completeness(report, required_topics_for_summary(summary))

        provenance = self.provenance({
            "dataset": summary.get("dataset"),
            "summary_run_id": summary.get("provenance", {}).get("run_id"),
            "generator_model": generation.get("model"),
            "n_report_chars": len(report),
        })
        return {
            "provenance": provenance.to_dict(),
            "dataset": summary.get("dataset"),
            "backend": self.explainer_config.backend,
            "model": generation.get("model"),
            "report": report,
            "fact_sheet": sheet,
            "evaluation": {
                "factual_consistency": factual.to_dict(),
                "completeness": completeness,
                "judgements": factual.judgements,
            },
            "generation_metadata": {
                key: value
                for key, value in generation.items()
                if key not in ("text",)
            },
        }


def save_report(payload: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Write the structured payload and the report itself side by side."""
    write_agent_output(payload, json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(payload["report"] + "\n")