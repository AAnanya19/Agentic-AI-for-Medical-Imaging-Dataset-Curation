"""Factual-consistency evaluation for generated audit reports.

**Why this module exists before the generator.** "The reports read well" is not a
result. The only defensible claim available here is a measured one: extract every
number a report asserts, check each against the source data, and report a
hallucination rate. That is objective, automatable, and reproducible - unlike a
readability judgement, which on a single-rater Masters project carries very
little weight.

Three verdicts per extracted number, not two. A strict match-or-hallucination
split would be dishonest, because a model that writes "roughly a third" for 0.348
or correctly sums two counts is not fabricating anything:

  * ``verified``     - the value appears in the source data, exactly or in an
                       accepted equivalent form (rounded, or a percentage of a
                       source fraction).
  * ``derived``      - the value is a correct arithmetic consequence of source
                       values (a sum or difference of two of them). Reported
                       separately because it is legitimate but not directly
                       checkable against a single field.
  * ``unverifiable`` - the value does not correspond to anything in the source.
                       This is what the hallucination rate counts.

Only ``unverifiable`` counts against the model. Conflating ``derived`` with
hallucination would penalise correct reasoning and make the metric useless for
comparing a template against an LLM, which is the point of the exercise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable


# Numbers written as digits, with optional thousands separators, decimals, and a
# trailing percent sign. Deliberately does not match numbers inside words
# (identifiers like "resnet18" or "ChestX-ray14") - those are names, not claims.
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(%?)(?![\w])"
)

# Values so common that finding them in the source proves nothing. A report
# saying "0" or "1" is not evidence of grounding, so these are excluded from the
# rate in both directions rather than inflating the verified count.
TRIVIAL_VALUES = {0.0, 1.0, 2.0, 3.0, 100.0}


@dataclass
class ExtractedNumber:
    """One numeric claim found in a generated report."""

    raw: str
    value: float
    is_percentage: bool
    context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "value": self.value,
            "is_percentage": self.is_percentage,
            "context": self.context,
        }


@dataclass
class FactCheckResult:
    """Per-report factual consistency, with every judgement itemised."""

    n_numbers: int
    n_verified: int
    n_derived: int
    n_unverifiable: int
    n_trivial_skipped: int
    hallucination_rate: float
    unverifiable_examples: list[dict[str, Any]] = field(default_factory=list)
    judgements: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_numbers": self.n_numbers,
            "n_verified": self.n_verified,
            "n_derived": self.n_derived,
            "n_unverifiable": self.n_unverifiable,
            "n_trivial_skipped": self.n_trivial_skipped,
            "hallucination_rate": self.hallucination_rate,
            "unverifiable_examples": self.unverifiable_examples,
        }


def extract_numbers(text: str, context_window: int = 60) -> list[ExtractedNumber]:
    """Every numeric claim in a report, with surrounding text for diagnosis.

    The context is kept so an unverifiable number can be inspected rather than
    merely counted - a hallucination rate with no examples is not actionable.
    """
    found: list[ExtractedNumber] = []
    for match in NUMBER_PATTERN.finditer(text):
        digits, percent = match.group(1), match.group(2)
        try:
            value = float(digits.replace(",", ""))
        except ValueError:  # pragma: no cover - the pattern guarantees a number
            continue
        start = max(0, match.start() - context_window)
        end = min(len(text), match.end() + context_window)
        found.append(ExtractedNumber(
            raw=match.group(0).strip(),
            value=value,
            is_percentage=bool(percent),
            context=text[start:end].replace("\n", " ").strip(),
        ))
    return found


def collect_source_values(payload: Any, path: str = "") -> dict[float, list[str]]:
    """Every numeric value in the source, mapped to where it came from.

    Walks the whole nested structure rather than a hand-listed set of fields, so
    a report citing any figure the summary contains is credited - and adding a
    field to the summary never silently turns a true statement into a
    hallucination.
    """
    values: dict[float, list[str]] = {}

    def record(value: float, where: str) -> None:
        values.setdefault(round(float(value), 6), []).append(where)

    def walk(node: Any, where: str) -> None:
        if isinstance(node, bool):
            return  # bools are not numeric claims
        if isinstance(node, (int, float)):
            record(node, where)
            return
        if isinstance(node, str):
            # **Numbers inside source strings count as source values.** The
            # Summary Agent emits pre-formatted findings such as "Edema loses 5
            # of 19 image(s) (26%)", and a report quoting that verbatim is
            # perfectly grounded. Walking only numeric fields made the template
            # backend - which copies those strings directly - score a 0.390
            # hallucination rate on real data, which is impossible by
            # construction and revealed the gap.
            for number in NUMBER_PATTERN.finditer(node):
                try:
                    value = float(number.group(1).replace(",", ""))
                except ValueError:  # pragma: no cover
                    continue
                record(value, f"{where} (in text)")
                if number.group(2):
                    # "26%" in prose also grounds a stored fraction of 0.26.
                    record(value / 100.0, f"{where} (percent in text)")
            return
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{where}.{key}" if where else str(key))
            return
        if isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                walk(child, f"{where}[{index}]")

    walk(payload, path)
    return values


def _equivalent_forms(value: float) -> set[float]:
    """Accepted alternative renderings of one source value.

    A fraction may legitimately be written as a percentage, and a long decimal
    may be rounded for readability. Accepting these is not leniency - rejecting
    them would measure formatting rather than factuality.

    **Rounding to zero decimal places is deliberately excluded for values below
    10.** It collapses distinct rates onto the same integer: round(0.912, 0) and
    round(0.887, 0) are both 1.0, so a fabricated recall of 0.912 verified
    against a stored severity-3 recall of 1.0. Sub-10 values are therefore
    matched only at one decimal place or finer, which still accepts "0.75" for a
    stored 0.753 while rejecting "0.9" for a stored 1.0.
    """
    forms = {round(value, 6)}
    forms.add(round(value * 100, 6))       # 0.348 -> 34.8
    forms.add(round(value / 100, 6))       # 34.8  -> 0.348
    decimal_places = (1, 2, 3) if abs(value) < 10.0 else (0, 1, 2, 3)
    for places in decimal_places:
        forms.add(round(value, places))
    for places in (0, 1, 2, 3):
        # A percentage rendering is a value at least 100x larger, so rounding it
        # to a whole number does not collapse distinct rates in the same way.
        forms.add(round(value * 100, places))
    return forms


def _tolerance_for(value: float, relative_tolerance: float) -> float:
    """Absolute tolerance permitted when matching a value.

    A purely relative tolerance is wrong for small numbers. With a 1% relative
    tolerance floored at 1.0, anything within 0.01 of a stored 1.0 was accepted -
    so a fabricated recall of 0.912 verified against a stored severity-3 recall
    of 1.0. Rates and probabilities need a tolerance proportional to their own
    magnitude, and counts need one proportional to theirs, so the floor is
    removed and small values are matched tightly.
    """
    magnitude = abs(value)
    if magnitude < 10.0:
        # Rates, fractions and small counts: allow only rounding-level slack, so
        # 0.753 matches 0.7531 but not 0.76.
        return max(5e-4, magnitude * relative_tolerance)
    return magnitude * relative_tolerance


def _matches_source(
    number: ExtractedNumber, source_values: dict[float, list[str]], tolerance: float
) -> str | None:
    """The source path a number matches, or None."""
    candidates = [number.value]
    if number.is_percentage:
        # "34.8%" may correspond to a stored 0.348 or a stored 34.8.
        candidates.append(number.value / 100.0)

    for candidate in candidates:
        for source_value, paths in source_values.items():
            if source_value in _equivalent_forms(candidate) or candidate in (
                _equivalent_forms(source_value)
            ):
                return paths[0]
            if abs(source_value - candidate) <= _tolerance_for(source_value, tolerance):
                return paths[0]
    return None


# Arithmetic credit is given only to values at or above this magnitude.
#
# **Why derivation cannot be credited for rates.** A summary contains dozens of
# numbers, and pairwise sums and differences of a few dozen values in [0, 1]
# densely cover that interval - so almost any fabricated rate appears
# "derivable". Measured directly: fabricated recalls of 0.912 and 0.899 were both
# credited as sums of unrelated fact-sheet values. Counts are different: their
# magnitudes are spread widely enough that hitting one by coincidence is
# genuinely unlikely, and "244 images were not accepted" from 100 + 144 is a
# claim worth crediting.
MIN_DERIVABLE_MAGNITUDE = 10.0


def _is_derived(
    number: ExtractedNumber, source_values: dict[float, list[str]], tolerance: float
) -> str | None:
    """Whether a number is the sum or difference of two source COUNTS.

    Restricted to values at or above MIN_DERIVABLE_MAGNITUDE for the reason
    documented there. Checked against a bounded sample because the pairwise space
    grows quadratically.
    """
    if abs(number.value) < MIN_DERIVABLE_MAGNITUDE or number.is_percentage:
        return None
    sample = [
        value for value in sorted(source_values)
        if abs(value) >= MIN_DERIVABLE_MAGNITUDE
    ][:120]
    for first, second in combinations(sample, 2):
        for combined, label in ((first + second, "sum"), (abs(first - second), "diff")):
            if abs(combined - number.value) <= _tolerance_for(combined, tolerance):
                return f"{label} of {first} and {second}"
    return None


def check_factual_consistency(
    report: str,
    source: Any,
    tolerance: float = 0.01,
    include_judgements: bool = False,
) -> FactCheckResult:
    """Verify every number in a report against the source it was generated from.

    ``tolerance`` is relative, so a 1% deviation is accepted on large counts and
    on small rates alike. The returned hallucination rate is
    unverifiable / (non-trivial numbers), so a report that cites nothing scores
    zero rather than dividing by zero.
    """
    numbers = extract_numbers(report)
    source_values = collect_source_values(source)

    verified = derived = unverifiable = trivial = 0
    unverifiable_examples: list[dict[str, Any]] = []
    judgements: list[dict[str, Any]] = []

    for number in numbers:
        if number.value in TRIVIAL_VALUES and not number.is_percentage:
            trivial += 1
            if include_judgements:
                judgements.append({**number.to_dict(), "verdict": "trivial"})
            continue

        matched = _matches_source(number, source_values, tolerance)
        if matched is not None:
            verified += 1
            if include_judgements:
                judgements.append(
                    {**number.to_dict(), "verdict": "verified", "source": matched}
                )
            continue

        derivation = _is_derived(number, source_values, tolerance)
        if derivation is not None:
            derived += 1
            if include_judgements:
                judgements.append(
                    {**number.to_dict(), "verdict": "derived", "source": derivation}
                )
            continue

        unverifiable += 1
        unverifiable_examples.append(number.to_dict())
        if include_judgements:
            judgements.append({**number.to_dict(), "verdict": "unverifiable"})

    assessed = verified + derived + unverifiable
    return FactCheckResult(
        n_numbers=len(numbers),
        n_verified=verified,
        n_derived=derived,
        n_unverifiable=unverifiable,
        n_trivial_skipped=trivial,
        hallucination_rate=(unverifiable / assessed) if assessed else 0.0,
        unverifiable_examples=unverifiable_examples[:15],
        judgements=judgements,
    )


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def check_completeness(
    report: str, required_topics: dict[str, Iterable[str]]
) -> dict[str, Any]:
    """Whether the report mentions every topic it is required to cover.

    Factual consistency alone can be gamed by saying almost nothing: a report
    containing one verified number scores a perfect hallucination rate. Coverage
    is therefore measured alongside it, and the two must be read together.

    Each topic maps to a set of keywords, any one of which counts as covering it.
    """
    lowered = report.lower()
    covered: dict[str, bool] = {}
    for topic, keywords in required_topics.items():
        covered[topic] = any(keyword.lower() in lowered for keyword in keywords)
    n_covered = sum(covered.values())
    return {
        "topics": covered,
        "n_required": len(covered),
        "n_covered": n_covered,
        "coverage": n_covered / len(covered) if covered else 0.0,
        "missing": [topic for topic, present in covered.items() if not present],
    }


def required_topics_for_summary(summary: dict[str, Any]) -> dict[str, list[str]]:
    """Topics a report about this particular summary must address.

    Derived from the summary's own content rather than fixed, so a report is
    only required to discuss findings that actually exist. Demanding coverage of
    a stratum collapse that did not happen would penalise an accurate report.
    """
    topics: dict[str, list[str]] = {
        "decision_counts": ["accept", "reject", "flag"],
        "dataset_name": [str(summary.get("dataset", "dataset"))],
    }
    decisions = summary.get("decisions", {})
    if decisions.get("by_primary_issue"):
        leading = list(decisions["by_primary_issue"])[0]
        topics["leading_issue"] = [leading.replace("_", " "), leading]
    if decisions.get("evaluation"):
        topics["detection_performance"] = [
            "recall", "false positive", "false-positive", "fpr",
        ]
    balance = summary.get("class_balance_impact", {})
    if balance.get("became_non_viable"):
        labels = [entry["label"] for entry in balance["became_non_viable"]]
        topics["stratum_collapse"] = [*labels, "viable", "stratum", "class"]
    if summary.get("recommendations"):
        topics["recommendations"] = [
            "recommend", "should", "consider", "oversample", "retain",
        ]
    if summary.get("split_integrity", {}).get("uneven"):
        topics["split_imbalance"] = ["split", "train", "test"]
    return topics