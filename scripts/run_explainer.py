#!/usr/bin/env python3
"""Run the LLM Explainer Agent over a Dataset Summary and evaluate the report.

Reads ``logs/<dataset>_summary.json`` and writes both the report itself
(``logs/<dataset>_audit_report[_<backend>].md``) and the structured payload
carrying its evaluation.

``--compare`` runs several backends over the same summary and tabulates their
factual consistency and coverage. **This comparison is the stage's result.** The
template backend is a zero-hallucination control by construction, so an LLM
report is only worth using if it matches the template on factuality while
reading better - and if it does not, that is a finding worth reporting rather
than a problem to hide.

Because you have llama3.2:3b and qwen3:8b available locally, a model-size
ablation costs one extra flag:

    --compare template,ollama:llama3.2:3b,ollama:qwen3:8b
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from scripts.common import configure_logging
    from agents.base import read_agent_output
    from agents.llm_explainer import (
        BACKENDS,
        DEFAULT_MODEL,
        ExplainerConfig,
        LLMExplainerAgent,
        LLMUnavailableError,
        save_report,
    )
except ModuleNotFoundError:  # Allow `python scripts/run_explainer.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from agents.base import read_agent_output  # type: ignore[no-redef]
    from agents.llm_explainer import (  # type: ignore[no-redef]
        BACKENDS,
        DEFAULT_MODEL,
        ExplainerConfig,
        LLMExplainerAgent,
        LLMUnavailableError,
        save_report,
    )


def parse_spec(spec: str) -> ExplainerConfig:
    """Turn 'template' or 'ollama:llama3.2:3b' into a config.

    Model tags contain colons, so only the first is a separator.
    """
    spec = spec.strip()
    if spec == "template":
        return ExplainerConfig(backend="template")
    if spec.startswith("ollama"):
        _, _, model = spec.partition(":")
        return ExplainerConfig(backend="ollama", model=model or DEFAULT_MODEL)
    raise ValueError(
        f"unrecognised backend spec {spec!r}; expected 'template' or "
        "'ollama[:model-tag]'"
    )


def label_for(config: ExplainerConfig) -> str:
    return "template" if config.backend == "template" else f"ollama:{config.model}"


def slug_for(config: ExplainerConfig) -> str:
    if config.backend == "template":
        return "template"
    return "ollama_" + config.model.replace(":", "-").replace("/", "-")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--backend", default="template", choices=BACKENDS,
        help="Single backend to run. Ignored when --compare is given.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Ollama model tag, used when --backend ollama",
    )
    parser.add_argument(
        "--compare",
        help="Comma-separated backend specs to compare, e.g. "
        "'template,ollama:llama3.2:3b,ollama:qwen3:8b'",
    )
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args(argv)


def run_one(
    config: ExplainerConfig, summary: dict[str, Any], logs_dir: Path, dataset: str, logger
) -> dict[str, Any] | None:
    label = label_for(config)
    logger.info("")
    logger.info("=" * 84)
    logger.info("BACKEND: %s", label)
    logger.info("=" * 84)
    try:
        payload = LLMExplainerAgent(config).run(summary)
    except LLMUnavailableError as error:
        logger.error("  %s unavailable: %s", label, error)
        return None

    factual = payload["evaluation"]["factual_consistency"]
    completeness = payload["evaluation"]["completeness"]
    logger.info(
        "  report length %d chars; %d numeric claim(s)",
        len(payload["report"]), factual["n_numbers"],
    )
    logger.info(
        "  verified %d, derived %d, unverifiable %d, trivial %d",
        factual["n_verified"], factual["n_derived"],
        factual["n_unverifiable"], factual["n_trivial_skipped"],
    )
    logger.info(
        "  HALLUCINATION RATE %.3f   COVERAGE %.2f (%d/%d)",
        factual["hallucination_rate"], completeness["coverage"],
        completeness["n_covered"], completeness["n_required"],
    )
    if completeness["missing"]:
        logger.info("  topics not covered: %s", ", ".join(completeness["missing"]))
    for example in factual["unverifiable_examples"][:5]:
        logger.warning(
            "  unverifiable value %s in: ...%s...",
            example["raw"], example["context"][:90],
        )
    metadata = payload.get("generation_metadata", {})
    if metadata.get("total_duration_ns"):
        logger.info(
            "  generated in %.1fs (%s tokens)",
            metadata["total_duration_ns"] / 1e9, metadata.get("eval_count"),
        )

    slug = slug_for(config)
    save_report(
        payload,
        logs_dir / f"{dataset}_explainer_{slug}.json",
        logs_dir / f"{dataset}_audit_report_{slug}.md",
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_explainer.log"
    )
    logs_dir = args.data_dir / "logs"

    summary = read_agent_output(
        logs_dir / f"{args.dataset}_summary.json", "llm_explainer"
    )
    logger.info(
        "Read summary for %s, produced by run %s",
        summary.get("dataset"), summary["provenance"]["run_id"],
    )

    if args.compare:
        configs = [parse_spec(spec) for spec in args.compare.split(",") if spec.strip()]
    else:
        configs = [
            ExplainerConfig(
                backend=args.backend, model=args.model, timeout=args.timeout
            )
        ]
    for config in configs:
        config.timeout = args.timeout

    results: list[tuple[str, dict[str, Any]]] = []
    for config in configs:
        payload = run_one(config, summary, logs_dir, args.dataset, logger)
        if payload is not None:
            results.append((label_for(config), payload))

    if not results:
        raise RuntimeError(
            "no backend produced a report; if the local model is unreachable, "
            "run with --backend template"
        )

    if len(results) > 1:
        logger.info("")
        logger.info("=" * 84)
        logger.info("BACKEND COMPARISON")
        logger.info("=" * 84)
        logger.info(
            "  %-26s %10s %10s %10s %9s", "backend", "halluc.", "coverage",
            "claims", "chars",
        )
        for label, payload in results:
            factual = payload["evaluation"]["factual_consistency"]
            completeness = payload["evaluation"]["completeness"]
            logger.info(
                "  %-26s %10.3f %10.2f %10d %9d",
                label, factual["hallucination_rate"], completeness["coverage"],
                factual["n_numbers"], len(payload["report"]),
            )
        logger.info("")
        logger.info(
            "  The template backend is a zero-hallucination control by "
            "construction, so it is the baseline any model must match on "
            "factuality before its readability counts for anything."
        )

    logger.info("")
    logger.info("Wrote reports and evaluations to %s", logs_dir)


if __name__ == "__main__":
    main()