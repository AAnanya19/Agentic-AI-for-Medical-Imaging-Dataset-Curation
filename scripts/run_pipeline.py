#!/usr/bin/env python3
"""Run the complete three-agent audit pipeline with one command.

    python scripts/run_pipeline.py --dataset chestxray14 --data-dir data

Inspector -> Summary -> Explainer, each consuming the previous stage's JSON. A
run manifest recording every stage's provenance, timing, outputs and any failure
is written to ``logs/<dataset>_pipeline_manifest.json``.

**Partial success is a real outcome.** The Explainer depends on an external model
server; if that is unreachable the quality analysis is still valid and its outputs
are still written. The pipeline reports which stage failed rather than discarding
everything upstream, and exits non-zero only when a *required* stage fails.

Assumes Stages 2-6 have already produced their score files for the dataset:
``iqa_scores.csv`` is required, and ``anomaly_scores_<backbone>.csv``,
``processed_metadata.csv`` and the duplicate reports are used when present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts.common import configure_logging
    from scripts.run_inspector import (
        add_fusion_score,
        load_duplicate_map,
        load_scores,
    )
    from agents.base import read_agent_output
    from agents.data_inspector import (
        DataInspectorAgent,
        build_resolution_frame,
        save_inspection,
    )
    from agents.dataset_summary import (
        DatasetSummaryAgent,
        SummaryThresholds,
        save_summary,
    )
    from agents.inspector_config import InspectorConfig, default_config
    from agents.llm_explainer import ExplainerConfig, LLMExplainerAgent, save_report
    from agents.orchestrator import Orchestrator, summarise_manifest
except ModuleNotFoundError:  # Allow `python scripts/run_pipeline.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from scripts.run_inspector import (  # type: ignore[no-redef]
        add_fusion_score,
        load_duplicate_map,
        load_scores,
    )
    from agents.base import read_agent_output  # type: ignore[no-redef]
    from agents.data_inspector import (  # type: ignore[no-redef]
        DataInspectorAgent,
        build_resolution_frame,
        save_inspection,
    )
    from agents.dataset_summary import (  # type: ignore[no-redef]
        DatasetSummaryAgent,
        SummaryThresholds,
        save_summary,
    )
    from agents.inspector_config import InspectorConfig, default_config  # type: ignore[no-redef]
    from agents.llm_explainer import (  # type: ignore[no-redef]
        ExplainerConfig,
        LLMExplainerAgent,
        save_report,
    )
    from agents.orchestrator import Orchestrator, summarise_manifest  # type: ignore[no-redef]


DEFAULT_LABEL_COLUMNS = {
    "chestxray14": "source_labels",
    "isic": "diagnosis",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--config", type=Path, help="InspectorConfig JSON")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--label-column")
    parser.add_argument("--min-viable-stratum", type=int, default=10)
    parser.add_argument(
        "--explainer-backend", default="template", choices=("template", "ollama"),
        help="Defaults to template, which needs no model server and is the "
        "zero-hallucination control",
    )
    parser.add_argument("--explainer-model", default="llama3.2:3b")
    parser.add_argument("--explainer-timeout", type=int, default=600)
    parser.add_argument(
        "--skip-explainer", action="store_true",
        help="Run the inspector and summary only",
    )
    parser.add_argument(
        "--continue-on-failure", action="store_true",
        help="Attempt every stage rather than stopping at the first failure, to "
        "collect all errors in one run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_pipeline.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = InspectorConfig.load(args.config) if args.config else default_config()
    label_column = args.label_column or DEFAULT_LABEL_COLUMNS.get(args.dataset)

    orchestrator = Orchestrator(
        dataset=args.dataset, stop_on_failure=not args.continue_on_failure
    )
    inspection_path = logs_dir / f"{args.dataset}_inspection.json"
    summary_path = logs_dir / f"{args.dataset}_summary.json"

    logger.info("=" * 84)
    logger.info("PIPELINE: %s", args.dataset)
    logger.info("=" * 84)

    # ---------------------------------------------------------- inspector
    def run_inspector() -> dict[str, Any]:
        frame, iqa_columns, detector_columns = load_scores(
            dataset_dir, args.backbone, logger
        )
        frame = add_fusion_score(
            frame, iqa_columns, detector_columns, config.seed, logger
        )
        validation = frame[frame["split"] == "val"]
        evaluation = frame[frame["split"] == args.eval_split]
        if validation.empty or evaluation.empty:
            raise RuntimeError(
                f"need both val and {args.eval_split} rows; got "
                f"{len(validation)} and {len(evaluation)}"
            )

        resolution = None
        if config.enable_resolution_check:
            processed_path = dataset_dir / "processed_metadata.csv"
            if processed_path.is_file():
                resolution = build_resolution_frame(
                    pd.read_csv(processed_path, dtype=str, keep_default_na=False)
                )

        duplicates = (
            load_duplicate_map(logs_dir, args.dataset, logger)
            if config.enable_duplicate_check else {}
        )

        payload = DataInspectorAgent(config).run(
            dataset=args.dataset,
            validation=validation,
            test=evaluation,
            resolution=resolution,
            duplicate_map=duplicates,
            inputs={
                "backbone": args.backbone,
                "eval_split": args.eval_split,
                "iqa_columns": iqa_columns,
                "detector_columns": detector_columns,
            },
        )
        save_inspection(payload, inspection_path)
        return {
            "run_id": payload["provenance"]["run_id"],
            "outputs": [inspection_path],
            "n_images": len(payload["images"]),
            "decision_counts": payload["decision_counts"],
            "confidence_counts": payload.get("confidence_counts", {}),
        }

    result = orchestrator.run_stage("inspector", run_inspector)
    if result.status == "ok":
        logger.info(
            "  inspector: %d image(s), decisions %s",
            result.detail["n_images"], result.detail["decision_counts"],
        )

    # ------------------------------------------------------------ summary
    def run_summary() -> dict[str, Any]:
        inspection = read_agent_output(inspection_path, "dataset_summary")
        source_metadata = None
        processed_metadata = None
        if label_column and (dataset_dir / "metadata.csv").is_file():
            source_metadata = pd.read_csv(
                dataset_dir / "metadata.csv", dtype=str, keep_default_na=False
            )
            processed_path = dataset_dir / "processed_metadata.csv"
            if processed_path.is_file():
                processed_metadata = pd.read_csv(
                    processed_path, dtype=str, keep_default_na=False
                )

        payload = DatasetSummaryAgent(
            SummaryThresholds(min_viable_stratum=args.min_viable_stratum)
        ).run(
            inspection,
            source_metadata=source_metadata,
            processed_metadata=processed_metadata,
            label_column=label_column,
        )
        save_summary(payload, summary_path)
        balance = payload["class_balance_impact"]
        return {
            "run_id": payload["provenance"]["run_id"],
            "outputs": [summary_path],
            "n_recommendations": len(payload["recommendations"]),
            "n_strata_collapsed": (
                len(balance.get("became_non_viable", []))
                if "skipped" not in balance else None
            ),
        }

    result = orchestrator.run_stage("summary", run_summary)
    if result.status == "ok":
        logger.info(
            "  summary: %d recommendation(s), %s stratum/strata collapsed",
            result.detail["n_recommendations"],
            result.detail["n_strata_collapsed"],
        )

    # ---------------------------------------------------------- explainer
    if args.skip_explainer:
        logger.info("  explainer: skipped by request")
    else:
        def run_explainer() -> dict[str, Any]:
            summary = read_agent_output(summary_path, "llm_explainer")
            explainer_config = ExplainerConfig(
                backend=args.explainer_backend,
                model=args.explainer_model,
                timeout=args.explainer_timeout,
            )
            payload = LLMExplainerAgent(explainer_config).run(summary)
            slug = (
                "template" if args.explainer_backend == "template"
                else "ollama_" + args.explainer_model.replace(":", "-")
            )
            json_path = logs_dir / f"{args.dataset}_explainer_{slug}.json"
            markdown_path = logs_dir / f"{args.dataset}_audit_report_{slug}.md"
            save_report(payload, json_path, markdown_path)
            factual = payload["evaluation"]["factual_consistency"]
            return {
                "run_id": payload["provenance"]["run_id"],
                "outputs": [json_path, markdown_path],
                "backend": args.explainer_backend,
                "model": payload.get("model"),
                "hallucination_rate": factual["hallucination_rate"],
                "coverage": payload["evaluation"]["completeness"]["coverage"],
            }

        result = orchestrator.run_stage("explainer", run_explainer, optional=True)
        if result.status == "ok":
            logger.info(
                "  explainer: %s, hallucination rate %.3f, coverage %.2f",
                result.detail["model"], result.detail["hallucination_rate"],
                result.detail["coverage"],
            )
        elif result.status == "failed":
            logger.warning(
                "  explainer failed (%s), but the quality analysis above is "
                "unaffected and its outputs are written",
                result.error_type,
            )

    # ----------------------------------------------------------- manifest
    manifest_config = {
        "backbone": args.backbone,
        "eval_split": args.eval_split,
        "label_column": label_column,
        "min_viable_stratum": args.min_viable_stratum,
        "explainer_backend": args.explainer_backend,
        "explainer_model": (
            args.explainer_model if args.explainer_backend == "ollama" else None
        ),
        "inspector_config": config.to_dict(),
    }
    manifest_path = logs_dir / f"{args.dataset}_pipeline_manifest.json"
    orchestrator.write_manifest(manifest_path, manifest_config)

    logger.info("")
    logger.info("=" * 84)
    logger.info("RUN MANIFEST")
    logger.info("=" * 84)
    for line in summarise_manifest(orchestrator.manifest(manifest_config)):
        logger.info(line)
    logger.info("")
    logger.info("Wrote the manifest to %s", manifest_path)

    # Non-zero only when a required stage failed, so a missing model server does
    # not make the whole run look broken to a calling script.
    return 0 if orchestrator.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())