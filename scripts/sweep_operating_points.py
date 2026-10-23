#!/usr/bin/env python3
"""Sweep the Inspector's operating point and report the false-reject / recall curve.

Answers the question a clinical deployment actually asks: **given that I can
tolerate discarding at most X% of clean images, how many defects do I catch?**

Reports three rates separately at each point, because a single "false positive
rate" conflates two errors of very different cost:

  * ``false_reject_rate``  - clean images discarded. The expensive error.
  * ``false_flag_rate``    - clean images sent for human review. Cheap.
  * ``recall_any_action``  - degraded images either flagged or rejected.

Thresholds are re-derived from the same clean validation rows at each target and
re-applied to the evaluation split, exactly as the Inspector does. Nothing is
refitted, so the curve is a property of the scorer rather than of a new model per
point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.common import configure_logging
    from scripts.run_inspector import add_fusion_score, load_scores
    from agents.data_inspector import orient
    from agents.inspector_config import InspectorConfig, default_config
    from agents.operating_points import (
        DEFAULT_REJECT_TARGETS,
        points_to_frame,
        smallest_sufficient_point,
        sweep_operating_points,
        threshold_stability_warning,
    )
except ModuleNotFoundError:  # Allow `python scripts/sweep_operating_points.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from scripts.run_inspector import add_fusion_score, load_scores  # type: ignore[no-redef]
    from agents.data_inspector import orient  # type: ignore[no-redef]
    from agents.inspector_config import InspectorConfig, default_config  # type: ignore[no-redef]
    from agents.operating_points import (  # type: ignore[no-redef]
        DEFAULT_REJECT_TARGETS,
        points_to_frame,
        smallest_sufficient_point,
        sweep_operating_points,
        threshold_stability_warning,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--reject-targets",
        default=",".join(str(value) for value in DEFAULT_REJECT_TARGETS),
        help="Comma-separated target false-reject rates",
    )
    parser.add_argument(
        "--flag-multiplier", type=float, default=5.0,
        help="How much looser the flag threshold is than the reject threshold. "
        "The default of 5 mirrors the Inspector's 1%% reject / 5%% flag pairing.",
    )
    parser.add_argument(
        "--budgets", default="0.001,0.005,0.01,0.05",
        help="False-reject budgets to report the best achievable recall for",
    )
    parser.add_argument("--eval-split", default="test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_operating_points.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"

    config = InspectorConfig.load(args.config) if args.config else default_config()

    frame, iqa_columns, detector_columns = load_scores(
        dataset_dir, args.backbone, logger
    )
    frame = add_fusion_score(
        frame, iqa_columns, detector_columns, config.seed, logger
    )

    score_column = config.overall_check.score_column
    higher_is_better = config.overall_check.higher_is_better
    frame["_oriented"] = orient(frame[score_column].to_numpy(), higher_is_better)

    validation = frame[frame["split"] == "val"]
    evaluation = frame[frame["split"] == args.eval_split]
    validation_clean = validation[validation["defect_type"] == "clean"]
    if validation_clean.empty or evaluation.empty:
        raise RuntimeError(
            f"need clean val rows and {args.eval_split} rows; got "
            f"{len(validation_clean)} and {len(evaluation)}"
        )

    targets = [
        float(value) for value in args.reject_targets.split(",") if value.strip()
    ]
    points = sweep_operating_points(
        validation_clean_scores=validation_clean["_oriented"].to_numpy(),
        evaluation_scores=evaluation["_oriented"].to_numpy(),
        evaluation_is_degraded=(evaluation["defect_type"] != "clean").to_numpy(),
        evaluation_severity=evaluation["severity_level"].astype(str).tolist(),
        reject_targets=targets,
        flag_multiplier=args.flag_multiplier,
    )

    logger.info("")
    logger.info("=" * 96)
    logger.info(
        "OPERATING-POINT SWEEP - %s (%d clean val rows, %d %s rows)",
        args.dataset, len(validation_clean), len(evaluation), args.eval_split,
    )
    logger.info("=" * 96)
    logger.info(
        "  %8s %12s %12s %12s %10s %10s",
        "target", "false-rej", "false-flag", "combined", "recall", "rec@sev1",
    )
    for point in points:
        logger.info(
            "  %8.4f %12.4f %12.4f %12.4f %10.3f %10.3f",
            point.target_reject_fpr, point.false_reject_rate,
            point.false_flag_rate, point.combined_false_positive_rate,
            point.recall_any_action,
            point.recall_by_severity.get("1", float("nan")),
        )

    logger.info("")
    logger.info(
        "  A false FLAG routes a clean image to human review; a false REJECT "
        "discards it. The two are reported separately because their costs differ "
        "by orders of magnitude, and a combined rate overstates the harm."
    )

    logger.info("")
    logger.info("  best achievable recall within a false-reject budget:")
    budget_rows = []
    for budget in [float(v) for v in args.budgets.split(",") if v.strip()]:
        best = smallest_sufficient_point(points, budget)
        if best is None:
            logger.info(
                "    <= %.3f  no operating point achieves this budget", budget
            )
            continue
        logger.info(
            "    <= %.3f  recall %.3f  (actual false-reject %.4f, target %.4f)",
            budget, best.recall_any_action, best.false_reject_rate,
            best.target_reject_fpr,
        )
        budget_rows.append({
            "false_reject_budget": budget,
            "achievable_recall": best.recall_any_action,
            "actual_false_reject_rate": best.false_reject_rate,
            "target_used": best.target_reject_fpr,
        })

    warning = threshold_stability_warning(points)
    if warning:
        logger.warning("")
        logger.warning("  %s", warning)

    table = points_to_frame(points)
    table.to_csv(logs_dir / f"{args.dataset}_operating_points.csv", index=False)
    (logs_dir / f"{args.dataset}_operating_points.json").write_text(
        json.dumps({
            "dataset": args.dataset,
            "score_column": score_column,
            "n_clean_validation_rows": int(len(validation_clean)),
            "n_evaluation_rows": int(len(evaluation)),
            "eval_split": args.eval_split,
            "flag_multiplier": args.flag_multiplier,
            "points": [point.to_dict() for point in points],
            "budgets": budget_rows,
            "stability_warning": warning,
            "note": (
                "false_reject_rate is the rate of clean images DISCARDED; "
                "false_flag_rate is the rate routed to human review. Thresholds "
                "re-derived from clean validation rows at each target."
            ),
        }, indent=2, sort_keys=True) + "\n"
    )
    logger.info("")
    logger.info("Wrote the sweep to %s", logs_dir)


if __name__ == "__main__":
    main()