#!/usr/bin/env python3
"""Run the Dataset Summary Agent over a Data Inspector audit.

Reads ``logs/<dataset>_inspection.json``, aggregates the decisions, analyses what
acting on them would do to the dataset, and writes
``logs/<dataset>_summary.json`` for the LLM Explainer to narrate.

The label column defaults per dataset because the two corpora label differently:
ChestX-ray14 uses pipe-separated multi-label ``source_labels``, ISIC uses a
single ``diagnosis``. Pass ``--label-column`` to override.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts.common import configure_logging
    from agents.base import read_agent_output
    from agents.dataset_summary import (
        DatasetSummaryAgent,
        SummaryThresholds,
        save_summary,
    )
except ModuleNotFoundError:  # Allow `python scripts/run_summary.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from agents.base import read_agent_output  # type: ignore[no-redef]
    from agents.dataset_summary import (  # type: ignore[no-redef]
        DatasetSummaryAgent,
        SummaryThresholds,
        save_summary,
    )


# ChestX-ray14 findings are pipe-separated and multi-label; ISIC carries one
# diagnosis per lesion. Guessing wrongly would silently produce one stratum per
# label combination, hiding the size of the rare findings.
DEFAULT_LABEL_COLUMNS = {
    "chestxray14": "source_labels",
    "isic": "diagnosis",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--label-column",
        help="Column in metadata.csv holding class labels. Defaults to "
        "source_labels for chestxray14 and diagnosis for isic.",
    )
    parser.add_argument(
        "--min-viable-stratum",
        type=int,
        default=10,
        help="Images below which a stratum cannot support a stratified split",
    )
    parser.add_argument(
        "--material-loss-fraction",
        type=float,
        default=0.20,
        help="Fraction of a stratum lost to rejection that counts as material",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_summary.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"

    inspection = read_agent_output(
        logs_dir / f"{args.dataset}_inspection.json", "dataset_summary"
    )
    logger.info(
        "Read inspection of %d image(s), produced by run %s",
        len(inspection["images"]),
        inspection["provenance"]["run_id"],
    )

    label_column = args.label_column or DEFAULT_LABEL_COLUMNS.get(args.dataset)
    source_metadata = None
    processed_metadata = None
    source_path = dataset_dir / "metadata.csv"
    processed_path = dataset_dir / "processed_metadata.csv"
    if label_column and source_path.is_file():
        source_metadata = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        if processed_path.is_file():
            processed_metadata = pd.read_csv(
                processed_path, dtype=str, keep_default_na=False
            )
        logger.info("Class-balance analysis will use label column %r", label_column)
    else:
        logger.warning(
            "No label column resolved for %r, or %s missing; class-balance "
            "impact will be skipped. Pass --label-column to enable it.",
            args.dataset, source_path,
        )

    agent = DatasetSummaryAgent(
        SummaryThresholds(
            min_viable_stratum=args.min_viable_stratum,
            material_loss_fraction=args.material_loss_fraction,
        )
    )
    payload = agent.run(
        inspection,
        source_metadata=source_metadata,
        processed_metadata=processed_metadata,
        label_column=label_column,
    )

    decisions = payload["decisions"]
    logger.info("")
    logger.info("=" * 84)
    logger.info("DECISIONS (%d image(s))", decisions["n_images"])
    logger.info("=" * 84)
    for decision in ("accept", "flag", "reject"):
        count = decisions["by_decision"].get(decision, 0)
        logger.info(
            "  %-8s %6d  (%5.1f%%)", decision, count,
            100 * count / decisions["n_images"] if decisions["n_images"] else 0.0,
        )
    if decisions["by_primary_issue"]:
        logger.info("")
        logger.info("  primary issue breakdown:")
        for name, count in list(decisions["by_primary_issue"].items())[:8]:
            logger.info("    %-20s %6d", name, count)

    evaluation = decisions.get("evaluation")
    if evaluation:
        logger.info("")
        logger.info(
            "  realised FPR %.3f on %d clean row(s); recall %.3f on %d degraded",
            evaluation["realised_false_positive_rate"] or 0.0,
            evaluation["n_clean"],
            evaluation["recall_on_degraded"] or 0.0,
            evaluation["n_degraded"],
        )
        if evaluation["recall_by_severity"]:
            logger.info("  recall by severity:")
            for severity, recall in sorted(evaluation["recall_by_severity"].items()):
                logger.info("    severity %-4s %.3f", severity, recall)

    correctness = payload.get("decision_correctness", {})
    if "skipped" not in correctness:
        counts = correctness["counts"]
        logger.info("")
        logger.info("  were the decisions right?")
        logger.info(
            "    correctly actioned %d, correctly accepted %d",
            counts["true_positive"], counts["true_negative"],
        )
        logger.info(
            "    wrongly actioned   %d (of which %d with HIGH confidence)",
            counts["false_positive"],
            correctness["confident_mistakes"]["false_positive"],
        )
        logger.info(
            "    defects missed     %d (of which %d with HIGH confidence)",
            counts["false_negative"],
            correctness["confident_mistakes"]["false_negative"],
        )
        if correctness.get("least_detected_defects"):
            logger.info("    weakest detection:")
            for entry in correctness["least_detected_defects"]:
                logger.info(
                    "      %-22s %d of %d (recall %.3f)", entry["defect"],
                    entry["detected"], entry["detected"] + entry["missed"],
                    entry["recall"],
                )

    bands = payload["decisions"].get("by_confidence", {})
    if bands:
        logger.info("")
        logger.info(
            "  decision confidence: %d clear, %d borderline, %d unscored",
            bands.get("clear", 0), bands.get("borderline", 0),
            bands.get("unscored", 0),
        )

    examples = payload.get("worked_examples", {})
    if examples:
        logger.info("")
        logger.info("  worked examples:")
        for category, entries in examples.items():
            for entry in entries[:1]:
                correct = entry.get("was_correct")
                verdict = (
                    "correct" if correct else "INCORRECT" if correct is False
                    else "unknown"
                )
                logger.info(
                    "    %-22s %-22s %-7s %-11s (%s)",
                    category, entry["image_id"], entry["decision"],
                    entry["confidence_band"], verdict,
                )

    cooccurrence = payload["issue_cooccurrence"]
    if cooccurrence["top_cooccurring_pairs"]:
        logger.info("")
        logger.info("  most co-occurring check pairs (Jaccard):")
        for pair in cooccurrence["top_cooccurring_pairs"][:5]:
            logger.info(
                "    %-16s + %-16s %4d together, J=%.2f",
                pair["check_a"], pair["check_b"], pair["n_together"], pair["jaccard"],
            )

    balance = payload["class_balance_impact"]
    logger.info("")
    logger.info("=" * 84)
    logger.info("CLASS-BALANCE IMPACT OF REJECTION")
    logger.info("=" * 84)
    if "skipped" in balance:
        logger.warning("  skipped: %s", balance["skipped"])
    else:
        logger.info(
            "  %d stratum/strata across %d joined image(s)",
            balance["n_strata"], balance["n_images_joined"],
        )
        logger.info("")
        logger.info("  smallest strata (most at risk):")
        for entry in balance["strata"][:10]:
            marker = ""
            if entry in balance["became_non_viable"]:
                marker = "  <-- BECOMES NON-VIABLE"
            elif not entry["viable_before"]:
                marker = "  (already below floor)"
            logger.info(
                "    %-26s %5d -> %5d  (%5.1f%% lost)%s",
                entry["label"][:26], entry["n_before"], entry["n_after"],
                100 * entry["fraction_lost"], marker,
            )
        if balance["became_non_viable"]:
            logger.warning("")
            logger.warning(
                "  %d stratum/strata fall below the viability floor of %d after "
                "rejection: %s",
                len(balance["became_non_viable"]),
                balance["thresholds"]["min_viable_stratum"],
                ", ".join(e["label"] for e in balance["became_non_viable"]),
            )

    integrity = payload["split_integrity"]
    if "skipped" not in integrity:
        logger.info("")
        logger.info("  split integrity:")
        for entry in integrity["by_split"]:
            logger.info(
                "    %-6s %5d -> %5d  (rejection rate %.3f)",
                entry["split"], entry["n_before"], entry["n_after"],
                entry["rejection_rate"],
            )
        if integrity["uneven"]:
            logger.warning(
                "    rejection rates differ by %.1f points across splits",
                100 * integrity["rejection_rate_spread"],
            )

    logger.info("")
    logger.info("=" * 84)
    logger.info("RECOMMENDATIONS (%d)", len(payload["recommendations"]))
    logger.info("=" * 84)
    for action in payload["recommendations"]:
        logger.info("  [%s] %s", action["priority"].upper(), action["finding"])
        for option in action.get("options", []):
            logger.info("      - %s", option)
    if not payload["recommendations"]:
        logger.info("  none; no action required at this operating point")

    output_path = logs_dir / f"{args.dataset}_summary.json"
    save_summary(payload, output_path)
    logger.info("")
    logger.info("Wrote summary to %s", output_path)


if __name__ == "__main__":
    main()