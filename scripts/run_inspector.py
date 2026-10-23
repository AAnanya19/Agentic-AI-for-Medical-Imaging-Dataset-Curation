#!/usr/bin/env python3
"""Run the Data Inspector Agent over a dataset and emit its structured audit.

Assembles the scores produced by Stages 2-6 into one frame, derives decision
thresholds from clean validation rows, decides on every test row, and writes a
JSON audit that the Dataset Summary Agent consumes directly.

Inputs, all optional except the IQA scores, so the agent degrades gracefully
rather than failing when a stage has not been run:

  * ``<dataset>/iqa_scores.csv``            - required (Stage 2)
  * ``<dataset>/anomaly_scores_<backbone>.csv`` - embedding detectors (Stage 3)
  * ``<dataset>/processed_metadata.csv``    - resolution provenance (Stage 6)
  * ``logs/<dataset>_dup_exact_pairs.csv``  - duplicate pairs (Stage 6)

The fusion score that backs the overall decision is fitted here on validation
rows rather than being read from a file, so the Inspector is self-contained: it
can score a dataset given only its raw metric columns.

**Policy lives in a config file, not in this script.** ``--config`` loads an
``InspectorConfig``; ``--write-default-config`` emits the current defaults for
editing. Which checks run, what false-positive rates they target, and how flag
and reject are separated are all editable without touching code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.common import configure_logging
    from agents.data_inspector import (
        DataInspectorAgent,
        build_resolution_frame,
        orient,
        save_inspection,
    )
    from agents.inspector_config import InspectorConfig, default_config
    from pipeline.fusion import LogisticFusion
    from pipeline.iqa_metrics import METRICS
except ModuleNotFoundError:  # Allow `python scripts/run_inspector.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from agents.data_inspector import (  # type: ignore[no-redef]
        DataInspectorAgent,
        build_resolution_frame,
        orient,
        save_inspection,
    )
    from agents.inspector_config import InspectorConfig, default_config  # type: ignore[no-redef]
    from pipeline.fusion import LogisticFusion  # type: ignore[no-redef]
    from pipeline.iqa_metrics import METRICS  # type: ignore[no-redef]


DETECTOR_PREFIXES = ("knn_", "mahalanobis", "isolation_forest")


def metric_polarity(name: str) -> bool | None:
    if name == "brightness_deviation":
        return False
    spec = METRICS.get(name)
    return spec.higher_is_better if spec else None


def load_scores(
    dataset_dir: Path, backbone: str, logger
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join IQA and (where available) embedding scores into one frame."""
    iqa_path = dataset_dir / "iqa_scores.csv"
    if not iqa_path.is_file():
        raise FileNotFoundError(
            f"{iqa_path} not found; run compute_iqa.py for this dataset first"
        )
    frame = pd.read_csv(iqa_path)
    iqa_columns = [name for name in METRICS if name in frame.columns]
    if "brightness_deviation" in frame.columns:
        iqa_columns.append("brightness_deviation")

    detector_columns: list[str] = []
    anomaly_path = dataset_dir / f"anomaly_scores_{backbone}.csv"
    if anomaly_path.is_file():
        anomaly = pd.read_csv(anomaly_path)
        detector_columns = [
            column
            for column in anomaly.columns
            if column.startswith(DETECTOR_PREFIXES)
        ]
        before = len(frame)
        frame = frame.merge(
            anomaly[["image_id", *detector_columns]], on="image_id", how="left"
        )
        matched = frame[detector_columns[0]].notna().sum() if detector_columns else 0
        logger.info(
            "Merged %d detector column(s) from %s; %d of %d row(s) matched",
            len(detector_columns), anomaly_path.name, matched, before,
        )
    else:
        logger.warning(
            "No embedding scores at %s. The Inspector will run on IQA metrics "
            "alone - which Stage 4b found loses only ~0.006 AUROC - but the "
            "fusion score will not include detector features.",
            anomaly_path,
        )
    return frame, iqa_columns, detector_columns


def add_fusion_score(
    frame: pd.DataFrame,
    iqa_columns: list[str],
    detector_columns: list[str],
    seed: int,
    logger,
) -> pd.DataFrame:
    """Fit logistic fusion on validation rows and score every row with it.

    Fitted here rather than loaded so the Inspector is self-contained. Fitting
    uses validation rows only; the thresholds derived later use the same rows,
    and decisions are reported on test, so no test row influences either.
    """
    feature_columns: list[str] = []
    oriented_frame = pd.DataFrame(index=frame.index)
    for name in iqa_columns:
        polarity = metric_polarity(name)
        if polarity is None:
            continue  # non-directional metrics cannot be oriented
        oriented_frame[f"iqa:{name}"] = orient(frame[name].to_numpy(), polarity)
        feature_columns.append(f"iqa:{name}")
    for name in detector_columns:
        # Detector scores are already higher = more anomalous.
        oriented_frame[f"emb:{name}"] = frame[name].to_numpy()
        feature_columns.append(f"emb:{name}")

    usable = oriented_frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    complete = usable.notna().all(axis=1)
    validation = frame["split"] == "val"
    fit_mask = complete & validation
    if fit_mask.sum() < 20:
        raise RuntimeError(
            f"only {int(fit_mask.sum())} complete validation row(s) available to "
            "fit the fusion scorer; at least 20 are needed"
        )

    labels = (frame.loc[fit_mask, "defect_type"] != "clean").astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            "validation rows contain only one class, so the fusion scorer cannot "
            "be fitted; check that inject_defects.py produced both clean and "
            "degraded validation rows"
        )
    model = LogisticFusion(random_state=seed).fit(
        usable.loc[fit_mask].to_numpy(), labels, feature_names=feature_columns
    )
    logger.info(
        "Fitted fusion scorer on %d complete validation row(s), %d feature(s)",
        int(fit_mask.sum()), len(feature_columns),
    )

    scores = np.full(len(frame), np.nan)
    scores[complete.to_numpy()] = model.score(usable.loc[complete].to_numpy())
    frame = frame.copy()
    frame["fused_logistic"] = scores
    incomplete = int((~complete).sum())
    if incomplete:
        logger.warning(
            "%d row(s) had a missing or non-finite feature and carry no fusion "
            "score; they are reported as not-scoreable rather than as passing",
            incomplete,
        )
    return frame


def load_duplicate_map(logs_dir: Path, dataset: str, logger) -> dict[str, str]:
    """Map each image to one partner it duplicates, from Stage 6 output."""
    mapping: dict[str, str] = {}
    for task in ("exact", "transformation"):
        path = logs_dir / f"{dataset}_dup_{task}_pairs.csv"
        if not path.is_file():
            continue
        pairs = pd.read_csv(path)
        exact_only = pairs[pairs.get("is_exact", False) == True]  # noqa: E712
        for row in exact_only.itertuples(index=False):
            # Record one direction only, so a pair yields a single flag rather
            # than both members flagging each other.
            if row.image_id_b not in mapping:
                mapping[row.image_id_b] = row.image_id_a
        logger.info(
            "Loaded %d exact duplicate pair(s) from %s", len(exact_only), path.name
        )
    if not mapping:
        logger.info(
            "No exact duplicate pairs found; the duplicate check will flag nothing"
        )
    return mapping


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument(
        "--config", type=Path, help="InspectorConfig JSON. Defaults are used if omitted."
    )
    parser.add_argument(
        "--write-default-config", type=Path,
        help="Write the default configuration to this path and exit, for editing",
    )
    parser.add_argument(
        "--eval-split", default="test",
        help="Split to decide on. Thresholds always come from val.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.write_default_config:
        default_config().save(args.write_default_config)
        print(f"Wrote default Inspector configuration to {args.write_default_config}")
        return

    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_inspector.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"

    config = (
        InspectorConfig.load(args.config) if args.config else default_config()
    )
    logger.info(
        "Configuration: %d check(s) enabled, overall decision from %s",
        len(config.enabled_checks()), config.overall_check.score_column,
    )
    if args.config:
        logger.info("Loaded policy from %s", args.config)

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
            f"need both val and {args.eval_split} rows; got {len(validation)} and "
            f"{len(evaluation)}"
        )

    resolution = None
    if config.enable_resolution_check:
        processed_path = dataset_dir / "processed_metadata.csv"
        if processed_path.is_file():
            processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
            resolution = build_resolution_frame(processed)
            logger.info("Resolution provenance available for %d row(s)", len(resolution))
        else:
            logger.warning(
                "No %s; the resolution check will be skipped for every image",
                processed_path.name,
            )

    duplicate_map = (
        load_duplicate_map(logs_dir, args.dataset, logger)
        if config.enable_duplicate_check
        else {}
    )

    agent = DataInspectorAgent(config)
    payload = agent.run(
        dataset=args.dataset,
        validation=validation,
        test=evaluation,
        resolution=resolution,
        duplicate_map=duplicate_map,
        inputs={
            "iqa_columns": iqa_columns,
            "detector_columns": detector_columns,
            "backbone": args.backbone,
            "eval_split": args.eval_split,
        },
    )

    logger.info("")
    logger.info("=" * 84)
    logger.info("DERIVED THRESHOLDS (from clean validation rows)")
    logger.info("=" * 84)
    for name, entry in payload["thresholds"].items():
        logger.info(
            "  %-18s flag %10.4f (target FPR %.0f%%)   reject %10.4f (%.0f%%)   "
            "from %d clean row(s)",
            name, entry["flag_threshold"], 100 * entry["flag_fpr_target"],
            entry["reject_threshold"], 100 * entry["reject_fpr_target"],
            entry["n_clean_validation_rows"],
        )
    if payload["checks_unavailable"]:
        logger.warning(
            "  checks with no available score column: %s",
            ", ".join(payload["checks_unavailable"]),
        )

    counts = payload["decision_counts"]
    total = sum(counts.values())
    logger.info("")
    logger.info("=" * 84)
    logger.info("DECISIONS on the %s split (%d image(s))", args.eval_split, total)
    logger.info("=" * 84)
    for decision in ("accept", "flag", "reject"):
        logger.info(
            "  %-8s %5d  (%5.1f%%)", decision, counts[decision],
            100 * counts[decision] / total if total else 0.0,
        )

    # Realised false-positive rate on clean evaluation rows: the number that
    # says whether the derived thresholds actually held on unseen data.
    clean_images = [
        image for image in payload["images"]
        if image["ground_truth_defect_type"] == "clean"
    ]
    if clean_images:
        false_positives = sum(
            1 for image in clean_images if image["decision"] != "accept"
        )
        logger.info("")
        logger.info(
            "  realised FPR on clean %s rows: %.3f (%d of %d) against a %.3f target",
            args.eval_split, false_positives / len(clean_images), false_positives,
            len(clean_images), config.overall_check.flag_fpr,
        )
    degraded = [
        image for image in payload["images"]
        if image["ground_truth_defect_type"] not in (None, "clean")
    ]
    if degraded:
        caught = sum(1 for image in degraded if image["decision"] != "accept")
        logger.info(
            "  recall on degraded %s rows:    %.3f (%d of %d)",
            args.eval_split, caught / len(degraded), caught, len(degraded),
        )

    # Structural findings are reported separately from the quality decision,
    # because an extreme source aspect ratio is a real curation issue but not a
    # quality defect - counting it against a quality FPR would be a category
    # error. See structural_findings_affect_decision in the config.
    structural = [
        image for image in payload["images"] if image.get("structural_findings")
    ]
    if structural:
        from collections import Counter

        counts = Counter(
            name for image in structural for name in image["structural_findings"]
        )
        logger.info("")
        logger.info(
            "  structural findings on %d image(s) (%s the quality decision):",
            len(structural),
            "affecting" if config.structural_findings_affect_decision else "separate from",
        )
        for name, count in counts.most_common():
            logger.info("    %-18s %d", name, count)

    issues = pd.Series(
        [image["primary_issue"] for image in payload["images"] if image["primary_issue"]]
    )
    if not issues.empty:
        logger.info("")
        logger.info("  most common primary issue:")
        for name, count in issues.value_counts().head(6).items():
            logger.info("    %-18s %d", name, count)

    output_path = logs_dir / f"{args.dataset}_inspection.json"
    save_inspection(payload, output_path)
    resolved_config = logs_dir / f"{args.dataset}_inspector_config_used.json"
    config.save(resolved_config)
    logger.info("")
    logger.info("Wrote audit to %s", output_path)
    logger.info("Wrote the exact configuration used to %s", resolved_config)


if __name__ == "__main__":
    main()