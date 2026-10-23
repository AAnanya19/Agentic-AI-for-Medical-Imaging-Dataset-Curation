#!/usr/bin/env python3
"""Regenerate the per-distortion head-to-head comparison of the two method families.

For each distortion this identifies the best-performing classical metric and the
best-performing embedding detector, then tests the difference between them by
paired bootstrap on the rows both scored. The result is the central experiment of
Chapter 4 and was previously computed interactively without being written to
disk; this script produces it as a reproducible artifact.

    python scripts/compare_families.py --dataset chestxray14 --data-dir data

Writes ``logs/<dataset>_headtohead_<backbone>.csv`` and a copy at
``logs/<dataset>_headtohead.csv`` for the figure script to read.

**Selection and testing use the same rows.** The winning metric in each family is
chosen on the same data the comparison is then made on, which favours whichever
family has more candidates to choose from. Since the classical family has nine
candidates and the embedding family six, any bias from this runs against the
conclusion that classical metrics perform better, so the result is conservative
in the direction that matters. This is stated rather than corrected because a
clean correction would require a further split, and the corpus is not large
enough to afford one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from scripts.common import configure_logging
    from pipeline.iqa_metrics import METRICS
    from pipeline.significance import paired_bootstrap_diff
except ModuleNotFoundError:  # Allow `python scripts/compare_families.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.iqa_metrics import METRICS  # type: ignore[no-redef]
    from pipeline.significance import paired_bootstrap_diff  # type: ignore[no-redef]


DETECTOR_PREFIXES = ("knn_", "mahalanobis", "isolation_forest")


def metric_polarity(name: str) -> bool | None:
    """True when a higher raw value means better quality, None if undirected."""
    if name == "brightness_deviation":
        return False
    spec = METRICS.get(name)
    return spec.higher_is_better if spec else None


def orient(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Flip sign so that, uniformly, a higher value means worse quality."""
    values = np.asarray(values, dtype=float)
    return -values if higher_is_better else values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits", default="val,test",
        help="Splits to evaluate on; the training split fits the detectors",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_headtohead.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    iqa_path = dataset_dir / "iqa_scores.csv"
    anomaly_path = dataset_dir / f"anomaly_scores_{args.backbone}.csv"
    for path in (iqa_path, anomaly_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")

    iqa = pd.read_csv(iqa_path)
    anomaly = pd.read_csv(anomaly_path)
    detector_columns = [
        column for column in anomaly.columns
        if column.startswith(DETECTOR_PREFIXES)
    ]
    frame = iqa.merge(
        anomaly[["image_id", *detector_columns]], on="image_id", how="inner"
    )

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    frame = frame[frame["split"].isin(splits)]
    logger.info(
        "%d row(s) scored by both families across split(s) %s",
        len(frame), ", ".join(splits),
    )

    metric_columns = [name for name in METRICS if name in frame.columns]
    if "brightness_deviation" in frame.columns:
        metric_columns.append("brightness_deviation")
    metric_columns = [
        name for name in metric_columns if metric_polarity(name) is not None
    ]
    logger.info(
        "%d classical metric(s), %d embedding detector(s)",
        len(metric_columns), len(detector_columns),
    )

    clean = frame[frame["defect_type"] == "clean"]
    defects = sorted(
        value for value in frame["defect_type"].unique() if value != "clean"
    )

    records = []
    for defect in defects:
        degraded = frame[frame["defect_type"] == defect]
        subset = pd.concat([clean, degraded])
        labels = (subset["defect_type"] != "clean").astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            logger.warning("skipping %s: only one class present", defect)
            continue

        def best_of(columns: list[str], orient_by_polarity: bool) -> tuple[str, float, np.ndarray]:
            """Highest-AUROC column in a family, with its oriented scores."""
            best_name, best_auroc, best_scores = None, -np.inf, None
            for name in columns:
                values = subset[name].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    continue
                if orient_by_polarity:
                    scores = orient(values, metric_polarity(name))
                else:
                    # Detector scores are already higher = more anomalous.
                    scores = values
                auroc = float(roc_auc_score(labels, scores))
                if auroc > best_auroc:
                    best_name, best_auroc, best_scores = name, auroc, scores
            return best_name, best_auroc, best_scores

        metric_name, metric_auroc, metric_scores = best_of(metric_columns, True)
        detector_name, detector_auroc, detector_scores = best_of(
            detector_columns, False
        )
        if metric_scores is None or detector_scores is None:
            logger.warning("skipping %s: a family produced no usable score", defect)
            continue

        # Difference is expressed as embedding minus classical, so a negative
        # value favours the classical family.
        comparison = paired_bootstrap_diff(
            labels, detector_scores, metric_scores,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        verdict = (
            "embedding" if comparison.significant_at_95 and comparison.diff > 0
            else "classical" if comparison.significant_at_95
            else "no difference"
        )
        records.append({
            "defect": defect,
            "n_clean": int((labels == 0).sum()),
            "n_degraded": int((labels == 1).sum()),
            "best_metric": metric_name,
            "metric_auroc": round(metric_auroc, 4),
            "best_detector": detector_name,
            "detector_auroc": round(detector_auroc, 4),
            "diff": round(comparison.diff, 4),
            "ci_low": round(comparison.ci_low, 4),
            "ci_high": round(comparison.ci_high, 4),
            "p_diff_le_zero": round(comparison.p_diff_le_zero, 4),
            "significant": comparison.significant_at_95,
            "favours": verdict,
        })

    if not records:
        raise RuntimeError("no distortion produced a usable comparison")

    table = pd.DataFrame(records).sort_values("diff")
    primary = logs_dir / f"{args.dataset}_headtohead_{args.backbone}.csv"
    table.to_csv(primary, index=False)
    table.to_csv(logs_dir / f"{args.dataset}_headtohead.csv", index=False)

    logger.info("")
    logger.info("=" * 96)
    logger.info("HEAD-TO-HEAD: best embedding detector against best classical metric")
    logger.info("=" * 96)
    logger.info(
        "  %-20s %10s %10s %9s %20s  %s",
        "distortion", "embedding", "classical", "diff", "95% interval", "favours",
    )
    for row in table.itertuples(index=False):
        logger.info(
            "  %-20s %10.3f %10.3f %+9.3f  [%+.3f, %+.3f]  %s",
            row.defect, row.detector_auroc, row.metric_auroc, row.diff,
            row.ci_low, row.ci_high, row.favours,
        )

    counts = table["favours"].value_counts().to_dict()
    logger.info("")
    logger.info(
        "  classical significantly better on %d of %d distortion(s); "
        "embedding on %d; no difference on %d",
        counts.get("classical", 0), len(table),
        counts.get("embedding", 0), counts.get("no difference", 0),
    )
    logger.info("")
    logger.info("Wrote %s", primary)


if __name__ == "__main__":
    main()