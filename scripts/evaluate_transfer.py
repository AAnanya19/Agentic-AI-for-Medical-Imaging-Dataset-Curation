#!/usr/bin/env python3
"""Stage 5: cross-modality transfer of a fitted quality scorer.

Stages 2-4 answered "does the method work on each modality independently".
This answers the different and more practical question: **does a scorer fitted
on one modality still work on the other, unchanged?** That is what determines
whether the Data Inspector Agent needs recalibrating per modality or can ship
one set of weights.

Three experiments:

  1. **Fusion transfer.** Fit logistic fusion on modality A's validation split,
     apply it unchanged to modality B's test split. Reported against the
     same-modality baseline so the degradation is interpretable rather than an
     isolated number.
  2. **Threshold transfer.** Take the operating point that yields a target FPR
     on A's clean images and apply that raw threshold to B. The realised FPR on
     B is the number the agent actually cares about: if a 5%-FPR threshold from
     radiography produces 40% FPR on dermoscopy, per-modality calibration is
     mandatory and must be stated as a deployment constraint.
  3. **Feature-space comparison.** Mean and spread of each score family on each
     modality's clean images, to show *why* transfer succeeds or fails - a
     threshold cannot transfer if the underlying score distributions do not
     overlap.

Both modalities must share an identical feature layout for a fitted model to be
applied across them, so the intersection of available IQA metrics and detectors
is used and reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.common import configure_logging
    from scripts.evaluate_fusion import (
        FEATURE_SETS,
        load_scores,
        metric_polarity,
        oriented_matrix,
    )
    from pipeline.fusion import LogisticFusion
    from pipeline.iqa_metrics import METRICS
    from pipeline.significance import bootstrap_auroc_ci, paired_bootstrap_diff
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:  # Allow `python scripts/evaluate_transfer.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from scripts.evaluate_fusion import (  # type: ignore[no-redef]
        FEATURE_SETS,
        load_scores,
        metric_polarity,
        oriented_matrix,
    )
    from pipeline.fusion import LogisticFusion  # type: ignore[no-redef]
    from pipeline.iqa_metrics import METRICS  # type: ignore[no-redef]
    from pipeline.significance import bootstrap_auroc_ci, paired_bootstrap_diff  # type: ignore[no-redef]
    from sklearn.metrics import roc_auc_score


def shared_feature_columns(
    frames: dict[str, pd.DataFrame]
) -> tuple[list[str], list[str]]:
    """IQA metrics and detectors present in *every* dataset, in a stable order.

    A model fitted on one modality can only be applied to another if the feature
    vector means the same thing in both, so anything missing from either side is
    dropped rather than zero-filled - zero-filling would silently feed the model
    a value it never saw during fitting.
    """
    iqa_candidates = [name for name in METRICS] + ["brightness_deviation"]
    detector_prefixes = ("knn_", "mahalanobis", "isolation_forest")

    shared_iqa = [
        name
        for name in iqa_candidates
        if all(name in frame.columns for frame in frames.values())
        and metric_polarity(name) is not None
    ]
    detector_names = sorted(
        {
            column
            for frame in frames.values()
            for column in frame.columns
            if column.startswith(detector_prefixes)
        }
    )
    shared_detectors = [
        name
        for name in detector_names
        if all(name in frame.columns for frame in frames.values())
    ]
    return shared_iqa, shared_detectors


def threshold_at_target_fpr(
    clean_scores: np.ndarray, target_fpr: float
) -> float:
    """The score threshold that flags exactly target_fpr of clean images."""
    return float(np.percentile(clean_scores, 100.0 * (1.0 - target_fpr)))


def realised_rates(
    clean_scores: np.ndarray, degraded_scores: np.ndarray, threshold: float
) -> tuple[float, float]:
    """(false positive rate, recall) when a raw threshold is applied."""
    fpr = float(np.mean(clean_scores > threshold))
    recall = float(np.mean(degraded_scores > threshold))
    return fpr, recall


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="chestxray14,isic",
        help="Comma-separated dataset names to transfer between (exactly two)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument(
        "--feature-set",
        default="all",
        choices=sorted(FEATURE_SETS),
        help="Which score families the transferred model may use",
    )
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    if len(dataset_names) != 2:
        raise ValueError(f"expected exactly two datasets, got {dataset_names}")

    logger = configure_logging(
        args.data_dir, dataset_names[0], False,
        f"transfer_{'_'.join(dataset_names)}.log",
    )

    frames = {
        name: load_scores(args.data_dir / name, args.backbone)
        for name in dataset_names
    }
    for name, frame in frames.items():
        logger.info("%s: %d matched row(s)", name, len(frame))

    shared_iqa, shared_detectors = shared_feature_columns(frames)
    logger.info(
        "Shared feature layout: %d IQA metric(s), %d detector(s)",
        len(shared_iqa), len(shared_detectors),
    )
    if not shared_iqa and not shared_detectors:
        raise RuntimeError("the two datasets share no usable feature columns")

    # Fit one model per source modality on that modality's validation split.
    models: dict[str, LogisticFusion] = {}
    for name, frame in frames.items():
        val = frame[frame["split"] == "val"]
        if val.empty:
            raise RuntimeError(f"{name} has no val rows to fit on")
        matrix, feature_names = oriented_matrix(
            val, shared_iqa, shared_detectors, args.feature_set
        )
        labels = (val["defect_type"] != "clean").astype(int).to_numpy()
        models[name] = LogisticFusion(random_state=args.seed).fit(
            matrix, labels, feature_names=feature_names
        )
        logger.info("Fitted logistic fusion on %s val (%d rows)", name, len(val))

    # Score every (source model, target dataset) pair on the target's test split.
    results: dict[str, Any] = {}
    scored: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for target in dataset_names:
        test = frames[target][frames[target]["split"] == "test"]
        matrix, _ = oriented_matrix(
            test, shared_iqa, shared_detectors, args.feature_set
        )
        labels = (test["defect_type"] != "clean").astype(int).to_numpy()
        for source in dataset_names:
            values = models[source].score(matrix)
            scored[(source, target)] = {
                "scores": values, "labels": labels,
                "clean": values[labels == 0], "degraded": values[labels == 1],
            }
            ci = bootstrap_auroc_ci(
                labels, values, n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            results[f"fit_{source}__test_{target}"] = {
                "auroc": ci.point, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                "n_test_rows": int(len(labels)),
                "same_modality": source == target,
            }

    logger.info("")
    logger.info("=" * 96)
    logger.info("EXPERIMENT 1 - FUSION TRANSFER (AUROC on the target's test split)")
    logger.info("=" * 96)
    header = "  " + " " * 22 + "".join(f"test on {name:<18s}" for name in dataset_names)
    logger.info(header)
    for source in dataset_names:
        cells = []
        for target in dataset_names:
            entry = results[f"fit_{source}__test_{target}"]
            marker = " *" if entry["same_modality"] else "  "
            cells.append(f"{entry['auroc']:.3f}{marker}            ")
        logger.info("  fit on %-16s %s", source, "".join(cells))
    logger.info("  (* = same modality, i.e. the no-transfer baseline)")

    # Degradation from transferring, tested properly: the two scorers are
    # compared on the same target rows, so a paired bootstrap applies.
    logger.info("")
    for target in dataset_names:
        other = [name for name in dataset_names if name != target][0]
        native = scored[(target, target)]
        foreign = scored[(other, target)]
        comparison = paired_bootstrap_diff(
            native["labels"], native["scores"], foreign["scores"],
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        verdict = (
            "transfer LOSES significantly"
            if comparison.significant_at_95 and comparison.diff > 0
            else "transfer GAINS significantly"
            if comparison.significant_at_95
            else "no significant loss from transfer"
        )
        logger.info(
            "  on %-14s native %.3f vs transferred-from-%s %.3f  "
            "diff %+.3f CI [%+.3f, %+.3f]  ->  %s",
            target, comparison.auroc_a, other, comparison.auroc_b,
            comparison.diff, comparison.ci_low, comparison.ci_high, verdict,
        )
        results[f"transfer_penalty_on_{target}"] = {
            "native_auroc": comparison.auroc_a,
            "transferred_auroc": comparison.auroc_b,
            "diff": comparison.diff,
            "ci_low": comparison.ci_low,
            "ci_high": comparison.ci_high,
            "significant": comparison.significant_at_95,
            "verdict": verdict,
        }

    logger.info("")
    logger.info("=" * 96)
    logger.info(
        "EXPERIMENT 2 - THRESHOLD TRANSFER (target FPR %.0f%% set on the source)",
        100 * args.target_fpr,
    )
    logger.info("=" * 96)
    threshold_results: dict[str, Any] = {}
    for source in dataset_names:
        # Calibrate on the SOURCE's own test-split clean images using the
        # source's own model, which is what a deployed agent would have done.
        source_native = scored[(source, source)]
        threshold = threshold_at_target_fpr(source_native["clean"], args.target_fpr)
        source_fpr, source_recall = realised_rates(
            source_native["clean"], source_native["degraded"], threshold
        )
        logger.info(
            "  threshold from %-14s = %.4f  ->  on %-14s FPR %.1f%%  recall %.1f%%  (calibrated)",
            source, threshold, source, 100 * source_fpr, 100 * source_recall,
        )
        for target in dataset_names:
            if target == source:
                continue
            # Same source model, applied to the other modality, same threshold.
            transferred = scored[(source, target)]
            target_fpr, target_recall = realised_rates(
                transferred["clean"], transferred["degraded"], threshold
            )
            inflation = target_fpr / args.target_fpr if args.target_fpr else float("nan")
            logger.info(
                "  %-31s  ->  on %-14s FPR %.1f%%  recall %.1f%%  "
                "(FPR inflated %.1fx)",
                "", target, 100 * target_fpr, 100 * target_recall, inflation,
            )
            threshold_results[f"from_{source}_to_{target}"] = {
                "threshold": threshold,
                "source_realised_fpr": source_fpr,
                "source_recall": source_recall,
                "target_realised_fpr": target_fpr,
                "target_recall": target_recall,
                "fpr_inflation_factor": inflation,
                "recalibration_required": abs(target_fpr - args.target_fpr)
                > 0.5 * args.target_fpr,
            }
    results["threshold_transfer"] = threshold_results

    logger.info("")
    logger.info("=" * 96)
    logger.info("EXPERIMENT 3 - CLEAN-IMAGE SCORE DISTRIBUTIONS (why transfer works or not)")
    logger.info("=" * 96)
    distributions: dict[str, Any] = {}
    for target in dataset_names:
        native = scored[(target, target)]
        logger.info(
            "  %-14s native model on clean images: mean %.3f  sd %.3f  "
            "p95 %.3f",
            target, native["clean"].mean(), native["clean"].std(),
            np.percentile(native["clean"], 95),
        )
        distributions[target] = {
            "clean_mean": float(native["clean"].mean()),
            "clean_sd": float(native["clean"].std()),
            "clean_p95": float(np.percentile(native["clean"], 95)),
            "degraded_mean": float(native["degraded"].mean()),
        }
    results["clean_score_distributions"] = distributions

    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "datasets": dataset_names,
        "backbone": args.backbone,
        "feature_set": args.feature_set,
        "shared_iqa_metrics": shared_iqa,
        "shared_detectors": shared_detectors,
        "target_fpr": args.target_fpr,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "results": results,
        "coefficients_by_source": {
            name: models[name].describe().get("coefficients")
            for name in dataset_names
        },
        "protocol": (
            "Logistic fusion fitted on each modality's val split using only "
            "features present in both datasets, then applied unchanged to the "
            "other modality's test split. Transfer penalty tested by paired "
            "bootstrap on identical target rows. Thresholds calibrated to the "
            "target FPR on the source's clean images and applied raw to the "
            "other modality."
        ),
    }
    output_path = logs_dir / f"transfer_{'_'.join(dataset_names)}_{args.backbone}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    logger.info("")
    logger.info("Wrote transfer results to %s", output_path)


if __name__ == "__main__":
    main()