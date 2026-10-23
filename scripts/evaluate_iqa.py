#!/usr/bin/env python3
"""Evaluate scored IQA metrics: severity correlation, AUROC, per-defect matrix.

Reads ``<dataset>/iqa_scores.csv`` (from compute_iqa.py) and the ground-truth
validation report (from validate_ground_truth.py, which declares which defects
have SSIM-ordinal severity and are therefore eligible for correlation analysis
- rotation_crop is excluded here for the same reason it was excluded there).

Every comparison respects each metric's registered ``higher_is_better``
polarity, so a metric that moves the "wrong" way (e.g. Laplacian variance
rising under noise, because noise looks like high-frequency detail to a
derivative-based sharpness proxy - confirmed during development against a real
photograph) is reported as a low AUROC rather than silently inverted or hidden.
That is the result, not a bug in the evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from scripts.common import configure_logging
    from pipeline.iqa_metrics import METRICS
    from pipeline.significance import bootstrap_auroc_ci, orient_scores
except ModuleNotFoundError:  # Allow `python scripts/evaluate_iqa.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.iqa_metrics import METRICS  # type: ignore[no-redef]
    from pipeline.significance import bootstrap_auroc_ci, orient_scores  # type: ignore[no-redef]


def metric_polarity(metric_name: str) -> bool | None:
    """higher_is_better for a registered metric, or None for brightness_deviation."""
    if metric_name == "brightness_deviation":
        return False  # larger deviation from typical brightness = worse
    return METRICS[metric_name].higher_is_better


def oriented_scores(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Deprecated alias for pipeline.significance.orient_scores. Kept because
    external callers (e.g. Stage 4 experiments not in this repo yet) may still
    import this symbol; new code should use orient_scores directly."""
    return orient_scores(scores, higher_is_better)


def severity_correlation(
    scored: pd.DataFrame, metric_name: str, defect_name: str
) -> dict[str, Any]:
    """Spearman rho between a metric and injected severity, for one defect."""
    subset = scored[scored["defect_type"] == defect_name]
    severities = pd.to_numeric(subset["severity_level"], errors="coerce")
    valid = severities.notna()
    if valid.sum() < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(valid.sum())}
    rho, p_value = spearmanr(severities[valid], subset.loc[valid, metric_name])
    return {"rho": float(rho), "p_value": float(p_value), "n": int(valid.sum())}


def binary_detection(
    scored: pd.DataFrame, metric_name: str, defect_name: str
) -> dict[str, Any]:
    """Clean-vs-degraded AUROC/AUPRC for one metric on one defect, with a CI."""
    clean = scored[scored["defect_type"] == "clean"]
    degraded = scored[scored["defect_type"] == defect_name]
    if clean.empty or degraded.empty:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "auprc": float("nan"), "n_clean": len(clean), "n_degraded": len(degraded)}

    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(degraded))])
    raw_scores = np.concatenate(
        [clean[metric_name].to_numpy(), degraded[metric_name].to_numpy()]
    )
    higher_is_better = metric_polarity(metric_name)
    if higher_is_better is None:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "auprc": float("nan"), "n_clean": len(clean), "n_degraded": len(degraded),
                "note": "no directional polarity registered"}
    oriented = orient_scores(raw_scores, higher_is_better)
    ci = bootstrap_auroc_ci(labels, oriented)
    auprc = float(average_precision_score(labels, oriented))
    return {
        "auroc": ci.point, "ci_low": ci.ci_low, "ci_high": ci.ci_high, "auprc": auprc,
        "n_clean": len(clean), "n_degraded": len(degraded),
    }


def recall_at_fixed_fpr(
    scored: pd.DataFrame, metric_name: str, defect_name: str, target_fpr: float = 0.05
) -> float:
    """Recall on a defect at the threshold giving target_fpr on clean images."""
    clean = scored[scored["defect_type"] == "clean"]
    degraded = scored[scored["defect_type"] == defect_name]
    higher_is_better = metric_polarity(metric_name)
    if clean.empty or degraded.empty or higher_is_better is None:
        return float("nan")
    clean_oriented = oriented_scores(clean[metric_name].to_numpy(), higher_is_better)
    degraded_oriented = oriented_scores(degraded[metric_name].to_numpy(), higher_is_better)
    threshold = np.percentile(clean_oriented, 100 * (1 - target_fpr))
    return float(np.mean(degraded_oriented > threshold))


def build_matrices(
    scored: pd.DataFrame,
    metric_names: list[str],
    defect_names: list[str],
    eligible_defects: set[str],
) -> dict[str, pd.DataFrame]:
    """AUROC, AUPRC, recall@5%FPR, and severity-rho matrices, metric x defect."""
    auroc = pd.DataFrame(index=metric_names, columns=defect_names, dtype=float)
    auprc = pd.DataFrame(index=metric_names, columns=defect_names, dtype=float)
    recall5 = pd.DataFrame(index=metric_names, columns=defect_names, dtype=float)
    rho = pd.DataFrame(
        index=metric_names,
        columns=[d for d in defect_names if d in eligible_defects],
        dtype=float,
    )

    for metric_name in metric_names:
        for defect_name in defect_names:
            detection = binary_detection(scored, metric_name, defect_name)
            auroc.loc[metric_name, defect_name] = detection["auroc"]
            auprc.loc[metric_name, defect_name] = detection["auprc"]
            recall5.loc[metric_name, defect_name] = recall_at_fixed_fpr(
                scored, metric_name, defect_name
            )
            if defect_name in eligible_defects:
                rho.loc[metric_name, defect_name] = severity_correlation(
                    scored, metric_name, defect_name
                )["rho"]
    return {"auroc": auroc, "auprc": auprc, "recall_at_5pct_fpr": recall5, "severity_rho": rho}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--eval-splits",
        default="all",
        help="Comma-separated splits to evaluate on, or 'all' (default). Stage 2 "
        "metrics need no fitting so 'all' is valid here, but Stage 3's detectors "
        "are fitted on clean train and therefore evaluate on val,test only. "
        "Pass --eval-splits val,test to produce IQA numbers directly comparable "
        "to the anomaly-detection results before fusing them in Stage 4",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Appended to output filenames, so a restricted-split run does not "
        "overwrite the full-split matrices",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_iqa_evaluation.log"
    )
    dataset_dir = args.data_dir / args.dataset

    scored = pd.read_csv(dataset_dir / "iqa_scores.csv", dtype={"severity_level": str})
    if args.eval_splits.strip().lower() != "all":
        requested = [part.strip() for part in args.eval_splits.split(",") if part.strip()]
        before = len(scored)
        scored = scored[scored["split"].isin(requested)]
        logger.info(
            "Restricted evaluation to splits %s: %d of %d row(s)",
            requested, len(scored), before,
        )
        if scored.empty:
            raise ValueError(f"no rows remain after filtering to splits {requested}")
    else:
        logger.info("Evaluating on all splits (%d rows)", len(scored))
    validation_path = args.data_dir / "logs" / f"{args.dataset}_ground_truth_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"{validation_path} not found; run validate_ground_truth.py first so "
            "eligible-for-correlation defects are read from data, not hardcoded"
        )
    validation = json.loads(validation_path.read_text())
    eligible_defects = set(validation["severity_correlation_eligible"])
    logger.info("Severity-correlation-eligible defects: %s", sorted(eligible_defects))

    metric_names = [name for name in METRICS if name in scored.columns]
    metric_names.append("brightness_deviation")
    defect_names = sorted(d for d in scored["defect_type"].unique() if d != "clean")
    logger.info("Metrics: %s", metric_names)
    logger.info("Defects: %s", defect_names)

    matrices = build_matrices(scored, metric_names, defect_names, eligible_defects)

    logger.info("")
    logger.info("AUROC (clean vs degraded), oriented so higher = correctly detects degradation:")
    for metric_name in metric_names:
        cells = "  ".join(
            f"{defect}={matrices['auroc'].loc[metric_name, defect]:.3f}"
            for defect in defect_names
        )
        logger.info("  %-20s %s", metric_name, cells)

    logger.info("")
    logger.info("Spearman rho vs severity (eligible defects only):")
    for metric_name in metric_names:
        cells = "  ".join(
            f"{defect}={matrices['severity_rho'].loc[metric_name, defect]:.3f}"
            for defect in defect_names
            if defect in eligible_defects
        )
        logger.info("  %-20s %s", metric_name, cells)

    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ci_low_matrix = matrices["auroc"].copy() * float("nan")
    ci_high_matrix = matrices["auroc"].copy() * float("nan")
    for metric_name in metric_names:
        for defect_name in defect_names:
            # binary_detection runs a full bootstrap; running it twice would
            # double the wall time for no additional information.
            detection = binary_detection(scored, metric_name, defect_name)
            ci_low_matrix.loc[metric_name, defect_name] = detection["ci_low"]
            ci_high_matrix.loc[metric_name, defect_name] = detection["ci_high"]
    matrices["auroc_ci_low"] = ci_low_matrix
    matrices["auroc_ci_high"] = ci_high_matrix

    for matrix_name, matrix in matrices.items():
        matrix.to_csv(logs_dir / f"{args.dataset}_iqa_{matrix_name}{args.output_suffix}.csv")
    logger.info("")
    logger.info("Wrote matrices (auroc, auprc, recall_at_5pct_fpr, severity_rho) to %s", logs_dir)

    # A per-metric summary makes the sanity-check gate ("Laplacian variance
    # should have |rho| > 0.7 against blur") a single number to look up.
    summary_path = logs_dir / f"{args.dataset}_iqa_summary{args.output_suffix}.json"
    summary = {
        "eval_splits": args.eval_splits,
        "n_rows_evaluated": int(len(scored)),
        "metrics": metric_names,
        "defects": defect_names,
        "severity_correlation_eligible": sorted(eligible_defects),
        "best_metric_per_defect_by_auroc": {
            defect: matrices["auroc"][defect].idxmax()
            for defect in defect_names
            if matrices["auroc"][defect].notna().any()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()