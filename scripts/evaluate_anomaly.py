#!/usr/bin/env python3
"""Fit anomaly detectors on clean training embeddings and evaluate per defect.

Produces the embedding-side half of the Stage 4 complementarity comparison, in
the same shape as ``evaluate_iqa.py``'s output so the two can be placed side by
side: AUROC/AUPRC/recall-at-5%-FPR per (detector x defect), plus severity
correlation for the defects that ``validate_ground_truth.py`` declared
SSIM-ordinal.

**Evaluation split.** Detectors are fitted on clean *train*-split embeddings, so
scoring the train split would be evaluating on fitted data. Evaluation therefore
defaults to ``val,test`` only. This matters for Stage 4: ``evaluate_iqa.py``
currently evaluates over all splits, so before comparing or fusing the two
paradigms, both must be restricted to the same rows. The rows actually used are
recorded in the summary JSON so the comparison can be checked rather than
assumed.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.common import configure_logging
    from pipeline.anomaly import DEFAULT_KNN_NEIGHBOURS, DetectorSuite
    from pipeline.embeddings import available_backbones, load_embeddings
    from pipeline.significance import bootstrap_auroc_ci
except ModuleNotFoundError:  # Allow `python scripts/evaluate_anomaly.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.anomaly import DEFAULT_KNN_NEIGHBOURS, DetectorSuite  # type: ignore[no-redef]
    from pipeline.embeddings import available_backbones, load_embeddings  # type: ignore[no-redef]
    from pipeline.significance import bootstrap_auroc_ci  # type: ignore[no-redef]

from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import average_precision_score


def build_scored_frame(
    processed: pd.DataFrame,
    image_ids: list[str],
    scores_by_detector: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Join detector scores back onto their metadata rows by image_id."""
    frame = pd.DataFrame({"image_id": image_ids})
    for name, values in scores_by_detector.items():
        frame[name] = values
    merged = frame.merge(
        processed[["image_id", "defect_type", "severity_level", "split"]],
        on="image_id",
        how="inner",
    )
    if len(merged) != len(frame):
        raise RuntimeError(
            f"{len(frame) - len(merged)} embedding row(s) had no matching metadata row"
        )
    return merged


def detection_metrics(
    scored: pd.DataFrame, detector: str, defect: str
) -> dict[str, Any]:
    """Clean-vs-one-defect AUROC (with bootstrap CI), AUPRC, and counts.

    Detector scores are already higher = more anomalous, so no orientation flip
    is needed - oriented_scores is called with higher_is_better=False purely to
    keep the code path identical to the IQA side.
    """
    clean = scored[scored["defect_type"] == "clean"]
    degraded = scored[scored["defect_type"] == defect]
    if clean.empty or degraded.empty:
        return {
            "auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "auprc": float("nan"), "n_clean": len(clean), "n_degraded": len(degraded),
        }
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(degraded))])
    # Detector scores are higher = more anomalous by construction (see
    # pipeline.anomaly), so no reorientation is needed on this side.
    oriented = np.concatenate([clean[detector].to_numpy(), degraded[detector].to_numpy()])
    ci = bootstrap_auroc_ci(labels, oriented)
    return {
        "auroc": ci.point, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
        "auprc": float(average_precision_score(labels, oriented)),
        "n_clean": len(clean), "n_degraded": len(degraded),
    }


def recall_at_fixed_fpr(
    scored: pd.DataFrame, detector: str, defect: str, target_fpr: float = 0.05
) -> float:
    """Recall on a defect at the threshold giving target_fpr on clean images."""
    clean = scored[scored["defect_type"] == "clean"]
    degraded = scored[scored["defect_type"] == defect]
    if clean.empty or degraded.empty:
        return float("nan")
    threshold = np.percentile(clean[detector].to_numpy(), 100 * (1 - target_fpr))
    return float(np.mean(degraded[detector].to_numpy() > threshold))


def severity_rho(scored: pd.DataFrame, detector: str, defect: str) -> float:
    """Spearman rho between an anomaly score and injected severity."""
    subset = scored[scored["defect_type"] == defect]
    severities = pd.to_numeric(subset["severity_level"], errors="coerce")
    valid = severities.notna()
    if valid.sum() < 3 or subset.loc[valid, detector].nunique() < 2:
        return float("nan")
    with warnings.catch_warnings():
        # A constant score column within one defect makes rho genuinely
        # undefined; nan is the correct answer and is returned below, so the
        # warning would only add noise to an otherwise readable log.
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        rho, _ = spearmanr(severities[valid], subset.loc[valid, detector])
    return float(rho) if np.isfinite(rho) else float("nan")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--backbone",
        default="all",
        help="Backbone to evaluate, or 'all' for every cached backbone",
    )
    parser.add_argument(
        "--eval-splits",
        default="val,test",
        help="Splits to evaluate on. Detectors are fitted on clean TRAIN only, so "
        "including 'train' here would evaluate on fitted data",
    )
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def evaluate_backbone(
    dataset: str,
    backbone: str,
    processed: pd.DataFrame,
    embeddings_dir: Path,
    eval_splits: list[str],
    eligible_defects: set[str],
    seed: int,
    logger,
) -> dict[str, Any]:
    cache_path = embeddings_dir / f"{dataset}_{backbone}.npz"
    if not cache_path.is_file():
        logger.warning("No embedding cache at %s; skipping %s", cache_path, backbone)
        return {}

    image_ids, embeddings, provenance = load_embeddings(cache_path)
    logger.info(
        "%s: loaded %s embeddings (dim=%d) from %s",
        backbone, embeddings.shape, embeddings.shape[1], cache_path,
    )

    lookup = processed.set_index("image_id")
    aligned = pd.DataFrame({"image_id": image_ids, "row": range(len(image_ids))})
    aligned = aligned[aligned["image_id"].isin(lookup.index)]
    aligned["defect_type"] = lookup.loc[aligned["image_id"], "defect_type"].to_numpy()
    aligned["split"] = lookup.loc[aligned["image_id"], "split"].to_numpy()

    fit_mask = (aligned["defect_type"] == "clean") & (aligned["split"] == "train")
    fit_rows = aligned.loc[fit_mask, "row"].to_numpy()
    if len(fit_rows) < 10:
        raise RuntimeError(
            f"only {len(fit_rows)} clean train embedding(s) available for {backbone}; "
            "too few to fit anomaly detectors. Increase --fraction-clean when "
            "running inject_defects.py, or use a larger dataset"
        )
    if len(fit_rows) < embeddings.shape[1]:
        logger.warning(
            "%s: only %d clean train embedding(s) for %d dimensions (n/d=%.2f). "
            "Mahalanobis is relying heavily on shrinkage; kNN and Isolation "
            "Forest are unaffected by dimensionality in the same way.",
            backbone, len(fit_rows), embeddings.shape[1],
            len(fit_rows) / embeddings.shape[1],
        )
    logger.info(
        "%s: fitting detectors on %d clean train embedding(s) "
        "(%.1f samples per dimension)",
        backbone, len(fit_rows), len(fit_rows) / embeddings.shape[1],
    )

    suite = DetectorSuite(random_state=seed).fit(embeddings[fit_rows])
    for name, description in suite.describe().items():
        if name == "mahalanobis":
            logger.info(
                "  mahalanobis: Ledoit-Wolf shrinkage=%.4f over %d dimensions",
                description["shrinkage_coefficient"], description["n_dimensions"],
            )

    eval_mask = aligned["split"].isin(eval_splits)
    eval_rows = aligned.loc[eval_mask, "row"].to_numpy()
    eval_ids = aligned.loc[eval_mask, "image_id"].tolist()
    scores = suite.score_all(embeddings[eval_rows])
    scored = build_scored_frame(processed, eval_ids, scores)
    logger.info(
        "%s: scored %d row(s) across splits %s", backbone, len(scored), eval_splits
    )

    detector_list = list(scores)
    defects = sorted(d for d in scored["defect_type"].unique() if d != "clean")

    auroc = pd.DataFrame(index=detector_list, columns=defects, dtype=float)
    auprc = pd.DataFrame(index=detector_list, columns=defects, dtype=float)
    recall5 = pd.DataFrame(index=detector_list, columns=defects, dtype=float)
    # CIs are captured in the same pass as AUROC so the bootstrap runs once per
    # cell, not once here and again during output. Storing them in the matrices
    # dict keeps the writing side agnostic to which matrix came from where.
    ci_low = pd.DataFrame(index=detector_list, columns=defects, dtype=float)
    ci_high = pd.DataFrame(index=detector_list, columns=defects, dtype=float)
    rho = pd.DataFrame(
        index=detector_list,
        columns=[d for d in defects if d in eligible_defects],
        dtype=float,
    )
    for detector in detector_list:
        for defect in defects:
            metrics = detection_metrics(scored, detector, defect)
            auroc.loc[detector, defect] = metrics["auroc"]
            auprc.loc[detector, defect] = metrics["auprc"]
            ci_low.loc[detector, defect] = metrics["ci_low"]
            ci_high.loc[detector, defect] = metrics["ci_high"]
            recall5.loc[detector, defect] = recall_at_fixed_fpr(scored, detector, defect)
            if defect in eligible_defects:
                rho.loc[detector, defect] = severity_rho(scored, detector, defect)

    logger.info("")
    logger.info("%s AUROC (clean vs degraded), higher = better detection:", backbone)
    for detector in detector_list:
        cells = "  ".join(f"{d}={auroc.loc[detector, d]:.3f}" for d in defects)
        logger.info("  %-18s %s", detector, cells)

    return {
        "backbone": backbone,
        "provenance": provenance,
        "detectors": suite.describe(),
        "n_clean_train_fitted": int(len(fit_rows)),
        "n_evaluated": int(len(scored)),
        "eval_splits": eval_splits,
        "matrices": {
            "auroc": auroc, "auprc": auprc,
            "auroc_ci_low": ci_low, "auroc_ci_high": ci_high,
            "recall_at_5pct_fpr": recall5, "severity_rho": rho,
        },
        "scored": scored,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_anomaly_evaluation.log"
    )
    dataset_dir = args.data_dir / args.dataset
    embeddings_dir = args.embeddings_dir or (args.data_dir / "embeddings")
    eval_splits = [part.strip() for part in args.eval_splits.split(",") if part.strip()]
    if "train" in eval_splits:
        logger.warning(
            "'train' is in --eval-splits, but detectors are FITTED on clean train "
            "embeddings; these results will be optimistically biased"
        )

    processed = pd.read_csv(
        dataset_dir / "processed_metadata.csv", dtype=str, keep_default_na=False
    )
    validation_path = args.data_dir / "logs" / f"{args.dataset}_ground_truth_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"{validation_path} not found; run validate_ground_truth.py first"
        )
    eligible_defects = set(
        json.loads(validation_path.read_text())["severity_correlation_eligible"]
    )
    logger.info("Severity-correlation-eligible defects: %s", sorted(eligible_defects))

    backbones = (
        available_backbones() if args.backbone == "all" else (args.backbone,)
    )
    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "eval_splits": eval_splits,
        "seed": args.seed,
        "note": (
            "Detectors fitted on clean train-split embeddings only. For Stage 4, "
            "restrict evaluate_iqa.py output to these same eval_splits before "
            "comparing or fusing."
        ),
        "backbones": {},
    }

    for backbone in backbones:
        result = evaluate_backbone(
            args.dataset, backbone, processed, embeddings_dir,
            eval_splits, eligible_defects, args.seed, logger,
        )
        if not result:
            continue
        for matrix_name, matrix in result["matrices"].items():
            matrix.to_csv(logs_dir / f"{args.dataset}_anomaly_{backbone}_{matrix_name}.csv")
        result["scored"].to_csv(
            dataset_dir / f"anomaly_scores_{backbone}.csv", index=False
        )
        auroc = result["matrices"]["auroc"]
        summary["backbones"][backbone] = {
            "provenance": result["provenance"],
            "detectors": result["detectors"],
            "n_clean_train_fitted": result["n_clean_train_fitted"],
            "n_evaluated": result["n_evaluated"],
            "best_detector_per_defect_by_auroc": {
                defect: str(auroc[defect].idxmax())
                for defect in auroc.columns
                if auroc[defect].notna().any()
            },
            "best_auroc_per_defect": {
                defect: float(auroc[defect].max())
                for defect in auroc.columns
                if auroc[defect].notna().any()
            },
        }

    summary_path = logs_dir / f"{args.dataset}_anomaly_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    logger.info("")
    logger.info("Wrote per-backbone matrices and summary to %s", logs_dir)


if __name__ == "__main__":
    main()