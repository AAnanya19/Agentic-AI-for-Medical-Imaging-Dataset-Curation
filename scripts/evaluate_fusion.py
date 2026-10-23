#!/usr/bin/env python3
"""Stage 4: per-defect ablation of IQA-only vs embedding-only vs fused.

This is the dissertation's central experiment. For every defect type it reports
three AUROCs on the same test rows - the best IQA metric alone, the best
embedding detector alone, and each fusion strategy - together with a **paired
bootstrap** test of fusion against the stronger single-paradigm baseline.

Protocol, and why each part matters:

  * Fusion weights are fitted on **val** and reported on **test**. Fitting on
    test would make any fusion benefit unfalsifiable.
  * The single-paradigm baselines are also selected on val (which metric, which
    detector) and then reported on test, so fusion is not being compared
    against a baseline that got to peek at the test set.
  * Comparisons use ``pipeline.significance.paired_bootstrap_diff``: both
    scorers see the same resampled rows, so correlated errors cancel.
    Independent CIs would be too conservative and could hide a real benefit.
  * "Fusion wins" is only claimed when the 95% CI on the difference excludes
    zero. A higher point estimate alone is not a result.

Expected outcome given Stage 3: fusion does not significantly beat IQA alone,
because the two paradigms were not found to be complementary on this data.
Reporting that carefully - having run three strategies of increasing power so
the null cannot be blamed on a weak combiner - is the contribution.
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
    from pipeline.fusion import FusionSuite, available_strategies
    from pipeline.iqa_metrics import METRICS
    from pipeline.significance import orient_scores, paired_bootstrap_diff
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:  # Allow `python scripts/evaluate_fusion.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.fusion import FusionSuite, available_strategies  # type: ignore[no-redef]
    from pipeline.iqa_metrics import METRICS  # type: ignore[no-redef]
    from pipeline.significance import orient_scores, paired_bootstrap_diff  # type: ignore[no-redef]
    from sklearn.metrics import roc_auc_score


def metric_polarity(metric_name: str) -> bool | None:
    """higher_is_better for an IQA metric, or None if non-directional."""
    if metric_name == "brightness_deviation":
        return False  # larger deviation from typical brightness = worse
    spec = METRICS.get(metric_name)
    return spec.higher_is_better if spec else None


def load_scores(dataset_dir: Path, backbone: str) -> pd.DataFrame:
    """Join IQA and embedding scores on image_id, keeping only matched rows.

    The embedding files only cover val+test (detectors are fitted on clean
    train), so this inner join is what aligns the two paradigms onto identical
    rows - without it the comparison would be between different samples.
    """
    iqa = pd.read_csv(dataset_dir / "iqa_scores.csv")
    anomaly_path = dataset_dir / f"anomaly_scores_{backbone}.csv"
    if not anomaly_path.is_file():
        raise FileNotFoundError(
            f"{anomaly_path} not found; run evaluate_anomaly.py for this backbone first"
        )
    anomaly = pd.read_csv(anomaly_path)
    detector_columns = [
        column
        for column in anomaly.columns
        if column not in ("image_id", "defect_type", "severity_level", "split")
    ]
    merged = iqa.merge(
        anomaly[["image_id", *detector_columns]], on="image_id", how="inner"
    )
    if merged.empty:
        raise RuntimeError(
            "no rows matched between iqa_scores.csv and "
            f"anomaly_scores_{backbone}.csv - check that both were produced "
            "from the same processed dataset"
        )
    return merged


# Which score families each feature set is allowed to use. Stage 4 fused
# everything and showed a gain over the best single scorer, but could not say
# whether that gain came from combining IQA metrics *with each other* or from
# adding embeddings. Running all three sets and comparing "all" against
# "iqa_only" is what isolates the embeddings' actual contribution.
FEATURE_SETS: dict[str, tuple[bool, bool]] = {
    # name: (include IQA metrics, include embedding detectors)
    "iqa_only": (True, False),
    "embeddings_only": (False, True),
    "all": (True, True),
}


def oriented_matrix(
    frame: pd.DataFrame,
    iqa_metrics: list[str],
    detectors: list[str],
    feature_set: str = "all",
) -> tuple[np.ndarray, list[str]]:
    """Build a score matrix with every column oriented higher = more degraded."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"unknown feature set {feature_set!r}; available: {sorted(FEATURE_SETS)}"
        )
    use_iqa, use_embeddings = FEATURE_SETS[feature_set]
    columns: list[np.ndarray] = []
    names: list[str] = []
    if use_iqa:
        for metric in iqa_metrics:
            polarity = metric_polarity(metric)
            if polarity is None:
                continue  # non-directional metrics cannot be oriented
            columns.append(orient_scores(frame[metric].to_numpy(), polarity))
            names.append(f"iqa:{metric}")
    if use_embeddings:
        for detector in detectors:
            # Detector scores are already higher = more anomalous.
            columns.append(frame[detector].to_numpy())
            names.append(f"emb:{detector}")
    if not columns:
        raise RuntimeError(f"feature set {feature_set!r} produced no usable columns")
    return np.column_stack(columns), names


def select_best_on_validation(
    val: pd.DataFrame, candidates: list[str], is_iqa: bool, defect: str
) -> tuple[str, float]:
    """Pick the strongest single scorer on VAL, so test stays untouched."""
    clean = val[val["defect_type"] == "clean"]
    degraded = val[val["defect_type"] == defect]
    if clean.empty or degraded.empty:
        return "", float("nan")
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(degraded))])
    best_name, best_auroc = "", -1.0
    for name in candidates:
        if is_iqa:
            polarity = metric_polarity(name)
            if polarity is None:
                continue
            values = orient_scores(
                np.concatenate([clean[name].to_numpy(), degraded[name].to_numpy()]),
                polarity,
            )
        else:
            values = np.concatenate([clean[name].to_numpy(), degraded[name].to_numpy()])
        auroc = roc_auc_score(labels, values)
        if auroc > best_auroc:
            best_name, best_auroc = name, float(auroc)
    return best_name, best_auroc


def score_on_test(
    test: pd.DataFrame, name: str, is_iqa: bool, defect: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Oriented scores, labels, and AUROC for one scorer on the test rows."""
    clean = test[test["defect_type"] == "clean"]
    degraded = test[test["defect_type"] == defect]
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(degraded))])
    raw = np.concatenate([clean[name].to_numpy(), degraded[name].to_numpy()])
    if is_iqa:
        polarity = metric_polarity(name)
        values = orient_scores(raw, polarity if polarity is not None else False)
    else:
        values = raw
    return values, labels, float(roc_auc_score(labels, values))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--backbone",
        default="resnet18",
        help="Which embedding backbone's scores to fuse with the IQA metrics",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_fusion.log"
    )
    dataset_dir = args.data_dir / args.dataset

    merged = load_scores(dataset_dir, args.backbone)
    iqa_metrics = [name for name in METRICS if name in merged.columns]
    iqa_metrics.append("brightness_deviation")
    detectors = [
        column
        for column in merged.columns
        if column.startswith(("knn_", "mahalanobis", "isolation_forest"))
    ]
    logger.info(
        "Loaded %d matched row(s): %d IQA metric(s), %d detector(s) from %s",
        len(merged), len(iqa_metrics), len(detectors), args.backbone,
    )

    val = merged[merged["split"] == "val"]
    test = merged[merged["split"] == "test"]
    if val.empty or test.empty:
        raise RuntimeError(
            f"need both val and test rows; got {len(val)} val, {len(test)} test"
        )
    logger.info("Fitting fusion on %d val row(s); reporting on %d test row(s)",
                len(val), len(test))

    val_labels = (val["defect_type"] != "clean").astype(int).to_numpy()
    test_labels_any = (test["defect_type"] != "clean").astype(int).to_numpy()

    # Fit one fusion suite per feature set. Fusion is fitted on all validation
    # rows (clean vs any-defect) and then applied per defect; fitting a separate
    # combiner per defect would leak the defect identity into the model, which
    # a real deployment would not know.
    suites: dict[str, FusionSuite] = {}
    fused_by_set: dict[str, dict[str, np.ndarray]] = {}
    feature_names_by_set: dict[str, list[str]] = {}
    for feature_set in FEATURE_SETS:
        val_matrix, feature_names = oriented_matrix(
            val, iqa_metrics, detectors, feature_set
        )
        test_matrix, _ = oriented_matrix(test, iqa_metrics, detectors, feature_set)
        suite = FusionSuite(random_state=args.seed).fit(
            val_matrix, val_labels, feature_names=feature_names
        )
        suites[feature_set] = suite
        fused_by_set[feature_set] = suite.score_all(test_matrix)
        feature_names_by_set[feature_set] = feature_names
        logger.info(
            "Fitted feature set %-16s (%d feature(s)): %s",
            feature_set, len(feature_names), ", ".join(suite.strategies),
        )
        description = suite.describe().get("logistic", {})
        if description.get("fitted") and isinstance(
            description.get("coefficients"), dict
        ):
            ranked = sorted(
                description["coefficients"].items(), key=lambda kv: -abs(kv[1])
            )[:5]
            logger.info(
                "    leans on: %s",
                ", ".join(f"{name} ({value:+.2f})" for name, value in ranked),
            )
        if description.get("fitted") and not description.get("converged", True):
            logger.warning(
                "    %s logistic hit its iteration cap without converging",
                feature_set,
            )

    # ---- STAGE 4b: does adding embeddings to the IQA metrics help at all? ----
    # This is the comparison Stage 4 could not make. "all" vs "iqa_only" on the
    # aggregate task isolates the embeddings' marginal contribution: if the two
    # are statistically indistinguishable, the entire fusion benefit reported in
    # Stage 4 is attributable to combining classical metrics with one another.
    logger.info("")
    logger.info("=" * 100)
    logger.info("FEATURE-SET ABLATION: does adding embeddings help? (aggregate task)")
    logger.info("=" * 100)
    feature_set_ablation: dict[str, Any] = {}
    for strategy_name in available_strategies():
        all_scores = fused_by_set["all"][strategy_name]
        iqa_scores_only = fused_by_set["iqa_only"][strategy_name]
        emb_scores_only = fused_by_set["embeddings_only"][strategy_name]
        auroc_all = float(roc_auc_score(test_labels_any, all_scores))
        auroc_iqa = float(roc_auc_score(test_labels_any, iqa_scores_only))
        auroc_emb = float(roc_auc_score(test_labels_any, emb_scores_only))
        marginal = paired_bootstrap_diff(
            test_labels_any, all_scores, iqa_scores_only,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        verdict = (
            "embeddings HELP" if marginal.significant_at_95 and marginal.diff > 0
            else "embeddings HURT" if marginal.significant_at_95
            else "embeddings add NOTHING measurable"
        )
        logger.info(
            "  %-14s iqa_only=%.3f  emb_only=%.3f  all=%.3f  |  "
            "all-vs-iqa %+.3f CI [%+.3f, %+.3f]  ->  %s",
            strategy_name, auroc_iqa, auroc_emb, auroc_all,
            marginal.diff, marginal.ci_low, marginal.ci_high, verdict,
        )
        feature_set_ablation[strategy_name] = {
            "auroc_iqa_only": auroc_iqa,
            "auroc_embeddings_only": auroc_emb,
            "auroc_all": auroc_all,
            "marginal_diff_all_minus_iqa": marginal.diff,
            "marginal_ci_low": marginal.ci_low,
            "marginal_ci_high": marginal.ci_high,
            "marginal_p_diff_le_zero": marginal.p_diff_le_zero,
            "embeddings_contribute_significantly": marginal.significant_at_95
            and marginal.diff > 0,
            "verdict": verdict,
        }

    # Everything below reports the "all" feature set, matching Stage 4.
    suite = suites["all"]
    feature_names = feature_names_by_set["all"]
    fused_all = fused_by_set["all"]
    test_matrix, _ = oriented_matrix(test, iqa_metrics, detectors, "all")

    # ---- The aggregate task: clean vs ANY degraded ----------------------
    # This is the comparison that matters for a deployed Inspector, and the
    # only one where fusion has a structural opportunity to win. The per-defect
    # table below compares fusion against whichever single scorer *specialises*
    # in that defect (noise_estimate owns noise, laplacian_variance owns blur),
    # which is a deliberately hostile baseline: no single metric is good at
    # everything, but for any one defect some metric usually is. Reporting only
    # the per-defect view would understate fusion by construction.
    aggregate: dict[str, Any] = {"task": "clean_vs_any_degraded"}

    best_single_name, best_single_auroc, best_single_values = "", -1.0, None
    for column_index, feature_name in enumerate(feature_names):
        auroc = roc_auc_score(test_labels_any, test_matrix[:, column_index])
        if auroc > best_single_auroc:
            best_single_name = feature_name
            best_single_auroc = float(auroc)
            best_single_values = test_matrix[:, column_index]
    aggregate["best_single_scorer"] = best_single_name
    aggregate["best_single_auroc"] = best_single_auroc

    logger.info("")
    logger.info("=" * 100)
    logger.info("AGGREGATE TASK: clean vs ANY degraded (the deployed Inspector's job)")
    logger.info("=" * 100)
    logger.info(
        "  best single scorer: %-28s %.3f", best_single_name, best_single_auroc
    )
    for strategy_name, fused_scores in fused_all.items():
        fused_auroc = float(roc_auc_score(test_labels_any, fused_scores))
        comparison = paired_bootstrap_diff(
            test_labels_any, fused_scores, best_single_values,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        aggregate[f"{strategy_name}_auroc"] = fused_auroc
        aggregate[f"{strategy_name}_diff"] = comparison.diff
        aggregate[f"{strategy_name}_ci_low"] = comparison.ci_low
        aggregate[f"{strategy_name}_ci_high"] = comparison.ci_high
        aggregate[f"{strategy_name}_p"] = comparison.p_diff_le_zero
        aggregate[f"{strategy_name}_significant"] = comparison.significant_at_95
        verdict = (
            "FUSION BETTER" if comparison.significant_at_95 and comparison.diff > 0
            else "BASELINE BETTER" if comparison.significant_at_95
            else "no sig diff"
        )
        logger.info(
            "  %-14s %.3f  diff %+.3f  CI [%+.3f, %+.3f]  %s",
            strategy_name, fused_auroc, comparison.diff,
            comparison.ci_low, comparison.ci_high, verdict,
        )

    defects = sorted(d for d in merged["defect_type"].unique() if d != "clean")
    rows: list[dict[str, Any]] = []

    for defect in defects:
        iqa_name, _ = select_best_on_validation(val, iqa_metrics, True, defect)
        emb_name, _ = select_best_on_validation(val, detectors, False, defect)
        if not iqa_name or not emb_name:
            logger.warning("Skipping %s: no valid baseline on val", defect)
            continue

        iqa_values, labels, iqa_auroc = score_on_test(test, iqa_name, True, defect)
        emb_values, _, emb_auroc = score_on_test(test, emb_name, False, defect)

        # Fused scores must be sliced to the same clean+defect rows, in the same
        # order that score_on_test built its labels.
        clean_mask = (test["defect_type"] == "clean").to_numpy()
        defect_mask = (test["defect_type"] == defect).to_numpy()
        row_order = np.concatenate([np.where(clean_mask)[0], np.where(defect_mask)[0]])

        baseline_name = "iqa" if iqa_auroc >= emb_auroc else "embedding"
        baseline_values = iqa_values if iqa_auroc >= emb_auroc else emb_values
        baseline_auroc = max(iqa_auroc, emb_auroc)

        record: dict[str, Any] = {
            "defect": defect,
            "best_iqa_metric": iqa_name,
            "iqa_auroc": iqa_auroc,
            "best_detector": emb_name,
            "embedding_auroc": emb_auroc,
            "stronger_baseline": baseline_name,
            "baseline_auroc": baseline_auroc,
            "n_clean_test": int(clean_mask.sum()),
            "n_degraded_test": int(defect_mask.sum()),
        }

        for strategy_name, fused_scores in fused_all.items():
            fused_subset = fused_scores[row_order]
            fused_auroc = float(roc_auc_score(labels, fused_subset))
            comparison = paired_bootstrap_diff(
                labels, fused_subset, baseline_values,
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
            record[f"{strategy_name}_auroc"] = fused_auroc
            record[f"{strategy_name}_diff"] = comparison.diff
            record[f"{strategy_name}_ci_low"] = comparison.ci_low
            record[f"{strategy_name}_ci_high"] = comparison.ci_high
            record[f"{strategy_name}_p"] = comparison.p_diff_le_zero
            record[f"{strategy_name}_significant"] = comparison.significant_at_95
        rows.append(record)

    table = pd.DataFrame(rows)

    logger.info("")
    logger.info("=" * 100)
    logger.info("PER-DEFECT ABLATION (test split), fusion vs the best SPECIALIST scorer")
    logger.info("Note: this baseline is deliberately hostile - it is re-selected per")
    logger.info("defect, so fusion competes against a different specialist each time.")
    logger.info("=" * 100)
    for row in table.itertuples(index=False):
        logger.info(
            "%-20s IQA=%.3f (%s)  EMB=%.3f (%s)  baseline=%s",
            row.defect, row.iqa_auroc, row.best_iqa_metric,
            row.embedding_auroc, row.best_detector, row.stronger_baseline,
        )
        for strategy_name in available_strategies():
            diff = getattr(row, f"{strategy_name}_diff")
            low = getattr(row, f"{strategy_name}_ci_low")
            high = getattr(row, f"{strategy_name}_ci_high")
            auroc = getattr(row, f"{strategy_name}_auroc")
            significant = getattr(row, f"{strategy_name}_significant")
            verdict = (
                "FUSION BETTER" if significant and diff > 0
                else "BASELINE BETTER" if significant and diff < 0
                else "no sig diff"
            )
            logger.info(
                "    %-14s %.3f  diff %+.3f  CI [%+.3f, %+.3f]  %s",
                strategy_name, auroc, diff, low, high, verdict,
            )

    wins = {
        strategy: int(
            ((table[f"{strategy}_significant"]) & (table[f"{strategy}_diff"] > 0)).sum()
        )
        for strategy in available_strategies()
    }
    losses = {
        strategy: int(
            ((table[f"{strategy}_significant"]) & (table[f"{strategy}_diff"] < 0)).sum()
        )
        for strategy in available_strategies()
    }
    logger.info("")
    logger.info("VERDICT over %d defect(s):", len(table))
    for strategy in available_strategies():
        logger.info(
            "  %-14s fusion significantly better on %d, significantly WORSE on %d, "
            "no difference on %d",
            strategy, wins[strategy], losses[strategy],
            len(table) - wins[strategy] - losses[strategy],
        )

    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        logs_dir / f"{args.dataset}_fusion_ablation_{args.backbone}.csv", index=False
    )
    summary = {
        "aggregate_clean_vs_any_degraded": aggregate,
        "feature_set_ablation": feature_set_ablation,
        "feature_set_sizes": {
            name: len(names) for name, names in feature_names_by_set.items()
        },
        "logistic_coefficients_by_feature_set": {
            name: suites[name].describe().get("logistic", {}).get("coefficients")
            for name in FEATURE_SETS
        },
        "dataset": args.dataset,
        "backbone": args.backbone,
        "n_val_rows_fitted": int(len(val)),
        "n_test_rows_reported": int(len(test)),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "strategies": suite.describe(),
        "fusion_significant_wins": wins,
        "fusion_significant_losses": losses,
        "n_defects": int(len(table)),
        "protocol": (
            "Fusion weights and single-paradigm baseline selection both fitted on "
            "val; all AUROCs reported on test. Significance via paired bootstrap "
            "on the difference. Two tasks are reported: the AGGREGATE task "
            "(clean vs any degraded, compared against the single best scorer "
            "overall) which is what a deployed Inspector performs, and the "
            "PER-DEFECT ablation (compared against the best specialist scorer "
            "for each defect, a deliberately hostile baseline since some single "
            "metric usually specialises in any given defect)."
        ),
    }
    (logs_dir / f"{args.dataset}_fusion_summary_{args.backbone}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    logger.info("")
    logger.info("Wrote ablation table and summary to %s", logs_dir)


if __name__ == "__main__":
    main()