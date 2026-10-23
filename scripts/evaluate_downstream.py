#!/usr/bin/env python3
"""Downstream evaluation: does quality curation improve a diagnostic classifier?

Every result up to this point measures how well the system detects degradation.
None shows that acting on those detections improves anything. This script closes
that gap with two experiments.

**Experiment 1 - performance by inspection verdict.** A classifier is trained on
clean training images and evaluated separately on the test images the Inspector
accepted and those it did not. This shows the gap directly.

Its weakness is circularity, stated here rather than buried: the rejected images
are overwhelmingly ones this project degraded synthetically, so a large gap
partly re-measures that a blurred image is harder to classify, which is true by
construction. It is reported because it establishes that the verdicts track
downstream difficulty at all, not as evidence that curation helps.

**Experiment 2 - curated against uncurated training.** The same classifier is
trained three times: on the full training split, on the Inspector-accepted
subset, and on a random subset matched in size to the accepted one. All three
are evaluated on the same clean test images.

The size-matched control is what makes this interpretable. Without it an
improvement cannot be distinguished from the effect of training on fewer images,
and a degradation cannot be distinguished from having discarded useful data. The
comparison of interest is accepted against size-matched, not accepted against
full.

**Classifier.** Logistic regression on frozen backbone features, the standard
linear-probe protocol. Training uses precomputed embeddings, so each fit takes
seconds and the control can be repeated across seeds without GPU time. The
absolute performance of such a probe is modest; the comparison between
conditions is the result, not the absolute figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    from scripts.common import configure_logging
    from pipeline.significance import bootstrap_auroc_ci, paired_bootstrap_diff
except ModuleNotFoundError:  # Allow `python scripts/evaluate_downstream.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.significance import (  # type: ignore[no-redef]
        bootstrap_auroc_ci,
        paired_bootstrap_diff,
    )


DEFAULT_LABEL_COLUMNS = {
    "chestxray14": "source_labels",
    "isic": "diagnosis",
}
# ChestX-ray14 marks images with no pathology using this exact string, so the
# binary task is its presence or absence.
NEGATIVE_LABELS = {"No Finding", "no finding", ""}


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return list(payload["image_ids"]), payload["embeddings"]


def build_binary_labels(values: pd.Series, dataset: str) -> tuple[np.ndarray, str]:
    """Reduce the diagnostic label to a binary target, with its description.

    ChestX-ray14 is multi-label across fifteen strata, several containing fewer
    than twenty images, so a multi-label probe would estimate some classes from
    almost no data. The binary presence-of-any-finding task is stable at this
    sample size. The choice limits the downstream evaluation rather than the
    curation system, and is stated as such.
    """
    if dataset == "chestxray14":
        positive = ~values.fillna("").astype(str).isin(NEGATIVE_LABELS)
        return positive.to_numpy().astype(int), "any finding against no finding"
    majority = values.value_counts().idxmax()
    positive = (values != majority).to_numpy().astype(int)
    return positive, f"any diagnosis other than {majority}"


def fit_and_score(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Fit a linear probe and score it."""
    if len(np.unique(train_labels)) < 2:
        return {"error": "training subset contains a single class"}
    if len(np.unique(test_labels)) < 2:
        return {"error": "evaluation subset contains a single class"}

    # Standardisation is fitted on the training subset only, so each condition is
    # a self-contained model rather than sharing statistics with the others.
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        max_iter=2000, random_state=seed, class_weight="balanced"
    ).fit(scaler.transform(train_features), train_labels)

    scores = model.predict_proba(scaler.transform(test_features))[:, 1]
    predictions = (scores >= 0.5).astype(int)
    interval = bootstrap_auroc_ci(test_labels, scores, n_bootstrap=1000, seed=seed)
    return {
        "n_train": int(len(train_labels)),
        "n_train_positive": int(train_labels.sum()),
        "n_test": int(len(test_labels)),
        "auroc": interval.point,
        "auroc_ci_low": interval.ci_low,
        "auroc_ci_high": interval.ci_high,
        "average_precision": float(average_precision_score(test_labels, scores)),
        "f1": float(f1_score(test_labels, predictions, zero_division=0)),
        "scores": scores,
        "labels": test_labels,
    }


def strip_arrays(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k not in ("scores", "labels")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--label-column")
    parser.add_argument(
        "--seeds", default="0,1,2,3,4",
        help="Seeds for the size-matched control; its results are averaged",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_downstream.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"

    embedding_path = (
        args.data_dir / "embeddings" / f"{args.dataset}_{args.backbone}.npz"
    )
    if not embedding_path.is_file():
        raise FileNotFoundError(
            f"{embedding_path} not found; run extract_embeddings.py first"
        )
    image_ids, embeddings = load_embeddings(embedding_path)
    logger.info(
        "Loaded %d embedding(s) of dimension %d", len(image_ids), embeddings.shape[1]
    )

    processed = pd.read_csv(
        dataset_dir / "processed_metadata.csv", dtype=str, keep_default_na=False
    )
    source = pd.read_csv(
        dataset_dir / "metadata.csv", dtype=str, keep_default_na=False
    )
    label_column = args.label_column or DEFAULT_LABEL_COLUMNS.get(args.dataset)
    if label_column not in source.columns:
        raise ValueError(
            f"label column {label_column!r} not in metadata.csv; available: "
            f"{sorted(source.columns)}"
        )

    index = {image_id: position for position, image_id in enumerate(image_ids)}
    labels_by_source = dict(zip(source["image_id"], source[label_column]))

    frame = processed[["image_id", "original_image_id", "split", "defect_type"]].copy()
    frame["row"] = frame["image_id"].map(index)
    frame["raw_label"] = frame["original_image_id"].map(labels_by_source)
    before = len(frame)
    frame = frame[frame["row"].notna() & frame["raw_label"].notna()].copy()
    frame["row"] = frame["row"].astype(int)
    if len(frame) < before:
        logger.warning(
            "%d row(s) dropped for want of an embedding or a label", before - len(frame)
        )

    targets, task = build_binary_labels(frame["raw_label"], args.dataset)
    frame["target"] = targets
    logger.info("Downstream task: %s", task)
    logger.info(
        "Positive rate %.3f across %d image(s)", frame["target"].mean(), len(frame)
    )

    inspection_path = logs_dir / f"{args.dataset}_inspection.json"
    if not inspection_path.is_file():
        raise FileNotFoundError(
            f"{inspection_path} not found; run run_pipeline.py first"
        )
    inspection = json.loads(inspection_path.read_text())
    verdicts = {image["image_id"]: image["decision"] for image in inspection["images"]}
    frame["decision"] = frame["image_id"].map(verdicts)
    logger.info(
        "Inspection verdicts on %d of %d row(s); split coverage %s",
        int(frame["decision"].notna().sum()), len(frame),
        frame.loc[frame["decision"].notna(), "split"].value_counts().to_dict(),
    )

    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    features = embeddings
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    results: dict[str, Any] = {
        "dataset": args.dataset, "backbone": args.backbone, "task": task,
        "label_column": label_column,
    }

    # ============================================== EXPERIMENT 1 =========
    logger.info("")
    logger.info("=" * 92)
    logger.info("EXPERIMENT 1 - classifier performance by inspection verdict")
    logger.info("=" * 92)

    clean_train = train[train["defect_type"] == "clean"]
    scored_test = test[test["decision"].notna()]
    accepted = scored_test[scored_test["decision"] == "accept"]
    actioned = scored_test[scored_test["decision"] != "accept"]

    if clean_train.empty or accepted.empty or actioned.empty:
        logger.warning(
            "  skipped: need clean training rows and both verdict groups in test; "
            "have %d, %d, %d", len(clean_train), len(accepted), len(actioned),
        )
    else:
        accepted_result = fit_and_score(
            features[clean_train["row"]], clean_train["target"].to_numpy(),
            features[accepted["row"]], accepted["target"].to_numpy(), seeds[0],
        )
        actioned_result = fit_and_score(
            features[clean_train["row"]], clean_train["target"].to_numpy(),
            features[actioned["row"]], actioned["target"].to_numpy(), seeds[0],
        )
        for name, result in (("accepted", accepted_result),
                             ("flagged or rejected", actioned_result)):
            if "error" in result:
                logger.warning("  %s: %s", name, result["error"])
                continue
            logger.info(
                "  %-20s n=%4d  AUROC %.3f [%.3f, %.3f]  AP %.3f  F1 %.3f",
                name, result["n_test"], result["auroc"], result["auroc_ci_low"],
                result["auroc_ci_high"], result["average_precision"], result["f1"],
            )
        if "error" not in accepted_result and "error" not in actioned_result:
            gap = accepted_result["auroc"] - actioned_result["auroc"]
            logger.info("")
            logger.info("  difference in AUROC, accepted minus actioned: %+.3f", gap)
            logger.info(
                "  NOTE: the two groups are different images, so this is an "
                "unpaired comparison. Most actioned images carry an injected "
                "defect, so the gap partly reflects that degraded images are "
                "harder to classify by construction."
            )
        results["experiment_1"] = {
            "accepted": strip_arrays(accepted_result),
            "actioned": strip_arrays(actioned_result),
            "caveat": (
                "unpaired comparison between different image sets; actioned "
                "images are predominantly synthetically degraded, so the gap is "
                "partly true by construction"
            ),
        }

    # ============================================== EXPERIMENT 2 =========
    logger.info("")
    logger.info("=" * 92)
    logger.info("EXPERIMENT 2 - curated against uncurated training data")
    logger.info("=" * 92)

    scored_train = train[train["decision"].notna()]
    if scored_train.empty:
        logger.error(
            "  cannot run: no inspection verdicts on the training split. Run the "
            "inspector over that split so the training images carry decisions:\n"
            "    python scripts/run_inspector.py --dataset %s --data-dir %s "
            "--eval-split train", args.dataset, args.data_dir,
        )
        results["experiment_2"] = {"skipped": "no training-split verdicts"}
    else:
        clean_test = test[test["defect_type"] == "clean"]
        if clean_test.empty:
            logger.error("  cannot run: no clean test rows to evaluate on")
            results["experiment_2"] = {"skipped": "no clean test rows"}
        else:
            accepted_train = scored_train[scored_train["decision"] == "accept"]
            logger.info(
                "  training pools: full %d, accepted %d (%.1f%% retained); "
                "evaluating on %d clean test image(s)",
                len(scored_train), len(accepted_train),
                100 * len(accepted_train) / len(scored_train), len(clean_test),
            )

            test_features = features[clean_test["row"]]
            test_targets = clean_test["target"].to_numpy()

            full_result = fit_and_score(
                features[scored_train["row"]], scored_train["target"].to_numpy(),
                test_features, test_targets, seeds[0],
            )
            accepted_result = fit_and_score(
                features[accepted_train["row"]], accepted_train["target"].to_numpy(),
                test_features, test_targets, seeds[0],
            )

            # Repeated over several seeds, because one random subset of this size
            # is itself a noisy draw and a comparison against a single draw would
            # not be interpretable.
            matched_results = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                chosen = rng.choice(
                    len(scored_train), size=len(accepted_train), replace=False
                )
                subset = scored_train.iloc[chosen]
                outcome = fit_and_score(
                    features[subset["row"]], subset["target"].to_numpy(),
                    test_features, test_targets, seed,
                )
                if "error" not in outcome:
                    matched_results.append(outcome)

            logger.info("")
            for name, result in (("full training set", full_result),
                                 ("accepted only", accepted_result)):
                if "error" in result:
                    logger.warning("  %s: %s", name, result["error"])
                    continue
                logger.info(
                    "  %-22s n=%4d  AUROC %.3f [%.3f, %.3f]  AP %.3f  F1 %.3f",
                    name, result["n_train"], result["auroc"], result["auroc_ci_low"],
                    result["auroc_ci_high"], result["average_precision"], result["f1"],
                )
            if matched_results:
                aurocs = [r["auroc"] for r in matched_results]
                logger.info(
                    "  %-22s n=%4d  AUROC %.3f (sd %.3f over %d seed(s))",
                    "size-matched random", matched_results[0]["n_train"],
                    float(np.mean(aurocs)), float(np.std(aurocs)), len(matched_results),
                )

            if matched_results and "error" not in accepted_result:
                order = np.argsort([r["auroc"] for r in matched_results])
                reference = matched_results[order[len(matched_results) // 2]]
                comparison = paired_bootstrap_diff(
                    test_targets, accepted_result["scores"], reference["scores"],
                    n_bootstrap=args.n_bootstrap, seed=seeds[0],
                )
                verdict = (
                    "curation HELPS"
                    if comparison.significant_at_95 and comparison.diff > 0
                    else "curation HURTS" if comparison.significant_at_95
                    else "no significant effect of curation"
                )
                logger.info("")
                logger.info(
                    "  accepted minus size-matched: %+.3f, 95%% CI [%+.3f, %+.3f]"
                    "  ->  %s",
                    comparison.diff, comparison.ci_low, comparison.ci_high, verdict,
                )
                logger.info(
                    "  This is the comparison that isolates curation from sample "
                    "size, both conditions having the same number of training "
                    "images."
                )
                results["experiment_2"] = {
                    "full": strip_arrays(full_result),
                    "accepted": strip_arrays(accepted_result),
                    "size_matched_mean_auroc": float(np.mean(aurocs)),
                    "size_matched_sd_auroc": float(np.std(aurocs)),
                    "size_matched_n_seeds": len(matched_results),
                    "accepted_minus_matched_diff": comparison.diff,
                    "accepted_minus_matched_ci_low": comparison.ci_low,
                    "accepted_minus_matched_ci_high": comparison.ci_high,
                    "significant": comparison.significant_at_95,
                    "verdict": verdict,
                    "retention_rate": len(accepted_train) / len(scored_train),
                }

    output_path = logs_dir / f"{args.dataset}_downstream_{args.backbone}.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n"
    )
    logger.info("")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()