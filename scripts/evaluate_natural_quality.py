#!/usr/bin/env python3
"""Experiment 3: does the quality score predict difficulty on images that were
never synthetically degraded?

Experiments 1 and 2 both compare groups that differ in whether this project
injected a defect into them, so part of any gap they measure is true by
construction. This experiment removes that confound entirely by restricting
attention to images with no injected defect, and asking whether the quality
assessment still carries information about downstream difficulty.

Two forms of the question are reported, because they trade power against
directness.

**Form A, the binary comparison.** Among images with no injected defect, compare
classifier performance on those the Inspector accepted against those it flagged
or rejected. This is the most direct statement of the hypothesis, and it is
badly underpowered: the Inspector flags roughly five per cent of clean images by
construction, so on a corpus of this size the flagged group numbers in the tens.
The result is reported with its confidence interval and its detection floor
rather than being presented as conclusive.

**Form B, the continuous association.** Among the same images, test whether the
continuous quality score predicts which ones the classifier gets wrong. This
uses every clean image rather than only the flagged minority, so it is far
better powered, and it tests the same underlying claim: that the quality
assessment carries information about downstream difficulty independent of any
injected degradation. The association is measured both as the area under the
curve for the quality score predicting misclassification, and as a rank
correlation against the classifier's probability of the true class.

**Interpretation.** A positive result in Form B would indicate that the system
detects naturally occurring quality variation, not merely the defects it was
shown. A null result is equally interpretable, and would indicate that the
scores carry no downstream signal beyond the synthetic degradations, which is a
meaningful limitation to report.

Clean images flagged by the Inspector are not verified to be genuinely poor;
distinguishing a real acquisition artefact from a threshold crossing at the
configured false-positive rate would require expert review that this project
does not include. The experiment therefore tests association, not detection
accuracy, and is described in those terms throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from scripts.common import configure_logging
    from pipeline.significance import bootstrap_auroc_ci
except ModuleNotFoundError:  # Allow direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.significance import bootstrap_auroc_ci  # type: ignore[no-redef]


DEFAULT_LABEL_COLUMNS = {"chestxray14": "source_labels", "isic": "diagnosis"}
NEGATIVE_LABELS = {"No Finding", "no finding", ""}


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return list(payload["image_ids"]), payload["embeddings"]


def build_binary_labels(values: pd.Series, dataset: str) -> tuple[np.ndarray, str]:
    if dataset == "chestxray14":
        positive = ~values.fillna("").astype(str).isin(NEGATIVE_LABELS)
        return positive.to_numpy().astype(int), "any finding against no finding"
    majority = values.value_counts().idxmax()
    return (values != majority).to_numpy().astype(int), (
        f"any diagnosis other than {majority}"
    )


def read_inspections(paths: list[Path]) -> tuple[dict[str, str], dict[str, float]]:
    """Merge verdicts and composite scores from one or more inspection files.

    The Inspector writes to a fixed filename per dataset, so evaluating several
    splits requires several runs. Accepting a list here lets the clean images of
    the validation and test splits be pooled, which roughly doubles the size of
    the flagged group that Form A depends on.
    """
    verdicts: dict[str, str] = {}
    scores: dict[str, float] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for image in payload["images"]:
            verdicts[image["image_id"]] = image["decision"]
            entry = image.get("checks", {}).get("overall_quality", {})
            value = entry.get("oriented_score", entry.get("score"))
            if value is not None and np.isfinite(value):
                scores[image["image_id"]] = float(value)
    return verdicts, scores


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--label-column")
    parser.add_argument(
        "--inspections", required=True,
        help="Comma-separated inspection JSON files covering the evaluation "
        "splits, for example "
        "logs/chestxray14_inspection_val.json,logs/chestxray14_inspection_test.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_natural_quality.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"

    image_ids, embeddings = load_embeddings(
        args.data_dir / "embeddings" / f"{args.dataset}_{args.backbone}.npz"
    )
    processed = pd.read_csv(
        dataset_dir / "processed_metadata.csv", dtype=str, keep_default_na=False
    )
    source = pd.read_csv(
        dataset_dir / "metadata.csv", dtype=str, keep_default_na=False
    )
    label_column = args.label_column or DEFAULT_LABEL_COLUMNS.get(args.dataset)

    inspection_paths = [
        Path(part.strip()) for part in args.inspections.split(",") if part.strip()
    ]
    inspection_paths = [
        path if path.is_absolute() or path.exists() else args.data_dir / path
        for path in inspection_paths
    ]
    missing = [path for path in inspection_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"inspection file(s) not found: {missing}")
    verdicts, quality = read_inspections(inspection_paths)
    logger.info(
        "Pooled %d verdict(s) and %d composite score(s) from %d inspection file(s)",
        len(verdicts), len(quality), len(inspection_paths),
    )

    index = {image_id: position for position, image_id in enumerate(image_ids)}
    labels_by_source = dict(zip(source["image_id"], source[label_column]))

    frame = processed[["image_id", "original_image_id", "split", "defect_type"]].copy()
    frame["row"] = frame["image_id"].map(index)
    frame["raw_label"] = frame["original_image_id"].map(labels_by_source)
    frame = frame[frame["row"].notna() & frame["raw_label"].notna()].copy()
    frame["row"] = frame["row"].astype(int)
    frame["target"], task = build_binary_labels(frame["raw_label"], args.dataset)
    frame["decision"] = frame["image_id"].map(verdicts)
    frame["quality"] = frame["image_id"].map(quality)

    logger.info("Downstream task: %s", task)

    # The classifier is trained on clean training images only, so no image in
    # the evaluation pool contributed to fitting it.
    train = frame[(frame["split"] == "train") & (frame["defect_type"] == "clean")]
    evaluation = frame[
        (frame["split"] != "train")
        & (frame["defect_type"] == "clean")
        & frame["decision"].notna()
    ]
    if train.empty or evaluation.empty:
        raise RuntimeError(
            f"need clean training rows and evaluated clean rows; have "
            f"{len(train)} and {len(evaluation)}"
        )

    logger.info(
        "Trained on %d clean training image(s); evaluating %d clean image(s) "
        "from split(s) %s",
        len(train), len(evaluation),
        ", ".join(sorted(evaluation["split"].unique())),
    )

    scaler = StandardScaler().fit(embeddings[train["row"]])
    model = LogisticRegression(
        max_iter=2000, random_state=args.seed, class_weight="balanced"
    ).fit(scaler.transform(embeddings[train["row"]]), train["target"].to_numpy())

    features = scaler.transform(embeddings[evaluation["row"]])
    probabilities = model.predict_proba(features)[:, 1]
    targets = evaluation["target"].to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    # Probability assigned to the true class: high means the classifier found the
    # image easy, low means it struggled.
    confidence = np.where(targets == 1, probabilities, 1.0 - probabilities)
    incorrect = (predictions != targets).astype(int)

    results: dict[str, Any] = {
        "dataset": args.dataset, "backbone": args.backbone, "task": task,
        "n_train_clean": int(len(train)),
        "n_evaluated_clean": int(len(evaluation)),
        "overall_accuracy": float((predictions == targets).mean()),
    }

    # ------------------------------------------------------------ Form A ----
    logger.info("")
    logger.info("=" * 94)
    logger.info("FORM A - accepted against flagged, among images with no injected defect")
    logger.info("=" * 94)

    accepted = evaluation["decision"] == "accept"
    n_flagged = int((~accepted).sum())
    logger.info(
        "  %d accepted, %d flagged or rejected (%.1f%% of clean images actioned)",
        int(accepted.sum()), n_flagged, 100 * n_flagged / len(evaluation),
    )

    if n_flagged < 5 or accepted.sum() < 5 or len(np.unique(targets[~accepted])) < 2:
        logger.warning(
            "  not evaluable: the flagged group has %d image(s) and %d class(es). "
            "The Inspector flags clean images at its configured false-positive "
            "rate, so this group is small by construction.",
            n_flagged, len(np.unique(targets[~accepted])) if n_flagged else 0,
        )
        results["form_a"] = {"skipped": "flagged group too small", "n_flagged": n_flagged}
    else:
        accepted_ci = bootstrap_auroc_ci(
            targets[accepted.to_numpy()], probabilities[accepted.to_numpy()],
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        flagged_ci = bootstrap_auroc_ci(
            targets[~accepted.to_numpy()], probabilities[~accepted.to_numpy()],
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        logger.info(
            "  accepted   n=%4d  AUROC %.3f [%.3f, %.3f]  accuracy %.3f",
            int(accepted.sum()), accepted_ci.point, accepted_ci.ci_low,
            accepted_ci.ci_high,
            float((predictions[accepted.to_numpy()] == targets[accepted.to_numpy()]).mean()),
        )
        logger.info(
            "  flagged    n=%4d  AUROC %.3f [%.3f, %.3f]  accuracy %.3f",
            n_flagged, flagged_ci.point, flagged_ci.ci_low, flagged_ci.ci_high,
            float((predictions[~accepted.to_numpy()] == targets[~accepted.to_numpy()]).mean()),
        )
        # With a group this small the interval is the result, not the point
        # estimate, so the detection floor is stated alongside it.
        floor = 2.8 * float(np.sqrt(0.25 / n_flagged))
        logger.info("")
        logger.info(
            "  difference %+.3f; with %d flagged image(s) the smallest reliably "
            "detectable difference is approximately %.2f, so this comparison "
            "cannot resolve an effect of plausible size",
            accepted_ci.point - flagged_ci.point, n_flagged, floor,
        )
        results["form_a"] = {
            "n_accepted": int(accepted.sum()), "n_flagged": n_flagged,
            "accepted_auroc": accepted_ci.point,
            "accepted_ci": [accepted_ci.ci_low, accepted_ci.ci_high],
            "flagged_auroc": flagged_ci.point,
            "flagged_ci": [flagged_ci.ci_low, flagged_ci.ci_high],
            "difference": accepted_ci.point - flagged_ci.point,
            "approximate_detection_floor": floor,
        }

    # ------------------------------------------------------------ Form B ----
    logger.info("")
    logger.info("=" * 94)
    logger.info("FORM B - does the quality score predict classifier difficulty?")
    logger.info("=" * 94)

    scored = evaluation["quality"].notna().to_numpy()
    if scored.sum() < 30 or len(np.unique(incorrect[scored])) < 2:
        logger.warning(
            "  not evaluable: %d scored image(s), %d outcome class(es)",
            int(scored.sum()), len(np.unique(incorrect[scored])),
        )
        results["form_b"] = {"skipped": "insufficient scored images"}
    else:
        quality_scores = evaluation["quality"].to_numpy()[scored]
        wrong = incorrect[scored]
        logger.info(
            "  %d clean image(s) carry a composite quality score; the classifier "
            "misclassifies %d of them (%.1f%%)",
            int(scored.sum()), int(wrong.sum()), 100 * wrong.mean(),
        )

        # Quality scores are oriented so that higher means worse. If the score
        # carries downstream signal, it should discriminate misclassified images
        # above chance.
        association = bootstrap_auroc_ci(
            wrong, quality_scores, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        correlation, p_value = spearmanr(quality_scores, confidence[scored])
        significant = not (association.ci_low <= 0.5 <= association.ci_high)
        verdict = (
            "quality score PREDICTS difficulty" if significant and association.point > 0.5
            else "quality score is INVERSELY related to difficulty"
            if significant else "no detectable association"
        )
        logger.info("")
        logger.info(
            "  AUROC of the quality score predicting misclassification: "
            "%.3f [%.3f, %.3f]", association.point, association.ci_low,
            association.ci_high,
        )
        logger.info(
            "  Spearman correlation with the classifier's confidence in the true "
            "class: %+.3f (p = %.3g)", correlation, p_value,
        )
        logger.info("  ->  %s", verdict)
        logger.info("")
        logger.info(
            "  This uses every clean image rather than the flagged minority, so "
            "it tests the same hypothesis as Form A with substantially more "
            "power. No image in this pool carries an injected defect, so the "
            "association cannot be an artefact of this project's own "
            "degradation."
        )
        results["form_b"] = {
            "n_scored": int(scored.sum()),
            "n_misclassified": int(wrong.sum()),
            "auroc_quality_predicts_error": association.point,
            "auroc_ci": [association.ci_low, association.ci_high],
            "spearman_with_confidence": float(correlation),
            "spearman_p": float(p_value),
            "significant": significant,
            "verdict": verdict,
        }

    output_path = logs_dir / f"{args.dataset}_natural_quality_{args.backbone}.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n"
    )
    logger.info("")
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()