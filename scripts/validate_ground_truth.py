#!/usr/bin/env python3
"""Validate the ground truth of a processed dataset that has already been written.

Calibration only ever saw a 60-image sample. This checks the dataset actually
delivered - every row of ``processed_metadata.csv`` - and answers the questions
that determine whether the graded ground truth is trustworthy:

  * Is severity monotonic? Severity 1 must be measurably milder than 2, and 2
    than 3, for every defect on every dataset. If it is not, "severity" is not
    ordinal and any correlation-against-severity result is meaningless.
  * Do the achieved medians match what calibration targeted? Calibration
    optimised a median over 60 images; the full set may drift.
  * Is anything destroyed rather than degraded? A defect that removes the
    subject entirely is detected perfectly by every method and silently inflates
    benchmark scores, so it needs flagging rather than averaging away.

Run this before treating a processed dataset as final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

try:
    from scripts.common import configure_logging
    from scripts.inject_defects import load_image, prepare_cli_source
    from pipeline.defects import SSIM_NON_ORDINAL_DEFECTS
except ModuleNotFoundError:  # Allow `python scripts/validate_ground_truth.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import configure_logging  # type: ignore[no-redef]
    from inject_defects import load_image, prepare_cli_source  # type: ignore[no-redef]
    from pipeline.defects import SSIM_NON_ORDINAL_DEFECTS  # type: ignore[no-redef]


# Below this SSIM an image has generally lost its subject rather than merely
# being degraded. Used only to flag rows for inspection, never to alter them.
DESTROYED_SSIM_THRESHOLD = 0.35
# How far a defect's achieved median may sit from the calibration target before
# it is worth reporting. Wider than the calibrator's own tolerance because the
# full set is a different, larger sample than the 60 images it optimised on.
MEDIAN_DRIFT_TOLERANCE = 0.05


def measure_ssim(original: np.ndarray, degraded: np.ndarray) -> float:
    multichannel = original.ndim == 3 and original.shape[2] > 1
    return float(
        structural_similarity(
            original,
            degraded,
            data_range=255,
            channel_axis=2 if multichannel else None,
        )
    )


def score_rows(
    dataset_dir: Path,
    processed: pd.DataFrame,
    sources: dict[str, str],
    canonical_size: int,
    logger,
) -> pd.DataFrame:
    """Recompute SSIM for every processed row against its own clean source."""
    # Each source is decoded once and reused across all of its degraded
    # variants; ISIC sources reach 6000x4000, so re-decoding per row would
    # dominate the runtime.
    cache: dict[str, np.ndarray] = {}
    records = []
    for row in tqdm(list(processed.itertuples(index=False)), desc="Scoring"):
        original_id = str(row.original_image_id)
        source_relative = sources.get(original_id)
        if source_relative is None:
            logger.warning("No source row for %s; skipping", original_id)
            continue
        if original_id not in cache:
            try:
                canonical, _ = prepare_cli_source(
                    load_image(dataset_dir / source_relative),
                    dataset_dir / source_relative,
                    canonical_size,
                )
            except Exception as error:  # noqa: BLE001
                logger.warning("Could not load source %s: %s", original_id, error)
                continue
            cache[original_id] = np.asarray(canonical)
        degraded_path = dataset_dir / str(row.path)
        if not degraded_path.is_file():
            logger.warning("Missing processed file for %s", row.image_id)
            continue
        with Image.open(degraded_path) as handle:
            degraded = np.asarray(handle.convert(
                "L" if cache[original_id].ndim == 2 else "RGB"
            ))
        records.append(
            {
                "image_id": row.image_id,
                "defect_type": str(row.defect_type),
                "severity_level": str(row.severity_level),
                "split": str(row.split),
                "ssim": measure_ssim(cache[original_id], degraded),
            }
        )
    return pd.DataFrame(records)


def check_monotonicity(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Median SSIM must strictly decrease as severity rises, for each defect.

    Defects listed in ``SSIM_NON_ORDINAL_DEFECTS`` are checked but reported
    separately: their severity is known not to be ordinal under SSIM, so a
    violation there is an expected property rather than a fault in the data.
    """
    degraded = scored[scored["defect_type"] != "clean"]
    summary = (
        degraded.groupby(["defect_type", "severity_level"])["ssim"]
        .agg(["count", "median", "mean", "std", "min", "max"])
        .reset_index()
    )
    violations: list[str] = []
    expected: list[str] = []
    for defect, group in summary.groupby("defect_type"):
        ordered = group.sort_values("severity_level")
        medians = ordered["median"].tolist()
        levels = ordered["severity_level"].tolist()
        for index in range(len(medians) - 1):
            if medians[index] <= medians[index + 1]:
                message = (
                    f"{defect}: severity {levels[index]} (SSIM "
                    f"{medians[index]:.3f}) is not milder than severity "
                    f"{levels[index + 1]} (SSIM {medians[index + 1]:.3f})"
                )
                if defect in SSIM_NON_ORDINAL_DEFECTS:
                    expected.append(message)
                else:
                    violations.append(message)
    return summary, violations, expected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--canonical-size", type=int, default=256)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Score only a random subset (default: every processed row)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_ground_truth_validation.log"
    )
    dataset_dir = args.data_dir / args.dataset

    processed = pd.read_csv(
        dataset_dir / "processed_metadata.csv", dtype=str, keep_default_na=False
    )
    source_metadata = pd.read_csv(
        dataset_dir / "metadata.csv", dtype=str, keep_default_na=False
    )
    sources = dict(zip(source_metadata["image_id"], source_metadata["path"]))

    if args.sample_size and args.sample_size < len(processed):
        processed = processed.sample(n=args.sample_size, random_state=args.seed)
        logger.info("Scoring a %d-row sample of the processed set", len(processed))
    else:
        logger.info("Scoring all %d processed row(s)", len(processed))

    scored = score_rows(
        dataset_dir, processed, sources, args.canonical_size, logger
    )
    if scored.empty:
        logger.error("Nothing could be scored")
        return 1

    summary, violations, expected_violations = check_monotonicity(scored)

    logger.info("")
    logger.info("Median SSIM by defect and severity:")
    for defect, group in summary.groupby("defect_type"):
        cells = "  ".join(
            f"sev{row.severity_level}={row.median:.3f} (n={int(row.count)})"
            for row in group.sort_values("severity_level").itertuples(index=False)
        )
        logger.info("  %-20s %s", defect, cells)

    clean = scored[scored["defect_type"] == "clean"]
    if not clean.empty:
        logger.info(
            "  %-20s median=%.4f (n=%d) - should be ~1.0; lower means the clean "
            "baseline is not lossless",
            "clean", clean["ssim"].median(), len(clean),
        )

    destroyed = scored[scored["ssim"] < DESTROYED_SSIM_THRESHOLD]
    logger.info("")
    ordinal_defects = sorted(
        set(summary["defect_type"]) - set(SSIM_NON_ORDINAL_DEFECTS)
    )
    if violations:
        logger.error("MONOTONICITY VIOLATIONS - severity is not ordinal:")
        for violation in violations:
            logger.error("  %s", violation)
    else:
        logger.info(
            "Monotonicity: OK across all %d severity-ordinal defect(s)",
            len(ordinal_defects),
        )
    if expected_violations:
        logger.info(
            "Expected non-ordinality in %s (documented; SSIM measures "
            "misalignment for these, not quality loss):",
            ", ".join(SSIM_NON_ORDINAL_DEFECTS),
        )
        for message in expected_violations:
            logger.info("  %s", message)
    logger.info(
        "Eligible for severity-correlation analysis: %s", ", ".join(ordinal_defects)
    )
    logger.info(
        "Binary detection only (severity not SSIM-ordinal): %s",
        ", ".join(SSIM_NON_ORDINAL_DEFECTS),
    )

    if not destroyed.empty:
        breakdown = (
            destroyed.groupby(["defect_type", "severity_level"])
            .size()
            .sort_values(ascending=False)
        )
        logger.warning(
            "%d row(s) fall below SSIM %.2f and may be destroyed rather than "
            "degraded:", len(destroyed), DESTROYED_SSIM_THRESHOLD,
        )
        for (defect, severity), count in breakdown.items():
            logger.warning("  %s severity %s: %d row(s)", defect, severity, count)
    else:
        logger.info(
            "No rows below SSIM %.2f - nothing looks destroyed",
            DESTROYED_SSIM_THRESHOLD,
        )

    # Compare against what calibration aimed for, if a report exists.
    calibration_path = args.data_dir / "logs" / f"{args.dataset}_severity_calibration.json"
    drifts: list[str] = []
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())
        targets = {str(k): float(v) for k, v in calibration["targets"].items()}
        # occlusion_patch and rotation_crop are deliberately not calibrated
        # against SSIM, so comparing them to an SSIM target is meaningless -
        # a 24-degree rotation legitimately scores far below any target.
        uncalibrated = set(calibration.get("uncalibrated_defects", {}))
        for row in summary.itertuples(index=False):
            if row.defect_type in uncalibrated:
                continue
            target = targets.get(str(row.severity_level))
            if target is None:
                continue
            if abs(row.median - target) > MEDIAN_DRIFT_TOLERANCE:
                drifts.append(
                    f"{row.defect_type} severity {row.severity_level}: "
                    f"median {row.median:.3f} vs target {target:.3f} "
                    f"(drift {row.median - target:+.3f})"
                )
        logger.info("")
        if drifts:
            logger.warning(
                "Achieved medians drifting more than %.2f from the calibration "
                "target on the full set:", MEDIAN_DRIFT_TOLERANCE,
            )
            for drift in drifts:
                logger.warning("  %s", drift)
        else:
            logger.info(
                "All achieved medians are within %.2f of their calibration target",
                MEDIAN_DRIFT_TOLERANCE,
            )

        if uncalibrated:
            logger.info(
                "Not drift-checked (uncalibrated by design): %s",
                ", ".join(sorted(uncalibrated)),
            )

    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(logs_dir / f"{args.dataset}_ground_truth_summary.csv", index=False)
    scored.to_csv(logs_dir / f"{args.dataset}_ground_truth_scores.csv", index=False)
    (logs_dir / f"{args.dataset}_ground_truth_validation.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "rows_scored": len(scored),
                "monotonicity_violations": violations,
                "expected_non_ordinal": expected_violations,
                # Stage 2 should read these rather than hardcoding a list.
                "severity_correlation_eligible": ordinal_defects,
                "binary_detection_only": list(SSIM_NON_ORDINAL_DEFECTS),
                "median_drifts": drifts,
                "rows_below_destroyed_threshold": int(len(destroyed)),
                "destroyed_threshold": DESTROYED_SSIM_THRESHOLD,
                "clean_median_ssim": (
                    float(clean["ssim"].median()) if not clean.empty else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info("Wrote per-defect summary and per-row scores to %s", logs_dir)

    # Non-zero exit only for a violation that invalidates the ground truth.
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())