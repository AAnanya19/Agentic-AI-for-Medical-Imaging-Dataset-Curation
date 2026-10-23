#!/usr/bin/env python3
"""Calibrate defect severity to a perceptual target so levels mean the same thing.

The hand-chosen ranges in ``pipeline/defects.py`` were tuned by eye on
dermoscopy. Applied unchanged to chest radiographs they misbehave in both
directions: signal-dependent Poisson noise obliterates an X-ray at severity 2-3
(because radiographs have large near-black regions), while JPEG artifacts stay
almost invisible even at quality 10 (because radiographs are smooth and
low-texture). A defect that destroys the image is trivially detectable by every
metric and contributes no discriminative power to a benchmark; one that is
invisible contributes none either.

This script instead defines each severity level as a *target degradation band*
and solves for the parameter that reaches it on a given dataset, by bisection
on the median SSIM across a sample of images. The result is written as a JSON
profile that ``inject_defects.py --severity-profile`` consumes.

Two consequences worth stating in a write-up:
  * severity is comparable across modalities, which any cross-dataset
    generalisation claim depends on; and
  * severity is comparable across defect types, which makes a per-defect
    detection comparison interpretable rather than an artifact of how hard each
    distortion happened to be tuned.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

try:
    from scripts.common import configure_logging
    from scripts.inject_defects import canonical_resize, load_image, prepare_cli_source
    from pipeline.defects import (
        CALIBRATION_SPECS,
        DEFECT_CONFIG,
        SEVERITY_NAMES,
        UNCALIBRATED_DEFECTS,
        CalibrationSpec,
        apply_defect,
        kernel_size_for_sigma,
        severity_profile,
    )
except ModuleNotFoundError:  # Allow `python scripts/calibrate_severity.py`.
    from common import configure_logging  # type: ignore[no-redef]
    from inject_defects import (  # type: ignore[no-redef]
        canonical_resize,
        load_image,
        prepare_cli_source,
    )
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.defects import (  # type: ignore[no-redef]
        CALIBRATION_SPECS,
        DEFECT_CONFIG,
        SEVERITY_NAMES,
        UNCALIBRATED_DEFECTS,
        CalibrationSpec,
        apply_defect,
        kernel_size_for_sigma,
        severity_profile,
    )


# Target median SSIM per severity level. The bands are wide enough apart that
# severity stays clearly ordinal, and none of them sits so low that the image is
# destroyed - severe should be obviously degraded but still recognisably the
# same study.
DEFAULT_TARGETS = {1: 0.90, 2: 0.75, 3: 0.55}
BISECTION_ITERATIONS = 18
SSIM_TOLERANCE = 0.01


def sample_source_images(
    dataset_dir: Path,
    sample_size: int,
    canonical_size: int,
    seed: int,
    logger: logging.Logger,
) -> list[np.ndarray]:
    """Load a deterministic, split-spanning sample as canonical uint8 arrays."""
    metadata = pd.read_csv(dataset_dir / "metadata.csv", dtype=str, keep_default_na=False)
    if metadata.empty:
        raise RuntimeError(f"{dataset_dir/'metadata.csv'} is empty")

    # Sample proportionally across splits so calibration is not driven by one
    # subset of the cohort.
    take = min(sample_size, len(metadata))
    chosen = metadata.sample(n=take, random_state=seed)

    images: list[np.ndarray] = []
    for row in tqdm(list(chosen.itertuples(index=False)), desc="Loading sample"):
        path = dataset_dir / str(row.path)
        if not path.is_file():
            continue
        try:
            canonical, _ = prepare_cli_source(load_image(path), path, canonical_size)
        except Exception as error:  # noqa: BLE001 - calibration must not abort
            logger.warning("Skipping %s during calibration: %s", path.name, error)
            continue
        images.append(np.asarray(canonical))
    if not images:
        raise RuntimeError("No usable images were loaded for calibration")
    logger.info("Loaded %d image(s) for calibration", len(images))
    return images


def measure_ssim(original: np.ndarray, degraded: np.ndarray) -> float:
    """Structural similarity between a clean and degraded canonical image."""
    multichannel = original.ndim == 3 and original.shape[2] > 1
    return float(
        structural_similarity(
            original,
            degraded,
            data_range=255,
            channel_axis=2 if multichannel else None,
        )
    )


def settings_for_value(
    defect_name: str,
    severity: int,
    spec: CalibrationSpec,
    value: float,
) -> dict[str, Any]:
    """Build a config override that centres the defect on ``value``.

    The original range's *relative* width is preserved, so a calibrated level
    keeps the same stochastic spread it had before - calibration moves the
    centre of the distribution, it does not collapse it to a constant.
    """
    default = DEFECT_CONFIG[defect_name][severity]
    settings: dict[str, Any] = {}

    if not spec.is_range:
        settings[spec.config_key] = int(round(value)) if spec.integer else float(value)
        return settings

    low, high = default[spec.config_key]
    centre = (float(low) + float(high)) / 2.0
    relative_half_width = ((float(high) - float(low)) / 2.0 / centre) if centre else 0.0
    # Keep the calibrated value as the centre and narrow the spread symmetrically
    # if the bounds are close. Clipping just one side would shift the centre away
    # from the value the search actually solved for, and letting the range cross
    # a bound is worse still: a contrast factor above 1.0 stops being "contrast
    # reduction" and starts increasing contrast - a different defect wearing the
    # same label.
    bound_low, bound_high = min(spec.bounds), max(spec.bounds)
    value = max(bound_low, min(bound_high, value))
    half_width = min(
        value * relative_half_width,
        value - bound_low,
        bound_high - value,
    )
    half_width = max(0.0, half_width)
    new_low = value - half_width
    new_high = value + half_width

    if spec.integer:
        new_low_i, new_high_i = int(round(new_low)), int(round(new_high))
        if spec.odd_only:
            new_low_i = max(3, new_low_i | 1)
            new_high_i = max(new_low_i, new_high_i | 1)
        new_high_i = max(new_low_i, new_high_i)
        settings[spec.config_key] = (new_low_i, new_high_i)
    else:
        settings[spec.config_key] = (float(new_low), float(new_high))

    if spec.couples_kernel_to_sigma:
        settings["kernel_sizes"] = (
            kernel_size_for_sigma(new_low),
            kernel_size_for_sigma(new_high),
        )
    return settings


def median_ssim_for_value(
    images: list[np.ndarray],
    defect_name: str,
    severity: int,
    spec: CalibrationSpec,
    value: float,
    seed: int,
) -> float:
    """Median SSIM across the sample when the defect is centred on ``value``."""
    override = {defect_name: {severity: settings_for_value(defect_name, severity, spec, value)}}
    scores = []
    with severity_profile(override):
        for index, image in enumerate(images):
            degraded, _ = apply_defect(
                Image.fromarray(image),
                defect_name,
                severity,
                # Fixed per-image seed: the same realisation is compared at every
                # bisection step, so the search sees a smooth function of the
                # parameter rather than sampling noise.
                rng=seed + index,
            )
            scores.append(measure_ssim(image, np.asarray(degraded)))
    return float(np.median(scores))


def calibrate_one(
    images: list[np.ndarray],
    defect_name: str,
    severity: int,
    spec: CalibrationSpec,
    target: float,
    seed: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Bisect the intensity parameter until median SSIM meets ``target``."""
    low, high = spec.bounds
    # Orient the search so `high` always denotes "more degradation".
    if not spec.increasing:
        low, high = high, low

    ssim_at_low = median_ssim_for_value(images, defect_name, severity, spec, low, seed)
    ssim_at_high = median_ssim_for_value(images, defect_name, severity, spec, high, seed)

    reachable = ssim_at_high <= target <= ssim_at_low
    if not reachable:
        # Clamp to whichever bound is closest and say so, rather than silently
        # returning a parameter that misses the target.
        chosen = high if abs(ssim_at_high - target) < abs(ssim_at_low - target) else low
        achieved = ssim_at_high if chosen == high else ssim_at_low
        logger.warning(
            "%s severity %d: target SSIM %.2f unreachable within bounds "
            "(SSIM spans %.3f..%.3f); clamping to %.4g at SSIM %.3f",
            defect_name, severity, target, ssim_at_high, ssim_at_low, chosen, achieved,
        )
        return {
            "settings": settings_for_value(defect_name, severity, spec, chosen),
            "value": chosen,
            "achieved_ssim": achieved,
            "target_ssim": target,
            "target_reached": False,
            "status": "clamped_to_parameter_bound",
        }

    achieved = ssim_at_low
    midpoint = low
    for _ in range(BISECTION_ITERATIONS):
        midpoint = (low + high) / 2.0
        achieved = median_ssim_for_value(
            images, defect_name, severity, spec, midpoint, seed
        )
        if abs(achieved - target) <= SSIM_TOLERANCE:
            break
        if achieved > target:  # not degraded enough yet
            low = midpoint
        else:
            high = midpoint

    # Integer-valued parameters (kernel sizes, JPEG quality, brightness offset)
    # move in discrete jumps, so the continuous bisection can land between two
    # achievable values. Search the neighbouring integers and keep the closest.
    if spec.integer:
        candidates = {
            int(round(midpoint)) + delta for delta in range(-2, 3)
        }
        best_value, best_ssim = midpoint, achieved
        for candidate in sorted(candidates):
            if not (min(spec.bounds) <= candidate <= max(spec.bounds)):
                continue
            score = median_ssim_for_value(
                images, defect_name, severity, spec, float(candidate), seed
            )
            if abs(score - target) < abs(best_ssim - target):
                best_value, best_ssim = float(candidate), score
        midpoint, achieved = best_value, best_ssim

    if abs(achieved - target) <= SSIM_TOLERANCE:
        status = "reached"
    elif spec.integer:
        # The parameter simply has no value that lands on the target.
        status = "nearest_achievable_at_integer_granularity"
    else:
        status = "converged_outside_tolerance"

    return {
        "settings": settings_for_value(defect_name, severity, spec, midpoint),
        "value": midpoint,
        "achieved_ssim": achieved,
        "target_ssim": target,
        "target_reached": abs(achieved - target) <= SSIM_TOLERANCE,
        "status": status,
    }



def probe_achievable_envelope(
    images: list[np.ndarray],
    selected: list[str],
    seed: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Measure the SSIM range each defect can actually reach on this dataset.

    Target bands are only meaningful if every defect can reach them. Some limits
    are physical rather than a matter of tuning: JPEG stops at quality 1 and
    contrast reduction stops at a uniform grey image, so neither can be pushed
    past a certain SSIM no matter how the bounds are set. Probing first turns
    "pick targets and hope" into "pick targets inside the feasible region", and
    the feasible region is a property of the data, not a guess.
    """
    envelope: dict[str, Any] = {}
    for defect_name in selected:
        spec = CALIBRATION_SPECS[defect_name]
        gentle, harsh = (spec.bounds if spec.increasing else spec.bounds[::-1])
        highest = median_ssim_for_value(images, defect_name, 1, spec, gentle, seed)
        lowest = median_ssim_for_value(images, defect_name, 3, spec, harsh, seed)
        envelope[defect_name] = {
            "max_ssim_at_gentlest": highest,
            "min_ssim_at_harshest": lowest,
            "gentlest_value": gentle,
            "harshest_value": harsh,
        }
        logger.info(
            "%-20s can reach SSIM %.3f (gentlest) down to %.3f (harshest)",
            defect_name, highest, lowest,
        )

    # A target is feasible for every defect only inside this intersection.
    feasible_high = min(v["max_ssim_at_gentlest"] for v in envelope.values())
    feasible_low = max(v["min_ssim_at_harshest"] for v in envelope.values())
    binding_high = min(envelope, key=lambda k: envelope[k]["max_ssim_at_gentlest"])
    binding_low = max(envelope, key=lambda k: envelope[k]["min_ssim_at_harshest"])

    logger.info("")
    logger.info(
        "Feasible common SSIM band: %.3f (limited by %s) .. %.3f (limited by %s)",
        feasible_low, binding_low, feasible_high, binding_high,
    )
    if feasible_low >= feasible_high:
        logger.warning(
            "No SSIM band is reachable by every defect; severity cannot be made "
            "comparable across all of them on this dataset."
        )
        suggested = None
    else:
        # Keep a small margin inside the band so bisection is not fighting a
        # bound, then space the three levels evenly across what remains.
        margin = 0.02 * (feasible_high - feasible_low)
        high = feasible_high - margin
        low = feasible_low + margin
        suggested = {
            1: round(high, 3),
            2: round((high + low) / 2.0, 3),
            3: round(low, 3),
        }
        logger.info(
            "Suggested targets (evenly spaced inside the feasible band): %s",
            ",".join(f"{suggested[k]:.3f}" for k in sorted(suggested)),
        )
        logger.info(
            "Re-run with: --targets %s",
            ",".join(f"{suggested[k]:.3f}" for k in sorted(suggested)),
        )
    return {
        "per_defect": envelope,
        "feasible_band": {"low": feasible_low, "high": feasible_high},
        "binding_constraints": {"low": binding_low, "high": binding_high},
        "suggested_targets": suggested,
    }


def parse_targets(raw: str | None) -> dict[int, float]:
    if not raw:
        return dict(DEFAULT_TARGETS)
    values = [float(part) for part in raw.split(",")]
    if len(values) != len(SEVERITY_NAMES):
        raise ValueError(f"--targets needs {len(SEVERITY_NAMES)} comma-separated values")
    return dict(zip(sorted(SEVERITY_NAMES), values))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--canonical-size", type=int, default=256)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=60,
        help="Images used to estimate median SSIM at each search step",
    )
    parser.add_argument(
        "--targets",
        help="Comma-separated target SSIM per severity, e.g. '0.90,0.75,0.55'",
    )
    parser.add_argument(
        "--defects",
        help="Comma-separated subset to calibrate (default: all calibratable)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Only measure each defect's achievable SSIM range and suggest "
        "targets that every defect can actually reach; write no profile",
    )
    parser.add_argument("--output", type=Path, help="Defaults to the dataset directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_severity_calibration.log"
    )
    targets = parse_targets(args.targets)
    dataset_dir = args.data_dir / args.dataset

    selected = (
        [name.strip() for name in args.defects.split(",")]
        if args.defects
        else list(CALIBRATION_SPECS)
    )
    unknown = [name for name in selected if name not in CALIBRATION_SPECS]
    if unknown:
        raise ValueError(
            f"not calibratable: {unknown}. Calibratable: {sorted(CALIBRATION_SPECS)}. "
            f"Deliberately excluded: {list(UNCALIBRATED_DEFECTS)}"
        )

    logger.info(
        "Calibrating %s on %s | targets=%s | sample=%d | canonical=%d",
        ",".join(selected), args.dataset, targets, args.sample_size, args.canonical_size,
    )
    images = sample_source_images(
        dataset_dir, args.sample_size, args.canonical_size, args.seed, logger
    )

    if args.probe:
        envelope = probe_achievable_envelope(images, selected, args.seed, logger)
        probe_path = args.data_dir / "logs" / f"{args.dataset}_severity_probe.json"
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        logger.info("Wrote probe results to %s", probe_path)
        return

    profile: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "metric": "SSIM (median over sample, against the clean canonical image)",
        "targets": {str(k): v for k, v in targets.items()},
        "sample_size": len(images),
        "canonical_size": args.canonical_size,
        "seed": args.seed,
        "uncalibrated_defects": {
            name: "parameterized in modality-independent units (area fraction / "
                  "degrees); SSIM is not a meaningful target for these"
            for name in UNCALIBRATED_DEFECTS
        },
        "results": {},
    }

    for defect_name in selected:
        spec = CALIBRATION_SPECS[defect_name]
        profile[defect_name] = {}
        report["results"][defect_name] = {}
        for severity in sorted(SEVERITY_NAMES):
            outcome = calibrate_one(
                images, defect_name, severity, spec, targets[severity], args.seed, logger
            )
            settings = outcome.pop("settings")
            profile[defect_name][str(severity)] = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in settings.items()
            }
            report["results"][defect_name][str(severity)] = {
                **outcome,
                "settings": profile[defect_name][str(severity)],
                "default_settings": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in DEFECT_CONFIG[defect_name][severity].items()
                },
            }
            flag = "" if outcome["target_reached"] else f"  [{outcome['status']}]"
            logger.info(
                "%-20s sev %d  target SSIM %.2f -> achieved %.3f  %s%s",
                defect_name, severity, outcome["target_ssim"],
                outcome["achieved_ssim"], profile[defect_name][str(severity)], flag,
            )

    output_dir = args.output or dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "severity_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    report_path = args.data_dir / "logs" / f"{args.dataset}_severity_calibration.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    logger.info("Wrote calibrated profile to %s", profile_path)
    logger.info("Wrote full calibration report to %s", report_path)
    logger.info(
        "Apply it with: python scripts/inject_defects.py --dataset %s "
        "--severity-profile %s", args.dataset, profile_path,
    )


if __name__ == "__main__":
    main()