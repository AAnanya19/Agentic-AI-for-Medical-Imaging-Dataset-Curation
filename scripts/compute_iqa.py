#!/usr/bin/env python3
"""Score a processed dataset with every registered no-reference IQA metric.

Reads ``<dataset>/processed_metadata.csv`` (written by inject_defects.py),
computes every metric in pipeline.iqa_metrics on every processed image, and
writes one row per image to ``<dataset>/iqa_scores.csv``.

brightness_deviation needs a reference "typical brightness" fitted from clean
training-split images before anything else can be scored - see
pipeline.iqa_metrics.BrightnessReference. The fitted reference is cached to
``<dataset>/brightness_reference.json`` so re-running with --skip-existing does
not require re-fitting it, and so the exact reference used is auditable.
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
from tqdm import tqdm

try:
    from scripts.common import configure_logging
    from pipeline.iqa_metrics import (
        METRICS,
        BrightnessReference,
        available_metrics,
        brightness_deviation,
        compute_all_metrics,
    )
except ModuleNotFoundError:  # Allow `python scripts/compute_iqa.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.iqa_metrics import (  # type: ignore[no-redef]
        METRICS,
        BrightnessReference,
        available_metrics,
        brightness_deviation,
        compute_all_metrics,
    )


OUTPUT_COLUMNS = [
    "image_id",
    "original_image_id",
    "defect_type",
    "severity_level",
    "split",
    *sorted(METRICS),
    "brightness_deviation",
]


def load_array(path: Path) -> np.ndarray:
    """Load an image exactly as saved, preserving its mode (L or RGB)."""
    with Image.open(path) as handle:
        return np.asarray(handle.copy())


def fit_brightness_reference(
    dataset_dir: Path,
    processed: pd.DataFrame,
    logger: logging.Logger,
) -> BrightnessReference:
    """Fit the brightness reference from clean, train-split processed images.

    Uses the *processed* (canonically resized) clean images rather than the raw
    sources, so the reference is on the same footing - same resolution, same
    preprocessing - as everything it will be compared against.
    """
    clean_train = processed[
        (processed["defect_type"] == "clean") & (processed["split"] == "train")
    ]
    if clean_train.empty:
        raise RuntimeError(
            "No clean training-split rows found; cannot fit a brightness "
            "reference. Check processed_metadata.csv for a 'clean' defect_type."
        )
    images = [load_array(dataset_dir / row.path) for row in clean_train.itertuples(index=False)]
    reference = BrightnessReference.fit(images)
    logger.info(
        "Fitted brightness reference from %d clean train-split image(s): "
        "median mean intensity = %.2f",
        reference.n_images,
        reference.reference_mean,
    )
    return reference


def score_dataset(
    dataset_dir: Path,
    processed: pd.DataFrame,
    reference: BrightnessReference,
    metric_names: tuple[str, ...],
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for row in tqdm(list(processed.itertuples(index=False)), desc="Scoring IQA metrics"):
        path = dataset_dir / row.path
        try:
            array = load_array(path)
            scores = compute_all_metrics(array, metric_names)
            scores["brightness_deviation"] = brightness_deviation(array, reference)
        except Exception as error:  # noqa: BLE001 - one bad image must not abort the run
            failures.append((row.image_id, str(error)))
            continue
        rows.append(
            {
                "image_id": row.image_id,
                "original_image_id": row.original_image_id,
                "defect_type": row.defect_type,
                "severity_level": row.severity_level,
                "split": row.split,
                **scores,
            }
        )
    if failures:
        logger.warning(
            "%d image(s) failed scoring and were skipped, e.g. %s: %s",
            len(failures), failures[0][0], failures[0][1],
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset directory name under --data-dir")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--no-torch-metrics",
        action="store_true",
        help="Skip BRISQUE and CLIP-IQA (useful if piq/torch is unavailable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(args.data_dir, args.dataset, False, f"{args.dataset}_iqa.log")
    dataset_dir = args.data_dir / args.dataset

    processed_path = dataset_dir / "processed_metadata.csv"
    if not processed_path.is_file():
        raise FileNotFoundError(
            f"{processed_path} not found; run inject_defects.py for this dataset first"
        )
    processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
    logger.info("Loaded %d processed row(s) from %s", len(processed), processed_path)

    metric_names = available_metrics(include_torch=not args.no_torch_metrics)
    logger.info("Computing metrics: %s", ", ".join(metric_names))

    reference = fit_brightness_reference(dataset_dir, processed, logger)
    (dataset_dir / "brightness_reference.json").write_text(
        json.dumps(reference.to_dict(), indent=2) + "\n"
    )

    scored = score_dataset(dataset_dir, processed, reference, metric_names, logger)
    output_path = dataset_dir / "iqa_scores.csv"
    scored.to_csv(output_path, index=False)
    logger.info("Wrote %d scored row(s) to %s", len(scored), output_path)


if __name__ == "__main__":
    main()