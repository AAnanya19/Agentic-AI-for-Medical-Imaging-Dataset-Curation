#!/usr/bin/env python3
"""Build a dedicated transformation near-duplicate benchmark set.

**Why this exists.** ``inject_defects.py`` applies exactly one treatment per
source image - clean, or one defect at one severity - which is correct for the
detection experiments in Stages 2-4 but means ``original_image_id`` groups are
all singletons. There are therefore no transformation near-duplicate *pairs* in
the main processed corpus, and the ``transformation`` task in
``detect_duplicates.py`` has nothing to validate against.

This script generates the missing evaluation set the same way published
medical-imaging deduplication benchmarks do: take a sample of source images and
emit several transformed copies of each, so every group has a known set of
mutually near-duplicate members. The transformations are the mild,
preprocessing-like ones a real pipeline would introduce (re-encoding, rescaling,
mild blur or noise), not the severe defects used for quality detection - a
near-duplicate is a copy that survived processing, not an image destroyed by it.

Output goes to ``<dataset>/duplicate_benchmark/`` with its own
``benchmark_metadata.csv`` carrying ``group_id``, leaving the main corpus
untouched.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from scripts.common import configure_logging
    from scripts.inject_defects import load_image, prepare_cli_source, to_uint8_pil
except ModuleNotFoundError:  # Allow `python scripts/build_duplicate_benchmark.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from scripts.inject_defects import (  # type: ignore[no-redef]
        load_image,
        prepare_cli_source,
        to_uint8_pil,
    )


def transform_identity(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """The unmodified canonical image: the group's reference member."""
    return image.copy()


def transform_jpeg_reencode(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Re-encode at high quality: the most common real near-duplicate cause."""
    buffer = io.BytesIO()
    to_uint8_pil(image).save(buffer, format="JPEG", quality=int(rng.integers(80, 96)))
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return handle.copy()


def transform_rescale(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Downscale then restore: simulates a resized copy re-entering a corpus."""
    width, height = image.size
    factor = float(rng.uniform(0.6, 0.85))
    small = image.resize(
        (max(1, int(width * factor)), max(1, int(height * factor))),
        Image.Resampling.LANCZOS,
    )
    return small.resize((width, height), Image.Resampling.LANCZOS)


def transform_mild_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    from PIL import ImageFilter

    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 0.9))))


def transform_mild_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(to_uint8_pil(image), dtype=np.float64)
    noisy = np.clip(array + rng.normal(0, rng.uniform(1.5, 3.5), array.shape), 0, 255)
    return Image.fromarray(noisy.astype(np.uint8), mode=to_uint8_pil(image).mode)


def transform_brightness_nudge(
    image: Image.Image, rng: np.random.Generator
) -> Image.Image:
    array = np.asarray(to_uint8_pil(image), dtype=np.int16)
    offset = int(rng.integers(4, 12)) * (1 if rng.random() < 0.5 else -1)
    return Image.fromarray(
        np.clip(array + offset, 0, 255).astype(np.uint8),
        mode=to_uint8_pil(image).mode,
    )


# Mild, preprocessing-like transformations only. A near-duplicate is a copy that
# survived processing; the severe defects used for quality detection would
# produce images no reasonable system should call duplicates.
TRANSFORMATIONS: dict[str, Callable[[Image.Image, np.random.Generator], Image.Image]] = {
    "identity": transform_identity,
    "jpeg_reencode": transform_jpeg_reencode,
    "rescale": transform_rescale,
    "mild_blur": transform_mild_blur,
    "mild_noise": transform_mild_noise,
    "brightness_nudge": transform_brightness_nudge,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--n-sources", type=int, default=120,
        help="Source images to sample. Pair search is O(n^2) over "
        "n_sources x n_transformations rows, so 120 x 6 = 720 rows is a "
        "reasonable ceiling.",
    )
    parser.add_argument(
        "--n-distractors", type=int, default=240,
        help="Additional single-copy images with no duplicates, so precision is "
        "measured against a realistic population rather than only true pairs.",
    )
    parser.add_argument("--canonical-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_duplicate_benchmark.log"
    )
    dataset_dir = args.data_dir / args.dataset
    output_dir = dataset_dir / "duplicate_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(
        dataset_dir / "metadata.csv", dtype=str, keep_default_na=False
    )
    rng = np.random.default_rng(args.seed)

    total_needed = args.n_sources + args.n_distractors
    if len(source) < total_needed:
        raise ValueError(
            f"need {total_needed} source image(s) but metadata.csv has {len(source)}"
        )
    chosen = source.sample(n=total_needed, random_state=args.seed).reset_index(drop=True)
    duplicated = chosen.iloc[: args.n_sources]
    distractors = chosen.iloc[args.n_sources :]

    logger.info(
        "Building benchmark: %d source(s) x %d transformation(s) + %d distractor(s)",
        args.n_sources, len(TRANSFORMATIONS), args.n_distractors,
    )

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    def emit(row, group_id: str, names: list[str]) -> None:
        source_path = dataset_dir / row.path
        try:
            # prepare_cli_source, not canonical_resize: it flattens alpha and
            # normalises to the shared 8-bit working domain first. ChestX-ray14
            # contains 23 RGBA images that fail downstream without this.
            canonical, _ = prepare_cli_source(
                load_image(source_path), source_path, args.canonical_size
            )
        except Exception as error:  # noqa: BLE001
            failures.append((row.image_id, str(error)))
            return
        for name in names:
            variant = TRANSFORMATIONS[name](canonical, rng)
            image_id = f"{Path(row.image_id).stem}__{name}"
            relative = f"duplicate_benchmark/{image_id}.png"
            to_uint8_pil(variant).save(dataset_dir / relative, format="PNG")
            rows.append({
                "image_id": image_id,
                "original_image_id": row.image_id,
                "group_id": group_id,
                "transformation": name,
                "path": relative,
                "split": row.split,
                "is_duplicate_group": group_id.startswith("dup:"),
            })

    for row in tqdm(list(duplicated.itertuples(index=False)), desc="Duplicate groups"):
        emit(row, f"dup:{row.image_id}", list(TRANSFORMATIONS))
    for row in tqdm(list(distractors.itertuples(index=False)), desc="Distractors"):
        # One copy each, and a group id unique to that image, so a distractor
        # has no true duplicate partner anywhere in the set.
        emit(row, f"solo:{row.image_id}", ["identity"])

    if failures:
        logger.warning(
            "%d source(s) could not be processed, e.g. %s: %s",
            len(failures), failures[0][0], failures[0][1],
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "benchmark_metadata.csv", index=False)

    group_sizes = frame[frame["is_duplicate_group"]].groupby("group_id").size()
    n_true_pairs = int((group_sizes * (group_sizes - 1) // 2).sum())
    manifest = {
        "dataset": args.dataset,
        "n_rows": int(len(frame)),
        "n_duplicate_groups": int(frame["is_duplicate_group"].sum() // len(TRANSFORMATIONS))
        if len(TRANSFORMATIONS) else 0,
        "n_distractors": int((~frame["is_duplicate_group"]).sum()),
        "n_true_duplicate_pairs": n_true_pairs,
        "transformations": list(TRANSFORMATIONS),
        "canonical_size": args.canonical_size,
        "seed": args.seed,
        "n_failed_sources": len(failures),
        "note": (
            "Mild, preprocessing-like transformations only. Ground truth is "
            "group_id: every pair within a 'dup:' group is a true near-duplicate, "
            "and 'solo:' rows have no duplicate partner. Evaluate with "
            "detect_duplicates.py --tasks transformation."
        ),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    logger.info("")
    logger.info(
        "Wrote %d row(s): %d duplicate group(s) yielding %d true pair(s), "
        "plus %d distractor(s)",
        manifest["n_rows"], manifest["n_duplicate_groups"],
        manifest["n_true_duplicate_pairs"], manifest["n_distractors"],
    )
    logger.info("Benchmark at %s", output_dir)


if __name__ == "__main__":
    main()