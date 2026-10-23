#!/usr/bin/env python3
"""Extract and cache frozen-backbone embeddings for a processed dataset.

Reads ``<dataset>/processed_metadata.csv`` and writes one compressed .npz per
backbone to ``embeddings/<dataset>_<backbone>.npz``, alongside the provenance of
exactly which model, input size and normalisation produced them.

Extraction is by far the slowest step of Stage 3 - DINOv2 at 252x252 measured
~205 ms/image on CPU - so results are cached and never recomputed. Re-running
skips a backbone whose cache already exists unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from scripts.common import configure_logging
    from pipeline.embeddings import (
        BACKBONES,
        FeatureExtractor,
        available_backbones,
        save_embeddings,
    )
except ModuleNotFoundError:  # Allow `python scripts/extract_embeddings.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.embeddings import (  # type: ignore[no-redef]
        BACKBONES,
        FeatureExtractor,
        available_backbones,
        save_embeddings,
    )


def parse_backbones(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return available_backbones()
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    unknown = [name for name in names if name not in BACKBONES]
    if unknown:
        raise ValueError(f"unknown backbone(s) {unknown}; available: {sorted(BACKBONES)}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--backbones",
        type=parse_backbones,
        default=available_backbones(),
        help="Comma-separated backbone names, or 'all'",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device; 'cuda' or 'mps' massively speeds up DINOv2 if available",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write .npz caches (default: <data-dir>/embeddings)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use randomly-initialised weights. TESTING ONLY - features are "
        "meaningless; exists so the pipeline can be smoke-tested offline",
    )
    parser.add_argument(
        "--subset",
        default="processed",
        choices=("processed", "benchmark"),
        help="Which manifest to embed. 'processed' is the main corpus; "
        "'benchmark' is the duplicate-detection set from "
        "build_duplicate_benchmark.py, whose image_ids do not exist in the "
        "main cache and so need their own.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if a cache already exists for this backbone",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_embeddings.log"
    )
    dataset_dir = args.data_dir / args.dataset
    output_dir = args.output_dir or (args.data_dir / "embeddings")

    if args.subset == "benchmark":
        processed_path = (
            dataset_dir / "duplicate_benchmark" / "benchmark_metadata.csv"
        )
        if not processed_path.is_file():
            raise FileNotFoundError(
                f"{processed_path} not found; run build_duplicate_benchmark.py "
                "for this dataset first"
            )
        cache_suffix = "_benchmark"
    else:
        processed_path = dataset_dir / "processed_metadata.csv"
        if not processed_path.is_file():
            raise FileNotFoundError(
                f"{processed_path} not found; run inject_defects.py for this "
                "dataset first"
            )
        cache_suffix = ""
    processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
    logger.info("Loaded %d row(s) from %s", len(processed), processed_path)

    for backbone in args.backbones:
        cache_path = output_dir / f"{args.dataset}{cache_suffix}_{backbone}.npz"
        if cache_path.is_file() and not args.overwrite:
            logger.info(
                "Skipping %s: cache already exists at %s (use --overwrite to force)",
                backbone, cache_path,
            )
            continue

        logger.info("Loading backbone %s on device %s", backbone, args.device)
        extractor = FeatureExtractor(
            backbone, device=args.device, pretrained=not args.no_pretrained
        )
        if args.no_pretrained:
            logger.warning(
                "--no-pretrained: %s has RANDOM weights; results are meaningless",
                backbone,
            )
        description = extractor.describe()
        logger.info(
            "%s: timm=%s input=%dx%d dim=%d",
            backbone, description["timm_name"],
            description["input_size"], description["input_size"],
            description["expected_dim"],
        )

        image_ids: list[str] = []
        blocks: list[np.ndarray] = []
        failures: list[tuple[str, str]] = []
        rows = list(processed.itertuples(index=False))

        for start in tqdm(
            range(0, len(rows), args.batch_size), desc=f"Embedding ({backbone})"
        ):
            chunk = rows[start : start + args.batch_size]
            batch_images: list[Image.Image] = []
            batch_ids: list[str] = []
            for row in chunk:
                try:
                    with Image.open(dataset_dir / row.path) as handle:
                        batch_images.append(handle.copy())
                    batch_ids.append(row.image_id)
                except Exception as error:  # noqa: BLE001
                    failures.append((row.image_id, str(error)))
            if not batch_images:
                continue
            blocks.append(extractor.embed_batch(batch_images))
            image_ids.extend(batch_ids)

        if failures:
            logger.warning(
                "%d image(s) failed to load and were skipped, e.g. %s: %s",
                len(failures), failures[0][0], failures[0][1],
            )
        embeddings = (
            np.concatenate(blocks, axis=0)
            if blocks
            else np.zeros((0, description["expected_dim"]), dtype=np.float32)
        )
        save_embeddings(
            cache_path,
            image_ids,
            embeddings,
            {
                "dataset": args.dataset,
                "n_images": len(image_ids),
                "n_failed": len(failures),
                **description,
            },
        )
        logger.info(
            "Wrote %s embeddings of shape %s to %s",
            backbone, embeddings.shape, cache_path,
        )


if __name__ == "__main__":
    main()