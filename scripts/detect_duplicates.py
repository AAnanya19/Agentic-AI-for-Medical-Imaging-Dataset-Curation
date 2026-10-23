#!/usr/bin/env python3
"""Stage 6: resolution checks and duplicate detection across three distinct tasks.

The deduplication literature separates three problems that are routinely
conflated. They need different methods, different ground truth, and different
expectations - reporting them as one "duplicate detection" number is what makes
a correct result look like a failure.

  1. ``exact`` - pixel-identical images. Needs no external ground truth:
     equality of a perceptual hash is itself the evidence. Hashing is known to
     be near-perfect here.

  2. ``transformation`` - the same source image after a transformation (resize,
     compression, noise, blur). **Ground truth is ``original_image_id``**: this
     project's injected defects are exactly transformation-induced near
     duplicates, nine types at three severities per source, with perfect
     provenance. This mirrors the methodology of published medical-imaging
     deduplication benchmarks, which generate transformed copies to evaluate
     against, so the numbers here are directly comparable to them. Embedding
     similarity is expected to beat hashing.

  3. ``reidentification`` - different acquisitions of the same subject (a
     patient's follow-up radiograph). **Ground truth is a subject column such as
     ``patient_id``.** Two studies of one patient months apart are *not* near
     duplicates; they are semantically related but pixel-distinct, so a
     perceptual hash cannot detect them at any threshold and frozen generic
     embeddings do so only weakly. Purpose-trained deep metric learning reaches
     AUC ~0.994 on ChestX-ray14 (Packhaeuser et al., Sci. Rep. 2022), which is
     the appropriate baseline to cite - not a bar untrained features should be
     expected to clear. A near-zero result here is a correct, citable negative
     finding, not a broken detector.

Row selection follows the task automatically: ``transformation`` needs the
degraded rows (they *are* the transformed copies), while ``exact`` and
``reidentification`` use clean rows only, since degraded variants would
otherwise be rediscovered as duplicates by construction.
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
    from pipeline.duplicates import (
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_HASH_SIZE,
        DEFAULT_NEAR_HASH_DISTANCE,
        HASH_TYPES,
        check_split_leakage,
        compute_hashes,
        evaluate_against_groups,
        find_embedding_duplicates,
        find_hash_duplicates,
        hash_saturation,
        pairs_to_frame,
        sweep_cosine_thresholds,
        sweep_thresholds,
    )
    from pipeline.embeddings import load_embeddings
    from pipeline.resolution import (
        ResolutionThresholds,
        check_resolution,
        summarise_resolution,
    )
except ModuleNotFoundError:  # Allow `python scripts/detect_duplicates.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.common import configure_logging  # type: ignore[no-redef]
    from pipeline.duplicates import (  # type: ignore[no-redef]
        DEFAULT_COSINE_THRESHOLD,
        DEFAULT_HASH_SIZE,
        DEFAULT_NEAR_HASH_DISTANCE,
        HASH_TYPES,
        check_split_leakage,
        compute_hashes,
        evaluate_against_groups,
        find_embedding_duplicates,
        find_hash_duplicates,
        hash_saturation,
        pairs_to_frame,
        sweep_cosine_thresholds,
        sweep_thresholds,
    )
    from pipeline.embeddings import load_embeddings  # type: ignore[no-redef]
    from pipeline.resolution import (  # type: ignore[no-redef]
        ResolutionThresholds,
        check_resolution,
        summarise_resolution,
    )


TASKS = ("exact", "transformation", "reidentification")

TASK_NOTES = {
    "exact": (
        "pixel-identical images; hash equality is self-evidencing so no external "
        "ground truth is required"
    ),
    "transformation": (
        "same source image after a transformation; ground truth = "
        "original_image_id, comparable to published dedup benchmarks"
    ),
    "reidentification": (
        "different acquisitions of the same subject; ground truth = subject "
        "column. Not a near-duplicate problem - expect hashing to score zero"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--tasks", default="all",
        help=f"Comma-separated subset of {TASKS}, or 'all'",
    )
    parser.add_argument(
        "--group-column",
        help="Column in metadata.csv identifying same-subject groups "
        "(patient_id for ChestX-ray14). Required for the reidentification task; "
        "ISIC persists none, so that task is skipped there rather than faked.",
    )
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--hash-distance", type=int, default=DEFAULT_NEAR_HASH_DISTANCE)
    parser.add_argument("--hash-size", type=int, default=DEFAULT_HASH_SIZE)
    parser.add_argument(
        "--cosine-threshold", type=float, default=DEFAULT_COSINE_THRESHOLD
    )
    parser.add_argument("--skip-hashing", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument(
        "--max-transformation-sources", type=int, default=150,
        help="Cap on source images for the transformation task. Pair search is "
        "O(n^2) and each source contributes up to 27 degraded variants, so the "
        "full corpus is intractable without sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_tasks(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return TASKS
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [task for task in requested if task not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s) {unknown}; available: {list(TASKS)}")
    return requested


def rows_for_task(
    task: str,
    processed: pd.DataFrame,
    dataset_dir: Path,
    max_sources: int,
    seed: int,
    logger,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Candidate rows and ground-truth mapping appropriate to one task."""
    if task == "transformation":
        # inject_defects.py applies exactly ONE treatment per source image, so
        # original_image_id groups in the main corpus are all singletons and
        # contain no near-duplicate pairs at all. The transformation task
        # therefore reads a dedicated benchmark set built by
        # build_duplicate_benchmark.py, whose group_id column gives every pair
        # within a group as known ground truth.
        benchmark_path = (
            dataset_dir / "duplicate_benchmark" / "benchmark_metadata.csv"
        )
        if not benchmark_path.is_file():
            raise FileNotFoundError(
                f"{benchmark_path} not found. The transformation task needs a "
                "dedicated near-duplicate set, because the main corpus has one "
                "treatment per source image and therefore no duplicate pairs. "
                "Run: python scripts/build_duplicate_benchmark.py --dataset "
                f"{dataset_dir.name}"
            )
        benchmark = pd.read_csv(benchmark_path, dtype=str, keep_default_na=False)
        truth = dict(zip(benchmark["image_id"], benchmark["group_id"]))
        return benchmark, truth

    # exact and reidentification use clean rows only: including degraded rows
    # would rediscover this project's own defect variants as duplicates.
    return processed[processed["defect_type"] == "clean"], {}


def subject_groups(
    dataset_dir: Path, candidates: pd.DataFrame, group_column: str
) -> dict[str, str]:
    source_path = dataset_dir / "metadata.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    if group_column not in source.columns:
        raise ValueError(
            f"--group-column {group_column!r} not in {source_path}; "
            f"available: {sorted(source.columns)}"
        )
    lookup = dict(zip(source["image_id"], source[group_column]))
    return {
        row.image_id: lookup[row.original_image_id]
        for row in candidates.itertuples(index=False)
        if row.original_image_id in lookup
    }


def run_task(
    task: str,
    args: argparse.Namespace,
    dataset_dir: Path,
    processed: pd.DataFrame,
    logs_dir: Path,
    logger,
) -> dict[str, Any]:
    logger.info("")
    logger.info("=" * 88)
    logger.info("TASK: %s", task.upper())
    logger.info("  %s", TASK_NOTES[task])
    logger.info("=" * 88)

    candidates, truth = rows_for_task(
        task, processed, dataset_dir, args.max_transformation_sources,
        args.seed, logger,
    )
    if candidates.empty:
        logger.warning("  no candidate rows for this task; skipping")
        return {"skipped": "no candidate rows"}

    if task == "reidentification":
        if not args.group_column:
            logger.warning(
                "  SKIPPED: no --group-column supplied. This dataset persists no "
                "subject identifier, so same-subject ground truth does not exist "
                "and any precision/recall would be fabricated."
            )
            return {"skipped": "no subject column available in this dataset"}
        truth = subject_groups(dataset_dir, candidates, args.group_column)
        counts = pd.Series(list(truth.values())).value_counts()
        logger.info(
            "  %d row(s); grouped by %s into %d group(s), %d with >1 image",
            len(candidates), args.group_column, len(counts),
            int((counts > 1).sum()),
        )
    else:
        logger.info("  %d candidate row(s)", len(candidates))
        if truth:
            counts = pd.Series(list(truth.values())).value_counts()
            n_true_pairs = int((counts * (counts - 1) // 2).sum())
            logger.info(
                "  ground truth: %d group(s), %d with >1 member, %d true pair(s); "
                "%d single-copy distractor(s)",
                len(counts), int((counts > 1).sum()), n_true_pairs,
                int((counts == 1).sum()),
            )

    result: dict[str, Any] = {
        "n_candidate_rows": int(len(candidates)),
        "ground_truth_available": bool(truth),
    }
    all_pairs: list[Any] = []
    image_ids = list(candidates["image_id"])

    # ---------------- Hashing ----------------
    if not args.skip_hashing:
        paths = {
            row.image_id: dataset_dir / row.path
            for row in candidates.itertuples(index=False)
        }
        hashes, failures = compute_hashes(paths, hash_size=args.hash_size)
        logger.info(
            "  hashed %d image(s) at hash_size=%d; %d failure(s)",
            len(hashes), args.hash_size, len(failures),
        )
        saturation = hash_saturation(hashes, "phash")
        # Saturation only indicates a problem where hash collisions between
        # DIFFERENT subjects would be false positives. On the transformation
        # task, collisions between variants of the same source are precisely
        # what is being detected, so a low distinct ratio is expected there and
        # the diagnostic is reported without a warning.
        if saturation["saturated"] and task != "transformation":
            logger.warning(
                "  pHash SATURATED (%d distinct value(s) for %d image(s)): "
                "different images share a hash, so verdicts below are "
                "unreliable - raise --hash-size",
                saturation["n_distinct_hashes"], saturation["n_images"],
            )
        elif task == "transformation":
            logger.info(
                "  hash resolution: %d distinct value(s) for %d image(s) "
                "(collisions within a group are expected on this task)",
                saturation["n_distinct_hashes"], saturation["n_images"],
            )
        result["hash_saturation"] = saturation

        logger.info("")
        logger.info(
            "  %-7s %8s %10s %11s %8s",
            "hash", "exact", f"d<={args.hash_distance}", "precision", "recall",
        )
        for hash_type in HASH_TYPES:
            exact = find_hash_duplicates(hashes, 0, hash_type)
            near = find_hash_duplicates(hashes, args.hash_distance, hash_type)
            entry: dict[str, Any] = {
                "n_exact_pairs": len(exact),
                "n_near_pairs": len(near) - len(exact),
            }
            if task == "exact":
                logger.info(
                    "  %-7s %8d %10s %11s %8s",
                    hash_type, len(exact), "-", "self-evident", "-",
                )
            elif truth:
                metrics = evaluate_against_groups(near, truth, image_ids)
                entry["validation"] = metrics
                logger.info(
                    "  %-7s %8d %10d %11.3f %8.3f",
                    hash_type, len(exact), entry["n_near_pairs"],
                    metrics["precision"], metrics["recall"],
                )
            result[f"hash_{hash_type}"] = entry
            if hash_type == "phash":
                all_pairs.extend(exact if task == "exact" else near)

        if truth and task != "exact":
            sweep = sweep_thresholds(hashes, truth, image_ids)
            sweep.to_csv(
                logs_dir / f"{args.dataset}_dup_{task}_hash_sweep.csv", index=False
            )
            if sweep["f1"].notna().any():
                best = sweep.loc[sweep["f1"].idxmax()]
                logger.info(
                    "  best pHash distance by F1: %d (precision %.3f, recall %.3f)",
                    int(best["hash_distance"]), best["precision"], best["recall"],
                )
                result["best_hash_distance_by_f1"] = int(best["hash_distance"])

    # ---------------- Embeddings ----------------
    if not args.skip_embeddings:
        # The transformation task runs on the benchmark set, whose image_ids do
        # not exist in the main corpus cache, so it needs its own extraction:
        #   extract_embeddings.py --dataset <name> --subset benchmark
        suffix = "_benchmark" if task == "transformation" else ""
        cache_path = (
            args.data_dir / "embeddings"
            / f"{args.dataset}{suffix}_{args.backbone}.npz"
        )
        if not cache_path.is_file():
            logger.warning(
                "  no embedding cache at %s; the embedding method will be "
                "MISSING from this task's comparison. Build it with: "
                "python scripts/extract_embeddings.py --dataset %s%s",
                cache_path, args.dataset,
                " --subset benchmark" if task == "transformation" else "",
            )
        else:
            cached_ids, embeddings, _ = load_embeddings(cache_path)
            wanted = set(image_ids)
            keep = [i for i, name in enumerate(cached_ids) if name in wanted]
            subset_ids = [cached_ids[i] for i in keep]
            subset = embeddings[keep]
            pairs = find_embedding_duplicates(
                subset_ids, subset, args.cosine_threshold
            )
            logger.info("")
            logger.info(
                "  embedding (%s): %d pair(s) at cosine >= %.2f, from %d matched "
                "embedding(s)",
                args.backbone, len(pairs), args.cosine_threshold, len(subset_ids),
            )
            entry: dict[str, Any] = {
                "n_pairs": len(pairs),
                "n_embeddings_matched": len(subset_ids),
            }
            if truth:
                metrics = evaluate_against_groups(pairs, truth, subset_ids)
                entry["validation"] = metrics
                logger.info(
                    "  embedding precision %.3f  recall %.3f  f1 %.3f",
                    metrics["precision"], metrics["recall"], metrics["f1"],
                )
                sweep = sweep_cosine_thresholds(subset_ids, subset, truth)
                sweep.to_csv(
                    logs_dir / f"{args.dataset}_dup_{task}_cosine_sweep.csv",
                    index=False,
                )
                for row in sweep.itertuples(index=False):
                    logger.info(
                        "    cosine %.2f: precision %.3f  recall %.3f  (%d pair(s))",
                        row.cosine_threshold, row.precision, row.recall,
                        int(row.n_detected_pairs),
                    )
                if sweep["f1"].notna().any():
                    best = sweep.loc[sweep["f1"].idxmax()]
                    entry["best_cosine_by_f1"] = float(best["cosine_threshold"])
                    logger.info(
                        "  best cosine by F1: %.2f (precision %.3f, recall %.3f)",
                        best["cosine_threshold"], best["precision"], best["recall"],
                    )
            result["embedding"] = entry
            all_pairs.extend(pairs)

    # ---------------- Split leakage ----------------
    if all_pairs:
        split_by_image = dict(zip(candidates["image_id"], candidates["split"]))
        leakage = check_split_leakage(all_pairs, split_by_image)
        logger.info("")
        logger.info(
            "  split leakage: %d pair(s) within one split, %d CROSSING",
            leakage["n_pairs_within_one_split"],
            leakage["n_pairs_crossing_splits"],
        )
        if leakage["n_pairs_crossing_splits"]:
            example = leakage["crossing_examples"][0]
            logger.warning(
                "  cross-split duplicates inflate held-out performance; e.g. "
                "%s (%s) <-> %s (%s)",
                example["image_id_a"], example["split_a"],
                example["image_id_b"], example["split_b"],
            )
        result["split_leakage"] = leakage
        pairs_to_frame(all_pairs).to_csv(
            logs_dir / f"{args.dataset}_dup_{task}_pairs.csv", index=False
        )

    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = resolve_tasks(args.tasks)
    logger = configure_logging(
        args.data_dir, args.dataset, False, f"{args.dataset}_duplicates.log"
    )
    dataset_dir = args.data_dir / args.dataset
    logs_dir = args.data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    processed_path = dataset_dir / "processed_metadata.csv"
    if not processed_path.is_file():
        raise FileNotFoundError(
            f"{processed_path} not found; run inject_defects.py first"
        )
    processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
    logger.info("Loaded %d processed row(s)", len(processed))

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "tasks_requested": list(tasks),
        "task_definitions": TASK_NOTES,
        "config": {
            "group_column": args.group_column,
            "hash_size": args.hash_size,
            "hash_distance": args.hash_distance,
            "cosine_threshold": args.cosine_threshold,
            "hash_types": list(HASH_TYPES),
            "max_transformation_sources": args.max_transformation_sources,
            "seed": args.seed,
        },
    }

    # ---------------- Resolution ----------------
    logger.info("")
    logger.info("=" * 88)
    logger.info("RESOLUTION CHECK (from recorded source dimensions, not pixels)")
    logger.info("=" * 88)
    resolution = check_resolution(processed, ResolutionThresholds())
    resolution_summary = summarise_resolution(resolution)
    logger.info(
        "  measurable %d/%d; %d flagged  (below min side %d, extreme aspect %d, "
        "small for dataset %d)",
        resolution_summary["n_measurable"], resolution_summary["n_rows"],
        resolution_summary["n_flagged"], resolution_summary["n_below_min_side"],
        resolution_summary["n_extreme_aspect_ratio"],
        resolution_summary["n_small_for_dataset"],
    )
    if resolution_summary["outlier_check_skipped_uniform_resolution"]:
        logger.info(
            "  small-for-dataset check skipped: this dataset has uniform source "
            "resolution, so it has no size outliers by definition"
        )
    megapixels = resolution_summary["megapixels"]
    if megapixels["median"] is not None:
        logger.info(
            "  source megapixels: min %.2f  median %.2f  max %.2f",
            megapixels["min"], megapixels["median"], megapixels["max"],
        )
    resolution.to_csv(logs_dir / f"{args.dataset}_resolution.csv", index=False)
    summary["resolution"] = resolution_summary

    # ---------------- Duplicate tasks ----------------
    summary["tasks"] = {}
    for task in tasks:
        summary["tasks"][task] = run_task(
            task, args, dataset_dir, processed, logs_dir, logger
        )

    # ---------------- Cross-task comparison ----------------
    logger.info("")
    logger.info("=" * 88)
    logger.info("SUMMARY ACROSS TASKS")
    logger.info("=" * 88)
    for task in tasks:
        entry = summary["tasks"][task]
        if "skipped" in entry:
            logger.info("  %-17s SKIPPED (%s)", task, entry["skipped"])
            continue
        if task == "exact":
            logger.info(
                "  %-17s pHash exact pairs %d (no ground truth required)",
                task, entry.get("hash_phash", {}).get("n_exact_pairs", 0),
            )
            continue
        phash = entry.get("hash_phash", {}).get("validation", {})
        embedding = entry.get("embedding", {}).get("validation", {})
        logger.info(
            "  %-17s pHash f1 %-7s  embedding f1 %-7s",
            task,
            f"{phash['f1']:.3f}" if phash and np.isfinite(phash.get("f1", np.nan))
            else "n/a",
            f"{embedding['f1']:.3f}"
            if embedding and np.isfinite(embedding.get("f1", np.nan)) else "n/a",
        )
    logger.info("")
    logger.info(
        "Note: a near-zero reidentification result is expected and citable. "
        "Purpose-trained deep metric learning reaches AUC ~0.994 on "
        "ChestX-ray14 (Packhaeuser et al., Sci. Rep. 2022); frozen generic "
        "features are not expected to match it."
    )

    (logs_dir / f"{args.dataset}_duplicates_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    logger.info("")
    logger.info("Wrote per-task reports and summary to %s", logs_dir)


if __name__ == "__main__":
    main()