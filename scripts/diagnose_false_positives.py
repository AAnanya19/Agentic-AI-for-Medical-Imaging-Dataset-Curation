#!/usr/bin/env python3
"""Diagnose confident false positives: clean images rejected far past threshold.

A marginal false positive is the operating point behaving as configured. A
*confident* one means the scorer is reacting to some property of those images
that is not a quality defect - which is either a real finding about the dataset
or a bug, and the two need distinguishing before any of it is written up.

Four hypotheses are tested against the Inspector's own output:

  1. **One check dominates.** If a single check fires on every confident false
     positive, the problem is that check, not the composite.
  2. **A shared source property.** If the affected images share a resolution,
     aspect ratio or class label, the scorer is keying on a source characteristic
     rather than on quality.
  3. **Threshold instability.** If the composite scores of the affected images sit
     within the spread of the clean validation distribution, the threshold is
     simply set too tight for the available sample.
  4. **Genuinely atypical images.** If they are far from the clean distribution on
     several independent checks, they may be real outliers that the injected-defect
     ground truth mislabels as clean - a limitation of synthetic ground truth
     rather than of the detector.

Run:
    python scripts/diagnose_false_positives.py --dataset isic --data-dir data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--top", type=int, default=8, help="How many affected images to detail"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logs_dir = args.data_dir / "logs"
    dataset_dir = args.data_dir / args.dataset

    inspection = json.loads(
        (logs_dir / f"{args.dataset}_inspection.json").read_text()
    )
    images = inspection["images"]
    thresholds = inspection["thresholds"]

    clean = [i for i in images if i.get("ground_truth_defect_type") == "clean"]
    false_positives = [i for i in clean if i["decision"] != "accept"]
    confident = [
        i for i in false_positives
        if i.get("confidence", {}).get("band") == "clear"
    ]
    marginal = [
        i for i in false_positives
        if i.get("confidence", {}).get("band") == "borderline"
    ]

    print(f"=== {args.dataset} ===")
    print(f"clean images:            {len(clean)}")
    print(f"false positives:         {len(false_positives)}")
    print(f"  of which confident:    {len(confident)}")
    print(f"  of which borderline:   {len(marginal)}")
    if not confident:
        print("\nNo confident false positives; nothing to diagnose.")
        return

    # ---- Hypothesis 1: one check dominates -----------------------------
    print("\n--- H1: does one check drive them? ---")
    fired = Counter()
    for image in confident:
        for name in image.get("flagged_checks", []) + image.get(
            "attribution_only_checks", []
        ):
            fired[name] += 1
    for name, count in fired.most_common():
        share = count / len(confident)
        marker = "  <-- fires on ALL of them" if count == len(confident) else ""
        print(f"  {name:20s} {count:3d}/{len(confident)} ({share:.0%}){marker}")

    print("\n  primary issue attributed:")
    for name, count in Counter(
        i.get("primary_issue") for i in confident
    ).most_common():
        print(f"    {str(name):20s} {count}")

    # ---- Hypothesis 3: threshold instability ---------------------------
    print("\n--- H3: are they inside the clean validation spread? ---")
    overall = thresholds.get("overall_quality", {})
    distances = [
        i["confidence"]["distance_from_boundary"] for i in confident
        if i.get("confidence", {}).get("distance_from_boundary") is not None
    ]
    if distances:
        print(f"  distance past the boundary (threshold spans): "
              f"min {min(distances):.2f}, median {np.median(distances):.2f}, "
              f"max {max(distances):.2f}")
        print(f"  thresholds derived from {overall.get('n_clean_validation_rows')} "
              "clean validation image(s)")
        if overall.get("low_sample_warning"):
            print(f"  NOTE: {overall['low_sample_warning']}")
        if max(distances) < 1.0:
            print("  -> all within one threshold-span: consistent with a tight "
                  "threshold rather than genuinely atypical images")
        else:
            print("  -> some sit well beyond the boundary: not explained by "
                  "threshold placement alone")

    # ---- Per-image detail ---------------------------------------------
    print(f"\n--- worst {min(args.top, len(confident))} cases in detail ---")
    ordered = sorted(
        confident,
        key=lambda i: i.get("confidence", {}).get("distance_from_boundary") or 0.0,
        reverse=True,
    )
    for image in ordered[: args.top]:
        distance = image.get("confidence", {}).get("distance_from_boundary")
        print(f"\n  {image['image_id']}  decision={image['decision']}  "
              f"distance={distance:.2f}" if distance is not None
              else f"\n  {image['image_id']}  decision={image['decision']}")
        for name, result in image.get("checks", {}).items():
            if not result.get("flagged") or result.get("score") is None:
                continue
            print(f"    {name:20s} score {result['score']:>12.4f}  "
                  f"threshold {result.get('flag_threshold', float('nan')):>12.4f}")

    # ---- Hypothesis 2: a shared source property ------------------------
    print("\n--- H2: do they share a source property? ---")
    processed_path = dataset_dir / "processed_metadata.csv"
    source_path = dataset_dir / "metadata.csv"
    if not (processed_path.is_file() and source_path.is_file()):
        print("  metadata not available; skipping")
        return

    processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    affected_ids = {i["image_id"] for i in confident}
    clean_ids = {i["image_id"] for i in clean}

    link = processed[processed["image_id"].isin(clean_ids)][
        ["image_id", "original_image_id", "defect_params_json"]
    ].copy()
    link["is_affected"] = link["image_id"].isin(affected_ids)

    def source_dimensions(raw: str) -> tuple[float, float]:
        try:
            info = json.loads(raw)["baseline_preprocessing"]["canonical_resize"]
            return float(info["original_width"]), float(info["original_height"])
        except Exception:  # noqa: BLE001
            return float("nan"), float("nan")

    dimensions = link["defect_params_json"].apply(source_dimensions)
    link["width"] = [d[0] for d in dimensions]
    link["height"] = [d[1] for d in dimensions]
    link["megapixels"] = link["width"] * link["height"] / 1e6
    link["aspect_ratio"] = link[["width", "height"]].max(axis=1) / link[
        ["width", "height"]
    ].min(axis=1)

    affected = link[link["is_affected"]]
    rest = link[~link["is_affected"]]
    print(f"  {'property':16s} {'affected':>18s} {'other clean':>18s}")
    for column in ("megapixels", "aspect_ratio", "width"):
        a, b = affected[column].dropna(), rest[column].dropna()
        if a.empty or b.empty:
            continue
        print(f"  {column:16s} {a.median():>18.3f} {b.median():>18.3f}")
        # A large median gap on a source property is the signature of the
        # scorer keying on acquisition characteristics rather than quality.
        if b.median() and abs(a.median() - b.median()) / b.median() > 0.5:
            print(f"    -> affected images differ markedly on {column}")

    label_column = "diagnosis" if "diagnosis" in source.columns else "source_labels"
    if label_column in source.columns:
        labels = dict(zip(source["image_id"], source[label_column]))
        affected_labels = Counter(
            labels.get(row.original_image_id, "?") for row in affected.itertuples()
        )
        all_labels = Counter(
            labels.get(row.original_image_id, "?") for row in link.itertuples()
        )
        print(f"\n  {label_column} of affected images:")
        for label, count in affected_labels.most_common(6):
            total = all_labels.get(label, 0)
            share = count / total if total else 0.0
            print(f"    {label[:40]:42s} {count}/{total} ({share:.0%} of that class)")


if __name__ == "__main__":
    main()