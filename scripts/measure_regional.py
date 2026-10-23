#!/usr/bin/env python3
"""Measure regional quality across a processed corpus and derive thresholds.

    python scripts/measure_regional.py --dataset chestxray14 --data-dir data

Unlike the annotation measurement, this can be calibrated in the ordinary way.
The clean validation images are clean with respect to regional contrast in the
sense that matters: whatever regional variation they carry is the variation the
corpus naturally exhibits, so a threshold at a stated percentile of their
distribution flags images that are unusual relative to it. The distinction from
annotation is that regional contrast is a continuous property every image has,
rather than a defect that is either present or absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from regional_metrics import regional_contrast, regional_sharpness
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.regional import regional_contrast, regional_sharpness


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_dir = args.data_dir / args.dataset
    processed = pd.read_csv(dataset_dir / "processed_metadata.csv",
                            dtype=str, keep_default_na=False)
    if args.limit:
        processed = processed.head(args.limit)

    rows = []
    for i, row in enumerate(processed.itertuples(index=False), 1):
        candidates = list((dataset_dir / "processed").glob(f"{row.image_id}*"))
        if not candidates:
            continue
        try:
            with Image.open(candidates[0]) as handle:
                array = np.asarray(handle.convert("L"), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "image_id": row.image_id,
            "split": row.split,
            "defect_type": row.defect_type,
            "severity_level": row.severity_level,
            "regional_contrast": regional_contrast(array),
            "regional_sharpness": regional_sharpness(array),
        })
        if i % 500 == 0:
            print(f"  {i} of {len(processed)}")

    frame = pd.DataFrame(rows)
    out = args.data_dir / "logs" / f"{args.dataset}_regional.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    clean = frame[frame.defect_type == "clean"]
    print()
    print("=" * 70)
    print(f"REGIONAL QUALITY, {args.dataset}")
    print("=" * 70)
    print(f"  images measured     : {len(frame)}")
    print(f"  of which undegraded : {len(clean)}")
    print()

    thresholds = {}
    for column, direction in (("regional_contrast", "lower is worse"),
                              ("regional_sharpness", "lower is worse")):
        values = clean[column].dropna()
        print(f"  {column}  ({direction})")
        print(f"     min {values.min():.4f}   5th pct {values.quantile(0.05):.4f}"
              f"   median {values.median():.4f}   max {values.max():.4f}")
        val = clean[clean.split == "val"][column].dropna()
        if len(val) >= 30:
            t = float(val.quantile(args.target_fpr))
            thresholds[column] = {
                "flag": t,
                "polarity": "lower_is_worse",
                "target_fpr": args.target_fpr,
                "n_clean_validation_rows": int(len(val)),
            }
            print(f"     flag below {t:.4f} (from {len(val)} clean validation rows)")
        print()

    # Does it detect the injected defects the global metrics detect?
    print("  detection of injected defects, worst-tile contrast:")
    if "regional_contrast" in thresholds:
        t = thresholds["regional_contrast"]["flag"]
        for defect in sorted(d for d in frame.defect_type.unique() if d != "clean"):
            sub = frame[frame.defect_type == defect]["regional_contrast"].dropna()
            if len(sub):
                print(f"     {defect:20s} fires on {(sub < t).mean():6.1%}")

    (args.data_dir / "logs" /
     f"{args.dataset}_regional_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n")
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()