#!/usr/bin/env python3
"""Measure burned-in annotation across a processed corpus.

    python scripts/measure_annotation.py --dataset chestxray14 --data-dir data

Two things are established here. The first is descriptive and requires no ground
truth: how much of the corpus carries burned-in annotation at all, and at what
extent. The second is a decision threshold derived the same way as every other
threshold in the system, from the clean validation distribution at a stated
false-positive rate.

The second carries a caveat that must be stated wherever it is used. Every other
check is calibrated against images known to be undegraded, because the pipeline
injected the degradation and knows which images it touched. No such ground truth
exists for annotation: the clean validation images are clean only in the sense
that nothing was injected into them, and many carry burned-in text already. A
threshold at the fifth percentile of that distribution therefore flags the most
heavily annotated images relative to the corpus, not images that a radiologist
would call inadequate. It is a relative measure, and reporting it as anything
else would overstate what it establishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from annotation_metric import annotation_fraction, saturated_border_fraction
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.annotation import annotation_fraction, saturated_border_fraction


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0,
                   help="Process only this many rows, for a quick check")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_dir = args.data_dir / args.dataset
    processed = pd.read_csv(dataset_dir / "processed_metadata.csv",
                            dtype=str, keep_default_na=False)
    if args.limit:
        processed = processed.head(args.limit)

    records = []
    for i, row in enumerate(processed.itertuples(index=False), 1):
        path = dataset_dir / "processed" / row.filename \
            if hasattr(row, "filename") else None
        if path is None or not path.is_file():
            candidates = list((dataset_dir / "processed").glob(f"{row.image_id}*"))
            if not candidates:
                continue
            path = candidates[0]
        try:
            with Image.open(path) as handle:
                array = np.asarray(handle.convert("L"), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        records.append({
            "image_id": row.image_id,
            "split": row.split,
            "defect_type": row.defect_type,
            "annotation_fraction": annotation_fraction(array),
            "saturated_border_fraction": saturated_border_fraction(array),
        })
        if i % 500 == 0:
            print(f"  {i} of {len(processed)}")

    frame = pd.DataFrame(records)
    out = args.data_dir / "logs" / f"{args.dataset}_annotation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    clean = frame[frame.defect_type == "clean"]
    print()
    print("=" * 68)
    print(f"BURNED-IN ANNOTATION, {args.dataset}")
    print("=" * 68)
    print(f"  images measured            : {len(frame)}")
    print(f"  of which undegraded        : {len(clean)}")
    print()
    for column in ("annotation_fraction", "saturated_border_fraction"):
        values = clean[column]
        print(f"  {column}")
        print(f"     median {values.median():.4f}   mean {values.mean():.4f}"
              f"   90th pct {values.quantile(0.90):.4f}"
              f"   max {values.max():.4f}")
        for level in (0.001, 0.01, 0.05):
            share = (values > level).mean()
            print(f"     above {level:.3f}: {share:6.1%} of undegraded images")
        print()

    thresholds = {}
    for column in ("annotation_fraction", "saturated_border_fraction"):
        clean_values = clean[clean.split == "val"][column]
        if len(clean_values) >= 30:
            t = float(clean_values.quantile(1 - args.target_fpr))
            thresholds[column] = {
                "flag": t,
                "target_fpr": args.target_fpr,
                "n_clean_validation_rows": int(len(clean_values)),
                "caveat": ("derived from images that were not injected with a "
                           "defect but that may carry annotation already; this "
                           "is a relative measure, not a clinical standard"),
            }
            print(f"  {column}: flag above {t:.4f} "
                  f"(from {len(clean_values)} clean validation rows)")

    (args.data_dir / "logs" / f"{args.dataset}_annotation_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n")
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
