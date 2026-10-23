#!/usr/bin/env python3
"""Diagnose whether severe defects survive application and JPEG saving.

This deliberately bypasses ``scripts/inject_defects.py`` and its Matplotlib
visualization. Direct comparisons call the named functions in
``pipeline.defects``; comparison images are assembled with Pillow alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.defects import (  # noqa: E402
    gaussian_blur,
    gaussian_noise,
    motion_blur,
    prepare_uint8_pil,
    rotation_crop,
    to_uint8_pil,
)


SEVERITY = 3
DEFAULT_SOURCE_IMAGE_ID = "ISIC_0069884"
DIRECT_DEFECTS: dict[str, Callable[..., Any]] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "rotation_crop": rotation_crop,
}
SAVED_DEFECTS = ("gaussian_noise", "gaussian_blur")


@dataclass(frozen=True)
class PixelDifference:
    """Full-resolution, per-channel pixel-difference statistics."""

    mean_absolute_difference: float
    max_absolute_difference: float
    difference_standard_deviation: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="isic")
    parser.add_argument(
        "--source-image-id",
        default=DEFAULT_SOURCE_IMAGE_ID,
        help="One raw source used for all four direct severity-3 calls",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <data-dir>/logs/defect_application_diagnostics",
    )
    return parser.parse_args(argv)


def resolve_metadata_path(path_value: str, dataset_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else dataset_dir / path


def load_image(path: Path) -> Image.Image:
    """Fully decode an image and detach it from the underlying file handle."""
    with Image.open(path) as image:
        image.load()
        return image.copy()


def comparable_pair(
    original: Image.Image | np.ndarray,
    degraded: Image.Image | np.ndarray,
) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """Return like-shaped uint8 images and float arrays without wraparound."""
    original_image = to_uint8_pil(original)
    degraded_image = to_uint8_pil(degraded)
    if original_image.size != degraded_image.size:
        raise ValueError(
            f"image dimensions differ: {original_image.size} vs {degraded_image.size}"
        )
    if original_image.mode != degraded_image.mode:
        comparison_mode = (
            "RGB"
            if "RGB" in {original_image.mode, degraded_image.mode}
            else "L"
        )
        original_image = original_image.convert(comparison_mode)
        degraded_image = degraded_image.convert(comparison_mode)

    original_array = np.asarray(original_image, dtype=np.float32)
    degraded_array = np.asarray(degraded_image, dtype=np.float32)
    if original_array.shape != degraded_array.shape:
        raise ValueError(
            f"image array shapes differ: {original_array.shape} vs {degraded_array.shape}"
        )
    return original_image, degraded_image, original_array, degraded_array


def pixel_difference(
    original: Image.Image | np.ndarray,
    degraded: Image.Image | np.ndarray,
) -> PixelDifference:
    """Measure signed and absolute differences at the original resolution."""
    _, _, original_array, degraded_array = comparable_pair(original, degraded)
    difference = degraded_array - original_array
    absolute_difference = np.abs(difference)
    return PixelDifference(
        mean_absolute_difference=float(absolute_difference.mean()),
        max_absolute_difference=float(absolute_difference.max()),
        difference_standard_deviation=float(difference.std()),
    )


def save_side_by_side(
    original: Image.Image | np.ndarray,
    degraded: Image.Image | np.ndarray,
    destination: Path,
) -> None:
    """Save a full-resolution comparison using Pillow, not Matplotlib."""
    original_image, degraded_image, _, _ = comparable_pair(original, degraded)
    original_display = original_image.convert("RGB")
    degraded_display = degraded_image.convert("RGB")
    width, height = original_display.size
    gap = 8
    heading_height = 28
    canvas = Image.new(
        "RGB",
        (width * 2 + gap, height + heading_height),
        color="white",
    )
    canvas.paste(original_display, (0, heading_height))
    canvas.paste(degraded_display, (width + gap, heading_height))
    labels = ImageDraw.Draw(canvas)
    labels.text((8, 7), "Original", fill="black")
    labels.text((width + gap + 8, 7), "Degraded", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def source_row_for_id(metadata: pd.DataFrame, image_id: str) -> pd.Series:
    matches = metadata.loc[metadata["image_id"] == image_id]
    if matches.empty:
        raise ValueError(f"source image_id not found in metadata.csv: {image_id}")
    if len(matches) != 1:
        raise ValueError(f"source image_id is not unique in metadata.csv: {image_id}")
    return matches.iloc[0]


def first_saved_example(
    processed_metadata: pd.DataFrame,
    source_metadata: pd.DataFrame,
    dataset_dir: Path,
    defect_name: str,
) -> tuple[pd.Series, pd.Series, Path, Path]:
    """Find the first loadable real saved example using CSV paths verbatim."""
    candidates = processed_metadata.loc[
        (processed_metadata["defect_type"] == defect_name)
        & (processed_metadata["severity_level"] == str(SEVERITY))
    ]
    if candidates.empty:
        raise ValueError(f"no saved severity-{SEVERITY} {defect_name} row found")

    source_by_id = source_metadata.set_index("image_id", drop=False)
    errors: list[str] = []
    for _, processed_row in candidates.iterrows():
        original_id = str(processed_row["original_image_id"])
        if original_id not in source_by_id.index:
            errors.append(f"{original_id}: absent from source metadata")
            continue
        source_row = source_by_id.loc[original_id]
        if isinstance(source_row, pd.DataFrame):
            errors.append(f"{original_id}: duplicate source metadata rows")
            continue
        source_path = resolve_metadata_path(str(source_row["path"]), dataset_dir)
        processed_path = resolve_metadata_path(str(processed_row["path"]), dataset_dir)
        if not source_path.is_file() or not processed_path.is_file():
            errors.append(f"{original_id}: source or processed file missing")
            continue
        try:
            with Image.open(source_path) as source_header, Image.open(
                processed_path
            ) as processed_header:
                if source_header.size != processed_header.size:
                    errors.append(f"{original_id}: source/processed dimensions differ")
                    continue
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
            errors.append(f"{original_id}: {error}")
            continue
        return processed_row, source_row, source_path, processed_path

    detail = "; ".join(errors[:5])
    raise RuntimeError(f"no loadable saved {defect_name} example found: {detail}")


def recorded_seed(processed_row: pd.Series) -> int | None:
    """Read the original operation seed for an optional save-path replay check."""
    payload = json.loads(str(processed_row["defect_params_json"]))
    steps = payload.get("steps", [])
    if len(steps) != 1 or steps[0].get("random_seed") is None:
        return None
    return int(steps[0]["random_seed"])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    dataset_dir = data_dir / args.dataset
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_dir / "logs" / "defect_application_diagnostics"
    )
    source_metadata_path = dataset_dir / "metadata.csv"
    processed_metadata_path = dataset_dir / "processed_metadata.csv"
    source_metadata = pd.read_csv(source_metadata_path, dtype=str, keep_default_na=False)
    processed_metadata = pd.read_csv(
        processed_metadata_path,
        dtype=str,
        keep_default_na=False,
    )

    source_row = source_row_for_id(source_metadata, args.source_image_id)
    direct_source_path = resolve_metadata_path(str(source_row["path"]), dataset_dir)
    direct_source = load_image(direct_source_path)
    print(
        f"Direct-call source: {args.source_image_id} | {direct_source_path} | "
        f"mode={direct_source.mode} size={direct_source.size}"
    )

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    direct_stats: dict[str, PixelDifference] = {}

    print("\nDirect severity-3 function calls (visualization and CLI bypassed):")
    for defect_name, function in DIRECT_DEFECTS.items():
        degraded, params = function(
            direct_source,
            SEVERITY,
            rng=args.seed,
            return_params=True,
        )
        stats = pixel_difference(direct_source, degraded)
        direct_stats[defect_name] = stats
        comparison_path = (
            output_dir
            / f"direct_{args.source_image_id}_{defect_name}_severity{SEVERITY}.png"
        )
        save_side_by_side(direct_source, degraded, comparison_path)
        print(
            f"  {defect_name:18s} MAD={stats.mean_absolute_difference:9.4f} "
            f"max={stats.max_absolute_difference:6.1f} "
            f"diff_std={stats.difference_standard_deviation:9.4f} "
            f"params={json.dumps(params, sort_keys=True)}"
        )
        results.append(
            {
                "comparison": "direct_function",
                "defect_type": defect_name,
                "severity_level": SEVERITY,
                "original_image_id": args.source_image_id,
                "original_path": str(direct_source_path),
                "degraded_path": None,
                "comparison_png": str(comparison_path),
                "parameters": params,
                **asdict(stats),
            }
        )

    saved_stats: dict[str, PixelDifference] = {}
    saved_replay_stats: dict[str, PixelDifference | None] = {}
    print("\nActual files already saved under processed/:")
    for defect_name in SAVED_DEFECTS:
        processed_row, actual_source_row, source_path, processed_path = first_saved_example(
            processed_metadata,
            source_metadata,
            dataset_dir,
            defect_name,
        )
        original = load_image(source_path)
        # This is the exact one-time uint8 baseline used before CLI defect application.
        original_baseline, _ = prepare_uint8_pil(original)
        saved = load_image(processed_path)
        stats = pixel_difference(original_baseline, saved)
        saved_stats[defect_name] = stats
        comparison_path = (
            output_dir
            / f"saved_{actual_source_row['image_id']}_{defect_name}_severity{SEVERITY}.png"
        )
        save_side_by_side(original_baseline, saved, comparison_path)

        operation_seed = recorded_seed(processed_row)
        replay_stats: PixelDifference | None = None
        if operation_seed is not None:
            replayed, _ = DIRECT_DEFECTS[defect_name](
                original_baseline,
                SEVERITY,
                rng=operation_seed,
                return_params=True,
            )
            replay_stats = pixel_difference(replayed, saved)
        saved_replay_stats[defect_name] = replay_stats
        replay_text = (
            f" saved_vs_replay_MAD={replay_stats.mean_absolute_difference:.4f}"
            if replay_stats is not None
            else " saved_vs_replay_MAD=N/A"
        )
        print(
            f"  {defect_name:18s} id={actual_source_row['image_id']} "
            f"MAD={stats.mean_absolute_difference:9.4f} "
            f"max={stats.max_absolute_difference:6.1f} "
            f"diff_std={stats.difference_standard_deviation:9.4f}"
            f"{replay_text}"
        )
        results.append(
            {
                "comparison": "actual_saved_file",
                "defect_type": defect_name,
                "severity_level": SEVERITY,
                "original_image_id": str(actual_source_row["image_id"]),
                "original_path": str(source_path),
                "degraded_path": str(processed_path),
                "comparison_png": str(comparison_path),
                "parameters": json.loads(str(processed_row["defect_params_json"])),
                **asdict(stats),
                "saved_vs_replayed_direct_mean_absolute_difference": (
                    replay_stats.mean_absolute_difference
                    if replay_stats is not None
                    else None
                ),
            }
        )

    for defect_name in DIRECT_DEFECTS:
        saved_result = saved_stats.get(defect_name)
        replay_result = saved_replay_stats.get(defect_name)
        summary_rows.append(
            {
                "defect_type": defect_name,
                "direct_function_MAD": direct_stats[
                    defect_name
                ].mean_absolute_difference,
                "actual_saved_file_MAD": (
                    saved_result.mean_absolute_difference
                    if saved_result is not None
                    else np.nan
                ),
                "saved_vs_replayed_direct_MAD": (
                    replay_result.mean_absolute_difference
                    if replay_result is not None
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    details_path = output_dir / "details.json"
    summary.to_csv(summary_path, index=False)
    details_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\nSummary (full-resolution mean absolute pixel difference):")
    print(
        summary.to_string(
            index=False,
            na_rep="N/A",
            formatters={
                "direct_function_MAD": "{:.4f}".format,
                "actual_saved_file_MAD": "{:.4f}".format,
                "saved_vs_replayed_direct_MAD": "{:.4f}".format,
            },
        )
    )
    print(f"\nDiagnostic artifacts: {output_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"Detailed JSON: {details_path}")


if __name__ == "__main__":
    main()
