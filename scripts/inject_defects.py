#!/usr/bin/env python3
"""Create a ground-truth-labeled dataset with synthetic image defects."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.defects import (  # noqa: E402
    SEVERITY_NAMES,
    apply_defect,
    available_defects,
    prepare_uint8_pil,
    to_uint8_pil,
    set_severity_profile,
)
from scripts.common import configure_logging, relative_metadata_path  # noqa: E402


OUTPUT_COLUMNS = [
    "image_id",
    "original_image_id",
    "path",
    "split",
    "defect_type",
    "severity_level",
    "canonical_size",
    "defect_params_json",
]
MAX_ACCEPTABLE_FAILURE_RATE = 0.05
VISUALIZATION_CROP_SIZE = 400
VISUALIZATION_CONTEXT_DEFECTS = frozenset({"occlusion_patch", "rotation_crop"})
VISUALIZATION_DISPLAY_UPSCALE = 3
# Pillow's own default (see PIL.Image.MAX_IMAGE_PIXELS); used only to decide
# whether to log a note when an unusually large source image is encountered,
# not to gate or restrict loading.
PIL_DEFAULT_DECOMPRESSION_BOMB_LIMIT = 178_956_970


@dataclass(frozen=True)
class Assignment:
    row_index: int
    clean: bool
    defect_names: tuple[str, ...]
    severities: tuple[int, ...]
    seeds: tuple[int, ...]


def parse_defects(value: str) -> tuple[str, ...]:
    supported = available_defects()
    if value.strip().lower() == "all":
        return supported
    requested = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not requested:
        raise argparse.ArgumentTypeError("--defects must not be empty")
    unknown = sorted(set(requested).difference(supported))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown defect(s): {', '.join(unknown)}; choose from {', '.join(supported)}"
        )
    if len(set(requested)) != len(requested):
        raise argparse.ArgumentTypeError("--defects must not contain duplicates")
    return requested


def parse_severities(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(SEVERITY_NAMES)
    name_to_level = {name: level for level, name in SEVERITY_NAMES.items()}
    levels: list[int] = []
    for item in (part.strip().lower() for part in value.split(",") if part.strip()):
        try:
            level = name_to_level[item] if item in name_to_level else int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "--severities accepts 1,2,3 or mild,moderate,severe"
            ) from error
        if level not in SEVERITY_NAMES:
            raise argparse.ArgumentTypeError("severity levels must be 1, 2, or 3")
        levels.append(level)
    if not levels:
        raise argparse.ArgumentTypeError("--severities must not be empty")
    if len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("--severities must not contain duplicates")
    return tuple(levels)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory name under --data-dir")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output image directory (default: data/<dataset>/processed)",
    )
    parser.add_argument(
        "--canonical-size",
        type=int,
        default=256,
        metavar="N",
        help=(
            "Center-crop to square and resize every source image to NxN pixels "
            "before any defect is applied, so kernel-size- and pixel-based "
            "severity parameters have a consistent effective meaning regardless "
            "of native source resolution"
        ),
    )
    parser.add_argument(
        "--severity-profile",
        type=Path,
        help=(
            "JSON profile from scripts/calibrate_severity.py that overrides the "
            "default severity ranges so each level hits a target perceptual "
            "degradation on this dataset"
        ),
    )
    parser.add_argument(
        "--defects",
        type=parse_defects,
        default=available_defects(),
        help="Comma-separated defect names, or 'all'",
    )
    parser.add_argument(
        "--severities",
        type=parse_severities,
        default=tuple(SEVERITY_NAMES),
        help="Comma-separated levels/names, or 'all'",
    )
    parser.add_argument(
        "--fraction-clean",
        type=float,
        default=0.3,
        help="Clean proportion assigned independently in each split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master assignment seed")
    parser.add_argument(
        "--allow-combinations",
        action="store_true",
        help="Stack two distinct defects on every degraded image",
    )
    parser.add_argument(
        "--occlusion-fill",
        choices=("random", "black", "noise", "edge"),
        default="random",
        help="Fill used by occlusion_patch",
    )
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        metavar="N",
        help="Save N original/degraded example pairs per defect and severity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the assignment summary without writing",
    )
    return parser.parse_args(argv)


def validate_dataset_name(dataset: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset) or dataset in {".", ".."}:
        raise ValueError("--dataset must be a simple directory name")


def resolve_source_path(path_value: str, dataset_dir: Path, raw_dir: Path) -> Path:
    supplied = Path(path_value).expanduser()
    candidates = (
        [supplied]
        if supplied.is_absolute()
        else [dataset_dir / supplied, raw_dir / supplied]
    )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(raw_dir.resolve()):
                raise ValueError(
                    f"metadata path is outside the dataset raw directory: {path_value}"
                )
            return resolved
    raise FileNotFoundError(f"metadata image path does not exist under raw/: {path_value}")


def load_source_metadata(dataset_dir: Path) -> pd.DataFrame:
    metadata_path = dataset_dir / "metadata.csv"
    raw_dir = dataset_dir / "raw"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"source metadata not found: {metadata_path}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw image directory not found: {raw_dir}")

    metadata = pd.read_csv(metadata_path, dtype={"image_id": str, "path": str, "split": str})
    required = {"image_id", "path", "split"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"source metadata is missing columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError(f"source metadata contains no images: {metadata_path}")
    if metadata["image_id"].isna().any() or metadata["path"].isna().any():
        raise ValueError("source metadata contains blank image_id or path values")
    if metadata["image_id"].duplicated().any():
        duplicate = metadata.loc[metadata["image_id"].duplicated(), "image_id"].iloc[0]
        raise ValueError(f"source metadata contains duplicate image_id: {duplicate}")

    metadata = metadata.copy()
    metadata["split"] = metadata["split"].fillna("unassigned").replace("", "unassigned")
    metadata["_source_path"] = [
        resolve_source_path(value, dataset_dir, raw_dir) for value in metadata["path"]
    ]
    return metadata


def build_assignments(
    metadata: pd.DataFrame,
    defects: tuple[str, ...],
    severities: tuple[int, ...],
    fraction_clean: float,
    seed: int,
    allow_combinations: bool,
) -> list[Assignment]:
    """Assign an exact rounded clean count independently within each split."""
    if not 0.0 <= fraction_clean <= 1.0:
        raise ValueError("--fraction-clean must be between 0 and 1 inclusive")
    if allow_combinations and len(defects) < 2:
        raise ValueError("--allow-combinations requires at least two selected defects")

    generator = np.random.default_rng(seed)
    assignments: dict[int, Assignment] = {}
    for _, group in metadata.groupby("split", sort=True, dropna=False):
        indices = np.asarray(group.index, dtype=int)
        shuffled = generator.permutation(indices)
        clean_count = int(round(len(indices) * fraction_clean))
        clean_indices = set(int(index) for index in shuffled[:clean_count])
        for row_index in indices:
            index = int(row_index)
            if index in clean_indices:
                assignments[index] = Assignment(index, True, (), (), ())
                continue
            step_count = 2 if allow_combinations else 1
            selected = generator.choice(defects, size=step_count, replace=False)
            selected_defects = tuple(str(name) for name in np.atleast_1d(selected))
            selected_severities = tuple(
                int(level)
                for level in np.atleast_1d(
                    generator.choice(severities, size=step_count, replace=True)
                )
            )
            selected_seeds = tuple(
                int(value)
                for value in generator.integers(
                    0,
                    np.iinfo(np.uint32).max,
                    size=step_count,
                    endpoint=True,
                    dtype=np.uint64,
                )
            )
            assignments[index] = Assignment(
                index,
                False,
                selected_defects,
                selected_severities,
                selected_seeds,
            )
    return [assignments[int(index)] for index in metadata.index]


def safe_image_id(image_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id).strip("._") or "image"
    if len(safe) > 120:
        digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:10]
        safe = f"{safe[:109]}_{digest}"
    return safe


def output_filename(original_image_id: str, assignment: Assignment) -> str:
    prefix = safe_image_id(original_image_id)
    if assignment.clean:
        return f"{prefix}_clean.jpg"
    defect_label = "+".join(assignment.defect_names)
    severity_label = "+".join(str(level) for level in assignment.severities)
    return f"{prefix}_{defect_label}_{severity_label}.jpg"


@contextlib.contextmanager
def _trusted_source_image_guard():
    """Temporarily relax Pillow's decompression-bomb guard for trusted sources.

    Pillow enforces this guard independently in more than one place —
    ``Image.open()`` checks the declared header size, and ``Image.crop()``
    separately checks the requested crop box size. Both must be relaxed for
    unusually large but legitimate dataset images (curated academic archives
    resolved through this project's own downloaders, not arbitrary untrusted
    uploads) to load and canonically resize successfully. The guard is
    restored immediately afterward; every image passed through this window is
    canonically resized down before any further processing, so no oversized
    array is retained in memory once the window closes.
    """
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def load_image(path: Path) -> Image.Image:
    """Open a source image, tolerating unusually large declared dimensions."""
    with _trusted_source_image_guard():
        with Image.open(path) as image:
            image.load()
            result = image.copy()
            result.format = image.format
            result.info = image.info.copy()
            return result


def canonical_resize(
    image: Image.Image,
    canonical_size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Center-crop to a square aspect ratio, then resize to a fixed canonical size.

    Applied uniformly to every image, for every defect type, before any defect
    is applied. Source images in datasets like ISIC vary enormously in native
    resolution (roughly 1,872x1,053 up to 6,000x4,000+ pixels observed), which
    made fixed-pixel-kernel defects (gaussian_blur, motion_blur) and fixed-pixel
    context views inconsistent in effective severity across images. Fixing the
    input resolution first makes every configured severity level comparable
    regardless of source resolution, and gives downstream feature-extraction
    stages a single consistent input size as well.
    """
    if canonical_size < 1:
        raise ValueError("canonical_size must be a positive integer")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    # Image.crop() enforces the decompression-bomb guard on its own, separately
    # from Image.open(), so an oversized source still needs the guard relaxed
    # here even though load_image() already relaxed it for the initial open.
    with _trusted_source_image_guard():
        square = image.crop((left, top, left + side, top + side))
        resized = square.resize((canonical_size, canonical_size), Image.Resampling.LANCZOS)
    metadata = {
        "canonical_size": canonical_size,
        "original_width": width,
        "original_height": height,
        "square_crop_box_xyxy": [left, top, left + side, top + side],
        "resize_interpolation": "lanczos",
        "operation": "center_crop_square_then_resize",
    }
    return resized, metadata


def flatten_alpha_channel(
    image: Image.Image,
) -> tuple[Image.Image, dict[str, Any] | None]:
    """Remove an alpha channel before the shared defect pipeline sees the image.

    A small number of ChestX-ray14 PNGs are stored as RGBA even though they are
    radiographs (23 of 5,606 in the official sample release). ``pipeline.defects``
    deliberately refuses to guess how alpha should be handled, so the decision is
    made here, at the CLI boundary, and recorded in ``defect_params_json``.

    Fully-opaque alpha - the usual case for these files - is simply dropped,
    which is lossless. Genuinely transparent pixels are composited onto black,
    which matches the unexposed background of a radiograph rather than inventing
    a bright region that a quality metric would read as signal.
    """
    if image.mode not in {"RGBA", "LA", "PA"}:
        return image, None

    prepared = image.convert("RGBA") if image.mode == "PA" else image
    alpha_minimum, alpha_maximum = prepared.getchannel("A").getextrema()
    fully_opaque = alpha_minimum == alpha_maximum == 255
    target_mode = "L" if prepared.mode == "LA" else "RGB"

    if fully_opaque:
        flattened = prepared.convert(target_mode)
        strategy = "dropped_fully_opaque_alpha"
    else:
        backdrop = Image.new("RGBA", prepared.size, (0, 0, 0, 255))
        flattened = Image.alpha_composite(
            backdrop, prepared.convert("RGBA")
        ).convert(target_mode)
        strategy = "composited_onto_black"

    # A radiograph stored as RGBA carries the same value in R, G and B. Leaving
    # it as 3-channel RGB would make these few images the odd ones out in an
    # otherwise single-channel dataset, which downstream feature extraction
    # would have to special-case. Collapsing identical channels to L is lossless
    # and keeps the processed dataset uniform; genuinely coloured images are
    # left as RGB.
    collapsed_to_grayscale = False
    if flattened.mode == "RGB":
        channels = flattened.split()
        if channels[0] == channels[1] == channels[2]:
            flattened = channels[0]
            target_mode = "L"
            collapsed_to_grayscale = True

    return flattened, {
        "source_mode": image.mode,
        "flattened_to": target_mode,
        "alpha_extrema": [int(alpha_minimum), int(alpha_maximum)],
        "fully_opaque": bool(fully_opaque),
        "collapsed_identical_rgb_to_grayscale": collapsed_to_grayscale,
        "strategy": strategy,
    }


def prepare_cli_source(
    image: Image.Image,
    path: Path,
    canonical_size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Convert once to the shared 8-bit working domain, then to canonical size."""
    source_mode = image.mode
    source_format = image.format
    source_info = image.info
    exif_orientation = image.getexif().get(274)

    flattened, alpha_info = flatten_alpha_channel(image)
    working, conversion = prepare_uint8_pil(flattened)
    conversion.update(
        {
            "source_format": source_format or path.suffix.lstrip(".").upper() or "unknown",
            "source_suffix": path.suffix.lower(),
            "source_pil_mode": source_mode,
            "icc_profile_present": bool(source_info.get("icc_profile")),
            "exif_orientation": int(exif_orientation) if exif_orientation is not None else None,
            "operation": "convert_once_to_uint8_working_domain",
        }
    )
    if alpha_info is not None:
        conversion["alpha_flattening"] = alpha_info
    canonical, resize_info = canonical_resize(working, canonical_size)
    conversion["canonical_resize"] = resize_info
    return canonical, conversion


def apply_assignment(
    image: Image.Image,
    assignment: Assignment,
    occlusion_fill: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    current: Image.Image | np.ndarray = image.copy()
    steps: list[dict[str, Any]] = []
    for defect_name, severity, operation_seed in zip(
        assignment.defect_names,
        assignment.severities,
        assignment.seeds,
    ):
        current, params = apply_defect(
            current,
            defect_name,
            severity,
            rng=operation_seed,
            occlusion_fill=occlusion_fill,
        )
        steps.append(
            {
                "defect_type": defect_name,
                "severity_level": severity,
                "severity_name": SEVERITY_NAMES[severity],
                **params,
            }
        )
    return to_uint8_pil(current), steps


def save_jpeg(image: Image.Image, path: Path) -> dict[str, Any]:
    output = to_uint8_pil(image)
    save_options: dict[str, Any] = {"format": "JPEG", "quality": 100}
    encoding = {"format": "JPEG", "quality": 100, "mode": output.mode}
    if output.mode == "RGB":
        save_options["subsampling"] = 0
        encoding["subsampling"] = 0
    output.save(path, **save_options)
    return encoding


def metadata_output_path(path: Path, dataset_dir: Path) -> str:
    try:
        return relative_metadata_path(path, dataset_dir)
    except ValueError:
        return path.resolve().as_posix()


def visualization_parameter_summary(
    defect_name: str,
    params: dict[str, Any],
) -> str:
    """Format the exact applied parameters as a compact figure annotation."""
    if defect_name == "gaussian_noise":
        return f"σ={float(params['sigma']):.2f}"
    if defect_name == "poisson_noise":
        return f"scale={float(params['scale']):.3f}"
    if defect_name == "gaussian_blur":
        kernel = int(params["kernel_size"])
        return f"kernel={kernel}×{kernel} · σ={float(params['sigma']):.2f}"
    if defect_name == "motion_blur":
        kernel = int(params["kernel_size"])
        return (
            f"kernel={kernel}×{kernel} · "
            f"angle={float(params['angle_degrees']):+.1f}°"
        )
    if defect_name == "brightness_shift":
        return f"offset={int(params['offset']):+d}"
    if defect_name == "contrast_reduction":
        return f"factor={float(params['contrast_factor']):.3f}"
    if defect_name == "jpeg_compression":
        return f"quality={int(params['quality'])}"
    if defect_name == "occlusion_patch":
        return (
            f"area={float(params['actual_area_fraction']):.1%} · "
            f"patch={int(params['width'])}×{int(params['height'])}\n"
            f"fill={params['fill_mode']}"
        )
    if defect_name == "rotation_crop":
        return f"angle={float(params['angle_degrees']):+.1f}°"
    return "parameters recorded in metadata"


def visualization_center_crop_box(
    width: int,
    height: int,
    crop_size: int = VISUALIZATION_CROP_SIZE,
) -> tuple[int, int, int, int]:
    """Return a deterministic native-pixel center crop as x0/y0/x1/y1."""
    if width < 1 or height < 1:
        raise ValueError("visualization source dimensions must be positive")
    if crop_size < 1:
        raise ValueError("visualization crop size must be positive")
    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return x0, y0, x0 + crop_width, y0 + crop_height


def crop_visualization_array(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop an image array without resizing or interpolating its pixels."""
    x0, y0, x1, y1 = box
    if image.ndim not in {2, 3}:
        raise ValueError(f"unsupported visualization array shape: {image.shape}")
    height, width = image.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop box {box} is outside image dimensions {width}x{height}")
    return image[y0:y1, x0:x1]


def upscale_nearest(image: np.ndarray, factor: int) -> np.ndarray:
    """Magnify an array by repeating pixels (no interpolation/averaging).

    Used only for display: now that source images are canonically resized to
    a fixed, comparatively small size before defect injection, pixel-level
    defects (noise, blur kernels) can otherwise render too small to see once
    rasterized into a multi-row comparison figure.
    """
    if factor < 1:
        raise ValueError("upscale factor must be a positive integer")
    if factor == 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def build_visualization_figure(
    metadata: pd.DataFrame,
    defects: tuple[str, ...],
    severities: tuple[int, ...],
    examples_per_cell: int,
    dataset: str,
    seed: int,
    occlusion_fill: str,
    canonical_size: int,
) -> Any:
    """Build a defect-major comparison grid without writing or closing it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metadata.empty:
        raise ValueError("cannot visualize an empty metadata selection")
    if examples_per_cell < 1:
        raise ValueError("visualization requires at least one source image")

    generator = np.random.default_rng(seed ^ 0x5EEDBEEF)
    selected_positions = np.atleast_1d(
        generator.choice(
            len(metadata),
            size=examples_per_cell,
            replace=len(metadata) < examples_per_cell,
        )
    )
    samples: list[
        tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]
    ] = []
    for position in selected_positions:
        source_path = Path(metadata.iloc[int(position)]["_source_path"])
        source, _ = prepare_cli_source(load_image(source_path), source_path, canonical_size)
        original_display = np.asarray(to_uint8_pil(source))
        height, width = original_display.shape[:2]
        samples.append(
            (
                source,
                original_display,
                visualization_center_crop_box(width, height),
            )
        )

    # Reusing a seed for a source/defect pair aligns random quantiles, signs,
    # directions, and fill choices across severity levels. Only the configured
    # severity range changes, making visual progression easier to compare.
    operation_seeds = {
        (defect_name, sample_number): int(
            generator.integers(0, np.iinfo(np.uint32).max, endpoint=True)
        )
        for defect_name in defects
        for sample_number in range(len(samples))
    }

    row_count = len(defects) * len(severities)
    column_count = examples_per_cell * 2
    figure_width = max(10.5, column_count * 2.1 + 3.0)
    figure_height = max(7.0, row_count * 1.85 + 1.2)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    row_number = 0
    for defect_name in defects:
        for severity in severities:
            for example_number, (source, original_display, crop_box) in enumerate(
                samples
            ):
                degraded, params = apply_defect(
                    source,
                    defect_name,
                    severity,
                    rng=operation_seeds[(defect_name, example_number)],
                    occlusion_fill=occlusion_fill,
                )
                degraded_display = np.asarray(to_uint8_pil(degraded))
                if degraded_display.shape[:2] != original_display.shape[:2]:
                    raise ValueError(
                        f"{defect_name} changed visualization dimensions from "
                        f"{original_display.shape[:2]} to {degraded_display.shape[:2]}"
                    )
                if defect_name in VISUALIZATION_CONTEXT_DEFECTS:
                    original_view = original_display
                    degraded_view = degraded_display
                    height, width = original_display.shape[:2]
                    view_label = f"full frame · scaled\n{width}×{height}"
                    interpolation = "antialiased"
                else:
                    original_view = crop_visualization_array(original_display, crop_box)
                    degraded_view = crop_visualization_array(degraded_display, crop_box)
                    view_height, view_width = original_view.shape[:2]
                    full_height, full_width = original_display.shape[:2]
                    is_full_frame = (view_width, view_height) == (full_width, full_height)
                    original_view = upscale_nearest(original_view, VISUALIZATION_DISPLAY_UPSCALE)
                    degraded_view = upscale_nearest(degraded_view, VISUALIZATION_DISPLAY_UPSCALE)
                    region_label = "full frame" if is_full_frame else "center crop"
                    view_label = (
                        f"{view_width}×{view_height} {region_label} · "
                        f"×{VISUALIZATION_DISPLAY_UPSCALE} nearest"
                    )
                    # Nearest-neighbor display/upscale avoids averaging pixel-level
                    # noise and small blur kernels when the grid itself is
                    # rasterized at a fixed figure DPI.
                    interpolation = "nearest"
                original_axis = axes[row_number, example_number * 2]
                degraded_axis = axes[row_number, example_number * 2 + 1]
                display_options: dict[str, Any] = {"interpolation": interpolation}
                if original_view.ndim == 2:
                    display_options.update({"cmap": "gray", "vmin": 0, "vmax": 255})
                original_axis.imshow(original_view, **display_options)
                degraded_options: dict[str, Any] = {"interpolation": interpolation}
                if degraded_view.ndim == 2:
                    degraded_options.update({"cmap": "gray", "vmin": 0, "vmax": 255})
                degraded_axis.imshow(degraded_view, **degraded_options)
                if row_number == 0:
                    original_axis.set_title(
                        f"Source {example_number + 1}\nOriginal",
                        fontsize=10,
                    )
                    degraded_axis.set_title(
                        f"Source {example_number + 1}\nDegraded",
                        fontsize=10,
                    )
                original_axis.axis("off")
                degraded_axis.axis("off")
                original_axis.text(
                    0.5,
                    -0.07,
                    view_label,
                    transform=original_axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    clip_on=False,
                )
                degraded_axis.text(
                    0.5,
                    -0.07,
                    visualization_parameter_summary(defect_name, params),
                    transform=degraded_axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    linespacing=1.15,
                    clip_on=False,
                )
            axes[row_number, 0].text(
                -0.08,
                0.5,
                f"{defect_name} — severity {severity} ({SEVERITY_NAMES[severity]})",
                transform=axes[row_number, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8,
                fontweight="medium",
                clip_on=False,
            )
            row_number += 1

    left_margin = min(0.30, max(0.12, 3.0 / figure_width))
    figure.subplots_adjust(
        left=left_margin,
        right=0.995,
        top=0.965,
        bottom=0.01,
        hspace=0.66,
        wspace=0.05,
    )
    figure.suptitle(
        f"Synthetic defect examples: {dataset}\n"
        f"Canonical {canonical_size}×{canonical_size} input "
        f"(center-crop + resize applied before every defect); "
        f"pixel-level crops shown at ×{VISUALIZATION_DISPLAY_UPSCALE}; "
        "context rows show the full canonical frame",
        fontsize=13,
        y=0.995,
    )
    return figure


def save_visualization(
    metadata: pd.DataFrame,
    defects: tuple[str, ...],
    severities: tuple[int, ...],
    examples_per_cell: int,
    dataset: str,
    data_dir: Path,
    seed: int,
    occlusion_fill: str,
    canonical_size: int,
) -> Path:
    """Build and save the labeled comparison grid."""
    import matplotlib.pyplot as plt

    figure = build_visualization_figure(
        metadata,
        defects,
        severities,
        examples_per_cell,
        dataset,
        seed,
        occlusion_fill,
        canonical_size,
    )
    destination = data_dir / "logs" / f"defect_examples_{dataset}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    return destination


def _run_injection(
    args: argparse.Namespace,
    logger: logging.Logger,
    data_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> None:
    metadata = load_source_metadata(dataset_dir)
    assignments = build_assignments(
        metadata,
        args.defects,
        args.severities,
        args.fraction_clean,
        args.seed,
        args.allow_combinations,
    )

    for split, group in metadata.groupby("split", sort=True):
        group_indices = set(int(index) for index in group.index)
        clean_count = sum(item.clean for item in assignments if item.row_index in group_indices)
        logger.info(
            "Split %s: %d images (%d clean, %d degraded)",
            split,
            len(group),
            clean_count,
            len(group) - clean_count,
        )
    if args.severity_profile is not None:
        if not args.severity_profile.is_file():
            raise FileNotFoundError(
                f"severity profile not found: {args.severity_profile}"
            )
        profile = json.loads(args.severity_profile.read_text())
        set_severity_profile(profile)
        logger.info(
            "Applied calibrated severity profile from %s covering %s",
            args.severity_profile,
            ",".join(sorted(profile)),
        )
    else:
        set_severity_profile(None)
        logger.info(
            "Using default hand-tuned severity ranges (no --severity-profile given)"
        )

    logger.info(
        "Selected defects=%s severities=%s combinations=%s canonical_size=%dx%d output=%s",
        ",".join(args.defects),
        ",".join(str(level) for level in args.severities),
        args.allow_combinations,
        args.canonical_size,
        args.canonical_size,
        output_dir,
    )

    if args.dry_run:
        logger.info(
            "DRY RUN: validated %d source paths; no images, metadata, logs, "
            "or figures were written",
            len(metadata),
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_jpegs = sum(1 for _ in output_dir.glob("*.jpg"))
    if existing_jpegs:
        logger.warning(
            "%d existing JPEG(s) are present in %s; this run overwrites matching names "
            "but does not delete stale files. processed_metadata.csv is authoritative.",
            existing_jpegs,
            output_dir,
        )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    successful_row_indices: list[int] = []
    used_filenames: dict[str, str] = {}
    iterator = zip(metadata.iterrows(), assignments)
    for (row_index, row), assignment in tqdm(
        iterator,
        total=len(metadata),
        desc=f"Injecting defects ({args.dataset})",
        unit="image",
    ):
        if int(row_index) != assignment.row_index:
            raise RuntimeError("internal assignment order mismatch")
        original_image_id = str(row["image_id"])
        filename = output_filename(original_image_id, assignment)
        if filename in used_filenames and used_filenames[filename] != original_image_id:
            digest = hashlib.sha1(original_image_id.encode("utf-8")).hexdigest()[:10]
            filename = f"{Path(filename).stem}_{digest}.jpg"
        used_filenames[filename] = original_image_id
        destination = output_dir / filename
        try:
            source_path = Path(row["_source_path"])
            source, baseline = prepare_cli_source(
                load_image(source_path), source_path, args.canonical_size
            )
            resize_info = baseline["canonical_resize"]
            original_pixels = resize_info["original_width"] * resize_info["original_height"]
            if original_pixels > PIL_DEFAULT_DECOMPRESSION_BOMB_LIMIT:
                logger.info(
                    "%s: source is %dx%d (%d px), above Pillow's default "
                    "decompression-bomb threshold; loaded via trusted-dataset "
                    "override and canonically resized to %dx%d",
                    original_image_id,
                    resize_info["original_width"],
                    resize_info["original_height"],
                    original_pixels,
                    args.canonical_size,
                    args.canonical_size,
                )
            output, steps = apply_assignment(source, assignment, args.occlusion_fill)
            encoding = save_jpeg(output, destination)
        except Image.DecompressionBombError as error:
            failures.append(
                {
                    "image_id": original_image_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            logger.exception("Failed to process %s", original_image_id)
            continue
        except Exception as error:  # Continue indexing other independent images.
            failures.append(
                {
                    "image_id": original_image_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            logger.exception("Failed to process %s", original_image_id)
            continue

        defect_type = "clean" if assignment.clean else "+".join(assignment.defect_names)
        severity_level = "0" if assignment.clean else "+".join(
            str(level) for level in assignment.severities
        )
        parameter_payload = {
            "baseline_preprocessing": baseline,
            "steps": steps,
            "output_encoding": encoding,
            # Which severity definitions produced these parameters. Without this
            # a calibrated run and a default run are indistinguishable after the
            # fact, and the numbers in `steps` cannot be interpreted.
            "severity_profile": (
                str(args.severity_profile)
                if args.severity_profile is not None
                else "default_hand_tuned_ranges"
            ),
        }
        rows.append(
            {
                "image_id": destination.stem,
                "original_image_id": original_image_id,
                "path": metadata_output_path(destination, dataset_dir),
                "split": str(row["split"]),
                "defect_type": defect_type,
                "severity_level": severity_level,
                "canonical_size": args.canonical_size,

                "defect_params_json": json.dumps(
                    parameter_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        successful_row_indices.append(int(row_index))

    processed_metadata = dataset_dir / "processed_metadata.csv"
    temporary_metadata = processed_metadata.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(temporary_metadata, index=False)
    temporary_metadata.replace(processed_metadata)
    logger.info("Wrote %d rows to %s", len(rows), processed_metadata)

    failure_report = (
        data_dir / "logs" / f"{args.dataset}_defect_injection_failures.json"
    )
    failure_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_failure_report = failure_report.with_suffix(".json.tmp")
    temporary_failure_report.write_text(
        json.dumps(failures, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_failure_report.replace(failure_report)
    logger.info("Wrote %d failure record(s) to %s", len(failures), failure_report)

    if failures:
        failure_rate = len(failures) / len(metadata)
        summary = (
            f"{len(failures)}/{len(metadata)} images failed "
            f"({failure_rate:.2%}) — see {failure_report.name}"
        )
        if failure_rate > MAX_ACCEPTABLE_FAILURE_RATE:
            raise RuntimeError(
                f"{summary}; exceeds the acceptable "
                f"{MAX_ACCEPTABLE_FAILURE_RATE:.0%} failure rate"
            )
        logger.warning(summary)

    if args.visualize:
        figure_path = save_visualization(
            metadata.loc[successful_row_indices],
            args.defects,
            args.severities,
            args.visualize,
            args.dataset,
            data_dir,
            args.seed,
            args.occlusion_fill,
            args.canonical_size,
        )
        logger.info("Saved defect visualization to %s", figure_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_dataset_name(args.dataset)
    if args.visualize < 0:
        raise ValueError("--visualize must be zero or a positive integer")
    if args.canonical_size < 1:
        raise ValueError("--canonical-size must be a positive integer")

    data_dir = args.data_dir.resolve()
    dataset_dir = data_dir / args.dataset
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset_dir / "processed"
    )
    raw_dir = dataset_dir / "raw"
    output_is_raw = output_dir.resolve() == raw_dir.resolve()
    output_is_under_raw = output_dir.resolve().is_relative_to(raw_dir.resolve())
    if output_is_raw or output_is_under_raw:
        raise ValueError("--output-dir must not be the source raw directory or one of its children")

    logger = configure_logging(
        data_dir,
        args.dataset,
        args.dry_run,
        log_filename=f"{args.dataset}_defect_injection.log",
    )
    try:
        _run_injection(args, logger, data_dir, dataset_dir, output_dir)
    except Exception:
        logger.exception("Defect injection failed for dataset %s", args.dataset)
        raise


if __name__ == "__main__":
    main()