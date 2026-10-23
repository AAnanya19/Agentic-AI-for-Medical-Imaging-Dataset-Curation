"""Resolution quality checks, read from recorded metadata rather than pixels.

Canonical preprocessing centre-crops every image to a square and resizes it to
256x256 before any defect is applied, so **every processed image has identical
dimensions and resolution information is erased from the pixels**. A resolution
check therefore cannot operate on the processed image at all; it reads the
source dimensions that ``inject_defects.canonical_resize`` recorded into
``defect_params_json -> baseline_preprocessing -> canonical_resize``.

This is a deliberate architectural split, not an inconsistency: resolution is
assessed *pre-canonicalisation* while every other check runs
*post-canonicalisation*. Stating it explicitly matters because a reader will
otherwise wonder how resolution can vary when all inputs are 256x256.

Three independent failure modes are flagged, because they have different causes
and different remedies:

  * **Below an absolute floor** - the image was never large enough to carry the
    detail the task needs, and upscaling to canonical size fabricated pixels.
  * **Extreme aspect ratio** - the square centre-crop discarded a large
    fraction of the frame, so the processed image may be missing anatomy that
    was present in the source.
  * **A distributional outlier** - unremarkable in absolute terms but far from
    what the rest of *this* dataset looks like, which is what actually predicts
    inconsistent downstream behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Below this, a 256x256 canonical image has been upscaled rather than
# downsampled, so its apparent detail is interpolation rather than measurement.
DEFAULT_MIN_SIDE = 256
# A square centre-crop of a frame this elongated discards over a third of it.
DEFAULT_MAX_ASPECT_RATIO = 1.5
# Percentile below which an image counts as small *for this dataset*.
DEFAULT_OUTLIER_PERCENTILE = 1.0


@dataclass(frozen=True)
class ResolutionThresholds:
    """Configuration for the resolution check, recorded alongside its output."""

    min_side: int = DEFAULT_MIN_SIDE
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO
    outlier_percentile: float = DEFAULT_OUTLIER_PERCENTILE

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_side": self.min_side,
            "max_aspect_ratio": self.max_aspect_ratio,
            "outlier_percentile": self.outlier_percentile,
        }


def extract_source_dimensions(processed: pd.DataFrame) -> pd.DataFrame:
    """Pull original width/height out of each row's recorded defect parameters.

    Returns one row per input row with ``original_width``, ``original_height``,
    ``megapixels``, ``aspect_ratio`` and ``min_side``. Rows whose provenance is
    missing or malformed get NaN rather than being silently dropped - a row that
    cannot be checked is a different thing from a row that passes, and the
    caller needs to be able to tell them apart.
    """
    widths: list[float] = []
    heights: list[float] = []
    for raw in processed["defect_params_json"]:
        width = height = float("nan")
        try:
            payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
            resize_info = payload["baseline_preprocessing"]["canonical_resize"]
            width = float(resize_info["original_width"])
            height = float(resize_info["original_height"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass  # NaN, handled by the caller
        widths.append(width)
        heights.append(height)

    frame = pd.DataFrame({
        "image_id": processed["image_id"].to_numpy(),
        "original_width": widths,
        "original_height": heights,
    })
    frame["megapixels"] = frame["original_width"] * frame["original_height"] / 1e6
    # Aspect ratio is expressed as long-side / short-side so it is always >= 1
    # and orientation does not matter - a portrait and a landscape frame of the
    # same elongation lose the same amount to a square crop.
    long_side = frame[["original_width", "original_height"]].max(axis=1)
    short_side = frame[["original_width", "original_height"]].min(axis=1)
    frame["aspect_ratio"] = long_side / short_side
    frame["min_side"] = short_side
    return frame


def check_resolution(
    processed: pd.DataFrame,
    thresholds: ResolutionThresholds | None = None,
) -> pd.DataFrame:
    """Flag each row for undersize, extreme aspect ratio, and small-for-dataset.

    The distributional threshold is computed from this dataset's own sources, so
    the check adapts to a corpus of uniformly large microscopy mosaics as
    readily as to one of small thumbnails.
    """
    settings = thresholds or ResolutionThresholds()
    frame = extract_source_dimensions(processed)

    measurable = frame["min_side"].notna()
    if measurable.any():
        sides = frame.loc[measurable, "min_side"]
        percentile_floor = float(np.percentile(sides, settings.outlier_percentile))
        # A dataset of uniform resolution has no outliers to find. ChestX-ray14
        # is entirely 1024x1024, so its 1st percentile *is* 1024 and a
        # "<= percentile" test would flag all 5,606 rows as unusually small.
        # The check is only meaningful where the distribution has spread below
        # the percentile, so it is skipped outright when it does not.
        distribution_is_degenerate = bool(sides.min() >= percentile_floor)
    else:
        percentile_floor = float("nan")
        distribution_is_degenerate = True

    frame["below_min_side"] = frame["min_side"] < settings.min_side
    frame["extreme_aspect_ratio"] = frame["aspect_ratio"] > settings.max_aspect_ratio
    # Strict less-than, so an image exactly at the percentile boundary is not
    # called an outlier against itself.
    frame["small_for_dataset"] = (
        False if distribution_is_degenerate else frame["min_side"] < percentile_floor
    )
    frame["provenance_missing"] = ~measurable

    # A row with no recorded provenance is "not checkable", which must not be
    # reported as "passed": every boolean above is False for such a row, so the
    # flag is driven off provenance_missing instead.
    frame["flagged"] = (
        frame["below_min_side"]
        | frame["extreme_aspect_ratio"]
        | frame["small_for_dataset"]
    ) & measurable

    frame.attrs["thresholds"] = settings.to_dict()
    frame.attrs["percentile_floor_min_side"] = percentile_floor
    frame.attrs["distribution_is_degenerate"] = distribution_is_degenerate
    return frame


def summarise_resolution(flagged: pd.DataFrame) -> dict[str, Any]:
    """Counts and distribution summary, for the Dataset Summary Agent."""
    measurable = flagged["min_side"].notna()
    return {
        "n_rows": int(len(flagged)),
        "n_measurable": int(measurable.sum()),
        "n_provenance_missing": int((~measurable).sum()),
        "n_flagged": int(flagged["flagged"].sum()),
        "n_below_min_side": int(flagged["below_min_side"].sum()),
        "n_extreme_aspect_ratio": int(flagged["extreme_aspect_ratio"].sum()),
        "n_small_for_dataset": int(flagged["small_for_dataset"].sum()),
        "thresholds": flagged.attrs.get("thresholds", {}),
        "percentile_floor_min_side": flagged.attrs.get("percentile_floor_min_side"),
        "outlier_check_skipped_uniform_resolution": flagged.attrs.get(
            "distribution_is_degenerate", False
        ),
        "megapixels": {
            "min": float(flagged.loc[measurable, "megapixels"].min())
            if measurable.any() else None,
            "median": float(flagged.loc[measurable, "megapixels"].median())
            if measurable.any() else None,
            "max": float(flagged.loc[measurable, "megapixels"].max())
            if measurable.any() else None,
        },
        "aspect_ratio": {
            "min": float(flagged.loc[measurable, "aspect_ratio"].min())
            if measurable.any() else None,
            "median": float(flagged.loc[measurable, "aspect_ratio"].median())
            if measurable.any() else None,
            "max": float(flagged.loc[measurable, "aspect_ratio"].max())
            if measurable.any() else None,
        },
    }