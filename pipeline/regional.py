"""Regional quality metrics.

The nine metrics of Chapter 4 each reduce an image to a single value computed
over the whole frame. That makes them insensitive to degradation confined to
part of the image: a radiograph whose bony structures are sharp and whose
global intensity range is wide scores well even where contrast is lost within
one anatomical region. Diagnostic reading is regional, so a globally acceptable
score does not establish that the region of interest is readable.

These functions compute the same underlying quantities over overlapping tiles
and report the worst tile rather than the frame. Tiles consisting mostly of
background are excluded: a radiograph's corners are unexposed and carry no
contrast by construction, so including them would make every image fail.
"""

from __future__ import annotations

import numpy as np

TILE = 64
STRIDE = 32
# A tile is informative when enough of it lies inside the exposed field. The
# bounds exclude unexposed background at one end and saturated annotation or
# collimation at the other.
TISSUE_LOW = 15
TISSUE_HIGH = 250
MIN_TISSUE_FRACTION = 0.60


def _tiles(array: np.ndarray):
    h, w = array.shape
    for y in range(0, max(1, h - TILE + 1), STRIDE):
        for x in range(0, max(1, w - TILE + 1), STRIDE):
            yield array[y:y + TILE, x:x + TILE]


def _informative(tile: np.ndarray) -> bool:
    inside = (tile >= TISSUE_LOW) & (tile <= TISSUE_HIGH)
    return inside.mean() >= MIN_TISSUE_FRACTION


def _grey(image) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        array = array.mean(axis=2)
    return array


def regional_contrast(image) -> float:
    """Lowest local contrast among tiles lying inside the exposed field.

    Local contrast is the robust intensity range within the tile, taken between
    percentiles rather than extrema so that a single bright pixel does not
    conceal a washed-out region.
    """
    array = _grey(image)
    values = []
    for tile in _tiles(array):
        if not _informative(tile):
            continue
        low, high = np.percentile(tile, [5, 95])
        denominator = high + low
        values.append(0.0 if denominator <= 0 else (high - low) / denominator)
    return float(min(values)) if values else float("nan")


def regional_sharpness(image) -> float:
    """Lowest local sharpness among tiles lying inside the exposed field."""
    array = _grey(image)
    values = []
    for tile in _tiles(array):
        if not _informative(tile):
            continue
        gy, gx = np.gradient(tile)
        values.append(float(np.var(np.hypot(gx, gy))))
    return float(min(values)) if values else float("nan")


def regional_contrast_profile(image) -> dict:
    """The distribution of local contrast, for diagnosis rather than decision."""
    array = _grey(image)
    values = []
    for tile in _tiles(array):
        if _informative(tile):
            low, high = np.percentile(tile, [5, 95])
            denominator = high + low
            values.append(0.0 if denominator <= 0 else (high - low) / denominator)
    if not values:
        return {"n_informative_tiles": 0}
    a = np.asarray(values)
    return {
        "n_informative_tiles": int(a.size),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "fraction_below_0_10": float((a < 0.10).mean()),
    }