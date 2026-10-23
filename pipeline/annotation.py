"""Detection of burned-in annotation and masked regions.

Added in response to a defect found during demonstration: radiographs carrying
burned-in acquisition text, laterality markers and masked-out corners were
accepted without comment, because none of the nine injected distortions
resembles them. Those features are structural rather than statistical, so the
existing metrics do not respond to them.

The measurement is the fraction of the frame occupied by saturated connected
regions. Burned-in text is rendered at full white and masking is rendered at
full black, both by the acquisition or archival software rather than by the
imaging physics, so both saturate. Anatomy rarely does: a radiograph's dynamic
range is compressed well inside the extremes by attenuation.

Isolated saturated pixels are excluded, since sensor noise produces them and
they carry no annotation. Only components above a minimum area count, which is
what distinguishes a text glyph or a masking bar from speckle.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# A glyph at 256 by 256 occupies on the order of ten pixels; speckle occupies
# one or two. Twelve separates them without excluding small markers.
MIN_COMPONENT_PIXELS = 12
DARK_LEVEL = 4
BRIGHT_LEVEL = 251


def annotation_fraction(image: np.ndarray) -> float:
    """Fraction of the frame occupied by saturated connected regions.

    Higher means more of the image is annotation or mask rather than anatomy,
    so the metric is oriented such that higher is worse.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.size == 0:
        return 0.0

    total = 0
    for mask in (array <= DARK_LEVEL, array >= BRIGHT_LEVEL):
        if not mask.any():
            continue
        labelled, count = ndimage.label(mask)
        if count == 0:
            continue
        sizes = ndimage.sum(mask, labelled, range(1, count + 1))
        total += int(sizes[sizes >= MIN_COMPONENT_PIXELS].sum())

    return float(total) / float(array.size)


def saturated_border_fraction(image: np.ndarray) -> float:
    """Fraction of the outer frame that is saturated.

    Masking and collimation appear at the edges, whereas anatomy does not
    normally reach saturation there. Reported separately because an edge
    artefact is a different curation decision from text over the lung fields.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        array = array.mean(axis=2)
    h, w = array.shape
    band = max(2, min(h, w) // 16)
    border = np.concatenate([
        array[:band, :].ravel(), array[-band:, :].ravel(),
        array[:, :band].ravel(), array[:, -band:].ravel(),
    ])
    if border.size == 0:
        return 0.0
    saturated = (border <= DARK_LEVEL) | (border >= BRIGHT_LEVEL)
    return float(saturated.mean())