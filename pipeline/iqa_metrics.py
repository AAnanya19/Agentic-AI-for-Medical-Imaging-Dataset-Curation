"""No-reference image quality metrics: classical statistics plus learned models.

Every metric takes a uint8 numpy array (HxW grayscale or HxWx3 RGB, matching
whatever mode the source image was saved in - inject_defects.py preserves mode,
so ChestX-ray14 arrives as HxW and ISIC as HxWx3) and returns a single float.

Each metric is registered with an explicit ``higher_is_better`` polarity so
downstream evaluation code never has to guess which direction is "worse" -
getting this backwards silently inverts every correlation and AUROC computed
from it. Metrics whose direction is not a simple quality statement (entropy can
rise or fall under either better or worse conditions) are registered with
``higher_is_better=None`` and excluded from directional evaluation, though still
reported descriptively.

``brightness_shift`` in pipeline.defects applies a *randomly signed* offset, so
raw mean intensity does not correlate monotonically with severity - a +40
offset and a -40 offset are both "severity 3" but move the mean in opposite
directions. brightness_deviation therefore needs a reference "typical
brightness" fitted from clean images, in the same spirit as Stage 3's anomaly
detectors being fit on clean embeddings only. See BrightnessReference below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover - environment guard
    raise ImportError(
        "opencv-python (cv2) is required for pipeline.iqa_metrics"
    ) from error

try:
    import torch
    import piq
except ImportError:  # pragma: no cover - degrade gracefully without piq
    torch = None
    piq = None


# ---------------------------------------------------------------------------
# Shared conversion helpers
# ---------------------------------------------------------------------------

def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Return a float64 HxW array in [0, 255], regardless of source mode."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        # ITU-R BT.601 luma weights, matching how JPEG/most tools derive
        # grayscale from RGB - consistent with what a human would perceive as
        # brightness, rather than a naive channel average.
        rgb = array[:, :, :3].astype(np.float64)
        return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    raise ValueError(f"unsupported image shape for grayscale conversion: {array.shape}")


def to_torch_tensor(image: np.ndarray, replicate_to_rgb: bool = False) -> "torch.Tensor":
    """Convert a uint8 array to an NCHW float tensor in [0, 1] for piq."""
    array = np.asarray(image)
    if array.ndim == 2:
        tensor = torch.from_numpy(array.astype(np.float32) / 255.0)[None, None]
    elif array.ndim == 3:
        tensor = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    else:
        raise ValueError(f"unsupported image shape for tensor conversion: {array.shape}")
    if replicate_to_rgb and tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor


# ---------------------------------------------------------------------------
# Classical metrics
# ---------------------------------------------------------------------------

def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian: a standard, fast blur proxy.

    A sharp image has strong, spatially varied second derivatives at edges; a
    blurred image's edges are smoothed out, so the variance of the Laplacian
    response drops. Higher = sharper = better.
    """
    gray = to_grayscale_float(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(image: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude: a second, independent blur proxy.

    Included alongside Laplacian variance because they respond somewhat
    differently to noise (Tenengrad is less sensitive to isolated pixel-level
    noise than the Laplacian, which is a second derivative). Higher = sharper.
    """
    gray = to_grayscale_float(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(grad_x**2 + grad_y**2))


def noise_estimate(image: np.ndarray) -> float:
    """Fast noise-level estimate (Immerkaer 1996).

    Convolves with a Laplacian-of-Gaussian-like kernel designed so that its
    response to smooth image content is near zero, leaving mostly the noise
    contribution. This is a real estimator (not just "apply a blur filter and
    call the difference noise"), commonly used as a cheap sigma estimate.
    Higher = noisier = worse.
    """
    gray = to_grayscale_float(image)
    height, width = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    convolved = cv2.filter2D(gray, -1, kernel)
    sigma = np.sum(np.abs(convolved))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * max(width - 2, 1) * max(height - 2, 1))
    return float(sigma)


def rms_contrast(image: np.ndarray) -> float:
    """Standard deviation of pixel intensity. Higher = more contrast = better."""
    gray = to_grayscale_float(image)
    return float(np.std(gray))


def michelson_contrast(image: np.ndarray) -> float:
    """(max-min)/(max+min) using robust 1st/99th percentiles.

    Percentiles rather than true min/max avoid a single hot or dead pixel
    dominating the score. An epsilon guards the degenerate all-uniform case
    (max+min == 0), which would otherwise divide by zero. Higher = better.
    """
    gray = to_grayscale_float(image)
    low = np.percentile(gray, 1)
    high = np.percentile(gray, 99)
    return float((high - low) / (high + low + 1e-8))


def shannon_entropy(image: np.ndarray) -> float:
    """Shannon entropy of the intensity histogram, in bits.

    Reported descriptively rather than as a directional quality signal: both a
    genuinely detailed clean image and a noisy image can have high entropy,
    while both a flat clean region and a heavily contrast-reduced image can
    have low entropy. Registered with higher_is_better=None.
    """
    gray = to_grayscale_float(image)
    histogram, _ = np.histogram(gray, bins=256, range=(0, 255))
    probabilities = histogram / max(histogram.sum(), 1)
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


# ---------------------------------------------------------------------------
# Brightness deviation - the one metric that needs a fitted reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrightnessReference:
    """A dataset's typical mean intensity, fitted from clean images only.

    Mirrors Stage 3's principle of fitting anomaly detectors on clean data:
    "normal" is defined by the training distribution, not assumed as a fixed
    constant (e.g. 127.5), which would not hold for a chest X-ray corpus that
    is mostly dark background versus a dermoscopy corpus that is mostly
    mid-tone skin.
    """

    reference_mean: float
    n_images: int

    def to_dict(self) -> dict[str, Any]:
        return {"reference_mean": self.reference_mean, "n_images": self.n_images}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BrightnessReference":
        return cls(
            reference_mean=float(payload["reference_mean"]),
            n_images=int(payload["n_images"]),
        )

    @classmethod
    def fit(cls, clean_images: list[np.ndarray]) -> "BrightnessReference":
        if not clean_images:
            raise ValueError("cannot fit a brightness reference from zero images")
        means = [float(np.mean(to_grayscale_float(image))) for image in clean_images]
        return cls(reference_mean=float(np.median(means)), n_images=len(clean_images))


def brightness_deviation(image: np.ndarray, reference: BrightnessReference) -> float:
    """Absolute deviation from the dataset's typical clean-image brightness.

    Necessary rather than plain mean intensity because brightness_shift's
    offset is randomly signed (see module docstring) - only the *distance* from
    typical brightness is monotonic with severity. Higher = worse.
    """
    gray = to_grayscale_float(image)
    return float(abs(np.mean(gray) - reference.reference_mean))


# ---------------------------------------------------------------------------
# Learned models (piq)
# ---------------------------------------------------------------------------

def brisque(image: np.ndarray) -> float:
    """BRISQUE natural-scene-statistics score. Higher = worse quality.

    Calibrated on the LIVE database of natural photographs; its assumptions are
    only partially transferable to radiographs and dermoscopy (Gupta et al.,
    J. Testing & Eval. 2018, found NSS models partially but not fully transfer
    to X-ray). Treat as a relative ranking signal on medical images, not an
    absolute quality score - see the evaluation report for this dataset's
    empirical behaviour.
    """
    if piq is None:
        raise ImportError("piq is not installed; brisque() is unavailable")
    tensor = to_torch_tensor(image, replicate_to_rgb=False)
    with torch.no_grad():
        score = piq.brisque(tensor, data_range=1.0)
    return float(score.item())


def clip_iqa(image: np.ndarray) -> float:
    """CLIP-IQA perceptual quality score. Higher = better quality.

    Validated by its authors (Wang, Chan & Loy, AAAI 2023) on natural-image
    NR-IQA datasets only (SPAQ, KonIQ); not validated on medical images.
    Always replicated to 3 channels, matching CLIP's RGB training domain,
    rather than relying on undocumented single-channel behaviour.
    """
    if piq is None:
        raise ImportError("piq is not installed; clip_iqa() is unavailable")
    tensor = to_torch_tensor(image, replicate_to_rgb=True)
    model = _get_clip_iqa_model()
    with torch.no_grad():
        score = model(tensor)
    return float(score.item())


_CLIP_IQA_MODEL = None


def _get_clip_iqa_model():
    """Load CLIP-IQA once and reuse it; construction downloads/loads weights."""
    global _CLIP_IQA_MODEL
    if _CLIP_IQA_MODEL is None:
        _CLIP_IQA_MODEL = piq.CLIPIQA()
    return _CLIP_IQA_MODEL


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """One registered metric: how to compute it and how to interpret it."""

    function: Callable[[np.ndarray], float]
    higher_is_better: bool | None
    family: str
    requires_torch: bool = False


METRICS: dict[str, MetricSpec] = {
    "laplacian_variance": MetricSpec(laplacian_variance, True, "blur"),
    "tenengrad": MetricSpec(tenengrad, True, "blur"),
    "noise_estimate": MetricSpec(noise_estimate, False, "noise"),
    "rms_contrast": MetricSpec(rms_contrast, True, "contrast"),
    "michelson_contrast": MetricSpec(michelson_contrast, True, "contrast"),
    "shannon_entropy": MetricSpec(shannon_entropy, None, "general"),
    "brisque": MetricSpec(brisque, False, "general", requires_torch=True),
    "clip_iqa": MetricSpec(clip_iqa, True, "general", requires_torch=True),
}
# brightness_deviation is deliberately not in METRICS: it takes an extra
# BrightnessReference argument and is computed separately in compute_iqa.py
# once the reference has been fitted on that dataset's clean training images.


def available_metrics(include_torch: bool = True) -> tuple[str, ...]:
    """Names of registered metrics, optionally excluding piq-based ones."""
    return tuple(
        name
        for name, spec in METRICS.items()
        if include_torch or not spec.requires_torch
    )


def compute_all_metrics(
    image: np.ndarray,
    metric_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Compute every requested metric on one image."""
    names = metric_names if metric_names is not None else available_metrics()
    return {name: METRICS[name].function(image) for name in names}