"""Deterministic synthetic image-quality defects for medical IQA datasets.

All user-tunable severity ranges live in :data:`DEFECT_CONFIG`. Public defect
functions accept either a PIL image or NumPy array and return the same container
type. Pass ``return_params=True`` to also receive the exact sampled parameters.
For exact stochastic replay, pass an integer ``rng`` seed; it is included in the
returned parameter dictionary.

Albumentations operates on uint8/float32 data. Non-uint8 inputs are mapped to an
8-bit working range and mapped back to their original dtype afterward. The CLI
saves JPEG outputs (therefore 8-bit), but reusable functions preserve NumPy dtype
and PIL grayscale/RGB mode wherever the format permits it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

# Avoid import-time update checks in offline/reproducible pipeline runs.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from PIL import Image


SEVERITY_NAMES = {1: "mild", 2: "moderate", 3: "severe"}

# Ranges are deliberately non-overlapping where practical so severity has a
# clear ordinal interpretation. Intensity-valued settings assume an 8-bit
# working range; higher-bit-depth inputs are mapped back after degradation.
DEFECT_CONFIG: dict[str, dict[int, dict[str, Any]]] = {
    "gaussian_noise": {
        1: {"sigma_range": (4.0, 8.0)},
        2: {"sigma_range": (14.0, 22.0)},
        3: {"sigma_range": (32.0, 45.0)},
    },
    "poisson_noise": {
        1: {"scale_range": (0.05, 0.12)},
        2: {"scale_range": (0.35, 0.70)},
        3: {"scale_range": (1.25, 2.50)},
    },
    "gaussian_blur": {
        1: {"kernel_sizes": (3, 5), "sigma_range": (0.6, 1.0)},
        2: {"kernel_sizes": (7, 9), "sigma_range": (1.8, 2.8)},
        3: {"kernel_sizes": (13, 17), "sigma_range": (4.0, 6.0)},
    },
    "motion_blur": {
        1: {"kernel_sizes": (5, 7), "angle_range": (0.0, 180.0)},
        2: {"kernel_sizes": (11, 15), "angle_range": (0.0, 180.0)},
        3: {"kernel_sizes": (21, 31), "angle_range": (0.0, 180.0)},
    },
    "brightness_shift": {
        1: {"offset_range": (8, 18)},
        2: {"offset_range": (25, 45)},
        3: {"offset_range": (55, 85)},
    },
    "contrast_reduction": {
        1: {"factor_range": (0.80, 0.90), "mid_gray": 127.5},
        2: {"factor_range": (0.55, 0.70), "mid_gray": 127.5},
        3: {"factor_range": (0.25, 0.40), "mid_gray": 127.5},
    },
    "jpeg_compression": {
        1: {"quality": 50},
        2: {"quality": 25},
        3: {"quality": 10},
    },
    "occlusion_patch": {
        1: {
            "area_fraction_range": (0.02, 0.04),
            "aspect_ratio_range": (0.5, 2.0),
            "fill_modes": ("black", "noise", "edge"),
            "noise_std_floor": 8.0,
        },
        2: {
            "area_fraction_range": (0.08, 0.12),
            "aspect_ratio_range": (0.5, 2.0),
            "fill_modes": ("black", "noise", "edge"),
            "noise_std_floor": 8.0,
        },
        3: {
            "area_fraction_range": (0.20, 0.30),
            "aspect_ratio_range": (0.5, 2.0),
            "fill_modes": ("black", "noise", "edge"),
            "noise_std_floor": 8.0,
        },
    },
    "rotation_crop": {
        1: {"absolute_angle_range": (2.0, 5.0)},
        2: {"absolute_angle_range": (6.0, 12.0)},
        3: {"absolute_angle_range": (14.0, 25.0)},
    },
}

ImageInput = Image.Image | np.ndarray
RngInput = np.random.Generator | int | None
DefectOutput = ImageInput | tuple[ImageInput, dict[str, Any]]


@dataclass(frozen=True)
class CalibrationSpec:
    """How to search one defect's intensity parameter toward a perceptual target.

    ``DEFECT_CONFIG`` above holds hand-chosen ranges. Those are a reasonable
    default, but the same absolute parameter produces very different perceptual
    degradation on different modalities - Poisson noise that merely darkens a
    dermoscopy image can obliterate a chest radiograph, because shot noise is
    signal-dependent and radiographs have large near-black regions. Calibrating
    each severity to a target SSIM instead makes "severity 2" mean the same
    thing across datasets and across defect types.

    ``increasing`` records whether a larger parameter means more degradation;
    it is False for contrast factor and JPEG quality, where smaller is worse.
    """

    config_key: str
    bounds: tuple[float, float]
    increasing: bool
    is_range: bool = True
    integer: bool = False
    odd_only: bool = False
    couples_kernel_to_sigma: bool = False


# Spatial defects are deliberately excluded. `occlusion_patch` is parameterized
# by fraction of image area and `rotation_crop` by degrees; both are already
# modality-independent and semantically meaningful, and SSIM is a poor target
# for them - a few degrees of rotation collapses SSIM without corresponding to
# a comparable loss of diagnostic quality, so calibrating against it would
# drive the angles to implausibly small values.
# A blur kernel spanning a large fraction of the frame stops modelling
# acquisition blur and starts erasing the subject. Calibrating ISIC against SSIM
# alone drove Gaussian blur to a 109x109 kernel on a 256px image - 43% of the
# frame - which hit the SSIM target while removing the lesion completely. ISIC
# images are downsampled ~20x to canonical size, so they arrive band-limited:
# blur removes little of the *local* structure SSIM measures while perceptually
# destroying the content. Capping the kernel keeps the defect physically
# meaningful; when a severity target then cannot be reached, the calibrator
# reports it honestly rather than producing an unusable image.
MAX_BLUR_KERNEL_FRACTION = 0.20
REFERENCE_CANONICAL_SIZE = 256
# 20% of a 256px frame, rounded down to an odd kernel.
_MAX_BLUR_KERNEL = 51
# Inverse of kernel_size_for_sigma: the sigma whose kernel hits that cap.
_MAX_BLUR_SIGMA = 16.5

# Other bounds were widened after a first calibration run clamped against them:
# Poisson could not be made mild enough (SSIM 0.78-0.82 even at its minimum).
# Bounds should constrain only what is physically sensible.
CALIBRATION_SPECS: dict[str, CalibrationSpec] = {
    "gaussian_noise": CalibrationSpec("sigma_range", (0.1, 160.0), True),
    "poisson_noise": CalibrationSpec("scale_range", (0.00002, 12.0), True),
    "gaussian_blur": CalibrationSpec(
        "sigma_range", (0.2, _MAX_BLUR_SIGMA), True, couples_kernel_to_sigma=True
    ),
    "motion_blur": CalibrationSpec(
        "kernel_sizes", (3, _MAX_BLUR_KERNEL), True, integer=True, odd_only=True
    ),
    "brightness_shift": CalibrationSpec("offset_range", (1, 200), True, integer=True),
    "contrast_reduction": CalibrationSpec("factor_range", (0.02, 0.995), False),
    "jpeg_compression": CalibrationSpec(
        "quality", (1, 95), False, is_range=False, integer=True
    ),
}
UNCALIBRATED_DEFECTS = ("occlusion_patch", "rotation_crop")

# Defects whose severity levels are not reliably ordinal under a pixel-alignment
# metric such as SSIM, and which therefore must be excluded from any
# severity-correlation analysis.
#
# Rotation displaces every pixel, so SSIM collapses on the first few degrees and
# then saturates: it is measuring misalignment, not loss of image quality. Once
# saturated, the ordering between severity levels is decided by the crop-and-
# rescale interaction and by sampling noise rather than by angle. Full-dataset
# validation confirmed this - on ISIC, rotation severity 2 scored 0.637 and
# severity 3 scored 0.641, a tie within noise and a formal monotonicity
# violation.
#
# The defect itself is realistic and worth detecting, so it stays in the corpus
# and remains valid for *binary* clean-vs-degraded detection. Its severity
# labels are ordinal by construction in degrees (roughly 4 / 10 / 21), so any
# severity analysis for rotation should use the recorded angle, not SSIM.
SSIM_NON_ORDINAL_DEFECTS = ("rotation_crop",)


def kernel_size_for_sigma(sigma: float) -> int:
    """Pick an odd Gaussian kernel that tracks sigma the way the defaults do.

    The hand-chosen defaults pair sigma 0.8/2.3/5.0 with kernels 3/7/17, which
    this reproduces, so calibrated blur keeps the same kernel-to-sigma
    relationship rather than inventing a new one.
    """
    kernel = 2 * int(round(1.5 * float(sigma))) + 1
    return max(3, kernel)


# An override profile, normally produced by scripts/calibrate_severity.py and
# scoped with `severity_profile(...)`. None means "use the defaults above".
_ACTIVE_PROFILE: dict[str, dict[int, dict[str, Any]]] | None = None


def _normalize_profile(
    profile: dict[str, Any] | None,
) -> dict[str, dict[int, dict[str, Any]]] | None:
    """Accept JSON-shaped profiles whose severity keys are strings."""
    if profile is None:
        return None
    normalized: dict[str, dict[int, dict[str, Any]]] = {}
    for defect_name, severities in profile.items():
        if defect_name not in DEFECT_CONFIG:
            raise ValueError(f"profile names unknown defect {defect_name!r}")
        normalized[defect_name] = {}
        for severity, settings in severities.items():
            level = int(severity)
            if level not in SEVERITY_NAMES:
                raise ValueError(f"profile has invalid severity {severity!r}")
            merged = dict(DEFECT_CONFIG[defect_name][level])
            for key, value in settings.items():
                merged[key] = tuple(value) if isinstance(value, list) else value
            normalized[defect_name][level] = merged
    return normalized


def set_severity_profile(profile: dict[str, Any] | None) -> None:
    """Install (or clear, with None) a calibrated severity profile globally."""
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = _normalize_profile(profile)


def get_severity_profile() -> dict[str, dict[int, dict[str, Any]]] | None:
    """Return the active override profile, or None when defaults are in use."""
    return _ACTIVE_PROFILE


@contextmanager
def severity_profile(profile: dict[str, Any] | None) -> Iterator[None]:
    """Temporarily apply a severity profile, restoring the previous one after."""
    global _ACTIVE_PROFILE
    previous = _ACTIVE_PROFILE
    _ACTIVE_PROFILE = _normalize_profile(profile)
    try:
        yield
    finally:
        _ACTIVE_PROFILE = previous


@dataclass
class _ImageState:
    was_pil: bool
    original_mode: str | None
    original_dtype: np.dtype[Any]
    value_min: float
    value_max: float
    observed_min: float
    observed_max: float
    singleton_channel: bool


def _validate_severity(defect_name: str, severity: int) -> dict[str, Any]:
    if severity not in SEVERITY_NAMES:
        raise ValueError(f"severity must be one of {sorted(SEVERITY_NAMES)}, got {severity}")
    # A calibrated profile, when active, overrides the hand-chosen defaults.
    # This is the single point where severity settings are resolved, so every
    # defect function picks the override up without further plumbing.
    if _ACTIVE_PROFILE is not None:
        override = _ACTIVE_PROFILE.get(defect_name, {}).get(severity)
        if override is not None:
            return override
    return DEFECT_CONFIG[defect_name][severity]


def _coerce_rng(rng: RngInput) -> tuple[np.random.Generator, int | None]:
    if isinstance(rng, np.random.Generator):
        return rng, None
    if rng is None:
        return np.random.default_rng(), None
    if isinstance(rng, (int, np.integer)):
        seed = int(rng)
        return np.random.default_rng(seed), seed
    raise TypeError("rng must be an integer seed, numpy.random.Generator, or None")


def _prepare_image(image: ImageInput) -> tuple[np.ndarray, _ImageState]:
    if isinstance(image, Image.Image):
        was_pil = True
        original_mode = image.mode
        if image.mode == "RGBA":
            raise ValueError("RGBA images are not supported; convert explicitly to RGB first")
        if image.mode == "1":
            array = np.asarray(image.convert("L"))
            original_mode = "L"
        elif image.mode == "P" or image.mode not in {
            "L",
            "RGB",
            "I",
            "I;16",
            "F",
        }:
            array = np.asarray(image.convert("RGB"))
            original_mode = "RGB"
        else:
            array = np.asarray(image)
    elif isinstance(image, np.ndarray):
        was_pil = False
        original_mode = None
        array = np.asarray(image)
    else:
        raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")

    if array.ndim not in {2, 3}:
        raise ValueError(f"expected a 2-D or 3-D image array, got shape {array.shape}")
    if array.ndim == 3 and array.shape[2] not in {1, 3}:
        raise ValueError(f"expected 1 or 3 channels, got shape {array.shape}")

    original_dtype = array.dtype
    singleton_channel = array.ndim == 3 and array.shape[2] == 1
    if singleton_channel:
        array = array[..., 0]

    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"unsupported image dtype: {array.dtype}")

    finite = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(finite)):
        raise ValueError("image contains NaN or infinite values")
    observed_min = float(finite.min())
    observed_max = float(finite.max())

    if array.dtype == np.uint8:
        value_min, value_max = 0.0, 255.0
        working = np.array(array, dtype=np.uint8, copy=True)
    elif array.dtype == np.bool_:
        value_min, value_max = 0.0, 1.0
        working = array.astype(np.uint8) * 255
    else:
        if np.issubdtype(array.dtype, np.floating) and observed_min >= 0.0 and observed_max <= 1.0:
            value_min, value_max = 0.0, 1.0
        elif observed_max > observed_min:
            value_min, value_max = observed_min, observed_max
        elif np.issubdtype(array.dtype, np.integer):
            limits = np.iinfo(array.dtype)
            value_min, value_max = float(limits.min), float(limits.max)
        else:
            value_min, value_max = 0.0, 1.0
        scaled = (finite - value_min) / max(value_max - value_min, np.finfo(float).eps)
        working = np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)

    state = _ImageState(
        was_pil=was_pil,
        original_mode=original_mode,
        original_dtype=np.dtype(original_dtype),
        value_min=value_min,
        value_max=value_max,
        observed_min=observed_min,
        observed_max=observed_max,
        singleton_channel=singleton_channel,
    )
    return working, state


def _restore_image(working: np.ndarray, state: _ImageState) -> ImageInput:
    working = np.asarray(working)
    if state.original_dtype == np.dtype(np.uint8):
        restored = np.clip(working, 0, 255).astype(np.uint8)
    elif state.original_dtype == np.dtype(np.bool_):
        restored = np.asarray(working) >= 128
    else:
        scaled = np.asarray(working, dtype=np.float64) / 255.0
        restored_float = scaled * (state.value_max - state.value_min) + state.value_min
        if np.issubdtype(state.original_dtype, np.integer):
            limits = np.iinfo(state.original_dtype)
            restored_float = np.rint(np.clip(restored_float, limits.min, limits.max))
        restored = restored_float.astype(state.original_dtype)

    if state.singleton_channel:
        restored = restored[..., np.newaxis]

    if not state.was_pil:
        return restored

    if state.singleton_channel:
        restored = restored[..., 0]
    result = Image.fromarray(restored)
    if state.original_mode in {"L", "RGB", "RGBA"} and result.mode != state.original_mode:
        result = result.convert(state.original_mode)
    return result


def to_uint8_pil(image: ImageInput) -> Image.Image:
    """Convert a supported input to JPEG-compatible 8-bit PIL without RGB coercion."""
    working, _ = _prepare_image(image)
    if working.ndim == 2:
        return Image.fromarray(working, mode="L")
    return Image.fromarray(working, mode="RGB")


def prepare_uint8_pil(image: ImageInput) -> tuple[Image.Image, dict[str, Any]]:
    """Create one fixed CLI working image and describe the intensity mapping."""
    working, state = _prepare_image(image)
    output = Image.fromarray(working, mode="L" if working.ndim == 2 else "RGB")
    identity = state.original_dtype == np.dtype(np.uint8) and state.original_mode in {
        "L",
        "RGB",
        None,
    }
    metadata = {
        "original_mode": state.original_mode or "numpy",
        "original_dtype": state.original_dtype.name,
        "original_observed_min": state.observed_min,
        "original_observed_max": state.observed_max,
        "mapping_min": state.value_min,
        "mapping_max": state.value_max,
        "working_mode": output.mode,
        "working_dtype": "uint8",
        "intensity_mapping": "identity" if identity else "linear_minmax_to_uint8",
    }
    return output, metadata


def _base_params(seed: int | None) -> dict[str, Any]:
    return {"random_seed": seed} if seed is not None else {}


def _albumentations_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))


def _apply_albumentations(image: np.ndarray, transform: A.BasicTransform, seed: int) -> np.ndarray:
    pipeline = A.Compose([transform], seed=seed)
    return np.asarray(pipeline(image=image)["image"])


def _result(
    working: np.ndarray,
    state: _ImageState,
    params: dict[str, Any],
    return_params: bool,
) -> DefectOutput:
    output = _restore_image(working, state)
    return (output, params) if return_params else output


def gaussian_noise(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Add independent Gaussian noise using Albumentations."""
    config = _validate_severity("gaussian_noise", severity)
    generator, source_seed = _coerce_rng(rng)
    sigma = float(generator.uniform(*config["sigma_range"]))
    transform_seed = _albumentations_seed(generator)
    working, state = _prepare_image(image)
    transform = A.GaussNoise(
        std_range=(sigma / 255.0, sigma / 255.0),
        mean_range=(0.0, 0.0),
        per_channel=True,
        p=1.0,
    )
    degraded = _apply_albumentations(working, transform, transform_seed)
    params = _base_params(source_seed)
    params.update({"sigma": sigma, "albumentations_seed": transform_seed})
    return _result(degraded, state, params, return_params)


def poisson_noise(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Add signal-dependent Poisson shot noise using Albumentations."""
    config = _validate_severity("poisson_noise", severity)
    generator, source_seed = _coerce_rng(rng)
    scale = float(generator.uniform(*config["scale_range"]))
    transform_seed = _albumentations_seed(generator)
    working, state = _prepare_image(image)
    transform = A.ShotNoise(scale_range=(scale, scale), p=1.0)
    degraded = _apply_albumentations(working, transform, transform_seed)
    params = _base_params(source_seed)
    params.update({"scale": scale, "albumentations_seed": transform_seed})
    return _result(degraded, state, params, return_params)


def gaussian_blur(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Apply Gaussian blur using a fixed sampled odd kernel and sigma."""
    config = _validate_severity("gaussian_blur", severity)
    generator, source_seed = _coerce_rng(rng)
    kernel_size = int(generator.choice(config["kernel_sizes"]))
    sigma = float(generator.uniform(*config["sigma_range"]))
    transform_seed = _albumentations_seed(generator)
    working, state = _prepare_image(image)
    transform = A.GaussianBlur(
        blur_limit=(kernel_size, kernel_size),
        sigma_limit=(sigma, sigma),
        p=1.0,
    )
    degraded = _apply_albumentations(working, transform, transform_seed)
    params = _base_params(source_seed)
    params.update(
        {
            "kernel_size": kernel_size,
            "sigma": sigma,
            "albumentations_seed": transform_seed,
        }
    )
    return _result(degraded, state, params, return_params)


def motion_blur(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Apply a manually constructed directional line-spread kernel."""
    config = _validate_severity("motion_blur", severity)
    generator, source_seed = _coerce_rng(rng)
    working, state = _prepare_image(image)
    max_kernel = max(1, min(working.shape[:2]))
    if max_kernel % 2 == 0:
        max_kernel -= 1
    eligible = [size for size in config["kernel_sizes"] if size <= max_kernel]
    kernel_size = int(generator.choice(eligible)) if eligible else max_kernel
    angle = float(generator.uniform(*config["angle_range"]))

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    radius = center
    radians = np.deg2rad(angle)
    dx, dy = radius * np.cos(radians), radius * np.sin(radians)
    start = (int(round(center - dx)), int(round(center - dy)))
    end = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, start, end, color=1.0, thickness=1, lineType=cv2.LINE_AA)
    kernel_sum = float(kernel.sum())
    if kernel_sum == 0.0:
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    else:
        kernel /= kernel_sum
    degraded = cv2.filter2D(working, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    params = _base_params(source_seed)
    params.update({"kernel_size": kernel_size, "angle_degrees": angle})
    return _result(degraded, state, params, return_params)


def brightness_shift(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Shift all intensities by a randomly signed offset."""
    config = _validate_severity("brightness_shift", severity)
    generator, source_seed = _coerce_rng(rng)
    low, high = config["offset_range"]
    magnitude = int(generator.integers(low, high + 1))
    signed_offset = magnitude if bool(generator.integers(0, 2)) else -magnitude
    working, state = _prepare_image(image)
    degraded = np.clip(working.astype(np.int16) + signed_offset, 0, 255).astype(np.uint8)
    params = _base_params(source_seed)
    params.update({"offset": signed_offset})
    return _result(degraded, state, params, return_params)


def contrast_reduction(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Compress intensities toward 8-bit mid-gray."""
    config = _validate_severity("contrast_reduction", severity)
    generator, source_seed = _coerce_rng(rng)
    factor = float(generator.uniform(*config["factor_range"]))
    midpoint = float(config["mid_gray"])
    working, state = _prepare_image(image)
    degraded = np.rint((working.astype(np.float32) - midpoint) * factor + midpoint)
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    params = _base_params(source_seed)
    params.update({"contrast_factor": factor, "mid_gray": midpoint})
    return _result(degraded, state, params, return_params)


def jpeg_compression(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Perform a real JPEG encode/decode cycle through Albumentations/OpenCV."""
    config = _validate_severity("jpeg_compression", severity)
    generator, source_seed = _coerce_rng(rng)
    quality = int(config["quality"])
    transform_seed = _albumentations_seed(generator)
    working, state = _prepare_image(image)
    transform = A.ImageCompression(
        compression_type="jpeg",
        quality_range=(quality, quality),
        p=1.0,
    )
    degraded = _apply_albumentations(working, transform, transform_seed)
    params = _base_params(source_seed)
    params.update({"quality": quality, "albumentations_seed": transform_seed})
    return _result(degraded, state, params, return_params)


def occlusion_patch(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    fill_mode: str = "random",
    return_params: bool = False,
) -> DefectOutput:
    """Place a random rectangle filled with black, noise, or an edge color."""
    config = _validate_severity("occlusion_patch", severity)
    if fill_mode not in {"random", "black", "noise", "edge"}:
        raise ValueError("fill_mode must be one of: random, black, noise, edge")
    generator, source_seed = _coerce_rng(rng)
    working, state = _prepare_image(image)
    height, width = working.shape[:2]
    fraction = float(generator.uniform(*config["area_fraction_range"]))
    aspect_ratio = float(generator.uniform(*config["aspect_ratio_range"]))
    target_area = max(1.0, fraction * height * width)
    patch_width = max(1, min(width, int(round(np.sqrt(target_area * aspect_ratio)))))
    patch_height = max(1, min(height, int(round(target_area / patch_width))))
    x = int(generator.integers(0, width - patch_width + 1))
    y = int(generator.integers(0, height - patch_height + 1))
    chosen_fill = (
        str(generator.choice(config["fill_modes"])) if fill_mode == "random" else fill_mode
    )

    degraded = working.copy()
    fill_params: dict[str, Any] = {}
    patch_shape = (patch_height, patch_width) + (() if working.ndim == 2 else (working.shape[2],))
    if chosen_fill == "black":
        patch = np.zeros(patch_shape, dtype=np.uint8)
        fill_params["fill_value"] = 0 if working.ndim == 2 else [0] * working.shape[2]
    elif chosen_fill == "noise":
        mean = np.mean(working, axis=(0, 1))
        std = np.maximum(np.std(working, axis=(0, 1)), config["noise_std_floor"])
        patch = generator.normal(mean, std, size=patch_shape)
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        fill_params["noise_mean"] = np.asarray(mean).round(6).tolist()
        fill_params["noise_std"] = np.asarray(std).round(6).tolist()
    else:
        if working.ndim == 2:
            edges = np.concatenate((working[0, :], working[-1, :], working[:, 0], working[:, -1]))
        else:
            edges = np.concatenate(
                (working[0, :, :], working[-1, :, :], working[:, 0, :], working[:, -1, :]),
                axis=0,
            )
        sampled_color = edges[int(generator.integers(0, len(edges)))]
        patch = np.empty(patch_shape, dtype=np.uint8)
        patch[...] = sampled_color
        fill_params["sampled_edge_value"] = np.asarray(sampled_color).tolist()

    degraded[y : y + patch_height, x : x + patch_width] = patch
    params = _base_params(source_seed)
    params.update(
        {
            "x": x,
            "y": y,
            "width": patch_width,
            "height": patch_height,
            "target_area_fraction": fraction,
            "actual_area_fraction": (patch_width * patch_height) / (width * height),
            "aspect_ratio": aspect_ratio,
            "fill_mode": chosen_fill,
            **fill_params,
        }
    )
    return _result(degraded, state, params, return_params)


def rotation_crop(
    image: ImageInput,
    severity: int,
    *,
    rng: RngInput = None,
    return_params: bool = False,
) -> DefectOutput:
    """Rotate, crop away border pixels, then resize to the original dimensions."""
    config = _validate_severity("rotation_crop", severity)
    generator, source_seed = _coerce_rng(rng)
    magnitude = float(generator.uniform(*config["absolute_angle_range"]))
    angle = magnitude if bool(generator.integers(0, 2)) else -magnitude
    transform_seed = _albumentations_seed(generator)
    working, state = _prepare_image(image)
    height, width = working.shape[:2]
    pipeline = A.Compose(
        [
            A.Rotate(
                limit=(angle, angle),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                crop_border=True,
                p=1.0,
            ),
            A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, p=1.0),
        ],
        seed=transform_seed,
        save_applied_params=True,
    )
    result = pipeline(image=working)
    degraded = np.asarray(result["image"])
    rotate_params = result["applied_transforms"][0][1]
    crop_box = [
        int(rotate_params["x_min"]),
        int(rotate_params["y_min"]),
        int(rotate_params["x_max"]),
        int(rotate_params["y_max"]),
    ]
    params = _base_params(source_seed)
    params.update(
        {
            "angle_degrees": angle,
            "crop_border": True,
            "crop_box_xyxy": crop_box,
            "resized_to": [width, height],
            "interpolation": "linear",
            "albumentations_seed": transform_seed,
        }
    )
    return _result(degraded, state, params, return_params)


DEFECT_FUNCTIONS: dict[str, Callable[..., DefectOutput]] = {
    "gaussian_noise": gaussian_noise,
    "poisson_noise": poisson_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness_shift": brightness_shift,
    "contrast_reduction": contrast_reduction,
    "jpeg_compression": jpeg_compression,
    "occlusion_patch": occlusion_patch,
    "rotation_crop": rotation_crop,
}


def available_defects() -> tuple[str, ...]:
    """Return supported defect names in stable display/CLI order."""
    return tuple(DEFECT_FUNCTIONS)


def apply_defect(
    image: ImageInput,
    defect_name: str,
    severity: int,
    *,
    rng: RngInput = None,
    occlusion_fill: str = "random",
) -> tuple[ImageInput, dict[str, Any]]:
    """Apply a named defect and always return its replay parameters."""
    try:
        function = DEFECT_FUNCTIONS[defect_name]
    except KeyError as error:
        raise ValueError(
            f"unknown defect {defect_name!r}; choose from {', '.join(available_defects())}"
        ) from error
    kwargs: dict[str, Any] = {"rng": rng, "return_params": True}
    if defect_name == "occlusion_patch":
        kwargs["fill_mode"] = occlusion_fill
    output, params = function(image, severity, **kwargs)
    return output, params