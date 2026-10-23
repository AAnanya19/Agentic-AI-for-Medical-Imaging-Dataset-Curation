"""Single-image inspection: score one arbitrary image and return a decision.

The batch pipeline processes a corpus that has already been curated, validated
and canonicalised. This module handles the opposite case - **one image the system
has never seen** - which is what a demonstration interface needs and what a
deployed inspector would actually receive.

That difference is almost entirely about input handling. Every mode that has
caused a failure in this project appears here: RGBA with an opaque alpha channel
(23 images in ChestX-ray14), 16-bit greyscale, CMYK, palette images, sources
spanning 0.01 to 302 megapixels, and files that are not images at all. Rather
than writing a second loading path, this reuses ``prepare_cli_source`` from
``inject_defects`` - the same function the corpus went through - because a second
path that looks equivalent but is not is precisely how the RGBA bug reached the
duplicate benchmark.

**Thresholds are loaded, never derived.** A single image cannot supply a clean
validation distribution, so the thresholds come from a completed batch run. An
interface that invented thresholds per upload would produce numbers that look
authoritative and mean nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    from scripts.inject_defects import prepare_cli_source
    from agents.data_inspector import DataInspectorAgent
    from agents.inspector_config import InspectorConfig
    from pipeline.iqa_metrics import (
        METRICS,
        BrightnessReference,
        brightness_deviation,
        compute_all_metrics,
    )
    from pipeline.resolution import ResolutionThresholds
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.inject_defects import prepare_cli_source  # type: ignore[no-redef]
    from agents.data_inspector import DataInspectorAgent  # type: ignore[no-redef]
    from agents.inspector_config import InspectorConfig  # type: ignore[no-redef]
    from pipeline.iqa_metrics import (  # type: ignore[no-redef]
        METRICS,
        BrightnessReference,
        brightness_deviation,
        compute_all_metrics,
    )
    from pipeline.resolution import ResolutionThresholds  # type: ignore[no-redef]


# Refuse absurdly large uploads before decoding. Pillow's own decompression-bomb
# guard is relaxed inside the pipeline for legitimately large medical sources, so
# an explicit ceiling is needed here where the input is untrusted.
MAX_UPLOAD_MEGAPIXELS = 400.0
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class UnsupportedImageError(ValueError):
    """Raised when an upload cannot be inspected, with a reason for the user."""


@dataclass
class InspectionContext:
    """Everything needed to inspect one image, loaded from a completed batch run.

    Thresholds and the brightness reference are properties of a *dataset*, not of
    an image, so they must be carried in rather than computed. ``dataset`` is
    recorded so the interface can state which corpus the thresholds came from -
    applying ChestX-ray14 thresholds to a dermoscopy upload would produce a
    confident but meaningless verdict.
    """

    dataset: str
    thresholds: dict[str, dict[str, float]]
    config: InspectorConfig
    brightness_reference: BrightnessReference
    fusion_model: Any | None = None
    fusion_feature_names: list[str] = None  # type: ignore[assignment]
    inspection_run_id: str | None = None

    @classmethod
    def fit_fusion(
        cls, dataset: str, data_dir: Path, config: InspectorConfig
    ) -> tuple[Any | None, list[str]]:
        """Refit the composite scorer from the dataset's own validation rows.

        The batch pipeline fits this during a run and does not persist the model,
        so it is refitted here from the same rows using the same seed - which
        yields an identical model rather than an approximation. Without it the
        overall verdict is unavailable, and the interface would be reduced to
        per-check attribution with no decision.
        """
        try:
            from pipeline.fusion import LogisticFusion
            from scripts.run_inspector import metric_polarity

            scores_path = data_dir / dataset / "iqa_scores.csv"
            if not scores_path.is_file():
                return None, []
            frame = pd.read_csv(scores_path)
            validation = frame[frame["split"] == "val"]
            if validation.empty:
                return None, []

            feature_names: list[str] = []
            columns: list[np.ndarray] = []
            for name in [*METRICS, "brightness_deviation"]:
                if name not in validation.columns:
                    continue
                polarity = metric_polarity(name)
                if polarity is None:
                    continue   # non-directional metrics cannot be oriented
                values = validation[name].to_numpy(dtype=float)
                columns.append(-values if polarity else values)
                feature_names.append(f"iqa:{name}")
            if not columns:
                return None, []

            matrix = np.column_stack(columns)
            complete = np.isfinite(matrix).all(axis=1)
            labels = (
                validation["defect_type"] != "clean"
            ).astype(int).to_numpy()[complete]
            if complete.sum() < 20 or len(np.unique(labels)) < 2:
                return None, []
            model = LogisticFusion(random_state=config.seed).fit(
                matrix[complete], labels, feature_names=feature_names
            )
            return model, feature_names
        except Exception:  # noqa: BLE001
            # A missing scorer degrades the interface; it must not break it.
            return None, []

    @classmethod
    def load(
        cls, dataset: str, data_dir: Path, logs_dir: Path | None = None
    ) -> "InspectionContext":
        logs_dir = logs_dir or (data_dir / "logs")
        inspection_path = logs_dir / f"{dataset}_inspection.json"
        if not inspection_path.is_file():
            raise FileNotFoundError(
                f"{inspection_path} not found. Single-image inspection needs "
                "thresholds from a completed batch run: "
                f"python scripts/run_pipeline.py --dataset {dataset}"
            )
        inspection = json.loads(inspection_path.read_text())

        reference_path = data_dir / dataset / "brightness_reference.json"
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"{reference_path} not found; run compute_iqa.py for {dataset}"
            )
        reference = BrightnessReference.from_dict(
            json.loads(reference_path.read_text())
        )

        config_path = logs_dir / f"{dataset}_inspector_config_used.json"
        config = (
            InspectorConfig.load(config_path) if config_path.is_file()
            else InspectorConfig()
        )
        model, feature_names = cls.fit_fusion(dataset, data_dir, config)
        return cls(
            dataset=dataset,
            thresholds=inspection["thresholds"],
            config=config,
            brightness_reference=reference,
            fusion_model=model,
            fusion_feature_names=feature_names or [],
            inspection_run_id=inspection.get("provenance", {}).get("run_id"),
        )


def load_upload(handle: BinaryIO | Path, canonical_size: int = 256) -> tuple[
    Image.Image, dict[str, Any]
]:
    """Decode an arbitrary upload and canonicalise it, or explain why not.

    Returns the canonical image alongside the source properties a resolution
    check needs. Every failure mode gets a message a user can act on rather than
    a traceback.
    """
    try:
        if isinstance(handle, (str, Path)):
            path = Path(handle)
            if path.stat().st_size > MAX_UPLOAD_BYTES:
                raise UnsupportedImageError(
                    f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                )
            image = Image.open(path)
        else:
            image = Image.open(handle)
        # Force a read so a truncated file fails here rather than mid-metric.
        image.load()
    except UnidentifiedImageError as error:
        raise UnsupportedImageError(
            "this file is not a readable image. Supported formats include PNG, "
            "JPEG, TIFF and BMP."
        ) from error
    except UnsupportedImageError:
        raise
    except Exception as error:  # noqa: BLE001
        raise UnsupportedImageError(f"could not read the image: {error}") from error

    width, height = image.size
    megapixels = width * height / 1e6
    if megapixels > MAX_UPLOAD_MEGAPIXELS:
        raise UnsupportedImageError(
            f"image is {megapixels:.0f} megapixels, above the "
            f"{MAX_UPLOAD_MEGAPIXELS:.0f} MP ceiling for interactive use"
        )
    if min(width, height) < 8:
        raise UnsupportedImageError(
            f"image is only {width}x{height}; too small to assess"
        )

    source = {
        "original_width": width,
        "original_height": height,
        "original_mode": image.mode,
        "megapixels": megapixels,
        "aspect_ratio": max(width, height) / min(width, height),
        # An upload smaller than the canonical size was upscaled, so its apparent
        # detail is interpolation. Worth surfacing, since every metric below then
        # measures fabricated structure.
        "upscaled_to_canonical": min(width, height) < canonical_size,
    }

    try:
        # The same preparation the corpus went through: alpha flattening,
        # normalisation to 8-bit, centre-crop and resize. Reused deliberately
        # rather than reimplemented.
        canonical, resize_info = prepare_cli_source(
            image, Path("upload"), canonical_size
        )
    except Exception as error:  # noqa: BLE001
        raise UnsupportedImageError(
            f"the image could not be canonicalised: {error}"
        ) from error
    source["canonical_mode"] = canonical.mode
    source["resize_info"] = resize_info
    return canonical, source


def score_image(
    canonical: Image.Image, context: InspectionContext
) -> dict[str, float]:
    """Every registered IQA metric plus brightness deviation, on one image."""
    array = np.asarray(canonical)
    scores = compute_all_metrics(array)
    scores["brightness_deviation"] = brightness_deviation(
        array, context.brightness_reference
    )
    return scores


def check_resolution_only(
    source: dict[str, Any], thresholds: ResolutionThresholds | None = None
) -> dict[str, Any]:
    """Resolution assessment for a single image, from its source dimensions.

    The batch check compares against the dataset's own distribution, which a lone
    image has no access to - so only the absolute criteria apply here, and that
    limitation is stated in the returned payload.
    """
    settings = thresholds or ResolutionThresholds()
    minimum_side = min(source["original_width"], source["original_height"])
    return {
        "flagged": bool(
            minimum_side < settings.min_side
            or source["aspect_ratio"] > settings.max_aspect_ratio
        ),
        "below_min_side": bool(minimum_side < settings.min_side),
        "extreme_aspect_ratio": bool(
            source["aspect_ratio"] > settings.max_aspect_ratio
        ),
        "original_width": source["original_width"],
        "original_height": source["original_height"],
        "min_side_threshold": settings.min_side,
        "max_aspect_ratio_threshold": settings.max_aspect_ratio,
        "scoreable": True,
        "note": (
            "absolute criteria only; the distributional check needs a dataset "
            "and is not available for a single image"
        ),
    }


def inspect_single_image(
    handle: BinaryIO | Path,
    context: InspectionContext,
    canonical_size: int = 256,
) -> dict[str, Any]:
    """Decide on one arbitrary image, with the evidence behind the decision.

    Reuses ``DataInspectorAgent.inspect_image`` rather than reimplementing the
    decision logic, so the interface and the batch pipeline cannot drift apart -
    a demo that disagreed with the pipeline would be worse than no demo.
    """
    canonical, source = load_upload(handle, canonical_size)
    scores = score_image(canonical, context)

    agent = DataInspectorAgent(context.config)
    agent.thresholds = context.thresholds
    agent._threshold_source_dataset = context.dataset

    row = pd.Series({
        "image_id": "upload",
        "split": None,
        "defect_type": None,      # unknown: no ground truth for an upload
        "severity_level": None,
        **scores,
    })

    overall_column = context.config.overall_check.score_column
    if overall_column not in row.index:
        row[overall_column] = _composite_score(scores, context)

    resolution_row = None
    if context.config.enable_resolution_check:
        resolution_row = pd.Series(check_resolution_only(source))

    result = agent.inspect_image(row, resolution_row, duplicate_of=None)
    result["source"] = source
    result["raw_scores"] = {
        name: float(value) for name, value in scores.items()
    }
    result["thresholds_from"] = {
        "dataset": context.dataset,
        "inspection_run_id": context.inspection_run_id,
    }
    result["canonical_size"] = canonical_size
    return result, canonical


def _composite_score(
    scores: dict[str, float], context: InspectionContext
) -> float:
    """The fused quality score for one image, or NaN when unavailable.

    NaN rather than a substituted single metric: the Inspector reports an
    unscoreable image as such, which is honest, whereas presenting one metric as
    though it were the composite would give a confident verdict on weaker
    evidence than the label implies.
    """
    if context.fusion_model is None or not context.fusion_feature_names:
        return float("nan")
    try:
        from scripts.run_inspector import metric_polarity

        values = []
        for name in context.fusion_feature_names:
            metric = name.removeprefix("iqa:")
            if metric not in scores:
                return float("nan")
            polarity = metric_polarity(metric)
            raw = float(scores[metric])
            values.append(-raw if polarity else raw)
        vector = np.asarray(values, dtype=float).reshape(1, -1)
        if not np.isfinite(vector).all():
            return float("nan")
        return float(context.fusion_model.score(vector)[0])
    except Exception:  # noqa: BLE001
        return float("nan")


def per_check_table(result: dict[str, Any]) -> pd.DataFrame:
    """Score beside threshold for every check, ready to display.

    **Both columns are in oriented space**, where higher always means worse. The
    stored thresholds are oriented (a metric with higher_is_better=True has its
    sign flipped when the threshold is derived), so showing a raw score against an
    oriented threshold would present two incomparable numbers - a clean image with
    Laplacian variance 610 would appear to fail a threshold of -614. The raw
    values are still available separately for anyone who wants them.
    """
    rows = []
    for name, entry in result.get("checks", {}).items():
        raw = entry.get("score")
        oriented = entry.get("oriented_score")
        rows.append({
            "check": name,
            "score (oriented)": oriented,
            "flag threshold": entry.get("flag_threshold"),
            "raw score": raw,
            "flagged": bool(entry.get("flagged")),
            "assessed": bool(entry.get("scoreable", True)),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Flagged checks first, so the explanation for a verdict is at the top.
    return frame.sort_values(
        ["flagged", "check"], ascending=[False, True]
    ).reset_index(drop=True)


def available_datasets(data_dir: Path, logs_dir: Path | None = None) -> list[str]:
    """Datasets with a completed batch run, and therefore usable thresholds."""
    logs_dir = logs_dir or (data_dir / "logs")
    if not logs_dir.is_dir():
        return []
    found = []
    for path in sorted(logs_dir.glob("*_inspection.json")):
        dataset = path.name.removesuffix("_inspection.json")
        if (data_dir / dataset / "brightness_reference.json").is_file():
            found.append(dataset)
    return found