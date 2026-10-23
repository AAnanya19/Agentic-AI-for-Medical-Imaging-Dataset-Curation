"""Tests for synthetic defect generation and CLI safety."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import scripts.inject_defects as inject_defects_module
from pipeline.defects import DEFECT_FUNCTIONS, gaussian_blur, gaussian_noise
from scripts.inject_defects import (
    Assignment,
    apply_assignment,
    build_assignments,
    main as inject_main,
    prepare_cli_source,
)


@pytest.fixture
def textured_grayscale() -> Image.Image:
    y, x = np.indices((128, 128))
    array = (
        ((x // 4 + y // 4) % 2) * 170
        + ((x * 3 + y * 5) % 80)
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


@pytest.mark.parametrize("defect_name", tuple(DEFECT_FUNCTIONS))
def test_each_defect_changes_image_and_preserves_dimensions(
    defect_name: str,
    textured_grayscale: Image.Image,
) -> None:
    function = DEFECT_FUNCTIONS[defect_name]
    degraded, params = function(
        textured_grayscale,
        2,
        rng=123,
        return_params=True,
    )
    assert degraded.size == textured_grayscale.size
    assert degraded.mode == "L"
    assert not np.array_equal(np.asarray(degraded), np.asarray(textured_grayscale))
    assert params["random_seed"] == 123
    replayed, replayed_params = function(
        textured_grayscale,
        2,
        rng=123,
        return_params=True,
    )
    assert np.array_equal(np.asarray(degraded), np.asarray(replayed))
    assert params == replayed_params


def global_structural_index(first: np.ndarray, second: np.ndarray) -> float:
    """Small dependency-free global SSIM suitable for ordinal test assertions."""
    first_float = first.astype(np.float64)
    second_float = second.astype(np.float64)
    mean_first = first_float.mean()
    mean_second = second_float.mean()
    variance_first = first_float.var()
    variance_second = second_float.var()
    covariance = ((first_float - mean_first) * (second_float - mean_second)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (
        (mean_first**2 + mean_second**2 + c1)
        * (variance_first + variance_second + c2)
    )
    return float(numerator / denominator)


@pytest.mark.parametrize("function", (gaussian_noise, gaussian_blur))
def test_noise_and_blur_severity_monotonically_reduce_ssim(
    function,
    textured_grayscale: Image.Image,
) -> None:
    original = np.asarray(textured_grayscale)
    scores = [
        global_structural_index(
            original, np.asarray(function(textured_grayscale, severity, rng=123))
        )
        for severity in (1, 2, 3)
    ]
    assert scores[0] > scores[1] > scores[2]


def test_reusable_function_preserves_numpy_uint16_shape_and_dtype() -> None:
    source = np.linspace(0, 4095, num=64 * 64, dtype=np.uint16).reshape(64, 64)
    degraded = gaussian_noise(source, 1, rng=9)
    assert isinstance(degraded, np.ndarray)
    assert degraded.shape == source.shape
    assert degraded.dtype == source.dtype


def test_cli_prepares_uint16_once_before_stacked_defects() -> None:
    source_array = np.linspace(0, 4095, num=64 * 64, dtype=np.uint16).reshape(64, 64)
    source = Image.fromarray(source_array)
    working, baseline = prepare_cli_source(source, Path("example.png"), 64)
    assignment = Assignment(
        row_index=0,
        clean=False,
        defect_names=("brightness_shift", "contrast_reduction"),
        severities=(2, 2),
        seeds=(11, 12),
    )
    degraded, steps = apply_assignment(working, assignment, "random")
    assert baseline["original_dtype"] == "uint16"
    assert baseline["intensity_mapping"] == "linear_minmax_to_uint8"
    assert working.mode == degraded.mode == "L"
    assert len(steps) == 2


def test_clean_fraction_is_applied_independently_per_split() -> None:
    metadata = pd.DataFrame(
        {
            "image_id": [f"image-{index}" for index in range(20)],
            "split": ["train"] * 10 + ["test"] * 10,
        }
    )
    assignments = build_assignments(
        metadata,
        defects=("gaussian_noise", "gaussian_blur"),
        severities=(1, 2, 3),
        fraction_clean=0.3,
        seed=42,
        allow_combinations=False,
    )
    assert sum(item.clean for item in assignments[:10]) == 3
    assert sum(item.clean for item in assignments[10:]) == 3


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    image_path = raw_dir / "example.png"
    Image.new("L", (32, 32), color=127).save(image_path)
    pd.DataFrame(
        [
            {
                "image_id": "example",
                "path": "raw/example.png",
                "split": "test",
            }
        ]
    ).to_csv(dataset_dir / "metadata.csv", index=False)

    inject_main(
        [
            "--dataset",
            "demo",
            "--data-dir",
            str(data_dir),
            "--defects",
            "gaussian_noise",
            "--dry-run",
        ]
    )

    assert not (dataset_dir / "processed").exists()
    assert not (dataset_dir / "processed_metadata.csv").exists()
    assert not (data_dir / "logs").exists()


def test_cli_reads_mixed_source_extensions_from_metadata_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    Image.new("L", (24, 24), color=80).save(raw_dir / "source-a.jpg")
    Image.new("L", (24, 24), color=160).save(raw_dir / "source-b.png")
    pd.DataFrame(
        [
            {
                "image_id": "ID_DOES_NOT_MATCH_JPEG_NAME",
                "path": "raw/source-a.jpg",
                "split": "test",
            },
            {
                "image_id": "ID_DOES_NOT_MATCH_PNG_NAME",
                "path": "raw/source-b.png",
                "split": "test",
            },
        ]
    ).to_csv(dataset_dir / "metadata.csv", index=False)

    inject_main(
        [
            "--dataset",
            "demo",
            "--data-dir",
            str(data_dir),
            "--defects",
            "gaussian_noise",
            "--fraction-clean",
            "1",
            "--seed",
            "5",
        ]
    )

    processed = pd.read_csv(dataset_dir / "processed_metadata.csv", dtype=str)
    assert set(processed["original_image_id"]) == {
        "ID_DOES_NOT_MATCH_JPEG_NAME",
        "ID_DOES_NOT_MATCH_PNG_NAME",
    }
    source_suffixes = {
        json.loads(payload)["baseline_preprocessing"]["source_suffix"]
        for payload in processed["defect_params_json"]
    }
    assert source_suffixes == {".jpg", ".png"}


def _build_cli_failure_batch(data_dir: Path, image_count: int) -> Path:
    dataset_dir = data_dir / "demo"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    rows = []
    for index in range(image_count):
        image_id = f"image-{index:03d}"
        path = raw_dir / f"{image_id}.png"
        Image.new("L", (8, 8), color=index % 256).save(path)
        rows.append(
            {
                "image_id": image_id,
                "path": f"raw/{path.name}",
                "split": "train",
            }
        )
    pd.DataFrame(rows).to_csv(dataset_dir / "metadata.csv", index=False)
    return dataset_dir


def _simulate_decompression_bombs(monkeypatch, failing_ids: set[str]) -> None:
    real_load_image = inject_defects_module.load_image

    def load_with_failures(path: Path) -> Image.Image:
        if path.stem in failing_ids:
            raise Image.DecompressionBombError("simulated oversized image")
        return real_load_image(path)

    monkeypatch.setattr(inject_defects_module, "load_image", load_with_failures)


def test_cli_accepts_small_failure_rate_and_writes_failure_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    dataset_dir = _build_cli_failure_batch(data_dir, image_count=50)
    failing_ids = {"image-003", "image-017"}
    _simulate_decompression_bombs(monkeypatch, failing_ids)

    inject_main(
        [
            "--dataset",
            "demo",
            "--data-dir",
            str(data_dir),
            "--defects",
            "gaussian_noise",
            "--fraction-clean",
            "1",
            "--visualize",
            "1",
        ]
    )

    processed = pd.read_csv(dataset_dir / "processed_metadata.csv", dtype=str)
    assert len(processed) == 48
    report_path = data_dir / "logs" / "demo_defect_injection_failures.json"
    failures = json.loads(report_path.read_text(encoding="utf-8"))
    assert {failure["image_id"] for failure in failures} == failing_ids
    assert all(
        failure["error"].startswith("DecompressionBombError:")
        for failure in failures
    )
    assert (data_dir / "logs" / "defect_examples_demo.png").is_file()


def test_cli_raises_when_failure_rate_exceeds_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    dataset_dir = _build_cli_failure_batch(data_dir, image_count=20)
    failing_ids = {"image-003", "image-017"}
    _simulate_decompression_bombs(monkeypatch, failing_ids)

    with pytest.raises(RuntimeError, match=r"2/20 images failed \(10\.00%\).+5%"):
        inject_main(
            [
                "--dataset",
                "demo",
                "--data-dir",
                str(data_dir),
                "--defects",
                "gaussian_noise",
                "--fraction-clean",
                "1",
            ]
        )

    processed = pd.read_csv(dataset_dir / "processed_metadata.csv", dtype=str)
    assert len(processed) == 18
    report_path = data_dir / "logs" / "demo_defect_injection_failures.json"
    failures = json.loads(report_path.read_text(encoding="utf-8"))
    assert {failure["image_id"] for failure in failures} == failing_ids


def test_visualization_contains_defect_severity_rows_and_applied_params(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.png"
    y, x = np.indices((600, 800))
    pixels = ((x // 4 + y // 3) % 128).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(source_path)
    metadata = pd.DataFrame(
        [{"image_id": "source", "_source_path": source_path}]
    )

    def mock_apply_defect(
        image: Image.Image,
        defect_name: str,
        severity: int,
        **_kwargs,
    ) -> tuple[Image.Image, dict[str, float | int]]:
        if defect_name == "gaussian_noise":
            params: dict[str, float | int] = {"sigma": 4.25 + severity}
        else:
            params = {
                "kernel_size": 5 + 2 * severity,
                "angle_degrees": 12.5 * severity,
            }
        degraded = np.asarray(image, dtype=np.uint8).astype(np.int16) + severity
        return Image.fromarray(degraded.astype(np.uint8), mode="L"), params

    monkeypatch.setattr(
        inject_defects_module,
        "apply_defect",
        mock_apply_defect,
    )
    figure = inject_defects_module.build_visualization_figure(
        metadata=metadata,
        defects=("gaussian_noise", "motion_blur"),
        severities=(1, 2),
        examples_per_cell=1,
        dataset="demo",
        seed=11,
        occlusion_fill="black",
        canonical_size=600,
    )
    try:
        texts = [text.get_text() for axis in figure.axes for text in axis.texts]
        assert "gaussian_noise — severity 1 (mild)" in texts
        assert "gaussian_noise — severity 2 (moderate)" in texts
        assert "motion_blur — severity 1 (mild)" in texts
        assert "motion_blur — severity 2 (moderate)" in texts
        assert "σ=5.25" in texts
        assert "kernel=9×9 · angle=+25.0°" in texts
        assert "400×400 center crop · ×3 nearest" in texts

        original_axes = figure.axes[0::2]
        degraded_axes = figure.axes[1::2]
        assert len(original_axes) == len(degraded_axes) == 4
        for row_number, (original_axis, degraded_axis) in enumerate(
            zip(original_axes, degraded_axes)
        ):
            severity = 1 if row_number % 2 == 0 else 2
            original_view = np.asarray(original_axis.images[0].get_array())
            degraded_view = np.asarray(degraded_axis.images[0].get_array())
            assert original_view.shape == degraded_view.shape == (1200, 1200)
            assert np.array_equal(degraded_view, original_view + severity)
            assert original_axis.images[0].get_interpolation() == "nearest"
            assert degraded_axis.images[0].get_interpolation() == "nearest"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_visualization_center_crop_handles_images_smaller_than_crop() -> None:
    box = inject_defects_module.visualization_center_crop_box(120, 300)
    assert box == (0, 0, 120, 300)
    pixels = np.arange(300 * 120, dtype=np.int32).reshape(300, 120)
    assert np.array_equal(
        inject_defects_module.crop_visualization_array(pixels, box),
        pixels,
    )


def test_visualization_labels_full_frame_context_defects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "context.png"
    Image.new("L", (700, 500), color=96).save(source_path)
    metadata = pd.DataFrame([{"image_id": "context", "_source_path": source_path}])

    def mock_apply_defect(
        image: Image.Image,
        defect_name: str,
        _severity: int,
        **_kwargs,
    ) -> tuple[Image.Image, dict[str, float | int | str]]:
        if defect_name == "rotation_crop":
            params: dict[str, float | int | str] = {"angle_degrees": 18.0}
        else:
            params = {
                "actual_area_fraction": 0.25,
                "width": 300,
                "height": 290,
                "fill_mode": "black",
            }
        return image.copy(), params

    monkeypatch.setattr(inject_defects_module, "apply_defect", mock_apply_defect)
    figure = inject_defects_module.build_visualization_figure(
        metadata=metadata,
        defects=("rotation_crop", "occlusion_patch"),
        severities=(3,),
        examples_per_cell=1,
        dataset="demo",
        seed=11,
        occlusion_fill="black",
        canonical_size=500,
    )
    try:
        texts = [text.get_text() for axis in figure.axes for text in axis.texts]
        assert any("full frame · scaled\n500×500" in text for text in texts)
        for axis in figure.axes:
            assert np.asarray(axis.images[0].get_array()).shape == (500, 500)
            assert axis.images[0].get_interpolation() == "antialiased"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_cli_writes_combinations_metadata_and_visualization(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)
    source_rows = []
    y, x = np.indices((48, 48))
    for index, split in enumerate(("train", "train", "val", "val", "test", "test")):
        image_id = f"example-{index}"
        pixels = ((x * (index + 2) + y * 3 + (x // 5 % 2) * 90) % 256).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(raw_dir / f"{image_id}.png")
        source_rows.append(
            {
                "image_id": image_id,
                "path": f"raw/{image_id}.png",
                "split": split,
            }
        )
    pd.DataFrame(source_rows).to_csv(dataset_dir / "metadata.csv", index=False)

    inject_main(
        [
            "--dataset",
            "demo",
            "--data-dir",
            str(data_dir),
            "--defects",
            "gaussian_noise,occlusion_patch",
            "--severities",
            "1,2",
            "--fraction-clean",
            "0.5",
            "--allow-combinations",
            "--visualize",
            "1",
            "--seed",
            "7",
        ]
    )

    processed = pd.read_csv(dataset_dir / "processed_metadata.csv", dtype=str)
    assert list(processed.columns) == [
        "image_id",
        "original_image_id",
        "path",
        "split",
        "defect_type",
        "severity_level",
        "canonical_size",
        "defect_params_json",
    ]
    assert len(processed) == len(source_rows)
    clean_per_split = processed.groupby("split")["defect_type"].apply(
        lambda values: (values == "clean").sum()
    )
    assert (clean_per_split == 1).all()
    combined = processed[processed["defect_type"] != "clean"]
    assert combined["defect_type"].str.contains("+", regex=False).all()
    for _, row in processed.iterrows():
        output_path = dataset_dir / row["path"]
        assert output_path.is_file()
        with Image.open(output_path) as output:
            assert output.size == (256, 256)
            assert output.mode == "L"
        payload = json.loads(row["defect_params_json"])
        assert payload["baseline_preprocessing"]["working_dtype"] == "uint8"
        assert "steps" in payload
        assert payload["output_encoding"]["format"] == "JPEG"
    assert (data_dir / "logs" / "demo_defect_injection.log").is_file()
    assert (data_dir / "logs" / "defect_examples_demo.png").is_file()
# Append these tests to tests/test_defects.py
#
# They cover the RGBA handling added for ChestX-ray14, where 23 of 5,606 images
# in the official sample release are stored as 4-channel PNGs despite being
# radiographs.

def test_fully_opaque_rgba_is_flattened_losslessly() -> None:
    """The real ChestX-ray14 case: RGBA whose alpha is entirely opaque."""
    base = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
    opaque = np.full((64, 64), 255, dtype=np.uint8)
    source = Image.fromarray(np.dstack([base, base, base, opaque]), mode="RGBA")

    flattened, info = inject_defects_module.flatten_alpha_channel(source)

    assert info["strategy"] == "dropped_fully_opaque_alpha"
    assert info["fully_opaque"] is True
    # Identical RGB channels collapse to grayscale so the dataset stays uniform.
    assert flattened.mode == "L"
    assert info["collapsed_identical_rgb_to_grayscale"] is True
    assert np.array_equal(np.asarray(flattened), base)


def test_transparent_rgba_is_composited_onto_black() -> None:
    """Transparent pixels must not become bright regions a metric reads as signal."""
    alpha = np.full((64, 64), 255, dtype=np.uint8)
    alpha[:32, :] = 0
    value = np.full((64, 64), 200, dtype=np.uint8)
    source = Image.fromarray(
        np.dstack([value, value, value, alpha]), mode="RGBA"
    )

    flattened, info = inject_defects_module.flatten_alpha_channel(source)
    pixels = np.asarray(flattened)

    assert info["strategy"] == "composited_onto_black"
    assert info["fully_opaque"] is False
    assert pixels[:32].max() == 0
    assert pixels[32:].min() == 200


def test_genuinely_coloured_rgba_stays_rgb() -> None:
    """Only redundant identical channels collapse; real colour is preserved."""
    base = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
    source = Image.fromarray(
        np.dstack(
            [
                base,
                np.roll(base, 5, axis=0),
                np.roll(base, 9, axis=1),
                np.full((64, 64), 255, dtype=np.uint8),
            ]
        ),
        mode="RGBA",
    )

    flattened, info = inject_defects_module.flatten_alpha_channel(source)

    assert flattened.mode == "RGB"
    assert info["collapsed_identical_rgb_to_grayscale"] is False


def test_images_without_alpha_are_returned_untouched() -> None:
    base = np.zeros((16, 16), dtype=np.uint8)
    source = Image.fromarray(base, mode="L")

    flattened, info = inject_defects_module.flatten_alpha_channel(source)

    assert flattened is source
    assert info is None


def test_cli_indexes_rgba_sources_and_records_provenance(tmp_path: Path) -> None:
    """An RGBA source must be processed, not skipped, and say so in metadata."""
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir(parents=True)

    rows = []
    base = np.linspace(0, 255, 48 * 48, dtype=np.uint8).reshape(48, 48)
    opaque = np.full((48, 48), 255, dtype=np.uint8)
    for index, split in enumerate(("train", "train", "val", "val", "test", "test")):
        image_id = f"example-{index}"
        if index % 2:
            Image.fromarray(
                np.dstack([base, base, base, opaque]), mode="RGBA"
            ).save(raw_dir / f"{image_id}.png")
        else:
            Image.fromarray(base, mode="L").save(raw_dir / f"{image_id}.png")
        rows.append(
            {"image_id": image_id, "path": f"raw/{image_id}.png", "split": split}
        )
    pd.DataFrame(rows).to_csv(dataset_dir / "metadata.csv", index=False)

    inject_main(
        [
            "--dataset", "demo",
            "--data-dir", str(data_dir),
            "--defects", "gaussian_noise",
            "--severities", "1",
            "--fraction-clean", "0.5",
            "--canonical-size", "64",
            "--seed", "7",
        ]
    )

    processed = pd.read_csv(dataset_dir / "processed_metadata.csv", dtype=str)
    # Nothing was dropped for being RGBA.
    assert len(processed) == len(rows)

    # Every output has the same channel count, whatever the source mode was.
    for row in processed.itertuples(index=False):
        with Image.open(dataset_dir / row.path) as output:
            assert output.mode == "L"

    # The flattening decision is recorded for the RGBA sources only.
    flattened_rows = [
        row
        for row in processed.itertuples(index=False)
        if "alpha_flattening" in row.defect_params_json
    ]
    assert len(flattened_rows) == 3