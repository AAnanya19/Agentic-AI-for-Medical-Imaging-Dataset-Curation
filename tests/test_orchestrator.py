"""Tests for the orchestrator and the end-to-end pipeline.

Four invariants:

1. **An optional stage's failure must not discard upstream work.** The Explainer
   depends on an external model server; losing an entire scoring run because that
   server was down would be an avoidable own goal, and its availability says
   nothing about whether the quality analysis worked.

2. **A required stage's failure must block its dependents.** The Summary Agent
   cannot run without the Inspector's JSON, so attempting it would produce a
   confusing FileNotFoundError instead of a clear dependency message.

3. **The manifest must be sufficient to audit the run.** Every stage's run_id,
   duration, outputs, configuration and library versions, so any number in the
   final report can be traced to what produced it.

4. **The exit code must distinguish a broken pipeline from a missing model.**
   A calling script needs to know the difference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.orchestrator import (
    REQUIRED_STAGES, Orchestrator, StageResult, summarise_manifest,
)


# ---------------------------------------------------------------------------
# Stage sequencing
# ---------------------------------------------------------------------------

def test_all_stages_succeeding_gives_a_successful_pipeline() -> None:
    orchestrator = Orchestrator("demo")
    orchestrator.run_stage("inspector", lambda: {"run_id": "a"})
    orchestrator.run_stage("summary", lambda: {"run_id": "b"})
    orchestrator.run_stage("explainer", lambda: {"run_id": "c"}, optional=True)
    assert orchestrator.succeeded is True
    assert orchestrator.manifest()["failed_stages"] == []


def test_an_optional_failure_leaves_the_pipeline_successful() -> None:
    """The core partial-success guarantee."""
    def unreachable() -> dict:
        raise RuntimeError("could not reach the local model")

    orchestrator = Orchestrator("demo")
    orchestrator.run_stage("inspector", lambda: {"run_id": "a"})
    orchestrator.run_stage("summary", lambda: {"run_id": "b"})
    orchestrator.run_stage("explainer", unreachable, optional=True)

    manifest = orchestrator.manifest()
    assert manifest["pipeline_succeeded"] is True
    assert manifest["failed_stages"] == ["explainer"]
    # And the successful stages are still recorded with their outputs.
    statuses = {stage["stage"]: stage["status"] for stage in manifest["stages"]}
    assert statuses["inspector"] == "ok"
    assert statuses["summary"] == "ok"


def test_a_required_failure_fails_the_pipeline() -> None:
    def broken() -> dict:
        raise ValueError("no clean validation rows")

    orchestrator = Orchestrator("demo")
    orchestrator.run_stage("inspector", broken)
    assert orchestrator.succeeded is False
    assert orchestrator.manifest()["failed_stages"] == ["inspector"]


def test_a_required_failure_skips_its_dependents_with_a_clear_reason() -> None:
    """Attempting the Summary without the Inspector's output would raise a
    confusing FileNotFoundError instead of naming the actual cause."""
    def broken() -> dict:
        raise ValueError("no clean validation rows")

    orchestrator = Orchestrator("demo")
    orchestrator.run_stage("inspector", broken)
    orchestrator.run_stage("summary", lambda: {"run_id": "b"})
    orchestrator.run_stage("explainer", lambda: {"run_id": "c"}, optional=True)

    manifest = orchestrator.manifest()
    assert manifest["skipped_stages"] == ["summary", "explainer"]
    skipped = [s for s in manifest["stages"] if s["status"] == "skipped"]
    assert all(s["error_type"] == "SkippedDependency" for s in skipped)


def test_continue_on_failure_attempts_every_stage() -> None:
    """Useful when debugging a fresh setup: collect all errors in one run."""
    def broken() -> dict:
        raise ValueError("first problem")

    def also_broken() -> dict:
        raise KeyError("second problem")

    orchestrator = Orchestrator("demo", stop_on_failure=False)
    orchestrator.run_stage("inspector", broken)
    orchestrator.run_stage("summary", also_broken)
    manifest = orchestrator.manifest()
    assert manifest["failed_stages"] == ["inspector", "summary"]
    assert manifest["skipped_stages"] == []


def test_an_optional_failure_does_not_block_later_stages() -> None:
    """Only required stages gate the chain."""
    def unreachable() -> dict:
        raise RuntimeError("model down")

    orchestrator = Orchestrator(
        "demo", required_stages=("inspector",), stop_on_failure=True
    )
    orchestrator.run_stage("explainer", unreachable, optional=True)
    orchestrator.run_stage("inspector", lambda: {"run_id": "a"})
    assert orchestrator.succeeded is True


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

def test_the_manifest_records_run_ids_and_outputs() -> None:
    orchestrator = Orchestrator("demo")
    orchestrator.run_stage(
        "inspector", lambda: {"run_id": "abc123", "outputs": [Path("a.json")]}
    )
    stage = orchestrator.manifest()["stages"][0]
    assert stage["run_id"] == "abc123"
    assert stage["outputs"] == ["a.json"]


def test_the_manifest_records_the_configuration_used() -> None:
    """Traceability: a number in the report must be attributable to parameters."""
    orchestrator = Orchestrator("demo")
    orchestrator.run_stage("inspector", lambda: {"run_id": "a"})
    manifest = orchestrator.manifest({"backbone": "resnet18", "fpr": 0.05})
    assert manifest["provenance"]["config"]["backbone"] == "resnet18"
    assert manifest["provenance"]["agent"] == "orchestrator"


def test_the_manifest_records_library_versions() -> None:
    orchestrator = Orchestrator("demo")
    orchestrator.run_stage("inspector", lambda: {"run_id": "a"})
    assert "numpy" in orchestrator.manifest()["libraries"]


def test_stage_detail_is_preserved_in_the_manifest() -> None:
    orchestrator = Orchestrator("demo")
    orchestrator.run_stage(
        "inspector", lambda: {"run_id": "a", "n_images": 844, "extra": "kept"}
    )
    detail = orchestrator.manifest()["stages"][0]["detail"]
    assert detail["n_images"] == 844
    assert detail["extra"] == "kept"


def test_duration_excludes_time_spent_before_the_first_stage() -> None:
    """Measured by summing stages rather than from construction, so a caller's
    data loading is not attributed to the pipeline."""
    orchestrator = Orchestrator("demo")
    time.sleep(0.2)
    orchestrator.run_stage("inspector", lambda: {"run_id": "a"})
    manifest = orchestrator.manifest()
    assert manifest["total_duration_seconds"] < 0.15
    assert manifest["wall_clock_seconds"] >= 0.2


def test_the_error_type_is_recorded_alongside_the_message() -> None:
    """A schema mismatch and a missing model server need different responses."""
    def broken() -> dict:
        raise FileNotFoundError("iqa_scores.csv not found")

    orchestrator = Orchestrator("demo")
    orchestrator.run_stage("inspector", broken)
    stage = orchestrator.manifest()["stages"][0]
    assert stage["error_type"] == "FileNotFoundError"
    assert "iqa_scores.csv" in stage["error"]


def test_the_manifest_is_json_serialisable(tmp_path: Path) -> None:
    orchestrator = Orchestrator("demo")
    orchestrator.run_stage(
        "inspector", lambda: {"run_id": "a", "outputs": [tmp_path / "x.json"]}
    )
    path = tmp_path / "manifest.json"
    orchestrator.write_manifest(path, {"backbone": "resnet18"})
    restored = json.loads(path.read_text())
    assert restored["dataset"] == "demo"


def test_the_summary_lines_name_the_failure() -> None:
    def broken() -> dict:
        raise RuntimeError("model down")

    # required_stages must name only the stages this test actually runs;
    # otherwise the never-run "summary" counts as incomplete.
    orchestrator = Orchestrator("demo", required_stages=("inspector",))
    orchestrator.run_stage("inspector", lambda: {"run_id": "a"})
    orchestrator.run_stage("explainer", broken, optional=True)
    text = "\n".join(summarise_manifest(orchestrator.manifest()))
    assert "FAIL" in text
    assert "model down" in text
    assert "pipeline completed" in text     # optional failure, still a success


def test_required_stages_are_the_two_analysis_agents() -> None:
    """The Explainer is deliberately excluded: it depends on an external server."""
    assert set(REQUIRED_STAGES) == {"inspector", "summary"}


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def _build_dataset(tmp_path: Path, rare_stratum: bool = True) -> Path:
    """A scored dataset the pipeline can run on, with no images needed."""
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "demo"
    dataset_dir.mkdir(parents=True)
    (data_dir / "logs").mkdir()

    rng = np.random.default_rng(5)
    rows = []
    source_rows = []
    metrics = [
        "laplacian_variance", "tenengrad", "noise_estimate", "rms_contrast",
        "michelson_contrast", "shannon_entropy", "brisque", "clip_iqa",
        "brightness_deviation",
    ]
    for split, n_clean, n_degraded in (("val", 120, 60), ("test", 80, 40)):
        for index in range(n_clean):
            image_id = f"{split}_clean_{index}"
            row = {
                "image_id": image_id, "original_image_id": f"src_{image_id}",
                "split": split, "defect_type": "clean", "severity_level": "0",
            }
            for name in metrics:
                row[name] = rng.normal(0, 1)
            rows.append(row)
            label = (
                "Hernia" if rare_stratum and index < 6 else "No Finding"
            )
            source_rows.append({"image_id": f"src_{image_id}", "source_labels": label})
        for index in range(n_degraded):
            image_id = f"{split}_blur_{index}"
            row = {
                "image_id": image_id, "original_image_id": f"src_{image_id}",
                "split": split, "defect_type": "gaussian_blur",
                "severity_level": str(1 + index % 3),
            }
            for name in metrics:
                row[name] = rng.normal(3, 1)
            rows.append(row)
            source_rows.append({
                "image_id": f"src_{image_id}", "source_labels": "No Finding",
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(dataset_dir / "iqa_scores.csv", index=False)
    pd.DataFrame(source_rows).to_csv(dataset_dir / "metadata.csv", index=False)

    processed = frame[[
        "image_id", "original_image_id", "split", "defect_type", "severity_level",
    ]].copy()
    processed["path"] = "processed/" + processed["image_id"] + ".jpg"
    processed["canonical_size"] = "256"
    processed["defect_params_json"] = json.dumps({
        "baseline_preprocessing": {
            "canonical_resize": {"original_width": 1024, "original_height": 1024}
        }
    })
    processed.to_csv(dataset_dir / "processed_metadata.csv", index=False)
    return data_dir


def test_the_pipeline_runs_end_to_end_and_writes_every_output(tmp_path: Path) -> None:
    from scripts.run_pipeline import main as pipeline_main

    data_dir = _build_dataset(tmp_path)
    exit_code = pipeline_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--label-column", "source_labels",
    ])
    assert exit_code == 0

    logs = data_dir / "logs"
    for name in (
        "demo_inspection.json", "demo_summary.json",
        "demo_explainer_template.json", "demo_audit_report_template.md",
        "demo_pipeline_manifest.json",
    ):
        assert (logs / name).is_file(), name


def test_the_pipeline_manifest_links_the_three_stages(tmp_path: Path) -> None:
    """Each stage's run_id must be recorded, so the chain is traceable."""
    from scripts.run_pipeline import main as pipeline_main

    data_dir = _build_dataset(tmp_path)
    pipeline_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--label-column", "source_labels",
    ])
    manifest = json.loads(
        (data_dir / "logs" / "demo_pipeline_manifest.json").read_text()
    )
    assert manifest["pipeline_succeeded"] is True
    stages = {stage["stage"]: stage for stage in manifest["stages"]}
    assert set(stages) == {"inspector", "summary", "explainer"}
    assert all(stages[name].get("run_id") for name in stages)


def test_an_unreachable_model_still_yields_a_zero_exit_code(tmp_path: Path) -> None:
    """A missing model server must not make the run look broken to a caller."""
    from scripts.run_pipeline import main as pipeline_main

    data_dir = _build_dataset(tmp_path)
    exit_code = pipeline_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--label-column", "source_labels",
        "--explainer-backend", "ollama",
        "--explainer-model", "definitely-not-pulled",
        "--explainer-timeout", "2",
    ])
    assert exit_code == 0

    manifest = json.loads(
        (data_dir / "logs" / "demo_pipeline_manifest.json").read_text()
    )
    assert manifest["pipeline_succeeded"] is True
    assert manifest["failed_stages"] == ["explainer"]
    # Upstream outputs survive the downstream failure.
    assert (data_dir / "logs" / "demo_inspection.json").is_file()
    assert (data_dir / "logs" / "demo_summary.json").is_file()


def test_a_missing_score_file_fails_the_pipeline(tmp_path: Path) -> None:
    from scripts.run_pipeline import main as pipeline_main

    data_dir = _build_dataset(tmp_path)
    (data_dir / "demo" / "iqa_scores.csv").unlink()
    exit_code = pipeline_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--label-column", "source_labels",
    ])
    assert exit_code == 1

    manifest = json.loads(
        (data_dir / "logs" / "demo_pipeline_manifest.json").read_text()
    )
    assert manifest["pipeline_succeeded"] is False
    assert manifest["failed_stages"] == ["inspector"]
    assert manifest["skipped_stages"] == ["summary", "explainer"]


def test_skipping_the_explainer_still_succeeds(tmp_path: Path) -> None:
    from scripts.run_pipeline import main as pipeline_main

    data_dir = _build_dataset(tmp_path)
    exit_code = pipeline_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--label-column", "source_labels", "--skip-explainer",
    ])
    assert exit_code == 0
    manifest = json.loads(
        (data_dir / "logs" / "demo_pipeline_manifest.json").read_text()
    )
    assert {s["stage"] for s in manifest["stages"]} == {"inspector", "summary"}


def test_the_pipeline_surfaces_a_stratum_collapse(tmp_path: Path) -> None:
    """The substantive finding must survive the full chain, not just the agent."""
    from scripts.run_pipeline import main as pipeline_main

    data_dir = _build_dataset(tmp_path, rare_stratum=True)
    pipeline_main([
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--label-column", "source_labels", "--min-viable-stratum", "10",
    ])
    summary = json.loads((data_dir / "logs" / "demo_summary.json").read_text())
    balance = summary["class_balance_impact"]
    assert "skipped" not in balance
    assert balance["n_strata"] >= 2


def test_the_pipeline_is_reproducible(tmp_path: Path) -> None:
    """Same inputs must give the same decisions; only run_ids and timings differ."""
    from scripts.run_pipeline import main as pipeline_main

    data_dir = _build_dataset(tmp_path)
    arguments = [
        "--dataset", "demo", "--data-dir", str(data_dir),
        "--label-column", "source_labels", "--skip-explainer",
    ]
    pipeline_main(arguments)
    first = json.loads((data_dir / "logs" / "demo_inspection.json").read_text())
    pipeline_main(arguments)
    second = json.loads((data_dir / "logs" / "demo_inspection.json").read_text())

    assert first["decision_counts"] == second["decision_counts"]
    assert first["thresholds"] == second["thresholds"]
    assert [i["decision"] for i in first["images"]] == [
        i["decision"] for i in second["images"]
    ]