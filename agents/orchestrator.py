"""Sequential orchestration of the three agents, with an auditable run manifest.

Inspector -> Summary -> Explainer, each consuming the previous agent's structured
output. Deliberately a plain sequential pipeline rather than an agent framework:
the dependency graph is a straight line, the handoff is a JSON file with a
versioned schema, and an explicit script is easier to defend in a viva than a
framework whose scheduling behaviour has to be explained before the research can
be.

Three properties that make this more than a shell script:

  * **Partial failure is survivable and visible.** If the Explainer's local model
    is unreachable, the Inspector and Summary results are still written and the
    manifest records which stage failed and why. Losing an hour of scoring
    because a model server was down would be an avoidable own goal.
  * **Every stage records provenance.** The manifest carries each agent's run_id,
    duration, output paths and configuration, so any number in the final report
    can be traced to the parameters and library versions that produced it.
  * **The schema contract is enforced at each boundary**, so a stale output from
    an earlier schema fails loudly rather than being silently misread.

The Explainer is treated as optional by default. It depends on an external model
server, and a failure there says nothing about the quality analysis - so the
pipeline reports a partial success rather than discarding everything upstream.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from agents.base import (
        AgentProvenance,
        SchemaMismatchError,
        library_versions,
        write_agent_output,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.base import (  # type: ignore[no-redef]
        AgentProvenance,
        SchemaMismatchError,
        library_versions,
        write_agent_output,
    )


# Stages that must succeed for the pipeline to be considered successful. The
# Explainer is excluded because it depends on an external model server whose
# availability is unrelated to whether the quality analysis worked.
REQUIRED_STAGES = ("inspector", "summary")


@dataclass
class StageResult:
    """What happened when one stage ran."""

    name: str
    status: str  # "ok", "failed", or "skipped"
    duration_seconds: float = 0.0
    run_id: str | None = None
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.name,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.run_id:
            payload["run_id"] = self.run_id
        if self.outputs:
            payload["outputs"] = self.outputs
        if self.error:
            payload["error"] = self.error
            payload["error_type"] = self.error_type
        if self.detail:
            payload["detail"] = self.detail
        return payload


class Orchestrator:
    """Runs the agent chain in order, recording what happened at each step."""

    name = "orchestrator"

    def __init__(
        self,
        dataset: str,
        required_stages: tuple[str, ...] = REQUIRED_STAGES,
        stop_on_failure: bool = True,
    ) -> None:
        self.dataset = dataset
        self.required_stages = required_stages
        # A downstream stage cannot run without its input, so by default a
        # failure stops the chain. Set False to attempt every stage and collect
        # all errors at once, which is useful when debugging a fresh setup.
        self.stop_on_failure = stop_on_failure
        self.results: list[StageResult] = []
        self.started_at = time.time()

    def run_stage(
        self,
        name: str,
        action: Callable[[], dict[str, Any]],
        optional: bool = False,
    ) -> StageResult:
        """Run one stage, capturing timing, provenance and any failure.

        A stage returns a dict describing what it produced; anything it raises is
        recorded rather than propagated, so an early failure does not discard the
        outputs of stages that already succeeded.
        """
        if self.stop_on_failure and self._blocked_by_failure(name):
            result = StageResult(
                name=name, status="skipped",
                error="an earlier required stage failed, so its input is missing",
                error_type="SkippedDependency",
            )
            self.results.append(result)
            return result

        started = time.perf_counter()
        try:
            detail = action()
            result = StageResult(
                name=name,
                status="ok",
                duration_seconds=time.perf_counter() - started,
                run_id=detail.pop("run_id", None),
                outputs=[str(path) for path in detail.pop("outputs", [])],
                detail=detail,
            )
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            result = StageResult(
                name=name,
                status="failed",
                duration_seconds=time.perf_counter() - started,
                error=str(error),
                error_type=type(error).__name__,
                detail={
                    "optional": optional,
                    # Kept because a schema mismatch and a missing model server
                    # need very different responses, and the type alone does not
                    # always make that clear from a log.
                    "traceback_tail": traceback.format_exc().strip().split("\n")[-1],
                },
            )
        self.results.append(result)
        return result

    def _blocked_by_failure(self, name: str) -> bool:
        """Whether an earlier required stage failed, making this one impossible."""
        for result in self.results:
            if result.status == "failed" and result.name in self.required_stages:
                return True
        return False

    @property
    def succeeded(self) -> bool:
        """True when every required stage completed."""
        by_name = {result.name: result.status for result in self.results}
        return all(
            by_name.get(stage) == "ok" for stage in self.required_stages
        )

    def manifest(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """A complete record of the run, sufficient to reproduce or audit it."""
        provenance = AgentProvenance(
            agent=self.name,
            config=dict(config or {}),
            inputs={"dataset": self.dataset},
        )
        failed = [r.name for r in self.results if r.status == "failed"]
        skipped = [r.name for r in self.results if r.status == "skipped"]
        return {
            "provenance": provenance.to_dict(),
            "dataset": self.dataset,
            "pipeline_succeeded": self.succeeded,
            "required_stages": list(self.required_stages),
            "failed_stages": failed,
            "skipped_stages": skipped,
            # Summed across stages rather than measured from construction, so
            # the figure reflects pipeline work rather than however long the
            # caller spent loading data before the first stage began.
            "total_duration_seconds": round(
                sum(result.duration_seconds for result in self.results), 3
            ),
            "wall_clock_seconds": round(time.time() - self.started_at, 3),
            "stages": [result.to_dict() for result in self.results],
            "libraries": library_versions(),
            "note": (
                "Stages run in order, each consuming the previous stage's JSON "
                "output. The explainer is optional: it depends on an external "
                "model server, so its failure does not invalidate the quality "
                "analysis and the pipeline still reports partial success."
            ),
        }

    def write_manifest(self, path: Path, config: dict[str, Any] | None = None) -> None:
        write_agent_output(self.manifest(config), path)


def summarise_manifest(manifest: dict[str, Any]) -> list[str]:
    """Human-readable lines describing a completed run, for logging."""
    lines: list[str] = []
    for stage in manifest["stages"]:
        marker = {"ok": " ok ", "failed": "FAIL", "skipped": "skip"}.get(
            stage["status"], "????"
        )
        line = f"  [{marker}] {stage['stage']:<12} {stage['duration_seconds']:>7.2f}s"
        if stage.get("run_id"):
            line += f"  run {stage['run_id']}"
        lines.append(line)
        if stage.get("error"):
            lines.append(f"         {stage['error_type']}: {stage['error']}")
        for output in stage.get("outputs", []):
            lines.append(f"         -> {output}")
    verdict = (
        "pipeline completed"
        if manifest["pipeline_succeeded"]
        else "pipeline FAILED on a required stage"
    )
    lines.append("")
    lines.append(
        f"  {verdict} in {manifest['total_duration_seconds']:.1f}s"
    )
    if manifest["failed_stages"]:
        lines.append(f"  failed: {', '.join(manifest['failed_stages'])}")
    if manifest["skipped_stages"]:
        lines.append(f"  skipped: {', '.join(manifest['skipped_stages'])}")
    return lines