"""Shared interface and output contract for every agent in the pipeline.

Three agents run in sequence - Inspector, Summary, Explainer - each consuming
the previous one's structured output. That handoff is the whole point of the
architecture, so the output contract is defined here once rather than being
re-invented per agent.

**The schema version is frozen at the end of Stage 7, before the Summary Agent
begins consuming it.** Schema churn across three dependent agents is the most
likely way this part of the project overruns, so a consumer that finds an
unexpected major version fails loudly rather than silently misreading fields.

Every agent output carries a provenance block recording what produced it: agent
name, schema version, run identifier, timestamp, and the complete configuration.
This exists so that a decision in a report can be traced back to the exact
thresholds and model versions that produced it, without re-running anything -
which is the difference between an auditable system and one that merely emits
numbers.
"""

from __future__ import annotations

import json
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Major version bumps break consumers; minor versions add optional fields only.
SCHEMA_VERSION = "1.0"


def schema_major(version: str) -> int:
    return int(version.split(".", 1)[0])


class SchemaMismatchError(RuntimeError):
    """Raised when a consumer is handed output from an incompatible schema."""


def require_compatible_schema(payload: dict[str, Any], consumer: str) -> None:
    """Fail loudly when an agent is handed output it cannot safely read.

    Silently proceeding on a mismatched schema produces a report that looks
    complete while having dropped or misread fields, which is worse than an
    error - the whole point of the structured handoff is that downstream
    agents can trust the shape of what they receive.
    """
    found = payload.get("provenance", {}).get("schema_version")
    if found is None:
        raise SchemaMismatchError(
            f"{consumer} received a payload with no provenance.schema_version; "
            "it was probably not produced by an agent in this pipeline"
        )
    if schema_major(found) != schema_major(SCHEMA_VERSION):
        raise SchemaMismatchError(
            f"{consumer} expects schema major version "
            f"{schema_major(SCHEMA_VERSION)} but received {found}. The agent "
            "output must be regenerated."
        )


def library_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour affects agent decisions."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "scipy", "torch", "timm", "PIL"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


@dataclass
class AgentProvenance:
    """What produced an agent output, recorded so decisions can be traced."""

    agent: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: str = SCHEMA_VERSION
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    libraries: dict[str, str] = field(default_factory=library_versions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "config": self.config,
            "inputs": self.inputs,
            "libraries": self.libraries,
        }


class Agent:
    """Base class: a named component that produces a provenance-carrying dict."""

    name: str = "agent"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def provenance(self, inputs: dict[str, Any] | None = None) -> AgentProvenance:
        return AgentProvenance(
            agent=self.name, config=self.config, inputs=dict(inputs or {})
        )

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


def write_agent_output(payload: dict[str, Any], path: Path) -> None:
    """Write an agent payload as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def read_agent_output(path: Path, consumer: str) -> dict[str, Any]:
    """Read a prior agent's output, checking schema compatibility first."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{consumer} needs {path}, which does not exist; run the upstream "
            "agent first"
        )
    payload = json.loads(path.read_text())
    require_compatible_schema(payload, consumer)
    return payload