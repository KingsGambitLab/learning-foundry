"""Oracle pass: boot the reference impl, capture per-scenario ground truth.

The oracle pass is the second step in the simplified single-outcome
course pipeline (see
``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``).
It receives the parsed scenarios and a path to the materialized
reference implementation bundle, boots the impl in a sandbox (Docker
in production, a fake in tests), runs every scenario against it via
:func:`app.services.scenario_trace_runner.run_scenario`, and returns
an :class:`OraclePassResult` carrying the captured outputs + the
oracle's own rubric verdicts.

That artifact (and its on-disk JSON form, persisted via
:func:`persist_oracle_outputs`) is the **interface contract** the
downstream ``oracle_validation`` node and the at-grade-time grader
runner both read. Changing the shape requires cross-coordination.

Design notes
------------

* **Sandbox protocol is duck-typed.** The injected ``sandbox_runner``
  only needs ``boot(reference_impl_dir) -> handle`` and
  ``teardown(handle)``. The real Docker runner lives in
  ``docker_sandbox_runner`` and exposes a richer surface; we intentionally
  don't bind to its specific class so tests can substitute a
  ``FakeSandboxRunner`` and so the eventual production wiring has
  freedom to choose any compatible adapter. ``handle`` only needs a
  ``base_url: str`` attribute.

* **Hashes are stable cache keys.** ``reference_impl_hash`` walks the
  reference bundle in sorted-path order, hashing path + content of
  every regular file. ``scenario_set_hash`` hashes the canonical YAML
  serialization of each scenario (sorted by id, ``sort_keys=True``)
  so re-ordering scenarios doesn't change the key.

* **Per-scenario isolation.** A broken scenario (HTTP raises, rubric
  raises, etc.) does NOT abort the whole pass — it produces an
  ``aborted=True`` :class:`OracleScenarioOutput` with the exception
  message and the pass continues. Only a sandbox-level failure stops
  everything, and even then the ``finally`` block guarantees teardown.

* **Persistence is a separate helper.** ``run`` returns a Pydantic
  model; the caller decides whether / where to write it. The standard
  destination is ``private/grader/_oracle/outputs.json``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.services.scenario_loader import Scenario
from app.services.scenario_trace_runner import (
    ScenarioVerdictReport,
    run_scenario,
)


# ---------------- Public Pydantic models ----------------


class OracleScenarioOutput(BaseModel):
    """Per-scenario ground truth captured from the reference impl."""

    scenario_id: str
    category: str
    captures: dict[str, Any] = Field(default_factory=dict)
    """The trace runner's captures dict from the ref impl run.

    Keys are step ids (and capture aliases). Each value is
    ``{"status": int, "headers": dict, "body": Any}``.
    """

    verdicts: list[tuple[str, dict[str, Any]]] = Field(default_factory=list)
    """Serialized ``[(rubric_kind, Verdict.model_dump()), ...]``.

    Kept as a plain list-of-tuples so on-disk JSON round-trips cleanly
    without depending on the ``Verdict`` model at read time.
    """

    aborted: bool = False
    abort_reason: str | None = None


class OraclePassResult(BaseModel):
    """The full ground-truth artifact persisted to disk.

    This is the **stable interface contract** with ``oracle_validation``
    (the validation step that ensures the oracle outputs themselves
    are internally consistent) and with the on-disk grader runner. Any
    breaking change here needs cross-coordination.
    """

    reference_impl_hash: str
    """sha256 of the sorted-by-path file contents of the reference impl dir."""

    scenario_set_hash: str
    """sha256 of the canonical YAML serialization of every scenario, sorted by id."""

    generated_at: str
    """ISO-8601 UTC timestamp of when the pass was produced."""

    scenario_outputs: list[OracleScenarioOutput] = Field(default_factory=list)
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    abstained_scenarios: int


# ---------------- Helpers ----------------


def _hash_reference_impl(reference_impl_dir: Path) -> str:
    """SHA-256 of the reference bundle.

    Walks every regular file under ``reference_impl_dir`` in sorted
    relative-path order, hashing path-then-bytes per file. Sort-order
    is deterministic across platforms because we use POSIX-style
    relative paths.
    """
    hasher = hashlib.sha256()
    root = Path(reference_impl_dir)
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    for f in files:
        rel = f.relative_to(root).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(f.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _hash_scenario_set(scenarios: list[Scenario]) -> str:
    """SHA-256 of the canonical YAML form of the scenario set.

    Scenarios are sorted by id before serialization so re-ordering the
    input list doesn't change the hash. ``yaml.safe_dump(sort_keys=True)``
    plus ``model_dump(mode='json')`` give a canonical text form that
    survives Pydantic round-trips.
    """
    hasher = hashlib.sha256()
    for scenario in sorted(scenarios, key=lambda s: s.id):
        payload = scenario.model_dump(mode="json")
        text = yaml.safe_dump(payload, sort_keys=True)
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _load_setup_data(setup_data_dir: Path | None) -> dict[str, Any]:
    """Walk a setup-data directory: JSON / JSONL parsed, everything else raw.

    Keys are file *stems* so ``gold.json`` and ``gold.txt`` would
    collide on purpose (authors should keep stems unique).

    JSONL handling (Bug 26): each line is parsed as a separate JSON
    object. The result is a LIST of those objects keyed by file stem
    — matching how callers index CRAG-shaped data
    (``setup_data.queries[0].query``). If any line fails to parse,
    the whole file falls back to raw text (so the operator can see
    what went wrong instead of a silent empty list).
    """
    if setup_data_dir is None:
        return {}
    result: dict[str, Any] = {}
    for f in sorted(Path(setup_data_dir).iterdir()):
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix == ".json":
            try:
                result[f.stem] = json.loads(f.read_text())
            except (ValueError, json.JSONDecodeError):
                result[f.stem] = f.read_text()
        elif suffix == ".jsonl":
            raw_text = f.read_text()
            try:
                lines = [
                    json.loads(line)
                    for line in raw_text.splitlines()
                    if line.strip()
                ]
            except (ValueError, json.JSONDecodeError):
                result[f.stem] = raw_text
            else:
                result[f.stem] = lines
        else:
            result[f.stem] = f.read_text()
    return result


def _aggregate_counts(reports: list[ScenarioVerdictReport]) -> tuple[int, int, int]:
    passed = sum(1 for r in reports if r.overall_status == "pass")
    failed = sum(1 for r in reports if r.overall_status == "fail")
    abstained = sum(1 for r in reports if r.overall_status == "abstain")
    return passed, failed, abstained


# ---------------- OraclePass ----------------


class OraclePass:
    """Orchestrate the oracle ground-truth capture.

    Parameters
    ----------
    sandbox_runner:
        Object exposing ``boot(reference_impl_dir) -> handle`` and
        ``teardown(handle)``. The handle must carry a ``base_url:
        str`` attribute pointing at the booted reference impl. In
        production this is the Docker sandbox runner; tests inject a
        ``FakeSandboxRunner``.
    http_client:
        Optional pre-built :class:`ScenarioHttpClient`. Defaults to
        the trace runner's own ``UrllibHttpClient`` when ``None``.
    timeout_per_scenario_s:
        Forwarded as the trace runner's ``timeout`` argument. Per-step
        HTTP requests inherit it. We deliberately do NOT add an
        outer-loop wall-clock kill switch in v1: the trace runner's
        own timeout is sufficient for typical scenarios, and signal /
        thread-based scenario kills add complexity that's not yet
        justified.
    """

    def __init__(
        self,
        *,
        sandbox_runner: Any,
        http_client: Any = None,
        timeout_per_scenario_s: float = 60.0,
    ) -> None:
        self.sandbox_runner = sandbox_runner
        self.http_client = http_client
        self.timeout_per_scenario_s = timeout_per_scenario_s

    def _boot_with_capabilities(
        self, reference_impl_dir: Path, capabilities: Any
    ) -> Any:
        """Call ``sandbox_runner.boot`` with or without ``capabilities``.

        The sandbox protocol is duck-typed. Production adapters (the
        ``WorkspaceBootSandboxAdapter``) accept ``capabilities=...`` as
        a keyword. Legacy fakes / test doubles only accept the positional
        directory. Falling back on ``TypeError`` keeps both shapes
        working without forcing every test fake to grow the new kwarg.
        """
        if capabilities is None:
            return self.sandbox_runner.boot(reference_impl_dir)
        try:
            return self.sandbox_runner.boot(
                reference_impl_dir, capabilities=capabilities
            )
        except TypeError:
            # Legacy adapter without the capability kwarg — fall back to
            # the original signature. The capability requirement is then
            # only enforced at the verifier layer.
            return self.sandbox_runner.boot(reference_impl_dir)

    def run(
        self,
        *,
        scenarios: list[Scenario],
        reference_impl_dir: Path,
        setup_data_dir: Path | None = None,
        course_meta: dict[str, Any] | None = None,
        router: Any = None,
        capabilities: Any = None,
    ) -> OraclePassResult:
        """Boot the reference impl, run every scenario, return the result.

        ``capabilities`` (a ``CourseOutcomeSpec.capabilities`` /
        ``CapabilityFlags``) threads through to the sandbox runner's
        ``boot`` call. Sandbox adapters that accept a ``capabilities``
        kwarg (e.g. ``WorkspaceBootSandboxAdapter``) provision the
        requested primitives (or fail loud) before starting the
        reference impl container. Adapters that don't yet accept the
        kwarg fall back to the legacy ``boot(dir)`` signature so existing
        FakeSandboxRunner test doubles keep working unchanged.
        """
        reference_impl_hash = _hash_reference_impl(reference_impl_dir)
        scenario_set_hash = _hash_scenario_set(scenarios)
        setup_data = _load_setup_data(setup_data_dir)

        scenario_outputs: list[OracleScenarioOutput] = []
        verdict_reports: list[ScenarioVerdictReport] = []

        handle = self._boot_with_capabilities(reference_impl_dir, capabilities)
        try:
            base_url = handle.base_url
            for scenario in scenarios:
                try:
                    report = run_scenario(
                        scenario=scenario,
                        base_url=base_url,
                        router=router,
                        setup_data=setup_data,
                        course_meta=course_meta or {},
                        http_client=self.http_client,
                        timeout=self.timeout_per_scenario_s,
                    )
                except Exception as exc:
                    # Per-scenario isolation: record the failure, keep going.
                    scenario_outputs.append(
                        OracleScenarioOutput(
                            scenario_id=scenario.id,
                            category=scenario.category,
                            captures={},
                            verdicts=[],
                            aborted=True,
                            abort_reason=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue

                verdict_reports.append(report)
                scenario_outputs.append(
                    OracleScenarioOutput(
                        scenario_id=report.scenario_id,
                        category=report.category,
                        captures=report.run_result.captures,
                        verdicts=[
                            (kind, verdict.model_dump())
                            for kind, verdict in report.verdicts
                        ],
                        aborted=report.run_result.aborted,
                        abort_reason=report.run_result.abort_reason,
                    )
                )
        finally:
            self.sandbox_runner.teardown(handle)

        passed, failed, abstained = _aggregate_counts(verdict_reports)

        return OraclePassResult(
            reference_impl_hash=reference_impl_hash,
            scenario_set_hash=scenario_set_hash,
            generated_at=datetime.now(UTC).isoformat(),
            scenario_outputs=scenario_outputs,
            total_scenarios=len(scenarios),
            passed_scenarios=passed,
            failed_scenarios=failed,
            abstained_scenarios=abstained,
        )


# ---------------- Persistence ----------------


def persist_oracle_outputs(result: OraclePassResult, output_path: Path) -> None:
    """Write ``result`` to ``output_path`` as pretty-printed JSON.

    Parent directories are created. Existing files are overwritten.
    The on-disk format is exactly ``OraclePassResult.model_dump(mode='json')``,
    so ``OraclePassResult.model_validate(json.loads(text))`` round-trips
    cleanly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


__all__ = [
    "OraclePass",
    "OraclePassResult",
    "OracleScenarioOutput",
    "persist_oracle_outputs",
]
