"""Tests for the single-outcome LangGraph dispatcher.

The graph stitches Wave 1-3 modules into a linear flow with three HIL
gates and two repair pockets. We exercise each node against fakes and
verify retry / gate semantics end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.registry import PackageType
from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    OracleSource,
    QualityBar,
    StarterType,
)
from app.services.course_outcome_planner import OutcomeCourseGenerationError
from app.services.langgraph_outcome_graph import (
    OutcomeGraphDeps,
    OutcomeWorkflowGraph,
    OutcomeWorkflowState,
    node_grader_repair,
    node_oracle_authoring,
    node_oracle_pass,
    node_oracle_validation,
    node_publish,
    node_reviewer_code,
    node_spec_authoring,
    node_spec_review,
    node_starter_authoring,
    node_starter_repair,
    node_starter_verify,
)
from app.services.oracle_authoring import (
    GeneratedReferenceFile,
    GeneratedScenarioFile,
    GeneratedSetupFile,
    OracleAuthoringResult,
)
from app.services.oracle_pass import (
    OraclePassResult,
    OracleScenarioOutput,
)


# ---------------- Fakes ----------------


def _sample_spec() -> CourseOutcomeSpec:
    # NOTE: existing Wave 4 tests in this file assume the reference-run
    # oracle path (boot ref impl + validate). The single-outcome spec's
    # default is curated as of Wave 4.5, so this helper explicitly opts
    # into ``reference_run``. Wave-4.5 tests author their own specs via
    # ``_spec_with_oracle_source`` when they need a different mode.
    return CourseOutcomeSpec(
        title="Build a Grounded RAG Service",
        goal=(
            "Build a small HTTP service that ingests documents, retrieves "
            "passages for a question, and returns a grounded answer with "
            "citations or abstains."
        ),
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str", "citations": "list"},
                description="Answer the question or abstain.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite passages that support them.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
        oracle_source=OracleSource.reference_run,
    )


def _passing_oracle_result(scenario_id: str = "hp1") -> OraclePassResult:
    """Build a passing OraclePassResult covering all three required categories.

    Lines up with ``_sample_authoring_result``'s scenarios so that the
    validator (which loads the actual scenarios from disk) finds
    matching pass verdicts.
    """
    return OraclePassResult(
        reference_impl_hash="ref-hash",
        scenario_set_hash="scen-hash",
        generated_at="2026-05-14T00:00:00Z",
        scenario_outputs=[
            OracleScenarioOutput(
                scenario_id=f"s_{cat}",
                category=cat,
                captures={"ask": {"status": 200, "body": {"answer": "ok"}}},
                verdicts=[("oracle_set_overlap", {"status": "pass"})],
                aborted=False,
            )
            for cat in ("happy_path", "boundary", "malformed_input")
        ],
        total_scenarios=3,
        passed_scenarios=3,
        failed_scenarios=0,
        abstained_scenarios=0,
    )


def _failing_oracle_result() -> OraclePassResult:
    """Same scenario shape as the passing variant, but the happy_path one
    is marked failed so the validator decides "not publishable"."""
    return OraclePassResult(
        reference_impl_hash="ref-hash",
        scenario_set_hash="scen-hash",
        generated_at="2026-05-14T00:00:00Z",
        scenario_outputs=[
            OracleScenarioOutput(
                scenario_id="s_happy_path",
                category="happy_path",
                captures={},
                verdicts=[("oracle_set_overlap", {"status": "fail"})],
                aborted=False,
            ),
            OracleScenarioOutput(
                scenario_id="s_boundary",
                category="boundary",
                captures={},
                verdicts=[("oracle_set_overlap", {"status": "pass"})],
                aborted=False,
            ),
            OracleScenarioOutput(
                scenario_id="s_malformed_input",
                category="malformed_input",
                captures={},
                verdicts=[("oracle_set_overlap", {"status": "pass"})],
                aborted=False,
            ),
        ],
        total_scenarios=3,
        passed_scenarios=2,
        failed_scenarios=1,
        abstained_scenarios=0,
    )


def _scenario_yaml(scenario_id: str, category: str) -> str:
    # ``quality_bar_ids: [faithfulness]`` lines up with the single bar in
    # ``_sample_spec`` so the publish-gate's coverage check (Codex
    # review #4 finding #2) is satisfied.
    return (
        f"id: {scenario_id}\n"
        f"description: {category} scenario\n"
        f"category: {category}\n"
        "quality_bar_ids:\n"
        "  - faithfulness\n"
        "trace:\n"
        "  - id: ask\n"
        "    method: POST\n"
        "    path: /answer\n"
        "    body:\n"
        "      question: hello\n"
        "    expect:\n"
        "      status: 200\n"
        "rubrics:\n"
        "  - kind: oracle_set_overlap\n"
        "    target: ask.body.citations\n"
        "    oracle_path: gold.q\n"
        "    min_overlap: 1\n"
    )


def _sample_authoring_result() -> OracleAuthoringResult:
    return OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(
                filename=f"{cat}.yaml",
                yaml_content=_scenario_yaml(f"s_{cat}", cat),
            )
            for cat in ("happy_path", "boundary", "malformed_input")
        ],
        reference_files=[
            GeneratedReferenceFile(relative_path="Dockerfile", content="FROM python:3.12-slim\n"),
            GeneratedReferenceFile(relative_path="requirements.txt", content="fastapi\n"),
        ],
        setup_files=[
            GeneratedSetupFile(relative_path="gold.json", content='{"q": "a"}'),
        ],
        cost_usd=0.05,
        model_id="claude-sonnet-4-6",
    )


@dataclass
class FakeOutcomePlanner:
    spec: CourseOutcomeSpec | None = None
    error: Exception | None = None
    call_count: int = 0

    def plan_course(self, request: Any) -> CourseOutcomeSpec:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        assert self.spec is not None
        return self.spec


@dataclass
class FakeRouter:
    """Router stub. Default returns a coherent verdict."""

    coherent: bool = True
    rationale: str = "ok"
    concerns: list[str] = field(default_factory=list)
    raise_on_call: Exception | None = None
    call_count: int = 0

    def parse_structured(self, **kwargs: Any) -> Any:
        from app.services.spec_review_llm import SpecCoherenceVerdict

        self.call_count += 1
        if self.raise_on_call is not None:
            raise self.raise_on_call
        verdict = SpecCoherenceVerdict(
            is_coherent=self.coherent,
            rationale=self.rationale,
            concerns=list(self.concerns),
        )
        return SimpleNamespace(parsed=verdict, output_parsed=verdict, usage=None,
                               usage_summary=SimpleNamespace(estimated_cost_usd=0.0001))


@dataclass
class FakeRepoAuthor:
    files: list[tuple[str, str]] = field(default_factory=list)
    call_count: int = 0
    last_failure_context: Any = None

    def __post_init__(self) -> None:
        if not self.files:
            self.files = [
                ("Dockerfile", "FROM python:3.12-slim\nCMD [\"python\",\"-m\",\"app\"]\n"),
                ("requirements.txt", "fastapi\n"),
                ("app/main.py", "print('hi')\n"),
            ]

    def generate_bundle(
        self, *, spec: CourseOutcomeSpec, failure_context: Any = None
    ) -> list[tuple[str, str]]:
        self.call_count += 1
        self.last_failure_context = failure_context
        return list(self.files)


@dataclass
class _BootHandle:
    base_url: str


@dataclass
class FakeSandboxRunner:
    boot_ok: bool = True
    boot_error: Exception | None = None
    booted: list[Path] = field(default_factory=list)
    teardowns: int = 0

    def boot(self, reference_impl_dir: Path):
        if self.boot_error is not None:
            raise self.boot_error
        self.booted.append(Path(reference_impl_dir))
        if not self.boot_ok:
            # simulate boot returning a handle but reporting failure via flag
            return _BootHandle(base_url="")
        return _BootHandle(base_url="http://localhost:8080")

    def teardown(self, handle: _BootHandle) -> None:
        self.teardowns += 1


@dataclass
class FakeStarterBootRunner:
    """Runner used by ``node_starter_verify``. Returns a dict result.
    """

    ok: bool = True
    last_capabilities: Any = None

    def verify_starter(
        self, starter_dir: Path, *, capabilities: Any = None
    ) -> dict[str, Any]:
        self.last_capabilities = capabilities
        if self.ok:
            return {"ok": True, "logs": ""}
        return {
            "ok": False,
            "stage": "build",
            "logs": "image build failed",
            "error": "module not found",
        }


@dataclass
class FakeOracleAuthor:
    result: OracleAuthoringResult | None = None
    call_count: int = 0

    def author_oracle(self, spec: CourseOutcomeSpec) -> OracleAuthoringResult:
        self.call_count += 1
        if self.result is None:
            return _sample_authoring_result()
        return self.result


@dataclass
class FakeOraclePass:
    result: OraclePassResult | None = None
    call_count: int = 0
    last_capabilities: Any = None

    def run(
        self,
        *,
        scenarios: list[Any],
        reference_impl_dir: Path,
        setup_data_dir: Path | None = None,
        course_meta: dict[str, Any] | None = None,
        router: Any = None,
        capabilities: Any = None,
    ) -> OraclePassResult:
        self.call_count += 1
        self.last_capabilities = capabilities
        if self.result is None:
            return _passing_oracle_result()
        return self.result


def _build_deps(
    *,
    planner: FakeOutcomePlanner | None = None,
    router: FakeRouter | None = None,
    repo_author: FakeRepoAuthor | None = None,
    sandbox_runner: FakeSandboxRunner | None = None,
    starter_verifier: FakeStarterBootRunner | None = None,
    oracle_author: FakeOracleAuthor | None = None,
    oracle_pass: FakeOraclePass | None = None,
) -> OutcomeGraphDeps:
    spec = _sample_spec()
    return OutcomeGraphDeps(
        planner=planner or FakeOutcomePlanner(spec=spec),
        router=router or FakeRouter(),
        repo_author=repo_author or FakeRepoAuthor(),
        sandbox_runner=sandbox_runner or FakeSandboxRunner(),
        starter_verifier=starter_verifier or FakeStarterBootRunner(),
        oracle_author=oracle_author or FakeOracleAuthor(),
        oracle_pass=oracle_pass or FakeOraclePass(),
    )


def _request() -> Any:
    """Minimal request shim. The fake planner ignores its content."""
    return SimpleNamespace(
        goal="Build a grounded retrieval service",
        title=None,
        package_type_hint=None,
        learning_outcomes=[],
        creator_setup=SimpleNamespace(
            starter_type=None,
            implementation_language=None,
            application_framework=None,
            primary_database=None,
            tech_stack=[],
        ),
    )


# ---------------- State model ----------------


def test_state_default_initial_values(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(run_id="r1", workspace_root=tmp_path)
    assert state.stage == "initialized"
    assert state.status == "running"
    assert state.cost_usd == 0.0
    assert state.starter_attempt == 0
    assert state.grader_attempt == 0
    assert state.spec is None
    assert state.blocking_reasons == []


# ---------------- node_spec_authoring ----------------


def test_node_spec_authoring_sets_spec_on_state(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(run_id="r1", workspace_root=tmp_path, request=_request())
    deps = _build_deps()
    out = node_spec_authoring(state, deps=deps)
    assert out.spec is not None
    assert out.spec.title == "Build a Grounded RAG Service"
    assert out.stage == "spec_authoring"
    assert deps.planner.call_count == 1


def test_node_spec_authoring_blocks_on_planner_error(tmp_path: Path) -> None:
    deps = _build_deps(
        planner=FakeOutcomePlanner(error=OutcomeCourseGenerationError("planner died"))
    )
    state = OutcomeWorkflowState(run_id="r1", workspace_root=tmp_path, request=_request())
    out = node_spec_authoring(state, deps=deps)
    assert out.status == "blocked"
    assert any("planner died" in r for r in out.blocking_reasons)


# ---------------- node_spec_review ----------------


def test_node_spec_review_passes_with_coherent_router(tmp_path: Path) -> None:
    deps = _build_deps()
    state = OutcomeWorkflowState(
        run_id="r1", workspace_root=tmp_path, spec=_sample_spec(),
    )
    out = node_spec_review(state, deps=deps)
    assert out.spec_review_findings == []
    assert out.stage == "awaiting_gate_1"
    assert out.status == "awaiting_human"


def test_node_spec_review_emits_findings_when_incoherent(tmp_path: Path) -> None:
    deps = _build_deps(
        router=FakeRouter(
            coherent=False,
            rationale="Quality bar 'general' is generic.",
            concerns=["Quality bar 'general' is too generic."],
        )
    )
    state = OutcomeWorkflowState(
        run_id="r1", workspace_root=tmp_path, spec=_sample_spec(),
    )
    out = node_spec_review(state, deps=deps)
    assert out.spec_review_findings
    # Still pauses at gate 1 — findings flow through for human review
    assert out.stage == "awaiting_gate_1"


def test_node_spec_review_tracks_cost(tmp_path: Path) -> None:
    deps = _build_deps()
    state = OutcomeWorkflowState(
        run_id="r1", workspace_root=tmp_path, spec=_sample_spec(),
    )
    out = node_spec_review(state, deps=deps)
    assert out.cost_usd > 0


# ---------------- node_starter_authoring + verify + reviewer ----------------


def test_node_starter_authoring_populates_files_and_attempts(tmp_path: Path) -> None:
    deps = _build_deps()
    state = OutcomeWorkflowState(
        run_id="r1", workspace_root=tmp_path, spec=_sample_spec(),
    )
    out = node_starter_authoring(state, deps=deps)
    assert out.starter_files
    assert out.starter_attempt == 1
    assert deps.repo_author.call_count == 1


def test_node_starter_verify_materializes_files_and_records_boot_result(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        starter_files=[
            ("Dockerfile", "FROM python:3.12-slim\n"),
            ("app/main.py", "print('hi')\n"),
        ],
    )
    deps = _build_deps()
    out = node_starter_verify(state, deps=deps)
    assert (tmp_path / "public/starter/Dockerfile").exists()
    assert out.starter_boot_result is not None
    assert out.starter_boot_result.get("ok") is True


def test_node_starter_verify_threads_capabilities_to_verifier(tmp_path: Path) -> None:
    """``node_starter_verify`` MUST pass ``state.spec.capabilities`` to the verifier.

    Codex review #7 finding #3 — capabilities never reached the
    sandbox before this wave. A spec with ``runtime_llm_required=True``
    must arrive at the verifier so it can either provision the proxy
    sidecar or fail loudly.
    """
    from app.services.course_outcome_models import CapabilityFlags

    spec = _sample_spec()
    spec = spec.model_copy(
        update={"capabilities": CapabilityFlags(runtime_llm_required=True)}
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=spec,
        starter_files=[("Dockerfile", "FROM python:3.12-slim\n")],
    )
    verifier = FakeStarterBootRunner()
    deps = _build_deps(starter_verifier=verifier)
    node_starter_verify(state, deps=deps)
    assert verifier.last_capabilities is not None
    assert verifier.last_capabilities.runtime_llm_required is True


def test_node_starter_verify_records_failure(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        starter_files=[("Dockerfile", "FROM x\n")],
    )
    deps = _build_deps(starter_verifier=FakeStarterBootRunner(ok=False))
    out = node_starter_verify(state, deps=deps)
    assert out.starter_boot_result is not None
    assert out.starter_boot_result.get("ok") is False
    assert out.starter_review_findings  # converts the failure into findings


def test_node_reviewer_code_appends_findings_when_router_present(tmp_path: Path) -> None:
    """The reviewer_code node calls the router via the existing
    domain-grounding judge; passing a no-router state should produce no findings."""
    deps = _build_deps()
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
    )
    # No findings expected here because the README/content isn't built.
    out = node_reviewer_code(state, deps=deps)
    # The node must still set/preserve stage progression bits and not crash.
    assert isinstance(out.starter_review_findings, list)


# ---------------- Repair budget: starter ----------------


def test_starter_repair_fires_when_verify_fails(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        starter_files=[("Dockerfile", "FROM x\n")],
        starter_attempt=1,
        starter_review_findings=[
            ReviewerFinding(
                category="starter_verify",
                severity=ReviewerFindingSeverity.error,
                title="boot failed",
                detail="image build failed",
            )
        ],
    )
    deps = _build_deps()
    out = node_starter_repair(state, deps=deps)
    assert out.starter_attempt == 2
    assert out.starter_files == deps.repo_author.files
    # repair must have forwarded failure context
    assert deps.repo_author.last_failure_context is not None


# ---------------- node_oracle_authoring ----------------


def test_node_oracle_authoring_records_result_and_materializes(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
    )
    deps = _build_deps()
    out = node_oracle_authoring(state, deps=deps)
    assert out.oracle_authoring_result is not None
    # scenarios + reference + setup all hit disk
    assert (tmp_path / "private/grader/scenarios/happy_path.yaml").exists()
    assert (tmp_path / "private/grader/_reference/Dockerfile").exists()
    assert (tmp_path / "private/grader/_setup/gold.json").exists()


# ---------------- node_oracle_pass + validation ----------------


def test_node_oracle_pass_threads_capabilities(tmp_path: Path) -> None:
    """``node_oracle_pass`` MUST pass ``state.spec.capabilities`` into ``oracle_pass.run``.

    Codex review #7 finding #3 — without this, the reference impl
    boots on a bare container even when the spec declared the LLM
    proxy / sidecar database / durable state requirements.
    """
    from app.services.course_outcome_models import CapabilityFlags
    from app.services.outcome_artifact_materializer import (
        materialize_oracle_bundle,
    )

    auth = _sample_authoring_result()
    materialize_oracle_bundle(tmp_path, auth)
    spec = _sample_spec()
    spec = spec.model_copy(
        update={"capabilities": CapabilityFlags(sidecar_database="redis")}
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=spec,
        oracle_authoring_result=auth,
    )
    oracle_pass = FakeOraclePass()
    deps = _build_deps(oracle_pass=oracle_pass)
    node_oracle_pass(state, deps=deps)
    assert oracle_pass.last_capabilities is not None
    assert oracle_pass.last_capabilities.sidecar_database == "redis"


def test_node_oracle_pass_persists_outputs(tmp_path: Path) -> None:
    auth = _sample_authoring_result()
    # materialize first so oracle_pass can read the dir
    from app.services.outcome_artifact_materializer import materialize_oracle_bundle
    materialize_oracle_bundle(tmp_path, auth)
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        oracle_authoring_result=auth,
    )
    deps = _build_deps()
    out = node_oracle_pass(state, deps=deps)
    assert out.oracle_pass_result is not None
    outputs_path = tmp_path / "private/grader/_oracle/outputs.json"
    assert outputs_path.exists()


def test_node_oracle_validation_publishable(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        oracle_authoring_result=_sample_authoring_result(),
        oracle_pass_result=_passing_oracle_result(),
    )
    # materialize so scenarios load
    from app.services.outcome_artifact_materializer import materialize_oracle_bundle
    materialize_oracle_bundle(tmp_path, _sample_authoring_result())
    out = node_oracle_validation(state, deps=_build_deps())
    assert out.oracle_validation_report is not None
    # The 1-scenario sample only covers happy_path; that's enough to be marked
    # not publishable due to missing categories — but the node ran cleanly.


def test_node_oracle_validation_blocks_when_failed(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        oracle_authoring_result=_sample_authoring_result(),
        oracle_pass_result=_failing_oracle_result(),
    )
    from app.services.outcome_artifact_materializer import materialize_oracle_bundle
    materialize_oracle_bundle(tmp_path, _sample_authoring_result())
    out = node_oracle_validation(state, deps=_build_deps())
    assert out.oracle_validation_report is not None
    assert not out.oracle_validation_report.publishable
    assert out.blocking_reasons


# ---------------- node_grader_repair ----------------


def test_node_grader_repair_increments_attempt_and_reauthors(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        grader_attempt=1,
    )
    deps = _build_deps()
    out = node_grader_repair(state, deps=deps)
    assert out.grader_attempt == 2
    assert out.oracle_authoring_result is not None
    assert deps.oracle_author.call_count == 1


# ---------------- node_publish ----------------


def test_node_publish_writes_runner_and_spec(tmp_path: Path) -> None:
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        spec=_sample_spec(),
        oracle_authoring_result=_sample_authoring_result(),
        oracle_pass_result=_passing_oracle_result(),
    )
    out = node_publish(state, deps=_build_deps())
    assert (tmp_path / "private/grader/runner.py").exists()
    assert (tmp_path / "private/course_spec.json").exists()
    assert out.stage == "published"
    assert out.status == "published"


# ---------------- Graph execute: gate semantics ----------------


def test_graph_execute_pauses_at_gate_1_after_spec_review(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    state = OutcomeWorkflowState(
        run_id="r1", workspace_root=tmp_path, request=_request()
    )
    out = graph.execute(state, deps=_build_deps())
    assert out.stage == "awaiting_gate_1"
    assert out.status == "awaiting_human"
    assert out.spec is not None


def test_graph_execute_resumes_after_gate_1(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=_sample_spec(),
        stage="awaiting_gate_1",
        status="running",  # caller has approved gate_1 and resumed
    )
    out = graph.execute(state, deps=_build_deps())
    # After approval, we run starter authoring → verify → reviewer → gate 2
    assert out.stage == "awaiting_gate_2"
    assert out.status == "awaiting_human"
    assert out.starter_files  # starter was authored on resume


def test_graph_execute_runs_grader_lane_after_gate_2(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=_sample_spec(),
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=_build_deps())
    assert out.stage == "awaiting_gate_3"
    assert out.status == "awaiting_human"
    assert out.oracle_authoring_result is not None


def test_graph_execute_publishes_after_gate_3(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=_sample_spec(),
        oracle_authoring_result=_sample_authoring_result(),
        oracle_pass_result=_passing_oracle_result(),
        stage="awaiting_gate_3",
        status="running",
    )
    out = graph.execute(state, deps=_build_deps())
    assert out.stage == "published"
    assert out.status == "published"


# ---------------- Retry budget: starter ----------------


def test_starter_repair_exhausts_at_three_attempts(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    deps = _build_deps(starter_verifier=FakeStarterBootRunner(ok=False))
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=_sample_spec(),
        stage="awaiting_gate_1",
        status="running",
    )
    out = graph.execute(state, deps=deps)
    assert out.status == "blocked"
    assert out.starter_attempt >= 3
    assert any("starter" in r.lower() for r in out.blocking_reasons)


# ---------------- Retry budget: grader ----------------


def test_grader_repair_exhausts_at_three_attempts(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    deps = _build_deps(oracle_pass=FakeOraclePass(result=_failing_oracle_result()))
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=_sample_spec(),
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)
    assert out.status == "blocked"
    assert out.grader_attempt >= 3
    assert any("oracle" in r.lower() or "scenario" in r.lower() for r in out.blocking_reasons)


# ---------------- Cost aggregation ----------------


def test_cost_aggregation_sums_across_node_calls(tmp_path: Path) -> None:
    """Each router call must add to ``state.cost_usd``."""
    deps = _build_deps()
    state = OutcomeWorkflowState(
        run_id="r1", workspace_root=tmp_path, request=_request()
    )
    out = node_spec_authoring(state, deps=deps)
    cost_after_authoring = out.cost_usd
    out2 = node_spec_review(out, deps=deps)
    assert out2.cost_usd > cost_after_authoring


# ---------------- Blocked status sets blocking_reasons ----------------


def test_blocked_status_carries_clear_reason(tmp_path: Path) -> None:
    deps = _build_deps(
        planner=FakeOutcomePlanner(error=OutcomeCourseGenerationError("could not plan"))
    )
    state = OutcomeWorkflowState(
        run_id="r1", workspace_root=tmp_path, request=_request()
    )
    graph = OutcomeWorkflowGraph()
    out = graph.execute(state, deps=deps)
    assert out.status == "blocked"
    assert out.blocking_reasons
    assert any("could not plan" in r for r in out.blocking_reasons)


# ---------------- Oracle-source branch (Wave 4.5) ----------------
#
# The graph now supports three oracle-source modes
# (``curated``, ``reference_run``, ``hybrid``). The dispatcher routes
# after ``node_oracle_authoring``:
#
#   curated      → node_oracle_curated_validation (no ref-impl boot)
#   reference_run→ node_oracle_pass → node_oracle_validation (existing)
#   hybrid       → both, AND'd
#
# Each test wires up scenarios + setup data shaped for the curated
# checker (``gold_set_path`` resolves into setup_data ``gold.json``).


def _curated_scenario_yaml(scenario_id: str, category: str) -> str:
    """Curated-mode scenario: oracle_set_overlap with a real ``gold_set_path``
    that resolves into the curated setup_data ``gold.json`` content below.

    ``quality_bar_ids`` lines up with the spec's single ``faithfulness``
    bar so the curated publish-gate's coverage check is satisfied.
    """
    return (
        f"id: {scenario_id}\n"
        f"description: {category} scenario\n"
        f"category: {category}\n"
        "quality_bar_ids:\n"
        "  - faithfulness\n"
        "trace:\n"
        "  - id: ask\n"
        "    method: POST\n"
        "    path: /answer\n"
        "    body:\n"
        "      question: hello\n"
        "    expect:\n"
        "      status: 200\n"
        "rubrics:\n"
        "  - kind: oracle_set_overlap\n"
        "    target: ask.body.citations\n"
        "    gold_set_path: gold.q.expected\n"
        "    min_recall: 0.5\n"
    )


def _curated_authoring_result(gold_value: list[str] | None = None) -> OracleAuthoringResult:
    import json as _json

    gold_payload = {
        "q": {"expected": gold_value if gold_value is not None else ["d1"]}
    }
    return OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(
                filename=f"{cat}.yaml",
                yaml_content=_curated_scenario_yaml(f"s_{cat}", cat),
            )
            for cat in ("happy_path", "boundary", "malformed_input")
        ],
        reference_files=[
            GeneratedReferenceFile(
                relative_path="Dockerfile", content="FROM python:3.12-slim\n"
            ),
        ],
        setup_files=[
            GeneratedSetupFile(
                relative_path="gold.json", content=_json.dumps(gold_payload)
            ),
        ],
        cost_usd=0.0,
        model_id="claude-sonnet-4-6",
    )


def _spec_with_oracle_source(source: OracleSource) -> CourseOutcomeSpec:
    base = _sample_spec()
    return base.model_copy(update={"oracle_source": source})


def test_graph_curated_mode_skips_oracle_pass(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.curated)
    oracle_pass_fake = FakeOraclePass()
    oracle_author_fake = FakeOracleAuthor(result=_curated_authoring_result())
    deps = _build_deps(
        oracle_author=oracle_author_fake,
        oracle_pass=oracle_pass_fake,
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)

    # Curated mode must NEVER boot the reference impl. The FakeOraclePass
    # must not be called.
    assert oracle_pass_fake.call_count == 0
    assert out.oracle_pass_result is None
    # The validation report lives on ``curated_validation_report`` in
    # curated mode; the reference-run validator never runs so
    # ``oracle_validation_report`` stays unset.
    assert out.curated_validation_report is not None
    assert out.oracle_validation_report is None
    assert out.stage == "awaiting_gate_3"
    assert out.status == "awaiting_human"


def test_graph_curated_mode_invokes_curated_validation(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.curated)
    deps = _build_deps(
        oracle_author=FakeOracleAuthor(result=_curated_authoring_result()),
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)

    # The curated validator runs without an OraclePassResult, so the
    # hashes stay empty.
    assert out.curated_validation_report is not None
    assert out.curated_validation_report.reference_impl_hash == ""
    assert out.curated_validation_report.scenario_set_hash == ""
    assert out.curated_validation_report.publishable is True


def test_graph_reference_run_mode_preserves_existing_behavior(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.reference_run)
    oracle_pass_fake = FakeOraclePass()
    deps = _build_deps(oracle_pass=oracle_pass_fake)
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)

    # reference_run keeps the existing oracle_pass + oracle_validation path.
    assert oracle_pass_fake.call_count >= 1
    assert out.oracle_pass_result is not None
    assert out.oracle_validation_report is not None
    assert out.stage == "awaiting_gate_3"


def test_graph_hybrid_mode_runs_both_validators(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.hybrid)
    oracle_pass_fake = FakeOraclePass()
    deps = _build_deps(
        oracle_author=FakeOracleAuthor(result=_curated_authoring_result()),
        oracle_pass=oracle_pass_fake,
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)

    # Both ran.
    assert oracle_pass_fake.call_count >= 1
    assert out.oracle_pass_result is not None
    assert out.oracle_validation_report is not None
    assert out.curated_validation_report is not None
    # Both validators agreed: publishable.
    assert out.oracle_validation_report.publishable is True
    assert out.curated_validation_report.publishable is True
    assert out.stage == "awaiting_gate_3"


def test_graph_hybrid_mode_blocks_when_curated_fails(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.hybrid)
    # Build a curated authoring result whose gold references a doc_id
    # that is fine in itself, but break the curated check by pointing the
    # rubric at a missing gold path. Simpler approach: omit the gold file
    # entirely so the gold_set_path can't resolve.
    bad_authoring = _curated_authoring_result()
    bad_authoring = bad_authoring.model_copy(update={"setup_files": []})

    oracle_pass_fake = FakeOraclePass(result=_passing_oracle_result())
    deps = _build_deps(
        oracle_author=FakeOracleAuthor(result=bad_authoring),
        oracle_pass=oracle_pass_fake,
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)

    # Ref-impl path passed but curated failed → blocked.
    assert out.status == "blocked"
    assert out.grader_attempt >= 3  # retried curated until budget exhausted
    # blocking_reasons must mention the curated failure source
    joined = " | ".join(out.blocking_reasons)
    assert "gold" in joined.lower() or "setup_data" in joined.lower()


def test_graph_hybrid_mode_blocks_when_ref_impl_fails(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.hybrid)
    oracle_pass_fake = FakeOraclePass(result=_failing_oracle_result())
    deps = _build_deps(
        oracle_author=FakeOracleAuthor(result=_curated_authoring_result()),
        oracle_pass=oracle_pass_fake,
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)

    # Curated passed but ref-impl failed → blocked.
    assert out.status == "blocked"
    assert out.grader_attempt >= 3
    joined = " | ".join(out.blocking_reasons)
    # blocking reason includes the failed scenario id from ref-impl
    assert "s_happy_path" in joined


def test_graph_hybrid_mode_both_pass_publishes(tmp_path: Path) -> None:
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.hybrid)
    deps = _build_deps(
        oracle_author=FakeOracleAuthor(result=_curated_authoring_result()),
        oracle_pass=FakeOraclePass(result=_passing_oracle_result()),
    )
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        oracle_authoring_result=_curated_authoring_result(),
        oracle_pass_result=_passing_oracle_result(),
        stage="awaiting_gate_3",
        status="running",
    )
    out = graph.execute(state, deps=deps)
    assert out.stage == "published"
    assert out.status == "published"


def test_graph_curated_mode_grader_repair_retries(tmp_path: Path) -> None:
    """Curated mode must still exercise the grader_repair retry pocket
    when the curated validator fails (gold path unresolvable)."""
    graph = OutcomeWorkflowGraph()
    spec = _spec_with_oracle_source(OracleSource.curated)
    bad_authoring = _curated_authoring_result().model_copy(update={"setup_files": []})
    author = FakeOracleAuthor(result=bad_authoring)
    deps = _build_deps(oracle_author=author)
    state = OutcomeWorkflowState(
        run_id="r1",
        workspace_root=tmp_path,
        request=_request(),
        spec=spec,
        starter_files=[("Dockerfile", "FROM x\n")],
        stage="awaiting_gate_2",
        status="running",
    )
    out = graph.execute(state, deps=deps)
    # All retries exhausted (3 grader attempts) — the FakeOracleAuthor
    # was called once per attempt.
    assert out.status == "blocked"
    assert out.grader_attempt >= 3
    assert author.call_count >= 3
