"""Tests for the outcome-mode entry point in ``CourseGenerationService``.

Wave 5 retired the legacy per-deliverable path. Every brief is now
routed through the single-outcome graph
(``langgraph_outcome_graph.OutcomeWorkflowGraph``) and the resulting
``OutcomeWorkflowState`` is adapted back to a
``GenerateCourseFromBriefResponse`` so the API surface is unchanged.

Tests in this module cover:

* ``generate_outcome_course_from_brief`` — the module-level helper used
  directly by tests / scripts to drive the graph against an injected
  fake planner.
* ``CourseGenerationService.generate_course_run`` — the public service
  entry point the FastAPI app uses. The integration tests at the bottom
  exercise that adapter end-to-end against fakes.
"""
from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.course import (
    CourseGenerationSource,
    CourseGenerationStatus,
    CreatorCourseSetupInput,
    GenerateCourseFromBriefRequest,
    GenerateCourseFromBriefResponse,
)
from app.domain.registry import PackageType
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.course_generation_service import (
    CourseGenerationService,
)
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    QualityBar,
    StarterType,
)
from app.services.course_outcome_planner import OutcomeCourseGenerationError
from app.services.course_workflow_service import CourseWorkflowService
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


# ---------------- Fixtures ----------------


def _spec() -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title="Build a Grounded RAG Service",
        goal=(
            "Build a small HTTP service that ingests documents, retrieves "
            "passages, and returns a grounded answer."
        ),
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str"},
                description="Answer the question.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite supporting passages.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
    )


@dataclass
class _FakePlanner:
    spec: CourseOutcomeSpec | None = None
    error: Exception | None = None
    call_count: int = 0

    def plan_course(self, request: Any) -> CourseOutcomeSpec:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        assert self.spec is not None
        return self.spec


# ---------------- generate_outcome_course_from_brief ----------------


def test_outcome_mode_happy_path_through_gate_1(tmp_path: Path) -> None:
    from app.services.course_generation_service import (
        generate_outcome_course_from_brief,
    )
    from app.domain.course import (
        CreatorCourseSetupInput,
        GenerateCourseFromBriefRequest,
    )

    request = GenerateCourseFromBriefRequest(
        goal="Build a grounded retrieval service",
        creator_setup=CreatorCourseSetupInput(),
    )
    planner = _FakePlanner(spec=_spec())
    state = generate_outcome_course_from_brief(
        request,
        planner=planner,
        workspace_root=tmp_path,
        run_id="run-1",
    )
    assert state.stage == "awaiting_gate_1"
    assert state.status == "awaiting_human"
    assert state.spec is not None
    assert planner.call_count == 1


def test_outcome_mode_blocks_on_planner_failure(tmp_path: Path) -> None:
    from app.services.course_generation_service import (
        generate_outcome_course_from_brief,
    )
    from app.domain.course import (
        CreatorCourseSetupInput,
        GenerateCourseFromBriefRequest,
    )

    request = GenerateCourseFromBriefRequest(
        goal="Build a grounded retrieval service",
        creator_setup=CreatorCourseSetupInput(),
    )
    planner = _FakePlanner(error=OutcomeCourseGenerationError("planner gave up"))
    state = generate_outcome_course_from_brief(
        request,
        planner=planner,
        workspace_root=tmp_path,
        run_id="run-1",
    )
    assert state.status == "blocked"
    assert any("planner gave up" in r for r in state.blocking_reasons)


def test_outcome_mode_state_has_run_id_and_workspace_set(tmp_path: Path) -> None:
    from app.services.course_generation_service import (
        generate_outcome_course_from_brief,
    )
    from app.domain.course import (
        CreatorCourseSetupInput,
        GenerateCourseFromBriefRequest,
    )

    request = GenerateCourseFromBriefRequest(
        goal="Build a grounded retrieval service",
        creator_setup=CreatorCourseSetupInput(),
    )
    planner = _FakePlanner(spec=_spec())
    state = generate_outcome_course_from_brief(
        request,
        planner=planner,
        workspace_root=tmp_path,
        run_id="custom-run-id",
    )
    assert state.run_id == "custom-run-id"
    assert state.workspace_root == tmp_path


# ---------------- generate_course_run public entry point ----------------
#
# These tests target the public service entry point. After Wave 5, every
# brief routes through the outcome graph; the response adapter converts
# the resulting ``OutcomeWorkflowState`` back to the legacy response
# shape so the API surface is unchanged.


class _FakeLegacyPlanner:
    """Drop-in replacement for ``OpenAICoursePlanner`` that records calls.

    The real legacy planner has a ``status()`` accessor and a
    ``plan_course()`` method; both are exercised by
    ``CourseGenerationService._generate_normalized_plan``. We disable
    the planner (``available=False``) so the legacy path falls through to
    the deterministic fallback inside the service — that way the legacy
    branch still produces a valid response without any real network I/O.
    """

    def __init__(self) -> None:
        self.status_calls = 0
        self.plan_calls = 0

    def status(self) -> CourseGenerationStatus:
        self.status_calls += 1
        return CourseGenerationStatus(
            provider="openai",
            available=False,
            source=CourseGenerationSource.deterministic_fallback,
            message="Live planning disabled for tests.",
            sdk_installed=False,
            api_key_present=False,
            model_id=None,
            env_file=None,
        )

    def plan_course(self, request: Any):  # pragma: no cover - not reachable when available=False
        self.plan_calls += 1
        raise AssertionError(
            "Legacy plan_course() should not be invoked when the planner is unavailable."
        )

    def suggest_learning_outcomes(self, request: Any):  # pragma: no cover
        raise AssertionError("suggest_learning_outcomes should not be called in these tests.")


@dataclass
class _FakeOutcomePlannerFactory:
    """Records that the planner factory was called and what it returned.

    The service injects an ``outcome_planner`` (or a factory that returns
    one); tests use this fake to verify the outcome path was chosen.
    """

    spec: CourseOutcomeSpec | None = None
    error: Exception | None = None
    call_count: int = 0

    def plan_course(self, request: Any) -> CourseOutcomeSpec:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        assert self.spec is not None
        return self.spec


def _build_service(
    tmp_dir: str,
    *,
    legacy_planner: _FakeLegacyPlanner | None = None,
    outcome_planner: Any = None,
    outcome_deps_overrides: dict[str, Any] | None = None,
) -> tuple[CourseGenerationService, _FakeLegacyPlanner]:
    legacy_planner = legacy_planner or _FakeLegacyPlanner()
    store = SQLiteWorkflowStore(db_path=f"{tmp_dir}/test.db")
    workspace_manager = AssignmentWorkspaceManager(base_dir=f"{tmp_dir}/workspaces")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{tmp_dir}/generated"),
        runner=TaskAgentBlackBoxRunner(),
        workspace_manager=workspace_manager,
    )
    course_workflow_service = CourseWorkflowService(
        store,
        workflow_service,
        job_runner=lambda job: job(),
    )
    # Defaults: disable the real LLM router AND the real Docker starter
    # verifier / oracle-pass sandbox so tests don't hit Anthropic and
    # don't spin up containers. ``repo_author`` is forced to the
    # deterministic-shell fallback so the adapter doesn't internally
    # try to reach the LLM router. Tests that exercise the real wiring
    # should override these explicitly via ``outcome_deps_overrides``.
    from app.services.outcome_graph_deps import (
        PlaceholderReferenceImplSandbox,
        PlaceholderStarterVerifier,
    )
    from app.services.outcome_repo_author_adapter import (
        DeterministicStarterShellFallback,
    )
    from app.services.oracle_pass import OraclePass

    defaults: dict[str, Any] = {
        "router": None,
        "repo_author": DeterministicStarterShellFallback(),
        "starter_verifier": PlaceholderStarterVerifier(),
        "oracle_pass": OraclePass(sandbox_runner=PlaceholderReferenceImplSandbox()),
    }
    if outcome_deps_overrides:
        defaults.update(outcome_deps_overrides)
    service = CourseGenerationService(
        course_workflow_service,
        live_planner=legacy_planner,  # type: ignore[arg-type]
        outcome_planner=outcome_planner,
        job_runner=lambda job: job(),
        outcome_workspace_root=Path(tmp_dir) / "outcome_workspaces",
        outcome_deps_overrides=defaults,
    )
    return service, legacy_planner


class GenerateCourseRunFlagWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.request = GenerateCourseFromBriefRequest(
            goal="Build a grounded retrieval service for engineering teams.",
            creator_setup=CreatorCourseSetupInput(),
        )

    def test_runs_outcome_path_and_skips_legacy_planner(self) -> None:
        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, legacy_planner = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(self.request)
        # Same legacy response shape — the adapter conforms.
        self.assertIsInstance(response, GenerateCourseFromBriefResponse)
        # The outcome planner WAS called, exactly once.
        self.assertEqual(outcome_planner.call_count, 1)
        # The legacy planner's plan_course is NEVER called from the
        # outcome entry point (status() may still be sampled for the
        # response status field, which is fine).
        self.assertEqual(legacy_planner.plan_calls, 0)

    def test_blocked_when_outcome_planner_raises(self) -> None:
        outcome_planner = _FakeOutcomePlannerFactory(
            error=OutcomeCourseGenerationError("planner gave up")
        )
        service, _legacy_planner = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(self.request)
        self.assertIsInstance(response, GenerateCourseFromBriefResponse)
        # The course_run on the response should carry a blocked / failed
        # signal (we treat outcome planner failure as a generation
        # failure for the API contract).
        self.assertEqual(response.course_run.status.value, "blocked")
        # ``last_error`` records the planner failure verbatim so the UI
        # can surface it.
        self.assertIsNotNone(response.course_run.last_error)
        self.assertIn("planner gave up", response.course_run.last_error or "")

    def test_response_reflects_gate_1_pause(self) -> None:
        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _legacy_planner = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(self.request)
        # Gate 1 pause maps to awaiting_human / awaiting_course_review.
        self.assertEqual(response.course_run.status.value, "awaiting_human")
        # The course_run.goal should mirror the brief's goal.
        self.assertEqual(response.course_run.goal, self.request.goal)
        # The plan title comes from the outcome spec (adapter copies it
        # over so legacy consumers still see a meaningful title).
        self.assertEqual(response.plan.title, _spec().title)

    def test_response_carries_single_synthetic_outcome_deliverable(self) -> None:
        """Outcome path emits exactly one synthetic "outcome" deliverable.

        The legacy ``GeneratedCoursePlan`` schema requires
        ``min_length=1`` deliverables; the adapter satisfies that
        without fabricating a per-deliverable plan.
        """
        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _legacy_planner = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(self.request)
        self.assertIsInstance(response, GenerateCourseFromBriefResponse)
        self.assertEqual(len(response.plan.deliverables), 1)
        self.assertEqual(response.plan.deliverables[0].deliverable_slug, "outcome")


class GenerateCourseRunOutcomeIntegrationTests(unittest.TestCase):
    """End-to-end: outcome path runs the planner and returns adapted response."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_outcome_path_returns_legacy_shape_with_outcome_data(self) -> None:
        spec = _spec()
        outcome_planner = _FakeOutcomePlannerFactory(spec=spec)
        service, _legacy_planner = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        request = GenerateCourseFromBriefRequest(
            goal="Build a grounded retrieval service for engineering teams.",
            creator_setup=CreatorCourseSetupInput(),
        )
        response = service.generate_course_run(request)

        self.assertIsInstance(response, GenerateCourseFromBriefResponse)
        # Source label tells callers this came from the outcome path.
        self.assertIn(
            response.source,
            {CourseGenerationSource.openai_live, CourseGenerationSource.deterministic_fallback},
        )
        # Title is sourced from the spec the planner produced.
        self.assertEqual(response.plan.title, spec.title)
        # Plan summary is non-empty so the UI has something to render.
        self.assertTrue(response.plan.summary)
        # Course_run.shared_design_spec is None in the outcome flow (the
        # outcome spec lives elsewhere); the adapter must NOT fabricate one.
        # We assert at minimum the course_run exists and has a stable id.
        self.assertIsNotNone(response.course_run.id)
        self.assertEqual(response.course_run.goal, request.goal)
        # The legacy review object is still produced (empty / minimal is
        # fine — the API contract just needs it present).
        self.assertEqual(response.review.course_run_id, response.course_run.id)


class GenerateCourseRunOutcomePersistenceTests(unittest.TestCase):
    """Outcome-mode persistence: the kickoff response is durable across reload.

    Pre-fix, ``_kick_off_outcome_workflow`` discarded the
    ``OutcomeWorkflowState`` after building the response — the persisted
    ``CourseRun`` had zero deliverables and ``_compute_refreshed_run``
    would flip subsequent reads to ``blocked``. These tests pin the new
    "state survives" contract end-to-end through the public API.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_kickoff_persists_outcome_state_payload(self) -> None:
        """The course_run row carries an outcome_state blob after kick-off."""
        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _legacy_planner = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )
        request = GenerateCourseFromBriefRequest(
            goal="Build a grounded retrieval service for engineering teams.",
            creator_setup=CreatorCourseSetupInput(),
        )
        response = service.generate_course_run(request)

        # Confirm the course_run carries the outcome state.
        self.assertIn("outcome_state", response.course_run.payload_json)
        blob = response.course_run.payload_json["outcome_state"]
        self.assertEqual(blob["status"], "awaiting_human")
        self.assertEqual(blob["stage"], "awaiting_gate_1")


# ---------------- Production OutcomeGraphDeps wiring ----------------
#
# Codex review #6 finding #2: ``resume_outcome_workflow_after_gate`` and
# ``_kick_off_outcome_workflow`` previously built ``OutcomeGraphDeps``
# with only ``planner=...``. The graph nodes downstream of gate 1 assert
# on the other collaborators, so any approval at gate 1 crashed with
# ``AssertionError``. The fix introduces
# ``_build_production_outcome_deps`` which constructs a deps object with
# every collaborator wired (real or placeholder when production wiring
# is not yet available). These tests pin the contract.


class ProductionOutcomeDepsTests(unittest.TestCase):
    """Pin the production-wired ``OutcomeGraphDeps`` shape."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_production_outcome_deps_includes_all_collaborators(self) -> None:
        """``_build_production_outcome_deps()`` returns deps with all required slots set."""
        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _ = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )
        deps = service._build_production_outcome_deps()
        # Every collaborator the dispatcher asserts on must be non-None.
        self.assertIsNotNone(deps.planner)
        self.assertIsNotNone(deps.repo_author)
        self.assertIsNotNone(deps.starter_verifier)
        self.assertIsNotNone(deps.oracle_author)
        self.assertIsNotNone(deps.oracle_pass)

    def test_production_outcome_deps_defaults_wire_real_adapters(self) -> None:
        """Without overrides the production deps wire the real adapters.

        Wave 5e Agent A left two TODOs: the deterministic stub
        ``OutcomeRepoAuthorAdapter`` and the ``Placeholder*`` verifier /
        sandbox. Both are now real wrappers around production-grade
        primitives. The default deps (no overrides) MUST surface them
        — tests can still inject the placeholders explicitly via
        ``outcome_deps_overrides``.
        """
        from app.services.outcome_graph_deps import (
            PlaceholderReferenceImplSandbox,
            PlaceholderStarterVerifier,
            RealStarterVerifier,
        )
        from app.services.outcome_repo_author_adapter import (
            OutcomeRepoAuthorAdapter,
        )
        from app.services.workspace_boot import WorkspaceBootSandboxAdapter

        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        # Construct the service WITHOUT the test-default placeholders,
        # so we observe the production wiring the live API path uses.
        service = CourseGenerationService(
            course_workflow_service=CourseWorkflowService(
                SQLiteWorkflowStore(db_path=f"{self.tmp.name}/test.db"),
                WorkflowService(
                    SQLiteWorkflowStore(db_path=f"{self.tmp.name}/test.db"),
                    materializer=ArtifactMaterializer(base_dir=f"{self.tmp.name}/generated"),
                    runner=TaskAgentBlackBoxRunner(),
                    workspace_manager=AssignmentWorkspaceManager(
                        base_dir=f"{self.tmp.name}/workspaces"
                    ),
                ),
                job_runner=lambda job: job(),
            ),
            live_planner=_FakeLegacyPlanner(),  # type: ignore[arg-type]
            outcome_planner=outcome_planner,
            job_runner=lambda job: job(),
            outcome_workspace_root=Path(self.tmp.name) / "outcome_workspaces",
            outcome_deps_overrides={"router": None},
        )

        deps = service._build_production_outcome_deps()
        # repo_author is the real adapter, not the deterministic stub.
        self.assertIsInstance(deps.repo_author, OutcomeRepoAuthorAdapter)
        # starter_verifier is the real wrapper, NOT the placeholder.
        self.assertIsInstance(deps.starter_verifier, RealStarterVerifier)
        self.assertNotIsInstance(deps.starter_verifier, PlaceholderStarterVerifier)
        # oracle_pass uses the real sandbox adapter, not the placeholder.
        self.assertIsInstance(
            deps.oracle_pass.sandbox_runner, WorkspaceBootSandboxAdapter
        )
        self.assertNotIsInstance(
            deps.oracle_pass.sandbox_runner, PlaceholderReferenceImplSandbox
        )

    def test_real_starter_verifier_returns_ok_dict_on_successful_boot(self) -> None:
        """When ``boot_and_verify`` succeeds, the verifier reports ``ok=True``.

        Mocks the underlying ``boot_and_verify`` so no real Docker runs.
        """
        from contextlib import contextmanager

        from app.services.outcome_graph_deps import RealStarterVerifier
        from app.services.workspace_boot import WorkspaceBootHandle

        @contextmanager
        def fake_boot(*args, **kwargs):
            yield WorkspaceBootHandle(
                base_url="http://127.0.0.1:55555",
                container_id="fake-cid",
                image_tag="fake-tag",
            )

        verifier = RealStarterVerifier()
        with unittest.mock.patch(
            "app.services.workspace_boot.boot_and_verify",
            fake_boot,
        ):
            result = verifier.verify_starter(Path(self.tmp.name))

        self.assertTrue(result["ok"])
        self.assertEqual(result["base_url"], "http://127.0.0.1:55555")
        self.assertEqual(result["stage"], "boot")

    def test_real_starter_verifier_returns_failure_dict_on_boot_error(self) -> None:
        """A ``WorkspaceBootError`` is translated to ``{"ok": False, ...}``."""
        from app.services.outcome_graph_deps import RealStarterVerifier
        from app.services.workspace_boot import WorkspaceBootError

        def raise_boot_error(*args, **kwargs):
            raise WorkspaceBootError("docker build failed: missing Dockerfile")

        verifier = RealStarterVerifier()
        with unittest.mock.patch(
            "app.services.workspace_boot.boot_and_verify",
            raise_boot_error,
        ):
            result = verifier.verify_starter(Path(self.tmp.name))

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "build")
        self.assertIn("docker build failed", result["error"])

    def test_real_starter_verifier_threads_capabilities_to_boot(self) -> None:
        """``RealStarterVerifier.verify_starter(dir, capabilities=...)`` forwards
        the capability flags into ``boot_and_verify``.

        Codex review #7 finding #3 — capabilities must reach sandbox
        provisioning. The verifier is the entry point for the starter
        boot; if it drops capabilities, the sandbox boots a bare
        container regardless of what the spec asked for.
        """
        from contextlib import contextmanager

        from app.services.course_outcome_models import CapabilityFlags
        from app.services.outcome_graph_deps import RealStarterVerifier
        from app.services.workspace_boot import WorkspaceBootHandle

        captured_kwargs: dict = {}

        @contextmanager
        def fake_boot(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield WorkspaceBootHandle(
                base_url="http://127.0.0.1:55555",
                container_id="fake-cid",
                image_tag="fake-tag",
            )

        caps = CapabilityFlags(durable_state_required=True)
        verifier = RealStarterVerifier(capabilities=caps)
        with unittest.mock.patch(
            "app.services.workspace_boot.boot_and_verify",
            fake_boot,
        ):
            result = verifier.verify_starter(Path(self.tmp.name))

        self.assertTrue(result["ok"])
        self.assertIn("capabilities", captured_kwargs)
        self.assertIs(captured_kwargs["capabilities"], caps)

    def test_workspace_boot_sandbox_adapter_threads_capabilities(self) -> None:
        """``WorkspaceBootSandboxAdapter(capabilities=...).boot(dir)`` forwards
        the capability flags into the underlying ``boot_and_verify`` /
        ``_provision_capabilities`` calls.

        The oracle-pass sandbox protocol is duck-typed (``boot(dir)``)
        so the adapter must pin the capabilities up front via its
        constructor — they then thread into the boot internals every
        call.
        """
        from app.services.course_outcome_models import CapabilityFlags
        from app.services.workspace_boot import WorkspaceBootSandboxAdapter

        provisioned_calls: list = []

        def fake_provision(capabilities):
            provisioned_calls.append(capabilities)
            # default-flag check still applies; no error raised here.

        class _FakeRun:
            def __init__(self, stdout: str = "cid\n") -> None:
                self.returncode = 0
                self.stdout = stdout
                self.stderr = ""

        caps = CapabilityFlags(structured_logging_required=True)
        adapter = WorkspaceBootSandboxAdapter(capabilities=caps)
        # All the docker plumbing is mocked; we only check the
        # capability hook fired with the configured flags.
        with unittest.mock.patch(
            "app.services.workspace_boot._provision_capabilities",
            fake_provision,
        ), unittest.mock.patch(
            "app.services.workspace_boot.subprocess.run",
            return_value=_FakeRun(),
        ), unittest.mock.patch(
            "app.services.workspace_boot._allocate_port", return_value=58001
        ), unittest.mock.patch(
            "app.services.workspace_boot._poll_health"
        ), unittest.mock.patch(
            "app.services.workspace_boot._teardown_container"
        ):
            handle = adapter.boot(Path(self.tmp.name))
            adapter.teardown(handle)

        self.assertEqual(len(provisioned_calls), 1)
        self.assertIs(provisioned_calls[0], caps)

    def test_outcome_deps_overrides_still_accepted(self) -> None:
        """Tests can still inject placeholder collaborators via overrides.

        The existing test seam (``outcome_deps_overrides``) must keep
        working — without it, every outcome-mode unit test would have
        to mock Docker.
        """
        from app.services.outcome_graph_deps import (
            PlaceholderReferenceImplSandbox,
            PlaceholderStarterVerifier,
        )
        from app.services.oracle_pass import OraclePass

        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _ = _build_service(
            self.tmp.name,
            outcome_planner=outcome_planner,
            outcome_deps_overrides={
                "starter_verifier": PlaceholderStarterVerifier(),
                "oracle_pass": OraclePass(
                    sandbox_runner=PlaceholderReferenceImplSandbox()
                ),
            },
        )
        deps = service._build_production_outcome_deps()
        self.assertIsInstance(deps.starter_verifier, PlaceholderStarterVerifier)
        self.assertIsInstance(
            deps.oracle_pass.sandbox_runner, PlaceholderReferenceImplSandbox
        )

    def test_repo_author_adapter_satisfies_protocol(self) -> None:
        """``deps.repo_author.generate_bundle(spec=...)`` returns ``list[tuple[str, str]]``."""
        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _ = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )
        deps = service._build_production_outcome_deps()
        spec = _spec()
        files = deps.repo_author.generate_bundle(spec=spec)
        self.assertIsInstance(files, list)
        # Either non-empty (placeholder synthesizes a starter shell) or
        # empty (the LLM service refused and the adapter degrades);
        # either way every element must be a (str, str) tuple so the
        # graph's ``materialize_starter(workspace_root, files)`` works.
        for entry in files:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)
            self.assertIsInstance(entry[0], str)
            self.assertIsInstance(entry[1], str)
        # The adapter also accepts a ``failure_context`` kwarg for the
        # graph's starter_repair node.
        repaired = deps.repo_author.generate_bundle(
            spec=spec, failure_context={"findings": []}
        )
        self.assertIsInstance(repaired, list)

    def test_starter_verifier_adapter_satisfies_protocol(self) -> None:
        """``deps.starter_verifier.verify_starter(dir)`` returns a dict with an ``ok`` key."""
        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _ = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )
        deps = service._build_production_outcome_deps()
        # A bogus path is fine for the protocol contract — adapter must
        # return a dict no matter what.
        result = deps.starter_verifier.verify_starter(Path(self.tmp.name))
        self.assertIsInstance(result, dict)
        # Every dict returned by ``starter_verifier`` MUST carry an
        # ``ok`` key so the graph's pass/fail branch is deterministic.
        self.assertIn("ok", result)

    def test_resume_with_real_deps_advances_past_gate_1(self) -> None:
        """Resume after gate 1 uses production deps; state advances past awaiting_gate_1.

        We don't run a real LLM, sandbox, or oracle here — the deps the
        service builds carry placeholder adapters that block gracefully
        rather than crashing the dispatcher. The assertion: ``stage``
        is no longer ``awaiting_gate_1`` after the resume call.
        """
        from app.domain.workflow import DecisionOutcome, HILGate

        outcome_planner = _FakeOutcomePlannerFactory(spec=_spec())
        service, _ = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )
        request = GenerateCourseFromBriefRequest(
            goal="Build a grounded retrieval service for engineering teams.",
            creator_setup=CreatorCourseSetupInput(),
        )
        kicked = service.generate_course_run(request)
        course_run_id = kicked.course_run.id

        service.resume_outcome_workflow_after_gate(
            course_run_id,
            gate=HILGate.gate_1_spec_review,
            decision=DecisionOutcome.approve,
        )

        loaded = service._load_outcome_state(course_run_id)
        assert loaded is not None
        # Stage MUST have moved beyond gate 1 — the previous bug was
        # an ``AssertionError`` crash in ``node_starter_authoring`` due
        # to a missing ``repo_author`` dep. The new deps either advance
        # cleanly (placeholder produced a bundle), block with a clean
        # ``starter verification not yet wired in production`` reason,
        # or pause at a downstream gate. Any of those proves the
        # dispatcher accepted the deps without crashing.
        self.assertNotEqual(loaded.stage, "awaiting_gate_1")
        # And the persisted state record must reflect the new stage.
        from app.services.course_workflow_service import CourseWorkflowService
        cws: CourseWorkflowService = service.course_workflow_service
        stored = cws.store.get_course_run(course_run_id)
        assert stored is not None
        self.assertEqual(
            stored.payload_json["outcome_state"]["stage"], loaded.stage
        )
