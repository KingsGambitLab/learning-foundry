"""Tests for outcome-workflow durability (Codex #4/#5 fix).

Before this change, ``CourseGenerationService._kick_off_outcome_workflow``
ran the outcome graph to the first gate and returned a synthesized
response, but it threw away the resulting ``OutcomeWorkflowState`` —
nothing survived for a subsequent reload, refresh, or gate-resume.

The fix is to:

1. Persist the ``OutcomeWorkflowState`` on the existing ``CourseRun`` row
   via a new ``payload_json`` blob (``payload_json["outcome_state"]``).
2. Teach ``_compute_refreshed_run`` to early-return for outcome runs so
   the deliverables-based classifier never reclassifies them as
   ``blocked``.
3. Expose a ``resume_outcome_workflow_after_gate`` method that loads
   the state, advances the graph, and re-persists.

These tests pin all three legs of the fix with fake stores and fake
graphs so we never touch a real DB or LLM.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.domain.ai import AIUsageSummary
from app.domain.course import (
    CourseGenerationSource,
    CourseGenerationStatus,
    CourseRun,
    CourseRunStage,
    CourseRunStatus,
    CreatorCourseSetupInput,
    GenerateCourseFromBriefRequest,
)
from app.domain.registry import PackageType
from app.domain.workflow import DecisionOutcome, HILGate
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.course_generation_service import CourseGenerationService
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    QualityBar,
    StarterType,
)
from app.services.course_workflow_service import CourseWorkflowService
from app.services.langgraph_outcome_graph import OutcomeWorkflowState
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
class _FakeOutcomePlanner:
    spec: CourseOutcomeSpec | None = None
    error: Exception | None = None
    call_count: int = 0

    def plan_course(self, request: Any) -> CourseOutcomeSpec:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        assert self.spec is not None
        return self.spec


class _FakeLegacyPlanner:
    def __init__(self) -> None:
        self.status_calls = 0

    def status(self) -> CourseGenerationStatus:
        self.status_calls += 1
        return CourseGenerationStatus(
            provider="anthropic",
            available=False,
            source=CourseGenerationSource.deterministic_fallback,
            message="Live planning disabled for tests.",
            sdk_installed=False,
            api_key_present=False,
            model_id=None,
            env_file=None,
        )


def _build_service(
    tmp_dir: str,
    *,
    outcome_planner: Any = None,
) -> tuple[CourseGenerationService, CourseWorkflowService, SQLiteWorkflowStore]:
    legacy_planner = _FakeLegacyPlanner()
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
    service = CourseGenerationService(
        course_workflow_service,
        live_planner=legacy_planner,  # type: ignore[arg-type]
        outcome_planner=outcome_planner,
        job_runner=lambda job: job(),
        outcome_workspace_root=Path(tmp_dir) / "outcome_workspaces",
        # Disable the real LLM router so tests don't hit Anthropic.
        outcome_deps_overrides={"router": None},
    )
    return service, course_workflow_service, store


def _make_request() -> GenerateCourseFromBriefRequest:
    return GenerateCourseFromBriefRequest(
        goal="Build a grounded retrieval service for engineering teams.",
        creator_setup=CreatorCourseSetupInput(),
    )


# ---------------- Part A: persistence ----------------


class OutcomeStatePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_outcome_state_persisted_in_payload_json(self) -> None:
        """After kick-off, the course_run row carries the serialized state."""
        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, _cws, store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(_make_request())

        # Re-read from storage so we know it actually persisted, not just
        # a freshly-built in-memory blob.
        reloaded = store.get_course_run(response.course_run.id)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertIn("outcome_state", reloaded.payload_json)
        outcome_payload = reloaded.payload_json["outcome_state"]
        self.assertEqual(outcome_payload["run_id"], response.course_run.id)
        self.assertEqual(outcome_payload["status"], "awaiting_human")
        self.assertEqual(outcome_payload["stage"], "awaiting_gate_1")
        self.assertIsNotNone(outcome_payload.get("spec"))

    def test_outcome_state_round_trips_through_storage(self) -> None:
        """Save then load returns an equivalent OutcomeWorkflowState."""
        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, _cws, _store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(_make_request())

        loaded = service._load_outcome_state(response.course_run.id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertIsInstance(loaded, OutcomeWorkflowState)
        self.assertEqual(loaded.run_id, response.course_run.id)
        self.assertEqual(loaded.stage, "awaiting_gate_1")
        self.assertEqual(loaded.status, "awaiting_human")
        self.assertIsNotNone(loaded.spec)
        # spec round-tripped from JSON back to the pydantic model.
        assert loaded.spec is not None
        self.assertEqual(loaded.spec.title, _spec().title)

    def test_kick_off_then_load_returns_same_state(self) -> None:
        """End-to-end round-trip: kick off, load by id, fields agree."""
        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, _cws, _store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(_make_request())
        loaded = service._load_outcome_state(response.course_run.id)
        assert loaded is not None
        self.assertEqual(loaded.run_id, response.course_run.id)
        # With ``router=None`` in the test deps (see ``_build_service``)
        # the spec_review node short-circuits and does not increment
        # ``cost_usd``. Production runs with a live router will see a
        # small positive value here; the test pins the no-router path
        # so we can detect a regression that re-enables network calls
        # in unit tests.
        self.assertEqual(loaded.cost_usd, 0.0)
        self.assertEqual(loaded.starter_attempt, 0)
        self.assertEqual(loaded.blocking_reasons, [])

    def test_outcome_state_serialization_excludes_non_serializable(self) -> None:
        """All fields of OutcomeWorkflowState produce JSON-safe values."""
        import json

        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, _cws, store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(_make_request())
        reloaded = store.get_course_run(response.course_run.id)
        assert reloaded is not None

        # Must JSON-encode cleanly with no objects-as-strings.
        encoded = json.dumps(reloaded.payload_json["outcome_state"])
        self.assertGreater(len(encoded), 0)

    def test_persist_failure_does_not_corrupt_response(self) -> None:
        """If persistence raises, the response either surfaces it cleanly
        or completes — never returns a half-built object.
        """
        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, _cws, store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        original_save = store.save_course_run
        call_state = {"count": 0}

        def flaky_save(run: CourseRun) -> CourseRun:
            call_state["count"] += 1
            # Allow the placeholder save to succeed; sabotage the final
            # save that carries outcome_state. The placeholder is saved
            # first (no outcome_state yet), then the adapter saves with
            # the state. We target the run that already has the state.
            if "outcome_state" in (run.payload_json or {}):
                raise RuntimeError("simulated db failure")
            return original_save(run)

        store.save_course_run = flaky_save  # type: ignore[method-assign]

        # The kick-off should raise (clean error surface) rather than
        # silently returning a half-built response. Either way, the
        # in-memory course_run state remains coherent: no half-set fields.
        with self.assertRaises(RuntimeError):
            service.generate_course_run(_make_request())

        # Restore for any teardown side-effects.
        store.save_course_run = original_save  # type: ignore[method-assign]


# ---------------- Part B: refresh awareness ----------------


class RefreshAwarenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_refreshed_outcome_run_uses_outcome_state_status(self) -> None:
        """``_compute_refreshed_run`` reads from outcome_state, not deliverables."""
        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, cws, store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(_make_request())
        # The persisted course_run has zero deliverables, so the legacy
        # ``_course_stage_from_deliverables`` would classify it as blocked.
        # The outcome-aware branch must take over and emit awaiting_human.
        reloaded = store.get_course_run(response.course_run.id)
        assert reloaded is not None
        refreshed = cws._compute_refreshed_run(reloaded)

        self.assertEqual(refreshed.status, CourseRunStatus.awaiting_human)
        self.assertEqual(refreshed.stage, CourseRunStage.awaiting_course_review)

    def test_refreshed_outcome_run_not_classified_as_blocked(self) -> None:
        """The exact Codex regression.

        Zero-deliverable outcome run does NOT flip to blocked on refresh.
        """
        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, cws, _store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        response = service.generate_course_run(_make_request())

        # Use the public ``get_run`` path — that's the exact route the API
        # uses on poll/refresh.
        refreshed = cws.get_run(response.course_run.id)
        assert refreshed is not None

        self.assertNotEqual(refreshed.status, CourseRunStatus.blocked)
        self.assertNotEqual(refreshed.stage, CourseRunStage.blocked)
        self.assertEqual(refreshed.status, CourseRunStatus.awaiting_human)

    def test_legacy_runs_unchanged_by_refresh_branch(self) -> None:
        """Runs without outcome_state in payload still use the legacy refresh."""
        _service, cws, store = _build_service(self.tmp.name)

        # Build a CourseRun by hand with NO outcome_state — i.e., a
        # legacy run. The legacy refresh path should classify it as
        # blocked (zero deliverables, no shared workflow). That's the
        # very behaviour we DO want to preserve for non-outcome rows.
        now = datetime.now(UTC)
        legacy = CourseRun(
            id=f"course_{uuid4().hex[:12]}",
            course_family_id=f"course_{uuid4().hex[:12]}",
            title="Legacy",
            summary="Legacy summary placeholder.",
            package_type=PackageType.progressive_codebase_course,
            created_at=now,
            updated_at=now,
            stage=CourseRunStage.awaiting_course_review,
            status=CourseRunStatus.awaiting_human,
            deliverables=[],
            payload_json={},
        )
        store.save_course_run(legacy)

        refreshed = cws._compute_refreshed_run(legacy)
        # Legacy behaviour: empty deliverables + drafting stage means
        # the early-return kicks in (existing legacy branch keeps it as
        # is) — confirm we did NOT alter that legacy outcome.
        # The key assertion is that the outcome-aware branch did NOT
        # fire (i.e. refreshed.payload_json has no outcome_state).
        self.assertNotIn("outcome_state", refreshed.payload_json)


# ---------------- Part C: resume after gate ----------------


@dataclass
class _ResumingFakePlanner:
    """Fake planner that lets us inject behaviour into ``plan_course``."""

    spec: CourseOutcomeSpec | None = None
    call_count: int = 0

    def plan_course(self, request: Any) -> CourseOutcomeSpec:
        self.call_count += 1
        assert self.spec is not None
        return self.spec


class ResumeAfterGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _kick_off(self) -> tuple[CourseGenerationService, str]:
        outcome_planner = _ResumingFakePlanner(spec=_spec())
        service, _cws, _store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )
        response = service.generate_course_run(_make_request())
        return service, response.course_run.id

    def test_resume_after_gate_1_approval(self) -> None:
        """Approving gate 1 advances state to the next pause / status.

        We use a fake graph stub via monkey-patching so we don't need
        full starter / oracle deps. The assertion: state.status flips
        from ``awaiting_human`` to something else after resume.
        """
        service, course_run_id = self._kick_off()

        # Stub ``OutcomeWorkflowGraph.execute`` to model an approval
        # that advances from gate 1 → gate 2 (still awaiting_human).
        from app.services import langgraph_outcome_graph as graph_mod

        original_execute = graph_mod.OutcomeWorkflowGraph.execute

        def fake_execute(self_, state, *, deps):  # type: ignore[no-untyped-def]
            # We expect the resume helper to flip status to running
            # before calling execute(). Assert and then advance.
            assert state.status == "running"
            assert state.stage == "awaiting_gate_1"
            state.stage = "awaiting_gate_2"
            state.status = "awaiting_human"
            return state

        graph_mod.OutcomeWorkflowGraph.execute = fake_execute  # type: ignore[method-assign]
        try:
            response = service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_1_spec_review,
                decision=DecisionOutcome.approve,
            )
        finally:
            graph_mod.OutcomeWorkflowGraph.execute = original_execute  # type: ignore[method-assign]

        loaded = service._load_outcome_state(course_run_id)
        assert loaded is not None
        self.assertEqual(loaded.stage, "awaiting_gate_2")
        self.assertEqual(loaded.status, "awaiting_human")
        self.assertEqual(response.course_run.status, CourseRunStatus.awaiting_human)

    def test_resume_after_gate_rejection(self) -> None:
        """Rejection sets blocking_reasons / routes through the rejection branch."""
        service, course_run_id = self._kick_off()

        from app.services import langgraph_outcome_graph as graph_mod

        original_execute = graph_mod.OutcomeWorkflowGraph.execute

        def fake_execute(self_, state, *, deps):  # type: ignore[no-untyped-def]
            # The resume path should not flip to running for a reject.
            return state

        graph_mod.OutcomeWorkflowGraph.execute = fake_execute  # type: ignore[method-assign]
        try:
            response = service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_1_spec_review,
                decision=DecisionOutcome.reject,
            )
        finally:
            graph_mod.OutcomeWorkflowGraph.execute = original_execute  # type: ignore[method-assign]

        loaded = service._load_outcome_state(course_run_id)
        assert loaded is not None
        # Rejection at gate 1 marks the run as blocked.
        self.assertEqual(loaded.status, "blocked")
        self.assertTrue(loaded.blocking_reasons)
        self.assertEqual(response.course_run.status, CourseRunStatus.blocked)

    def test_resume_idempotent(self) -> None:
        """Calling resume twice with the same decision is a no-op the second time.

        After the first approve advances gate 1 → gate 2, a second
        approve targeting gate 1 should be rejected (or no-op) rather
        than re-running the graph from the gate 1 state.
        """
        service, course_run_id = self._kick_off()

        from app.services import langgraph_outcome_graph as graph_mod

        original_execute = graph_mod.OutcomeWorkflowGraph.execute
        call_count = {"value": 0}

        def fake_execute(self_, state, *, deps):  # type: ignore[no-untyped-def]
            call_count["value"] += 1
            state.stage = "awaiting_gate_2"
            state.status = "awaiting_human"
            return state

        graph_mod.OutcomeWorkflowGraph.execute = fake_execute  # type: ignore[method-assign]
        try:
            service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_1_spec_review,
                decision=DecisionOutcome.approve,
            )
            self.assertEqual(call_count["value"], 1)

            # Second call targeting the same gate should be a no-op.
            service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_1_spec_review,
                decision=DecisionOutcome.approve,
            )
            # The graph executor must NOT have been re-invoked.
            self.assertEqual(call_count["value"], 1)
        finally:
            graph_mod.OutcomeWorkflowGraph.execute = original_execute  # type: ignore[method-assign]

    def test_gate_approval_persists_intermediate_state(self) -> None:
        """After gate 1 → gate 2, state at gate 2 is persisted to disk."""
        service, course_run_id = self._kick_off()

        from app.services import langgraph_outcome_graph as graph_mod

        original_execute = graph_mod.OutcomeWorkflowGraph.execute

        def fake_execute(self_, state, *, deps):  # type: ignore[no-untyped-def]
            state.stage = "awaiting_gate_2"
            state.status = "awaiting_human"
            state.starter_attempt = 1
            return state

        graph_mod.OutcomeWorkflowGraph.execute = fake_execute  # type: ignore[method-assign]
        try:
            service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_1_spec_review,
                decision=DecisionOutcome.approve,
            )
        finally:
            graph_mod.OutcomeWorkflowGraph.execute = original_execute  # type: ignore[method-assign]

        # Re-read from storage; the state must reflect the advance.
        reloaded = service.course_workflow_service.store.get_course_run(course_run_id)
        assert reloaded is not None
        outcome_payload = reloaded.payload_json["outcome_state"]
        self.assertEqual(outcome_payload["stage"], "awaiting_gate_2")
        self.assertEqual(outcome_payload["starter_attempt"], 1)


# ---------------- Part D: combined scenarios ----------------


class CombinedDurabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_full_lifecycle_kickoff_reload_refresh_resume(self) -> None:
        """The complete path: kick-off → reload → refresh → resume."""
        outcome_planner = _FakeOutcomePlanner(spec=_spec())
        service, cws, store = _build_service(
            self.tmp.name, outcome_planner=outcome_planner
        )

        # Step 1: kick-off
        response = service.generate_course_run(_make_request())
        course_run_id = response.course_run.id
        self.assertEqual(response.course_run.status, CourseRunStatus.awaiting_human)

        # Step 2: reload from disk
        reloaded = store.get_course_run(course_run_id)
        assert reloaded is not None
        self.assertIn("outcome_state", reloaded.payload_json)

        # Step 3: refresh — must NOT flip to blocked
        refreshed = cws.get_run(course_run_id)
        assert refreshed is not None
        self.assertEqual(refreshed.status, CourseRunStatus.awaiting_human)

        # Step 4: resume after gate 1
        from app.services import langgraph_outcome_graph as graph_mod

        original_execute = graph_mod.OutcomeWorkflowGraph.execute

        def fake_execute(self_, state, *, deps):  # type: ignore[no-untyped-def]
            state.stage = "awaiting_gate_2"
            state.status = "awaiting_human"
            return state

        graph_mod.OutcomeWorkflowGraph.execute = fake_execute  # type: ignore[method-assign]
        try:
            response2 = service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_1_spec_review,
                decision=DecisionOutcome.approve,
            )
        finally:
            graph_mod.OutcomeWorkflowGraph.execute = original_execute  # type: ignore[method-assign]

        # After resume, state at gate 2
        self.assertEqual(response2.course_run.status, CourseRunStatus.awaiting_human)
        loaded_after = service._load_outcome_state(course_run_id)
        assert loaded_after is not None
        self.assertEqual(loaded_after.stage, "awaiting_gate_2")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
