"""Tests for the course-run gate-decision API route (Codex review #6 finding #1).

Before this change, the only gate-decision route — ``POST
/v1/workflow-runs/{run_id}/decisions`` — operated on a *workflow*
run-id. Outcome runs are *course-run* scoped, so HTTP clients had no
way to send an approve / reject to advance a paused outcome run past
gate 1. The fix adds ``POST /v1/course-runs/{course_run_id}/decisions``
which dispatches to ``CourseGenerationService.resume_outcome_workflow_after_gate``.

These tests pin the route's contract: it 404s on unknown ids, 409s
when the run isn't paused at a gate, and delegates to the service for
valid decisions. The legacy workflow-run route is exercised indirectly
to confirm it still uses the existing ``WorkflowService.apply_gate_decision``
path.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.domain.course import (
    CourseGenerationSource,
    CourseGenerationStatus,
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
    call_count: int = 0

    def plan_course(self, request: Any) -> CourseOutcomeSpec:
        self.call_count += 1
        assert self.spec is not None
        return self.spec


class _FakeLegacyPlanner:
    """Drop-in replacement for ``OpenAICoursePlanner`` (unavailable)."""

    def status(self) -> CourseGenerationStatus:
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


def _build_app(tmp_dir: str) -> tuple[FastAPI, CourseGenerationService, str]:
    """Build a FastAPI app with just enough state for the gate-decision route.

    Returns the app, the service, and the course_run_id of a freshly
    kicked-off outcome run that is paused at gate 1 awaiting human
    decision.
    """
    legacy_planner = _FakeLegacyPlanner()
    outcome_planner = _FakeOutcomePlanner(spec=_spec())
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

    request = GenerateCourseFromBriefRequest(
        goal="Build a grounded retrieval service for engineering teams.",
        creator_setup=CreatorCourseSetupInput(),
    )
    kicked = service.generate_course_run(request)

    app = FastAPI()
    app.include_router(router)
    # Only attach the bits the gate-decision route reads. Anything else
    # the router declares but the test doesn't hit stays unset; FastAPI
    # won't read it unless that route fires.
    app.state.workflow_service = workflow_service
    app.state.course_workflow_service = course_workflow_service
    app.state.course_generation_service = service

    return app, service, kicked.course_run.id


# ---------------- Route: course-run gate decision ----------------


class CourseRunGateDecisionRouteTests(unittest.TestCase):
    """End-to-end tests for ``POST /v1/course-runs/{course_run_id}/decisions``."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_course_run_gate_decision_endpoint_approves(self) -> None:
        """POST gate-1 approve → service.resume_outcome_workflow_after_gate is invoked."""
        app, service, course_run_id = _build_app(self.tmp.name)

        # Use MagicMock to verify the service method is invoked exactly
        # once with the right args. Stub it to return a synthetic
        # response so we don't drive the graph.
        from app.domain.course import (
            CreateCourseDeliverableRequest,
            GenerateCourseFromBriefResponse,
            GeneratedCoursePlan,
        )

        stub_response = GenerateCourseFromBriefResponse(
            source=CourseGenerationSource.deterministic_fallback,
            status=service.live_planner.status(),
            plan=GeneratedCoursePlan(
                title="t",
                summary="s" * 20,
                package_type=PackageType.progressive_codebase_course,
                deliverables=[
                    CreateCourseDeliverableRequest(
                        deliverable_slug="outcome",
                        title="t",
                        summary="s" * 20,
                    )
                ],
            ),
            course_run=service.course_workflow_service.store.get_course_run(
                course_run_id
            ),
            review=service.course_workflow_service.review_run(course_run_id),
        )
        service.resume_outcome_workflow_after_gate = MagicMock(  # type: ignore[method-assign]
            return_value=stub_response
        )

        with TestClient(app) as client:
            resp = client.post(
                f"/v1/course-runs/{course_run_id}/decisions",
                json={
                    "gate": "gate_1_spec_review",
                    "decision": "approve",
                },
            )

        self.assertEqual(resp.status_code, 200)
        service.resume_outcome_workflow_after_gate.assert_called_once()
        _args, kwargs = service.resume_outcome_workflow_after_gate.call_args
        # Allow either positional or keyword passing of course_run_id;
        # in either case the gate/decision must be the right enum
        # values (the route may pass strings — both shapes are
        # acceptable since ``resume_outcome_workflow_after_gate``
        # tolerates both).
        gate_arg = kwargs.get("gate") or service.resume_outcome_workflow_after_gate.call_args[0][1] if len(service.resume_outcome_workflow_after_gate.call_args[0]) > 1 else kwargs.get("gate")
        decision_arg = kwargs.get("decision")
        # Accept either enum or string.
        gate_value = gate_arg.value if hasattr(gate_arg, "value") else gate_arg
        decision_value = decision_arg.value if hasattr(decision_arg, "value") else decision_arg
        self.assertEqual(gate_value, HILGate.gate_1_spec_review.value)
        self.assertEqual(decision_value, DecisionOutcome.approve.value)

    def test_course_run_gate_decision_endpoint_404_on_unknown_id(self) -> None:
        """An unknown course_run_id returns 404."""
        app, _service, _course_run_id = _build_app(self.tmp.name)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/course-runs/does-not-exist/decisions",
                json={
                    "gate": "gate_1_spec_review",
                    "decision": "approve",
                },
            )
        self.assertEqual(resp.status_code, 404)

    def test_course_run_gate_decision_endpoint_409_when_not_awaiting(self) -> None:
        """A course_run that is not in awaiting_human at any gate returns 409."""
        app, service, course_run_id = _build_app(self.tmp.name)

        # Mutate the persisted state into a non-awaiting status so the
        # service refuses to advance.
        run = service.course_workflow_service.store.get_course_run(course_run_id)
        assert run is not None
        # Forcefully overwrite the outcome_state to something terminal.
        blob = dict(run.payload_json.get("outcome_state", {}))
        blob["status"] = "published"
        blob["stage"] = "published"
        run.payload_json = {**run.payload_json, "outcome_state": blob}
        service.course_workflow_service.store.save_course_run(run)

        with TestClient(app) as client:
            resp = client.post(
                f"/v1/course-runs/{course_run_id}/decisions",
                json={
                    "gate": "gate_1_spec_review",
                    "decision": "approve",
                },
            )
        self.assertEqual(resp.status_code, 409)

    def test_course_run_gate_decision_endpoint_404_for_non_outcome_run(self) -> None:
        """A course_run with no outcome_state blob is not an outcome run; 404."""
        app, service, _course_run_id = _build_app(self.tmp.name)

        # Build a legacy-shaped course_run by hand with no ``outcome_state``
        # under ``payload_json``.
        from datetime import UTC, datetime
        from app.domain.course import (
            CourseRun,
            CourseRunStage,
            CourseRunStatus,
        )

        legacy = CourseRun(
            id="legacy-course-run",
            course_family_id="legacy-course-run",
            title="Legacy run",
            summary="A legacy progressive course run; not outcome mode.",
            package_type=PackageType.progressive_codebase_course,
            stage=CourseRunStage.drafting,
            status=CourseRunStatus.active,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        service.course_workflow_service.store.save_course_run(legacy)

        with TestClient(app) as client:
            resp = client.post(
                "/v1/course-runs/legacy-course-run/decisions",
                json={
                    "gate": "gate_1_spec_review",
                    "decision": "approve",
                },
            )
        # 404 — the course-run route is outcome-specific and a legacy
        # row is "unknown" from its perspective.
        self.assertEqual(resp.status_code, 404)


# ---------------- Route: legacy workflow-run path unchanged ----------------


class LegacyWorkflowRunDecisionRouteTests(unittest.TestCase):
    """The legacy ``/v1/workflow-runs/{run_id}/decisions`` route is unchanged.

    Codex review #6 asked for the new course-run route to *coexist*
    with the legacy one. This test confirms that posting a gate
    decision against an unknown workflow run id still hits the
    ``WorkflowService.apply_gate_decision`` path and returns 404 —
    not the new ``resume_outcome_workflow_after_gate`` path. (The full
    happy-path test for the legacy route lives in test_api.py and is
    not duplicated here.)
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_legacy_workflow_run_decisions_unchanged(self) -> None:
        app, service, _course_run_id = _build_app(self.tmp.name)

        # Wire a MagicMock onto the service so we can detect if the
        # course-run path was incorrectly fired.
        service.resume_outcome_workflow_after_gate = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("legacy route must not dispatch to outcome")
        )

        with TestClient(app) as client:
            resp = client.post(
                "/v1/workflow-runs/does-not-exist/decisions",
                json={
                    "gate": "gate_1_spec_review",
                    "decision": "approve",
                },
            )
        # The legacy ``apply_gate_decision`` raises KeyError → 404.
        self.assertEqual(resp.status_code, 404)
        # The outcome-mode resume helper was NOT consulted.
        service.resume_outcome_workflow_after_gate.assert_not_called()
