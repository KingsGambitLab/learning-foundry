"""Tests for Codex review #7 critical: legacy gate execution-evidence guard.

Background
----------
`WorkflowService.apply_gate_decision` used to advance a legacy workflow
run from gate 1 → gate 2 → gate 3 → ``published`` while only validating
the task-agent spec at gate 1. After Wave 5b's deletion of the
per-deliverable LangGraph node loop, gate 2 and gate 3 never run the
sandbox / tests / reviewer that used to gate publish, so a legacy run
could reach ``published`` with no execution evidence at all.

This module pins the new guard: the legacy approve path now refuses to
advance a run unless it has clear execution evidence
(``node_executions`` with at least one passed runtime/tests/reviewer
node, OR a materialized ``workspace_snapshot``). The reject path is
unchanged so the publish-certification recovery flow in
``course_workflow_service._route_publish_failure_to_shared_workflow_revision``
keeps working.

Outcome-mode runs are unaffected — they advance through the
``/v1/course-runs/{id}/decisions`` route, not through
``apply_gate_decision``.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.domain.workflow import (
    DecisionOutcome,
    DraftKind,
    GateDecisionRequest,
    HILGate,
    MaterializedBundle,
    WorkflowArtifacts,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowNodeStatus,
    WorkflowRun,
    WorkflowStage,
    WorkflowStatus,
)
from app.services.assignment_design_inference import GenerationIntake
from app.services.workflow_service import (
    WorkflowGateRefused,
    WorkflowService,
)
from app.storage.sqlite_store import SQLiteWorkflowStore


# ---------------- Fixture helpers ----------------


def _intake() -> GenerationIntake:
    return GenerationIntake(
        title="Sample legacy workflow",
        problem_statement=(
            "Build a small service that ingests requests, calls a tool, "
            "and produces a structured reply for the calling agent."
        ),
        learning_outcomes=["service contract"],
    )


def _run_at_gate(
    *,
    pending_gate: HILGate,
    stage: WorkflowStage,
    node_executions: list[WorkflowNodeExecution] | None = None,
    workspace_snapshot: MaterializedBundle | None = None,
    valid_spec: bool = True,
) -> WorkflowRun:
    """Hand-craft a legacy WorkflowRun for guard testing.

    The default shape mirrors a fresh post-create run: a valid task-agent
    spec (so gate 1 schema-validation passes), no node_executions and no
    workspace_snapshot (so the new execution-evidence guard refuses).
    """
    now = datetime.now(UTC)
    artifacts = WorkflowArtifacts(
        draft_kind=DraftKind.task_agent_spec,
        validation_summary={"valid": valid_spec},
        node_executions=node_executions or [],
        workspace_snapshot=workspace_snapshot,
        # ``task_agent_spec`` stays None — every test in this module
        # exercises gate 2 or gate 3, where ``apply_gate_decision``
        # does not read the spec. The new execution-evidence guard is
        # the only thing under test here.
        task_agent_spec=None,
    )
    return WorkflowRun(
        id=f"run_test_{pending_gate.value}",
        title="Sample legacy workflow",
        created_at=now,
        updated_at=now,
        stage=stage,
        status=WorkflowStatus.awaiting_human,
        pending_gate=pending_gate,
        intake=_intake(),
        artifacts=artifacts,
        notes=[],
    )


def _passed_runtime_node() -> WorkflowNodeExecution:
    return WorkflowNodeExecution(
        node_id="node_authoring_runtime_1",
        kind=WorkflowNodeKind.authoring_runtime,
        iteration=1,
        attempt=1,
        status=WorkflowNodeStatus.passed,
        summary="Runtime came up cleanly for all deliverables.",
        created_at=datetime.now(UTC),
    )


def _failed_runtime_node() -> WorkflowNodeExecution:
    return WorkflowNodeExecution(
        node_id="node_authoring_runtime_1",
        kind=WorkflowNodeKind.authoring_runtime,
        iteration=1,
        attempt=1,
        status=WorkflowNodeStatus.failed,
        summary="Runtime did not come up.",
        created_at=datetime.now(UTC),
    )


def _materialized_workspace() -> MaterializedBundle:
    return MaterializedBundle(
        bundle_id="workspace_snapshot_1",
        generated_at=datetime.now(UTC),
        root_dir="/tmp/workspace",
        public_dir="/tmp/workspace/public",
        private_dir="/tmp/workspace/private",
        manifest_path="/tmp/workspace/manifest.json",
        files=[],
    )


def _build_service(tmp_dir: str) -> WorkflowService:
    store = SQLiteWorkflowStore(db_path=f"{tmp_dir}/test.db")
    return WorkflowService(store)


# ---------------- Unit tests: _run_has_execution_evidence ----------------


class RunHasExecutionEvidenceTests(unittest.TestCase):
    """Direct tests for the new ``_run_has_execution_evidence`` helper."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service = _build_service(self.tmp.name)

    def test_returns_false_for_fresh_placeholder_run(self) -> None:
        """A fresh run has no node_executions and no workspace_snapshot."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
        )
        self.assertFalse(self.service._run_has_execution_evidence(run))

    def test_returns_true_after_any_node_passed(self) -> None:
        """At least one passed runtime/tests/reviewer node counts as evidence."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
            node_executions=[_passed_runtime_node()],
        )
        self.assertTrue(self.service._run_has_execution_evidence(run))

    def test_returns_false_when_only_failed_nodes_present(self) -> None:
        """Failed nodes alone are not evidence — sandbox did not pass."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
            node_executions=[_failed_runtime_node()],
        )
        self.assertFalse(self.service._run_has_execution_evidence(run))

    def test_returns_true_when_workspace_snapshot_has_files(self) -> None:
        """A materialized workspace counts as execution evidence."""
        # A workspace_snapshot is created by ``materialize`` after a
        # successful build; treat its presence as evidence.
        snapshot = _materialized_workspace()
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
            workspace_snapshot=snapshot,
        )
        self.assertTrue(self.service._run_has_execution_evidence(run))


# ---------------- apply_gate_decision: advancement guard ----------------


class ApplyGateDecisionGuardTests(unittest.TestCase):
    """Pin the new ``WorkflowGateRefused`` behaviour on the approve path."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.service = _build_service(self.tmp.name)

    def test_approve_gate_2_refused_without_execution_evidence(self) -> None:
        """A run with no execution cannot advance past gate 1."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
        )
        self.service.store.save_run(run)

        decision = GateDecisionRequest(
            gate=HILGate.gate_2_progression_review,
            decision=DecisionOutcome.approve,
        )

        with self.assertRaises(WorkflowGateRefused):
            self.service.apply_gate_decision(run.id, decision)

        # State is unchanged — the guard refuses before persisting.
        reloaded = self.service.store.get_run(run.id)
        assert reloaded is not None
        self.assertEqual(reloaded.stage, WorkflowStage.awaiting_hil_gate_2)
        self.assertEqual(reloaded.pending_gate, HILGate.gate_2_progression_review)

    def test_approve_gate_3_refused_without_execution_evidence(self) -> None:
        """Gate 3 advancement to ``published`` is the publish-without-execution risk."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_3_pre_publish,
            stage=WorkflowStage.awaiting_hil_gate_3,
        )
        self.service.store.save_run(run)

        decision = GateDecisionRequest(
            gate=HILGate.gate_3_pre_publish,
            decision=DecisionOutcome.approve,
        )

        with self.assertRaises(WorkflowGateRefused):
            self.service.apply_gate_decision(run.id, decision)

        reloaded = self.service.store.get_run(run.id)
        assert reloaded is not None
        self.assertNotEqual(reloaded.stage, WorkflowStage.published)
        self.assertNotEqual(reloaded.status, WorkflowStatus.published)

    def test_approve_allowed_with_passed_node_execution(self) -> None:
        """A passed runtime node is execution evidence; advancement proceeds."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
            node_executions=[_passed_runtime_node()],
        )
        self.service.store.save_run(run)

        decision = GateDecisionRequest(
            gate=HILGate.gate_2_progression_review,
            decision=DecisionOutcome.approve,
        )

        result = self.service.apply_gate_decision(run.id, decision)
        self.assertEqual(result.stage, WorkflowStage.awaiting_hil_gate_3)
        self.assertEqual(result.pending_gate, HILGate.gate_3_pre_publish)

    def test_approve_allowed_with_workspace_snapshot(self) -> None:
        """A materialized workspace_snapshot is execution evidence."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
            workspace_snapshot=_materialized_workspace(),
        )
        self.service.store.save_run(run)

        decision = GateDecisionRequest(
            gate=HILGate.gate_2_progression_review,
            decision=DecisionOutcome.approve,
        )

        result = self.service.apply_gate_decision(run.id, decision)
        self.assertEqual(result.stage, WorkflowStage.awaiting_hil_gate_3)

    def test_reject_path_unaffected_by_guard(self) -> None:
        """Reject must keep working — it powers the publish-certification recovery flow."""
        run = _run_at_gate(
            pending_gate=HILGate.gate_3_pre_publish,
            stage=WorkflowStage.awaiting_hil_gate_3,
        )
        self.service.store.save_run(run)

        decision = GateDecisionRequest(
            gate=HILGate.gate_3_pre_publish,
            decision=DecisionOutcome.reject,
            comment="Certification failed; reopen for revision.",
        )

        # Should NOT raise. The reject path doesn't advance toward
        # publish, so it doesn't need execution evidence.
        result = self.service.apply_gate_decision(run.id, decision)
        self.assertEqual(result.stage, WorkflowStage.needs_revision)


# ---------------- HTTP layer: 409 mapping ----------------


class WorkflowGateRefusedHttpStatusTests(unittest.TestCase):
    """Verify ``WorkflowGateRefused`` maps to HTTP 409 on the legacy route."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_legacy_decisions_route_returns_409_when_evidence_missing(self) -> None:
        service = _build_service(self.tmp.name)
        run = _run_at_gate(
            pending_gate=HILGate.gate_2_progression_review,
            stage=WorkflowStage.awaiting_hil_gate_2,
        )
        service.store.save_run(run)

        app = FastAPI()
        app.include_router(router)
        app.state.workflow_service = service

        with TestClient(app) as client:
            resp = client.post(
                f"/v1/workflow-runs/{run.id}/decisions",
                json={
                    "gate": "gate_2_progression_review",
                    "decision": "approve",
                },
            )

        self.assertEqual(resp.status_code, 409)
        body = resp.json()
        self.assertIn("detail", body)
        # The detail should mention execution evidence so the caller
        # has a clue why the gate is refusing.
        self.assertIn("execution", body["detail"].lower())


if __name__ == "__main__":
    unittest.main()
