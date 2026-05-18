"""Tests for outcome-run review report surfacing (Codex #6 finding #3).

Before this fix, ``_build_review_report()`` only derived blockers from
``course_run.last_error`` + persisted deliverables + linked workflow runs.
Outcome runs persist their findings in
``course_run.payload_json["outcome_state"]``
(``spec_review_findings``, ``starter_review_findings``,
``oracle_validation_report.blocking_reasons``,
``curated_validation_report.blocking_reasons``, and the top-level
``blocking_reasons``) — none of which the legacy review path consulted.

The fix is to detect outcome runs and surface their findings via a new
``outcome_findings`` field on ``CourseReviewReport`` while also folding
the same information into the legacy ``blockers`` / ``next_actions``
fields so existing consumers continue to see something actionable.

These tests pin the contract with hand-crafted ``payload_json`` blobs
carrying a serialized ``OutcomeWorkflowState`` — no real LLM, no real
Docker, no real SQLite (a tiny fake store keeps the boot fast).
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.domain.course import (
    CourseAsyncOperation,
    CourseRun,
    CourseRunStage,
    CourseRunStatus,
)
from app.domain.registry import PackageType
from app.domain.workflow import (
    ReviewerFinding,
    ReviewerFindingSeverity,
)
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
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
from app.services.oracle_validation import OracleValidationReport
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


def _build_outcome_state(
    *,
    run_id: str,
    workspace_root: Path,
    spec_review_findings: list[ReviewerFinding] | None = None,
    starter_review_findings: list[ReviewerFinding] | None = None,
    oracle_validation_report: OracleValidationReport | None = None,
    curated_validation_report: OracleValidationReport | None = None,
    blocking_reasons: list[str] | None = None,
    stage: str = "awaiting_gate_1",
    status: str = "awaiting_human",
) -> OutcomeWorkflowState:
    return OutcomeWorkflowState(
        run_id=run_id,
        workspace_root=workspace_root,
        spec=_spec(),
        spec_review_findings=spec_review_findings or [],
        starter_review_findings=starter_review_findings or [],
        oracle_validation_report=oracle_validation_report,
        curated_validation_report=curated_validation_report,
        blocking_reasons=blocking_reasons or [],
        stage=stage,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
    )


def _course_run_with_outcome_state(
    state: OutcomeWorkflowState | None,
    *,
    stage: CourseRunStage = CourseRunStage.awaiting_course_review,
    status: CourseRunStatus = CourseRunStatus.awaiting_human,
) -> CourseRun:
    """Hand-craft a CourseRun with a serialized outcome state in payload_json.

    ``state`` may be ``None`` to model a legacy (non-outcome) run with an
    empty payload_json.
    """
    now = datetime.now(UTC)
    payload_json: dict[str, Any] = {}
    if state is not None:
        payload_json["outcome_state"] = state.model_dump(mode="json")
    return CourseRun(
        id=f"course_{uuid4().hex[:12]}",
        course_family_id=f"family_{uuid4().hex[:12]}",
        title="Outcome RAG course",
        summary="Outcome-mode RAG course for review surfacing tests.",
        package_type=PackageType.progressive_codebase_course,
        created_at=now,
        updated_at=now,
        stage=stage,
        status=status,
        deliverables=[],
        payload_json=payload_json,
    )


def _build_service(tmp_dir: str) -> CourseWorkflowService:
    store = SQLiteWorkflowStore(db_path=f"{tmp_dir}/test.db")
    workspace_manager = AssignmentWorkspaceManager(base_dir=f"{tmp_dir}/workspaces")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{tmp_dir}/generated"),
        runner=TaskAgentBlackBoxRunner(),
        workspace_manager=workspace_manager,
    )
    return CourseWorkflowService(
        store,
        workflow_service,
        job_runner=lambda job: job(),
    )


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace_root = Path(self.tmp.name) / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.service = _build_service(self.tmp.name)

    def _save(self, course_run: CourseRun) -> None:
        self.service.store.save_course_run(course_run)


# ---------------- Tests ----------------


class OutcomeRunReviewReportTests(_Base):
    def test_outcome_run_review_surfaces_spec_review_findings(self) -> None:
        """A spec-review error in outcome_state appears in the review report."""
        finding = ReviewerFinding(
            category="spec_coherence",
            severity=ReviewerFindingSeverity.error,
            title="Spec missing abstention scenario",
            detail="Quality bar mandates refusal coverage but no out-of-scope scenario was authored.",
            hint="Add an out_of_scope scenario to the spec before approving gate 1.",
        )
        state = _build_outcome_state(
            run_id="r1",
            workspace_root=self.workspace_root,
            spec_review_findings=[finding],
        )
        course_run = _course_run_with_outcome_state(state)
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        self.assertIsNotNone(review.outcome_findings)
        assert review.outcome_findings is not None
        self.assertEqual(len(review.outcome_findings.spec_review), 1)
        self.assertEqual(
            review.outcome_findings.spec_review[0].title,
            "Spec missing abstention scenario",
        )
        # The error severity also bubbles into legacy ``blockers``.
        self.assertTrue(
            any(
                "Spec missing abstention scenario" in blocker
                for blocker in review.blockers
            )
        )

    def test_outcome_run_review_surfaces_starter_review_findings(self) -> None:
        """A starter-review error in outcome_state appears in the review report."""
        finding = ReviewerFinding(
            category="starter_verify",
            severity=ReviewerFindingSeverity.error,
            title="Starter sandbox failed during build",
            detail="Pip install crashed: ModuleNotFoundError: requests",
            hint="Add 'requests' to requirements.txt before retrying verification.",
        )
        state = _build_outcome_state(
            run_id="r2",
            workspace_root=self.workspace_root,
            starter_review_findings=[finding],
        )
        course_run = _course_run_with_outcome_state(state)
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        self.assertIsNotNone(review.outcome_findings)
        assert review.outcome_findings is not None
        self.assertEqual(len(review.outcome_findings.starter_review), 1)
        self.assertEqual(
            review.outcome_findings.starter_review[0].category,
            "starter_verify",
        )
        # Hint flows into next_actions for the repair LLM / UI.
        self.assertTrue(
            any(
                "requirements.txt" in action
                for action in review.next_actions
            )
        )

    def test_outcome_run_review_surfaces_oracle_validation_blockers(self) -> None:
        """oracle_validation_report's blocking_reasons appear in the review."""
        report = OracleValidationReport(
            publishable=False,
            reference_impl_hash="abc123",
            scenario_set_hash="def456",
            blocking_reasons=[
                "Scenario 'happy_path_01' failed during reference replay.",
                "Missing required category: malformed_input.",
            ],
        )
        state = _build_outcome_state(
            run_id="r3",
            workspace_root=self.workspace_root,
            oracle_validation_report=report,
        )
        course_run = _course_run_with_outcome_state(state)
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        self.assertIsNotNone(review.outcome_findings)
        assert review.outcome_findings is not None
        self.assertEqual(
            review.outcome_findings.oracle_validation_failures,
            [
                "Scenario 'happy_path_01' failed during reference replay.",
                "Missing required category: malformed_input.",
            ],
        )
        # Legacy blockers also see the oracle failures.
        joined = "\n".join(review.blockers)
        self.assertIn("happy_path_01", joined)
        self.assertIn("malformed_input", joined)

    def test_outcome_run_review_surfaces_curated_validation_blockers(self) -> None:
        """curated_validation_report's blocking_reasons appear in the review."""
        report = OracleValidationReport(
            publishable=False,
            reference_impl_hash="abc",
            scenario_set_hash="def",
            blocking_reasons=[
                "Curated scenario 'cur_42' lacks any non-trivial rubric.",
            ],
        )
        state = _build_outcome_state(
            run_id="r4",
            workspace_root=self.workspace_root,
            curated_validation_report=report,
        )
        course_run = _course_run_with_outcome_state(state)
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        self.assertIsNotNone(review.outcome_findings)
        assert review.outcome_findings is not None
        self.assertEqual(
            review.outcome_findings.curated_validation_failures,
            ["Curated scenario 'cur_42' lacks any non-trivial rubric."],
        )

    def test_outcome_run_review_includes_overall_status(self) -> None:
        """review.outcome_findings.overall_status mirrors state.status."""
        state = _build_outcome_state(
            run_id="r5",
            workspace_root=self.workspace_root,
            status="awaiting_human",
            stage="awaiting_gate_2",
        )
        course_run = _course_run_with_outcome_state(state)
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        self.assertIsNotNone(review.outcome_findings)
        assert review.outcome_findings is not None
        self.assertEqual(review.outcome_findings.overall_status, "awaiting_human")
        self.assertEqual(review.outcome_findings.stage, "awaiting_gate_2")

    def test_legacy_run_review_unchanged_when_payload_json_empty(self) -> None:
        """A run without outcome_state gets the existing review behavior."""
        course_run = _course_run_with_outcome_state(None)
        # Give it a last_error so we know the legacy ``blockers`` branch fires.
        course_run.last_error = "Some legacy error."
        course_run.stage = CourseRunStage.blocked
        course_run.status = CourseRunStatus.blocked
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        self.assertIsNone(review.outcome_findings)
        # Legacy behaviour preserved.
        self.assertIn("Some legacy error.", review.blockers)

    def test_outcome_run_blocked_status_surfaces_blocking_reasons(self) -> None:
        """A blocked outcome run with multiple blocking reasons shows them all."""
        state = _build_outcome_state(
            run_id="r6",
            workspace_root=self.workspace_root,
            blocking_reasons=[
                "Outcome planner failed: missing endpoint contract.",
                "Spec review rejected at gate 1.",
            ],
            status="blocked",
            stage="blocked",
        )
        course_run = _course_run_with_outcome_state(
            state,
            stage=CourseRunStage.blocked,
            status=CourseRunStatus.blocked,
        )
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        self.assertIsNotNone(review.outcome_findings)
        assert review.outcome_findings is not None
        self.assertEqual(len(review.outcome_findings.blocking_reasons), 2)
        # All blocking reasons land in the legacy ``blockers`` field too.
        joined = "\n".join(review.blockers)
        self.assertIn("Outcome planner failed", joined)
        self.assertIn("Spec review rejected", joined)

    def test_outcome_findings_severity_order_errors_first(self) -> None:
        """Errors come before warnings when merging findings into legacy blockers."""
        warning = ReviewerFinding(
            category="spec_coherence",
            severity=ReviewerFindingSeverity.warning,
            title="Spec uses ambiguous wording",
            detail="The quality bar phrasing could be clearer.",
        )
        error = ReviewerFinding(
            category="starter_verify",
            severity=ReviewerFindingSeverity.error,
            title="Starter sandbox build crashed",
            detail="Build stage exited with code 1.",
        )
        state = _build_outcome_state(
            run_id="r7",
            workspace_root=self.workspace_root,
            spec_review_findings=[warning],
            starter_review_findings=[error],
        )
        course_run = _course_run_with_outcome_state(state)
        self._save(course_run)

        review = self.service.review_run(course_run.id)

        # The error must precede the warning in the legacy ``blockers``.
        joined_text = "\n".join(review.blockers)
        error_index = joined_text.find("Starter sandbox build crashed")
        warning_index = joined_text.find("Spec uses ambiguous wording")
        self.assertGreater(error_index, -1, "error finding must appear in blockers")
        # The warning is non-blocking, so it may not appear in blockers at
        # all — but if both do appear, the error must come first.
        if warning_index >= 0:
            self.assertLess(error_index, warning_index)

    def test_outcome_findings_handle_malformed_outcome_state_defensively(self) -> None:
        """A malformed outcome_state blob does not crash review_run().

        If ``payload_json["outcome_state"]`` is present but cannot be
        deserialized, the review path must degrade gracefully: emit no
        ``outcome_findings`` and fall back to the legacy behaviour
        without raising.
        """
        course_run = _course_run_with_outcome_state(None)
        course_run.payload_json = {"outcome_state": {"not_a_real_field": 42}}
        self._save(course_run)

        # Must not raise.
        review = self.service.review_run(course_run.id)

        # Outcome findings are not populated because the blob was bad.
        self.assertIsNone(review.outcome_findings)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
