from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.ai import AIUsageSummary, merge_ai_usage
from app.domain.registry import PackageType, RiskClass
from app.domain.grader import DeliverableGraderPlan, TaskAgentGraderPlanCollection
from app.domain.grading import LiveGradeTaskAgentRequest, LiveTaskAgentGradeReport, DeliverableGradeReport, TaskAgentSubmission
from app.domain.workflow import (
    BundleFileContent,
    DecisionOutcome,
    DraftKind,
    GateDecisionRequest,
    HILGate,
    MaterializedBundle,
    MaterializeBundleRequest,
    WorkflowArtifacts,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowLoopPhaseSummary,
    WorkflowLoopPolicy,
    WorkflowReviewSummary,
    WorkflowNodeStatus,
    WorkflowRun,
    WorkflowRunList,
    WorkflowStage,
    WorkflowStatus,
)
from app.domain.task_agent import AssignmentDesignSpec, DeliverableSpec, TaskAgentServiceSpec
from app.services.public_surface_quality import meaningful_domain_entities
from app.services.assignment_design_inference import (
    DesignSupportStatus,
    GenerationIntake,
    infer_assignment_design,
)
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.bundle_validation import validate_materialized_bundle
from app.services.coursegen_logging import log_coursegen_event
from app.services.grader_planner import build_all_task_agent_grader_plans, build_task_agent_grader_plan
from app.services.failure_context_builder import build_failure_context
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs
from app.services.openai_task_agent_authoring import (
    OpenAITaskAgentAuthoringService,
    TaskAgentAuthoringSource,
    TaskAgentAuthoringStatus,
)
from app.services.spec_validation import validate_task_agent_spec
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner, TaskAgentRunnerError
from app.services.task_agent_grader import grade_task_agent_submission
from app.storage.workflow_store import WorkflowStore

_GRAPH_EXECUTION_NODE_KINDS = {
    WorkflowNodeKind.authoring_runtime,
    WorkflowNodeKind.authoring_tests,
    WorkflowNodeKind.authoring_repair,
    WorkflowNodeKind.reviewer_runtime,
    WorkflowNodeKind.reviewer_repair,
    WorkflowNodeKind.reviewer_code,
    WorkflowNodeKind.reviewer_pedagogy,
    WorkflowNodeKind.reviewer_tests,
}


class WorkflowConflictError(ValueError):
    """Raised when a workflow transition is invalid."""


def _default_planner_deliverables(design_spec: AssignmentDesignSpec) -> list[DeliverableSpec]:
    """Build a minimal planner deliverable list when no planner is wired in.

    Pass 10 Job A: ``build_task_agent_scaffold`` always needs a planner-shaped
    deliverable list. Callers that come directly through ``WorkflowService``
    without going through the course planner (legacy single-assignment flows,
    most direct tests) get a small default list derived from the design
    spec's primary domain entity. Progressive courses still need at least
    two deliverables to satisfy the spec validator, so the default size is
    based on the package type.
    """
    entities = meaningful_domain_entities(design_spec.project_contract.core_entities)
    entity = entities[0] if entities else (design_spec.project_contract.system_kind or "resource")
    package_type = design_spec.course_structure.package_type
    titles = [
        f"{entity.title()} contract and public surface",
    ]
    if package_type.value == "progressive_codebase_course":
        titles.append(f"{entity.title()} production hardening")
    objectives = {
        titles[0]: f"Define and implement a stable public surface for {entity}.",
    }
    if len(titles) > 1:
        objectives[titles[1]] = (
            f"Raise the {entity} service to a production-minded bar for reliability, latency, and diagnostics."
        )
    return [
        DeliverableSpec(
            id=f"deliverable_{index}",
            title=title,
            objective=objectives[title],
            learning_outcomes=[],
            overlay_ids=[],
        )
        for index, title in enumerate(titles, start=1)
    ]


class WorkflowService:
    def __init__(
        self,
        store: WorkflowStore,
        materializer: ArtifactMaterializer | None = None,
        runner: TaskAgentBlackBoxRunner | None = None,
        node_runtime: LangGraphAssignmentGraph | None = None,
        task_agent_authoring_service: OpenAITaskAgentAuthoringService | None = None,
        workspace_manager: AssignmentWorkspaceManager | None = None,
    ) -> None:
        self.store = store
        self.materializer = materializer or ArtifactMaterializer()
        self.runner = runner or TaskAgentBlackBoxRunner()
        self.node_runtime = node_runtime
        self.task_agent_authoring_service = task_agent_authoring_service or OpenAITaskAgentAuthoringService(enabled=False)
        self.workspace_manager = workspace_manager or AssignmentWorkspaceManager()

    def list_runs(self, limit: int = 50) -> WorkflowRunList:
        return WorkflowRunList(runs=self.store.list_runs(limit=limit))

    def get_run(self, run_id: str) -> WorkflowRun | None:
        return self.store.get_run(run_id)

    def list_events(self, run_id: str):
        return self.store.list_events(run_id)

    def list_node_executions(self, run_id: str) -> list[WorkflowNodeExecution]:
        run = self._require_run(run_id)
        return run.artifacts.node_executions

    def task_agent_authoring_status(self) -> TaskAgentAuthoringStatus:
        return self.task_agent_authoring_service.status()

    def get_workspace(self, run_id: str) -> MaterializedBundle:
        run = self._require_run(run_id)
        if run.artifacts.workspace_snapshot is None:
            raise WorkflowConflictError("This run does not have a prepared workspace yet.")
        return run.artifacts.workspace_snapshot

    def read_workspace_file(self, run_id: str, relative_path: str) -> BundleFileContent:
        run = self._require_run(run_id)
        if run.artifacts.workspace_snapshot is None:
            raise WorkflowConflictError("This run does not have a prepared workspace yet.")
        return self.workspace_manager.read_workspace_file(run.artifacts.workspace_snapshot, relative_path)

    def get_review_summary(self, run_id: str) -> WorkflowReviewSummary:
        run = self._require_run(run_id)
        self._refresh_review_summary(run)
        self.store.save_run(run)
        return run.artifacts.review_summary or self._empty_review_summary()

    def materialize_run(self, run_id: str, request: MaterializeBundleRequest) -> WorkflowRun:
        run = self._require_run(run_id)
        if run.artifacts.task_agent_spec is not None:
            validation = validate_task_agent_spec(run.artifacts.task_agent_spec)
            run.artifacts.validation_summary = validation.model_dump(mode="json")
            run.artifacts.progression_preview = [summary.model_dump(mode="json") for summary in validation.deliverable_gates]
            if not validation.valid:
                self._refresh_review_summary(run)
                self.store.save_run(run)
                raise WorkflowConflictError("Task-agent draft is invalid. Fix validation errors before materializing artifacts.")

        self._refresh_review_summary(run)
        bundle = self.materializer.materialize_run(run, overwrite=request.overwrite)
        if run.artifacts.task_agent_spec is not None:
            bundle_validation = validate_materialized_bundle(run.artifacts.task_agent_spec, bundle)
            if not bundle_validation.valid:
                raise WorkflowConflictError(
                    "The materialized learner bundle is inconsistent with the assignment contract. "
                    + "; ".join(issue.code for issue in bundle_validation.errors[:3])
                )
        run.artifacts.materialized_bundle = bundle
        run.updated_at = datetime.now(UTC)
        self.store.save_run(run)
        self.store.append_event(
            run.id,
            "artifacts_materialized",
            {"bundle_id": bundle.bundle_id, "file_count": len(bundle.files)},
        )
        return run

    def read_bundle_file(self, run_id: str, relative_path: str) -> BundleFileContent:
        run = self._require_run(run_id)
        if run.artifacts.materialized_bundle is None:
            raise WorkflowConflictError("This run has not been materialized yet.")
        return self.materializer.read_bundle_file(run.artifacts.materialized_bundle, relative_path)

    def create_run_from_explicit_plan(
        self,
        *,
        intake: GenerationIntake,
        design_spec: AssignmentDesignSpec,
        reasons: list[str] | None = None,
        warnings: list[str] | None = None,
        notes: list[str] | None = None,
        execute_nodes: bool = True,
        planner_deliverables: list[DeliverableSpec] | None = None,
    ) -> WorkflowRun:
        reasons = reasons or []
        warnings = warnings or []
        notes = notes or []
        status = (
            DesignSupportStatus.manual_review
            if design_spec.risk_class != RiskClass.standard
            else DesignSupportStatus.supported
        )
        return self._create_run_from_design(
            intake,
            design_spec=design_spec,
            status=status,
            reasons=reasons or ["Created from an explicit assignment design specification."],
            warnings=warnings,
            notes=notes,
            execute_nodes=execute_nodes,
            planner_deliverables=planner_deliverables,
        )

    def list_task_agent_grader_plans(self, run_id: str) -> TaskAgentGraderPlanCollection:
        spec = self._require_task_agent_spec(run_id)
        return build_all_task_agent_grader_plans(spec)

    def get_task_agent_grader_plan(self, run_id: str, deliverable_id: str) -> DeliverableGraderPlan:
        spec = self._require_task_agent_spec(run_id)
        return build_task_agent_grader_plan(spec, deliverable_id)

    def grade_task_agent_run(self, run_id: str, deliverable_id: str, submission: TaskAgentSubmission) -> DeliverableGradeReport:
        spec = self._require_task_agent_spec(run_id)
        report = grade_task_agent_submission(spec, deliverable_id, submission)
        self.store.append_event(
            run_id,
            "submission_graded",
            {
                "deliverable_id": deliverable_id,
                "submission_id": submission.submission_id,
                "status": report.status.value,
                "passed_tests": report.passed_tests,
                "total_tests": report.total_tests,
            },
        )
        return report

    def grade_task_agent_run_live(
        self,
        run_id: str,
        deliverable_id: str,
        request: LiveGradeTaskAgentRequest,
    ) -> LiveTaskAgentGradeReport:
        spec = self._require_task_agent_spec(run_id)
        report = self.runner.grade_live(spec, deliverable_id, request)
        self.store.append_event(
            run_id,
            "submission_graded_live",
            {
                "deliverable_id": deliverable_id,
                "base_url": request.base_url,
                "status": report.grade_report.status.value,
                "passed_tests": report.grade_report.passed_tests,
                "total_tests": report.grade_report.total_tests,
            },
        )
        return report

    def create_run(
        self,
        intake: GenerationIntake,
        *,
        execute_nodes: bool = True,
        planner_deliverables: list[DeliverableSpec] | None = None,
    ) -> WorkflowRun:
        inferred = infer_assignment_design(
            title=intake.title,
            problem_statement=intake.problem_statement,
            package_type_hint=intake.package_type_hint,
            starter_type=intake.starter_type,
            implementation_language=intake.implementation_language,
            language_version=intake.language_version,
            application_framework=intake.application_framework,
            framework_version=intake.framework_version,
            package_manager=intake.package_manager,
            primary_database=intake.primary_database,
            primary_database_version=intake.primary_database_version,
            cache_backend=intake.cache_backend,
            cache_backend_version=intake.cache_backend_version,
            tech_stack=intake.tech_stack,
            data_sources=intake.data_sources,
        )
        if inferred.design_spec is None:
            return self._create_blocked_run(
                intake,
                notes=[
                    *inferred.reasons,
                    *inferred.warnings,
                ] or ["The brief is outside the current learner-ready generation scope."],
            )
        return self._create_run_from_design(
            intake,
            design_spec=inferred.design_spec,
            status=inferred.status,
            reasons=inferred.reasons,
            warnings=inferred.warnings,
            notes=[],
            execute_nodes=execute_nodes,
            planner_deliverables=planner_deliverables,
        )

    def create_revision_from_run(self, run_id: str) -> WorkflowRun:
        source = self._require_run(run_id)
        now = datetime.now(UTC)
        revision = source.model_copy(deep=True)
        revision.id = f"run_{uuid4().hex[:12]}"
        revision.created_at = now
        revision.updated_at = now
        revision.stage = WorkflowStage.needs_revision
        revision.status = WorkflowStatus.active
        revision.pending_gate = None
        revision.artifacts.workspace_snapshot = None
        revision.artifacts.materialized_bundle = None
        revision.artifacts.node_executions = []
        revision.artifacts.review_summary = None
        revision.notes = [
            *revision.notes,
            f"Revision draft created from published workflow `{source.id}`.",
        ]
        revision.artifacts.notes = [
            *revision.artifacts.notes,
            f"Revision draft cloned from `{source.id}` and reset for a fresh author/reviewer pass.",
        ]
        self.store.save_run(revision)
        self.store.append_event(
            revision.id,
            "run_revision_created",
            {"source_run_id": source.id},
        )
        if revision.artifacts.task_agent_spec is not None and self.node_runtime is not None:
            revision = self.execute_langgraph_nodes(revision.id)
        else:
            revision.stage = WorkflowStage.awaiting_hil_gate_1
            revision.status = WorkflowStatus.awaiting_human
            revision.pending_gate = HILGate.gate_1_spec_review
            self.store.save_run(revision)
        return revision

    def _create_blocked_run(
        self,
        intake: GenerationIntake,
        *,
        notes: list[str],
    ) -> WorkflowRun:
        now = datetime.now(UTC)
        run_id = f"run_{uuid4().hex[:12]}"
        artifacts = WorkflowArtifacts(
            draft_kind=DraftKind.scope_blocked,
            notes=["No learner-ready starter project was created for this brief."],
        )
        run = WorkflowRun(
            id=run_id,
            title=intake.title,
            created_at=now,
            updated_at=now,
            stage=WorkflowStage.blocked,
            status=WorkflowStatus.blocked,
            pending_gate=None,
            intake=intake,
            artifacts=artifacts,
            notes=notes,
        )
        self.store.save_run(run)
        self.store.append_event(run.id, "run_created", {"status": run.status.value})
        return run

    def _create_run_from_design(
        self,
        intake: GenerationIntake,
        *,
        design_spec: AssignmentDesignSpec,
        status: DesignSupportStatus,
        reasons: list[str],
        warnings: list[str],
        notes: list[str],
        execute_nodes: bool = True,
        planner_deliverables: list[DeliverableSpec] | None = None,
    ) -> WorkflowRun:
        now = datetime.now(UTC)
        run_id = f"run_{uuid4().hex[:12]}"
        log_coursegen_event(
            "workflow_authoring_started",
            workflow_run_id=run_id,
            title=intake.title,
            package_type=design_spec.course_structure.package_type.value,
            implementation_language=design_spec.runtime_dependencies.implementation_language,
            application_framework=design_spec.runtime_dependencies.application_framework,
        )
        resolved_planner_deliverables = (
            planner_deliverables
            if planner_deliverables is not None
            else _default_planner_deliverables(design_spec)
        )
        authoring_result = self.task_agent_authoring_service.generate_scaffold(
            title=intake.title,
            summary=intake.problem_statement,
            design_spec=design_spec,
            planner_deliverables=resolved_planner_deliverables,
        )
        log_coursegen_event(
            "workflow_authoring_scaffold_generated",
            workflow_run_id=run_id,
            title=intake.title,
            source=authoring_result.source.value,
            origin_template=authoring_result.origin_template,
            note_count=len(authoring_result.notes),
        )
        if self._should_block_on_live_authoring_failure(authoring_result):
            log_coursegen_event(
                "workflow_authoring_failed",
                workflow_run_id=run_id,
                title=intake.title,
                reason="live_authoring_fallback_blocked",
                notes=authoring_result.notes[:5],
            )
            return self._create_authoring_failure_run(
                intake,
                notes=[
                    "Live assignment authoring did not complete successfully, so the workflow stopped before reviewer execution.",
                    *authoring_result.notes,
                ],
            )
        task_agent_spec = authoring_result.spec
        origin_template = authoring_result.origin_template
        validation = validate_task_agent_spec(task_agent_spec)
        artifacts = WorkflowArtifacts(
            draft_kind=DraftKind.task_agent_spec,
            task_agent_spec=task_agent_spec,
            ai_usage=authoring_result.usage or AIUsageSummary(),
            validation_summary=validation.model_dump(mode="json"),
            progression_preview=[summary.model_dump(mode="json") for summary in validation.deliverable_gates],
            artifact_plan=self._artifact_plan_for_task_agent(task_agent_spec),
            origin_template=origin_template,
            notes=[
                "Draft starter project created from the explicit assignment design spec.",
                *authoring_result.notes,
                "Edit the learner-ready assignment spec, revalidate it, then move through the HIL gates.",
            ],
        )

        run = WorkflowRun(
            id=run_id,
            title=intake.title,
            created_at=now,
            updated_at=now,
            stage=WorkflowStage.awaiting_hil_gate_1,
            status=WorkflowStatus.awaiting_human,
            pending_gate=HILGate.gate_1_spec_review,
            intake=intake,
            artifacts=artifacts,
            notes=[*reasons, *warnings, *notes],
        )
        self.store.save_run(run)
        self.store.append_event(
            run.id,
            "run_created",
            {
                "stage": run.stage.value,
                "design_status": status.value,
                "draft_kind": run.artifacts.draft_kind.value,
                "origin_template": run.artifacts.origin_template,
                "ai_usage": (
                    authoring_result.usage.model_dump(mode="json")
                    if authoring_result.usage is not None
                    else None
                ),
            },
        )
        self.store.append_event(
            run.id,
            "workflow_authoring_completed",
            {
                "message": (
                    f"Generated the first learner-ready assignment draft with "
                    f"{len(task_agent_spec.deliverables)} deliverable"
                    f"{'' if len(task_agent_spec.deliverables) == 1 else 's'}."
                ),
                "origin_template": run.artifacts.origin_template,
                "deliverable_count": len(task_agent_spec.deliverables),
            },
        )
        log_coursegen_event(
            "workflow_authoring_completed",
            workflow_run_id=run.id,
            title=run.title,
            deliverable_count=len(task_agent_spec.deliverables),
            execute_nodes=execute_nodes,
        )
        if execute_nodes and run.artifacts.task_agent_spec is not None and self.node_runtime is not None:
            run = self.execute_langgraph_nodes(run.id)
        return run

    def update_task_agent_spec(
        self,
        run_id: str,
        spec: TaskAgentServiceSpec,
        *,
        execute_nodes: bool = True,
    ) -> WorkflowRun:
        run = self._require_run(run_id)
        if run.stage == WorkflowStage.published or run.status == WorkflowStatus.published:
            raise WorkflowConflictError("Published workflow runs are immutable. Create a new revision before editing the spec.")
        if run.artifacts.task_agent_spec is None:
            raise WorkflowConflictError("This run does not contain a task-agent draft spec.")

        spec = ensure_task_agent_deliverable_briefs(spec, overwrite=False)
        validation = validate_task_agent_spec(spec)
        run.artifacts.task_agent_spec = spec
        self._invalidate_generated_artifacts(run, reason="task-agent spec update")
        run.artifacts.validation_summary = validation.model_dump(mode="json")
        run.artifacts.progression_preview = [summary.model_dump(mode="json") for summary in validation.deliverable_gates]
        run.updated_at = datetime.now(UTC)
        self.store.save_run(run)
        self.store.append_event(
            run.id,
            "task_agent_spec_updated",
            {
                "valid": validation.valid,
                "error_count": len(validation.errors),
                "warning_count": len(validation.warnings),
            },
        )
        if execute_nodes and self.node_runtime is not None:
            run = self.execute_langgraph_nodes(run.id)
        return run

    def _should_block_on_live_authoring_failure(self, authoring_result) -> bool:
        if authoring_result.source != TaskAgentAuthoringSource.deterministic_fallback:
            return False
        status = authoring_result.status
        if not status.sdk_installed or not status.api_key_present:
            return False
        message = " ".join([status.message, *authoring_result.notes]).lower()
        return "fell back to the deterministic starter template" in message

    def _create_authoring_failure_run(
        self,
        intake: GenerationIntake,
        *,
        notes: list[str],
    ) -> WorkflowRun:
        now = datetime.now(UTC)
        run_id = f"run_{uuid4().hex[:12]}"
        artifacts = WorkflowArtifacts(
            draft_kind=DraftKind.scope_blocked,
            notes=[
                "No learner-ready starter project was created because live assignment authoring failed.",
                *notes,
            ],
        )
        run = WorkflowRun(
            id=run_id,
            title=intake.title,
            created_at=now,
            updated_at=now,
            stage=WorkflowStage.blocked,
            status=WorkflowStatus.blocked,
            pending_gate=None,
            intake=intake,
            artifacts=artifacts,
            notes=notes,
        )
        self.store.save_run(run)
        self.store.append_event(
            run.id,
            "workflow_authoring_failed",
            {"status": run.status.value, "message": notes[0] if notes else "Live assignment authoring failed."},
        )
        return run

    def execute_langgraph_nodes(self, run_id: str) -> WorkflowRun:
        run = self._require_run(run_id)
        if run.stage == WorkflowStage.published or run.status == WorkflowStatus.published:
            raise WorkflowConflictError("Published workflow runs are immutable. Create a new revision before rerunning authoring or review.")
        if run.artifacts.task_agent_spec is None:
            raise WorkflowConflictError("This run does not contain a task-agent draft spec.")
        if self.node_runtime is None:
            raise WorkflowConflictError("LangGraph node execution is not configured for this app instance.")

        if run.artifacts.workspace_snapshot is None:
            run.artifacts.workspace_snapshot = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
        log_coursegen_event(
            "workflow_node_loop_started",
            workflow_run_id=run.id,
            title=run.title,
            stage=run.stage.value,
            status=run.status.value,
        )
        try:
            run = self.node_runtime.execute(
                run,
                on_node_started=self._record_node_started,
                on_node_finished=self._record_node_finished,
            )
        except Exception as exc:
            self.store.append_event(
                run.id,
                "langgraph_nodes_failed",
                {
                    "error": str(exc),
                    "stage": run.stage.value,
                    "status": run.status.value,
                },
            )
            log_coursegen_event(
                "workflow_node_loop_failed",
                workflow_run_id=run.id,
                title=run.title,
                error=str(exc),
            )
            raise
        validation = validate_task_agent_spec(run.artifacts.task_agent_spec)
        run.artifacts.validation_summary = validation.model_dump(mode="json")
        run.artifacts.progression_preview = [summary.model_dump(mode="json") for summary in validation.deliverable_gates]
        self._refresh_review_summary(run)
        if run.artifacts.workspace_snapshot is None:
            run.artifacts.workspace_snapshot = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
        self._apply_node_stage(run)
        run.updated_at = datetime.now(UTC)
        self._persist_run_progress(run)
        latest_node_by_kind = {
            node.kind: node
            for node in run.artifacts.node_executions
        }
        self.store.append_event(
            run.id,
            "langgraph_nodes_executed",
            {
                "node_count": len(run.artifacts.node_executions),
                "failed_nodes": [
                    kind.value
                    for kind, node in latest_node_by_kind.items()
                    if node.status != WorkflowNodeStatus.passed
                ],
                "stage": run.stage.value,
                "pending_gate": run.pending_gate.value if run.pending_gate else None,
                "workspace_root": run.artifacts.workspace_snapshot.root_dir if run.artifacts.workspace_snapshot is not None else None,
            },
        )
        log_coursegen_event(
            "workflow_node_loop_completed",
            workflow_run_id=run.id,
            title=run.title,
            stage=run.stage.value,
            status=run.status.value,
            node_count=len(run.artifacts.node_executions),
        )
        return run

    def _record_node_started(
        self,
        run: WorkflowRun,
        kind: WorkflowNodeKind,
        attempt: int,
    ) -> None:
        self.store.append_event(
            run.id,
            "workflow_node_started",
            {
                "node_kind": kind.value,
                "attempt": attempt,
                "stage": run.stage.value,
                "status": run.status.value,
            },
        )
        log_coursegen_event(
            "workflow_node_started",
            workflow_run_id=run.id,
            title=run.title,
            node_kind=kind.value,
            attempt=attempt,
            stage=run.stage.value,
            status=run.status.value,
        )

    def _record_node_finished(
        self,
        run: WorkflowRun,
        node: WorkflowNodeExecution,
    ) -> None:
        run.updated_at = datetime.now(UTC)
        self._persist_run_progress(run)
        self.store.append_event(
            run.id,
            "workflow_node_completed",
            {
                "node_kind": node.kind.value,
                "iteration": node.iteration,
                "attempt": node.attempt,
                "status": node.status.value,
                "summary": node.summary,
            },
        )
        log_coursegen_event(
            "workflow_node_completed",
            workflow_run_id=run.id,
            title=run.title,
            node_kind=node.kind.value,
            attempt=node.attempt,
            node_status=node.status.value,
            summary=node.summary,
        )

    def _persist_run_progress(self, run: WorkflowRun) -> None:
        self.store.save_run(run)
        self._write_workspace_progress_files(run)

    def _write_workspace_progress_files(self, run: WorkflowRun) -> None:
        workspace = run.artifacts.workspace_snapshot
        if workspace is None:
            return
        private_dir = Path(workspace.private_dir)
        private_dir.mkdir(parents=True, exist_ok=True)
        spec_payload = (
            run.artifacts.task_agent_spec.model_dump(mode="json")
            if run.artifacts.task_agent_spec is not None
            else None
        )
        self._write_progress_json(private_dir / "task_agent_spec.json", spec_payload)
        self._write_progress_json(private_dir / "validation_summary.json", run.artifacts.validation_summary or {})
        self._write_progress_json(private_dir / "progression_preview.json", run.artifacts.progression_preview)
        self._write_progress_json(
            private_dir / "workflow_snapshot.json",
            {
                "run_id": run.id,
                "title": run.title,
                "stage": run.stage.value,
                "status": run.status.value,
                "pending_gate": run.pending_gate.value if run.pending_gate else None,
                "origin_template": run.artifacts.origin_template,
            },
        )
        self._write_progress_json(
            private_dir / "node_executions.json",
            [node.model_dump(mode="json") for node in run.artifacts.node_executions],
        )
        self._write_progress_json(
            private_dir / "review_summary.json",
            run.artifacts.review_summary.model_dump(mode="json") if run.artifacts.review_summary is not None else {},
        )

    def _write_progress_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def apply_gate_decision(self, run_id: str, decision: GateDecisionRequest) -> WorkflowRun:
        run = self._require_run(run_id)
        if run.pending_gate != decision.gate:
            raise WorkflowConflictError(
                f"Run is waiting on '{run.pending_gate.value if run.pending_gate else None}', not '{decision.gate.value}'."
            )

        now = datetime.now(UTC)
        if decision.decision == DecisionOutcome.reject:
            run.stage = WorkflowStage.needs_revision
            run.status = WorkflowStatus.awaiting_human
            if decision.comment:
                run.notes.append(decision.comment)
                run.artifacts.notes.append("Human review feedback captured for the next authoring/review pass.")
                run = self._apply_human_feedback_revision(run, decision.comment)
            run.updated_at = now
            self.store.save_run(run)
            self.store.append_event(
                run.id,
                "gate_rejected",
                {
                    "gate": decision.gate.value,
                    "comment": decision.comment or "",
                    "rerun_requested": bool(decision.comment and self.node_runtime is not None),
                },
            )
            if decision.comment and self.node_runtime is not None and run.artifacts.task_agent_spec is not None:
                run = self.execute_langgraph_nodes(run.id)
            return run

        if decision.gate == HILGate.gate_1_spec_review:
            if run.artifacts.task_agent_spec is not None:
                validation = run.artifacts.validation_summary or {}
                if not validation.get("valid", False):
                    raise WorkflowConflictError("The task-agent draft is not valid yet. Fix validation errors before requesting review.")
                if run.artifacts.workspace_snapshot is None:
                    raise WorkflowConflictError(
                        "Authoring must rerun after the latest spec change before the spec review gate can be approved."
                    )
                author_node = self._node_by_kind(run, WorkflowNodeKind.authoring_runtime)
                authoring_tests_node = self._node_by_kind(run, WorkflowNodeKind.authoring_tests)
                if author_node is None or author_node.status != WorkflowNodeStatus.passed:
                    raise WorkflowConflictError("Authoring must pass Docker sandbox verification before the spec review gate can be approved.")
                if authoring_tests_node is None or authoring_tests_node.status != WorkflowNodeStatus.passed:
                    raise WorkflowConflictError("Generated visible and hidden tests must be authored successfully before the spec review gate can be approved.")
            else:
                raise WorkflowConflictError("This workflow does not have a reviewable assignment spec yet.")
            run.stage = WorkflowStage.awaiting_hil_gate_2
            run.pending_gate = HILGate.gate_2_progression_review
        elif decision.gate == HILGate.gate_2_progression_review:
            run.stage = WorkflowStage.awaiting_hil_gate_3
            run.pending_gate = HILGate.gate_3_pre_publish
        elif decision.gate == HILGate.gate_3_pre_publish:
            run.stage = WorkflowStage.published
            run.status = WorkflowStatus.published
            run.pending_gate = None
            run.artifacts.notes.append("Marked published by the HIL workflow.")
        run.updated_at = now
        if decision.comment:
            run.notes.append(decision.comment)
        self.store.save_run(run)
        self.store.append_event(
            run.id,
            "gate_approved",
            {"gate": decision.gate.value, "comment": decision.comment or "", "stage": run.stage.value},
        )
        return run

    def _apply_human_feedback_revision(self, run: WorkflowRun, feedback: str) -> WorkflowRun:
        spec = run.artifacts.task_agent_spec
        if spec is None:
            return run

        latest_node = run.artifacts.node_executions[-1] if run.artifacts.node_executions else None
        revision = self.task_agent_authoring_service.revise_spec(
            spec=spec,
            title=run.title,
            summary=run.intake.problem_statement,
            package_type=spec.course_structure.package_type,
            domain_pack=spec.domain_pack,
            risk_class=spec.risk_class,
            overlays=spec.overlays,
            feedback=feedback,
            failure_context=build_failure_context(run, latest_node) if latest_node is not None else None,
            origin_template=run.artifacts.origin_template,
        )
        revised_spec = ensure_task_agent_deliverable_briefs(revision.spec, overwrite=False)
        validation = validate_task_agent_spec(revised_spec)
        if not validation.valid:
            raise WorkflowConflictError(
                "Human feedback revision produced an invalid task-agent draft. "
                "Fix the generated spec before rerunning review."
            )
        run.artifacts.task_agent_spec = revised_spec
        run.artifacts.origin_template = revision.origin_template
        self._invalidate_generated_artifacts(run, reason="human-feedback revision")
        run.artifacts.ai_usage = merge_ai_usage(run.artifacts.ai_usage, revision.usage)
        run.artifacts.validation_summary = validation.model_dump(mode="json")
        run.artifacts.progression_preview = [
            summary.model_dump(mode="json")
            for summary in validation.deliverable_gates
        ]
        if revision.notes:
            run.notes.extend(revision.notes)
        self.store.append_event(
            run.id,
            "workflow_authoring_revised",
            {
                "message": "Updated the learner-ready assignment draft after human review feedback.",
                "deliverable_count": len(revised_spec.deliverables),
                "origin_template": revision.origin_template,
            },
        )
        return run

    def _invalidate_generated_artifacts(self, run: WorkflowRun, *, reason: str) -> None:
        invalidated: list[str] = []
        if run.artifacts.workspace_snapshot is not None:
            invalidated.append("workspace snapshot")
        if run.artifacts.materialized_bundle is not None:
            invalidated.append("materialized bundle")
        run.artifacts.workspace_snapshot = None
        run.artifacts.materialized_bundle = None
        if invalidated:
            run.artifacts.notes.append(
                f"Invalidated the {' and '.join(invalidated)} after {reason} so the next execution rematerializes learner-facing artifacts from the latest spec."
            )

    def _artifact_plan_for_task_agent(self, spec: TaskAgentServiceSpec) -> list[str]:
        lines = [
            "learner-ready runtime plan and public endpoint contract",
            "deliverable activation plan derived from visible checks",
            "starter plan per deliverable based on learner-owned files",
            "per-deliverable learning content and evaluation bundle",
        ]
        if spec.capabilities.approval_flow_required:
            lines.append("approval-sensitive behavior covered in the final learner bundle")
        return lines

    def _require_run(self, run_id: str) -> WorkflowRun:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def _require_task_agent_spec(self, run_id: str) -> TaskAgentServiceSpec:
        run = self._require_run(run_id)
        if run.artifacts.task_agent_spec is None:
            raise WorkflowConflictError("This run does not contain a task-agent draft spec.")
        return run.artifacts.task_agent_spec

    def _node_by_kind(self, run: WorkflowRun, kind: WorkflowNodeKind) -> WorkflowNodeExecution | None:
        for node in reversed(run.artifacts.node_executions):
            if node.kind == kind:
                return node
        return None

    def _latest_graph_iteration(self, run: WorkflowRun) -> int:
        return max(
            (
                node.iteration
                for node in run.artifacts.node_executions
                if node.kind in _GRAPH_EXECUTION_NODE_KINDS
            ),
            default=0,
        )

    def _refresh_review_summary(self, run: WorkflowRun) -> WorkflowReviewSummary:
        policy = self._loop_policy()
        if run.artifacts.task_agent_spec is None:
            review_summary = WorkflowReviewSummary(
                review_ready=False,
                blockers=["No learner-ready assignment spec is attached to this workflow run."],
                policy=policy,
                authoring=WorkflowLoopPhaseSummary(
                    attempts_used=0,
                    max_attempts=policy.max_authoring_attempts,
                    remaining_attempts=policy.max_authoring_attempts,
                    latest_node_kind=None,
                    latest_status=None,
                    exhausted=False,
                    passed=False,
                ),
                reviewer=WorkflowLoopPhaseSummary(
                    attempts_used=0,
                    max_attempts=policy.max_reviewer_attempts,
                    remaining_attempts=policy.max_reviewer_attempts,
                    latest_node_kind=None,
                    latest_status=None,
                    exhausted=False,
                    passed=False,
                ),
            )
            run.artifacts.review_summary = review_summary
            return review_summary

        latest_iteration = self._latest_graph_iteration(run)
        authoring_nodes = [
            node
            for node in run.artifacts.node_executions
            if node.kind in {WorkflowNodeKind.authoring_runtime, WorkflowNodeKind.authoring_tests, WorkflowNodeKind.authoring_repair}
            and (latest_iteration == 0 or node.iteration == latest_iteration)
        ]
        reviewer_nodes = [
            node
            for node in run.artifacts.node_executions
            if node.kind in {
                WorkflowNodeKind.reviewer_runtime,
                WorkflowNodeKind.reviewer_repair,
                WorkflowNodeKind.reviewer_code,
                WorkflowNodeKind.reviewer_pedagogy,
                WorkflowNodeKind.reviewer_tests,
            }
            and (latest_iteration == 0 or node.iteration == latest_iteration)
        ]

        authoring_summary = self._phase_summary(authoring_nodes, policy.max_authoring_attempts)
        reviewer_summary = self._phase_summary(reviewer_nodes, policy.max_reviewer_attempts)
        validation = run.artifacts.validation_summary or {}
        valid = validation.get("valid", False)

        reviewer_kinds = (
            WorkflowNodeKind.reviewer_runtime,
            WorkflowNodeKind.reviewer_code,
            WorkflowNodeKind.reviewer_pedagogy,
            WorkflowNodeKind.reviewer_tests,
        )
        authoring_runtime = self._node_by_kind(run, WorkflowNodeKind.authoring_runtime)
        authoring_tests = self._node_by_kind(run, WorkflowNodeKind.authoring_tests)
        authoring_passed = (
            run.artifacts.workspace_snapshot is not None
            and authoring_runtime is not None
            and authoring_runtime.status == WorkflowNodeStatus.passed
            and authoring_tests is not None
            and authoring_tests.status == WorkflowNodeStatus.passed
        )
        reviewer_passed = all(
            (node := self._node_by_kind(run, kind)) is not None and node.status == WorkflowNodeStatus.passed
            for kind in reviewer_kinds
        )

        blockers: list[str] = []
        if not valid:
            blockers.append("Spec validation is still failing.")
        if not authoring_passed:
            if authoring_summary.exhausted:
                if (
                    authoring_summary.latest_node_kind == WorkflowNodeKind.authoring_repair
                    and authoring_summary.latest_status == WorkflowNodeStatus.failed
                    and authoring_nodes
                ):
                    blockers.append(authoring_nodes[-1].summary)
                else:
                    blockers.append(
                        f"Authoring loop exhausted after {authoring_summary.attempts_used}/{authoring_summary.max_attempts} attempts."
                    )
            else:
                blockers.append("Authoring runtime verification has not passed yet.")
        if authoring_passed and not reviewer_passed:
            if reviewer_summary.exhausted:
                if (
                    reviewer_summary.latest_node_kind == WorkflowNodeKind.reviewer_repair
                    and reviewer_summary.latest_status == WorkflowNodeStatus.failed
                    and reviewer_nodes
                ):
                    blockers.append(reviewer_nodes[-1].summary)
                else:
                    blockers.append(
                        f"Reviewer loop exhausted after {reviewer_summary.attempts_used}/{reviewer_summary.max_attempts} attempts."
                    )
            else:
                blockers.append("Reviewer nodes have not all passed yet.")

        review_summary = WorkflowReviewSummary(
            review_ready=valid and authoring_passed and reviewer_passed,
            blockers=blockers,
            policy=policy,
            authoring=authoring_summary,
            reviewer=reviewer_summary,
        )
        run.artifacts.review_summary = review_summary
        return review_summary

    def _phase_summary(
        self,
        nodes: list[WorkflowNodeExecution],
        max_attempts: int,
    ) -> WorkflowLoopPhaseSummary:
        if not nodes:
            return WorkflowLoopPhaseSummary(
                attempts_used=0,
                max_attempts=max_attempts,
                remaining_attempts=max_attempts,
                latest_node_kind=None,
                latest_status=None,
                exhausted=False,
                passed=False,
            )

        latest = nodes[-1]
        attempts_used = max(node.attempt for node in nodes)
        passed = latest.status == WorkflowNodeStatus.passed
        remaining = max(max_attempts - attempts_used, 0)
        stopped_early = latest.kind in {
            WorkflowNodeKind.authoring_repair,
            WorkflowNodeKind.reviewer_repair,
        } and latest.status == WorkflowNodeStatus.failed
        exhausted = not passed and (attempts_used >= max_attempts or stopped_early)
        return WorkflowLoopPhaseSummary(
            attempts_used=attempts_used,
            max_attempts=max_attempts,
            remaining_attempts=remaining,
            latest_node_kind=latest.kind,
            latest_status=latest.status,
            exhausted=exhausted,
            passed=passed,
        )

    def _loop_policy(self) -> WorkflowLoopPolicy:
        if self.node_runtime is not None:
            return self.node_runtime.policy()
        return WorkflowLoopPolicy(max_authoring_attempts=1, max_reviewer_attempts=1)

    def _empty_review_summary(self) -> WorkflowReviewSummary:
        policy = self._loop_policy()
        return WorkflowReviewSummary(
            review_ready=False,
            blockers=["Workflow review has not run yet."],
            policy=policy,
            authoring=WorkflowLoopPhaseSummary(max_attempts=policy.max_authoring_attempts),
            reviewer=WorkflowLoopPhaseSummary(max_attempts=policy.max_reviewer_attempts),
        )

    def _apply_node_stage(self, run: WorkflowRun) -> None:
        summary = self._refresh_review_summary(run)
        if summary.authoring.exhausted or summary.reviewer.exhausted:
            run.stage = WorkflowStage.blocked
            run.status = WorkflowStatus.blocked
            run.pending_gate = None
            return

        if not summary.review_ready:
            run.stage = WorkflowStage.needs_revision
            run.status = WorkflowStatus.active
            run.pending_gate = None
            return

        if run.stage == WorkflowStage.published:
            return

        if run.pending_gate is None or run.stage == WorkflowStage.needs_revision:
            run.stage = WorkflowStage.awaiting_hil_gate_1
            run.status = WorkflowStatus.awaiting_human
            run.pending_gate = HILGate.gate_1_spec_review
