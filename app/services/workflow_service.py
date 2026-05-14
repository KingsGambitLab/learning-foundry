from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.ai import AIUsageSummary
from app.domain.registry import RiskClass
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
    WorkflowNodeStatus,
    WorkflowLoopPhaseSummary,
    WorkflowLoopPolicy,
    WorkflowReviewSummary,
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
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs
from app.services.spec_validation import validate_task_agent_spec
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.storage.sqlite_store import SQLiteWorkflowStore


class WorkflowConflictError(ValueError):
    """Raised when a workflow transition is invalid."""


class WorkflowGateRefused(ValueError):
    """Raised when a legacy gate-approval cannot proceed without execution evidence.

    Codex review #7 critical finding: after Wave 5b deleted the
    per-deliverable LangGraph node loop, ``apply_gate_decision`` would
    advance gate 2 and gate 3 with no sandbox / tests / reviewer pass,
    letting a legacy run reach ``published`` on the back of nothing more
    than a structurally-valid task-agent spec. The guard refuses the
    approve path when the run has no concrete execution evidence
    (no passed runtime / tests / reviewer node and no materialized
    workspace snapshot). Routes translate this to HTTP 409.

    The reject path is unaffected — that is how
    ``CourseWorkflowService._route_publish_failure_to_shared_workflow_revision``
    legitimately reopens a shared workflow after certification fails.
    """


def _default_planner_deliverables(design_spec: AssignmentDesignSpec) -> list[DeliverableSpec]:
    """Build a minimal planner deliverable list when no planner is wired in.

    The deterministic scaffold builder always needs a planner-shaped
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
    """Persistence-and-materialization service for legacy workflow runs.

    Wave 5b retired the per-deliverable LangGraph authoring/reviewer loop;
    the only live entry point is the outcome graph driven from
    ``CourseGenerationService``. What remains here:

    * ``create_run`` / ``create_run_from_explicit_plan`` — build a workflow
      run shell with a deterministic task-agent scaffold (no LLM).
    * ``create_revision_from_run`` — clone-and-reset a run for course
      certification follow-up (used by ``CourseWorkflowService``).
    * ``apply_gate_decision`` — record HIL decisions; needed by the
      course publish/certification path.
    * ``materialize_run`` / ``read_bundle_file`` / ``get_workspace`` /
      ``read_workspace_file`` — produce learner-facing artifacts.
    * ``list_runs`` / ``get_run`` / ``list_events`` /
      ``list_node_executions`` / ``get_review_summary`` — API queries.

    All of the per-deliverable authoring loop methods
    (``execute_langgraph_nodes``, ``update_task_agent_spec``,
    ``grade_task_agent_run*``, ``list_task_agent_grader_plans``, etc.) were
    removed; the outcome graph owns that surface now.
    """

    def __init__(
        self,
        store: SQLiteWorkflowStore,
        materializer: ArtifactMaterializer | None = None,
        runner: TaskAgentBlackBoxRunner | None = None,
        workspace_manager: AssignmentWorkspaceManager | None = None,
    ) -> None:
        self.store = store
        self.materializer = materializer or ArtifactMaterializer()
        self.runner = runner or TaskAgentBlackBoxRunner()
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
        # The per-deliverable authoring/reviewer loop is retired; surface a
        # static empty summary so the legacy ``/v1/workflow-runs/{id}/review``
        # endpoint keeps a stable shape for any external pollers.
        if run.artifacts.review_summary is not None:
            return run.artifacts.review_summary
        return self._empty_review_summary()

    def materialize_run(self, run_id: str, request: MaterializeBundleRequest) -> WorkflowRun:
        run = self._require_run(run_id)
        if run.artifacts.task_agent_spec is not None:
            validation = validate_task_agent_spec(run.artifacts.task_agent_spec)
            run.artifacts.validation_summary = validation.model_dump(mode="json")
            run.artifacts.progression_preview = [summary.model_dump(mode="json") for summary in validation.deliverable_gates]
            if not validation.valid:
                self.store.save_run(run)
                raise WorkflowConflictError("Task-agent draft is invalid. Fix validation errors before materializing artifacts.")

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
        # ``execute_nodes`` is preserved as a kwarg only so callers in
        # course_workflow_service compile without modification. The
        # per-deliverable LangGraph node loop is retired (Wave 5b); this
        # arg is now a no-op.
        del execute_nodes
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
            planner_deliverables=planner_deliverables,
        )

    def create_run(
        self,
        intake: GenerationIntake,
        *,
        execute_nodes: bool = True,
        planner_deliverables: list[DeliverableSpec] | None = None,
    ) -> WorkflowRun:
        # ``execute_nodes`` is now a no-op (see ``create_run_from_explicit_plan``).
        del execute_nodes
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
            planner_deliverables=planner_deliverables,
        )

    def create_revision_from_run(self, run_id: str) -> WorkflowRun:
        source = self._require_run(run_id)
        now = datetime.now(UTC)
        revision = source.model_copy(deep=True)
        revision.id = f"run_{uuid4().hex[:12]}"
        revision.created_at = now
        revision.updated_at = now
        revision.stage = WorkflowStage.awaiting_hil_gate_1
        revision.status = WorkflowStatus.awaiting_human
        revision.pending_gate = HILGate.gate_1_spec_review
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
            f"Revision draft cloned from `{source.id}` and reset for a fresh authoring pass.",
        ]
        self.store.save_run(revision)
        self.store.append_event(
            revision.id,
            "run_revision_created",
            {"source_run_id": source.id},
        )
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
        # Deterministic scaffold builder — replaces the LLM authoring
        # service that was deleted in Wave 5b. The outcome graph is now
        # the live LLM-authoring surface; the legacy workflow run shell
        # only needs a structurally-valid TaskAgentServiceSpec.
        task_agent_spec, origin_template = build_task_agent_scaffold(
            title=intake.title,
            summary=intake.problem_statement,
            design_spec=design_spec,
            planner_deliverables=resolved_planner_deliverables,
        )
        task_agent_spec = ensure_task_agent_deliverable_briefs(task_agent_spec, overwrite=False)
        log_coursegen_event(
            "workflow_authoring_scaffold_generated",
            workflow_run_id=run_id,
            title=intake.title,
            source="deterministic_scaffold",
            origin_template=origin_template,
        )
        validation = validate_task_agent_spec(task_agent_spec)
        artifacts = WorkflowArtifacts(
            draft_kind=DraftKind.task_agent_spec,
            task_agent_spec=task_agent_spec,
            ai_usage=AIUsageSummary(),
            validation_summary=validation.model_dump(mode="json"),
            progression_preview=[summary.model_dump(mode="json") for summary in validation.deliverable_gates],
            artifact_plan=self._artifact_plan_for_task_agent(task_agent_spec),
            origin_template=origin_template,
            notes=[
                "Draft starter project created from the explicit assignment design spec.",
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
                "ai_usage": None,
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
        )
        return run

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

    def _write_progress_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # Node kinds that count as "real execution" for the legacy gate guard.
    # A passed status on any of these means the sandbox, tests, or reviewer
    # actually ran against the authored artifact — i.e. the artifact has
    # been observed working, not just declared schema-valid.
    _EXECUTION_EVIDENCE_KINDS: frozenset[WorkflowNodeKind] = frozenset(
        {
            WorkflowNodeKind.authoring_runtime,
            WorkflowNodeKind.authoring_tests,
            WorkflowNodeKind.reviewer_runtime,
            WorkflowNodeKind.reviewer_tests,
            WorkflowNodeKind.reviewer_code,
            WorkflowNodeKind.reviewer_pedagogy,
            WorkflowNodeKind.reviewer_learner_runtime,
        }
    )

    def _run_has_execution_evidence(self, run: WorkflowRun) -> bool:
        """Return True iff ``run`` carries proof that real execution happened.

        Two paths qualify:

        1. ``artifacts.node_executions`` contains at least one node of an
           execution-bearing kind (runtime, tests, reviewer) with
           ``status == passed``. Failed-only history doesn't count — the
           sandbox never confirmed the artifact works.
        2. ``artifacts.workspace_snapshot`` is present. The snapshot is
           only written after a successful materialize, so its presence
           is treated as evidence that a working bundle was produced.

        Returning False blocks approve-side gate advancement past gate 1
        on the legacy ``apply_gate_decision`` path (Codex review #7).
        """
        if run.artifacts.workspace_snapshot is not None:
            return True
        for node in run.artifacts.node_executions:
            if (
                node.kind in self._EXECUTION_EVIDENCE_KINDS
                and node.status == WorkflowNodeStatus.passed
            ):
                return True
        return False

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
                run.artifacts.notes.append("Human review feedback captured for the next authoring pass.")
            run.updated_at = now
            self.store.save_run(run)
            self.store.append_event(
                run.id,
                "gate_rejected",
                {
                    "gate": decision.gate.value,
                    "comment": decision.comment or "",
                    "rerun_requested": False,
                },
            )
            return run

        if decision.gate == HILGate.gate_1_spec_review:
            if run.artifacts.task_agent_spec is None:
                raise WorkflowConflictError("This workflow does not have a reviewable assignment spec yet.")
            validation = run.artifacts.validation_summary or {}
            if not validation.get("valid", False):
                raise WorkflowConflictError("The task-agent draft is not valid yet. Fix validation errors before requesting review.")
            run.stage = WorkflowStage.awaiting_hil_gate_2
            run.pending_gate = HILGate.gate_2_progression_review
        elif decision.gate == HILGate.gate_2_progression_review:
            # Codex review #7: gate 2 used to advance blindly. Wave 5b
            # removed the per-deliverable LangGraph loop, so a legacy
            # run can reach this point without any sandbox / tests /
            # reviewer pass. Refuse to advance toward publish unless
            # the run carries concrete execution evidence.
            if not self._run_has_execution_evidence(run):
                raise WorkflowGateRefused(
                    f"Cannot advance run '{run.id}' past gate 2 without execution "
                    "evidence. The legacy per-deliverable LangGraph loop was "
                    "retired in Wave 5b; outcome-mode generation is the live "
                    "publishing path. Use POST /v1/course-runs/<id>/decisions "
                    "for outcome runs."
                )
            run.stage = WorkflowStage.awaiting_hil_gate_3
            run.pending_gate = HILGate.gate_3_pre_publish
        elif decision.gate == HILGate.gate_3_pre_publish:
            # Same guard at gate 3 — this is the actual publish step,
            # so blocking without execution evidence is the load-bearing
            # check that prevents un-executed scaffolds from going live.
            if not self._run_has_execution_evidence(run):
                raise WorkflowGateRefused(
                    f"Cannot publish run '{run.id}' without execution evidence. "
                    "The legacy per-deliverable LangGraph loop was retired in "
                    "Wave 5b; outcome-mode generation is the live publishing "
                    "path. Use POST /v1/course-runs/<id>/decisions for outcome runs."
                )
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

    def _empty_review_summary(self) -> WorkflowReviewSummary:
        policy = WorkflowLoopPolicy(max_authoring_attempts=1, max_reviewer_attempts=1)
        return WorkflowReviewSummary(
            review_ready=False,
            blockers=["Workflow review has not run yet."],
            policy=policy,
            authoring=WorkflowLoopPhaseSummary(max_attempts=policy.max_authoring_attempts),
            reviewer=WorkflowLoopPhaseSummary(max_attempts=policy.max_reviewer_attempts),
        )
