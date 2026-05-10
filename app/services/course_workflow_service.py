from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.ai import AIUsageSummary, merge_ai_usage
from app.domain.course import (
    CourseAsyncOperation,
    CourseLinkedBundleSummary,
    CourseLinkedWorkflowSummary,
    CourseGenerationSource,
    CourseGenerationStatus,
    CreatorCourseSetupChoices,
    DraftTimelineItem,
    DraftTimelineResponse,
    DraftTimelineSourceKind,
    LocalDraftResetResult,
    CourseDeliverableDraft,
    CourseDeliverableReview,
    CourseReviewCounts,
    CourseReviewReport,
    CourseRun,
    CourseRunList,
    CourseRunSummary,
    CourseRunStage,
    CourseRunStatus,
    CreateCourseDeliverableRequest,
    CreateCourseRunRequest,
    GeneratedCoursePlan,
    QueueCourseOperationResponse,
    QueueCourseRevisionResponse,
)
from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import AssignmentDesignSpec, RetrievalMode
from app.domain.publish import PublishSnapshot, PublishedVersionList, PublishedVersionSummary
from app.domain.publish import PublishCertificationCheckStatus, PublishCertificationFailureOrigin, PublishLearnerCertificationReport
from app.domain.testing import (
    CreateCreatorFeedbackRequest,
    CreateLearnerEvaluationReportRequest,
    CreatorFeedbackList,
    CreatorFeedbackRecord,
    CreatorTestingView,
    LearnerCourseEvaluationReport,
    TestingDiagnostic,
    TestingDiagnosticSeverity,
)
from app.domain.workflow import (
    ArtifactVisibility,
    BundleFileContent,
    DecisionOutcome,
    DraftKind,
    GateDecisionRequest,
    HILGate,
    MaterializeBundleRequest,
    ReviewerFinding,
    ReviewerFindingSeverity,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowNodeStatus,
    WorkflowRun,
    WorkflowEvent,
    WorkflowStatus,
)
from app.services.course_artifact_materializer import CourseArtifactMaterializer
from app.services.course_patterns import CoursePattern, course_pattern_by_slug
from app.services.creator_asset_service import CreatorAssetService
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design, infer_risk_class
from app.services.lms_service import default_learner_workspace_dir
from app.services.publish_learner_certification_service import PublishLearnerCertificationService
from app.services.publish_snapshot_service import PublishSnapshotService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class CourseWorkflowConflictError(ValueError):
    """Raised when a course workflow transition is invalid."""


ASSIGNMENT_PACKAGE_TYPE = PackageType.progressive_codebase_course


class CourseWorkflowService:
    def __init__(
        self,
        store: SQLiteWorkflowStore,
        workflow_service: WorkflowService,
        materializer: CourseArtifactMaterializer | None = None,
        publish_snapshot_service: PublishSnapshotService | None = None,
        publish_certification_service: PublishLearnerCertificationService | None = None,
        creator_asset_service: CreatorAssetService | None = None,
        job_runner: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.store = store
        self.workflow_service = workflow_service
        self.materializer = materializer or CourseArtifactMaterializer()
        self.publish_snapshot_service = publish_snapshot_service or PublishSnapshotService(store, workflow_service)
        self.publish_certification_service = publish_certification_service or PublishLearnerCertificationService()
        self.creator_asset_service = creator_asset_service
        self.job_runner = job_runner or self._run_job_in_background

    def list_runs(self, limit: int = 50) -> CourseRunList:
        refreshed_runs: list[CourseRunSummary] = []
        for summary in self.store.list_course_runs(limit=limit):
            run = self.store.get_course_run(summary.id)
            if run is None:
                continue
            refreshed = self._compute_refreshed_run(run)
            self.store.save_course_run(refreshed)
            refreshed_runs.append(CourseRunSummary.from_run(refreshed))
        return CourseRunList(runs=refreshed_runs)

    def get_run(self, course_run_id: str) -> CourseRun | None:
        run = self.store.get_course_run(course_run_id)
        if run is None:
            return None
        refreshed = self._compute_refreshed_run(run)
        self.store.save_course_run(refreshed)
        return refreshed

    def list_events(self, course_run_id: str):
        return self.store.list_course_events(course_run_id)

    def generated_plan_from_run(
        self,
        course_run: CourseRun,
        *,
        notes: list[str] | None = None,
    ) -> GeneratedCoursePlan:
        return GeneratedCoursePlan(
            title=course_run.title,
            summary=course_run.summary,
            package_type=course_run.package_type,
            shared_design_spec=course_run.shared_design_spec,
            deliverables=[self._request_from_deliverable_draft(deliverable) for deliverable in course_run.deliverables],
            notes=list(notes or []),
        )

    def reset_local_state(self) -> LocalDraftResetResult:
        counts = self.store.reset_all()
        cleared_directories: list[str] = []
        candidate_dirs = {
            self.materializer.base_dir.resolve(),
            self.workflow_service.materializer.base_dir.resolve(),
            self.workflow_service.workspace_manager.base_dir.resolve(),
            default_learner_workspace_dir().resolve(),
        }
        if self.creator_asset_service is not None:
            candidate_dirs.add(self.creator_asset_service.base_dir.resolve())
        for directory in sorted(candidate_dirs):
            path = Path(directory)
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            cleared_directories.append(str(path))

        return LocalDraftResetResult(
            deleted_course_runs=counts["deleted_course_runs"],
            deleted_course_events=counts["deleted_course_events"],
            deleted_workflow_runs=counts["deleted_workflow_runs"],
            deleted_workflow_events=counts["deleted_workflow_events"],
            deleted_publish_snapshots=counts["deleted_publish_snapshots"],
            deleted_learner_enrollments=counts["deleted_learner_enrollments"],
            deleted_learner_submissions=counts["deleted_learner_submissions"],
            deleted_learner_workspace_sessions=counts["deleted_learner_workspace_sessions"],
            deleted_creator_feedback=counts.get("deleted_creator_feedback", 0),
            deleted_learner_feedback=counts.get("deleted_learner_feedback", 0),
            deleted_learner_eval_reports=counts.get("deleted_learner_eval_reports", 0),
            deleted_creator_assets=counts.get("deleted_creator_assets", 0),
            cleared_directories=cleared_directories,
        )

    def create_run(self, request: CreateCourseRunRequest) -> CourseRun:
        course_run_id = f"course_{uuid4().hex[:12]}"
        now = datetime.now(UTC)
        course_run = self._build_course_run(
            course_run_id=course_run_id,
            created_at=now,
            updated_at=now,
            request=request,
        )
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_run_created",
            {
                "package_type": course_run.package_type.value,
                "deliverable_count": len(course_run.deliverables),
                "shared_workflow_run_id": course_run.shared_workflow_run_id,
            },
        )
        return course_run

    def create_generation_placeholder(
        self,
        *,
        title: str,
        goal: str,
        learning_outcomes: list[str],
        package_type_hint: PackageType | None = None,
        creator_choices: CreatorCourseSetupChoices | None = None,
        generation_status: CourseGenerationStatus | None = None,
    ) -> CourseRun:
        intake = GenerationIntake(
            title=title,
            problem_statement=goal,
            learning_outcomes=learning_outcomes,
            package_type_hint=package_type_hint,
            starter_type=(creator_choices.starter_type if creator_choices is not None else None),
            implementation_language=(creator_choices.implementation_language if creator_choices is not None else None),
            language_version=(creator_choices.language_version if creator_choices is not None else None),
            application_framework=(creator_choices.application_framework if creator_choices is not None else None),
            framework_version=(creator_choices.framework_version if creator_choices is not None else None),
            package_manager=(creator_choices.package_manager if creator_choices is not None else None),
            primary_database=(creator_choices.primary_database if creator_choices is not None else None),
            primary_database_version=(creator_choices.primary_database_version if creator_choices is not None else None),
            cache_backend=(creator_choices.cache_backend if creator_choices is not None else None),
            cache_backend_version=(creator_choices.cache_backend_version if creator_choices is not None else None),
            tech_stack=(list(creator_choices.tech_stack) if creator_choices is not None else []),
            data_sources=(list(creator_choices.data_sources) if creator_choices is not None else []),
        )
        inferred = infer_assignment_design(
            title=title,
            problem_statement=goal,
            learning_outcomes=learning_outcomes,
            package_type_hint=package_type_hint,
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
        now = datetime.now(UTC)
        course_run_id = f"course_{uuid4().hex[:12]}"
        course_run = CourseRun(
            id=course_run_id,
            course_family_id=course_run_id,
            title=title,
            summary=goal.strip(),
            package_type=package_type_hint or inferred.package_type,
            creator_choices=creator_choices.model_copy(deep=True) if creator_choices is not None else None,
            shared_design_spec=inferred.design_spec,
            shared_workflow_run_id=None,
            created_at=now,
            updated_at=now,
            stage=CourseRunStage.drafting,
            status=CourseRunStatus.active,
            deliverables=[],
            notes=[
                "Draft created from the brief.",
                "Generation is in progress and the draft will update in place.",
            ],
            goal=goal,
            requested_learning_outcomes=list(learning_outcomes),
            generation_status=generation_status,
            active_operation=CourseAsyncOperation.generation,
        )
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_generation_queued",
            {
                "goal": goal,
                "learning_outcome_count": len(learning_outcomes),
                "package_type_hint": package_type_hint.value if package_type_hint is not None else None,
                "starter_type": creator_choices.starter_type.value if creator_choices is not None else None,
                "primary_database": creator_choices.primary_database if creator_choices is not None else None,
                "cache_backend": creator_choices.cache_backend if creator_choices is not None else None,
            },
        )
        return course_run

    def apply_generated_plan(
        self,
        course_run_id: str,
        *,
        plan: GeneratedCoursePlan,
        source: CourseGenerationSource,
        generation_status: CourseGenerationStatus,
        usage: AIUsageSummary | None = None,
        execute_shared_workflow_nodes: bool = True,
        clear_active_operation: bool = True,
    ) -> CourseRun:
        existing = self._require_run(course_run_id)
        course_run = self._build_course_run(
            course_run_id=existing.id,
            created_at=existing.created_at,
            updated_at=datetime.now(UTC),
            execute_shared_workflow_nodes=execute_shared_workflow_nodes,
            request=CreateCourseRunRequest(
                title=plan.title,
                summary=plan.summary,
                package_type=plan.package_type,
                creator_choices=existing.creator_choices,
                shared_design_spec=plan.shared_design_spec,
                course_family_id=existing.course_family_id,
                deliverables=plan.deliverables,
            ),
        )
        notes = [note for note in existing.notes if "Generation is in progress" not in note]
        notes.extend(
            [
                (
                    f"Course brief generated via `{source.value}`."
                    if source == CourseGenerationSource.openai_live
                    else "Course brief generated via deterministic fallback planning."
                ),
            ]
        )
        if generation_status.model_id:
            notes.append(f"Planner model: `{generation_status.model_id}`.")
        course_run.notes = list(dict.fromkeys(notes))
        course_run.goal = existing.goal
        course_run.requested_learning_outcomes = existing.requested_learning_outcomes
        course_run.generated_plan = self.generated_plan_from_run(course_run, notes=plan.notes)
        course_run.generation_source = source
        course_run.generation_status = generation_status
        course_run.own_ai_usage = merge_ai_usage(existing.own_ai_usage, usage)
        course_run.ai_usage = course_run.own_ai_usage
        course_run.active_operation = None if clear_active_operation else existing.active_operation
        course_run.last_error = None
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_brief_generated",
            {
                "source": source.value,
                "provider": generation_status.provider,
                "model_id": generation_status.model_id,
                "message": generation_status.message,
                "deliverable_count": len(course_run.deliverables),
                "ai_usage": (usage.model_dump(mode="json") if usage is not None else None),
            },
        )
        return course_run

    def mark_generation_failed(
        self,
        course_run_id: str,
        *,
        error: str,
        generation_status: CourseGenerationStatus,
    ) -> CourseRun:
        course_run = self._require_run(course_run_id)
        course_run.stage = CourseRunStage.blocked
        course_run.status = CourseRunStatus.blocked
        course_run.updated_at = datetime.now(UTC)
        course_run.generation_status = generation_status
        course_run.active_operation = None
        course_run.last_error = error
        course_run.notes = list(
            dict.fromkeys(
                [
                    *course_run.notes,
                    "Generation failed before the draft could be fully built.",
                ]
            )
        )
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_generation_failed",
            {
                "error": error,
                "message": generation_status.message,
                "provider": generation_status.provider,
                "model_id": generation_status.model_id,
            },
        )
        return course_run

    def _build_course_run(
        self,
        *,
        course_run_id: str,
        created_at: datetime,
        updated_at: datetime,
        execute_shared_workflow_nodes: bool = True,
        request: CreateCourseRunRequest,
    ) -> CourseRun:
        pattern = course_pattern_by_slug(request.pattern_slug) if request.pattern_slug else None
        if request.pattern_slug and pattern is None:
            raise ValueError(f"Unknown course pattern '{request.pattern_slug}'.")

        if pattern is None and not request.deliverables:
            raise ValueError("Custom course creation requires at least one deliverable.")

        title = request.title or (pattern.course_title if pattern else None)
        if not title:
            raise ValueError("Course title is required.")
        summary = request.summary or self._default_course_summary(pattern, title)
        package_type = request.package_type or (pattern.package_type if pattern else PackageType.survey_course)
        shared_design_spec = request.shared_design_spec or (pattern.shared_design_spec if pattern else None)

        deliverables = self._deliverable_requests_from_pattern(pattern, summary) if pattern else request.deliverables
        if not deliverables:
            raise ValueError("Course must contain at least one deliverable.")

        if package_type == PackageType.survey_course:
            deliverable_drafts = [self._create_survey_deliverable(deliverable, title, summary) for deliverable in deliverables]
            shared_workflow_run_id = None
        else:
            shared_run = self._create_progressive_workflow(
                title,
                summary,
                deliverables,
                shared_design_spec,
                execute_nodes=execute_shared_workflow_nodes,
            )
            shared_run = self._ensure_progressive_workflow_matches_deliverables(
                shared_run,
                deliverables,
                execute_nodes=execute_shared_workflow_nodes,
            )
            aligned_deliverables = self._align_progressive_deliverables(deliverables, shared_run)
            deliverable_drafts = [
                self._deliverable_draft_from_workflow(
                    deliverable,
                    shared_run.id,
                    shared_run.stage.value,
                    shared_run.status.value,
                    shared_run.artifacts.draft_kind.value,
                    self._design_spec_from_workflow(shared_run, deliverable.design_spec or shared_design_spec),
                    self._workflow_design_status(shared_run),
                    extra_notes=["Shared progressive workflow run for the whole course."],
                )
                for deliverable in aligned_deliverables
            ]
            shared_workflow_run_id = shared_run.id
            shared_design_spec = self._design_spec_from_workflow(shared_run, shared_design_spec)

        stage, status = self._course_stage_from_deliverables(deliverable_drafts)
        course_run = CourseRun(
            id=course_run_id,
            title=title,
            summary=summary,
            package_type=package_type,
            pattern_slug=pattern.course_slug if pattern else request.pattern_slug,
            course_family_id=request.course_family_id or course_run_id,
            creator_choices=request.creator_choices.model_copy(deep=True) if request.creator_choices is not None else None,
            shared_design_spec=shared_design_spec,
            shared_workflow_run_id=shared_workflow_run_id,
            created_at=created_at,
            updated_at=updated_at,
            stage=stage,
            status=status,
            deliverables=deliverable_drafts,
            notes=[
                "Course draft created from the course workflow layer.",
                "Each survey deliverable maps to its own assignment workflow run; progressive courses share one assignment workflow run.",
            ],
        )
        return course_run

    def create_revision(self, course_run_id: str) -> CourseRun:
        source = self._require_published_run(course_run_id)
        revision = self._build_revision_placeholder(source, queued=False)
        deliverable_drafts, shared_workflow_run_id = self._prepare_revision_drafts(source)
        return self._finalize_revision(
            revision=revision,
            source=source,
            deliverable_drafts=deliverable_drafts,
            shared_workflow_run_id=shared_workflow_run_id,
            event_type="course_revision_created",
        )

    def queue_revision(self, course_run_id: str) -> QueueCourseRevisionResponse:
        source = self._require_published_run(course_run_id)
        revision = self._build_revision_placeholder(source, queued=True)
        self.store.save_course_run(revision)
        self.store.append_course_event(
            revision.id,
            "course_revision_queued",
            {
                "source_course_run_id": source.id,
                "course_family_id": revision.course_family_id,
                "message": "New version draft queued. We are cloning the published course into a fresh draft.",
            },
        )
        self.store.append_course_event(
            revision.id,
            "course_revision_started",
            {
                "source_course_run_id": source.id,
                "message": "Cloning linked workflows and running assignment checks for the new course version.",
            },
        )
        self.job_runner(lambda: self._finish_queued_revision(revision.id, source.id))
        latest = self.get_run(revision.id) or revision
        return QueueCourseRevisionResponse(queued=True, course_run=latest)

    def sync_run(self, course_run_id: str) -> CourseRun:
        course_run = self._require_run(course_run_id)
        course_run = self._compute_refreshed_run(course_run)
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_run_synced",
            {
                "stage": course_run.stage.value,
                "status": course_run.status.value,
            },
        )
        return course_run

    def review_run(self, course_run_id: str) -> CourseReviewReport:
        course_run = self._compute_refreshed_run(self._require_run(course_run_id))
        linked_runs = self._linked_runs(course_run)
        return self._build_review_report(course_run, linked_runs)

    def creator_view(self, course_run_id: str) -> CreatorTestingView:
        course_run = self._compute_refreshed_run(self._require_run(course_run_id))
        linked_runs = self._linked_runs(course_run)
        review = self._build_review_report(course_run, linked_runs)
        return CreatorTestingView(
            course_run=CourseRunSummary.from_run(course_run),
            review=review,
            goal=course_run.goal,
            requested_learning_outcomes=list(course_run.requested_learning_outcomes),
            creator_choices=self._creator_choices(course_run),
            published_versions=self.list_published_versions(course_run_id),
            diagnostics=self._creator_diagnostics(course_run, review),
            creator_feedback=self.store.list_creator_feedback(course_run_id),
            latest_learner_evaluation=self.store.get_latest_learner_eval_report(
                course_run_id,
                course_run.latest_publish_snapshot_id,
            ),
        )

    def timeline(self, course_run_id: str) -> DraftTimelineResponse:
        course_run = self._compute_refreshed_run(self._require_run(course_run_id))
        self.store.save_course_run(course_run)
        linked_runs = self._linked_runs_with_shared(course_run)
        items: list[DraftTimelineItem] = []
        for event in self.store.list_course_events(course_run_id):
            if event.event_type == "course_run_synced":
                continue
            items.append(self._timeline_item_from_course_event(course_run, event))
        for workflow_run in linked_runs.values():
            for event in self.workflow_service.list_events(workflow_run.id):
                items.append(self._timeline_item_from_workflow_event(workflow_run, event))
            for node in workflow_run.artifacts.node_executions:
                items.append(self._timeline_item_from_workflow_node(workflow_run, node))
        items.sort(
            key=lambda item: (
                item.created_at,
                item.source_title,
                item.sequence_no if item.sequence_no is not None else 10_000,
                item.attempt if item.attempt is not None else 10_000,
                item.id,
            )
        )
        return DraftTimelineResponse(
            course_run=CourseRunSummary.from_run(course_run),
            shared_workflow_run_id=course_run.shared_workflow_run_id,
            linked_workflow_run_ids=list(linked_runs.keys()),
            items=items,
        )

    def record_creator_feedback(
        self,
        course_run_id: str,
        request: CreateCreatorFeedbackRequest,
    ) -> CreatorFeedbackRecord:
        course_run = self._compute_refreshed_run(self._require_run(course_run_id))
        if request.deliverable_slug is not None and all(deliverable.deliverable_slug != request.deliverable_slug for deliverable in course_run.deliverables):
            raise ValueError(f"Unknown deliverable '{request.deliverable_slug}' for course '{course_run_id}'.")
        valid_workflow_ids = {
            workflow_id
            for workflow_id in [course_run.shared_workflow_run_id, *[deliverable.workflow_run_id for deliverable in course_run.deliverables]]
            if workflow_id is not None
        }
        if request.workflow_run_id is not None and request.workflow_run_id not in valid_workflow_ids:
            raise ValueError(f"Workflow '{request.workflow_run_id}' is not linked to course '{course_run_id}'.")

        feedback = CreatorFeedbackRecord(
            id=f"creator_feedback_{uuid4().hex[:12]}",
            course_run_id=course_run_id,
            created_at=datetime.now(UTC),
            category=request.category.strip() or "general",
            summary=request.summary.strip(),
            details=request.details.strip() if request.details else None,
            rating=request.rating,
            deliverable_slug=request.deliverable_slug,
            workflow_run_id=request.workflow_run_id,
            stage=course_run.stage.value,
            status=course_run.status.value,
            context=request.context,
        )
        self.store.save_creator_feedback(feedback)
        self.store.append_course_event(
            course_run_id,
            "creator_feedback_recorded",
            {
                "feedback_id": feedback.id,
                "category": feedback.category,
                "deliverable_slug": feedback.deliverable_slug,
                "workflow_run_id": feedback.workflow_run_id,
                "rating": feedback.rating,
            },
        )
        return feedback

    def list_creator_feedback(self, course_run_id: str) -> CreatorFeedbackList:
        self._require_run(course_run_id)
        return CreatorFeedbackList(items=self.store.list_creator_feedback(course_run_id))

    def record_learner_evaluation(
        self,
        course_run_id: str,
        request: CreateLearnerEvaluationReportRequest,
    ) -> LearnerCourseEvaluationReport:
        course_run = self._require_run(course_run_id)
        latest_snapshot = self.store.get_latest_publish_snapshot(course_run_id=course_run_id)
        publish_snapshot_id = (
            request.publish_snapshot_id
            or course_run.latest_publish_snapshot_id
            or (latest_snapshot.id if latest_snapshot is not None else None)
        )
        if publish_snapshot_id is None:
            raise CourseWorkflowConflictError("This course does not have a publish snapshot to attach a learner evaluation report to.")
        snapshot = self.store.get_publish_snapshot(publish_snapshot_id)
        if snapshot is None or snapshot.course_run_id != course_run_id:
            raise CourseWorkflowConflictError(
                f"Publish snapshot '{publish_snapshot_id}' does not belong to course '{course_run_id}'."
            )
        if snapshot.learner_package is None:
            raise CourseWorkflowConflictError(
                f"Publish snapshot '{publish_snapshot_id}' does not have a learner package to evaluate."
            )

        learner_deliverable_ids = {
            deliverable.deliverable_id
            for deliverable in snapshot.learner_package.deliverables
        }
        unknown_deliverables = [
            result.deliverable_id
            for result in request.deliverable_results
            if result.deliverable_id not in learner_deliverable_ids
        ]
        if unknown_deliverables:
            raise CourseWorkflowConflictError(
                f"Learner evaluation report includes unknown deliverable ids for snapshot '{publish_snapshot_id}': {', '.join(unknown_deliverables)}."
            )

        learner_id = request.learner_id
        if request.enrollment_id is not None:
            enrollment = self.store.get_learner_enrollment(request.enrollment_id)
            if enrollment is None:
                raise KeyError(request.enrollment_id)
            if enrollment.course_run_id != course_run_id:
                raise CourseWorkflowConflictError(
                    f"Enrollment '{request.enrollment_id}' does not belong to course '{course_run_id}'."
                )
            if enrollment.publish_snapshot_id != publish_snapshot_id:
                raise CourseWorkflowConflictError(
                    f"Enrollment '{request.enrollment_id}' is pinned to snapshot '{enrollment.publish_snapshot_id}', not '{publish_snapshot_id}'."
                )
            learner_id = learner_id or enrollment.learner_id

        overall_status = (
            "passed"
            if all(
                result.good_attempt.status == "passed" and result.progression_observed
                for result in request.deliverable_results
            )
            else "failed"
        )
        report = LearnerCourseEvaluationReport(
            id=f"learner_eval_{uuid4().hex[:12]}",
            course_run_id=course_run_id,
            publish_snapshot_id=publish_snapshot_id,
            learner_id=learner_id,
            enrollment_id=request.enrollment_id,
            created_at=datetime.now(UTC),
            overall_status=overall_status,
            notes=list(request.notes),
            deliverable_results=request.deliverable_results,
        )
        self.store.save_learner_eval_report(report)
        self.store.append_course_event(
            course_run_id,
            "learner_evaluation_recorded",
            {
                "report_id": report.id,
                "publish_snapshot_id": publish_snapshot_id,
                "overall_status": report.overall_status,
                "deliverable_count": len(report.deliverable_results),
                "enrollment_id": request.enrollment_id,
            },
        )
        return report

    def get_latest_learner_evaluation(self, course_run_id: str) -> LearnerCourseEvaluationReport | None:
        course_run = self._require_run(course_run_id)
        return self.store.get_latest_learner_eval_report(course_run_id, course_run.latest_publish_snapshot_id)

    def list_published_versions(self, course_run_id: str) -> PublishedVersionList:
        course_run = self._require_run(course_run_id)
        snapshots = [
            self.store.get_publish_snapshot(summary.id)
            for summary in self.store.list_publish_snapshots(course_family_id=course_run.course_family_id, limit=100)
        ]
        materialized = [snapshot for snapshot in snapshots if snapshot is not None]
        enrollments = self.store.list_learner_enrollments(learner_id=None, limit=1000)
        enrollment_counts: dict[str, int] = {}
        for enrollment_summary in enrollments:
            enrollment = self.store.get_learner_enrollment(enrollment_summary.id)
            if enrollment is None:
                continue
            enrollment_counts[enrollment.publish_snapshot_id] = enrollment_counts.get(enrollment.publish_snapshot_id, 0) + 1

        versions: list[PublishedVersionSummary] = []
        ordered = sorted(materialized, key=lambda item: item.version, reverse=True)
        for index, snapshot in enumerate(ordered):
            previous = ordered[index + 1] if index + 1 < len(ordered) else None
            learner_package = snapshot.learner_package
            versions.append(
                PublishedVersionSummary(
                    snapshot_id=snapshot.id,
                    course_run_id=course_run.id,
                    version=snapshot.version,
                    created_at=snapshot.created_at,
                    is_latest=index == 0,
                    default_for_new_enrollments=course_run.latest_publish_snapshot_id == snapshot.id or (index == 0 and course_run.latest_publish_snapshot_id is None),
                    learner_count=enrollment_counts.get(snapshot.id, 0),
                    deliverable_count=len(learner_package.deliverables) if learner_package is not None else 0,
                    compatibility="new_enrollments_only",
                    changes=self._published_version_changes(snapshot, previous),
                )
            )
        return PublishedVersionList(versions=versions)

    def queue_publish_run(self, course_run_id: str) -> QueueCourseOperationResponse:
        course_run = self.sync_run(course_run_id)
        self._ensure_no_active_operation(course_run)
        if course_run.stage != CourseRunStage.ready_to_publish:
            raise CourseWorkflowConflictError(self._publish_readiness_error(course_run))
        course_run.active_operation = CourseAsyncOperation.publish
        course_run.updated_at = datetime.now(UTC)
        course_run.last_error = None
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_publish_queued",
            {
                "message": "Publishing is queued. We are preparing the learner-facing snapshot now.",
            },
        )
        self.store.append_course_event(
            course_run.id,
            "course_publish_started",
            {
                "message": "Creating the publish snapshot and learner package for this course.",
            },
        )
        self.job_runner(lambda: self._finish_queued_publish(course_run.id))
        latest = self.get_run(course_run.id) or course_run
        return QueueCourseOperationResponse(
            queued=True,
            operation=CourseAsyncOperation.publish,
            course_run=latest,
        )

    def publish_run(self, course_run_id: str) -> CourseRun:
        course_run = self.sync_run(course_run_id)
        if course_run.active_operation not in (None, CourseAsyncOperation.publish):
            raise CourseWorkflowConflictError(
                f"This course is already busy with `{course_run.active_operation.value}`. Wait for it to finish before starting another author action."
            )
        return self._execute_publish_run(course_run_id)

    def queue_materialize_run(
        self,
        course_run_id: str,
        request: MaterializeBundleRequest,
    ) -> QueueCourseOperationResponse:
        course_run = self.sync_run(course_run_id)
        self._ensure_no_active_operation(course_run)
        course_run.active_operation = CourseAsyncOperation.materialize
        course_run.updated_at = datetime.now(UTC)
        course_run.last_error = None
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_materialize_queued",
            {
                "overwrite": request.overwrite,
                "message": "Bundle materialization is queued. We are preparing the author review package now.",
            },
        )
        self.store.append_course_event(
            course_run.id,
            "course_materialize_started",
            {
                "overwrite": request.overwrite,
                "message": "Building the author review bundle and collecting linked workflow outputs.",
            },
        )
        self.job_runner(lambda: self._finish_queued_materialize(course_run.id, request))
        latest = self.get_run(course_run.id) or course_run
        return QueueCourseOperationResponse(
            queued=True,
            operation=CourseAsyncOperation.materialize,
            course_run=latest,
        )

    def materialize_run(self, course_run_id: str, request: MaterializeBundleRequest) -> CourseRun:
        course_run = self.sync_run(course_run_id)
        if course_run.active_operation not in (None, CourseAsyncOperation.materialize):
            raise CourseWorkflowConflictError(
                f"This course is already busy with `{course_run.active_operation.value}`. Wait for it to finish before starting another author action."
            )
        return self._execute_materialize_run(course_run_id, request)

    def read_bundle_file(self, course_run_id: str, relative_path: str) -> BundleFileContent:
        course_run = self._require_run(course_run_id)
        if course_run.materialized_bundle is None:
            raise CourseWorkflowConflictError("This course run has not been materialized yet.")
        return self.materializer.read_bundle_file(course_run.materialized_bundle, relative_path)

    def _compute_refreshed_run(self, course_run: CourseRun) -> CourseRun:
        if course_run.stage == CourseRunStage.drafting and not course_run.deliverables:
            course_run.ai_usage = course_run.own_ai_usage
            return course_run
        refreshed_run = course_run.model_copy(deep=True)
        deliverable_drafts: list[CourseDeliverableDraft] = []
        shared_run = None
        linked_runs: list[WorkflowRun] = []
        if refreshed_run.shared_workflow_run_id is not None:
            shared_run = self.workflow_service.get_run(refreshed_run.shared_workflow_run_id)

        if shared_run is not None:
            linked_runs.append(shared_run)
            aligned_deliverables = self._align_progressive_deliverables(
                [self._request_from_deliverable_draft(deliverable) for deliverable in refreshed_run.deliverables],
                shared_run,
            )
            deliverable_drafts = [
                self._deliverable_draft_from_workflow(
                    deliverable,
                    shared_run.id,
                    shared_run.stage.value,
                    shared_run.status.value,
                    shared_run.artifacts.draft_kind.value,
                    self._design_spec_from_workflow(shared_run, deliverable.design_spec or refreshed_run.shared_design_spec),
                    self._workflow_design_status(shared_run),
                    extra_notes=["Shared progressive workflow run for the whole course."],
                )
                for deliverable in aligned_deliverables
            ]
            refreshed_run.shared_design_spec = self._design_spec_from_workflow(shared_run, refreshed_run.shared_design_spec)
        else:
            for deliverable in refreshed_run.deliverables:
                if not deliverable.workflow_run_id:
                    deliverable_drafts.append(deliverable)
                    continue
                child_run = self.workflow_service.get_run(deliverable.workflow_run_id)
                if child_run is None:
                    refreshed = deliverable.model_copy(deep=True)
                    refreshed.workflow_stage = "missing"
                    refreshed.workflow_status = "blocked"
                    refreshed.notes.append("Linked assignment workflow run is missing.")
                    deliverable_drafts.append(refreshed)
                    continue
                linked_runs.append(child_run)
                deliverable_drafts.append(
                    self._deliverable_draft_from_workflow(
                        self._request_from_deliverable_draft(deliverable),
                        child_run.id,
                        child_run.stage.value,
                        child_run.status.value,
                        child_run.artifacts.draft_kind.value,
                        self._design_spec_from_workflow(child_run, deliverable.design_spec),
                        self._workflow_design_status(child_run),
                        extra_notes=[note for note in deliverable.notes if "Shared progressive workflow run" in note],
                    )
                )

        generation_in_progress = refreshed_run.active_operation == CourseAsyncOperation.generation
        stage, status = self._course_stage_from_deliverables(deliverable_drafts)
        if generation_in_progress:
            stage = CourseRunStage.drafting
            status = CourseRunStatus.active
        refreshed_run.deliverables = deliverable_drafts
        refreshed_run.stage = stage if refreshed_run.status != CourseRunStatus.published else CourseRunStage.published
        refreshed_run.status = status if refreshed_run.status != CourseRunStatus.published else CourseRunStatus.published
        refreshed_run.updated_at = datetime.now(UTC)
        refreshed_run.ai_usage = merge_ai_usage(
            refreshed_run.own_ai_usage,
            *[run.artifacts.ai_usage for run in linked_runs],
        )
        if refreshed_run.generated_plan is not None:
            refreshed_run.generated_plan = self.generated_plan_from_run(
                refreshed_run,
                notes=refreshed_run.generated_plan.notes,
            )
        return refreshed_run

    def _execute_publish_run(self, course_run_id: str) -> CourseRun:
        course_run = self.sync_run(course_run_id)
        if course_run.stage != CourseRunStage.ready_to_publish:
            raise CourseWorkflowConflictError(self._publish_readiness_error(course_run))

        linked_runs = self._linked_runs(course_run)
        snapshot = self.publish_snapshot_service.create_snapshot(course_run, linked_runs, persist=False)
        if snapshot is None:
            raise CourseWorkflowConflictError(
                "This course cannot publish yet because a linked assignment workflow does not have a learner-ready spec."
            )
        certification = self.publish_certification_service.certify_snapshot(snapshot)
        snapshot.learner_certification = certification
        if not certification.passed:
            self._handle_publish_certification_failure(course_run, linked_runs, certification)
        self.store.save_publish_snapshot(snapshot)
        course_run.stage = CourseRunStage.published
        course_run.status = CourseRunStatus.published
        course_run.updated_at = datetime.now(UTC)
        course_run.active_operation = None
        course_run.last_error = None
        course_run.latest_publish_snapshot_id = snapshot.id if snapshot is not None else None
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_run_published",
            {
                "deliverable_count": len(course_run.deliverables),
                "publish_snapshot_id": course_run.latest_publish_snapshot_id,
            },
        )
        return course_run

    def _handle_publish_certification_failure(
        self,
        course_run: CourseRun,
        linked_runs: dict[str, WorkflowRun],
        certification: PublishLearnerCertificationReport,
    ) -> None:
        blocking_failures = certification.blocking_failures
        primary_failure = blocking_failures[0] if blocking_failures else None
        detail = primary_failure.detail if primary_failure is not None else None
        summary = primary_failure.summary if primary_failure is not None else "Learner-path certification failed."
        message = summary if detail is None else f"{summary} {detail}"

        if (
            certification.failure_origin == PublishCertificationFailureOrigin.repairable_generation
            and course_run.shared_workflow_run_id is not None
        ):
            self._route_publish_failure_to_shared_workflow_revision(
                course_run,
                linked_runs,
                certification,
            )
            self.store.append_course_event(
                course_run.id,
                "course_publish_certification_failed",
                {
                    "message": "Learner-path certification failed and the shared assignment workflow was sent back for revision.",
                    "failure_origin": certification.failure_origin.value,
                    "summary": summary,
                    "detail": detail,
                },
            )
            raise CourseWorkflowConflictError(
                "Learner-path certification failed on the exact publish snapshot. We routed the shared assignment workflow back into revision with the certification findings."
            )

        course_run.updated_at = datetime.now(UTC)
        course_run.active_operation = None
        course_run.last_error = message
        course_run.notes = list(
            dict.fromkeys(
                [
                    *course_run.notes,
                    "Publish was blocked because the exact learner path failed certification.",
                ]
            )
        )
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_publish_certification_failed",
            {
                "message": "Learner-path certification failed and publish was blocked.",
                "failure_origin": (
                    certification.failure_origin.value
                    if certification.failure_origin is not None
                    else None
                ),
                "summary": summary,
                "detail": detail,
            },
        )
        raise CourseWorkflowConflictError(message)

    def _route_publish_failure_to_shared_workflow_revision(
        self,
        course_run: CourseRun,
        linked_runs: dict[str, WorkflowRun],
        certification: PublishLearnerCertificationReport,
    ) -> None:
        shared_workflow_run_id = course_run.shared_workflow_run_id
        if shared_workflow_run_id is None:
            raise CourseWorkflowConflictError("This course is missing the shared workflow needed for repair routing.")

        shared_run = linked_runs.get(shared_workflow_run_id) or self.workflow_service.get_run(shared_workflow_run_id)
        if shared_run is None:
            raise CourseWorkflowConflictError(
                f"Shared workflow '{shared_workflow_run_id}' is missing, so certification findings cannot be routed back into revision."
            )

        revision = self.workflow_service.create_revision_from_run(shared_run.id)
        revision = self._append_publish_certification_failure_node(revision, certification)
        self.store.save_run(revision)

        pending_gate = revision.pending_gate or HILGate.gate_1_spec_review
        revision = self.workflow_service.apply_gate_decision(
            revision.id,
            GateDecisionRequest(
                gate=pending_gate,
                decision=DecisionOutcome.reject,
                comment=self._publish_certification_feedback_comment(certification),
            ),
        )
        revision = self._append_publish_certification_failure_node(revision, certification)
        self.store.save_run(revision)

        aligned_deliverables = self._align_progressive_deliverables(
            [self._request_from_deliverable_draft(deliverable) for deliverable in course_run.deliverables],
            revision,
        )
        deliverable_drafts = [
            self._deliverable_draft_from_workflow(
                deliverable,
                revision.id,
                revision.stage.value,
                revision.status.value,
                revision.artifacts.draft_kind.value,
                self._design_spec_from_workflow(revision, deliverable.design_spec or course_run.shared_design_spec),
                self._workflow_design_status(revision),
                extra_notes=[
                    "Shared workflow revision created after learner-path certification failed before publish.",
                ],
            )
            for deliverable in aligned_deliverables
        ]
        stage, status = self._course_stage_from_deliverables(deliverable_drafts)
        course_run.shared_workflow_run_id = revision.id
        course_run.shared_design_spec = self._design_spec_from_workflow(revision, course_run.shared_design_spec)
        course_run.deliverables = deliverable_drafts
        course_run.stage = stage
        course_run.status = status
        course_run.updated_at = datetime.now(UTC)
        course_run.active_operation = None
        course_run.last_error = "Learner-path certification failed before publish; the linked workflow was reopened for revision."
        course_run.notes = list(
            dict.fromkeys(
                [
                    *course_run.notes,
                    "Publish certification found a repairable learner-path problem and reopened the shared workflow for revision.",
                ]
            )
        )
        self.store.save_course_run(course_run)

    def _append_publish_certification_failure_node(
        self,
        run: WorkflowRun,
        certification: PublishLearnerCertificationReport,
    ) -> WorkflowRun:
        findings = [
            ReviewerFinding(
                category="learner_runtime_review",
                severity=(
                    ReviewerFindingSeverity.error
                    if check.status == PublishCertificationCheckStatus.failed
                    else ReviewerFindingSeverity.info
                ),
                title=check.summary,
                detail=check.detail or check.summary,
            )
            for check in certification.checks
            if check.status != PublishCertificationCheckStatus.skipped
        ]
        node_executions = list(run.artifacts.node_executions)
        next_iteration = max((node.iteration for node in node_executions), default=0) + 1
        node_executions.append(
            WorkflowNodeExecution(
                node_id=f"{WorkflowNodeKind.reviewer_learner_runtime.value}_{len(node_executions) + 1}",
                kind=WorkflowNodeKind.reviewer_learner_runtime,
                iteration=next_iteration,
                attempt=1,
                status=WorkflowNodeStatus.failed,
                summary="Final learner-path certification failed before publish.",
                created_at=datetime.now(UTC),
                sandbox_result=None,
                findings=findings,
            )
        )
        run.artifacts.node_executions = node_executions
        run.artifacts.notes = list(
            dict.fromkeys(
                [
                    *run.artifacts.notes,
                    "Final learner-path certification failed before publish and was attached as reviewer context.",
                ]
            )
        )
        return run

    def _publish_certification_feedback_comment(
        self,
        certification: PublishLearnerCertificationReport,
    ) -> str:
        failed_checks = [check for check in certification.checks if check.status == PublishCertificationCheckStatus.failed]
        lines = [
            "Final learner-path certification failed before publish.",
            "Fix the generated learner package and runtime path so the exact published learner experience works cleanly.",
        ]
        for check in failed_checks[:5]:
            lines.append(f"- {check.summary}")
            if check.detail:
                lines.append(f"  Detail: {check.detail}")
        return "\n".join(lines)

    def _execute_materialize_run(self, course_run_id: str, request: MaterializeBundleRequest) -> CourseRun:
        course_run = self.sync_run(course_run_id)
        linked_runs = self._linked_runs(course_run)
        review_report = self._build_review_report(course_run, linked_runs)
        bundle = self.materializer.materialize_course_run(
            course_run,
            linked_runs,
            review_report=review_report,
            overwrite=request.overwrite,
        )
        course_run.materialized_bundle = bundle
        course_run.updated_at = datetime.now(UTC)
        course_run.active_operation = None
        course_run.last_error = None
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            "course_bundle_materialized",
            {"bundle_id": bundle.bundle_id, "file_count": len(bundle.files)},
        )
        return course_run

    def _finish_queued_publish(self, course_run_id: str) -> None:
        try:
            self._execute_publish_run(course_run_id)
            self.store.append_course_event(
                course_run_id,
                "course_publish_completed",
                {
                    "message": "The course is published and new learners will now get this latest version.",
                },
            )
        except Exception as exc:
            self._mark_operation_failed(
                course_run_id,
                operation=CourseAsyncOperation.publish,
                event_type="course_publish_failed",
                error=str(exc),
                message="Publishing the course failed.",
            )

    def _finish_queued_materialize(self, course_run_id: str, request: MaterializeBundleRequest) -> None:
        try:
            self._execute_materialize_run(course_run_id, request)
            self.store.append_course_event(
                course_run_id,
                "course_materialize_completed",
                {
                    "overwrite": request.overwrite,
                    "message": "The author review bundle is ready.",
                },
            )
        except Exception as exc:
            self._mark_operation_failed(
                course_run_id,
                operation=CourseAsyncOperation.materialize,
                event_type="course_materialize_failed",
                error=str(exc),
                message="Materializing the course bundle failed.",
            )

    def _mark_operation_failed(
        self,
        course_run_id: str,
        *,
        operation: CourseAsyncOperation,
        event_type: str,
        error: str,
        message: str,
    ) -> None:
        course_run = self._require_run(course_run_id)
        course_run.active_operation = None
        course_run.updated_at = datetime.now(UTC)
        course_run.last_error = error
        course_run.notes = list(
            dict.fromkeys(
                [
                    *course_run.notes,
                    f"{operation.value.capitalize()} action failed and needs attention before you retry it.",
                ]
            )
        )
        self.store.save_course_run(course_run)
        self.store.append_course_event(
            course_run.id,
            event_type,
            {
                "error": error,
                "message": message,
            },
        )

    def _ensure_no_active_operation(self, course_run: CourseRun) -> None:
        if course_run.active_operation is not None:
            raise CourseWorkflowConflictError(
                f"This course is already busy with `{course_run.active_operation.value}`. Wait for it to finish before starting another author action."
            )

    def _build_review_report(
        self,
        course_run: CourseRun,
        linked_runs: dict[str, WorkflowRun],
    ) -> CourseReviewReport:
        generation_in_progress = course_run.active_operation == CourseAsyncOperation.generation
        deliverable_reports: list[CourseDeliverableReview] = []
        linked_workflows: list[CourseLinkedWorkflowSummary] = []
        linked_workflow_ids: set[str] = set()
        ready_deliverables = 0
        blocked_deliverables = 0
        deliverables_with_bundle = 0
        blockers: list[str] = [course_run.last_error] if course_run.last_error else []

        for position, deliverable in enumerate(course_run.deliverables, start=1):
            linked_run = linked_runs.get(deliverable.workflow_run_id) if deliverable.workflow_run_id else None
            linked_summary = self._linked_workflow_summary(linked_run)
            deliverable_blockers = self._deliverable_blockers(
                deliverable,
                linked_run,
                generation_in_progress=generation_in_progress,
            )
            bundle_available = bool(
                linked_run and linked_run.artifacts.materialized_bundle is not None
            )
            ready_for_publish = (
                linked_run is not None
                and linked_run.status == WorkflowStatus.published
                and linked_run.artifacts.task_agent_spec is not None
            )

            if bundle_available:
                deliverables_with_bundle += 1
            if ready_for_publish:
                ready_deliverables += 1
            if deliverable_blockers:
                blocked_deliverables += 1
                blockers.extend(
                    f"Deliverable {position} ({deliverable.title}): {reason}"
                    for reason in deliverable_blockers
                )
            if linked_summary is not None and linked_summary.run_id not in linked_workflow_ids:
                linked_workflows.append(linked_summary)
                linked_workflow_ids.add(linked_summary.run_id)

            deliverable_reports.append(
                CourseDeliverableReview(
                    position=position,
                    deliverable_slug=deliverable.deliverable_slug,
                    title=deliverable.title,
                    summary=deliverable.summary,
                    design_spec=deliverable.design_spec,
                    domain_pack=deliverable.domain_pack,
                    overlays=deliverable.overlays,
                    learning_outcomes=deliverable.learning_outcomes,
                    workflow_run_id=deliverable.workflow_run_id,
                    workflow_stage=deliverable.workflow_stage,
                    workflow_status=deliverable.workflow_status,
                    recommendation_status=deliverable.recommendation_status,
                    ready_for_publish=ready_for_publish,
                    bundle_available=bundle_available,
                    blockers=deliverable_blockers,
                    linked_workflow=linked_summary,
                    notes=deliverable.notes,
                )
            )

        counts = CourseReviewCounts(
            total_deliverables=len(course_run.deliverables),
            ready_deliverables=ready_deliverables,
            deliverables_with_blockers=blocked_deliverables,
            deliverables_with_bundle=deliverables_with_bundle,
            linked_workflow_runs=len(linked_workflows),
            published_workflow_runs=sum(1 for item in linked_workflows if item.status == WorkflowStatus.published),
            workflow_runs_with_bundle=sum(1 for item in linked_workflows if item.bundle is not None),
        )

        return CourseReviewReport(
            course_run_id=course_run.id,
            title=course_run.title,
            package_type=course_run.package_type,
            stage=course_run.stage,
            status=course_run.status,
            shared_design_spec=course_run.shared_design_spec,
            shared_workflow_run_id=course_run.shared_workflow_run_id,
            materialized_bundle=course_run.materialized_bundle,
            counts=counts,
            blockers=blockers,
            next_actions=self._next_actions(course_run, linked_workflows),
            linked_workflows=linked_workflows,
            deliverables=deliverable_reports,
        )

    def _create_survey_deliverable(self, deliverable: CreateCourseDeliverableRequest, course_title: str, course_summary: str) -> CourseDeliverableDraft:
        intake = self._generation_intake_from_design_context(
            title=deliverable.title,
            problem_statement=(
                deliverable.summary
                or f"Build the '{deliverable.title}' assignment for the '{course_title}' course. {course_summary}"
            ),
            learning_outcomes=deliverable.learning_outcomes,
            package_type_hint=ASSIGNMENT_PACKAGE_TYPE,
            design_spec=deliverable.design_spec,
        )
        if deliverable.design_spec is not None:
            child_run = self.workflow_service.create_run_from_explicit_plan(
                intake=intake,
                design_spec=deliverable.design_spec,
                reasons=[f"Seeded from course deliverable '{deliverable.title}'."],
                notes=["Created from the survey-course deliverable planner."],
            )
        else:
            child_run = self.workflow_service.create_run(intake)

        return self._deliverable_draft_from_workflow(
            deliverable,
            child_run.id,
            child_run.stage.value,
            child_run.status.value,
            child_run.artifacts.draft_kind.value,
            self._design_spec_from_workflow(child_run, deliverable.design_spec),
            self._workflow_design_status(child_run),
        )

    def _create_progressive_workflow(
        self,
        title: str,
        summary: str,
        deliverables: list[CreateCourseDeliverableRequest],
        shared_design_spec: AssignmentDesignSpec | None,
        *,
        execute_nodes: bool = True,
    ):
        course_intake = self._generation_intake_from_design_context(
            title=title,
            problem_statement=summary,
            learning_outcomes=self._combined_learning_outcomes(deliverables),
            package_type_hint=ASSIGNMENT_PACKAGE_TYPE,
            design_spec=shared_design_spec,
        )
        if shared_design_spec is not None:
            return self.workflow_service.create_run_from_explicit_plan(
                intake=course_intake,
                design_spec=shared_design_spec,
                reasons=["Created from the progressive course planner."],
                notes=["This workflow run is shared across all course deliverables."],
                execute_nodes=execute_nodes,
            )

        return self.workflow_service.create_run(course_intake, execute_nodes=execute_nodes)

    def _deliverable_draft_from_workflow(
        self,
        deliverable: CreateCourseDeliverableRequest,
        workflow_run_id: str,
        workflow_stage: str,
        workflow_status: str,
        draft_kind: str,
        design_spec: AssignmentDesignSpec | None,
        recommendation_status: str | None,
        extra_notes: list[str] | None = None,
    ) -> CourseDeliverableDraft:
        return CourseDeliverableDraft(
            deliverable_slug=deliverable.deliverable_slug or self._slugify(deliverable.title),
            title=deliverable.title,
            summary=deliverable.summary or deliverable.title,
            learning_outcomes=deliverable.learning_outcomes,
            design_spec=design_spec,
            domain_pack=(design_spec.domain_pack if design_spec is not None else deliverable.domain_pack_hint),
            overlays=list(design_spec.overlays if design_spec is not None else deliverable.overlays_hint),
            workflow_run_id=workflow_run_id,
            workflow_stage=workflow_stage,
            workflow_status=workflow_status,
            draft_kind=draft_kind,
            recommendation_status=recommendation_status,
            notes=(extra_notes or []),
        )

    def _deliverable_requests_from_pattern(self, pattern: CoursePattern | None, course_summary: str) -> list[CreateCourseDeliverableRequest]:
        if pattern is None:
            return []
        deliverables: list[CreateCourseDeliverableRequest] = []
        for deliverable in pattern.deliverables:
            deliverables.append(
                CreateCourseDeliverableRequest(
                    deliverable_slug=deliverable.deliverable_slug,
                    title=deliverable.title,
                    summary=f"Build the '{deliverable.title}' deliverable for the '{pattern.course_title}' course. {course_summary}",
                    learning_outcomes=self._default_learning_outcomes(deliverable.design_spec),
                    design_spec=deliverable.design_spec,
                    domain_pack_hint=deliverable.domain_pack,
                    overlays_hint=deliverable.overlays,
                )
            )
        return deliverables

    def _combined_learning_outcomes(self, deliverables: list[CreateCourseDeliverableRequest]) -> list[str]:
        seen: list[str] = []
        for deliverable in deliverables:
            for outcome in deliverable.learning_outcomes or self._default_learning_outcomes(deliverable.design_spec):
                if outcome not in seen:
                    seen.append(outcome)
        return seen[:8]

    def _request_from_deliverable_draft(self, deliverable: CourseDeliverableDraft) -> CreateCourseDeliverableRequest:
        return CreateCourseDeliverableRequest(
            deliverable_slug=deliverable.deliverable_slug,
            title=deliverable.title,
            summary=deliverable.summary,
            learning_outcomes=deliverable.learning_outcomes,
            design_spec=deliverable.design_spec,
            domain_pack_hint=deliverable.domain_pack,
            overlays_hint=deliverable.overlays,
        )

    def _align_progressive_deliverables(
        self,
        deliverables: list[CreateCourseDeliverableRequest],
        shared_run: WorkflowRun,
    ) -> list[CreateCourseDeliverableRequest]:
        if shared_run.artifacts.task_agent_spec is None or not deliverables:
            return deliverables
        authored_deliverables = list(shared_run.artifacts.task_agent_spec.deliverables)
        aligned: list[CreateCourseDeliverableRequest] = []
        for index, deliverable in enumerate(deliverables):
            if index >= len(authored_deliverables):
                aligned.append(deliverable)
                continue
            authored = authored_deliverables[index]
            aligned.append(
                deliverable.model_copy(
                    update={
                        "deliverable_slug": deliverable.deliverable_slug or authored.id,
                        "title": authored.title.strip(),
                        "summary": authored.objective.strip(),
                        "learning_outcomes": list(authored.learning_outcomes or deliverable.learning_outcomes),
                    }
                )
            )
        return aligned

    def _ensure_progressive_workflow_matches_deliverables(
        self,
        shared_run: WorkflowRun,
        deliverables: list[CreateCourseDeliverableRequest],
        *,
        execute_nodes: bool = True,
    ) -> WorkflowRun:
        del deliverables, execute_nodes
        return shared_run

    def _course_stage_from_deliverables(self, deliverables: list[CourseDeliverableDraft]) -> tuple[CourseRunStage, CourseRunStatus]:
        statuses = {deliverable.workflow_status for deliverable in deliverables}
        if not deliverables or None in statuses:
            return CourseRunStage.blocked, CourseRunStatus.blocked
        if "blocked" in statuses:
            return CourseRunStage.blocked, CourseRunStatus.blocked
        if statuses == {"published"}:
            if any(deliverable.draft_kind != DraftKind.task_agent_spec.value for deliverable in deliverables):
                return CourseRunStage.awaiting_course_review, CourseRunStatus.awaiting_human
            return CourseRunStage.ready_to_publish, CourseRunStatus.awaiting_human
        return CourseRunStage.awaiting_course_review, CourseRunStatus.awaiting_human

    def _publish_readiness_error(self, course_run: CourseRun) -> str:
        if any(deliverable.draft_kind != DraftKind.task_agent_spec.value for deliverable in course_run.deliverables):
            return "This course cannot publish yet because a linked assignment workflow does not have a learner-ready spec."
        return "All linked assignment workflow runs must be published before the course can be published."

    def _creator_choices(self, course_run: CourseRun):
        if course_run.creator_choices is not None:
            return course_run.creator_choices.model_copy(deep=True)
        shared_design = course_run.shared_design_spec
        runtime_dependencies = shared_design.runtime_dependencies if shared_design is not None else None
        runtime_plan = (
            shared_design.project_contract.runtime_plan
            if shared_design is not None
            else None
        )
        if runtime_dependencies is None:
            return None
        from app.domain.course import CreatorCourseSetupChoices

        def service_version(service_id: str | None) -> str | None:
            if runtime_plan is None or not service_id:
                return None
            service = next(
                (candidate for candidate in runtime_plan.services if candidate.service_id == service_id),
                None,
            )
            return service.version_hint if service is not None else None

        return CreatorCourseSetupChoices(
            starter_type=runtime_dependencies.starter_type,
            implementation_language=runtime_dependencies.implementation_language,
            language_version=runtime_dependencies.language_version or (runtime_plan.language_version if runtime_plan else None),
            application_framework=runtime_dependencies.application_framework,
            framework_version=runtime_dependencies.framework_version or (runtime_plan.framework_version if runtime_plan else None),
            package_manager=runtime_dependencies.package_manager or (runtime_plan.package_manager if runtime_plan else None),
            primary_database=runtime_dependencies.primary_database,
            primary_database_version=runtime_dependencies.primary_database_version or service_version(runtime_dependencies.primary_database),
            cache_backend=runtime_dependencies.cache_backend,
            cache_backend_version=runtime_dependencies.cache_backend_version or service_version(runtime_dependencies.cache_backend),
            tech_stack=list(runtime_dependencies.tech_stack),
            data_sources=list(runtime_dependencies.data_sources),
        )

    def _generation_intake_from_design_context(
        self,
        *,
        title: str,
        problem_statement: str,
        learning_outcomes: list[str],
        package_type_hint: PackageType | None,
        design_spec: AssignmentDesignSpec | None,
    ) -> GenerationIntake:
        runtime_dependencies = design_spec.runtime_dependencies if design_spec is not None else None
        runtime_plan = design_spec.project_contract.runtime_plan if design_spec is not None else None

        def service_version(service_id: str | None) -> str | None:
            if runtime_plan is None or not service_id:
                return None
            service = next(
                (candidate for candidate in runtime_plan.services if candidate.service_id == service_id),
                None,
            )
            return service.version_hint if service is not None else None

        return GenerationIntake(
            title=title,
            problem_statement=problem_statement,
            learning_outcomes=learning_outcomes,
            package_type_hint=package_type_hint,
            starter_type=(runtime_dependencies.starter_type if runtime_dependencies is not None else None),
            implementation_language=(
                runtime_dependencies.implementation_language if runtime_dependencies is not None else None
            ),
            language_version=(
                (runtime_dependencies.language_version if runtime_dependencies is not None else None)
                or (runtime_plan.language_version if runtime_plan is not None else None)
            ),
            application_framework=(
                runtime_dependencies.application_framework if runtime_dependencies is not None else None
            ),
            framework_version=(
                (runtime_dependencies.framework_version if runtime_dependencies is not None else None)
                or (runtime_plan.framework_version if runtime_plan is not None else None)
            ),
            package_manager=(
                (runtime_dependencies.package_manager if runtime_dependencies is not None else None)
                or (runtime_plan.package_manager if runtime_plan is not None else None)
            ),
            primary_database=(runtime_dependencies.primary_database if runtime_dependencies is not None else None),
            primary_database_version=(
                (runtime_dependencies.primary_database_version if runtime_dependencies is not None else None)
                or service_version(runtime_dependencies.primary_database if runtime_dependencies is not None else None)
            ),
            cache_backend=(runtime_dependencies.cache_backend if runtime_dependencies is not None else None),
            cache_backend_version=(
                (runtime_dependencies.cache_backend_version if runtime_dependencies is not None else None)
                or service_version(runtime_dependencies.cache_backend if runtime_dependencies is not None else None)
            ),
            tech_stack=(list(runtime_dependencies.tech_stack) if runtime_dependencies is not None else []),
            data_sources=(list(runtime_dependencies.data_sources) if runtime_dependencies is not None else []),
        )

    def _creator_diagnostics(
        self,
        course_run: CourseRun,
        review: CourseReviewReport,
    ) -> list[TestingDiagnostic]:
        diagnostics: list[TestingDiagnostic] = []

        if course_run.active_operation is not None:
            diagnostics.append(
                TestingDiagnostic(
                    code="course_operation_in_progress",
                    severity=TestingDiagnosticSeverity.info,
                    summary=f"Course action `{course_run.active_operation.value}` is still running.",
                    detail="The draft is being updated in the background. Wait for the current action to finish before starting another one.",
                    recommended_action="Refresh the draft status after the current action completes.",
                    blocking=False,
                    context={"operation": course_run.active_operation.value},
                )
            )

        if course_run.last_error:
            diagnostics.append(
                TestingDiagnostic(
                    code="course_action_failed",
                    severity=TestingDiagnosticSeverity.error,
                    summary="The last course action failed.",
                    detail=course_run.last_error,
                    recommended_action="Review the blocking reason, fix the linked draft issue, and retry the action.",
                    blocking=True,
                    context={"stage": course_run.stage.value, "status": course_run.status.value},
                )
            )

        if review.blockers and course_run.active_operation is None:
            diagnostics.append(
                TestingDiagnostic(
                    code="review_blocked",
                    severity=TestingDiagnosticSeverity.error,
                    summary="This draft still has blocking issues.",
                    detail=review.blockers[0],
                    recommended_action=review.next_actions[0] if review.next_actions else "Inspect the blocking deliverable or linked workflow before continuing.",
                    blocking=True,
                    context={"blocker_count": len(review.blockers)},
                )
            )

        pending_workflows = [
            workflow
            for workflow in review.linked_workflows
            if workflow.pending_gate is not None
        ]
        if pending_workflows and course_run.active_operation is None:
            pending = pending_workflows[0]
            diagnostics.append(
                TestingDiagnostic(
                    code="linked_workflow_review_pending",
                    severity=TestingDiagnosticSeverity.warning,
                    summary="A linked assignment workflow is waiting on review.",
                    detail=f"Workflow `{pending.title}` is paused at `{pending.pending_gate.value}`.",
                    recommended_action="Review the linked assignment step and approve or request changes.",
                    blocking=False,
                    context={
                        "workflow_run_id": pending.run_id,
                        "pending_gate": pending.pending_gate.value,
                    },
                )
            )

        if (
            course_run.stage == CourseRunStage.ready_to_publish
            and course_run.latest_publish_snapshot_id is None
        ):
            diagnostics.append(
                TestingDiagnostic(
                    code="ready_for_publish",
                    severity=TestingDiagnosticSeverity.info,
                    summary="The draft is ready for publish review.",
                    detail="All linked assignment workflows are published and the course can be materialized or published.",
                    recommended_action="Open the draft playground, test the learner path, then publish when it feels ready.",
                    blocking=False,
                )
            )

        latest_eval = self.store.get_latest_learner_eval_report(
            course_run.id,
            course_run.latest_publish_snapshot_id,
        )
        if latest_eval is not None and latest_eval.overall_status != "passed":
            diagnostics.append(
                TestingDiagnostic(
                    code="learner_eval_failed",
                    severity=TestingDiagnosticSeverity.error,
                    summary="The latest learner-path evaluation did not pass.",
                    detail=f"Latest learner evaluation report `{latest_eval.id}` finished with `{latest_eval.overall_status}`.",
                    recommended_action="Inspect the learner evaluation report before publishing this draft.",
                    blocking=True,
                    context={
                        "report_id": latest_eval.id,
                        "publish_snapshot_id": latest_eval.publish_snapshot_id,
                    },
                )
            )

        if not diagnostics:
            diagnostics.append(
                TestingDiagnostic(
                    code="creator_view_healthy",
                    severity=TestingDiagnosticSeverity.info,
                    summary="No blocking backend issues are currently recorded for this draft.",
                    detail="You can focus on reviewing the deliverable plan, linked workflow details, and learner experience.",
                    blocking=False,
                )
            )

        return diagnostics

    def _default_course_summary(self, pattern: CoursePattern | None, title: str) -> str:
        if pattern is None:
            return f"Course draft for '{title}' generated from the explicit assignment-design planner."
        if pattern.package_type == PackageType.survey_course:
            return f"Survey course covering multiple assignment designs under '{title}'."
        return f"Progressive codebase course for '{title}', with one inherited system evolving across deliverables."

    def _default_learning_outcomes(
        self,
        design_spec: AssignmentDesignSpec | None,
    ) -> list[str]:
        if design_spec is None:
            return ["contract correctness", "error handling"]
        overlays = design_spec.overlays
        domain_pack = design_spec.domain_pack
        if design_spec.capabilities.retrieval_mode == RetrievalMode.none and design_spec.capabilities.tool_use_required:
            outcomes = ["tool selection", "structured output", "safe escalation"]
            if "productionization_overlay" in overlays:
                outcomes.append("observability")
            if "scale_slo_overlay" in overlays:
                outcomes.append("latency and cost tuning")
            return outcomes
        if design_spec.capabilities.retrieval_mode == RetrievalMode.grounded_answers:
            outcomes = ["retrieval quality", "citation correctness", "faithfulness"]
            if "scale_slo_overlay" in overlays:
                outcomes.append("latency and cost tuning")
            if "freshness_overlay" in overlays:
                outcomes.append("index freshness")
            return outcomes
        if design_spec.capabilities.retrieval_mode == RetrievalMode.ranked_results:
            return ["ranking quality", "metadata filtering", "latency"]
        if design_spec.capabilities.durable_state_required and not design_spec.capabilities.tool_use_required:
            return ["state invariants", "concurrency safety", "idempotency"]
        return ["contract correctness", "error handling"]

    def _design_spec_from_workflow(
        self,
        workflow_run: WorkflowRun,
        fallback: AssignmentDesignSpec | None = None,
    ) -> AssignmentDesignSpec | None:
        spec = workflow_run.artifacts.task_agent_spec
        if spec is not None:
            return AssignmentDesignSpec(
                course_structure=spec.course_structure,
                runtime_dependencies=spec.runtime_dependencies,
                capabilities=spec.capabilities,
                assessment_strategy=spec.assessment_strategy,
                project_contract=spec.project_contract,
                risk_class=spec.risk_class,
                domain_pack=spec.domain_pack,
                overlays=list(spec.overlays),
            )
        return fallback

    def _workflow_design_status(self, workflow_run: WorkflowRun) -> str:
        if workflow_run.status == WorkflowStatus.blocked:
            return "unsupported"
        if workflow_run.artifacts.task_agent_spec is None:
            return "manual_review"
        if workflow_run.status == WorkflowStatus.published:
            return "supported"
        return "manual_review"

    def _slugify(self, value: str) -> str:
        return value.lower().replace(" ", "-").replace("/", "-")

    def _run_job_in_background(self, job: Callable[[], None]) -> None:
        thread = threading.Thread(target=job, daemon=True)
        thread.start()

    def _require_run(self, course_run_id: str) -> CourseRun:
        run = self.store.get_course_run(course_run_id)
        if run is None:
            raise KeyError(course_run_id)
        return run

    def _require_published_run(self, course_run_id: str) -> CourseRun:
        run = self._require_run(course_run_id)
        if run.status != CourseRunStatus.published:
            raise CourseWorkflowConflictError("Only published courses can start a new version draft.")
        return run

    def _build_revision_placeholder(self, source: CourseRun, *, queued: bool) -> CourseRun:
        now = datetime.now(UTC)
        revision = source.model_copy(deep=True)
        revision.id = f"course_{uuid4().hex[:12]}"
        revision.created_at = now
        revision.updated_at = now
        revision.stage = CourseRunStage.drafting
        revision.status = CourseRunStatus.active
        revision.shared_workflow_run_id = None
        revision.materialized_bundle = None
        revision.latest_publish_snapshot_id = None
        revision.deliverables = []
        revision.active_operation = CourseAsyncOperation.revision if queued else None
        revision.last_error = None
        revision.notes = [*source.notes]
        if queued:
            revision.notes.extend(
                [
                    f"New version draft queued from published course `{source.id}`.",
                    "We are cloning linked assignment workflows and rebuilding review state in the background.",
                ]
            )
        else:
            revision.notes.append(f"New version draft created from published course `{source.id}`.")
        revision.notes.append("Existing learners stay pinned to the previously published snapshot until this revision is published.")
        return revision

    def _prepare_revision_drafts(self, source: CourseRun) -> tuple[list[CourseDeliverableDraft], str | None]:
        workflow_revision_map: dict[str, WorkflowRun] = {}
        for linked_run_id in self._linked_runs(source):
            workflow_revision_map[linked_run_id] = self.workflow_service.create_revision_from_run(linked_run_id)

        deliverable_drafts: list[CourseDeliverableDraft] = []
        if source.shared_workflow_run_id and source.shared_workflow_run_id in workflow_revision_map:
            cloned_run = workflow_revision_map[source.shared_workflow_run_id]
            cloned_run = self._ensure_progressive_workflow_matches_deliverables(
                cloned_run,
                [self._request_from_deliverable_draft(deliverable) for deliverable in source.deliverables],
            )
            aligned_deliverables = self._align_progressive_deliverables(
                [self._request_from_deliverable_draft(deliverable) for deliverable in source.deliverables],
                cloned_run,
            )
            deliverable_drafts = [
                self._deliverable_draft_from_workflow(
                    deliverable,
                    cloned_run.id,
                    cloned_run.stage.value,
                    cloned_run.status.value,
                    cloned_run.artifacts.draft_kind.value,
                    self._design_spec_from_workflow(cloned_run, deliverable.design_spec or source.shared_design_spec),
                    self._workflow_design_status(cloned_run),
                    extra_notes=[f"Revision draft created from published course `{source.id}`."],
                )
                for deliverable in aligned_deliverables
            ]
        else:
            for deliverable in source.deliverables:
                cloned_run = workflow_revision_map.get(deliverable.workflow_run_id) if deliverable.workflow_run_id else None
                if cloned_run is None:
                    deliverable_drafts.append(deliverable.model_copy(deep=True))
                    continue
                deliverable_drafts.append(
                    self._deliverable_draft_from_workflow(
                        self._request_from_deliverable_draft(deliverable),
                        cloned_run.id,
                        cloned_run.stage.value,
                        cloned_run.status.value,
                        cloned_run.artifacts.draft_kind.value,
                        self._design_spec_from_workflow(cloned_run, deliverable.design_spec),
                        self._workflow_design_status(cloned_run),
                        extra_notes=[f"Revision draft created from published course `{source.id}`."],
                    )
                )

        shared_workflow_run_id = (
            workflow_revision_map[source.shared_workflow_run_id].id
            if source.shared_workflow_run_id and source.shared_workflow_run_id in workflow_revision_map
            else None
        )
        return deliverable_drafts, shared_workflow_run_id

    def _finalize_revision(
        self,
        *,
        revision: CourseRun,
        source: CourseRun,
        deliverable_drafts: list[CourseDeliverableDraft],
        shared_workflow_run_id: str | None,
        event_type: str,
    ) -> CourseRun:
        stage, status = self._course_stage_from_deliverables(deliverable_drafts)
        revision.updated_at = datetime.now(UTC)
        revision.stage = stage
        revision.status = status
        revision.shared_workflow_run_id = shared_workflow_run_id
        revision.materialized_bundle = None
        revision.latest_publish_snapshot_id = None
        revision.deliverables = deliverable_drafts
        revision.active_operation = None
        revision.last_error = None
        if source.generated_plan is not None:
            revision.generated_plan = self.generated_plan_from_run(revision, notes=source.generated_plan.notes)
        revision.notes = list(
            dict.fromkeys(
                [
                    *revision.notes,
                    f"New version draft created from published course `{source.id}`.",
                    "Existing learners stay pinned to the previously published snapshot until this revision is published.",
                ]
            )
        )
        self.store.save_course_run(revision)
        self.store.append_course_event(
            revision.id,
            event_type,
            {
                "source_course_run_id": source.id,
                "course_family_id": revision.course_family_id,
                "shared_workflow_run_id": revision.shared_workflow_run_id,
                "message": "New version draft is ready for review.",
            },
        )
        return revision

    def _finish_queued_revision(self, revision_id: str, source_course_run_id: str) -> None:
        try:
            source = self._require_published_run(source_course_run_id)
            revision = self._require_run(revision_id)
            deliverable_drafts, shared_workflow_run_id = self._prepare_revision_drafts(source)
            self._finalize_revision(
                revision=revision,
                source=source,
                deliverable_drafts=deliverable_drafts,
                shared_workflow_run_id=shared_workflow_run_id,
                event_type="course_revision_completed",
            )
        except Exception as exc:
            self._mark_revision_failed(revision_id, source_course_run_id, str(exc))

    def _mark_revision_failed(self, revision_id: str, source_course_run_id: str, error: str) -> None:
        revision = self._require_run(revision_id)
        revision.stage = CourseRunStage.blocked
        revision.status = CourseRunStatus.blocked
        revision.updated_at = datetime.now(UTC)
        revision.active_operation = None
        revision.last_error = error
        revision.notes = list(
            dict.fromkeys(
                [
                    *revision.notes,
                    "Creating the new version draft failed before the revision could be fully prepared.",
                ]
            )
        )
        self.store.save_course_run(revision)
        self.store.append_course_event(
            revision.id,
            "course_revision_failed",
            {
                "source_course_run_id": source_course_run_id,
                "error": error,
                "message": "Creating the new version draft failed.",
            },
        )

    def _deliverable_blockers(
        self,
        deliverable: CourseDeliverableDraft,
        linked_run: WorkflowRun | None,
        *,
        generation_in_progress: bool = False,
    ) -> list[str]:
        blockers: list[str] = []
        if not deliverable.workflow_run_id:
            blockers.append("No linked assignment workflow run.")
            return blockers
        if linked_run is None:
            blockers.append("Linked assignment workflow run is missing.")
            return blockers
        if generation_in_progress:
            blockers.append("Linked assignment workflow is still running automated generation and review.")
            return blockers
        if linked_run.status == WorkflowStatus.blocked:
            blockers.append("Linked assignment workflow is blocked.")
        if linked_run.artifacts.task_agent_spec is None:
            blockers.append("Linked assignment workflow does not have a learner-ready spec yet.")
        if linked_run.artifacts.review_summary is not None:
            blockers.extend(linked_run.artifacts.review_summary.blockers)
        return blockers

    def _linked_workflow_summary(self, linked_run: WorkflowRun | None) -> CourseLinkedWorkflowSummary | None:
        if linked_run is None:
            return None
        bundle = None
        if linked_run.artifacts.materialized_bundle is not None:
            materialized_bundle = linked_run.artifacts.materialized_bundle
            public_files = [
                bundle_file.relative_path
                for bundle_file in materialized_bundle.files
                if bundle_file.visibility == ArtifactVisibility.public
            ]
            private_count = sum(
                1
                for bundle_file in materialized_bundle.files
                if bundle_file.visibility == ArtifactVisibility.private
            )
            bundle = CourseLinkedBundleSummary(
                bundle_id=materialized_bundle.bundle_id,
                root_dir=materialized_bundle.root_dir,
                public_dir=materialized_bundle.public_dir,
                manifest_path=materialized_bundle.manifest_path,
                total_file_count=len(materialized_bundle.files),
                public_files=public_files,
                private_file_count=private_count,
            )
        return CourseLinkedWorkflowSummary(
            run_id=linked_run.id,
            title=linked_run.title,
            stage=linked_run.stage,
            status=linked_run.status,
            pending_gate=linked_run.pending_gate,
            draft_kind=linked_run.artifacts.draft_kind,
            bundle=bundle,
            review_summary=linked_run.artifacts.review_summary,
        )

    def _next_actions(
        self,
        course_run: CourseRun,
        linked_workflows: list[CourseLinkedWorkflowSummary],
    ) -> list[str]:
        if course_run.active_operation == CourseAsyncOperation.generation:
            return [
                "Wait for background generation to finish. Live workflow node progress now appears in the draft timeline and the course generation log.",
            ]
        actions: list[str] = []
        if course_run.stage == CourseRunStage.drafting:
            actions.append("We are still building this draft. Keep this page open or reload it later to see fresh progress.")
            return actions
        if course_run.last_error:
            actions.append("This draft is blocked. Review the activity log or the draft details, then update the brief and try again.")
        pending_gates = [workflow.pending_gate for workflow in linked_workflows if workflow.pending_gate is not None]
        if pending_gates:
            unique_gates = []
            for gate in pending_gates:
                assert gate is not None
                if gate.value not in unique_gates:
                    unique_gates.append(gate.value)
            actions.append(
                "Advance the linked assignment workflow HIL gates: "
                + ", ".join(f"`{gate}`" for gate in unique_gates)
                + "."
            )
        if any(workflow.status == WorkflowStatus.blocked for workflow in linked_workflows):
            actions.append("Repair or replace blocked assignment workflow runs before publishing the course.")
        if any(workflow.draft_kind != DraftKind.task_agent_spec for workflow in linked_workflows):
            actions.append("Every linked workflow must generate a learner-ready assignment spec before the course can publish.")
        if any(workflow.bundle is None for workflow in linked_workflows):
            actions.append("Materialize linked assignment workflow bundles so authors can review generated artifacts.")
        if course_run.materialized_bundle is None:
            actions.append("Materialize the course bundle to review syllabus, deliverable sequencing, and aggregate status.")
        if course_run.stage == CourseRunStage.ready_to_publish and course_run.status != CourseRunStatus.published:
            actions.append("Publish the course run once the authoring review is complete.")
        if course_run.shared_workflow_run_id is not None and any(
            workflow.run_id == course_run.shared_workflow_run_id and workflow.status != WorkflowStatus.published
            for workflow in linked_workflows
        ):
            actions.append("The shared progressive assignment workflow gates the whole course; publishing it will unblock every deliverable.")
        return actions

    def _linked_runs(self, course_run: CourseRun) -> dict[str, WorkflowRun]:
        linked_runs: dict[str, WorkflowRun] = {}
        for deliverable in course_run.deliverables:
            if not deliverable.workflow_run_id:
                continue
            child_run = self.workflow_service.get_run(deliverable.workflow_run_id)
            if child_run is not None:
                linked_runs[deliverable.workflow_run_id] = child_run
        return linked_runs

    def _linked_runs_with_shared(self, course_run: CourseRun) -> dict[str, WorkflowRun]:
        linked_runs = self._linked_runs(course_run)
        if course_run.shared_workflow_run_id and course_run.shared_workflow_run_id not in linked_runs:
            shared_run = self.workflow_service.get_run(course_run.shared_workflow_run_id)
            if shared_run is not None:
                linked_runs[course_run.shared_workflow_run_id] = shared_run
        return linked_runs

    def _timeline_item_from_course_event(
        self,
        course_run: CourseRun,
        event,
    ) -> DraftTimelineItem:
        return DraftTimelineItem(
            id=f"course:{course_run.id}:{event.sequence_no}",
            created_at=event.created_at,
            source_kind=DraftTimelineSourceKind.course_event,
            source_id=course_run.id,
            source_title=course_run.title,
            title=self._timeline_title_for_event(event.event_type),
            detail=self._timeline_detail_for_event(event.event_type, event.payload),
            event_type=event.event_type,
            stage=course_run.stage.value,
            status=course_run.status.value,
            sequence_no=event.sequence_no,
            payload=event.payload,
        )

    def _timeline_item_from_workflow_event(
        self,
        workflow_run: WorkflowRun,
        event: WorkflowEvent,
    ) -> DraftTimelineItem:
        source_kind = (
            DraftTimelineSourceKind.workflow_authoring
            if event.event_type in {"workflow_authoring_completed", "workflow_authoring_revised"}
            else DraftTimelineSourceKind.workflow_event
        )
        return DraftTimelineItem(
            id=f"workflow-event:{workflow_run.id}:{event.sequence_no}",
            created_at=event.created_at,
            source_kind=source_kind,
            source_id=workflow_run.id,
            source_title=workflow_run.title,
            title=self._timeline_title_for_event(event.event_type),
            detail=self._timeline_detail_for_event(event.event_type, event.payload),
            event_type=event.event_type,
            stage=workflow_run.stage.value,
            status=workflow_run.status.value,
            sequence_no=event.sequence_no,
            payload=event.payload,
        )

    def _timeline_item_from_workflow_node(
        self,
        workflow_run: WorkflowRun,
        node: WorkflowNodeExecution,
    ) -> DraftTimelineItem:
        findings = [
            finding.title.strip()
            for finding in node.findings
            if finding.title.strip()
        ]
        detail = node.summary.strip()
        if findings:
            detail = f"{detail} Key findings: {', '.join(findings[:3])}."
        return DraftTimelineItem(
            id=f"workflow-node:{workflow_run.id}:{node.node_id}",
            created_at=node.created_at,
            source_kind=DraftTimelineSourceKind.workflow_node,
            source_id=workflow_run.id,
            source_title=workflow_run.title,
            title=f"{self._timeline_title_for_node_kind(node.kind)} · {node.status.value.replace('_', ' ')}",
            detail=detail,
            event_type=node.kind.value,
            stage=workflow_run.stage.value,
            status=node.status.value,
            iteration=node.iteration,
            attempt=node.attempt,
            payload={
                "node_id": node.node_id,
                "kind": node.kind.value,
                "iteration": node.iteration,
                "status": node.status.value,
                "summary": node.summary,
                "findings": [finding.model_dump(mode="json") for finding in node.findings],
            },
        )

    def _timeline_title_for_event(self, event_type: str) -> str:
        overrides = {
            "course_run_created": "Draft created",
            "course_generation_queued": "Generation queued",
            "course_brief_generated": "Course brief generated",
            "course_generation_failed": "Generation failed",
            "course_publish_requested": "Publish requested",
            "course_published": "Course published",
            "course_publish_failed": "Publish failed",
            "course_materialize_requested": "Review package requested",
            "course_materialized": "Review package built",
            "course_materialize_failed": "Review package failed",
            "course_revision_queued": "Revision queued",
            "course_revision_started": "Revision started",
            "course_revision_created": "Revision created",
            "run_created": "Workflow run created",
            "workflow_authoring_completed": "Workflow authoring completed",
            "workflow_authoring_revised": "Workflow authoring revised",
            "workflow_node_started": "Workflow node started",
            "workflow_node_completed": "Workflow node completed",
            "run_revision_created": "Workflow revision created",
            "task_agent_spec_updated": "Assignment spec updated",
            "task_agent_workspace_materialized": "Workspace materialized",
            "bundle_materialized": "Bundle materialized",
            "hil_gate_decision": "Human review decision",
            "langgraph_nodes_executed": "Reviewer loop finished",
            "langgraph_nodes_failed": "Reviewer loop failed",
        }
        if event_type in overrides:
            return overrides[event_type]
        return event_type.replace("_", " ").strip().capitalize()

    def _timeline_title_for_node_kind(self, kind: WorkflowNodeKind) -> str:
        labels = {
            WorkflowNodeKind.authoring_runtime: "Authoring runtime",
            WorkflowNodeKind.authoring_repair: "Authoring repair",
            WorkflowNodeKind.reviewer_runtime: "Runtime review",
            WorkflowNodeKind.reviewer_repair: "Reviewer repair",
            WorkflowNodeKind.reviewer_code: "Code review",
            WorkflowNodeKind.reviewer_pedagogy: "Pedagogy review",
            WorkflowNodeKind.reviewer_tests: "Tests review",
            WorkflowNodeKind.reviewer_learner_runtime: "Learner runtime review",
        }
        return labels.get(kind, kind.value.replace("_", " ").title())

    def _timeline_detail_for_event(self, event_type: str, payload: dict) -> str | None:
        if not payload:
            return None
        if isinstance(payload.get("message"), str) and payload["message"].strip():
            return payload["message"].strip()
        if isinstance(payload.get("error"), str) and payload["error"].strip():
            return payload["error"].strip()
        if event_type == "course_run_created":
            deliverable_count = payload.get("deliverable_count")
            return (
                f"Created a draft with {deliverable_count} deliverable"
                f"{'' if deliverable_count == 1 else 's'}."
                if isinstance(deliverable_count, int)
                else "Created a new course draft."
            )
        if event_type == "course_brief_generated":
            deliverable_count = payload.get("deliverable_count")
            source = payload.get("source")
            parts: list[str] = []
            if isinstance(deliverable_count, int):
                parts.append(f"Planned {deliverable_count} deliverables")
            if source:
                parts.append(f"via {source}")
            return ". ".join(parts) + ("." if parts else "")
        if event_type == "run_created":
            draft_kind = payload.get("draft_kind")
            design_status = payload.get("design_status")
            parts = [part for part in [draft_kind, design_status] if part]
            return f"Initialized workflow run ({', '.join(parts)})." if parts else "Initialized workflow run."
        if event_type in {"workflow_node_started", "workflow_node_completed"}:
            node_kind = payload.get("node_kind")
            attempt = payload.get("attempt")
            status = payload.get("status")
            parts = []
            if node_kind:
                parts.append(node_kind)
            if attempt is not None:
                parts.append(f"attempt {attempt}")
            if status:
                parts.append(status)
            summary = payload.get("summary")
            head = " · ".join(parts) if parts else None
            if head and summary:
                return f"{head} · {summary}"
            return head or summary
        if event_type == "hil_gate_decision":
            gate = payload.get("gate")
            decision = payload.get("decision")
            if gate and decision:
                return f"{decision.capitalize()} on {gate}."
        summary_keys = ["stage", "status", "pending_gate", "workflow_run_id", "source_run_id"]
        fragments = [f"{key}: {payload[key]}" for key in summary_keys if payload.get(key) is not None]
        if fragments:
            return " · ".join(fragments)
        return None

    def _published_version_changes(
        self,
        snapshot: PublishSnapshot,
        previous: PublishSnapshot | None,
    ) -> list[str]:
        learner_package = snapshot.learner_package
        if previous is None:
            return ["Initial published version for learners."]
        previous_package = previous.learner_package
        changes: list[str] = []
        if learner_package is not None and previous_package is not None:
            if learner_package.title != previous_package.title:
                changes.append("Course title changed for learners.")
            if learner_package.summary != previous_package.summary:
                changes.append("Course summary changed.")
            if len(learner_package.deliverables) != len(previous_package.deliverables):
                changes.append(
                    f"Deliverable count changed from {len(previous_package.deliverables)} to {len(learner_package.deliverables)}."
                )
            changed_deliverables = 0
            previous_by_id = {
                deliverable.deliverable_id: deliverable
                for deliverable in previous_package.deliverables
            }
            for deliverable in learner_package.deliverables:
                old = previous_by_id.get(deliverable.deliverable_id)
                if old is None:
                    changed_deliverables += 1
                    continue
                if (
                    deliverable.title != old.title
                    or deliverable.objective != old.objective
                    or deliverable.content_markdown != old.content_markdown
                    or deliverable.starter_readme != old.starter_readme
                    or deliverable.learning_outcomes != old.learning_outcomes
                    or deliverable.active_test_ids != old.active_test_ids
                    or [file.relative_path for file in deliverable.workspace_seed_files]
                    != [file.relative_path for file in old.workspace_seed_files]
                ):
                    changed_deliverables += 1
            if changed_deliverables:
                changes.append(f"Learner package changed in {changed_deliverables} deliverable(s).")
        if snapshot.task_agent_spec is not None and previous.task_agent_spec is not None:
            if snapshot.task_agent_spec.model_dump(mode="json") != previous.task_agent_spec.model_dump(mode="json"):
                changes.append("Hidden grading contract changed.")
        if not changes:
            changes.append("No learner-visible changes from the previous published version.")
        return changes
