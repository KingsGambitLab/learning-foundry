from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.course import (
    CourseAsyncOperation,
    CourseLinkedBundleSummary,
    CourseLinkedWorkflowSummary,
    CourseGenerationSource,
    CourseGenerationStatus,
    LocalDraftResetResult,
    CourseModuleDraft,
    CourseModuleReview,
    CourseReviewCounts,
    CourseReviewReport,
    CourseRun,
    CourseRunList,
    CourseRunSummary,
    CourseRunStage,
    CourseRunStatus,
    CreateCourseModuleRequest,
    CreateCourseRunRequest,
    GeneratedCoursePlan,
    QueueCourseOperationResponse,
    QueueCourseRevisionResponse,
)
from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import AssignmentDesignSpec, RetrievalMode
from app.domain.publish import PublishSnapshot, PublishedVersionList, PublishedVersionSummary
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
    DraftKind,
    MaterializeBundleRequest,
    WorkflowRun,
    WorkflowStatus,
)
from app.services.course_artifact_materializer import CourseArtifactMaterializer
from app.services.course_patterns import CoursePattern, course_pattern_by_slug
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design, infer_risk_class
from app.services.lms_service import default_learner_workspace_dir
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
        job_runner: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.store = store
        self.workflow_service = workflow_service
        self.materializer = materializer or CourseArtifactMaterializer()
        self.publish_snapshot_service = publish_snapshot_service or PublishSnapshotService(store, workflow_service)
        self.job_runner = job_runner or self._run_job_in_background

    def list_runs(self, limit: int = 50) -> CourseRunList:
        return CourseRunList(runs=self.store.list_course_runs(limit=limit))

    def get_run(self, course_run_id: str) -> CourseRun | None:
        return self.store.get_course_run(course_run_id)

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
            modules=[self._request_from_module_draft(module) for module in course_run.modules],
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
                "module_count": len(course_run.modules),
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
        generation_status: CourseGenerationStatus | None = None,
    ) -> CourseRun:
        intake = GenerationIntake(
            title=title,
            problem_statement=goal,
            learning_outcomes=learning_outcomes,
            package_type_hint=package_type_hint,
        )
        inferred = infer_assignment_design(
            title=title,
            problem_statement=goal,
            learning_outcomes=learning_outcomes,
            package_type_hint=package_type_hint,
        )
        now = datetime.now(UTC)
        course_run_id = f"course_{uuid4().hex[:12]}"
        course_run = CourseRun(
            id=course_run_id,
            course_family_id=course_run_id,
            title=title,
            summary=goal.strip(),
            package_type=package_type_hint or inferred.package_type,
            shared_design_spec=inferred.design_spec,
            shared_workflow_run_id=None,
            created_at=now,
            updated_at=now,
            stage=CourseRunStage.drafting,
            status=CourseRunStatus.active,
            modules=[],
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
    ) -> CourseRun:
        existing = self._require_run(course_run_id)
        course_run = self._build_course_run(
            course_run_id=existing.id,
            created_at=existing.created_at,
            updated_at=datetime.now(UTC),
            request=CreateCourseRunRequest(
                title=plan.title,
                summary=plan.summary,
                package_type=plan.package_type,
                shared_design_spec=plan.shared_design_spec,
                course_family_id=existing.course_family_id,
                modules=plan.modules,
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
        course_run.active_operation = None
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
                "module_count": len(course_run.modules),
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
        request: CreateCourseRunRequest,
    ) -> CourseRun:
        pattern = course_pattern_by_slug(request.pattern_slug) if request.pattern_slug else None
        if request.pattern_slug and pattern is None:
            raise ValueError(f"Unknown course pattern '{request.pattern_slug}'.")

        if pattern is None and not request.modules:
            raise ValueError("Custom course creation requires at least one module.")

        title = request.title or (pattern.course_title if pattern else None)
        if not title:
            raise ValueError("Course title is required.")
        summary = request.summary or self._default_course_summary(pattern, title)
        package_type = request.package_type or (pattern.package_type if pattern else PackageType.survey_course)
        shared_design_spec = request.shared_design_spec or (pattern.shared_design_spec if pattern else None)

        modules = self._module_requests_from_pattern(pattern, summary) if pattern else request.modules
        if not modules:
            raise ValueError("Course must contain at least one module.")

        if package_type == PackageType.survey_course:
            module_drafts = [self._create_survey_module(module, title, summary) for module in modules]
            shared_workflow_run_id = None
        else:
            shared_run = self._create_progressive_workflow(title, summary, modules, shared_design_spec)
            aligned_modules = self._align_progressive_modules(modules, shared_run)
            module_drafts = [
                self._module_draft_from_workflow(
                    module,
                    shared_run.id,
                    shared_run.stage.value,
                    shared_run.status.value,
                    shared_run.artifacts.draft_kind.value,
                    self._design_spec_from_workflow(shared_run, module.design_spec or shared_design_spec),
                    self._workflow_design_status(shared_run),
                    extra_notes=["Shared progressive workflow run for the whole course."],
                )
                for module in aligned_modules
            ]
            shared_workflow_run_id = shared_run.id
            shared_design_spec = self._design_spec_from_workflow(shared_run, shared_design_spec)

        stage, status = self._course_stage_from_modules(module_drafts)
        course_run = CourseRun(
            id=course_run_id,
            title=title,
            summary=summary,
            package_type=package_type,
            pattern_slug=pattern.course_slug if pattern else request.pattern_slug,
            course_family_id=request.course_family_id or course_run_id,
            shared_design_spec=shared_design_spec,
            shared_workflow_run_id=shared_workflow_run_id,
            created_at=created_at,
            updated_at=updated_at,
            stage=stage,
            status=status,
            modules=module_drafts,
            notes=[
                "Course draft created from the course workflow layer.",
                "Each survey module maps to its own assignment workflow run; progressive courses share one assignment workflow run.",
            ],
        )
        return course_run

    def create_revision(self, course_run_id: str) -> CourseRun:
        source = self._require_published_run(course_run_id)
        revision = self._build_revision_placeholder(source, queued=False)
        module_drafts, shared_workflow_run_id = self._prepare_revision_drafts(source)
        return self._finalize_revision(
            revision=revision,
            source=source,
            module_drafts=module_drafts,
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

    def record_creator_feedback(
        self,
        course_run_id: str,
        request: CreateCreatorFeedbackRequest,
    ) -> CreatorFeedbackRecord:
        course_run = self._compute_refreshed_run(self._require_run(course_run_id))
        if request.module_slug is not None and all(module.module_slug != request.module_slug for module in course_run.modules):
            raise ValueError(f"Unknown module '{request.module_slug}' for course '{course_run_id}'.")
        valid_workflow_ids = {
            workflow_id
            for workflow_id in [course_run.shared_workflow_run_id, *[module.workflow_run_id for module in course_run.modules]]
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
            module_slug=request.module_slug,
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
                "module_slug": feedback.module_slug,
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

        learner_module_ids = {module.module_id for module in snapshot.learner_package.modules}
        unknown_modules = [result.module_id for result in request.module_results if result.module_id not in learner_module_ids]
        if unknown_modules:
            raise CourseWorkflowConflictError(
                f"Learner evaluation report includes unknown module ids for snapshot '{publish_snapshot_id}': {', '.join(unknown_modules)}."
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
                for result in request.module_results
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
            module_results=request.module_results,
        )
        self.store.save_learner_eval_report(report)
        self.store.append_course_event(
            course_run_id,
            "learner_evaluation_recorded",
            {
                "report_id": report.id,
                "publish_snapshot_id": publish_snapshot_id,
                "overall_status": report.overall_status,
                "module_count": len(report.module_results),
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
                    module_count=len(learner_package.modules) if learner_package is not None else 0,
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
        if course_run.stage == CourseRunStage.drafting and not course_run.modules:
            return course_run
        refreshed_run = course_run.model_copy(deep=True)
        module_drafts: list[CourseModuleDraft] = []
        shared_run = None
        if refreshed_run.shared_workflow_run_id is not None:
            shared_run = self.workflow_service.get_run(refreshed_run.shared_workflow_run_id)

        if shared_run is not None:
            aligned_modules = self._align_progressive_modules(
                [self._request_from_module_draft(module) for module in refreshed_run.modules],
                shared_run,
            )
            module_drafts = [
                self._module_draft_from_workflow(
                    module,
                    shared_run.id,
                    shared_run.stage.value,
                    shared_run.status.value,
                    shared_run.artifacts.draft_kind.value,
                    self._design_spec_from_workflow(shared_run, module.design_spec or refreshed_run.shared_design_spec),
                    self._workflow_design_status(shared_run),
                    extra_notes=["Shared progressive workflow run for the whole course."],
                )
                for module in aligned_modules
            ]
            refreshed_run.shared_design_spec = self._design_spec_from_workflow(shared_run, refreshed_run.shared_design_spec)
        else:
            for module in refreshed_run.modules:
                if not module.workflow_run_id:
                    module_drafts.append(module)
                    continue
                child_run = self.workflow_service.get_run(module.workflow_run_id)
                if child_run is None:
                    refreshed = module.model_copy(deep=True)
                    refreshed.workflow_stage = "missing"
                    refreshed.workflow_status = "blocked"
                    refreshed.notes.append("Linked assignment workflow run is missing.")
                    module_drafts.append(refreshed)
                    continue
                module_drafts.append(
                    self._module_draft_from_workflow(
                        self._request_from_module_draft(module),
                        child_run.id,
                        child_run.stage.value,
                        child_run.status.value,
                        child_run.artifacts.draft_kind.value,
                        self._design_spec_from_workflow(child_run, module.design_spec),
                        self._workflow_design_status(child_run),
                        extra_notes=[note for note in module.notes if "Shared progressive workflow run" in note],
                    )
                )

        stage, status = self._course_stage_from_modules(module_drafts)
        refreshed_run.modules = module_drafts
        refreshed_run.stage = stage if refreshed_run.status != CourseRunStatus.published else CourseRunStage.published
        refreshed_run.status = status if refreshed_run.status != CourseRunStatus.published else CourseRunStatus.published
        refreshed_run.updated_at = datetime.now(UTC)
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
        snapshot = self.publish_snapshot_service.create_snapshot(course_run, linked_runs)
        if snapshot is None:
            raise CourseWorkflowConflictError(
                "This course cannot publish yet because a linked assignment workflow does not have a learner-ready spec."
            )
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
                "module_count": len(course_run.modules),
                "publish_snapshot_id": course_run.latest_publish_snapshot_id,
            },
        )
        return course_run

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
        module_reports: list[CourseModuleReview] = []
        linked_workflows: list[CourseLinkedWorkflowSummary] = []
        linked_workflow_ids: set[str] = set()
        ready_modules = 0
        blocked_modules = 0
        modules_with_bundle = 0
        blockers: list[str] = [course_run.last_error] if course_run.last_error else []

        for position, module in enumerate(course_run.modules, start=1):
            linked_run = linked_runs.get(module.workflow_run_id) if module.workflow_run_id else None
            linked_summary = self._linked_workflow_summary(linked_run)
            module_blockers = self._module_blockers(module, linked_run)
            if course_run.shared_workflow_run_id is not None and not module.checkpoint_module_ids:
                module_blockers.append("Progressive module is not aligned to any shared assignment checkpoints yet.")
            bundle_available = bool(
                linked_run and linked_run.artifacts.materialized_bundle is not None
            )
            ready_for_publish = (
                linked_run is not None
                and linked_run.status == WorkflowStatus.published
                and linked_run.artifacts.task_agent_spec is not None
            )

            if bundle_available:
                modules_with_bundle += 1
            if ready_for_publish:
                ready_modules += 1
            if module_blockers:
                blocked_modules += 1
                blockers.extend(
                    f"Module {position} ({module.title}): {reason}"
                    for reason in module_blockers
                )
            if linked_summary is not None and linked_summary.run_id not in linked_workflow_ids:
                linked_workflows.append(linked_summary)
                linked_workflow_ids.add(linked_summary.run_id)

            module_reports.append(
                CourseModuleReview(
                    position=position,
                    module_slug=module.module_slug,
                    title=module.title,
                    summary=module.summary,
                    checkpoint_module_ids=list(module.checkpoint_module_ids),
                    design_spec=module.design_spec,
                    domain_pack=module.domain_pack,
                    overlays=module.overlays,
                    learning_outcomes=module.learning_outcomes,
                    workflow_run_id=module.workflow_run_id,
                    workflow_stage=module.workflow_stage,
                    workflow_status=module.workflow_status,
                    recommendation_status=module.recommendation_status,
                    ready_for_publish=ready_for_publish,
                    bundle_available=bundle_available,
                    blockers=module_blockers,
                    linked_workflow=linked_summary,
                    notes=module.notes,
                )
            )

        counts = CourseReviewCounts(
            total_modules=len(course_run.modules),
            ready_modules=ready_modules,
            modules_with_blockers=blocked_modules,
            modules_with_bundle=modules_with_bundle,
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
            modules=module_reports,
        )

    def _create_survey_module(self, module: CreateCourseModuleRequest, course_title: str, course_summary: str) -> CourseModuleDraft:
        intake = GenerationIntake(
            title=module.title,
            problem_statement=module.summary or f"Build the '{module.title}' assignment for the '{course_title}' course. {course_summary}",
            learning_outcomes=module.learning_outcomes,
            package_type_hint=ASSIGNMENT_PACKAGE_TYPE,
        )
        if module.design_spec is not None:
            child_run = self.workflow_service.create_run_from_explicit_plan(
                intake=intake,
                design_spec=module.design_spec,
                reasons=[f"Seeded from course module '{module.title}'."],
                notes=["Created from the survey-course module planner."],
            )
        else:
            child_run = self.workflow_service.create_run(intake)

        return self._module_draft_from_workflow(
            module,
            child_run.id,
            child_run.stage.value,
            child_run.status.value,
            child_run.artifacts.draft_kind.value,
            self._design_spec_from_workflow(child_run, module.design_spec),
            self._workflow_design_status(child_run),
        )

    def _create_progressive_workflow(
        self,
        title: str,
        summary: str,
        modules: list[CreateCourseModuleRequest],
        shared_design_spec: AssignmentDesignSpec | None,
    ):
        course_intake = GenerationIntake(
            title=title,
            problem_statement=summary,
            learning_outcomes=self._combined_learning_outcomes(modules),
            package_type_hint=ASSIGNMENT_PACKAGE_TYPE,
        )
        if shared_design_spec is not None:
            return self.workflow_service.create_run_from_explicit_plan(
                intake=course_intake,
                design_spec=shared_design_spec,
                reasons=["Created from the progressive course planner."],
                notes=["This workflow run is shared across all course modules."],
            )

        return self.workflow_service.create_run(course_intake)

    def _module_draft_from_workflow(
        self,
        module: CreateCourseModuleRequest,
        workflow_run_id: str,
        workflow_stage: str,
        workflow_status: str,
        draft_kind: str,
        design_spec: AssignmentDesignSpec | None,
        recommendation_status: str | None,
        extra_notes: list[str] | None = None,
    ) -> CourseModuleDraft:
        return CourseModuleDraft(
            module_slug=module.module_slug or self._slugify(module.title),
            title=module.title,
            summary=module.summary or module.title,
            learning_outcomes=module.learning_outcomes,
            checkpoint_module_ids=list(module.checkpoint_module_ids),
            design_spec=design_spec,
            domain_pack=(design_spec.domain_pack if design_spec is not None else module.domain_pack_hint),
            overlays=list(design_spec.overlays if design_spec is not None else module.overlays_hint),
            workflow_run_id=workflow_run_id,
            workflow_stage=workflow_stage,
            workflow_status=workflow_status,
            draft_kind=draft_kind,
            recommendation_status=recommendation_status,
            notes=(extra_notes or []),
        )

    def _module_requests_from_pattern(self, pattern: CoursePattern | None, course_summary: str) -> list[CreateCourseModuleRequest]:
        if pattern is None:
            return []
        modules: list[CreateCourseModuleRequest] = []
        for module in pattern.modules:
            modules.append(
                CreateCourseModuleRequest(
                    module_slug=module.module_slug,
                    title=module.title,
                    summary=f"Build the '{module.title}' module for the '{pattern.course_title}' course. {course_summary}",
                    learning_outcomes=self._default_learning_outcomes(module.design_spec),
                    design_spec=module.design_spec,
                    domain_pack_hint=module.domain_pack,
                    overlays_hint=module.overlays,
                )
            )
        return modules

    def _combined_learning_outcomes(self, modules: list[CreateCourseModuleRequest]) -> list[str]:
        seen: list[str] = []
        for module in modules:
            for outcome in module.learning_outcomes or self._default_learning_outcomes(module.design_spec):
                if outcome not in seen:
                    seen.append(outcome)
        return seen[:8]

    def _request_from_module_draft(self, module: CourseModuleDraft) -> CreateCourseModuleRequest:
        return CreateCourseModuleRequest(
            module_slug=module.module_slug,
            title=module.title,
            summary=module.summary,
            learning_outcomes=module.learning_outcomes,
            checkpoint_module_ids=list(module.checkpoint_module_ids),
            design_spec=module.design_spec,
            domain_pack_hint=module.domain_pack,
            overlays_hint=module.overlays,
        )

    def _align_progressive_modules(
        self,
        modules: list[CreateCourseModuleRequest],
        shared_run: WorkflowRun,
    ) -> list[CreateCourseModuleRequest]:
        spec = shared_run.artifacts.task_agent_spec
        if spec is None or not spec.modules or not modules:
            return modules

        checkpoint_ids = [module.id for module in spec.modules]
        checkpoint_modules_by_id = {module.id: module for module in spec.modules}
        checkpoint_groups = self._explicit_progressive_checkpoint_groups(modules, checkpoint_ids)
        if checkpoint_groups is None:
            checkpoint_groups = self._balanced_checkpoint_groups(len(modules), checkpoint_ids)

        aligned_modules: list[CreateCourseModuleRequest] = []
        for index, module in enumerate(modules):
            group_ids = checkpoint_groups[index] if index < len(checkpoint_groups) else []
            group_modules = [
                checkpoint_modules_by_id[checkpoint_id]
                for checkpoint_id in group_ids
                if checkpoint_id in checkpoint_modules_by_id
            ]
            title = module.title.strip()
            summary = (module.summary or module.title).strip()
            if not module.checkpoint_module_ids and group_modules:
                title = self._progressive_module_title(group_modules, fallback=title)
                summary = self._progressive_module_summary(group_modules, fallback=summary)
            aligned_modules.append(
                module.model_copy(
                    update={
                        "title": title,
                        "summary": summary,
                        "checkpoint_module_ids": group_ids,
                    }
                )
            )
        return aligned_modules

    def _explicit_progressive_checkpoint_groups(
        self,
        modules: list[CreateCourseModuleRequest],
        checkpoint_ids: list[str],
    ) -> list[list[str]] | None:
        groups: list[list[str]] = []
        assigned: list[str] = []
        for module in modules:
            if not module.checkpoint_module_ids:
                return None
            selected = [checkpoint_id for checkpoint_id in checkpoint_ids if checkpoint_id in module.checkpoint_module_ids]
            if not selected:
                return None
            groups.append(selected)
            assigned.extend(selected)
        if len(assigned) != len(set(assigned)):
            return None
        if assigned != checkpoint_ids:
            return None
        return groups

    def _balanced_checkpoint_groups(
        self,
        module_count: int,
        checkpoint_ids: list[str],
    ) -> list[list[str]]:
        if module_count <= 0:
            return []
        if not checkpoint_ids:
            return [[] for _ in range(module_count)]
        groups: list[list[str]] = []
        total_checkpoints = len(checkpoint_ids)
        for index in range(module_count):
            start = (index * total_checkpoints) // module_count
            end = ((index + 1) * total_checkpoints) // module_count
            groups.append(checkpoint_ids[start:end])
        return groups

    def _progressive_module_title(self, checkpoint_modules: list, *, fallback: str) -> str:
        if not checkpoint_modules:
            return fallback
        titles = [module.title.strip().rstrip(".") for module in checkpoint_modules]
        if len(titles) == 1:
            return titles[0]
        if len(titles) == 2:
            return f"{titles[0]} + {titles[1]}"
        return f"{titles[0]} + {titles[-1]}"

    def _progressive_module_summary(self, checkpoint_modules: list, *, fallback: str) -> str:
        if not checkpoint_modules:
            return fallback
        objectives = [module.objective.strip().rstrip(".") for module in checkpoint_modules]
        if len(objectives) == 1:
            return f"{objectives[0]}."
        if len(objectives) == 2:
            return f"{objectives[0]}. Then {self._sentence_tail(objectives[1])}."
        return (
            f"{objectives[0]}. Then extend the module through "
            f"{self._sentence_tail(objectives[1])} and {self._sentence_tail(objectives[-1])}."
        )

    def _sentence_tail(self, text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:] if text[0].isupper() else text

    def _course_stage_from_modules(self, modules: list[CourseModuleDraft]) -> tuple[CourseRunStage, CourseRunStatus]:
        statuses = {module.workflow_status for module in modules}
        if not modules or None in statuses:
            return CourseRunStage.blocked, CourseRunStatus.blocked
        if "blocked" in statuses:
            return CourseRunStage.blocked, CourseRunStatus.blocked
        if statuses == {"published"}:
            if any(module.draft_kind != DraftKind.task_agent_spec.value for module in modules):
                return CourseRunStage.awaiting_course_review, CourseRunStatus.awaiting_human
            return CourseRunStage.ready_to_publish, CourseRunStatus.awaiting_human
        return CourseRunStage.awaiting_course_review, CourseRunStatus.awaiting_human

    def _publish_readiness_error(self, course_run: CourseRun) -> str:
        if any(module.draft_kind != DraftKind.task_agent_spec.value for module in course_run.modules):
            return "This course cannot publish yet because a linked assignment workflow does not have a learner-ready spec."
        return "All linked assignment workflow runs must be published before the course can be published."

    def _creator_choices(self, course_run: CourseRun):
        runtime_dependencies = (
            course_run.shared_design_spec.runtime_dependencies
            if course_run.shared_design_spec is not None
            else None
        )
        if runtime_dependencies is None:
            return None
        from app.domain.course import CreatorCourseSetupChoices

        return CreatorCourseSetupChoices(
            starter_type=runtime_dependencies.starter_type,
            primary_database=runtime_dependencies.primary_database,
            cache_backend=runtime_dependencies.cache_backend,
            tech_stack=list(runtime_dependencies.tech_stack),
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

        if review.blockers:
            diagnostics.append(
                TestingDiagnostic(
                    code="review_blocked",
                    severity=TestingDiagnosticSeverity.error,
                    summary="This draft still has blocking issues.",
                    detail=review.blockers[0],
                    recommended_action=review.next_actions[0] if review.next_actions else "Inspect the blocking module or linked workflow before continuing.",
                    blocking=True,
                    context={"blocker_count": len(review.blockers)},
                )
            )

        pending_workflows = [
            workflow
            for workflow in review.linked_workflows
            if workflow.pending_gate is not None
        ]
        if pending_workflows:
            pending = pending_workflows[0]
            diagnostics.append(
                TestingDiagnostic(
                    code="linked_workflow_review_pending",
                    severity=TestingDiagnosticSeverity.warning,
                    summary="A linked assignment workflow is waiting on review.",
                    detail=f"Workflow `{pending.title}` is paused at `{pending.pending_gate.value}`.",
                    recommended_action="Review the linked assignment step and approve or request changes.",
                    blocking=True,
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
                    detail="You can focus on reviewing the module plan, linked workflow details, and learner experience.",
                    blocking=False,
                )
            )

        return diagnostics

    def _default_course_summary(self, pattern: CoursePattern | None, title: str) -> str:
        if pattern is None:
            return f"Course draft for '{title}' generated from the explicit assignment-design planner."
        if pattern.package_type == PackageType.survey_course:
            return f"Survey course covering multiple assignment designs under '{title}'."
        return f"Progressive codebase course for '{title}', with one inherited system evolving across modules."

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
            if domain_pack == "analyst_sql":
                outcomes.append("safe SQL execution")
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
        revision.modules = []
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

    def _prepare_revision_drafts(self, source: CourseRun) -> tuple[list[CourseModuleDraft], str | None]:
        workflow_revision_map: dict[str, WorkflowRun] = {}
        for linked_run_id in self._linked_runs(source):
            workflow_revision_map[linked_run_id] = self.workflow_service.create_revision_from_run(linked_run_id)

        module_drafts: list[CourseModuleDraft] = []
        if source.shared_workflow_run_id and source.shared_workflow_run_id in workflow_revision_map:
            cloned_run = workflow_revision_map[source.shared_workflow_run_id]
            aligned_modules = self._align_progressive_modules(
                [self._request_from_module_draft(module) for module in source.modules],
                cloned_run,
            )
            module_drafts = [
                self._module_draft_from_workflow(
                    module,
                    cloned_run.id,
                    cloned_run.stage.value,
                    cloned_run.status.value,
                    cloned_run.artifacts.draft_kind.value,
                    self._design_spec_from_workflow(cloned_run, module.design_spec or source.shared_design_spec),
                    self._workflow_design_status(cloned_run),
                    extra_notes=[f"Revision draft created from published course `{source.id}`."],
                )
                for module in aligned_modules
            ]
        else:
            for module in source.modules:
                cloned_run = workflow_revision_map.get(module.workflow_run_id) if module.workflow_run_id else None
                if cloned_run is None:
                    module_drafts.append(module.model_copy(deep=True))
                    continue
                module_drafts.append(
                    self._module_draft_from_workflow(
                        self._request_from_module_draft(module),
                        cloned_run.id,
                        cloned_run.stage.value,
                        cloned_run.status.value,
                        cloned_run.artifacts.draft_kind.value,
                        self._design_spec_from_workflow(cloned_run, module.design_spec),
                        self._workflow_design_status(cloned_run),
                        extra_notes=[f"Revision draft created from published course `{source.id}`."],
                    )
                )

        shared_workflow_run_id = (
            workflow_revision_map[source.shared_workflow_run_id].id
            if source.shared_workflow_run_id and source.shared_workflow_run_id in workflow_revision_map
            else None
        )
        return module_drafts, shared_workflow_run_id

    def _finalize_revision(
        self,
        *,
        revision: CourseRun,
        source: CourseRun,
        module_drafts: list[CourseModuleDraft],
        shared_workflow_run_id: str | None,
        event_type: str,
    ) -> CourseRun:
        stage, status = self._course_stage_from_modules(module_drafts)
        revision.updated_at = datetime.now(UTC)
        revision.stage = stage
        revision.status = status
        revision.shared_workflow_run_id = shared_workflow_run_id
        revision.materialized_bundle = None
        revision.latest_publish_snapshot_id = None
        revision.modules = module_drafts
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
            module_drafts, shared_workflow_run_id = self._prepare_revision_drafts(source)
            self._finalize_revision(
                revision=revision,
                source=source,
                module_drafts=module_drafts,
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

    def _module_blockers(
        self,
        module: CourseModuleDraft,
        linked_run: WorkflowRun | None,
    ) -> list[str]:
        blockers: list[str] = []
        if not module.workflow_run_id:
            blockers.append("No linked assignment workflow run.")
            return blockers
        if linked_run is None:
            blockers.append("Linked assignment workflow run is missing.")
            return blockers
        if linked_run.status == WorkflowStatus.blocked:
            blockers.append("Linked assignment workflow is blocked.")
        if linked_run.artifacts.task_agent_spec is None:
            blockers.append("Linked assignment workflow does not have a learner-ready spec yet.")
        if linked_run.artifacts.review_summary is not None:
            blockers.extend(linked_run.artifacts.review_summary.blockers)
        if linked_run.pending_gate is not None:
            blockers.append(f"Linked assignment workflow is waiting on `{linked_run.pending_gate.value}`.")
        if linked_run.status != WorkflowStatus.published:
            blockers.append("Linked assignment workflow is not published yet.")
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
            actions.append("Materialize the course bundle to review syllabus, module sequencing, and aggregate status.")
        if course_run.stage == CourseRunStage.ready_to_publish and course_run.status != CourseRunStatus.published:
            actions.append("Publish the course run once the authoring review is complete.")
        if course_run.shared_workflow_run_id is not None and any(
            workflow.run_id == course_run.shared_workflow_run_id and workflow.status != WorkflowStatus.published
            for workflow in linked_workflows
        ):
            actions.append("The shared progressive assignment workflow gates the whole course; publishing it will unblock every module.")
        return actions

    def _linked_runs(self, course_run: CourseRun) -> dict[str, WorkflowRun]:
        linked_runs: dict[str, WorkflowRun] = {}
        for module in course_run.modules:
            if not module.workflow_run_id:
                continue
            child_run = self.workflow_service.get_run(module.workflow_run_id)
            if child_run is not None:
                linked_runs[module.workflow_run_id] = child_run
        return linked_runs

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
            if len(learner_package.modules) != len(previous_package.modules):
                changes.append(
                    f"Module count changed from {len(previous_package.modules)} to {len(learner_package.modules)}."
                )
            changed_modules = 0
            previous_by_id = {module.module_id: module for module in previous_package.modules}
            for module in learner_package.modules:
                old = previous_by_id.get(module.module_id)
                if old is None:
                    changed_modules += 1
                    continue
                if (
                    module.title != old.title
                    or module.objective != old.objective
                    or module.content_markdown != old.content_markdown
                    or module.starter_readme != old.starter_readme
                    or module.learning_outcomes != old.learning_outcomes
                    or module.checkpoint_module_ids != old.checkpoint_module_ids
                    or module.completion_checkpoint_id != old.completion_checkpoint_id
                    or module.active_test_ids != old.active_test_ids
                    or [file.relative_path for file in module.workspace_seed_files]
                    != [file.relative_path for file in old.workspace_seed_files]
                ):
                    changed_modules += 1
            if changed_modules:
                changes.append(f"Learner package changed in {changed_modules} module(s).")
        if snapshot.task_agent_spec is not None and previous.task_agent_spec is not None:
            if snapshot.task_agent_spec.model_dump(mode="json") != previous.task_agent_spec.model_dump(mode="json"):
                changes.append("Hidden grading or checkpoint contract changed.")
        if not changes:
            changes.append("No learner-visible changes from the previous published version.")
        return changes
