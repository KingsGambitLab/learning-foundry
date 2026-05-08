from __future__ import annotations

import json
import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath
from uuid import uuid4

from app.domain.course import CourseRun, CourseRunStatus, CourseRunSummary
from app.domain.grading import AssignmentGradeReport, GradeStatus, ModuleGradeReport, ReviewAreaGradeReport
from app.domain.learner import (
    CreateEnrollmentRequest,
    LaunchWorkspaceRequest,
    LearnerEnrollment,
    LearnerEnrollmentList,
    LearnerEnrollmentStatus,
    LearnerEnrollmentSummary,
    LearnerModuleExperience,
    LearnerModuleProgress,
    LearnerModuleStatus,
    LearnerSubmissionRecord,
    LearnerWorkspaceFileContent,
    LearnerWorkspaceFileList,
    LearnerWorkspaceFileSummary,
    LearnerWorkspaceFileWriteResult,
    LearnerWorkspaceScope,
    PublishedCourseCatalog,
    PublishedCourseSummary,
    SubmitModuleRequest,
    WriteLearnerWorkspaceFileRequest,
)
from app.domain.publish import LearnerModulePackage, PublishSnapshot
from app.domain.testing import (
    CreateLearnerFeedbackRequest,
    LearnerFeedbackList,
    LearnerFeedbackRecord,
    LearnerTestingView,
)
from app.services.learner_studio_service import LearnerStudioService
from app.services.task_agent_starter_templates import (
    render_legacy_task_agent_root_app,
    render_task_agent_root_app,
)
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def default_learner_workspace_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "learner_workspaces"


class LMSConflictError(ValueError):
    """Raised when an LMS action is invalid for the current course or enrollment state."""


class LMSService:
    MAX_WORKSPACE_FILE_BYTES = 1_000_000

    def __init__(
        self,
        store: SQLiteWorkflowStore,
        workflow_service: WorkflowService,
        learner_studio_service: LearnerStudioService | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self.store = store
        self.workflow_service = workflow_service
        self.learner_studio_service = learner_studio_service or LearnerStudioService()
        self.base_dir = Path(base_dir or default_learner_workspace_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_catalog(self) -> PublishedCourseCatalog:
        latest_run_by_family: dict[str, tuple[CourseRun, PublishSnapshot | None]] = {}
        for summary in self.store.list_course_runs(limit=200):
            run = self.store.get_course_run(summary.id)
            if run is None or run.status != CourseRunStatus.published:
                continue
            snapshot = self._latest_snapshot(run)
            family_id = run.course_family_id
            current = latest_run_by_family.get(family_id)
            current_snapshot = current[1] if current is not None else None
            current_timestamp = current_snapshot.created_at if current_snapshot is not None else (current[0].updated_at if current is not None else None)
            candidate_timestamp = snapshot.created_at if snapshot is not None else run.updated_at
            if current is None or candidate_timestamp >= current_timestamp:
                latest_run_by_family[family_id] = (run, snapshot)

        courses: list[PublishedCourseSummary] = []
        for run, snapshot in latest_run_by_family.values():
            summary = CourseRunSummary.from_run(run)
            supported, reason = self._lms_support(run, snapshot)
            snapshot_package = snapshot.learner_package if snapshot is not None else None
            courses.append(
                PublishedCourseSummary.from_run(
                    summary,
                    title=snapshot_package.title if snapshot_package is not None else run.title,
                    summary=snapshot_package.summary if snapshot_package is not None else run.summary,
                    deliverable_count=len(snapshot_package.deliverables) if snapshot_package is not None else summary.deliverable_count,
                    shared_workflow_run_id=run.shared_workflow_run_id,
                    supported_for_lms=supported,
                    support_reason=reason,
                    publish_snapshot_id=snapshot.id if snapshot is not None else run.latest_publish_snapshot_id,
                    published_at=snapshot.created_at if snapshot is not None else run.updated_at,
                )
            )
        courses.sort(key=lambda item: item.published_at, reverse=True)
        return PublishedCourseCatalog(courses=courses)

    def list_enrollments(self, learner_id: str = "local-learner") -> LearnerEnrollmentList:
        return LearnerEnrollmentList(enrollments=self.store.list_learner_enrollments(learner_id=learner_id))

    def enroll(self, request: CreateEnrollmentRequest) -> LearnerEnrollment:
        existing = self.store.find_learner_enrollment(request.learner_id, request.course_run_id)
        if existing is not None:
            return self.get_enrollment(existing.id)

        course_run = self._require_published_course(request.course_run_id)
        snapshot = self._require_supported_snapshot(course_run)
        learner_package = snapshot.learner_package
        assert learner_package is not None
        now = datetime.now(UTC)

        deliverables = [self._deliverable_progress(item) for item in learner_package.deliverables]

        enrollment = LearnerEnrollment(
            id=f"enrollment_{uuid4().hex[:12]}",
            learner_id=request.learner_id,
            course_run_id=course_run.id,
            publish_snapshot_id=snapshot.id,
            course_title=learner_package.title,
            course_summary=learner_package.summary,
            package_type=learner_package.package_type,
            shared_workflow_run_id=snapshot.shared_workflow_run_id or course_run.shared_workflow_run_id or "shared_workflow",
            created_at=now,
            updated_at=now,
            status=LearnerEnrollmentStatus.active,
            workspace_scope=learner_package.workspace_scope,
            current_deliverable_id=deliverables[0].deliverable_id if deliverables else None,
            deliverables=deliverables,
            notes=[
                "Enrollment created for the published course.",
                f"Progress is pinned to publish snapshot `{snapshot.id}`.",
            ],
        )
        self.store.save_learner_enrollment(enrollment)
        self._ensure_workspace_seeded(enrollment, snapshot)
        return enrollment

    def get_enrollment(self, enrollment_id: str) -> LearnerEnrollment:
        enrollment = self._require_enrollment(enrollment_id)
        submissions = self.store.list_learner_submissions(enrollment.id)
        sessions = self.store.list_learner_workspace_sessions(enrollment.id)
        latest_session = sessions[0] if sessions else None
        latest_submissions: dict[str, LearnerSubmissionRecord] = {}
        for submission in submissions:
            current = latest_submissions.get(submission.deliverable_id)
            if current is None or submission.created_at > current.created_at:
                latest_submissions[submission.deliverable_id] = submission

        refreshed = enrollment.model_copy(deep=True)
        for deliverable in refreshed.deliverables:
            latest_submission = latest_submissions.get(deliverable.deliverable_id)
            deliverable.latest_submission = latest_submission
            if latest_submission is not None:
                deliverable.status = (
                    LearnerModuleStatus.passed
                    if latest_submission.grade_report is not None
                    and latest_submission.grade_report.status == GradeStatus.passed
                    else LearnerModuleStatus.available
                )
            deliverable.workspace_session = latest_session
        if all(deliverable.status == LearnerModuleStatus.passed for deliverable in refreshed.deliverables):
            refreshed.status = LearnerEnrollmentStatus.completed
            refreshed.current_deliverable_id = None
        if refreshed.model_dump(mode="json") != enrollment.model_dump(mode="json"):
            refreshed.updated_at = datetime.now(UTC)
            self.store.save_learner_enrollment(refreshed)
        return refreshed

    def get_deliverable_experience(self, enrollment_id: str, deliverable_id: str | None = None) -> LearnerModuleExperience:
        enrollment = self.get_enrollment(enrollment_id)
        active_deliverable = self._resolve_target_deliverable(enrollment, deliverable_id)
        all_submissions = self.store.list_learner_submissions(enrollment.id)
        latest_assignment_submission = self._latest_assignment_submission(all_submissions)
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)
        self._ensure_workspace_seeded(enrollment, snapshot)
        latest_session = self.store.list_learner_workspace_sessions(enrollment.id)
        workspace_session = latest_session[0] if latest_session else None
        project_brief_markdown = self._project_brief_markdown(snapshot)
        return LearnerModuleExperience(
            enrollment=LearnerEnrollmentSummary.from_enrollment(enrollment),
            project_brief_markdown=project_brief_markdown,
            workspace_session=workspace_session,
            latest_assignment_report=(
                latest_assignment_submission.assignment_report
                if latest_assignment_submission is not None
                else None
            ),
            latest_assignment_submission=latest_assignment_submission,
            active_deliverable=active_deliverable,
            deliverables=enrollment.deliverables,
            submissions=all_submissions,
        )

    def get_learner_view(self, enrollment_id: str, deliverable_id: str | None = None) -> LearnerTestingView:
        return LearnerTestingView(
            experience=self.get_deliverable_experience(enrollment_id, deliverable_id),
            feedback=self.store.list_learner_feedback(enrollment_id),
        )

    def record_feedback(
        self,
        enrollment_id: str,
        request: CreateLearnerFeedbackRequest,
    ) -> LearnerFeedbackRecord:
        enrollment = self.get_enrollment(enrollment_id)
        deliverable = self._resolve_target_deliverable(enrollment, request.deliverable_id)
        feedback = LearnerFeedbackRecord(
            id=f"learner_feedback_{uuid4().hex[:12]}",
            enrollment_id=enrollment.id,
            course_run_id=enrollment.course_run_id,
            publish_snapshot_id=enrollment.publish_snapshot_id,
            learner_id=enrollment.learner_id,
            created_at=datetime.now(UTC),
            summary=request.summary.strip(),
            details=request.details.strip() if request.details else None,
            rating=request.rating,
            deliverable_id=deliverable.deliverable_id,
            context=request.context,
        )
        self.store.save_learner_feedback(feedback)
        return feedback

    def list_feedback(self, enrollment_id: str) -> LearnerFeedbackList:
        self._require_enrollment(enrollment_id)
        return LearnerFeedbackList(items=self.store.list_learner_feedback(enrollment_id))

    def launch_workspace(self, enrollment_id: str, request: LaunchWorkspaceRequest) -> LearnerEnrollment:
        enrollment, deliverable, _, workspace_root = self._workspace_context(
            enrollment_id,
            request.deliverable_id,
        )

        existing_sessions = self.store.list_learner_workspace_sessions(enrollment.id)
        latest_session = existing_sessions[0] if existing_sessions else None
        session = self.learner_studio_service.launch_editor(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=workspace_root,
            scope=enrollment.workspace_scope,
            existing_session=latest_session,
        )
        self.store.save_learner_workspace_session(session)
        if enrollment.current_deliverable_id != deliverable.deliverable_id:
            refreshed = enrollment.model_copy(deep=True)
            refreshed.current_deliverable_id = deliverable.deliverable_id
            refreshed.updated_at = datetime.now(UTC)
            self.store.save_learner_enrollment(refreshed)
        return self.get_enrollment(enrollment.id)

    def list_workspace_files(self, enrollment_id: str, deliverable_id: str | None = None) -> LearnerWorkspaceFileList:
        enrollment, deliverable, _, workspace_root = self._workspace_context(enrollment_id, deliverable_id)
        root = workspace_root.resolve()
        files = [
            LearnerWorkspaceFileSummary(
                relative_path=path.resolve().relative_to(root).as_posix(),
                media_type=self._guess_media_type(path),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        ]
        return LearnerWorkspaceFileList(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=str(root),
            files=files,
        )

    def read_workspace_file(
        self,
        enrollment_id: str,
        relative_path: str,
        deliverable_id: str | None = None,
    ) -> LearnerWorkspaceFileContent:
        enrollment, deliverable, _, workspace_root = self._workspace_context(enrollment_id, deliverable_id)
        root = workspace_root.resolve()
        target = self._resolve_workspace_file(workspace_root, relative_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(relative_path)
        return LearnerWorkspaceFileContent(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=str(root),
            relative_path=target.relative_to(root).as_posix(),
            media_type=self._guess_media_type(target),
            content=target.read_text(encoding="utf-8"),
        )

    def write_workspace_file(
        self,
        enrollment_id: str,
        payload: WriteLearnerWorkspaceFileRequest,
    ) -> LearnerWorkspaceFileWriteResult:
        if len(payload.content.encode("utf-8")) > self.MAX_WORKSPACE_FILE_BYTES:
            raise LMSConflictError("Workspace file payload is too large for the LMS prototype.")
        enrollment, deliverable, _, workspace_root = self._workspace_context(enrollment_id, payload.deliverable_id)
        root = workspace_root.resolve()
        target = self._resolve_workspace_file(workspace_root, payload.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload.content, encoding="utf-8")
        return LearnerWorkspaceFileWriteResult(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=str(root),
            relative_path=target.relative_to(root).as_posix(),
            media_type=self._guess_media_type(target),
            size_bytes=target.stat().st_size,
        )

    def submit_project(self, enrollment_id: str, request: SubmitModuleRequest) -> LearnerModuleExperience:
        enrollment, deliverable, deliverable_package, workspace_root = self._workspace_context(
            enrollment_id,
            request.deliverable_id,
        )
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)
        if snapshot.task_agent_spec is None:
            raise LMSConflictError("The publish snapshot is missing the internal grading spec.")

        report = self.learner_studio_service.grade_assignment(
            workspace_root=workspace_root,
            spec=snapshot.task_agent_spec,
        )
        submission_group_id = f"submission_{uuid4().hex[:12]}"
        created_at = datetime.now(UTC)
        assignment_report = self._learner_assignment_report(snapshot, report.assignment_report)
        submissions_by_deliverable: dict[str, LearnerSubmissionRecord] = {}
        for review_area in assignment_report.review_areas:
            submission = LearnerSubmissionRecord(
                id=f"{submission_group_id}_{review_area.deliverable_id.replace('/', '_')}",
                submission_group_id=submission_group_id,
                enrollment_id=enrollment.id,
                deliverable_id=review_area.deliverable_id,
                created_at=created_at,
                status=review_area.grade_report.status.value,
                passed_tests=review_area.grade_report.passed_tests,
                total_tests=review_area.grade_report.total_tests,
                pass_rate=review_area.grade_report.pass_rate,
                grade_report=review_area.grade_report,
                assignment_report=assignment_report,
            )
            self.store.save_learner_submission(submission)
            submissions_by_deliverable[review_area.deliverable_id] = submission

        refreshed = enrollment.model_copy(deep=True)
        for item in refreshed.deliverables:
            latest_submission = submissions_by_deliverable.get(item.deliverable_id)
            if latest_submission is not None:
                item.latest_submission = latest_submission
                item.status = (
                    LearnerModuleStatus.passed
                    if latest_submission.grade_report.status == GradeStatus.passed
                    else LearnerModuleStatus.available
                )
        refreshed.current_deliverable_id = deliverable.deliverable_id
        if assignment_report.status == GradeStatus.passed:
            refreshed.current_deliverable_id = None
            refreshed.status = LearnerEnrollmentStatus.completed
        else:
            refreshed.status = LearnerEnrollmentStatus.active
        refreshed.updated_at = datetime.now(UTC)
        self.store.save_learner_enrollment(refreshed)
        return self.get_deliverable_experience(refreshed.id, deliverable.deliverable_id)

    def _workspace_root(self, enrollment: LearnerEnrollment) -> Path:
        return self.base_dir / enrollment.id / "workspace"

    def _workspace_context(
        self,
        enrollment_id: str,
        deliverable_id: str | None = None,
    ) -> tuple[LearnerEnrollment, LearnerModuleProgress, LearnerModulePackage, Path]:
        enrollment = self.get_enrollment(enrollment_id)
        deliverable = self._resolve_target_deliverable(enrollment, deliverable_id)
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)
        deliverable_package = self._resolve_deliverable_package(snapshot, deliverable.deliverable_id)
        workspace_root = self._workspace_root(enrollment)
        self._ensure_workspace_seeded(enrollment, snapshot)
        return enrollment, deliverable, deliverable_package, workspace_root

    def _require_published_course(self, course_run_id: str) -> CourseRun:
        course_run = self.store.get_course_run(course_run_id)
        if course_run is None:
            raise KeyError(course_run_id)
        if course_run.status != CourseRunStatus.published:
            raise LMSConflictError("Only published courses can be enrolled.")
        return course_run

    def _require_supported_snapshot(self, course_run: CourseRun) -> PublishSnapshot:
        snapshot = self._latest_snapshot(course_run)
        if snapshot is None:
            raise LMSConflictError("This published course does not have a learner-ready publish snapshot yet.")
        if snapshot.learner_package is None or snapshot.task_agent_spec is None:
            raise LMSConflictError("This published course is not yet packaged for the LMS learner flow.")
        return snapshot

    def _require_snapshot(self, snapshot_id: str) -> PublishSnapshot:
        snapshot = self.store.get_publish_snapshot(snapshot_id)
        if snapshot is None:
            raise LMSConflictError(f"Publish snapshot '{snapshot_id}' is missing.")
        return snapshot

    def _latest_snapshot(self, course_run: CourseRun) -> PublishSnapshot | None:
        if course_run.latest_publish_snapshot_id:
            snapshot = self.store.get_publish_snapshot(course_run.latest_publish_snapshot_id)
            if snapshot is not None:
                return snapshot
        return self.store.get_latest_publish_snapshot(course_run.id)

    def _lms_support(self, course_run: CourseRun, snapshot: PublishSnapshot | None) -> tuple[bool, str | None]:
        if course_run.status != CourseRunStatus.published:
            return False, "This course is still being prepared and is not available to learners yet."
        if not course_run.shared_workflow_run_id:
            return False, "This course is still being prepared and is not available to learners yet."
        if snapshot is None:
            return False, "This course is being prepared and is not ready for learners yet."
        if snapshot.learner_package is None or snapshot.task_agent_spec is None:
            return False, "This course is being prepared and is not ready for learners yet."
        if not snapshot.learner_package.deliverables:
            return False, "This course is being prepared and is not ready for learners yet."
        return True, None

    def _deliverable_progress(self, deliverable_package: LearnerModulePackage) -> LearnerModuleProgress:
        return LearnerModuleProgress(
            deliverable_id=deliverable_package.deliverable_id,
            title=deliverable_package.title,
            objective=deliverable_package.objective,
            status=LearnerModuleStatus.available,
            deliverable_index=deliverable_package.deliverable_index,
            content_markdown=deliverable_package.content_markdown,
            starter_readme=deliverable_package.starter_readme,
            visible_files=list(deliverable_package.visible_files),
        )

    def _ensure_workspace_seeded(self, enrollment: LearnerEnrollment, snapshot: PublishSnapshot) -> Path:
        workspace_root = self._workspace_root(enrollment)
        workspace_root.mkdir(parents=True, exist_ok=True)
        seed_marker = workspace_root / ".coursegen" / "workspace_seeded.txt"

        learner_package = snapshot.learner_package
        if learner_package is None or not learner_package.deliverables:
            raise LMSConflictError("This publish snapshot is missing learner workspace seed files.")

        files_to_write: dict[str, str] = {}
        for deliverable in learner_package.deliverables:
            for file in deliverable.workspace_seed_files:
                files_to_write.setdefault(file.relative_path, file.content)
        legacy_app = render_legacy_task_agent_root_app()
        current_app = render_task_agent_root_app()
        if files_to_write.get("app.py") == legacy_app:
            files_to_write["app.py"] = current_app
        project_brief_markdown = self._project_brief_markdown(snapshot)
        files_to_write["README.md"] = project_brief_markdown
        files_to_write["project_brief.md"] = project_brief_markdown
        files_to_write["deliverables.md"] = self._deliverables_markdown(learner_package.deliverables)
        files_to_write[".coursegen/workspace_seeded.txt"] = snapshot.id + "\n"
        files_to_write[".coursegen/review_areas/index.json"] = self._review_area_index_json(learner_package.deliverables)
        files_to_write[".coursegen/deliverables/index.json"] = self._review_area_index_json(learner_package.deliverables)
        for deliverable in learner_package.deliverables:
            files_to_write[f".coursegen/review_areas/{deliverable.deliverable_id}/README.md"] = deliverable.starter_readme
            files_to_write[f".coursegen/review_areas/{deliverable.deliverable_id}/module_content.md"] = deliverable.content_markdown
            files_to_write[f".coursegen/deliverables/{deliverable.deliverable_id}/README.md"] = deliverable.starter_readme
            files_to_write[f".coursegen/deliverables/{deliverable.deliverable_id}/deliverable.md"] = deliverable.content_markdown

        for relative_path, content in files_to_write.items():
            target = workspace_root / relative_path
            if target.exists():
                if relative_path == "app.py" and target.read_text(encoding="utf-8") == legacy_app:
                    target.write_text(current_app, encoding="utf-8")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return workspace_root

    def _project_brief_markdown(self, snapshot: PublishSnapshot) -> str:
        learner_package = snapshot.learner_package
        if learner_package is None:
            return ""
        brief = (learner_package.project_brief_markdown or "").strip()
        if brief:
            return brief
        lines = [
            f"# {learner_package.title}",
            "",
            learner_package.summary or "Build the shared project in the workspace.",
            "",
            "## What review will look at",
            "",
        ]
        for index, deliverable in enumerate(learner_package.deliverables, start=1):
            objective = (deliverable.objective or deliverable.title).strip()
            lines.append(f"{index}. **{deliverable.title}** - {objective}")
        lines.extend(
            [
                "",
                "## How to work",
                "",
                "- Open the shared VS Code workspace.",
                "- Run the visible checks while you iterate.",
                "- Submit the full project for review.",
                "- Use the deliverable scorecard to see what still needs work.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def _review_area_index_json(self, deliverables: list[LearnerModulePackage]) -> str:
        rows = [
            {
                "deliverable_id": deliverable.deliverable_id,
                "title": deliverable.title,
                "objective": deliverable.objective,
                "deliverable_index": deliverable.deliverable_index,
            }
            for deliverable in deliverables
        ]
        return json.dumps({"review_areas": rows, "deliverables": rows}, indent=2, ensure_ascii=True) + "\n"

    def _deliverables_markdown(self, deliverables: list[LearnerModulePackage]) -> str:
        lines = [
            "# Project deliverables",
            "",
            "Use this as the checklist for what review will look at on submission.",
            "",
        ]
        for deliverable in deliverables:
            lines.extend(
                [
                    f"## {deliverable.deliverable_index}. {deliverable.title}",
                    "",
                    deliverable.objective,
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _learner_assignment_report(
        self,
        snapshot: PublishSnapshot,
        assignment_report: AssignmentGradeReport,
    ) -> AssignmentGradeReport:
        learner_package = snapshot.learner_package
        if learner_package is None:
            return assignment_report

        spec_deliverable_order = (
            {
                deliverable.id: index
                for index, deliverable in enumerate(snapshot.task_agent_spec.modules)
            }
            if snapshot.task_agent_spec is not None
            else {}
        )
        deliverable_positions = {
            deliverable.deliverable_id: position
            for position, deliverable in enumerate(learner_package.deliverables)
        }
        aggregated: dict[str, ReviewAreaGradeReport] = {}
        for review_area in assignment_report.review_areas:
            learner_deliverable = self._resolve_review_area_deliverable(
                learner_package.deliverables,
                review_area.deliverable_id,
                spec_deliverable_order,
            )
            learner_deliverable_id = (
                learner_deliverable.deliverable_id if learner_deliverable is not None else review_area.deliverable_id
            )
            learner_title = learner_deliverable.title if learner_deliverable is not None else review_area.title
            learner_objective = learner_deliverable.objective if learner_deliverable is not None else review_area.objective
            learner_index = (
                learner_deliverable.deliverable_index
                if learner_deliverable is not None
                else review_area.deliverable_index
            )

            existing = aggregated.get(learner_deliverable_id)
            if existing is None:
                aggregated[learner_deliverable_id] = ReviewAreaGradeReport(
                    deliverable_id=learner_deliverable_id,
                    title=learner_title,
                    objective=learner_objective,
                    deliverable_index=learner_index,
                    grade_report=review_area.grade_report.model_copy(update={"deliverable_id": learner_deliverable_id}),
                )
                continue

            merged_results = [*existing.grade_report.results, *review_area.grade_report.results]
            merged_warnings = list(dict.fromkeys([
                *existing.grade_report.submission_warnings,
                *review_area.grade_report.submission_warnings,
            ]))
            passed_tests = existing.grade_report.passed_tests + review_area.grade_report.passed_tests
            total_tests = existing.grade_report.total_tests + review_area.grade_report.total_tests
            failed_tests = total_tests - passed_tests
            pass_rate = passed_tests / total_tests if total_tests else 0.0
            aggregated[learner_deliverable_id] = ReviewAreaGradeReport(
                deliverable_id=learner_deliverable_id,
                title=learner_title,
                objective=learner_objective,
                deliverable_index=learner_index,
                grade_report=ModuleGradeReport(
                    deliverable_id=learner_deliverable_id,
                    total_tests=total_tests,
                    passed_tests=passed_tests,
                    failed_tests=failed_tests,
                    pass_rate=pass_rate,
                    status=GradeStatus.passed if failed_tests == 0 else GradeStatus.failed,
                    results=merged_results,
                    submission_warnings=merged_warnings,
                ),
            )

        ordered_review_areas = sorted(
            aggregated.values(),
            key=lambda item: (
                deliverable_positions.get(item.deliverable_id, item.deliverable_index - 1),
                item.deliverable_index,
                item.deliverable_id,
            ),
        )
        passed_tests = sum(item.grade_report.passed_tests for item in ordered_review_areas)
        total_tests = sum(item.grade_report.total_tests for item in ordered_review_areas)
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests else 0.0
        return AssignmentGradeReport(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            pass_rate=pass_rate,
            status=GradeStatus.passed if failed_tests == 0 else GradeStatus.failed,
            review_areas=ordered_review_areas,
            submission_warnings=assignment_report.submission_warnings,
        )

    def _resolve_review_area_deliverable(
        self,
        deliverables: list[LearnerModulePackage],
        spec_deliverable_id: str,
        spec_deliverable_order: dict[str, int] | None = None,
    ) -> LearnerModulePackage | None:
        for deliverable in deliverables:
            if deliverable.deliverable_id == spec_deliverable_id:
                return deliverable
        if spec_deliverable_order is not None:
            deliverable_position = spec_deliverable_order.get(spec_deliverable_id)
            if deliverable_position is not None and 0 <= deliverable_position < len(deliverables):
                return deliverables[deliverable_position]
        return None

    def _resolve_workspace_file(self, workspace_root: Path, relative_path: str) -> Path:
        normalized = PurePosixPath(relative_path.strip())
        if not normalized.parts:
            raise LMSConflictError("Workspace file path is required.")
        if normalized.is_absolute():
            raise LMSConflictError("Workspace file path must be relative.")
        if ".." in normalized.parts:
            raise LMSConflictError("Workspace file path must stay inside the learner workspace.")
        target = (workspace_root / Path(*normalized.parts)).resolve()
        root = workspace_root.resolve()
        if root != target and root not in target.parents:
            raise LMSConflictError("Workspace file path must stay inside the learner workspace.")
        return target

    def _guess_media_type(self, path: Path) -> str:
        media_type, _ = mimetypes.guess_type(str(path))
        return media_type or "text/plain"

    def _resolve_target_deliverable(
        self,
        enrollment: LearnerEnrollment,
        requested_deliverable_id: str | None,
    ) -> LearnerModuleProgress:
        target_deliverable_id = requested_deliverable_id or enrollment.current_deliverable_id
        if target_deliverable_id is None and enrollment.deliverables:
            target_deliverable_id = enrollment.deliverables[-1].deliverable_id
        for deliverable in enrollment.deliverables:
            if deliverable.deliverable_id == target_deliverable_id:
                return deliverable
        raise LMSConflictError("Could not resolve the active learner deliverable.")

    def _latest_assignment_submission(
        self,
        submissions: list[LearnerSubmissionRecord],
    ) -> LearnerSubmissionRecord | None:
        grouped: dict[str, LearnerSubmissionRecord] = {}
        for submission in submissions:
            group_id = submission.submission_group_id or submission.id
            current = grouped.get(group_id)
            if current is None or submission.created_at > current.created_at:
                grouped[group_id] = submission
        if not grouped:
            return None
        return max(grouped.values(), key=lambda item: item.created_at)

    def _resolve_deliverable_package(self, snapshot: PublishSnapshot, deliverable_id: str) -> LearnerModulePackage:
        learner_package = snapshot.learner_package
        if learner_package is None:
            raise LMSConflictError("This publish snapshot is missing its learner package.")
        for deliverable in learner_package.deliverables:
            if deliverable.deliverable_id == deliverable_id:
                return deliverable
        raise LMSConflictError(f"Deliverable '{deliverable_id}' is not present in publish snapshot '{snapshot.id}'.")

    def _require_enrollment(self, enrollment_id: str) -> LearnerEnrollment:
        enrollment = self.store.get_learner_enrollment(enrollment_id)
        if enrollment is None:
            raise KeyError(enrollment_id)
        return enrollment
