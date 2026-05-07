from __future__ import annotations

import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath
from uuid import uuid4

from app.domain.course import CourseRun, CourseRunStatus, CourseRunSummary
from app.domain.grading import GradeStatus
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
from app.services.learner_studio_service import LearnerStudioService
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
                    module_count=len(snapshot_package.modules) if snapshot_package is not None else summary.module_count,
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

        modules = [self._module_progress(module_package) for module_package in learner_package.modules]
        if modules:
            modules[0].status = LearnerModuleStatus.available

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
            current_module_id=modules[0].module_id if modules else None,
            modules=modules,
            notes=[
                "Enrollment created for the published course.",
                f"Progress is pinned to publish snapshot `{snapshot.id}`.",
            ],
        )
        self.store.save_learner_enrollment(enrollment)
        return enrollment

    def get_enrollment(self, enrollment_id: str) -> LearnerEnrollment:
        enrollment = self._require_enrollment(enrollment_id)
        submissions = self.store.list_learner_submissions(enrollment.id)
        sessions = self.store.list_learner_workspace_sessions(enrollment.id)
        latest_sessions = {session.module_id: session for session in sessions}
        latest_submissions = {submission.module_id: submission for submission in submissions}

        refreshed = enrollment.model_copy(deep=True)
        for module in refreshed.modules:
            module.latest_submission = latest_submissions.get(module.module_id)
            module.workspace_session = latest_sessions.get(module.module_id) or latest_sessions.get(refreshed.current_module_id or "")
        if all(module.status == LearnerModuleStatus.passed for module in refreshed.modules):
            refreshed.status = LearnerEnrollmentStatus.completed
            refreshed.current_module_id = None
        if refreshed.model_dump(mode="json") != enrollment.model_dump(mode="json"):
            refreshed.updated_at = datetime.now(UTC)
            self.store.save_learner_enrollment(refreshed)
        return refreshed

    def get_module_experience(self, enrollment_id: str, module_id: str | None = None) -> LearnerModuleExperience:
        enrollment = self.get_enrollment(enrollment_id)
        active_module = self._resolve_target_module(enrollment, module_id)
        submissions = self.store.list_learner_submissions(enrollment.id, module_id=active_module.module_id)
        return LearnerModuleExperience(
            enrollment=LearnerEnrollmentSummary.from_enrollment(enrollment),
            active_module=active_module,
            modules=enrollment.modules,
            submissions=submissions,
        )

    def launch_workspace(self, enrollment_id: str, request: LaunchWorkspaceRequest) -> LearnerEnrollment:
        enrollment, module, module_package, workspace_root = self._workspace_context(
            enrollment_id,
            request.module_id,
        )

        existing_sessions = self.store.list_learner_workspace_sessions(enrollment.id)
        latest_session = existing_sessions[0] if existing_sessions else None
        session = self.learner_studio_service.launch_editor(
            enrollment_id=enrollment.id,
            module_id=module.module_id,
            workspace_root=workspace_root,
            scope=enrollment.workspace_scope,
            existing_session=latest_session,
        )
        self.store.save_learner_workspace_session(session)
        return self.get_enrollment(enrollment.id)

    def list_workspace_files(self, enrollment_id: str, module_id: str | None = None) -> LearnerWorkspaceFileList:
        enrollment, module, _, workspace_root = self._workspace_context(enrollment_id, module_id)
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
            module_id=module.module_id,
            workspace_root=str(root),
            files=files,
        )

    def read_workspace_file(
        self,
        enrollment_id: str,
        relative_path: str,
        module_id: str | None = None,
    ) -> LearnerWorkspaceFileContent:
        enrollment, module, _, workspace_root = self._workspace_context(enrollment_id, module_id)
        root = workspace_root.resolve()
        target = self._resolve_workspace_file(workspace_root, relative_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(relative_path)
        return LearnerWorkspaceFileContent(
            enrollment_id=enrollment.id,
            module_id=module.module_id,
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
        enrollment, module, _, workspace_root = self._workspace_context(enrollment_id, payload.module_id)
        root = workspace_root.resolve()
        target = self._resolve_workspace_file(workspace_root, payload.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload.content, encoding="utf-8")
        return LearnerWorkspaceFileWriteResult(
            enrollment_id=enrollment.id,
            module_id=module.module_id,
            workspace_root=str(root),
            relative_path=target.relative_to(root).as_posix(),
            media_type=self._guess_media_type(target),
            size_bytes=target.stat().st_size,
        )

    def submit_module(self, enrollment_id: str, request: SubmitModuleRequest) -> LearnerModuleExperience:
        enrollment, module, module_package, workspace_root = self._workspace_context(
            enrollment_id,
            request.module_id,
        )
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)
        if snapshot.task_agent_spec is None:
            raise LMSConflictError("The publish snapshot is missing the internal grading spec.")

        report = self.learner_studio_service.grade_workspace(
            workspace_root=workspace_root,
            spec=snapshot.task_agent_spec,
            module_id=module_package.completion_checkpoint_id or module.module_id,
        )
        submission = LearnerSubmissionRecord(
            id=f"submission_{uuid4().hex[:12]}",
            enrollment_id=enrollment.id,
            module_id=module.module_id,
            created_at=datetime.now(UTC),
            status=report.grade_report.status.value,
            passed_tests=report.grade_report.passed_tests,
            total_tests=report.grade_report.total_tests,
            pass_rate=report.grade_report.pass_rate,
            grade_report=report.grade_report,
        )
        self.store.save_learner_submission(submission)

        refreshed = enrollment.model_copy(deep=True)
        target_index = next(index for index, item in enumerate(refreshed.modules) if item.module_id == module.module_id)
        refreshed.modules[target_index].latest_submission = submission
        if report.grade_report.status == GradeStatus.passed:
            refreshed.modules[target_index].status = LearnerModuleStatus.passed
            if target_index + 1 < len(refreshed.modules):
                next_module = refreshed.modules[target_index + 1]
                if next_module.status == LearnerModuleStatus.locked:
                    next_module.status = LearnerModuleStatus.available
                refreshed.current_module_id = next_module.module_id
                self._sync_workspace_for_module(
                    self._resolve_module_package(snapshot, next_module.module_id),
                    workspace_root,
                    preserve_app=True,
                )
            else:
                refreshed.current_module_id = None
                refreshed.status = LearnerEnrollmentStatus.completed
        else:
            refreshed.current_module_id = module.module_id
        refreshed.updated_at = datetime.now(UTC)
        self.store.save_learner_enrollment(refreshed)
        return self.get_module_experience(refreshed.id, module.module_id)

    def _workspace_root(self, enrollment: LearnerEnrollment) -> Path:
        return self.base_dir / enrollment.id / "workspace"

    def _workspace_context(
        self,
        enrollment_id: str,
        module_id: str | None = None,
    ) -> tuple[LearnerEnrollment, LearnerModuleProgress, LearnerModulePackage, Path]:
        enrollment = self.get_enrollment(enrollment_id)
        module = self._resolve_target_module(enrollment, module_id)
        if module.status == LearnerModuleStatus.locked:
            raise LMSConflictError(f"Module '{module.module_id}' is still locked.")
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)
        module_package = self._resolve_module_package(snapshot, module.module_id)
        workspace_root = self._workspace_root(enrollment)
        self._sync_workspace_for_module(module_package, workspace_root)
        return enrollment, module, module_package, workspace_root

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
        if any(module.completion_checkpoint_id is None for module in snapshot.learner_package.modules):
            return False, "This course is being prepared and is not ready for learners yet."
        return True, None

    def _module_progress(self, module_package: LearnerModulePackage) -> LearnerModuleProgress:
        return LearnerModuleProgress(
            module_id=module_package.module_id,
            title=module_package.title,
            objective=module_package.objective,
            status=LearnerModuleStatus.locked,
            module_index=module_package.module_index,
            content_markdown=module_package.content_markdown,
            starter_readme=module_package.starter_readme,
            visible_files=list(module_package.visible_files),
        )

    def _sync_workspace_for_module(
        self,
        module_package: LearnerModulePackage,
        workspace_root: Path,
        *,
        preserve_app: bool = True,
    ) -> None:
        workspace_root.mkdir(parents=True, exist_ok=True)
        files_to_write = {
            file.relative_path: file.content
            for file in module_package.workspace_seed_files
        }
        files_to_write[".coursegen/current_module.txt"] = module_package.module_id + "\n"
        if preserve_app and (workspace_root / "app.py").exists():
            files_to_write.pop("app.py", None)

        for relative_path, content in files_to_write.items():
            target = workspace_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

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

    def _resolve_target_module(self, enrollment: LearnerEnrollment, requested_module_id: str | None) -> LearnerModuleProgress:
        target_module_id = requested_module_id or enrollment.current_module_id
        if target_module_id is None and enrollment.modules:
            target_module_id = enrollment.modules[-1].module_id
        for module in enrollment.modules:
            if module.module_id == target_module_id:
                return module
        raise LMSConflictError("Could not resolve the active learner module.")

    def _resolve_module_package(self, snapshot: PublishSnapshot, module_id: str) -> LearnerModulePackage:
        learner_package = snapshot.learner_package
        if learner_package is None:
            raise LMSConflictError("This publish snapshot is missing its learner package.")
        for module in learner_package.modules:
            if module.module_id == module_id:
                return module
        raise LMSConflictError(f"Module '{module_id}' is not present in publish snapshot '{snapshot.id}'.")

    def _require_enrollment(self, enrollment_id: str) -> LearnerEnrollment:
        enrollment = self.store.get_learner_enrollment(enrollment_id)
        if enrollment is None:
            raise KeyError(enrollment_id)
        return enrollment
