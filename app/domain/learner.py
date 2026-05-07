from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.domain.course import CourseRunStatus, CourseRunSummary
from app.domain.grading import ModuleGradeReport
from app.domain.registry import PackageType


class LearnerEnrollmentStatus(str, Enum):
    active = "active"
    completed = "completed"
    blocked = "blocked"


class LearnerModuleStatus(str, Enum):
    locked = "locked"
    available = "available"
    passed = "passed"


class LearnerWorkspaceScope(str, Enum):
    shared_course = "shared_course"
    per_module = "per_module"


class LearnerWorkspaceSessionStatus(str, Enum):
    starting = "starting"
    running = "running"
    stopped = "stopped"
    failed = "failed"


class PublishedCourseSummary(BaseModel):
    course_run_id: str
    publish_snapshot_id: str | None = None
    title: str
    summary: str
    package_type: PackageType
    module_count: int
    shared_workflow_run_id: str | None = None
    supported_for_lms: bool = False
    support_reason: str | None = None
    course_run_status: CourseRunStatus
    published_at: datetime

    @classmethod
    def from_run(
        cls,
        run: CourseRunSummary,
        *,
        title: str | None = None,
        summary: str,
        module_count: int | None = None,
        shared_workflow_run_id: str | None,
        supported_for_lms: bool,
        support_reason: str | None,
        publish_snapshot_id: str | None,
        published_at: datetime | None = None,
    ) -> "PublishedCourseSummary":
        return cls(
            course_run_id=run.id,
            publish_snapshot_id=publish_snapshot_id,
            title=title or run.title,
            summary=summary,
            package_type=run.package_type,
            module_count=module_count if module_count is not None else run.module_count,
            shared_workflow_run_id=shared_workflow_run_id,
            supported_for_lms=supported_for_lms,
            support_reason=support_reason,
            course_run_status=run.status,
            published_at=published_at or run.updated_at,
        )


class PublishedCourseCatalog(BaseModel):
    courses: list[PublishedCourseSummary] = Field(default_factory=list)


class LearnerWorkspaceSession(BaseModel):
    id: str
    enrollment_id: str
    module_id: str
    scope: LearnerWorkspaceScope
    created_at: datetime
    updated_at: datetime
    status: LearnerWorkspaceSessionStatus
    workspace_root: str
    container_name: str | None = None
    host_port: int | None = None
    editor_url: str | None = None
    image_name: str | None = None
    notes: list[str] = Field(default_factory=list)


class LearnerSubmissionRecord(BaseModel):
    id: str
    enrollment_id: str
    module_id: str
    created_at: datetime
    status: str
    passed_tests: int
    total_tests: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    grade_report: ModuleGradeReport


class LearnerModuleProgress(BaseModel):
    module_id: str
    title: str
    objective: str
    status: LearnerModuleStatus
    module_index: int
    content_markdown: str = ""
    starter_readme: str = ""
    visible_files: list[str] = Field(default_factory=list)
    latest_submission: LearnerSubmissionRecord | None = None
    workspace_session: LearnerWorkspaceSession | None = None


class LearnerEnrollment(BaseModel):
    id: str
    learner_id: str
    course_run_id: str
    publish_snapshot_id: str
    course_title: str
    course_summary: str
    package_type: PackageType
    shared_workflow_run_id: str
    created_at: datetime
    updated_at: datetime
    status: LearnerEnrollmentStatus
    workspace_scope: LearnerWorkspaceScope
    current_module_id: str | None = None
    modules: list[LearnerModuleProgress] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LearnerEnrollmentSummary(BaseModel):
    id: str
    learner_id: str
    course_run_id: str
    course_title: str
    course_summary: str
    status: LearnerEnrollmentStatus
    module_count: int
    completed_module_count: int
    current_module_id: str | None = None
    current_module_title: str | None = None
    current_module_index: int | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_enrollment(cls, enrollment: LearnerEnrollment) -> "LearnerEnrollmentSummary":
        current_module = next(
            (module for module in enrollment.modules if module.module_id == enrollment.current_module_id),
            None,
        )
        return cls(
            id=enrollment.id,
            learner_id=enrollment.learner_id,
            course_run_id=enrollment.course_run_id,
            course_title=enrollment.course_title,
            course_summary=enrollment.course_summary,
            status=enrollment.status,
            module_count=len(enrollment.modules),
            completed_module_count=sum(1 for module in enrollment.modules if module.status == LearnerModuleStatus.passed),
            current_module_id=enrollment.current_module_id,
            current_module_title=current_module.title if current_module is not None else None,
            current_module_index=current_module.module_index if current_module is not None else None,
            created_at=enrollment.created_at,
            updated_at=enrollment.updated_at,
        )


class LearnerEnrollmentList(BaseModel):
    enrollments: list[LearnerEnrollmentSummary] = Field(default_factory=list)


class CreateEnrollmentRequest(BaseModel):
    course_run_id: str
    learner_id: str = "local-learner"


class LaunchWorkspaceRequest(BaseModel):
    module_id: str | None = None


class SubmitModuleRequest(BaseModel):
    module_id: str | None = None


class LearnerWorkspaceFileSummary(BaseModel):
    relative_path: str
    media_type: str
    size_bytes: int = Field(ge=0)


class LearnerWorkspaceFileList(BaseModel):
    enrollment_id: str
    module_id: str
    workspace_root: str
    files: list[LearnerWorkspaceFileSummary] = Field(default_factory=list)


class LearnerWorkspaceFileContent(BaseModel):
    enrollment_id: str
    module_id: str
    workspace_root: str
    relative_path: str
    media_type: str
    content: str


class WriteLearnerWorkspaceFileRequest(BaseModel):
    module_id: str | None = None
    relative_path: str
    content: str


class LearnerWorkspaceFileWriteResult(BaseModel):
    enrollment_id: str
    module_id: str
    workspace_root: str
    relative_path: str
    media_type: str
    size_bytes: int = Field(ge=0)


class LearnerModuleExperience(BaseModel):
    enrollment: LearnerEnrollmentSummary
    active_module: LearnerModuleProgress
    modules: list[LearnerModuleProgress] = Field(default_factory=list)
    submissions: list[LearnerSubmissionRecord] = Field(default_factory=list)
