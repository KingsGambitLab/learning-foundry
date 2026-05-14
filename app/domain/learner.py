from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field

from app.domain.course import CourseRunStatus, CourseRunSummary
from app.domain.grading import AssignmentGradeReport, DeliverableGradeReport
from app.domain.registry import PackageType


class LearnerEnrollmentStatus(str, Enum):
    active = "active"
    completed = "completed"
    blocked = "blocked"


class LearnerDeliverableStatus(str, Enum):
    available = "available"
    passed = "passed"


class LearnerWorkspaceScope(str, Enum):
    shared_course = "shared_course"


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
    deliverable_count: int = Field(validation_alias="deliverable_count")
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
        deliverable_count: int | None = None,
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
            deliverable_count=deliverable_count if deliverable_count is not None else run.deliverable_count,
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
    deliverable_id: str = Field(validation_alias="deliverable_id")
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
    submission_group_id: str | None = None
    enrollment_id: str
    deliverable_id: str = Field(validation_alias="deliverable_id")
    created_at: datetime
    status: str
    passed_tests: int
    total_tests: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    grade_report: DeliverableGradeReport
    assignment_report: AssignmentGradeReport | None = None


class LearnerDeliverableProgress(BaseModel):
    deliverable_id: str = Field(validation_alias="deliverable_id")
    title: str
    objective: str
    status: LearnerDeliverableStatus
    deliverable_index: int = Field(validation_alias="deliverable_index")
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
    current_deliverable_id: str | None = Field(
        default=None,
        validation_alias="current_deliverable_id",
    )
    deliverables: list[LearnerDeliverableProgress] = Field(
        default_factory=list,
        validation_alias="deliverables",
    )
    notes: list[str] = Field(default_factory=list)


class LearnerEnrollmentSummary(BaseModel):
    id: str
    learner_id: str
    course_run_id: str
    course_title: str
    course_summary: str
    status: LearnerEnrollmentStatus
    deliverable_count: int = Field(validation_alias="deliverable_count")
    completed_deliverable_count: int = Field(
        validation_alias="completed_deliverable_count",
    )
    current_deliverable_id: str | None = Field(
        default=None,
        validation_alias="current_deliverable_id",
    )
    current_deliverable_title: str | None = Field(
        default=None,
        validation_alias="current_deliverable_title",
    )
    current_deliverable_index: int | None = Field(
        default=None,
        validation_alias="current_deliverable_index",
    )
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_enrollment(cls, enrollment: LearnerEnrollment) -> "LearnerEnrollmentSummary":
        current_deliverable = next(
            (deliverable for deliverable in enrollment.deliverables if deliverable.deliverable_id == enrollment.current_deliverable_id),
            None,
        )
        return cls(
            id=enrollment.id,
            learner_id=enrollment.learner_id,
            course_run_id=enrollment.course_run_id,
            course_title=enrollment.course_title,
            course_summary=enrollment.course_summary,
            status=enrollment.status,
            deliverable_count=len(enrollment.deliverables),
            completed_deliverable_count=sum(
                1
                for deliverable in enrollment.deliverables
                if deliverable.status == LearnerDeliverableStatus.passed
            ),
            current_deliverable_id=enrollment.current_deliverable_id,
            current_deliverable_title=current_deliverable.title if current_deliverable is not None else None,
            current_deliverable_index=current_deliverable.deliverable_index if current_deliverable is not None else None,
            created_at=enrollment.created_at,
            updated_at=enrollment.updated_at,
        )


class LearnerEnrollmentList(BaseModel):
    enrollments: list[LearnerEnrollmentSummary] = Field(default_factory=list)


class CreateEnrollmentRequest(BaseModel):
    course_run_id: str
    learner_id: str = "local-learner"


class LaunchWorkspaceRequest(BaseModel):
    deliverable_id: str | None = Field(default=None, validation_alias="deliverable_id")


class SubmitDeliverableRequest(BaseModel):
    deliverable_id: str | None = Field(default=None, validation_alias="deliverable_id")


class LearnerWorkspaceFileSummary(BaseModel):
    relative_path: str
    media_type: str
    size_bytes: int = Field(ge=0)


class LearnerWorkspaceFileList(BaseModel):
    enrollment_id: str
    deliverable_id: str = Field(validation_alias="deliverable_id")
    workspace_root: str
    files: list[LearnerWorkspaceFileSummary] = Field(default_factory=list)


class LearnerWorkspaceFileContent(BaseModel):
    enrollment_id: str
    deliverable_id: str = Field(validation_alias="deliverable_id")
    workspace_root: str
    relative_path: str
    media_type: str
    content: str


class WriteLearnerWorkspaceFileRequest(BaseModel):
    deliverable_id: str | None = Field(default=None, validation_alias="deliverable_id")
    relative_path: str
    content: str


class LearnerWorkspaceFileWriteResult(BaseModel):
    enrollment_id: str
    deliverable_id: str = Field(validation_alias="deliverable_id")
    workspace_root: str
    relative_path: str
    media_type: str
    size_bytes: int = Field(ge=0)


class LearnerDeliverableExperience(BaseModel):
    enrollment: LearnerEnrollmentSummary
    project_brief_markdown: str = ""
    workspace_session: LearnerWorkspaceSession | None = None
    latest_assignment_report: AssignmentGradeReport | None = None
    latest_assignment_submission: LearnerSubmissionRecord | None = None
    active_deliverable: LearnerDeliverableProgress = Field(
        validation_alias="active_deliverable",
    )
    deliverables: list[LearnerDeliverableProgress] = Field(
        default_factory=list,
        validation_alias="deliverables",
    )
    submissions: list[LearnerSubmissionRecord] = Field(default_factory=list)
