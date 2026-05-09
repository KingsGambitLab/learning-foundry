from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.domain.course import CourseReviewReport, CourseRunSummary, CreatorCourseSetupChoices
from app.domain.learner import LearnerDeliverableExperience
from app.domain.publish import PublishedVersionList


class TestingDiagnosticSeverity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class TestingDiagnostic(BaseModel):
    code: str
    severity: TestingDiagnosticSeverity
    summary: str
    detail: str | None = None
    recommended_action: str | None = None
    blocking: bool = False
    context: dict = Field(default_factory=dict)


class CreateCreatorFeedbackRequest(BaseModel):
    summary: str = Field(min_length=3)
    details: str | None = None
    category: str = "general"
    rating: int | None = Field(default=None, ge=1, le=5)
    deliverable_slug: str | None = Field(default=None, validation_alias="deliverable_slug")
    workflow_run_id: str | None = None
    context: dict = Field(default_factory=dict)


class CreatorFeedbackRecord(BaseModel):
    id: str
    course_run_id: str
    created_at: datetime
    category: str
    summary: str
    details: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    deliverable_slug: str | None = Field(default=None, validation_alias="deliverable_slug")
    workflow_run_id: str | None = None
    stage: str | None = None
    status: str | None = None
    context: dict = Field(default_factory=dict)


class CreatorFeedbackList(BaseModel):
    items: list[CreatorFeedbackRecord] = Field(default_factory=list)


class CreateLearnerFeedbackRequest(BaseModel):
    summary: str = Field(min_length=3)
    details: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    deliverable_id: str | None = Field(default=None, validation_alias="deliverable_id")
    context: dict = Field(default_factory=dict)


class LearnerFeedbackRecord(BaseModel):
    id: str
    enrollment_id: str
    course_run_id: str
    publish_snapshot_id: str
    learner_id: str
    created_at: datetime
    summary: str
    details: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    deliverable_id: str | None = Field(default=None, validation_alias="deliverable_id")
    context: dict = Field(default_factory=dict)


class LearnerFeedbackList(BaseModel):
    items: list[LearnerFeedbackRecord] = Field(default_factory=list)


class LearnerEvalSubmissionSummary(BaseModel):
    status: str
    passed_tests: int = Field(ge=0)
    total_tests: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)


class LearnerDeliverableEvaluationResult(BaseModel):
    deliverable_id: str = Field(validation_alias="deliverable_id")
    title: str
    deliverable_index: int = Field(
        ge=1,
        validation_alias="deliverable_index",
    )
    learner_visible_files: list[str] = Field(default_factory=list)
    bad_attempt: LearnerEvalSubmissionSummary
    good_attempt: LearnerEvalSubmissionSummary
    next_deliverable_id: str | None = Field(
        default=None,
        validation_alias="next_deliverable_id",
    )
    progression_observed: bool = False
    course_completed: bool = False
    notes: list[str] = Field(default_factory=list)


class CreateLearnerEvaluationReportRequest(BaseModel):
    publish_snapshot_id: str | None = None
    learner_id: str | None = None
    enrollment_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    deliverable_results: list[LearnerDeliverableEvaluationResult] = Field(
        default_factory=list,
        min_length=1,
        validation_alias="deliverable_results",
    )


class LearnerCourseEvaluationReport(BaseModel):
    id: str
    course_run_id: str
    publish_snapshot_id: str
    learner_id: str | None = None
    enrollment_id: str | None = None
    created_at: datetime
    overall_status: str
    notes: list[str] = Field(default_factory=list)
    deliverable_results: list[LearnerDeliverableEvaluationResult] = Field(
        default_factory=list,
        validation_alias="deliverable_results",
    )


class CreatorTestingView(BaseModel):
    course_run: CourseRunSummary
    review: CourseReviewReport
    goal: str | None = None
    requested_learning_outcomes: list[str] = Field(default_factory=list)
    creator_choices: CreatorCourseSetupChoices | None = None
    published_versions: PublishedVersionList
    diagnostics: list[TestingDiagnostic] = Field(default_factory=list)
    creator_feedback: list[CreatorFeedbackRecord] = Field(default_factory=list)
    latest_learner_evaluation: LearnerCourseEvaluationReport | None = None


class LearnerTestingView(BaseModel):
    experience: LearnerDeliverableExperience
    diagnostics: list[TestingDiagnostic] = Field(default_factory=list)
    feedback: list[LearnerFeedbackRecord] = Field(default_factory=list)
