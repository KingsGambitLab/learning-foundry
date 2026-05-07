from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.course import CourseReviewReport, CourseRunSummary
from app.domain.learner import LearnerModuleExperience
from app.domain.publish import PublishedVersionList


class CreateCreatorFeedbackRequest(BaseModel):
    summary: str = Field(min_length=3)
    details: str | None = None
    category: str = "general"
    rating: int | None = Field(default=None, ge=1, le=5)
    module_slug: str | None = None
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
    module_slug: str | None = None
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
    module_id: str | None = None
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
    module_id: str | None = None
    context: dict = Field(default_factory=dict)


class LearnerFeedbackList(BaseModel):
    items: list[LearnerFeedbackRecord] = Field(default_factory=list)


class LearnerEvalSubmissionSummary(BaseModel):
    status: str
    passed_tests: int = Field(ge=0)
    total_tests: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)


class LearnerModuleEvaluationResult(BaseModel):
    module_id: str
    title: str
    module_index: int = Field(ge=1)
    learner_visible_files: list[str] = Field(default_factory=list)
    bad_attempt: LearnerEvalSubmissionSummary
    good_attempt: LearnerEvalSubmissionSummary
    next_module_id: str | None = None
    progression_observed: bool = False
    course_completed: bool = False
    notes: list[str] = Field(default_factory=list)


class CreateLearnerEvaluationReportRequest(BaseModel):
    publish_snapshot_id: str | None = None
    learner_id: str | None = None
    enrollment_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    module_results: list[LearnerModuleEvaluationResult] = Field(default_factory=list, min_length=1)


class LearnerCourseEvaluationReport(BaseModel):
    id: str
    course_run_id: str
    publish_snapshot_id: str
    learner_id: str | None = None
    enrollment_id: str | None = None
    created_at: datetime
    overall_status: str
    notes: list[str] = Field(default_factory=list)
    module_results: list[LearnerModuleEvaluationResult] = Field(default_factory=list)


class CreatorTestingView(BaseModel):
    course_run: CourseRunSummary
    review: CourseReviewReport
    published_versions: PublishedVersionList
    creator_feedback: list[CreatorFeedbackRecord] = Field(default_factory=list)
    latest_learner_evaluation: LearnerCourseEvaluationReport | None = None


class LearnerTestingView(BaseModel):
    experience: LearnerModuleExperience
    feedback: list[LearnerFeedbackRecord] = Field(default_factory=list)
