from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from app.domain.task_agent import TaskAgentServiceSpec


JsonObject = dict[str, Any]


class ToolCallStatus(str, Enum):
    ok = "ok"
    timeout = "timeout"
    error = "error"
    skipped = "skipped"
    deduplicated = "deduplicated"
    preview = "preview"


class ToolCallRecord(BaseModel):
    order: int = Field(ge=0)
    tool_id: str
    args: JsonObject = Field(default_factory=dict)
    status: ToolCallStatus
    idempotency_key: str | None = None
    deduplicated: bool = False
    approval_id: str | None = None


class ApprovalRecord(BaseModel):
    approval_id: str
    order: int = Field(ge=0)
    tool_id: str
    approved: bool = True


class EscalationRecord(BaseModel):
    order: int = Field(ge=0)
    reason: str


class FailureInjectionRecord(BaseModel):
    target: str
    target_id: str
    failure_mode: str


class FallbackActionRecord(BaseModel):
    trigger: str
    action: str
    target_id: str | None = None


class EvalRunEvidence(BaseModel):
    run_id: str
    case_id: str
    dry_run: bool = False
    output: JsonObject = Field(default_factory=dict)
    trace_events: list[str] = Field(default_factory=list)
    step_count: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    escalations: list[EscalationRecord] = Field(default_factory=list)
    failure_injections: list[FailureInjectionRecord] = Field(default_factory=list)
    fallback_actions: list[FallbackActionRecord] = Field(default_factory=list)
    resumed_after_pause: bool = False
    success: bool
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class TaskAgentSubmission(BaseModel):
    submission_id: str
    runs: list[EvalRunEvidence] = Field(min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class GradeTaskAgentRequest(BaseModel):
    spec: TaskAgentServiceSpec
    submission: TaskAgentSubmission


class LiveGradeTaskAgentRequest(BaseModel):
    base_url: str
    workspace_root: str | None = None
    timeout_ms: int = Field(default=10_000, ge=100, le=120_000)
    poll_interval_ms: int = Field(default=50, ge=0, le=5_000)
    max_poll_attempts: int = Field(default=25, ge=1, le=500)
    auto_approve: bool = True
    include_dry_runs: bool = True


class LiveGradeTaskAgentSpecRequest(BaseModel):
    spec: TaskAgentServiceSpec
    live: LiveGradeTaskAgentRequest


class LiveTaskAgentGradeReport(BaseModel):
    base_url: str
    submission: TaskAgentSubmission
    grade_report: "DeliverableGradeReport"


class GradeStatus(str, Enum):
    passed = "passed"
    failed = "failed"


class TestGradeResult(BaseModel):
    test_id: str
    test_type: str
    kind: str
    status: GradeStatus
    score: float = Field(ge=0.0, le=1.0)
    summary: str
    diagnostics: list[str] = Field(default_factory=list)
    # Worked example for FAILED scenarios so a learner can fix in one
    # pass instead of guessing against hidden inputs. All optional /
    # defaulted: reports serialized before this field still validate.
    failing_rubric: str | None = None
    example_question: str | None = None
    example_expected: str | None = None
    example_actual: str | None = None


class LearnerReviewGuidance(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    fundamental_gap: str = ""
    why_it_matters: list[str] = Field(default_factory=list)
    likely_root_cause: list[str] = Field(default_factory=list)
    investigation_steps: list[str] = Field(default_factory=list)
    learner_feedback: str = ""


class DeliverableGradeReport(BaseModel):
    deliverable_id: str = Field(validation_alias="deliverable_id")
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    status: GradeStatus
    results: list[TestGradeResult] = Field(default_factory=list)
    submission_warnings: list[str] = Field(default_factory=list)


class ReviewAreaGradeReport(BaseModel):
    deliverable_id: str = Field(validation_alias="deliverable_id")
    title: str
    objective: str
    deliverable_index: int = Field(
        ge=1,
        validation_alias="deliverable_index",
    )
    grade_report: DeliverableGradeReport
    feedback: LearnerReviewGuidance | None = None


class AssignmentGradeReport(BaseModel):
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    status: GradeStatus
    review_areas: list[ReviewAreaGradeReport] = Field(default_factory=list)
    submission_warnings: list[str] = Field(default_factory=list)


class LiveAssignmentGradeReport(BaseModel):
    base_url: str
    submission: TaskAgentSubmission
    assignment_report: AssignmentGradeReport


LiveTaskAgentGradeReport.model_rebuild()
LiveAssignmentGradeReport.model_rebuild()
