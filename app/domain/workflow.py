from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.ai import AIUsageSummary
from app.domain.sandbox import SandboxExecutionResult
from app.domain.task_agent import TaskAgentServiceSpec
from app.services.assignment_design_inference import GenerationIntake

JsonObject = dict[str, Any]


class WorkflowStage(str, Enum):
    intake_review = "intake_review"
    awaiting_hil_gate_1 = "awaiting_hil_gate_1"
    awaiting_hil_gate_2 = "awaiting_hil_gate_2"
    awaiting_hil_gate_3 = "awaiting_hil_gate_3"
    needs_revision = "needs_revision"
    published = "published"
    blocked = "blocked"


class WorkflowStatus(str, Enum):
    active = "active"
    awaiting_human = "awaiting_human"
    published = "published"
    blocked = "blocked"


class HILGate(str, Enum):
    gate_1_spec_review = "gate_1_spec_review"
    gate_2_progression_review = "gate_2_progression_review"
    gate_3_pre_publish = "gate_3_pre_publish"


class DecisionOutcome(str, Enum):
    approve = "approve"
    reject = "reject"


class DraftKind(str, Enum):
    task_agent_spec = "task_agent_spec"
    scope_blocked = "scope_blocked"


class ArtifactVisibility(str, Enum):
    public = "public"
    private = "private"


class WorkflowNodeKind(str, Enum):
    authoring_runtime = "authoring_runtime"
    authoring_tests = "authoring_tests"
    authoring_repair = "authoring_repair"
    reviewer_runtime = "reviewer_runtime"
    reviewer_repair = "reviewer_repair"
    reviewer_code = "reviewer_code"
    reviewer_pedagogy = "reviewer_pedagogy"
    reviewer_tests = "reviewer_tests"
    reviewer_learner_runtime = "reviewer_learner_runtime"


class WorkflowNodeStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    blocked = "blocked"


class ReviewerFindingSeverity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class WorkflowFailureOwnerHint(str, Enum):
    authored_artifact = "authored_artifact"
    platform_runtime = "platform_runtime"
    ambiguous = "ambiguous"


class ReviewerFinding(BaseModel):
    category: str
    severity: ReviewerFindingSeverity
    title: str
    detail: str
    code: str | None = None
    location: str | None = None


class FailureContextValidationIssue(BaseModel):
    level: str
    code: str
    location: str
    message: str


class FailureContextDeliverableReport(BaseModel):
    deliverable_id: str
    compile_succeeded: bool
    runtime_succeeded: bool
    error: str | None = None
    stderr_excerpt: str | None = None


class FailureContextDependencyContract(BaseModel):
    deliverable_id: str
    starter_root: str | None = None
    implementation_language: str | None = None
    language_version: str | None = None
    application_framework: str | None = None
    framework_version: str | None = None
    package_manager: str | None = None
    container_image: str | None = None
    root_files: list[str] = Field(default_factory=list)
    expected_manifest_paths: list[str] = Field(default_factory=list)
    present_manifest_paths: list[str] = Field(default_factory=list)
    expected_lockfile_paths: list[str] = Field(default_factory=list)
    present_lockfile_paths: list[str] = Field(default_factory=list)
    expected_toolchain_paths: list[str] = Field(default_factory=list)
    present_toolchain_paths: list[str] = Field(default_factory=list)
    runtime_protocol_paths_present: list[str] = Field(default_factory=list)
    runtime_bundle_complete: bool = False


class FailureContextSandboxSummary(BaseModel):
    error: str | None = None
    build_stdout_excerpt: str | None = None
    build_stderr_excerpt: str | None = None
    run_stdout_excerpt: str | None = None
    run_stderr_excerpt: str | None = None
    failed_deliverables: list[str] = Field(default_factory=list)
    deliverable_reports: list[FailureContextDeliverableReport] = Field(default_factory=list)


class FailureContext(BaseModel):
    source_node_kind: WorkflowNodeKind
    source_node_attempt: int
    source_summary: str
    owner_hint: WorkflowFailureOwnerHint = WorkflowFailureOwnerHint.ambiguous
    failure_signature: str | None = None
    phase: str | None = None
    findings: list[ReviewerFinding] = Field(default_factory=list)
    validation_issues: list[FailureContextValidationIssue] = Field(default_factory=list)
    sandbox: FailureContextSandboxSummary | None = None
    dependency_contracts: list[FailureContextDependencyContract] = Field(default_factory=list)


class WorkflowNodeExecution(BaseModel):
    node_id: str
    kind: WorkflowNodeKind
    iteration: int = 1
    status: WorkflowNodeStatus
    attempt: int = 1
    summary: str
    created_at: datetime
    sandbox_result: SandboxExecutionResult | None = None
    findings: list[ReviewerFinding] = Field(default_factory=list)


class WorkflowLoopPolicy(BaseModel):
    max_authoring_attempts: int = Field(ge=1)
    max_reviewer_attempts: int = Field(ge=1)


class WorkflowLoopPhaseSummary(BaseModel):
    attempts_used: int = 0
    max_attempts: int = Field(ge=1)
    remaining_attempts: int = 0
    latest_node_kind: WorkflowNodeKind | None = None
    latest_status: WorkflowNodeStatus | None = None
    exhausted: bool = False
    passed: bool = False


class WorkflowReviewSummary(BaseModel):
    review_ready: bool = False
    blockers: list[str] = Field(default_factory=list)
    policy: WorkflowLoopPolicy
    authoring: WorkflowLoopPhaseSummary
    reviewer: WorkflowLoopPhaseSummary


class BundleFile(BaseModel):
    relative_path: str
    visibility: ArtifactVisibility
    media_type: str
    size_bytes: int
    role: str | None = None
    audience: str | None = None
    deliverable_id: str | None = None
    semantic_source: str | None = None


class MaterializedBundle(BaseModel):
    bundle_id: str
    generated_at: datetime
    root_dir: str
    public_dir: str
    private_dir: str
    manifest_path: str
    files: list[BundleFile] = Field(default_factory=list)


class WorkflowArtifacts(BaseModel):
    draft_kind: DraftKind
    task_agent_spec: TaskAgentServiceSpec | None = None
    ai_usage: AIUsageSummary = Field(default_factory=AIUsageSummary)
    validation_summary: JsonObject | None = None
    progression_preview: list[JsonObject] = Field(default_factory=list)
    artifact_plan: list[str] = Field(default_factory=list)
    origin_template: str | None = None
    workspace_snapshot: MaterializedBundle | None = None
    materialized_bundle: MaterializedBundle | None = None
    node_executions: list[WorkflowNodeExecution] = Field(default_factory=list)
    review_summary: WorkflowReviewSummary | None = None
    notes: list[str] = Field(default_factory=list)


class WorkflowRun(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    stage: WorkflowStage
    status: WorkflowStatus
    pending_gate: HILGate | None = None
    intake: GenerationIntake
    artifacts: WorkflowArtifacts
    notes: list[str] = Field(default_factory=list)


class WorkflowRunSummary(BaseModel):
    id: str
    title: str
    stage: WorkflowStage
    status: WorkflowStatus
    pending_gate: HILGate | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_run(cls, run: WorkflowRun) -> "WorkflowRunSummary":
        return cls(
            id=run.id,
            title=run.title,
            stage=run.stage,
            status=run.status,
            pending_gate=run.pending_gate,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class WorkflowEvent(BaseModel):
    run_id: str
    sequence_no: int
    event_type: str
    created_at: datetime
    payload: JsonObject = Field(default_factory=dict)


class WorkflowRunList(BaseModel):
    runs: list[WorkflowRunSummary]


class CreateWorkflowRunRequest(BaseModel):
    intake: GenerationIntake


class GateDecisionRequest(BaseModel):
    gate: HILGate
    decision: DecisionOutcome
    comment: str | None = None


class MaterializeBundleRequest(BaseModel):
    overwrite: bool = True


class BundleFileContent(BaseModel):
    relative_path: str
    media_type: str
    content: str
