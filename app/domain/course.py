from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.domain.registry import PackageType, StarterType
from app.domain.task_agent import AssignmentDesignSpec
from app.domain.workflow import DraftKind, HILGate, MaterializedBundle, WorkflowReviewSummary, WorkflowStage, WorkflowStatus


class CourseRunStage(str, Enum):
    drafting = "drafting"
    awaiting_course_review = "awaiting_course_review"
    ready_to_publish = "ready_to_publish"
    published = "published"
    blocked = "blocked"


class CourseRunStatus(str, Enum):
    active = "active"
    awaiting_human = "awaiting_human"
    published = "published"
    blocked = "blocked"


class CourseModuleDraft(BaseModel):
    module_slug: str
    title: str
    summary: str
    learning_outcomes: list[str] = Field(default_factory=list)
    checkpoint_module_ids: list[str] = Field(default_factory=list)
    design_spec: AssignmentDesignSpec | None = None
    domain_pack: str | None = None
    overlays: list[str] = Field(default_factory=list)
    workflow_run_id: str | None = None
    workflow_stage: str | None = None
    workflow_status: str | None = None
    draft_kind: str | None = None
    recommendation_status: str | None = None
    notes: list[str] = Field(default_factory=list)


class CourseRun(BaseModel):
    id: str
    course_family_id: str
    title: str
    summary: str
    package_type: PackageType
    pattern_slug: str | None = None
    shared_design_spec: AssignmentDesignSpec | None = None
    shared_workflow_run_id: str | None = None
    latest_publish_snapshot_id: str | None = None
    active_operation: CourseAsyncOperation | None = None
    created_at: datetime
    updated_at: datetime
    stage: CourseRunStage
    status: CourseRunStatus
    materialized_bundle: MaterializedBundle | None = None
    modules: list[CourseModuleDraft] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    goal: str | None = None
    requested_learning_outcomes: list[str] = Field(default_factory=list)
    generated_plan: GeneratedCoursePlan | None = None
    generation_source: CourseGenerationSource | None = None
    generation_status: CourseGenerationStatus | None = None
    last_error: str | None = None


class CourseRunSummary(BaseModel):
    id: str
    course_family_id: str
    title: str
    package_type: PackageType
    stage: CourseRunStage
    status: CourseRunStatus
    module_count: int
    shared_workflow_run_id: str | None = None
    latest_publish_snapshot_id: str | None = None
    active_operation: CourseAsyncOperation | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_run(cls, run: CourseRun) -> "CourseRunSummary":
        return cls(
            id=run.id,
            course_family_id=run.course_family_id,
            title=run.title,
            package_type=run.package_type,
            stage=run.stage,
            status=run.status,
            module_count=len(run.modules),
            shared_workflow_run_id=run.shared_workflow_run_id,
            latest_publish_snapshot_id=run.latest_publish_snapshot_id,
            active_operation=run.active_operation,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class CourseEvent(BaseModel):
    course_run_id: str
    sequence_no: int
    event_type: str
    created_at: datetime
    payload: dict = Field(default_factory=dict)


class CourseRunList(BaseModel):
    runs: list[CourseRunSummary]


class LocalDraftResetResult(BaseModel):
    deleted_course_runs: int
    deleted_course_events: int
    deleted_workflow_runs: int
    deleted_workflow_events: int
    deleted_publish_snapshots: int = 0
    deleted_learner_enrollments: int = 0
    deleted_learner_submissions: int = 0
    deleted_learner_workspace_sessions: int = 0
    deleted_creator_feedback: int = 0
    deleted_learner_feedback: int = 0
    deleted_learner_eval_reports: int = 0
    cleared_directories: list[str] = Field(default_factory=list)


class CourseGenerationSource(str, Enum):
    openai_live = "openai_live"
    deterministic_fallback = "deterministic_fallback"


class CourseAsyncOperation(str, Enum):
    generation = "generation"
    revision = "revision"
    materialize = "materialize"
    publish = "publish"


class CourseGenerationStatus(BaseModel):
    provider: str
    available: bool
    source: CourseGenerationSource
    message: str
    sdk_installed: bool = False
    api_key_present: bool = False
    model_id: str | None = None
    env_file: str | None = None


class CourseLinkedBundleSummary(BaseModel):
    bundle_id: str
    root_dir: str
    public_dir: str
    manifest_path: str
    total_file_count: int
    public_files: list[str] = Field(default_factory=list)
    private_file_count: int = 0


class CourseLinkedWorkflowSummary(BaseModel):
    run_id: str
    title: str
    stage: WorkflowStage
    status: WorkflowStatus
    pending_gate: HILGate | None = None
    draft_kind: DraftKind
    bundle: CourseLinkedBundleSummary | None = None
    review_summary: WorkflowReviewSummary | None = None


class CourseModuleReview(BaseModel):
    position: int
    module_slug: str
    title: str
    summary: str
    checkpoint_module_ids: list[str] = Field(default_factory=list)
    design_spec: AssignmentDesignSpec | None = None
    domain_pack: str | None = None
    overlays: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)
    workflow_run_id: str | None = None
    workflow_stage: str | None = None
    workflow_status: str | None = None
    recommendation_status: str | None = None
    ready_for_publish: bool = False
    bundle_available: bool = False
    blockers: list[str] = Field(default_factory=list)
    linked_workflow: CourseLinkedWorkflowSummary | None = None
    notes: list[str] = Field(default_factory=list)


class CourseReviewCounts(BaseModel):
    total_modules: int
    ready_modules: int
    modules_with_blockers: int
    modules_with_bundle: int
    linked_workflow_runs: int
    published_workflow_runs: int
    workflow_runs_with_bundle: int


class CourseReviewReport(BaseModel):
    course_run_id: str
    title: str
    package_type: PackageType
    stage: CourseRunStage
    status: CourseRunStatus
    shared_design_spec: AssignmentDesignSpec | None = None
    shared_workflow_run_id: str | None = None
    materialized_bundle: MaterializedBundle | None = None
    counts: CourseReviewCounts
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    linked_workflows: list[CourseLinkedWorkflowSummary] = Field(default_factory=list)
    modules: list[CourseModuleReview] = Field(default_factory=list)


class CreateCourseModuleRequest(BaseModel):
    module_slug: str | None = None
    title: str
    summary: str | None = None
    learning_outcomes: list[str] = Field(default_factory=list)
    checkpoint_module_ids: list[str] = Field(default_factory=list)
    design_spec: AssignmentDesignSpec | None = None
    domain_pack_hint: str | None = None
    overlays_hint: list[str] = Field(default_factory=list)


class GeneratedCoursePlan(BaseModel):
    title: str
    summary: str
    package_type: PackageType
    shared_design_spec: AssignmentDesignSpec | None = None
    modules: list[CreateCourseModuleRequest] = Field(default_factory=list, min_length=1)
    notes: list[str] = Field(default_factory=list)


class CreateCourseRunRequest(BaseModel):
    pattern_slug: str | None = None
    title: str | None = None
    summary: str | None = None
    package_type: PackageType | None = None
    shared_design_spec: AssignmentDesignSpec | None = None
    course_family_id: str | None = None
    modules: list[CreateCourseModuleRequest] = Field(default_factory=list)


class GenerateCourseFromBriefRequest(BaseModel):
    goal: str = Field(min_length=10)
    learning_outcomes: list[str] = Field(default_factory=list, min_length=1, max_length=10)
    title: str | None = None
    package_type_hint: PackageType | None = None


class GenerateCourseFromBriefResponse(BaseModel):
    source: CourseGenerationSource
    status: CourseGenerationStatus
    plan: GeneratedCoursePlan
    course_run: CourseRun
    review: CourseReviewReport


class QueueCourseGenerationResponse(BaseModel):
    queued: bool = True
    status: CourseGenerationStatus
    course_run: CourseRun


class QueueCourseRevisionResponse(BaseModel):
    queued: bool = True
    course_run: CourseRun


class QueueCourseOperationResponse(BaseModel):
    queued: bool = True
    operation: CourseAsyncOperation
    course_run: CourseRun


class SuggestLearningOutcomesRequest(BaseModel):
    goal: str = Field(min_length=10)
    title: str | None = None


class SuggestLearningOutcomesResponse(BaseModel):
    source: CourseGenerationSource
    status: CourseGenerationStatus
    learning_outcomes: list[str] = Field(default_factory=list)


class CreatorCourseSetupChoices(BaseModel):
    starter_type: StarterType = StarterType.partial_implementation
    primary_database: str | None = None
    cache_backend: str | None = None
    tech_stack: list[str] = Field(default_factory=list)


class CreatorCourseModulePlan(BaseModel):
    module_slug: str
    title: str
    summary: str
    learning_outcomes: list[str] = Field(default_factory=list)
    creator_notes: list[str] = Field(default_factory=list)
    checkpoint_module_ids: list[str] = Field(default_factory=list)
    design_spec: AssignmentDesignSpec | None = None


class CreatorCoursePlan(BaseModel):
    title: str
    summary: str
    package_type: PackageType
    creator_choices: CreatorCourseSetupChoices
    shared_design_spec: AssignmentDesignSpec | None = None
    modules: list[CreatorCourseModulePlan] = Field(default_factory=list, min_length=1)
    creator_summary: str | None = None
    notes: list[str] = Field(default_factory=list)


class GenerateCreatorCoursePlanRequest(BaseModel):
    goal: str = Field(min_length=10)
    learning_outcomes: list[str] = Field(default_factory=list)
    title: str | None = None
    package_type_hint: PackageType | None = None
    creator_choices: CreatorCourseSetupChoices = Field(default_factory=CreatorCourseSetupChoices)


class GenerateCreatorCoursePlanResponse(BaseModel):
    source: CourseGenerationSource
    status: CourseGenerationStatus
    learning_outcomes: list[str] = Field(default_factory=list)
    plan: CreatorCoursePlan


class CreateCourseFromCreatorPlanRequest(BaseModel):
    plan: CreatorCoursePlan
