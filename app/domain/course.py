from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field

from app.domain.ai import AIUsageSummary
from app.domain.registry import PackageType, StarterType
from app.domain.task_agent import AssignmentDesignSpec, DataSourceSpec
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
    deliverable_slug: str = Field(validation_alias=AliasChoices("deliverable_slug", "module_slug"))
    title: str
    summary: str
    learning_outcomes: list[str] = Field(default_factory=list)
    design_spec: AssignmentDesignSpec | None = None
    domain_pack: str | None = None
    overlays: list[str] = Field(default_factory=list)
    workflow_run_id: str | None = None
    workflow_stage: str | None = None
    workflow_status: str | None = None
    draft_kind: str | None = None
    recommendation_status: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def module_slug(self) -> str:
        return self.deliverable_slug

    @module_slug.setter
    def module_slug(self, value: str) -> None:
        self.deliverable_slug = value


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
    deliverables: list[CourseModuleDraft] = Field(default_factory=list, validation_alias=AliasChoices("deliverables", "modules"))
    notes: list[str] = Field(default_factory=list)
    goal: str | None = None
    requested_learning_outcomes: list[str] = Field(default_factory=list)
    generated_plan: GeneratedCoursePlan | None = None
    generation_source: CourseGenerationSource | None = None
    generation_status: CourseGenerationStatus | None = None
    own_ai_usage: AIUsageSummary = Field(default_factory=AIUsageSummary)
    ai_usage: AIUsageSummary = Field(default_factory=AIUsageSummary)
    last_error: str | None = None

    @property
    def modules(self) -> list[CourseModuleDraft]:
        return self.deliverables

    @modules.setter
    def modules(self, value: list[CourseModuleDraft]) -> None:
        self.deliverables = value


class CourseRunSummary(BaseModel):
    id: str
    course_family_id: str
    title: str
    summary: str
    goal: str | None = None
    package_type: PackageType
    stage: CourseRunStage
    status: CourseRunStatus
    deliverable_count: int = Field(validation_alias=AliasChoices("deliverable_count", "module_count"))
    shared_workflow_run_id: str | None = None
    latest_publish_snapshot_id: str | None = None
    active_operation: CourseAsyncOperation | None = None
    ai_usage: AIUsageSummary = Field(default_factory=AIUsageSummary)
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_run(cls, run: CourseRun) -> "CourseRunSummary":
        return cls(
            id=run.id,
            course_family_id=run.course_family_id,
            title=run.title,
            summary=run.summary,
            goal=run.goal,
            package_type=run.package_type,
            stage=run.stage,
            status=run.status,
            deliverable_count=len(run.deliverables),
            shared_workflow_run_id=run.shared_workflow_run_id,
            latest_publish_snapshot_id=run.latest_publish_snapshot_id,
            active_operation=run.active_operation,
            ai_usage=run.ai_usage,
            last_error=run.last_error,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )

    @property
    def module_count(self) -> int:
        return self.deliverable_count


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
    deleted_creator_assets: int = 0
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
    deliverable_slug: str = Field(validation_alias=AliasChoices("deliverable_slug", "module_slug"))
    title: str
    summary: str
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

    @property
    def module_slug(self) -> str:
        return self.deliverable_slug

    @module_slug.setter
    def module_slug(self, value: str) -> None:
        self.deliverable_slug = value


class CourseReviewCounts(BaseModel):
    total_deliverables: int = Field(validation_alias=AliasChoices("total_deliverables", "total_modules"))
    ready_deliverables: int = Field(validation_alias=AliasChoices("ready_deliverables", "ready_modules"))
    deliverables_with_blockers: int = Field(validation_alias=AliasChoices("deliverables_with_blockers", "modules_with_blockers"))
    deliverables_with_bundle: int = Field(validation_alias=AliasChoices("deliverables_with_bundle", "modules_with_bundle"))
    linked_workflow_runs: int
    published_workflow_runs: int
    workflow_runs_with_bundle: int

    @property
    def total_modules(self) -> int:
        return self.total_deliverables

    @property
    def ready_modules(self) -> int:
        return self.ready_deliverables

    @property
    def modules_with_blockers(self) -> int:
        return self.deliverables_with_blockers

    @property
    def modules_with_bundle(self) -> int:
        return self.deliverables_with_bundle


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
    deliverables: list[CourseModuleReview] = Field(default_factory=list, validation_alias=AliasChoices("deliverables", "modules"))

    @property
    def modules(self) -> list[CourseModuleReview]:
        return self.deliverables

    @modules.setter
    def modules(self, value: list[CourseModuleReview]) -> None:
        self.deliverables = value


class CreateCourseModuleRequest(BaseModel):
    deliverable_slug: str | None = Field(default=None, validation_alias=AliasChoices("deliverable_slug", "module_slug"))
    title: str
    summary: str | None = None
    learning_outcomes: list[str] = Field(default_factory=list)
    design_spec: AssignmentDesignSpec | None = None
    domain_pack_hint: str | None = None
    overlays_hint: list[str] = Field(default_factory=list)

    @property
    def module_slug(self) -> str | None:
        return self.deliverable_slug

    @module_slug.setter
    def module_slug(self, value: str | None) -> None:
        self.deliverable_slug = value


class GeneratedCoursePlan(BaseModel):
    title: str
    summary: str
    package_type: PackageType
    shared_design_spec: AssignmentDesignSpec | None = None
    deliverables: list[CreateCourseModuleRequest] = Field(default_factory=list, min_length=1, validation_alias=AliasChoices("deliverables", "modules"))
    notes: list[str] = Field(default_factory=list)

    @property
    def modules(self) -> list[CreateCourseModuleRequest]:
        return self.deliverables

    @modules.setter
    def modules(self, value: list[CreateCourseModuleRequest]) -> None:
        self.deliverables = value


class CreateCourseRunRequest(BaseModel):
    pattern_slug: str | None = None
    title: str | None = None
    summary: str | None = None
    package_type: PackageType | None = None
    shared_design_spec: AssignmentDesignSpec | None = None
    course_family_id: str | None = None
    deliverables: list[CreateCourseModuleRequest] = Field(default_factory=list, validation_alias=AliasChoices("deliverables", "modules"))

    @property
    def modules(self) -> list[CreateCourseModuleRequest]:
        return self.deliverables

    @modules.setter
    def modules(self, value: list[CreateCourseModuleRequest]) -> None:
        self.deliverables = value


class CreatorCourseSetupInput(BaseModel):
    starter_type: StarterType | None = None
    primary_database: str | None = None
    cache_backend: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    data_sources: list[DataSourceSpec] = Field(default_factory=list)


class CreatorCourseSetupChoices(BaseModel):
    starter_type: StarterType = StarterType.partial_implementation
    primary_database: str | None = None
    cache_backend: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    data_sources: list[DataSourceSpec] = Field(default_factory=list)


class GenerateCourseFromBriefRequest(BaseModel):
    goal: str = Field(min_length=10)
    learning_outcomes: list[str] = Field(default_factory=list, max_length=10)
    title: str | None = None
    package_type_hint: PackageType | None = None
    creator_setup: CreatorCourseSetupInput = Field(default_factory=CreatorCourseSetupInput)


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


class CreatorCourseModulePlan(BaseModel):
    deliverable_slug: str = Field(validation_alias=AliasChoices("deliverable_slug", "module_slug"))
    title: str
    summary: str
    learning_outcomes: list[str] = Field(default_factory=list)
    creator_notes: list[str] = Field(default_factory=list)
    design_spec: AssignmentDesignSpec | None = None

    @property
    def module_slug(self) -> str:
        return self.deliverable_slug

    @module_slug.setter
    def module_slug(self, value: str) -> None:
        self.deliverable_slug = value


class CreatorCoursePlan(BaseModel):
    goal: str | None = None
    learning_outcomes: list[str] = Field(default_factory=list)
    title: str
    summary: str
    package_type: PackageType
    creator_choices: CreatorCourseSetupChoices
    shared_design_spec: AssignmentDesignSpec | None = None
    deliverables: list[CreatorCourseModulePlan] = Field(default_factory=list, min_length=1, validation_alias=AliasChoices("deliverables", "modules"))
    creator_summary: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def modules(self) -> list[CreatorCourseModulePlan]:
        return self.deliverables

    @modules.setter
    def modules(self, value: list[CreatorCourseModulePlan]) -> None:
        self.deliverables = value


class GenerateCreatorCoursePlanRequest(BaseModel):
    goal: str = Field(min_length=10)
    learning_outcomes: list[str] = Field(default_factory=list)
    title: str | None = None
    package_type_hint: PackageType | None = None
    creator_choices: CreatorCourseSetupInput = Field(default_factory=CreatorCourseSetupInput)


class GenerateCreatorCoursePlanResponse(BaseModel):
    source: CourseGenerationSource
    status: CourseGenerationStatus
    learning_outcomes: list[str] = Field(default_factory=list)
    plan: CreatorCoursePlan


class CreateCourseFromCreatorPlanRequest(BaseModel):
    plan: CreatorCoursePlan
