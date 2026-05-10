from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.domain.registry import PackageType, RiskClass, StarterType

JsonObject = dict[str, Any]
JsonSchema = dict[str, Any]


class EndpointSpec(BaseModel):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    required: bool = True


class WorkspaceScope(str, Enum):
    shared_course_workspace = "shared_course_workspace"
    per_deliverable_workspace = "per_deliverable_workspace"


class ProgressionMode(str, Enum):
    cumulative_deliverable_gates = "cumulative_deliverable_gates"
    independent_deliverables = "independent_deliverables"


class ExecutionSurface(str, Enum):
    http_service = "http_service"
    cli = "cli"
    protocol_server = "protocol_server"


class DataSourceKind(str, Enum):
    uploaded_file = "uploaded_file"
    corpus_bundle = "corpus_bundle"
    seed_database = "seed_database"
    mock_api = "mock_api"
    object_store = "object_store"


class DataSourcePurpose(str, Enum):
    retrieval = "retrieval"
    reference_data = "reference_data"
    seed_state = "seed_state"
    external_mock = "external_mock"


class RetrievalMode(str, Enum):
    none = "none"
    ranked_results = "ranked_results"
    grounded_answers = "grounded_answers"


class ProjectFamily(str, Enum):
    generic_backend_service = "generic_backend_service"
    workflow_agent_service = "workflow_agent_service"
    ranked_retrieval_service = "ranked_retrieval_service"
    grounded_retrieval_service = "grounded_retrieval_service"
    transactional_stateful_service = "transactional_stateful_service"
    control_plane_service = "control_plane_service"


class CourseStructureSpec(BaseModel):
    package_type: PackageType
    workspace_scope: WorkspaceScope
    progression_mode: ProgressionMode
    shared_codebase: bool = True


class DataSourceSpec(BaseModel):
    id: str
    kind: DataSourceKind
    title: str
    purpose: DataSourcePurpose = DataSourcePurpose.reference_data
    learner_visible: bool = True
    format: str | None = None
    workspace_path: str | None = None
    asset_id: str | None = None
    description: str | None = None


class RuntimeDependencySpec(BaseModel):
    execution_surface: ExecutionSurface
    starter_type: StarterType = StarterType.partial_implementation
    implementation_language: str | None = None
    language_version: str | None = None
    application_framework: str | None = None
    framework_version: str | None = None
    package_manager: str | None = None
    editable_files: list[str] = Field(default_factory=list)
    visible_fixture_files: list[str] = Field(default_factory=list)
    data_sources: list[DataSourceSpec] = Field(default_factory=list)
    primary_database: str | None = None
    primary_database_version: str | None = None
    cache_backend: str | None = None
    cache_backend_version: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    local_run_command: str | None = None
    visible_check_command: str | None = None
    preview_command: str | None = None


class CapabilitySpec(BaseModel):
    retrieval_mode: RetrievalMode = RetrievalMode.none
    answer_synthesis_required: bool = False
    citations_required: bool = False
    abstention_required: bool = False
    tool_use_required: bool = False
    traceability_required: bool = False
    durable_state_required: bool = False
    approval_flow_required: bool = False

    def summary_labels(self) -> list[str]:
        labels: list[str] = []
        if self.retrieval_mode == RetrievalMode.ranked_results:
            labels.append("ranked retrieval")
        elif self.retrieval_mode == RetrievalMode.grounded_answers:
            labels.append("grounded retrieval")
        if self.answer_synthesis_required:
            labels.append("answer synthesis")
        if self.citations_required:
            labels.append("citations")
        if self.abstention_required:
            labels.append("abstention")
        if self.tool_use_required:
            labels.append("tool use")
        if self.traceability_required:
            labels.append("traceability")
        if self.durable_state_required:
            labels.append("durable state")
        if self.approval_flow_required:
            labels.append("approval flow")
        return labels or ["general service workflow"]

    @property
    def is_grounded_answer_system(self) -> bool:
        return self.retrieval_mode == RetrievalMode.grounded_answers or (
            self.answer_synthesis_required and self.citations_required
        )


class AssessmentStrategySpec(BaseModel):
    public_checks_required: bool = True
    hidden_grader_required: bool = True
    cumulative_deliverable_gates: bool = True
    learner_submission_enabled: bool = True


class ProjectServiceBinding(BaseModel):
    service_id: str
    role: str
    technology: str | None = None
    learner_managed: bool = False


class ProjectRuntimeBindingSpec(BaseModel):
    implementation_language: str | None = None
    application_framework: str | None = None
    backing_services: list[ProjectServiceBinding] = Field(default_factory=list)
    seed_artifacts: list[str] = Field(default_factory=list)
    integration_points: list[str] = Field(default_factory=list)


class ProjectRuntimeCommandSpec(BaseModel):
    phase: Literal["install", "build", "seed", "run", "check", "verify"]
    command: str
    target_service_id: str | None = None
    notes: str | None = None


class ProjectRuntimeServiceSpec(BaseModel):
    service_id: str
    role: str
    technology: str | None = None
    version_hint: str | None = None
    package_manager: str | None = None
    entrypoint_path: str | None = None
    container_image: str | None = None
    learner_managed: bool = False
    run_command: str | None = None
    healthcheck_path: str | None = None
    default_port: int | None = None


class ProjectRuntimePlanSpec(BaseModel):
    implementation_language: str | None = None
    language_version: str | None = None
    application_framework: str | None = None
    framework_version: str | None = None
    package_manager: str | None = None
    services: list[ProjectRuntimeServiceSpec] = Field(default_factory=list)
    setup_steps: list[ProjectRuntimeCommandSpec] = Field(default_factory=list)
    seed_steps: list[ProjectRuntimeCommandSpec] = Field(default_factory=list)
    verify_steps: list[ProjectRuntimeCommandSpec] = Field(default_factory=list)
    run_steps: list[ProjectRuntimeCommandSpec] = Field(default_factory=list)
    check_steps: list[ProjectRuntimeCommandSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProjectContractSpec(BaseModel):
    family: ProjectFamily = ProjectFamily.generic_backend_service
    system_kind: str = "General backend service"
    core_entities: list[str] = Field(default_factory=list)
    primary_read_paths: list[str] = Field(default_factory=list)
    primary_write_paths: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)
    operational_concerns: list[str] = Field(default_factory=list)
    runtime_binding: ProjectRuntimeBindingSpec = Field(default_factory=ProjectRuntimeBindingSpec)
    runtime_plan: ProjectRuntimePlanSpec = Field(default_factory=ProjectRuntimePlanSpec)


def default_project_contract() -> ProjectContractSpec:
    return ProjectContractSpec(
        family=ProjectFamily.generic_backend_service,
        system_kind="General backend service",
        invariants=["The service returns a stable response contract for supported requests."],
        operational_concerns=["Keep failures observable enough to debug quickly."],
    )


class AssignmentDesignSpec(BaseModel):
    course_structure: CourseStructureSpec
    runtime_dependencies: RuntimeDependencySpec
    capabilities: CapabilitySpec
    assessment_strategy: AssessmentStrategySpec
    project_contract: ProjectContractSpec = Field(default_factory=default_project_contract)
    risk_class: RiskClass = RiskClass.standard
    domain_pack: str | None = None
    overlays: list[str] = Field(default_factory=list)


class LearnerDeliverableBrief(BaseModel):
    why_this_deliverable_matters: str
    task_to_build: str
    files_to_edit: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)
    example_scenarios: list[str] = Field(default_factory=list)
    implementation_hints: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)


class StarterScenarioSpec(BaseModel):
    id: str
    title: str
    request_summary: str
    expected_behavior: str


class LearnerStarterSurfaceSpec(BaseModel):
    starter_summary: str
    primary_editable_paths: list[str] = Field(default_factory=list)
    support_paths: list[str] = Field(default_factory=list)
    required_endpoints: list[EndpointSpec] = Field(default_factory=list)
    implementation_checklist: list[str] = Field(default_factory=list)
    domain_scenarios: list[StarterScenarioSpec] = Field(default_factory=list)


class PublicCheckSpec(BaseModel):
    id: str
    title: str
    learner_goal: str
    request_method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    request_path: str
    request_body: JsonObject = Field(default_factory=dict)
    expected_status: int = 200
    expected_response_contains: list[str] = Field(default_factory=list)
    files_to_use: list[str] = Field(default_factory=list)


class DeliverableSpec(BaseModel):
    id: str
    title: str
    objective: str
    starter_type: StarterType
    overlay_ids: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)
    learner_starter_surface: LearnerStarterSurfaceSpec | None = None
    learner_brief: LearnerDeliverableBrief | None = None
    public_checks: list[PublicCheckSpec] = Field(default_factory=list)


class DeliverableGate(BaseModel):
    deliverable_id: str
    cumulative_deliverables: list[str]
    active_public_check_ids: list[str]
    active_test_ids: list[str]


class TaskAgentServiceSpec(BaseModel):
    title: str
    summary: str
    package_type: PackageType
    risk_class: RiskClass = RiskClass.standard
    domain_pack: str | None = None
    overlays: list[str] = Field(default_factory=list)
    course_structure: CourseStructureSpec
    runtime_dependencies: RuntimeDependencySpec
    capabilities: CapabilitySpec
    assessment_strategy: AssessmentStrategySpec
    project_contract: ProjectContractSpec = Field(default_factory=default_project_contract)
    public_endpoints: list[EndpointSpec] = Field(default_factory=list)
    deliverables: list[DeliverableSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "TaskAgentServiceSpec":
        deliverable_ids = [deliverable.id for deliverable in self.deliverables]
        if len(deliverable_ids) != len(set(deliverable_ids)):
            raise ValueError("deliverable ids must be unique")
        if self.course_structure.package_type != self.package_type:
            raise ValueError("course_structure.package_type must match package_type")
        if (
            self.package_type == PackageType.progressive_codebase_course
            and len(self.deliverables) < 2
        ):
            raise ValueError("progressive codebase courses need at least two deliverables")
        if (
            self.course_structure.progression_mode == ProgressionMode.cumulative_deliverable_gates
            and not self.course_structure.shared_codebase
        ):
            raise ValueError(
                "cumulative deliverable gates require a shared codebase"
            )
        return self

    @property
    def deliverable_order(self) -> dict[str, int]:
        return {deliverable.id: index for index, deliverable in enumerate(self.deliverables)}

    def gate_for(self, deliverable_id: str) -> DeliverableGate:
        order = self.deliverable_order
        if deliverable_id not in order:
            raise ValueError(f"unknown deliverable id: {deliverable_id}")
        cutoff = order[deliverable_id]
        if self.course_structure.progression_mode == ProgressionMode.independent_deliverables:
            cumulative_deliverables = [deliverable_id]
        else:
            cumulative_deliverables = [
                deliverable.id for deliverable in self.deliverables[: cutoff + 1]
            ]
        active_public_check_ids = [
            check.id
            for deliverable in self.deliverables
            if deliverable.id in cumulative_deliverables
            for check in deliverable.public_checks
        ]
        return DeliverableGate(
            deliverable_id=deliverable_id,
            cumulative_deliverables=cumulative_deliverables,
            active_public_check_ids=active_public_check_ids,
            active_test_ids=active_public_check_ids,
        )
