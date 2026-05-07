from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.registry import PackageType, RiskClass, StarterType

JsonObject = dict[str, Any]
JsonSchema = dict[str, Any]


class AgentMode(str, Enum):
    routed_single_step = "routed_single_step"
    tool_using_single_run = "tool_using_single_run"
    multi_step_workflow = "multi_step_workflow"
    retrieval_plus_action = "retrieval_plus_action"
    async_human_in_loop = "async_human_in_loop"


class ToolSafety(str, Enum):
    read = "read"
    write = "write"
    irreversible = "irreversible"


class TraceEventType(str, Enum):
    run_started = "run_started"
    model_called = "model_called"
    tool_selected = "tool_selected"
    tool_called = "tool_called"
    tool_result = "tool_result"
    approval_requested = "approval_requested"
    escalated = "escalated"
    fallback_used = "fallback_used"
    run_completed = "run_completed"
    run_failed = "run_failed"


class EndpointSpec(BaseModel):
    method: Literal["GET", "POST"]
    path: str
    required: bool = True


class ToolSpec(BaseModel):
    id: str
    description: str
    safety: ToolSafety
    input_schema: JsonSchema
    output_schema: JsonSchema
    grader_fixture_id: str
    dry_run_supported: bool = True
    approval_required: bool = False
    idempotency_key_arg: str | None = None
    timeout_ms: int = Field(default=10_000, ge=100, le=120_000)
    max_calls_per_run: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_irreversible_tool(self) -> "ToolSpec":
        if self.safety == ToolSafety.irreversible and not self.approval_required:
            raise ValueError("irreversible tools must require approval")
        return self


class ToolRegistry(BaseModel):
    tools: list[ToolSpec] = Field(min_length=1)
    max_steps_per_run: int = Field(default=8, ge=1, le=50)
    max_parallel_tool_calls: int = Field(default=1, ge=1, le=16)
    allow_tool_retries: bool = True

    @field_validator("tools")
    @classmethod
    def unique_tool_ids(cls, tools: list[ToolSpec]) -> list[ToolSpec]:
        ids = [tool.id for tool in tools]
        if len(ids) != len(set(ids)):
            raise ValueError("tool ids must be unique")
        return tools


class BudgetPolicy(BaseModel):
    max_steps: int = Field(ge=1, le=50)
    max_tool_calls: int = Field(ge=1, le=200)
    max_runtime_ms: int = Field(ge=100, le=600_000)
    max_cost_usd: float = Field(ge=0.0)


class ApprovalPolicy(BaseModel):
    require_for_irreversible: bool = True
    require_for_tools: list[str] = Field(default_factory=list)
    require_for_risk_labels: list[str] = Field(default_factory=list)


class EscalationRule(BaseModel):
    reason: Literal[
        "low_confidence",
        "ambiguous_request",
        "missing_tool",
        "tool_failure",
        "policy_block",
        "budget_exhausted",
    ]
    action: Literal["escalate", "abort", "request_approval"]


class FallbackStep(BaseModel):
    trigger: Literal[
        "tool_timeout",
        "tool_5xx",
        "invalid_tool_output",
        "model_failure",
        "low_confidence",
    ]
    action: Literal[
        "retry_same_tool",
        "switch_tool",
        "switch_model",
        "escalate",
        "return_partial",
    ]
    target_id: str | None = None
    max_retries: int = Field(default=1, ge=0, le=3)


class FallbackPolicy(BaseModel):
    steps: list[FallbackStep] = Field(default_factory=list)


class TraceContract(BaseModel):
    required_events: list[TraceEventType] = Field(
        default_factory=lambda: [
            TraceEventType.run_started,
            TraceEventType.model_called,
            TraceEventType.run_completed,
        ]
    )
    require_prompt_version: bool = True
    require_model_name: bool = True
    require_token_usage: bool = True
    require_cost_usd: bool = True


class SLOProfile(BaseModel):
    p95_run_latency_ms: int | None = Field(default=None, ge=1)
    min_task_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_cost_per_success_usd: float | None = Field(default=None, ge=0.0)
    min_escalation_precision: float | None = Field(default=None, ge=0.0, le=1.0)


def default_agent_endpoints() -> list[EndpointSpec]:
    return [
        EndpointSpec(method="POST", path="/run"),
        EndpointSpec(method="GET", path="/runs/{id}"),
        EndpointSpec(method="GET", path="/trace/{id}"),
        EndpointSpec(method="POST", path="/approve/{id}"),
        EndpointSpec(method="POST", path="/eval"),
        EndpointSpec(method="GET", path="/health"),
    ]


class ProductionContract(BaseModel):
    canonical_endpoints: list[EndpointSpec] = Field(default_factory=default_agent_endpoints)
    supports_async_runs: bool = True
    supports_resume: bool = True
    supports_dry_run: bool = True
    state_backend: Literal["memory", "sqlite", "postgres", "redis"]
    trace_retention_days: int = Field(default=14, ge=1, le=365)
    budget_policy: BudgetPolicy
    approval_policy: ApprovalPolicy
    escalation_policy: list[EscalationRule] = Field(default_factory=list)
    fallback_policy: FallbackPolicy
    trace_contract: TraceContract
    slos: SLOProfile


class WorkspaceScope(str, Enum):
    shared_course_workspace = "shared_course_workspace"
    per_module_workspace = "per_module_workspace"


class ProgressionMode(str, Enum):
    cumulative_module_gates = "cumulative_module_gates"
    independent_modules = "independent_modules"


class ExecutionSurface(str, Enum):
    http_service = "http_service"
    cli = "cli"
    protocol_server = "protocol_server"


class RetrievalMode(str, Enum):
    none = "none"
    ranked_results = "ranked_results"
    grounded_answers = "grounded_answers"


class CourseStructureSpec(BaseModel):
    package_type: PackageType
    workspace_scope: WorkspaceScope
    progression_mode: ProgressionMode
    shared_codebase: bool = True


class RuntimeDependencySpec(BaseModel):
    execution_surface: ExecutionSurface
    editable_files: list[str] = Field(default_factory=list)
    visible_fixture_files: list[str] = Field(default_factory=list)
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
    cumulative_module_gates: bool = True
    learner_submission_enabled: bool = True


class AssignmentDesignSpec(BaseModel):
    course_structure: CourseStructureSpec
    runtime_dependencies: RuntimeDependencySpec
    capabilities: CapabilitySpec
    assessment_strategy: AssessmentStrategySpec
    risk_class: RiskClass = RiskClass.standard
    domain_pack: str | None = None
    overlays: list[str] = Field(default_factory=list)


class TaskEvalCase(BaseModel):
    id: str
    input: JsonObject
    expected_output: JsonObject | None = None
    should_escalate: bool | None = None
    requires_approval: bool | None = None
    must_use_any_of_tools: list[str] = Field(default_factory=list)
    must_not_use_tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    id: str
    cases: list[TaskEvalCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def unique_case_ids(cls, cases: list[TaskEvalCase]) -> list[TaskEvalCase]:
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("eval case ids must be unique")
        return cases


class ToolChoiceExpectation(BaseModel):
    case_id: str
    must_call_any_of: list[str] = Field(default_factory=list)
    must_call_all_of: list[str] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)


class ToolInvocationExpectation(BaseModel):
    case_id: str
    tool_id: str
    required_args_subset: JsonObject = Field(default_factory=dict)


class EscalationExpectation(BaseModel):
    case_id: str
    must_escalate: bool
    allowed_reasons: list[str] = Field(default_factory=list)


class ApprovalExpectation(BaseModel):
    case_id: str
    tool_id: str
    requires_approval: bool = True


class FaultInjection(BaseModel):
    case_id: str
    target: Literal["model", "tool"]
    target_id: str
    failure_mode: Literal["timeout", "5xx", "invalid_json", "empty_result"]


class OutputSchemaTestParams(BaseModel):
    type: Literal["output_schema_test"]
    case_ids: list[str] = Field(min_length=1)


class TraceSchemaTestParams(BaseModel):
    type: Literal["trace_schema_test"]
    case_ids: list[str] = Field(min_length=1)
    required_events: list[TraceEventType] = Field(min_length=1)


class ToolSelectionTestParams(BaseModel):
    type: Literal["tool_selection_test"]
    expectations: list[ToolChoiceExpectation] = Field(min_length=1)


class ToolInvocationCorrectnessTestParams(BaseModel):
    type: Literal["tool_invocation_correctness_test"]
    expectations: list[ToolInvocationExpectation] = Field(min_length=1)


class StepBudgetEnforcementTestParams(BaseModel):
    type: Literal["step_budget_enforcement_test"]
    case_ids: list[str] = Field(min_length=1)
    max_steps: int = Field(ge=1, le=50)


class EscalationPolicyTestParams(BaseModel):
    type: Literal["escalation_policy_test"]
    expectations: list[EscalationExpectation] = Field(min_length=1)


class ApprovalGateTestParams(BaseModel):
    type: Literal["approval_gate_test"]
    expectations: list[ApprovalExpectation] = Field(min_length=1)


class FallbackPolicyTestParams(BaseModel):
    type: Literal["fallback_policy_test"]
    injections: list[FaultInjection] = Field(min_length=1)
    min_success_after_fallback: float = Field(ge=0.0, le=1.0)


class DurableResumeTestParams(BaseModel):
    type: Literal["durable_resume_test"]
    case_id: str
    interrupt_after_event: TraceEventType


class DryRunSemanticsTestParams(BaseModel):
    type: Literal["dry_run_semantics_test"]
    case_ids: list[str] = Field(min_length=1)
    mutating_tool_ids: list[str] = Field(min_length=1)


class IdempotentActionTestParams(BaseModel):
    type: Literal["idempotent_action_test"]
    case_ids: list[str] = Field(min_length=1)
    idempotency_key_field: str


BehaviorTest = Annotated[
    Union[
        OutputSchemaTestParams,
        TraceSchemaTestParams,
        ToolSelectionTestParams,
        ToolInvocationCorrectnessTestParams,
        StepBudgetEnforcementTestParams,
        EscalationPolicyTestParams,
        ApprovalGateTestParams,
        FallbackPolicyTestParams,
        DurableResumeTestParams,
        DryRunSemanticsTestParams,
        IdempotentActionTestParams,
    ],
    Field(discriminator="type"),
]


class TaskSuccessRateTestParams(BaseModel):
    type: Literal["task_success_rate_test"]
    dataset_id: str
    min_success_rate: float = Field(ge=0.0, le=1.0)


class P95RunLatencyTestParams(BaseModel):
    type: Literal["p95_run_latency_test"]
    dataset_id: str
    concurrency: int = Field(ge=1, le=256)
    p95_ms: int = Field(ge=1)


class CostPerSuccessTestParams(BaseModel):
    type: Literal["cost_per_success_test"]
    dataset_id: str
    max_cost_usd: float = Field(ge=0.0)


class RecoveryAfterToolFailureTestParams(BaseModel):
    type: Literal["recovery_after_tool_failure_test"]
    dataset_id: str
    injections: list[FaultInjection] = Field(min_length=1)
    min_success_rate_after_faults: float = Field(ge=0.0, le=1.0)


class EscalationPrecisionTestParams(BaseModel):
    type: Literal["escalation_precision_test"]
    dataset_id: str
    min_precision: float = Field(ge=0.0, le=1.0)


class TaskOutputQualityJudgeTestParams(BaseModel):
    type: Literal["task_output_quality_judge_test"]
    dataset_id: str
    judge_id: str
    rubric_id: str
    min_avg_score: float = Field(ge=0.0, le=1.0)


class ConfidenceCalibrationJudgeTestParams(BaseModel):
    type: Literal["confidence_calibration_judge_test"]
    dataset_id: str
    max_expected_calibration_error: float = Field(ge=0.0, le=1.0)


AssessmentTest = Annotated[
    Union[
        TaskSuccessRateTestParams,
        P95RunLatencyTestParams,
        CostPerSuccessTestParams,
        RecoveryAfterToolFailureTestParams,
        EscalationPrecisionTestParams,
        TaskOutputQualityJudgeTestParams,
        ConfidenceCalibrationJudgeTestParams,
    ],
    Field(discriminator="type"),
]


class LearnerModuleBrief(BaseModel):
    why_this_module_matters: str
    task_to_build: str
    files_to_edit: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)
    example_scenarios: list[str] = Field(default_factory=list)
    implementation_hints: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)


class PublicCheckSpec(BaseModel):
    id: str
    title: str
    learner_goal: str
    case_id: str
    files_to_use: list[str] = Field(default_factory=list)
    expected_assertions: list[str] = Field(default_factory=list)
    covers_behavior_ids: list[str] = Field(default_factory=list)
    covers_quality_ids: list[str] = Field(default_factory=list)


class ModuleSpec(BaseModel):
    id: str
    title: str
    objective: str
    starter_type: StarterType
    overlay_ids: list[str] = Field(default_factory=list)
    learner_brief: LearnerModuleBrief | None = None
    public_checks: list[PublicCheckSpec] = Field(default_factory=list)


class BehaviorSpec(BaseModel):
    id: str
    description: str
    first_required_in: str
    exploratory: bool = False
    test: BehaviorTest


class QualitySpec(BaseModel):
    id: str
    description: str
    first_required_in: str
    test: AssessmentTest


class ModuleGate(BaseModel):
    module_id: str
    cumulative_modules: list[str]
    active_behavior_ids: list[str]
    active_quality_ids: list[str]
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
    supported_modes: list[AgentMode] = Field(min_length=1)
    modules: list[ModuleSpec] = Field(min_length=1)
    task_schema: JsonSchema
    output_schema: JsonSchema
    trace_schema: JsonSchema
    run_state_schema: JsonSchema
    tool_registry: ToolRegistry
    eval_dataset: EvalDataset
    production_contract: ProductionContract
    behaviors: list[BehaviorSpec] = Field(min_length=1)
    qualities: list[QualitySpec] = Field(min_length=1)

    @field_validator("supported_modes")
    @classmethod
    def unique_supported_modes(cls, modes: list[AgentMode]) -> list[AgentMode]:
        if len(modes) != len(set(modes)):
            raise ValueError("supported modes must be unique")
        return modes

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "TaskAgentServiceSpec":
        module_ids = [module.id for module in self.modules]
        behavior_ids = [behavior.id for behavior in self.behaviors]
        quality_ids = [quality.id for quality in self.qualities]

        if len(module_ids) != len(set(module_ids)):
            raise ValueError("module ids must be unique")
        if len(behavior_ids) != len(set(behavior_ids)):
            raise ValueError("behavior ids must be unique")
        if len(quality_ids) != len(set(quality_ids)):
            raise ValueError("quality ids must be unique")
        if self.course_structure.package_type != self.package_type:
            raise ValueError("course_structure.package_type must match package_type")
        if self.package_type == PackageType.progressive_codebase_course and len(self.modules) < 2:
            raise ValueError("progressive codebase courses need at least two modules")
        if self.course_structure.progression_mode == ProgressionMode.cumulative_module_gates and not self.course_structure.shared_codebase:
            raise ValueError("cumulative module gates require a shared codebase")
        return self

    @property
    def module_order(self) -> dict[str, int]:
        return {module.id: index for index, module in enumerate(self.modules)}

    @property
    def tool_ids(self) -> set[str]:
        return {tool.id for tool in self.tool_registry.tools}

    @property
    def eval_case_ids(self) -> set[str]:
        return {case.id for case in self.eval_dataset.cases}

    def gate_for(self, module_id: str) -> ModuleGate:
        order = self.module_order
        if module_id not in order:
            raise ValueError(f"unknown module id: {module_id}")

        cutoff = order[module_id]
        cumulative_modules = [module.id for module in self.modules[: cutoff + 1]]
        active_behaviors = [
            behavior.id
            for behavior in self.behaviors
            if order.get(behavior.first_required_in, cutoff + 1) <= cutoff
        ]
        active_qualities = [
            quality.id
            for quality in self.qualities
            if order.get(quality.first_required_in, cutoff + 1) <= cutoff
        ]
        return ModuleGate(
            module_id=module_id,
            cumulative_modules=cumulative_modules,
            active_behavior_ids=active_behaviors,
            active_quality_ids=active_qualities,
            active_test_ids=active_behaviors + active_qualities,
        )
