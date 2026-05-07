from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.registry import DESIGN_CATALOG, PackageType, RiskClass
from app.domain.task_agent import (
    AgentMode,
    ApprovalGateTestParams,
    CostPerSuccessTestParams,
    DryRunSemanticsTestParams,
    DurableResumeTestParams,
    EscalationPolicyTestParams,
    FallbackPolicyTestParams,
    IdempotentActionTestParams,
    ModuleGate,
    OutputSchemaTestParams,
    P95RunLatencyTestParams,
    QualitySpec,
    RecoveryAfterToolFailureTestParams,
    StepBudgetEnforcementTestParams,
    TaskAgentServiceSpec,
    TaskOutputQualityJudgeTestParams,
    TaskSuccessRateTestParams,
    ToolInvocationCorrectnessTestParams,
    ToolSafety,
    ToolSelectionTestParams,
    TraceSchemaTestParams,
    TraceEventType,
    ConfidenceCalibrationJudgeTestParams,
    EscalationPrecisionTestParams,
)


class ValidationLevel(str, Enum):
    error = "error"
    warning = "warning"


class ValidationIssue(BaseModel):
    level: ValidationLevel
    code: str
    location: str
    message: str


class ModuleGateSummary(BaseModel):
    module_id: str
    active_behavior_ids: list[str]
    active_quality_ids: list[str]
    active_test_count: int


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    module_gates: list[ModuleGateSummary] = Field(default_factory=list)


def _gate_summaries(spec: TaskAgentServiceSpec) -> list[ModuleGateSummary]:
    summaries: list[ModuleGateSummary] = []
    for module in spec.modules:
        gate = spec.gate_for(module.id)
        summaries.append(
            ModuleGateSummary(
                module_id=gate.module_id,
                active_behavior_ids=gate.active_behavior_ids,
                active_quality_ids=gate.active_quality_ids,
                active_test_count=len(gate.active_test_ids),
            )
        )
    return summaries


def validate_task_agent_spec(spec: TaskAgentServiceSpec) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    module_ids = set(spec.module_order.keys())
    tool_ids = spec.tool_ids
    eval_case_ids = spec.eval_case_ids
    endpoint_paths = {endpoint.path for endpoint in spec.production_contract.canonical_endpoints}
    overlay_ids = {overlay.id for overlay in DESIGN_CATALOG.overlays}
    escalation_reasons = {rule.reason for rule in spec.production_contract.escalation_policy}
    tool_by_id = {tool.id: tool for tool in spec.tool_registry.tools}

    def add_issue(level: ValidationLevel, code: str, location: str, message: str) -> None:
        issue = ValidationIssue(level=level, code=code, location=location, message=message)
        if level == ValidationLevel.error:
            errors.append(issue)
        else:
            warnings.append(issue)

    if spec.domain_pack:
        domain_pack = DESIGN_CATALOG.domain_pack_by_id(spec.domain_pack)
        if domain_pack is None:
            add_issue(
                ValidationLevel.error,
                "unknown_domain_pack",
                "domain_pack",
                f"Unknown domain pack '{spec.domain_pack}'.",
            )
        elif domain_pack.risk_class != spec.risk_class:
            add_issue(
                ValidationLevel.warning,
                "domain_pack_risk_mismatch",
                "risk_class",
                f"Domain pack '{spec.domain_pack}' is marked '{domain_pack.risk_class.value}'.",
            )

    if not spec.runtime_dependencies.editable_files:
        add_issue(
            ValidationLevel.error,
            "missing_editable_files",
            "runtime_dependencies.editable_files",
            "The runtime dependency spec must declare which learner files are editable.",
        )
    if spec.runtime_dependencies.visible_check_command is None:
        add_issue(
            ValidationLevel.warning,
            "missing_visible_check_command",
            "runtime_dependencies.visible_check_command",
            "Add a visible check command so learners can debug before submitting to the grader.",
        )
    if spec.course_structure.workspace_scope.value == "per_module_workspace" and spec.course_structure.shared_codebase:
        add_issue(
            ValidationLevel.error,
            "workspace_scope_conflict",
            "course_structure.workspace_scope",
            "Per-module workspaces cannot be paired with a shared codebase course structure.",
        )
    if spec.capabilities.answer_synthesis_required and spec.capabilities.retrieval_mode.value == "none":
        add_issue(
            ValidationLevel.error,
            "answer_synthesis_without_retrieval",
            "capabilities.answer_synthesis_required",
            "Answer synthesis requires retrieval support to be declared in the capability spec.",
        )
    if spec.capabilities.citations_required and spec.capabilities.retrieval_mode.value == "none":
        add_issue(
            ValidationLevel.error,
            "citations_without_retrieval",
            "capabilities.citations_required",
            "Citation requirements need retrieval support in the capability spec.",
        )
    if spec.assessment_strategy.public_checks_required and spec.runtime_dependencies.visible_check_command is None:
        add_issue(
            ValidationLevel.error,
            "public_checks_without_command",
            "assessment_strategy.public_checks_required",
            "Public/basic checks are required, but the runtime dependency spec has no visible check command.",
        )

    if spec.package_type == PackageType.survey_course and len(spec.modules) > 3:
        add_issue(
            ValidationLevel.warning,
            "survey_many_modules",
            "modules",
            "This looks more like a progressive codebase than a survey assignment.",
        )

    if spec.risk_class in {RiskClass.review_required, RiskClass.high_stakes}:
        add_issue(
            ValidationLevel.warning,
            "manual_review_required",
            "risk_class",
            "This spec should require human review before publish.",
        )

    for module in spec.modules:
        gate = spec.gate_for(module.id)
        for overlay_id in module.overlay_ids:
            if overlay_id not in overlay_ids:
                add_issue(
                    ValidationLevel.error,
                    "unknown_overlay",
                    f"modules.{module.id}.overlay_ids",
                    f"Unknown overlay '{overlay_id}'.",
                )
        if module.learner_brief is None:
            add_issue(
                ValidationLevel.error,
                "missing_learner_brief",
                f"modules.{module.id}.learner_brief",
                "Each module needs a learner-facing brief before it can be reviewed or published.",
            )
            continue
        if not module.learner_brief.files_to_edit:
            add_issue(
                ValidationLevel.error,
                "missing_files_to_edit",
                f"modules.{module.id}.learner_brief.files_to_edit",
                "Learner briefs must tell the learner which files to edit.",
            )
        if not module.learner_brief.definition_of_done:
            add_issue(
                ValidationLevel.error,
                "missing_definition_of_done",
                f"modules.{module.id}.learner_brief.definition_of_done",
                "Learner briefs must explain what done looks like for the module.",
            )
        if not module.learner_brief.example_scenarios:
            add_issue(
                ValidationLevel.warning,
                "missing_examples",
                f"modules.{module.id}.learner_brief.example_scenarios",
                "Add at least one concrete learner-facing example or scenario.",
            )
        learner_brief_text = " ".join(
            [
                module.learner_brief.why_this_module_matters,
                module.learner_brief.task_to_build,
                *module.learner_brief.example_scenarios,
            ]
        ).lower()
        if "hidden checkpoint" in learner_brief_text or "active checks" in learner_brief_text:
            add_issue(
                ValidationLevel.error,
                "internal_jargon_in_brief",
                f"modules.{module.id}.learner_brief",
                "Learner briefs should explain the task without internal checkpoint or grader jargon.",
            )
        if not module.public_checks:
            add_issue(
                ValidationLevel.error,
                "missing_public_checks",
                f"modules.{module.id}.public_checks",
                "Each module needs learner-visible public checks before it can be reviewed or published.",
            )
        seen_public_check_ids: set[str] = set()
        active_behavior_ids = set(gate.active_behavior_ids)
        active_quality_ids = set(gate.active_quality_ids)
        for index, public_check in enumerate(module.public_checks):
            location = f"modules.{module.id}.public_checks[{index}]"
            if public_check.id in seen_public_check_ids:
                add_issue(
                    ValidationLevel.error,
                    "duplicate_public_check_id",
                    f"{location}.id",
                    f"Duplicate public check id '{public_check.id}'.",
                )
            seen_public_check_ids.add(public_check.id)
            if public_check.case_id not in eval_case_ids:
                add_issue(
                    ValidationLevel.error,
                    "unknown_public_check_case",
                    f"{location}.case_id",
                    f"Unknown eval case '{public_check.case_id}' in public check '{public_check.id}'.",
                )
            if not public_check.learner_goal.strip():
                add_issue(
                    ValidationLevel.error,
                    "missing_public_check_goal",
                    f"{location}.learner_goal",
                    "Public checks should explain what the learner is trying to prove.",
                )
            if not public_check.expected_assertions:
                add_issue(
                    ValidationLevel.warning,
                    "missing_public_check_assertions",
                    f"{location}.expected_assertions",
                    "Public checks should tell the learner what the visible check is asserting.",
                )
            public_check_text = " ".join(
                [
                    public_check.title,
                    public_check.learner_goal,
                    *public_check.expected_assertions,
                ]
            ).lower()
            if "hidden checkpoint" in public_check_text or "internal grader" in public_check_text:
                add_issue(
                    ValidationLevel.error,
                    "internal_jargon_in_public_check",
                    location,
                    "Public checks should describe learner-visible expectations, not hidden grader machinery.",
                )
            invalid_behavior_ids = sorted(set(public_check.covers_behavior_ids) - active_behavior_ids)
            if invalid_behavior_ids:
                add_issue(
                    ValidationLevel.error,
                    "inactive_public_check_behavior",
                    f"{location}.covers_behavior_ids",
                    "Public checks can only reference active behaviors for the module: "
                    + ", ".join(f"'{item}'" for item in invalid_behavior_ids),
                )
            invalid_quality_ids = sorted(set(public_check.covers_quality_ids) - active_quality_ids)
            if invalid_quality_ids:
                add_issue(
                    ValidationLevel.error,
                    "inactive_public_check_quality",
                    f"{location}.covers_quality_ids",
                    "Public checks can only reference active qualities for the module: "
                    + ", ".join(f"'{item}'" for item in invalid_quality_ids),
                )
            if not public_check.covers_behavior_ids and not public_check.covers_quality_ids:
                add_issue(
                    ValidationLevel.warning,
                    "unmapped_public_check",
                    location,
                    "Public checks should map to at least one active behavior or quality so reviewers can verify intent.",
                )
        if len(module.public_checks) > 4:
            add_issue(
                ValidationLevel.warning,
                "too_many_public_checks",
                f"modules.{module.id}.public_checks",
                "Keep learner-visible checks focused. More than four checks usually signals that the module should rely on the hidden grader for depth.",
            )

    for behavior in spec.behaviors:
        if behavior.first_required_in not in module_ids:
            add_issue(
                ValidationLevel.error,
                "unknown_behavior_module",
                f"behaviors.{behavior.id}.first_required_in",
                f"Unknown module '{behavior.first_required_in}'.",
            )

        test = behavior.test
        if isinstance(test, (OutputSchemaTestParams, TraceSchemaTestParams, StepBudgetEnforcementTestParams, DryRunSemanticsTestParams)):
            for case_id in test.case_ids:
                if case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"behaviors.{behavior.id}.test.case_ids",
                        f"Unknown eval case '{case_id}'.",
                    )
        if isinstance(test, ToolSelectionTestParams):
            for expectation in test.expectations:
                if expectation.case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"behaviors.{behavior.id}.test.expectations",
                        f"Unknown eval case '{expectation.case_id}'.",
                    )
                referenced_tools = set(expectation.must_call_any_of + expectation.must_call_all_of + expectation.must_not_call)
                for tool_id in referenced_tools:
                    if tool_id not in tool_ids:
                        add_issue(
                            ValidationLevel.error,
                            "unknown_tool_reference",
                            f"behaviors.{behavior.id}.test.expectations",
                            f"Unknown tool '{tool_id}'.",
                        )
        if isinstance(test, ToolInvocationCorrectnessTestParams):
            for expectation in test.expectations:
                if expectation.case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"behaviors.{behavior.id}.test.expectations",
                        f"Unknown eval case '{expectation.case_id}'.",
                    )
                if expectation.tool_id not in tool_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_tool_reference",
                        f"behaviors.{behavior.id}.test.expectations",
                        f"Unknown tool '{expectation.tool_id}'.",
                    )
        if isinstance(test, EscalationPolicyTestParams):
            for expectation in test.expectations:
                if expectation.case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"behaviors.{behavior.id}.test.expectations",
                        f"Unknown eval case '{expectation.case_id}'.",
                    )
                for reason in expectation.allowed_reasons:
                    if reason not in escalation_reasons:
                        add_issue(
                            ValidationLevel.warning,
                            "undeclared_escalation_reason",
                            f"behaviors.{behavior.id}.test.expectations",
                            f"Escalation reason '{reason}' is not declared in the production contract.",
                        )
        if isinstance(test, ApprovalGateTestParams):
            for expectation in test.expectations:
                if expectation.case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"behaviors.{behavior.id}.test.expectations",
                        f"Unknown eval case '{expectation.case_id}'.",
                    )
                if expectation.tool_id not in tool_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_tool_reference",
                        f"behaviors.{behavior.id}.test.expectations",
                        f"Unknown tool '{expectation.tool_id}'.",
                    )
                elif not tool_by_id[expectation.tool_id].approval_required:
                    add_issue(
                        ValidationLevel.warning,
                        "approval_not_required_by_tool",
                        f"behaviors.{behavior.id}.test.expectations",
                        f"Tool '{expectation.tool_id}' does not itself require approval.",
                    )
        if isinstance(test, FallbackPolicyTestParams):
            for injection in test.injections:
                if injection.case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"behaviors.{behavior.id}.test.injections",
                        f"Unknown eval case '{injection.case_id}'.",
                    )
                if injection.target == "tool" and injection.target_id not in tool_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_tool_reference",
                        f"behaviors.{behavior.id}.test.injections",
                        f"Unknown tool '{injection.target_id}'.",
                    )
        if isinstance(test, DurableResumeTestParams):
            if test.case_id not in eval_case_ids:
                add_issue(
                    ValidationLevel.error,
                    "unknown_eval_case",
                    f"behaviors.{behavior.id}.test.case_id",
                    f"Unknown eval case '{test.case_id}'.",
                )
            if not spec.production_contract.supports_resume:
                add_issue(
                    ValidationLevel.error,
                    "resume_not_supported",
                    f"behaviors.{behavior.id}.test",
                    "Durable resume tests require supports_resume=True.",
                )
        if isinstance(test, DryRunSemanticsTestParams):
            if not spec.production_contract.supports_dry_run:
                add_issue(
                    ValidationLevel.error,
                    "dry_run_not_supported",
                    f"behaviors.{behavior.id}.test",
                    "Dry-run tests require supports_dry_run=True.",
                )
            for tool_id in test.mutating_tool_ids:
                if tool_id not in tool_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_tool_reference",
                        f"behaviors.{behavior.id}.test.mutating_tool_ids",
                        f"Unknown tool '{tool_id}'.",
                    )
                elif tool_by_id[tool_id].safety == ToolSafety.read:
                    add_issue(
                        ValidationLevel.warning,
                        "read_tool_marked_mutating",
                        f"behaviors.{behavior.id}.test.mutating_tool_ids",
                        f"Tool '{tool_id}' is marked read-only.",
                    )
        if isinstance(test, IdempotentActionTestParams):
            for case_id in test.case_ids:
                if case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"behaviors.{behavior.id}.test.case_ids",
                        f"Unknown eval case '{case_id}'.",
                    )

    for quality in spec.qualities:
        if quality.first_required_in not in module_ids:
            add_issue(
                ValidationLevel.error,
                "unknown_quality_module",
                f"qualities.{quality.id}.first_required_in",
                f"Unknown module '{quality.first_required_in}'.",
            )

        test = quality.test
        if isinstance(test, (TaskSuccessRateTestParams, P95RunLatencyTestParams, CostPerSuccessTestParams, EscalationPrecisionTestParams, TaskOutputQualityJudgeTestParams, ConfidenceCalibrationJudgeTestParams)):
            if test.dataset_id != spec.eval_dataset.id:
                add_issue(
                    ValidationLevel.error,
                    "unknown_dataset",
                    f"qualities.{quality.id}.test.dataset_id",
                    f"Dataset '{test.dataset_id}' does not match eval dataset '{spec.eval_dataset.id}'.",
                )
        if isinstance(test, RecoveryAfterToolFailureTestParams):
            if test.dataset_id != spec.eval_dataset.id:
                add_issue(
                    ValidationLevel.error,
                    "unknown_dataset",
                    f"qualities.{quality.id}.test.dataset_id",
                    f"Dataset '{test.dataset_id}' does not match eval dataset '{spec.eval_dataset.id}'.",
                )
            for injection in test.injections:
                if injection.case_id not in eval_case_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_eval_case",
                        f"qualities.{quality.id}.test.injections",
                        f"Unknown eval case '{injection.case_id}'.",
                    )
                if injection.target == "tool" and injection.target_id not in tool_ids:
                    add_issue(
                        ValidationLevel.error,
                        "unknown_tool_reference",
                        f"qualities.{quality.id}.test.injections",
                        f"Unknown tool '{injection.target_id}'.",
                    )

    if AgentMode.async_human_in_loop in spec.supported_modes:
        if "/approve/{id}" not in endpoint_paths:
            add_issue(
                ValidationLevel.error,
                "missing_approval_endpoint",
                "production_contract.canonical_endpoints",
                "Async human-in-loop agents require '/approve/{id}'.",
            )
        if not spec.production_contract.supports_async_runs:
            add_issue(
                ValidationLevel.error,
                "async_mode_disabled",
                "production_contract.supports_async_runs",
                "Async human-in-loop mode requires supports_async_runs=True.",
            )

    if any(tool.approval_required for tool in spec.tool_registry.tools) and "/approve/{id}" not in endpoint_paths:
        add_issue(
            ValidationLevel.error,
            "approval_endpoint_missing",
            "production_contract.canonical_endpoints",
            "Approval-required tools need '/approve/{id}'.",
        )

    if any(tool.safety == ToolSafety.irreversible for tool in spec.tool_registry.tools):
        if not spec.production_contract.supports_dry_run:
            add_issue(
                ValidationLevel.warning,
                "irreversible_without_dry_run",
                "production_contract.supports_dry_run",
                "Irreversible actions should usually support dry-run mode.",
            )
        if not spec.production_contract.approval_policy.require_for_irreversible:
            add_issue(
                ValidationLevel.error,
                "irreversible_without_policy",
                "production_contract.approval_policy",
                "Irreversible actions must require approval in policy.",
            )

    if TraceEventType.run_completed not in spec.production_contract.trace_contract.required_events:
        add_issue(
            ValidationLevel.warning,
            "trace_missing_run_completed",
            "production_contract.trace_contract.required_events",
            "Run completion should usually be part of the trace contract.",
        )

    if not spec.behaviors:
        add_issue(
            ValidationLevel.error,
            "missing_behaviors",
            "behaviors",
            "At least one behavior is required.",
        )
    if not spec.qualities:
        add_issue(
            ValidationLevel.error,
            "missing_qualities",
            "qualities",
            "At least one quality target is required.",
        )

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        module_gates=_gate_summaries(spec),
    )


def compute_task_agent_gate(spec: TaskAgentServiceSpec, module_id: str) -> ModuleGate:
    return spec.gate_for(module_id)
