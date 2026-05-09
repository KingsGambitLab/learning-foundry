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
    DeliverableGate,
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
from app.services.review_area_coverage import (
    RESERVED_REVIEW_AREA_TAGS,
    infer_review_area_case_tags,
    summarize_review_area_hidden_coverage,
)


class ValidationLevel(str, Enum):
    error = "error"
    warning = "warning"


class ValidationIssue(BaseModel):
    level: ValidationLevel
    code: str
    location: str
    message: str


class DeliverableGateSummary(BaseModel):
    deliverable_id: str
    active_behavior_ids: list[str]
    active_quality_ids: list[str]
    active_test_count: int


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    deliverable_gates: list[DeliverableGateSummary] = Field(default_factory=list)


def _gate_summaries(spec: TaskAgentServiceSpec) -> list[DeliverableGateSummary]:
    summaries: list[DeliverableGateSummary] = []
    for deliverable in spec.deliverables:
        gate = spec.gate_for(deliverable.id)
        summaries.append(
            DeliverableGateSummary(
                deliverable_id=gate.deliverable_id,
                active_behavior_ids=gate.active_behavior_ids,
                active_quality_ids=gate.active_quality_ids,
                active_test_count=len(gate.active_test_ids),
            )
        )
    return summaries


def validate_task_agent_spec(spec: TaskAgentServiceSpec) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    deliverable_ids = set(spec.deliverable_order.keys())
    tool_ids = spec.tool_ids
    eval_case_ids = spec.eval_case_ids
    inferred_case_tags = infer_review_area_case_tags(spec)
    hidden_coverage = {
        summary.deliverable_id: summary
        for summary in summarize_review_area_hidden_coverage(spec)
    }
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
    if spec.course_structure.workspace_scope.value == "per_deliverable_workspace" and spec.course_structure.shared_codebase:
        add_issue(
            ValidationLevel.error,
            "workspace_scope_conflict",
            "course_structure.workspace_scope",
            "Per-deliverable workspaces cannot be paired with a shared codebase course structure.",
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

    if spec.package_type == PackageType.survey_course and len(spec.deliverables) > 3:
        add_issue(
            ValidationLevel.warning,
            "survey_many_deliverables",
            "deliverables",
            "This looks more like a progressive codebase than a survey assignment.",
        )

    if spec.risk_class in {RiskClass.review_required, RiskClass.high_stakes}:
        add_issue(
            ValidationLevel.warning,
            "manual_review_required",
            "risk_class",
            "This spec should require human review before publish.",
        )

    for deliverable in spec.deliverables:
        gate = spec.gate_for(deliverable.id)
        coverage = hidden_coverage[deliverable.id]
        for overlay_id in deliverable.overlay_ids:
            if overlay_id not in overlay_ids:
                add_issue(
                    ValidationLevel.error,
                    "unknown_overlay",
                    f"deliverables.{deliverable.id}.overlay_ids",
                    f"Unknown overlay '{overlay_id}'.",
                )
        if not deliverable.learning_outcomes:
            add_issue(
                ValidationLevel.error,
                "missing_deliverable_learning_outcomes",
                f"deliverables.{deliverable.id}.learning_outcomes",
                "Each deliverable should publish concrete learning outcomes derived from the learner task.",
            )
        else:
            seen_outcomes: set[str] = set()
            for index, outcome in enumerate(deliverable.learning_outcomes):
                location = f"deliverables.{deliverable.id}.learning_outcomes[{index}]"
                normalized = outcome.strip()
                if not normalized:
                    add_issue(
                        ValidationLevel.error,
                        "blank_deliverable_learning_outcome",
                        location,
                        "Learning outcomes must not be blank.",
                    )
                    continue
                lowered = normalized.lower()
                if lowered in seen_outcomes:
                    add_issue(
                        ValidationLevel.warning,
                        "duplicate_deliverable_learning_outcome",
                        location,
                        f"Duplicate learning outcome '{normalized}'.",
                    )
                seen_outcomes.add(lowered)
                if any(phrase in lowered for phrase in ["understand", "learn about", "be familiar"]):
                    add_issue(
                        ValidationLevel.warning,
                        "vague_deliverable_learning_outcome",
                        location,
                        "Learning outcomes should describe an observable learner capability, not vague understanding.",
                    )
            if len(deliverable.learning_outcomes) > 4:
                add_issue(
                    ValidationLevel.warning,
                    "too_many_deliverable_learning_outcomes",
                    f"deliverables.{deliverable.id}.learning_outcomes",
                    "Keep deliverable outcomes focused. More than four usually means the deliverable is trying to teach too much at once.",
                )
        if deliverable.learner_brief is None:
            add_issue(
                ValidationLevel.error,
                "missing_learner_brief",
                f"deliverables.{deliverable.id}.learner_brief",
                "Each deliverable needs a learner-facing brief before it can be reviewed or published.",
            )
            continue
        starter_surface = deliverable.learner_starter_surface
        if starter_surface is None:
            add_issue(
                ValidationLevel.error,
                "missing_learner_starter_surface",
                f"deliverables.{deliverable.id}.learner_starter_surface",
                "Each deliverable needs an authored learner starter surface that explains the real files, endpoints, and scenarios the learner owns.",
            )
        else:
            if not starter_surface.primary_editable_paths:
                add_issue(
                    ValidationLevel.error,
                    "missing_primary_editable_paths",
                    f"deliverables.{deliverable.id}.learner_starter_surface.primary_editable_paths",
                    "The learner starter surface must identify at least one primary learner-owned file.",
                )
            if not starter_surface.required_endpoints:
                add_issue(
                    ValidationLevel.error,
                    "missing_required_endpoints",
                    f"deliverables.{deliverable.id}.learner_starter_surface.required_endpoints",
                    "The learner starter surface must list the required public endpoints or commands it preserves.",
                )
            if not starter_surface.domain_scenarios:
                add_issue(
                    ValidationLevel.warning,
                    "missing_domain_scenarios",
                    f"deliverables.{deliverable.id}.learner_starter_surface.domain_scenarios",
                    "Add concrete learner-visible scenarios so the learner can connect the brief to real cases.",
                )
            else:
                for index, scenario in enumerate(starter_surface.domain_scenarios):
                    location = f"deliverables.{deliverable.id}.learner_starter_surface.domain_scenarios[{index}]"
                    if _looks_like_placeholder_scenario(scenario.title, scenario.request_summary, scenario.expected_behavior):
                        add_issue(
                            ValidationLevel.error,
                            "placeholder_domain_scenario",
                            location,
                            "Replace generic placeholder scenarios with domain-specific learner cases.",
                        )
            if deliverable.learner_brief.files_to_edit:
                missing_paths = sorted(
                    set(starter_surface.primary_editable_paths) - set(deliverable.learner_brief.files_to_edit)
                )
                if missing_paths:
                    add_issue(
                        ValidationLevel.warning,
                        "brief_starter_surface_drift",
                        f"deliverables.{deliverable.id}.learner_brief.files_to_edit",
                        "The learner brief should point at the same primary files as the starter surface: "
                        + ", ".join(f"'{path}'" for path in missing_paths),
                    )
        if not deliverable.learner_brief.files_to_edit:
            add_issue(
                ValidationLevel.error,
                "missing_files_to_edit",
                f"deliverables.{deliverable.id}.learner_brief.files_to_edit",
                "Learner briefs must tell the learner which files to edit.",
            )
        if not deliverable.learner_brief.definition_of_done:
            add_issue(
                ValidationLevel.error,
                "missing_definition_of_done",
                f"deliverables.{deliverable.id}.learner_brief.definition_of_done",
                "Learner briefs must explain what done looks like for the deliverable.",
            )
        if not deliverable.learner_brief.example_scenarios:
            add_issue(
                ValidationLevel.warning,
                "missing_examples",
                f"deliverables.{deliverable.id}.learner_brief.example_scenarios",
                "Add at least one concrete learner-facing example or scenario.",
            )
        learner_brief_text = " ".join(
            [
                deliverable.learner_brief.why_this_deliverable_matters,
                deliverable.learner_brief.task_to_build,
                *deliverable.learner_brief.example_scenarios,
            ]
        ).lower()
        if "hidden checkpoint" in learner_brief_text or "active checks" in learner_brief_text:
            add_issue(
                ValidationLevel.error,
                "internal_jargon_in_brief",
                f"deliverables.{deliverable.id}.learner_brief",
                "Learner briefs should explain the task without internal checkpoint or grader jargon.",
            )
        if deliverable.learning_outcomes:
            alignment_text = " ".join(
                [
                    deliverable.title,
                    deliverable.objective,
                    deliverable.learner_brief.task_to_build,
                    *deliverable.learner_brief.definition_of_done,
                    *deliverable.learner_brief.example_scenarios,
                ]
            ).lower()
            if not any(token in alignment_text for token in _alignment_tokens(deliverable.learning_outcomes)):
                add_issue(
                    ValidationLevel.warning,
                    "deliverable_learning_outcomes_need_alignment_review",
                    f"deliverables.{deliverable.id}.learning_outcomes",
                    "The learning outcomes do not obviously line up with the learner brief. Review them before publish.",
                )
        if not deliverable.public_checks:
            add_issue(
                ValidationLevel.error,
                "missing_public_checks",
                f"deliverables.{deliverable.id}.public_checks",
                "Each deliverable needs learner-visible public checks before it can be reviewed or published.",
            )
        seen_public_check_ids: set[str] = set()
        active_behavior_ids = set(gate.active_behavior_ids)
        active_quality_ids = set(gate.active_quality_ids)
        for index, public_check in enumerate(deliverable.public_checks):
            location = f"deliverables.{deliverable.id}.public_checks[{index}]"
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
            if _looks_like_placeholder_scenario(public_check.title, public_check.learner_goal, " ".join(public_check.expected_assertions)):
                add_issue(
                    ValidationLevel.error,
                    "placeholder_public_check",
                    location,
                    "Replace generic placeholder public checks with domain-specific learner-visible checks.",
                )
            invalid_behavior_ids = sorted(set(public_check.covers_behavior_ids) - active_behavior_ids)
            if invalid_behavior_ids:
                add_issue(
                    ValidationLevel.error,
                    "inactive_public_check_behavior",
                    f"{location}.covers_behavior_ids",
                    "Public checks can only reference active behaviors for the deliverable: "
                    + ", ".join(f"'{item}'" for item in invalid_behavior_ids),
                )
            invalid_quality_ids = sorted(set(public_check.covers_quality_ids) - active_quality_ids)
            if invalid_quality_ids:
                add_issue(
                    ValidationLevel.error,
                    "inactive_public_check_quality",
                    f"{location}.covers_quality_ids",
                    "Public checks can only reference active qualities for the deliverable: "
                    + ", ".join(f"'{item}'" for item in invalid_quality_ids),
                )
            if not public_check.covers_behavior_ids and not public_check.covers_quality_ids:
                add_issue(
                    ValidationLevel.warning,
                    "unmapped_public_check",
                    location,
                    "Public checks should map to at least one active behavior or quality so reviewers can verify intent.",
                )
        if len(deliverable.public_checks) > 4:
            add_issue(
                ValidationLevel.warning,
                "too_many_public_checks",
                f"deliverables.{deliverable.id}.public_checks",
                "Keep learner-visible checks focused. More than four checks usually signals that the deliverable should rely on the hidden grader for depth.",
            )
        if spec.assessment_strategy.hidden_grader_required and not coverage.hidden_test_ids:
            add_issue(
                ValidationLevel.error,
                "missing_hidden_grader_coverage",
                f"deliverables.{deliverable.id}",
                "Each review area needs at least one hidden grader test so submission feedback can go deeper than the public checks.",
            )
        elif spec.assessment_strategy.hidden_grader_required and not coverage.hidden_case_ids:
            add_issue(
                ValidationLevel.error,
                "missing_hidden_eval_cases",
                f"deliverables.{deliverable.id}",
                "Hidden grader coverage must exercise at least one tagged eval case for this review area.",
            )

    for case in spec.eval_dataset.cases:
        explicit_tags = set(case.tags)
        invalid_tags = sorted(explicit_tags - deliverable_ids - RESERVED_REVIEW_AREA_TAGS)
        if invalid_tags:
            add_issue(
                ValidationLevel.error,
                "unknown_eval_case_tag",
                f"eval_dataset.cases.{case.id}.tags",
                "Eval case tags must reference known review areas: "
                + ", ".join(f"'{tag}'" for tag in invalid_tags),
            )
        effective_tags = explicit_tags | set(inferred_case_tags.get(case.id, []))
        if not effective_tags:
            add_issue(
                ValidationLevel.error,
                "unmapped_eval_case",
                f"eval_dataset.cases.{case.id}",
                "Every hidden eval case must belong to at least one review area so grader feedback can map back to the learner-visible deliverable.",
            )

    for behavior in spec.behaviors:
        if behavior.first_required_in not in deliverable_ids:
            add_issue(
                ValidationLevel.error,
                "unknown_behavior_deliverable",
                f"behaviors.{behavior.id}.first_required_in",
                f"Unknown deliverable '{behavior.first_required_in}'.",
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
        if quality.first_required_in not in deliverable_ids:
            add_issue(
                ValidationLevel.error,
                "unknown_quality_deliverable",
                f"qualities.{quality.id}.first_required_in",
                f"Unknown deliverable '{quality.first_required_in}'.",
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
        deliverable_gates=_gate_summaries(spec),
    )


def compute_task_agent_gate(spec: TaskAgentServiceSpec, deliverable_id: str) -> DeliverableGate:
    return spec.gate_for(deliverable_id)


def _alignment_tokens(outcomes: list[str]) -> set[str]:
    stop_words = {
        "the",
        "and",
        "with",
        "into",
        "from",
        "that",
        "this",
        "will",
        "your",
        "each",
        "learner",
        "deliverable",
        "service",
        "system",
        "visible",
        "public",
    }
    tokens: set[str] = set()
    for outcome in outcomes:
        for token in outcome.lower().replace("`", "").replace("/", " ").split():
            cleaned = token.strip(".,:;()[]{}")
            if len(cleaned) < 5 or cleaned in stop_words:
                continue
            tokens.add(cleaned)
    return tokens


def _looks_like_placeholder_scenario(*parts: str) -> bool:
    text = " ".join(part.lower() for part in parts if part).strip()
    if not text:
        return True
    placeholder_phrases = (
        "routine case",
        "ambiguous or risky case",
        "handle the routine case cleanly",
        "handle the ambiguous or risky case",
        "generic request",
        "placeholder",
    )
    return any(phrase in text for phrase in placeholder_phrases)
