from __future__ import annotations

from collections.abc import Iterable

from app.domain.grader import (
    ControlFlag,
    GraderEntryKind,
    GraderPlanEntry,
    DeliverableGraderPlan,
    TaskAgentGraderPlanCollection,
    TestDependencies,
)
from app.domain.task_agent import (
    ApprovalGateTestParams,
    ConfidenceCalibrationJudgeTestParams,
    CostPerSuccessTestParams,
    DryRunSemanticsTestParams,
    DurableResumeTestParams,
    EscalationPolicyTestParams,
    EscalationPrecisionTestParams,
    FallbackPolicyTestParams,
    IdempotentActionTestParams,
    OutputSchemaTestParams,
    P95RunLatencyTestParams,
    QualitySpec,
    RecoveryAfterToolFailureTestParams,
    StepBudgetEnforcementTestParams,
    TaskAgentServiceSpec,
    TaskOutputQualityJudgeTestParams,
    TaskSuccessRateTestParams,
    ToolInvocationCorrectnessTestParams,
    ToolSelectionTestParams,
    TraceSchemaTestParams,
)


def build_task_agent_grader_plan(spec: TaskAgentServiceSpec, deliverable_id: str) -> DeliverableGraderPlan:
    deliverable = next((item for item in spec.deliverables if item.id == deliverable_id), None)
    if deliverable is None:
        raise ValueError(f"unknown deliverable id: {deliverable_id}")

    gate = spec.gate_for(deliverable_id)
    behaviors_by_id = {behavior.id: behavior for behavior in spec.behaviors}
    qualities_by_id = {quality.id: quality for quality in spec.qualities}

    entries: list[GraderPlanEntry] = []
    for behavior_id in gate.active_behavior_ids:
        entries.append(_behavior_entry(spec, behaviors_by_id[behavior_id]))
    for quality_id in gate.active_quality_ids:
        entries.append(_quality_entry(spec, qualities_by_id[quality_id]))

    endpoint_paths = sorted({path for entry in entries for path in entry.dependencies.endpoint_paths})
    tool_ids = sorted({tool_id for entry in entries for tool_id in entry.dependencies.tool_ids})
    controls = sorted(
        {control for entry in entries for control in entry.controls},
        key=lambda item: item.value,
    )

    return DeliverableGraderPlan(
        deliverable_id=deliverable.id,
        deliverable_title=deliverable.title,
        deliverable_objective=deliverable.objective,
        starter_type=deliverable.starter_type.value,
        overlay_ids=deliverable.overlay_ids,
        cumulative_deliverables=gate.cumulative_deliverables,
        active_behavior_ids=gate.active_behavior_ids,
        active_quality_ids=gate.active_quality_ids,
        total_tests=len(entries),
        endpoint_paths=endpoint_paths,
        tool_ids=tool_ids,
        controls=controls,
        entries=entries,
    )


def build_task_agent_review_area_plan(spec: TaskAgentServiceSpec, deliverable_id: str) -> DeliverableGraderPlan:
    deliverable = next((item for item in spec.deliverables if item.id == deliverable_id), None)
    if deliverable is None:
        raise ValueError(f"unknown deliverable id: {deliverable_id}")

    behaviors_by_id = {behavior.id: behavior for behavior in spec.behaviors}
    qualities_by_id = {quality.id: quality for quality in spec.qualities}
    active_behavior_ids = [
        behavior.id
        for behavior in spec.behaviors
        if behavior.first_required_in == deliverable_id
    ]
    active_quality_ids = [
        quality.id
        for quality in spec.qualities
        if quality.first_required_in == deliverable_id
    ]

    entries: list[GraderPlanEntry] = []
    for behavior_id in active_behavior_ids:
        entries.append(_behavior_entry(spec, behaviors_by_id[behavior_id]))
    for quality_id in active_quality_ids:
        entries.append(_quality_entry(spec, qualities_by_id[quality_id]))

    endpoint_paths = sorted({path for entry in entries for path in entry.dependencies.endpoint_paths})
    tool_ids = sorted({tool_id for entry in entries for tool_id in entry.dependencies.tool_ids})
    controls = sorted(
        {control for entry in entries for control in entry.controls},
        key=lambda item: item.value,
    )

    return DeliverableGraderPlan(
        deliverable_id=deliverable.id,
        deliverable_title=deliverable.title,
        deliverable_objective=deliverable.objective,
        starter_type=deliverable.starter_type.value,
        overlay_ids=deliverable.overlay_ids,
        cumulative_deliverables=[deliverable.id],
        active_behavior_ids=active_behavior_ids,
        active_quality_ids=active_quality_ids,
        total_tests=len(entries),
        endpoint_paths=endpoint_paths,
        tool_ids=tool_ids,
        controls=controls,
        entries=entries,
    )


def build_all_task_agent_grader_plans(spec: TaskAgentServiceSpec) -> TaskAgentGraderPlanCollection:
    return TaskAgentGraderPlanCollection(
        title=spec.title,
        eval_dataset_id=spec.eval_dataset.id,
        system_profile=spec.capabilities.summary_labels(),
        deliverable_plans=[build_task_agent_grader_plan(spec, deliverable.id) for deliverable in spec.deliverables],
    )


def build_all_task_agent_review_area_plans(spec: TaskAgentServiceSpec) -> TaskAgentGraderPlanCollection:
    return TaskAgentGraderPlanCollection(
        title=spec.title,
        eval_dataset_id=spec.eval_dataset.id,
        system_profile=spec.capabilities.summary_labels(),
        deliverable_plans=[build_task_agent_review_area_plan(spec, deliverable.id) for deliverable in spec.deliverables],
    )


def _behavior_entry(spec: TaskAgentServiceSpec, behavior) -> GraderPlanEntry:
    test = behavior.test
    dependencies = _dependencies_for_test(spec, test, kind=GraderEntryKind.behavior)
    return GraderPlanEntry(
        test_id=behavior.id,
        kind=GraderEntryKind.behavior,
        test_type=test.type,
        description=behavior.description,
        first_required_in=behavior.first_required_in,
        controls=_controls_for_test(test),
        dependencies=dependencies,
        config=test.model_dump(mode="json"),
    )


def _quality_entry(spec: TaskAgentServiceSpec, quality: QualitySpec) -> GraderPlanEntry:
    test = quality.test
    dependencies = _dependencies_for_test(spec, test, kind=GraderEntryKind.quality)
    return GraderPlanEntry(
        test_id=quality.id,
        kind=GraderEntryKind.quality,
        test_type=test.type,
        description=quality.description,
        first_required_in=quality.first_required_in,
        controls=_controls_for_test(test),
        dependencies=dependencies,
        config=test.model_dump(mode="json"),
    )


def _dependencies_for_test(spec: TaskAgentServiceSpec, test, *, kind: GraderEntryKind) -> TestDependencies:
    eval_case_ids: list[str] = []
    dataset_id: str | None = None
    tool_ids: list[str] = []
    endpoint_paths: set[str] = set()
    required_events: list[str] = []
    allowed_reasons: list[str] = []
    mutating_tool_ids: list[str] = []
    injected_failures: list[str] = []
    idempotency_key_field: str | None = None

    if kind == GraderEntryKind.behavior:
        endpoint_paths.add("/run")
    else:
        endpoint_paths.add("/eval")

    if isinstance(test, (OutputSchemaTestParams, TraceSchemaTestParams, StepBudgetEnforcementTestParams, DryRunSemanticsTestParams, IdempotentActionTestParams)):
        eval_case_ids.extend(test.case_ids)
    if isinstance(test, DurableResumeTestParams):
        eval_case_ids.append(test.case_id)
    if isinstance(test, ToolSelectionTestParams):
        eval_case_ids.extend(expectation.case_id for expectation in test.expectations)
        for expectation in test.expectations:
            tool_ids.extend(expectation.must_call_any_of)
            tool_ids.extend(expectation.must_call_all_of)
            tool_ids.extend(expectation.must_not_call)
    if isinstance(test, ToolInvocationCorrectnessTestParams):
        eval_case_ids.extend(expectation.case_id for expectation in test.expectations)
        tool_ids.extend(expectation.tool_id for expectation in test.expectations)
    if isinstance(test, EscalationPolicyTestParams):
        eval_case_ids.extend(expectation.case_id for expectation in test.expectations)
        for expectation in test.expectations:
            allowed_reasons.extend(expectation.allowed_reasons)
    if isinstance(test, ApprovalGateTestParams):
        eval_case_ids.extend(expectation.case_id for expectation in test.expectations)
        tool_ids.extend(expectation.tool_id for expectation in test.expectations)
        endpoint_paths.add("/approve/{id}")
    if isinstance(test, FallbackPolicyTestParams):
        for injection in test.injections:
            eval_case_ids.append(injection.case_id)
            injected_failures.append(
                f"{injection.target}:{injection.target_id}:{injection.failure_mode}@{injection.case_id}"
            )
            if injection.target == "tool":
                tool_ids.append(injection.target_id)
    if isinstance(test, DryRunSemanticsTestParams):
        mutating_tool_ids.extend(test.mutating_tool_ids)
        tool_ids.extend(test.mutating_tool_ids)
    if isinstance(test, DurableResumeTestParams):
        endpoint_paths.update({"/runs/{id}", "/trace/{id}"})
        if "/approve/{id}" in {endpoint.path for endpoint in spec.production_contract.canonical_endpoints}:
            endpoint_paths.add("/approve/{id}")
        required_events.append(test.interrupt_after_event.value)
    if isinstance(test, TraceSchemaTestParams):
        endpoint_paths.add("/trace/{id}")
        required_events.extend(event.value for event in test.required_events)
    if isinstance(test, IdempotentActionTestParams):
        idempotency_key_field = test.idempotency_key_field
        tool_ids.extend(
            tool.id
            for tool in spec.tool_registry.tools
            if tool.idempotency_key_arg == test.idempotency_key_field and tool.safety.value != "read"
        )
    if isinstance(test, (TaskSuccessRateTestParams, P95RunLatencyTestParams, CostPerSuccessTestParams, EscalationPrecisionTestParams, TaskOutputQualityJudgeTestParams, ConfidenceCalibrationJudgeTestParams)):
        dataset_id = test.dataset_id
    if isinstance(test, RecoveryAfterToolFailureTestParams):
        dataset_id = test.dataset_id
        for injection in test.injections:
            eval_case_ids.append(injection.case_id)
            injected_failures.append(
                f"{injection.target}:{injection.target_id}:{injection.failure_mode}@{injection.case_id}"
            )
            if injection.target == "tool":
                tool_ids.append(injection.target_id)

    return TestDependencies(
        eval_case_ids=_sorted_unique(eval_case_ids),
        dataset_id=dataset_id,
        tool_ids=_sorted_unique(tool_ids),
        endpoint_paths=sorted(endpoint_paths),
        required_events=_sorted_unique(required_events),
        allowed_reasons=_sorted_unique(allowed_reasons),
        mutating_tool_ids=_sorted_unique(mutating_tool_ids),
        injected_failures=_sorted_unique(injected_failures),
        idempotency_key_field=idempotency_key_field,
    )


def _controls_for_test(test) -> list[ControlFlag]:
    controls: set[ControlFlag] = set()
    if isinstance(test, ApprovalGateTestParams):
        controls.add(ControlFlag.approval)
    if isinstance(test, StepBudgetEnforcementTestParams):
        controls.add(ControlFlag.budget)
    if isinstance(test, CostPerSuccessTestParams):
        controls.add(ControlFlag.cost)
    if isinstance(test, DryRunSemanticsTestParams):
        controls.add(ControlFlag.dry_run)
    if isinstance(test, (EscalationPolicyTestParams, EscalationPrecisionTestParams)):
        controls.add(ControlFlag.escalation)
    if isinstance(test, (FallbackPolicyTestParams, RecoveryAfterToolFailureTestParams)):
        controls.add(ControlFlag.fault_injection)
    if isinstance(test, IdempotentActionTestParams):
        controls.add(ControlFlag.idempotency)
    if isinstance(test, (TaskOutputQualityJudgeTestParams, ConfidenceCalibrationJudgeTestParams)):
        controls.add(ControlFlag.judge)
    if isinstance(test, P95RunLatencyTestParams):
        controls.add(ControlFlag.latency)
    if isinstance(test, DurableResumeTestParams):
        controls.add(ControlFlag.resume)
    if isinstance(test, (TraceSchemaTestParams, DurableResumeTestParams)):
        controls.add(ControlFlag.trace)
    return sorted(controls, key=lambda item: item.value)


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted({item for item in items if item})
