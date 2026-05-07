from __future__ import annotations

from app.domain.task_agent import (
    ApprovalGateTestParams,
    DryRunSemanticsTestParams,
    DurableResumeTestParams,
    EndpointSpec,
    EscalationPolicyTestParams,
    FallbackPolicyTestParams,
    IdempotentActionTestParams,
    OutputSchemaTestParams,
    P95RunLatencyTestParams,
    RecoveryAfterToolFailureTestParams,
    StepBudgetEnforcementTestParams,
    TaskOutputQualityJudgeTestParams,
    TaskSuccessRateTestParams,
    ToolInvocationCorrectnessTestParams,
    ToolSafety,
    ToolSelectionTestParams,
    TraceSchemaTestParams,
    ConfidenceCalibrationJudgeTestParams,
    CostPerSuccessTestParams,
    EscalationPrecisionTestParams,
)
from app.domain.workflow import WorkflowNodeExecution, WorkflowRun


class TaskAgentRepairService:
    def apply(self, run: WorkflowRun, latest_node: WorkflowNodeExecution) -> tuple[WorkflowRun, bool, str]:
        spec = run.artifacts.task_agent_spec
        if spec is None:
            return run, False, "No task-agent spec is available to repair."

        changed = False
        notes: list[str] = []
        eval_case_ids = [case.id for case in spec.eval_dataset.cases]
        fallback_case = eval_case_ids[0] if eval_case_ids else None
        tool_ids = [tool.id for tool in spec.tool_registry.tools]
        fallback_tool = tool_ids[0] if tool_ids else None
        fallback_mutating_tool = next((tool.id for tool in spec.tool_registry.tools if tool.safety != ToolSafety.read), fallback_tool)
        module_ids = [module.id for module in spec.modules]
        first_module = module_ids[0] if module_ids else None
        last_module = module_ids[-1] if module_ids else None
        escalation_reasons = [rule.reason for rule in spec.production_contract.escalation_policy]
        fallback_reason = escalation_reasons[0] if escalation_reasons else None

        for tool in spec.tool_registry.tools:
            if tool.safety == ToolSafety.irreversible and not tool.approval_required:
                tool.approval_required = True
                changed = True
                notes.append(f"Marked irreversible tool `{tool.id}` as approval-required.")
            if tool.approval_required and tool.id not in spec.production_contract.approval_policy.require_for_tools:
                spec.production_contract.approval_policy.require_for_tools.append(tool.id)
                changed = True
                notes.append(f"Added `{tool.id}` to approval policy.")

        endpoint_paths = {endpoint.path for endpoint in spec.production_contract.canonical_endpoints}
        if any(tool.approval_required for tool in spec.tool_registry.tools) and "/approve/{id}" not in endpoint_paths:
            spec.production_contract.canonical_endpoints.append(
                EndpointSpec(method="POST", path="/approve/{id}", required=True)
            )
            changed = True
            notes.append("Added the missing approval endpoint required by the tool policy.")

        for behavior in spec.behaviors:
            if behavior.first_required_in not in module_ids and first_module is not None:
                behavior.first_required_in = first_module
                changed = True
                notes.append(f"Moved behavior `{behavior.id}` onto a valid module checkpoint.")

            test = behavior.test
            if isinstance(test, (OutputSchemaTestParams, TraceSchemaTestParams, StepBudgetEnforcementTestParams, DryRunSemanticsTestParams, IdempotentActionTestParams)):
                if fallback_case is not None:
                    valid_cases = [case_id for case_id in test.case_ids if case_id in eval_case_ids]
                    if len(valid_cases) != len(test.case_ids):
                        test.case_ids = valid_cases or [fallback_case]
                        changed = True
                        notes.append(f"Repaired eval-case references for behavior `{behavior.id}`.")

            if isinstance(test, ToolSelectionTestParams):
                for expectation in test.expectations:
                    if fallback_case is not None and expectation.case_id not in eval_case_ids:
                        expectation.case_id = fallback_case
                        changed = True
                    expectation.must_call_any_of = [tool_id for tool_id in expectation.must_call_any_of if tool_id in tool_ids]
                    expectation.must_call_all_of = [tool_id for tool_id in expectation.must_call_all_of if tool_id in tool_ids]
                    expectation.must_not_call = [tool_id for tool_id in expectation.must_not_call if tool_id in tool_ids]
                    if not expectation.must_call_any_of and not expectation.must_call_all_of and fallback_tool is not None:
                        expectation.must_call_any_of = [fallback_tool]
                        changed = True

            if isinstance(test, ToolInvocationCorrectnessTestParams):
                for expectation in test.expectations:
                    if fallback_case is not None and expectation.case_id not in eval_case_ids:
                        expectation.case_id = fallback_case
                        changed = True
                    if fallback_tool is not None and expectation.tool_id not in tool_ids:
                        expectation.tool_id = fallback_tool
                        changed = True

            if isinstance(test, EscalationPolicyTestParams):
                for expectation in test.expectations:
                    if fallback_case is not None and expectation.case_id not in eval_case_ids:
                        expectation.case_id = fallback_case
                        changed = True
                    valid_reasons = [reason for reason in expectation.allowed_reasons if reason in escalation_reasons]
                    if len(valid_reasons) != len(expectation.allowed_reasons):
                        expectation.allowed_reasons = valid_reasons or ([fallback_reason] if fallback_reason else [])
                        changed = True

            if isinstance(test, ApprovalGateTestParams):
                for expectation in test.expectations:
                    if fallback_case is not None and expectation.case_id not in eval_case_ids:
                        expectation.case_id = fallback_case
                        changed = True
                    if fallback_mutating_tool is not None and expectation.tool_id not in tool_ids:
                        expectation.tool_id = fallback_mutating_tool
                        changed = True
                    tool = next((item for item in spec.tool_registry.tools if item.id == expectation.tool_id), None)
                    if tool is not None and not tool.approval_required:
                        tool.approval_required = True
                        changed = True

            if isinstance(test, FallbackPolicyTestParams):
                for injection in test.injections:
                    if fallback_case is not None and injection.case_id not in eval_case_ids:
                        injection.case_id = fallback_case
                        changed = True
                    if injection.target == "tool" and fallback_tool is not None and injection.target_id not in tool_ids:
                        injection.target_id = fallback_tool
                        changed = True

            if isinstance(test, DurableResumeTestParams):
                if fallback_case is not None and test.case_id not in eval_case_ids:
                    test.case_id = fallback_case
                    changed = True
                if not spec.production_contract.supports_resume:
                    spec.production_contract.supports_resume = True
                    changed = True

            if isinstance(test, DryRunSemanticsTestParams):
                if not spec.production_contract.supports_dry_run:
                    spec.production_contract.supports_dry_run = True
                    changed = True
                valid_mutating_tools = [
                    tool_id
                    for tool_id in test.mutating_tool_ids
                    if tool_id in tool_ids and next((tool for tool in spec.tool_registry.tools if tool.id == tool_id), None).safety != ToolSafety.read
                ]
                if len(valid_mutating_tools) != len(test.mutating_tool_ids):
                    if fallback_mutating_tool is not None and fallback_mutating_tool not in valid_mutating_tools:
                        valid_mutating_tools.append(fallback_mutating_tool)
                    test.mutating_tool_ids = valid_mutating_tools
                    changed = True

        for quality in spec.qualities:
            if quality.first_required_in not in module_ids and last_module is not None:
                quality.first_required_in = last_module
                changed = True
                notes.append(f"Moved quality `{quality.id}` onto the final valid module.")

            test = quality.test
            if isinstance(
                test,
                (
                    TaskSuccessRateTestParams,
                    P95RunLatencyTestParams,
                    CostPerSuccessTestParams,
                    EscalationPrecisionTestParams,
                    TaskOutputQualityJudgeTestParams,
                    ConfidenceCalibrationJudgeTestParams,
                ),
            ):
                if test.dataset_id != spec.eval_dataset.id:
                    test.dataset_id = spec.eval_dataset.id
                    changed = True
                    notes.append(f"Repaired dataset binding for quality `{quality.id}`.")

            if isinstance(test, RecoveryAfterToolFailureTestParams):
                if test.dataset_id != spec.eval_dataset.id:
                    test.dataset_id = spec.eval_dataset.id
                    changed = True
                for injection in test.injections:
                    if fallback_case is not None and injection.case_id not in eval_case_ids:
                        injection.case_id = fallback_case
                        changed = True
                    if injection.target == "tool" and fallback_tool is not None and injection.target_id not in tool_ids:
                        injection.target_id = fallback_tool
                        changed = True

        if changed:
            run.notes.append(
                f"Automatic repair applied after `{latest_node.kind.value}` attempt {latest_node.attempt}: {'; '.join(dict.fromkeys(notes)) or 'generic spec normalization.'}"
            )
            run.artifacts.notes.append(
                f"Auto-repair completed for `{latest_node.kind.value}` attempt {latest_node.attempt}."
            )
            return run, True, notes[0] if notes else "Applied automatic task-agent repairs."

        return run, False, "No deterministic repair was available for the current failures."
