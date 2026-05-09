from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

from app.domain.grading import (
    AssignmentGradeReport,
    EvalRunEvidence,
    GradeStatus,
    DeliverableGradeReport,
    ReviewAreaGradeReport,
    TaskAgentSubmission,
    TestGradeResult,
    ToolCallStatus,
)
from app.domain.task_agent import TaskAgentServiceSpec
from app.services.grader_planner import build_all_task_agent_review_area_plans, build_task_agent_grader_plan


def grade_task_agent_submission(
    spec: TaskAgentServiceSpec,
    deliverable_id: str,
    submission: TaskAgentSubmission,
) -> DeliverableGradeReport:
    plan = build_task_agent_grader_plan(spec, deliverable_id)
    grouped_runs = _group_runs_by_case(submission)
    primary_runs = _primary_runs(spec, grouped_runs)
    warnings = _submission_warnings(spec, submission, primary_runs)

    results: list[TestGradeResult] = []
    for entry in plan.entries:
        results.append(_grade_entry(spec, entry, grouped_runs, primary_runs))

    passed_tests = sum(1 for result in results if result.status == GradeStatus.passed)
    total_tests = len(results)
    failed_tests = total_tests - passed_tests
    pass_rate = passed_tests / total_tests if total_tests else 0.0
    status = GradeStatus.passed if failed_tests == 0 else GradeStatus.failed

    return DeliverableGradeReport(
        deliverable_id=deliverable_id,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        pass_rate=pass_rate,
        status=status,
        results=results,
        submission_warnings=warnings,
    )


def grade_assignment_submission(
    spec: TaskAgentServiceSpec,
    submission: TaskAgentSubmission,
) -> AssignmentGradeReport:
    plan_collection = build_all_task_agent_review_area_plans(spec)
    grouped_runs = _group_runs_by_case(submission)
    primary_runs = _primary_runs(spec, grouped_runs)
    warnings = _submission_warnings(spec, submission, primary_runs)

    review_areas: list[ReviewAreaGradeReport] = []
    for plan in plan_collection.deliverable_plans:
        results: list[TestGradeResult] = []
        for entry in plan.entries:
            results.append(_grade_entry(spec, entry, grouped_runs, primary_runs))

        passed_tests = sum(1 for result in results if result.status == GradeStatus.passed)
        total_tests = len(results)
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests else 0.0
        status = GradeStatus.passed if failed_tests == 0 else GradeStatus.failed
        review_areas.append(
            ReviewAreaGradeReport(
                deliverable_id=plan.deliverable_id,
                title=plan.deliverable_title,
                objective=plan.deliverable_objective,
                deliverable_index=spec.deliverable_order[plan.deliverable_id] + 1,
                grade_report=DeliverableGradeReport(
                    deliverable_id=plan.deliverable_id,
                    total_tests=total_tests,
                    passed_tests=passed_tests,
                    failed_tests=failed_tests,
                    pass_rate=pass_rate,
                    status=status,
                    results=results,
                    submission_warnings=list(warnings),
                ),
            )
        )

    passed_tests = sum(area.grade_report.passed_tests for area in review_areas)
    total_tests = sum(area.grade_report.total_tests for area in review_areas)
    failed_tests = total_tests - passed_tests
    pass_rate = passed_tests / total_tests if total_tests else 0.0
    status = GradeStatus.passed if failed_tests == 0 else GradeStatus.failed
    return AssignmentGradeReport(
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        pass_rate=pass_rate,
        status=status,
        review_areas=review_areas,
        submission_warnings=warnings,
    )


def _grade_entry(spec: TaskAgentServiceSpec, entry, grouped_runs, primary_runs) -> TestGradeResult:
    config = entry.config
    test_type = entry.test_type
    if test_type == "output_schema_test":
        return _grade_output_schema(spec, entry, grouped_runs, primary_runs, config)
    if test_type == "trace_schema_test":
        return _grade_trace_schema(entry, primary_runs, config)
    if test_type == "tool_selection_test":
        return _grade_tool_selection(entry, primary_runs, config)
    if test_type == "tool_invocation_correctness_test":
        return _grade_tool_invocation(entry, primary_runs, config)
    if test_type == "step_budget_enforcement_test":
        return _grade_step_budget(entry, primary_runs, config)
    if test_type == "escalation_policy_test":
        return _grade_escalation(entry, primary_runs, config)
    if test_type == "approval_gate_test":
        return _grade_approval(entry, primary_runs, config)
    if test_type == "fallback_policy_test":
        return _grade_fallback(entry, primary_runs, config)
    if test_type == "durable_resume_test":
        return _grade_durable_resume(entry, primary_runs, config)
    if test_type == "dry_run_semantics_test":
        return _grade_dry_run(entry, grouped_runs, config)
    if test_type == "idempotent_action_test":
        return _grade_idempotency(entry, primary_runs, config)
    if test_type == "task_success_rate_test":
        return _grade_success_rate(spec, entry, primary_runs, config)
    if test_type == "p95_run_latency_test":
        return _grade_latency(spec, entry, primary_runs, config)
    if test_type == "cost_per_success_test":
        return _grade_cost(spec, entry, primary_runs, config)
    if test_type == "recovery_after_tool_failure_test":
        return _grade_recovery_after_fault(entry, primary_runs, config)
    if test_type == "escalation_precision_test":
        return _grade_escalation_precision(spec, entry, primary_runs, config)
    if test_type == "task_output_quality_judge_test":
        return _grade_quality(entry, primary_runs, config)
    if test_type == "confidence_calibration_judge_test":
        return _grade_confidence_calibration(spec, entry, primary_runs, config)

    return _result(
        entry=entry,
        passed=False,
        score=0.0,
        summary=f"Unsupported test type '{test_type}'.",
        diagnostics=[f"No runtime evaluator exists yet for '{test_type}'."],
    )


def _grade_output_schema(spec: TaskAgentServiceSpec, entry, grouped_runs, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    case_ids = config.get("case_ids", [])
    case_map = {case.id: case for case in spec.eval_dataset.cases}
    diagnostics: list[str] = []
    passed = 0
    for case_id in case_ids:
        run = primary_runs.get(case_id) or _first_run(grouped_runs, case_id)
        if run is None:
            diagnostics.append(f"Missing run evidence for case '{case_id}'.")
            continue
        schema_errors = _validate_object_against_schema(spec.output_schema, run.output)
        if schema_errors:
            diagnostics.append(f"Case '{case_id}' output failed schema validation: {', '.join(schema_errors)}")
            continue
        expected_output = case_map[case_id].expected_output or {}
        if expected_output and not _contains_subset(run.output, expected_output):
            diagnostics.append(f"Case '{case_id}' output does not match expected subset {expected_output}.")
            continue
        passed += 1
    score = passed / len(case_ids) if case_ids else 0.0
    return _result(entry, passed == len(case_ids), score, f"{passed}/{len(case_ids)} outputs satisfy the contract.", diagnostics)


def _grade_trace_schema(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    case_ids = config.get("case_ids", [])
    required_events = set(config.get("required_events", []))
    diagnostics: list[str] = []
    passed = 0
    for case_id in case_ids:
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        seen = set(run.trace_events)
        missing = required_events - seen
        if missing:
            diagnostics.append(f"Case '{case_id}' is missing trace events: {sorted(missing)}")
            continue
        passed += 1
    score = passed / len(case_ids) if case_ids else 0.0
    return _result(entry, passed == len(case_ids), score, f"{passed}/{len(case_ids)} runs expose the required trace events.", diagnostics)


def _grade_tool_selection(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    expectations = config.get("expectations", [])
    diagnostics: list[str] = []
    passed = 0
    for expectation in expectations:
        case_id = expectation["case_id"]
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        called_tools = {call.tool_id for call in run.tool_calls if call.status != ToolCallStatus.skipped}
        any_of = set(expectation.get("must_call_any_of", []))
        all_of = set(expectation.get("must_call_all_of", []))
        must_not = set(expectation.get("must_not_call", []))
        if any_of and not (called_tools & any_of):
            diagnostics.append(f"Case '{case_id}' did not call any of {sorted(any_of)}.")
            continue
        if all_of and not all_of.issubset(called_tools):
            diagnostics.append(f"Case '{case_id}' did not call all of {sorted(all_of)}.")
            continue
        if must_not & called_tools:
            diagnostics.append(f"Case '{case_id}' called forbidden tools {sorted(must_not & called_tools)}.")
            continue
        passed += 1
    score = passed / len(expectations) if expectations else 0.0
    return _result(entry, passed == len(expectations), score, f"{passed}/{len(expectations)} tool-choice expectations passed.", diagnostics)


def _grade_tool_invocation(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    expectations = config.get("expectations", [])
    diagnostics: list[str] = []
    passed = 0
    for expectation in expectations:
        case_id = expectation["case_id"]
        tool_id = expectation["tool_id"]
        required_subset = expectation.get("required_args_subset", {})
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        matching_call = next(
            (
                call
                for call in run.tool_calls
                if call.tool_id == tool_id and _contains_subset(call.args, required_subset)
            ),
            None,
        )
        if matching_call is None:
            diagnostics.append(f"Case '{case_id}' never invoked '{tool_id}' with args matching {required_subset}.")
            continue
        passed += 1
    score = passed / len(expectations) if expectations else 0.0
    return _result(entry, passed == len(expectations), score, f"{passed}/{len(expectations)} tool-invocation checks passed.", diagnostics)


def _grade_step_budget(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    case_ids = config.get("case_ids", [])
    max_steps = config.get("max_steps", 0)
    diagnostics: list[str] = []
    passed = 0
    for case_id in case_ids:
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        if run.step_count > max_steps:
            diagnostics.append(f"Case '{case_id}' used {run.step_count} steps, above the cap of {max_steps}.")
            continue
        passed += 1
    score = passed / len(case_ids) if case_ids else 0.0
    return _result(entry, passed == len(case_ids), score, f"{passed}/{len(case_ids)} runs stayed within budget.", diagnostics)


def _grade_escalation(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    expectations = config.get("expectations", [])
    diagnostics: list[str] = []
    passed = 0
    for expectation in expectations:
        case_id = expectation["case_id"]
        must_escalate = expectation["must_escalate"]
        allowed_reasons = set(expectation.get("allowed_reasons", []))
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        reasons = {record.reason for record in run.escalations}
        if must_escalate and not reasons:
            diagnostics.append(f"Case '{case_id}' did not escalate.")
            continue
        if not must_escalate and reasons:
            diagnostics.append(f"Case '{case_id}' escalated unexpectedly.")
            continue
        if allowed_reasons and reasons and not (reasons & allowed_reasons):
            diagnostics.append(f"Case '{case_id}' escalated with {sorted(reasons)}, outside allowed reasons {sorted(allowed_reasons)}.")
            continue
        passed += 1
    score = passed / len(expectations) if expectations else 0.0
    return _result(entry, passed == len(expectations), score, f"{passed}/{len(expectations)} escalation checks passed.", diagnostics)


def _grade_approval(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    expectations = config.get("expectations", [])
    diagnostics: list[str] = []
    passed = 0
    for expectation in expectations:
        case_id = expectation["case_id"]
        tool_id = expectation["tool_id"]
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        call = next((item for item in run.tool_calls if item.tool_id == tool_id and item.status in {ToolCallStatus.ok, ToolCallStatus.deduplicated}), None)
        if call is None:
            diagnostics.append(f"Case '{case_id}' never executed '{tool_id}'.")
            continue
        approved = any(
            record.tool_id == tool_id and record.approved and record.order <= call.order
            for record in run.approvals
        )
        if not approved:
            diagnostics.append(f"Case '{case_id}' called '{tool_id}' before approval.")
            continue
        passed += 1
    score = passed / len(expectations) if expectations else 0.0
    return _result(entry, passed == len(expectations), score, f"{passed}/{len(expectations)} approval checks passed.", diagnostics)


def _grade_fallback(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    injections = config.get("injections", [])
    minimum = config.get("min_success_after_fallback", 1.0)
    diagnostics: list[str] = []
    matched_runs: list[EvalRunEvidence] = []
    for injection in injections:
        case_id = injection["case_id"]
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        found = any(
            item.target == injection["target"]
            and item.target_id == injection["target_id"]
            and item.failure_mode == injection["failure_mode"]
            for item in run.failure_injections
        )
        if not found:
            diagnostics.append(
                f"Case '{case_id}' is missing injected failure {injection['target']}:{injection['target_id']}:{injection['failure_mode']}."
            )
            continue
        if not run.fallback_actions:
            diagnostics.append(f"Case '{case_id}' has no fallback action recorded.")
            continue
        matched_runs.append(run)
    success_rate = (sum(1 for run in matched_runs if run.success) / len(matched_runs)) if matched_runs else 0.0
    if matched_runs and success_rate < minimum:
        diagnostics.append(f"Fallback success rate was {success_rate:.2f}, below {minimum:.2f}.")
    passed = bool(matched_runs) and not diagnostics and success_rate >= minimum
    return _result(entry, passed, success_rate, f"Fallback success rate was {success_rate:.2f}.", diagnostics)


def _grade_durable_resume(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    case_id = config.get("case_id")
    interrupt_after = config.get("interrupt_after_event")
    run = primary_runs.get(case_id)
    diagnostics: list[str] = []
    if run is None:
        diagnostics.append(f"Missing primary run for case '{case_id}'.")
        return _result(entry, False, 0.0, "No matching run was submitted.", diagnostics)
    if interrupt_after not in run.trace_events:
        diagnostics.append(f"Run never reached interrupt event '{interrupt_after}'.")
    if "run_completed" not in run.trace_events:
        diagnostics.append("Run never completed after the pause.")
    if not run.resumed_after_pause:
        diagnostics.append("Run is not marked as resumed after pause.")
    return _result(entry, not diagnostics, 1.0 if not diagnostics else 0.0, "Resume semantics verified." if not diagnostics else "Resume semantics failed.", diagnostics)


def _grade_dry_run(entry, grouped_runs, config: dict[str, Any]) -> TestGradeResult:
    case_ids = config.get("case_ids", [])
    mutating_tool_ids = set(config.get("mutating_tool_ids", []))
    diagnostics: list[str] = []
    passed = 0
    for case_id in case_ids:
        run = next((item for item in grouped_runs.get(case_id, []) if item.dry_run), None)
        if run is None:
            diagnostics.append(f"Missing dry-run evidence for case '{case_id}'.")
            continue
        illegal_calls = [
            call.tool_id
            for call in run.tool_calls
            if call.tool_id in mutating_tool_ids and call.status not in {ToolCallStatus.skipped, ToolCallStatus.preview}
        ]
        if illegal_calls:
            diagnostics.append(f"Case '{case_id}' executed mutating tools during dry-run: {sorted(set(illegal_calls))}.")
            continue
        passed += 1
    score = passed / len(case_ids) if case_ids else 0.0
    return _result(entry, passed == len(case_ids), score, f"{passed}/{len(case_ids)} dry-run checks passed.", diagnostics)


def _grade_idempotency(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    case_ids = config.get("case_ids", [])
    diagnostics: list[str] = []
    passed = 0
    for case_id in case_ids:
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        key_values = defaultdict(list)
        for call in run.tool_calls:
            if call.idempotency_key:
                key_values[(call.tool_id, call.idempotency_key)].append(call)
        duplicate_found = any(
            len(calls) > 1 and any(call.deduplicated or call.status == ToolCallStatus.deduplicated for call in calls)
            for calls in key_values.values()
        )
        if not duplicate_found:
            diagnostics.append(f"Case '{case_id}' does not show duplicate suppression for any idempotent action.")
            continue
        passed += 1
    score = passed / len(case_ids) if case_ids else 0.0
    return _result(entry, passed == len(case_ids), score, f"{passed}/{len(case_ids)} idempotency checks passed.", diagnostics)


def _grade_success_rate(spec: TaskAgentServiceSpec, entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    runs = _ordered_primary_runs(spec, primary_runs)
    success_rate = (sum(1 for run in runs if run.success) / len(runs)) if runs else 0.0
    minimum = config.get("min_success_rate", 1.0)
    diagnostics = [] if runs else ["No primary runs were submitted for the eval dataset."]
    if runs and success_rate < minimum:
        diagnostics.append(f"Success rate was {success_rate:.2f}, below {minimum:.2f}.")
    return _result(entry, bool(runs) and success_rate >= minimum, success_rate, f"Success rate was {success_rate:.2f}.", diagnostics)


def _grade_latency(spec: TaskAgentServiceSpec, entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    runs = _ordered_primary_runs(spec, primary_runs)
    diagnostics: list[str] = []
    if not runs:
        diagnostics.append("No primary runs were submitted for latency measurement.")
        return _result(entry, False, 0.0, "No latency evidence submitted.", diagnostics)
    latencies = [run.latency_ms for run in runs]
    p95 = _percentile(latencies, 0.95)
    threshold = config.get("p95_ms", 0)
    if p95 > threshold:
        diagnostics.append(f"Observed p95 latency {p95:.0f}ms exceeds threshold {threshold}ms.")
    score = min(1.0, threshold / p95) if p95 else 1.0
    return _result(entry, p95 <= threshold, score, f"Observed p95 latency was {p95:.0f}ms.", diagnostics)


def _grade_cost(spec: TaskAgentServiceSpec, entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    runs = [run for run in _ordered_primary_runs(spec, primary_runs) if run.success]
    diagnostics: list[str] = []
    if not runs:
        diagnostics.append("No successful runs were submitted for cost measurement.")
        return _result(entry, False, 0.0, "No successful runs available for cost measurement.", diagnostics)
    average_cost = mean(run.cost_usd for run in runs)
    threshold = config.get("max_cost_usd", 0.0)
    if average_cost > threshold:
        diagnostics.append(f"Average cost per successful run was ${average_cost:.3f}, above ${threshold:.3f}.")
    score = min(1.0, threshold / average_cost) if average_cost else 1.0
    return _result(entry, average_cost <= threshold, score, f"Average cost per successful run was ${average_cost:.3f}.", diagnostics)


def _grade_recovery_after_fault(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    injections = config.get("injections", [])
    minimum = config.get("min_success_rate_after_faults", 1.0)
    diagnostics: list[str] = []
    matched_runs: list[EvalRunEvidence] = []
    for injection in injections:
        case_id = injection["case_id"]
        run = primary_runs.get(case_id)
        if run is None:
            diagnostics.append(f"Missing primary run for case '{case_id}'.")
            continue
        found = any(
            item.target == injection["target"]
            and item.target_id == injection["target_id"]
            and item.failure_mode == injection["failure_mode"]
            for item in run.failure_injections
        )
        if not found:
            diagnostics.append(
                f"Case '{case_id}' is missing fault evidence {injection['target']}:{injection['target_id']}:{injection['failure_mode']}."
            )
            continue
        matched_runs.append(run)
    success_rate = (sum(1 for run in matched_runs if run.success) / len(matched_runs)) if matched_runs else 0.0
    if matched_runs and success_rate < minimum:
        diagnostics.append(f"Recovery success rate was {success_rate:.2f}, below {minimum:.2f}.")
    passed = bool(matched_runs) and not diagnostics and success_rate >= minimum
    return _result(entry, passed, success_rate, f"Recovery success rate was {success_rate:.2f}.", diagnostics)


def _grade_escalation_precision(spec: TaskAgentServiceSpec, entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    case_map = {case.id: case for case in spec.eval_dataset.cases}
    escalated_runs = [run for run in _ordered_primary_runs(spec, primary_runs) if run.escalations]
    diagnostics: list[str] = []
    if not escalated_runs:
        diagnostics.append("No escalations were submitted, so escalation precision is undefined.")
        return _result(entry, False, 0.0, "No escalations available for precision measurement.", diagnostics)
    precision = sum(1 for run in escalated_runs if case_map[run.case_id].should_escalate) / len(escalated_runs)
    minimum = config.get("min_precision", 1.0)
    if precision < minimum:
        diagnostics.append(f"Escalation precision was {precision:.2f}, below {minimum:.2f}.")
    return _result(entry, precision >= minimum, precision, f"Escalation precision was {precision:.2f}.", diagnostics)


def _grade_quality(entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    runs = list(primary_runs.values())
    diagnostics: list[str] = []
    scores = [run.quality_score for run in runs if run.quality_score is not None]
    if len(scores) != len(runs) or not scores:
        diagnostics.append("Every primary run needs a quality_score for this judge-based test.")
        average_score = mean(scores) if scores else 0.0
        return _result(entry, False, average_score, f"Average quality score was {average_score:.2f}.", diagnostics)
    average_score = mean(scores)
    minimum = config.get("min_avg_score", 1.0)
    if average_score < minimum:
        diagnostics.append(f"Average quality score was {average_score:.2f}, below {minimum:.2f}.")
    return _result(entry, average_score >= minimum, average_score, f"Average quality score was {average_score:.2f}.", diagnostics)


def _grade_confidence_calibration(spec: TaskAgentServiceSpec, entry, primary_runs, config: dict[str, Any]) -> TestGradeResult:
    runs = _ordered_primary_runs(spec, primary_runs)
    diagnostics: list[str] = []
    errors: list[float] = []
    for run in runs:
        confidence = run.output.get("confidence")
        if not isinstance(confidence, (int, float)):
            diagnostics.append(f"Case '{run.case_id}' does not include numeric confidence.")
            continue
        outcome = 1.0 if run.success else 0.0
        errors.append(abs(float(confidence) - outcome))
    if not errors:
        return _result(entry, False, 0.0, "No calibration evidence submitted.", diagnostics or ["No numeric confidence values were found."])
    average_error = mean(errors)
    maximum = config.get("max_expected_calibration_error", 0.0)
    if average_error > maximum:
        diagnostics.append(f"Average calibration error was {average_error:.2f}, above {maximum:.2f}.")
    score = max(0.0, 1.0 - average_error)
    return _result(entry, average_error <= maximum, score, f"Average calibration error was {average_error:.2f}.", diagnostics)


def _group_runs_by_case(submission: TaskAgentSubmission) -> dict[str, list[EvalRunEvidence]]:
    grouped: dict[str, list[EvalRunEvidence]] = defaultdict(list)
    for run in submission.runs:
        grouped[run.case_id].append(run)
    return grouped


def _primary_runs(spec: TaskAgentServiceSpec, grouped_runs: dict[str, list[EvalRunEvidence]]) -> dict[str, EvalRunEvidence]:
    primary: dict[str, EvalRunEvidence] = {}
    for case in spec.eval_dataset.cases:
        run = next((item for item in grouped_runs.get(case.id, []) if not item.dry_run), None)
        if run is not None:
            primary[case.id] = run
    return primary


def _ordered_primary_runs(spec: TaskAgentServiceSpec, primary_runs: dict[str, EvalRunEvidence]) -> list[EvalRunEvidence]:
    return [primary_runs[case.id] for case in spec.eval_dataset.cases if case.id in primary_runs]


def _submission_warnings(
    spec: TaskAgentServiceSpec,
    submission: TaskAgentSubmission,
    primary_runs: dict[str, EvalRunEvidence],
) -> list[str]:
    warnings: list[str] = []
    known_cases = {case.id for case in spec.eval_dataset.cases}
    known_tools = spec.tool_ids

    extra_cases = sorted({run.case_id for run in submission.runs if run.case_id not in known_cases})
    if extra_cases:
        warnings.append(f"Submission contains runs for unknown eval cases: {extra_cases}")

    missing_cases = [case.id for case in spec.eval_dataset.cases if case.id not in primary_runs]
    if missing_cases:
        warnings.append(f"Submission is missing primary non-dry-run evidence for: {missing_cases}")

    unknown_tools = sorted(
        {
            call.tool_id
            for run in submission.runs
            for call in run.tool_calls
            if call.tool_id not in known_tools
        }
    )
    if unknown_tools:
        warnings.append(f"Submission references unknown tools: {unknown_tools}")

    return warnings


def _first_run(grouped_runs: dict[str, list[EvalRunEvidence]], case_id: str) -> EvalRunEvidence | None:
    runs = grouped_runs.get(case_id, [])
    return runs[0] if runs else None


def _validate_object_against_schema(schema: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    for field in required:
        if field not in payload:
            errors.append(f"missing required field '{field}'")

    for field, rules in properties.items():
        if field not in payload:
            continue
        value = payload[field]
        expected_type = rules.get("type")
        if expected_type and not _matches_type(expected_type, value):
            errors.append(f"field '{field}' should be {expected_type}")
            continue
        if "enum" in rules and value not in rules["enum"]:
            errors.append(f"field '{field}' must be one of {rules['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rules and float(value) < float(rules["minimum"]):
                errors.append(f"field '{field}' must be >= {rules['minimum']}")
            if "maximum" in rules and float(value) > float(rules["maximum"]):
                errors.append(f"field '{field}' must be <= {rules['maximum']}")
    return errors


def _matches_type(expected_type: str, value: Any) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def _contains_subset(candidate: Any, expected_subset: Any) -> bool:
    if isinstance(expected_subset, dict):
        if not isinstance(candidate, dict):
            return False
        return all(
            key in candidate and _contains_subset(candidate[key], value)
            for key, value in expected_subset.items()
        )
    if isinstance(expected_subset, list):
        if not isinstance(candidate, list) or len(candidate) < len(expected_subset):
            return False
        return all(_contains_subset(actual, expected) for actual, expected in zip(candidate, expected_subset, strict=False))
    return candidate == expected_subset


def _percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[rank])


def _result(entry, passed: bool, score: float, summary: str, diagnostics: list[str]) -> TestGradeResult:
    return TestGradeResult(
        test_id=entry.test_id,
        test_type=entry.test_type,
        kind=entry.kind.value,
        status=GradeStatus.passed if passed else GradeStatus.failed,
        score=max(0.0, min(1.0, score)),
        summary=summary,
        diagnostics=diagnostics,
    )
