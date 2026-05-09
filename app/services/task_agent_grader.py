from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from app.domain.grading import (
    AssignmentGradeReport,
    DeliverableGradeReport,
    EvalRunEvidence,
    GradeStatus,
    ReviewAreaGradeReport,
    TaskAgentSubmission,
    TestGradeResult,
)
from app.domain.task_agent import PublicCheckSpec, TaskAgentServiceSpec
from app.services.grader_planner import build_all_task_agent_review_area_plans, build_task_agent_grader_plan


def grade_task_agent_submission(
    spec: TaskAgentServiceSpec,
    deliverable_id: str,
    submission: TaskAgentSubmission,
) -> DeliverableGradeReport:
    deliverable = _deliverable_by_id(spec, deliverable_id)
    plan = build_task_agent_grader_plan(spec, deliverable_id)
    grouped_runs = _group_runs_by_case(submission)
    primary_runs = _primary_runs(grouped_runs)
    warnings = _submission_warnings(deliverable.public_checks, grouped_runs)

    results = [
        _grade_public_check(entry.test_id, entry.config, grouped_runs, primary_runs)
        for entry in plan.entries
    ]

    return _deliverable_grade_report(
        deliverable_id=deliverable_id,
        results=results,
        warnings=warnings,
    )


def grade_assignment_submission(
    spec: TaskAgentServiceSpec,
    submission: TaskAgentSubmission,
) -> AssignmentGradeReport:
    plans = build_all_task_agent_review_area_plans(spec)
    grouped_runs = _group_runs_by_case(submission)
    primary_runs = _primary_runs(grouped_runs)

    review_areas: list[ReviewAreaGradeReport] = []
    warnings_by_deliverable: dict[str, list[str]] = defaultdict(list)
    for deliverable in spec.deliverables:
        warnings_by_deliverable[deliverable.id] = _submission_warnings(deliverable.public_checks, grouped_runs)

    for plan in plans.deliverable_plans:
        results = [
            _grade_public_check(entry.test_id, entry.config, grouped_runs, primary_runs)
            for entry in plan.entries
        ]
        grade_report = _deliverable_grade_report(
            deliverable_id=plan.deliverable_id,
            results=results,
            warnings=warnings_by_deliverable.get(plan.deliverable_id, []),
        )
        review_areas.append(
            ReviewAreaGradeReport(
                deliverable_id=plan.deliverable_id,
                title=plan.deliverable_title,
                objective=plan.deliverable_objective,
                deliverable_index=spec.deliverable_order[plan.deliverable_id] + 1,
                grade_report=grade_report,
            )
        )

    passed_tests = sum(area.grade_report.passed_tests for area in review_areas)
    total_tests = sum(area.grade_report.total_tests for area in review_areas)
    failed_tests = total_tests - passed_tests
    pass_rate = passed_tests / total_tests if total_tests else 0.0
    status = GradeStatus.passed if failed_tests == 0 else GradeStatus.failed
    submission_warnings: list[str] = []
    for warnings in warnings_by_deliverable.values():
        for warning in warnings:
            if warning not in submission_warnings:
                submission_warnings.append(warning)

    return AssignmentGradeReport(
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        pass_rate=pass_rate,
        status=status,
        review_areas=review_areas,
        submission_warnings=submission_warnings,
    )


def _deliverable_by_id(spec: TaskAgentServiceSpec, deliverable_id: str):
    deliverable = next((item for item in spec.deliverables if item.id == deliverable_id), None)
    if deliverable is None:
        raise ValueError(f"unknown deliverable id: {deliverable_id}")
    return deliverable


def _deliverable_grade_report(
    *,
    deliverable_id: str,
    results: list[TestGradeResult],
    warnings: list[str],
) -> DeliverableGradeReport:
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


def _grade_public_check(
    test_id: str,
    config: dict[str, Any],
    grouped_runs: dict[str, list[EvalRunEvidence]],
    primary_runs: dict[str, EvalRunEvidence],
) -> TestGradeResult:
    run = primary_runs.get(test_id) or _first_run(grouped_runs, test_id)
    if run is None:
        return _result(
            test_id=test_id,
            passed=False,
            summary="No submission evidence was recorded for this visible check.",
            diagnostics=[f"Missing run evidence for public check '{test_id}'."],
        )

    expected_status = int(config.get("expected_status") or 200)
    expected_snippets = [str(item).strip() for item in config.get("expected_response_contains", []) if str(item).strip()]
    diagnostics: list[str] = []

    actual_status = _response_status(run)
    if actual_status is not None and actual_status != expected_status:
        diagnostics.append(f"Expected HTTP {expected_status} but observed HTTP {actual_status}.")

    haystack = _response_haystack(run)
    missing_snippets = [snippet for snippet in expected_snippets if snippet.lower() not in haystack.lower()]
    if missing_snippets:
        diagnostics.append(
            "Response is missing expected content: "
            + ", ".join(sorted(missing_snippets))
            + "."
        )

    if not run.success and not diagnostics:
        diagnostics.append(run.notes[0] if run.notes else "The submitted behavior did not satisfy the visible check.")

    passed = not diagnostics
    score = 1.0 if passed else 0.0
    summary = "Visible check passed." if passed else "Visible check failed."
    return _result(
        test_id=test_id,
        passed=passed,
        score=score,
        summary=summary,
        diagnostics=diagnostics or list(run.notes),
    )


def _response_status(run: EvalRunEvidence) -> int | None:
    status = run.output.get("_coursegen_http_status")
    if status is None:
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _response_haystack(run: EvalRunEvidence) -> str:
    if "_coursegen_body_text" in run.output:
        return str(run.output.get("_coursegen_body_text") or "")
    return json.dumps(run.output, sort_keys=True, ensure_ascii=True)


def _submission_warnings(
    public_checks: list[PublicCheckSpec],
    grouped_runs: dict[str, list[EvalRunEvidence]],
) -> list[str]:
    warnings: list[str] = []
    expected_ids = {check.id for check in public_checks}
    observed_ids = set(grouped_runs)

    unexpected = sorted(observed_ids - expected_ids)
    if unexpected:
        warnings.append(
            "Ignoring submission evidence that does not map to a visible check: "
            + ", ".join(unexpected)
            + "."
        )

    missing = sorted(check.id for check in public_checks if check.id not in grouped_runs)
    if missing:
        warnings.append(
            "Some visible checks have no submission evidence yet: "
            + ", ".join(missing)
            + "."
        )
    return warnings


def _group_runs_by_case(submission: TaskAgentSubmission) -> dict[str, list[EvalRunEvidence]]:
    grouped: dict[str, list[EvalRunEvidence]] = defaultdict(list)
    for run in submission.runs:
        grouped[run.case_id].append(run)
    return grouped


def _primary_runs(grouped_runs: dict[str, list[EvalRunEvidence]]) -> dict[str, EvalRunEvidence]:
    return {
        case_id: next((run for run in runs if not run.dry_run), runs[0])
        for case_id, runs in grouped_runs.items()
        if runs
    }


def _first_run(grouped_runs: dict[str, list[EvalRunEvidence]], case_id: str) -> EvalRunEvidence | None:
    runs = grouped_runs.get(case_id) or []
    return runs[0] if runs else None


def _result(
    *,
    test_id: str,
    passed: bool,
    summary: str,
    diagnostics: list[str],
    score: float | None = None,
) -> TestGradeResult:
    return TestGradeResult(
        test_id=test_id,
        test_type="public_check",
        kind="behavior",
        status=GradeStatus.passed if passed else GradeStatus.failed,
        score=1.0 if passed and score is None else (0.0 if score is None else score),
        summary=summary,
        diagnostics=diagnostics,
    )
