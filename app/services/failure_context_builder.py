from __future__ import annotations

from collections.abc import Iterable

from app.domain.workflow import (
    FailureContext,
    FailureContextModuleReport,
    FailureContextSandboxSummary,
    FailureContextValidationIssue,
    ReviewerFinding,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowRun,
)

_REVIEWER_KINDS = {
    WorkflowNodeKind.reviewer_runtime,
    WorkflowNodeKind.reviewer_code,
    WorkflowNodeKind.reviewer_pedagogy,
    WorkflowNodeKind.reviewer_tests,
}
_AUTHORING_KINDS = {
    WorkflowNodeKind.authoring_runtime,
}


def build_failure_context(
    run: WorkflowRun,
    latest_node: WorkflowNodeExecution,
) -> FailureContext:
    related_nodes = _related_nodes(run, latest_node)
    findings = _dedupe_findings(
        finding
        for node in related_nodes
        for finding in node.findings
    )
    validation_issues = _validation_issues(run.artifacts.validation_summary or {})
    sandbox = _sandbox_summary(latest_node)
    return FailureContext(
        source_node_kind=latest_node.kind,
        source_node_attempt=latest_node.attempt,
        source_summary=latest_node.summary,
        findings=findings,
        validation_issues=validation_issues,
        sandbox=sandbox,
    )


def _related_nodes(run: WorkflowRun, latest_node: WorkflowNodeExecution) -> list[WorkflowNodeExecution]:
    if latest_node.kind in _REVIEWER_KINDS:
        allowed_kinds = _REVIEWER_KINDS
    elif latest_node.kind in _AUTHORING_KINDS:
        allowed_kinds = _AUTHORING_KINDS
    else:
        allowed_kinds = {latest_node.kind}
    related = [
        node
        for node in run.artifacts.node_executions
        if node.attempt == latest_node.attempt and node.kind in allowed_kinds
    ]
    return related or [latest_node]


def _dedupe_findings(findings: Iterable[ReviewerFinding]) -> list[ReviewerFinding]:
    deduped: list[ReviewerFinding] = []
    seen: set[tuple[str, str, str, str]] = set()
    for finding in findings:
        key = (finding.category, finding.severity.value, finding.title, finding.detail)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _validation_issues(validation_summary: dict) -> list[FailureContextValidationIssue]:
    issues: list[FailureContextValidationIssue] = []
    for bucket_name in ("errors", "warnings"):
        for issue in validation_summary.get(bucket_name, [])[:12]:
            issues.append(
                FailureContextValidationIssue(
                    level=str(issue.get("level") or bucket_name[:-1]),
                    code=str(issue.get("code") or "unknown_validation_issue"),
                    location=str(issue.get("location") or ""),
                    message=str(issue.get("message") or ""),
                )
            )
    return issues


def _sandbox_summary(latest_node: WorkflowNodeExecution) -> FailureContextSandboxSummary | None:
    sandbox_result = latest_node.sandbox_result
    if sandbox_result is None:
        return None

    failed_reports = [
        report
        for report in sandbox_result.module_reports
        if not report.compile_succeeded or not report.runtime_succeeded or report.error or report.stderr
    ]
    return FailureContextSandboxSummary(
        error=sandbox_result.error,
        build_stdout_excerpt=_excerpt(sandbox_result.build_stdout),
        build_stderr_excerpt=_excerpt(sandbox_result.build_stderr),
        run_stdout_excerpt=_excerpt(sandbox_result.run_stdout),
        run_stderr_excerpt=_excerpt(sandbox_result.run_stderr),
        failed_modules=[report.module_id for report in failed_reports],
        module_reports=[
            FailureContextModuleReport(
                module_id=report.module_id,
                compile_succeeded=report.compile_succeeded,
                runtime_succeeded=report.runtime_succeeded,
                error=report.error,
                stderr_excerpt=_excerpt(report.stderr),
            )
            for report in failed_reports[:8]
        ],
    )


def _excerpt(text: str, *, max_chars: int = 1200) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    tail = cleaned[-max_chars:]
    return f"...{tail}"
