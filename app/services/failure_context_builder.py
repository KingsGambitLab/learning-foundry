from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from app.domain.workflow import (
    FailureContext,
    FailureContextDeliverableReport,
    FailureContextSandboxSummary,
    FailureContextValidationIssue,
    ReviewerFinding,
    WorkflowFailureOwnerHint,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowRun,
)
from app.services.runtime_contract_surface import dependency_contract_facts_for_deliverables

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
    sandbox = _sandbox_summary(latest_node, related_nodes)
    dependency_contracts = dependency_contract_facts_for_deliverables(
        public_root=(run.artifacts.workspace_snapshot.public_dir if run.artifacts.workspace_snapshot else None),
        runtime_plan=run.artifacts.task_agent_spec.project_contract.runtime_plan
        if run.artifacts.task_agent_spec is not None
        else None,
        deliverable_ids=list(sandbox.failed_deliverables if sandbox is not None else []),
    )
    owner_hint = _owner_hint(latest_node, validation_issues, sandbox)
    phase = _phase(latest_node, validation_issues, sandbox)
    failure_signature = _failure_signature(latest_node, validation_issues, sandbox)
    return FailureContext(
        source_node_kind=latest_node.kind,
        source_node_attempt=latest_node.attempt,
        source_summary=latest_node.summary,
        owner_hint=owner_hint,
        failure_signature=failure_signature,
        phase=phase,
        findings=findings,
        validation_issues=validation_issues,
        sandbox=sandbox,
        dependency_contracts=dependency_contracts,
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
        if node.iteration == latest_node.iteration
        and node.attempt == latest_node.attempt
        and node.kind in allowed_kinds
    ]
    return related or [latest_node]


def _dedupe_findings(findings: Iterable[ReviewerFinding]) -> list[ReviewerFinding]:
    deduped: list[ReviewerFinding] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for finding in findings:
        key = (
            finding.category,
            finding.severity.value,
            finding.title,
            finding.detail,
            finding.code or "",
            finding.location or "",
        )
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


def _sandbox_summary(
    latest_node: WorkflowNodeExecution,
    related_nodes: list[WorkflowNodeExecution],
) -> FailureContextSandboxSummary | None:
    sandbox_result = latest_node.sandbox_result
    if sandbox_result is None:
        sandbox_result = next(
            (
                node.sandbox_result
                for node in reversed(related_nodes)
                if node.sandbox_result is not None
            ),
            None,
        )
    if sandbox_result is None:
        return None

    failed_reports = [
        report
        for report in sandbox_result.deliverable_reports
        if not report.compile_succeeded or not report.runtime_succeeded or report.error or report.stderr
    ]
    return FailureContextSandboxSummary(
        error=sandbox_result.error,
        build_stdout_excerpt=_excerpt(sandbox_result.build_stdout),
        build_stderr_excerpt=_excerpt(sandbox_result.build_stderr),
        run_stdout_excerpt=_excerpt(sandbox_result.run_stdout),
        run_stderr_excerpt=_excerpt(sandbox_result.run_stderr),
        failed_deliverables=[report.deliverable_id for report in failed_reports],
        deliverable_reports=[
            FailureContextDeliverableReport(
                deliverable_id=report.deliverable_id,
                compile_succeeded=report.compile_succeeded,
                runtime_succeeded=report.runtime_succeeded,
                failed_stage=(report.failed_stage.value if report.failed_stage is not None else None),
                stage_command=list(report.stage_command),
                stage_exit_code=report.stage_exit_code,
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


def _owner_hint(
    latest_node: WorkflowNodeExecution,
    validation_issues: list[FailureContextValidationIssue],
    sandbox: FailureContextSandboxSummary | None,
) -> WorkflowFailureOwnerHint:
    issue_codes = {issue.code for issue in validation_issues}
    authored_issue_codes = {
        "missing_learner_starter_surface",
        "missing_primary_editable_paths",
        "missing_required_endpoints",
        "brief_starter_surface_drift",
        "placeholder_domain_scenario",
        "placeholder_public_check",
        "missing_deliverable_learning_outcomes",
        "missing_definition_of_done",
    }
    if issue_codes & authored_issue_codes:
        return WorkflowFailureOwnerHint.authored_artifact

    text = _combined_failure_text(latest_node, sandbox)
    platform_markers = (
        "cannot connect to the docker daemon",
        "docker daemon",
        "no such network",
        "network not found",
        "port is already allocated",
        "failed to set up container networking",
        "could not build learner studio image",
        "could not start learner editor container",
        "could not start grading container",
        "could not build learner runtime image",
        "image not found",
        "pull access denied",
    )
    if any(marker in text for marker in platform_markers):
        return WorkflowFailureOwnerHint.platform_runtime

    platform_compiler_markers = (
        "create_app_from_manifest",
        "app.add_api_route",
        "[coursegen] verify step",
    )
    if "invalid args for response field" in text and any(
        marker in text for marker in platform_compiler_markers
    ):
        return WorkflowFailureOwnerHint.platform_runtime

    if latest_node.kind in {
        WorkflowNodeKind.reviewer_code,
        WorkflowNodeKind.reviewer_pedagogy,
        WorkflowNodeKind.reviewer_tests,
    } and latest_node.findings:
        return WorkflowFailureOwnerHint.authored_artifact

    structured_stage = _structured_failed_stage(sandbox)
    if latest_node.kind in {
        WorkflowNodeKind.authoring_runtime,
        WorkflowNodeKind.reviewer_runtime,
    } and structured_stage in {
        "missing_workspace",
        "dependency_materialization",
        "install",
        "verify",
        "boot",
        "contract",
        "checks",
        "runtime",
        "container_launch",
    }:
        return WorkflowFailureOwnerHint.authored_artifact

    authored_markers = (
        "traceback (most recent call last)",
        "fastapierror",
        "syntaxerror",
        "indentationerror",
        "modulenotfounderror",
        "importerror",
        "attributeerror",
        "typeerror",
        "nameerror",
        "assertionerror",
        "route registration",
        "invalid args for response field",
        "starter manifest",
        "public check",
        "public checks",
        "required endpoint",
    )
    if any(marker in text for marker in authored_markers):
        return WorkflowFailureOwnerHint.authored_artifact

    if latest_node.kind == WorkflowNodeKind.authoring_runtime and sandbox is not None:
        if sandbox.error or sandbox.deliverable_reports:
            return WorkflowFailureOwnerHint.authored_artifact

    return WorkflowFailureOwnerHint.ambiguous


def _phase(
    latest_node: WorkflowNodeExecution,
    validation_issues: list[FailureContextValidationIssue],
    sandbox: FailureContextSandboxSummary | None,
) -> str | None:
    if validation_issues:
        return "validation"
    if sandbox is None:
        return None
    structured_stage = _structured_failed_stage(sandbox)
    if structured_stage:
        return structured_stage
    text = _combined_failure_text(latest_node, sandbox)
    if "[coursegen] verify step" in text:
        return "verify"
    if "pip install" in text or "npm install" in text or "pnpm install" in text or "cargo build" in text:
        return "install"
    if "timed out waiting for" in text or "became healthy" in text or "/health" in text:
        return "boot"
    if "public check" in text or "success_rate" in text:
        return "checks"
    return "runtime"


def _failure_signature(
    latest_node: WorkflowNodeExecution,
    validation_issues: list[FailureContextValidationIssue],
    sandbox: FailureContextSandboxSummary | None,
) -> str:
    parts = [latest_node.kind.value]
    parts.extend(sorted(issue.code for issue in validation_issues))
    if sandbox is not None:
        if sandbox.error:
            parts.append(_normalize_for_signature(sandbox.error))
        for report in sandbox.deliverable_reports:
            status = report.failed_stage or ("compile" if not report.compile_succeeded else "runtime")
            parts.append(
                f"{report.deliverable_id}:{status}:{report.stage_exit_code or ''}:{_normalize_for_signature(report.error or report.stderr_excerpt or '')}"
            )
    payload = "|".join(part for part in parts if part)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _combined_failure_text(
    latest_node: WorkflowNodeExecution,
    sandbox: FailureContextSandboxSummary | None,
) -> str:
    parts: list[str] = [latest_node.summary]
    parts.extend(f"{finding.title} {finding.detail}" for finding in latest_node.findings)
    if sandbox is not None:
        parts.extend(
            item
            for item in (
                sandbox.error,
                sandbox.build_stdout_excerpt,
                sandbox.build_stderr_excerpt,
                sandbox.run_stdout_excerpt,
                sandbox.run_stderr_excerpt,
            )
            if item
        )
        parts.extend(
            f"{report.deliverable_id} {report.error or ''} {report.stderr_excerpt or ''}"
            for report in sandbox.deliverable_reports
        )
    return " ".join(parts).lower()


def _normalize_for_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    normalized = re.sub(r"0x[0-9a-f]+", "0xaddr", normalized)
    normalized = re.sub(r"\b\d+\b", "#", normalized)
    return normalized[:240]


def _structured_failed_stage(
    sandbox: FailureContextSandboxSummary | None,
) -> str | None:
    if sandbox is None:
        return None
    for report in sandbox.deliverable_reports:
        if report.failed_stage:
            return report.failed_stage
    return None
