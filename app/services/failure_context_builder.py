from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from app.domain.workflow import (
    FailureContext,
    FailureContextDeliverableReport,
    FailureContextSandboxSummary,
    FailureContextValidationIssue,
    FailureContextLastAttemptedRuntime,
    FailureContextVerifiedRuntime,
    FailureContextVerifiedRuntimeFile,
    ReviewerFinding,
    WorkflowFailureOwnerHint,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowRun,
)
from app.services.runtime_contract_surface import (
    STARTER_RUNTIME_PROTOCOL_PATHS,
    dependency_contract_facts_for_deliverables,
    load_starter_manifest,
    starter_dependency_contract_paths,
    starter_verified_support_paths,
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
    sandbox = _sandbox_summary(latest_node, related_nodes)
    dependency_contracts = dependency_contract_facts_for_deliverables(
        public_root=(run.artifacts.workspace_snapshot.public_dir if run.artifacts.workspace_snapshot else None),
        runtime_plan=run.artifacts.task_agent_spec.project_contract.runtime_plan
        if run.artifacts.task_agent_spec is not None
        else None,
        deliverable_ids=list(sandbox.failed_deliverables if sandbox is not None else []),
        workspace_root=(run.artifacts.workspace_snapshot.root_dir if run.artifacts.workspace_snapshot else None),
        shared_codebase=bool(
            run.artifacts.task_agent_spec is not None
            and run.artifacts.task_agent_spec.course_structure.shared_codebase
        ),
    )
    owner_hint = _owner_hint(latest_node, validation_issues, sandbox)
    phase = _phase(latest_node, validation_issues, sandbox)
    failure_signature = _failure_signature(latest_node, validation_issues, sandbox)
    previously_verified_runtime = _previously_verified_runtime(run, latest_node)
    last_attempted_runtime = _last_attempted_runtime(run, latest_node)
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
        previously_verified_runtime=previously_verified_runtime,
        last_attempted_runtime=last_attempted_runtime,
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

    # Real failures: anything that didn't compile, didn't run, or carries an
    # explicit error. Plain stderr content alone is not failure — boot logs
    # always have stderr output even on a passing sandbox.
    failed_reports = [
        report
        for report in sandbox_result.deliverable_reports
        if not report.compile_succeeded or not report.runtime_succeeded or report.error
    ]
    failed_deliverable_ids = [report.deliverable_id for report in failed_reports]

    # When reviewer-side findings flag specific deliverables via
    # `location=starter/deliverable_X`, treat those as the authoritative
    # failed set for this attempt — the sandbox itself may have passed earlier
    # but the reviewer disagreed about a specific deliverable's quality.
    for finding in latest_node.findings:
        if finding.severity.value != "error":
            continue
        deliverable_id = _deliverable_id_from_location(finding.location)
        if deliverable_id and deliverable_id not in failed_deliverable_ids:
            failed_deliverable_ids.append(deliverable_id)

    return FailureContextSandboxSummary(
        error=sandbox_result.error,
        build_stdout_excerpt=_excerpt(sandbox_result.build_stdout),
        build_stderr_excerpt=_excerpt(sandbox_result.build_stderr),
        run_stdout_excerpt=_excerpt(sandbox_result.run_stdout),
        run_stderr_excerpt=_excerpt(sandbox_result.run_stderr),
        failed_deliverables=failed_deliverable_ids,
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


def _deliverable_id_from_location(location: str | None) -> str | None:
    """Extract a deliverable id from a reviewer finding location.

    Reviewer findings point at concrete files using locations like
    `starter/deliverable_2`, `starter/deliverable_2/.coursegen/...`, or
    `public/starter/deliverable_4/README.md`. Return the deliverable id when
    present, else None.
    """
    if not location:
        return None
    parts = [part for part in location.replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts):
        if part == "starter" and index + 1 < len(parts):
            candidate = parts[index + 1]
            if candidate.startswith("deliverable"):
                return candidate
    return None


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
    # Reviewer-side failures classify by the reviewer node, not by stage-mining
    # the carried-over passing sandbox. reviewer_tests in particular surfaces
    # baseline-matrix discrimination issues, which are a test-authoring concern
    # — not the runtime layer.
    if latest_node.kind == WorkflowNodeKind.reviewer_tests:
        return "tests"
    if latest_node.kind == WorkflowNodeKind.reviewer_code:
        return "code_review"
    if latest_node.kind == WorkflowNodeKind.reviewer_pedagogy:
        return "pedagogy_review"
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


def _previously_verified_runtime(
    run: WorkflowRun,
    latest_node: WorkflowNodeExecution,
) -> FailureContextVerifiedRuntime | None:
    history = run.artifacts.node_executions
    latest_index = next(
        (
            index
            for index, node in reversed(list(enumerate(history)))
            if node.node_id == latest_node.node_id
            and node.kind == latest_node.kind
            and node.attempt == latest_node.attempt
            and node.created_at == latest_node.created_at
        ),
        len(history),
    )
    runtime_node = next(
        (
            node
            for node in reversed(history[:latest_index])
            if node.kind in {WorkflowNodeKind.authoring_runtime, WorkflowNodeKind.reviewer_runtime}
            and node.status.value == "passed"
            and node.sandbox_result is not None
            and node.sandbox_result.status.value == "passed"
        ),
        None,
    )
    if runtime_node is None or runtime_node.sandbox_result is None:
        return None

    passed_deliverables = [
        report.deliverable_id
        for report in runtime_node.sandbox_result.deliverable_reports
        if report.compile_succeeded and report.runtime_succeeded
    ]
    if not passed_deliverables:
        return None

    current_failed_deliverables = [
        report.deliverable_id
        for report in (latest_node.sandbox_result.deliverable_reports if latest_node.sandbox_result is not None else [])
        if not report.compile_succeeded or not report.runtime_succeeded or bool(report.error)
    ]

    public_root = run.artifacts.workspace_snapshot.public_dir if run.artifacts.workspace_snapshot else None
    workspace_root = (
        run.artifacts.workspace_snapshot.root_dir if run.artifacts.workspace_snapshot else None
    )
    runtime_plan = (
        run.artifacts.task_agent_spec.project_contract.runtime_plan
        if run.artifacts.task_agent_spec is not None
        else None
    )
    shared_codebase = bool(
        run.artifacts.task_agent_spec is not None
        and run.artifacts.task_agent_spec.course_structure.shared_codebase
    )
    dependency_contracts = dependency_contract_facts_for_deliverables(
        public_root=public_root,
        runtime_plan=runtime_plan,
        deliverable_ids=passed_deliverables,
        workspace_root=workspace_root,
        shared_codebase=shared_codebase,
    )
    verified_files = _verified_runtime_files(
        public_root=public_root,
        source_deliverable_id=passed_deliverables[0],
        shared_codebase=shared_codebase,
        workspace_root=workspace_root,
    )
    return FailureContextVerifiedRuntime(
        source_node_kind=runtime_node.kind,
        source_node_attempt=runtime_node.attempt,
        verified_at=runtime_node.created_at,
        source_deliverable_id=passed_deliverables[0],
        passed_deliverables=passed_deliverables,
        current_failed_deliverables=current_failed_deliverables,
        verified_files=verified_files,
        dependency_contracts=dependency_contracts,
    )


_STAGE_ORDER = ("image_build", "install", "verify", "boot", "contract", "checks")


def _stage_outcomes_for_report(report) -> dict[str, str]:
    """Infer per-stage outcomes from a DeliverableSandboxReport.

    Stages before `failed_stage` are inferred as `passed`; the failed stage is
    `failed`; later stages are `not_run`. When the report has no `failed_stage`
    but `compile_succeeded` / `runtime_succeeded` / `public_checks_passed` are
    set, we fill those in directly.
    """
    failed_stage_value = report.failed_stage.value if report.failed_stage else None
    outcomes: dict[str, str] = {}

    if failed_stage_value is not None:
        for stage in _STAGE_ORDER:
            if stage == failed_stage_value:
                outcomes[stage] = "failed"
                break
            outcomes[stage] = "passed"
        # remaining stages didn't run
        seen = set(outcomes)
        for stage in _STAGE_ORDER:
            if stage not in seen:
                outcomes[stage] = "not_run"
        return outcomes

    # No structured failed_stage — fall back to boolean signals.
    if report.compile_succeeded:
        outcomes["image_build"] = "passed"
        outcomes["install"] = "passed"
    if report.runtime_succeeded:
        outcomes["verify"] = "passed"
        outcomes["boot"] = "passed"
    if report.public_checks_passed is True:
        outcomes["contract"] = "passed"
        outcomes["checks"] = "passed"
    elif report.public_checks_passed is False:
        outcomes["contract"] = "failed"
    return outcomes


def _last_attempted_runtime(
    run: WorkflowRun,
    latest_node: WorkflowNodeExecution,
) -> FailureContextLastAttemptedRuntime | None:
    history = run.artifacts.node_executions
    latest_index = next(
        (
            index
            for index, node in reversed(list(enumerate(history)))
            if node.node_id == latest_node.node_id
            and node.kind == latest_node.kind
            and node.attempt == latest_node.attempt
            and node.created_at == latest_node.created_at
        ),
        len(history),
    )
    # Include the latest_node itself in the search window: when repair runs
    # after a failed authoring_runtime, latest_node IS the most recent attempt
    # whose stage outcomes we want to carry forward.
    runtime_node = next(
        (
            node
            for node in reversed(history[: latest_index + 1])
            if node.kind in {WorkflowNodeKind.authoring_runtime, WorkflowNodeKind.reviewer_runtime}
            and node.sandbox_result is not None
            and node.sandbox_result.deliverable_reports
        ),
        None,
    )
    if runtime_node is None or runtime_node.sandbox_result is None:
        return None

    reports = list(runtime_node.sandbox_result.deliverable_reports)
    if not reports:
        return None
    source_report = reports[0]
    source_deliverable_id = source_report.deliverable_id

    stage_outcomes = _stage_outcomes_for_report(source_report)

    public_root = run.artifacts.workspace_snapshot.public_dir if run.artifacts.workspace_snapshot else None
    workspace_root = (
        run.artifacts.workspace_snapshot.root_dir if run.artifacts.workspace_snapshot else None
    )
    shared_codebase = bool(
        run.artifacts.task_agent_spec is not None
        and run.artifacts.task_agent_spec.course_structure.shared_codebase
    )
    verified_files = _verified_runtime_files(
        public_root=public_root,
        source_deliverable_id=source_deliverable_id,
        shared_codebase=shared_codebase,
        workspace_root=workspace_root,
    )

    # Whether each file should be preserved depends on which stages of the
    # harness it contributed to. If `boot` passed, the entire runtime bundle
    # (Dockerfile + install/verify/run.sh) is verified by the harness; if
    # `install` passed, dependency-contract files are verified.
    runtime_bundle_verified = stage_outcomes.get("boot") == "passed"
    deps_verified = stage_outcomes.get("install") == "passed"
    for file in verified_files:
        if file.role == "runtime_protocol":
            file.preserve_verbatim = runtime_bundle_verified
        elif file.role == "dependency_contract":
            file.preserve_verbatim = deps_verified
        else:
            file.preserve_verbatim = False

    return FailureContextLastAttemptedRuntime(
        source_node_kind=runtime_node.kind,
        source_node_attempt=runtime_node.attempt,
        attempted_at=runtime_node.created_at,
        source_deliverable_id=source_deliverable_id,
        stage_outcomes=stage_outcomes,
        verified_files=verified_files,
    )


def _verified_runtime_files(
    *,
    public_root: str | None,
    source_deliverable_id: str,
    shared_codebase: bool = False,
    workspace_root: str | None = None,
) -> list[FailureContextVerifiedRuntimeFile]:
    if not public_root:
        return []
    if shared_codebase:
        starter_root = Path(public_root) / "starter"
        if workspace_root:
            manifest_path = (
                Path(workspace_root) / "private" / "grader" / source_deliverable_id / "deliverable.json"
            )
        else:
            manifest_path = (
                Path(public_root).parent / "private" / "grader" / source_deliverable_id / "deliverable.json"
            )
        manifest: dict | None = None
        if manifest_path.exists():
            try:
                import json as _json
                manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = None
    else:
        starter_root = Path(public_root) / "starter" / source_deliverable_id
        manifest = load_starter_manifest(starter_root)
    if manifest is None:
        return []

    per_file_limit = 24 * 1024
    total_limit = 64 * 1024
    total_bytes = 0
    files: list[FailureContextVerifiedRuntimeFile] = []
    path_roles: list[tuple[str, str]] = [
        *(("runtime_protocol", path) for path in STARTER_RUNTIME_PROTOCOL_PATHS),
        *(
            ("dependency_contract", path)
            for path in starter_dependency_contract_paths(
                manifest=manifest,
                include_lockfiles=False,
                include_build_support=True,
            )
        ),
        *(
            ("repo_support", path)
            for path in starter_verified_support_paths(
                starter_root=starter_root,
                manifest=manifest,
            )
        ),
    ]
    seen_paths: set[str] = set()
    for role, relative_path in path_roles:
        if relative_path in seen_paths:
            continue
        seen_paths.add(relative_path)
        target = starter_root / relative_path
        content = _read_verified_text_file(
            target,
            max_file_bytes=per_file_limit,
            remaining_total_bytes=max(0, total_limit - total_bytes),
        )
        if content is None:
            continue
        byte_length = len(content.encode("utf-8"))
        total_bytes += byte_length
        files.append(
            FailureContextVerifiedRuntimeFile(
                path=relative_path,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                role=role,
                content=content,
                preserve_verbatim=True,
            )
        )
        if total_bytes >= total_limit:
            break
    return files


def _read_verified_text_file(
    path: Path,
    *,
    max_file_bytes: int,
    remaining_total_bytes: int,
) -> str | None:
    if remaining_total_bytes <= 0:
        return None
    if not path.exists() or not path.is_file():
        return None
    try:
        file_size = path.stat().st_size
    except OSError:
        return None
    if file_size > max_file_bytes or file_size > remaining_total_bytes:
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for character in content:
        codepoint = ord(character)
        if codepoint < 32 and character not in {"\n", "\r", "\t"}:
            return None
    return content
