from __future__ import annotations

import json
import re
from enum import Enum
from hashlib import sha256

from pydantic import BaseModel

from app.domain.ai import merge_ai_usage
from app.domain.workflow import (
    FailureContext,
    FailureContextValidationIssue,
    ReviewerFinding,
    WorkflowNodeKind,
    WorkflowNodeStatus,
    WorkflowFailureOwnerHint,
    WorkflowNodeExecution,
    WorkflowRun,
)
from app.services.bundle_validation import validate_materialized_bundle
from app.services.failure_context_builder import build_failure_context
from app.services.openai_task_agent_authoring import OpenAITaskAgentAuthoringService
from app.services.spec_validation import ValidationResult, validate_task_agent_spec
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService


class TaskAgentRetryAction(str, Enum):
    revised = "revised"
    blocked_platform = "blocked_platform"
    no_material_change = "no_material_change"
    unresolved_blocker = "unresolved_blocker"


class TaskAgentRetryResult(BaseModel):
    action: TaskAgentRetryAction
    applied: bool
    should_continue: bool
    skip_workspace_authoring: bool = False
    owner_hint: WorkflowFailureOwnerHint
    failure_signature: str | None = None
    before_spec_hash: str | None = None
    after_spec_hash: str | None = None
    summary: str
    detail: str


class TaskAgentRetryService:
    def __init__(
        self,
        *,
        authoring_service: OpenAITaskAgentAuthoringService | None = None,
        workspace_authoring_service: TaskAgentWorkspaceAuthoringService | None = None,
    ) -> None:
        self.authoring_service = authoring_service or OpenAITaskAgentAuthoringService(enabled=False)
        self.workspace_authoring_service = workspace_authoring_service or TaskAgentWorkspaceAuthoringService()

    def retry(
        self,
        run: WorkflowRun,
        latest_node: WorkflowNodeExecution,
        *,
        failure_context: FailureContext | None = None,
    ) -> tuple[WorkflowRun, TaskAgentRetryResult]:
        spec = run.artifacts.task_agent_spec
        if spec is None:
            return run, TaskAgentRetryResult(
                action=TaskAgentRetryAction.no_material_change,
                applied=False,
                should_continue=False,
                owner_hint=WorkflowFailureOwnerHint.ambiguous,
                summary="Retry could not run because the assignment spec is missing.",
                detail="No task-agent spec was attached to the workflow run.",
            )

        failure_context = failure_context or build_failure_context(run, latest_node)
        before_spec_hash = _spec_hash(spec)
        if failure_context.owner_hint == WorkflowFailureOwnerHint.platform_runtime:
            return run, TaskAgentRetryResult(
                action=TaskAgentRetryAction.blocked_platform,
                applied=False,
                should_continue=False,
                owner_hint=failure_context.owner_hint,
                failure_signature=failure_context.failure_signature,
                before_spec_hash=before_spec_hash,
                summary="Retry stopped because the failure packet points to a platform/runtime blocker.",
                detail=_failure_packet_summary(failure_context),
            )

        if _should_retry_workspace(failure_context, latest_node):
            if _workspace_blocker_repeated(run, latest_node, failure_context):
                return run, TaskAgentRetryResult(
                    action=TaskAgentRetryAction.unresolved_blocker,
                    applied=False,
                    should_continue=False,
                    owner_hint=failure_context.owner_hint,
                    failure_signature=failure_context.failure_signature,
                    before_spec_hash=before_spec_hash,
                    after_spec_hash=before_spec_hash,
                    summary=(
                        "Retry stopped because the same workspace blocker survived the previous repair attempt."
                    ),
                    detail=_failure_packet_summary(failure_context),
                )
            run, repaired, repair_detail = self.workspace_authoring_service.repair_workspace(
                run,
                latest_node,
                failure_context=failure_context,
            )
            if repaired:
                run.artifacts.materialized_bundle = None
                run.artifacts.review_summary = None
                run.artifacts.notes.append(
                    (
                        "Automated retry re-authored the learner workspace from the latest failure packet "
                        f"({failure_context.owner_hint.value}, signature {failure_context.failure_signature})."
                    )
                )
                return run, TaskAgentRetryResult(
                    action=TaskAgentRetryAction.revised,
                    applied=True,
                    should_continue=True,
                    skip_workspace_authoring=True,
                    owner_hint=failure_context.owner_hint,
                    failure_signature=failure_context.failure_signature,
                    before_spec_hash=before_spec_hash,
                    after_spec_hash=before_spec_hash,
                    summary="Retry re-authored the learner workspace from the latest failure packet.",
                    detail=repair_detail,
                )

        feedback = _build_feedback(run, latest_node, failure_context)
        revision = self.authoring_service.revise_spec(
            spec=spec,
            title=run.title,
            summary=run.intake.problem_statement,
            package_type=spec.course_structure.package_type,
            domain_pack=spec.domain_pack,
            risk_class=spec.risk_class,
            overlays=spec.overlays,
            feedback=feedback,
            failure_context=failure_context,
            origin_template=run.artifacts.origin_template,
        )
        revised_spec = revision.spec
        after_spec_hash = _spec_hash(revised_spec)
        if after_spec_hash == before_spec_hash:
            return run, TaskAgentRetryResult(
                action=TaskAgentRetryAction.no_material_change,
                applied=False,
                should_continue=False,
                owner_hint=failure_context.owner_hint,
                failure_signature=failure_context.failure_signature,
                before_spec_hash=before_spec_hash,
                after_spec_hash=after_spec_hash,
                summary="Retry produced no material spec changes, so the loop stopped early.",
                detail=_failure_packet_summary(failure_context),
            )

        validation = validate_task_agent_spec(revised_spec)
        run.artifacts.task_agent_spec = revised_spec
        run.artifacts.origin_template = revision.origin_template
        run.artifacts.ai_usage = merge_ai_usage(run.artifacts.ai_usage, revision.usage)
        run.artifacts.validation_summary = validation.model_dump(mode="json")
        run.artifacts.progression_preview = [
            summary.model_dump(mode="json")
            for summary in validation.deliverable_gates
        ]
        run.artifacts.materialized_bundle = None
        run.artifacts.review_summary = None
        run.artifacts.notes.append(
            (
                "Automated retry revised the learner-ready assignment draft from the latest failure packet "
                f"({failure_context.owner_hint.value}, signature {failure_context.failure_signature})."
            )
        )
        if revision.notes:
            run.notes.extend(revision.notes)
        run = self.workspace_authoring_service.sync_workspace(run)
        unresolved = _persisting_blockers(
            failure_context=failure_context,
            validation_summary=validation,
            run=run,
        )
        if unresolved:
            return run, TaskAgentRetryResult(
                action=TaskAgentRetryAction.unresolved_blocker,
                applied=True,
                should_continue=False,
                owner_hint=failure_context.owner_hint,
                failure_signature=failure_context.failure_signature,
                before_spec_hash=before_spec_hash,
                after_spec_hash=after_spec_hash,
                summary=(
                    "Retry revised the assignment spec, but the rematerialized learner workspace "
                    "still failed the same blocking checks."
                ),
                detail=_unresolved_blocker_detail(failure_context, unresolved),
            )
        return run, TaskAgentRetryResult(
            action=TaskAgentRetryAction.revised,
            applied=True,
            should_continue=True,
            owner_hint=failure_context.owner_hint,
            failure_signature=failure_context.failure_signature,
            before_spec_hash=before_spec_hash,
            after_spec_hash=after_spec_hash,
            summary="Retry revised the assignment spec from the latest failure packet.",
            detail=_failure_packet_summary(failure_context),
        )


def _spec_hash(spec) -> str:
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def _should_retry_workspace(
    failure_context: FailureContext,
    latest_node: WorkflowNodeExecution,
) -> bool:
    if failure_context.owner_hint == WorkflowFailureOwnerHint.platform_runtime:
        return False
    if latest_node.kind.value in {"authoring_runtime", "reviewer_runtime"}:
        return True
    if failure_context.sandbox is not None:
        return True
    authored_codes = {
        "starter_repo_bundle_not_authored",
        "starter_repo_bundle_incomplete",
        "runtime_protocol_bundle_not_authored",
        "runtime_protocol_bundle_incomplete",
        "starter_entrypoint_embeds_internal_runtime",
        "starter_entrypoint_simulates_from_manifest",
        "starter_primary_editable_missing",
    }
    if any(finding.code in authored_codes for finding in failure_context.findings):
        return True
    return False


def _workspace_blocker_repeated(
    run: WorkflowRun,
    latest_node: WorkflowNodeExecution,
    failure_context: FailureContext,
) -> bool:
    if not failure_context.failure_signature:
        return False
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
        None,
    )
    if latest_index is None:
        return False

    for prior_index in range(latest_index - 1, -1, -1):
        prior_node = history[prior_index]
        if prior_node.kind != latest_node.kind or prior_node.status != latest_node.status:
            continue
        prior_context = build_failure_context(run, prior_node)
        if prior_context.failure_signature != failure_context.failure_signature:
            continue
        intervening_repairs = [
            node
            for node in history[prior_index + 1 : latest_index]
            if node.kind in {WorkflowNodeKind.authoring_repair, WorkflowNodeKind.reviewer_repair}
            and node.status == WorkflowNodeStatus.passed
        ]
        if intervening_repairs:
            return True
    return False


def _build_feedback(
    run: WorkflowRun,
    latest_node: WorkflowNodeExecution,
    failure_context: FailureContext,
) -> str:
    packet = {
        "source_node_kind": failure_context.source_node_kind.value,
        "source_node_attempt": failure_context.source_node_attempt,
        "phase": failure_context.phase,
        "owner_hint": failure_context.owner_hint.value,
        "failure_signature": failure_context.failure_signature,
        "source_summary": failure_context.source_summary,
        "validation_issues": [
            issue.model_dump(mode="json")
            for issue in failure_context.validation_issues
        ],
        "findings": [
            finding.model_dump(mode="json")
            for finding in failure_context.findings[:12]
        ],
        "sandbox": (
            failure_context.sandbox.model_dump(mode="json")
            if failure_context.sandbox is not None
            else None
        ),
        "dependency_contracts": [
            contract.model_dump(mode="json")
            for contract in failure_context.dependency_contracts
        ],
    }
    return (
        "Revise the learner-ready assignment draft using the failure packet below. "
        "Fix the root cause revealed by the harness and reviewer feedback. "
        "Keep the creator-selected stack and infrastructure constraints intact. "
        "Do not leave the draft materially unchanged. "
        f"Workflow title: {run.title}. Latest failed node: {latest_node.kind.value}.\n\n"
        f"Failure packet:\n{json.dumps(packet, indent=2)}"
    )


def _failure_packet_summary(failure_context: FailureContext) -> str:
    parts = [
        f"owner_hint={failure_context.owner_hint.value}",
        f"signature={failure_context.failure_signature or 'unknown'}",
    ]
    if failure_context.phase:
        parts.append(f"phase={failure_context.phase}")
    if failure_context.sandbox is not None and failure_context.sandbox.error:
        parts.append(f"error={failure_context.sandbox.error}")
    elif failure_context.findings:
        parts.append(f"finding={failure_context.findings[0].detail}")
    return "; ".join(parts)


def _persisting_blockers(
    *,
    failure_context: FailureContext,
    validation_summary: ValidationResult,
    run: WorkflowRun,
) -> list[tuple[str, str | None]]:
    prior_blockers = _blocking_issue_keys(
        findings=failure_context.findings,
        validation_issues=failure_context.validation_issues,
    )
    if not prior_blockers:
        return []

    current_blockers = {
        *(
            (issue.code, issue.location or None)
            for issue in validation_summary.errors
        ),
    }
    workspace = run.artifacts.workspace_snapshot
    spec = run.artifacts.task_agent_spec
    if workspace is not None and spec is not None:
        bundle_validation = validate_materialized_bundle(spec, workspace)
        current_blockers.update(
            (issue.code, issue.relative_path or None)
            for issue in bundle_validation.errors
        )

    unresolved: list[tuple[str, str | None]] = []
    for prior_code, prior_location in sorted(prior_blockers):
        for current_code, current_location in current_blockers:
            if prior_code != current_code:
                continue
            if prior_location is not None and current_location != prior_location:
                continue
            unresolved.append((current_code, current_location))
            break
    return unresolved


def _blocking_issue_keys(
    *,
    findings: list[ReviewerFinding],
    validation_issues: list[FailureContextValidationIssue],
) -> set[tuple[str, str | None]]:
    keys: set[tuple[str, str | None]] = set()
    for issue in validation_issues:
        keys.add((issue.code, issue.location or None))
    for finding in findings:
        if finding.severity.value != "error":
            continue
        code = finding.code or _normalize_finding_code(finding.title)
        if code:
            keys.add((code, finding.location or None))
    return keys


def _normalize_finding_code(title: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
    return normalized or None


def _unresolved_blocker_detail(
    failure_context: FailureContext,
    unresolved: list[tuple[str, str | None]],
) -> str:
    issue_bits = [
        f"{code} @ {location}" if location else code
        for code, location in unresolved[:5]
    ]
    return (
        _failure_packet_summary(failure_context)
        + "; unresolved="
        + ", ".join(issue_bits)
    )
