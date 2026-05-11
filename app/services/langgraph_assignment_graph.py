from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain.sandbox import SandboxExecutionResult, SandboxExecutionStatus
from app.domain.workflow import (
    ReviewerFinding,
    ReviewerFindingSeverity,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowLoopPolicy,
    WorkflowNodeStatus,
    WorkflowRun,
)
from app.services.coursegen_logging import log_coursegen_event
from app.services.bundle_validation import inspect_materialized_starter_surface, validate_materialized_bundle
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.failure_context_builder import build_failure_context
from app.services.generated_test_harness import GeneratedTestBaselineVerifier
from app.services.openai_test_script_authoring import OpenAITestScriptAuthoringService
from app.services.public_surface_quality import starter_surface_markers
from app.services.spec_validation import validate_task_agent_spec
from app.services.task_agent_retry_service import TaskAgentRetryService
from app.services.task_agent_repair_service import TaskAgentRepairService
from app.services.task_agent_starter_templates import (
    HIDDEN_GRADER_SCRIPT_PATH,
    HIDDEN_MANIFEST_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
)
from app.services.task_agent_contract_surface import (
    learner_editable_paths_for_deliverable,
    learner_editable_paths_for_spec,
)
from app.services.task_agent_workspace_authoring import (
    TaskAgentWorkspaceAuthoringService,
    WorkspaceAuthoringResult,
)


AuthoringRoute = Literal["authoring_repair", "authoring_tests", "reviewer_runtime", "end"]
ReviewerRoute = Literal["reviewer_repair", "reviewer_code", "reviewer_pedagogy", "reviewer_tests", "end"]
RetryRoute = Literal["authoring_runtime", "authoring_tests", "reviewer_tests", "end"]
_UNSET = object()


class AssignmentGraphState(TypedDict):
    run: WorkflowRun
    node_executions: list[WorkflowNodeExecution]
    active_iteration: int
    authoring_attempt: int
    reviewer_attempt: int
    cached_sandbox_result: SandboxExecutionResult | None
    next_retry_node: str | None
    skip_workspace_authoring: bool


class LangGraphAssignmentGraph:
    def __init__(
        self,
        sandbox_runner: DockerSandboxRunner,
        *,
        repair_service: TaskAgentRepairService | None = None,
        workspace_authoring_service: TaskAgentWorkspaceAuthoringService | None = None,
        authoring_service=None,
        test_authoring_service: OpenAITestScriptAuthoringService | None = None,
        retry_service: TaskAgentRetryService | None = None,
        baseline_verifier: GeneratedTestBaselineVerifier | None = None,
        max_authoring_attempts: int = 3,
        max_reviewer_attempts: int = 2,
    ) -> None:
        self.sandbox_runner = sandbox_runner
        self.repair_service = repair_service or TaskAgentRepairService()
        self.workspace_authoring_service = workspace_authoring_service or TaskAgentWorkspaceAuthoringService()
        self.test_authoring_service = test_authoring_service or OpenAITestScriptAuthoringService(enabled=False)
        self.retry_service = retry_service or TaskAgentRetryService(
            authoring_service=authoring_service,
            workspace_authoring_service=self.workspace_authoring_service,
        )
        self.baseline_verifier = baseline_verifier or GeneratedTestBaselineVerifier()
        self.max_authoring_attempts = max_authoring_attempts
        self.max_reviewer_attempts = max_reviewer_attempts
        self.graph = self._build_graph().compile()

    def status(self):
        return self.sandbox_runner.status()

    def policy(self) -> WorkflowLoopPolicy:
        return WorkflowLoopPolicy(
            max_authoring_attempts=self.max_authoring_attempts,
            max_reviewer_attempts=self.max_reviewer_attempts,
        )

    def execute(
        self,
        run: WorkflowRun,
        *,
        on_node_started: Callable[[WorkflowRun, WorkflowNodeKind, int], None] | None = None,
        on_node_finished: Callable[[WorkflowRun, WorkflowNodeExecution], None] | None = None,
    ) -> WorkflowRun:
        if run.artifacts.task_agent_spec is None:
            return run
        existing_executions = [
            node.model_copy(deep=True)
            for node in run.artifacts.node_executions
        ]
        state: AssignmentGraphState = {
            "run": run.model_copy(deep=True),
            "node_executions": existing_executions,
            "active_iteration": max(
                (node.iteration for node in existing_executions),
                default=0,
            )
            + 1,
            "authoring_attempt": 0,
            "reviewer_attempt": 0,
            "cached_sandbox_result": None,
            "next_retry_node": None,
            "skip_workspace_authoring": False,
        }
        current_node = "authoring_runtime"
        while current_node is not None:
            kind = self._kind_for_node_name(current_node)
            attempt = self._planned_attempt(current_node, state)
            if on_node_started is not None:
                on_node_started(self._run_snapshot(state), kind, attempt)
            state = self._invoke_node(current_node, state)
            latest = state["node_executions"][-1]
            if on_node_finished is not None:
                on_node_finished(self._run_snapshot(state), latest)
            current_node = self._next_node(current_node, state)
        return self._run_snapshot(state)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AssignmentGraphState)
        graph.add_node("authoring_runtime", self._authoring_runtime_node)
        graph.add_node("authoring_tests", self._authoring_tests_node)
        graph.add_node("authoring_repair", self._authoring_repair_node)
        graph.add_node("reviewer_runtime", self._reviewer_runtime_node)
        graph.add_node("reviewer_repair", self._reviewer_repair_node)
        graph.add_node("reviewer_code", self._reviewer_code_node)
        graph.add_node("reviewer_pedagogy", self._reviewer_pedagogy_node)
        graph.add_node("reviewer_tests", self._reviewer_tests_node)

        graph.add_edge(START, "authoring_runtime")
        graph.add_conditional_edges(
            "authoring_runtime",
            self._after_authoring_runtime,
            {
                "authoring_repair": "authoring_repair",
                "authoring_tests": "authoring_tests",
                "reviewer_runtime": "reviewer_runtime",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "authoring_tests",
            self._after_authoring_tests,
            {
                "authoring_repair": "authoring_repair",
                "reviewer_runtime": "reviewer_runtime",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "authoring_repair",
            self._after_authoring_repair,
            {
                "authoring_runtime": "authoring_runtime",
                "authoring_tests": "authoring_tests",
                "end": END,
            },
        )

        graph.add_conditional_edges(
            "reviewer_runtime",
            self._after_reviewer_runtime,
            {
                "reviewer_repair": "reviewer_repair",
                "reviewer_code": "reviewer_code",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "reviewer_code",
            self._after_reviewer_code,
            {
                "reviewer_repair": "reviewer_repair",
                "reviewer_pedagogy": "reviewer_pedagogy",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "reviewer_pedagogy",
            self._after_reviewer_pedagogy,
            {
                "reviewer_repair": "reviewer_repair",
                "reviewer_tests": "reviewer_tests",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "reviewer_tests",
            self._after_reviewer_tests,
            {
                "reviewer_repair": "reviewer_repair",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "reviewer_repair",
            self._after_reviewer_repair,
            {
                "authoring_runtime": "authoring_runtime",
                "reviewer_tests": "reviewer_tests",
                "end": END,
            },
        )
        return graph

    def _invoke_node(self, node_name: str, state: AssignmentGraphState) -> AssignmentGraphState:
        node_map = {
            "authoring_runtime": self._authoring_runtime_node,
            "authoring_tests": self._authoring_tests_node,
            "authoring_repair": self._authoring_repair_node,
            "reviewer_runtime": self._reviewer_runtime_node,
            "reviewer_repair": self._reviewer_repair_node,
            "reviewer_code": self._reviewer_code_node,
            "reviewer_pedagogy": self._reviewer_pedagogy_node,
            "reviewer_tests": self._reviewer_tests_node,
        }
        return node_map[node_name](state)

    def _next_node(self, node_name: str, state: AssignmentGraphState) -> str | None:
        if node_name == "authoring_runtime":
            route = self._after_authoring_runtime(state)
        elif node_name == "authoring_tests":
            route = self._after_authoring_tests(state)
        elif node_name == "authoring_repair":
            route = self._after_authoring_repair(state)
        elif node_name == "reviewer_runtime":
            route = self._after_reviewer_runtime(state)
        elif node_name == "reviewer_code":
            route = self._after_reviewer_code(state)
        elif node_name == "reviewer_pedagogy":
            route = self._after_reviewer_pedagogy(state)
        elif node_name == "reviewer_tests":
            route = self._after_reviewer_tests(state)
        elif node_name == "reviewer_repair":
            route = self._after_reviewer_repair(state)
        else:
            raise KeyError(f"Unknown node '{node_name}'.")
        return None if route == "end" else route

    def _run_snapshot(self, state: AssignmentGraphState) -> WorkflowRun:
        run = state["run"].model_copy(deep=True)
        run.artifacts.node_executions = list(state["node_executions"])
        return run

    def _planned_attempt(self, node_name: str, state: AssignmentGraphState) -> int:
        if node_name == "authoring_runtime":
            return state["authoring_attempt"] + 1
        if node_name == "authoring_tests":
            return state["authoring_attempt"] or 1
        if node_name == "authoring_repair":
            return state["authoring_attempt"]
        if node_name == "reviewer_runtime":
            return state["reviewer_attempt"] + 1
        return state["reviewer_attempt"]

    def _kind_for_node_name(self, node_name: str) -> WorkflowNodeKind:
        return WorkflowNodeKind(node_name)

    def _after_authoring_runtime(self, state: AssignmentGraphState) -> AuthoringRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "authoring_tests"
        if state["authoring_attempt"] < self.max_authoring_attempts:
            return "authoring_repair"
        return "end"

    def _after_authoring_tests(self, state: AssignmentGraphState) -> AuthoringRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "reviewer_runtime"
        if state["authoring_attempt"] < self.max_authoring_attempts:
            return "authoring_repair"
        return "end"

    def _after_reviewer_runtime(self, state: AssignmentGraphState) -> ReviewerRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "reviewer_code"
        if state["reviewer_attempt"] < self.max_reviewer_attempts:
            return "reviewer_repair"
        return "end"

    def _after_reviewer_code(self, state: AssignmentGraphState) -> ReviewerRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "reviewer_pedagogy"
        if state["reviewer_attempt"] < self.max_reviewer_attempts:
            return self._reviewer_failure_route(state, latest)
        return "end"

    def _after_reviewer_pedagogy(self, state: AssignmentGraphState) -> ReviewerRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "reviewer_tests"
        if state["reviewer_attempt"] < self.max_reviewer_attempts:
            return self._reviewer_failure_route(state, latest)
        return "end"

    def _after_reviewer_tests(self, state: AssignmentGraphState) -> ReviewerRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "end"
        if state["reviewer_attempt"] < self.max_reviewer_attempts:
            return self._reviewer_failure_route(state, latest)
        return "end"

    def _reviewer_failure_route(
        self,
        state: AssignmentGraphState,
        latest: WorkflowNodeExecution,
    ) -> ReviewerRoute:
        return "reviewer_repair"

    def _after_authoring_repair(self, state: AssignmentGraphState) -> RetryRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return state.get("next_retry_node") or "authoring_runtime"
        return "end"

    def _after_reviewer_repair(self, state: AssignmentGraphState) -> RetryRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return state.get("next_retry_node") or "authoring_runtime"
        return "end"

    def _authoring_runtime_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        attempt = state["authoring_attempt"] + 1
        skip_workspace_authoring = state.get("skip_workspace_authoring", False)
        if skip_workspace_authoring and state["run"].artifacts.workspace_snapshot is not None:
            run = state["run"]
            authoring_result = WorkspaceAuthoringResult(
                updated_files=[],
                notes=[],
                message="Reused the repaired learner workspace for runtime re-verification.",
            )
            log_coursegen_event(
                "authoring_runtime_workspace_authoring_skipped",
                workflow_run_id=run.id,
                title=run.title,
                attempt=attempt,
            )
        else:
            log_coursegen_event(
                "authoring_runtime_workspace_authoring_started",
                workflow_run_id=state["run"].id,
                title=state["run"].title,
                attempt=attempt,
            )
            try:
                run, authoring_result = self.workspace_authoring_service.author_workspace(state["run"])
            except Exception as exc:
                log_coursegen_event(
                    "authoring_runtime_workspace_authoring_failed",
                    workflow_run_id=state["run"].id,
                    title=state["run"].title,
                    attempt=attempt,
                    error=str(exc),
                )
                raise
            log_coursegen_event(
                "authoring_runtime_workspace_authoring_completed",
                workflow_run_id=run.id,
                title=run.title,
                attempt=attempt,
                updated_file_count=len(authoring_result.updated_files),
                updated_files=authoring_result.updated_files[:10],
            )
        state_with_workspace = {
            **state,
            "run": run,
            "cached_sandbox_result": None,
            "skip_workspace_authoring": False,
        }
        log_coursegen_event(
            "authoring_runtime_sandbox_started",
            workflow_run_id=run.id,
            title=run.title,
            attempt=attempt,
            workspace_root=(
                run.artifacts.workspace_snapshot.root_dir
                if run.artifacts.workspace_snapshot is not None
                else None
            ),
        )
        try:
            state_with_sandbox, sandbox_result = self._sandbox_result(state_with_workspace, force=True)
        except Exception as exc:
            log_coursegen_event(
                "authoring_runtime_sandbox_failed",
                workflow_run_id=run.id,
                title=run.title,
                attempt=attempt,
                error=str(exc),
            )
            raise
        log_coursegen_event(
            "authoring_runtime_sandbox_completed",
            workflow_run_id=run.id,
            title=run.title,
            attempt=attempt,
            sandbox_status=sandbox_result.status.value,
            deliverable_report_count=len(sandbox_result.deliverable_reports),
            duration_ms=sandbox_result.duration_ms,
        )
        findings: list[ReviewerFinding] = [
            ReviewerFinding(
                category="authoring_runtime",
                severity=ReviewerFindingSeverity.info,
                title="Workspace prepared",
                detail=authoring_result.message,
            )
        ]
        if authoring_result.updated_files:
            findings.append(
                ReviewerFinding(
                    category="authoring_runtime",
                    severity=ReviewerFindingSeverity.info,
                    title="Workspace files updated",
                    detail=", ".join(authoring_result.updated_files[:5]),
                )
            )
        status = WorkflowNodeStatus.passed
        summary = "Generated assignment compiled and booted inside the Docker sandbox."

        if sandbox_result.status != SandboxExecutionStatus.passed:
            status = WorkflowNodeStatus.failed
            summary = "Generated assignment failed to compile or boot inside the Docker sandbox."
            findings.append(
                ReviewerFinding(
                    category="runtime",
                    severity=ReviewerFindingSeverity.error,
                    title="Sandbox verification failed",
                    detail=sandbox_result.error or "Inspect sandbox stdout/stderr for the failure cause.",
                )
            )

        return self._append_node(
            state_with_sandbox,
            kind=WorkflowNodeKind.authoring_runtime,
            attempt=attempt,
            status=status,
            summary=summary,
            findings=findings,
            sandbox_result=sandbox_result,
            authoring_attempt=attempt,
            cached_sandbox_result=sandbox_result,
        )

    def _authoring_tests_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        attempt = state["authoring_attempt"] or 1
        try:
            run, test_result = self.test_authoring_service.author_workspace_tests(state["run"])
        except Exception as exc:
            return self._append_node(
                state,
                kind=WorkflowNodeKind.authoring_tests,
                attempt=attempt,
                status=WorkflowNodeStatus.failed,
                summary="Authoring tests failed to generate learner-visible and hidden scripts.",
                findings=[
                    ReviewerFinding(
                        category="test_authoring",
                        severity=ReviewerFindingSeverity.error,
                        title="Generated test scripts could not be authored",
                        detail=str(exc),
                        code="generated_test_authoring_failed",
                    )
                ],
                sandbox_result=None,
                authoring_attempt=attempt,
                cached_sandbox_result=state.get("cached_sandbox_result"),
                next_retry_node=None,
            )

        findings: list[ReviewerFinding] = []
        status = WorkflowNodeStatus.passed
        workspace = run.artifacts.workspace_snapshot
        if workspace is None:
            status = WorkflowNodeStatus.failed
            findings.append(
                ReviewerFinding(
                    category="test_authoring",
                    severity=ReviewerFindingSeverity.error,
                    title="Workspace missing for generated tests",
                    detail="Test authoring needs a materialized learner workspace before it can write scripts.",
                    code="generated_test_workspace_missing",
                )
            )
        else:
            public_dir = Path(workspace.public_dir)
            for deliverable in run.artifacts.task_agent_spec.deliverables:  # type: ignore[union-attr]
                starter_root = public_dir / "starter" / deliverable.id
                visible_path = starter_root / "checks" / "run_visible_checks.py"
                hidden_path = starter_root / HIDDEN_GRADER_SCRIPT_PATH
                if not visible_path.exists() or not hidden_path.exists():
                    status = WorkflowNodeStatus.failed
                    findings.append(
                        ReviewerFinding(
                            category="test_authoring",
                            severity=ReviewerFindingSeverity.error,
                            title=f"Generated tests missing for {deliverable.id}",
                            detail="Both visible and hidden test scripts must exist in the materialized starter workspace.",
                            code="generated_test_scripts_missing",
                            location=f"starter/{deliverable.id}",
                        )
                    )
            if status == WorkflowNodeStatus.passed:
                findings.append(
                    ReviewerFinding(
                        category="test_authoring",
                        severity=ReviewerFindingSeverity.info,
                        title="Generated test scripts materialized",
                        detail=test_result.message,
                    )
                )
                if not test_result.available:
                    findings.append(
                        ReviewerFinding(
                            category="test_authoring",
                            severity=ReviewerFindingSeverity.warning,
                            title="Generated tests stayed on the existing workspace scripts",
                            detail=test_result.message,
                            code="generated_test_authoring_unavailable",
                        )
                    )

        return self._append_node(
            {**state, "run": run},
            kind=WorkflowNodeKind.authoring_tests,
            attempt=attempt,
            status=status,
            summary="Authoring tests generated runnable visible and hidden scripts against the materialized starter workspace.",
            findings=findings,
            sandbox_result=None,
            authoring_attempt=state["authoring_attempt"] or attempt,
            cached_sandbox_result=state.get("cached_sandbox_result"),
            next_retry_node=None,
        )

    def _authoring_repair_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        latest = state["node_executions"][-1]
        failure_context = build_failure_context(state["run"], latest)
        if latest.kind == WorkflowNodeKind.authoring_tests:
            return self._repair_generated_tests(
                state,
                latest=latest,
                failure_context=failure_context,
                kind=WorkflowNodeKind.authoring_repair,
                attempt=state["authoring_attempt"],
                next_retry_node="authoring_tests",
            )
        run, retry_result = self.retry_service.retry(
            state["run"],
            latest,
            failure_context=failure_context,
        )
        status = WorkflowNodeStatus.passed if retry_result.should_continue else WorkflowNodeStatus.failed
        return self._append_node(
            {**state, "run": run},
            kind=WorkflowNodeKind.authoring_repair,
            attempt=state["authoring_attempt"],
            status=status,
            summary=retry_result.summary,
            findings=[
                ReviewerFinding(
                    category="authoring_repair",
                    severity=ReviewerFindingSeverity.info if retry_result.should_continue else ReviewerFindingSeverity.error,
                    title="Retry decision",
                    detail=retry_result.detail,
                )
            ],
            sandbox_result=None,
            cached_sandbox_result=None,
            next_retry_node="authoring_runtime" if retry_result.should_continue else None,
            skip_workspace_authoring=retry_result.skip_workspace_authoring if retry_result.should_continue else False,
        )

    def _reviewer_runtime_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        state, sandbox_result = self._sandbox_result(state)
        attempt = state["reviewer_attempt"] + 1
        findings: list[ReviewerFinding] = []
        status = WorkflowNodeStatus.passed
        summary = "Reviewer runtime node confirmed the assignment still boots in Docker."

        if sandbox_result.status != SandboxExecutionStatus.passed:
            status = WorkflowNodeStatus.failed
            summary = "Reviewer runtime node saw a Docker execution failure."
            findings.append(
                ReviewerFinding(
                    category="runtime_review",
                    severity=ReviewerFindingSeverity.error,
                    title="Runtime verification failed",
                    detail=sandbox_result.error or "The reviewer sandbox run failed.",
                )
            )
        else:
            findings.append(
                ReviewerFinding(
                    category="runtime_review",
                    severity=ReviewerFindingSeverity.info,
                    title="Runtime verification passed",
                    detail=f"Verified {len(sandbox_result.deliverable_reports)} deliverable starter(s) in Docker.",
                )
            )
            starter_check_gaps = [
                report.deliverable_id
                for report in sandbox_result.deliverable_reports
                if report.runtime_succeeded and report.public_checks_passed is False
            ]
            if starter_check_gaps:
                findings.append(
                    ReviewerFinding(
                        category="runtime_review",
                        severity=ReviewerFindingSeverity.info,
                        title="Starter still leaves visible work for the learner",
                        detail=(
                            "The starter keeps the public contract stable in Docker, but visible checks still fail for: "
                            + ", ".join(starter_check_gaps)
                        ),
                    )
                )

        return self._append_node(
            state,
            kind=WorkflowNodeKind.reviewer_runtime,
            attempt=attempt,
            status=status,
            summary=summary,
            findings=findings,
            sandbox_result=sandbox_result,
            reviewer_attempt=attempt,
            cached_sandbox_result=sandbox_result,
        )

    def _reviewer_repair_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        latest = state["node_executions"][-1]
        failure_context = build_failure_context(state["run"], latest)
        if latest.kind == WorkflowNodeKind.reviewer_tests:
            return self._repair_generated_tests(
                state,
                latest=latest,
                failure_context=failure_context,
                kind=WorkflowNodeKind.reviewer_repair,
                attempt=state["reviewer_attempt"],
                next_retry_node="reviewer_tests",
            )
        run, retry_result = self.retry_service.retry(
            state["run"],
            latest,
            failure_context=failure_context,
        )
        status = WorkflowNodeStatus.passed if retry_result.should_continue else WorkflowNodeStatus.failed
        return self._append_node(
            {**state, "run": run},
            kind=WorkflowNodeKind.reviewer_repair,
            attempt=state["reviewer_attempt"],
            status=status,
            summary=retry_result.summary,
            findings=[
                ReviewerFinding(
                    category="reviewer_repair",
                    severity=ReviewerFindingSeverity.info if retry_result.should_continue else ReviewerFindingSeverity.error,
                    title="Retry decision",
                    detail=retry_result.detail,
                )
            ],
            sandbox_result=None,
            cached_sandbox_result=None,
            next_retry_node="authoring_runtime" if retry_result.should_continue else None,
            skip_workspace_authoring=retry_result.skip_workspace_authoring if retry_result.should_continue else False,
        )

    def _repair_generated_tests(
        self,
        state: AssignmentGraphState,
        *,
        latest: WorkflowNodeExecution,
        failure_context,
        kind: WorkflowNodeKind,
        attempt: int,
        next_retry_node: str,
    ) -> AssignmentGraphState:
        deliverable_ids = self._target_deliverable_ids(state["run"], failure_context)
        try:
            run, test_result = self.test_authoring_service.author_workspace_tests(
                state["run"],
                failure_context=failure_context,
                deliverable_ids=deliverable_ids or None,
            )
        except Exception as exc:
            return self._append_node(
                state,
                kind=kind,
                attempt=attempt,
                status=WorkflowNodeStatus.failed,
                summary="Repair could not regenerate the authored test scripts.",
                findings=[
                    ReviewerFinding(
                        category="test_authoring",
                        severity=ReviewerFindingSeverity.error,
                        title="Generated test repair failed",
                        detail=str(exc),
                        code="generated_test_repair_failed",
                    )
                ],
                sandbox_result=None,
                cached_sandbox_result=None,
                next_retry_node=None,
            )

        should_continue = bool(test_result.updated_files) or test_result.available
        unresolved = (
            self._persisting_generated_test_blockers(
                run=run,
                failure_context=failure_context,
                deliverable_ids=deliverable_ids,
            )
            if should_continue
            else []
        )
        if unresolved:
            should_continue = False
        severity = ReviewerFindingSeverity.info if should_continue else ReviewerFindingSeverity.error
        return self._append_node(
            {**state, "run": run},
            kind=kind,
            attempt=attempt,
            status=WorkflowNodeStatus.passed if should_continue else WorkflowNodeStatus.failed,
            summary=(
                "Repair regenerated learner-visible and hidden test scripts."
                if should_continue
                else (
                    "Repair regenerated the test scripts, but the same blocking issues still remain."
                    if unresolved
                    else "Repair could not improve the generated test scripts."
                )
            ),
            findings=[
                ReviewerFinding(
                    category="test_authoring",
                    severity=severity,
                    title="Generated test repair",
                    detail=(
                        test_result.message
                        if not unresolved
                        else test_result.message
                        + " Unresolved blockers: "
                        + ", ".join(
                            f"{code} @ {location}" if location else code
                            for code, location in unresolved[:5]
                        )
                    ),
                    code=(
                        None
                        if should_continue
                        else ("generated_test_repair_unresolved" if unresolved else "generated_test_repair_unavailable")
                    ),
                )
            ],
            sandbox_result=None,
            cached_sandbox_result=None,
            next_retry_node=next_retry_node if should_continue else None,
        )

    def _target_deliverable_ids(self, run: WorkflowRun, failure_context) -> list[str]:
        spec = run.artifacts.task_agent_spec
        if spec is None:
            return []
        known_ids = {deliverable.id for deliverable in spec.deliverables}
        target_ids: set[str] = set()
        if failure_context.sandbox is not None:
            target_ids.update(
                deliverable_id
                for deliverable_id in failure_context.sandbox.failed_deliverables
                if deliverable_id in known_ids
            )
        for finding in failure_context.findings:
            location = (finding.location or "").replace("\\", "/")
            for deliverable_id in known_ids:
                if f"/{deliverable_id}/" in f"/{location}/":
                    target_ids.add(deliverable_id)
        return sorted(target_ids)

    def _persisting_generated_test_blockers(
        self,
        *,
        run: WorkflowRun,
        failure_context,
        deliverable_ids: list[str],
    ) -> list[tuple[str, str | None]]:
        prior_blockers = self._blocking_issue_keys(failure_context.findings, failure_context.validation_issues)
        if not prior_blockers:
            return []

        current_blockers = self._current_generated_test_blockers(
            run=run,
            deliverable_ids=deliverable_ids,
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

    def _current_generated_test_blockers(
        self,
        *,
        run: WorkflowRun,
        deliverable_ids: list[str],
    ) -> set[tuple[str, str | None]]:
        spec = run.artifacts.task_agent_spec
        workspace = run.artifacts.workspace_snapshot
        if spec is None or workspace is None:
            return {("generated_test_workspace_missing", None)}

        public_dir = Path(workspace.public_dir)
        known_ids = {deliverable.id for deliverable in spec.deliverables}
        target_ids = set(deliverable_ids) & known_ids if deliverable_ids else known_ids
        blockers: set[tuple[str, str | None]] = set()
        for deliverable in spec.deliverables:
            if deliverable.id not in target_ids:
                continue
            deliverable_dir = public_dir / "starter" / deliverable.id
            manifest_path = deliverable_dir / HIDDEN_MANIFEST_PATH
            visible_check_path = deliverable_dir / "checks" / "run_visible_checks.py"
            hidden_check_path = deliverable_dir / HIDDEN_GRADER_SCRIPT_PATH
            if not manifest_path.exists():
                blockers.add(("generated_test_manifest_missing", f"starter/{deliverable.id}/{HIDDEN_MANIFEST_PATH}"))
                continue
            if not visible_check_path.exists() or not hidden_check_path.exists():
                blockers.add(("generated_test_scripts_missing", f"starter/{deliverable.id}"))
                continue

            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            generated_test_scripts = manifest_payload.get("generated_test_scripts") or {}
            generated_test_source = str(generated_test_scripts.get("source") or "").strip().lower()
            if generated_test_source in {"", "starter_default"}:
                blockers.add(("generated_test_scripts_not_authored", f"starter/{deliverable.id}/{HIDDEN_MANIFEST_PATH}"))

            baseline = self.baseline_verifier.verify_deliverable(
                workspace_root=deliverable_dir,
                spec=spec,
                starter_type=spec.runtime_dependencies.starter_type,
            )
            blockers.update(
                (issue.code, issue.relative_path or f"starter/{deliverable.id}")
                for issue in baseline.errors
            )
        return blockers

    def _blocking_issue_keys(
        self,
        findings,
        validation_issues,
    ) -> set[tuple[str, str | None]]:
        keys: set[tuple[str, str | None]] = set()
        for issue in validation_issues:
            keys.add((issue.code, issue.location or None))
        for finding in findings:
            if finding.severity != ReviewerFindingSeverity.error:
                continue
            code = finding.code or self._normalize_finding_code(finding.title)
            if code:
                keys.add((code, finding.location or None))
        return keys

    def _normalize_finding_code(self, title: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", "_", title.strip().lower()).strip("_")
        return normalized or None

    def _reviewer_code_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        state, sandbox_result = self._sandbox_result(state)
        spec = state["run"].artifacts.task_agent_spec
        assert spec is not None
        primary_editable_paths = learner_editable_paths_for_spec(spec)
        entrypoint_path = primary_editable_paths[0] if primary_editable_paths else "the learner-owned repo surface"

        findings: list[ReviewerFinding] = [
            ReviewerFinding(
                category="code_review",
                severity=ReviewerFindingSeverity.info,
                title="Starter runtime entrypoint present",
                detail=(
                    "The generated starter surface includes the canonical endpoints required by the published contract "
                    f"through `{entrypoint_path}`."
                ),
            )
        ]
        if state["run"].artifacts.workspace_snapshot is not None:
            placeholder_deliverables = []
            wrapper_deliverables = []
            for deliverable in spec.deliverables:
                deliverable_dir = Path(state["run"].artifacts.workspace_snapshot.public_dir) / "starter" / deliverable.id
                editable_paths = learner_editable_paths_for_deliverable(spec, deliverable)
                deliverable_has_placeholder = False
                deliverable_has_wrapper = False
                for relative_path in editable_paths:
                    deliverable_app = deliverable_dir / relative_path
                    try:
                        source = deliverable_app.read_text(encoding="utf-8")
                    except OSError:
                        deliverable_has_placeholder = True
                        continue
                    if "Implement /run" in source or "status_code=501" in source:
                        deliverable_has_placeholder = True
                    if "from runtime.task_agent_runtime import" in source or (
                        "app = create_app_from_manifest(" in source
                        and "def create_app_from_manifest(" not in source
                    ):
                        deliverable_has_wrapper = True
                if deliverable_has_placeholder:
                    placeholder_deliverables.append(deliverable.id)
                if deliverable_has_wrapper:
                    wrapper_deliverables.append(deliverable.id)
            if placeholder_deliverables:
                findings.append(
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.error,
                        title="Placeholder starter endpoints remain",
                        detail="The workspace still contains placeholder starter code for: "
                        + ", ".join(placeholder_deliverables),
                        code="placeholder_starter_endpoints_remain",
                        location="starter",
                    )
                )
            if wrapper_deliverables:
                findings.append(
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.error,
                        title="Primary starter surface is still a thin wrapper",
                        detail=(
                            "The learner-owned files should contain the real application flow, not just import a generated "
                            "runtime wrapper. Affected review areas: " + ", ".join(wrapper_deliverables)
                        ),
                        code="starter_surface_thin_wrapper",
                        location="starter",
                    )
                )
            if not placeholder_deliverables and not wrapper_deliverables:
                findings.append(
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.info,
                        title="Workspace starter apps expose a learner-owned implementation surface",
                        detail="The workspace starter files expose substantive learner-owned entrypoints instead of placeholder handlers or thin runtime wrappers.",
                    )
                )
            starter_surface_review = inspect_materialized_starter_surface(spec, state["run"].artifacts.workspace_snapshot)
            for issue in starter_surface_review.errors:
                findings.append(
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.error,
                        title=issue.code,
                        detail=issue.message,
                        code=issue.code,
                        location=issue.relative_path,
                    )
                )
            for issue in starter_surface_review.warnings:
                findings.append(
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.warning,
                        title=issue.code,
                        detail=issue.message,
                        code=issue.code,
                        location=issue.relative_path,
                    )
                )
        findings.append(
            ReviewerFinding(
                category="code_review",
                severity=ReviewerFindingSeverity.info,
                title="Starter surface reviewed as real application code",
                detail="Reviewer code is checking the learner-owned entrypoint directly instead of an internal workflow simulator.",
            )
        )

        status = WorkflowNodeStatus.passed
        if sandbox_result.status != SandboxExecutionStatus.passed or any(
            finding.severity == ReviewerFindingSeverity.error for finding in findings
        ):
            status = WorkflowNodeStatus.failed

        return self._append_node(
            state,
            kind=WorkflowNodeKind.reviewer_code,
            attempt=state["reviewer_attempt"],
            status=status,
            summary="Reviewer code node checked starter project shape, safety rails, and Docker execution.",
            findings=findings,
            sandbox_result=None,
            cached_sandbox_result=sandbox_result,
        )

    def _reviewer_pedagogy_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        state, sandbox_result = self._sandbox_result(state)
        spec = state["run"].artifacts.task_agent_spec
        assert spec is not None
        workspace = state["run"].artifacts.workspace_snapshot

        findings: list[ReviewerFinding] = []
        deliverable_count = len(spec.deliverables)
        if spec.package_type.value == "progressive_codebase_course" and deliverable_count < 3:
            findings.append(
                ReviewerFinding(
                    category="pedagogy_review",
                    severity=ReviewerFindingSeverity.warning,
                    title="Short progressive deliverable plan",
                    detail="Progressive courses are easier to teach when they have at least three meaningful review areas.",
                )
            )
        else:
            findings.append(
                ReviewerFinding(
                    category="pedagogy_review",
                    severity=ReviewerFindingSeverity.info,
                    title="Deliverable plan present",
                    detail=f"The assignment defines {deliverable_count} learner review area(s).",
                )
            )

        for deliverable in spec.deliverables:
            gate = spec.gate_for(deliverable.id)
            if not gate.active_test_ids:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.warning,
                        title=f"{deliverable.id} has no active gate",
                        detail="Each deliverable should light up at least one behavior or quality bar.",
                    )
                )
            if not deliverable.learning_outcomes:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=f"{deliverable.id} is missing learning outcomes",
                        detail="Derive concrete deliverable outcomes from the learner task before sending this draft to human review.",
                    )
                )
            elif any(
                phrase in outcome.lower()
                for outcome in deliverable.learning_outcomes
                for phrase in ["understand", "learn about", "be familiar"]
            ):
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.warning,
                        title=f"{deliverable.id} outcomes are vague",
                        detail="Rewrite learning outcomes as observable capabilities, not general understanding goals.",
                    )
                )
            brief = deliverable.learner_brief
            starter_surface = deliverable.learner_starter_surface
            if starter_surface is None:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=f"{deliverable.id} is missing a starter surface",
                        detail="Authoring must describe the real learner-owned files, endpoints, and scenarios before this draft can be reviewed.",
                    )
                )
            elif not starter_surface.primary_editable_paths:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=f"{deliverable.id} starter surface has no primary files",
                        detail="Learners need a clear primary implementation surface, not just supporting files.",
                    )
                )
            elif any(
                phrase in " ".join(
                    [
                        scenario.title,
                        scenario.request_summary,
                        scenario.expected_behavior,
                    ]
                ).lower()
                for scenario in starter_surface.domain_scenarios
                for phrase in ["routine case", "ambiguous or risky case", "placeholder", *starter_surface_markers()]
            ):
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=f"{deliverable.id} starter scenarios are still generic",
                        detail="Replace placeholder scenarios with real domain cases that help the learner connect the prompt to the implementation.",
                    )
                )
            if brief is None:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=f"{deliverable.id} is missing a learner brief",
                        detail="Learners need a concrete task statement, files-to-edit guidance, examples, and a definition of done.",
                    )
                )
                continue
            if not brief.files_to_edit or not brief.definition_of_done:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=f"{deliverable.id} brief is underspecified",
                        detail="Call out the files to edit and what done looks like before asking a learner to work in the starter.",
                    )
                )
            if not brief.example_scenarios:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.warning,
                        title=f"{deliverable.id} needs concrete examples",
                        detail="Add at least one learner-facing example so the expected behavior is easier to visualize.",
                    )
                )
            elif starter_surface is not None and starter_surface.primary_editable_paths:
                missing_primary_paths = sorted(
                    set(starter_surface.primary_editable_paths) - set(brief.files_to_edit)
                )
                if missing_primary_paths:
                    findings.append(
                        ReviewerFinding(
                            category="pedagogy_review",
                            severity=ReviewerFindingSeverity.error,
                            title=f"{deliverable.id} brief drifts from the starter surface",
                            detail="The learner brief should point at the primary learner-owned files: "
                            + ", ".join(missing_primary_paths),
                        )
                    )
            brief_text = " ".join(
                [brief.why_this_deliverable_matters, brief.task_to_build, *brief.example_scenarios]
            ).lower()
            if "hidden checkpoint" in brief_text or "active checks" in brief_text:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=f"{deliverable.id} leaks internal grading language",
                        detail="Rewrite the learner brief in task language instead of referencing hidden checkpoints or active checks.",
                    )
                )
            elif deliverable.learning_outcomes:
                alignment_text = " ".join(
                    [brief.task_to_build, *brief.definition_of_done, *brief.example_scenarios]
                ).lower()
                if not any(
                    token in alignment_text
                    for outcome in deliverable.learning_outcomes
                    for token in outcome.lower().replace("`", "").replace("/", " ").split()
                    if len(token.strip(".,:;()[]{}")) >= 5
                ):
                    findings.append(
                        ReviewerFinding(
                            category="pedagogy_review",
                            severity=ReviewerFindingSeverity.warning,
                            title=f"{deliverable.id} outcomes may not match the learner task",
                            detail="The stated outcomes do not obviously match the current learner brief. Review them before approval.",
                        )
                    )

        if workspace is None:
            findings.append(
                ReviewerFinding(
                    category="pedagogy_review",
                    severity=ReviewerFindingSeverity.error,
                    title="Learner bundle missing",
                    detail="The materialized learner bundle is missing, so the public packaging could not be reviewed.",
                )
            )
        else:
            bundle_validation = validate_materialized_bundle(spec, workspace)
            for issue in bundle_validation.errors:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.error,
                        title=issue.code,
                        detail=issue.message,
                        code=issue.code,
                        location=issue.relative_path,
                    )
                )
            for issue in bundle_validation.warnings:
                findings.append(
                    ReviewerFinding(
                        category="pedagogy_review",
                        severity=ReviewerFindingSeverity.warning,
                        title=issue.code,
                        detail=issue.message,
                        code=issue.code,
                        location=issue.relative_path,
                    )
                )

        status = WorkflowNodeStatus.passed
        if sandbox_result.status != SandboxExecutionStatus.passed or any(
            finding.severity == ReviewerFindingSeverity.error for finding in findings
        ):
            status = WorkflowNodeStatus.failed

        return self._append_node(
            state,
            kind=WorkflowNodeKind.reviewer_pedagogy,
            attempt=state["reviewer_attempt"],
            status=status,
            summary="Reviewer pedagogy node checked the deliverable plan after verifying the assignment in Docker.",
            findings=findings,
            sandbox_result=None,
            cached_sandbox_result=sandbox_result,
        )

    def _reviewer_tests_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        state, sandbox_result = self._sandbox_result(state)
        spec = state["run"].artifacts.task_agent_spec
        assert spec is not None

        validation = validate_task_agent_spec(spec)
        findings: list[ReviewerFinding] = []
        learner_checks_valid = True

        if not validation.valid:
            findings.extend(
                ReviewerFinding(
                    category="tests_review",
                    severity=ReviewerFindingSeverity.error,
                    title=error.code,
                    detail=error.message,
                    code=error.code,
                    location=error.location,
                )
                for error in validation.errors
            )
        else:
            findings.append(
                ReviewerFinding(
                    category="tests_review",
                    severity=ReviewerFindingSeverity.info,
                    title="Spec validation passed",
                    detail="The assignment spec passed deterministic validation before review.",
                )
            )

        workspace = state["run"].artifacts.workspace_snapshot
        if workspace is None:
            learner_checks_valid = False
            findings.append(
                ReviewerFinding(
                    category="tests_review",
                    severity=ReviewerFindingSeverity.error,
                    title="Learner workspace missing",
                    detail="The persistent workspace snapshot is missing, so learner-visible checks could not be verified.",
                )
            )
        else:
            public_dir = Path(workspace.public_dir)
            for deliverable in spec.deliverables:
                deliverable_dir = public_dir / "starter" / deliverable.id
                manifest_path = deliverable_dir / HIDDEN_MANIFEST_PATH
                visible_check_path = deliverable_dir / "checks" / "run_visible_checks.py"
                hidden_check_path = deliverable_dir / HIDDEN_GRADER_SCRIPT_PATH
                tasks_path = deliverable_dir / ".vscode" / "tasks.json"
                missing_paths = [
                    path.relative_to(public_dir).as_posix()
                    for path in (manifest_path, visible_check_path, hidden_check_path, tasks_path)
                    if not path.exists()
                ]
                if missing_paths:
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title=f"Learner checks missing for {deliverable.id}",
                            detail="Missing learner-visible assets: " + ", ".join(missing_paths),
                        )
                    )
                    continue

                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                starter_surface = manifest_payload.get("learner_starter_surface") or {}
                visible_check_command = manifest_payload.get("visible_check_command")
                hidden_check_command = manifest_payload.get("hidden_check_command")
                generated_test_scripts = manifest_payload.get("generated_test_scripts") or {}
                generated_test_source = str(generated_test_scripts.get("source") or "").strip().lower()
                starter_repo_bundle = manifest_payload.get("starter_repo_bundle") or {}
                starter_repo_source = str(starter_repo_bundle.get("source") or "").strip().lower()
                runtime_protocol_bundle = manifest_payload.get("runtime_protocol_bundle") or {}
                runtime_protocol_source = str(runtime_protocol_bundle.get("source") or "").strip().lower()
                if (
                    visible_check_command != f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"
                    or hidden_check_command != f"sh {RUNTIME_HIDDEN_CHECK_SCRIPT_PATH}"
                    or not starter_surface.get("primary_editable_paths")
                    or not starter_surface.get("required_endpoints")
                ):
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title=f"Generated test commands incomplete for {deliverable.id}",
                            detail=(
                                "The starter manifest must include the standard visible and hidden test commands "
                                "plus a real learner starter surface."
                            ),
                            code="generated_test_commands_incomplete",
                            location=f"starter/{deliverable.id}/{HIDDEN_MANIFEST_PATH}",
                        )
                    )
                    continue
                if starter_repo_source in {"", "starter_default"}:
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title="starter_repo_bundle_not_authored",
                            detail=(
                                "The starter repo is still using the default protocol-only files. "
                                "Authoring must produce the learner-owned repo bundle before review."
                            ),
                            code="starter_repo_bundle_not_authored",
                            location=f"starter/{deliverable.id}/{HIDDEN_MANIFEST_PATH}",
                        )
                    )
                    continue
                if runtime_protocol_source in {"", "starter_default"}:
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title="runtime_protocol_bundle_not_authored",
                            detail=(
                                "The runtime install, verify, and run protocol is still using the default placeholders. "
                                "Authoring must produce the real runtime scripts before review."
                            ),
                            code="runtime_protocol_bundle_not_authored",
                            location=f"starter/{deliverable.id}/{HIDDEN_MANIFEST_PATH}",
                        )
                    )
                    continue
                if generated_test_source in {"", "starter_default"}:
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title="generated_test_scripts_not_authored",
                            detail=(
                                "The starter is still using the default generated test placeholders. "
                                "Authoring must produce real visible and hidden scripts from the materialized workspace."
                            ),
                            code="generated_test_scripts_not_authored",
                            location=f"starter/{deliverable.id}/{HIDDEN_MANIFEST_PATH}",
                        )
                    )
                    continue

                baseline = self.baseline_verifier.verify_deliverable(
                    workspace_root=deliverable_dir,
                    spec=spec,
                    starter_type=spec.runtime_dependencies.starter_type,
                )
                for issue in baseline.errors:
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title=issue.code,
                            detail=issue.message,
                            code=issue.code,
                            location=issue.relative_path or f"starter/{deliverable.id}",
                        )
                    )
                for issue in baseline.warnings:
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.warning,
                            title=issue.code,
                            detail=issue.message,
                            code=issue.code,
                            location=issue.relative_path or f"starter/{deliverable.id}",
                        )
                    )
                if baseline.valid:
                    outcome_bits = [
                        f"{outcome.baseline}:{outcome.suite_type}={len(outcome.report.tests)}"
                        for outcome in baseline.outcomes
                    ]
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.info,
                            title=f"Generated tests discriminate correctly for {deliverable.id}",
                            detail=(
                                "Baseline matrix verified the empty repo and untouched starter against the generated visible "
                                "and hidden scripts: " + ", ".join(outcome_bits)
                            ),
                        )
                    )

        status = WorkflowNodeStatus.passed
        if sandbox_result.status != SandboxExecutionStatus.passed or not validation.valid or not learner_checks_valid:
            status = WorkflowNodeStatus.failed

        return self._append_node(
            state,
            kind=WorkflowNodeKind.reviewer_tests,
            attempt=state["reviewer_attempt"],
            status=status,
            summary="Reviewer test node verified deterministic validation plus the generated visible/hidden test baseline matrix.",
            findings=findings,
            sandbox_result=None,
            cached_sandbox_result=sandbox_result,
        )

    def _sandbox_result(
        self,
        state: AssignmentGraphState,
        *,
        force: bool = False,
    ) -> tuple[AssignmentGraphState, SandboxExecutionResult]:
        if not force and state.get("cached_sandbox_result") is not None:
            return state, state["cached_sandbox_result"]
        sandbox_result = self.sandbox_runner.execute(state["run"])
        return {**state, "cached_sandbox_result": sandbox_result}, sandbox_result

    def _append_node(
        self,
        state: AssignmentGraphState,
        *,
        kind: WorkflowNodeKind,
        attempt: int,
        status: WorkflowNodeStatus,
        summary: str,
        findings: list[ReviewerFinding],
        sandbox_result,
        authoring_attempt: int | None = None,
        reviewer_attempt: int | None = None,
        cached_sandbox_result: SandboxExecutionResult | None | object = _UNSET,
        next_retry_node: str | None | object = _UNSET,
        skip_workspace_authoring: bool | object = _UNSET,
    ) -> AssignmentGraphState:
        executions = list(state["node_executions"])
        executions.append(
            WorkflowNodeExecution(
                node_id=f"{kind.value}_{len(executions) + 1}",
                kind=kind,
                iteration=state["active_iteration"],
                attempt=attempt,
                status=status,
                summary=summary,
                created_at=datetime.now(UTC),
                sandbox_result=sandbox_result,
                findings=findings,
            )
        )
        return {
            "run": state["run"],
            "node_executions": executions,
            "active_iteration": state["active_iteration"],
            "authoring_attempt": state["authoring_attempt"] if authoring_attempt is None else authoring_attempt,
            "reviewer_attempt": state["reviewer_attempt"] if reviewer_attempt is None else reviewer_attempt,
            "cached_sandbox_result": (
                state.get("cached_sandbox_result")
                if cached_sandbox_result is _UNSET
                else cached_sandbox_result
            ),
            "next_retry_node": (
                state.get("next_retry_node")
                if next_retry_node is _UNSET
                else next_retry_node
            ),
            "skip_workspace_authoring": (
                state.get("skip_workspace_authoring", False)
                if skip_workspace_authoring is _UNSET
                else skip_workspace_authoring
            ),
        }
