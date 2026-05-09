from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.domain.sandbox import SandboxExecutionStatus
from app.domain.workflow import (
    ReviewerFinding,
    ReviewerFindingSeverity,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowLoopPolicy,
    WorkflowNodeStatus,
    WorkflowRun,
)
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.failure_context_builder import build_failure_context
from app.services.grader_planner import build_all_task_agent_review_area_plans
from app.services.review_area_coverage import summarize_review_area_hidden_coverage
from app.services.spec_validation import validate_task_agent_spec
from app.services.task_agent_repair_service import TaskAgentRepairService
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService


AuthoringRoute = Literal["authoring_repair", "reviewer_runtime", "end"]
ReviewerRoute = Literal["reviewer_repair", "reviewer_code", "reviewer_pedagogy", "reviewer_tests", "end"]


class AssignmentGraphState(TypedDict):
    run: WorkflowRun
    node_executions: list[WorkflowNodeExecution]
    authoring_attempt: int
    reviewer_attempt: int


class LangGraphAssignmentGraph:
    def __init__(
        self,
        sandbox_runner: DockerSandboxRunner,
        *,
        repair_service: TaskAgentRepairService | None = None,
        workspace_authoring_service: TaskAgentWorkspaceAuthoringService | None = None,
        max_authoring_attempts: int = 3,
        max_reviewer_attempts: int = 2,
    ) -> None:
        self.sandbox_runner = sandbox_runner
        self.repair_service = repair_service or TaskAgentRepairService()
        self.workspace_authoring_service = workspace_authoring_service or TaskAgentWorkspaceAuthoringService()
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

    def execute(self, run: WorkflowRun) -> WorkflowRun:
        if run.artifacts.task_agent_spec is None:
            return run
        state: AssignmentGraphState = {
            "run": run.model_copy(deep=True),
            "node_executions": [],
            "authoring_attempt": 0,
            "reviewer_attempt": 0,
        }
        result = self.graph.invoke(state)
        updated_run = result["run"]
        updated_run.artifacts.node_executions = result["node_executions"]
        return updated_run

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AssignmentGraphState)
        graph.add_node("authoring_runtime", self._authoring_runtime_node)
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
                "reviewer_runtime": "reviewer_runtime",
                "end": END,
            },
        )
        graph.add_edge("authoring_repair", "authoring_runtime")

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
        graph.add_edge("reviewer_repair", "reviewer_runtime")
        return graph

    def _after_authoring_runtime(self, state: AssignmentGraphState) -> AuthoringRoute:
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
            return "reviewer_repair"
        return "end"

    def _after_reviewer_pedagogy(self, state: AssignmentGraphState) -> ReviewerRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "reviewer_tests"
        if state["reviewer_attempt"] < self.max_reviewer_attempts:
            return "reviewer_repair"
        return "end"

    def _after_reviewer_tests(self, state: AssignmentGraphState) -> ReviewerRoute:
        latest = state["node_executions"][-1]
        if latest.status == WorkflowNodeStatus.passed:
            return "end"
        if state["reviewer_attempt"] < self.max_reviewer_attempts:
            return "reviewer_repair"
        return "end"

    def _authoring_runtime_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        run, authoring_result = self.workspace_authoring_service.author_workspace(state["run"])
        sandbox_result = self.sandbox_runner.execute(run)
        attempt = state["authoring_attempt"] + 1
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
            {**state, "run": run},
            kind=WorkflowNodeKind.authoring_runtime,
            attempt=attempt,
            status=status,
            summary=summary,
            findings=findings,
            sandbox_result=sandbox_result,
            authoring_attempt=attempt,
        )

    def _authoring_repair_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        latest = state["node_executions"][-1]
        failure_context = build_failure_context(state["run"], latest)
        run, workspace_repaired, workspace_message = self.workspace_authoring_service.repair_workspace(
            state["run"],
            latest,
            failure_context=failure_context,
        )
        run, spec_repaired, spec_message = self.repair_service.apply(
            run,
            latest,
            failure_context=failure_context,
        )
        if spec_repaired:
            run = self.workspace_authoring_service.sync_workspace(run)
        repaired = workspace_repaired or spec_repaired
        status = WorkflowNodeStatus.passed if repaired else WorkflowNodeStatus.failed
        detail_lines = []
        if workspace_message:
            detail_lines.append(workspace_message)
        if spec_message and spec_message != workspace_message:
            detail_lines.append(spec_message)
        return self._append_node(
            {**state, "run": run},
            kind=WorkflowNodeKind.authoring_repair,
            attempt=state["authoring_attempt"],
            status=status,
            summary=f"Authoring repair {'applied' if repaired else 'could not repair'} after runtime failure.",
            findings=[
                ReviewerFinding(
                    category="authoring_repair",
                    severity=ReviewerFindingSeverity.info if repaired else ReviewerFindingSeverity.error,
                    title="Authoring repair step",
                    detail=" ".join(detail_lines) if detail_lines else "No repair detail was available.",
                )
            ],
            sandbox_result=latest.sandbox_result,
        )

    def _reviewer_runtime_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        sandbox_result = self.sandbox_runner.execute(state["run"])
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

        return self._append_node(
            state,
            kind=WorkflowNodeKind.reviewer_runtime,
            attempt=attempt,
            status=status,
            summary=summary,
            findings=findings,
            sandbox_result=sandbox_result,
            reviewer_attempt=attempt,
        )

    def _reviewer_repair_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        latest = state["node_executions"][-1]
        failure_context = build_failure_context(state["run"], latest)
        run, workspace_repaired, workspace_message = self.workspace_authoring_service.repair_workspace(
            state["run"],
            latest,
            failure_context=failure_context,
        )
        run, spec_repaired, spec_message = self.repair_service.apply(
            run,
            latest,
            failure_context=failure_context,
        )
        if spec_repaired:
            run = self.workspace_authoring_service.sync_workspace(run)
        repaired = workspace_repaired or spec_repaired
        status = WorkflowNodeStatus.passed if repaired else WorkflowNodeStatus.failed
        detail_lines = []
        if workspace_message:
            detail_lines.append(workspace_message)
        if spec_message and spec_message != workspace_message:
            detail_lines.append(spec_message)
        return self._append_node(
            {**state, "run": run},
            kind=WorkflowNodeKind.reviewer_repair,
            attempt=state["reviewer_attempt"],
            status=status,
            summary=f"Reviewer repair {'applied' if repaired else 'could not repair'} the latest issue.",
            findings=[
                ReviewerFinding(
                    category="reviewer_repair",
                    severity=ReviewerFindingSeverity.info if repaired else ReviewerFindingSeverity.error,
                    title="Reviewer repair step",
                    detail=" ".join(detail_lines) if detail_lines else "No repair detail was available.",
                )
            ],
            sandbox_result=latest.sandbox_result,
        )

    def _reviewer_code_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        sandbox_result = self.sandbox_runner.execute(state["run"])
        spec = state["run"].artifacts.task_agent_spec
        assert spec is not None

        findings: list[ReviewerFinding] = [
            ReviewerFinding(
                category="code_review",
                severity=ReviewerFindingSeverity.info,
                title="FastAPI starter present",
                detail="The generated starter surface includes the canonical FastAPI endpoints required by the agent contract.",
            )
        ]
        if state["run"].artifacts.workspace_snapshot is not None:
            placeholder_deliverables = []
            for deliverable in spec.deliverables:
                deliverable_app = (
                    Path(state["run"].artifacts.workspace_snapshot.public_dir)
                    / "starter"
                    / deliverable.id
                    / "app.py"
                )
                try:
                    source = deliverable_app.read_text(encoding="utf-8")
                except OSError:
                    placeholder_deliverables.append(deliverable.id)
                    continue
                if "Implement /run" in source or "status_code=501" in source:
                    placeholder_deliverables.append(deliverable.id)
            if placeholder_deliverables:
                findings.append(
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.error,
                        title="Placeholder starter endpoints remain",
                        detail="The workspace still contains placeholder starter code for: "
                        + ", ".join(placeholder_deliverables),
                    )
                )
            else:
                findings.append(
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.info,
                        title="Workspace starter apps are runnable",
                        detail="The workspace starter files expose the runtime-backed wrapper instead of placeholder 501 handlers.",
                    )
                )
        if spec.production_contract.supports_dry_run:
            findings.append(
                ReviewerFinding(
                    category="code_review",
                    severity=ReviewerFindingSeverity.info,
                    title="Dry-run contract preserved",
                    detail="The production contract keeps dry-run support visible in the generated assignment.",
                )
            )
        if any(tool.safety.value == "irreversible" and not tool.approval_required for tool in spec.tool_registry.tools):
            findings.append(
                ReviewerFinding(
                    category="code_review",
                    severity=ReviewerFindingSeverity.error,
                    title="Irreversible tool without approval",
                    detail="At least one irreversible tool is missing an approval gate.",
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
            sandbox_result=sandbox_result,
        )

    def _reviewer_pedagogy_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        sandbox_result = self.sandbox_runner.execute(state["run"])
        spec = state["run"].artifacts.task_agent_spec
        assert spec is not None

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
            sandbox_result=sandbox_result,
        )

    def _reviewer_tests_node(self, state: AssignmentGraphState) -> AssignmentGraphState:
        sandbox_result = self.sandbox_runner.execute(state["run"])
        spec = state["run"].artifacts.task_agent_spec
        assert spec is not None

        validation = validate_task_agent_spec(spec)
        grader_plans = build_all_task_agent_review_area_plans(spec)
        hidden_coverage = {
            summary.deliverable_id: summary
            for summary in summarize_review_area_hidden_coverage(spec)
        }
        findings: list[ReviewerFinding] = []
        learner_checks_valid = True

        if not validation.valid:
            findings.extend(
                ReviewerFinding(
                    category="tests_review",
                    severity=ReviewerFindingSeverity.error,
                    title=error.code,
                    detail=error.message,
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

        for plan in grader_plans.deliverable_plans:
            coverage = hidden_coverage.get(plan.deliverable_id)
            hidden_case_count = len(coverage.hidden_case_ids) if coverage is not None else 0
            findings.append(
                ReviewerFinding(
                    category="tests_review",
                    severity=ReviewerFindingSeverity.info,
                    title=f"Hidden grader coverage ready for {plan.deliverable_id}",
                    detail=(
                        f"This review area activates {plan.total_tests} hidden test(s) across "
                        f"{hidden_case_count} tagged eval case(s)."
                    ),
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
                manifest_path = deliverable_dir / "starter_manifest.json"
                visible_check_path = deliverable_dir / "checks" / "run_visible_checks.py"
                tasks_path = deliverable_dir / ".vscode" / "tasks.json"
                missing_paths = [
                    path.relative_to(public_dir).as_posix()
                    for path in (manifest_path, visible_check_path, tasks_path)
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
                public_checks = manifest_payload.get("public_checks") or []
                public_check_cases = manifest_payload.get("public_check_cases") or []
                visible_check_command = manifest_payload.get("visible_check_command")
                if (
                    not public_checks
                    or not public_check_cases
                    or len(public_checks) != len(public_check_cases)
                    or visible_check_command != "python checks/run_visible_checks.py"
                ):
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title=f"Visible learner checks incomplete for {deliverable.id}",
                            detail=(
                                "Starter manifest must include reviewed public_checks, matching public_check_cases, "
                                "and the standard visible_check_command."
                            ),
                        )
                    )
                    continue

                malformed_check = next(
                    (
                        check
                        for check in public_checks
                        if not check.get("title")
                        or not check.get("learner_goal")
                        or not check.get("expected_assertions")
                    ),
                    None,
                )
                if malformed_check is not None:
                    learner_checks_valid = False
                    findings.append(
                        ReviewerFinding(
                            category="tests_review",
                            severity=ReviewerFindingSeverity.error,
                            title=f"Reviewed public checks are incomplete for {deliverable.id}",
                            detail=(
                                "Each learner-visible public check must include a title, learner goal, and expected assertions "
                                "before the deliverable can pass review."
                            ),
                        )
                    )
                    continue

                findings.append(
                    ReviewerFinding(
                        category="tests_review",
                        severity=ReviewerFindingSeverity.info,
                        title=f"Visible learner checks ready for {deliverable.id}",
                        detail=(
                            f"Learners can run `{visible_check_command}` with {len(public_checks)} reviewed public check(s) "
                            "before submitting to the deeper hidden grader."
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
            summary="Reviewer test node verified Docker execution, deterministic spec validation, and grader coverage.",
            findings=findings,
            sandbox_result=sandbox_result,
        )

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
    ) -> AssignmentGraphState:
        executions = list(state["node_executions"])
        executions.append(
            WorkflowNodeExecution(
                node_id=f"{kind.value}_{len(executions) + 1}",
                kind=kind,
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
            "authoring_attempt": state["authoring_attempt"] if authoring_attempt is None else authoring_attempt,
            "reviewer_attempt": state["reviewer_attempt"] if reviewer_attempt is None else reviewer_attempt,
        }
