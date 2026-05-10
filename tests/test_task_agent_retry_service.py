from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from app.domain.ai import AIUsageSummary
from app.domain.sandbox import DeliverableSandboxReport, SandboxExecutionResult, SandboxExecutionStatus
from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity, WorkflowNodeExecution, WorkflowNodeKind, WorkflowNodeStatus
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.failure_context_builder import build_failure_context
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
from app.services.openai_task_agent_authoring import (
    OpenAITaskAgentAuthoringService,
    TaskAgentAuthoringResult,
    TaskAgentAuthoringSource,
    TaskAgentAuthoringStatus,
)
from app.services.task_agent_retry_service import TaskAgentRetryAction, TaskAgentRetryService
from app.services.task_agent_workspace_authoring import (
    TaskAgentWorkspaceAuthoringService,
    WorkspaceRepairSmokeResult,
)
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class _RevisingAuthoringService:
    def __init__(self) -> None:
        self.last_failure_context = None

    def revise_spec(self, *, spec, failure_context=None, origin_template=None, **kwargs):  # noqa: ANN001
        self.last_failure_context = failure_context
        revised = spec.model_copy(deep=True)
        revised.summary = spec.summary + " Revised from failure packet."
        revised.deliverables[0].title = revised.deliverables[0].title + " Revised"
        status = TaskAgentAuthoringStatus(
            available=True,
            source=TaskAgentAuthoringSource.openai_live,
            message="fake reviser",
            sdk_installed=True,
            api_key_present=True,
            model_id="fake-model",
            env_file=None,
        )
        return TaskAgentAuthoringResult(
            spec=revised,
            origin_template=f"openai_revision:{origin_template or 'task_agent_spec'}",
            source=TaskAgentAuthoringSource.openai_live,
            notes=["revised by fake authoring service"],
            status=status,
            usage=AIUsageSummary(request_count=1, input_tokens=10, output_tokens=10, total_tokens=20),
        )


class _NoChangeAuthoringService:
    def revise_spec(self, *, spec, origin_template=None, **kwargs):  # noqa: ANN001
        status = TaskAgentAuthoringStatus(
            available=False,
            source=TaskAgentAuthoringSource.deterministic_fallback,
            message="no-op reviser",
            sdk_installed=True,
            api_key_present=False,
            model_id="fake-model",
            env_file=None,
        )
        return TaskAgentAuthoringResult(
            spec=spec,
            origin_template=origin_template or "task_agent_spec",
            source=TaskAgentAuthoringSource.deterministic_fallback,
            notes=["left unchanged"],
            status=status,
            usage=None,
        )


class _AlwaysFailSandboxRunner:
    def status(self):
        return None

    def execute(self, run):  # noqa: ANN001
        return _authored_failure_sandbox_result(run.id)


class _RepairingWorkspaceAuthoringService(TaskAgentWorkspaceAuthoringService):
    def __init__(self, workspace_manager: AssignmentWorkspaceManager, *, smoke_passed: bool = True) -> None:
        super().__init__(workspace_manager)
        self.repair_calls = 0
        self.smoke_checks = 0
        self.smoke_passed = smoke_passed

    def repair_workspace(self, run, latest_node, failure_context=None):  # noqa: ANN001
        self.repair_calls += 1
        run = self.ensure_workspace(run)
        return run, True, "Fake workspace repair rewrote the learner repo."

    def smoke_verify_repair(self, run, latest_node, *, failure_context=None):  # noqa: ANN001
        self.smoke_checks += 1
        if self.smoke_passed:
            return WorkspaceRepairSmokeResult(
                passed=True,
                summary="Fake workspace smoke verification passed.",
            )
        return WorkspaceRepairSmokeResult(
            passed=False,
            summary="Fake workspace smoke verification still fails with the same build blocker.",
        )


def _make_workflow_service(temp_dir: Path, *, node_runtime=None, task_agent_authoring_service=None) -> WorkflowService:
    store = SQLiteWorkflowStore(db_path=temp_dir / "course_gen.db")
    workspace_manager = AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
    return WorkflowService(
        store,
        node_runtime=node_runtime,
        task_agent_authoring_service=task_agent_authoring_service or OpenAITaskAgentAuthoringService(enabled=False),
        workspace_manager=workspace_manager,
    )


def _make_run(temp_dir: Path):
    workflow_service = _make_workflow_service(temp_dir)
    intake = GenerationIntake(
        title="Inventory reservations",
        problem_statement="Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis.",
        learning_outcomes=["keep reservations correct under concurrency"],
        implementation_language="python",
        application_framework="fastapi",
        primary_database="postgres",
        cache_backend="redis",
        tech_stack=["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"],
    )
    inferred = infer_assignment_design(
        title=intake.title,
        problem_statement=intake.problem_statement,
        package_type_hint=intake.package_type_hint,
        starter_type=intake.starter_type,
        implementation_language=intake.implementation_language,
        application_framework=intake.application_framework,
        primary_database=intake.primary_database,
        cache_backend=intake.cache_backend,
        tech_stack=intake.tech_stack,
        data_sources=intake.data_sources,
    )
    assert inferred.design_spec is not None
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake,
        design_spec=inferred.design_spec,
        execute_nodes=False,
    )
    return run


def _platform_failure_node() -> WorkflowNodeExecution:
    return WorkflowNodeExecution(
        node_id="authoring_runtime_1",
        kind=WorkflowNodeKind.authoring_runtime,
        status=WorkflowNodeStatus.failed,
        attempt=1,
        summary="Generated assignment failed to boot in Docker.",
        created_at=datetime.now(UTC),
        sandbox_result=SandboxExecutionResult(
            status=SandboxExecutionStatus.failed,
            available=True,
            build_succeeded=True,
            run_succeeded=False,
            generated_at=datetime.now(UTC),
            run_stderr="Error response from daemon: No such network: course-gen-sandbox-net",
            error="Error response from daemon: No such network: course-gen-sandbox-net",
            deliverable_reports=[
                DeliverableSandboxReport(
                    deliverable_id="deliverable_1",
                    compile_succeeded=True,
                    runtime_succeeded=False,
                    error="Error response from daemon: No such network: course-gen-sandbox-net",
                )
            ],
        ),
        findings=[
            ReviewerFinding(
                category="runtime",
                severity=ReviewerFindingSeverity.error,
                title="Sandbox verification failed",
                detail="No such network: course-gen-sandbox-net",
            )
        ],
    )


def _authored_failure_node() -> WorkflowNodeExecution:
    return WorkflowNodeExecution(
        node_id="authoring_runtime_1",
        kind=WorkflowNodeKind.authoring_runtime,
        status=WorkflowNodeStatus.failed,
        attempt=1,
        summary="Generated assignment failed to boot in Docker.",
        created_at=datetime.now(UTC),
        sandbox_result=_authored_failure_sandbox_result("run_test"),
        findings=[
            ReviewerFinding(
                category="runtime",
                severity=ReviewerFindingSeverity.error,
                title="Sandbox verification failed",
                detail="FastAPIError during route registration.",
            )
        ],
    )


def _approval_claim_failure_node() -> WorkflowNodeExecution:
    return WorkflowNodeExecution(
        node_id="reviewer_code_1",
        kind=WorkflowNodeKind.reviewer_code,
        status=WorkflowNodeStatus.failed,
        attempt=1,
        summary="Reviewer code node found an unsupported approval claim in the learner-facing README.",
        created_at=datetime.now(UTC),
        sandbox_result=None,
        findings=[
            ReviewerFinding(
                category="code_review",
                severity=ReviewerFindingSeverity.error,
                title="course_readme_unbacked_approval_claim",
                detail="Course README mentions approval flow semantics that are not part of this project contract.",
                code="course_readme_unbacked_approval_claim",
                location="public/README.md",
            )
        ],
    )


def _pedagogy_failure_node() -> WorkflowNodeExecution:
    return WorkflowNodeExecution(
        node_id="reviewer_pedagogy_1",
        kind=WorkflowNodeKind.reviewer_pedagogy,
        status=WorkflowNodeStatus.failed,
        attempt=1,
        summary="Reviewer pedagogy node found a weak learner brief.",
        created_at=datetime.now(UTC),
        sandbox_result=None,
        findings=[
            ReviewerFinding(
                category="pedagogy_review",
                severity=ReviewerFindingSeverity.error,
                title="Learner brief needs a clearer opening task",
                detail="Deliverable 1 should name the first learner outcome more explicitly in the README.",
                code="learner_brief_opening_unclear",
                location="public/starter/deliverable_1/README.md",
            )
        ],
    )


def _platform_compiler_failure_node() -> WorkflowNodeExecution:
    return WorkflowNodeExecution(
        node_id="authoring_runtime_1",
        kind=WorkflowNodeKind.authoring_runtime,
        status=WorkflowNodeStatus.failed,
        attempt=1,
        summary="Generated assignment failed to boot in Docker.",
        created_at=datetime.now(UTC),
        sandbox_result=SandboxExecutionResult(
            status=SandboxExecutionStatus.failed,
            available=True,
            build_succeeded=True,
            run_succeeded=False,
            generated_at=datetime.now(UTC),
            workspace_root="/tmp/run_test",
            run_stderr=(
                "[coursegen] verify step 1 started\n"
                "Traceback (most recent call last):\n"
                "  File \"/workspace/starter/deliverable_1/app.py\", line 599, in <module>\n"
                "    app = create_app_from_manifest(MANIFEST_PATH)\n"
                "  File \"/workspace/starter/deliverable_1/app.py\", line 590, in create_app_from_manifest\n"
                "    app.add_api_route(\n"
                "fastapi.exceptions.FastAPIError: Invalid args for response field!"
            ),
            error="Starter deliverable verification failed on the authored runtime harness.",
            deliverable_reports=[
                DeliverableSandboxReport(
                    deliverable_id="deliverable_1",
                    compile_succeeded=True,
                    runtime_succeeded=False,
                    error="FastAPIError: Invalid args for response field!",
                    stderr=(
                        "[coursegen] verify step 1 started\n"
                        "create_app_from_manifest\n"
                        "app.add_api_route\n"
                        "fastapi.exceptions.FastAPIError: Invalid args for response field!"
                    ),
                )
            ],
        ),
        findings=[
            ReviewerFinding(
                category="runtime",
                severity=ReviewerFindingSeverity.error,
                title="Sandbox verification failed",
                detail="FastAPIError during route registration in create_app_from_manifest.",
            )
        ],
    )


def _authored_failure_sandbox_result(run_id: str) -> SandboxExecutionResult:
    return SandboxExecutionResult(
        status=SandboxExecutionStatus.failed,
        available=True,
        build_succeeded=True,
        run_succeeded=False,
        generated_at=datetime.now(UTC),
        workspace_root=f"/tmp/{run_id}",
        run_stderr=(
            "Traceback (most recent call last):\n"
            "fastapi.exceptions.FastAPIError: Invalid args for response field!"
        ),
        error="Starter deliverable verification failed on the authored runtime harness.",
        deliverable_reports=[
            DeliverableSandboxReport(
                deliverable_id="deliverable_1",
                compile_succeeded=True,
                runtime_succeeded=False,
                error="FastAPIError: Invalid args for response field!",
                stderr="fastapi.exceptions.FastAPIError: Invalid args for response field!",
            )
        ],
    )


class TaskAgentRetryServiceTests(TestCase):
    def test_retry_service_stops_on_platform_runtime_failure(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            run.artifacts.validation_summary = {"valid": True, "errors": [], "warnings": []}
            retry_service = TaskAgentRetryService(
                authoring_service=OpenAITaskAgentAuthoringService(enabled=False),
                workspace_authoring_service=TaskAgentWorkspaceAuthoringService(
                    AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
                ),
            )

            updated_run, result = retry_service.retry(run, _platform_failure_node())

            self.assertIs(updated_run, run)
            self.assertFalse(result.should_continue)
            self.assertFalse(result.applied)
            self.assertEqual(result.action, TaskAgentRetryAction.blocked_platform)
            self.assertEqual(result.owner_hint.value, "platform_runtime")

    def test_retry_service_repairs_workspace_from_runtime_failure_packet(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            reviser = _RevisingAuthoringService()
            workspace_authoring = _RepairingWorkspaceAuthoringService(
                AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
            )
            retry_service = TaskAgentRetryService(
                authoring_service=reviser,
                workspace_authoring_service=workspace_authoring,
            )

            updated_run, result = retry_service.retry(run, _authored_failure_node())

            self.assertTrue(result.should_continue)
            self.assertTrue(result.applied)
            self.assertEqual(result.action, TaskAgentRetryAction.revised)
            self.assertEqual(result.owner_hint.value, "authored_artifact")
            self.assertEqual(result.before_spec_hash, result.after_spec_hash)
            self.assertIsNone(reviser.last_failure_context)
            self.assertIsNotNone(updated_run.artifacts.workspace_snapshot)
            self.assertEqual(workspace_authoring.repair_calls, 1)
            self.assertEqual(workspace_authoring.smoke_checks, 0)
            self.assertIn("re-authored the learner workspace", result.summary)

    def test_retry_service_workspace_repair_continues_even_without_smoke_gate(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            reviser = _RevisingAuthoringService()
            workspace_authoring = _RepairingWorkspaceAuthoringService(
                AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces"),
                smoke_passed=False,
            )
            retry_service = TaskAgentRetryService(
                authoring_service=reviser,
                workspace_authoring_service=workspace_authoring,
            )

            updated_run, result = retry_service.retry(run, _authored_failure_node())

            self.assertTrue(result.applied)
            self.assertTrue(result.should_continue)
            self.assertEqual(result.action, TaskAgentRetryAction.revised)
            self.assertEqual(result.owner_hint.value, "authored_artifact")
            self.assertIsNone(reviser.last_failure_context)
            self.assertIsNotNone(updated_run.artifacts.workspace_snapshot)
            self.assertEqual(workspace_authoring.repair_calls, 1)
            self.assertEqual(workspace_authoring.smoke_checks, 0)
            self.assertIn("re-authored the learner workspace", result.summary.lower())

    def test_retry_service_revises_spec_from_non_runtime_failure_packet(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            reviser = _RevisingAuthoringService()
            retry_service = TaskAgentRetryService(
                authoring_service=reviser,
                workspace_authoring_service=TaskAgentWorkspaceAuthoringService(
                    AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
                ),
            )

            updated_run, result = retry_service.retry(run, _pedagogy_failure_node())

            self.assertTrue(result.should_continue)
            self.assertTrue(result.applied)
            self.assertEqual(result.action, TaskAgentRetryAction.revised)
            self.assertEqual(result.owner_hint.value, "authored_artifact")
            self.assertNotEqual(result.before_spec_hash, result.after_spec_hash)
            self.assertIsNotNone(reviser.last_failure_context)
            self.assertTrue(updated_run.artifacts.task_agent_spec.deliverables[0].title.endswith("Revised"))
            self.assertIsNotNone(updated_run.artifacts.workspace_snapshot)

    def test_retry_service_blocks_platform_owned_starter_compiler_failure(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            run.artifacts.validation_summary = {"valid": True, "errors": [], "warnings": []}
            reviser = _RevisingAuthoringService()
            retry_service = TaskAgentRetryService(
                authoring_service=reviser,
                workspace_authoring_service=TaskAgentWorkspaceAuthoringService(
                    AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
                ),
            )

            updated_run, result = retry_service.retry(run, _platform_compiler_failure_node())

            self.assertIs(updated_run, run)
            self.assertFalse(result.should_continue)
            self.assertFalse(result.applied)
            self.assertEqual(result.action, TaskAgentRetryAction.blocked_platform)
            self.assertEqual(result.owner_hint.value, "platform_runtime")
            self.assertIsNone(reviser.last_failure_context)

    def test_retry_service_stops_when_same_bundle_blocker_persists_after_revision(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            assert run.artifacts.task_agent_spec is not None
            run.artifacts.task_agent_spec.title = "Inventory Reservation Service with Approval"
            reviser = _RevisingAuthoringService()
            retry_service = TaskAgentRetryService(
                authoring_service=reviser,
                workspace_authoring_service=TaskAgentWorkspaceAuthoringService(
                    AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
                ),
            )

            updated_run, result = retry_service.retry(run, _approval_claim_failure_node())

            self.assertTrue(result.applied)
            self.assertFalse(result.should_continue)
            self.assertEqual(result.action, TaskAgentRetryAction.unresolved_blocker)
            self.assertIn("course_readme_unbacked_approval_claim", result.detail)
            self.assertTrue(updated_run.artifacts.task_agent_spec.deliverables[0].title.endswith("Revised"))
            self.assertIsNotNone(updated_run.artifacts.workspace_snapshot)

    def test_langgraph_stops_after_no_material_retry(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            workspace_manager = AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
            graph = LangGraphAssignmentGraph(
                _AlwaysFailSandboxRunner(),
                authoring_service=_NoChangeAuthoringService(),
                workspace_authoring_service=TaskAgentWorkspaceAuthoringService(workspace_manager),
                max_authoring_attempts=3,
                max_reviewer_attempts=1,
            )
            workflow_service = WorkflowService(
                SQLiteWorkflowStore(db_path=temp_dir / "course_gen.db"),
                node_runtime=graph,
                task_agent_authoring_service=OpenAITaskAgentAuthoringService(enabled=False),
                workspace_manager=workspace_manager,
            )

            created = workflow_service.create_run(
                GenerationIntake(
                    title="Inventory reservations",
                    problem_statement="Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis.",
                    learning_outcomes=["keep reservations correct under concurrency"],
                    implementation_language="python",
                    application_framework="fastapi",
                    primary_database="postgres",
                    cache_backend="redis",
                    tech_stack=["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"],
                )
            )

            node_kinds = [node.kind.value for node in created.artifacts.node_executions]
            self.assertEqual(node_kinds, ["authoring_runtime", "authoring_repair"])
            self.assertEqual(created.stage.value, "blocked")
            self.assertEqual(created.status.value, "blocked")
            self.assertIn(
                "Retry produced no material spec changes",
                "\n".join(created.artifacts.review_summary.blockers),
            )

    def test_failure_context_reuses_runtime_sandbox_for_non_runtime_reviewer_nodes(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            runtime_node = WorkflowNodeExecution(
                node_id="reviewer_runtime_1",
                kind=WorkflowNodeKind.reviewer_runtime,
                iteration=2,
                attempt=1,
                status=WorkflowNodeStatus.passed,
                summary="Reviewer runtime node confirmed the assignment still boots in Docker.",
                created_at=datetime.now(UTC),
                sandbox_result=_authored_failure_sandbox_result(run.id),
                findings=[],
            )
            code_node = WorkflowNodeExecution(
                node_id="reviewer_code_2",
                kind=WorkflowNodeKind.reviewer_code,
                iteration=2,
                attempt=1,
                status=WorkflowNodeStatus.failed,
                summary="Reviewer code node failed after inspecting the starter surface.",
                created_at=datetime.now(UTC),
                sandbox_result=None,
                findings=[
                    ReviewerFinding(
                        category="code_review",
                        severity=ReviewerFindingSeverity.error,
                        title="Primary starter surface is still a thin wrapper",
                        detail="The learner-owned files still delegate to a generated wrapper.",
                    )
                ],
            )
            run.artifacts.node_executions = [runtime_node, code_node]
            run.artifacts.validation_summary = {"valid": True, "errors": [], "warnings": []}

            failure_context = build_failure_context(run, code_node)

            self.assertIsNotNone(failure_context.sandbox)
            assert failure_context.sandbox is not None
            self.assertIn("Starter deliverable verification failed", failure_context.sandbox.error or "")

    def test_failure_context_includes_dependency_contract_facts_for_failed_deliverables(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            assert spec is not None
            spec.project_contract.runtime_plan.implementation_language = "rust"
            spec.project_contract.runtime_plan.language_version = "1.82"
            spec.project_contract.runtime_plan.application_framework = "axum"
            spec.project_contract.runtime_plan.framework_version = "0.8.9"
            spec.project_contract.runtime_plan.package_manager = "cargo"
            if spec.project_contract.runtime_plan.services:
                spec.project_contract.runtime_plan.services[0].container_image = "rust:1.82-bookworm"
                spec.project_contract.runtime_plan.services[0].learner_managed = True
            run.artifacts.task_agent_spec = spec

            workspace_authoring = TaskAgentWorkspaceAuthoringService()
            run, _ = workspace_authoring.author_workspace(run)

            workspace = run.artifacts.workspace_snapshot
            assert workspace is not None
            starter_root = Path(workspace.public_dir) / "starter" / "deliverable_1"
            (starter_root / "Cargo.toml").write_text(
                "[package]\nname = 'demo'\nversion = '0.1.0'\nedition = '2021'\n",
                encoding="utf-8",
            )
            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")

            failure_node = WorkflowNodeExecution(
                node_id="authoring_runtime_1",
                kind=WorkflowNodeKind.authoring_runtime,
                iteration=1,
                attempt=1,
                status=WorkflowNodeStatus.failed,
                summary="Rust starter failed to build in Docker.",
                created_at=datetime.now(UTC),
                sandbox_result=SandboxExecutionResult(
                    status=SandboxExecutionStatus.failed,
                    available=True,
                    build_succeeded=False,
                    run_succeeded=False,
                    generated_at=datetime.now(UTC),
                    build_stderr="cargo build failed",
                    error="cargo build failed",
                    deliverable_reports=[
                        DeliverableSandboxReport(
                            deliverable_id="deliverable_1",
                            compile_succeeded=False,
                            runtime_succeeded=False,
                            error="edition2024 is required",
                        )
                    ],
                ),
                findings=[],
            )
            run.artifacts.node_executions = [failure_node]

            failure_context = build_failure_context(run, failure_node)

            self.assertEqual(len(failure_context.dependency_contracts), 1)
            contract = failure_context.dependency_contracts[0]
            self.assertEqual(contract.deliverable_id, "deliverable_1")
            self.assertEqual(contract.package_manager, "cargo")
            self.assertEqual(contract.expected_manifest_paths, ["Cargo.toml"])
            self.assertEqual(contract.present_manifest_paths, ["Cargo.toml"])
            self.assertEqual(contract.expected_lockfile_paths, ["Cargo.lock"])
            self.assertEqual(contract.present_lockfile_paths, [])
            self.assertTrue(contract.runtime_bundle_complete)

    def test_langgraph_execute_appends_history_across_iterations(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            graph = LangGraphAssignmentGraph(_AlwaysFailSandboxRunner())

            def _single_pass(node_name, state):  # noqa: ANN001
                return graph._append_node(
                    state,
                    kind=WorkflowNodeKind.authoring_runtime,
                    attempt=1,
                    status=WorkflowNodeStatus.passed,
                    summary=f"Synthetic {node_name} pass.",
                    findings=[],
                    sandbox_result=None,
                    authoring_attempt=1,
                    cached_sandbox_result=None,
                )

            graph._invoke_node = _single_pass  # type: ignore[method-assign]
            graph._next_node = lambda node_name, state: None  # type: ignore[method-assign]

            first = graph.execute(run)
            second = graph.execute(first)

            self.assertEqual(len(first.artifacts.node_executions), 1)
            self.assertEqual(len(second.artifacts.node_executions), 2)
            self.assertEqual(
                [node.iteration for node in second.artifacts.node_executions],
                [1, 2],
            )
