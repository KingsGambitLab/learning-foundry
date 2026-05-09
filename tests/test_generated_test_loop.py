from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from app.domain.registry import PackageType
from app.domain.sandbox import SandboxExecutionResult, SandboxExecutionStatus
from app.domain.workflow import MaterializeBundleRequest, WorkflowNodeStatus
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.generated_test_harness import BaselineValidationResult, GeneratedTestBaselineVerifier
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
from app.services.openai_test_script_authoring import (
    TestScriptAuthoringResult as _TestScriptAuthoringResult,
    TestScriptAuthoringSource as _TestScriptAuthoringSource,
)
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class _BaselineAlwaysValid:
    def verify_deliverable(self, **kwargs):  # noqa: ANN003
        return BaselineValidationResult(valid=True)


class _NoOpTestAuthoringService:
    def author_workspace_tests(self, run, **kwargs):  # noqa: ANN003
        return run, _TestScriptAuthoringResult(
            source=_TestScriptAuthoringSource.unavailable,
            updated_files=[],
            usage=None,
            notes=[],
            message="Left the current generated test scripts in place.",
            available=False,
        )


def _materialized_run(temp_dir: str):
    store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
    )
    intake = GenerationIntake(
        title="Inventory Reservation Service",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    inferred = infer_assignment_design(
        title=intake.title,
        problem_statement=intake.problem_statement,
        package_type_hint=intake.package_type_hint,
    )
    assert inferred.design_spec is not None
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake,
        design_spec=inferred.design_spec,
        execute_nodes=False,
    )
    workflow_service.materialize_run(run.id, MaterializeBundleRequest(overwrite=True))
    run = workflow_service.get_run(run.id)
    assert run is not None
    run, _ = TaskAgentWorkspaceAuthoringService().author_workspace(run)
    return run


class GeneratedTestLoopTests(unittest.TestCase):
    def test_baseline_verifier_rejects_identical_hidden_and_visible_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            assert spec is not None
            workspace = run.artifacts.workspace_snapshot
            assert workspace is not None
            deliverable = spec.deliverables[0]
            deliverable_root = Path(workspace.public_dir) / "starter" / deliverable.id
            visible_path = deliverable_root / "checks" / "run_visible_checks.py"
            hidden_path = deliverable_root / ".coursegen" / "grader" / "run_hidden_checks.py"
            hidden_path.write_text(visible_path.read_text(encoding="utf-8"), encoding="utf-8")

            verifier = GeneratedTestBaselineVerifier()
            result = verifier.verify_deliverable(
                workspace_root=deliverable_root,
                spec=spec,
                starter_type=deliverable.starter_type,
            )

        self.assertFalse(result.valid)
        self.assertTrue(any(issue.code == "hidden_tests_match_visible_tests" for issue in result.errors))

    def test_reviewer_tests_blocks_default_generated_test_placeholder_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            graph = LangGraphAssignmentGraph(
                DockerSandboxRunner(),
                baseline_verifier=_BaselineAlwaysValid(),
            )
            state = {
                "run": run,
                "node_executions": [],
                "active_iteration": 1,
                "authoring_attempt": 1,
                "reviewer_attempt": 1,
                "cached_sandbox_result": SandboxExecutionResult(
                    status=SandboxExecutionStatus.passed,
                    available=True,
                    build_succeeded=True,
                    run_succeeded=True,
                    generated_at=datetime.now(UTC),
                ),
                "next_retry_node": None,
            }

            updated = graph._reviewer_tests_node(state)
            latest = updated["node_executions"][-1]

        self.assertEqual(latest.status, WorkflowNodeStatus.failed)
        self.assertTrue(any(finding.code == "generated_test_scripts_not_authored" for finding in latest.findings))

    def test_authoring_tests_stays_on_the_current_authoring_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            graph = LangGraphAssignmentGraph(
                DockerSandboxRunner(),
                test_authoring_service=_NoOpTestAuthoringService(),
            )
            state = {
                "run": run,
                "node_executions": [],
                "active_iteration": 1,
                "authoring_attempt": 1,
                "reviewer_attempt": 0,
                "cached_sandbox_result": None,
                "next_retry_node": None,
            }

            updated = graph._authoring_tests_node(state)
            latest = updated["node_executions"][-1]

        self.assertEqual(latest.attempt, 1)
        self.assertEqual(updated["authoring_attempt"], 1)


if __name__ == "__main__":
    unittest.main()
