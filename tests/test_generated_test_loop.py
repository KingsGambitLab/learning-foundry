from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.domain.registry import PackageType
from app.domain.sandbox import SandboxExecutionResult, SandboxExecutionStatus, SandboxFailureStage
from app.domain.workflow import MaterializeBundleRequest, WorkflowNodeStatus
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.dependency_contract_materializer import DependencyContractMaterializationResult
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.generated_test_harness import BaselineValidationResult, GeneratedTestBaselineVerifier
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
from app.services.learner_studio_service import LearnerStudioError
from app.services.openai_test_script_authoring import (
    TestScriptAuthoringResult as _TestScriptAuthoringResult,
    TestScriptAuthoringSource as _TestScriptAuthoringSource,
)
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class _BaselineAlwaysValid:
    def verify_course(self, **kwargs):  # noqa: ANN003
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


class _FailingDependencyContractMaterializer:
    def materialize(self, *, deliverable_id, **kwargs):  # noqa: ANN003
        return DependencyContractMaterializationResult(
            deliverable_id=deliverable_id,
            attempted=True,
            succeeded=False,
            image_name="rust:1.82-bookworm",
            stderr="lockfile generation failed",
            error="Dependency contract materialization failed before runtime boot.",
        )


class _NoOpDependencyContractMaterializer:
    def materialize(self, *, deliverable_id, **kwargs):  # noqa: ANN003
        return DependencyContractMaterializationResult(
            deliverable_id=deliverable_id,
            attempted=False,
            succeeded=True,
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
            public_root = Path(workspace.public_dir)
            private_root = Path(workspace.private_dir)
            deliverable = spec.deliverables[0]
            visible_path = (
                public_root / "checks" / deliverable.id / "run_visible_checks.py"
            )
            hidden_path = (
                private_root / "grader" / deliverable.id / "run_hidden_checks.py"
            )
            hidden_path.write_text(visible_path.read_text(encoding="utf-8"), encoding="utf-8")

            verifier = GeneratedTestBaselineVerifier()
            result = verifier.verify_course(
                workspace_root=public_root,
                private_root=private_root,
                spec=spec,
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
        self.assertTrue(any(finding.code == "starter_repo_bundle_not_authored" for finding in latest.findings))

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

    def test_sandbox_runner_converts_runtime_image_build_failure_into_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None

            runner = DockerSandboxRunner(
                dependency_contract_materializer=_NoOpDependencyContractMaterializer()
            )
            with patch.object(
                runner.runtime_harness,
                "_workspace_runtime_image_name",
                side_effect=LearnerStudioError("runtime image build failed"),
            ):
                result = runner._execute_starter_harness(
                    workspace_root=Path(workspace.public_dir),
                    spec=spec,
                    workflow_run_id=run.id,
                    now=datetime.now(UTC),
                    started=0.0,
                )

        self.assertEqual(result.status, SandboxExecutionStatus.failed)
        self.assertFalse(result.build_succeeded)
        self.assertTrue(result.deliverable_reports)
        self.assertIn("runtime image build failed", result.deliverable_reports[0].error or "")

    def test_sandbox_runner_converts_dependency_contract_materialization_failure_into_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None

            runner = DockerSandboxRunner(
                dependency_contract_materializer=_FailingDependencyContractMaterializer()
            )
            result = runner._execute_starter_harness(
                workspace_root=Path(workspace.public_dir),
                spec=spec,
                workflow_run_id=run.id,
                now=datetime.now(UTC),
                started=0.0,
            )

        self.assertEqual(result.status, SandboxExecutionStatus.failed)
        self.assertFalse(result.build_succeeded)
        self.assertTrue(result.deliverable_reports)
        self.assertIn(
            "Dependency contract materialization failed before runtime boot.",
            result.deliverable_reports[0].error or "",
        )
        self.assertEqual(
            result.deliverable_reports[0].failed_stage,
            SandboxFailureStage.dependency_materialization,
        )


class MaxAuthoringAttemptsPolicyTests(unittest.TestCase):
    """Pass 9 Job B: shared-codebase courses need more authoring retries.

    Each retry advances the runtime through a distinct failure mode
    (install → verify → contract → …). Three attempts run out before
    convergence for shared-codebase courses; non-shared courses are
    smaller and three is still appropriate.
    """

    def _make_spec(self, *, shared_codebase: bool):
        from app.services.assignment_design_inference import (
            GenerationIntake,
            infer_assignment_design,
        )

        intake = GenerationIntake(
            title="Inventory reservations",
            problem_statement=(
                "Build a multi-warehouse inventory reservation service with "
                "FastAPI, Postgres, and Redis."
            ),
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
        spec = inferred.design_spec
        spec.course_structure.shared_codebase = shared_codebase
        return spec

    def test_max_authoring_attempts_returns_5_for_shared_codebase(self) -> None:
        from app.services.langgraph_assignment_graph import max_authoring_attempts

        spec = self._make_spec(shared_codebase=True)

        self.assertEqual(max_authoring_attempts(spec), 5)

    def test_max_authoring_attempts_returns_3_for_non_shared_codebase(self) -> None:
        from app.services.langgraph_assignment_graph import max_authoring_attempts

        spec = self._make_spec(shared_codebase=False)

        self.assertEqual(max_authoring_attempts(spec), 3)

    def test_langgraph_uses_5_attempts_for_shared_codebase_run(self) -> None:
        """The graph's runtime cap during ``execute()`` must reflect the
        per-spec policy. For a shared-codebase run the graph must allow
        up to five authoring attempts even if the constructor was given
        the default 3.
        """
        from app.domain.workflow import WorkflowRun

        graph = LangGraphAssignmentGraph(
            DockerSandboxRunner(),
            max_authoring_attempts=3,
        )
        spec = self._make_spec(shared_codebase=True)
        run = WorkflowRun.__fake_for_test__ if False else None  # placeholder; not used

        # The helper exposed on the graph must reflect the spec, not the
        # constructor default, when the spec says shared_codebase.
        self.assertEqual(graph._max_authoring_attempts_for_spec(spec), 5)

    def test_langgraph_uses_3_attempts_for_non_shared_codebase_run(self) -> None:
        graph = LangGraphAssignmentGraph(
            DockerSandboxRunner(),
            max_authoring_attempts=3,
        )
        spec = self._make_spec(shared_codebase=False)

        self.assertEqual(graph._max_authoring_attempts_for_spec(spec), 3)


if __name__ == "__main__":
    unittest.main()
