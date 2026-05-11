from __future__ import annotations

import time
import unittest
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.domain.sandbox import SandboxFailureStage
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.openai_task_agent_authoring import OpenAITaskAgentAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _make_run(temp_dir: Path):
    store = SQLiteWorkflowStore(db_path=temp_dir / "course_gen.db")
    workspace_manager = AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
    workflow_service = WorkflowService(
        store,
        task_agent_authoring_service=OpenAITaskAgentAuthoringService(enabled=False),
        workspace_manager=workspace_manager,
    )
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
    run.artifacts.workspace_snapshot = workspace_manager.prepare_run_workspace(run, overwrite=True)
    return run


class DockerSandboxRunnerTests(unittest.TestCase):
    def test_boot_failure_summary_prefers_useful_container_log_line(self) -> None:
        runner = DockerSandboxRunner()

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.boot,
            error_text=(
                "Timed out waiting for 'http://127.0.0.1:18001/health' during 'boot'. "
                "Last error: [Errno 61] Connection refused"
            ),
            logs=(
                "2026-05-11T10:00:00Z ERROR com.zaxxer.hikari.pool.HikariPool: "
                "HikariPool-1 - Exception during pool initialization.\n"
                "org.postgresql.util.PSQLException: Connection to postgres:5432 refused"
            ),
            default="boot failed",
        )

        self.assertEqual(
            summary,
            (
                "deliverable_1 failed during boot: "
                "2026-05-11T10:00:00Z ERROR com.zaxxer.hikari.pool.HikariPool: "
                "HikariPool-1 - Exception during pool initialization."
            ),
        )

    def test_shared_codebase_runtime_stops_after_first_failed_deliverable(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None

            runner = DockerSandboxRunner()
            runner.runtime_harness = Mock()
            runner.runtime_harness._allocate_port.side_effect = [18001, 18002, 18003, 18004]
            runner.runtime_harness._runtime_manifest.return_value = {}
            runner.runtime_harness._ephemeral_runtime_workspace.side_effect = lambda starter_root: nullcontext(starter_root)
            runner.runtime_harness._workspace_runtime_image_name.return_value = "course-gen-runtime:test"
            runner.runtime_harness._ensure_runtime_image_available.return_value = None
            runner.runtime_harness._image_exists.return_value = True
            runner.runtime_harness._dependency_services.return_value = []
            runner.runtime_harness._start_runtime_support_services.return_value = None
            runner.runtime_harness._docker_env_args.return_value = []
            runner.runtime_harness._app_runtime_environment.return_value = {}
            runner.runtime_harness._runtime_shell_command.return_value = ["sh", "-c", "echo run"]
            runner.runtime_harness._runtime_launch_script.return_value = "echo run"
            runner.runtime_harness._healthcheck_path.return_value = "/health"
            runner.runtime_harness._wait_for_http.side_effect = RuntimeError("connection refused")
            runner.runtime_harness._container_logs.return_value = "HikariPool boot failed"
            runner.runtime_harness._runtime_stage_from_logs.return_value = "boot"
            runner.runtime_harness._runtime_stage_command.return_value = ["sh", ".coursegen/runtime/run.sh"]
            runner.runtime_harness._remove_runtime_support.return_value = None
            runner.runtime_harness._RUNTIME_STAGE_MARKER_PREFIX = "[coursegen-runtime-stage] "
            runner.dependency_contract_materializer.materialize = Mock(
                return_value=SimpleNamespace(
                    attempted=False,
                    succeeded=True,
                    stdout="",
                    stderr="",
                    image_name=None,
                    synced_paths=[],
                    command=[],
                    return_code=0,
                    error=None,
                )
            )

            with (
                patch("app.services.docker_sandbox_runner.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="container-id", stderr="")),
                patch.object(runner, "_probe_contract_smoke") as mock_contract_smoke,
                patch.object(runner, "_run_visible_suite") as mock_visible_suite,
            ):
                result = runner._execute_starter_harness(
                    workspace_root=Path(workspace.public_dir),
                    spec=spec,
                    workflow_run_id=run.id,
                    now=datetime.now(UTC),
                    started=time.perf_counter(),
                )

            self.assertEqual(len(result.deliverable_reports), 1)
            self.assertEqual(result.deliverable_reports[0].deliverable_id, "deliverable_1")
            self.assertEqual(result.deliverable_reports[0].failed_stage, SandboxFailureStage.boot)
            mock_contract_smoke.assert_not_called()
            mock_visible_suite.assert_not_called()
            self.assertEqual(runner.runtime_harness._wait_for_http.call_count, 1)


if __name__ == "__main__":
    unittest.main()
