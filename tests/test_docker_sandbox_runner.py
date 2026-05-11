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
    if not inferred.design_spec.runtime_dependencies.editable_files:
        inferred.design_spec.runtime_dependencies.editable_files = ["app.py"]
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake,
        design_spec=inferred.design_spec,
        execute_nodes=False,
    )
    run.artifacts.workspace_snapshot = workspace_manager.prepare_run_workspace(run, overwrite=True)
    return run


class DockerSandboxRunnerTests(unittest.TestCase):
    def test_image_build_failure_summary_includes_buildkit_diagnostic(self) -> None:
        """For docker buildkit failures, the stderr tail naturally contains
        the real diagnostic (ERROR: ... checksum/not found / requires X / ...).
        The headline is a stage-anchored teaser; the FULL stderr_excerpt is
        the canonical diagnostic shipped to the LLM.
        """
        runner = DockerSandboxRunner()

        build_log = "\n".join(
            [
                "#10 [6/8] RUN sh .coursegen/runtime/install.sh",
                "#10 1.060 -Dmaven.multiModuleProjectDirectory system property is not set.",
                "#10 ERROR: process \"/bin/sh -c sh .coursegen/runtime/install.sh\" did not complete successfully: exit code: 1",
                "ERROR: failed to build: failed to solve: process \"/bin/sh -c sh .coursegen/runtime/install.sh\" did not complete successfully: exit code: 1",
            ]
        )

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.image_build,
            error_text="Docker build failed (exit 1).",
            logs=build_log,
            default="image build failed",
        )

        # The buildkit error line IS in the tail and IS the canonical signal.
        self.assertIn("deliverable_1 failed during image build", summary)
        self.assertIn("failed to solve", summary)

    def test_image_build_failure_summary_uses_log_tail(self) -> None:
        runner = DockerSandboxRunner()

        build_log = "\n".join(
            [
                "#0 building with \"desktop-linux\" instance using docker driver",
                "",
                "#7 [4/6] COPY mvnw ./mvnw",
                "#7 ERROR: failed to calculate checksum of ref: \"/mvnw\": not found",
                "ERROR: failed to solve: failed to compute cache key: failed to calculate checksum of ref: \"/mvnw\": not found",
            ]
        )

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.image_build,
            error_text="Docker build for deliverable_1 failed (exit 1).",
            logs=build_log,
            default="image build failed",
        )

        # The tail naturally surfaces the diagnostic.
        self.assertIn("failed to calculate checksum", summary)
        self.assertIn("/mvnw", summary)
        self.assertIn("deliverable_1 failed during image build", summary)

    def test_image_build_failure_classifies_with_command_and_exit_code(self) -> None:
        from app.services.learner_studio_service import RuntimeImageBuildError

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None

            build_command = [
                "docker",
                "build",
                "-t",
                "course-gen-runtime:abc123",
                ".",
            ]
            build_stderr = "\n".join(
                [
                    "#0 building with \"desktop-linux\" instance using docker driver",
                    "#7 [4/6] COPY mvnw ./mvnw",
                    "ERROR: failed to compute cache key: \"/mvnw\": not found",
                    "ERROR: failed to solve: failed to compute cache key",
                ]
            )

            runner = DockerSandboxRunner()
            runner.runtime_harness = Mock()
            runner.runtime_harness._allocate_port.side_effect = [
                18001,
                18002,
                18003,
                18004,
            ]
            runner.runtime_harness._runtime_manifest.return_value = {}
            runner.runtime_harness._ephemeral_runtime_workspace.side_effect = lambda starter_root: nullcontext(starter_root)
            runner.runtime_harness._workspace_runtime_image_name.side_effect = RuntimeImageBuildError(
                "Docker build failed",
                command=build_command,
                returncode=1,
                stdout="",
                stderr=build_stderr,
            )
            runner.runtime_harness._remove_runtime_support.return_value = None
            runner.runtime_harness._container_logs.return_value = ""
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

            result = runner._execute_starter_harness(
                workspace_root=Path(workspace.public_dir),
                spec=spec,
                workflow_run_id=run.id,
                now=datetime.now(UTC),
                started=time.perf_counter(),
            )

            self.assertGreaterEqual(len(result.deliverable_reports), 1)
            report = result.deliverable_reports[0]
            self.assertEqual(report.failed_stage, SandboxFailureStage.image_build)
            self.assertEqual(report.stage_command, build_command)
            self.assertEqual(report.stage_exit_code, 1)
            self.assertFalse(report.compile_succeeded)
            self.assertFalse(report.runtime_succeeded)
            # The summary should surface the real diagnostic (failed to compute
            # cache key for /mvnw), not Docker's generic "failed to solve" footer.
            self.assertIn("failed to compute cache key", report.error or "")
            # The full stderr is preserved (tail-truncated) so the model still
            # sees the buildkit footer alongside the real cause if it wants.
            self.assertIn("ERROR: failed to solve:", report.stderr)

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

        # The summary must surface the actual diagnostic, even if a different
        # line is chosen than before. Both signal-bearing lines are acceptable
        # — what matters is that one of them is in the headline.
        self.assertIn("deliverable_1 failed during boot", summary)
        self.assertTrue(
            "PSQLException" in summary or "HikariPool" in summary,
            f"Boot summary should contain a real diagnostic, got: {summary!r}",
        )

    def test_install_failure_summary_includes_stderr_tail_for_go_toolchain(self) -> None:
        """The Go toolchain prints its version-mismatch error to stderr at the
        very end of `go mod tidy`. The harness summary must surface that line
        verbatim — not collapse it to "stopped during install".
        """
        runner = DockerSandboxRunner()

        go_stderr = "\n".join(
            [
                "go: downloading github.com/rogpeppe/go-internal v1.14.1",
                "go: downloading golang.org/x/tools v0.30.0",
                "go: github.com/rogpeppe/go-internal@v1.14.1 requires go >= 1.23 (running go 1.22.4)",
            ]
        )

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_1",
            failed_stage=SandboxFailureStage.install,
            error_text="Container stopped during 'install' before health became healthy",
            logs=go_stderr,
            default="install failed",
        )

        self.assertIn("requires go >= 1.23", summary)
        self.assertIn("deliverable_1 failed during install", summary)
        # The generic stopped-during headline must NOT replace the diagnostic.
        self.assertNotIn("stopped during 'install' before health", summary)

    def test_summary_falls_back_to_stage_label_when_no_text_available(self) -> None:
        runner = DockerSandboxRunner()

        summary = runner._summarize_stage_failure(
            deliverable_id="deliverable_3",
            failed_stage=SandboxFailureStage.runtime,
            error_text=None,
            logs=None,
            default="something happened",
        )

        self.assertEqual(summary, "deliverable_3 failed during runtime.")

    def test_shared_codebase_boots_runtime_once_for_all_deliverables(self) -> None:
        """For shared_codebase courses, the runtime image must be built once and
        the container booted once. The per-deliverable visible script lives at
        public/checks/<id>/run_visible_checks.py and must be invoked once per
        deliverable against the single running app.
        """
        from app.services.generated_test_harness import GeneratedTestSuiteReport

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            assert spec.course_structure.shared_codebase is True
            # _make_run produces 4 deliverables for the shared-codebase course.
            self.assertGreaterEqual(len(spec.deliverables), 3)

            runner = DockerSandboxRunner()
            runner.runtime_harness = Mock()
            runner.runtime_harness._allocate_port.return_value = 18001
            runner.runtime_harness._runtime_manifest.return_value = {}
            runner.runtime_harness._ephemeral_runtime_workspace.side_effect = (
                lambda starter_root: nullcontext(starter_root)
            )
            runner.runtime_harness._workspace_runtime_image_name.return_value = (
                "course-gen-runtime:test"
            )
            runner.runtime_harness._ensure_runtime_image_available.return_value = None
            runner.runtime_harness._image_exists.return_value = True
            runner.runtime_harness._dependency_services.return_value = []
            runner.runtime_harness._start_runtime_support_services.return_value = None
            runner.runtime_harness._docker_env_args.return_value = []
            runner.runtime_harness._app_runtime_environment.return_value = {}
            runner.runtime_harness._runtime_shell_command.return_value = [
                "sh",
                "-c",
                "echo run",
            ]
            runner.runtime_harness._runtime_launch_script.return_value = "echo run"
            runner.runtime_harness._healthcheck_path.return_value = "/health"
            runner.runtime_harness._wait_for_http.return_value = None
            runner.runtime_harness._container_logs.return_value = ""
            runner.runtime_harness._runtime_stage_from_logs.return_value = ""
            runner.runtime_harness._remove_runtime_support.return_value = None
            runner.runtime_harness._RUNTIME_STAGE_MARKER_PREFIX = (
                "[coursegen-runtime-stage] "
            )
            runner.dependency_contract_materializer.materialize = Mock(
                return_value=SimpleNamespace(
                    attempted=True,
                    succeeded=True,
                    stdout="",
                    stderr="",
                    image_name="course-gen-runtime:test",
                    synced_paths=[],
                    command=[],
                    return_code=0,
                    error=None,
                )
            )

            # Make test_script_runner.run_suite return a passing report each time.
            def _ok_report(*, workspace_root, command, base_url, suite_type):
                return GeneratedTestSuiteReport(
                    suite_type=suite_type,
                    command=command,
                    exit_code=0,
                    valid=True,
                    passed=True,
                    tests=[],
                    summary="ok",
                )

            runner.test_script_runner = Mock()
            runner.test_script_runner.run_suite.side_effect = _ok_report

            with (
                patch(
                    "app.services.docker_sandbox_runner.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout="container-id", stderr=""
                    ),
                ),
                patch.object(runner, "_probe_contract_smoke", return_value=(True, "", None)),
            ):
                result = runner._execute_starter_harness(
                    workspace_root=Path(workspace.public_dir),
                    spec=spec,
                    workflow_run_id=run.id,
                    now=datetime.now(UTC),
                    started=time.perf_counter(),
                )

            # Single image build, single container boot.
            self.assertEqual(
                runner.runtime_harness._ensure_runtime_image_available.call_count,
                1,
                "Shared-codebase course must build the runtime image exactly once.",
            )
            # The single `docker run -d ...` command for the container is the
            # only subprocess.run call to the docker binary for boot.
            # Easier assertion: dependency-contract materialization is called once.
            self.assertEqual(
                runner.dependency_contract_materializer.materialize.call_count,
                1,
                "Shared-codebase course must materialize the dependency contract exactly once.",
            )
            # Per-deliverable visible-suite calls = len(deliverables), each with
            # a distinct command pointing at public/checks/<id>/run_visible_checks.py.
            self.assertEqual(
                runner.test_script_runner.run_suite.call_count,
                len(spec.deliverables),
                "Visible suite must run once per deliverable against the single running app.",
            )
            invoked_commands = [
                call.kwargs.get("command") or (call.args[1] if len(call.args) > 1 else None)
                for call in runner.test_script_runner.run_suite.call_args_list
            ]
            for deliverable in spec.deliverables:
                expected_fragment = f"../checks/{deliverable.id}/run_visible_checks.py"
                self.assertTrue(
                    any(expected_fragment in (cmd or "") for cmd in invoked_commands),
                    f"Visible suite command for {deliverable.id} should reference {expected_fragment}; got {invoked_commands!r}",
                )
            # All four reports produced; all passed.
            self.assertEqual(len(result.deliverable_reports), len(spec.deliverables))
            for report in result.deliverable_reports:
                self.assertTrue(report.runtime_succeeded, report.error)
                self.assertTrue(report.public_checks_passed, report.error)

    def test_shared_codebase_missing_visible_script_fails_only_that_deliverable(self) -> None:
        """If a deliverable's visible script does not exist in public/checks/<id>/,
        the runner produces a DeliverableSandboxReport with checks_passed=False
        and a clear error. The other deliverables still run (no fail-fast on
        missing visible script, unless the contract probe also fails).
        """
        from app.services.generated_test_harness import GeneratedTestSuiteReport

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            assert spec.course_structure.shared_codebase is True

            # Delete the visible script for the second deliverable so the runner
            # observes a missing per-deliverable visible suite.
            missing_id = spec.deliverables[1].id
            missing_path = (
                Path(workspace.public_dir)
                / "checks"
                / missing_id
                / "run_visible_checks.py"
            )
            self.assertTrue(missing_path.exists())
            missing_path.unlink()

            runner = DockerSandboxRunner()
            runner.runtime_harness = Mock()
            runner.runtime_harness._allocate_port.return_value = 18001
            runner.runtime_harness._runtime_manifest.return_value = {}
            runner.runtime_harness._ephemeral_runtime_workspace.side_effect = (
                lambda starter_root: nullcontext(starter_root)
            )
            runner.runtime_harness._workspace_runtime_image_name.return_value = (
                "course-gen-runtime:test"
            )
            runner.runtime_harness._ensure_runtime_image_available.return_value = None
            runner.runtime_harness._image_exists.return_value = True
            runner.runtime_harness._dependency_services.return_value = []
            runner.runtime_harness._start_runtime_support_services.return_value = None
            runner.runtime_harness._docker_env_args.return_value = []
            runner.runtime_harness._app_runtime_environment.return_value = {}
            runner.runtime_harness._runtime_shell_command.return_value = [
                "sh",
                "-c",
                "echo run",
            ]
            runner.runtime_harness._runtime_launch_script.return_value = "echo run"
            runner.runtime_harness._healthcheck_path.return_value = "/health"
            runner.runtime_harness._wait_for_http.return_value = None
            runner.runtime_harness._container_logs.return_value = ""
            runner.runtime_harness._runtime_stage_from_logs.return_value = ""
            runner.runtime_harness._remove_runtime_support.return_value = None
            runner.runtime_harness._RUNTIME_STAGE_MARKER_PREFIX = (
                "[coursegen-runtime-stage] "
            )
            runner.dependency_contract_materializer.materialize = Mock(
                return_value=SimpleNamespace(
                    attempted=True,
                    succeeded=True,
                    stdout="",
                    stderr="",
                    image_name="course-gen-runtime:test",
                    synced_paths=[],
                    command=[],
                    return_code=0,
                    error=None,
                )
            )

            def _ok_report(*, workspace_root, command, base_url, suite_type):
                return GeneratedTestSuiteReport(
                    suite_type=suite_type,
                    command=command,
                    exit_code=0,
                    valid=True,
                    passed=True,
                    tests=[],
                    summary="ok",
                )

            runner.test_script_runner = Mock()
            runner.test_script_runner.run_suite.side_effect = _ok_report

            with (
                patch(
                    "app.services.docker_sandbox_runner.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout="container-id", stderr=""
                    ),
                ),
                patch.object(runner, "_probe_contract_smoke", return_value=(True, "", None)),
            ):
                result = runner._execute_starter_harness(
                    workspace_root=Path(workspace.public_dir),
                    spec=spec,
                    workflow_run_id=run.id,
                    now=datetime.now(UTC),
                    started=time.perf_counter(),
                )

            # The missing-script deliverable must yield a failed report with a
            # clear error. The other deliverables should still produce reports.
            ids_to_reports = {r.deliverable_id: r for r in result.deliverable_reports}
            self.assertIn(missing_id, ids_to_reports)
            missing_report = ids_to_reports[missing_id]
            self.assertFalse(missing_report.public_checks_passed)
            self.assertTrue(
                missing_report.error
                and "run_visible_checks.py" in missing_report.error,
                f"Error should mention the missing visible script; got {missing_report.error!r}",
            )
            # run_suite should NOT have been called for the missing deliverable
            # (one fewer call than total deliverables).
            self.assertEqual(
                runner.test_script_runner.run_suite.call_count,
                len(spec.deliverables) - 1,
            )

    def test_non_shared_codebase_runs_legacy_per_deliverable_boot_loop(self) -> None:
        """Non-shared courses must still go through the per-deliverable boot
        loop. The dispatcher must not route them to the single-boot variant.
        """
        from app.services.learner_studio_service import RuntimeImageBuildError

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            run = _make_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None

            # Flip to non-shared and create per-deliverable starter dirs so the
            # legacy path finds something to operate on.
            spec.course_structure.shared_codebase = False
            shared_starter = Path(workspace.public_dir) / "starter"
            for deliverable in spec.deliverables:
                per_dir = shared_starter / deliverable.id
                per_dir.mkdir(parents=True, exist_ok=True)

            runner = DockerSandboxRunner()
            runner.runtime_harness = Mock()
            runner.runtime_harness._allocate_port.return_value = 18001
            runner.runtime_harness._runtime_manifest.return_value = {}
            # Make image build raise so we don't have to mock the entire boot.
            # We only need to confirm the LEGACY path is taken — i.e., the loop
            # body runs per-deliverable, not the single-boot variant.
            runner.runtime_harness._ephemeral_runtime_workspace.side_effect = (
                lambda starter_root: nullcontext(starter_root)
            )
            runner.runtime_harness._workspace_runtime_image_name.side_effect = (
                RuntimeImageBuildError(
                    "build failed",
                    command=["docker", "build"],
                    returncode=1,
                    stdout="",
                    stderr="ERROR: build failed",
                )
            )
            runner.runtime_harness._remove_runtime_support.return_value = None
            runner.runtime_harness._container_logs.return_value = ""
            runner.runtime_harness._RUNTIME_STAGE_MARKER_PREFIX = (
                "[coursegen-runtime-stage] "
            )
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

            result = runner._execute_starter_harness(
                workspace_root=Path(workspace.public_dir),
                spec=spec,
                workflow_run_id=run.id,
                now=datetime.now(UTC),
                started=time.perf_counter(),
            )

            # Legacy path: per-deliverable boot loop calls materialize once per
            # deliverable (in contrast to the shared-codebase single-boot
            # variant which only calls it once total).
            self.assertEqual(
                runner.dependency_contract_materializer.materialize.call_count,
                len(spec.deliverables),
                "Non-shared course must use the legacy per-deliverable loop "
                "(one materialize call per deliverable).",
            )
            # All deliverable reports show image build failure (the same mock).
            self.assertEqual(len(result.deliverable_reports), len(spec.deliverables))

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
