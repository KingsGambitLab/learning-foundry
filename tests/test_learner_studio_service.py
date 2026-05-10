from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.domain.learner import LearnerWorkspaceScope, LearnerWorkspaceSession, LearnerWorkspaceSessionStatus
from app.services.task_agent_starter_templates import HIDDEN_MANIFEST_PATH
from app.services.learner_studio_service import LearnerStudioError, LearnerStudioService


class LearnerStudioServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name) / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        self.existing_session = LearnerWorkspaceSession(
            id="studio_test_session",
            enrollment_id="enrollment_test",
            deliverable_id="exercise/01-contract",
            scope=LearnerWorkspaceScope.shared_course,
            created_at=now,
            updated_at=now,
            status=LearnerWorkspaceSessionStatus.running,
            workspace_root=str(self.workspace_root),
            container_name="course-gen-studio-existing",
            host_port=18080,
            editor_url="http://127.0.0.1:18080/",
            image_name="course-gen-learner-studio:latest",
            notes=["Existing learner studio session."],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_launch_editor_reuses_running_session_when_workspace_matches(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")

        with (
            patch.object(service, "_ensure_image"),
            patch.object(service, "_container_running", return_value=True),
            patch("app.services.learner_studio_service.subprocess.run") as mock_run,
        ):
            refreshed = service.launch_editor(
                enrollment_id="enrollment_test",
                deliverable_id="exercise/01-contract",
                workspace_root=self.workspace_root,
                scope=LearnerWorkspaceScope.shared_course,
                existing_session=self.existing_session,
            )

        self.assertEqual(mock_run.call_count, 0)
        self.assertEqual(refreshed.id, self.existing_session.id)
        self.assertEqual(refreshed.container_name, self.existing_session.container_name)
        self.assertEqual(refreshed.host_port, self.existing_session.host_port)
        self.assertEqual(refreshed.deliverable_id, "exercise/01-contract")
        self.assertEqual(refreshed.status, LearnerWorkspaceSessionStatus.running)

    def test_launch_editor_restarts_running_session_when_container_is_not_running(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")

        with (
            patch.object(service, "_ensure_image"),
            patch.object(service, "_container_running", return_value=False),
            patch.object(service, "_allocate_port", return_value=19191),
            patch.object(service, "_wait_for_http"),
            patch.object(service, "_remove_container") as mock_remove_container,
            patch(
                "app.services.learner_studio_service.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ) as mock_run,
        ):
            refreshed = service.launch_editor(
                enrollment_id="enrollment_test",
                deliverable_id="exercise/01-contract",
                workspace_root=self.workspace_root,
                scope=LearnerWorkspaceScope.shared_course,
                existing_session=self.existing_session,
            )

        mock_remove_container.assert_called_once_with(self.existing_session.container_name)
        self.assertGreaterEqual(mock_run.call_count, 1)
        docker_command = mock_run.call_args_list[-1].args[0]
        self.assertIn("run", docker_command)
        self.assertIn("-v", docker_command)
        self.assertIn(f"{self.workspace_root.resolve()}:/workspace", docker_command)
        self.assertNotIn("--rm", docker_command)
        self.assertNotIn("--network", docker_command)
        self.assertNotIn("--network-alias", docker_command)
        self.assertEqual(refreshed.container_name, self.existing_session.container_name)
        self.assertEqual(refreshed.host_port, 19191)
        self.assertEqual(refreshed.editor_url, "http://127.0.0.1:19191/")
        self.assertEqual(refreshed.deliverable_id, "exercise/01-contract")
        self.assertEqual(refreshed.status, LearnerWorkspaceSessionStatus.running)

    def test_grade_assignment_skips_docker_network_when_workspace_has_no_dependency_services(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")
        spec = SimpleNamespace(
            runtime_dependencies=SimpleNamespace(preview_command="python .coursegen/preview_app.py --host 0.0.0.0"),
            project_contract=SimpleNamespace(runtime_plan=SimpleNamespace(services=[])),
        )

        with (
            patch.object(service, "_allocate_port", return_value=18001),
            patch.object(service, "_ensure_runtime_image_available"),
            patch.object(service, "_workspace_runtime_image_name", return_value="course-gen-runtime:test"),
            patch.object(service, "_wait_for_http"),
            patch.object(service, "_remove_runtime_support"),
            patch.object(service.runner, "grade_assignment_live", return_value=SimpleNamespace(status="failed")),
            patch("app.services.learner_studio_service.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as mock_run,
        ):
            service.grade_assignment(
                workspace_root=self.workspace_root,
                spec=spec,
            )

        docker_command = mock_run.call_args.args[0]
        self.assertIn("run", docker_command)
        self.assertNotIn("--rm", docker_command)
        self.assertNotIn("--network", docker_command)
        self.assertNotIn("--network-alias", docker_command)

    def test_grade_assignment_uses_non_login_shell_for_runtime_protocol(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")
        spec = SimpleNamespace(
            runtime_dependencies=SimpleNamespace(preview_command="sh .coursegen/runtime/run.sh"),
            project_contract=SimpleNamespace(runtime_plan=SimpleNamespace(services=[])),
        )

        with (
            patch.object(service, "_allocate_port", return_value=18001),
            patch.object(service, "_ensure_runtime_image_available"),
            patch.object(service, "_workspace_runtime_image_name", return_value="course-gen-runtime:test"),
            patch.object(service, "_wait_for_http"),
            patch.object(service, "_remove_runtime_support"),
            patch.object(service.runner, "grade_assignment_live", return_value=SimpleNamespace(status="failed")),
            patch("app.services.learner_studio_service.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as mock_run,
        ):
            service.grade_assignment(
                workspace_root=self.workspace_root,
                spec=spec,
            )

        docker_command = mock_run.call_args.args[0]
        self.assertIn("sh", docker_command)
        self.assertIn("-c", docker_command)
        self.assertNotIn("-lc", docker_command)

    def test_workspace_runtime_image_name_builds_workspace_image_when_dockerfile_exists(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")
        manifest_path = self.workspace_root / HIDDEN_MANIFEST_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            '{"runtime_plan": {"services": [{"service_id": "app", "container_image": null}]}}',
            encoding="utf-8",
        )
        (self.workspace_root / "Dockerfile").write_text("FROM rust:1.82-bookworm\n", encoding="utf-8")

        with patch.object(service, "_ensure_workspace_runtime_image", return_value="course-gen-runtime:workspace") as mock_build:
            image_name = service._workspace_runtime_image_name(self.workspace_root)

        self.assertEqual(image_name, "course-gen-runtime:workspace")
        mock_build.assert_called_once_with(self.workspace_root)

    def test_workspace_runtime_image_name_prefers_authored_dockerfile_over_manifest_image(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")
        manifest_path = self.workspace_root / HIDDEN_MANIFEST_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            '{"runtime_plan": {"services": [{"service_id": "app", "container_image": "rust:1.82-bookworm"}]}}',
            encoding="utf-8",
        )
        (self.workspace_root / "Dockerfile").write_text("FROM rust:1.82-bookworm\n", encoding="utf-8")

        with patch.object(service, "_ensure_workspace_runtime_image", return_value="course-gen-runtime:workspace") as mock_build:
            image_name = service._workspace_runtime_image_name(self.workspace_root)

        self.assertEqual(image_name, "course-gen-runtime:workspace")
        mock_build.assert_called_once_with(self.workspace_root)

    def test_workspace_runtime_image_name_uses_manifest_image_when_no_authored_dockerfile_exists(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")
        manifest_path = self.workspace_root / HIDDEN_MANIFEST_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            '{"runtime_plan": {"services": [{"service_id": "app", "container_image": "rust:1.82-bookworm"}]}}',
            encoding="utf-8",
        )

        with patch.object(service, "_ensure_workspace_runtime_image") as mock_build:
            image_name = service._workspace_runtime_image_name(self.workspace_root)

        self.assertEqual(image_name, "rust:1.82-bookworm")
        mock_build.assert_not_called()

    def test_ephemeral_runtime_workspace_keeps_generated_artifacts_off_host_workspace(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")
        (self.workspace_root / "src").mkdir(parents=True, exist_ok=True)
        (self.workspace_root / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")

        with service._ephemeral_runtime_workspace(self.workspace_root) as runtime_workspace:
            self.assertNotEqual(runtime_workspace, self.workspace_root)
            self.assertTrue((runtime_workspace / "src" / "main.rs").exists())
            (runtime_workspace / "target" / "debug").mkdir(parents=True, exist_ok=True)
            (runtime_workspace / "target" / "debug" / "demo").write_text("compiled\n", encoding="utf-8")

        self.assertFalse((self.workspace_root / "target").exists())

    def test_workspace_runtime_image_build_reclaims_managed_space_and_labels_image_when_disk_is_low(self) -> None:
        service = LearnerStudioService(
            image_name="course-gen-learner-studio:test",
            minimum_free_disk_bytes=3 * 1024 * 1024 * 1024,
        )
        (self.workspace_root / "Dockerfile").write_text("FROM rust:1.82-bookworm\n", encoding="utf-8")

        with (
            patch.object(service, "_image_exists", return_value=False),
            patch(
                "app.services.learner_studio_service.shutil.disk_usage",
                side_effect=[
                    SimpleNamespace(total=10, used=9, free=512 * 1024 * 1024),
                    SimpleNamespace(total=10, used=4, free=6 * 1024 * 1024 * 1024),
                ],
            ),
            patch(
                "app.services.learner_studio_service.subprocess.run",
                side_effect=[
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                ],
            ) as mock_run,
        ):
            image_name = service._ensure_workspace_runtime_image(self.workspace_root)

        self.assertTrue(image_name.startswith("course-gen-runtime:"))
        prune_commands = [call.args[0] for call in mock_run.call_args_list[:2]]
        self.assertEqual(prune_commands[0][:4], ["docker", "builder", "prune", "-af"])
        self.assertEqual(prune_commands[1][:4], ["docker", "image", "prune", "-af"])
        build_command = mock_run.call_args_list[-1].args[0]
        self.assertIn("--label", build_command)
        self.assertIn("coursegen.managed=true", build_command)
        self.assertIn("coursegen.kind=runtime", build_command)

    def test_workspace_editor_image_build_labels_image(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test")

        with (
            patch.object(service, "_image_exists", return_value=False),
            patch.object(service, "_ensure_docker_build_capacity"),
            patch(
                "app.services.learner_studio_service.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ) as mock_run,
        ):
            image_name = service._ensure_workspace_editor_image("course-gen-runtime:test")

        self.assertTrue(image_name.startswith("course-gen-editor:"))
        build_command = mock_run.call_args.args[0]
        self.assertIn("--label", build_command)
        self.assertIn("coursegen.managed=true", build_command)
        self.assertIn("coursegen.kind=editor", build_command)

    def test_workspace_runtime_image_build_fails_loudly_when_disk_stays_full_after_cleanup(self) -> None:
        service = LearnerStudioService(
            image_name="course-gen-learner-studio:test",
            minimum_free_disk_bytes=3 * 1024 * 1024 * 1024,
        )
        (self.workspace_root / "Dockerfile").write_text("FROM rust:1.82-bookworm\n", encoding="utf-8")

        with (
            patch.object(service, "_image_exists", return_value=False),
            patch(
                "app.services.learner_studio_service.shutil.disk_usage",
                side_effect=[
                    SimpleNamespace(total=10, used=9, free=512 * 1024 * 1024),
                    SimpleNamespace(total=10, used=9, free=512 * 1024 * 1024),
                    SimpleNamespace(total=10, used=9, free=512 * 1024 * 1024),
                ],
            ),
            patch(
                "app.services.learner_studio_service.subprocess.run",
                side_effect=[
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                    SimpleNamespace(returncode=0, stdout="", stderr=""),
                ],
            ),
        ):
            with self.assertRaisesRegex(LearnerStudioError, "Insufficient free disk space for Docker builds"):
                service._ensure_workspace_runtime_image(self.workspace_root)

    def test_wait_for_http_fails_fast_when_container_exits_before_healthcheck(self) -> None:
        service = LearnerStudioService(image_name="course-gen-learner-studio:test", start_timeout_s=90)

        with (
            patch("app.services.learner_studio_service.httpx.get", side_effect=RuntimeError("connection refused")),
            patch.object(service, "_container_running", return_value=False),
            patch.object(service, "_container_logs", return_value="app boot failed"),
        ):
            with self.assertRaisesRegex(
                LearnerStudioError,
                "stopped before 'http://127.0.0.1:18001/health' became healthy",
            ):
                service._wait_for_http(
                    "http://127.0.0.1:18001/health",
                    container_name="course-gen-sandbox-deliverable_1-deadbeef",
                )


if __name__ == "__main__":
    unittest.main()
