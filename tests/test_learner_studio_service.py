from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.domain.learner import LearnerWorkspaceScope, LearnerWorkspaceSession, LearnerWorkspaceSessionStatus
from app.services.learner_studio_service import LearnerStudioService


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
        self.assertEqual(mock_run.call_count, 1)
        docker_command = mock_run.call_args.args[0]
        self.assertIn("run", docker_command)
        self.assertIn("-v", docker_command)
        self.assertIn(f"{self.workspace_root.resolve()}:/workspace", docker_command)
        self.assertEqual(refreshed.container_name, self.existing_session.container_name)
        self.assertEqual(refreshed.host_port, 19191)
        self.assertEqual(refreshed.editor_url, "http://127.0.0.1:19191/")
        self.assertEqual(refreshed.deliverable_id, "exercise/01-contract")
        self.assertEqual(refreshed.status, LearnerWorkspaceSessionStatus.running)


if __name__ == "__main__":
    unittest.main()
