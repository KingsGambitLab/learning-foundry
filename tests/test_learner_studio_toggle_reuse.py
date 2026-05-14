"""Tests that the lab_tutor_enabled toggle mismatch prevents session reuse.

When an existing session was launched with a different lab_tutor_enabled value
than the requested launch, the launcher must tear down the old container and
start a fresh one. When the toggle matches and the workspace is reusable, the
existing session should be returned without a new docker run.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from app.domain.learner import (
    LearnerWorkspaceScope,
    LearnerWorkspaceSession,
    LearnerWorkspaceSessionStatus,
)
from app.services.learner_studio_service import LearnerStudioService


def _make_session(
    workspace_root: str,
    lab_tutor_enabled: bool = False,
) -> LearnerWorkspaceSession:
    now = datetime.now(UTC)
    return LearnerWorkspaceSession(
        id="studio_test_toggle",
        enrollment_id="enrollment_toggle",
        deliverable_id="exercise/01-contract",
        scope=LearnerWorkspaceScope.shared_course,
        created_at=now,
        updated_at=now,
        status=LearnerWorkspaceSessionStatus.running,
        workspace_root=workspace_root,
        container_name="course-gen-studio-toggle-test",
        host_port=18090,
        editor_url="http://127.0.0.1:18090/",
        image_name="course-gen-learner-studio:latest",
        lab_tutor_enabled=lab_tutor_enabled,
        notes=["Existing session."],
    )


class LabTutorToggleReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name) / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_toggle_mismatch_forces_new_docker_run(self) -> None:
        """Existing tutor-enabled session + launch with disabled → new container."""
        existing = _make_session(str(self.workspace_root), lab_tutor_enabled=True)
        svc = LearnerStudioService(image_name="course-gen-learner-studio:test")

        captured: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
            if "run" in cmd and "-d" in cmd:
                captured.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch.object(svc, "_ensure_image"),
            patch.object(svc, "_allocate_port", return_value=19292),
            patch.object(svc, "_wait_for_http"),
            patch.object(svc, "_dependency_services", return_value=[]),
            patch.object(svc, "_remove_runtime_support") as mock_remove,
            patch("app.services.learner_studio_service.subprocess.run", side_effect=fake_run),
        ):
            result = svc.launch_editor(
                enrollment_id="enrollment_toggle",
                deliverable_id="exercise/01-contract",
                workspace_root=self.workspace_root,
                scope=LearnerWorkspaceScope.shared_course,
                existing_session=existing,
                lab_tutor_enabled=False,
            )

        # A new docker run must have been issued.
        self.assertEqual(len(captured), 1, "Expected exactly one docker run command")
        # The teardown of the old container must have been called.
        self.assertTrue(mock_remove.called, "Old container must be torn down on toggle mismatch")
        # The new session must track the updated toggle state.
        self.assertFalse(result.lab_tutor_enabled)
        # Tutor env vars must be absent from the new command.
        cmd_str = " ".join(captured[0])
        self.assertNotIn("LAB_TUTOR_BASE_URL", cmd_str)
        self.assertNotIn("LAB_TUTOR_SESSION_ID", cmd_str)
        self.assertNotIn("--extensions-dir", cmd_str)

    def test_toggle_match_and_running_container_returns_existing_session(self) -> None:
        """Existing disabled session + launch with disabled + running container → reuse."""
        existing = _make_session(str(self.workspace_root), lab_tutor_enabled=False)
        svc = LearnerStudioService(image_name="course-gen-learner-studio:test")

        with (
            patch.object(svc, "_ensure_image"),
            patch.object(svc, "_container_running", return_value=True),
            patch("app.services.learner_studio_service.subprocess.run") as mock_run,
        ):
            result = svc.launch_editor(
                enrollment_id="enrollment_toggle",
                deliverable_id="exercise/01-contract",
                workspace_root=self.workspace_root,
                scope=LearnerWorkspaceScope.shared_course,
                existing_session=existing,
                lab_tutor_enabled=False,
            )

        # No new docker run when reusing.
        self.assertEqual(mock_run.call_count, 0)
        self.assertEqual(result.id, existing.id)
        self.assertEqual(result.container_name, existing.container_name)
        self.assertFalse(result.lab_tutor_enabled)


if __name__ == "__main__":
    unittest.main()
