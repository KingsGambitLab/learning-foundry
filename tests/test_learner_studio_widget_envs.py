from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from app.domain.learner import LearnerWorkspaceScope
from app.services.learner_studio_service import LearnerStudioService


class LearnerStudioWidgetEnvsTest(unittest.TestCase):
    def _run_launch(self, *, lab_tutor_enabled: bool, assignment_title: str | None = None, enrollment_id: str = "enr-1"):
        captured: dict[str, object] = {}

        def fake_run(cmd, *args, **kwargs):
            if "run" in cmd and "-d" in cmd:
                captured["cmd"] = cmd
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with TemporaryDirectory() as td:
            workspace = Path(td) / "ws"
            workspace.mkdir()
            svc = LearnerStudioService(tutor_base_url="http://lab-tutor.svc:8012")
            with patch.object(svc, "_allocate_port", return_value=8765), \
                 patch.object(svc, "_ensure_image"), \
                 patch.object(svc, "_remove_runtime_support"), \
                 patch.object(svc, "_dependency_services", return_value=[]), \
                 patch.object(svc, "_wait_for_http"), \
                 patch("app.services.learner_studio_service.subprocess.run", side_effect=fake_run):
                svc.launch_editor(
                    enrollment_id=enrollment_id,
                    deliverable_id="del-1",
                    workspace_root=workspace,
                    scope=LearnerWorkspaceScope.shared_course,
                    lab_tutor_enabled=lab_tutor_enabled,
                    assignment_title=assignment_title,
                )
            return captured.get("cmd")

    def test_lab_tutor_env_vars_pass_when_enabled(self) -> None:
        cmd = self._run_launch(lab_tutor_enabled=True, assignment_title="Build a thing")
        self.assertIsNotNone(cmd)
        assert isinstance(cmd, list)
        base_url_value = "LAB_TUTOR_BASE_URL=http://lab-tutor.svc:8012"
        self.assertIn(base_url_value, cmd)
        self.assertEqual(cmd[cmd.index(base_url_value) - 1], "-e")
        title_value = "LAB_TUTOR_ASSIGNMENT_TITLE=Build a thing"
        self.assertIn(title_value, cmd)
        self.assertEqual(cmd[cmd.index(title_value) - 1], "-e")

    def test_title_omitted_when_not_provided(self) -> None:
        cmd = self._run_launch(lab_tutor_enabled=True)
        assert isinstance(cmd, list)
        self.assertFalse(any(c.startswith("LAB_TUTOR_ASSIGNMENT_TITLE=") for c in cmd))

    def test_all_env_vars_omitted_when_disabled(self) -> None:
        cmd = self._run_launch(lab_tutor_enabled=False, assignment_title="Build a thing")
        assert isinstance(cmd, list)
        self.assertFalse(any(c.startswith("LAB_TUTOR_") for c in cmd))

    def test_enrollment_id_env_var_passes_when_enabled(self) -> None:
        cmd = self._run_launch(lab_tutor_enabled=True, assignment_title="Build a thing", enrollment_id="enr-test")
        self.assertIsNotNone(cmd)
        assert isinstance(cmd, list)
        enrollment_value = "LAB_TUTOR_ENROLLMENT_ID=enr-test"
        self.assertIn(enrollment_value, cmd)
        self.assertEqual(cmd[cmd.index(enrollment_value) - 1], "-e")


if __name__ == "__main__":
    unittest.main()
