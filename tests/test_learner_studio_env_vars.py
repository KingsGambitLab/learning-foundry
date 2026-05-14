from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from app.domain.learner import LearnerWorkspaceScope
from app.services.learner_studio_service import LearnerStudioService


class LearnerStudioEnvVarsTest(unittest.TestCase):
    def test_launch_passes_lab_tutor_env_vars_and_extensions_dir(self) -> None:
        with TemporaryDirectory() as td:
            workspace = Path(td) / "ws"
            workspace.mkdir()
            svc = LearnerStudioService(tutor_base_url="http://lab-tutor.svc:8000")

            captured: dict[str, object] = {}

            def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
                # First docker call is `docker run` — capture it; later mock calls return success.
                if "run" in cmd and "-d" in cmd:
                    captured["cmd"] = cmd
                result = MagicMock()
                result.returncode = 0
                result.stdout = ""
                result.stderr = ""
                return result

            with patch.object(svc, "_allocate_port", return_value=8765), \
                 patch.object(svc, "_ensure_image"), \
                 patch.object(svc, "_remove_runtime_support"), \
                 patch.object(svc, "_dependency_services", return_value=[]), \
                 patch.object(svc, "_wait_for_http"), \
                 patch("app.services.learner_studio_service.subprocess.run", side_effect=fake_run):
                svc.launch_editor(
                    enrollment_id="enr-1",
                    deliverable_id="del-1",
                    workspace_root=workspace,
                    scope=LearnerWorkspaceScope.shared_course,
                )

            cmd = captured.get("cmd")
            self.assertIsNotNone(cmd, "docker run command must be captured")
            assert isinstance(cmd, list)
            self.assertIn("-e", cmd)
            self.assertIn("LAB_TUTOR_BASE_URL=http://lab-tutor.svc:8000", cmd)
            tutor_session_env = [c for c in cmd if c.startswith("LAB_TUTOR_SESSION_ID=")]
            self.assertEqual(len(tutor_session_env), 1)
            self.assertTrue(tutor_session_env[0].startswith("LAB_TUTOR_SESSION_ID=studio_"))
            self.assertIn("--extensions-dir", cmd)
            self.assertIn("/opt/lab-tutor/extensions", cmd)


if __name__ == "__main__":
    unittest.main()
