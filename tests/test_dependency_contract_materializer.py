from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.domain.task_agent import ProjectRuntimePlanSpec, ProjectRuntimeServiceSpec
from app.services.dependency_contract_materializer import DependencyContractMaterializer
from app.services.task_agent_starter_templates import HIDDEN_MANIFEST_PATH, RUNTIME_INSTALL_SCRIPT_PATH


class DependencyContractMaterializerTests(unittest.TestCase):
    def test_materialize_syncs_generated_contract_files_back_to_starter_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            starter_root = Path(temp_dir) / "starter"
            starter_root.mkdir(parents=True, exist_ok=True)
            (starter_root / ".coursegen" / "runtime").mkdir(parents=True, exist_ok=True)
            (starter_root / "Cargo.toml").write_text(
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                encoding="utf-8",
            )
            (starter_root / RUNTIME_INSTALL_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo generate-lockfile\n",
                encoding="utf-8",
            )
            (starter_root / HIDDEN_MANIFEST_PATH).write_text(
                (
                    '{"runtime_plan": {"services": [{"service_id": "app", '
                    '"container_image": "rust:1.82-bookworm"}]}, '
                    '"dependency_contract": {"manifest_paths": ["Cargo.toml"], '
                    '"lockfile_paths": ["Cargo.lock"], '
                    '"toolchain_paths": ["rust-toolchain.toml"], '
                    '"build_support_paths": [], '
                    '"reproducibility_mode": "locked"}}'
                ),
                encoding="utf-8",
            )
            runtime_plan = ProjectRuntimePlanSpec(
                package_manager="cargo",
                services=[
                    ProjectRuntimeServiceSpec(
                        service_id="app",
                        role="application",
                        technology="rust",
                        container_image="rust:1.82-bookworm",
                    )
                ],
            )

            def _fake_run(command, **kwargs):  # noqa: ANN001
                mounted_root = Path(command[4].split(":", 1)[0])
                (mounted_root / "Cargo.lock").write_text("# generated lockfile\n", encoding="utf-8")
                (mounted_root / "rust-toolchain.toml").write_text(
                    '[toolchain]\nchannel = "1.82.0"\n',
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="generated", stderr="")

            with patch("app.services.dependency_contract_materializer.subprocess.run", side_effect=_fake_run):
                result = DependencyContractMaterializer().materialize(
                    starter_root=starter_root,
                    runtime_plan=runtime_plan,
                    deliverable_id="deliverable_1",
                )

            self.assertTrue(result.attempted)
            self.assertTrue(result.succeeded)
            self.assertEqual(result.image_name, "rust:1.82-bookworm")
            self.assertIn("Cargo.lock", result.synced_paths)
            self.assertIn("rust-toolchain.toml", result.synced_paths)
            self.assertEqual((starter_root / "Cargo.lock").read_text(encoding="utf-8"), "# generated lockfile\n")
            self.assertEqual(
                (starter_root / "rust-toolchain.toml").read_text(encoding="utf-8"),
                '[toolchain]\nchannel = "1.82.0"\n',
            )

    def test_materialize_returns_failed_result_when_install_step_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            starter_root = Path(temp_dir) / "starter"
            starter_root.mkdir(parents=True, exist_ok=True)
            (starter_root / ".coursegen" / "runtime").mkdir(parents=True, exist_ok=True)
            (starter_root / "Cargo.toml").write_text(
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                encoding="utf-8",
            )
            (starter_root / RUNTIME_INSTALL_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo generate-lockfile\n",
                encoding="utf-8",
            )
            (starter_root / HIDDEN_MANIFEST_PATH).write_text(
                (
                    '{"dependency_contract": {"manifest_paths": ["Cargo.toml"], '
                    '"lockfile_paths": ["Cargo.lock"], '
                    '"toolchain_paths": [], '
                    '"build_support_paths": [], '
                    '"reproducibility_mode": "locked"}}'
                ),
                encoding="utf-8",
            )
            runtime_plan = ProjectRuntimePlanSpec(
                package_manager="cargo",
                services=[
                    ProjectRuntimeServiceSpec(
                        service_id="app",
                        role="application",
                        technology="rust",
                        container_image="rust:1.82-bookworm",
                    )
                ],
            )

            with patch(
                "app.services.dependency_contract_materializer.subprocess.run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr="lockfile resolution failed"),
            ):
                result = DependencyContractMaterializer().materialize(
                    starter_root=starter_root,
                    runtime_plan=runtime_plan,
                    deliverable_id="deliverable_1",
                )

            self.assertTrue(result.attempted)
            self.assertFalse(result.succeeded)
            self.assertIn("lockfile resolution failed", result.error or "")
            self.assertFalse((starter_root / "Cargo.lock").exists())


if __name__ == "__main__":
    unittest.main()
