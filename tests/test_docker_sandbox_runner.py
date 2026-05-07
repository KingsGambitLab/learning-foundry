from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.domain.sandbox import SandboxAvailability
from app.domain.workflow import ArtifactVisibility, BundleFile, MaterializedBundle
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.examples import get_support_triage_example


class AvailableDockerSandboxRunner(DockerSandboxRunner):
    def status(self) -> SandboxAvailability:
        return SandboxAvailability(
            available=True,
            message="Docker available for tests.",
            docker_version="test",
        )


def _make_run(workspace_public_dir: Path):
    spec = get_support_triage_example()
    bundle = MaterializedBundle(
        bundle_id="bundle_test",
        generated_at=datetime.now(UTC),
        root_dir=str(workspace_public_dir.parent),
        public_dir=str(workspace_public_dir),
        private_dir=str(workspace_public_dir.parent / "private"),
        manifest_path=str(workspace_public_dir.parent / "manifest.json"),
        files=[
            BundleFile(
                relative_path="public/runtime/Dockerfile",
                visibility=ArtifactVisibility.public,
                media_type="text/plain",
                size_bytes=0,
            )
        ],
    )
    return SimpleNamespace(
        id="run_cache_test",
        artifacts=SimpleNamespace(
            task_agent_spec=spec,
            workspace_snapshot=bundle,
        ),
    )


def _write_workspace(public_dir: Path, *, marker: str = "v1") -> None:
    (public_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (public_dir / "starter" / "module_1").mkdir(parents=True, exist_ok=True)
    (public_dir / "runtime" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\n# {marker}\n",
        encoding="utf-8",
    )
    (public_dir / "runtime" / "verify_assignment.py").write_text(
        "print('hello')\n",
        encoding="utf-8",
    )
    (public_dir / "starter" / "module_1" / "app.py").write_text(
        f"print('{marker}')\n",
        encoding="utf-8",
    )


class DockerSandboxRunnerCacheTests(unittest.TestCase):
    def test_execute_reuses_cached_image_for_identical_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            public_dir = Path(temp_dir) / "bundle" / "public"
            _write_workspace(public_dir, marker="same")
            run = _make_run(public_dir)
            runner = AvailableDockerSandboxRunner(cache_images=True)

            images: set[str] = set()
            build_tags: list[str] = []

            def fake_run(cmd, **kwargs):
                if cmd[:3] == ["docker", "image", "inspect"]:
                    tag = cmd[3]
                    return _completed(cmd, 0 if tag in images else 1, "", "")
                if cmd[:2] == ["docker", "build"]:
                    tag = cmd[cmd.index("-t") + 1]
                    images.add(tag)
                    build_tags.append(tag)
                    return _completed(cmd, 0, "built", "")
                if cmd[:2] == ["docker", "run"]:
                    return _completed(
                        cmd,
                        0,
                        json.dumps({"success": True, "module_reports": []}),
                        "",
                    )
                raise AssertionError(f"Unexpected docker command: {cmd}")

            with patch("app.services.docker_sandbox_runner.subprocess.run", side_effect=fake_run):
                first = runner.execute(run)
                second = runner.execute(run)

            self.assertFalse(first.build_cached)
            self.assertTrue(second.build_cached)
            self.assertEqual(len(build_tags), 1)
            self.assertEqual(first.cache_key, second.cache_key)
            self.assertEqual(first.image_tag, second.image_tag)

    def test_execute_rebuilds_when_workspace_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            public_dir = Path(temp_dir) / "bundle" / "public"
            _write_workspace(public_dir, marker="v1")
            run = _make_run(public_dir)
            runner = AvailableDockerSandboxRunner(cache_images=True)

            images: set[str] = set()
            build_tags: list[str] = []

            def fake_run(cmd, **kwargs):
                if cmd[:3] == ["docker", "image", "inspect"]:
                    tag = cmd[3]
                    return _completed(cmd, 0 if tag in images else 1, "", "")
                if cmd[:2] == ["docker", "build"]:
                    tag = cmd[cmd.index("-t") + 1]
                    images.add(tag)
                    build_tags.append(tag)
                    return _completed(cmd, 0, "built", "")
                if cmd[:2] == ["docker", "run"]:
                    return _completed(
                        cmd,
                        0,
                        json.dumps({"success": True, "module_reports": []}),
                        "",
                    )
                raise AssertionError(f"Unexpected docker command: {cmd}")

            with patch("app.services.docker_sandbox_runner.subprocess.run", side_effect=fake_run):
                first = runner.execute(run)
                (public_dir / "starter" / "module_1" / "app.py").write_text(
                    "print('v2')\n",
                    encoding="utf-8",
                )
                second = runner.execute(run)

            self.assertEqual(len(build_tags), 2)
            self.assertNotEqual(first.cache_key, second.cache_key)
            self.assertNotEqual(first.image_tag, second.image_tag)
            self.assertFalse(second.build_cached)


def _completed(cmd, returncode: int, stdout: str, stderr: str):
    return __import__("subprocess").CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    unittest.main()
