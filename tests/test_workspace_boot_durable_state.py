"""Regression tests for durable_state_required volume mounting.

The promptfoo prompt-pipeline live smoke (2026-05-15, course
``course_25d08a784cad``) blocked at starter_verify because
``_provision_capabilities`` unconditionally rejected
``durable_state_required=True``. The boot reported::

    sandbox cannot provision capability 'durable_state_required':
    spec requests durable_state_required=True but the sandbox does
    not yet mount a persistent /data volume

The course legitimately needs persistent state — versioned prompts +
their evaluation reports must survive container restarts within the
verification cycle.

Fix:
- ``_start_container`` accepts ``data_volume_host_dir`` and adds
  ``-v <host>:/data`` to the docker invocation when set.
- ``boot_and_verify`` provisions ``<workspace_dir>/.coursegen_data`` as
  the host path when ``capabilities.durable_state_required`` is True.
- ``_provision_capabilities`` no longer rejects the capability.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.course_outcome_models import CapabilityFlags
from app.services.workspace_boot import (
    WorkspaceBootCapabilityError,
    _provision_capabilities,
    boot_and_verify,
)


def _success_run(stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)


class _ScriptedSubprocess:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.results.pop(0)


class _ScriptedHealth:
    """Mirror tests/test_workspace_boot.py's _ScriptedHealth (MagicMock-based
    so urllib's context-manager invocation works)."""

    def __init__(self, statuses) -> None:
        self.statuses = list(statuses)

    def __call__(self, *args, **kwargs):
        status = self.statuses.pop(0)
        resp = MagicMock()
        resp.status = status
        resp.__enter__ = lambda self_: resp
        resp.__exit__ = lambda self_, *a: False
        return resp


class DurableStateVolumeTests(unittest.TestCase):
    def test_provision_capabilities_no_longer_rejects_durable_state(self) -> None:
        """The previous behavior was an unconditional raise; the fix
        removes that — the volume mount happens at start time."""
        try:
            _provision_capabilities(
                CapabilityFlags(durable_state_required=True)
            )
        except WorkspaceBootCapabilityError as exc:  # pragma: no cover
            self.fail(
                f"durable_state_required should no longer raise; got {exc}"
            )

    def test_docker_run_includes_volume_mount_when_durable_state_required(
        self,
    ) -> None:
        sub = _ScriptedSubprocess([
            _success_run(),                       # docker build
            _success_run(stdout="container-id\n"),  # docker run
            _success_run(),                       # docker stop (teardown)
            _success_run(),                       # docker rm
        ])
        health = _ScriptedHealth([200])
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with patch(
                "app.services.workspace_boot.subprocess.run", sub
            ), patch(
                "app.services.workspace_boot.urllib.request.urlopen", health
            ), patch(
                "app.services.workspace_boot._allocate_port", return_value=49200
            ), patch("app.services.workspace_boot.time.sleep"):
                with boot_and_verify(
                    workspace,
                    capabilities=CapabilityFlags(durable_state_required=True),
                ):
                    pass
            # Inspect inside the tempdir lifetime so the host data dir
            # is still on disk for the existence check.
            run_args = sub.calls[1][0][0]
            self.assertIn("-v", run_args)
            v_idx = run_args.index("-v")
            mount = run_args[v_idx + 1]
            self.assertTrue(
                mount.endswith(":/data"),
                f"expected ``...:/data`` mount, got {mount!r}",
            )
            host_side = mount[: -len(":/data")]
            self.assertTrue(host_side.endswith(".coursegen_data"))
            # Pre-created so docker doesn't auto-create it owned by root.
            self.assertTrue(Path(host_side).exists())

    def test_no_volume_mount_when_durable_state_off(self) -> None:
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="c\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with patch(
                "app.services.workspace_boot.subprocess.run", sub
            ), patch(
                "app.services.workspace_boot.urllib.request.urlopen", health
            ), patch(
                "app.services.workspace_boot._allocate_port", return_value=49201
            ), patch("app.services.workspace_boot.time.sleep"):
                with boot_and_verify(workspace, capabilities=CapabilityFlags()):
                    pass

        run_args = sub.calls[1][0][0]
        self.assertNotIn("-v", run_args)


if __name__ == "__main__":
    unittest.main()
