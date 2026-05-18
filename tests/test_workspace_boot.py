"""Tests for ``app.services.workspace_boot``.

The ``boot_and_verify`` helper builds a Docker image from a workspace,
launches the resulting container on a free local port, and polls the
``/health`` endpoint until it returns a 2xx/3xx/4xx response. It exposes
the booted handle as a context manager so callers (``RealStarterVerifier``,
the oracle-pass sandbox adapter) can rely on teardown happening even when
the body raises.

These tests use ``unittest.mock.patch`` to stub ``subprocess.run`` and
``urllib.request.urlopen``. No real Docker daemon is involved.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.course_outcome_models import CapabilityFlags
from app.services.workspace_boot import (
    WorkspaceBootCapabilityError,
    WorkspaceBootError,
    WorkspaceBootHandle,
    boot_and_verify,
)


def _success_run(stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)


def _failure_run(stderr: str, returncode: int = 1) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)


class _ScriptedSubprocess:
    """Sequenced ``subprocess.run`` stand-in.

    Each call returns the next ``SimpleNamespace`` from the supplied list
    and records the (args, kwargs) for assertion.
    """

    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self.results:
            raise AssertionError(
                "subprocess.run called more times than scripted "
                f"(call #{len(self.calls)}, args={args})"
            )
        return self.results.pop(0)


class _ScriptedHealth:
    """Sequenced /health stand-in for ``urllib.request.urlopen``."""

    def __init__(self, statuses: list[int | type[Exception]]) -> None:
        self.statuses = list(statuses)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if not self.statuses:
            raise AssertionError(
                f"urlopen called more times than scripted (call #{self.calls})"
            )
        item = self.statuses.pop(0)
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("scripted connection failure")
        resp = MagicMock()
        resp.status = item
        resp.__enter__ = lambda self_: resp
        resp.__exit__ = lambda self_, *a: False
        return resp


class WorkspaceBootTests(unittest.TestCase):
    """Cover the build/run/poll/teardown flow with all I/O mocked."""

    def setUp(self) -> None:
        self.workspace_dir = Path("/tmp/fake-workspace-dir")

    # ---- Build step ----

    def test_boot_and_verify_invokes_docker_build_with_workspace_dir(self) -> None:
        """``docker build -t <tag> <workspace_dir>`` runs first."""
        sub = _ScriptedSubprocess([
            _success_run(),  # docker build
            _success_run(stdout="container-abc\n"),  # docker run -d
            _success_run(),  # docker stop (teardown)
            _success_run(),  # docker rm (teardown)
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=54321):
            with boot_and_verify(self.workspace_dir, image_tag="course-gen-test:abc") as handle:
                self.assertEqual(handle.base_url, "http://127.0.0.1:54321")

        # First subprocess call should be the docker build pointed at the workspace dir.
        first_args, _ = sub.calls[0]
        cmd = first_args[0]
        self.assertEqual(cmd[0], "docker")
        self.assertEqual(cmd[1], "build")
        self.assertIn("-t", cmd)
        tag_idx = cmd.index("-t") + 1
        self.assertEqual(cmd[tag_idx], "course-gen-test:abc")
        # The build context (last positional) MUST be the workspace dir.
        self.assertEqual(cmd[-1], str(self.workspace_dir))

    def test_boot_and_verify_starts_container_on_allocated_port(self) -> None:
        """``docker run -d`` publishes the allocated port and uses the built tag."""
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="abc1234\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=49152):
            with boot_and_verify(self.workspace_dir, image_tag="img:v1") as handle:
                self.assertEqual(handle.base_url, "http://127.0.0.1:49152")
                self.assertEqual(handle.container_id, "abc1234")

        run_args = sub.calls[1][0][0]
        self.assertEqual(run_args[0], "docker")
        self.assertEqual(run_args[1], "run")
        self.assertIn("-d", run_args)
        # The port publish argument carries the host-side port we allocated.
        publish_idx = run_args.index("-p")
        self.assertEqual(run_args[publish_idx + 1].split(":")[0], "49152")
        # The image tag is the build tag we asked for.
        self.assertIn("img:v1", run_args)

    def test_boot_and_verify_polls_health_until_ok(self) -> None:
        """Polls /health repeatedly until a non-5xx response is observed."""
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="container-id\n"),
            _success_run(),
            _success_run(),
        ])
        # First two polls connection-refused; third returns 200.
        health = _ScriptedHealth([ConnectionRefusedError, ConnectionRefusedError, 200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=51000), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir) as handle:
                self.assertEqual(handle.base_url, "http://127.0.0.1:51000")
        self.assertEqual(health.calls, 3)

    def test_boot_and_verify_raises_on_health_timeout(self) -> None:
        """When /health never returns 2xx within the budget, raises ``WorkspaceBootError``."""
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="container-id\n"),
            _success_run(),  # teardown stop
            _success_run(),  # teardown rm
        ])
        # Always raise — never reaches a 200.
        health = _ScriptedHealth([ConnectionRefusedError] * 100)
        # Patch time so the deadline triggers after a fixed number of iterations.
        clock = [0.0]

        def fake_monotonic() -> float:
            return clock[0]

        def fake_sleep(_: float) -> None:
            # Advance the clock past the readiness deadline after a few polls.
            clock[0] += 5.0

        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=51001), patch(
            "app.services.workspace_boot.time.monotonic", fake_monotonic
        ), patch("app.services.workspace_boot.time.sleep", fake_sleep):
            with self.assertRaises(WorkspaceBootError) as cm:
                with boot_and_verify(self.workspace_dir, readiness_timeout_s=10.0):
                    self.fail("context body should not run when readiness times out")
        self.assertIn("health", str(cm.exception).lower())
        # Teardown still happened — last two subprocess calls are stop + rm.
        last_cmds = [call[0][0][:2] for call in sub.calls[-2:]]
        self.assertIn(["docker", "stop"], last_cmds)
        self.assertIn(["docker", "rm"], last_cmds)

    def test_context_manager_calls_teardown_even_when_body_raises(self) -> None:
        """``__exit__`` tears down the container even on exceptions in the body."""
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="container-id\n"),
            _success_run(),  # docker stop
            _success_run(),  # docker rm
        ])
        health = _ScriptedHealth([200])

        class BodyError(RuntimeError):
            pass

        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=52000), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with self.assertRaises(BodyError):
                with boot_and_verify(self.workspace_dir):
                    raise BodyError("oops")
        # The last two subprocess calls are the teardown stop/rm.
        last_two = [call[0][0][:2] for call in sub.calls[-2:]]
        self.assertIn(["docker", "stop"], last_two)
        self.assertIn(["docker", "rm"], last_two)

    def test_handle_base_url_matches_bound_port(self) -> None:
        """The handle's ``base_url`` reflects the host-side port we bound."""
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=53789), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir) as handle:
                self.assertIsInstance(handle, WorkspaceBootHandle)
                self.assertEqual(handle.base_url, "http://127.0.0.1:53789")

    def test_build_failure_raises_workspace_boot_error_with_stderr(self) -> None:
        """``docker build`` failure surfaces the stderr text in the error."""
        sub = _ScriptedSubprocess([
            _failure_run(stderr="failed to read Dockerfile: not found"),
        ])
        health = _ScriptedHealth([])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=54000):
            with self.assertRaises(WorkspaceBootError) as cm:
                with boot_and_verify(self.workspace_dir):
                    self.fail("context body should not run when build fails")
        self.assertIn("failed to read Dockerfile", str(cm.exception))

    def test_container_start_failure_raises_workspace_boot_error_with_stderr(self) -> None:
        """``docker run`` failure surfaces the stderr text in the error."""
        sub = _ScriptedSubprocess([
            _success_run(),  # build ok
            _failure_run(stderr="port already allocated"),  # run -d fails
        ])
        health = _ScriptedHealth([])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=54001):
            with self.assertRaises(WorkspaceBootError) as cm:
                with boot_and_verify(self.workspace_dir):
                    self.fail("context body should not run when container start fails")
        self.assertIn("port already allocated", str(cm.exception))

    # ---- Capability threading (Codex review #7 #3) ----

    def test_boot_and_verify_capabilities_none_is_default_behavior(self) -> None:
        """``capabilities=None`` (the default) preserves the current boot path.

        No capability checks run, no proxy lookup, the existing
        build/run/poll/teardown sequence executes unchanged.
        """
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=56001), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir, capabilities=None) as handle:
                self.assertIsInstance(handle, WorkspaceBootHandle)

    def test_boot_and_verify_capabilities_all_default_is_unchanged(self) -> None:
        """A default-valued ``CapabilityFlags()`` requests nothing — boot proceeds normally."""
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=56002), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir, capabilities=CapabilityFlags()) as handle:
                self.assertIsInstance(handle, WorkspaceBootHandle)

    def test_boot_and_verify_runtime_llm_required_raises_when_proxy_unavailable(self) -> None:
        """``runtime_llm_required=True`` MUST refuse to boot when the proxy isn't pre-started.

        The harness contract is: the deployment owner pre-starts the
        sandbox LLM proxy on the docker network with DNS name
        ``coursegen-llm``. If a probe says it's unavailable we surface a
        ``WorkspaceBootCapabilityError`` naming the missing capability
        rather than silently booting a bare container.
        """
        sub = _ScriptedSubprocess([])  # we should never reach docker build
        health = _ScriptedHealth([])
        caps = CapabilityFlags(runtime_llm_required=True)
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch(
            "app.services.workspace_boot._is_llm_proxy_available", return_value=False
        ):
            with self.assertRaises(WorkspaceBootCapabilityError) as cm:
                with boot_and_verify(self.workspace_dir, capabilities=caps):
                    self.fail("body should not run when capability provisioning fails")
        # The error MUST name the missing capability so downstream
        # repair / blocking reasons surface the cause.
        self.assertIn("runtime_llm_required", str(cm.exception))
        self.assertEqual(cm.exception.capability, "runtime_llm_required")

    def test_boot_and_verify_sidecar_database_postgres_raises(self) -> None:
        """``sidecar_database='postgres'`` raises until full provisioning lands.

        The error names the missing capability so the graph can route
        the failure into the same repair / blocking surface as the
        proxy-missing case.
        """
        sub = _ScriptedSubprocess([])
        health = _ScriptedHealth([])
        caps = CapabilityFlags(sidecar_database="postgres")
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ):
            with self.assertRaises(WorkspaceBootCapabilityError) as cm:
                with boot_and_verify(self.workspace_dir, capabilities=caps):
                    self.fail("body should not run when sidecar provisioning fails")
        self.assertIn("sidecar_database", str(cm.exception))
        self.assertEqual(cm.exception.capability, "sidecar_database")

    def test_default_image_tag_is_synthesized_when_not_supplied(self) -> None:
        """When the caller omits ``image_tag``, the helper synthesizes one."""
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=55001), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir) as handle:
                self.assertTrue(handle.image_tag)
                # Synthesized tag has the conventional prefix.
                self.assertTrue(handle.image_tag.startswith("course-gen-outcome:"))


class DockerfileExposePortTests(unittest.TestCase):
    """Boot must publish the host port to whatever the Dockerfile ``EXPOSE``s.

    Live-run finding (2026-05-14): the LLM-generated starter binds uvicorn to
    port 8000 and ``EXPOSE 8000`` in its Dockerfile, but ``workspace_boot``
    used to hardcode ``CONTAINER_PORT = 8080``. ``docker run -p host:8080``
    accepts the host TCP connection then RSTs because nothing inside the
    container listens on 8080 — the harness reports ``ConnectionResetError``
    and the starter never verifies.

    The fix parses ``EXPOSE`` from the workspace's Dockerfile and uses that
    port for ``-p host:<internal>``. When no Dockerfile or no EXPOSE is
    present, the legacy 8080 default is preserved so existing callers
    keep working.
    """

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace_dir = Path(self._tmp.name)

    def _write_dockerfile(self, contents: str) -> None:
        (self.workspace_dir / "Dockerfile").write_text(contents)

    def test_boot_publishes_port_from_dockerfile_expose_8000(self) -> None:
        """Dockerfile ``EXPOSE 8000`` → ``docker run -p <host>:8000``."""
        self._write_dockerfile("FROM python:3.11-slim\nEXPOSE 8000\nCMD [\"x\"]\n")
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=49160), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir):
                pass
        run_args = sub.calls[1][0][0]
        publish_idx = run_args.index("-p")
        self.assertEqual(run_args[publish_idx + 1], "49160:8000")

    def test_boot_publishes_port_from_dockerfile_expose_8080(self) -> None:
        """Legacy ``EXPOSE 8080`` keeps working unchanged."""
        self._write_dockerfile("FROM x\nEXPOSE 8080\n")
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=49161), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir):
                pass
        run_args = sub.calls[1][0][0]
        publish_idx = run_args.index("-p")
        self.assertEqual(run_args[publish_idx + 1], "49161:8080")

    def test_boot_falls_back_to_8080_when_dockerfile_has_no_expose(self) -> None:
        """No ``EXPOSE`` directive — fall back to the legacy 8080 default."""
        self._write_dockerfile("FROM x\nCMD [\"y\"]\n")
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=49162), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir):
                pass
        run_args = sub.calls[1][0][0]
        publish_idx = run_args.index("-p")
        self.assertEqual(run_args[publish_idx + 1], "49162:8080")

    def test_boot_falls_back_to_8080_when_no_dockerfile_present(self) -> None:
        """Missing Dockerfile (legacy test setup) — fall back to 8080."""
        # Don't write a Dockerfile.
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=49163), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir):
                pass
        run_args = sub.calls[1][0][0]
        publish_idx = run_args.index("-p")
        self.assertEqual(run_args[publish_idx + 1], "49163:8080")

    def test_boot_handles_expose_with_protocol_suffix(self) -> None:
        """``EXPOSE 9000/tcp`` is a valid Docker form — strip the protocol."""
        self._write_dockerfile("FROM x\nEXPOSE 9000/tcp\n")
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=49164), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir):
                pass
        run_args = sub.calls[1][0][0]
        publish_idx = run_args.index("-p")
        self.assertEqual(run_args[publish_idx + 1], "49164:9000")

    def test_boot_uses_first_expose_when_multiple_declared(self) -> None:
        """When Dockerfile declares multiple EXPOSE ports, use the first one.

        Multi-port containers are rare in our generated starters and the
        first port is the conventional ``/health`` port.
        """
        self._write_dockerfile("FROM x\nEXPOSE 8000\nEXPOSE 9090\n")
        sub = _ScriptedSubprocess([
            _success_run(),
            _success_run(stdout="cid\n"),
            _success_run(),
            _success_run(),
        ])
        health = _ScriptedHealth([200])
        with patch("app.services.workspace_boot.subprocess.run", sub), patch(
            "app.services.workspace_boot.urllib.request.urlopen", health
        ), patch("app.services.workspace_boot._allocate_port", return_value=49165), patch(
            "app.services.workspace_boot.time.sleep"
        ):
            with boot_and_verify(self.workspace_dir):
                pass
        run_args = sub.calls[1][0][0]
        publish_idx = run_args.index("-p")
        self.assertEqual(run_args[publish_idx + 1], "49165:8000")


if __name__ == "__main__":
    unittest.main()
