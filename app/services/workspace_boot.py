"""Boot a workspace's Dockerfile into a verified runtime.

This module exposes a single public entry point: ``boot_and_verify``.
It is a thin self-contained wrapper around ``docker build`` /
``docker run -d`` / a /health poll, used by the outcome graph's
production starter verifier and the oracle-pass reference-impl sandbox
adapter.

Why a fresh module and not ``DockerSandboxRunner``?
----------------------------------------------------

``DockerSandboxRunner`` is geared toward the legacy per-deliverable
workflow: it consumes ``WorkflowRun`` + ``WorkspaceBundle``, materializes
deliverable manifests, runs visible/hidden check scripts, and reports
back through ``SandboxExecutionResult``. The outcome path only needs a
much smaller contract — *build this workspace's Dockerfile, run it on a
free local port, confirm /health is up, and tear it down on exit*. The
existing primitives in ``DockerSandboxRunner`` / ``LearnerStudioService``
are tied to that legacy lifecycle and not easy to invoke standalone.
This module duplicates ~30 lines of subprocess plumbing rather than
plumbing a new lifecycle through the legacy runner.

Failure handling
----------------

Every failure surface raises ``WorkspaceBootError`` with the captured
stderr (or, for the readiness poll, the URL + the elapsed deadline).
Teardown is unconditional: the context manager protocol guarantees
``__exit__`` runs the ``docker stop`` + ``docker rm`` pair even when the
body raises.

Capability provisioning (Codex review #7 finding #3)
----------------------------------------------------

``boot_and_verify`` accepts an optional ``capabilities: CapabilityFlags``
argument. When supplied, ``_provision_capabilities`` runs **before**
``docker build`` and refuses the boot — via
``WorkspaceBootCapabilityError`` — when the spec asks for sandbox
primitives the deployment hasn't pre-provisioned.

The Wave 5b contract (option (b) in the design brief) is:

* ``runtime_llm_required=True`` requires the harness to have started
  the sandbox LLM proxy on the docker network with DNS name
  ``coursegen-llm`` (configurable via ``COURSEGEN_LLM_PROXY_URL``). We
  probe ``/health`` on that URL; an unreachable proxy aborts boot.
* ``sidecar_database in ("postgres", "redis")`` and
  ``durable_state_required=True`` raise unconditionally — full sidecar
  provisioning is post-Wave-5b work, and a silent bare-container boot
  would let the learner's service silently mis-call the missing
  dependency.

Failing loud is the explicit P0 contract: the CRAG smoke test (next
wave) will get a clear ``WorkspaceBootCapabilityError`` if its proxy
isn't pre-started, rather than booting a bare container that
mysteriously returns garbage at the rubric step.
"""
from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from uuid import uuid4


if TYPE_CHECKING:
    from app.services.course_outcome_models import CapabilityFlags


__all__ = [
    "WorkspaceBootCapabilityError",
    "WorkspaceBootError",
    "WorkspaceBootHandle",
    "WorkspaceBootSandboxAdapter",
    "boot_and_verify",
]


# Docker default health probe — the application's /health endpoint in
# the booted container. The trace runner / starter verifier both treat
# anything non-5xx as ready, mirroring ``LearnerStudioService._wait_for_http``.
HEALTH_PATH = "/health"
DEFAULT_READINESS_TIMEOUT_S = 30.0
DEFAULT_BUILD_TIMEOUT_S = 600
DEFAULT_RUN_TIMEOUT_S = 120
DEFAULT_TEARDOWN_TIMEOUT_S = 30
POLL_INTERVAL_S = 1.0
# Legacy default. The actual container port is now read from the workspace's
# Dockerfile ``EXPOSE`` directive (see ``_detect_container_port``); this constant
# is only used when no Dockerfile exists or it has no EXPOSE line.
DEFAULT_CONTAINER_PORT = 8080
CONTAINER_PORT = DEFAULT_CONTAINER_PORT  # back-compat alias for tests/imports.
HOST = "127.0.0.1"

# ``EXPOSE 9000`` or ``EXPOSE 9000/tcp`` — Dockerfile is case-insensitive on
# directives and the protocol suffix is optional. We extract the first integer
# port, which is the conventional health-probe target in our generated
# starters.
_EXPOSE_RE = re.compile(r"^\s*EXPOSE\s+(\d+)(?:/\w+)?", re.IGNORECASE | re.MULTILINE)

# Sandbox LLM proxy contract — the harness deployment is expected to
# pre-start the proxy on the docker network with this DNS name. Tests
# override via ``COURSEGEN_LLM_PROXY_URL`` when they want to point at a
# different probe target.
DEFAULT_LLM_PROXY_URL = "http://coursegen-llm:8080/health"
LLM_PROXY_URL_ENV = "COURSEGEN_LLM_PROXY_URL"
LLM_PROXY_PROBE_TIMEOUT_S = 2.0


class WorkspaceBootError(RuntimeError):
    """Raised when ``boot_and_verify`` cannot deliver a healthy handle.

    Three failure modes share this exception:

    1. ``docker build`` non-zero return (stderr surfaced verbatim).
    2. ``docker run -d`` non-zero return (stderr surfaced verbatim).
    3. ``/health`` never returns a non-5xx response within the timeout.
    """


class WorkspaceBootCapabilityError(WorkspaceBootError):
    """Raised when ``boot_and_verify`` can't provision a requested capability.

    Subclasses ``WorkspaceBootError`` so the verifier's existing
    classify-and-report path picks it up automatically, BUT it also
    carries structured data (``capability``, ``detail``) so the graph's
    blocking-reason surface can name *which* capability the spec asked
    for that the sandbox couldn't deliver.

    Triggered when:

    * ``runtime_llm_required=True`` but the sandbox LLM proxy isn't
      reachable on the docker network (harness deployment owns
      pre-starting it; see module docstring).
    * ``sidecar_database in {"postgres","redis"}`` — full sidecar
      provisioning is post-Wave-5b.
    * ``durable_state_required=True`` — volume-mount provisioning is
      post-Wave-5b.

    Attributes
    ----------
    capability:
        The capability flag name (e.g. ``"runtime_llm_required"``,
        ``"sidecar_database"``). Always a single field name from
        ``CapabilityFlags``.
    detail:
        Human-readable explanation of what the deployment needs to do
        to satisfy the capability (e.g. "start the LLM proxy sidecar on
        the coursegen-llm DNS name").
    """

    def __init__(self, capability: str, detail: str) -> None:
        self.capability = capability
        self.detail = detail
        super().__init__(
            f"sandbox cannot provision capability '{capability}': {detail}"
        )


@dataclass
class WorkspaceBootHandle:
    """Handle to a booted workspace container.

    The graph's oracle-pass adapter only consumes ``base_url``; the
    ``container_id`` / ``image_tag`` fields are kept for diagnostics and
    teardown.
    """

    base_url: str
    container_id: str
    image_tag: str


def _allocate_port() -> int:
    """Return a free TCP port on the loopback interface.

    Mirrors ``LearnerStudioService._allocate_port`` — bind a socket to
    port 0, capture the kernel-assigned port, release the socket.
    There's a tiny TOCTOU window before the container claims it; in
    practice the loopback rebind race is benign and is also what the
    legacy runner accepts.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _synthesize_image_tag(workspace_dir: Path) -> str:
    """Deterministic tag for re-builds of the same workspace path."""
    digest = hashlib.sha256(str(workspace_dir).encode("utf-8")).hexdigest()[:16]
    return f"course-gen-outcome:{digest}"


def _detect_container_port(workspace_dir: Path) -> int:
    """Return the first ``EXPOSE``d port in the workspace's Dockerfile.

    Falls back to ``DEFAULT_CONTAINER_PORT`` (8080) when:
    - the Dockerfile doesn't exist (legacy callers, mocked tests),
    - the Dockerfile is unreadable,
    - the Dockerfile declares no ``EXPOSE`` directive.

    Why this exists
    ---------------
    The live RAG smoke (2026-05-14) revealed that LLM-authored starters
    pick port 8000 (uvicorn default) rather than the 8080 the harness used
    to hardcode. ``docker run -p host:8080`` then accepts the TCP handshake
    and RSTs because nothing inside the container listens on 8080, surfacing
    as ``ConnectionResetError`` in the /health poll. Parsing ``EXPOSE``
    lets each starter pick its own port without coordination with the
    harness.

    Multiple ``EXPOSE`` directives: we take the first one. Multi-port
    containers are rare in our generated starters and the first is
    conventionally the HTTP /health port.
    """
    dockerfile = workspace_dir / "Dockerfile"
    try:
        contents = dockerfile.read_text()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return DEFAULT_CONTAINER_PORT
    match = _EXPOSE_RE.search(contents)
    if match is None:
        return DEFAULT_CONTAINER_PORT
    try:
        return int(match.group(1))
    except ValueError:
        return DEFAULT_CONTAINER_PORT


def _build_image(
    workspace_dir: Path,
    image_tag: str,
    *,
    docker_binary: str,
    build_timeout_s: int,
) -> None:
    """Run ``docker build -t <tag> <workspace_dir>``.

    Raises ``WorkspaceBootError`` on non-zero return.
    """
    cmd = [docker_binary, "build", "-t", image_tag, str(workspace_dir)]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=build_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceBootError(
            f"docker build timed out after {build_timeout_s}s for {workspace_dir}: {exc}"
        ) from exc
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise WorkspaceBootError(
            f"docker build failed for {workspace_dir} (exit {result.returncode}): {stderr}"
        )


def _start_container(
    image_tag: str,
    host_port: int,
    *,
    docker_binary: str,
    run_timeout_s: int,
    container_port: int = DEFAULT_CONTAINER_PORT,
    data_volume_host_dir: Path | None = None,
) -> str:
    """Run ``docker run -d -p <host_port>:<container_port> <image_tag>``.

    ``container_port`` defaults to the legacy 8080 so callers that haven't
    been updated keep the previous behavior. The outcome graph path now
    passes the workspace's actual EXPOSE'd port via
    ``_detect_container_port`` (see ``boot_and_verify``).

    ``data_volume_host_dir`` is the host path to mount at ``/data`` inside
    the container. Used to satisfy ``durable_state_required=True``
    capability so the learner's service can persist state across the
    boot-verify lifecycle. When ``None`` (default), no volume is mounted.

    Returns the container id (first line of stdout). Raises
    ``WorkspaceBootError`` on non-zero return.
    """
    cmd = [
        docker_binary,
        "run",
        "-d",
        "--rm",
        # Bind to loopback only. Workspace-boot sandboxes are
        # short-lived test containers; even on staging EC2 they must
        # never be reachable from the public interface.
        "-p",
        f"{HOST}:{host_port}:{container_port}",
    ]
    if data_volume_host_dir is not None:
        data_volume_host_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-v", f"{data_volume_host_dir.absolute()}:/data"])
    # Lab LLM proxy bridge (Customer Support Bot course). Additive and
    # opt-in: the grader sets LAB_LLM_BASE_URL/LAB_LLM_TOKEN in its own
    # process env for the duration of one submission; for every other
    # course these are absent and nothing changes. The container reaches
    # the host-bound proxy via the docker host-gateway alias; cost/abuse
    # is bounded by the proxy's per-submission token cap + global USD
    # hard-stop, not by this hop.
    cmd.extend(["--add-host", "host.docker.internal:host-gateway"])
    for _ev in ("LAB_LLM_BASE_URL", "LAB_LLM_TOKEN"):
        _val = os.environ.get(_ev)
        if _val:
            cmd.extend(["-e", f"{_ev}={_val}"])
    cmd.append(image_tag)
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=run_timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceBootError(
            f"docker run timed out after {run_timeout_s}s for image {image_tag}: {exc}"
        ) from exc
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise WorkspaceBootError(
            f"docker run failed for image {image_tag} (exit {result.returncode}): {stderr}"
        )
    container_id = (result.stdout or "").strip().splitlines()[0] if result.stdout else ""
    if not container_id:
        raise WorkspaceBootError(
            f"docker run returned no container id for image {image_tag}"
        )
    return container_id


def _poll_health(base_url: str, *, readiness_timeout_s: float) -> None:
    """Poll ``<base_url>/health`` until non-5xx or timeout.

    Raises ``WorkspaceBootError`` on timeout. Any individual request
    failure (connection refused, DNS, HTTPError 5xx, etc.) is treated
    as "not ready yet" and the loop continues until the deadline.
    """
    url = f"{base_url}{HEALTH_PATH}"
    deadline = time.monotonic() + readiness_timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                status = int(getattr(response, "status", 200))
                if status < 500:
                    return
                last_error = f"HTTP {status}"
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return
            last_error = f"HTTPError {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(POLL_INTERVAL_S)
    raise WorkspaceBootError(
        f"workspace /health never became ready at {url} "
        f"within {readiness_timeout_s:.1f}s (last_error={last_error})"
    )


def _teardown_container(container_id: str, *, docker_binary: str) -> None:
    """Best-effort stop+rm of the container.

    Both commands ignore non-zero exits — teardown happens during
    ``__exit__`` and we don't want to mask the original exception with a
    teardown failure. Diagnostics survive in the docker daemon logs.
    """
    for cmd in (
        [docker_binary, "stop", container_id],
        [docker_binary, "rm", container_id],
    ):
        try:
            subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TEARDOWN_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            # Teardown timeout is non-fatal — log via stderr would help
            # production debugging but the contract here is "best
            # effort"; surfacing it via the context manager exit would
            # mask the original exception.
            continue


def _is_llm_proxy_available() -> bool:
    """Probe the sandbox LLM proxy /health endpoint.

    Returns True iff the proxy answers a non-5xx response. Any other
    outcome (connection refused, DNS failure, 5xx, timeout) returns
    False so the boot path raises ``WorkspaceBootCapabilityError`` with
    a clear capability name rather than a generic boot timeout further
    down.

    The probe target defaults to ``http://coursegen-llm:8080/health``
    and can be overridden via ``COURSEGEN_LLM_PROXY_URL`` (mainly for
    tests + alternative deployment topologies).
    """
    url = os.environ.get(LLM_PROXY_URL_ENV, DEFAULT_LLM_PROXY_URL)
    try:
        with urllib.request.urlopen(url, timeout=LLM_PROXY_PROBE_TIMEOUT_S) as resp:
            status = int(getattr(resp, "status", 200))
            return status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except Exception:  # noqa: BLE001
        return False


def _provision_capabilities(capabilities: "CapabilityFlags | None") -> None:
    """Pre-flight check before ``docker build``: every requested capability
    must be satisfiable, else raise ``WorkspaceBootCapabilityError`` naming
    the missing capability.

    Wave 5b minimum: option (b) from the Codex review #7 design brief —
    fail loud rather than silently boot a bare container that the
    learner's service will mis-call. Full sidecar provisioning (Postgres,
    Redis, durable volumes) lands post-Wave-5b. For now those flags raise
    unconditionally.

    ``capabilities=None`` is the no-op default — preserves the existing
    contract for callers that never plumbed capabilities through.
    """
    if capabilities is None:
        return
    if getattr(capabilities, "runtime_llm_required", False):
        if not _is_llm_proxy_available():
            raise WorkspaceBootCapabilityError(
                capability="runtime_llm_required",
                detail=(
                    "the sandbox LLM proxy is not reachable. The harness "
                    "deployment must pre-start the proxy sidecar on the "
                    "docker network with DNS name 'coursegen-llm' (override "
                    f"via the {LLM_PROXY_URL_ENV} env var)."
                ),
            )
    sidecar = getattr(capabilities, "sidecar_database", "none")
    if sidecar and sidecar != "none":
        raise WorkspaceBootCapabilityError(
            capability="sidecar_database",
            detail=(
                f"spec requests sidecar_database='{sidecar}' but sandbox "
                "sidecar provisioning (postgres/redis) is not yet "
                "implemented — pre-start the sidecar on the docker network "
                f"with DNS name '{sidecar}' and document the harness "
                "deployment requirement."
            ),
        )
    # ``durable_state_required=True`` is now satisfied by ``boot_and_verify``
    # mounting ``<workspace_dir>/.coursegen_data`` at ``/data`` inside the
    # container (see the ``data_volume_host_dir`` arg threaded through
    # ``_start_container``). Per-container scoping keeps state isolated
    # between scenarios while still persisting across the build/boot
    # lifecycle. Documented in the boot docstring.


@contextmanager
def boot_and_verify(
    workspace_dir: Path,
    *,
    image_tag: str | None = None,
    readiness_timeout_s: float = DEFAULT_READINESS_TIMEOUT_S,
    docker_binary: str = "docker",
    build_timeout_s: int = DEFAULT_BUILD_TIMEOUT_S,
    run_timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    capabilities: "CapabilityFlags | None" = None,
) -> Iterator[WorkspaceBootHandle]:
    """Build, run, and health-check a workspace's Dockerfile.

    Parameters
    ----------
    workspace_dir:
        Directory containing the Dockerfile. Passed verbatim to
        ``docker build`` as the build context.
    image_tag:
        Optional explicit tag. When omitted the helper synthesizes a
        deterministic tag derived from ``workspace_dir``.
    readiness_timeout_s:
        How long to wait for ``/health`` to return a non-5xx response.
    docker_binary:
        Override the docker binary (used by tests that don't want the
        real ``docker`` on $PATH).
    capabilities:
        Optional ``CapabilityFlags`` from the course spec. When
        supplied, the helper runs ``_provision_capabilities`` BEFORE
        ``docker build`` and refuses to boot if any requested capability
        isn't pre-provisioned (see ``WorkspaceBootCapabilityError``).
        ``None`` (the default) is the unchanged legacy path — no
        capability checks run.

    Yields
    ------
    WorkspaceBootHandle
        Carries the base_url, container id, and tag. Closing the
        context manager tears down the container.

    Raises
    ------
    WorkspaceBootError
        On build failure, container start failure, or readiness timeout.
        Build/start failures short-circuit before any teardown.
        Readiness timeout DOES tear down the half-started container.
    WorkspaceBootCapabilityError
        When ``capabilities`` requests a primitive the sandbox can't
        provision. Raised BEFORE ``docker build`` so no container is
        ever started.
    """
    _provision_capabilities(capabilities)
    resolved_tag = image_tag or _synthesize_image_tag(workspace_dir)
    _build_image(
        workspace_dir,
        resolved_tag,
        docker_binary=docker_binary,
        build_timeout_s=build_timeout_s,
    )
    host_port = _allocate_port()
    container_port = _detect_container_port(workspace_dir)
    # Provision a per-workspace persistent ``/data`` volume when the
    # spec requests durable state. The host path lives under the
    # workspace dir so it gets torn down with the workspace (and
    # survives across container restarts within a verification cycle).
    data_volume_host_dir: Path | None = None
    if capabilities is not None and getattr(
        capabilities, "durable_state_required", False
    ):
        data_volume_host_dir = workspace_dir / ".coursegen_data"
    container_id = _start_container(
        resolved_tag,
        host_port,
        docker_binary=docker_binary,
        run_timeout_s=run_timeout_s,
        container_port=container_port,
        data_volume_host_dir=data_volume_host_dir,
    )
    base_url = f"http://{HOST}:{host_port}"
    handle = WorkspaceBootHandle(
        base_url=base_url,
        container_id=container_id,
        image_tag=resolved_tag,
    )
    try:
        _poll_health(base_url, readiness_timeout_s=readiness_timeout_s)
        try:
            yield handle
        finally:
            _teardown_container(container_id, docker_binary=docker_binary)
    except WorkspaceBootError:
        # Health timed out before yielding — tear down the half-started
        # container so the failure doesn't leave a Docker resource.
        _teardown_container(container_id, docker_binary=docker_binary)
        raise


# ---------------- Adapters consumed by the outcome graph ----------------


class WorkspaceBootSandboxAdapter:
    """Satisfies ``OraclePass``'s duck-typed sandbox protocol.

    ``OraclePass`` calls ``boot(reference_impl_dir)`` to get a handle
    with a ``base_url``, then ``teardown(handle)`` when done. Our
    ``boot_and_verify`` is a context manager, so this adapter opens the
    context inside ``boot`` and closes it inside ``teardown`` — keyed by
    the handle's container id so ``teardown`` knows which context to
    close.

    This is deliberately a thin pair of methods so the existing
    ``OraclePass.run`` (which wraps ``teardown`` in a ``finally``) needs
    no changes.
    """

    def __init__(
        self,
        *,
        readiness_timeout_s: float = DEFAULT_READINESS_TIMEOUT_S,
        capabilities: "CapabilityFlags | None" = None,
    ) -> None:
        self._readiness_timeout_s = readiness_timeout_s
        self._capabilities = capabilities
        # Map container_id → (image_tag,) so teardown can reach the docker binary.
        self._active: dict[str, str] = {}

    def boot(
        self,
        reference_impl_dir: Path,
        *,
        capabilities: "CapabilityFlags | None" = None,
    ) -> WorkspaceBootHandle:
        # Per-call override beats constructor-pinned flags so the oracle
        # pass can pass the current spec's capabilities explicitly while
        # the existing duck-typed ``boot(dir)`` callers stay working.
        effective = capabilities if capabilities is not None else self._capabilities
        _provision_capabilities(effective)
        resolved_tag = _synthesize_image_tag(reference_impl_dir) + "-" + uuid4().hex[:6]
        _build_image(
            reference_impl_dir,
            resolved_tag,
            docker_binary="docker",
            build_timeout_s=DEFAULT_BUILD_TIMEOUT_S,
        )
        host_port = _allocate_port()
        container_port = _detect_container_port(reference_impl_dir)
        data_volume_host_dir: Path | None = None
        if effective is not None and getattr(
            effective, "durable_state_required", False
        ):
            data_volume_host_dir = reference_impl_dir / ".coursegen_data"
        container_id = _start_container(
            resolved_tag,
            host_port,
            docker_binary="docker",
            run_timeout_s=DEFAULT_RUN_TIMEOUT_S,
            data_volume_host_dir=data_volume_host_dir,
            container_port=container_port,
        )
        base_url = f"http://{HOST}:{host_port}"
        try:
            _poll_health(base_url, readiness_timeout_s=self._readiness_timeout_s)
        except WorkspaceBootError:
            _teardown_container(container_id, docker_binary="docker")
            raise
        self._active[container_id] = resolved_tag
        return WorkspaceBootHandle(
            base_url=base_url,
            container_id=container_id,
            image_tag=resolved_tag,
        )

    def teardown(self, handle: WorkspaceBootHandle) -> None:
        container_id = getattr(handle, "container_id", None)
        if not container_id:
            return
        self._active.pop(container_id, None)
        _teardown_container(container_id, docker_binary="docker")
