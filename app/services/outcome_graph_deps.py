"""Production adapters that satisfy the ``OutcomeGraphDeps`` protocol.

The outcome graph (``langgraph_outcome_graph.py``) drives spec →
starter → grader → publish via a small set of injected collaborators:

* ``repo_author`` — must provide
  ``generate_bundle(spec, failure_context=None) -> list[tuple[str, str]]``
  where each tuple is ``(relative_path, content)``.
* ``starter_verifier`` — must provide
  ``verify_starter(workspace_dir) -> dict`` returning at minimum
  ``{"ok": bool}`` plus optional ``stage`` / ``error`` / ``logs`` keys.
* ``oracle_author`` — real ``OracleAuthor`` from ``oracle_authoring``.
* ``oracle_pass`` — real ``OraclePass`` from ``oracle_pass``.

What's wired (post-Wave 5e Agent A)
-----------------------------------

* ``oracle_author`` and ``oracle_pass`` are wired as real production
  classes — they accept ``CourseOutcomeSpec`` directly.
* ``repo_author`` is wired via ``OutcomeRepoAuthorAdapter``
  (``outcome_repo_author_adapter.py``), which calls
  ``OpenAIStarterRepoAuthoringService._generate_bundle`` directly with
  a payload synthesized from the spec. Strategy A from the wave 5e
  brief — bypasses the legacy ``WorkflowRun`` / on-disk manifest
  surface entirely.
* ``starter_verifier`` is a ``RealStarterVerifier`` that wraps
  ``workspace_boot.boot_and_verify``. The verifier builds the
  starter's Dockerfile, runs the container on a free local port, polls
  ``/health``, and reports back via the same ``{"ok": bool, ...}``
  contract the graph expects.
* ``oracle_pass.sandbox_runner`` is a ``WorkspaceBootSandboxAdapter``
  that satisfies the duck-typed ``boot(dir) -> handle`` / ``teardown(handle)``
  protocol the oracle pass already consumes.
* ``router`` is the default LLM router. Spec review and oracle author
  both consult it via ``parse_structured``.

The two ``Placeholder*`` classes are kept here as historical fallbacks
for tests that want to disable the real boot path without mocking
Docker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.course_outcome_models import CapabilityFlags, CourseOutcomeSpec


__all__ = [
    "OutcomeRepoAuthorAdapter",
    "RealStarterVerifier",
    "PlaceholderStarterVerifier",
    "PlaceholderReferenceImplSandbox",
    "build_production_outcome_deps",
]


# Re-export the new dedicated adapter so existing imports continue to work.
from app.services.outcome_repo_author_adapter import (  # noqa: E402
    OutcomeRepoAuthorAdapter,
)


class RealStarterVerifier:
    """Production starter verifier — boots the starter via ``workspace_boot``.

    The graph's ``node_starter_verify`` writes the authored starter
    files under ``<workspace_root>/public/starter/`` and then calls
    ``verify_starter(<that dir>)``. We build the Dockerfile inside that
    directory, run the resulting container on a free loopback port,
    and consider the starter "verified" when ``/health`` returns a
    non-5xx response.

    Failures are translated to the
    ``{"ok": False, "stage": ..., "error": ..., "logs": ...}`` contract
    the graph already understands so the starter_repair loop runs
    cleanly. A successful boot returns
    ``{"ok": True, "base_url": ..., "stage": "boot"}``.
    """

    def __init__(
        self,
        *,
        readiness_timeout_s: float | None = None,
        capabilities: CapabilityFlags | None = None,
    ) -> None:
        self._readiness_timeout_s = readiness_timeout_s
        self._capabilities = capabilities

    def verify_starter(
        self,
        starter_dir: Path,
        *,
        capabilities: CapabilityFlags | None = None,
    ) -> dict[str, Any]:
        # Lazy import keeps the docker-subprocess surface out of the
        # import graph for tests that never invoke the verifier.
        from app.services.workspace_boot import (
            WorkspaceBootError,
            boot_and_verify,
        )

        kwargs: dict[str, Any] = {}
        if self._readiness_timeout_s is not None:
            kwargs["readiness_timeout_s"] = self._readiness_timeout_s
        # Per-call ``capabilities`` (passed in by the graph node from
        # ``state.spec.capabilities``) wins over the constructor default.
        effective_caps = capabilities if capabilities is not None else self._capabilities
        if effective_caps is not None:
            kwargs["capabilities"] = effective_caps
        try:
            with boot_and_verify(Path(starter_dir), **kwargs) as handle:
                return {
                    "ok": True,
                    "stage": "boot",
                    "base_url": handle.base_url,
                    "container_id": handle.container_id,
                    "image_tag": handle.image_tag,
                }
        except WorkspaceBootError as exc:
            return {
                "ok": False,
                "stage": _classify_boot_failure_stage(str(exc)),
                "logs": "",
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            # Any other unexpected failure (e.g. docker binary missing,
            # permission issues) is reported in the same shape so the
            # graph doesn't crash. ``stage="boot"`` is the safe default.
            return {
                "ok": False,
                "stage": "boot",
                "logs": "",
                "error": f"unexpected verifier failure: {type(exc).__name__}: {exc}",
            }


def _classify_boot_failure_stage(message: str) -> str:
    """Best-effort stage classification from the WorkspaceBootError text.

    The graph uses ``stage`` to thread the failure into the right
    repair bucket — distinguishing a build failure from a runtime/boot
    failure helps the LLM author the right kind of fix.
    """
    lowered = message.lower()
    # Capability failures are a distinct stage so blocking reasons can
    # name the missing capability instead of looking like a generic
    # boot timeout.
    if "cannot provision capability" in lowered:
        return "capability"
    if "docker build" in lowered:
        return "build"
    if "docker run" in lowered:
        return "start"
    if "/health" in lowered or "health" in lowered:
        return "boot"
    return "boot"


class PlaceholderReferenceImplSandbox:
    """Legacy placeholder — kept for tests that don't want real Docker boots."""

    PLACEHOLDER_MESSAGE = "reference-impl sandbox boot not yet wired in production"

    def boot(
        self,
        reference_impl_dir: Path,
        *,
        capabilities: CapabilityFlags | None = None,
    ) -> Any:
        del reference_impl_dir
        del capabilities
        raise RuntimeError(self.PLACEHOLDER_MESSAGE)

    def teardown(self, handle: Any) -> None:
        del handle


class PlaceholderStarterVerifier:
    """Legacy placeholder — kept for tests that don't want real Docker boots."""

    PLACEHOLDER_ERROR = "starter verification not yet wired in production"

    def verify_starter(
        self,
        starter_dir: Path,
        *,
        capabilities: CapabilityFlags | None = None,
    ) -> dict[str, Any]:
        del starter_dir
        del capabilities
        return {
            "ok": False,
            "stage": "build",
            "logs": "",
            "error": self.PLACEHOLDER_ERROR,
        }


# ---------------- production builder ----------------


_UNSET: Any = object()


def build_production_outcome_deps(
    *,
    planner: Any,
    router: Any = _UNSET,
    repo_author: Any | None = None,
    starter_verifier: Any | None = None,
    oracle_author: Any | None = None,
    oracle_pass: Any | None = None,
) -> Any:
    """Construct an ``OutcomeGraphDeps`` with every collaborator wired.

    Real wiring for every slot:

    * ``repo_author``: ``OutcomeRepoAuthorAdapter`` wrapping
      ``OpenAIStarterRepoAuthoringService._generate_bundle``.
    * ``starter_verifier``: ``RealStarterVerifier`` that boots the
      starter Dockerfile via ``workspace_boot.boot_and_verify``.
    * ``oracle_author`` / ``oracle_pass``: real production classes.
    * ``oracle_pass``'s sandbox runner is the
      ``WorkspaceBootSandboxAdapter`` that satisfies the duck-typed
      ``boot(dir) -> handle`` / ``teardown(handle)`` protocol.

    Callers may override any slot via keyword args (tests pass fakes
    through this seam to exercise the resume-with-real-deps contract
    without booting Docker).

    ``router`` uses a sentinel default so an explicit ``router=None``
    (the test path that disables LLM calls) is preserved verbatim
    rather than re-resolved to the default router.
    """
    # Local imports keep the LLM router / graph types out of the
    # import graph for callers that never flip the outcome flag.
    from app.services.langgraph_outcome_graph import OutcomeGraphDeps
    from app.services.oracle_authoring import OracleAuthor
    from app.services.oracle_pass import OraclePass
    from app.services.workspace_boot import WorkspaceBootSandboxAdapter

    if router is _UNSET:
        try:
            from app.services.llm_router import get_default_router

            router = get_default_router()
        except Exception:
            # The router is consulted only by spec_review +
            # oracle_authoring; if it's unavailable we proceed without
            # it (those nodes degrade gracefully — see
            # ``node_spec_review``'s ``if deps.router is not None``).
            router = None

    return OutcomeGraphDeps(
        planner=planner,
        router=router,
        repo_author=repo_author or OutcomeRepoAuthorAdapter(),
        starter_verifier=starter_verifier or RealStarterVerifier(),
        oracle_author=oracle_author or OracleAuthor(router=router),
        oracle_pass=oracle_pass
        or OraclePass(sandbox_runner=WorkspaceBootSandboxAdapter()),
    )
