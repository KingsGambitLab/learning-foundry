"""Sandbox LLM proxy — harness-managed HTTP gateway to Anthropic.

A tiny FastAPI service that runs on the sandbox internal Docker network
(``http://coursegen-llm:8080`` by default). Both the learner submission
and the reference implementation reach it via plain HTTP; the proxy
itself holds the real Anthropic credentials and dispatches calls through
the platform's :class:`LLMRouter`.

The network boundary is the auth: nothing outside the sandbox Docker
network can reach this service. Inside the sandbox, anything can call it
— but the proxy applies per-submission rate-limits to bound cost so a
runaway learner script can't drain the budget.

Rate-limit identity is server-derived. The harness sets
``COURSEGEN_SANDBOX_SUBMISSION_TOKEN`` (and friends) on the proxy
container at startup; the proxy reads it once via
:meth:`SandboxLLMProxyConfig.from_env` and uses it as the bucket key for
every request. The caller-supplied ``submission_token`` on
:class:`SandboxLLMRequest` is IGNORED — a malicious learner cannot
rotate it to escape the per-submission cap.

Docker sidecar wiring is a separate follow-up; this module is the proxy
in isolation.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


# ----- env-var contract -----
#
# The harness sets these via Docker ``-e`` flags on the proxy container.
# Names live here as constants so the test suite and the harness wiring
# code share one source of truth.
ENV_SUBMISSION_TOKEN = "COURSEGEN_SANDBOX_SUBMISSION_TOKEN"
ENV_MAX_CALLS = "COURSEGEN_SANDBOX_MAX_CALLS"
ENV_MAX_TOKENS = "COURSEGEN_SANDBOX_MAX_TOKENS"
ENV_ALLOWED_TIERS = "COURSEGEN_SANDBOX_ALLOWED_TIERS"
ENV_MAX_INPUT_TOKENS_PER_CALL = "COURSEGEN_SANDBOX_MAX_INPUT_TOKENS_PER_CALL"
ENV_MAX_TOKENS_PER_SUBMISSION = "COURSEGEN_SANDBOX_MAX_TOKENS_PER_SUBMISSION"


# ----- pricing -----
# Per-million-token prices in USD. Kept local to the proxy because the
# proxy is what writes the cost number back to the caller; the rest of
# the platform reads cost out of the AIUsageSummary path.
_PRICING_PER_MILLION = {
    "haiku": {"input": 1.0, "output": 5.0},
    "sonnet": {"input": 3.0, "output": 15.0},
}


def _compute_cost_usd(tier: str, input_tokens: int, output_tokens: int) -> float:
    rates = _PRICING_PER_MILLION.get(tier)
    if rates is None:
        return 0.0
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0


# ----- config -----


class SandboxLLMProxyConfig(BaseModel):
    """Knobs that bound proxy cost and surface area.

    - ``max_calls_per_submission`` caps the total LLM calls a single
      grading run can make. The harness scopes one proxy container per
      submission so this counter is the per-run budget.
    - ``max_tokens_per_call`` is the hard ceiling on each individual
      ``max_tokens``. Requests above this get a 400 — they don't silently
      get clamped, because silent clamping makes test failures confusing.
    - ``allowed_tiers`` defaults to Haiku-only so runaway loops in
      learner code don't reach for Sonnet.
    - ``fixed_submission_token`` is the **server-derived** rate-limit
      bucket key. The harness injects it via the
      ``COURSEGEN_SANDBOX_SUBMISSION_TOKEN`` env var at proxy startup;
      tests can also pass it directly. It is NEVER pulled from request
      bodies — see :class:`SandboxLLMRequest` for the threat model.
    - ``max_input_tokens_per_call`` is a rough heuristic ceiling on
      prompt size. The proxy estimates input tokens as
      ``(len(system) + sum(len(content) for content in messages)) // 4``
      and rejects the call with 400 if the estimate exceeds this cap.
      The chars/4 heuristic is intentional: it doesn't require a real
      tokenizer, and a learner who tries to smuggle a megaprompt past
      it would still bite the per-submission token budget once the
      router reports actual usage.
    - ``max_tokens_per_submission`` is the cumulative cap on
      ``input_tokens + output_tokens`` across every call in the bucket.
      The check uses the estimated input tokens + the requested
      ``max_tokens`` for the pre-flight projection, and accumulates the
      router's actual reported usage after success.
    """

    max_calls_per_submission: int = 50
    max_tokens_per_call: int = 2000
    max_input_tokens_per_call: int = 4000
    max_tokens_per_submission: int = 100_000
    allowed_tiers: list[Literal["haiku", "sonnet"]] = Field(default_factory=lambda: ["haiku"])
    fixed_submission_token: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SandboxLLMProxyConfig":
        """Read the harness-supplied env vars at proxy startup.

        Env-var contract (set by the harness in the Docker ``-e`` flags):

        - ``COURSEGEN_SANDBOX_SUBMISSION_TOKEN``: opaque per-submission
          identifier. Used as the rate-limit bucket key. If unset the
          proxy falls back to a single named startup bucket — the cap
          still bites, but every request shares one counter.
        - ``COURSEGEN_SANDBOX_MAX_CALLS``: integer, defaults to 50.
        - ``COURSEGEN_SANDBOX_MAX_TOKENS``: integer, defaults to 2000.
        - ``COURSEGEN_SANDBOX_ALLOWED_TIERS``: comma-separated list
          (e.g. ``"haiku"`` or ``"haiku,sonnet"``); defaults to
          ``["haiku"]``.

        Threat model: the Docker daemon controls who can set env vars on
        a container. Learner / reference code runs in a separate process
        (and typically a separate container from the proxy sidecar) and
        cannot read or rewrite the proxy's env. Hence env-var-derived
        identity is trustworthy in a way that request-body-derived
        identity is not.
        """
        source = env if env is not None else os.environ
        max_calls_raw = source.get(ENV_MAX_CALLS)
        max_tokens_raw = source.get(ENV_MAX_TOKENS)
        tiers_raw = source.get(ENV_ALLOWED_TIERS)
        max_input_tokens_raw = source.get(ENV_MAX_INPUT_TOKENS_PER_CALL)
        max_total_tokens_raw = source.get(ENV_MAX_TOKENS_PER_SUBMISSION)

        kwargs: dict[str, Any] = {}
        token = source.get(ENV_SUBMISSION_TOKEN)
        if token:
            kwargs["fixed_submission_token"] = token
        if max_calls_raw is not None and max_calls_raw.strip():
            kwargs["max_calls_per_submission"] = int(max_calls_raw)
        if max_tokens_raw is not None and max_tokens_raw.strip():
            kwargs["max_tokens_per_call"] = int(max_tokens_raw)
        if max_input_tokens_raw is not None and max_input_tokens_raw.strip():
            kwargs["max_input_tokens_per_call"] = int(max_input_tokens_raw)
        if max_total_tokens_raw is not None and max_total_tokens_raw.strip():
            kwargs["max_tokens_per_submission"] = int(max_total_tokens_raw)
        if tiers_raw is not None and tiers_raw.strip():
            kwargs["allowed_tiers"] = [
                t.strip() for t in tiers_raw.split(",") if t.strip()
            ]
        return cls(**kwargs)


# ----- rate limiter -----


_DEFAULT_BUCKET_KEY = "__default__"
# Stable bucket key the proxy uses when the harness didn't supply a
# server-derived token. Distinct from ``_DEFAULT_BUCKET_KEY`` so logs can
# tell "called with None at the limiter API" apart from "proxy was booted
# without a harness token".
_STARTUP_BUCKET_KEY = "__startup__"


class SandboxRateLimiter:
    """In-memory per-submission call + token counters.

    Threaded sandbox workers may hit the proxy concurrently, so all
    mutations are guarded by a single lock. The implementation is
    intentionally tiny: two dicts (``submission_token -> count`` and
    ``submission_token -> tokens``) plus a ``threading.Lock``. No expiry
    — the harness is expected to tear down the proxy with the sandbox,
    so the dicts' lifetime is bounded by the grading run.

    The two counters live on the same lock and the same bucket key so
    a single ``reset()`` clears both. The token counter is read /
    written by the handler around each successful router call: it's a
    cumulative cap on the actual usage the router reports, so a learner
    can't drain the budget by sending a few huge prompts under the
    call-count cap.
    """

    def __init__(self, *, max_calls: int) -> None:
        self._max_calls = max_calls
        self._counts: dict[str, int] = {}
        self._tokens: dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(submission_token: str | None) -> str:
        return submission_token if submission_token else _DEFAULT_BUCKET_KEY

    def check_and_increment_calls(self, submission_token: str | None) -> bool:
        key = self._key(submission_token)
        with self._lock:
            current = self._counts.get(key, 0)
            if current >= self._max_calls:
                return False
            self._counts[key] = current + 1
            return True

    # Backwards-compatible alias — older call sites and the existing test
    # suite still use ``check_and_increment``.
    def check_and_increment(self, submission_token: str | None) -> bool:
        return self.check_and_increment_calls(submission_token)

    def add_tokens(self, submission_token: str | None, tokens: int) -> None:
        if tokens <= 0:
            return
        key = self._key(submission_token)
        with self._lock:
            self._tokens[key] = self._tokens.get(key, 0) + tokens

    def tokens_used(self, submission_token: str | None) -> int:
        key = self._key(submission_token)
        with self._lock:
            return self._tokens.get(key, 0)

    def reserve_tokens(
        self, submission_token: str | None, projected_tokens: int, cap: int
    ) -> bool:
        """Atomically check ``used + projected <= cap`` and reserve.

        If the projection fits, increment the bucket by ``projected_tokens``
        and return ``True``. Otherwise leave the bucket untouched and
        return ``False``.

        This is the primitive that closes the parallel-request race: two
        concurrent callers cannot both pass the check against the same
        starting balance, because the check and the increment happen under
        the same lock.

        ``projected_tokens <= 0`` is a no-op that returns ``True`` (an
        empty reservation always fits and changes nothing).
        """
        if projected_tokens <= 0:
            return True
        key = self._key(submission_token)
        with self._lock:
            current = self._tokens.get(key, 0)
            if current + projected_tokens > cap:
                return False
            self._tokens[key] = current + projected_tokens
            return True

    def reconcile_tokens(
        self, submission_token: str | None, reserved: int, actual: int
    ) -> None:
        """Adjust a prior reservation by ``actual - reserved``.

        Called after the upstream succeeds. When ``actual < reserved`` the
        unused portion of the reservation is returned to the bucket; when
        ``actual > reserved`` (shouldn't happen if the projection was
        honest — the router would have to return more output tokens than
        ``max_tokens`` allowed) the bucket gets the real usage and a
        warning is logged for follow-up.
        """
        key = self._key(submission_token)
        delta = actual - reserved
        with self._lock:
            current = self._tokens.get(key, 0)
            new = current + delta
            if new < 0:
                # Defensive floor: a caller that double-reconciles or
                # passes a bogus pair could otherwise drive the bucket
                # negative. Clamp at zero so future reservations behave.
                new = 0
            self._tokens[key] = new
        if actual > reserved:
            _log.warning(
                "sandbox_llm_proxy: actual token usage %d exceeded "
                "reservation of %d for bucket %r — metering the actual "
                "value but the projection was wrong",
                actual,
                reserved,
                key,
            )

    def release_reservation(
        self, submission_token: str | None, reserved: int
    ) -> None:
        """Return a full reservation to the bucket.

        Called when the upstream call failed (no usage was incurred) or
        when a downstream check (e.g. call-count cap) rejects the request
        after the reservation was taken. Equivalent to
        ``reconcile_tokens(submission_token, reserved, 0)`` but spelled
        out as its own method so the call sites read clearly.
        """
        if reserved <= 0:
            return
        key = self._key(submission_token)
        with self._lock:
            current = self._tokens.get(key, 0)
            new = current - reserved
            if new < 0:
                new = 0
            self._tokens[key] = new

    def reset(self, submission_token: str | None) -> None:
        key = self._key(submission_token)
        with self._lock:
            self._counts.pop(key, None)
            self._tokens.pop(key, None)


# ----- request / response shapes -----


class SandboxLLMRequest(BaseModel):
    tier: Literal["haiku", "sonnet"]
    system: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    submission_token: str | None = Field(
        default=None,
        description=(
            "Ignored; retained for backwards compatibility only. "
            "Rate limiting uses server-derived identity from the "
            "COURSEGEN_SANDBOX_SUBMISSION_TOKEN env var that the harness "
            "sets on the proxy container at startup. Any value supplied "
            "by the caller is logged as a warning and discarded so a "
            "malicious learner cannot rotate this field to escape the "
            "per-submission rate-limit cap."
        ),
    )


class SandboxLLMResponse(BaseModel):
    content: str
    usage: dict[str, int]
    model_id: str
    cost_usd: float


# ----- LLM call helpers -----


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off either an attribute-style object (SDK / namespace)
    or a dict (test fixture / pre-decoded payload)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_text_from_first_block(response: Any) -> str:
    for block in _attr(response, "content") or []:
        if _attr(block, "type") == "text":
            text = _attr(block, "text")
            if text:
                return str(text)
    return ""


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = _attr(response, "usage")
    if usage is None:
        return 0, 0
    return int(_attr(usage, "input_tokens") or 0), int(_attr(usage, "output_tokens") or 0)


def _resolve_model_id(response: Any, router: Any, tier: str) -> str:
    model = _attr(response, "model")
    if model:
        return str(model)
    lookup = getattr(router, "model_id_for", None)
    if callable(lookup):
        try:
            return str(lookup(tier))
        except Exception:
            return ""
    return ""


# ----- app factory -----


def _estimate_input_tokens(system: str, messages: list[dict[str, Any]]) -> int:
    """Rough chars/4 heuristic for input-token count.

    The real Anthropic tokenizer is non-trivial; the proxy doesn't need
    precision, just an upper-bound guardrail. We sum the system prompt
    plus every message ``content`` field. A ``content`` that is itself a
    list of content blocks (Anthropic's multimodal shape) is handled by
    summing the ``text`` of each text block and treating non-text blocks
    as a fixed per-block character allowance — better to slightly
    over-count than to give a free pass to a multipart payload.
    """
    total_chars = len(system or "")
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        total_chars += len(text)
                    else:
                        # Non-text block (image, tool result, etc.):
                        # charge a fixed surcharge so the cap still
                        # bites multipart payloads.
                        total_chars += 256
                elif isinstance(block, str):
                    total_chars += len(block)
    return total_chars // 4


def build_sandbox_llm_proxy_app(
    *,
    config: SandboxLLMProxyConfig,
    router: Any,
    limiter: SandboxRateLimiter | None = None,
) -> FastAPI:
    """Construct a FastAPI app that proxies LLM calls through ``router``.

    ``router`` is duck-typed: it must expose ``messages_raw(tier, system,
    messages, max_tokens)`` returning an Anthropic-shaped response, and
    optionally ``model_id_for(tier)`` so the proxy can fill in
    ``model_id`` when the response doesn't carry one (test fakes often
    don't bother). The real :class:`app.services.llm_router.LLMRouter`
    integration is a follow-up — that wiring is intentionally out of
    scope here so the proxy stays trivially testable.

    ``limiter`` may be supplied for tests that need to inspect the bucket
    after handling a request; in production it defaults to a fresh
    in-process :class:`SandboxRateLimiter`.
    """
    app = FastAPI()
    if limiter is None:
        limiter = SandboxRateLimiter(max_calls=config.max_calls_per_submission)

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/messages")
    def messages(request: SandboxLLMRequest) -> Any:
        # tier allow-list — strict 403 so the proxy refuses to even ask
        # the router for a disallowed tier.
        if request.tier not in config.allowed_tiers:
            return JSONResponse(status_code=403, content={"error": "tier not allowed"})

        # cap individual call size — refuse rather than clamp so the
        # caller learns about the cap immediately.
        if request.max_tokens > config.max_tokens_per_call:
            return JSONResponse(status_code=400, content={"error": "max_tokens exceeds cap"})

        # Estimate input-token size via the chars/4 heuristic and refuse
        # the call if a single prompt would blow the per-call cap. This
        # blocks the "few huge prompts" bypass that would otherwise let a
        # learner stay under ``max_calls_per_submission`` while still
        # draining the budget.
        #
        # 400 is the right status here: it's a client-side validation
        # failure on the payload (the request itself is malformed for
        # the proxy's contract). 413 is tempting but is HTTP-level
        # body-size semantics; this is a token-budget check, not a wire
        # size check. 429 belongs to the cumulative bucket check below.
        estimated_input_tokens = _estimate_input_tokens(
            request.system, request.messages
        )
        if estimated_input_tokens > config.max_input_tokens_per_call:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"estimated input tokens {estimated_input_tokens} "
                        f"exceeds per-call cap {config.max_input_tokens_per_call}"
                    ),
                },
            )

        # Per-submission budget. The rate-limit bucket key is
        # SERVER-DERIVED: it comes from the harness-set
        # ``COURSEGEN_SANDBOX_SUBMISSION_TOKEN`` env var (surfaced as
        # ``config.fixed_submission_token``). The caller-supplied
        # ``request.submission_token`` is deliberately ignored — a
        # malicious learner could otherwise mint a fresh random token on
        # every call and never hit the cap. When the harness didn't set
        # a token (early dev / tests), fall back to a single named
        # startup bucket so the cap still bites.
        if request.submission_token is not None:
            _log.warning(
                "sandbox_llm_proxy: ignoring caller-supplied submission_token; "
                "rate limiting uses server-derived identity",
            )
        token = config.fixed_submission_token or _STARTUP_BUCKET_KEY

        # Cumulative token cap — ATOMIC reservation.
        #
        # The naive flow ("read tokens_used, project, if-over-cap-429,
        # then call upstream, then add_tokens") is racy: two parallel
        # requests both pass the pre-flight check against the same
        # starting balance, both call upstream, both commit, and the
        # combined spend blows the cap. Codex review #4 finding #3.
        #
        # The fix: reserve the projected tokens BEFORE the upstream
        # call, atomically under the limiter lock. On success reconcile
        # the reservation against the router's reported actual usage; on
        # failure (or on a downstream check that rejects the request)
        # release the reservation so the bucket doesn't leak.
        #
        # Reservation may slightly over-commit (we reserve ``max_tokens``
        # worth of output even when actual output is smaller) but it
        # will NEVER under-commit, which is the safety property we need.
        projected = estimated_input_tokens + request.max_tokens
        if not limiter.reserve_tokens(
            token, projected, config.max_tokens_per_submission
        ):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "submission token budget would be exceeded",
                    "max_tokens_per_submission": config.max_tokens_per_submission,
                    "tokens_used": limiter.tokens_used(token),
                },
            )

        # Call-count check runs AFTER token reservation. If it trips, the
        # reservation must be released so the token bucket doesn't leak.
        if not limiter.check_and_increment_calls(token):
            limiter.release_reservation(token, projected)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate limit exceeded",
                    "max_calls": config.max_calls_per_submission,
                },
            )

        try:
            response = router.messages_raw(
                tier=request.tier,
                system=request.system,
                messages=request.messages,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            # Upstream borked — no actual usage was incurred. Return the
            # reservation to the bucket so it can be used by a later
            # request.
            limiter.release_reservation(token, projected)
            return JSONResponse(
                status_code=502,
                content={"error": f"upstream LLM call failed: {exc}"},
            )

        content_text = _extract_text_from_first_block(response)
        input_tokens, output_tokens = _extract_usage(response)
        # Reconcile the reservation against actual reported usage. When
        # actual < reserved (the common case — we reserved the worst-case
        # max_tokens for output), the unused portion is returned to the
        # bucket. When actual > reserved (shouldn't happen, but
        # defensive), the bucket gets the real usage and a warning is
        # logged from the limiter.
        actual = input_tokens + output_tokens
        limiter.reconcile_tokens(token, reserved=projected, actual=actual)
        cost_usd = _compute_cost_usd(request.tier, input_tokens, output_tokens)
        model_id = _resolve_model_id(response, router, request.tier)

        payload = SandboxLLMResponse(
            content=content_text,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            model_id=model_id,
            cost_usd=cost_usd,
        )
        return payload

    return app
