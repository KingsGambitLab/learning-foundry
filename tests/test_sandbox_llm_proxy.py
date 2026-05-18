"""Tests for the sandbox LLM proxy service.

The proxy runs as a small FastAPI HTTP service on the sandbox internal
Docker network so learner code and the reference implementation can both
make LLM calls through a shared, harness-controlled gateway without ever
seeing the real Anthropic API key. The proxy applies per-submission
rate-limiting to bound cost.

These tests cover the proxy in isolation. Docker network wiring is a
separate follow-up.
"""
from __future__ import annotations

import logging
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.services.sandbox_llm_proxy import (
    _STARTUP_BUCKET_KEY,
    SandboxLLMProxyConfig,
    SandboxLLMRequest,
    SandboxLLMResponse,
    SandboxRateLimiter,
    build_sandbox_llm_proxy_app,
)


# ----- fake router -----


class _FakeAnthropicResponse:
    """Mimics the shape of an anthropic SDK ``Message`` enough for the
    proxy to extract content + usage."""

    def __init__(
        self,
        *,
        text: str = "hi from the fake LLM",
        input_tokens: int = 10,
        output_tokens: int = 5,
        model: str = "claude-haiku-fake",
    ) -> None:
        self.content = [SimpleNamespace(type="text", text=text)]
        self.usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.model = model


class _FakeRouter:
    def __init__(
        self,
        *,
        response: _FakeAnthropicResponse | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.response = response or _FakeAnthropicResponse()
        self.raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    def model_id_for(self, tier: str) -> str:
        return {"haiku": "claude-haiku-fake", "sonnet": "claude-sonnet-fake"}[tier]

    def messages_raw(
        self,
        *,
        tier: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> _FakeAnthropicResponse:
        self.calls.append(
            {
                "tier": tier,
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


# ----- config -----


class SandboxLLMProxyConfigTests(unittest.TestCase):
    def test_defaults_are_sane(self) -> None:
        config = SandboxLLMProxyConfig()
        self.assertEqual(config.max_calls_per_submission, 50)
        self.assertEqual(config.max_tokens_per_call, 2000)
        self.assertEqual(config.allowed_tiers, ["haiku"])
        self.assertIsNone(config.fixed_submission_token)
        # New token-budget fields — sane defaults so tests / dev boot
        # don't choke on missing env vars.
        self.assertEqual(config.max_input_tokens_per_call, 4000)
        self.assertEqual(config.max_tokens_per_submission, 100_000)

    def test_from_env_reads_token_budget_vars(self) -> None:
        # ``COURSEGEN_SANDBOX_MAX_INPUT_TOKENS_PER_CALL`` and
        # ``COURSEGEN_SANDBOX_MAX_TOKENS_PER_SUBMISSION`` are the env
        # contract for the new per-call and per-submission token caps.
        import os

        keys = (
            "COURSEGEN_SANDBOX_MAX_INPUT_TOKENS_PER_CALL",
            "COURSEGEN_SANDBOX_MAX_TOKENS_PER_SUBMISSION",
        )
        old_env = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["COURSEGEN_SANDBOX_MAX_INPUT_TOKENS_PER_CALL"] = "1234"
            os.environ["COURSEGEN_SANDBOX_MAX_TOKENS_PER_SUBMISSION"] = "56789"
            config = SandboxLLMProxyConfig.from_env()
            self.assertEqual(config.max_input_tokens_per_call, 1234)
            self.assertEqual(config.max_tokens_per_submission, 56789)
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_from_env_reads_documented_env_vars(self) -> None:
        # The contract: from_env() reads COURSEGEN_SANDBOX_SUBMISSION_TOKEN,
        # COURSEGEN_SANDBOX_MAX_CALLS, COURSEGEN_SANDBOX_MAX_TOKENS, and
        # COURSEGEN_SANDBOX_ALLOWED_TIERS. The harness sets these via
        # Docker -e flags when spinning up the sandbox sidecar; the proxy
        # reads them once at process start.
        import os

        old_env = {
            k: os.environ.get(k)
            for k in (
                "COURSEGEN_SANDBOX_SUBMISSION_TOKEN",
                "COURSEGEN_SANDBOX_MAX_CALLS",
                "COURSEGEN_SANDBOX_MAX_TOKENS",
                "COURSEGEN_SANDBOX_ALLOWED_TIERS",
            )
        }
        try:
            os.environ["COURSEGEN_SANDBOX_SUBMISSION_TOKEN"] = "harness-token-XYZ"
            os.environ["COURSEGEN_SANDBOX_MAX_CALLS"] = "17"
            os.environ["COURSEGEN_SANDBOX_MAX_TOKENS"] = "777"
            os.environ["COURSEGEN_SANDBOX_ALLOWED_TIERS"] = "haiku,sonnet"
            config = SandboxLLMProxyConfig.from_env()
            self.assertEqual(config.fixed_submission_token, "harness-token-XYZ")
            self.assertEqual(config.max_calls_per_submission, 17)
            self.assertEqual(config.max_tokens_per_call, 777)
            self.assertEqual(config.allowed_tiers, ["haiku", "sonnet"])
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_from_env_falls_back_to_defaults_when_unset(self) -> None:
        import os

        old_env = {
            k: os.environ.get(k)
            for k in (
                "COURSEGEN_SANDBOX_SUBMISSION_TOKEN",
                "COURSEGEN_SANDBOX_MAX_CALLS",
                "COURSEGEN_SANDBOX_MAX_TOKENS",
                "COURSEGEN_SANDBOX_ALLOWED_TIERS",
            )
        }
        try:
            for k in old_env:
                os.environ.pop(k, None)
            config = SandboxLLMProxyConfig.from_env()
            self.assertIsNone(config.fixed_submission_token)
            self.assertEqual(config.max_calls_per_submission, 50)
            self.assertEqual(config.max_tokens_per_call, 2000)
            self.assertEqual(config.allowed_tiers, ["haiku"])
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_request_submission_token_documented_as_ignored(self) -> None:
        # The field is kept in the schema for backwards compatibility,
        # but its description must say it is ignored so downstream callers
        # discover the new contract via the OpenAPI schema.
        schema = SandboxLLMRequest.model_json_schema()
        props = schema["properties"]
        self.assertIn("submission_token", props)
        description = (props["submission_token"].get("description") or "").lower()
        self.assertIn("ignored", description)


# ----- rate limiter -----


class SandboxRateLimiterTests(unittest.TestCase):
    def test_counts_correctly_under_threshold(self) -> None:
        limiter = SandboxRateLimiter(max_calls=5)
        for _ in range(5):
            self.assertTrue(limiter.check_and_increment("submission-A"))

    def test_tokens_used_zero_at_startup(self) -> None:
        limiter = SandboxRateLimiter(max_calls=5)
        self.assertEqual(limiter.tokens_used("submission-A"), 0)
        self.assertEqual(limiter.tokens_used(None), 0)

    def test_add_tokens_accumulates(self) -> None:
        limiter = SandboxRateLimiter(max_calls=5)
        limiter.add_tokens("submission-A", 100)
        limiter.add_tokens("submission-A", 250)
        self.assertEqual(limiter.tokens_used("submission-A"), 350)
        # Other buckets stay isolated.
        self.assertEqual(limiter.tokens_used("submission-B"), 0)

    def test_reset_clears_tokens_and_calls(self) -> None:
        limiter = SandboxRateLimiter(max_calls=2)
        self.assertTrue(limiter.check_and_increment("submission-A"))
        limiter.add_tokens("submission-A", 500)
        self.assertEqual(limiter.tokens_used("submission-A"), 500)
        limiter.reset("submission-A")
        # Both counters cleared.
        self.assertEqual(limiter.tokens_used("submission-A"), 0)
        # And the call-count bucket is also fresh.
        self.assertTrue(limiter.check_and_increment("submission-A"))
        self.assertTrue(limiter.check_and_increment("submission-A"))
        self.assertFalse(limiter.check_and_increment("submission-A"))

    def test_thread_safe_token_accumulation(self) -> None:
        # 10 threads × 10 add_tokens calls of 7 tokens each → 700 total.
        limiter = SandboxRateLimiter(max_calls=10_000)

        def worker() -> None:
            for _ in range(10):
                limiter.add_tokens("contended", 7)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(limiter.tokens_used("contended"), 700)

    def test_blocks_over_threshold(self) -> None:
        limiter = SandboxRateLimiter(max_calls=3)
        for _ in range(3):
            self.assertTrue(limiter.check_and_increment("submission-A"))
        self.assertFalse(limiter.check_and_increment("submission-A"))
        self.assertFalse(limiter.check_and_increment("submission-A"))

    def test_per_submission_isolation(self) -> None:
        limiter = SandboxRateLimiter(max_calls=2)
        self.assertTrue(limiter.check_and_increment("submission-A"))
        self.assertTrue(limiter.check_and_increment("submission-A"))
        # submission-B has its own counter
        self.assertTrue(limiter.check_and_increment("submission-B"))
        self.assertTrue(limiter.check_and_increment("submission-B"))
        # both are now exhausted
        self.assertFalse(limiter.check_and_increment("submission-A"))
        self.assertFalse(limiter.check_and_increment("submission-B"))

    def test_reset_zeros_the_counter(self) -> None:
        limiter = SandboxRateLimiter(max_calls=2)
        self.assertTrue(limiter.check_and_increment("submission-A"))
        self.assertTrue(limiter.check_and_increment("submission-A"))
        self.assertFalse(limiter.check_and_increment("submission-A"))
        limiter.reset("submission-A")
        self.assertTrue(limiter.check_and_increment("submission-A"))

    def test_none_token_uses_default_bucket(self) -> None:
        limiter = SandboxRateLimiter(max_calls=2)
        self.assertTrue(limiter.check_and_increment(None))
        self.assertTrue(limiter.check_and_increment(None))
        self.assertFalse(limiter.check_and_increment(None))

    def test_thread_safe_under_contention(self) -> None:
        # 10 threads each try 100 increments against a cap of 25. Final
        # observed-True count must be exactly the cap.
        limiter = SandboxRateLimiter(max_calls=25)
        successes: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            local = 0
            for _ in range(100):
                if limiter.check_and_increment("contended"):
                    local += 1
            with lock:
                successes.append(local)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(successes), 25)


# ----- HTTP endpoints -----


class SandboxLLMProxyAppTests(unittest.TestCase):
    def _client(
        self,
        *,
        config: SandboxLLMProxyConfig | None = None,
        router: _FakeRouter | None = None,
    ) -> tuple[TestClient, _FakeRouter, SandboxLLMProxyConfig]:
        config = config or SandboxLLMProxyConfig()
        router = router or _FakeRouter()
        app = build_sandbox_llm_proxy_app(config=config, router=router)
        return TestClient(app), router, config

    def test_health_endpoint(self) -> None:
        client, _, _ = self._client()
        resp = client.get("/v1/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_messages_happy_path(self) -> None:
        router = _FakeRouter(
            response=_FakeAnthropicResponse(
                text="answer is 42",
                input_tokens=1000,
                output_tokens=500,
                model="claude-haiku-fake",
            )
        )
        client, router, _ = self._client(router=router)
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "you are a helper",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 256,
                "submission_token": "sub-1",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["content"], "answer is 42")
        self.assertEqual(body["usage"]["input_tokens"], 1000)
        self.assertEqual(body["usage"]["output_tokens"], 500)
        self.assertEqual(body["usage"]["total_tokens"], 1500)
        self.assertEqual(body["model_id"], "claude-haiku-fake")
        # 1000 * 1/1M + 500 * 5/1M = 0.001 + 0.0025 = 0.0035
        self.assertAlmostEqual(body["cost_usd"], 0.0035, places=6)
        # the router saw the correct args
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(router.calls[0]["tier"], "haiku")
        self.assertEqual(router.calls[0]["max_tokens"], 256)

    def test_messages_disallowed_tier_returns_403(self) -> None:
        config = SandboxLLMProxyConfig(allowed_tiers=["haiku"])
        client, _, _ = self._client(config=config)
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "sonnet",
                "system": "s",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json(), {"error": "tier not allowed"})

    def test_messages_max_tokens_over_cap_returns_400(self) -> None:
        config = SandboxLLMProxyConfig(max_tokens_per_call=500)
        client, _, _ = self._client(config=config)
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 501,
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json(), {"error": "max_tokens exceeds cap"})

    def test_messages_rate_limit_returns_429(self) -> None:
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=2,
            fixed_submission_token="sub-1",
        )
        client, _, _ = self._client(config=config)
        payload = {
            "tier": "haiku",
            "system": "s",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        self.assertEqual(client.post("/v1/messages", json=payload).status_code, 200)
        self.assertEqual(client.post("/v1/messages", json=payload).status_code, 200)
        third = client.post("/v1/messages", json=payload)
        self.assertEqual(third.status_code, 429)
        body = third.json()
        self.assertEqual(body["error"], "rate limit exceeded")
        self.assertEqual(body["max_calls"], 2)

    def test_rate_limit_bypass_via_caller_chosen_token_is_blocked(self) -> None:
        # Threat model: a malicious learner mints a fresh random token on
        # every request. Before the fix this kept each request in its own
        # rate-limit bucket and never hit the cap, letting them drain the
        # LLM budget. After the fix, the proxy uses a server-derived
        # bucket key and IGNORES the caller-supplied submission_token, so
        # all 100 attempts share the same bucket.
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=5,
            fixed_submission_token="harness-derived-token",
        )
        client, _, _ = self._client(config=config)
        successes = 0
        for i in range(100):
            resp = client.post(
                "/v1/messages",
                json={
                    "tier": "haiku",
                    "system": "s",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "submission_token": f"forged-{i}-{i * 7}-fresh",
                },
            )
            if resp.status_code == 200:
                successes += 1
        self.assertEqual(successes, 5)

    def test_caller_supplied_submission_token_is_ignored(self) -> None:
        # Regardless of what submission_token the caller supplies, the
        # proxy meters off the server-derived identity. Sending the same
        # 5 requests with 5 different caller-tokens must drain the same
        # bucket, leaving the 6th call rejected.
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=5,
            fixed_submission_token="harness-derived-token",
        )
        client, _, _ = self._client(config=config)
        for tag in ("alice", "bob", "carol", "dave", "eve"):
            resp = client.post(
                "/v1/messages",
                json={
                    "tier": "haiku",
                    "system": "s",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "submission_token": tag,
                },
            )
            self.assertEqual(resp.status_code, 200, resp.text)
        sixth = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "submission_token": "frank",
            },
        )
        self.assertEqual(sixth.status_code, 429)

    def test_missing_fixed_token_falls_back_to_startup_bucket(self) -> None:
        # When the harness didn't set an env var (e.g. dev / test boot),
        # the proxy must NOT silently switch to per-request buckets. It
        # falls back to a single named bucket so the cap still bites.
        config = SandboxLLMProxyConfig(max_calls_per_submission=3)
        # sanity: fixed_submission_token defaults to None
        self.assertIsNone(config.fixed_submission_token)
        client, _, _ = self._client(config=config)
        for _ in range(3):
            resp = client.post(
                "/v1/messages",
                json={
                    "tier": "haiku",
                    "system": "s",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "submission_token": "ignored-anyway",
                },
            )
            self.assertEqual(resp.status_code, 200)
        fourth = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "submission_token": "still-ignored",
            },
        )
        self.assertEqual(fourth.status_code, 429)

    def test_startup_bucket_key_constant_is_used(self) -> None:
        # Defence-in-depth: the well-known fallback bucket has a stable,
        # non-empty name so logs / counters always point somewhere.
        self.assertIsInstance(_STARTUP_BUCKET_KEY, str)
        self.assertTrue(_STARTUP_BUCKET_KEY)

    def test_messages_upstream_failure_returns_502(self) -> None:
        router = _FakeRouter(raise_exc=RuntimeError("the upstream is on fire"))
        client, _, _ = self._client(router=router)
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        self.assertEqual(resp.status_code, 502)
        body = resp.json()
        self.assertIn("upstream LLM call failed", body["error"])
        self.assertIn("the upstream is on fire", body["error"])

    def test_estimated_input_tokens_exceeded(self) -> None:
        # A learner who sends a 100k-char system prompt would burn the
        # budget in a single call. The proxy estimates input tokens via
        # the chars/4 heuristic and rejects with 400 BEFORE forwarding to
        # the router, so the upstream is never called.
        router = _FakeRouter()
        config = SandboxLLMProxyConfig(max_input_tokens_per_call=1000)
        client, router, _ = self._client(config=config, router=router)
        big_system = "x" * 100_000  # ~25_000 estimated input tokens
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": big_system,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertIn("estimated input tokens", body["error"])
        # Crucially: the router never got the call.
        self.assertEqual(router.calls, [])

    def test_cumulative_tokens_block(self) -> None:
        # Five small successful calls each report 1000 input + 1000 output
        # tokens via the fake router → cumulative 10_000 actual tokens.
        # Sixth call estimates an additional ~250 input tokens + 1000
        # max_tokens = 1250, which combined with the 10_000 already used
        # exceeds the 10_000 submission cap → 429 with token-budget error
        # (NOT a call-count rate-limit error, because the call-count cap
        # is generous here).
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=100,
            max_input_tokens_per_call=10_000,
            max_tokens_per_submission=10_000,
            fixed_submission_token="sub-budget",
        )
        router = _FakeRouter(
            response=_FakeAnthropicResponse(
                text="ok",
                input_tokens=1000,
                output_tokens=1000,
                model="claude-haiku-fake",
            )
        )
        client, router, _ = self._client(config=config, router=router)
        payload = {
            "tier": "haiku",
            "system": "s" * 1000,  # ~250 estimated input tokens
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1000,
        }
        for _ in range(5):
            resp = client.post("/v1/messages", json=payload)
            self.assertEqual(resp.status_code, 200, resp.text)
        sixth = client.post("/v1/messages", json=payload)
        self.assertEqual(sixth.status_code, 429)
        body = sixth.json()
        self.assertIn("token budget", body["error"])
        # Confirm the router was NOT called for the sixth attempt.
        self.assertEqual(len(router.calls), 5)

    def test_actual_usage_added_to_bucket_after_success(self) -> None:
        # After a successful call, the limiter's tokens_used must reflect
        # the actual reported input+output usage (so a router that
        # actually charges more than estimated still bites the cap).
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=100,
            max_input_tokens_per_call=10_000,
            max_tokens_per_submission=100_000,
            fixed_submission_token="sub-meter",
        )
        router = _FakeRouter(
            response=_FakeAnthropicResponse(
                text="ok",
                input_tokens=2000,
                output_tokens=1500,
                model="claude-haiku-fake",
            )
        )
        # Build the app and inspect the limiter directly. Easiest path is
        # to attach the limiter to the app for tests; instead we rebuild
        # the app ourselves so we can reach the limiter.
        from app.services.sandbox_llm_proxy import SandboxRateLimiter

        limiter = SandboxRateLimiter(max_calls=config.max_calls_per_submission)
        # Patch the factory to use our limiter by monkey-injecting:
        # we just re-implement the small build using our own limiter.
        from app.services.sandbox_llm_proxy import build_sandbox_llm_proxy_app

        app = build_sandbox_llm_proxy_app(
            config=config, router=router, limiter=limiter
        )
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        # 2000 input + 1500 output = 3500 actual tokens charged.
        self.assertEqual(limiter.tokens_used("sub-meter"), 3500)

    def test_cost_computation_haiku(self) -> None:
        # Direct check via a small standalone request — Haiku 1k input,
        # 500 output → $0.0035.
        router = _FakeRouter(
            response=_FakeAnthropicResponse(
                text="hi",
                input_tokens=1000,
                output_tokens=500,
                model="claude-haiku-fake",
            )
        )
        client, _, _ = self._client(router=router)
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(resp.json()["cost_usd"], 0.0035, places=6)


# ----- request / response models -----


class SandboxLLMRequestResponseTests(unittest.TestCase):
    def test_request_model_validates(self) -> None:
        req = SandboxLLMRequest(
            tier="haiku",
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        self.assertEqual(req.tier, "haiku")
        self.assertIsNone(req.submission_token)

    def test_response_model_round_trips(self) -> None:
        r = SandboxLLMResponse(
            content="hello",
            usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            model_id="m",
            cost_usd=0.0,
        )
        self.assertEqual(r.content, "hello")
        self.assertEqual(r.model_id, "m")


# ----- atomic token reservation (Codex review #4 finding #3) -----
#
# The pre-fix flow read tokens_used, projected, and only AFTER the upstream
# call wrote actual usage back. Two concurrent requests would both pass the
# pre-flight check against the same starting balance, both call upstream,
# both commit — defeating the per-submission cap. The fix introduces an
# atomic reserve / reconcile / release dance on the limiter.


class SandboxRateLimiterReservationTests(unittest.TestCase):
    """The atomic reserve / reconcile / release API.

    These three methods are the building blocks the proxy uses to close
    the parallel-request race. Each takes the limiter lock once and
    operates atomically against the bucket counter.
    """

    def test_reserve_tokens_succeeds_under_cap(self) -> None:
        limiter = SandboxRateLimiter(max_calls=100)
        self.assertTrue(limiter.reserve_tokens("bucket-A", 500, cap=1000))
        self.assertEqual(limiter.tokens_used("bucket-A"), 500)

    def test_reserve_tokens_fails_over_cap_no_state_change(self) -> None:
        limiter = SandboxRateLimiter(max_calls=100)
        limiter.add_tokens("bucket-A", 800)
        self.assertEqual(limiter.tokens_used("bucket-A"), 800)
        # 800 + 300 = 1100 > cap of 1000
        self.assertFalse(limiter.reserve_tokens("bucket-A", 300, cap=1000))
        # No partial state change — bucket is unchanged.
        self.assertEqual(limiter.tokens_used("bucket-A"), 800)

    def test_reserve_fails_atomically(self) -> None:
        # Direct atomic-failure check: reserving an amount that would
        # exceed the cap MUST leave tokens_used untouched.
        limiter = SandboxRateLimiter(max_calls=100)
        limiter.add_tokens("bucket-A", 900)
        before = limiter.tokens_used("bucket-A")
        self.assertFalse(limiter.reserve_tokens("bucket-A", 200, cap=1000))
        self.assertEqual(limiter.tokens_used("bucket-A"), before)

    def test_reconcile_tokens_actual_less_than_reserved_returns_unused(self) -> None:
        limiter = SandboxRateLimiter(max_calls=100)
        limiter.reserve_tokens("bucket-A", 1000, cap=10_000)
        self.assertEqual(limiter.tokens_used("bucket-A"), 1000)
        limiter.reconcile_tokens("bucket-A", reserved=1000, actual=600)
        # 600 actually used; 400 returned to the bucket.
        self.assertEqual(limiter.tokens_used("bucket-A"), 600)

    def test_reconcile_tokens_actual_equals_reserved_no_change(self) -> None:
        limiter = SandboxRateLimiter(max_calls=100)
        limiter.reserve_tokens("bucket-A", 1000, cap=10_000)
        limiter.reconcile_tokens("bucket-A", reserved=1000, actual=1000)
        self.assertEqual(limiter.tokens_used("bucket-A"), 1000)

    def test_reconcile_tokens_actual_greater_than_reserved_logs_warning(
        self,
    ) -> None:
        limiter = SandboxRateLimiter(max_calls=100)
        limiter.reserve_tokens("bucket-A", 1000, cap=10_000)
        with self.assertLogs(
            "app.services.sandbox_llm_proxy", level=logging.WARNING
        ) as cm:
            limiter.reconcile_tokens("bucket-A", reserved=1000, actual=1500)
        # Final bucket reflects actual (1500), not reserved (1000).
        self.assertEqual(limiter.tokens_used("bucket-A"), 1500)
        # And a warning was emitted so operators can investigate.
        self.assertTrue(
            any(
                "exceed" in r.getMessage().lower()
                or "actual" in r.getMessage().lower()
                for r in cm.records
            ),
            cm.output,
        )

    def test_release_reservation_zeros_out_reserved(self) -> None:
        limiter = SandboxRateLimiter(max_calls=100)
        limiter.add_tokens("bucket-A", 200)
        limiter.reserve_tokens("bucket-A", 500, cap=10_000)
        self.assertEqual(limiter.tokens_used("bucket-A"), 700)
        limiter.release_reservation("bucket-A", 500)
        # Pre-reservation value restored.
        self.assertEqual(limiter.tokens_used("bucket-A"), 200)

    def test_reserve_tokens_atomic_under_lock(self) -> None:
        # 100 threads * 5 reservations each. Each reservation is for a
        # small fixed amount. The cap is set so only some succeed; the
        # final used count must be exactly the sum of the successful
        # reservations.
        limiter = SandboxRateLimiter(max_calls=10_000)
        cap = 1000
        per_reservation = 3
        # 100 * 5 = 500 attempts of 3 = max 1500 wanted; cap of 1000 caps
        # successes at floor(1000/3) = 333 successful reservations.
        successes: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            local = 0
            for _ in range(5):
                if limiter.reserve_tokens("contended", per_reservation, cap=cap):
                    local += 1
            with lock:
                successes.append(local)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_successes = sum(successes)
        # Final used count is exactly sum(successful) * per_reservation,
        # never above the cap.
        self.assertEqual(
            limiter.tokens_used("contended"), total_successes * per_reservation
        )
        self.assertLessEqual(limiter.tokens_used("contended"), cap)
        # And we must have hit (or been within one reservation of) the cap.
        self.assertGreaterEqual(
            limiter.tokens_used("contended"), cap - per_reservation + 1
        )


# ----- concurrency regression: parallel requests can't bypass the cap -----


class _SlowFakeRouter:
    """Fake router whose ``messages_raw`` sleeps to widen the race window.

    Wraps the same response shape as ``_FakeAnthropicResponse`` so the
    proxy's extraction logic doesn't have to change.
    """

    def __init__(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        sleep_seconds: float = 0.05,
        gate: threading.Event | None = None,
    ) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._sleep_seconds = sleep_seconds
        self._gate = gate
        self.calls: list[dict[str, Any]] = []
        self._calls_lock = threading.Lock()

    def model_id_for(self, tier: str) -> str:
        return {"haiku": "claude-haiku-fake", "sonnet": "claude-sonnet-fake"}[tier]

    def messages_raw(
        self,
        *,
        tier: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> _FakeAnthropicResponse:
        with self._calls_lock:
            self.calls.append(
                {
                    "tier": tier,
                    "system": system,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
            )
        # Wait until the test releases the gate (so all threads bunch up
        # in the upstream call before any of them gets to commit usage),
        # OR sleep a bit if no gate was wired.
        if self._gate is not None:
            self._gate.wait(timeout=5.0)
        else:
            time.sleep(self._sleep_seconds)
        return _FakeAnthropicResponse(
            text="ok",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model="claude-haiku-fake",
        )


class _FailingRouter:
    """Router that always raises — used to confirm releases on failure."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls: list[dict[str, Any]] = []

    def model_id_for(self, tier: str) -> str:
        return "claude-haiku-fake"

    def messages_raw(self, **_: Any) -> Any:
        self.calls.append(_)
        raise self._exc


class _OvershootRouter:
    """Router that reports MORE actual tokens than max_tokens. Used to
    verify the proxy still meters honestly and emits a warning."""

    def __init__(self, *, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[dict[str, Any]] = []

    def model_id_for(self, tier: str) -> str:
        return "claude-haiku-fake"

    def messages_raw(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.calls.append(kwargs)
        return _FakeAnthropicResponse(
            text="ok",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model="claude-haiku-fake",
        )


class SandboxLLMProxyConcurrencyTests(unittest.TestCase):
    def test_concurrent_requests_respect_token_cap(self) -> None:
        # 20 threads fire requests simultaneously. Each request, if it
        # ran in isolation, would consume ~250 estimated input tokens +
        # 1000 max_tokens = 1250 projected. The cap is 5000 → AT MOST
        # 4 of the 20 requests can fit. The race the fix targets would
        # otherwise let all 20 through.
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=100,
            max_input_tokens_per_call=10_000,
            max_tokens_per_call=2000,
            max_tokens_per_submission=5000,
            fixed_submission_token="sub-race",
        )
        # Gate the router so every thread is parked in upstream until we
        # release them all at once — that maximally widens the race
        # window the test is hunting for.
        gate = threading.Event()
        router = _SlowFakeRouter(
            input_tokens=200,
            output_tokens=400,
            gate=gate,
        )
        limiter = SandboxRateLimiter(max_calls=config.max_calls_per_submission)
        app = build_sandbox_llm_proxy_app(
            config=config, router=router, limiter=limiter
        )
        client = TestClient(app)

        payload = {
            "tier": "haiku",
            "system": "s" * 1000,  # ~250 estimated input tokens
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1000,
        }
        results: list[int] = []
        results_lock = threading.Lock()

        def worker() -> None:
            resp = client.post("/v1/messages", json=payload)
            with results_lock:
                results.append(resp.status_code)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        # Give the threads a moment to all enter upstream / be parked at
        # the gate or be rejected at the cap.
        time.sleep(0.2)
        gate.set()
        for t in threads:
            t.join()

        successes = [code for code in results if code == 200]
        rejections = [code for code in results if code == 429]
        # Every result is either a 200 or a 429 — nothing else.
        self.assertEqual(len(successes) + len(rejections), 20, results)
        # At least one request had to be rejected (the cap bites).
        self.assertGreater(len(rejections), 0, results)
        # Total spend across all completed requests <= cap.
        self.assertLessEqual(
            limiter.tokens_used("sub-race"),
            config.max_tokens_per_submission,
            f"used={limiter.tokens_used('sub-race')} cap="
            f"{config.max_tokens_per_submission}",
        )

    def test_reservation_released_on_upstream_failure(self) -> None:
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=100,
            max_input_tokens_per_call=10_000,
            max_tokens_per_submission=10_000,
            fixed_submission_token="sub-fail",
        )
        router = _FailingRouter(RuntimeError("upstream borked"))
        limiter = SandboxRateLimiter(max_calls=config.max_calls_per_submission)
        app = build_sandbox_llm_proxy_app(
            config=config, router=router, limiter=limiter
        )
        client = TestClient(app)

        before = limiter.tokens_used("sub-fail")
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 500,
            },
        )
        self.assertEqual(resp.status_code, 502, resp.text)
        # Reservation released — no tokens leaked.
        self.assertEqual(limiter.tokens_used("sub-fail"), before)

    def test_reservation_reconciled_when_actual_less_than_projected(self) -> None:
        # Projected = ~200 estimated input (800 char system / 4) + 800
        # max_tokens = 1000. Actual = 200 input + 400 output = 600. Final
        # tokens_used must be 600, not 1000.
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=100,
            max_input_tokens_per_call=10_000,
            max_tokens_per_submission=10_000,
            max_tokens_per_call=2000,
            fixed_submission_token="sub-reconcile",
        )
        router = _SlowFakeRouter(input_tokens=200, output_tokens=400, sleep_seconds=0)
        limiter = SandboxRateLimiter(max_calls=config.max_calls_per_submission)
        app = build_sandbox_llm_proxy_app(
            config=config, router=router, limiter=limiter
        )
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "tier": "haiku",
                "system": "s" * 800,  # ~200 estimated input tokens
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 800,
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        # Final reflects actual (600), not projected (1000).
        self.assertEqual(limiter.tokens_used("sub-reconcile"), 600)

    def test_reservation_over_projected_logs_warning_and_commits_actual(
        self,
    ) -> None:
        # Defensive: the router shouldn't return more output than
        # max_tokens, but if it does, the proxy must meter honestly
        # against the bucket AND log a warning so operators can dig in.
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=100,
            max_input_tokens_per_call=10_000,
            max_tokens_per_submission=100_000,
            max_tokens_per_call=2000,
            fixed_submission_token="sub-overshoot",
        )
        # estimated input ~0 (tiny system) + max_tokens 100 = projected 100.
        # Router reports 200 input + 500 output = 700 actual >> 100.
        router = _OvershootRouter(input_tokens=200, output_tokens=500)
        limiter = SandboxRateLimiter(max_calls=config.max_calls_per_submission)
        app = build_sandbox_llm_proxy_app(
            config=config, router=router, limiter=limiter
        )
        client = TestClient(app)
        with self.assertLogs(
            "app.services.sandbox_llm_proxy", level=logging.WARNING
        ) as cm:
            resp = client.post(
                "/v1/messages",
                json={
                    "tier": "haiku",
                    "system": "s",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        # Bucket gets the actual value (700), not the projection (100).
        self.assertEqual(limiter.tokens_used("sub-overshoot"), 700)
        self.assertTrue(
            any(
                "exceed" in r.getMessage().lower()
                or "actual" in r.getMessage().lower()
                for r in cm.records
            ),
            cm.output,
        )

    def test_call_count_limit_releases_token_reservation(self) -> None:
        # The call-count check sits AFTER the atomic token reservation in
        # the new flow. If the count cap trips, the reservation must be
        # released so the token bucket doesn't leak.
        config = SandboxLLMProxyConfig(
            max_calls_per_submission=2,
            max_input_tokens_per_call=10_000,
            max_tokens_per_submission=100_000,
            max_tokens_per_call=2000,
            fixed_submission_token="sub-callcap",
        )
        router = _SlowFakeRouter(input_tokens=100, output_tokens=100, sleep_seconds=0)
        limiter = SandboxRateLimiter(max_calls=config.max_calls_per_submission)
        app = build_sandbox_llm_proxy_app(
            config=config, router=router, limiter=limiter
        )
        client = TestClient(app)
        payload = {
            "tier": "haiku",
            "system": "s",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 500,
        }
        # Burn the call-count budget with two successful calls.
        self.assertEqual(client.post("/v1/messages", json=payload).status_code, 200)
        self.assertEqual(client.post("/v1/messages", json=payload).status_code, 200)
        tokens_before_third = limiter.tokens_used("sub-callcap")
        # Third call must 429 on call-count; reservation taken in step 3
        # of the pipeline must be released in the 429 branch.
        resp = client.post("/v1/messages", json=payload)
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.json()["error"], "rate limit exceeded")
        # No token leak from the released reservation.
        self.assertEqual(limiter.tokens_used("sub-callcap"), tokens_before_third)


if __name__ == "__main__":
    unittest.main()
