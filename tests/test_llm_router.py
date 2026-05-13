from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from app.services.llm_router import (
    LLMProvider,
    LLMRouter,
    LLMRouterConfigError,
    LLMTier,
    resolve_provider_from_env,
)


# ---------------- provider resolution ----------------


def test_resolve_provider_defaults_to_anthropic(monkeypatch) -> None:
    monkeypatch.delenv("COURSE_GEN_LLM_PROVIDER", raising=False)
    assert resolve_provider_from_env() == LLMProvider.anthropic


def test_resolve_provider_respects_env_var(monkeypatch) -> None:
    monkeypatch.setenv("COURSE_GEN_LLM_PROVIDER", "openai")
    assert resolve_provider_from_env() == LLMProvider.openai


def test_resolve_provider_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("COURSE_GEN_LLM_PROVIDER", "ANTHROPIC")
    assert resolve_provider_from_env() == LLMProvider.anthropic


def test_resolve_provider_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv("COURSE_GEN_LLM_PROVIDER", "gemini")
    with pytest.raises(LLMRouterConfigError):
        resolve_provider_from_env()


# ---------------- tier → model id mapping (Anthropic) ----------------


def _write_anthropic_env(tmp_path: Path, **overrides: str) -> Path:
    """Write an anthropic env file with sensible defaults and the user's
    overrides. Returns the file path."""
    base = {"ANTHROPIC_API_KEY": "sk-ant-fake"}
    base.update(overrides)
    p = tmp_path / "anthropic.env.keys"
    p.write_text("\n".join(f"{k}={v}" for k, v in base.items()) + "\n")
    return p


def test_anthropic_tier_uses_default_sonnet_when_env_var_absent(tmp_path: Path) -> None:
    env_path = _write_anthropic_env(tmp_path)
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))
    assert router.model_id_for(LLMTier.sonnet) == "claude-sonnet-4-6"


def test_anthropic_tier_uses_default_haiku_when_env_var_absent(tmp_path: Path) -> None:
    env_path = _write_anthropic_env(tmp_path)
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))
    assert router.model_id_for(LLMTier.haiku) == "claude-haiku-4-5"


def test_anthropic_tier_respects_env_var_override(tmp_path: Path) -> None:
    env_path = _write_anthropic_env(
        tmp_path,
        ANTHROPIC_MODEL_SONNET="claude-sonnet-4-6",
        ANTHROPIC_MODEL_HAIKU="claude-haiku-4-5-20251001",
    )
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))
    assert router.model_id_for(LLMTier.sonnet) == "claude-sonnet-4-6"
    assert router.model_id_for(LLMTier.haiku) == "claude-haiku-4-5-20251001"


# ---------------- structured-output routing (Anthropic) ----------------


class _DummySchema(BaseModel):
    answer: str = Field(description="A short answer.")


def test_router_dispatches_to_anthropic_subprocess(tmp_path: Path) -> None:
    """When provider=anthropic, parse_structured() must call the
    Anthropic subprocess wrapper, not the OpenAI one."""
    env_path = _write_anthropic_env(tmp_path)
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))

    fake_parsed = _DummySchema(answer="42")

    class _FakeResponse:
        def __init__(self, parsed: BaseModel) -> None:
            self.output_parsed = parsed
            self.usage = None

    with patch(
        "app.services.llm_router.parse_structured_anthropic_response_with_hard_timeout",
        return_value=_FakeResponse(fake_parsed),
    ) as anth_mock, patch(
        "app.services.llm_router.parse_structured_openai_response_with_hard_timeout"
    ) as openai_mock:
        result = router.parse_structured(
            tier=LLMTier.sonnet,
            system="You answer questions concisely.",
            user="What is 6 times 7?",
            text_format=_DummySchema,
            request_timeout_s=10.0,
            max_tokens=256,
        )

    assert isinstance(result.parsed, _DummySchema)
    assert result.parsed.answer == "42"
    anth_mock.assert_called_once()
    openai_mock.assert_not_called()
    # Verify the subprocess wrapper received the model id from the env file
    call = anth_mock.call_args
    assert call.kwargs["model"] == "claude-sonnet-4-6"
    assert call.kwargs["api_key"] == "sk-ant-fake"
    assert call.kwargs["text_format"] is _DummySchema


def test_router_dispatches_to_openai_when_provider_overridden(tmp_path: Path) -> None:
    """When provider=openai, parse_structured() must NOT touch Anthropic."""
    anth_env = _write_anthropic_env(tmp_path)
    openai_env = tmp_path / "openai.env.keys"
    openai_env.write_text("OPENAI_API_KEY=sk-fake\nOPENAI_MODEL=gpt-5.4\n")

    router = LLMRouter(
        provider=LLMProvider.openai,
        anthropic_env_file=str(anth_env),
        openai_env_file=str(openai_env),
    )

    class _FakeResponse:
        def __init__(self, parsed: BaseModel) -> None:
            self.output_parsed = parsed
            self.usage = None

    fake_parsed = _DummySchema(answer="ok")
    with patch(
        "app.services.llm_router.parse_structured_openai_response_with_hard_timeout",
        return_value=_FakeResponse(fake_parsed),
    ) as openai_mock, patch(
        "app.services.llm_router.parse_structured_anthropic_response_with_hard_timeout"
    ) as anth_mock:
        result = router.parse_structured(
            tier=LLMTier.sonnet,
            system="s",
            user="u",
            text_format=_DummySchema,
            request_timeout_s=10.0,
            max_tokens=256,
        )
    assert result.parsed.answer == "ok"
    openai_mock.assert_called_once()
    anth_mock.assert_not_called()


# ---------------- status ----------------


def test_router_status_reports_provider_and_models(tmp_path: Path) -> None:
    env_path = _write_anthropic_env(tmp_path)
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))
    status = router.status()
    assert status["provider"] == "anthropic"
    assert status["api_key_present"] is True
    assert status["model_id_sonnet"] == "claude-sonnet-4-6"
    assert status["model_id_haiku"] == "claude-haiku-4-5"
    assert status["sdk_installed"] is True
    # Status shape should remain backward-compatible with the existing endpoint
    assert status["available"] is True
    # `model_id` is preserved (older dashboard code reads it) — set to the sonnet id
    assert status["model_id"] == status["model_id_sonnet"]


def test_router_status_reports_unavailable_when_api_key_missing(tmp_path: Path) -> None:
    p = tmp_path / "empty.env.keys"
    p.write_text("# no key here\n")
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(p))
    status = router.status()
    assert status["available"] is False
    assert status["api_key_present"] is False


# ---------------- tier accepts strings too ----------------


def test_tier_accepts_string_alias(tmp_path: Path) -> None:
    env_path = _write_anthropic_env(tmp_path)
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))
    # Callers commonly pass a plain string; the router must accept it.
    assert router.model_id_for("sonnet") == "claude-sonnet-4-6"
    assert router.model_id_for("haiku") == "claude-haiku-4-5"


def test_tier_rejects_unknown_string(tmp_path: Path) -> None:
    env_path = _write_anthropic_env(tmp_path)
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))
    with pytest.raises(LLMRouterConfigError):
        router.model_id_for("opus")


# ---------------- messages_to_system_user helper ----------------


def test_messages_to_system_user_single_user_only() -> None:
    from app.services.llm_router import messages_to_system_user
    sys, user = messages_to_system_user([{"role": "user", "content": "hello"}])
    assert sys == ""
    assert user == "hello"


def test_messages_to_system_user_system_plus_user() -> None:
    from app.services.llm_router import messages_to_system_user
    sys, user = messages_to_system_user([
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "explain RAG"},
    ])
    assert sys == "you are helpful"
    assert user == "explain RAG"


def test_messages_to_system_user_joins_multiple_systems() -> None:
    from app.services.llm_router import messages_to_system_user
    sys, _ = messages_to_system_user([
        {"role": "system", "content": "rule one"},
        {"role": "system", "content": "rule two"},
        {"role": "user", "content": "go"},
    ])
    assert "rule one" in sys
    assert "rule two" in sys


def test_messages_to_system_user_flattens_content_blocks() -> None:
    from app.services.llm_router import messages_to_system_user
    sys, user = messages_to_system_user([
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "second"},
            ],
        }
    ])
    assert sys == ""
    assert "first" in user and "second" in user


# ---------------- ParsedResult alias ----------------


def test_parsed_result_exposes_output_parsed_alias(tmp_path: Path) -> None:
    """ParsedResult.output_parsed must be an alias of .parsed so existing
    callsites that read response.output_parsed continue to work."""
    from app.services.llm_router import ParsedResult
    schema = _DummySchema(answer="ok")
    r = ParsedResult(parsed=schema, usage=None)
    assert r.output_parsed is schema


# ---------------- usage_summary (per-course cost tracking) ----------------


def test_router_attaches_anthropic_usage_summary_with_anthropic_pricing(tmp_path: Path) -> None:
    """Calls dispatched to Anthropic must return an AIUsageSummary with
    provider='anthropic' and Anthropic Sonnet pricing applied, not the
    OpenAI pricing table that ``extract_openai_usage`` would otherwise
    silently use."""
    env_path = _write_anthropic_env(tmp_path)
    router = LLMRouter(provider=LLMProvider.anthropic, anthropic_env_file=str(env_path))

    from types import SimpleNamespace

    fake_parsed = _DummySchema(answer="42")
    fake_usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=200,
    )
    fake_response = SimpleNamespace(output_parsed=fake_parsed, usage=fake_usage)

    with patch(
        "app.services.llm_router.parse_structured_anthropic_response_with_hard_timeout",
        return_value=fake_response,
    ):
        result = router.parse_structured(
            tier=LLMTier.sonnet,
            system="s",
            user="u",
            text_format=_DummySchema,
            request_timeout_s=10.0,
            max_tokens=256,
        )
    assert result.usage_summary is not None
    assert result.usage_summary.provider == "anthropic"
    assert result.usage_summary.input_tokens == 1000
    assert result.usage_summary.output_tokens == 500
    # Cache read tokens are surfaced as cached_input_tokens
    assert result.usage_summary.cached_input_tokens == 200
    # Cost computed at Anthropic Sonnet rates (3 / 0.3 / 15 per 1M)
    expected = (1000 / 1_000_000.0) * 3.00 + (200 / 1_000_000.0) * 0.30 + (500 / 1_000_000.0) * 15.00
    assert abs(result.usage_summary.estimated_cost_usd - round(expected, 6)) < 1e-6
    assert "claude-sonnet-4-6" in (result.usage_summary.models or [])


def test_router_attaches_openai_usage_summary_when_provider_openai(tmp_path: Path) -> None:
    """When the router dispatches to OpenAI, the AIUsageSummary must
    carry provider='openai' and OpenAI pricing."""
    anth_env = _write_anthropic_env(tmp_path)
    openai_env = tmp_path / "openai.env.keys"
    openai_env.write_text("OPENAI_API_KEY=sk-fake\nOPENAI_MODEL=gpt-5.4\n")
    router = LLMRouter(
        provider=LLMProvider.openai,
        anthropic_env_file=str(anth_env),
        openai_env_file=str(openai_env),
    )

    from types import SimpleNamespace

    fake_parsed = _DummySchema(answer="ok")
    fake_usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    fake_response = SimpleNamespace(output_parsed=fake_parsed, usage=fake_usage)
    with patch(
        "app.services.llm_router.parse_structured_openai_response_with_hard_timeout",
        return_value=fake_response,
    ):
        result = router.parse_structured(
            tier=LLMTier.sonnet,
            system="s",
            user="u",
            text_format=_DummySchema,
            request_timeout_s=10.0,
            max_tokens=256,
        )
    assert result.usage_summary is not None
    assert result.usage_summary.provider == "openai"
    assert result.usage_summary.input_tokens == 1000
    assert result.usage_summary.output_tokens == 500


def test_usage_summary_from_response_prefers_router_attached_summary() -> None:
    """When a router-produced response is passed in, the helper must use
    the pre-computed AIUsageSummary verbatim — NOT re-run
    extract_openai_usage with the wrong provider's pricing."""
    from app.domain.ai import AIUsageSummary
    from app.services.llm_router import (
        ParsedResult,
        usage_summary_from_response,
    )

    pre = AIUsageSummary(
        provider="anthropic",
        request_count=1,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.0105,
        models=["claude-sonnet-4-6"],
    )
    parsed_result = ParsedResult(parsed=_DummySchema(answer="x"), usage=None, usage_summary=pre)
    summary = usage_summary_from_response(parsed_result, model_id="claude-sonnet-4-6")
    assert summary is pre


def test_usage_summary_from_response_falls_back_to_openai_extract_for_raw_sdk_response() -> None:
    """For the test-mode client_factory path (raw SDK response, no
    router involvement), the helper falls back to the existing OpenAI
    extractor so legacy tests continue to work."""
    from types import SimpleNamespace
    from app.services.llm_router import usage_summary_from_response

    raw_response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
    )
    summary = usage_summary_from_response(raw_response, model_id="gpt-5.4")
    assert summary is not None
    assert summary.provider == "openai"
    assert summary.input_tokens == 100
