from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from app.services.anthropic_runtime_support import (
    LLMStructuredOutputError,
    _call_anthropic_messages_parse,
    extract_anthropic_usage,
    load_anthropic_env_file,
    resolve_anthropic_env_file,
)


# ---------------- env-file helpers ----------------


def test_resolve_anthropic_env_file_explicit_path_wins(tmp_path: Path) -> None:
    p = tmp_path / "explicit.env"
    p.write_text("ANTHROPIC_API_KEY=x\n")
    assert resolve_anthropic_env_file(str(p)) == str(p)


def test_resolve_anthropic_env_file_via_env_var(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "via_env.env"
    p.write_text("ANTHROPIC_API_KEY=x\n")
    monkeypatch.setenv("COURSE_GEN_ANTHROPIC_ENV_FILE", str(p))
    assert resolve_anthropic_env_file() == str(p)


def test_resolve_anthropic_env_file_returns_none_when_no_match(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("COURSE_GEN_ANTHROPIC_ENV_FILE", raising=False)
    monkeypatch.delenv("ANTHROPIC_ENV_FILE", raising=False)
    # Point the default-Desktop fallback at a non-existent path so the test
    # is hermetic regardless of the developer's actual filesystem.
    monkeypatch.setattr("app.services.anthropic_runtime_support.DEFAULT_ANTHROPIC_ENV_FILES", (tmp_path / "nope.env",))
    assert resolve_anthropic_env_file() is None


def test_load_anthropic_env_file_basic(tmp_path: Path) -> None:
    p = tmp_path / "keys.env"
    p.write_text("ANTHROPIC_API_KEY=sk-ant-abc\nANTHROPIC_MODEL_SONNET=claude-sonnet-4-6\n")
    env = load_anthropic_env_file(str(p))
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-abc"
    assert env["ANTHROPIC_MODEL_SONNET"] == "claude-sonnet-4-6"


def test_load_anthropic_env_file_ignores_comments_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / "keys.env"
    p.write_text("# a comment\n\nANTHROPIC_API_KEY=k\n# another\n")
    env = load_anthropic_env_file(str(p))
    assert env == {"ANTHROPIC_API_KEY": "k"}


def test_load_anthropic_env_file_strips_quotes_and_export_prefix(tmp_path: Path) -> None:
    p = tmp_path / "keys.env"
    p.write_text('export ANTHROPIC_API_KEY="sk-ant-quoted"\n')
    env = load_anthropic_env_file(str(p))
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-quoted"


def test_load_anthropic_env_file_missing_returns_empty() -> None:
    assert load_anthropic_env_file(None) == {}
    assert load_anthropic_env_file("/no/such/file.env") == {}


# ---------------- usage extraction ----------------


def test_extract_anthropic_usage_maps_token_counts() -> None:
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=80,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=400,
    )
    response = SimpleNamespace(usage=usage)
    summary = extract_anthropic_usage(response, model_id="claude-sonnet-4-6")
    assert summary is not None
    assert summary.input_tokens == 120
    assert summary.output_tokens == 80
    assert summary.total_tokens == 200
    # Cache-read tokens are surfaced via the cached_input_tokens field
    assert summary.cached_input_tokens == 400


def test_extract_anthropic_usage_none_when_response_has_no_usage() -> None:
    response = SimpleNamespace(usage=None)
    assert extract_anthropic_usage(response, model_id="claude-sonnet-4-6") is None


# ---------------- structured-output call ----------------


class _Echo(BaseModel):
    """Tiny schema for testing the parse path."""
    text: str = Field(description="Echoed text.")
    count: int = Field(description="A small integer.")


def _make_mock_client_with_parsed(parsed_model: BaseModel) -> MagicMock:
    """Build a mock Anthropic client whose messages.parse returns the given
    parsed Pydantic instance."""
    client = MagicMock()
    response = SimpleNamespace(
        parsed_output=parsed_model,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    client.messages.parse.return_value = response
    return client


def test_call_anthropic_messages_parse_returns_validated_pydantic_and_usage() -> None:
    expected = _Echo(text="hello", count=42)
    client = _make_mock_client_with_parsed(expected)

    parsed, usage = _call_anthropic_messages_parse(
        client=client,
        model="claude-sonnet-4-6",
        system="You echo input.",
        user="hello",
        text_format=_Echo,
        request_timeout_s=10.0,
        max_tokens=512,
    )
    assert isinstance(parsed, _Echo)
    assert parsed.text == "hello"
    assert parsed.count == 42
    assert usage.input_tokens == 10
    # Verify the SDK was called with the right call shape
    call = client.messages.parse.call_args
    assert call.kwargs["model"] == "claude-sonnet-4-6"
    assert call.kwargs["output_format"] is _Echo
    assert call.kwargs["max_tokens"] == 512
    assert call.kwargs["messages"] == [{"role": "user", "content": "hello"}]
    # System content should be passed (string or list — must contain our text)
    sys = call.kwargs["system"]
    if isinstance(sys, str):
        assert "You echo input." in sys
    else:
        assert any("You echo input." in (b.get("text") or "") for b in sys)
    # Structured extraction should disable thinking
    assert call.kwargs["thinking"] == {"type": "disabled"}


def test_call_anthropic_messages_parse_raises_when_parsed_output_is_none() -> None:
    client = MagicMock()
    client.messages.parse.return_value = SimpleNamespace(parsed_output=None, usage=None)
    with pytest.raises(LLMStructuredOutputError):
        _call_anthropic_messages_parse(
            client=client,
            model="claude-sonnet-4-6",
            system="s",
            user="u",
            text_format=_Echo,
            request_timeout_s=10.0,
            max_tokens=512,
        )


def test_call_anthropic_messages_parse_enables_prompt_cache_on_system() -> None:
    expected = _Echo(text="cached", count=1)
    client = _make_mock_client_with_parsed(expected)
    _call_anthropic_messages_parse(
        client=client,
        model="claude-sonnet-4-6",
        system="A large stable preamble.",
        user="hello",
        text_format=_Echo,
        request_timeout_s=10.0,
        max_tokens=512,
    )
    call = client.messages.parse.call_args
    sys = call.kwargs["system"]
    # System must be a list of content blocks carrying cache_control on the last block
    assert isinstance(sys, list)
    assert any(
        (b.get("cache_control") == {"type": "ephemeral"}) for b in sys
    ), f"system content blocks should carry cache_control: {sys}"
