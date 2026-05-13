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
    _reset_too_complex_schema_cache,
    extract_anthropic_usage,
    load_anthropic_env_file,
    resolve_anthropic_env_file,
)


@pytest.fixture(autouse=True)
def _isolate_too_complex_cache():
    """The known-too-complex schema cache lives at module scope so it
    survives the subprocess boundary at runtime. For tests, clear it
    before and after each case so order doesn't matter."""
    _reset_too_complex_schema_cache()
    yield
    _reset_too_complex_schema_cache()


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


# ---------------- "Schema is too complex" fallback path ----------------


def _make_schema_too_complex_error() -> Exception:
    """Build the shape Anthropic returns when the output_format schema
    blows past their server-side complexity ceiling. The SDK raises
    ``anthropic.BadRequestError`` (a subclass of APIStatusError) with the
    message ``Schema is too complex.``."""
    from anthropic import BadRequestError

    request = MagicMock()
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Schema is too complex."},
    }
    # The SDK accepts (message, *, response, body) — pin shape loosely.
    try:
        return BadRequestError(
            message="Schema is too complex.",
            response=response,
            body=body,
        )
    except TypeError:
        return BadRequestError("Schema is too complex.")


def test_call_anthropic_messages_parse_falls_back_to_create_on_schema_too_complex() -> None:
    """When messages.parse() rejects the schema as too complex, the
    helper must retry via messages.create() + prompt-engineered JSON
    output, then validate the response text with Pydantic."""
    schema_err = _make_schema_too_complex_error()

    expected = _Echo(text="from-create", count=7)

    client = MagicMock()
    # messages.parse fails with the complexity error
    client.messages.parse.side_effect = schema_err
    # messages.create returns a text block carrying the JSON we want
    text_block = SimpleNamespace(type="text", text=expected.model_dump_json())
    client.messages.create.return_value = SimpleNamespace(
        content=[text_block],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    parsed, usage = _call_anthropic_messages_parse(
        client=client,
        model="claude-sonnet-4-6",
        system="You are a JSON extractor.",
        user="anything",
        text_format=_Echo,
        request_timeout_s=10.0,
        max_tokens=512,
    )
    assert isinstance(parsed, _Echo)
    assert parsed.text == "from-create"
    assert parsed.count == 7
    # The fallback must have actually invoked messages.create
    assert client.messages.create.called
    # The fallback's system prompt must mention JSON / schema so Claude
    # knows what shape to emit
    create_call = client.messages.create.call_args
    sys_blocks = create_call.kwargs["system"]
    sys_text = " ".join(
        (b.get("text") or "") for b in sys_blocks if isinstance(b, dict)
    ) if isinstance(sys_blocks, list) else str(sys_blocks)
    assert "JSON" in sys_text or "json" in sys_text
    # Usage namespace still surfaces
    assert usage.input_tokens == 10


def test_call_anthropic_messages_parse_raises_on_other_400() -> None:
    """A non-complexity BadRequestError must propagate — we only fall
    back on the specific 'Schema is too complex' signature."""
    from anthropic import BadRequestError

    client = MagicMock()
    other_err = BadRequestError(
        message="model is overloaded",
        response=MagicMock(status_code=400, headers={}),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": "model is overloaded"}},
    )
    client.messages.parse.side_effect = other_err
    with pytest.raises(BadRequestError):
        _call_anthropic_messages_parse(
            client=client,
            model="claude-sonnet-4-6",
            system="s",
            user="u",
            text_format=_Echo,
            request_timeout_s=10.0,
            max_tokens=512,
        )
    # The fallback must NOT have been called
    client.messages.create.assert_not_called()


def test_second_call_with_known_complex_schema_skips_parse_entirely() -> None:
    """Once Anthropic rejects a schema as too complex, every subsequent
    call with the SAME schema must skip ``messages.parse()`` and go
    straight to the create-fallback. Burning ~3 min waiting for a
    deterministic 400 per call is wasteful."""
    from app.services.anthropic_runtime_support import (
        _reset_too_complex_schema_cache,
    )

    _reset_too_complex_schema_cache()  # hermetic test isolation

    schema_err = _make_schema_too_complex_error()
    expected = _Echo(text="cached-fallback", count=9)
    fallback_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=expected.model_dump_json())],
        usage=None,
    )

    # First call: parse() fails, fallback runs.
    client = MagicMock()
    client.messages.parse.side_effect = schema_err
    client.messages.create.return_value = fallback_response
    _call_anthropic_messages_parse(
        client=client,
        model="claude-sonnet-4-6",
        system="s",
        user="u1",
        text_format=_Echo,
        request_timeout_s=10.0,
        max_tokens=512,
    )
    assert client.messages.parse.call_count == 1
    assert client.messages.create.call_count == 1

    # Second call with same schema on a fresh client: parse() MUST be
    # skipped (cache hit on fingerprint), fallback fires directly.
    client2 = MagicMock()
    client2.messages.create.return_value = fallback_response
    parsed, _ = _call_anthropic_messages_parse(
        client=client2,
        model="claude-sonnet-4-6",
        system="s",
        user="u2",
        text_format=_Echo,
        request_timeout_s=10.0,
        max_tokens=512,
    )
    assert parsed.text == "cached-fallback"
    client2.messages.parse.assert_not_called()
    assert client2.messages.create.call_count == 1


def test_cache_is_per_schema_not_global() -> None:
    """A different Pydantic class with a small schema must still go
    through messages.parse() even after a different (deep) schema has
    been marked too-complex."""
    from app.services.anthropic_runtime_support import (
        _reset_too_complex_schema_cache,
    )

    _reset_too_complex_schema_cache()

    class _OtherSchema(BaseModel):
        label: str
        score: int

    # 1) Mark _Echo's schema as too-complex via a real fallback.
    client = MagicMock()
    client.messages.parse.side_effect = _make_schema_too_complex_error()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=_Echo(text="x", count=0).model_dump_json())],
        usage=None,
    )
    _call_anthropic_messages_parse(
        client=client,
        model="claude-sonnet-4-6",
        system="s",
        user="u",
        text_format=_Echo,
        request_timeout_s=10.0,
        max_tokens=512,
    )

    # 2) Now call with _OtherSchema — parse() must still be the first
    # attempt because the cache is per-fingerprint.
    client2 = MagicMock()
    other_parsed = _OtherSchema(label="ok", score=1)
    client2.messages.parse.return_value = SimpleNamespace(parsed_output=other_parsed, usage=None)
    parsed, _ = _call_anthropic_messages_parse(
        client=client2,
        model="claude-sonnet-4-6",
        system="s",
        user="u",
        text_format=_OtherSchema,
        request_timeout_s=10.0,
        max_tokens=512,
    )
    assert isinstance(parsed, _OtherSchema)
    client2.messages.parse.assert_called_once()
    client2.messages.create.assert_not_called()


def test_outer_entry_point_passes_skip_parse_when_schema_cached_in_parent(monkeypatch) -> None:
    """The parent-process cache must be consulted BEFORE we spawn the
    subprocess. When the fingerprint is known-too-complex, the
    subprocess gets ``skip_parse=True`` so the worker bypasses
    ``messages.parse()`` and saves the ~3-minute toll."""
    from app.services import anthropic_runtime_support as ars

    ars._reset_too_complex_schema_cache()
    ars._mark_schema_too_complex(_Echo)

    captured: dict = {}

    def fake_runner(*, skip_parse, **rest):
        captured["skip_parse"] = skip_parse
        captured.update(rest)
        return {
            "ok": True,
            "parsed": _Echo(text="x", count=1).model_dump(mode="json"),
            "usage": None,
            "schema_too_complex_observed": False,
        }

    monkeypatch.setattr(ars, "_run_anthropic_call_in_subprocess", fake_runner)
    ars.parse_structured_anthropic_response_with_hard_timeout(
        api_key="k",
        base_url=None,
        model="claude-sonnet-4-6",
        system="s",
        user="u",
        text_format=_Echo,
        request_timeout_s=5.0,
        max_tokens=512,
    )
    assert captured["skip_parse"] is True


def test_outer_entry_point_does_not_pass_skip_parse_for_new_schema(monkeypatch) -> None:
    from app.services import anthropic_runtime_support as ars

    ars._reset_too_complex_schema_cache()

    captured: dict = {}

    def fake_runner(*, skip_parse, **rest):
        captured["skip_parse"] = skip_parse
        return {
            "ok": True,
            "parsed": _Echo(text="x", count=1).model_dump(mode="json"),
            "usage": None,
            "schema_too_complex_observed": False,
        }

    monkeypatch.setattr(ars, "_run_anthropic_call_in_subprocess", fake_runner)
    ars.parse_structured_anthropic_response_with_hard_timeout(
        api_key="k",
        base_url=None,
        model="claude-sonnet-4-6",
        system="s",
        user="u",
        text_format=_Echo,
        request_timeout_s=5.0,
        max_tokens=512,
    )
    assert captured["skip_parse"] is False


def test_outer_entry_point_marks_cache_when_worker_reports_observation(monkeypatch) -> None:
    """When the worker actually saw the schema-too-complex 400 and
    fell back, the parent must record that schema's fingerprint so the
    NEXT call skips ``parse()`` from the start."""
    from app.services import anthropic_runtime_support as ars

    ars._reset_too_complex_schema_cache()
    assert not ars._is_schema_known_too_complex(_Echo)

    def fake_runner(*, skip_parse, **rest):
        return {
            "ok": True,
            "parsed": _Echo(text="x", count=1).model_dump(mode="json"),
            "usage": None,
            "schema_too_complex_observed": True,
        }

    monkeypatch.setattr(ars, "_run_anthropic_call_in_subprocess", fake_runner)
    ars.parse_structured_anthropic_response_with_hard_timeout(
        api_key="k",
        base_url=None,
        model="claude-sonnet-4-6",
        system="s",
        user="u",
        text_format=_Echo,
        request_timeout_s=5.0,
        max_tokens=512,
    )
    assert ars._is_schema_known_too_complex(_Echo)


def test_call_anthropic_fallback_strips_markdown_code_fences() -> None:
    """messages.create() commonly wraps JSON in ```json ... ``` fences.
    The fallback must unwrap them before Pydantic validation."""
    schema_err = _make_schema_too_complex_error()
    expected = _Echo(text="fenced", count=3)

    client = MagicMock()
    client.messages.parse.side_effect = schema_err
    fenced = "```json\n" + expected.model_dump_json() + "\n```"
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=fenced)],
        usage=None,
    )

    parsed, _ = _call_anthropic_messages_parse(
        client=client,
        model="claude-sonnet-4-6",
        system="s",
        user="u",
        text_format=_Echo,
        request_timeout_s=10.0,
        max_tokens=512,
    )
    assert isinstance(parsed, _Echo)
    assert parsed.text == "fenced"
    assert parsed.count == 3
