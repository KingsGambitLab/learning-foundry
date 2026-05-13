"""Anthropic SDK adapter for the LLM router.

Mirrors the shape of `openai_runtime_support` so the router can swap
providers without callsites learning a second SDK. The public surface
is:

- ``resolve_anthropic_env_file`` / ``load_anthropic_env_file`` — same
  file-based key plumbing as the existing OpenAI module.
- ``extract_anthropic_usage`` — maps the Anthropic ``usage`` object onto
  ``AIUsageSummary`` so existing usage logging continues to work.
- ``parse_structured_anthropic_response_with_hard_timeout`` — calls
  ``client.messages.parse(output_format=PydanticModel, ...)`` inside a
  hard-kill subprocess wrapper, mirroring the OpenAI guardrail. Returns
  a ``_ParsedStructuredAnthropicResponse`` carrying the validated
  Pydantic instance and a usage namespace.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from app.domain.ai import AIUsageSummary


DEFAULT_ANTHROPIC_ENV_FILES = (
    Path.home() / "Desktop" / "anthropic.env.keys",
    Path.home() / "anthropic.env.keys",
)


# Local-dev pricing estimate, $/M tokens, in the shape
# (input_per_million, cached_input_per_million, output_per_million).
# Numbers below mirror Anthropic's public list pricing at the time of
# writing — override per-deployment by extending the table.
DEFAULT_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float, float]] = {
    "claude-sonnet-4-6": (3.00, 0.30, 15.00),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
}


# Default Sonnet/Haiku model ids — picked as exact aliases (no date
# suffix) per `anthropic-skills:claude-api`. Rotate by setting
# ANTHROPIC_MODEL_{SONNET,HAIKU} in the env file.
DEFAULT_SONNET_MODEL_ID = "claude-sonnet-4-6"
DEFAULT_HAIKU_MODEL_ID = "claude-haiku-4-5"


class LLMStructuredOutputError(RuntimeError):
    """Raised when the provider returns no parseable structured output
    (refusal, safety stop, max-tokens early cut, etc.). The callsite's
    existing retry loop catches this."""


def resolve_anthropic_env_file(explicit_path: str | None = None) -> str | None:
    candidates: list[str | Path | None] = [
        explicit_path,
        os.environ.get("COURSE_GEN_ANTHROPIC_ENV_FILE"),
        os.environ.get("ANTHROPIC_ENV_FILE"),
        *DEFAULT_ANTHROPIC_ENV_FILES,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    if explicit_path:
        return str(Path(explicit_path).expanduser())
    return None


def load_anthropic_env_file(path: str | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path:
        return env
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = _strip_quotes(value.strip())
    return env


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def extract_anthropic_usage(response: Any, model_id: str | None) -> AIUsageSummary | None:
    """Map Anthropic's usage shape onto the platform's AIUsageSummary."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    # Matches the existing OpenAI summary semantic: total = input + output
    # (cache_read is surfaced separately via cached_input_tokens).
    total_tokens = input_tokens + output_tokens
    estimated_cost_usd = _estimate_anthropic_cost(
        model_id=model_id,
        input_tokens=input_tokens,
        cached_input_tokens=cache_read,
        output_tokens=output_tokens,
    )
    return AIUsageSummary(
        provider="anthropic",
        request_count=1,
        input_tokens=input_tokens,
        cached_input_tokens=cache_read,
        output_tokens=output_tokens,
        reasoning_tokens=0,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        models=[model_id] if model_id else [],
    )


def _estimate_anthropic_cost(
    *,
    model_id: str | None,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float:
    in_rate, cached_rate, out_rate = _pricing_for_model(model_id)
    cost = (
        (input_tokens / 1_000_000.0) * in_rate
        + (cached_input_tokens / 1_000_000.0) * cached_rate
        + (output_tokens / 1_000_000.0) * out_rate
    )
    return round(cost, 6)


def _pricing_for_model(model_id: str | None) -> tuple[float, float, float]:
    if model_id and model_id in DEFAULT_PRICING_PER_1M_TOKENS:
        return DEFAULT_PRICING_PER_1M_TOKENS[model_id]
    # Sensible fallback to Sonnet's pricing for unknown ids.
    return DEFAULT_PRICING_PER_1M_TOKENS["claude-sonnet-4-6"]


def _build_system_blocks(system: str) -> list[dict[str, Any]]:
    """Wrap a system prompt string in a content-block list with prompt
    caching enabled on the last block. The block list shape is what lets
    us attach `cache_control` per the SDK contract."""
    return [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _call_anthropic_messages_parse(
    *,
    client: Any,
    model: str,
    system: str,
    user: str,
    text_format: type[BaseModel],
    request_timeout_s: float,
    max_tokens: int,
    extra_request_kwargs: dict[str, Any] | None = None,
) -> tuple[BaseModel, Any]:
    """One Anthropic structured-output call.

    Tries ``client.messages.parse(output_format=...)`` first. If
    Anthropic rejects the call with the specific
    ``BadRequestError: Schema is too complex.`` (deep Pydantic models
    like ``TaskAgentServiceSpec`` exceed the server-side complexity
    ceiling), falls back to ``client.messages.create(...)`` with the
    schema embedded in the system prompt, and validates the response
    text with Pydantic on our side.

    Raises ``LLMStructuredOutputError`` if neither path produces a
    parseable payload. Other ``BadRequestError`` shapes propagate
    untouched."""
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=_build_system_blocks(system),
            messages=[{"role": "user", "content": user}],
            output_format=text_format,
            thinking={"type": "disabled"},
            timeout=request_timeout_s,
            **(extra_request_kwargs or {}),
        )
    except Exception as exc:
        if _is_schema_too_complex_error(exc):
            return _call_anthropic_messages_create_with_json_fallback(
                client=client,
                model=model,
                system=system,
                user=user,
                text_format=text_format,
                request_timeout_s=request_timeout_s,
                max_tokens=max_tokens,
                extra_request_kwargs=extra_request_kwargs,
            )
        raise
    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise LLMStructuredOutputError(
            "Anthropic structured response returned no parsed payload "
            "(refusal, safety stop, or max-tokens early cut)."
        )
    return parsed, getattr(response, "usage", None)


def _is_schema_too_complex_error(exc: Exception) -> bool:
    """True for the specific Anthropic 400 the server returns when an
    output_format / tool input_schema exceeds the complexity ceiling.

    Detect on message substring rather than class identity so we don't
    care whether the caller saw ``BadRequestError`` or another
    ``APIStatusError`` subclass."""
    try:
        from anthropic import APIStatusError
    except Exception:
        APIStatusError = ()  # type: ignore[assignment]
    if APIStatusError and not isinstance(exc, APIStatusError):  # type: ignore[arg-type]
        return False
    text = str(exc).lower()
    return "schema is too complex" in text or "schema is too complex." in text


_MD_FENCE_PREFIX = "```"


def _unwrap_markdown_fence(text: str) -> str:
    """Strip a wrapping ```json ... ``` fence if present. The
    fallback prompt asks Claude for raw JSON, but the model still
    sometimes wraps it."""
    stripped = text.strip()
    if not stripped.startswith(_MD_FENCE_PREFIX):
        return stripped
    body = stripped[len(_MD_FENCE_PREFIX):]
    # Drop optional language tag on the first line.
    first_newline = body.find("\n")
    if first_newline != -1:
        first_line = body[:first_newline].strip().lower()
        if first_line in {"", "json"} or first_line.isalpha():
            body = body[first_newline + 1:]
    if body.rstrip().endswith(_MD_FENCE_PREFIX):
        body = body.rstrip()[: -len(_MD_FENCE_PREFIX)]
    return body.strip()


def _call_anthropic_messages_create_with_json_fallback(
    *,
    client: Any,
    model: str,
    system: str,
    user: str,
    text_format: type[BaseModel],
    request_timeout_s: float,
    max_tokens: int,
    extra_request_kwargs: dict[str, Any] | None,
) -> tuple[BaseModel, Any]:
    """Last-resort path when ``messages.parse()`` rejects the schema as
    too complex: ask the model for raw JSON inline, then validate with
    Pydantic on our side."""
    schema = text_format.model_json_schema()
    schema_text = json.dumps(schema, indent=2, sort_keys=True)
    fallback_system = (
        system
        + "\n\n"
        + "OUTPUT CONTRACT — your response MUST be a single JSON object that "
        + "validates against the schema below. No prose, no markdown fences, "
        + "no commentary. JSON only.\n\nSchema:\n"
        + schema_text
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_build_system_blocks(fallback_system),
        messages=[{"role": "user", "content": user}],
        thinking={"type": "disabled"},
        timeout=request_timeout_s,
        **(extra_request_kwargs or {}),
    )
    # Extract the first text block from response.content.
    text_payload: str | None = None
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text_payload = getattr(block, "text", None)
            if text_payload:
                break
    if not text_payload:
        raise LLMStructuredOutputError(
            "Anthropic create-fallback returned no text content (refusal, "
            "safety stop, or max-tokens early cut)."
        )
    unwrapped = _unwrap_markdown_fence(text_payload)
    try:
        parsed = text_format.model_validate_json(unwrapped)
    except Exception as exc:
        raise LLMStructuredOutputError(
            f"Anthropic create-fallback produced text that did not parse "
            f"into {text_format.__name__}: {exc}"
        ) from exc
    return parsed, getattr(response, "usage", None)


class _ParsedStructuredAnthropicResponse:
    def __init__(self, *, output_parsed: BaseModel, usage: Any) -> None:
        self.output_parsed = output_parsed
        self.usage = usage


def parse_structured_anthropic_response_with_hard_timeout(
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    system: str,
    user: str,
    text_format: type[BaseModel],
    request_timeout_s: float,
    max_tokens: int,
    extra_request_kwargs: dict[str, Any] | None = None,
) -> _ParsedStructuredAnthropicResponse:
    """Run one structured Anthropic parse call in a killable subprocess.

    Mirrors `openai_runtime_support.parse_structured_openai_response_with_hard_timeout`
    so the LLMRouter can dispatch to either provider with the same
    guardrail: the SDK call lives inside a spawned process, and the
    parent process holds the authoritative wall-clock deadline.
    """
    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_structured_anthropic_parse_worker,
        args=(
            result_queue,
            api_key,
            base_url,
            model,
            system,
            user,
            text_format,
            request_timeout_s,
            max_tokens,
            extra_request_kwargs or {},
        ),
    )
    process.start()
    process.join(request_timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError(
            f"Anthropic structured response exceeded {request_timeout_s:.0f}s and was terminated."
        )
    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(
            f"Anthropic structured response subprocess exited without returning a result (exit_code={process.exitcode})."
        ) from exc
    if not result.get("ok"):
        error_text = result.get("error") or "Anthropic structured response subprocess failed."
        trace = result.get("traceback")
        if trace:
            raise RuntimeError(f"{error_text}\n{trace}")
        raise RuntimeError(error_text)
    parsed_payload = result.get("parsed")
    parsed = text_format.model_validate(parsed_payload)
    return _ParsedStructuredAnthropicResponse(
        output_parsed=parsed,
        usage=_usage_namespace(result.get("usage")),
    )


def _structured_anthropic_parse_worker(
    result_queue,
    api_key: str,
    base_url: str | None,
    model: str,
    system: str,
    user: str,
    text_format: type[BaseModel],
    request_timeout_s: float,
    max_tokens: int,
    extra_request_kwargs: dict[str, Any],
) -> None:
    try:
        from anthropic import Anthropic

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        client = Anthropic(**client_kwargs)
        parsed, usage = _call_anthropic_messages_parse(
            client=client,
            model=model,
            system=system,
            user=user,
            text_format=text_format,
            request_timeout_s=request_timeout_s,
            max_tokens=max_tokens,
            extra_request_kwargs=extra_request_kwargs,
        )
        result_queue.put(
            {
                "ok": True,
                "parsed": parsed.model_dump(mode="json"),
                "usage": _usage_to_plain(usage),
            }
        )
    except Exception as exc:  # pragma: no cover - subprocess path varies by platform/network
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _usage_to_plain(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    }


def _usage_namespace(payload: dict[str, Any] | None) -> Any:
    if payload is None:
        return None
    return SimpleNamespace(**payload)
