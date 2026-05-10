from __future__ import annotations

import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.domain.ai import AIUsageSummary

DEFAULT_OPENAI_ENV_FILES = (
    Path.home() / "Desktop" / "openai.env.keys",
    Path.home() / "openai.env.keys",
)

# Local-dev estimate only. Override these with env vars if you want billing-grade numbers.
DEFAULT_PRICING_PER_1M_TOKENS: dict[str, tuple[float, float, float]] = {
    "gpt-5.4": (1.25, 0.125, 10.0),
}


def resolve_openai_env_file(explicit_path: str | None = None) -> str | None:
    candidates: list[str | Path | None] = [
        explicit_path,
        os.environ.get("COURSE_GEN_OPENAI_ENV_FILE"),
        os.environ.get("OPENAI_ENV_FILE"),
        *DEFAULT_OPENAI_ENV_FILES,
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


def load_openai_env_file(path: str | None) -> dict[str, str]:
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
        env[key.strip()] = strip_quotes(value.strip())
    return env


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def extract_openai_usage(response: Any, model_id: str | None) -> AIUsageSummary | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    cached_input_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
    reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)
    estimated_cost_usd = estimate_openai_cost(
        model_id=model_id,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )
    models = [model_id] if model_id else []
    return AIUsageSummary(
        request_count=1,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        models=models,
    )


def estimate_openai_cost(
    *,
    model_id: str | None,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float:
    input_rate, cached_input_rate, output_rate = _pricing_for_model(model_id)
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    estimated = (
        (uncached_input_tokens / 1_000_000) * input_rate
        + (cached_input_tokens / 1_000_000) * cached_input_rate
        + (output_tokens / 1_000_000) * output_rate
    )
    return round(estimated, 6)


def _pricing_for_model(model_id: str | None) -> tuple[float, float, float]:
    input_override = os.environ.get("COURSE_GEN_OPENAI_INPUT_PRICE_PER_1M_TOKENS")
    cached_override = os.environ.get("COURSE_GEN_OPENAI_CACHED_INPUT_PRICE_PER_1M_TOKENS")
    output_override = os.environ.get("COURSE_GEN_OPENAI_OUTPUT_PRICE_PER_1M_TOKENS")
    if input_override and output_override:
        return (
            float(input_override),
            float(cached_override or input_override),
            float(output_override),
        )

    normalized = (model_id or "gpt-5.4").strip().lower()
    for model_prefix, pricing in DEFAULT_PRICING_PER_1M_TOKENS.items():
        if normalized == model_prefix or normalized.startswith(f"{model_prefix}-"):
            return pricing
    return DEFAULT_PRICING_PER_1M_TOKENS["gpt-5.4"]


def parse_structured_openai_response_with_hard_timeout(
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    input: Any,
    text_format: type[Any],
    request_timeout_s: float,
    extra_request_kwargs: dict[str, Any] | None = None,
):
    """Run one structured OpenAI parse call in a killable subprocess.

    The SDK timeout is still passed through, but the subprocess boundary is the
    authoritative wall-clock deadline. If the SDK or network stack wedges, we
    terminate the child process and surface a clean timeout to the workflow.
    """

    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_structured_openai_parse_worker,
        args=(
            result_queue,
            api_key,
            base_url,
            model,
            input,
            text_format,
            request_timeout_s,
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
            f"OpenAI structured response exceeded {request_timeout_s:.0f}s and was terminated."
        )
    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(
            f"OpenAI structured response subprocess exited without returning a result (exit_code={process.exitcode})."
        ) from exc
    if not result.get("ok"):
        error_text = result.get("error") or "OpenAI structured response subprocess failed."
        trace = result.get("traceback")
        if trace:
            raise RuntimeError(f"{error_text}\n{trace}")
        raise RuntimeError(error_text)
    parsed_payload = result.get("parsed")
    parsed = text_format.model_validate(parsed_payload)
    return _ParsedStructuredOpenAIResponse(
        output_parsed=parsed,
        usage=_usage_namespace(result.get("usage")),
    )


class _ParsedStructuredOpenAIResponse:
    def __init__(self, *, output_parsed: Any, usage: Any) -> None:
        self.output_parsed = output_parsed
        self.usage = usage


def _structured_openai_parse_worker(
    result_queue,
    api_key: str,
    base_url: str | None,
    model: str,
    input: Any,
    text_format: type[Any],
    request_timeout_s: float,
    extra_request_kwargs: dict[str, Any],
) -> None:
    try:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        response = client.responses.parse(
            model=model,
            input=input,
            text_format=text_format,
            timeout=request_timeout_s,
            **extra_request_kwargs,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI structured response returned no parsed payload.")
        result_queue.put(
            {
                "ok": True,
                "parsed": parsed.model_dump(mode="json"),
                "usage": _usage_to_plain(getattr(response, "usage", None)),
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
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
    }


def _usage_namespace(payload: dict[str, Any] | None) -> Any:
    if payload is None:
        return None
    return SimpleNamespace(
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        total_tokens=int(payload.get("total_tokens", 0) or 0),
        input_tokens_details=SimpleNamespace(
            cached_tokens=int(payload.get("cached_tokens", 0) or 0),
        ),
        output_tokens_details=SimpleNamespace(
            reasoning_tokens=int(payload.get("reasoning_tokens", 0) or 0),
        ),
    )
