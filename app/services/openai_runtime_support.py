from __future__ import annotations

import os
from pathlib import Path
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
