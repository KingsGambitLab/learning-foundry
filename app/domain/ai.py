from __future__ import annotations

from pydantic import BaseModel, Field


class AIUsageSummary(BaseModel):
    provider: str = "openai"
    request_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    models: list[str] = Field(default_factory=list)


def merge_ai_usage(*summaries: AIUsageSummary | None) -> AIUsageSummary:
    merged = AIUsageSummary()
    seen_models: set[str] = set()
    for summary in summaries:
        if summary is None:
            continue
        merged.request_count += summary.request_count
        merged.input_tokens += summary.input_tokens
        merged.cached_input_tokens += summary.cached_input_tokens
        merged.output_tokens += summary.output_tokens
        merged.reasoning_tokens += summary.reasoning_tokens
        merged.total_tokens += summary.total_tokens
        merged.estimated_cost_usd = round(merged.estimated_cost_usd + summary.estimated_cost_usd, 6)
        for model in summary.models:
            if model and model not in seen_models:
                seen_models.add(model)
                merged.models.append(model)
    return merged
