"""LLM judge that screens a CourseOutcomeSpec for coherence.

The substring-based ``OutcomeCoursePlanner`` validates structural
invariants (unique IDs, hint references, non-empty endpoints). It does
NOT decide whether endpoints are coherent with the goal, whether
quality bars are measurable, or whether IDs are specific. This module
adds a small Haiku judge that reads the spec and decides exactly that,
returning a list of concrete concerns when the spec is not coherent.

Callers should treat ``None`` as "judge unavailable, accept the spec"
so offline / no-API-key environments keep working.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.services.course_outcome_models import CourseOutcomeSpec

__all__ = [
    "SpecCoherenceVerdict",
    "evaluate_spec_coherence",
]


_JUDGE_SYSTEM_PROMPT = (
    "You judge whether a single-outcome course spec is coherent. A spec "
    "is coherent when:\n"
    "  - every HTTP endpoint plausibly fits the stated goal,\n"
    "  - every quality bar is specific and measurable (no generic ids "
    "    like 'general_quality' or 'correctness_check'),\n"
    "  - the listed quality bars together cover the outcome the goal "
    "    describes — at minimum a schema/contract bar plus at least one "
    "    semantic bar (faithfulness / accuracy / overlap / etc.) when "
    "    the goal implies semantic correctness.\n\n"
    "When the spec is NOT coherent, list 1-5 short, copy-pasteable "
    "concerns that name the exact field at fault. Concerns must be "
    "specific to this spec — not generic advice."
)


class SpecCoherenceVerdict(BaseModel):
    """Judge verdict on whether a CourseOutcomeSpec is coherent."""

    is_coherent: bool = Field(
        ...,
        description="True when endpoints + quality bars fit the goal.",
    )
    rationale: str = Field(
        ...,
        description="One sentence explaining the verdict.",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, copy-pasteable concerns the planner should fix. "
            "Empty when is_coherent=True. 1-5 items otherwise."
        ),
    )


def _spec_to_prompt_payload(spec: CourseOutcomeSpec) -> dict[str, Any]:
    return {
        "title": spec.title,
        "goal": spec.goal,
        "starter_type": spec.starter_type.value,
        "endpoints": [
            {
                "method": ep.method.value,
                "path": ep.path,
                "description": ep.description,
                "request_schema": ep.request_schema,
                "response_schema": ep.response_schema,
            }
            for ep in spec.endpoints
        ],
        "quality_bars": [
            {
                "id": bar.id,
                "metric_description": bar.metric_description,
                "threshold": bar.threshold,
                "judged_by": bar.judged_by.value,
                "sample_size": bar.sample_size,
            }
            for bar in spec.quality_bars
        ],
        "learning_path": [
            {"on_metric_fail": h.on_metric_fail, "hint": h.hint}
            for h in spec.learning_path
        ],
    }


def evaluate_spec_coherence(
    *,
    spec: CourseOutcomeSpec,
    router: Any = None,
) -> SpecCoherenceVerdict | None:
    """Ask a Haiku judge whether ``spec`` is internally coherent.

    Returns ``None`` when the judge is unavailable so the caller can
    fall back to accepting the spec:

    - ``router`` is None (no provider configured, test mode), or
    - the LLM call raises.
    """
    if router is None:
        return None

    payload = _spec_to_prompt_payload(spec)
    user_prompt = (
        "Judge the following CourseOutcomeSpec for coherence.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nReturn your verdict as a SpecCoherenceVerdict."
    )

    try:
        from app.services.llm_router import LLMTier

        result = router.parse_structured(
            tier=LLMTier.haiku,
            system=_JUDGE_SYSTEM_PROMPT,
            user=user_prompt,
            text_format=SpecCoherenceVerdict,
            request_timeout_s=60.0,
            max_tokens=1024,
        )
    except Exception:
        return None

    parsed = getattr(result, "parsed", None) or getattr(result, "output_parsed", None)
    if not isinstance(parsed, SpecCoherenceVerdict):
        return None
    return parsed
