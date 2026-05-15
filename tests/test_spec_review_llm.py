"""Tests for the spec-coherence Haiku judge.

Mirrors ``test_public_surface_quality_llm.py``: confirm the verdict
shape, confirm the router is called at Haiku tier with the right
schema, and confirm graceful fallback (None) when the router is
absent or raises.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    QualityBar,
    StarterType,
)
from app.services.spec_review_llm import (
    SpecCoherenceVerdict,
    evaluate_spec_coherence,
)


# ---------------- Helper builders ----------------


def _sample_spec(
    *,
    quality_bars: list[QualityBar] | None = None,
    endpoints: list[EndpointContract] | None = None,
) -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title="Build a Grounded RAG Service",
        goal=(
            "Build a small HTTP service that ingests documents, retrieves "
            "passages for a question, and returns a grounded answer with "
            "citations or abstains."
        ),
        starter_type=StarterType.partial,
        endpoints=endpoints
        or [
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str", "citations": "list"},
                description="Answer the question or abstain.",
            ),
        ],
        quality_bars=quality_bars
        or [
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite passages that support them.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
    )


def _router_returning(verdict: SpecCoherenceVerdict) -> MagicMock:
    router = MagicMock()
    router.parse_structured.return_value = SimpleNamespace(
        parsed=verdict, output_parsed=verdict, usage=None, usage_summary=None
    )
    return router


# ---------------- Verdict shape ----------------


def test_verdict_carries_coherent_rationale_and_concerns() -> None:
    v = SpecCoherenceVerdict(
        is_coherent=False,
        rationale="The faithfulness bar is too generic to measure.",
        concerns=[
            "Quality bar 'general_quality' is generic.",
            "Endpoint /service is too vague.",
        ],
    )
    assert v.is_coherent is False
    assert len(v.concerns) == 2


def test_coherent_verdict_typically_empty_concerns() -> None:
    v = SpecCoherenceVerdict(is_coherent=True, rationale="All bars are specific and endpoints describe concrete shape.")
    assert v.concerns == []


# ---------------- evaluate_spec_coherence behavior ----------------


def test_evaluate_calls_router_with_haiku_tier_and_returns_verdict() -> None:
    expected = SpecCoherenceVerdict(
        is_coherent=True,
        rationale="Faithfulness is concretely measurable; /answer fits the goal.",
    )
    router = _router_returning(expected)
    spec = _sample_spec()

    result = evaluate_spec_coherence(spec=spec, router=router)

    assert isinstance(result, SpecCoherenceVerdict)
    assert result.is_coherent is True
    call = router.parse_structured.call_args
    tier = call.kwargs["tier"]
    assert str(tier) in {"LLMTier.haiku", "haiku"}, f"expected haiku tier, got {tier!r}"
    assert call.kwargs["text_format"] is SpecCoherenceVerdict
    full_prompt = (call.kwargs["system"] + "\n" + call.kwargs["user"]).lower()
    assert "faithfulness" in full_prompt
    assert "/answer" in full_prompt or "answer" in full_prompt


def test_evaluate_returns_none_when_router_is_none() -> None:
    result = evaluate_spec_coherence(spec=_sample_spec(), router=None)
    assert result is None


def test_evaluate_returns_none_when_router_raises() -> None:
    router = MagicMock()
    router.parse_structured.side_effect = RuntimeError("boom")
    result = evaluate_spec_coherence(spec=_sample_spec(), router=router)
    assert result is None


def test_evaluate_serializes_quality_bar_ids_and_endpoint_paths_into_prompt() -> None:
    """The judge needs to see the actual IDs / paths to spot generic bars."""
    expected = SpecCoherenceVerdict(is_coherent=True, rationale="ok")
    router = _router_returning(expected)
    spec = _sample_spec(
        quality_bars=[
            QualityBar(
                id="recall_at_5",
                metric_description="At least 5 of the top-5 hits are gold.",
                threshold=">= 0.7",
                judged_by=JudgeKind.oracle_set_overlap,
                sample_size=10,
            ),
            QualityBar(
                id="abstention_precision",
                metric_description="When abstaining, the system is correct to do so.",
                threshold=">= 0.95",
                judged_by=JudgeKind.llm_haiku,
                sample_size=10,
            ),
        ],
    )
    evaluate_spec_coherence(spec=spec, router=router)
    call = router.parse_structured.call_args
    user = call.kwargs["user"]
    assert "recall_at_5" in user
    assert "abstention_precision" in user
