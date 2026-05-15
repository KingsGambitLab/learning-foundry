from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.public_surface_quality_llm import (
    DomainGroundingVerdict,
    evaluate_domain_grounding,
)


# ---------------- DomainGroundingVerdict shape ----------------


def test_verdict_carries_grounded_rationale_and_suggestions() -> None:
    v = DomainGroundingVerdict(
        is_grounded=False,
        rationale="The README never names the corpus or grounded-response concepts.",
        suggested_revisions=[
            "Rename 'API endpoints' to 'retrieval-corpus endpoints'.",
            "Refer to outputs as 'grounded responses', not 'JSON results'.",
        ],
    )
    assert v.is_grounded is False
    assert "retrieval-corpus" in v.suggested_revisions[0]


def test_grounded_verdict_typically_has_empty_suggestions() -> None:
    v = DomainGroundingVerdict(is_grounded=True, rationale="README names corpus and grounded answers.")
    assert v.suggested_revisions == []


# ---------------- evaluate_domain_grounding via mocked router ----------------


def _router_returning(verdict: DomainGroundingVerdict) -> MagicMock:
    """Build a mock LLMRouter whose parse_structured returns ``verdict``."""
    router = MagicMock()
    router.parse_structured.return_value = SimpleNamespace(parsed=verdict, output_parsed=verdict, usage=None, usage_summary=None)
    return router


def test_evaluate_calls_router_with_haiku_tier_and_returns_verdict() -> None:
    expected = DomainGroundingVerdict(
        is_grounded=True,
        rationale="The README explicitly references the retrieval corpus and grounded answers.",
    )
    router = _router_returning(expected)

    result = evaluate_domain_grounding(
        content="The retrieval corpus stores chunks; grounded answers cite their sources.",
        entities=["retrieval corpus", "grounded response"],
        system_kind="Grounded retrieval and answer service",
        router=router,
    )

    assert isinstance(result, DomainGroundingVerdict)
    assert result.is_grounded is True
    # Haiku tier — this is a small judgment task, not deep reasoning
    call = router.parse_structured.call_args
    tier = call.kwargs["tier"]
    assert str(tier) in {"LLMTier.haiku", "haiku"}, f"expected haiku tier, got {tier!r}"
    # Schema asked for is DomainGroundingVerdict
    assert call.kwargs["text_format"] is DomainGroundingVerdict
    # Prompt must surface the actual content and the entities
    full_prompt = (call.kwargs["system"] + "\n" + call.kwargs["user"]).lower()
    assert "retrieval corpus" in full_prompt
    assert "grounded response" in full_prompt
    # The README content must be reachable by the model
    assert "retrieval corpus stores chunks" in full_prompt


def test_evaluate_returns_none_when_router_is_none() -> None:
    """No router available (test mode, no API key) → returns None so the
    caller can fall back to the substring check without surfacing an
    artificial failure."""
    result = evaluate_domain_grounding(
        content="anything",
        entities=["retrieval corpus"],
        system_kind=None,
        router=None,
    )
    assert result is None


def test_evaluate_returns_none_when_router_raises() -> None:
    """LLM call raised (network blip, quota, schema-too-complex on the
    judge schema itself, etc.) → returns None so the caller falls back."""
    router = MagicMock()
    router.parse_structured.side_effect = RuntimeError("temporary boom")
    result = evaluate_domain_grounding(
        content="anything",
        entities=["retrieval corpus"],
        system_kind=None,
        router=router,
    )
    assert result is None


def test_evaluate_returns_none_when_entities_empty() -> None:
    """If the spec didn't declare any core_entities, there's nothing to
    judge against — short-circuit before paying for an LLM call."""
    router = MagicMock()
    result = evaluate_domain_grounding(
        content="anything",
        entities=[],
        system_kind=None,
        router=router,
    )
    assert result is None
    router.parse_structured.assert_not_called()


def test_evaluate_pulls_actionable_suggestions_into_verdict() -> None:
    """When the README isn't grounded, the verdict's suggested_revisions
    list MUST be populated — that's the actionable signal we feed to the
    repair LLM via the finding's hint."""
    expected = DomainGroundingVerdict(
        is_grounded=False,
        rationale="The README only uses the literal endpoint plural, not the entity phrase.",
        suggested_revisions=[
            "Introduce 'retrieval corpus' as the canonical singular entity in the intro.",
            "Mention 'grounded response' alongside 'answer' so reviewers see both phrases.",
        ],
    )
    router = _router_returning(expected)
    result = evaluate_domain_grounding(
        content="POST /retrieval-corpuses ingests documents. The endpoint returns a JSON answer.",
        entities=["retrieval corpus", "grounded response"],
        system_kind=None,
        router=router,
    )
    assert result is not None
    assert result.is_grounded is False
    assert len(result.suggested_revisions) == 2
    assert any("retrieval corpus" in s for s in result.suggested_revisions)
