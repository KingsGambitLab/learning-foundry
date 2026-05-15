"""Tests for the ``LLMJudgeCoverage`` scenario rubric.

These tests pin the contract:

- Without an injected router the rubric MUST abstain — offline / no-API-
  key environments cannot have judge calls block grading.
- When a router is wired in, the rubric MUST call
  ``router.parse_structured`` at the Haiku tier with its own Pydantic
  verdict schema, route the parsed verdict into a ``Verdict``, and add a
  ``cost_usd`` line item derived from the returned ``usage_summary``.
- ``strict`` mode rejects any missing fact; ``lenient`` defers to the
  LLM's own pass/fail field.
- The rubric never raises on a learner-side bug (missing target path,
  judge raised) — it encodes those in the returned ``Verdict``.

The tests use a hand-rolled FakeRouter to avoid touching Anthropic / the
real ``LLMRouter`` machinery — same shape as ``ParsedResult``
(``.parsed``, ``.usage``, ``.usage_summary``).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from app.domain.ai import AIUsageSummary
from app.services.scenario_rubrics_base import RUBRIC_REGISTRY, RubricContext
from app.services.scenario_rubrics_llm import (
    LLMJudgeCoverage,
    LLMJudgeCoverageVerdict,
    LLMJudgeFalsePremise,
    LLMJudgeFalsePremiseVerdict,
    LLMJudgeSemanticEq,
    SemanticEqVerdict,
)


# ---------------- FakeRouter ----------------


@dataclass
class _FakeResult:
    """ParsedResult-shaped object the router test double returns."""

    parsed: Any
    usage: Any = None
    usage_summary: AIUsageSummary | None = None

    @property
    def output_parsed(self) -> Any:
        return self.parsed


class _FakeRouter:
    """Minimal LLMRouter stand-in for tests.

    Records the last ``parse_structured`` call so tests can assert on the
    tier, schema, system / user prompts; returns whatever was loaded via
    ``set_response``. ``set_exception`` makes the next call raise.
    """

    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None
        self._response: _FakeResult | None = None
        self._exception: Exception | None = None

    def set_response(self, result: _FakeResult) -> None:
        self._response = result
        self._exception = None

    def set_exception(self, exc: Exception) -> None:
        self._exception = exc
        self._response = None

    def parse_structured(self, **kwargs: Any) -> _FakeResult:
        self.last_call = kwargs
        if self._exception is not None:
            raise self._exception
        assert self._response is not None, "FakeRouter response not set"
        return self._response


def _usage(input_tokens: int, output_tokens: int) -> AIUsageSummary:
    """Build a usage summary with Haiku-shaped fields. ``estimated_cost_usd``
    is intentionally left at 0 so the rubric is forced to compute the
    cost line item from the token counts itself."""
    return AIUsageSummary(
        provider="anthropic",
        request_count=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        models=["claude-haiku-4-5"],
    )


# ---------------- Registration ----------------


def test_llm_judge_coverage_is_registered() -> None:
    assert "llm_judge_coverage" in RUBRIC_REGISTRY
    assert RUBRIC_REGISTRY["llm_judge_coverage"] is LLMJudgeCoverage


# ---------------- Abstain when no router ----------------


def test_abstains_when_router_is_none() -> None:
    """Without a router (test mode, no API key configured) the rubric
    must abstain — judge availability is never allowed to block grading."""
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF", "abstention"],
        router=None,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "anything"}},
    )
    out = r.judge(ctx)
    assert out.status == "abstain"
    assert "no LLM router" in out.rationale.lower() or "judge unavailable" in out.rationale.lower()
    assert out.cost_usd == 0.0


# ---------------- Happy path: pass ----------------


def test_passes_when_judge_returns_pass_verdict() -> None:
    verdict = LLMJudgeCoverageVerdict(
        covered_facts=["RRF", "abstention"],
        missing_facts=[],
        hallucinated_claims=[],
        verdict="pass",
        rationale="Answer covers both RRF and abstention semantics.",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(800, 200))
    )
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF", "abstention"],
        router=router,
    )
    ctx = RubricContext(
        captures={
            "answer_response": {
                "answer": "We blend BM25 and dense via RRF and abstain on low scores."
            },
        },
    )
    out = r.judge(ctx)
    assert out.status == "pass"
    assert "RRF" in out.rationale or "abstention" in out.rationale or "covers" in out.rationale.lower()
    # diagnostic surfaces the structured judge fields for the feedback synth
    assert out.diagnostic["covered_facts"] == ["RRF", "abstention"]
    assert out.diagnostic["missing_facts"] == []
    # cost computed from tokens (Haiku: $1/M input, $5/M output)
    expected_cost = (800 / 1_000_000) * 1.0 + (200 / 1_000_000) * 5.0
    assert out.cost_usd == pytest.approx(expected_cost, rel=1e-6)


# ---------------- Fail path with missing facts ----------------


def test_fails_when_judge_returns_fail_verdict_with_missing_facts() -> None:
    verdict = LLMJudgeCoverageVerdict(
        covered_facts=["RRF"],
        missing_facts=["abstention"],
        hallucinated_claims=[],
        verdict="fail",
        rationale="Answer never explains the abstention threshold.",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(900, 150))
    )
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF", "abstention"],
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "We use RRF."}},
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert out.diagnostic["missing_facts"] == ["abstention"]
    assert out.diagnostic["covered_facts"] == ["RRF"]
    # cost still computed even on fail
    expected_cost = (900 / 1_000_000) * 1.0 + (150 / 1_000_000) * 5.0
    assert out.cost_usd == pytest.approx(expected_cost, rel=1e-6)


# ---------------- Fail open when judge raises ----------------


def test_abstains_when_judge_call_raises() -> None:
    """Anthropic returned an error, schema rejection, network blip, etc.
    Judge availability must never block grading — abstain so the other
    rubrics still produce a verdict."""
    router = _FakeRouter()
    router.set_exception(RuntimeError("transient anthropic 5xx"))
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF"],
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "anything"}},
    )
    out = r.judge(ctx)
    assert out.status == "abstain"
    assert "judge failed" in out.rationale.lower() or "transient" in out.rationale.lower()
    assert out.cost_usd == 0.0


# ---------------- Missing target path = learner-side fail ----------------


def test_fails_with_missing_path_diagnostic_when_target_absent() -> None:
    """A missing dotted-path target is the learner's bug — the answer
    field they were supposed to populate isn't there. That's a FAIL with
    a missing_path diagnostic, not a judge issue."""
    router = _FakeRouter()
    # router.set_response intentionally NOT called: the rubric must short-
    # circuit before paying for an LLM call.
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF"],
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {}},  # answer key missing
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert out.diagnostic["missing_path"] == "answer_response.answer"
    assert out.cost_usd == 0.0
    assert router.last_call is None  # never called the LLM


# ---------------- Strict mode ----------------


def test_strict_mode_fails_when_any_fact_missing_even_if_llm_says_pass() -> None:
    """In strict mode the rubric overrides the LLM's pass verdict if
    missing_facts is non-empty. The LLM can be lenient; the rubric is
    not."""
    verdict = LLMJudgeCoverageVerdict(
        covered_facts=["RRF"],
        missing_facts=["abstention"],
        hallucinated_claims=[],
        verdict="pass",  # LLM was generous
        rationale="Close enough.",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(500, 100))
    )
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF", "abstention"],
        strictness="strict",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "We use RRF."}},
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert out.diagnostic["missing_facts"] == ["abstention"]


def test_lenient_mode_defers_to_llm_pass_verdict_with_missing_facts() -> None:
    """In lenient mode the rubric trusts the LLM's overall pass/fail
    judgment — missing_facts alone is not enough to fail."""
    verdict = LLMJudgeCoverageVerdict(
        covered_facts=["RRF"],
        missing_facts=["abstention"],
        hallucinated_claims=[],
        verdict="pass",
        rationale="Answer adequately covers the topic even if it skips abstention details.",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(500, 100))
    )
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF", "abstention"],
        strictness="lenient",  # default, but pin it explicitly
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "We use RRF for fusion."}},
    )
    out = r.judge(ctx)
    assert out.status == "pass"


# ---------------- Strictness defaults ----------------


def test_strictness_defaults_to_lenient() -> None:
    r = LLMJudgeCoverage(
        target="x",
        must_contain_facts=["a"],
    )
    assert r.strictness == "lenient"


# ---------------- Prompt assembly ----------------


def test_prompt_carries_facts_and_judge_context() -> None:
    verdict = LLMJudgeCoverageVerdict(
        covered_facts=[],
        missing_facts=[],
        hallucinated_claims=[],
        verdict="pass",
        rationale="ok",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(10, 10))
    )
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF fusion", "abstention threshold"],
        judge_context_path="retrieval_trace.ranked_chunks",
        router=router,
    )
    ctx = RubricContext(
        captures={
            "answer_response": {"answer": "Some answer."},
            "retrieval_trace": {
                "ranked_chunks": [
                    {"doc_id": "doc_001", "text": "RRF combines BM25 and dense."},
                ],
            },
        },
    )
    r.judge(ctx)
    assert router.last_call is not None
    # Tier must be Haiku — judging coverage is a small task.
    tier = router.last_call["tier"]
    assert str(tier) in {"LLMTier.haiku", "haiku"}
    # Schema asked for must be the rubric's own verdict model
    assert router.last_call["text_format"] is LLMJudgeCoverageVerdict
    full_prompt = (router.last_call["system"] + "\n" + router.last_call["user"]).lower()
    # Facts surfaced verbatim
    assert "rrf fusion" in full_prompt
    assert "abstention threshold" in full_prompt
    # Target content surfaced
    assert "some answer" in full_prompt
    # Judge context surfaced when provided
    assert "doc_001" in full_prompt or "rrf combines bm25" in full_prompt


def test_prompt_omits_judge_context_when_none() -> None:
    """When ``judge_context_path`` is None the rubric still has to call
    the judge, just without grounding context."""
    verdict = LLMJudgeCoverageVerdict(
        covered_facts=[],
        missing_facts=[],
        hallucinated_claims=[],
        verdict="pass",
        rationale="ok",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(10, 10))
    )
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF"],
        judge_context_path=None,
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "Some answer."}},
    )
    r.judge(ctx)
    assert router.last_call is not None
    # Did not blow up on the missing key; called through.


# ---------------- cost_usd from raw usage when usage_summary absent ----------------


def test_cost_usd_is_zero_when_usage_summary_missing() -> None:
    """If the provider response has no usage_summary attached, the rubric
    still must not crash — it reports 0 cost rather than guessing."""
    verdict = LLMJudgeCoverageVerdict(
        covered_facts=["RRF"],
        missing_facts=[],
        hallucinated_claims=[],
        verdict="pass",
        rationale="ok",
    )
    router = _FakeRouter()
    router.set_response(_FakeResult(parsed=verdict, usage=None, usage_summary=None))
    r = LLMJudgeCoverage(
        target="answer_response.answer",
        must_contain_facts=["RRF"],
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "RRF works."}},
    )
    out = r.judge(ctx)
    assert out.status == "pass"
    assert out.cost_usd == 0.0


# ============================================================
# LLMJudgeSemanticEq (CRAG semantic-equivalence rubric)
# ============================================================


def test_llm_judge_semantic_eq_is_registered() -> None:
    assert "llm_judge_semantic_eq" in RUBRIC_REGISTRY
    assert RUBRIC_REGISTRY["llm_judge_semantic_eq"] is LLMJudgeSemanticEq


def test_semantic_eq_abstains_when_router_is_none() -> None:
    r = LLMJudgeSemanticEq(
        target="answer_response.answer",
        gold_path="setup_data.gold_answers.q1.answer",
        router=None,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "Maurice Ravel."}},
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "Maurice Ravel.",
                    "alt_ans": ["Ravel"],
                    "answer_type": "valid",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "abstain"
    assert "judge unavailable" in out.rationale.lower() or "no LLM router" in out.rationale.lower()


def test_semantic_eq_passes_when_judge_marks_equivalent_to_primary() -> None:
    verdict = SemanticEqVerdict(
        is_equivalent=True,
        matched_against="primary",
        rationale="Learner answer matches the primary gold answer.",
        factual_drift=[],
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(400, 80))
    )
    r = LLMJudgeSemanticEq(
        target="answer_response.answer",
        gold_path="setup_data.gold_answers.q1.answer",
        alt_path="setup_data.gold_answers.q1.alt_ans",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "Ravel composed it."}},
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "Maurice Ravel composed Boléro.",
                    "alt_ans": ["Ravel"],
                    "answer_type": "valid",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "pass"
    assert out.diagnostic["matched_against"] == "primary"
    # Cost computed from tokens
    expected = (400 / 1_000_000) * 1.0 + (80 / 1_000_000) * 5.0
    assert out.cost_usd == pytest.approx(expected, rel=1e-6)


def test_semantic_eq_passes_when_matched_against_alt() -> None:
    """The judge can match against the alt list; rubric still passes and
    surfaces ``matched_against=alt`` in diagnostic."""
    verdict = SemanticEqVerdict(
        is_equivalent=True,
        matched_against="alt",
        rationale="Matches an alt phrasing.",
        factual_drift=[],
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(200, 40))
    )
    r = LLMJudgeSemanticEq(
        target="answer_response.answer",
        gold_path="setup_data.gold_answers.q1.answer",
        alt_path="setup_data.gold_answers.q1.alt_ans",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "$117.2 billion"}},
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "AAPL Q1 2023 revenue was $117.2B.",
                    "alt_ans": ["$117.2 billion", "117.2B"],
                    "answer_type": "valid",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "pass"
    assert out.diagnostic["matched_against"] == "alt"


def test_semantic_eq_fails_with_factual_drift_when_not_equivalent() -> None:
    verdict = SemanticEqVerdict(
        is_equivalent=False,
        matched_against="none",
        rationale="Learner names a different composer.",
        factual_drift=["names Stravinsky instead of Ravel"],
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(300, 60))
    )
    r = LLMJudgeSemanticEq(
        target="answer_response.answer",
        gold_path="setup_data.gold_answers.q1.answer",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "Igor Stravinsky composed Boléro."}},
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "Maurice Ravel.",
                    "alt_ans": [],
                    "answer_type": "valid",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert out.diagnostic["factual_drift"] == ["names Stravinsky instead of Ravel"]
    assert out.diagnostic["matched_against"] == "none"


def test_semantic_eq_strict_mode_rejects_partial_paraphrase() -> None:
    """In strict mode any non-empty ``factual_drift`` causes a fail even
    when the judge's overall ``is_equivalent=True``."""
    verdict = SemanticEqVerdict(
        is_equivalent=True,
        matched_against="primary",
        rationale="Mostly right, minor drift.",
        factual_drift=["dates the event to 2022 instead of 2023"],
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(200, 40))
    )
    r = LLMJudgeSemanticEq(
        target="answer_response.answer",
        gold_path="setup_data.gold_answers.q1.answer",
        strictness="strict",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "About 2022."}},
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "2023.",
                    "alt_ans": [],
                    "answer_type": "valid",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert out.diagnostic["factual_drift"]
    assert out.diagnostic["strictness"] == "strict"


def test_semantic_eq_target_missing_is_learner_fail() -> None:
    """Same convention as ``LLMJudgeCoverage``: a missing dotted-path
    target is a learner-side bug, not a judge availability issue."""
    router = _FakeRouter()
    r = LLMJudgeSemanticEq(
        target="answer_response.answer",
        gold_path="setup_data.gold_answers.q1.answer",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {}},  # answer missing
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "x",
                    "alt_ans": [],
                    "answer_type": "valid",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert out.diagnostic["missing_path"] == "answer_response.answer"
    # No LLM call paid for.
    assert router.last_call is None


def test_semantic_eq_abstains_when_judge_call_raises() -> None:
    router = _FakeRouter()
    router.set_exception(RuntimeError("network blip"))
    r = LLMJudgeSemanticEq(
        target="answer_response.answer",
        gold_path="setup_data.gold_answers.q1.answer",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "x"}},
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "x",
                    "alt_ans": [],
                    "answer_type": "valid",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "abstain"
    assert "judge failed" in out.rationale.lower() or "network blip" in out.rationale.lower()


# ============================================================
# LLMJudgeFalsePremise (CRAG false-premise rubric)
# ============================================================


def test_llm_judge_false_premise_is_registered() -> None:
    assert "llm_judge_false_premise" in RUBRIC_REGISTRY
    assert RUBRIC_REGISTRY["llm_judge_false_premise"] is LLMJudgeFalsePremise


def test_false_premise_abstains_when_router_is_none() -> None:
    r = LLMJudgeFalsePremise(
        target="answer_response.answer",
        router=None,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "anything"}},
    )
    out = r.judge(ctx)
    assert out.status == "abstain"


def test_false_premise_passes_on_explicit_refusal() -> None:
    verdict = LLMJudgeFalsePremiseVerdict(
        identifies_falsity=True,
        rationale="Answer correctly refuses and identifies the false premise.",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(150, 30))
    )
    r = LLMJudgeFalsePremise(
        target="answer_response.answer",
        router=router,
    )
    ctx = RubricContext(
        captures={
            "answer_response": {
                "answer": "Microsoft did not acquire OpenAI in 2024 — that premise is false."
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "pass"
    expected = (150 / 1_000_000) * 1.0 + (30 / 1_000_000) * 5.0
    assert out.cost_usd == pytest.approx(expected, rel=1e-6)


def test_false_premise_fails_when_answer_treats_premise_as_true() -> None:
    """The learner answered the question as if the false premise were
    true — that's the classic CRAG false-premise failure mode."""
    verdict = LLMJudgeFalsePremiseVerdict(
        identifies_falsity=False,
        rationale="Answer asserts a reason for an acquisition that never happened.",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(150, 30))
    )
    r = LLMJudgeFalsePremise(
        target="answer_response.answer",
        router=router,
    )
    ctx = RubricContext(
        captures={
            "answer_response": {
                "answer": "Microsoft acquired OpenAI to consolidate its AI position."
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert "identifies_falsity" in out.diagnostic
    assert out.diagnostic["identifies_falsity"] is False


def test_false_premise_abstains_on_judge_exception() -> None:
    router = _FakeRouter()
    router.set_exception(RuntimeError("provider 5xx"))
    r = LLMJudgeFalsePremise(
        target="answer_response.answer",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {"answer": "anything"}},
    )
    out = r.judge(ctx)
    assert out.status == "abstain"


def test_false_premise_consults_expected_falsity_path_when_present() -> None:
    """If ``expected_falsity_path`` is set, the rubric pulls a hint about
    which part of the question is false from setup_data and threads it
    into the user prompt so the LLM has explicit grounding."""
    verdict = LLMJudgeFalsePremiseVerdict(
        identifies_falsity=True,
        rationale="Refusal identifies the named false claim.",
    )
    router = _FakeRouter()
    router.set_response(
        _FakeResult(parsed=verdict, usage_summary=_usage(120, 30))
    )
    r = LLMJudgeFalsePremise(
        target="answer_response.answer",
        expected_falsity_path="setup_data.gold_answers.q1.alt_ans",
        router=router,
    )
    ctx = RubricContext(
        captures={
            "answer_response": {
                "answer": "MSFT did not acquire OpenAI in 2024."
            }
        },
        setup_data={
            "gold_answers": {
                "q1": {
                    "answer": "Microsoft did not acquire OpenAI in 2024.",
                    "alt_ans": ["MSFT did not acquire OpenAI in 2024"],
                    "answer_type": "valid",
                    "question_type": "false_premise",
                }
            }
        },
    )
    out = r.judge(ctx)
    assert out.status == "pass"
    # The user prompt must thread the expected falsity hint to the LLM.
    assert router.last_call is not None
    full_prompt = router.last_call["user"].lower()
    assert "msft" in full_prompt or "did not acquire" in full_prompt


def test_false_premise_target_missing_is_learner_fail() -> None:
    router = _FakeRouter()
    r = LLMJudgeFalsePremise(
        target="answer_response.answer",
        router=router,
    )
    ctx = RubricContext(
        captures={"answer_response": {}},
    )
    out = r.judge(ctx)
    assert out.status == "fail"
    assert out.diagnostic["missing_path"] == "answer_response.answer"
    assert router.last_call is None
