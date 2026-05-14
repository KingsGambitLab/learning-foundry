"""LLM-judge rubrics for the scenario-rubric library.

This module hosts judge-style rubrics that delegate the
``pass`` / ``fail`` decision to a Haiku-tier LLM through
:class:`app.services.llm_router.LLMRouter`. The first one,
:class:`LLMJudgeCoverage`, decides whether a generated answer covers a
list of required facts and is the workhorse rubric for RAG faithfulness
checks ("does the learner's answer reproduce the expected key points,
paraphrase OK?").

Design tenets, mirrored from
``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``:

- **Fail open on judge availability.** When the router is ``None`` or
  the LLM call raises, the rubric returns ``abstain`` so offline /
  no-API-key environments never have a judge call block grading.
- **Learner-side bugs are fail, not abstain.** A missing dotted-path
  target means the learner did not populate the response field they
  were supposed to — that's a real failure, not a judge problem.
- **Strict mode is a rubric-level override.** The LLM may be lenient
  and call something "close enough"; ``strictness="strict"`` makes the
  rubric reject any non-empty ``missing_facts`` regardless of what the
  judge concluded.
- **Cost is computed locally.** Haiku 4.5 pricing ($1/M input, $5/M
  output) is encoded here so the verdict carries an accurate cost line
  item; the caller doesn't need the raw usage object.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.scenario_rubrics_base import (
    Rubric,
    RubricContext,
    Verdict,
    register_rubric,
    resolve_path,
)


# Haiku 4.5 pricing as of the spec date. Update here when the price
# table moves; the rubric uses this to attribute per-grade cost.
_HAIKU_INPUT_USD_PER_TOKEN = 1.0 / 1_000_000
_HAIKU_OUTPUT_USD_PER_TOKEN = 5.0 / 1_000_000


_JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous reviewer judging whether a learner-written "
    "answer covers a list of required facts about the topic. You accept "
    "paraphrases, synonyms, and morphological variants — exact wording "
    "is not required. A fact is 'covered' when a reader who knows the "
    "domain would recognize the concept from the answer alone.\n\n"
    "If you are given retrieved-context chunks, use them as the source "
    "of truth: an answer that contradicts the retrieved chunks is a "
    "hallucination even if it sounds plausible.\n\n"
    "Return a structured verdict with: (1) the facts you found covered, "
    "(2) the facts you found missing, (3) any claims in the answer that "
    "are not supported by the retrieved context (hallucinations), "
    "(4) an overall pass/fail, and (5) a one-sentence rationale."
)


class LLMJudgeCoverageVerdict(BaseModel):
    """Structured verdict the Haiku judge returns for ``LLMJudgeCoverage``.

    Field semantics:

    - ``covered_facts``: subset of the input ``must_contain_facts`` the
      judge believes the answer covers (paraphrase OK).
    - ``missing_facts``: subset of the input ``must_contain_facts`` the
      judge believes the answer does NOT cover. The disjoint union of
      covered and missing should equal the input fact list, but the
      rubric tolerates the judge dropping or duplicating items.
    - ``hallucinated_claims``: free-form list of claims in the answer
      that contradict (or have no support in) the retrieved context.
      Empty when no judge_context was provided or no contradictions
      found.
    - ``verdict``: the judge's own pass/fail. In lenient mode the rubric
      defers to this; in strict mode the rubric overrides it on any
      missing fact.
    - ``rationale``: one short sentence — fed into the ``Verdict``'s
      rationale field so the learner / repair LLM gets actionable text.
    """

    covered_facts: list[str] = Field(
        default_factory=list,
        description="Required facts the answer covers (paraphrase OK).",
    )
    missing_facts: list[str] = Field(
        default_factory=list,
        description="Required facts the answer does NOT cover.",
    )
    hallucinated_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims in the answer that contradict or are unsupported by "
            "the retrieved context. Empty when no judge_context was "
            "supplied."
        ),
    )
    verdict: Literal["pass", "fail"] = Field(
        ...,
        description="Overall pass/fail from the judge.",
    )
    rationale: str = Field(
        ...,
        description="One short sentence explaining the verdict.",
    )


def _format_judge_context(value: Any) -> str:
    """Render the optional judge-context payload into a string the LLM
    can read. The payload is usually a list of ranked chunks (dicts
    with doc_id / text fields), but we accept anything and stringify
    each element so the rubric stays robust to upstream shape drift."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for index, element in enumerate(value):
            if isinstance(element, dict):
                # Surface common chunk fields first if present, fall
                # back to a full repr so nothing gets dropped silently.
                doc_id = element.get("doc_id") or element.get("id") or f"chunk_{index}"
                text = element.get("text") or element.get("content") or repr(element)
                parts.append(f"[{doc_id}] {text}")
            else:
                parts.append(f"[{index}] {element!r}")
        return "\n".join(parts)
    return repr(value)


@register_rubric
class LLMJudgeCoverage(Rubric):
    """Haiku-judged "did the answer cover these facts?" rubric.

    YAML config knobs are surfaced as ``__init__`` kwargs. ``router`` is
    injected by the rubric runner — when it's ``None`` the rubric
    abstains so offline grading still produces a verdict.
    """

    name = "llm_judge_coverage"

    def __init__(
        self,
        *,
        target: str,
        must_contain_facts: list[str],
        judge_context_path: str | None = None,
        strictness: Literal["lenient", "strict"] = "lenient",
        router: Any = None,
    ) -> None:
        self.target = target
        self.must_contain_facts = list(must_contain_facts)
        self.judge_context_path = judge_context_path
        self.strictness = strictness
        self.router = router

    def judge(self, ctx: RubricContext) -> Verdict:
        # 1. Pull the content to be judged out of captures. Missing key /
        #    out-of-range index is a learner-side bug → FAIL with the
        #    path so the feedback synth can name it.
        try:
            content = resolve_path(ctx.captures, self.target)
        except (KeyError, IndexError):
            return Verdict(
                status="fail",
                rationale=f"target path '{self.target}' not present in captures",
                diagnostic={"missing_path": self.target},
            )

        # 2. No router → abstain. Judge availability never blocks grading.
        if self.router is None:
            return Verdict(
                status="abstain",
                rationale="judge unavailable: no LLM router configured",
            )

        # 3. Optional judge-context lookup. Missing context is non-fatal —
        #    the judge can still rule on coverage without it.
        judge_context_value: Any = None
        if self.judge_context_path:
            try:
                judge_context_value = resolve_path(ctx.captures, self.judge_context_path)
            except (KeyError, IndexError):
                judge_context_value = None

        user_prompt = self._build_user_prompt(content, judge_context_value)

        # 4. Call the judge. Any exception → abstain (fail open).
        try:
            # Lazy import keeps the rubric importable in environments
            # where ``llm_router`` cannot import its provider deps.
            from app.services.llm_router import LLMTier

            result = self.router.parse_structured(
                tier=LLMTier.haiku,
                system=_JUDGE_SYSTEM_PROMPT,
                user=user_prompt,
                text_format=LLMJudgeCoverageVerdict,
                request_timeout_s=60.0,
                max_tokens=1000,
            )
        except Exception as exc:
            return Verdict(
                status="abstain",
                rationale=f"LLM judge failed: {exc}",
            )

        parsed = getattr(result, "parsed", None) or getattr(result, "output_parsed", None)
        if not isinstance(parsed, LLMJudgeCoverageVerdict):
            return Verdict(
                status="abstain",
                rationale="LLM judge returned an unexpected response shape",
            )

        cost_usd = self._compute_cost(getattr(result, "usage_summary", None))

        # 5. Decide pass/fail. In strict mode any missing fact → fail;
        #    in lenient mode trust the LLM's overall verdict.
        if self.strictness == "strict" and parsed.missing_facts:
            status: Literal["pass", "fail"] = "fail"
        else:
            status = "pass" if parsed.verdict == "pass" else "fail"

        return Verdict(
            status=status,
            rationale=parsed.rationale,
            diagnostic={
                "covered_facts": list(parsed.covered_facts),
                "missing_facts": list(parsed.missing_facts),
                "hallucinated_claims": list(parsed.hallucinated_claims),
                "judge_verdict": parsed.verdict,
                "strictness": self.strictness,
            },
            cost_usd=cost_usd,
        )

    # ----- helpers -----

    def _build_user_prompt(self, content: Any, judge_context_value: Any) -> str:
        facts_block = "\n".join(f"- {fact}" for fact in self.must_contain_facts)
        rendered_context = _format_judge_context(judge_context_value)
        context_block = (
            f"Retrieved context (source of truth):\n<<<CONTEXT>>>\n{rendered_context}\n<<<END>>>\n\n"
            if rendered_context
            else ""
        )
        return (
            f"Required facts the answer must cover (paraphrase OK):\n{facts_block}\n\n"
            f"{context_block}"
            "Answer to judge:\n"
            "<<<ANSWER>>>\n"
            f"{content}\n"
            "<<<END>>>\n\n"
            f"Strictness mode: {self.strictness}.\n"
            "Return your verdict as an LLMJudgeCoverageVerdict."
        )

    @staticmethod
    def _compute_cost(usage_summary: Any) -> float:
        """Derive a Haiku-priced cost from the usage summary.

        Prefers the summary's own ``estimated_cost_usd`` when it's
        already non-zero (the router's pricing is authoritative), and
        otherwise computes the cost from token counts so this rubric
        still reports something useful when the summary leaves cost at
        0 (e.g., when the model id isn't in the provider's price table)."""
        if usage_summary is None:
            return 0.0
        existing = getattr(usage_summary, "estimated_cost_usd", 0.0) or 0.0
        if existing > 0:
            return float(existing)
        input_tokens = int(getattr(usage_summary, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage_summary, "output_tokens", 0) or 0)
        return (
            input_tokens * _HAIKU_INPUT_USD_PER_TOKEN
            + output_tokens * _HAIKU_OUTPUT_USD_PER_TOKEN
        )


# ============================================================
# CRAG-style: semantic-equivalence + false-premise rubrics
# ============================================================


_SEMANTIC_EQ_SYSTEM_PROMPT = (
    "You are a meticulous reviewer judging whether a learner-emitted "
    "answer is SEMANTICALLY EQUIVALENT to a known-good reference "
    "answer. You accept paraphrase, abbreviation, word reordering, "
    "synonym substitution, and reasonable rounding (e.g., '$117.2B' vs "
    "'$117.2 billion'). You REJECT factual drift: different names, "
    "different dates, different magnitudes, different entities.\n\n"
    "You may also be given a list of alternative valid answers. The "
    "learner's answer passes if it is semantically equivalent to the "
    "primary reference OR to any of the alternatives.\n\n"
    "Return a structured verdict naming: (1) whether the answer is "
    "equivalent, (2) which reference it matched against ('primary', "
    "'alt', or 'none'), (3) a one-sentence rationale, and (4) any "
    "specific factual claims in the learner's answer that DO NOT match "
    "the reference (factual_drift). Even on an equivalent match, list "
    "minor drifts in factual_drift so a strict-mode rubric can reject "
    "them."
)


class SemanticEqVerdict(BaseModel):
    """Structured verdict the Haiku judge returns for
    :class:`LLMJudgeSemanticEq`.

    Field semantics:

    - ``is_equivalent``: overall pass/fail from the judge. In lenient
      mode the rubric defers to this; in strict mode the rubric
      overrides it whenever ``factual_drift`` is non-empty.
    - ``matched_against``: ``"primary"`` if the answer matched the gold
      reference; ``"alt"`` if it matched one of the alternatives;
      ``"none"`` when not equivalent.
    - ``rationale``: one short sentence used as the verdict rationale.
    - ``factual_drift``: list of factual claims in the answer that
      diverge from the reference, even when the judge calls the answer
      equivalent overall. Strict mode rejects on any non-empty drift.
    """

    is_equivalent: bool = Field(
        ...,
        description="True when the answer matches the primary or any alt.",
    )
    matched_against: Literal["primary", "alt", "none"] = Field(
        ...,
        description="Which reference the answer matched, or 'none'.",
    )
    rationale: str = Field(..., description="One short sentence.")
    factual_drift: list[str] = Field(
        default_factory=list,
        description=(
            "Specific claims in the answer that diverge from the "
            "reference. Surfaced even when is_equivalent=True so "
            "strict mode can reject minor drift."
        ),
    )


@register_rubric
class LLMJudgeSemanticEq(Rubric):
    """Haiku-judged semantic-equivalence rubric for CRAG-style courses.

    Configuration:

    - ``target`` (required): dotted path to the learner's answer in
      captures (e.g., ``"answer_response.answer"``).
    - ``gold_path`` (required): dotted path to the primary reference
      answer (typically
      ``"setup_data.gold_answers.<query_id>.answer"``).
    - ``alt_path`` (optional): dotted path to the alt-answer list. When
      omitted, only the primary reference is consulted.
    - ``strictness``: ``"lenient"`` (default) defers to the judge's
      ``is_equivalent`` field; ``"strict"`` rejects any non-empty
      ``factual_drift`` regardless of the judge's overall verdict.
    - ``router``: injected by the runner; ``None`` makes the rubric
      abstain so offline grading still produces a verdict.
    """

    name = "llm_judge_semantic_eq"

    def __init__(
        self,
        *,
        target: str,
        gold_path: str,
        alt_path: str | None = None,
        strictness: Literal["lenient", "strict"] = "lenient",
        router: Any = None,
    ) -> None:
        self.target = target
        self.gold_path = gold_path
        self.alt_path = alt_path
        self.strictness = strictness
        self.router = router

    def judge(self, ctx: RubricContext) -> Verdict:
        # 1. Pull the learner's answer. Missing dotted-path target is a
        #    learner-side bug → FAIL (same convention as LLMJudgeCoverage).
        try:
            content = resolve_path(ctx.captures, self.target)
        except (KeyError, IndexError):
            return Verdict(
                status="fail",
                rationale=f"target path '{self.target}' not present in captures",
                diagnostic={"missing_path": self.target},
            )

        # 2. No router → abstain.
        if self.router is None:
            return Verdict(
                status="abstain",
                rationale="judge unavailable: no LLM router configured",
            )

        # 3. Resolve gold / alt. Both consult a unified mapping that
        #    overlays setup_data on top of captures so "setup_data.x"
        #    paths work the same way as capture paths.
        merged = {**ctx.captures, "setup_data": ctx.setup_data, "course_meta": ctx.course_meta}
        gold_value: Any = None
        try:
            gold_value = resolve_path(merged, self.gold_path)
        except (KeyError, IndexError):
            gold_value = None
        alt_value: Any = None
        if self.alt_path is not None:
            try:
                alt_value = resolve_path(merged, self.alt_path)
            except (KeyError, IndexError):
                alt_value = None

        user_prompt = self._build_user_prompt(content, gold_value, alt_value)

        # 4. Call the judge. Any exception → abstain.
        try:
            from app.services.llm_router import LLMTier

            result = self.router.parse_structured(
                tier=LLMTier.haiku,
                system=_SEMANTIC_EQ_SYSTEM_PROMPT,
                user=user_prompt,
                text_format=SemanticEqVerdict,
                request_timeout_s=60.0,
                max_tokens=600,
            )
        except Exception as exc:
            return Verdict(
                status="abstain",
                rationale=f"LLM judge failed: {exc}",
            )

        parsed = getattr(result, "parsed", None) or getattr(result, "output_parsed", None)
        if not isinstance(parsed, SemanticEqVerdict):
            return Verdict(
                status="abstain",
                rationale="LLM judge returned an unexpected response shape",
            )

        cost_usd = LLMJudgeCoverage._compute_cost(getattr(result, "usage_summary", None))

        # 5. Decide pass/fail.
        if self.strictness == "strict" and parsed.factual_drift:
            status: Literal["pass", "fail"] = "fail"
        else:
            status = "pass" if parsed.is_equivalent else "fail"

        return Verdict(
            status=status,
            rationale=parsed.rationale,
            diagnostic={
                "is_equivalent": parsed.is_equivalent,
                "matched_against": parsed.matched_against,
                "factual_drift": list(parsed.factual_drift),
                "strictness": self.strictness,
            },
            cost_usd=cost_usd,
        )

    # ----- helpers -----

    def _build_user_prompt(
        self,
        content: Any,
        gold_value: Any,
        alt_value: Any,
    ) -> str:
        gold_block = "" if gold_value is None else str(gold_value)
        if isinstance(alt_value, list):
            alt_lines = "\n".join(f"- {item}" for item in alt_value)
        elif alt_value:
            alt_lines = f"- {alt_value}"
        else:
            alt_lines = "(none)"
        return (
            "Primary reference answer:\n"
            f"<<<PRIMARY>>>\n{gold_block}\n<<<END>>>\n\n"
            "Alternative valid answers:\n"
            f"{alt_lines}\n\n"
            "Learner's answer:\n"
            f"<<<ANSWER>>>\n{content}\n<<<END>>>\n\n"
            f"Strictness mode: {self.strictness}.\n"
            "Return your verdict as a SemanticEqVerdict."
        )


_FALSE_PREMISE_SYSTEM_PROMPT = (
    "You are a meticulous reviewer judging whether a learner's answer "
    "correctly handles a question that contains a FALSE PREMISE. A "
    "false-premise question presupposes something that is not true "
    "(e.g., 'Why did MSFT acquire OpenAI in 2024?' presupposes an "
    "acquisition that did not happen). The CORRECT answer either "
    "refuses to answer or explicitly identifies the false claim.\n\n"
    "Pass conditions (set identifies_falsity=true): the answer "
    "explicitly says the premise is false / never happened / is "
    "incorrect; or the answer refuses to answer on the grounds that "
    "the premise cannot be supported.\n\n"
    "Fail conditions (set identifies_falsity=false): the answer "
    "proceeds AS IF the false premise were true (e.g., gives reasons "
    "for the non-event); or the answer hedges without naming the "
    "falsity.\n\n"
    "If you are given an expected-falsity hint (a textual description "
    "of what part of the question is false), use it as guidance for "
    "what the answer should call out — but still pass any refusal that "
    "names the right falsity even if it phrases the rebuttal "
    "differently."
)


class LLMJudgeFalsePremiseVerdict(BaseModel):
    """Structured verdict for :class:`LLMJudgeFalsePremise`.

    - ``identifies_falsity``: True when the learner's answer correctly
      refuses or names the false premise; False when the learner
      answers as if the premise were true.
    - ``rationale``: one short sentence explaining the verdict.
    """

    identifies_falsity: bool = Field(
        ...,
        description="True when the answer refuses or identifies the false premise.",
    )
    rationale: str = Field(..., description="One short sentence.")


@register_rubric
class LLMJudgeFalsePremise(Rubric):
    """Haiku-judged false-premise rubric for CRAG-style courses.

    Configuration:

    - ``target`` (required): dotted path to the learner's answer.
    - ``expected_falsity_path`` (optional): dotted path to a textual
      hint about which part of the question is false (e.g.,
      ``"setup_data.gold_answers.q1.alt_ans"``). When set, the rubric
      threads the hint into the user prompt so the judge has explicit
      grounding.
    - ``router``: injected by the runner; ``None`` makes the rubric
      abstain.
    """

    name = "llm_judge_false_premise"

    def __init__(
        self,
        *,
        target: str,
        expected_falsity_path: str | None = None,
        router: Any = None,
    ) -> None:
        self.target = target
        self.expected_falsity_path = expected_falsity_path
        self.router = router

    def judge(self, ctx: RubricContext) -> Verdict:
        # 1. Pull the learner's answer.
        try:
            content = resolve_path(ctx.captures, self.target)
        except (KeyError, IndexError):
            return Verdict(
                status="fail",
                rationale=f"target path '{self.target}' not present in captures",
                diagnostic={"missing_path": self.target},
            )

        # 2. No router → abstain.
        if self.router is None:
            return Verdict(
                status="abstain",
                rationale="judge unavailable: no LLM router configured",
            )

        # 3. Optional falsity hint.
        falsity_hint: Any = None
        if self.expected_falsity_path is not None:
            merged = {**ctx.captures, "setup_data": ctx.setup_data, "course_meta": ctx.course_meta}
            try:
                falsity_hint = resolve_path(merged, self.expected_falsity_path)
            except (KeyError, IndexError):
                falsity_hint = None

        user_prompt = self._build_user_prompt(content, falsity_hint)

        # 4. Call the judge.
        try:
            from app.services.llm_router import LLMTier

            result = self.router.parse_structured(
                tier=LLMTier.haiku,
                system=_FALSE_PREMISE_SYSTEM_PROMPT,
                user=user_prompt,
                text_format=LLMJudgeFalsePremiseVerdict,
                request_timeout_s=60.0,
                max_tokens=400,
            )
        except Exception as exc:
            return Verdict(
                status="abstain",
                rationale=f"LLM judge failed: {exc}",
            )

        parsed = getattr(result, "parsed", None) or getattr(result, "output_parsed", None)
        if not isinstance(parsed, LLMJudgeFalsePremiseVerdict):
            return Verdict(
                status="abstain",
                rationale="LLM judge returned an unexpected response shape",
            )

        cost_usd = LLMJudgeCoverage._compute_cost(getattr(result, "usage_summary", None))
        status: Literal["pass", "fail"] = (
            "pass" if parsed.identifies_falsity else "fail"
        )
        return Verdict(
            status=status,
            rationale=parsed.rationale,
            diagnostic={
                "identifies_falsity": parsed.identifies_falsity,
            },
            cost_usd=cost_usd,
        )

    def _build_user_prompt(
        self, content: Any, falsity_hint: Any
    ) -> str:
        if falsity_hint is None:
            hint_block = ""
        elif isinstance(falsity_hint, list):
            hint_block = (
                "Expected-falsity hint (what the answer should call "
                "out as untrue):\n"
                + "\n".join(f"- {item}" for item in falsity_hint)
                + "\n\n"
            )
        else:
            hint_block = (
                "Expected-falsity hint (what the answer should call "
                "out as untrue):\n"
                f"{falsity_hint}\n\n"
            )
        return (
            "Question type: false_premise (the question presupposes "
            "something that is not true).\n\n"
            f"{hint_block}"
            "Learner's answer:\n"
            f"<<<ANSWER>>>\n{content}\n<<<END>>>\n\n"
            "Return your verdict as a LLMJudgeFalsePremiseVerdict."
        )
