"""Course planner for the single-outcome pipeline (Wave 2).

This planner runs in parallel with the legacy ``OpenAICoursePlanner``;
it is wired in behind a feature flag in a later wave. The legacy planner
emits a per-deliverable ``_CoursePlanPayload``. This one emits the
single-outcome ``CourseOutcomeSpec``: one goal, one starter, a list of
endpoints, a list of measurable quality bars, and an optional learning
path.

Design rules (mirrored in the system prompt):

- One outcome per course. NO deliverables.
- The outcome IS the contract: endpoints + measurable quality bars.
  Technologies (FAISS / BM25 / RRF / ...) are emergent consequences of
  the quality bars, not items to prescribe in the spec.
- Quality-bar IDs must be specific (``faithfulness``, ``recall_at_5``,
  ``stub_resistance``), not generic (``general_quality``).
- ``starter_type`` ∈ {empty, partial, buggy}; default ``partial``.
- ``learning_path`` is optional pedagogy keyed by quality_bar.id.

Retry / failure policy:

- Up to 3 attempts. On each attempt either the router call or the
  ``_normalize_payload`` validation may fail.
- After 3 failures, ``OutcomeCourseGenerationError`` is raised.
- NO regex/deterministic fallback. The pipeline above us decides what
  to do when planning fails.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.domain.course import GenerateCourseFromBriefRequest
from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CapabilityFlags,
    CourseOutcomeSpec,
    CRAGBenchmarkSource,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    LearningHint,
    OracleSource,
    QualityBar,
    QualityBarAggregation,
    StarterType,
)
from app.services.coursegen_logging import log_coursegen_event
from app.services.llm_router import LLMRouter, LLMTier, get_default_router


__all__ = [
    "OutcomeCoursePlanner",
    "OutcomeCourseGenerationError",
    "DEFAULT_OUTCOME_PACKAGE_TYPE",
]


# The single-outcome pipeline reuses the existing ``progressive_codebase_course``
# PackageType today — a richer ``outcome_assessment`` value is not yet in
# the registry. Centralize the choice here so callers and tests can pin
# the constant.
DEFAULT_OUTCOME_PACKAGE_TYPE: PackageType = PackageType.progressive_codebase_course


class OutcomeCourseGenerationError(RuntimeError):
    """Raised when single-outcome course generation fails after retries."""


def _parse_optional_json_object(
    raw: str | None, *, field_label: str
) -> dict[str, Any] | None:
    """Parse an LLM-emitted JSON-string into a dict, tolerating empty inputs.

    Returns ``None`` for None / "" / "null" so optional fields stay
    optional. Raises ``OutcomeCourseGenerationError`` on malformed JSON
    or non-object payloads so the planner's retry loop can re-prompt.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped.lower() == "null":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise OutcomeCourseGenerationError(
            f"{field_label} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise OutcomeCourseGenerationError(
            f"{field_label} must be a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def _parse_required_json_object(raw: str, *, field_label: str) -> dict[str, Any]:
    """Same as ``_parse_optional_json_object`` but disallows missing/empty."""
    parsed = _parse_optional_json_object(raw, field_label=field_label)
    if parsed is None:
        raise OutcomeCourseGenerationError(
            f"{field_label} is required but was empty/null."
        )
    return parsed


# ---------------- LLM-emitted payload shape ----------------


class _EndpointPayload(BaseModel):
    """Flatter, LLM-friendly shape for one ``EndpointContract``.

    ``request_schema`` and ``response_schema`` are JSON-stringified
    objects (NOT raw dicts) so the planner payload schema satisfies both
    Anthropic's structured-output mode and OpenAI's strict mode, which
    rejects untyped ``dict[str, Any]`` (it emits ``{"type":"object"}``
    without ``additionalProperties:false``). The ``_normalize_payload``
    step calls ``json.loads`` to recover the real schema mapping.
    """

    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    path: str
    request_schema_json: str | None = None
    response_schema_json: str
    description: str


class _QualityBarPayload(BaseModel):
    """Flatter, LLM-friendly shape for one ``QualityBar``.

    ``aggregation`` is optional — when the LLM omits it the planner
    defaults to ``ratio`` so legacy bars keep their pass-rate semantics.
    Count-style bars (``stub_resistance == 0``) must opt in by emitting
    ``"count_failing"`` (or ``"count_passing"`` / ``"categorical"``).
    """

    id: str
    metric_description: str
    threshold: str
    judged_by: Literal[
        "llm_haiku",
        "oracle_set_overlap",
        "behavioral_equivalence",
        "literal",
        "regex",
        "numeric",
        "load_test_harness",
    ]
    sample_size: int = Field(ge=1)
    aggregation: (
        Literal["ratio", "count_failing", "count_passing", "categorical"] | None
    ) = None


class _LearningHintPayload(BaseModel):
    on_metric_fail: str
    hint: str


class _OutcomePlanPayload(BaseModel):
    """Shape the LLM emits. Mirrors ``CourseOutcomeSpec`` but flatter so
    Sonnet does not have to navigate nested enums.

    ``oracle_source`` is intentionally optional — the planner defaults
    to ``curated`` when omitted. Advanced course authors who want the
    reference impl to be exercised at oracle time can opt in by
    emitting ``"reference_run"`` or ``"hybrid"`` explicitly.

    ``capabilities`` is a free-form ``dict[str, Any]`` so the planner
    can absorb additional capability flags in future waves without a
    schema migration. The ``_normalize_payload`` step coerces the dict
    into a :class:`CapabilityFlags`; unknown keys are tolerated by
    pydantic-ignoring the extra fields, and an omitted ``capabilities``
    map produces a default :class:`CapabilityFlags` with every flag
    off.
    """

    title: str
    goal: str
    starter_type: Literal["empty", "partial", "buggy"]
    endpoints: list[_EndpointPayload] = Field(default_factory=list)
    quality_bars: list[_QualityBarPayload] = Field(default_factory=list)
    learning_path: list[_LearningHintPayload] = Field(default_factory=list)
    oracle_source: Literal["curated", "reference_run", "hybrid"] | None = None
    # JSON-stringified ``CapabilityFlags`` (see ``_EndpointPayload`` docstring
    # for why we don't use ``dict[str, Any]`` here). Optional; omitted /
    # null / "" / "{}" all coerce to default flags (everything off).
    capabilities_json: str | None = None
    # JSON-stringified BenchmarkSource (HFBenchmarkSource | CRAGBenchmarkSource).
    # Optional; omitted / null / "{}" means "no benchmark — LLM-author the
    # setup data". When set, oracle_authoring pulls the dataset from HF and
    # serializes it under ``_setup/`` instead. Same stringification rule as
    # ``capabilities_json`` (a raw ``dict[str, Any]`` field would break
    # OpenAI's strict structured-output mode).
    #
    # Example for Quivr/CRAG:
    #   ``'{"kind":"crag","dataset":"Quivr/CRAG","max_queries":20}'``
    # Example for a BeIR dataset:
    #   ``'{"kind":"huggingface","corpus_dataset":"BeIR/scifact",
    #     "qrels_dataset":"BeIR/scifact-qrels","max_queries":50}'``
    benchmark_json: str | None = None


# ---------------- system prompt ----------------


_SYSTEM_PROMPT = (
    "You design ONE single-outcome engineering course. The course has "
    "NO deliverables — the learner builds a single working system that "
    "either passes the quality bars or it does not.\n\n"
    "The outcome IS the contract. Spell it out as:\n"
    "  - title: a learner-facing course name (8-16 words). Format: "
    "``\"<System being built>: <Skill 1>, <Skill 2> & <Skill 3>\"``. "
    "Examples: ``\"Production-Quality Finance RAG: BM25 Retrieval, "
    "Citation Grounding & False-Premise Abstention\"`` or "
    "``\"Real-Time Order Service: Idempotency Keys, "
    "Outbox-Pattern Eventing & Saga Compensation\"``. "
    "NEVER emit a goal-style \"Build a service that ...\" title — that's "
    "the brief, not a course name. The title must read like a course "
    "syllabus headline.\n"
    "  - goal: 6-10 sentence learner-facing course description with "
    "TWO sections: (1) a short paragraph naming the system the learner "
    "builds and why it's interesting, and (2) a ``Skills you'll learn:`` "
    "list (3-6 bullets) naming concrete competencies the learner will "
    "leave with — extracted from the quality bars and learning_path. "
    "DO NOT mention the brief verbatim. Each skill bullet should name "
    "a technique or capability, not just restate a bar.\n"
    "  - starter_type: empty | partial | buggy (default partial)\n"
    "  - endpoints: HTTP surfaces specific to this goal — never generic "
    "  paths like POST /service or GET /resource\n"
    "  - quality_bars: measurable bars that gate done\n"
    "  - learning_path: optional hints keyed by quality_bar.id\n\n"
    "Quality bars must include AT LEAST ONE bar from each of the "
    "following categories WHERE APPLICABLE to the goal:\n"
    "  - schema/contract correctness (literal / regex / numeric judge)\n"
    "  - recall-style retrieval accuracy (oracle_set_overlap judge)\n"
    "  - faithfulness / answer quality (llm_haiku judge), e.g. "
    "  ``faithfulness >= 0.8``\n"
    "  - abstention / refusal precision (llm_haiku judge), e.g. "
    "  ``abstention_precision >= 0.95``\n"
    "  - stub-resistance / anti-gaming (behavioral_equivalence judge)\n\n"
    "Example bars from the design doc: ``faithfulness >= 0.8``, "
    "``recall_at_5 >= 0.7``, ``abstention_precision >= 0.95``.\n\n"
    "Each quality bar carries an ``aggregation`` field that controls "
    "how per-scenario verdicts roll up. Default is ``ratio`` (pass-rate "
    "in [0, 1]; use for bars like ``faithfulness >= 0.8``). For bars "
    "that count failures (e.g. ``stub_resistance == 0`` meaning ZERO "
    "stub-resistant scenarios should fail), set "
    "``aggregation: count_failing`` and use integer thresholds like "
    "``== 0``, ``<= 2``. Use ``count_passing`` for minimum-pass-count "
    "bars (``>= 8`` of N) and ``categorical`` for ``== true``/"
    "``== false`` style bars. Do NOT use ``ratio`` (the default) for "
    "``== 0`` thresholds — that mis-grades a perfect learner as a "
    "stub-leak.\n\n"
    "Quality-bar IDs must be SPECIFIC. Do not emit ``general_quality``, "
    "``correctness_check``, or other generic IDs. Use names like "
    "``faithfulness``, ``recall_at_5``, ``abstention_precision``, "
    "``stub_resistance``, ``schema_conformance``.\n\n"
    "Do NOT prescribe implementation technology in any field. The spec "
    "describes the WHAT — endpoints and measurable bars. Tools like "
    "FAISS, BM25, or RRF are emergent consequences of the bars and "
    "must not appear in the spec. Mentioning FAISS, BM25, or RRF in "
    "title, goal, endpoints, quality_bars, or learning_path will fail "
    "spec review.\n\n"
    "Endpoint ``request_schema_json`` and ``response_schema_json`` MUST "
    "be **JSON STRINGS** (not raw objects) containing the JSON Schema "
    "for the endpoint payloads. Example: "
    "``\"response_schema_json\": \"{\\\"type\\\":\\\"object\\\",\\\"properties\\\":{...}}\"``. "
    "The ``request_schema_json`` is optional (omit for GET endpoints with "
    "no body); the ``response_schema_json`` is required.\n\n"
    "If the brief names a public benchmark dataset (e.g., "
    "Quivr/CRAG, BeIR/scifact, BeIR/nfcorpus, MS MARCO), emit "
    "``benchmark_json`` as a **JSON STRING** describing the source so "
    "the oracle authoring step can load the real gold data rather than "
    "synthesize curated scenarios. Examples:\n"
    "  - Quivr/CRAG: "
    "``\"benchmark_json\": \"{\\\"kind\\\":\\\"crag\\\","
    "\\\"dataset\\\":\\\"Quivr/CRAG\\\",\\\"max_queries\\\":20}\"``\n"
    "  - BeIR/scifact: "
    "``\"benchmark_json\": \"{\\\"kind\\\":\\\"huggingface\\\","
    "\\\"corpus_dataset\\\":\\\"BeIR/scifact\\\","
    "\\\"qrels_dataset\\\":\\\"BeIR/scifact-qrels\\\","
    "\\\"max_queries\\\":50}\"``\n"
    "Omit ``benchmark_json`` entirely when no dataset is named — the "
    "oracle author will then synthesize curated scenarios.\n\n"
    "Also emit ``capabilities_json`` as a **JSON STRING** (not a raw "
    "object) declaring what runtime primitives the learner's service "
    "needs. Example: "
    "``\"capabilities_json\": \"{\\\"runtime_llm_required\\\":false}\"``. "
    "Keys (all optional; defaults are false / \"none\"):\n"
    "  - ``runtime_llm_required`` (bool): set to ``true`` when the "
    "service synthesizes answers / responses using an LLM (RAG answer "
    "synthesis, classification justification, summarization, or any "
    "endpoint whose response is generated text grounded in an LLM "
    "call). The sandbox ships a managed Haiku endpoint; declaring "
    "this flag surfaces the proxy URL + request shape in the learner "
    "README.\n"
    "  - ``structured_logging_required`` (bool): set to ``true`` when "
    "a quality bar grades observability (latency percentiles, request "
    "tracing, audit trails). Off by default — most services log "
    "informally.\n"
    "  - ``durable_state_required`` (bool): set to ``true`` when the "
    "service must persist state across restarts (ingested corpora, "
    "tenant configs, learning records). Off by default — most "
    "MVP services keep state in memory.\n"
    "  - ``sidecar_database`` (string, one of ``\"postgres\"`` | "
    "``\"redis\"`` | ``\"none\"``): set when the service genuinely "
    "needs a relational or KV sidecar. Defaults to ``\"none\"``; "
    "prefer in-process state when the bars don't force a database.\n"
    "Be conservative — only enable a capability when the goal text or "
    "quality bars clearly require it.\n\n"
    "Return JSON matching the schema. Endpoints and quality_bars must "
    "each have at least one entry. Each learning_path hint must "
    "reference an existing quality_bar.id."
)


# ---------------- planner ----------------


class OutcomeCoursePlanner:
    """LLM-driven planner emitting ``CourseOutcomeSpec``."""

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        *,
        router: LLMRouter | None = None,
        model_id: str | None = None,
        # Live-run finding (2026-05-14): 240s was too tight for the
        # outcome spec's structured output. Sonnet legitimately needs
        # ~3-5 min to emit the spec when ``max_tokens`` allows room for
        # all the endpoint + quality_bar + benchmark detail. 480s gives
        # headroom; the planner still has 3 retry attempts on top.
        request_timeout_s: float = 480.0,
    ) -> None:
        self._router = router
        self.model_id = model_id
        self.request_timeout_s = request_timeout_s

    # ----- public API -----

    def plan_course(self, request: GenerateCourseFromBriefRequest) -> CourseOutcomeSpec:
        router = self._router if self._router is not None else get_default_router()
        user_prompt = json.dumps(self._user_prompt_payload(request), indent=2)

        last_error: Exception | None = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            log_coursegen_event(
                "outcome_planner_attempt_started",
                attempt=attempt,
                max_attempts=self.MAX_ATTEMPTS,
                goal=request.goal,
                title_hint=request.title,
            )
            try:
                response = router.parse_structured(
                    tier=LLMTier.sonnet,
                    system=_SYSTEM_PROMPT,
                    user=user_prompt,
                    text_format=_OutcomePlanPayload,
                    request_timeout_s=self.request_timeout_s,
                    # Live-run findings (2026-05-14):
                    #   - 16K max_tokens (default) truncated Sonnet's structured
                    #     output at ~17K tokens / 68KB JSON, causing a Pydantic
                    #     ValidationError ("EOF while parsing a string").
                    #   - 50K max_tokens bought enough room but the call then
                    #     exceeded the 240s router timeout (terminated mid-stream).
                    # 32K leaves comfortable headroom over the observed 17K
                    # output (≈2x) while keeping the per-attempt latency below
                    # the bumped 480s request timeout.
                    max_tokens=32_000,
                )
                payload = getattr(response, "parsed", None) or getattr(
                    response, "output_parsed", None
                )
                if payload is None:
                    raise OutcomeCourseGenerationError(
                        "Router returned no parsed outcome payload."
                    )
                spec = self._normalize_payload(request, payload)
            except OutcomeCourseGenerationError as exc:
                last_error = exc
                log_coursegen_event(
                    "outcome_planner_attempt_failed",
                    attempt=attempt,
                    max_attempts=self.MAX_ATTEMPTS,
                    error=str(exc),
                    error_kind="normalization",
                )
                if attempt >= self.MAX_ATTEMPTS:
                    raise
                continue
            except Exception as exc:
                last_error = exc
                log_coursegen_event(
                    "outcome_planner_attempt_failed",
                    attempt=attempt,
                    max_attempts=self.MAX_ATTEMPTS,
                    error=str(exc),
                    error_kind="router",
                )
                if attempt >= self.MAX_ATTEMPTS:
                    raise OutcomeCourseGenerationError(str(exc)) from exc
                continue
            log_coursegen_event(
                "outcome_planner_attempt_completed",
                attempt=attempt,
                quality_bar_count=len(spec.quality_bars),
                endpoint_count=len(spec.endpoints),
            )
            return spec

        # Defensive: the loop above always raises on the last attempt.
        if last_error is not None:
            raise OutcomeCourseGenerationError(str(last_error)) from last_error
        raise OutcomeCourseGenerationError("Outcome planner exhausted retries.")

    # ----- normalization -----

    def _normalize_payload(
        self,
        request: GenerateCourseFromBriefRequest,
        payload: _OutcomePlanPayload,
    ) -> CourseOutcomeSpec:
        if not payload.endpoints:
            raise OutcomeCourseGenerationError(
                "LLM payload had no endpoints; spec requires >= 1."
            )
        if not payload.quality_bars:
            raise OutcomeCourseGenerationError(
                "LLM payload had no quality_bars; spec requires >= 1."
            )

        try:
            starter = StarterType(payload.starter_type)
        except ValueError as exc:
            raise OutcomeCourseGenerationError(
                f"Unknown starter_type {payload.starter_type!r}: {exc}"
            ) from exc

        endpoints: list[EndpointContract] = []
        for raw_ep in payload.endpoints:
            try:
                req_schema = _parse_optional_json_object(
                    raw_ep.request_schema_json, field_label="request_schema_json"
                )
                resp_schema = _parse_required_json_object(
                    raw_ep.response_schema_json, field_label="response_schema_json"
                )
                endpoints.append(
                    EndpointContract(
                        method=HttpMethod(raw_ep.method),
                        path=raw_ep.path,
                        request_schema=req_schema,
                        response_schema=resp_schema,
                        description=raw_ep.description,
                    )
                )
            except (ValueError, ValidationError) as exc:
                raise OutcomeCourseGenerationError(
                    f"Endpoint conversion failed: {exc}"
                ) from exc

        quality_bars: list[QualityBar] = []
        for raw_bar in payload.quality_bars:
            try:
                if raw_bar.aggregation is None:
                    aggregation = QualityBarAggregation.ratio
                else:
                    aggregation = QualityBarAggregation(raw_bar.aggregation)
                quality_bars.append(
                    QualityBar(
                        id=raw_bar.id,
                        metric_description=raw_bar.metric_description,
                        threshold=raw_bar.threshold,
                        judged_by=JudgeKind(raw_bar.judged_by),
                        sample_size=raw_bar.sample_size,
                        aggregation=aggregation,
                    )
                )
            except (ValueError, ValidationError) as exc:
                raise OutcomeCourseGenerationError(
                    f"QualityBar conversion failed: {exc}"
                ) from exc

        learning_path: list[LearningHint] = []
        for raw_hint in payload.learning_path:
            try:
                learning_path.append(
                    LearningHint(
                        on_metric_fail=raw_hint.on_metric_fail,
                        hint=raw_hint.hint,
                    )
                )
            except ValidationError as exc:
                raise OutcomeCourseGenerationError(
                    f"LearningHint conversion failed: {exc}"
                ) from exc

        package_type = request.package_type_hint or DEFAULT_OUTCOME_PACKAGE_TYPE

        if payload.oracle_source is None:
            oracle_source = OracleSource.curated
        else:
            try:
                oracle_source = OracleSource(payload.oracle_source)
            except ValueError as exc:
                raise OutcomeCourseGenerationError(
                    f"Unknown oracle_source {payload.oracle_source!r}: {exc}"
                ) from exc

        # ``capabilities_json`` is a JSON-stringified flag dict (see the
        # ``_OutcomePlanPayload`` docstring for why we stringify). Empty /
        # null / "{}" coerce to default flags (everything off). Coerce to
        # ``CapabilityFlags`` so the spec downstream sees a typed bundle,
        # not a free-form dict. Pydantic raises on bad values (e.g.
        # unknown ``sidecar_database`` literal) and we surface those as
        # a normalization error so the planner can retry.
        try:
            caps_dict = _parse_optional_json_object(
                payload.capabilities_json, field_label="capabilities_json"
            ) or {}
            capabilities = CapabilityFlags.model_validate(caps_dict)
        except ValidationError as exc:
            raise OutcomeCourseGenerationError(
                f"CapabilityFlags conversion failed: {exc}"
            ) from exc

        # ---- BenchmarkSource resolution ----
        # The payload now carries an optional ``benchmark_json`` field
        # (added Bug 4 fix, 2026-05-15). When the LLM emits it, we parse
        # it as the discriminated union and use it. When it doesn't, we
        # fall back to the previous brief-text sniff so legacy briefs
        # that name Quivr/CRAG without the planner-payload field still
        # bind correctly.
        benchmark = None
        if payload.benchmark_json:
            try:
                bench_dict = _parse_optional_json_object(
                    payload.benchmark_json, field_label="benchmark_json"
                )
            except OutcomeCourseGenerationError:
                bench_dict = None
            if bench_dict:
                kind = bench_dict.get("kind")
                try:
                    if kind == "crag":
                        benchmark = CRAGBenchmarkSource.model_validate(bench_dict)
                    elif kind == "huggingface":
                        from app.services.course_outcome_models import (
                            HFBenchmarkSource,
                        )

                        benchmark = HFBenchmarkSource.model_validate(bench_dict)
                    else:
                        # Unknown kind — log and ignore, falling through
                        # to the sniff fallback below.
                        log_coursegen_event(
                            "outcome_planner_unknown_benchmark_kind",
                            kind=str(kind),
                        )
                except ValidationError as exc:
                    log_coursegen_event(
                        "outcome_planner_benchmark_json_invalid",
                        error=str(exc),
                        kind=str(kind),
                    )

        # Brief-text sniff fallback. Useful when the LLM doesn't emit
        # ``benchmark_json`` but the brief literally names a benchmark
        # (Quivr/CRAG). Skipped when the LLM already emitted a valid
        # benchmark above.
        if benchmark is None:
            brief_blob = " ".join(
                [
                    request.goal or "",
                    request.title or "",
                    *(request.learning_outcomes or []),
                ]
            ).lower()
            if "quivr/crag" in brief_blob or "quivr/ crag" in brief_blob:
                benchmark = CRAGBenchmarkSource(max_queries=20)
                log_coursegen_event(
                    "outcome_planner_benchmark_binding_injected",
                    kind="crag",
                    dataset="Quivr/CRAG",
                    max_queries=20,
                    detected_in="brief",
                )

        # Whenever ANY benchmark is bound (LLM- or sniff-emitted), the
        # oracle_source is upgraded to hybrid so the reference-impl pass
        # contributes while the benchmark rows ground the eval set.
        if benchmark is not None and oracle_source is OracleSource.curated:
            oracle_source = OracleSource.hybrid

        try:
            return CourseOutcomeSpec(
                title=payload.title,
                goal=payload.goal,
                starter_type=starter,
                endpoints=endpoints,
                quality_bars=quality_bars,
                learning_path=learning_path,
                package_type=package_type,
                oracle_source=oracle_source,
                capabilities=capabilities,
                benchmark=benchmark,
            )
        except ValidationError as exc:
            # Cross-field invariants (duplicate ids, unknown hint refs,
            # duplicate endpoints) surface here.
            raise OutcomeCourseGenerationError(
                f"CourseOutcomeSpec validation failed: {exc}"
            ) from exc

    # ----- prompt assembly -----

    def _user_prompt_payload(self, request: GenerateCourseFromBriefRequest) -> dict[str, Any]:
        setup = request.creator_setup
        return {
            "goal": request.goal,
            "title_hint": request.title,
            "package_type_hint": (
                request.package_type_hint.value if request.package_type_hint else None
            ),
            "learning_outcomes_hint": list(request.learning_outcomes),
            "creator_setup": {
                "starter_type": setup.starter_type.value if setup.starter_type else None,
                "implementation_language": setup.implementation_language,
                "application_framework": setup.application_framework,
                "primary_database": setup.primary_database,
                "tech_stack": list(setup.tech_stack),
            },
        }
