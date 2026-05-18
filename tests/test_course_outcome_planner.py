"""Tests for the single-outcome course planner (Wave 2).

The planner emits ``CourseOutcomeSpec`` rather than the legacy
per-deliverable ``_CoursePlanPayload``. Sonnet authors the new payload;
on validation / call failure we retry up to 3 times. There is NO
deterministic fallback — failure propagates as
``OutcomeCourseGenerationError``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.domain.course import GenerateCourseFromBriefRequest
from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CapabilityFlags,
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    LearningHint,
    OracleSource,
    QualityBar,
    QualityBarAggregation,
    StarterType,
)
from app.services.course_outcome_planner import (
    OutcomeCoursePlanner,
    OutcomeCourseGenerationError,
    _OutcomePlanPayload,
)


# ---------------- helpers ----------------


def _valid_llm_payload_dict() -> dict:
    return {
        "title": "Grounded retrieval over a small corpus",
        "goal": (
            "Build a retrieval-and-answer service that ingests a small "
            "document corpus and returns grounded answers with citations."
        ),
        "starter_type": "partial",
        "endpoints": [
            {
                "method": "POST",
                "path": "/ingest",
                "request_schema_json": '{"documents": "list[dict]"}',
                "response_schema_json": '{"corpus_id": "str", "chunk_count": "int"}',
                "description": "Ingest documents and produce retrievable chunks.",
            },
            {
                "method": "POST",
                "path": "/answer",
                "request_schema_json": '{"corpus_id": "str", "question": "str"}',
                "response_schema_json": '{"answer": "str", "citations": "list[str]"}',
                "description": "Return a grounded answer plus cited source chunks.",
            },
        ],
        "quality_bars": [
            {
                "id": "faithfulness",
                "metric_description": "Answer faithfulness to cited sources, LLM-judged.",
                "threshold": ">= 0.8",
                "judged_by": "llm_haiku",
                "sample_size": 20,
            },
            {
                "id": "recall_at_5",
                "metric_description": "Recall@5 over the labeled retrieval oracle.",
                "threshold": ">= 0.7",
                "judged_by": "oracle_set_overlap",
                "sample_size": 20,
            },
            {
                "id": "abstention_precision",
                "metric_description": (
                    "When the corpus does not support the question, the "
                    "system declines to answer."
                ),
                "threshold": ">= 0.95",
                "judged_by": "llm_haiku",
                "sample_size": 10,
            },
            {
                "id": "stub_resistance",
                "metric_description": (
                    "Hard-coded shortcuts that bypass retrieval do not "
                    "pass the suite."
                ),
                "threshold": "== 0",
                "judged_by": "behavioral_equivalence",
                "sample_size": 5,
            },
        ],
        "learning_path": [
            {
                "on_metric_fail": "recall_at_5",
                "hint": (
                    "When recall is low, inspect chunking granularity and "
                    "the embedding similarity threshold."
                ),
            },
            {
                "on_metric_fail": "faithfulness",
                "hint": (
                    "When faithfulness is low, ensure the answer prompt "
                    "actually grounds in the retrieved passages."
                ),
            },
        ],
    }


def _valid_request() -> GenerateCourseFromBriefRequest:
    return GenerateCourseFromBriefRequest(
        goal=(
            "Teach learners to build a grounded retrieval-and-answer "
            "service over a small document corpus, with measurable "
            "faithfulness and refusal behavior."
        ),
        title="Grounded retrieval service",
        package_type_hint=PackageType.progressive_codebase_course,
    )


def _router_returning(payload: _OutcomePlanPayload) -> MagicMock:
    router = MagicMock()
    router.parse_structured.return_value = SimpleNamespace(
        parsed=payload,
        output_parsed=payload,
        usage=None,
        usage_summary=None,
    )
    return router


def _router_sequence(*results) -> MagicMock:
    """Build a router where successive parse_structured calls yield the
    next entry from ``results``. Each entry is either an Exception (to
    raise) or a payload (to return wrapped in a SimpleNamespace)."""

    router = MagicMock()

    def _side_effect(*_args, **_kwargs):
        nxt = results_iter.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return SimpleNamespace(parsed=nxt, output_parsed=nxt, usage=None, usage_summary=None)

    results_iter = list(results)
    router.parse_structured.side_effect = _side_effect
    return router


# ---------------- _OutcomePlanPayload shape ----------------


def test_outcome_plan_payload_constructs_from_valid_dict() -> None:
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())
    assert payload.title == "Grounded retrieval over a small corpus"
    assert payload.starter_type == "partial"
    assert len(payload.endpoints) == 2
    assert payload.endpoints[0].method == "POST"
    assert payload.endpoints[0].path == "/ingest"
    assert len(payload.quality_bars) == 4
    assert payload.quality_bars[0].id == "faithfulness"
    assert payload.quality_bars[0].judged_by == "llm_haiku"
    assert len(payload.learning_path) == 2
    assert payload.learning_path[0].on_metric_fail == "recall_at_5"


# ---------------- _normalize_payload ----------------


def test_normalize_payload_converts_to_course_outcome_spec() -> None:
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())

    spec = planner._normalize_payload(_valid_request(), payload)

    assert isinstance(spec, CourseOutcomeSpec)
    assert spec.title == "Grounded retrieval over a small corpus"
    assert spec.starter_type == StarterType.partial
    assert len(spec.endpoints) == 2
    assert spec.endpoints[0].method == HttpMethod.POST
    assert spec.endpoints[0].path == "/ingest"
    assert len(spec.quality_bars) == 4
    assert spec.quality_bars[0].judged_by == JudgeKind.llm_haiku
    assert isinstance(spec.quality_bars[0], QualityBar)
    assert isinstance(spec.endpoints[0], EndpointContract)
    assert isinstance(spec.learning_path[0], LearningHint)
    assert spec.package_type == PackageType.progressive_codebase_course


def test_normalize_payload_raises_on_empty_endpoints() -> None:
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload_dict = _valid_llm_payload_dict()
    payload_dict["endpoints"] = []
    payload = _OutcomePlanPayload.model_validate(payload_dict)

    with pytest.raises(OutcomeCourseGenerationError):
        planner._normalize_payload(_valid_request(), payload)


def test_normalize_payload_raises_on_empty_quality_bars() -> None:
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload_dict = _valid_llm_payload_dict()
    payload_dict["quality_bars"] = []
    payload = _OutcomePlanPayload.model_validate(payload_dict)

    with pytest.raises(OutcomeCourseGenerationError):
        planner._normalize_payload(_valid_request(), payload)


def test_normalize_payload_raises_on_duplicate_quality_bar_ids() -> None:
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload_dict = _valid_llm_payload_dict()
    # Duplicate the first bar's id onto the second bar.
    payload_dict["quality_bars"][1]["id"] = payload_dict["quality_bars"][0]["id"]
    payload = _OutcomePlanPayload.model_validate(payload_dict)

    with pytest.raises(OutcomeCourseGenerationError):
        planner._normalize_payload(_valid_request(), payload)


def test_normalize_payload_raises_on_learning_path_unknown_bar() -> None:
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload_dict = _valid_llm_payload_dict()
    payload_dict["learning_path"].append(
        {"on_metric_fail": "no_such_bar", "hint": "won't match anything"}
    )
    payload = _OutcomePlanPayload.model_validate(payload_dict)

    with pytest.raises(OutcomeCourseGenerationError):
        planner._normalize_payload(_valid_request(), payload)


# ---------------- plan_course ----------------


def test_plan_course_happy_path_returns_course_outcome_spec() -> None:
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())
    router = _router_returning(payload)

    planner = OutcomeCoursePlanner(router=router)
    spec = planner.plan_course(_valid_request())

    assert isinstance(spec, CourseOutcomeSpec)
    assert spec.title == "Grounded retrieval over a small corpus"
    # Sonnet tier — this is the design task
    call = router.parse_structured.call_args
    tier = call.kwargs["tier"]
    assert str(tier) in {"LLMTier.sonnet", "sonnet"}, f"expected sonnet tier, got {tier!r}"
    # text_format is _OutcomePlanPayload
    assert call.kwargs["text_format"] is _OutcomePlanPayload
    assert router.parse_structured.call_count == 1


def test_plan_course_retries_then_succeeds() -> None:
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())
    # First call raises, second call returns a valid payload.
    router = _router_sequence(RuntimeError("temporary failure"), payload)

    planner = OutcomeCoursePlanner(router=router)
    spec = planner.plan_course(_valid_request())

    assert isinstance(spec, CourseOutcomeSpec)
    assert router.parse_structured.call_count == 2


def test_plan_course_propagates_after_three_validation_failures() -> None:
    bad_dict = _valid_llm_payload_dict()
    bad_dict["endpoints"] = []  # normalization will reject
    bad_payload = _OutcomePlanPayload.model_validate(bad_dict)
    router = _router_sequence(bad_payload, bad_payload, bad_payload)

    planner = OutcomeCoursePlanner(router=router)
    with pytest.raises(OutcomeCourseGenerationError):
        planner.plan_course(_valid_request())

    assert router.parse_structured.call_count == 3


def test_plan_course_propagates_after_three_router_exceptions() -> None:
    router = _router_sequence(
        RuntimeError("boom 1"),
        RuntimeError("boom 2"),
        RuntimeError("boom 3"),
    )

    planner = OutcomeCoursePlanner(router=router)
    with pytest.raises(OutcomeCourseGenerationError):
        planner.plan_course(_valid_request())

    assert router.parse_structured.call_count == 3


# ---------------- system prompt content ----------------


def test_system_prompt_states_no_deliverables_and_quality_bars() -> None:
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())
    router = _router_returning(payload)
    planner = OutcomeCoursePlanner(router=router)

    planner.plan_course(_valid_request())

    call = router.parse_structured.call_args
    system = call.kwargs["system"].lower()
    assert "no deliverables" in system
    assert "quality bars" in system or "quality_bars" in system
    # Forbidden tech prescriptions
    assert "faiss" in system
    assert "bm25" in system
    assert "rrf" in system


# ---------------- error class shape ----------------


def test_outcome_course_generation_error_is_runtime_error() -> None:
    assert issubclass(OutcomeCourseGenerationError, RuntimeError)


# ---------------- oracle_source ----------------


def test_normalize_payload_defaults_oracle_source_to_curated_when_missing() -> None:
    planner = OutcomeCoursePlanner(router=MagicMock())
    # Payload omits oracle_source — the planner must still produce a
    # valid spec defaulted to ``curated``.
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())
    spec = planner._normalize_payload(_valid_request(), payload)
    assert spec.oracle_source is OracleSource.curated


def test_normalize_payload_respects_explicit_oracle_source() -> None:
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload_dict = _valid_llm_payload_dict()
    payload_dict["oracle_source"] = "reference_run"
    payload = _OutcomePlanPayload.model_validate(payload_dict)
    spec = planner._normalize_payload(_valid_request(), payload)
    assert spec.oracle_source is OracleSource.reference_run


# ---------------- aggregation passthrough ----------------


def test_normalize_payload_forwards_count_failing_aggregation() -> None:
    """When the LLM emits a quality_bar with ``aggregation:
    count_failing`` (the right shape for ``stub_resistance == 0``),
    the planner must forward it onto the resulting ``QualityBar``."""
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload_dict = _valid_llm_payload_dict()
    # The stub_resistance bar in the fixture already uses ``== 0`` —
    # opt it in to count_failing so the synthesizer grades it correctly.
    for bar in payload_dict["quality_bars"]:
        if bar["id"] == "stub_resistance":
            bar["aggregation"] = "count_failing"
    payload = _OutcomePlanPayload.model_validate(payload_dict)
    spec = planner._normalize_payload(_valid_request(), payload)
    stub_bar = next(b for b in spec.quality_bars if b.id == "stub_resistance")
    assert stub_bar.aggregation is QualityBarAggregation.count_failing
    # Other bars keep the default.
    faith_bar = next(b for b in spec.quality_bars if b.id == "faithfulness")
    assert faith_bar.aggregation is QualityBarAggregation.ratio


def test_normalize_payload_defaults_missing_aggregation_to_ratio() -> None:
    """LLM payloads that omit ``aggregation`` (the common case) must
    default to ratio so legacy bars keep their existing semantics.
    """
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())
    spec = planner._normalize_payload(_valid_request(), payload)
    for bar in spec.quality_bars:
        assert bar.aggregation is QualityBarAggregation.ratio


# ---------------- capability flag passthrough ----------------


def test_normalize_payload_forwards_explicit_capability_flags() -> None:
    """When the LLM payload declares capability flags (e.g. the service
    needs a managed LLM endpoint), the planner forwards them verbatim
    onto the resulting :class:`CourseOutcomeSpec`."""
    import json as _json
    planner = OutcomeCoursePlanner(router=MagicMock())
    payload_dict = _valid_llm_payload_dict()
    # Capabilities is JSON-stringified — see ``_OutcomePlanPayload`` docstring.
    payload_dict["capabilities_json"] = _json.dumps({
        "runtime_llm_required": True,
        "structured_logging_required": True,
        "durable_state_required": False,
        "sidecar_database": "postgres",
    })
    payload = _OutcomePlanPayload.model_validate(payload_dict)
    spec = planner._normalize_payload(_valid_request(), payload)
    assert isinstance(spec.capabilities, CapabilityFlags)
    assert spec.capabilities.runtime_llm_required is True
    assert spec.capabilities.structured_logging_required is True
    assert spec.capabilities.durable_state_required is False
    assert spec.capabilities.sidecar_database == "postgres"


def test_normalize_payload_defaults_missing_capabilities_to_empty_flags() -> None:
    """LLM payloads that omit ``capabilities`` (the common case for
    courses with no runtime LLM / DB needs) must default to an empty
    :class:`CapabilityFlags` so the spec is still valid."""
    planner = OutcomeCoursePlanner(router=MagicMock())
    # Fixture omits ``capabilities`` — planner must still produce a spec.
    payload = _OutcomePlanPayload.model_validate(_valid_llm_payload_dict())
    spec = planner._normalize_payload(_valid_request(), payload)
    assert isinstance(spec.capabilities, CapabilityFlags)
    assert spec.capabilities.runtime_llm_required is False
    assert spec.capabilities.structured_logging_required is False
    assert spec.capabilities.durable_state_required is False
    assert spec.capabilities.sidecar_database == "none"
