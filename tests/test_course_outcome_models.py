"""Tests for the single-outcome course spec models.

Covers the seven types introduced in
``app/services/course_outcome_models.py``:

- ``StarterType`` (enum): starter-codebase shapes the learner can begin with.
- ``HttpMethod`` (enum): HTTP verbs the new endpoint contract accepts.
- ``EndpointContract``: one HTTP surface the learner must implement.
- ``JudgeKind`` (enum): rubric flavors that grade a quality bar.
- ``QualityBar``: one numerical/behavioral bar the deliverable must clear.
- ``LearningHint``: advisory hint surfaced when a specific bar fails.
- ``CourseOutcomeSpec``: top-level single-outcome spec, composed of the
  above plus ``PackageType`` and the existing project/runtime models.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.registry import PackageType
from app.domain.task_agent import ProjectContractSpec, ProjectRuntimePlanSpec
from app.services.course_outcome_models import (
    CapabilityFlags,
    CourseOutcomeSpec,
    CRAGBenchmarkSource,
    EndpointContract,
    HFBenchmarkSource,
    HttpMethod,
    JudgeKind,
    LearningHint,
    OracleSource,
    QualityBar,
    QualityBarAggregation,
    StarterType,
)


# ---------------- StarterType ----------------


def test_starter_type_has_three_values() -> None:
    assert {member.value for member in StarterType} == {"empty", "partial", "buggy"}


def test_starter_type_string_compat() -> None:
    # str enum: members compare equal to their raw string.
    assert StarterType.partial == "partial"
    assert StarterType("buggy") is StarterType.buggy


def test_starter_type_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        StarterType("legacy_template")


# ---------------- HttpMethod ----------------


def test_http_method_covers_five_verbs() -> None:
    assert {member.value for member in HttpMethod} == {
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
    }


def test_http_method_string_compat() -> None:
    assert HttpMethod.GET == "GET"
    assert HttpMethod("POST") is HttpMethod.POST


def test_http_method_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        HttpMethod("OPTIONS")


# ---------------- EndpointContract ----------------


def _endpoint(**overrides: object) -> EndpointContract:
    payload: dict[str, object] = {
        "method": HttpMethod.POST,
        "path": "/retrieval-corpuses/{id}/answers",
        "request_schema": {"question": "str"},
        "response_schema": {"answer": "str", "cited_chunk_ids": "list[str]"},
        "description": "Answer a question against a corpus.",
    }
    payload.update(overrides)
    return EndpointContract(**payload)  # type: ignore[arg-type]


def test_endpoint_contract_happy_path() -> None:
    ec = _endpoint()
    assert ec.method is HttpMethod.POST
    assert ec.path == "/retrieval-corpuses/{id}/answers"
    assert ec.request_schema == {"question": "str"}
    assert ec.response_schema["cited_chunk_ids"] == "list[str]"
    assert ec.description.startswith("Answer")


def test_endpoint_contract_request_schema_optional() -> None:
    ec = _endpoint(method=HttpMethod.GET, request_schema=None, path="/health")
    assert ec.request_schema is None
    assert ec.method is HttpMethod.GET


def test_endpoint_contract_accepts_method_as_string() -> None:
    # Pydantic should coerce the literal verb string into the enum.
    ec = _endpoint(method="DELETE")
    assert ec.method is HttpMethod.DELETE


def test_endpoint_contract_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        _endpoint(method="OPTIONS")


def test_endpoint_contract_response_schema_required() -> None:
    with pytest.raises(ValidationError):
        EndpointContract(  # type: ignore[call-arg]
            method=HttpMethod.GET,
            path="/x",
            description="missing response schema",
        )


# ---------------- JudgeKind ----------------


def test_judge_kind_covers_expected_flavors() -> None:
    assert {member.value for member in JudgeKind} == {
        "llm_haiku",
        "oracle_set_overlap",
        "behavioral_equivalence",
        "literal",
        "regex",
        "numeric",
        "load_test_harness",
    }


def test_judge_kind_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        JudgeKind("vibes")


# ---------------- QualityBar ----------------


def _bar(**overrides: object) -> QualityBar:
    payload: dict[str, object] = {
        "id": "groundedness",
        "metric_description": "fraction of answers grounded in cited chunks",
        "threshold": ">= 0.8",
        "judged_by": JudgeKind.llm_haiku,
        "sample_size": 20,
    }
    payload.update(overrides)
    return QualityBar(**payload)  # type: ignore[arg-type]


def test_quality_bar_happy_path() -> None:
    bar = _bar()
    assert bar.id == "groundedness"
    assert bar.threshold == ">= 0.8"
    assert bar.judged_by is JudgeKind.llm_haiku
    assert bar.sample_size == 20


def test_quality_bar_sample_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _bar(sample_size=0)
    with pytest.raises(ValidationError):
        _bar(sample_size=-3)


def test_quality_bar_accepts_judge_kind_as_string() -> None:
    bar = _bar(judged_by="oracle_set_overlap")
    assert bar.judged_by is JudgeKind.oracle_set_overlap


# ---------------- LearningHint ----------------


def test_learning_hint_happy_path() -> None:
    hint = LearningHint(
        on_metric_fail="groundedness",
        hint="Cite the chunk id you copied each sentence from.",
    )
    assert hint.on_metric_fail == "groundedness"
    assert hint.hint.startswith("Cite")


def test_learning_hint_requires_both_fields() -> None:
    with pytest.raises(ValidationError):
        LearningHint(on_metric_fail="groundedness")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        LearningHint(hint="text")  # type: ignore[call-arg]


# ---------------- CourseOutcomeSpec ----------------


def _spec(**overrides: object) -> CourseOutcomeSpec:
    payload: dict[str, object] = {
        "title": "Retrieval QA service",
        "goal": (
            "Build a small retrieval-augmented QA service that answers "
            "user questions grounded in a corpus of internal docs."
        ),
        "starter_type": StarterType.partial,
        "endpoints": [_endpoint()],
        "quality_bars": [_bar()],
        "learning_path": [
            LearningHint(
                on_metric_fail="groundedness",
                hint="Cite the chunk id you copied each sentence from.",
            )
        ],
        "package_type": PackageType.progressive_codebase_course,
    }
    payload.update(overrides)
    return CourseOutcomeSpec(**payload)  # type: ignore[arg-type]


def test_course_outcome_spec_happy_path() -> None:
    spec = _spec()
    assert spec.title == "Retrieval QA service"
    assert spec.starter_type is StarterType.partial
    assert spec.package_type is PackageType.progressive_codebase_course
    assert len(spec.endpoints) == 1
    assert len(spec.quality_bars) == 1
    assert len(spec.learning_path) == 1
    assert spec.project_contract is None
    assert spec.runtime_plan is None


def test_course_outcome_spec_title_minimum_length() -> None:
    with pytest.raises(ValidationError):
        _spec(title="abc")


def test_course_outcome_spec_goal_minimum_length() -> None:
    with pytest.raises(ValidationError):
        _spec(goal="too short")


def test_course_outcome_spec_requires_endpoint() -> None:
    with pytest.raises(ValidationError):
        _spec(endpoints=[])


def test_course_outcome_spec_requires_quality_bar() -> None:
    with pytest.raises(ValidationError):
        _spec(quality_bars=[])


def test_course_outcome_spec_learning_path_defaults_empty() -> None:
    spec = _spec(learning_path=[])
    assert spec.learning_path == []


def test_course_outcome_spec_learning_hint_must_reference_existing_bar() -> None:
    with pytest.raises(ValidationError) as exc_info:
        _spec(
            learning_path=[
                LearningHint(
                    on_metric_fail="latency",  # not in quality_bars
                    hint="Tighten your retrieval window.",
                )
            ]
        )
    assert "latency" in str(exc_info.value)


def test_course_outcome_spec_quality_bar_ids_must_be_unique() -> None:
    duplicate = _bar(id="groundedness")
    with pytest.raises(ValidationError) as exc_info:
        _spec(quality_bars=[_bar(), duplicate])
    assert "groundedness" in str(exc_info.value)


def test_course_outcome_spec_endpoints_must_be_unique() -> None:
    # Same (method, path) tuple appears twice.
    dup = _endpoint()
    with pytest.raises(ValidationError):
        _spec(endpoints=[_endpoint(), dup])


def test_course_outcome_spec_endpoints_unique_per_method() -> None:
    # Same path under different methods is allowed.
    spec = _spec(
        endpoints=[
            _endpoint(method=HttpMethod.POST, path="/items"),
            _endpoint(method=HttpMethod.GET, path="/items"),
        ]
    )
    assert {ep.method for ep in spec.endpoints} == {HttpMethod.POST, HttpMethod.GET}


def test_course_outcome_spec_accepts_project_contract_and_runtime_plan() -> None:
    contract = ProjectContractSpec()
    runtime = ProjectRuntimePlanSpec(implementation_language="python")
    spec = _spec(project_contract=contract, runtime_plan=runtime)
    assert spec.project_contract is contract
    assert spec.runtime_plan is not None
    assert spec.runtime_plan.implementation_language == "python"


def test_course_outcome_spec_learning_path_multiple_hints_all_must_resolve() -> None:
    bar_a = _bar(id="groundedness")
    bar_b = _bar(id="latency_ms", metric_description="p95 latency", threshold="<= 2000ms")
    spec = _spec(
        quality_bars=[bar_a, bar_b],
        learning_path=[
            LearningHint(on_metric_fail="groundedness", hint="Cite chunks."),
            LearningHint(on_metric_fail="latency_ms", hint="Cache retrieval."),
        ],
    )
    assert {hint.on_metric_fail for hint in spec.learning_path} == {
        "groundedness",
        "latency_ms",
    }


# ---------------- OracleSource ----------------


def test_oracle_source_has_three_values() -> None:
    assert {member.value for member in OracleSource} == {
        "curated",
        "reference_run",
        "hybrid",
    }


def test_course_outcome_spec_oracle_source_default_curated() -> None:
    spec = _spec()
    # Default is curated (the RAG-friendly default).
    assert spec.oracle_source is OracleSource.curated


def test_course_outcome_spec_accepts_explicit_oracle_source() -> None:
    spec_run = _spec(oracle_source=OracleSource.reference_run)
    assert spec_run.oracle_source is OracleSource.reference_run
    spec_hybrid = _spec(oracle_source=OracleSource.hybrid)
    assert spec_hybrid.oracle_source is OracleSource.hybrid


# ---------------- QualityBarAggregation ----------------


def test_quality_bar_aggregation_has_four_values() -> None:
    """The aggregation enum exists so count-style bars (e.g.
    ``stub_resistance == 0``) are not mis-interpreted as ratio bars
    by ``_build_bar_report``. Four values cover the design space:
    ratio (default), count_failing, count_passing, categorical.
    """
    assert {member.value for member in QualityBarAggregation} == {
        "ratio",
        "count_failing",
        "count_passing",
        "categorical",
    }


def test_quality_bar_aggregation_defaults_to_ratio() -> None:
    """Existing bars (authored before the aggregation field shipped)
    must keep grading as pass-rate ratios so the change is backward
    compatible.
    """
    bar = _bar()
    assert bar.aggregation is QualityBarAggregation.ratio


def test_quality_bar_aggregation_persists_through_model_validate() -> None:
    """A bar that opts in to ``count_failing`` keeps the field round-
    trip through pydantic — the planner forwards this verbatim and the
    grader synthesizer branches on it.
    """
    raw = {
        "id": "stub_resistance",
        "metric_description": "zero stub-resistant scenarios should fail",
        "threshold": "== 0",
        "judged_by": "behavioral_equivalence",
        "sample_size": 5,
        "aggregation": "count_failing",
    }
    bar = QualityBar.model_validate(raw)
    assert bar.aggregation is QualityBarAggregation.count_failing
    # And the categorical and count_passing values round-trip too.
    raw["aggregation"] = "count_passing"
    assert QualityBar.model_validate(raw).aggregation is (
        QualityBarAggregation.count_passing
    )
    raw["aggregation"] = "categorical"
    raw["threshold"] = "== true"
    assert QualityBar.model_validate(raw).aggregation is (
        QualityBarAggregation.categorical
    )


def test_course_outcome_spec_oracle_source_coexists_with_learning_path() -> None:
    # Make sure the new field does not interact with existing cross-field
    # validators (duplicate ids, unknown hint refs, etc.).
    spec = _spec(
        oracle_source=OracleSource.hybrid,
        learning_path=[
            LearningHint(
                on_metric_fail="groundedness",
                hint="Cite the chunk id you copied each sentence from.",
            )
        ],
    )
    assert spec.oracle_source is OracleSource.hybrid
    assert len(spec.learning_path) == 1


# ---------------- HFBenchmarkSource ----------------


def test_hf_benchmark_source_happy_path_defaults() -> None:
    """Construct with the minimum required fields; the rest of the field
    names should default to the BeIR layout (``_id`` / ``text`` /
    ``query-id`` / ``corpus-id`` / ``score``)."""
    src = HFBenchmarkSource(
        corpus_dataset="BeIR/scifact",
        qrels_dataset="BeIR/scifact-qrels",
    )
    assert src.kind == "huggingface"
    assert src.corpus_dataset == "BeIR/scifact"
    assert src.qrels_dataset == "BeIR/scifact-qrels"
    assert src.queries_dataset is None
    assert src.split == "test"
    assert src.max_corpus_docs is None
    assert src.max_queries is None
    # BeIR field-name defaults.
    assert src.corpus_id_field == "_id"
    assert src.corpus_text_field == "text"
    assert src.corpus_title_field == "title"
    assert src.query_id_field == "_id"
    assert src.query_text_field == "text"
    assert src.qrels_query_field == "query-id"
    assert src.qrels_corpus_field == "corpus-id"
    assert src.qrels_score_field == "score"


def test_hf_benchmark_source_accepts_custom_field_mapping() -> None:
    """Non-BeIR datasets often use ``id`` / ``content``. Override the
    field-name defaults to map them."""
    src = HFBenchmarkSource(
        corpus_dataset="some/other-corpus",
        queries_dataset="some/other-queries",
        qrels_dataset="some/other-qrels",
        split="train",
        corpus_id_field="id",
        corpus_text_field="content",
        corpus_title_field=None,
        query_id_field="qid",
        query_text_field="question",
        qrels_query_field="qid",
        qrels_corpus_field="docid",
        qrels_score_field="rel",
        max_corpus_docs=500,
        max_queries=50,
    )
    assert src.corpus_id_field == "id"
    assert src.corpus_text_field == "content"
    assert src.corpus_title_field is None
    assert src.query_id_field == "qid"
    assert src.query_text_field == "question"
    assert src.qrels_query_field == "qid"
    assert src.qrels_corpus_field == "docid"
    assert src.qrels_score_field == "rel"
    assert src.max_corpus_docs == 500
    assert src.max_queries == 50
    assert src.split == "train"


def test_course_outcome_spec_accepts_benchmark_field() -> None:
    """``CourseOutcomeSpec.benchmark`` is optional; setting it round-trips
    cleanly."""
    src = HFBenchmarkSource(
        corpus_dataset="BeIR/scifact",
        qrels_dataset="BeIR/scifact-qrels",
    )
    spec = _spec(benchmark=src)
    assert spec.benchmark is src
    assert spec.benchmark.corpus_dataset == "BeIR/scifact"


def test_course_outcome_spec_benchmark_defaults_to_none() -> None:
    """Back-compat: existing specs that don't declare a benchmark see the
    field default to None."""
    spec = _spec()
    assert spec.benchmark is None


# ---------------- CRAGBenchmarkSource ----------------


def test_crag_benchmark_source_happy_path_defaults() -> None:
    """Construct with no required overrides; defaults map to Quivr/CRAG's
    canonical schema (one dataset, per-query embedded retrieval pool,
    ``interaction_id`` keys, ``search_results`` field, etc.)."""
    src = CRAGBenchmarkSource()
    assert src.kind == "crag"
    assert src.dataset == "Quivr/CRAG"
    assert src.use_split == "validation"
    assert src.max_queries is None
    # Default filters: only ``answer_type_filter`` is set ("valid"); the
    # others are explicitly None so all rows pass when no filter is given.
    assert src.domain_filter is None
    assert src.question_type_filter is None
    assert src.answer_type_filter == ["valid"]
    # Field-mapping defaults match the Quivr/CRAG schema.
    assert src.query_id_field == "interaction_id"
    assert src.query_text_field == "query"
    assert src.answer_field == "answer"
    assert src.alt_ans_field == "alt_ans"
    assert src.search_results_field == "search_results"
    assert src.split_field == "split"
    assert src.domain_field == "domain"
    assert src.question_type_field == "question_type"
    assert src.answer_type_field == "answer_type"


def test_course_outcome_spec_accepts_crag_benchmark() -> None:
    """``CourseOutcomeSpec.benchmark`` accepts a ``CRAGBenchmarkSource``
    via the discriminated union — round-trip cleanly."""
    src = CRAGBenchmarkSource(
        use_split="test",
        domain_filter=["finance", "music"],
        max_queries=50,
    )
    spec = _spec(benchmark=src)
    assert isinstance(spec.benchmark, CRAGBenchmarkSource)
    assert spec.benchmark.use_split == "test"
    assert spec.benchmark.domain_filter == ["finance", "music"]
    assert spec.benchmark.max_queries == 50


def test_crag_benchmark_source_serializes_and_round_trips_via_union() -> None:
    """The discriminator (``kind``) MUST be present in the dump so the
    union deserializer picks ``CRAGBenchmarkSource`` and not
    ``HFBenchmarkSource``."""
    src = CRAGBenchmarkSource(
        domain_filter=["sports"],
        question_type_filter=["false_premise"],
    )
    spec = _spec(benchmark=src)
    dumped = spec.model_dump()
    assert dumped["benchmark"]["kind"] == "crag"
    # Re-validate the dump and confirm the union selected the CRAG side.
    reloaded = CourseOutcomeSpec.model_validate(dumped)
    assert isinstance(reloaded.benchmark, CRAGBenchmarkSource)
    assert reloaded.benchmark.domain_filter == ["sports"]
    assert reloaded.benchmark.question_type_filter == ["false_premise"]


def test_crag_benchmark_source_rejects_unknown_use_split() -> None:
    """Only ``validation`` / ``test`` are accepted (mirrors the int64
    ``split`` column on Quivr/CRAG: 0 -> validation, 1 -> test)."""
    with pytest.raises(ValidationError):
        CRAGBenchmarkSource(use_split="train")  # type: ignore[arg-type]


# ---------------- CapabilityFlags ----------------


def test_capability_flags_defaults_all_off() -> None:
    """A freshly constructed ``CapabilityFlags`` has every runtime
    primitive turned off and ``sidecar_database`` set to ``"none"``.
    This is the conservative default so a spec that omits capabilities
    does not silently advertise a sandbox LLM proxy or a database.
    """
    flags = CapabilityFlags()
    assert flags.runtime_llm_required is False
    assert flags.structured_logging_required is False
    assert flags.durable_state_required is False
    assert flags.sidecar_database == "none"


def test_course_outcome_spec_capabilities_default_empty_flags() -> None:
    """``CourseOutcomeSpec.capabilities`` is optional and defaults to an
    empty :class:`CapabilityFlags` so legacy callers that omit the
    field still construct a valid spec.
    """
    spec = _spec()
    assert isinstance(spec.capabilities, CapabilityFlags)
    assert spec.capabilities.runtime_llm_required is False
    assert spec.capabilities.sidecar_database == "none"


def test_course_outcome_spec_capabilities_round_trip() -> None:
    """Custom capability values persist through ``model_dump`` and
    ``model_validate`` — the materializer reads them back off the dumped
    spec and must see what the planner emitted."""
    flags = CapabilityFlags(
        runtime_llm_required=True,
        structured_logging_required=True,
        durable_state_required=True,
        sidecar_database="postgres",
    )
    spec = _spec(capabilities=flags)
    dumped = spec.model_dump()
    assert dumped["capabilities"]["runtime_llm_required"] is True
    assert dumped["capabilities"]["sidecar_database"] == "postgres"
    restored = CourseOutcomeSpec.model_validate(dumped)
    assert restored.capabilities.runtime_llm_required is True
    assert restored.capabilities.structured_logging_required is True
    assert restored.capabilities.durable_state_required is True
    assert restored.capabilities.sidecar_database == "postgres"
