"""Pydantic models for the single-outcome course pipeline.

The simplified pipeline collapses the legacy per-deliverable progressive
model down to one outcome per course. A ``CourseOutcomeSpec`` captures
everything downstream stages (planner, materializer, grader) need:

- A stakeholder-facing ``goal`` and human-readable ``title``.
- The starter shape the learner begins with (``StarterType``).
- A list of HTTP ``EndpointContract`` instances the learner must build.
- A list of ``QualityBar`` instances that gate "done" — each carries a
  threshold expression and a ``JudgeKind`` selector for the rubric.
- An optional ``learning_path`` of ``LearningHint`` rows mapped 1:1 to
  a quality bar; hints surface to the learner when that bar fails.
- The ``PackageType`` from ``app.domain.registry`` and optional
  composition with the existing ``ProjectContractSpec`` and
  ``ProjectRuntimePlanSpec`` from ``app.domain.task_agent``.

These models are intentionally additive; the legacy spec types are not
touched here.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

from app.domain.registry import PackageType
from app.domain.task_agent import ProjectContractSpec, ProjectRuntimePlanSpec

__all__ = [
    "StarterType",
    "HttpMethod",
    "EndpointContract",
    "JudgeKind",
    "QualityBar",
    "QualityBarAggregation",
    "LearningHint",
    "OracleSource",
    "HFBenchmarkSource",
    "CRAGBenchmarkSource",
    "BenchmarkSource",
    "CapabilityFlags",
    "CourseOutcomeSpec",
]


class StarterType(str, Enum):
    """Shapes the learner's starter codebase can take.

    ``empty`` — scaffolding only; learner writes the full implementation.
    ``partial`` — some pieces wired up, the learner finishes the rest.
        This is the default end-to-end implemented variant today.
    ``buggy`` — complete implementation with seeded defects; the learner
        diagnoses and repairs.
    """

    empty = "empty"
    partial = "partial"
    buggy = "buggy"


class HttpMethod(str, Enum):
    """HTTP verbs accepted by ``EndpointContract``."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class EndpointContract(BaseModel):
    """One HTTP surface the learner must implement.

    ``request_schema`` and ``response_schema`` are intentionally
    free-form ``dict[str, Any]`` shape maps (e.g. ``{"answer": "str"}``)
    rather than strict JSON Schema — they describe the contract at the
    granularity the planner and learner brief need.
    """

    method: HttpMethod
    path: str
    request_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any]
    description: str


class JudgeKind(str, Enum):
    """Rubric flavor that grades a ``QualityBar``.

    Names mirror the rubric class roster in
    ``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``
    plus the structural primitives (``literal``, ``regex``, ``numeric``).
    """

    llm_haiku = "llm_haiku"
    oracle_set_overlap = "oracle_set_overlap"
    behavioral_equivalence = "behavioral_equivalence"
    literal = "literal"
    regex = "regex"
    numeric = "numeric"
    load_test_harness = "load_test_harness"


class QualityBarAggregation(str, Enum):
    """How per-scenario verdicts roll up into a single observed value.

    ``ratio`` (default) — observed = passing_count / total. Thresholds
        live in ``[0, 1]`` (or the human percent form). This is the
        right shape for ``faithfulness >= 0.8``-style bars.
    ``count_failing`` — observed = number of failing scenarios.
        Thresholds are non-negative integers. The right shape for
        gaming-resistant bars like ``stub_resistance == 0`` (meaning
        "zero stub-resistant scenarios should fail"). Without this
        aggregation a perfect learner with ratio 1.0 would be graded
        ``1.0 == 0`` → fail, which is the bug this enum exists to fix.
    ``count_passing`` — observed = number of passing scenarios.
        Thresholds are non-negative integers, e.g. ``>= 8`` for "at
        least 8 of N scenarios must pass". Use when the bar is naturally
        expressed as a minimum count rather than a fraction.
    ``categorical`` — observed = ``"true"`` if all relevant scenarios
        passed, ``"false"`` otherwise. Thresholds are tokens like
        ``"== true"`` or ``"== json"``. Bars with a free-form
        non-numeric RHS land here.
    """

    ratio = "ratio"
    count_failing = "count_failing"
    count_passing = "count_passing"
    categorical = "categorical"


class QualityBar(BaseModel):
    """One numerical or behavioral bar the deliverable must clear.

    ``threshold`` is a free-form expression (``">= 0.8"``, ``"<= 2000ms"``)
    parsed downstream by the evaluator; we keep the raw string here so
    course planners can author bars without typing-up a comparator AST.

    ``aggregation`` selects how per-scenario verdicts roll up. The
    default ``ratio`` matches the original Wave 4 behavior; bars that
    count failures (``stub_resistance == 0``) opt in to
    ``count_failing`` so the threshold is interpreted as an integer
    failure count rather than a pass-rate ratio.
    """

    id: str
    metric_description: str
    threshold: str
    judged_by: JudgeKind
    sample_size: int = Field(ge=1)
    aggregation: QualityBarAggregation = QualityBarAggregation.ratio


class LearningHint(BaseModel):
    """Advisory hint surfaced to the learner when a specific bar fails.

    ``on_metric_fail`` references a ``QualityBar.id`` within the same
    spec; the cross-field check lives on ``CourseOutcomeSpec``.
    """

    on_metric_fail: str
    hint: str


class OracleSource(str, Enum):
    """Oracle-source mode for the single-outcome grader.

    Three values mirror the publish-gate matrix:

    - ``curated`` — humans (or an authoring LLM) hand-write gold sets
      shipped under ``private/grader/_setup/``. The reference impl is
      treated as documentation only; the grader never boots it. Default
      for RAG-style courses.
    - ``reference_run`` — the Wave 4 default. ``oracle_pass`` boots the
      reference implementation, runs every scenario through it, and
      captures the outputs as the oracle. Used when there is a single
      "right answer" the reference impl can compute deterministically.
    - ``hybrid`` — both. Curated gold is validated AND the reference
      impl runs; the course is publishable only when both pass.
    """

    curated = "curated"
    reference_run = "reference_run"
    hybrid = "hybrid"


class HFBenchmarkSource(BaseModel):
    """Declarative source for a Hugging Face dataset used as the
    course's benchmark instead of LLM-synthesized setup data.

    Scoped to BeIR-family layouts in v1; the loader recognizes that
    shape. Extensible to other layouts later via a ``layout`` field
    (deferred for now).

    Field-name defaults match the BeIR family
    (``BeIR/scifact``, ``BeIR/nfcorpus``, etc.): corpus rows look like
    ``{"_id": str, "title": str, "text": str}``; query rows look like
    ``{"_id": str, "text": str}``; qrels rows look like
    ``{"query-id": str, "corpus-id": str, "score": int}``. Override
    the field names per-dataset for non-BeIR layouts.
    """

    kind: Literal["huggingface"] = "huggingface"
    corpus_dataset: str
    queries_dataset: str | None = None
    qrels_dataset: str
    split: Literal["train", "test", "validation"] = "test"
    max_corpus_docs: int | None = None
    max_queries: int | None = None
    corpus_id_field: str = "_id"
    corpus_text_field: str = "text"
    corpus_title_field: str | None = "title"
    query_id_field: str = "_id"
    query_text_field: str = "text"
    qrels_query_field: str = "query-id"
    qrels_corpus_field: str = "corpus-id"
    qrels_score_field: str = "score"


class CRAGBenchmarkSource(BaseModel):
    """Declarative source for the Quivr/CRAG benchmark.

    Sibling to :class:`HFBenchmarkSource`. CRAG is a fundamentally
    different shape from the BeIR family:

    - Single HF dataset (no separate corpus / queries / qrels splits).
      Each row carries its own query, gold answer, alternative valid
      answers, and a per-query retrieval pool (``search_results``).
    - Gold labels are TEXT answers (``answer`` + ``alt_ans``), not
      doc-ID relevance judgments.
    - The split between validation and test lives in an in-row
      ``split: int64`` column (0 = validation, 1 = test) inside the
      single HF ``train`` split.
    - Rich metadata (``domain``, ``question_type``, ``answer_type``)
      lets the loader filter rows down to a course-relevant slice
      before sandbox-cost caps kick in.

    Field-name defaults match the Quivr/CRAG canonical schema; override
    them per-spec if a derived dataset reshapes columns.
    """

    kind: Literal["crag"] = "crag"
    dataset: str = "Quivr/CRAG"
    use_split: Literal["validation", "test"] = "validation"

    # Sandbox cost cap. Applied AFTER the split + filter passes so the
    # cap reflects "this many filtered rows" not "this many raw rows".
    max_queries: int | None = None

    # Filters. ``answer_type_filter`` defaults to ["valid"] so the loader
    # excludes the no_answer / invalid rows by default — those are
    # rarely useful for a "did the learner answer correctly?" course.
    # The other filters default to None (no restriction).
    domain_filter: list[str] | None = None
    question_type_filter: list[str] | None = None
    answer_type_filter: list[str] | None = Field(default_factory=lambda: ["valid"])

    # Field-name overrides. Defaults match the Quivr/CRAG schema.
    query_id_field: str = "interaction_id"
    query_text_field: str = "query"
    answer_field: str = "answer"
    alt_ans_field: str = "alt_ans"
    search_results_field: str = "search_results"
    split_field: str = "split"
    domain_field: str = "domain"
    question_type_field: str = "question_type"
    answer_type_field: str = "answer_type"


# Discriminated union of benchmark layouts. ``kind`` is the
# discriminator: ``"huggingface"`` -> BeIR-shape, ``"crag"`` -> CRAG.
BenchmarkSource = Annotated[
    Union[HFBenchmarkSource, CRAGBenchmarkSource],
    Field(discriminator="kind"),
]


class CapabilityFlags(BaseModel):
    """What runtime primitives the learner's service needs access to.

    Set by the course planner from the goal text + endpoint shape. Drives
    capability-gated README sections, sandbox sidecar wiring (future),
    and family-scaffold selection. Defaults are all conservative — a
    spec that omits ``capabilities`` is treated as "the service needs
    nothing beyond Python and HTTP", so README docs for a managed LLM
    proxy or a Postgres sidecar never appear by accident.

    The four primitives covered here are course-agnostic; domain-specific
    primitives (HTML parsing libraries for a RAG course, scoring tools
    for a classifier course, ...) belong in family scaffolds, not the
    framework. Extension hook: additional flags can be added to this
    model without a spec migration as long as defaults stay conservative.
    """

    runtime_llm_required: bool = False
    structured_logging_required: bool = False
    durable_state_required: bool = False
    sidecar_database: Literal["postgres", "redis", "none"] = "none"


class CourseOutcomeSpec(BaseModel):
    """Top-level spec for a single-outcome course.

    Composes the new endpoint/quality-bar surface with the existing
    ``PackageType`` (from ``app.domain.registry``) and the existing
    ``ProjectContractSpec`` / ``ProjectRuntimePlanSpec`` (from
    ``app.domain.task_agent``). The latter two are optional so the
    planner can populate them in a later stage without blocking spec
    creation.
    """

    title: str = Field(min_length=5)
    goal: str = Field(min_length=20)
    starter_type: StarterType
    endpoints: list[EndpointContract] = Field(min_length=1)
    quality_bars: list[QualityBar] = Field(min_length=1)
    learning_path: list[LearningHint] = Field(default_factory=list)
    package_type: PackageType
    oracle_source: OracleSource = OracleSource.curated
    benchmark: BenchmarkSource | None = None
    project_contract: ProjectContractSpec | None = None
    runtime_plan: ProjectRuntimePlanSpec | None = None
    capabilities: CapabilityFlags = Field(default_factory=CapabilityFlags)

    @model_validator(mode="after")
    def _check_quality_bar_ids_unique(self) -> "CourseOutcomeSpec":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for bar in self.quality_bars:
            if bar.id in seen:
                duplicates.add(bar.id)
            seen.add(bar.id)
        if duplicates:
            raise ValueError(
                f"duplicate quality_bar id(s): {sorted(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def _check_endpoint_uniqueness(self) -> "CourseOutcomeSpec":
        seen: set[tuple[HttpMethod, str]] = set()
        duplicates: set[tuple[HttpMethod, str]] = set()
        for ep in self.endpoints:
            key = (ep.method, ep.path)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            pretty = sorted(f"{m.value} {p}" for m, p in duplicates)
            raise ValueError(f"duplicate endpoint(s): {pretty}")
        return self

    @model_validator(mode="after")
    def _check_learning_hints_reference_bars(self) -> "CourseOutcomeSpec":
        bar_ids = {bar.id for bar in self.quality_bars}
        missing = [
            hint.on_metric_fail
            for hint in self.learning_path
            if hint.on_metric_fail not in bar_ids
        ]
        if missing:
            raise ValueError(
                f"learning_path hint(s) reference unknown quality_bar id(s): "
                f"{sorted(set(missing))}"
            )
        return self
