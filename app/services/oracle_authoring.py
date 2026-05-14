"""Oracle authoring node for the single-outcome course pipeline (Wave 3).

The oracle author consumes a ``CourseOutcomeSpec`` and produces the
three-part authoring bundle a downstream grader is built from:

1. **Scenario YAML files** — curated learner-service interactions, one
   YAML per scenario, parseable by ``scenario_loader``. Coverage spans
   the framework's category taxonomy with a minimum set derived from
   the spec's quality_bars and endpoint shape.
2. **Reference implementation files** — a flat ``(relative_path,
   content)`` list that, when materialized under
   ``private/grader/_reference/``, builds a Docker image that satisfies
   every endpoint. The reference impl is the platform's own learner
   submission: it boots in the same sandbox and runs the same Dockerfile
   contract, used by the validator to confirm the quality bars are
   actually achievable.
3. **Setup data files** — gold-label sets, hidden corpora, seed lists,
   anything the scenarios reference via ``setup_data.*``. Hidden from
   the learner; lives under ``private/grader/_setup/``.

This module is parse + validate + retry only. It does NOT materialize
files to disk (the LangGraph integration in Wave 4 does that), and it
does NOT run the reference impl (``oracle_pass``'s job). It owns:

- the system prompt (with the registered rubric registry inlined),
- the user prompt (driven by the spec; on retry, prior validation
  failures are appended so Sonnet can repair),
- the Pydantic payload schema Sonnet fills,
- the validator chain (scenario YAML parse, rubric kinds registered,
  required categories covered, Dockerfile present, install manifest
  present),
- the retry loop (up to 3 attempts).

Validation chain — applied IN ORDER on each Sonnet response:

  1. ``reference_files`` is non-empty.
  2. ``reference_files`` includes a ``Dockerfile`` (path match).
  3. ``reference_files`` includes ``requirements.txt`` or
     ``pyproject.toml``.
  4. Every scenario YAML parses via ``yaml.safe_load``.
  5. Every parsed scenario constructs a ``Scenario`` model
     (which transitively validates rubric ``kind`` against
     ``RUBRIC_REGISTRY``).
  6. The set of scenario categories covers the minimum set required by
     the spec (always: happy_path / boundary / malformed_input;
     conditionally: out_of_scope when the spec carries an abstention
     bar; idempotency when any endpoint is a non-GET creator; and
     composition when there is more than one endpoint).

On any check failing, the loop assembles a diagnostic message naming
EVERY problem found in that response (so Sonnet sees the full picture,
not a one-at-a-time drip) and retries with that diagnostic appended to
the user prompt. After 3 attempts, ``OracleAuthoringError`` carries the
accumulated diagnostics across all attempts so a human reviewing the
log can see the full failure history.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, Field, ValidationError

from app.services import scenario_rubrics_base
from app.services.benchmark_loader import (
    BenchmarkBundle,
    BenchmarkLoadError,
    CRAGBenchmarkBundle,
    beir_bundle_to_visible_payload,
    crag_bundle_to_visible_payload,
    load_benchmark,
    load_crag_benchmark,
    split_beir_for_visibility,
    split_crag_for_visibility,
)
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    CRAGBenchmarkSource,
    HFBenchmarkSource,
    HttpMethod,
    JudgeKind,
)
from app.services.coursegen_logging import log_coursegen_event
from app.services.llm_router import LLMRouter, LLMTier, get_default_router
from app.services.scenario_loader import (
    Scenario,
    ScenarioLoadError,
    _ensure_rubrics_registered,
)


__all__ = [
    "GeneratedScenarioFile",
    "GeneratedReferenceFile",
    "GeneratedSetupFile",
    "OracleAuthoringResult",
    "OracleAuthoringError",
    "OracleAuthor",
]


# ---------------- typed authoring artifacts ----------------


class GeneratedScenarioFile(BaseModel):
    """One scenario file ready to land under ``scenarios/``.

    ``yaml_content`` is the raw YAML string; ``filename`` is the leaf
    name (no directories) used when the bundle is materialized.
    """

    filename: str
    yaml_content: str


class GeneratedReferenceFile(BaseModel):
    """One file in the reference implementation bundle.

    ``relative_path`` is relative to ``private/grader/_reference/`` and
    may contain forward-slash directory parts (e.g., ``app/main.py``).
    """

    relative_path: str
    content: str


class GeneratedSetupFile(BaseModel):
    """One file under ``private/grader/_setup/`` (gold labels, corpora,
    anything hidden from the learner). ``content`` is pre-serialized:
    JSON content is a JSON string, text is plain."""

    relative_path: str
    content: str


class OracleAuthoringResult(BaseModel):
    """Typed result of one ``OracleAuthor.author_oracle`` call.

    ``visible_sample_queries_json`` carries the learner-visible sample
    payload for benchmark-backed courses (CRAG / BeIR) and is ``None``
    for everything else. When set, the materializer lands it at
    ``public/examples/sample_queries.json`` alongside the visible
    self-test script. See
    :func:`benchmark_loader.crag_bundle_to_visible_payload` and
    :func:`benchmark_loader.beir_bundle_to_visible_payload` for the
    schemas.
    """

    scenarios: list[GeneratedScenarioFile]
    reference_files: list[GeneratedReferenceFile]
    setup_files: list[GeneratedSetupFile]
    notes: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    model_id: str = ""
    visible_sample_queries_json: str | None = None


class OracleAuthoringError(RuntimeError):
    """Raised when oracle authoring fails after all retries.

    The message includes a per-attempt diagnostic summary so a human
    can see which validation gates rejected which Sonnet response.
    """


# ---------------- LLM-emitted payload shape ----------------


class _ScenarioFilePayload(BaseModel):
    filename: str
    yaml_content: str
    # IDs of ``CourseOutcomeSpec.quality_bars`` this scenario contributes
    # evidence toward. The publish gate in ``oracle_validation`` blocks
    # publication when any spec bar is uncovered by every scenario, so the
    # authoring schema makes the field explicit — the LLM is told (via the
    # system prompt and the spec dump in the user prompt) what the valid
    # IDs are, and ``_validate_payload`` rejects a bundle that leaves any
    # spec bar unreferenced (Codex review #5 finding).
    quality_bar_ids: list[str] = Field(default_factory=list)


class _ReferenceFilePayload(BaseModel):
    relative_path: str
    content: str


class _SetupFilePayload(BaseModel):
    relative_path: str
    content: str


class _OracleAuthoringPayload(BaseModel):
    """Flat shape Sonnet fills.

    A close mirror of ``OracleAuthoringResult`` minus the cost / model
    fields (which the author adds after the call). The post-parse
    validator chain converts each entry into the typed result objects
    and runs scenario YAML / category / Dockerfile checks.
    """

    scenarios: list[_ScenarioFilePayload] = Field(default_factory=list)
    reference_files: list[_ReferenceFilePayload] = Field(default_factory=list)
    setup_files: list[_SetupFilePayload] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------- system prompt ----------------


def _build_system_prompt() -> str:
    """Compose the author's system prompt.

    The registered rubric kinds are pulled live from
    ``RUBRIC_REGISTRY`` so we never drift from the available rubric
    library when a new kind is added. The prompt also names a small set
    of forbidden tokens (FAISS, BM25, RRF) that the planner already
    banned — the author MUST NOT re-introduce them into the scenarios
    or the reference impl's required-tech assertions.
    """
    _ensure_rubrics_registered()
    kinds = sorted(scenario_rubrics_base.RUBRIC_REGISTRY.keys())
    kinds_str = ", ".join(kinds)

    return (
        "You author the GRADER for a single-outcome engineering course. "
        "Your output is the oracle bundle: scenario YAML files, a "
        "reference implementation, and any hidden setup data the "
        "scenarios depend on.\n\n"
        "The reference implementation is a peer of the learner: it "
        "ships a Dockerfile, exposes the same endpoints from the spec, "
        "and runs in the same sandbox. It is the platform's own learner "
        "submission, used to verify that the quality bars are actually "
        "achievable.\n\n"
        "The scenarios you emit must be thorough and stub-resistant. "
        "They must catch hardcoded responses, cover the boundary of "
        "acceptable behavior, and exercise the spec's abstention bar "
        "where one exists. Include at least one scenario that fires "
        "the same query twice with subtle variation so a hard-coded "
        "implementation cannot pass.\n\n"
        "Scenario category taxonomy (set ``category`` to one of):\n"
        "  - happy_path (required)\n"
        "  - boundary (required)\n"
        "  - malformed_input (required)\n"
        "  - out_of_scope (required when the spec includes an "
        "abstention quality bar)\n"
        "  - idempotency (required when the spec includes a "
        "non-idempotent creator endpoint — POST creates, etc.)\n"
        "  - composition (required when endpoints chain — e.g., POST "
        "then GET then DELETE on the same resource)\n"
        "  - adversarial (optional but recommended for the stub_resistance bar)\n"
        "Aim for 15-25 scenarios total. Balance coverage against cost; "
        "omit scenarios that would not move any quality bar.\n\n"
        f"Allowed rubric kinds (the registered rubric library): {kinds_str}. "
        "Every ``kind`` in every scenario YAML MUST be one of these. "
        "For RAG-shaped courses include LLM-judged faithfulness "
        "rubrics (``llm_judge_coverage``) in answer scenarios and "
        "``oracle_set_overlap`` rubrics in retrieval scenarios; "
        "reference the gold Q+A pairs in ``_setup/`` via "
        "``setup_data.gold.<query_id>`` paths.\n\n"
        "Setup data files (``setup_files``) live under ``_setup/`` and "
        "are HIDDEN from the learner. Put every gold-label file there. "
        "Do NOT inline gold labels into the scenario YAML — the YAML "
        "may surface to the learner at grader-debug time.\n\n"
        "Quality-bar IDs and rubric ``kind`` values must already be "
        "specific (``faithfulness``, ``recall_at_5``, etc.). Do not "
        "invent new rubric kinds. Use only the registered list above.\n\n"
        "Scenario-to-quality-bar coverage. Every scenario carries a "
        "``quality_bar_ids: list[str]`` field naming which spec "
        "``quality_bars.id`` values it contributes evidence toward. Two "
        "rules MUST hold across the bundle you emit:\n"
        "  - every scenario references at least one spec quality_bar.id "
        "via ``quality_bar_ids`` (no scenario contributes zero evidence);\n"
        "  - every spec quality_bar.id is referenced by at least one "
        "scenario's ``quality_bar_ids`` (no spec bar is left uncovered).\n"
        "Multiple scenarios MAY reference the same bar id (and SHOULD, "
        "when the bar is hard to clear in a single scenario). The "
        "user prompt lists the valid spec bar IDs explicitly — use "
        "exactly those strings; do not invent new ones.\n\n"
        "Reference implementation: include a ``Dockerfile`` and a "
        "Python install manifest (``requirements.txt`` or "
        "``pyproject.toml``). Source files use forward-slash relative "
        "paths (e.g., ``app/main.py``).\n\n"
        "Forbidden tokens. Do not prescribe tech in either the scenario "
        "rubrics or the reference impl as a REQUIREMENT for the "
        "learner: do not require FAISS, do not require BM25, do not "
        "require RRF. Your reference impl may use any tech to clear "
        "the bars; the scenarios you author must judge the OUTCOME, "
        "not how the learner gets there.\n\n"
        "Return JSON matching the schema. Each scenario YAML must parse "
        "as a Scenario object (id, description, category, trace, "
        "rubrics). Each reference file uses a forward-slash "
        "``relative_path``. Each setup file likewise."
    )


# Note: _build_system_prompt() is called fresh on every author_oracle()
# invocation rather than snapshotted at module import. That keeps the
# prompt's rubric-kind list in sync with whatever rubrics are registered
# at call time — important under pytest where rubric-registry-touching
# tests may run before this module is exercised.


# ---------------- author ----------------


# Always-required scenario categories.
_BASE_REQUIRED_CATEGORIES: tuple[str, ...] = (
    "happy_path",
    "boundary",
    "malformed_input",
)

# Substrings that flag a quality bar as an abstention/refusal bar.
_ABSTENTION_MARKERS: tuple[str, ...] = (
    "abstain",
    "abstention",
    "refusal",
    "decline",
    "out_of_scope",
)


def _spec_implies_abstention(spec: CourseOutcomeSpec) -> bool:
    for bar in spec.quality_bars:
        haystack = f"{bar.id} {bar.metric_description}".lower()
        if any(marker in haystack for marker in _ABSTENTION_MARKERS):
            return True
    return False


def _spec_implies_idempotency(spec: CourseOutcomeSpec) -> bool:
    # Any non-GET / non-HEAD endpoint is a side-effecting endpoint
    # whose POST/PUT/DELETE behavior the suite should pin down.
    creator_methods = {HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH, HttpMethod.DELETE}
    return any(ep.method in creator_methods for ep in spec.endpoints)


def _spec_implies_composition(spec: CourseOutcomeSpec) -> bool:
    return len(spec.endpoints) > 1


def _required_categories(spec: CourseOutcomeSpec) -> set[str]:
    required = set(_BASE_REQUIRED_CATEGORIES)
    if _spec_implies_abstention(spec):
        required.add("out_of_scope")
    if _spec_implies_idempotency(spec):
        required.add("idempotency")
    if _spec_implies_composition(spec):
        required.add("composition")
    return required


def _benchmark_to_setup_files(
    bundle: BenchmarkBundle,
) -> list[GeneratedSetupFile]:
    """Render a :class:`BenchmarkBundle` as three ``GeneratedSetupFile``
    entries matching the shape ``serialize_benchmark_to_setup`` would
    write to disk.

    Kept in sync with ``serialize_benchmark_to_setup``: any change to
    one must mirror in the other so the in-memory result and the
    on-disk materialization agree.
    """
    corpus_lines = "\n".join(
        json.dumps(doc.model_dump(), sort_keys=True) for doc in bundle.corpus
    )
    if corpus_lines:
        corpus_lines += "\n"
    query_lines = "\n".join(
        json.dumps(q.model_dump(), sort_keys=True) for q in bundle.queries
    )
    if query_lines:
        query_lines += "\n"
    gold: dict[str, dict[str, list[str]]] = {}
    for query_id, doc_scores in bundle.qrels.items():
        gold[query_id] = {"expected_doc_ids": sorted(doc_scores.keys())}
    gold_text = json.dumps(gold, indent=2, sort_keys=True)
    return [
        GeneratedSetupFile(
            relative_path="corpus.jsonl", content=corpus_lines
        ),
        GeneratedSetupFile(
            relative_path="queries.jsonl", content=query_lines
        ),
        GeneratedSetupFile(
            relative_path="gold_qa.json", content=gold_text
        ),
    ]


def _crag_bundle_to_setup_files(
    bundle: CRAGBenchmarkBundle,
) -> list[GeneratedSetupFile]:
    """Render a :class:`CRAGBenchmarkBundle` as three
    ``GeneratedSetupFile`` entries matching the shape
    ``serialize_crag_to_setup`` would write to disk.

    Output: ``queries.jsonl`` (full per-row record),
    ``gold_answers.json`` (semantic-equivalence reference), and
    ``search_results_index.json`` (per-query retrieval pool). No global
    ``corpus.jsonl`` — CRAG's retrieval pool is per-query.
    """
    query_lines = "\n".join(
        json.dumps(q.model_dump(), sort_keys=True) for q in bundle.queries
    )
    if query_lines:
        query_lines += "\n"

    gold: dict[str, dict[str, Any]] = {}
    for q in bundle.queries:
        gold[q.query_id] = {
            "answer": q.answer,
            "alt_ans": list(q.alt_ans),
            "answer_type": q.answer_type,
            "question_type": q.question_type,
        }
    gold_text = json.dumps(gold, indent=2, sort_keys=True)

    index: dict[str, list[dict[str, Any]]] = {
        q.query_id: list(q.search_results) for q in bundle.queries
    }
    index_text = json.dumps(index, indent=2, sort_keys=True)

    return [
        GeneratedSetupFile(
            relative_path="queries.jsonl", content=query_lines
        ),
        GeneratedSetupFile(
            relative_path="gold_answers.json", content=gold_text
        ),
        GeneratedSetupFile(
            relative_path="search_results_index.json", content=index_text
        ),
    ]


class OracleAuthor:
    """LLM-driven author for the oracle bundle.

    One ``OracleAuthor`` instance can be reused across many specs. The
    router is injected for testability; the production wiring goes
    through ``get_default_router()``.
    """

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        *,
        router: LLMRouter | None = None,
        request_timeout_s: float = 600.0,
    ) -> None:
        self._router = router
        self.request_timeout_s = request_timeout_s

    # ----- public API -----

    def author_oracle(self, spec: CourseOutcomeSpec) -> OracleAuthoringResult:
        """Produce scenarios + reference impl + setup data for the spec.

        Retries up to 3 times. Each retry appends the accumulated
        validation failures to the user prompt so Sonnet can repair the
        bundle without re-deriving everything from scratch. Raises
        :class:`OracleAuthoringError` on unrecoverable failure.

        When ``spec.benchmark`` is set, the setup data step is taken over
        by the benchmark loader: the corpus, queries and qrels are pulled
        from Hugging Face and pre-emitted as ``GeneratedSetupFile``
        entries, the LLM prompt is told NOT to author setup files, and
        any ``setup_files`` the LLM emits anyway are discarded. The LLM
        still authors scenarios and the reference implementation.
        """
        router = self._router if self._router is not None else get_default_router()
        model_id = router.model_id_for(LLMTier.sonnet)

        # Preload the benchmark BEFORE invoking the LLM — if the loader
        # raises we fail loudly without paying for any Sonnet calls.
        # There is no silent fallback to LLM-synthesized setup data:
        # callers who set ``spec.benchmark`` are opting OUT of LLM
        # authorship for that artifact, and a load failure must surface.
        # Dispatch on the discriminated union: BeIR-shape goes through
        # ``load_benchmark``; CRAG-shape goes through
        # ``load_crag_benchmark`` and produces a different setup-file
        # set (no global corpus, per-query retrieval pool).
        benchmark_bundle: BenchmarkBundle | None = None
        crag_bundle: CRAGBenchmarkBundle | None = None
        benchmark_setup_files: list[GeneratedSetupFile] = []
        # Pre-serialized JSON payload for ``public/examples/sample_queries.json``.
        # Populated only for benchmark-backed courses; ``None`` otherwise.
        visible_sample_queries_json: str | None = None
        if isinstance(spec.benchmark, CRAGBenchmarkSource):
            try:
                crag_bundle = load_crag_benchmark(spec.benchmark)
            except BenchmarkLoadError as exc:
                log_coursegen_event(
                    "oracle_authoring_benchmark_load_failed",
                    benchmark=spec.benchmark.dataset,
                    error=str(exc),
                )
                raise OracleAuthoringError(
                    f"failed to load CRAG benchmark "
                    f"'{spec.benchmark.dataset}': {exc}"
                ) from exc
            # Split off a small visible sample BEFORE rendering the
            # hidden setup files so the hidden bundle holds only the
            # "kept" rows. The visible payload is JSON-serialized once
            # here and threaded through ``OracleAuthoringResult`` to
            # the materializer.
            visible_crag, hidden_crag = split_crag_for_visibility(crag_bundle)
            benchmark_setup_files = _crag_bundle_to_setup_files(hidden_crag)
            visible_sample_queries_json = json.dumps(
                crag_bundle_to_visible_payload(visible_crag),
                indent=2,
                sort_keys=True,
            )
            log_coursegen_event(
                "oracle_authoring_benchmark_loaded",
                benchmark=spec.benchmark.dataset,
                query_count=len(crag_bundle.queries),
                visible_sample_count=len(visible_crag.queries),
                hidden_query_count=len(hidden_crag.queries),
                layout="crag",
            )
        elif isinstance(spec.benchmark, HFBenchmarkSource):
            try:
                benchmark_bundle = load_benchmark(spec.benchmark)
            except BenchmarkLoadError as exc:
                log_coursegen_event(
                    "oracle_authoring_benchmark_load_failed",
                    benchmark=spec.benchmark.corpus_dataset,
                    error=str(exc),
                )
                raise OracleAuthoringError(
                    f"failed to load benchmark "
                    f"'{spec.benchmark.corpus_dataset}': {exc}"
                ) from exc
            visible_beir, hidden_beir = split_beir_for_visibility(benchmark_bundle)
            benchmark_setup_files = _benchmark_to_setup_files(hidden_beir)
            visible_sample_queries_json = json.dumps(
                beir_bundle_to_visible_payload(visible_beir),
                indent=2,
                sort_keys=True,
            )
            log_coursegen_event(
                "oracle_authoring_benchmark_loaded",
                benchmark=spec.benchmark.corpus_dataset,
                corpus_size=len(benchmark_bundle.corpus),
                query_count=len(benchmark_bundle.queries),
                visible_sample_count=len(visible_beir.queries),
                hidden_query_count=len(hidden_beir.queries),
                layout="beir",
            )

        base_user_prompt = self._build_user_prompt(
            spec,
            benchmark_loaded=benchmark_bundle is not None,
            crag_loaded=crag_bundle is not None,
        )
        required_categories = _required_categories(spec)

        total_cost = 0.0
        attempt_diagnostics: list[str] = []

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            user_prompt = base_user_prompt
            if attempt_diagnostics:
                user_prompt = (
                    base_user_prompt
                    + "\n\n## Prior validation failures\n"
                    + "Your last attempt failed the following validators. "
                    + "Repair the bundle and retry — keep the parts that "
                    + "passed, fix only the listed problems.\n\n"
                    + "\n".join(
                        f"Attempt {idx}:\n{diag}"
                        for idx, diag in enumerate(attempt_diagnostics, start=1)
                    )
                )

            log_coursegen_event(
                "oracle_authoring_attempt_started",
                attempt=attempt,
                max_attempts=self.MAX_ATTEMPTS,
                spec_title=spec.title,
            )

            try:
                response = router.parse_structured(
                    tier=LLMTier.sonnet,
                    system=_build_system_prompt(),
                    user=user_prompt,
                    text_format=_OracleAuthoringPayload,
                    request_timeout_s=self.request_timeout_s,
                )
            except Exception as exc:  # pragma: no cover - exercised in test 6
                attempt_diagnostics.append(f"router error: {exc}")
                log_coursegen_event(
                    "oracle_authoring_attempt_failed",
                    attempt=attempt,
                    error=str(exc),
                    error_kind="router",
                )
                if attempt >= self.MAX_ATTEMPTS:
                    raise OracleAuthoringError(
                        self._format_error_message(attempt_diagnostics)
                    ) from exc
                continue

            total_cost += self._cost_from_response(response)

            payload = getattr(response, "parsed", None) or getattr(
                response, "output_parsed", None
            )
            if payload is None:
                attempt_diagnostics.append("router returned no parsed payload")
                log_coursegen_event(
                    "oracle_authoring_attempt_failed",
                    attempt=attempt,
                    error_kind="no_payload",
                )
                if attempt >= self.MAX_ATTEMPTS:
                    raise OracleAuthoringError(
                        self._format_error_message(attempt_diagnostics)
                    )
                continue

            failures = self._validate_payload(
                payload,
                required_categories,
                spec=spec,
                benchmark_loaded=(
                    benchmark_bundle is not None or crag_bundle is not None
                ),
            )
            if failures:
                diag = "\n".join(f"- {msg}" for msg in failures)
                attempt_diagnostics.append(diag)
                log_coursegen_event(
                    "oracle_authoring_attempt_failed",
                    attempt=attempt,
                    error_kind="validation",
                    failure_count=len(failures),
                )
                if attempt >= self.MAX_ATTEMPTS:
                    raise OracleAuthoringError(
                        self._format_error_message(attempt_diagnostics)
                    )
                continue

            log_coursegen_event(
                "oracle_authoring_attempt_completed",
                attempt=attempt,
                scenarios=len(payload.scenarios),
                reference_files=len(payload.reference_files),
                setup_files=len(payload.setup_files),
            )
            return self._build_result(
                payload,
                cost_usd=total_cost,
                model_id=model_id,
                benchmark_setup_files=benchmark_setup_files,
                visible_sample_queries_json=visible_sample_queries_json,
            )

        # Defensive: every branch above either ``return``s or ``raise``s.
        raise OracleAuthoringError(
            "Oracle authoring exhausted retries without producing a result."
        )

    # ----- prompt assembly -----

    def _build_user_prompt(
        self,
        spec: CourseOutcomeSpec,
        *,
        benchmark_loaded: bool = False,
        crag_loaded: bool = False,
    ) -> str:
        payload = {
            "title": spec.title,
            "goal": spec.goal,
            "starter_type": spec.starter_type.value,
            "endpoints": [
                {
                    "method": ep.method.value,
                    "path": ep.path,
                    "request_schema": ep.request_schema,
                    "response_schema": ep.response_schema,
                    "description": ep.description,
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
                {"on_metric_fail": hint.on_metric_fail, "hint": hint.hint}
                for hint in spec.learning_path
            ],
            "package_type": spec.package_type.value,
            "implied_categories": sorted(_required_categories(spec)),
            "abstention_bar_present": _spec_implies_abstention(spec),
            # Flat list of the valid ``quality_bar_ids`` for the scenarios
            # you emit. Every scenario MUST tag itself with one or more of
            # these IDs via ``quality_bar_ids``; every ID below MUST be
            # referenced by at least one scenario.
            "valid_quality_bar_ids": [bar.id for bar in spec.quality_bars],
        }
        preamble = (
            "Author the oracle bundle for the following course spec. "
            "Cover every required category listed in "
            "``implied_categories``. The Dockerfile in your reference "
            "impl must boot a service that satisfies every endpoint. "
            "Every scenario YAML you emit MUST set ``quality_bar_ids`` "
            "to one or more entries from ``valid_quality_bar_ids`` "
            "below, and every entry in ``valid_quality_bar_ids`` MUST be "
            "referenced by at least one scenario."
        )
        if (
            crag_loaded
            and isinstance(spec.benchmark, CRAGBenchmarkSource)
        ):
            # CRAG path: setup files are preloaded from Quivr/CRAG and
            # have a fundamentally different shape from BeIR. Tell the
            # model where the files live, what's in them, and which
            # rubric to use per question_type.
            preamble += (
                "\n\nThe setup data has been preloaded from the "
                f"CRAG benchmark '{spec.benchmark.dataset}' "
                f"(use_split='{spec.benchmark.use_split}'). Do NOT "
                "author setup files; three files are already "
                "materialized under ``_setup/``:\n"
                "  - ``_setup/queries.jsonl`` — one row per query "
                "(query_id, query, answer, alt_ans, search_results, "
                "domain, question_type, answer_type).\n"
                "  - ``_setup/gold_answers.json`` — "
                "``{query_id: {answer, alt_ans, answer_type, "
                "question_type}}``, the grader's semantic-equivalence "
                "reference. Use this from the "
                "``llm_judge_semantic_eq`` rubric via "
                "``setup_data.gold_answers.<query_id>.answer`` (and "
                "``.alt_ans`` for the alternative list).\n"
                "  - ``_setup/search_results_index.json`` — "
                "``{query_id: [search_result, ...]}``, the per-query "
                "retrieval pool. Retrieval is PER-QUERY in CRAG (no "
                "global corpus); scenarios that exercise retrieval "
                "should drive the learner's service against the pool "
                "for the relevant query_id.\n\n"
                "Rubric selection guidance for CRAG scenarios:\n"
                "  - For scenarios judging a question with "
                "``answer_type == 'valid'`` (the common case), use "
                "the ``llm_judge_semantic_eq`` rubric. Set ``target`` "
                "to the learner's answer field; set ``gold_path`` to "
                "``setup_data.gold_answers.<query_id>.answer``; "
                "optionally set ``alt_path`` to "
                "``setup_data.gold_answers.<query_id>.alt_ans``.\n"
                "  - For scenarios judging a question with "
                "``question_type == 'false_premise'``, use the "
                "``llm_judge_false_premise`` rubric — the correct "
                "answer here is to REFUSE or identify the false "
                "premise, not to answer. Optionally set "
                "``expected_falsity_path`` to "
                "``setup_data.gold_answers.<query_id>.alt_ans`` so "
                "the judge has explicit grounding.\n\n"
                "Leave ``setup_files`` as an empty list in your "
                "response; any setup files you emit will be discarded."
            )
        elif benchmark_loaded and spec.benchmark is not None:
            # BeIR path: setup data is preloaded as a global corpus +
            # queries + qrels triple.
            preamble += (
                "\n\nThe setup data has been preloaded from the "
                f"Hugging Face benchmark '{spec.benchmark.corpus_dataset}' "
                f"(qrels: '{spec.benchmark.qrels_dataset}'). Do NOT "
                "author setup files; the corpus, queries, and gold "
                "relevance judgments are already materialized at "
                "``_setup/corpus.jsonl``, ``_setup/queries.jsonl``, and "
                "``_setup/gold_qa.json``. You author ONLY scenarios "
                "and the reference implementation. Reference the "
                "preloaded gold sets via ``setup_data.gold_qa.<query_id>"
                ".expected_doc_ids`` in your ``oracle_set_overlap`` "
                "rubrics. Leave ``setup_files`` as an empty list in "
                "your response; any setup files you emit will be "
                "discarded."
            )
        return preamble + "\n\n" + json.dumps(payload, indent=2)

    # ----- validation -----

    def _validate_payload(
        self,
        payload: _OracleAuthoringPayload,
        required_categories: set[str],
        *,
        spec: CourseOutcomeSpec,
        benchmark_loaded: bool = False,
    ) -> list[str]:
        """Return a list of human-readable failure messages.

        Empty list means the payload is good. Each failure is a one-line
        bullet ready to splice into the retry user prompt. Validation
        order is fixed (see module docstring) so the diagnostics read
        the same way every time.

        When ``benchmark_loaded`` is True, the LLM-emitted ``setup_files``
        slot is ignored entirely — the benchmark loader is authoritative
        for that artifact and any setup files the LLM emits will be
        discarded by ``_build_result``. The other validation steps
        (reference impl shape, scenario YAML, category coverage,
        quality-bar coverage) still apply.
        """
        # ``benchmark_loaded`` is consumed by ``_build_result`` (which
        # drops LLM-emitted setup files in favor of the benchmark
        # output); the current validation chain has nothing to skip
        # for it (we never validated setup file contents). The
        # parameter is kept in the signature so future setup-file
        # validators can branch on it without re-threading the call
        # sites.
        _ = benchmark_loaded
        failures: list[str] = []

        # 1-3: reference bundle shape.
        ref_paths = [f.relative_path for f in payload.reference_files]
        if not ref_paths:
            failures.append("reference_files is empty; must include at least Dockerfile + install manifest")
        if "Dockerfile" not in ref_paths:
            failures.append(
                "reference_files is missing 'Dockerfile' (exact path required)"
            )
        if not any(p in {"requirements.txt", "pyproject.toml"} for p in ref_paths):
            failures.append(
                "reference_files must include 'requirements.txt' or 'pyproject.toml'"
            )

        # 4-5: scenario YAML well-formedness and rubric kind validity.
        seen_categories: set[str] = set()
        # Tracks which spec ``quality_bars.id`` values any scenario
        # references — used by the coverage gate below. Repeats across
        # scenarios are allowed and expected (multiple scenarios may
        # contribute to the same bar); we just union them.
        referenced_bar_ids: set[str] = set()
        for sf in payload.scenarios:
            try:
                data = yaml.safe_load(sf.yaml_content)
            except yaml.YAMLError as exc:
                failures.append(
                    f"scenario '{sf.filename}' has malformed YAML: {exc}"
                )
                continue
            if not isinstance(data, dict):
                failures.append(
                    f"scenario '{sf.filename}' did not produce a YAML mapping"
                )
                continue
            try:
                scenario = Scenario.model_validate(data)
            except ScenarioLoadError as exc:
                # Unknown rubric kind surfaces here.
                failures.append(
                    f"scenario '{sf.filename}' failed validation: {exc}"
                )
                continue
            except ValidationError as exc:
                failures.append(
                    f"scenario '{sf.filename}' failed Pydantic validation: {exc}"
                )
                continue
            except Exception as exc:  # defensive: anything else
                failures.append(
                    f"scenario '{sf.filename}' could not be constructed: {exc}"
                )
                continue
            seen_categories.add(scenario.category)
            referenced_bar_ids.update(scenario.quality_bar_ids)

        # 6: category coverage.
        missing_categories = sorted(required_categories - seen_categories)
        if missing_categories:
            failures.append(
                "missing required scenario categories: "
                f"{missing_categories} — every spec needs happy_path, "
                f"boundary, malformed_input; abstention specs also need "
                f"out_of_scope; multi-endpoint specs need composition; "
                f"creator endpoints need idempotency"
            )

        # 7: quality_bar coverage gate (Codex review #5 finding).
        # The downstream publish gate in ``oracle_validation`` blocks
        # publication when any spec-declared bar is uncovered. Catch it
        # here so the LLM gets a targeted diagnostic and the next attempt
        # can repair, instead of letting the broken bundle slip through to
        # exhaust the grader-retry budget.
        spec_bar_ids = [bar.id for bar in spec.quality_bars]
        uncovered_bars = [
            bar_id for bar_id in spec_bar_ids if bar_id not in referenced_bar_ids
        ]
        if uncovered_bars:
            failures.append(
                "uncovered spec quality_bar.id values: "
                f"{uncovered_bars} — every spec quality_bar.id MUST be "
                "referenced by at least one scenario's quality_bar_ids. "
                "Add or amend scenarios so the listed bar IDs are each "
                "named in some scenario's quality_bar_ids."
            )

        return failures

    # ----- result construction -----

    def _build_result(
        self,
        payload: _OracleAuthoringPayload,
        *,
        cost_usd: float,
        model_id: str,
        benchmark_setup_files: list[GeneratedSetupFile] | None = None,
        visible_sample_queries_json: str | None = None,
    ) -> OracleAuthoringResult:
        # When the benchmark loader produced setup files, those are
        # authoritative — any setup files the LLM emitted are discarded
        # to keep the contract simple (one source of truth per
        # artifact).
        if benchmark_setup_files:
            setup_files = list(benchmark_setup_files)
        else:
            setup_files = [
                GeneratedSetupFile(
                    relative_path=f.relative_path, content=f.content
                )
                for f in payload.setup_files
            ]
        return OracleAuthoringResult(
            scenarios=[
                GeneratedScenarioFile(
                    filename=s.filename, yaml_content=s.yaml_content
                )
                for s in payload.scenarios
            ],
            reference_files=[
                GeneratedReferenceFile(
                    relative_path=f.relative_path, content=f.content
                )
                for f in payload.reference_files
            ],
            setup_files=setup_files,
            notes=list(payload.notes),
            cost_usd=cost_usd,
            model_id=model_id,
            visible_sample_queries_json=visible_sample_queries_json,
        )

    @staticmethod
    def _cost_from_response(response: Any) -> float:
        """Pull the per-call cost from the router's usage summary.

        Mirrors the convention used elsewhere in the codebase: prefer
        the summary's own ``estimated_cost_usd`` when present; otherwise
        treat as 0 (the caller may not have a price table for this
        model id, and we don't want to double-bill from token counts
        here)."""
        summary = getattr(response, "usage_summary", None)
        if summary is None:
            return 0.0
        cost = getattr(summary, "estimated_cost_usd", 0.0) or 0.0
        return float(cost)

    @staticmethod
    def _format_error_message(diagnostics: Iterable[str]) -> str:
        joined = "\n\n".join(
            f"Attempt {idx}:\n{diag}"
            for idx, diag in enumerate(diagnostics, start=1)
        )
        return (
            "Oracle authoring failed after "
            f"{OracleAuthor.MAX_ATTEMPTS} attempts.\n\n"
            f"{joined}"
        )


# Silence the unused-import lint: JudgeKind is exposed for type-hint
# consistency in callers that build specs alongside this module.
# ``HFBenchmarkSource`` is similarly re-exported so callers that build
# benchmark-bearing specs alongside this module can import it from
# either location.
_ = JudgeKind
_ = HFBenchmarkSource
