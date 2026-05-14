"""Tests for the ``oracle_authoring`` node (Wave 3).

The oracle author consumes a ``CourseOutcomeSpec`` and produces the
authoring bundle for a course's grader: scenario YAML files, a reference
implementation (Dockerfile + sources), and any setup data files (gold
labels, corpora, seeds). It runs the LLM router at Sonnet tier and
retries up to 3 times on malformed output. There is no deterministic
fallback — unrecoverable failures raise ``OracleAuthoringError``.

All tests use a fake router; no real LLM is contacted.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.domain.registry import PackageType
from app.services import scenario_loader
from app.services.benchmark_loader import (
    BenchmarkBundle,
    BenchmarkDocument,
    BenchmarkLoadError,
    BenchmarkQuery,
    CRAGBenchmarkBundle,
    CRAGQuery,
)
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    CRAGBenchmarkSource,
    EndpointContract,
    HFBenchmarkSource,
    HttpMethod,
    JudgeKind,
    QualityBar,
    StarterType,
)
from app.services.oracle_authoring import (
    GeneratedReferenceFile,
    GeneratedScenarioFile,
    GeneratedSetupFile,
    OracleAuthor,
    OracleAuthoringError,
    OracleAuthoringResult,
    _OracleAuthoringPayload,
)


# Ensure the rubric registry is populated before the author imports it.
scenario_loader._ensure_rubrics_registered()


# ---------------- helpers ----------------


def _spec_with_abstention() -> CourseOutcomeSpec:
    """A RAG-shaped spec with an abstention quality bar — this triggers
    the ``out_of_scope`` category requirement."""
    return CourseOutcomeSpec(
        title="Grounded retrieval over a small corpus",
        goal=(
            "Build a retrieval-and-answer service that ingests a small "
            "document corpus and returns grounded answers with citations."
        ),
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/ingest",
                request_schema={"documents": "list[dict]"},
                response_schema={"corpus_id": "str", "chunk_count": "int"},
                description="Ingest documents and produce retrievable chunks.",
            ),
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"corpus_id": "str", "question": "str"},
                response_schema={"answer": "str", "citations": "list[str]"},
                description="Return a grounded answer plus cited sources.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answer faithfulness to cited sources.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=10,
            ),
            QualityBar(
                id="recall_at_5",
                metric_description="Recall@5 over the labeled retrieval oracle.",
                threshold=">= 0.7",
                judged_by=JudgeKind.oracle_set_overlap,
                sample_size=10,
            ),
            QualityBar(
                id="abstention_precision",
                metric_description=(
                    "When the corpus does not support the question, the "
                    "system declines to answer."
                ),
                threshold=">= 0.95",
                judged_by=JudgeKind.llm_haiku,
                sample_size=5,
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
    )


def _scenario_yaml(
    scenario_id: str,
    *,
    category: str = "happy_path",
    rubric_kind: str = "schema_match",
    rubric_extras: dict | None = None,
    quality_bar_ids: list[str] | None = None,
) -> str:
    rubric_block: dict[str, object] = {"kind": rubric_kind}
    if rubric_kind == "schema_match":
        rubric_block.update({"target": "resp.body", "must_have_fields": ["answer"]})
    if rubric_extras:
        rubric_block.update(rubric_extras)
    doc: dict[str, object] = {
        "id": scenario_id,
        "description": f"Scenario {scenario_id} ({category}).",
        "category": category,
        "trace": [
            {
                "id": "ask",
                "method": "POST",
                "path": "/answer",
                "body": {"corpus_id": "c1", "question": "What is RAG?"},
                "expect": {"status": 200},
            }
        ],
        "rubrics": [rubric_block],
        "quality_bar_ids": list(quality_bar_ids) if quality_bar_ids else [],
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _valid_payload_dict(*, include_out_of_scope: bool = True) -> dict:
    """A complete, valid authoring payload covering the required
    categories for the abstention-bearing spec.

    The spec used by these tests has multiple endpoints AND non-GET
    creator endpoints AND an abstention bar, so the required category
    set is happy_path + boundary + malformed_input + out_of_scope +
    idempotency + composition. The fixture covers all of these so the
    happy-path test passes; individual tests then drop categories to
    exercise the validators.
    """
    scenarios: list[dict[str, str]] = [
        {
            "filename": "happy_qa.yaml",
            "yaml_content": _scenario_yaml(
                "happy_qa",
                category="happy_path",
                quality_bar_ids=["faithfulness", "recall_at_5"],
            ),
        },
        {
            "filename": "boundary_long_question.yaml",
            "yaml_content": _scenario_yaml(
                "boundary_long_question",
                category="boundary",
                quality_bar_ids=["faithfulness"],
            ),
        },
        {
            "filename": "malformed_missing_corpus_id.yaml",
            "yaml_content": _scenario_yaml(
                "malformed_missing_corpus_id",
                category="malformed_input",
                quality_bar_ids=["faithfulness"],
            ),
        },
        {
            "filename": "idempotency_repeat_ingest.yaml",
            "yaml_content": _scenario_yaml(
                "idempotency_repeat_ingest",
                category="idempotency",
                quality_bar_ids=["faithfulness"],
            ),
        },
        {
            "filename": "composition_ingest_then_answer.yaml",
            "yaml_content": _scenario_yaml(
                "composition_ingest_then_answer",
                category="composition",
                quality_bar_ids=["recall_at_5"],
            ),
        },
    ]
    if include_out_of_scope:
        scenarios.append(
            {
                "filename": "out_of_scope_unrelated.yaml",
                "yaml_content": _scenario_yaml(
                    "out_of_scope_unrelated",
                    category="out_of_scope",
                    quality_bar_ids=["abstention_precision"],
                ),
            }
        )
    return {
        "scenarios": scenarios,
        "reference_files": [
            {
                "relative_path": "Dockerfile",
                "content": "FROM python:3.12-slim\nCOPY . /app\nWORKDIR /app\nRUN pip install -r requirements.txt\nCMD [\"python\", \"-m\", \"app.main\"]\n",
            },
            {
                "relative_path": "requirements.txt",
                "content": "fastapi\nuvicorn\n",
            },
            {
                "relative_path": "app/main.py",
                "content": "from fastapi import FastAPI\napp = FastAPI()\n",
            },
        ],
        "setup_files": [
            {
                "relative_path": "gold_qa.json",
                "content": "{\"queries\": []}",
            }
        ],
        "notes": ["Coverage spans happy/boundary/malformed/abstention."],
    }


def _router_returning(payload_dict: dict) -> MagicMock:
    payload = _OracleAuthoringPayload.model_validate(payload_dict)
    router = MagicMock()
    router.parse_structured.return_value = SimpleNamespace(
        parsed=payload,
        output_parsed=payload,
        usage=None,
        usage_summary=SimpleNamespace(
            estimated_cost_usd=0.42,
            input_tokens=1000,
            output_tokens=2000,
        ),
    )
    return router


def _router_sequence(*payload_dicts_or_exceptions) -> MagicMock:
    """Build a router whose successive ``parse_structured`` calls yield
    the next entry — either a payload dict (validated to
    ``_OracleAuthoringPayload``) or an Exception to raise."""
    router = MagicMock()

    def _side_effect(*_args, **_kwargs):
        nxt = items.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        payload = _OracleAuthoringPayload.model_validate(nxt)
        return SimpleNamespace(
            parsed=payload,
            output_parsed=payload,
            usage=None,
            usage_summary=SimpleNamespace(
                estimated_cost_usd=0.10,
                input_tokens=500,
                output_tokens=1000,
            ),
        )

    items = list(payload_dicts_or_exceptions)
    router.parse_structured.side_effect = _side_effect
    return router


# ---------------- 1. result model ----------------


def test_oracle_authoring_result_model_construction() -> None:
    result = OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(filename="x.yaml", yaml_content="id: x")
        ],
        reference_files=[
            GeneratedReferenceFile(relative_path="Dockerfile", content="FROM scratch")
        ],
        setup_files=[
            GeneratedSetupFile(relative_path="g.json", content="{}")
        ],
        notes=["fine"],
        cost_usd=0.5,
        model_id="claude-sonnet",
    )

    assert result.scenarios[0].filename == "x.yaml"
    assert result.reference_files[0].relative_path == "Dockerfile"
    assert result.setup_files[0].relative_path == "g.json"
    assert result.notes == ["fine"]
    assert result.cost_usd == 0.5
    assert result.model_id == "claude-sonnet"


# ---------------- 2. happy path ----------------


def test_author_oracle_happy_path_returns_typed_result() -> None:
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet-4-5"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert len(result.scenarios) == 6
    assert all(isinstance(s, GeneratedScenarioFile) for s in result.scenarios)
    assert all(isinstance(r, GeneratedReferenceFile) for r in result.reference_files)
    assert all(isinstance(s, GeneratedSetupFile) for s in result.setup_files)
    # Sonnet tier expected
    call = router.parse_structured.call_args
    assert str(call.kwargs["tier"]) in {"LLMTier.sonnet", "sonnet"}
    assert call.kwargs["text_format"] is _OracleAuthoringPayload
    assert router.parse_structured.call_count == 1


# ---------------- 3. retry on missing Dockerfile ----------------


def test_author_oracle_retries_when_dockerfile_missing() -> None:
    bad = _valid_payload_dict()
    # Drop the Dockerfile from the reference bundle.
    bad["reference_files"] = [
        f for f in bad["reference_files"] if f["relative_path"] != "Dockerfile"
    ]
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet-4-5"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert router.parse_structured.call_count == 2
    # The retry's user prompt must mention the failure so Sonnet sees it.
    second_call_user = router.parse_structured.call_args_list[1].kwargs["user"]
    assert "Dockerfile" in second_call_user


# ---------------- 4. retry on unknown rubric kind ----------------


def test_author_oracle_retries_on_unknown_rubric_kind() -> None:
    bad = _valid_payload_dict()
    # Inject a scenario with a rubric kind that is not registered.
    bad["scenarios"][0]["yaml_content"] = _scenario_yaml(
        "happy_bogus_rubric", rubric_kind="this_kind_is_not_registered"
    )
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert router.parse_structured.call_count == 2


# ---------------- 5. retry on malformed YAML ----------------


def test_author_oracle_retries_on_malformed_yaml() -> None:
    bad = _valid_payload_dict()
    bad["scenarios"][0]["yaml_content"] = (
        "id: x\n  this is: malformed: yaml: ::\n   - and: weird"
    )
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert router.parse_structured.call_count == 2


# ---------------- 6. three failures -> error ----------------


def test_author_oracle_raises_after_three_failures() -> None:
    bad = _valid_payload_dict()
    # Strip Dockerfile so every attempt fails validation.
    bad["reference_files"] = [
        f for f in bad["reference_files"] if f["relative_path"] != "Dockerfile"
    ]
    router = _router_sequence(bad, bad, bad)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    with pytest.raises(OracleAuthoringError) as excinfo:
        author.author_oracle(_spec_with_abstention())

    assert router.parse_structured.call_count == 3
    # Diagnostics should be in the message.
    assert "Dockerfile" in str(excinfo.value)


# ---------------- 7. required categories ----------------


def test_author_oracle_validates_required_categories_for_abstention_spec() -> None:
    bad = _valid_payload_dict(include_out_of_scope=False)
    good = _valid_payload_dict(include_out_of_scope=True)
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert router.parse_structured.call_count == 2
    # Second attempt's user prompt should mention the missing out_of_scope.
    second_call_user = router.parse_structured.call_args_list[1].kwargs["user"]
    assert "out_of_scope" in second_call_user


# ---------------- 8. system prompt lists rubric kinds ----------------


def test_system_prompt_lists_registered_rubric_kinds() -> None:
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    author.author_oracle(_spec_with_abstention())

    call = router.parse_structured.call_args
    system = call.kwargs["system"]
    for kind in scenario_loader.scenario_rubrics_base.RUBRIC_REGISTRY:
        assert kind in system, f"system prompt missing rubric kind '{kind}'"


# ---------------- 9. system prompt forbids prescriptive tech ----------------


def test_system_prompt_forbids_prescriptive_tech() -> None:
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    author.author_oracle(_spec_with_abstention())

    call = router.parse_structured.call_args
    system = call.kwargs["system"].lower()
    # The author MUST NOT bake "FAISS" / "BM25" as required tech in the
    # scenarios it emits — the prompt names these as banned tokens.
    assert "faiss" in system
    assert "bm25" in system
    # And the prompt explicitly says they're forbidden / not required.
    assert (
        "do not require" in system
        or "must not require" in system
        or "do not prescribe" in system
        or "must not prescribe" in system
        or "forbidden" in system
    )


# ---------------- 10. reference impl includes Dockerfile ----------------


def test_author_oracle_reference_files_include_dockerfile() -> None:
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    paths = {f.relative_path for f in result.reference_files}
    assert "Dockerfile" in paths
    # And a Python install manifest.
    assert "requirements.txt" in paths or "pyproject.toml" in paths


# ---------------- 11. cost_usd computed from usage ----------------


def test_author_oracle_cost_usd_aggregates_router_usage() -> None:
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet-4-5"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    # ``_router_returning`` reports estimated_cost_usd=0.42 on the first
    # (and only) call. The author surfaces it on the result.
    assert result.cost_usd == pytest.approx(0.42)
    assert result.model_id == "claude-sonnet-4-5"


def test_author_oracle_cost_usd_aggregates_across_retries() -> None:
    bad = _valid_payload_dict()
    bad["reference_files"] = [
        f for f in bad["reference_files"] if f["relative_path"] != "Dockerfile"
    ]
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    # _router_sequence reports 0.10 per call; two calls = 0.20.
    assert result.cost_usd == pytest.approx(0.20)


# ---------------- 12. retry passes failure context ----------------


def test_author_oracle_retry_appends_failure_context_to_user_prompt() -> None:
    bad = _valid_payload_dict()
    bad["scenarios"][0]["yaml_content"] = _scenario_yaml(
        "happy_bogus_rubric", rubric_kind="this_kind_is_not_registered"
    )
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    author.author_oracle(_spec_with_abstention())

    first_user = router.parse_structured.call_args_list[0].kwargs["user"]
    second_user = router.parse_structured.call_args_list[1].kwargs["user"]
    assert second_user != first_user
    # Failure mentions the offending rubric kind so Sonnet can repair.
    assert (
        "this_kind_is_not_registered" in second_user
        or "Unknown rubric kind" in second_user
    )


# ---------------- 13. error class shape ----------------


def test_oracle_authoring_error_is_runtime_error() -> None:
    assert issubclass(OracleAuthoringError, RuntimeError)


# ---------------- 14. empty reference_files -> retry ----------------


def test_author_oracle_retries_on_empty_reference_files() -> None:
    bad = _valid_payload_dict()
    bad["reference_files"] = []
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert router.parse_structured.call_count == 2


# ---------------- 15. quality_bar coverage gate (Codex review #5) ----------------


def _payload_dropping_bar_coverage(uncovered_bar_ids: list[str]) -> dict:
    """Build a valid-shape payload, then strip the given bar IDs from every
    scenario's ``quality_bar_ids`` so the coverage validator fails for them.

    Used to drive the new tests that exercise the publish-gate coverage check
    inside ``OracleAuthor._validate_payload`` (Codex review #5 finding).
    """
    payload = _valid_payload_dict()
    drop = set(uncovered_bar_ids)
    for sf in payload["scenarios"]:
        doc = yaml.safe_load(sf["yaml_content"])
        doc["quality_bar_ids"] = [
            bar_id for bar_id in doc.get("quality_bar_ids", []) if bar_id not in drop
        ]
        sf["yaml_content"] = yaml.safe_dump(doc, sort_keys=False)
    return payload


def test_author_oracle_fails_when_spec_bar_uncovered() -> None:
    """``_validate_payload`` must reject a bundle that leaves a spec
    QualityBar unreferenced — even when every other gate (Dockerfile,
    install manifest, category coverage) passes. Otherwise oracle_authoring
    happily produces a bundle that ``oracle_validation`` will block at the
    publish gate, wasting the 3-attempt grader retry budget downstream.
    """
    bad = _payload_dropping_bar_coverage(["abstention_precision"])
    # Make every attempt fail the same way so retries exhaust.
    router = _router_sequence(bad, bad, bad)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    with pytest.raises(OracleAuthoringError) as excinfo:
        author.author_oracle(_spec_with_abstention())

    msg = str(excinfo.value)
    assert "abstention_precision" in msg
    # The diagnostic should make clear this is a coverage problem so a
    # human reading the run log can immediately tell what went wrong.
    assert "quality_bar" in msg or "uncovered" in msg


def test_author_oracle_retries_with_uncovered_bar_diagnostic() -> None:
    """When attempt N fails the coverage gate, attempt N+1's user prompt
    must include the offending bar IDs so Sonnet can repair targeted-ly."""
    bad = _payload_dropping_bar_coverage(["abstention_precision"])
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert router.parse_structured.call_count == 2
    second_user = router.parse_structured.call_args_list[1].kwargs["user"]
    assert "abstention_precision" in second_user
    # The retry context should mention what kind of failure this was.
    assert "quality_bar" in second_user or "uncovered" in second_user


def test_author_oracle_succeeds_when_all_bars_covered() -> None:
    """Sanity check: a payload that references every spec bar at least once
    passes the coverage gate on the first attempt."""
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet-4-5"

    author = OracleAuthor(router=router)
    result = author.author_oracle(_spec_with_abstention())

    assert isinstance(result, OracleAuthoringResult)
    assert router.parse_structured.call_count == 1


def test_payload_schema_includes_quality_bar_ids() -> None:
    """The LLM-emitted scenario payload must declare ``quality_bar_ids`` so
    Sonnet's structured-output schema teaches the model the field exists."""
    schema_fields = _OracleAuthoringPayload.model_fields
    scenarios_field = schema_fields["scenarios"]
    # The inner type for the scenarios list must be the per-scenario payload
    # model. Pull it out and verify it has a ``quality_bar_ids`` field.
    from app.services.oracle_authoring import _ScenarioFilePayload

    inner_fields = _ScenarioFilePayload.model_fields
    assert "quality_bar_ids" in inner_fields, (
        f"_ScenarioFilePayload must declare quality_bar_ids; "
        f"got fields: {list(inner_fields)}"
    )
    # Silence the unused-binding lint while keeping the assertion above
    # readable.
    _ = scenarios_field


def test_system_prompt_mentions_quality_bar_ids() -> None:
    """The system prompt must teach the LLM about the ``quality_bar_ids``
    field — both that scenarios carry it and that every spec bar must be
    referenced by at least one scenario."""
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    author.author_oracle(_spec_with_abstention())

    system = router.parse_structured.call_args.kwargs["system"]
    assert "quality_bar_ids" in system, (
        "system prompt must teach the LLM about the quality_bar_ids field"
    )


def test_user_prompt_lists_spec_bar_ids() -> None:
    """The user prompt must list the spec's bar IDs explicitly so the LLM
    knows the universe of valid ``quality_bar_ids`` values."""
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    spec = _spec_with_abstention()
    author.author_oracle(spec)

    user = router.parse_structured.call_args.kwargs["user"]
    for bar in spec.quality_bars:
        assert bar.id in user, (
            f"user prompt must list spec bar id '{bar.id}' so the LLM can "
            f"reference it from scenario quality_bar_ids"
        )


def test_repair_prompt_includes_uncovered_bar_failure_context() -> None:
    """On retry after a coverage failure, the failure-context block in the
    user prompt must mention the specific uncovered bar IDs."""
    bad = _payload_dropping_bar_coverage(
        ["abstention_precision", "recall_at_5"]
    )
    good = _valid_payload_dict()
    router = _router_sequence(bad, good)
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    author.author_oracle(_spec_with_abstention())

    second_user = router.parse_structured.call_args_list[1].kwargs["user"]
    # Both uncovered bars must appear in the retry user prompt's failure
    # context so the LLM repairs both, not just one.
    assert "abstention_precision" in second_user
    assert "recall_at_5" in second_user
    # And the prompt should explicitly mark this as a prior-validation
    # failure (the existing retry context header).
    assert "Prior validation failures" in second_user


# ---------------- HF benchmark integration ----------------


def _benchmark_bundle() -> BenchmarkBundle:
    return BenchmarkBundle(
        corpus=[
            BenchmarkDocument(doc_id="d1", title="T1", text="first"),
            BenchmarkDocument(doc_id="d2", title="T2", text="second"),
        ],
        queries=[BenchmarkQuery(query_id="q1", text="?")],
        qrels={"q1": {"d1": 1}},
        source=HFBenchmarkSource(
            corpus_dataset="BeIR/scifact",
            qrels_dataset="BeIR/scifact-qrels",
        ),
    )


def _spec_with_benchmark() -> CourseOutcomeSpec:
    spec = _spec_with_abstention()
    return spec.model_copy(
        update={
            "benchmark": HFBenchmarkSource(
                corpus_dataset="BeIR/scifact",
                qrels_dataset="BeIR/scifact-qrels",
            ),
        }
    )


def test_author_oracle_without_benchmark_uses_existing_path() -> None:
    """The existing LLM-synthesis path must remain untouched when the spec
    declares no benchmark.
    """
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet"

    author = OracleAuthor(router=router)
    spec = _spec_with_abstention()
    assert spec.benchmark is None
    result = author.author_oracle(spec)

    assert isinstance(result, OracleAuthoringResult)
    # The setup file from the LLM payload is preserved (gold_qa.json
    # with the stub queries content).
    assert any(f.relative_path == "gold_qa.json" for f in result.setup_files)


def test_author_oracle_calls_loader_when_benchmark_set() -> None:
    """When ``spec.benchmark`` is set, ``load_benchmark`` MUST be invoked
    with the declared source — and the LLM MUST still be called (for
    scenarios + reference impl).
    """
    bundle = _benchmark_bundle()
    payload = _valid_payload_dict()
    # The LLM payload should not include setup files in the benchmark path;
    # the loader provides them. Drop them here so we verify the result
    # files came from the benchmark, not the LLM.
    payload["setup_files"] = []
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_benchmark", return_value=bundle
    ) as loader:
        author = OracleAuthor(router=router)
        result = author.author_oracle(_spec_with_benchmark())

    assert loader.call_count == 1
    # The loader saw the same HFBenchmarkSource instance the spec carries.
    loader.assert_called_once()
    assert isinstance(result, OracleAuthoringResult)
    # Scenarios + reference came from the LLM; setup came from the loader.
    assert router.parse_structured.call_count == 1


def test_author_oracle_benchmark_prompt_excludes_setup_authoring() -> None:
    """The LLM user prompt MUST tell the model not to author setup files
    when the benchmark is preloaded — otherwise the model wastes tokens
    authoring data we're about to throw away.
    """
    bundle = _benchmark_bundle()
    payload = _valid_payload_dict()
    payload["setup_files"] = []
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_benchmark", return_value=bundle
    ):
        author = OracleAuthor(router=router)
        author.author_oracle(_spec_with_benchmark())

    user_prompt = router.parse_structured.call_args.kwargs["user"]
    # The prompt should name the benchmark and indicate setup is pre-loaded.
    assert "BeIR/scifact" in user_prompt
    lowered = user_prompt.lower()
    assert "preloaded" in lowered or "pre-loaded" in lowered or "pre-load" in lowered
    # And the prompt must explicitly tell the LLM not to author setup files.
    assert (
        "do not author setup" in lowered
        or "not author setup" in lowered
        or "no setup" in lowered
        or "skip setup" in lowered
    )


def test_author_oracle_result_setup_files_from_benchmark() -> None:
    """When the benchmark is loaded, the returned
    ``OracleAuthoringResult.setup_files`` MUST contain the three benchmark-
    derived files (corpus.jsonl, queries.jsonl, gold_qa.json) — regardless
    of what the LLM said in its ``setup_files`` slot.
    """
    bundle = _benchmark_bundle()
    payload = _valid_payload_dict()
    # Even if the LLM emitted unrelated setup files, the benchmark-derived
    # ones must take precedence (and ideally be the only ones returned).
    payload["setup_files"] = [
        {"relative_path": "bogus.json", "content": "should be discarded"}
    ]
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_benchmark", return_value=bundle
    ):
        author = OracleAuthor(router=router)
        result = author.author_oracle(_spec_with_benchmark())

    paths = {f.relative_path for f in result.setup_files}
    assert "corpus.jsonl" in paths
    assert "queries.jsonl" in paths
    assert "gold_qa.json" in paths
    # The LLM's bogus setup file must NOT be present.
    assert "bogus.json" not in paths


def test_author_oracle_raises_when_benchmark_loader_fails() -> None:
    """If ``load_benchmark`` raises, ``author_oracle`` must wrap the failure
    in ``OracleAuthoringError`` — there is no silent fallback to the
    LLM-synthesis path.
    """
    router = _router_returning(_valid_payload_dict())
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_benchmark",
        side_effect=BenchmarkLoadError("dataset not found"),
    ):
        author = OracleAuthor(router=router)
        with pytest.raises(OracleAuthoringError) as excinfo:
            author.author_oracle(_spec_with_benchmark())

    assert "dataset not found" in str(excinfo.value) or "benchmark" in str(
        excinfo.value
    ).lower()
    # The LLM must NOT be invoked when the benchmark load fails.
    assert router.parse_structured.call_count == 0


def test_author_oracle_benchmark_path_still_authors_scenarios_and_reference() -> None:
    """Even with a benchmark set, the LLM still authors scenarios and the
    reference impl — they're orthogonal to the setup data.
    """
    bundle = _benchmark_bundle()
    payload = _valid_payload_dict()
    payload["setup_files"] = []
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_benchmark", return_value=bundle
    ):
        author = OracleAuthor(router=router)
        result = author.author_oracle(_spec_with_benchmark())

    # Scenarios came from the LLM and they're still present.
    assert len(result.scenarios) >= 1
    # Reference files include the Dockerfile (LLM-authored too).
    paths = {f.relative_path for f in result.reference_files}
    assert "Dockerfile" in paths


# ---------------- CRAG benchmark integration ----------------


def _crag_bundle() -> CRAGBenchmarkBundle:
    return CRAGBenchmarkBundle(
        queries=[
            CRAGQuery(
                query_id="fin1",
                query="What was AAPL's Q1 2023 revenue?",
                answer="$117.2B.",
                alt_ans=["117.2 billion"],
                search_results=[
                    {
                        "page_url": "https://example.com/a",
                        "page_snippet": "AAPL Q1 2023 revenue 117.2B...",
                        "page_result": "<html>",
                    }
                ],
                domain="finance",
                question_type="simple",
                answer_type="valid",
            ),
            CRAGQuery(
                query_id="fp1",
                query="Why did MSFT acquire OpenAI in 2024?",
                answer="Microsoft did not acquire OpenAI in 2024.",
                alt_ans=["false premise"],
                search_results=[],
                domain="finance",
                question_type="false_premise",
                answer_type="valid",
            ),
        ],
        source=CRAGBenchmarkSource(),
    )


def _spec_with_crag_benchmark() -> CourseOutcomeSpec:
    spec = _spec_with_abstention()
    return spec.model_copy(
        update={"benchmark": CRAGBenchmarkSource()}
    )


def test_author_oracle_calls_crag_loader_when_crag_benchmark_set() -> None:
    """When ``spec.benchmark`` is a ``CRAGBenchmarkSource``, the author
    must call ``load_crag_benchmark`` (NOT the BeIR ``load_benchmark``)."""
    bundle = _crag_bundle()
    payload = _valid_payload_dict()
    payload["setup_files"] = []
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_crag_benchmark",
        return_value=bundle,
    ) as crag_loader, patch(
        "app.services.oracle_authoring.load_benchmark"
    ) as beir_loader:
        author = OracleAuthor(router=router)
        result = author.author_oracle(_spec_with_crag_benchmark())

    assert crag_loader.call_count == 1
    assert beir_loader.call_count == 0
    assert isinstance(result, OracleAuthoringResult)


def test_author_oracle_user_prompt_is_crag_aware() -> None:
    """The user prompt MUST tell Sonnet that:
    - retrieval is per-query (look up ``search_results_index.json``),
    - gold answers are text strings (``gold_answers.json``),
    - and the LLM should NOT author setup files (they're preloaded)."""
    bundle = _crag_bundle()
    payload = _valid_payload_dict()
    payload["setup_files"] = []
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_crag_benchmark",
        return_value=bundle,
    ):
        author = OracleAuthor(router=router)
        author.author_oracle(_spec_with_crag_benchmark())

    user_prompt = router.parse_structured.call_args.kwargs["user"]
    lowered = user_prompt.lower()
    # CRAG name + dataset identifier
    assert "crag" in lowered or "quivr" in lowered
    # The two CRAG setup files are named so Sonnet references them in scenarios
    assert "search_results_index" in user_prompt
    assert "gold_answers" in user_prompt
    # And the prompt must instruct the model NOT to author setup files
    assert (
        "do not author setup" in lowered
        or "not author setup" in lowered
        or "no setup" in lowered
        or "skip setup" in lowered
    )


def test_author_oracle_crag_prompt_recommends_semantic_eq_for_valid_answers() -> None:
    """The user prompt must teach Sonnet that ``llm_judge_semantic_eq``
    is the right rubric for scenarios judging ``valid``-answer
    questions."""
    bundle = _crag_bundle()
    payload = _valid_payload_dict()
    payload["setup_files"] = []
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_crag_benchmark",
        return_value=bundle,
    ):
        author = OracleAuthor(router=router)
        author.author_oracle(_spec_with_crag_benchmark())

    user_prompt = router.parse_structured.call_args.kwargs["user"]
    assert "llm_judge_semantic_eq" in user_prompt


def test_author_oracle_crag_prompt_recommends_false_premise_for_false_premise_type() -> None:
    """The user prompt must teach Sonnet that ``llm_judge_false_premise``
    is the right rubric for ``question_type == "false_premise"``
    scenarios."""
    bundle = _crag_bundle()
    payload = _valid_payload_dict()
    payload["setup_files"] = []
    router = _router_returning(payload)
    router.model_id_for.return_value = "claude-sonnet"

    with patch(
        "app.services.oracle_authoring.load_crag_benchmark",
        return_value=bundle,
    ):
        author = OracleAuthor(router=router)
        author.author_oracle(_spec_with_crag_benchmark())

    user_prompt = router.parse_structured.call_args.kwargs["user"]
    assert "llm_judge_false_premise" in user_prompt
    assert "false_premise" in user_prompt
