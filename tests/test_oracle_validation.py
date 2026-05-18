"""Tests for the oracle_validation hard publish-gate.

Covers ``_required_categories`` heuristics, ``validate_oracle`` decision
logic (per-scenario pass, abort, category coverage, anti-trivial-rubric
check), and the ``validation_failures_to_findings`` bridge into the
canonical reviewer-repair channel.

All inputs are hand-crafted Pydantic instances; no LLM, no Docker, no
filesystem.
"""
from __future__ import annotations

import pytest

from app.domain.registry import PackageType
from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    QualityBar,
    StarterType,
)
from app.services.oracle_validation import (
    CategoryCoverageStatus,
    OraclePassResult,
    OracleScenarioOutput,
    OracleValidationReport,
    _required_categories,
    validate_curated_gold,
    validate_oracle,
    validation_failures_to_findings,
)
from app.services.scenario_loader import RubricSpec, Scenario, TraceStep


# ---------------- helpers ----------------


def _trace_step(step_id: str = "step1") -> TraceStep:
    return TraceStep(id=step_id, method="GET", path="/ping")


def _rubric(kind: str) -> RubricSpec:
    # Each registered rubric has a different config surface; we test
    # validation-by-kind only, so minimal but-valid configs per kind.
    if kind == "schema_match":
        return RubricSpec(kind=kind, target="resp.body", must_have_fields=["x"])
    if kind == "literal_match":
        return RubricSpec(kind=kind, target="resp.body.x", expected="y")
    if kind == "regex_match":
        return RubricSpec(kind=kind, target="resp.body.x", pattern="^y$")
    if kind == "numeric_range":
        return RubricSpec(kind=kind, target="resp.body.x", min=0, max=1)
    if kind == "oracle_set_overlap":
        return RubricSpec(
            kind=kind,
            target="resp.body.items",
            oracle_key="gold",
            threshold=0.8,
        )
    if kind == "subset_match":
        return RubricSpec(
            kind=kind, target="resp.body.items", expected_subset=["a"]
        )
    if kind == "behavioral_equivalence":
        return RubricSpec(
            kind=kind, target="resp.body.x", expected_oracle_key="gold"
        )
    if kind == "llm_judge_coverage":
        return RubricSpec(
            kind=kind,
            target="resp.body.x",
            criteria="must mention X",
        )
    raise ValueError(f"unhandled rubric kind in test helper: {kind}")


def _scenario(
    scenario_id: str,
    category: str,
    rubric_kinds: list[str] | None = None,
    *,
    quality_bar_ids: list[str] | None = None,
) -> Scenario:
    # Default ``quality_bar_ids`` covers ``_basic_spec``'s single
    # ``latency`` bar so legacy tests don't trip the publish gate's
    # coverage check (Codex review #4 finding #2). Tests that want to
    # exercise an uncovered-bar scenario pass an explicit list.
    return Scenario(
        id=scenario_id,
        description=f"scenario {scenario_id}",
        category=category,  # type: ignore[arg-type]
        trace=[_trace_step()],
        rubrics=[_rubric(k) for k in (rubric_kinds or ["oracle_set_overlap"])],
        quality_bar_ids=(
            list(quality_bar_ids) if quality_bar_ids is not None else ["latency"]
        ),
    )


def _verdict(rubric_kind: str, status: str = "pass") -> tuple[str, dict]:
    """Construct a canonical ``(rubric_kind, verdict_payload_dict)`` tuple
    matching :class:`app.services.oracle_pass.OracleScenarioOutput.verdicts`.
    """
    return (
        rubric_kind,
        {
            "status": status,
            "rationale": "ok",
            "diagnostic": {},
            "cost_usd": 0.0,
        },
    )


def _scenario_output(
    scenario_id: str,
    category: str,
    verdicts: list[tuple[str, dict]] | None = None,
    aborted: bool = False,
    abort_reason: str | None = None,
) -> OracleScenarioOutput:
    return OracleScenarioOutput(
        scenario_id=scenario_id,
        category=category,
        captures={},
        verdicts=verdicts if verdicts is not None else [_verdict("oracle_set_overlap")],
        aborted=aborted,
        abort_reason=abort_reason,
    )


def _oracle_result(
    scenario_outputs: list[OracleScenarioOutput],
) -> OraclePassResult:
    total = len(scenario_outputs)
    passed = sum(
        1
        for s in scenario_outputs
        if not s.aborted
        and s.verdicts
        and all(v[1].get("status") == "pass" for v in s.verdicts)
    )
    aborted = sum(1 for s in scenario_outputs if s.aborted)
    failed = total - passed - aborted
    abstained = sum(
        1
        for s in scenario_outputs
        if not s.aborted
        and any(v[1].get("status") == "abstain" for v in s.verdicts)
        and all(v[1].get("status") != "fail" for v in s.verdicts)
    )
    return OraclePassResult(
        reference_impl_hash="ref-hash-abc",
        scenario_set_hash="set-hash-xyz",
        generated_at="2026-05-14T00:00:00Z",
        scenario_outputs=scenario_outputs,
        total_scenarios=total,
        passed_scenarios=passed,
        failed_scenarios=failed,
        abstained_scenarios=abstained,
    )


def _basic_spec(
    *,
    quality_bars: list[QualityBar] | None = None,
    endpoints: list[EndpointContract] | None = None,
) -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title="Basic Course Title",
        goal="Build a tiny service that answers ping requests for testing.",
        starter_type=StarterType.partial,
        endpoints=endpoints
        or [
            EndpointContract(
                method=HttpMethod.GET,
                path="/ping",
                response_schema={"ok": "bool"},
                description="Health check.",
            )
        ],
        quality_bars=quality_bars
        or [
            QualityBar(
                id="latency",
                metric_description="p95 response latency",
                threshold="<= 200ms",
                judged_by=JudgeKind.numeric,
                sample_size=10,
            )
        ],
        package_type=PackageType.progressive_codebase_course,
    )


def _required_categories_scenarios() -> list[Scenario]:
    """A minimal scenario set covering the always-required trio."""
    return [
        _scenario("s_happy", "happy_path", ["oracle_set_overlap"]),
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
        _scenario("s_malformed", "malformed_input", ["oracle_set_overlap"]),
    ]


def _passing_outputs(scenarios: list[Scenario]) -> list[OracleScenarioOutput]:
    return [
        _scenario_output(s.id, s.category, [_verdict(r.kind) for r in s.rubrics])
        for s in scenarios
    ]


# ---------------- validate_oracle happy path ----------------


def test_validate_oracle_happy_path_publishable() -> None:
    spec = _basic_spec()
    scenarios = _required_categories_scenarios()
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert isinstance(report, OracleValidationReport)
    assert report.publishable is True
    assert report.failed_scenarios == []
    assert report.aborted_scenarios == []
    assert report.missing_required_categories == []
    assert report.trivial_rubric_warnings == []
    assert report.blocking_reasons == []
    assert report.reference_impl_hash == "ref-hash-abc"
    assert report.scenario_set_hash == "set-hash-xyz"


# ---------------- per-scenario pass/fail ----------------


def test_validate_oracle_one_scenario_failed_blocks_publish() -> None:
    spec = _basic_spec()
    scenarios = _required_categories_scenarios()
    outputs = _passing_outputs(scenarios)
    # Fail the boundary scenario by setting one verdict to "fail".
    outputs[1] = _scenario_output(
        "s_boundary",
        "boundary",
        [_verdict("oracle_set_overlap", status="fail")],
    )
    result = _oracle_result(outputs)

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "s_boundary" in report.failed_scenarios
    assert any("s_boundary" in br for br in report.blocking_reasons)


def test_validate_oracle_one_scenario_aborted_blocks_publish() -> None:
    spec = _basic_spec()
    scenarios = _required_categories_scenarios()
    outputs = _passing_outputs(scenarios)
    outputs[0] = _scenario_output(
        "s_happy",
        "happy_path",
        verdicts=[],
        aborted=True,
        abort_reason="connection refused",
    )
    result = _oracle_result(outputs)

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "s_happy" in report.aborted_scenarios
    assert any("s_happy" in br for br in report.blocking_reasons)


# ---------------- per-category coverage ----------------


def test_validate_oracle_missing_happy_path_blocks() -> None:
    spec = _basic_spec()
    scenarios = [
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
        _scenario("s_malformed", "malformed_input", ["oracle_set_overlap"]),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "happy_path" in report.missing_required_categories
    assert any("happy_path" in br for br in report.blocking_reasons)


def test_validate_oracle_missing_boundary_blocks() -> None:
    spec = _basic_spec()
    scenarios = [
        _scenario("s_happy", "happy_path", ["oracle_set_overlap"]),
        _scenario("s_malformed", "malformed_input", ["oracle_set_overlap"]),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "boundary" in report.missing_required_categories


def test_validate_oracle_missing_malformed_input_blocks() -> None:
    spec = _basic_spec()
    scenarios = [
        _scenario("s_happy", "happy_path", ["oracle_set_overlap"]),
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "malformed_input" in report.missing_required_categories


# ---------------- abstention/out_of_scope conditional ----------------


def test_validate_oracle_abstention_bar_requires_out_of_scope() -> None:
    spec = _basic_spec(
        quality_bars=[
            QualityBar(
                id="abstention_precision",
                metric_description="precision when refusing out-of-scope",
                threshold=">= 0.9",
                judged_by=JudgeKind.llm_haiku,
                sample_size=10,
            )
        ]
    )
    scenarios = _required_categories_scenarios()
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "out_of_scope" in report.missing_required_categories


def test_validate_oracle_no_abstention_bar_allows_missing_out_of_scope() -> None:
    spec = _basic_spec()
    scenarios = _required_categories_scenarios()
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is True
    assert "out_of_scope" not in report.missing_required_categories


# ---------------- composition conditional (3+ endpoints) ----------------


def test_validate_oracle_three_endpoints_require_composition_scenario() -> None:
    spec = _basic_spec(
        endpoints=[
            EndpointContract(
                method=HttpMethod.GET,
                path="/a",
                response_schema={"ok": "bool"},
                description="get a",
            ),
            EndpointContract(
                method=HttpMethod.GET,
                path="/b",
                response_schema={"ok": "bool"},
                description="get b",
            ),
            EndpointContract(
                method=HttpMethod.GET,
                path="/c",
                response_schema={"ok": "bool"},
                description="get c",
            ),
        ]
    )
    scenarios = _required_categories_scenarios()
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "composition" in report.missing_required_categories


# ---------------- idempotency conditional (create-shaped POST/PUT) ----------------


def test_validate_oracle_create_endpoint_requires_idempotency_scenario() -> None:
    spec = _basic_spec(
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/items/{id}",
                request_schema={"name": "str"},
                response_schema={"id": "str"},
                description="Create a new item.",
            )
        ]
    )
    scenarios = _required_categories_scenarios()
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "idempotency" in report.missing_required_categories


# ---------------- trivial-rubric anti-pattern ----------------


def test_validate_oracle_trivial_rubric_only_blocks_publish() -> None:
    spec = _basic_spec()
    scenarios = [
        _scenario(
            "s_happy", "happy_path", ["schema_match", "literal_match"]
        ),
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
        _scenario("s_malformed", "malformed_input", ["oracle_set_overlap"]),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "s_happy" in report.trivial_rubric_warnings
    assert any("s_happy" in br for br in report.blocking_reasons)


def test_validate_oracle_schema_plus_oracle_set_overlap_is_not_trivial() -> None:
    spec = _basic_spec()
    scenarios = [
        _scenario(
            "s_happy", "happy_path", ["schema_match", "oracle_set_overlap"]
        ),
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
        _scenario("s_malformed", "malformed_input", ["oracle_set_overlap"]),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is True
    assert report.trivial_rubric_warnings == []


# ---------------- combined failures ----------------


def test_validate_oracle_combined_failures_all_surface_in_blocking_reasons() -> None:
    spec = _basic_spec()
    # Missing malformed_input + a failing happy-path + a trivial-rubric scenario.
    scenarios = [
        _scenario("s_happy", "happy_path", ["schema_match", "literal_match"]),
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
    ]
    outputs = _passing_outputs(scenarios)
    outputs[0] = _scenario_output(
        "s_happy",
        "happy_path",
        [
            _verdict("schema_match", status="fail"),
            _verdict("literal_match", status="pass"),
        ],
    )
    result = _oracle_result(outputs)

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    assert "s_happy" in report.failed_scenarios
    assert "s_happy" in report.trivial_rubric_warnings
    assert "malformed_input" in report.missing_required_categories
    # blocking_reasons must mention each independently
    joined = " | ".join(report.blocking_reasons)
    assert "s_happy" in joined
    assert "malformed_input" in joined


# ---------------- validation_failures_to_findings bridge ----------------


def test_validation_failures_to_findings_one_per_blocking_reason() -> None:
    spec = _basic_spec()
    scenarios = [
        _scenario("s_happy", "happy_path", ["oracle_set_overlap"]),
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
        # Missing malformed_input on purpose.
    ]
    outputs = _passing_outputs(scenarios)
    outputs[0] = _scenario_output(
        "s_happy",
        "happy_path",
        [_verdict("oracle_set_overlap", status="fail")],
    )
    result = _oracle_result(outputs)

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )
    findings = validation_failures_to_findings(report)

    assert len(findings) == len(report.blocking_reasons)
    for f in findings:
        assert isinstance(f, ReviewerFinding)
        assert f.severity == ReviewerFindingSeverity.error


def test_validation_failures_to_findings_populates_actionable_hint() -> None:
    spec = _basic_spec()
    scenarios = [
        _scenario("s_happy", "happy_path", ["oracle_set_overlap"]),
        _scenario("s_boundary", "boundary", ["oracle_set_overlap"]),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )
    findings = validation_failures_to_findings(report)

    assert findings, "expected at least one finding for missing malformed_input"
    missing_cat_finding = next(
        (f for f in findings if "malformed_input" in (f.detail or "")),
        None,
    )
    assert missing_cat_finding is not None
    assert missing_cat_finding.hint is not None
    assert "malformed_input" in missing_cat_finding.hint
    assert "scenario" in missing_cat_finding.hint.lower()


def test_validation_failures_to_findings_empty_when_publishable() -> None:
    spec = _basic_spec()
    scenarios = _required_categories_scenarios()
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )
    assert report.publishable is True

    findings = validation_failures_to_findings(report)
    assert findings == []


# ---------------- _required_categories ----------------


def test_required_categories_basic_spec_returns_always_required_trio() -> None:
    spec = _basic_spec()
    required = _required_categories(spec)

    assert required == {"happy_path", "boundary", "malformed_input"}


def test_required_categories_rag_shaped_spec_returns_full_set() -> None:
    spec = CourseOutcomeSpec(
        title="RAG Course Title",
        goal="Build a retrieval-augmented service with abstention support.",
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/corpus",
                request_schema={"docs": "list"},
                response_schema={"ok": "bool"},
                description="Register a new corpus.",
            ),
            EndpointContract(
                method=HttpMethod.POST,
                path="/items/{id}",
                request_schema={"name": "str"},
                response_schema={"id": "str"},
                description="Create a new item.",
            ),
            EndpointContract(
                method=HttpMethod.GET,
                path="/search",
                response_schema={"results": "list"},
                description="Search the corpus.",
            ),
            EndpointContract(
                method=HttpMethod.GET,
                path="/health",
                response_schema={"ok": "bool"},
                description="Health probe.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="abstention_precision",
                metric_description="abstain precisely on out-of-scope queries",
                threshold=">= 0.95",
                judged_by=JudgeKind.llm_haiku,
                sample_size=10,
            )
        ],
        package_type=PackageType.progressive_codebase_course,
    )

    required = _required_categories(spec)

    assert "happy_path" in required
    assert "boundary" in required
    assert "malformed_input" in required
    assert "out_of_scope" in required
    assert "composition" in required
    assert "idempotency" in required


# ---------------- coverage entries on the report ----------------


def test_validate_oracle_report_exposes_per_category_coverage() -> None:
    spec = _basic_spec()
    scenarios = _required_categories_scenarios()
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    by_cat = {c.category: c for c in report.coverage}
    assert by_cat["happy_path"].present is True
    assert by_cat["happy_path"].is_required is True
    assert by_cat["happy_path"].scenario_count == 1
    assert by_cat["boundary"].present is True
    assert by_cat["malformed_input"].present is True


# ---------------- validate_curated_gold ----------------
#
# These tests cover the curated-mode gate. The validator runs *without*
# an OraclePassResult — it inspects the loaded scenarios + setup_data
# bundle to confirm the hand-authored gold is consistent and the
# category-coverage / trivial-rubric checks pass.


def _curated_overlap_rubric(
    gold_set_path: str = "gold.q.expected",
    target: str = "ask.body.cited_chunk_ids",
) -> RubricSpec:
    return RubricSpec(
        kind="oracle_set_overlap",
        target=target,
        gold_set_path=gold_set_path,
        min_recall=0.5,
    )


def _curated_judge_rubric(
    must_contain_facts: list[str] | None = None,
) -> RubricSpec:
    return RubricSpec(
        kind="llm_judge_coverage",
        target="ask.body.answer",
        must_contain_facts=list(must_contain_facts)
        if must_contain_facts is not None
        else ["fact-a", "fact-b"],
    )


def _curated_scenario(
    scenario_id: str,
    category: str,
    rubrics: list[RubricSpec] | None = None,
    *,
    quality_bar_ids: list[str] | None = None,
) -> Scenario:
    # See ``_scenario`` for why ``quality_bar_ids`` defaults to the
    # single bar in ``_basic_spec``.
    return Scenario(
        id=scenario_id,
        description=f"scenario {scenario_id}",
        category=category,  # type: ignore[arg-type]
        trace=[_trace_step()],
        rubrics=rubrics or [_curated_overlap_rubric()],
        quality_bar_ids=(
            list(quality_bar_ids) if quality_bar_ids is not None else ["latency"]
        ),
    )


def _curated_setup_data(
    gold_value: list[str] | None = None,
    corpus: list[dict] | None = None,
) -> dict:
    """Build a setup_data dict shaped the way oracle_pass loads it from
    ``private/grader/_setup/``: file stems mapped to JSON contents.

    ``gold.json`` → ``{"q": {"expected": [...]}}`` so the rubric path
    ``gold.q.expected`` resolves to a gold doc-id list.

    ``corpus.json`` → ``[{"doc_id": ...}, ...]`` so the universe check
    can verify gold IDs reference real docs.
    """
    return {
        "gold": {"q": {"expected": gold_value if gold_value is not None else ["d1"]}},
        "corpus": corpus
        if corpus is not None
        else [{"doc_id": "d1", "text": "hi"}, {"doc_id": "d2", "text": "bye"}],
    }


def _curated_required_scenarios() -> list[Scenario]:
    return [
        _curated_scenario("s_happy", "happy_path", [_curated_overlap_rubric()]),
        _curated_scenario("s_boundary", "boundary", [_curated_overlap_rubric()]),
        _curated_scenario(
            "s_malformed", "malformed_input", [_curated_overlap_rubric()]
        ),
    ]


def test_validate_curated_gold_happy_path_publishable() -> None:
    spec = _basic_spec()
    scenarios = _curated_required_scenarios()
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert isinstance(report, OracleValidationReport)
    assert report.publishable is True
    assert report.blocking_reasons == []
    # In curated mode there is no ref-impl hash; the validator must
    # leave these fields empty without raising.
    assert report.reference_impl_hash == ""
    assert report.scenario_set_hash == ""


def test_validate_curated_gold_missing_gold_entry_blocks() -> None:
    spec = _basic_spec()
    # Rubric points at a gold path that does not exist in setup_data.
    bad_rubric = _curated_overlap_rubric(gold_set_path="gold.missing_q.expected")
    scenarios = [
        _curated_scenario("s_happy", "happy_path", [bad_rubric]),
        _curated_scenario("s_boundary", "boundary"),
        _curated_scenario("s_malformed", "malformed_input"),
    ]
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is False
    assert any("gold.missing_q.expected" in br for br in report.blocking_reasons)
    assert any("s_happy" in br for br in report.blocking_reasons)


def test_validate_curated_gold_empty_must_contain_facts_blocks() -> None:
    spec = _basic_spec()
    judge_rubric = _curated_judge_rubric(must_contain_facts=[])
    scenarios = [
        _curated_scenario("s_happy", "happy_path", [judge_rubric]),
        _curated_scenario("s_boundary", "boundary"),
        _curated_scenario("s_malformed", "malformed_input"),
    ]
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is False
    # Blocking reason names the scenario and points at must_contain_facts.
    assert any(
        "s_happy" in br and "must_contain_facts" in br
        for br in report.blocking_reasons
    )


def test_validate_curated_gold_gold_doc_id_outside_corpus_blocks() -> None:
    spec = _basic_spec()
    scenarios = _curated_required_scenarios()
    # Gold references doc_id "ghost" that is not in the corpus universe.
    setup_data = _curated_setup_data(gold_value=["ghost"])

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is False
    assert any(
        "ghost" in br and "corpus" in br for br in report.blocking_reasons
    )


def test_validate_curated_gold_no_corpus_universe_emits_warning_not_block() -> None:
    spec = _basic_spec()
    scenarios = _curated_required_scenarios()
    # No corpus.json — the validator should skip the universe check
    # (warning at most) and still mark the bundle publishable.
    setup_data = {
        "gold": {"q": {"expected": ["d1"]}},
    }

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is True
    # No corpus-related blocking reasons present.
    assert not any("corpus" in br.lower() for br in report.blocking_reasons)


def test_validate_curated_gold_missing_required_category_blocks() -> None:
    spec = _basic_spec()
    # Drop the malformed_input scenario.
    scenarios = [
        _curated_scenario("s_happy", "happy_path"),
        _curated_scenario("s_boundary", "boundary"),
    ]
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is False
    assert "malformed_input" in report.missing_required_categories
    assert any("malformed_input" in br for br in report.blocking_reasons)


def test_validate_curated_gold_trivial_rubric_scenario_blocks() -> None:
    spec = _basic_spec()
    # Make happy_path use only structural rubrics.
    trivial = [
        RubricSpec(kind="schema_match", target="resp.body", must_have_fields=["x"]),
        RubricSpec(kind="literal_match", target="resp.body.x", expected="y"),
    ]
    scenarios = [
        _curated_scenario("s_happy", "happy_path", trivial),
        _curated_scenario("s_boundary", "boundary"),
        _curated_scenario("s_malformed", "malformed_input"),
    ]
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is False
    assert "s_happy" in report.trivial_rubric_warnings
    assert any("s_happy" in br for br in report.blocking_reasons)


def test_validate_curated_gold_report_has_empty_hashes() -> None:
    """The curated validator does not hash a reference impl. Both hash
    fields on the report must be empty strings; ``validation_failures_to_findings``
    consumes the report regardless of hash presence."""
    spec = _basic_spec()
    scenarios = _curated_required_scenarios()
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.reference_impl_hash == ""
    assert report.scenario_set_hash == ""


def test_validate_curated_gold_combined_failures_each_surface_in_blocking_reasons() -> None:
    spec = _basic_spec()
    judge_rubric = _curated_judge_rubric(must_contain_facts=[])  # empty facts
    scenarios = [
        _curated_scenario("s_happy", "happy_path", [judge_rubric]),
        _curated_scenario("s_boundary", "boundary"),
        # missing malformed_input
    ]
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is False
    joined = " | ".join(report.blocking_reasons)
    assert "must_contain_facts" in joined
    assert "malformed_input" in joined


def test_validate_curated_gold_judge_rubric_with_facts_is_ok() -> None:
    spec = _basic_spec()
    judge_rubric = _curated_judge_rubric(must_contain_facts=["alpha", "beta"])
    scenarios = [
        _curated_scenario("s_happy", "happy_path", [judge_rubric]),
        _curated_scenario("s_boundary", "boundary"),
        _curated_scenario("s_malformed", "malformed_input"),
    ]
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    assert report.publishable is True


# ---------------- canonical contract: oracle_pass is the single source ----------------


def test_oracle_pass_result_is_re_exported_not_duplicated() -> None:
    """``oracle_validation`` must re-export the canonical types from
    ``oracle_pass`` (Finding B: contract drift). Importing from either
    module must yield the SAME class object, not a parallel definition.
    """
    from app.services import oracle_pass as op
    from app.services import oracle_validation as ov

    assert ov.OraclePassResult is op.OraclePassResult
    assert ov.OracleScenarioOutput is op.OracleScenarioOutput


def test_validate_oracle_consumes_canonical_tuple_verdicts() -> None:
    """Integration: feed a real ``oracle_pass.OraclePassResult`` (tuple
    verdicts) into ``validate_oracle`` and verify per-scenario pass/fail
    is decoded correctly.

    Before Finding B was fixed, ``_scenario_passed`` called ``.get()`` on
    each verdict tuple, raising ``AttributeError`` (or silently treating
    every scenario as failing because tuples have no ``"status"`` key).
    """
    from app.services.oracle_pass import (
        OraclePassResult as CanonicalPassResult,
    )
    from app.services.oracle_pass import (
        OracleScenarioOutput as CanonicalScenarioOutput,
    )

    spec = _basic_spec()
    scenarios = _required_categories_scenarios()

    # Two scenarios pass; one fails — emulate the on-the-wire tuple shape.
    canonical = CanonicalPassResult(
        reference_impl_hash="hash-real",
        scenario_set_hash="set-real",
        generated_at="2026-05-14T00:00:00Z",
        scenario_outputs=[
            CanonicalScenarioOutput(
                scenario_id="s_happy",
                category="happy_path",
                captures={},
                verdicts=[
                    (
                        "oracle_set_overlap",
                        {"status": "pass", "rationale": "ok"},
                    )
                ],
                aborted=False,
            ),
            CanonicalScenarioOutput(
                scenario_id="s_boundary",
                category="boundary",
                captures={},
                verdicts=[
                    (
                        "oracle_set_overlap",
                        {"status": "fail", "rationale": "off-by-one"},
                    )
                ],
                aborted=False,
            ),
            CanonicalScenarioOutput(
                scenario_id="s_malformed",
                category="malformed_input",
                captures={},
                verdicts=[
                    (
                        "oracle_set_overlap",
                        {"status": "pass", "rationale": "ok"},
                    )
                ],
                aborted=False,
            ),
        ],
        total_scenarios=3,
        passed_scenarios=2,
        failed_scenarios=1,
        abstained_scenarios=0,
    )

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=canonical
    )

    assert report.publishable is False
    assert "s_boundary" in report.failed_scenarios
    assert "s_happy" not in report.failed_scenarios
    assert "s_malformed" not in report.failed_scenarios
    assert any("s_boundary" in br for br in report.blocking_reasons)


# ---------------- quality-bar coverage (Codex review #4 / finding #2) ----------------
#
# A QualityBar declared in the spec but referenced by zero scenarios is
# a course-authoring configuration error: the grader would silently
# "abstain" on that contract and the synthesizer would mark the overall
# report ``pass``. The publish gate must block this so a broken grader
# never reaches learners. We layer the check at both validators
# (reference-run and curated-gold) for defense in depth.


def _scenario_with_bar_ids(
    scenario_id: str,
    category: str,
    *,
    quality_bar_ids: list[str],
    rubric_kinds: list[str] | None = None,
) -> Scenario:
    s = _scenario(scenario_id, category, rubric_kinds)
    # ``Scenario.quality_bar_ids`` defaults to ``[]`` on the loader; the
    # publish gate must check this against the spec's declared bars.
    s.quality_bar_ids = list(quality_bar_ids)
    return s


def test_validate_oracle_blocks_when_quality_bar_uncovered() -> None:
    """Spec declares bars A and B; every scenario references only A.
    ``validate_oracle`` must mark the bundle non-publishable with a
    blocking reason that names the uncovered bar B.
    """
    spec = _basic_spec(
        quality_bars=[
            QualityBar(
                id="bar_a",
                metric_description="covered bar",
                threshold=">= 0.5",
                judged_by=JudgeKind.llm_haiku,
                sample_size=1,
            ),
            QualityBar(
                id="bar_b",
                metric_description="uncovered bar",
                threshold=">= 0.5",
                judged_by=JudgeKind.llm_haiku,
                sample_size=1,
            ),
        ]
    )
    scenarios = [
        _scenario_with_bar_ids(
            "s_happy", "happy_path", quality_bar_ids=["bar_a"]
        ),
        _scenario_with_bar_ids(
            "s_boundary", "boundary", quality_bar_ids=["bar_a"]
        ),
        _scenario_with_bar_ids(
            "s_malformed", "malformed_input", quality_bar_ids=["bar_a"]
        ),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is False
    # The blocking reason must name the uncovered bar and explain it as
    # a coverage gap rather than a run-time failure.
    assert any(
        "bar_b" in br and "no scenario" in br.lower()
        for br in report.blocking_reasons
    ), report.blocking_reasons
    # The covered bar must NOT appear as a coverage failure.
    assert not any(
        "bar_a" in br and "no scenario" in br.lower()
        for br in report.blocking_reasons
    )


def test_validate_curated_gold_blocks_when_bar_uncovered() -> None:
    """Same coverage check, but on the curated-gold path. The
    curated-gold validator has no oracle-pass result to consult; the
    coverage check is purely structural over the scenario set's
    ``quality_bar_ids`` and the spec's declared bars.
    """
    spec = _basic_spec(
        quality_bars=[
            QualityBar(
                id="bar_a",
                metric_description="covered bar",
                threshold=">= 0.5",
                judged_by=JudgeKind.llm_haiku,
                sample_size=1,
            ),
            QualityBar(
                id="bar_b",
                metric_description="uncovered bar",
                threshold=">= 0.5",
                judged_by=JudgeKind.llm_haiku,
                sample_size=1,
            ),
        ]
    )
    base = _curated_required_scenarios()
    for s in base:
        s.quality_bar_ids = ["bar_a"]
    setup_data = _curated_setup_data()

    report = validate_curated_gold(
        spec=spec, scenarios=base, setup_data=setup_data
    )

    assert report.publishable is False
    assert any(
        "bar_b" in br and "no scenario" in br.lower()
        for br in report.blocking_reasons
    ), report.blocking_reasons


def test_validate_oracle_passes_with_full_coverage() -> None:
    """When every declared bar is referenced by at least one scenario's
    ``quality_bar_ids``, the coverage gate is satisfied and the bundle
    publishes as before.
    """
    spec = _basic_spec(
        quality_bars=[
            QualityBar(
                id="bar_a",
                metric_description="covered bar",
                threshold=">= 0.5",
                judged_by=JudgeKind.llm_haiku,
                sample_size=1,
            ),
            QualityBar(
                id="bar_b",
                metric_description="also covered",
                threshold=">= 0.5",
                judged_by=JudgeKind.llm_haiku,
                sample_size=1,
            ),
        ]
    )
    scenarios = [
        _scenario_with_bar_ids(
            "s_happy", "happy_path", quality_bar_ids=["bar_a"]
        ),
        _scenario_with_bar_ids(
            "s_boundary", "boundary", quality_bar_ids=["bar_b"]
        ),
        _scenario_with_bar_ids(
            "s_malformed",
            "malformed_input",
            quality_bar_ids=["bar_a", "bar_b"],
        ),
    ]
    result = _oracle_result(_passing_outputs(scenarios))

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )

    assert report.publishable is True
    assert not any(
        "no scenario" in br.lower() for br in report.blocking_reasons
    )


def test_validate_oracle_canonical_all_pass_is_publishable() -> None:
    """Positive companion to the all-pass / fail integration case: every
    scenario passes when fed via the canonical tuple shape."""
    from app.services.oracle_pass import (
        OraclePassResult as CanonicalPassResult,
    )
    from app.services.oracle_pass import (
        OracleScenarioOutput as CanonicalScenarioOutput,
    )

    spec = _basic_spec()
    scenarios = _required_categories_scenarios()

    canonical = CanonicalPassResult(
        reference_impl_hash="h",
        scenario_set_hash="s",
        generated_at="2026-05-14T00:00:00Z",
        scenario_outputs=[
            CanonicalScenarioOutput(
                scenario_id=s.id,
                category=s.category,
                captures={},
                verdicts=[
                    ("oracle_set_overlap", {"status": "pass"}),
                ],
                aborted=False,
            )
            for s in scenarios
        ],
        total_scenarios=3,
        passed_scenarios=3,
        failed_scenarios=0,
        abstained_scenarios=0,
    )

    report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=canonical
    )

    assert report.publishable is True
    assert report.failed_scenarios == []
