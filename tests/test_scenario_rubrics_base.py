"""Foundation tests for the scenario-rubric library.

These tests pin the public surface every rubric implementer codes
against: ``Verdict`` (the result), ``RubricContext`` (the input bundle),
``Rubric`` (the abstract base), ``resolve_path`` (dotted-path lookup
inside captures), and ``RUBRIC_REGISTRY`` + ``register_rubric``.

The agents implementing the eight RAG-needed rubrics rely on this
contract; if anything here changes, downstream tests must update too.
"""
from __future__ import annotations

import pytest

from app.services.scenario_rubrics_base import (
    RUBRIC_REGISTRY,
    Rubric,
    RubricContext,
    Verdict,
    register_rubric,
    resolve_path,
)


# ---------------- Verdict ----------------


def test_verdict_pass_minimal_fields() -> None:
    v = Verdict(status="pass", rationale="answer covers all expected facts")
    assert v.status == "pass"
    assert v.rationale == "answer covers all expected facts"
    assert v.diagnostic == {}
    assert v.cost_usd == 0.0


def test_verdict_fail_with_diagnostic() -> None:
    v = Verdict(
        status="fail",
        rationale="missing two required facts",
        diagnostic={"missing_facts": ["RRF", "abstention"]},
    )
    assert v.status == "fail"
    assert v.diagnostic["missing_facts"] == ["RRF", "abstention"]


def test_verdict_abstain_when_judge_unavailable() -> None:
    v = Verdict(status="abstain", rationale="no LLM router configured")
    assert v.status == "abstain"


def test_verdict_rejects_unknown_status() -> None:
    with pytest.raises(Exception):
        Verdict(status="error", rationale="...")  # type: ignore[arg-type]


def test_verdict_cost_usd_records_llm_spend() -> None:
    v = Verdict(status="pass", rationale="ok", cost_usd=0.0072)
    assert v.cost_usd == pytest.approx(0.0072)


# ---------------- RubricContext ----------------


def test_rubric_context_defaults() -> None:
    ctx = RubricContext(captures={"answer": "hello"})
    assert ctx.captures == {"answer": "hello"}
    assert ctx.setup_data == {}
    assert ctx.course_meta == {}


def test_rubric_context_carries_setup_and_meta() -> None:
    ctx = RubricContext(
        captures={"resp": {"chunks": []}},
        setup_data={"corpus": [{"doc_id": "doc_001"}]},
        course_meta={"core_entities": ["retrieval corpus"]},
    )
    assert ctx.setup_data["corpus"][0]["doc_id"] == "doc_001"
    assert "retrieval corpus" in ctx.course_meta["core_entities"]


# ---------------- resolve_path ----------------


def test_resolve_path_top_level_key() -> None:
    captures = {"resp": {"status": 200}}
    assert resolve_path(captures, "resp") == {"status": 200}


def test_resolve_path_nested_dict() -> None:
    captures = {"resp": {"body": {"answer": "hello"}}}
    assert resolve_path(captures, "resp.body.answer") == "hello"


def test_resolve_path_list_index() -> None:
    captures = {"chunks": [{"doc_id": "a"}, {"doc_id": "b"}]}
    assert resolve_path(captures, "chunks[1].doc_id") == "b"


def test_resolve_path_list_index_at_top() -> None:
    captures = {"results": [10, 20, 30]}
    assert resolve_path(captures, "results[2]") == 30


def test_resolve_path_missing_key_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        resolve_path({"a": 1}, "missing")


def test_resolve_path_missing_nested_key_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        resolve_path({"a": {"b": 1}}, "a.c")


def test_resolve_path_index_out_of_range_raises_indexerror() -> None:
    with pytest.raises(IndexError):
        resolve_path({"xs": [1, 2]}, "xs[5]")


def test_resolve_path_empty_path_returns_root() -> None:
    captures = {"a": 1}
    assert resolve_path(captures, "") == captures


# ---------------- Rubric ABC ----------------


def test_rubric_abc_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Rubric()  # type: ignore[abstract]


def test_rubric_subclass_must_implement_judge() -> None:
    class IncompleteRubric(Rubric):
        name = "incomplete"

    with pytest.raises(TypeError):
        IncompleteRubric()  # type: ignore[abstract]


def test_rubric_subclass_with_judge_works() -> None:
    class AlwaysPass(Rubric):
        name = "always_pass"

        def judge(self, ctx: RubricContext) -> Verdict:
            return Verdict(status="pass", rationale="trivially true")

    r = AlwaysPass()
    out = r.judge(RubricContext(captures={}))
    assert out.status == "pass"


# ---------------- Registry ----------------


def test_register_rubric_adds_to_registry() -> None:
    @register_rubric
    class FooRubric(Rubric):
        name = "foo_rubric_for_registry_test"

        def judge(self, ctx: RubricContext) -> Verdict:
            return Verdict(status="pass", rationale="")

    assert RUBRIC_REGISTRY["foo_rubric_for_registry_test"] is FooRubric


def test_register_rubric_returns_class_unchanged() -> None:
    @register_rubric
    class BarRubric(Rubric):
        name = "bar_rubric_for_registry_test"

        def judge(self, ctx: RubricContext) -> Verdict:
            return Verdict(status="pass", rationale="")

    assert BarRubric.name == "bar_rubric_for_registry_test"


def test_register_rubric_rejects_duplicate_name() -> None:
    @register_rubric
    class FirstRubric(Rubric):
        name = "duplicate_name_rubric_test"

        def judge(self, ctx: RubricContext) -> Verdict:
            return Verdict(status="pass", rationale="")

    with pytest.raises(ValueError):

        @register_rubric
        class SecondRubric(Rubric):  # type: ignore[unused-variable]
            name = "duplicate_name_rubric_test"

            def judge(self, ctx: RubricContext) -> Verdict:
                return Verdict(status="pass", rationale="")
