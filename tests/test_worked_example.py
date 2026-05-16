"""Worked-example feature coverage.

Audit (2026-05-16) found `_scenario_worked_example` only resolved an
"Expected" for 4 of 10 registered rubric kinds — the rest rendered a
blank Expected to the learner, including kinds used by deployed courses
(`subset_match`, `numeric_range`, `behavioral_equivalence`,
`llm_judge_false_premise`). These tests lock in that EVERY rubric kind
yields a non-empty Expected + a precise `<kind> on <target>` label,
that the question is field-agnostic (question/message/prompt/...), and
that Expected binds to the SPECIFIC failing rubric instance (not the
first of its kind).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.lms_service import (
    _rubric_kind_cfg,
    _scenario_worked_example,
    _short_target,
)


def _scenario(body: dict, rubrics: list[dict]):
    return SimpleNamespace(
        trace=[SimpleNamespace(body=body)],
        rubrics=rubrics,
    )


def _output():
    return SimpleNamespace(
        captures={"call_x": {"status": 200, "body": {"action": "answer", "redactions": 0}}}
    )


SETUP = {"gold": {"ans": "the canonical answer", "cites": ["a", "b"],
                  "fp": True, "rep": "stable reply"}}
BODY = {"message": "I need a refund of $480 for a duplicate charge."}

# (kind, config) — one representative per registered rubric kind.
CASES = [
    ("schema_match", {"target": "call_x.body", "must_have_fields": ["reply", "action"]}),
    ("literal_match", {"target": "call_x.body.action", "expected": "escalate"}),
    ("regex_match", {"target": "call_x.body.id", "pattern": "^kb_[a-z]+$"}),
    ("numeric_range", {"target": "call_x.body.redactions", "min_value": 1, "max_value": 5}),
    ("subset_match", {"target": "call_x.body.citations",
                      "acceptable_source": "call_x.request.body.kb_articles",
                      "acceptable_key": "article_id", "min_overlap": 1.0}),
    ("behavioral_equivalence", {"target": "call_x.body.action", "expected": "answer"}),
    ("behavioral_equivalence", {"target": "call_x.body.reply",
                                "expected_path": "setup_data.gold.rep"}),
    ("llm_judge_coverage", {"target": "call_x.body.reply",
                            "must_contain_facts": ["refund policy", "agent review"]}),
    ("llm_judge_semantic_eq", {"target": "call_x.body.reply",
                               "gold_path": "setup_data.gold.ans"}),
    ("llm_judge_false_premise", {"target": "call_x.body.abstained",
                                 "expected_falsity_path": "setup_data.gold.fp"}),
    ("oracle_set_overlap", {"target": "call_x.body.citations",
                            "gold_set_path": "setup_data.gold.cites", "min_recall": 0.5}),
]


@pytest.mark.parametrize("kind,cfg", CASES, ids=[f"{k}:{c['target']}" for k, c in CASES])
def test_every_rubric_kind_yields_expected_and_label(kind, cfg):
    rubric = {"kind": kind, **cfg}
    scen = _scenario(BODY, [rubric])
    q, expected, actual, label = _scenario_worked_example(
        scen, _output(), rubric, kind, SETUP
    )
    # 1. No kind renders a blank Expected (the core regression).
    assert expected, f"{kind} ({cfg['target']}) produced an empty Expected"
    assert isinstance(expected, str) and expected.strip()
    # 2. Precise label.
    st = _short_target(cfg["target"])
    assert label == (f"{kind} on {st}" if st != "response" else kind)
    # 3. Question is field-agnostic (body uses `message`, not `question`).
    assert q and "refund" in q
    # 4. Actual is the learner's captured response body.
    assert actual and "action" in actual


def test_no_registered_kind_is_unmapped():
    """If a new rubric kind is registered, this nudges to add coverage."""
    from app.services.scenario_rubrics_base import RUBRIC_REGISTRY

    covered = {k for k, _ in CASES}
    missing = set(RUBRIC_REGISTRY) - covered
    assert not missing, (
        f"Registered rubric kinds with no worked-example coverage test: "
        f"{sorted(missing)} — extend CASES and the resolver."
    )


@pytest.mark.parametrize("field", ["question", "message", "prompt", "query", "input"])
def test_question_is_field_agnostic(field):
    rubric = {"kind": "literal_match", "target": "call_x.body.action", "expected": "x"}
    scen = _scenario({field: "How do I export my data?"}, [rubric])
    q, _e, _a, _l = _scenario_worked_example(scen, _output(), rubric, "literal_match", SETUP)
    assert q == "How do I export my data?"


def test_question_falls_back_to_first_string_field():
    rubric = {"kind": "literal_match", "target": "call_x.body.action", "expected": "x"}
    scen = _scenario({"weird_key": "the only text here", "n": 3}, [rubric])
    q, *_ = _scenario_worked_example(scen, _output(), rubric, "literal_match", SETUP)
    assert q == "the only text here"


def test_expected_binds_to_specific_failing_instance_not_first_of_kind():
    """Two literal_match rubrics; the SECOND (redactions) is the failer.
    Expected must be the redactions gold, not the first instance's."""
    r_action = {"kind": "literal_match", "target": "call_x.body.action",
                "expected": "answer"}
    r_redactions = {"kind": "literal_match", "target": "call_x.body.redactions",
                    "expected": 2}
    scen = _scenario(BODY, [r_action, r_redactions])
    # caller passes the actual failing instance (r_redactions)
    q, expected, actual, label = _scenario_worked_example(
        scen, _output(), r_redactions, "literal_match", SETUP
    )
    assert expected == "2", f"bound to wrong instance: {expected!r}"
    assert label == "literal_match on redactions"


def test_short_target_helper():
    assert _short_target("call_x.body.action") == "action"
    assert _short_target("call_x.body.citations") == "citations"
    assert _short_target("call_x.body") == "response"
    assert _short_target("") == "response"


def test_rubric_kind_cfg_handles_dict_and_object():
    k, c = _rubric_kind_cfg({"kind": "literal_match", "target": "t", "expected": 1})
    assert k == "literal_match" and c["target"] == "t" and c["expected"] == 1
    obj = SimpleNamespace(kind="schema_match", config={"target": "call_x.body"})
    k2, c2 = _rubric_kind_cfg(obj)
    assert k2 == "schema_match" and c2["target"] == "call_x.body"
