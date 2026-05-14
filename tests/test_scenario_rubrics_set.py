"""Tests for set-style scenario rubrics: ``SubsetMatch`` and ``BehavioralEquivalence``.

These tests pin the public surface of the two set-style rubrics declared
in the scenario-rubric library design doc. Both rubrics are pure
functions of a ``RubricContext`` and must encode every reasonable
outcome in the returned ``Verdict`` rather than raising.
"""
from __future__ import annotations

from app.services.scenario_rubrics_base import (
    RUBRIC_REGISTRY,
    RubricContext,
)
from app.services.scenario_rubrics_set import BehavioralEquivalence, SubsetMatch


# ---------------- SubsetMatch ----------------


def test_subset_match_passes_when_all_target_values_in_acceptable_list_of_dicts() -> None:
    ctx = RubricContext(
        captures={
            "answer_response": {
                "cited_chunk_ids": ["doc_001_0", "doc_002_3"],
            },
            "corpus": [
                {"id": "doc_001_0", "text": "..."},
                {"id": "doc_001_1", "text": "..."},
                {"id": "doc_002_3", "text": "..."},
            ],
        },
    )
    rubric = SubsetMatch(
        target="answer_response.cited_chunk_ids",
        acceptable_source="corpus",
        acceptable_key="id",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"
    assert verdict.diagnostic["target_size"] == 2
    assert verdict.diagnostic["overlap_fraction"] == 1.0


def test_subset_match_fails_when_target_has_value_outside_acceptable() -> None:
    ctx = RubricContext(
        captures={
            "cited": ["doc_001_0", "ghost_chunk"],
            "corpus": [{"id": "doc_001_0"}, {"id": "doc_002_3"}],
        },
    )
    rubric = SubsetMatch(
        target="cited",
        acceptable_source="corpus",
        acceptable_key="id",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert "ghost_chunk" in verdict.diagnostic["invalid_values"]
    assert verdict.diagnostic["target_size"] == 2
    assert verdict.diagnostic["overlap_fraction"] == 0.5
    assert verdict.diagnostic["min_overlap"] == 1.0


def test_subset_match_partial_overlap_passes_when_threshold_lowered() -> None:
    ctx = RubricContext(
        captures={
            "cited": ["a", "b", "c", "d", "e"],
            "corpus": ["a", "b", "c", "d", "x"],
        },
    )
    rubric = SubsetMatch(
        target="cited",
        acceptable_source="corpus",
        acceptable_key=None,
        min_overlap=0.8,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"
    assert verdict.diagnostic["overlap_fraction"] == 0.8


def test_subset_match_partial_overlap_fails_below_threshold() -> None:
    ctx = RubricContext(
        captures={
            "cited": ["a", "b", "c", "d", "e"],
            "corpus": ["a", "b", "x", "y", "z"],
        },
    )
    rubric = SubsetMatch(
        target="cited",
        acceptable_source="corpus",
        acceptable_key=None,
        min_overlap=0.8,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["overlap_fraction"] == 0.4
    assert sorted(verdict.diagnostic["invalid_values"]) == ["c", "d", "e"]


def test_subset_match_resolves_acceptable_from_setup_data_prefix() -> None:
    ctx = RubricContext(
        captures={"cited": ["doc_001_0"]},
        setup_data={"hidden_corpus": [{"id": "doc_001_0"}, {"id": "doc_002"}]},
    )
    rubric = SubsetMatch(
        target="cited",
        acceptable_source="setup_data.hidden_corpus",
        acceptable_key="id",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"


def test_subset_match_fails_when_target_is_empty() -> None:
    ctx = RubricContext(
        captures={"cited": [], "corpus": [{"id": "doc_001_0"}]},
    )
    rubric = SubsetMatch(
        target="cited",
        acceptable_source="corpus",
        acceptable_key="id",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert "empty" in verdict.rationale.lower()


def test_subset_match_fails_when_acceptable_is_empty() -> None:
    ctx = RubricContext(
        captures={"cited": ["doc_001_0"], "corpus": []},
    )
    rubric = SubsetMatch(
        target="cited",
        acceptable_source="corpus",
        acceptable_key="id",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert "acceptable" in verdict.rationale.lower()


def test_subset_match_fails_when_target_path_missing() -> None:
    ctx = RubricContext(
        captures={"corpus": [{"id": "doc_001_0"}]},
    )
    rubric = SubsetMatch(
        target="answer.cited_chunk_ids",
        acceptable_source="corpus",
        acceptable_key="id",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["missing_path"] == "answer.cited_chunk_ids"


def test_subset_match_fails_when_acceptable_setup_path_missing() -> None:
    ctx = RubricContext(
        captures={"cited": ["doc_001_0"]},
        setup_data={},
    )
    rubric = SubsetMatch(
        target="cited",
        acceptable_source="setup_data.hidden_corpus",
        acceptable_key="id",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["missing_path"] == "setup_data.hidden_corpus"


def test_subset_match_is_registered() -> None:
    assert RUBRIC_REGISTRY["subset_match"] is SubsetMatch


# ---------------- BehavioralEquivalence ----------------


def test_behavioral_equivalence_passes_when_bool_matches() -> None:
    ctx = RubricContext(
        captures={"answer_response": {"abstained": True}},
    )
    rubric = BehavioralEquivalence(target="answer_response.abstained", expected=True)
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"


def test_behavioral_equivalence_fails_when_bool_mismatches() -> None:
    ctx = RubricContext(
        captures={"answer_response": {"abstained": False}},
    )
    rubric = BehavioralEquivalence(target="answer_response.abstained", expected=True)
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic == {"got": False, "expected": True}


def test_behavioral_equivalence_passes_for_int_status_code() -> None:
    ctx = RubricContext(captures={"resp": {"status": 404}})
    rubric = BehavioralEquivalence(target="resp.status", expected=404)
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"


def test_behavioral_equivalence_string_case_sensitive_fails_on_case_mismatch() -> None:
    ctx = RubricContext(captures={"resp": {"verdict": "REFUSE"}})
    rubric = BehavioralEquivalence(target="resp.verdict", expected="refuse")
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic == {"got": "REFUSE", "expected": "refuse"}


def test_behavioral_equivalence_string_case_insensitive_passes() -> None:
    ctx = RubricContext(captures={"resp": {"verdict": "REFUSE"}})
    rubric = BehavioralEquivalence(
        target="resp.verdict", expected="refuse", case_sensitive=False
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"


def test_behavioral_equivalence_fails_when_target_path_missing() -> None:
    ctx = RubricContext(captures={"resp": {}})
    rubric = BehavioralEquivalence(target="resp.status", expected=404)
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["missing_path"] == "resp.status"


def test_behavioral_equivalence_case_insensitive_does_not_affect_non_strings() -> None:
    # Setting case_sensitive=False with non-string values should still
    # require exact equality; it must not crash.
    ctx = RubricContext(captures={"x": 1})
    rubric = BehavioralEquivalence(target="x", expected=1, case_sensitive=False)
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"


def test_behavioral_equivalence_is_registered() -> None:
    assert RUBRIC_REGISTRY["behavioral_equivalence"] is BehavioralEquivalence
