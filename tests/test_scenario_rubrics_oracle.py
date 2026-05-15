"""Tests for oracle-comparison rubrics: ``OracleSetOverlap``.

These tests pin the public surface of the recall-against-gold rubric
declared in the scenario-rubric library design doc. The rubric is a
pure function of a ``RubricContext`` and must encode every reasonable
outcome in the returned ``Verdict`` rather than raising.

The canonical use is RAG retrieval evaluation: "for query Q, the gold
docs that should appear in top-k retrieval are {doc_001, doc_005}.
Did the learner's retrieval return at least one of these?".
"""
from __future__ import annotations

from app.services.scenario_rubrics_base import (
    RUBRIC_REGISTRY,
    RubricContext,
)
from app.services.scenario_rubrics_oracle import OracleSetOverlap


# ---------------- Registry ----------------


def test_oracle_set_overlap_is_registered() -> None:
    assert RUBRIC_REGISTRY["oracle_set_overlap"] is OracleSetOverlap
    assert OracleSetOverlap.name == "oracle_set_overlap"


# ---------------- Full recall ----------------


def test_full_recall_passes_with_plain_list_target() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_001", "doc_002", "doc_005"]}},
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"
    assert verdict.diagnostic["recall"] == 1.0
    assert verdict.diagnostic["gold_size"] == 2
    assert set(verdict.diagnostic["matched"]) == {"doc_001", "doc_005"}
    assert verdict.diagnostic["missed"] == []


# ---------------- Partial recall above threshold ----------------


def test_partial_recall_passes_when_above_threshold() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_001", "doc_007", "doc_009"]}},
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"
    assert verdict.diagnostic["recall"] == 0.5
    assert verdict.diagnostic["matched"] == ["doc_001"]
    assert verdict.diagnostic["missed"] == ["doc_005"]


# ---------------- Below threshold ----------------


def test_recall_below_threshold_fails() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_007", "doc_009"]}},
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["recall"] == 0.0
    assert verdict.diagnostic["min_recall"] == 0.5
    assert verdict.diagnostic["gold_size"] == 2
    assert verdict.diagnostic["matched"] == []
    assert sorted(verdict.diagnostic["missed"]) == ["doc_001", "doc_005"]


# ---------------- target_key extraction ----------------


def test_target_list_of_dicts_uses_target_key() -> None:
    ctx = RubricContext(
        captures={
            "retrieval": {
                "chunks": [
                    {"doc_id": "doc_001", "score": 0.91},
                    {"doc_id": "doc_003", "score": 0.78},
                    {"doc_id": "doc_005", "score": 0.42},
                ]
            }
        },
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.chunks",
        target_key="doc_id",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"
    assert verdict.diagnostic["recall"] == 1.0
    assert set(verdict.diagnostic["matched"]) == {"doc_001", "doc_005"}


# ---------------- top_k slicing ----------------


def test_top_k_limits_considered_slice() -> None:
    # Without top_k both gold items are in the list and we'd pass.
    # With top_k=2 we only consider the first two and lose doc_005.
    ctx = RubricContext(
        captures={
            "retrieval": {
                "doc_ids": ["doc_007", "doc_009", "doc_001", "doc_005"]
            }
        },
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
        top_k=2,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["recall"] == 0.0
    assert verdict.diagnostic["matched"] == []


def test_top_k_with_target_key_only_slices_target_then_extracts() -> None:
    ctx = RubricContext(
        captures={
            "retrieval": {
                "chunks": [
                    {"doc_id": "doc_001"},
                    {"doc_id": "doc_002"},
                    {"doc_id": "doc_005"},  # outside top_k
                ]
            }
        },
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.chunks",
        target_key="doc_id",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
        top_k=2,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"  # 1/2 = 0.5 >= 0.5
    assert verdict.diagnostic["recall"] == 0.5
    assert verdict.diagnostic["matched"] == ["doc_001"]


# ---------------- Empty gold set ----------------


def test_empty_gold_set_returns_abstain() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_001"]}},
        setup_data={"expected_doc_ids": []},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "abstain"
    assert "gold set is empty" in verdict.rationale


# ---------------- Missing gold path ----------------


def test_missing_gold_path_returns_abstain() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_001"]}},
        setup_data={"other_key": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "abstain"
    assert "expected_doc_ids" in verdict.rationale


def test_missing_gold_nested_path_returns_abstain() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_001"]}},
        setup_data={"gold_qa": {"q1": {"expected_doc_ids": ["doc_001"]}}},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        # q2 is not present
        gold_set_path="gold_qa.q2.expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "abstain"
    assert "gold_qa.q2.expected_doc_ids" in verdict.rationale


# ---------------- Missing target path ----------------


def test_missing_target_path_returns_fail() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"chunks": [{"doc_id": "doc_001"}]}},
        setup_data={"expected_doc_ids": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",  # captures has "chunks", not "doc_ids"
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["missing_path"] == "retrieval.doc_ids"


def test_target_index_out_of_range_returns_fail() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"groups": [["doc_001"]]}},
        setup_data={"expected_doc_ids": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.groups[3]",  # only one group present
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic["missing_path"] == "retrieval.groups[3]"


# ---------------- Threshold edge cases ----------------


def test_recall_exactly_at_threshold_passes() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_001", "doc_007"]}},
        setup_data={"expected_doc_ids": ["doc_001", "doc_002"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "pass"
    assert verdict.diagnostic["recall"] == 0.5


def test_default_min_recall_is_half() -> None:
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
    )
    assert rubric.min_recall == 0.5


# ---------------- Cost is zero (no LLM) ----------------


def test_verdict_has_zero_cost() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": ["doc_001"]}},
        setup_data={"expected_doc_ids": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
    )
    verdict = rubric.judge(ctx)
    assert verdict.cost_usd == 0.0


# ---------------- Malformed-learner-payload hardening ----------------
#
# These tests pin Codex review #6 finding #4: malformed shapes in the
# learner-produced target list must turn into a FAIL verdict, never a
# raised exception. The rubric library contract says pass/fail logic is
# encoded in the returned Verdict — never raised — so a bad learner
# payload must NOT crash the runner.


def test_target_not_a_list_returns_fail() -> None:
    """When ``target`` resolves to something that isn't a list (a string,
    int, None, etc.), the rubric must FAIL with a diagnostic rather than
    crash with ``TypeError`` inside ``list()`` or the comprehension."""
    ctx = RubricContext(
        # Learner returned a string instead of a list of doc_ids.
        captures={"retrieval": {"doc_ids": "not-a-list"}},
        setup_data={"expected_doc_ids": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert "not a list" in verdict.rationale
    # Diagnostic should record what type the learner actually produced.
    assert verdict.diagnostic.get("target_type") == "str"


def test_target_not_a_list_int_returns_fail() -> None:
    ctx = RubricContext(
        captures={"retrieval": {"doc_ids": 42}},
        setup_data={"expected_doc_ids": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.doc_ids",
        gold_set_path="expected_doc_ids",
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic.get("target_type") == "int"


def test_non_dict_element_with_target_key_returns_fail_with_diagnostic() -> None:
    """``target_key`` is set but elements are plain strings — the old
    code did ``"doc_001"["doc_id"]`` which raises ``TypeError``. The
    hardened rubric must FAIL with a diagnostic that names the count of
    invalid elements."""
    ctx = RubricContext(
        # Three strings — none of them are dicts.
        captures={"retrieval": {"chunks": ["doc_001", "doc_002", "doc_005"]}},
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.chunks",
        target_key="doc_id",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic.get("invalid_elements") == 3
    # Every element was invalid, so we can't compute any recall.
    assert "all" in verdict.rationale.lower() or "no valid" in verdict.rationale.lower()


def test_missing_target_key_in_element_returns_fail() -> None:
    """Dicts missing the configured key would raise ``KeyError`` under
    the old code. The hardened rubric must FAIL and skip those elements."""
    ctx = RubricContext(
        captures={
            "retrieval": {
                "chunks": [
                    {"score": 0.91},  # no doc_id
                    {"score": 0.78},  # no doc_id
                ]
            }
        },
        setup_data={"expected_doc_ids": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.chunks",
        target_key="doc_id",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic.get("invalid_elements") == 2


def test_unhashable_extracted_value_returns_fail() -> None:
    """The extracted value is a list/dict (unhashable). The old code
    would crash inside ``set(target_values)`` — the hardened rubric
    skips those elements as invalid."""
    ctx = RubricContext(
        captures={
            "retrieval": {
                "chunks": [
                    {"doc_id": ["nested", "list"]},  # unhashable
                    {"doc_id": {"a": 1}},  # unhashable
                ]
            }
        },
        setup_data={"expected_doc_ids": ["doc_001"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.chunks",
        target_key="doc_id",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    assert verdict.status == "fail"
    assert verdict.diagnostic.get("invalid_elements") == 2


def test_partial_invalid_elements_still_evaluates_valid_ones() -> None:
    """3 valid + 2 invalid → use the 3 valid for the overlap calc and
    expose the invalid count in the diagnostic. This is the
    defence-in-depth case: don't throw away the whole evaluation just
    because some entries are broken."""
    ctx = RubricContext(
        captures={
            "retrieval": {
                "chunks": [
                    {"doc_id": "doc_001"},  # valid
                    "string_garbage",  # invalid: not a dict
                    {"doc_id": "doc_005"},  # valid
                    {"score": 0.4},  # invalid: missing doc_id
                    {"doc_id": "doc_009"},  # valid (not in gold)
                ]
            }
        },
        setup_data={"expected_doc_ids": ["doc_001", "doc_005"]},
    )
    rubric = OracleSetOverlap(
        target="retrieval.chunks",
        target_key="doc_id",
        gold_set_path="expected_doc_ids",
        min_recall=0.5,
    )
    verdict = rubric.judge(ctx)
    # Both gold items are present in the 3 valid entries → recall = 1.0.
    assert verdict.status == "pass"
    assert verdict.diagnostic["recall"] == 1.0
    assert verdict.diagnostic.get("invalid_elements") == 2
    assert set(verdict.diagnostic["matched"]) == {"doc_001", "doc_005"}
