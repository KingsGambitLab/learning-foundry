"""Tests for the four 'structural' rubric classes.

These rubrics inspect shape and primitive values only — no network, no
LLM, no setup_data lookup. They share the same error path: when the
target dotted path cannot be resolved against captures, they emit a
FAIL verdict with the missing-path diagnostic rather than propagating
the underlying ``KeyError`` / ``IndexError``.
"""
from __future__ import annotations

import pytest

from app.services.scenario_rubrics_base import (
    RUBRIC_REGISTRY,
    RubricContext,
)


# ============================================================
# SchemaMatch
# ============================================================


class TestSchemaMatch:
    def test_pass_when_all_required_fields_present(self) -> None:
        from app.services.scenario_rubrics_structural import SchemaMatch

        rubric = SchemaMatch(
            target="resp.body",
            must_have_fields=["answer", "citations"],
        )
        ctx = RubricContext(
            captures={"resp": {"body": {"answer": "x", "citations": []}}}
        )
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_fail_when_required_field_missing(self) -> None:
        from app.services.scenario_rubrics_structural import SchemaMatch

        rubric = SchemaMatch(
            target="resp.body",
            must_have_fields=["answer", "citations"],
        )
        ctx = RubricContext(captures={"resp": {"body": {"answer": "x"}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["missing_fields"] == ["citations"]

    def test_fail_when_field_type_does_not_match(self) -> None:
        from app.services.scenario_rubrics_structural import SchemaMatch

        rubric = SchemaMatch(
            target="resp.body",
            must_have_fields=["answer", "citations"],
            field_types={"answer": str, "citations": list},
        )
        ctx = RubricContext(
            captures={"resp": {"body": {"answer": 42, "citations": []}}}
        )
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert "answer" in verdict.diagnostic["type_mismatches"]
        assert "got_int" in verdict.diagnostic["type_mismatches"]["answer"]
        assert "expected_str" in verdict.diagnostic["type_mismatches"]["answer"]

    def test_pass_with_field_types_matching(self) -> None:
        from app.services.scenario_rubrics_structural import SchemaMatch

        rubric = SchemaMatch(
            target="resp.body",
            must_have_fields=["answer", "score"],
            field_types={"answer": str, "score": float},
        )
        ctx = RubricContext(
            captures={"resp": {"body": {"answer": "yes", "score": 0.95}}}
        )
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_fail_when_target_is_not_a_dict(self) -> None:
        from app.services.scenario_rubrics_structural import SchemaMatch

        rubric = SchemaMatch(
            target="resp.body",
            must_have_fields=["answer"],
        )
        ctx = RubricContext(captures={"resp": {"body": "just a string"}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert "not a dict" in verdict.rationale.lower()

    def test_fail_with_missing_path(self) -> None:
        from app.services.scenario_rubrics_structural import SchemaMatch

        rubric = SchemaMatch(
            target="resp.body",
            must_have_fields=["answer"],
        )
        ctx = RubricContext(captures={"resp": {}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["missing_path"] == "resp.body"

    def test_registered_under_schema_match(self) -> None:
        import app.services.scenario_rubrics_structural  # noqa: F401

        assert "schema_match" in RUBRIC_REGISTRY


# ============================================================
# LiteralMatch
# ============================================================


class TestLiteralMatch:
    def test_pass_when_value_equals_expected(self) -> None:
        from app.services.scenario_rubrics_structural import LiteralMatch

        rubric = LiteralMatch(target="resp.status_code", expected=200)
        ctx = RubricContext(captures={"resp": {"status_code": 200}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_fail_when_value_differs(self) -> None:
        from app.services.scenario_rubrics_structural import LiteralMatch

        rubric = LiteralMatch(target="resp.status_code", expected=200)
        ctx = RubricContext(captures={"resp": {"status_code": 500}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["got"] == 500
        assert verdict.diagnostic["expected"] == 200

    def test_pass_with_string_expected(self) -> None:
        from app.services.scenario_rubrics_structural import LiteralMatch

        rubric = LiteralMatch(target="resp.body.mode", expected="strict")
        ctx = RubricContext(captures={"resp": {"body": {"mode": "strict"}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_fail_with_missing_path(self) -> None:
        from app.services.scenario_rubrics_structural import LiteralMatch

        rubric = LiteralMatch(target="resp.body.mode", expected="strict")
        ctx = RubricContext(captures={"resp": {"body": {}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["missing_path"] == "resp.body.mode"

    def test_registered_under_literal_match(self) -> None:
        import app.services.scenario_rubrics_structural  # noqa: F401

        assert "literal_match" in RUBRIC_REGISTRY


# ============================================================
# RegexMatch
# ============================================================


class TestRegexMatch:
    def test_pass_when_string_matches_pattern(self) -> None:
        from app.services.scenario_rubrics_structural import RegexMatch

        rubric = RegexMatch(target="resp.body.id", pattern=r"doc_\d{3}")
        ctx = RubricContext(captures={"resp": {"body": {"id": "doc_001"}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_fail_when_string_does_not_match_pattern(self) -> None:
        from app.services.scenario_rubrics_structural import RegexMatch

        rubric = RegexMatch(target="resp.body.id", pattern=r"doc_\d{3}")
        ctx = RubricContext(captures={"resp": {"body": {"id": "DOC-001"}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["got"] == "DOC-001"
        assert verdict.diagnostic["pattern"] == r"doc_\d{3}"

    def test_fail_when_pattern_only_partially_matches(self) -> None:
        # fullmatch means anchored at both ends — partial isn't enough.
        from app.services.scenario_rubrics_structural import RegexMatch

        rubric = RegexMatch(target="resp.body.id", pattern=r"doc_\d{3}")
        ctx = RubricContext(
            captures={"resp": {"body": {"id": "doc_001_extra"}}}
        )
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"

    def test_fail_when_value_is_not_a_string(self) -> None:
        from app.services.scenario_rubrics_structural import RegexMatch

        rubric = RegexMatch(target="resp.body.id", pattern=r"\d+")
        ctx = RubricContext(captures={"resp": {"body": {"id": 42}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert "not a string" in verdict.rationale.lower()

    def test_fail_with_missing_path(self) -> None:
        from app.services.scenario_rubrics_structural import RegexMatch

        rubric = RegexMatch(target="resp.body.id", pattern=r"doc_\d+")
        ctx = RubricContext(captures={"resp": {"body": {}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["missing_path"] == "resp.body.id"

    def test_registered_under_regex_match(self) -> None:
        import app.services.scenario_rubrics_structural  # noqa: F401

        assert "regex_match" in RUBRIC_REGISTRY


# ============================================================
# NumericRange
# ============================================================


class TestNumericRange:
    def test_pass_within_inclusive_bounds(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="resp.body.score", min_value=0.0, max_value=1.0)
        ctx = RubricContext(captures={"resp": {"body": {"score": 0.5}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_pass_at_lower_bound(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="resp.body.score", min_value=0.0, max_value=1.0)
        ctx = RubricContext(captures={"resp": {"body": {"score": 0.0}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_pass_at_upper_bound(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="resp.body.score", min_value=0.0, max_value=1.0)
        ctx = RubricContext(captures={"resp": {"body": {"score": 1.0}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_fail_below_min(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="resp.body.score", min_value=0.0, max_value=1.0)
        ctx = RubricContext(captures={"resp": {"body": {"score": -0.1}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["got"] == -0.1
        assert verdict.diagnostic["min_value"] == 0.0

    def test_fail_above_max(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="resp.body.score", min_value=0.0, max_value=1.0)
        ctx = RubricContext(captures={"resp": {"body": {"score": 1.5}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["got"] == 1.5

    def test_pass_with_only_min_bound(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="count", min_value=1, max_value=None)
        ctx = RubricContext(captures={"count": 9999})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_pass_with_only_max_bound(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="latency_ms", min_value=None, max_value=500)
        ctx = RubricContext(captures={"latency_ms": 12})
        verdict = rubric.judge(ctx)
        assert verdict.status == "pass"

    def test_constructor_rejects_no_bounds(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        with pytest.raises(ValueError):
            NumericRange(target="x", min_value=None, max_value=None)

    def test_fail_when_value_is_not_numeric(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="resp.body.score", min_value=0.0, max_value=1.0)
        ctx = RubricContext(captures={"resp": {"body": {"score": "high"}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert "not numeric" in verdict.rationale.lower()

    def test_fail_when_value_is_bool(self) -> None:
        # bool is a subclass of int in Python — explicitly excluded.
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="flag", min_value=0, max_value=1)
        ctx = RubricContext(captures={"flag": True})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert "not numeric" in verdict.rationale.lower()

    def test_fail_with_missing_path(self) -> None:
        from app.services.scenario_rubrics_structural import NumericRange

        rubric = NumericRange(target="resp.body.score", min_value=0.0, max_value=1.0)
        ctx = RubricContext(captures={"resp": {"body": {}}})
        verdict = rubric.judge(ctx)
        assert verdict.status == "fail"
        assert verdict.diagnostic["missing_path"] == "resp.body.score"

    def test_registered_under_numeric_range(self) -> None:
        import app.services.scenario_rubrics_structural  # noqa: F401

        assert "numeric_range" in RUBRIC_REGISTRY
