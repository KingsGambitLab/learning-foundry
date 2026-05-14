"""Regression tests for the rubric kwarg normalization layer.

The live RAG/CRAG smoke (2026-05-14) saw every scenario rubric fail to
construct with ``TypeError: <RubricClass>.__init__() got an unexpected
keyword argument '<wrong-kwarg>'``. The scenario-author LLM emits
semantic-flavored names like ``literal_match.value`` while the rubric
classes expect ``literal_match.expected``. ``_build_rubric`` now
normalizes the kwargs before invoking the class.

These tests use REAL kwarg-drift cases harvested from
``/tmp/coursegen-resume-*.log``::

    56  SchemaMatch.__init__() got an unexpected keyword argument 'schema'
    28  LiteralMatch.__init__() got an unexpected keyword argument 'value'
    26  OracleSetOverlap.__init__() got an unexpected keyword argument 'oracle_set'
    26  LLMJudgeSemanticEq.__init__() got an unexpected keyword argument 'gold'
    22  SchemaMatch.__init__() got an unexpected keyword argument 'value'
    22  NumericRange.__init__() got an unexpected keyword argument 'min'
    14  LLMJudgeCoverage.__init__() got an unexpected keyword argument 'question'
    12  SubsetMatch.__init__() got an unexpected keyword argument 'value'
    10  OracleSetOverlap.__init__() got an unexpected keyword argument 'gold_path'
    ...

Each test exercises one observed drift case from the log; the assertion
is that the rubric now CONSTRUCTS (no TypeError) and that the
canonical attribute on the instance carries the LLM-emitted value.
"""
from __future__ import annotations

import unittest

from app.services.scenario_rubrics_base import RUBRIC_REGISTRY
# Import the rubric modules so they self-register with RUBRIC_REGISTRY.
# In production paths these get imported via langgraph_outcome_graph and
# friends; the standalone unit-test process needs them explicit.
from app.services import (  # noqa: F401 — import-for-side-effect
    scenario_rubrics_structural,
    scenario_rubrics_set,
    scenario_rubrics_oracle,
    scenario_rubrics_llm,
)
from app.services.scenario_trace_runner import (
    _normalize_rubric_kwargs,
    _build_rubric,
)


class KwargNormalizationTests(unittest.TestCase):
    """Pin individual rename / drop rules."""

    def test_literal_match_value_renames_to_expected(self) -> None:
        # 28 occurrences in the log.
        out = _normalize_rubric_kwargs(
            "literal_match", {"target": "x.y", "value": True}
        )
        self.assertEqual(out, {"target": "x.y", "expected": True})

    def test_numeric_range_min_max_rename(self) -> None:
        # 22 occurrences for `min`.
        out = _normalize_rubric_kwargs(
            "numeric_range", {"target": "x.y", "min": 400, "max": 499}
        )
        self.assertEqual(
            out, {"target": "x.y", "min_value": 400, "max_value": 499}
        )

    def test_oracle_set_overlap_gold_path_renames_to_gold_set_path(self) -> None:
        # 10 occurrences.
        out = _normalize_rubric_kwargs(
            "oracle_set_overlap",
            {"target": "x.y", "gold_path": "gold.q1"},
        )
        self.assertIn("gold_set_path", out)
        self.assertEqual(out["gold_set_path"], "gold.q1")

    def test_oracle_set_overlap_min_overlap_renames_to_min_recall(self) -> None:
        out = _normalize_rubric_kwargs(
            "oracle_set_overlap",
            {"target": "x", "gold_set_path": "g", "min_overlap": 0.5},
        )
        self.assertEqual(out["min_recall"], 0.5)
        self.assertNotIn("min_overlap", out)

    def test_llm_judge_semantic_eq_gold_renames_to_gold_path(self) -> None:
        # 26 occurrences.
        out = _normalize_rubric_kwargs(
            "llm_judge_semantic_eq",
            {"target": "x.y", "gold": "setup_data.gold.q1.answer"},
        )
        self.assertEqual(out["gold_path"], "setup_data.gold.q1.answer")

    def test_llm_judge_coverage_drops_question_and_reference(self) -> None:
        # 14× question, 12× reference.
        out = _normalize_rubric_kwargs(
            "llm_judge_coverage",
            {
                "target": "x.y",
                "must_contain_facts": ["a"],
                "question": "What is the answer?",
                "reference": "ignore me",
            },
        )
        self.assertNotIn("question", out)
        self.assertNotIn("reference", out)
        self.assertEqual(out["target"], "x.y")
        self.assertEqual(out["must_contain_facts"], ["a"])

    def test_llm_judge_false_premise_drops_question_evidence(self) -> None:
        # 6× question.
        out = _normalize_rubric_kwargs(
            "llm_judge_false_premise",
            {
                "target": "x.y",
                "expected_falsity_path": "setup_data.gold.q1.alt_ans",
                "question": "When did X file?",
                "evidence": "no evidence",
            },
        )
        self.assertEqual(out["target"], "x.y")
        self.assertEqual(
            out["expected_falsity_path"], "setup_data.gold.q1.alt_ans"
        )
        self.assertNotIn("question", out)
        self.assertNotIn("evidence", out)

    def test_subset_match_value_and_subset_of_rename_to_acceptable_source(
        self,
    ) -> None:
        # 12× value, 6× subset_of.
        out_a = _normalize_rubric_kwargs(
            "subset_match", {"target": "x", "value": "setup_data.passages"}
        )
        out_b = _normalize_rubric_kwargs(
            "subset_match",
            {"target": "x", "subset_of": "setup_data.passages"},
        )
        self.assertEqual(out_a["acceptable_source"], "setup_data.passages")
        self.assertEqual(out_b["acceptable_source"], "setup_data.passages")

    def test_behavioral_equivalence_aliases(self) -> None:
        # 8× value, 8× reference_target, 6× target_a.
        for llm_name in ("value", "reference_target", "reference_trace"):
            out = _normalize_rubric_kwargs(
                "behavioral_equivalence",
                {"target": "x", llm_name: "y"},
            )
            self.assertEqual(out["expected"], "y", f"failed for {llm_name}")

    def test_schema_match_schema_dict_extracts_required(self) -> None:
        # 56× SchemaMatch.schema.
        json_schema = {
            "type": "object",
            "required": ["answer", "citations", "abstained"],
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array"},
                "abstained": {"type": "boolean"},
            },
        }
        out = _normalize_rubric_kwargs(
            "schema_match", {"target": "x.y", "schema": json_schema}
        )
        self.assertNotIn("schema", out)
        self.assertEqual(
            sorted(out["must_have_fields"]),
            ["abstained", "answer", "citations"],
        )

    def test_schema_match_value_is_dropped(self) -> None:
        # 22× SchemaMatch.value (also a JSON Schema dict, same as schema).
        out = _normalize_rubric_kwargs(
            "schema_match",
            {"target": "x.y", "value": {"type": "object"}},
        )
        self.assertNotIn("value", out)


class BuildRubricEndToEndTests(unittest.TestCase):
    """Verify ``_build_rubric`` produces a constructed rubric for every
    LLM-drift case from the log — the path that previously raised
    TypeError and turned the scenario into a fail verdict."""

    def test_literal_match_value_now_constructs(self) -> None:
        rubric = _build_rubric(
            "literal_match",
            {"target": "$.body.abstained", "value": True},
            router=None,
        )
        self.assertEqual(rubric.expected, True)
        self.assertEqual(rubric.target, "$.body.abstained")

    def test_numeric_range_min_now_constructs(self) -> None:
        rubric = _build_rubric(
            "numeric_range",
            {"target": "$.status", "min": 400, "max": 499},
            router=None,
        )
        self.assertEqual(rubric.min_value, 400)
        self.assertEqual(rubric.max_value, 499)

    def test_schema_match_schema_dict_now_constructs(self) -> None:
        rubric = _build_rubric(
            "schema_match",
            {
                "target": "$.body",
                "schema": {
                    "type": "object",
                    "required": ["answer", "citations", "abstained"],
                },
            },
            router=None,
        )
        self.assertEqual(
            sorted(rubric.must_have_fields),
            ["abstained", "answer", "citations"],
        )

    def test_llm_judge_semantic_eq_gold_now_constructs(self) -> None:
        rubric = _build_rubric(
            "llm_judge_semantic_eq",
            {
                "target": "$.body.answer",
                "gold": "setup_data.gold.q1.answer",
            },
            router=None,
        )
        self.assertEqual(rubric.gold_path, "setup_data.gold.q1.answer")

    def test_oracle_set_overlap_gold_path_now_constructs(self) -> None:
        rubric = _build_rubric(
            "oracle_set_overlap",
            {
                "target": "$.body.citations",
                "gold_path": "gold_supports.q1",
                "min_overlap": 0.6,
            },
            router=None,
        )
        self.assertEqual(rubric.gold_set_path, "gold_supports.q1")
        self.assertEqual(rubric.min_recall, 0.6)


class PathPrefixUnificationTests(unittest.TestCase):
    """Bug 16: unify the two path-prefix conventions.

    Before this fix, ``oracle_set_overlap.gold_set_path`` walked
    ``ctx.setup_data`` directly (NO prefix), but ``llm_judge_*`` paths
    used the merged context (REQUIRED prefix). The LLM kept emitting
    one form into the other's kwarg.

    After this fix:
    - ``oracle_set_overlap.gold_set_path`` accepts either form (strips
      a leading ``setup_data.`` if present).
    - ``llm_judge_*`` paths get auto-prefixed in ``_normalize_rubric_kwargs``
      when the LLM emits a bare path that obviously addresses setup_data.
    """

    def test_oracle_set_overlap_accepts_prefixed_path(self) -> None:
        from app.services.scenario_rubrics_base import RubricContext

        rubric = _build_rubric(
            "oracle_set_overlap",
            {
                "target": "call.body.citations",
                "gold_set_path": "setup_data.gold_supports.q1",
            },
            router=None,
        )
        ctx = RubricContext(
            captures={
                "call": {"body": {"citations": ["acme_q3_24_rev"]}, "status": 200, "headers": {}}
            },
            setup_data={"gold_supports": {"q1": ["acme_q3_24_rev"]}},
        )
        verdict = rubric.judge(ctx)
        self.assertEqual(verdict.status, "pass")

    def test_oracle_set_overlap_accepts_bare_path(self) -> None:
        """Original convention still works."""
        from app.services.scenario_rubrics_base import RubricContext

        rubric = _build_rubric(
            "oracle_set_overlap",
            {
                "target": "call.body.citations",
                "gold_set_path": "gold_supports.q1",
            },
            router=None,
        )
        ctx = RubricContext(
            captures={
                "call": {"body": {"citations": ["acme_q3_24_rev"]}, "status": 200, "headers": {}}
            },
            setup_data={"gold_supports": {"q1": ["acme_q3_24_rev"]}},
        )
        verdict = rubric.judge(ctx)
        self.assertEqual(verdict.status, "pass")

    def test_llm_judge_semantic_eq_bare_path_gets_prefixed(self) -> None:
        """LLM emits ``gold_answers.q1.answer`` (no prefix) → normalizer
        auto-prefixes to ``setup_data.gold_answers.q1.answer`` so the
        merged-context resolver can reach it.
        """
        out = _normalize_rubric_kwargs(
            "llm_judge_semantic_eq",
            {
                "target": "call.body.answer",
                "gold_path": "gold_answers.q1.answer",
                "alt_path": "gold_answers.q1.alt_ans",
            },
        )
        self.assertEqual(out["gold_path"], "setup_data.gold_answers.q1.answer")
        self.assertEqual(out["alt_path"], "setup_data.gold_answers.q1.alt_ans")

    def test_llm_judge_prefixed_path_left_alone(self) -> None:
        """Already-prefixed paths shouldn't get a double prefix."""
        out = _normalize_rubric_kwargs(
            "llm_judge_semantic_eq",
            {
                "target": "call.body.answer",
                "gold_path": "setup_data.gold_answers.q1.answer",
            },
        )
        self.assertEqual(out["gold_path"], "setup_data.gold_answers.q1.answer")

    def test_llm_judge_capture_path_not_prefixed(self) -> None:
        """Paths that obviously address captures (start with $., contain
        ``response`` / ``call_``) must NOT get a setup_data prefix.
        """
        for capture_path in (
            "$.call.body.answer",
            "call_q1.response.body.answer",
            "response.body",
        ):
            out = _normalize_rubric_kwargs(
                "llm_judge_semantic_eq",
                {"target": "x", "gold_path": capture_path},
            )
            self.assertEqual(
                out["gold_path"],
                capture_path,
                f"normalizer mis-prefixed capture path {capture_path!r}",
            )


if __name__ == "__main__":
    unittest.main()
