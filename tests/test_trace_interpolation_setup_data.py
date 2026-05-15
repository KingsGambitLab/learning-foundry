"""Regression tests for trace-body interpolation reading setup_data.

Bug 27 from docs/superpowers/bugs/2026-05-15-autonomous-fix-loop.md.

Before this fix, ``${setup_data.X.Y}`` placeholders in scenario YAML
trace bodies never resolved — the interpolator only read from
``captures``. Curated scenarios had to inline question + retrieval
pool into every trace, which is wasteful (the data lives in
``_setup/`` already).

After this fix, scenarios can interpolate directly::

    body:
      question: "${setup_data.queries.q_0001.query}"
      search_results: "${setup_data.search_results_index.q_0001}"

These tests use the EXACT setup_data shape that the LLM-emitted
curated scenarios in workspaces/outcome/course_f918e889a33c/private/
generate at materialize time, so we know the new interpolator works
against the actual data structure.
"""
from __future__ import annotations

import unittest

from app.services.scenario_trace_runner import (
    InterpolationError,
    _interpolate_body,
    interpolate,
)


# Real-data shape captured from
# workspaces/outcome/course_f918e889a33c/private/grader/_setup/
LIVE_SETUP_DATA = {
    "queries": {
        "q_0001": {"query": "What was Acme Corp's Q3 2024 total revenue?", "domain": "finance"},
        "q_fp_0001": {"query": "When did Acme Corp file bankruptcy in 2024?"},
    },
    "search_results_index": {
        "q_0001": [
            {"passage_id": "acme_q3_24_rev", "text": "Q3 2024 revenue was $12.4 billion."},
            {"passage_id": "acme_q3_24_margin", "text": "Operating margin was 18.6%."},
        ],
        "q_fp_0001": [
            {"passage_id": "acme_cash", "text": "Strong cash balance reported."},
        ],
    },
    "gold_answers": {
        "q_0001": {"answer": "$12.4 billion", "alt_ans": ["12.4B"]},
    },
    "gold_supports": {
        "q_0001": ["acme_q3_24_rev"],
    },
}


class StringInterpolationTests(unittest.TestCase):
    def test_setup_data_dotted_path_resolves(self) -> None:
        out = interpolate(
            "Question: ${setup_data.queries.q_0001.query}",
            captures={},
            setup_data=LIVE_SETUP_DATA,
        )
        self.assertEqual(
            out,
            "Question: What was Acme Corp's Q3 2024 total revenue?",
        )

    def test_setup_data_without_setup_data_arg_raises(self) -> None:
        with self.assertRaises(InterpolationError):
            interpolate("${setup_data.queries.q_0001.query}", captures={})

    def test_course_meta_routing(self) -> None:
        out = interpolate(
            "${course_meta.title}",
            captures={},
            course_meta={"title": "Finance RAG"},
        )
        self.assertEqual(out, "Finance RAG")

    def test_capture_path_still_works_when_setup_data_present(self) -> None:
        """Captures path resolution unchanged when setup_data is also passed."""
        out = interpolate(
            "${step1.body.answer}",
            captures={
                "step1": {
                    "body": {"answer": "Yes"},
                    "status": 200,
                    "headers": {},
                }
            },
            setup_data=LIVE_SETUP_DATA,
        )
        self.assertEqual(out, "Yes")

    def test_setup_data_missing_key_raises(self) -> None:
        with self.assertRaises(InterpolationError) as cm:
            interpolate(
                "${setup_data.queries.q_NOPE.query}",
                captures={},
                setup_data=LIVE_SETUP_DATA,
            )
        self.assertIn("setup_data", str(cm.exception))


class BodyInterpolationTests(unittest.TestCase):
    """The trace's request body is a dict whose leaf strings get
    interpolated. Strings that are EXACTLY one placeholder must return
    the resolved value (often a list/dict) verbatim, not its repr.
    """

    def test_body_with_setup_data_question_resolves_to_string(self) -> None:
        out = _interpolate_body(
            {
                "question": "${setup_data.queries.q_0001.query}",
                "extra": "hello",
            },
            captures={},
            setup_data=LIVE_SETUP_DATA,
        )
        self.assertEqual(out["question"], "What was Acme Corp's Q3 2024 total revenue?")
        self.assertEqual(out["extra"], "hello")

    def test_body_with_search_results_placeholder_resolves_to_list(self) -> None:
        """The KEY ergonomic win: ``${setup_data.search_results_index.q_0001}``
        returns the LIST verbatim, not its stringified form. Without
        this, scenarios that POST a list of passages would send
        ``"[{'passage_id': ...}, ...]"`` as a string — the service
        would 422 on schema validation."""
        out = _interpolate_body(
            {
                "question": "${setup_data.queries.q_0001.query}",
                "search_results": "${setup_data.search_results_index.q_0001}",
            },
            captures={},
            setup_data=LIVE_SETUP_DATA,
        )
        self.assertIsInstance(out["search_results"], list)
        self.assertEqual(len(out["search_results"]), 2)
        self.assertEqual(out["search_results"][0]["passage_id"], "acme_q3_24_rev")

    def test_nested_dict_recurses(self) -> None:
        out = _interpolate_body(
            {
                "outer": {
                    "inner": "${setup_data.queries.q_0001.query}",
                    "static": 42,
                }
            },
            captures={},
            setup_data=LIVE_SETUP_DATA,
        )
        self.assertEqual(
            out["outer"]["inner"], "What was Acme Corp's Q3 2024 total revenue?"
        )
        self.assertEqual(out["outer"]["static"], 42)


if __name__ == "__main__":
    unittest.main()
