"""Capture-target shorthand resolution for rubrics.

Bug surfaced 2026-05-15 grading the Versioned Prompt Eval course
(``enrollment_c161709f4b82``, 1/20 passing). LLM-authored scenarios
emit rubric targets like ``eval.summary.total_cases`` expecting the
trace-runner body-shorthand convention to apply — i.e.
``captures["eval"]["body"]["summary"]["total_cases"]``. But rubrics
walk captures via ``resolve_path`` which is a literal nested-dict
walker with no shorthand, so the leading capture id resolves to the
capture entry ``{status, headers, body, raw, request}`` and the
second segment (``summary``) is looked up there directly. KeyError.

Result: every rubric using the shorthand fails with
``X not found in captures`` even when the learner's response had
exactly the right shape.

``resolve_capture_target`` is the fix: expand the path with a
``body`` segment after the capture id whenever segment 2 isn't
already one of ``status`` / ``headers`` / ``body``, then call
``resolve_path``. The shorthand matches the trace-runner's
placeholder ``${X.Y}`` convention so scenarios author paths once.
"""
from __future__ import annotations

import unittest

from app.services.scenario_rubrics_base import (
    expand_capture_shorthand,
    resolve_capture_target,
)


_CAPTURES = {
    "eval": {
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": {
            "candidate_version": "v2.0.0",
            "summary": {
                "total_cases": 2,
                "passed_cases": 0,
                "pass_rate": 0.0,
            },
            "case_results": [
                {"id": "exact_1", "passed": False},
            ],
        },
    }
}


class ExpandCaptureShorthandTests(unittest.TestCase):
    def test_two_segment_field_gets_body_prefix(self) -> None:
        self.assertEqual(expand_capture_shorthand("eval.summary"), "eval.body.summary")

    def test_three_segment_field_gets_body_prefix(self) -> None:
        self.assertEqual(
            expand_capture_shorthand("eval.summary.total_cases"),
            "eval.body.summary.total_cases",
        )

    def test_explicit_body_left_alone(self) -> None:
        self.assertEqual(
            expand_capture_shorthand("eval.body.summary"),
            "eval.body.summary",
        )

    def test_status_left_alone(self) -> None:
        self.assertEqual(expand_capture_shorthand("eval.status"), "eval.status")

    def test_headers_left_alone(self) -> None:
        self.assertEqual(
            expand_capture_shorthand("eval.headers.location"),
            "eval.headers.location",
        )

    def test_single_segment_capture_id_left_alone(self) -> None:
        # ``eval`` (just the capture id) resolves to the whole capture
        # entry — schema_match callers explicitly want the entry for
        # top-level body inspection via ``must_have_fields``.
        # We DO inject body here so the must_have_fields contract works
        # on the body dict.
        self.assertEqual(expand_capture_shorthand("eval"), "eval.body")

    def test_setup_data_prefix_left_alone(self) -> None:
        self.assertEqual(
            expand_capture_shorthand("setup_data.queries.q1"),
            "setup_data.queries.q1",
        )

    def test_course_meta_prefix_left_alone(self) -> None:
        self.assertEqual(
            expand_capture_shorthand("course_meta.title"),
            "course_meta.title",
        )

    def test_empty_string_passes_through(self) -> None:
        self.assertEqual(expand_capture_shorthand(""), "")


class ResolveCaptureTargetTests(unittest.TestCase):
    def test_two_segment_resolves_through_body(self) -> None:
        got = resolve_capture_target(_CAPTURES, "eval.summary")
        self.assertEqual(got["total_cases"], 2)

    def test_three_segment_resolves_through_body(self) -> None:
        self.assertEqual(
            resolve_capture_target(_CAPTURES, "eval.summary.total_cases"),
            2,
        )

    def test_status_does_not_double_prefix(self) -> None:
        self.assertEqual(resolve_capture_target(_CAPTURES, "eval.status"), 200)

    def test_top_level_capture_id_returns_body(self) -> None:
        body = resolve_capture_target(_CAPTURES, "eval")
        self.assertIn("candidate_version", body)
        self.assertIn("summary", body)

    def test_list_index_after_body_works(self) -> None:
        # ``eval.case_results[0].id`` → ``captures.eval.body.case_results[0].id``
        self.assertEqual(
            resolve_capture_target(_CAPTURES, "eval.case_results[0].id"),
            "exact_1",
        )


if __name__ == "__main__":
    unittest.main()
