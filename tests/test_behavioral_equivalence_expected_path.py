"""Regression tests for BehavioralEquivalence path-vs-path support.

Bug 19 from docs/superpowers/bugs/2026-05-15-autonomous-fix-loop.md.

The canonical idempotency check ``first_call.body.answer ==
second_call.body.answer`` was impossible before this fix because
``BehavioralEquivalence`` treated ``expected`` as a literal value. The
LLM kept emitting path strings into ``expected`` (e.g. ``"trace.second_call.body.answer"``)
which then compared the actual response string to the literal path
string — guaranteed mismatch.

Fix: add an ``expected_path`` kwarg that resolves like ``target``. The
kwarg normalizer routes the LLM's common cross-step kwarg names
(``reference_target``, ``target_b``, ``reference_trace``) to
``expected_path``.
"""
from __future__ import annotations

import unittest

from app.services.scenario_rubrics_base import RubricContext
from app.services.scenario_rubrics_set import BehavioralEquivalence
from app.services import (  # noqa: F401 — register rubrics
    scenario_rubrics_set,
)
from app.services.scenario_trace_runner import _build_rubric


class BehavioralEquivalencePathTests(unittest.TestCase):
    def _ctx(self) -> RubricContext:
        return RubricContext(
            captures={
                "first": {
                    "body": {"answer": "Yes, the company beat expectations."},
                    "status": 200,
                    "headers": {},
                },
                "second": {
                    "body": {"answer": "Yes, the company beat expectations."},
                    "status": 200,
                    "headers": {},
                },
                "third": {
                    "body": {"answer": "No, the company missed forecasts."},
                    "status": 200,
                    "headers": {},
                },
            }
        )

    def test_expected_path_equal_passes(self) -> None:
        rubric = BehavioralEquivalence(
            target="first.body.answer",
            expected_path="second.body.answer",
        )
        verdict = rubric.judge(self._ctx())
        self.assertEqual(verdict.status, "pass")

    def test_expected_path_mismatch_fails(self) -> None:
        rubric = BehavioralEquivalence(
            target="first.body.answer",
            expected_path="third.body.answer",
        )
        verdict = rubric.judge(self._ctx())
        self.assertEqual(verdict.status, "fail")

    def test_expected_path_missing_returns_fail_not_raise(self) -> None:
        rubric = BehavioralEquivalence(
            target="first.body.answer",
            expected_path="nope.body.answer",
        )
        verdict = rubric.judge(self._ctx())
        self.assertEqual(verdict.status, "fail")
        self.assertIn("expected_path", verdict.rationale)

    def test_literal_expected_still_works(self) -> None:
        """Original literal-value form unchanged."""
        rubric = BehavioralEquivalence(
            target="first.body.answer",
            expected="Yes, the company beat expectations.",
        )
        verdict = rubric.judge(self._ctx())
        self.assertEqual(verdict.status, "pass")

    def test_neither_expected_nor_expected_path_raises_typeerror(self) -> None:
        with self.assertRaises(TypeError) as cm:
            BehavioralEquivalence(target="first.body.answer")
        self.assertIn("expected", str(cm.exception))

    def test_case_insensitive_path_vs_path(self) -> None:
        rubric = BehavioralEquivalence(
            target="first.body.answer",
            expected_path="second.body.answer",
            case_sensitive=False,
        )
        ctx = RubricContext(
            captures={
                "first": {
                    "body": {"answer": "yes, BEAT expectations."},
                    "status": 200,
                    "headers": {},
                },
                "second": {
                    "body": {"answer": "Yes, beat EXPECTATIONS."},
                    "status": 200,
                    "headers": {},
                },
            }
        )
        verdict = rubric.judge(ctx)
        self.assertEqual(verdict.status, "pass")


class KwargNormalizationToExpectedPathTests(unittest.TestCase):
    """The kwarg normalizer translates LLM-emitted cross-step kwarg
    names (``reference_target``, ``target_b``, ``reference_trace``) to
    ``expected_path``."""

    def _ctx(self) -> RubricContext:
        return RubricContext(
            captures={
                "first": {
                    "body": {"answer": "X"},
                    "status": 200,
                    "headers": {},
                },
                "second": {
                    "body": {"answer": "X"},
                    "status": 200,
                    "headers": {},
                },
            }
        )

    def test_reference_target_routes_to_expected_path(self) -> None:
        rubric = _build_rubric(
            "behavioral_equivalence",
            {
                "target": "first.body.answer",
                "reference_target": "second.body.answer",
            },
            router=None,
        )
        verdict = rubric.judge(self._ctx())
        self.assertEqual(verdict.status, "pass")

    def test_target_b_routes_to_expected_path(self) -> None:
        rubric = _build_rubric(
            "behavioral_equivalence",
            {
                "target_a": "first.body.answer",
                "target_b": "second.body.answer",
            },
            router=None,
        )
        verdict = rubric.judge(self._ctx())
        self.assertEqual(verdict.status, "pass")


if __name__ == "__main__":
    unittest.main()
