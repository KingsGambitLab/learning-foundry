"""Cluster outcome-submit diagnostics into tech-lead-style feedback.

After commit c8c9e7ce the per-scenario diagnostics were humanized but
the scorecard still listed all 19+ failures equally weighted. A real
reviewer wouldn't do that — they'd group root causes and prioritize:

    "1 of 20 passing. Most failures cluster around 2 missing response
    fields. Fix `eval.regression_diff` first (8 scenarios), then
    `eval.summary` (6 scenarios)."

``build_outcome_feedback`` produces that shape. It returns a
``LearnerReviewGuidance`` Pydantic model so the existing
``renderLearnerGuidance`` JS block can render it without any UI
changes beyond populating the field.
"""
from __future__ import annotations

import unittest

from app.domain.grading import GradeStatus, TestGradeResult
from app.services.lms_service import build_outcome_feedback


def _result(test_id: str, diagnostics: list[str], passed: bool = False) -> TestGradeResult:
    return TestGradeResult(
        test_id=test_id,
        test_type="scenario",
        kind="happy_path",
        status=GradeStatus.passed if passed else GradeStatus.failed,
        score=1.0 if passed else 0.0,
        summary=f"Scenario {test_id}",
        diagnostics=diagnostics,
    )


class BuildOutcomeFeedbackTests(unittest.TestCase):
    def test_returns_none_when_no_failures(self) -> None:
        results = [_result("a", [], passed=True), _result("b", [], passed=True)]
        self.assertIsNone(build_outcome_feedback(results))

    def test_clusters_missing_field_by_root_path(self) -> None:
        """Sub-field misses under the same root field collapse into one
        cluster — ``eval.regression_diff.baseline_present`` and
        ``eval.regression_diff.new_passes`` are both ``eval.regression_diff``
        being absent, not two independent fixes."""
        results = [
            _result("s1", [
                "Response is missing field `eval.regression_diff`",
                "Response is missing field `eval.regression_diff.baseline_present`",
            ]),
            _result("s2", [
                "Response is missing field `eval.regression_diff.new_passes`",
            ]),
            _result("s3", [
                "Response is missing field `eval.regression_diff.pass_rate_delta`",
            ]),
        ]
        feedback = build_outcome_feedback(results)
        self.assertIsNotNone(feedback)
        # 3 scenarios all hit the same root cause.
        self.assertEqual(len(feedback.likely_root_cause), 1)
        cause = feedback.likely_root_cause[0]
        self.assertIn("eval.regression_diff", cause)
        self.assertIn("3", cause)  # impact count

    def test_separate_root_fields_become_separate_clusters(self) -> None:
        results = [
            _result("s1", ["Response is missing field `eval.regression_diff`"]),
            _result("s2", ["Response is missing field `eval.regression_diff.new_passes`"]),
            _result("s3", ["Response is missing field `eval.summary.total_cases`"]),
            _result("s4", ["Response is missing field `eval.summary.pass_rate`"]),
        ]
        feedback = build_outcome_feedback(results)
        self.assertIsNotNone(feedback)
        self.assertEqual(len(feedback.likely_root_cause), 2)
        # Ranked by impact desc.
        self.assertIn("eval.regression_diff", feedback.likely_root_cause[0])
        self.assertIn("eval.summary", feedback.likely_root_cause[1])

    def test_behavior_diagnostic_becomes_its_own_cluster(self) -> None:
        results = [
            _result("s1", ["Expected `bad_eval.status` to be `422`, got `400`"]),
            _result("s2", ["Response is missing field `eval.summary.total_cases`"]),
        ]
        feedback = build_outcome_feedback(results)
        self.assertIsNotNone(feedback)
        causes = "\n".join(feedback.likely_root_cause)
        self.assertIn("bad_eval.status", causes)
        self.assertIn("422", causes)
        self.assertIn("eval.summary", causes)

    def test_schema_mismatch_clusters_together(self) -> None:
        results = [
            _result("s1", ["Response shape doesn't match the required schema"]),
            _result("s2", ["Response shape doesn't match the required schema"]),
            _result("s3", ["Response shape doesn't match the required schema"]),
        ]
        feedback = build_outcome_feedback(results)
        self.assertIsNotNone(feedback)
        self.assertEqual(len(feedback.likely_root_cause), 1)
        self.assertIn("3", feedback.likely_root_cause[0])
        self.assertIn("schema", feedback.likely_root_cause[0].lower())

    def test_top_n_capped(self) -> None:
        """Even when there are many distinct clusters, only the top
        few make the priority list — otherwise it's just noise again."""
        results = [
            _result(f"s{i}", [f"Response is missing field `field_{i}.subfield`"])
            for i in range(10)
        ]
        feedback = build_outcome_feedback(results)
        self.assertIsNotNone(feedback)
        # Top 5 max, sorted by impact (here all equal at 1 — order is stable).
        self.assertLessEqual(len(feedback.likely_root_cause), 5)

    def test_headline_includes_pass_count_and_top_cluster(self) -> None:
        results = (
            [_result(f"p{i}", [], passed=True) for i in range(2)]
            + [_result(f"f{i}", ["Response is missing field `eval.regression_diff`"]) for i in range(8)]
        )
        feedback = build_outcome_feedback(results)
        self.assertIsNotNone(feedback)
        self.assertIn("2", feedback.learner_feedback)  # passing count
        self.assertIn("10", feedback.learner_feedback)  # total
        # Top cluster surfaced in the headline.
        self.assertIn("eval.regression_diff", feedback.learner_feedback)


if __name__ == "__main__":
    unittest.main()
