r"""Test the rubric-diagnostic to plain-English translator.

After the outcome-mode submit branch landed, learners saw raw rubric
diagnostics like ``schema_match (fail): eval.regression_diff not found
in captures`` in the scorecard — mechanically correct but not
actionable for someone trying to fix their code. The humanizer
rewrites those messages as advice (``Response is missing field
`eval.regression_diff` ``) before they reach the UI.

Rules are deterministic and regex-based; we keep them in Python so
unit tests pin the exact output strings the UI depends on.
"""
from __future__ import annotations

import unittest

from app.services.lms_service import humanize_diagnostic


class HumanizeDiagnosticTests(unittest.TestCase):
    def test_schema_match_missing_field(self) -> None:
        out = humanize_diagnostic(
            "schema_match (fail): eval.regression_diff not found in captures"
        )
        self.assertEqual(out, "Response is missing field `eval.regression_diff`")

    def test_behavioral_equivalence_missing_target_path(self) -> None:
        out = humanize_diagnostic(
            "behavioral_equivalence (fail): target path 'eval.regression_diff.baseline_present' not found in captures"
        )
        self.assertEqual(out, "Response is missing field `eval.regression_diff.baseline_present`")

    def test_behavioral_equivalence_expected_got(self) -> None:
        out = humanize_diagnostic(
            "behavioral_equivalence (fail): expected 422 at 'bad_eval.status', got 400"
        )
        self.assertEqual(out, "Expected `bad_eval.status` to be `422`, got `400`")

    def test_schema_match_dict_failed(self) -> None:
        out = humanize_diagnostic(
            "schema_match (fail): target dict failed schema check"
        )
        self.assertEqual(out, "Response shape doesn't match the required schema")

    def test_oracle_set_overlap_recall(self) -> None:
        out = humanize_diagnostic(
            "oracle_set_overlap (fail): recall 0.00 < threshold 0.50 (0/1 gold items found)"
        )
        self.assertEqual(
            out,
            "Retrieval recall 0.00 is below threshold 0.50 (matched 0 of 1 expected items)",
        )

    def test_llm_judge_target_not_present(self) -> None:
        out = humanize_diagnostic(
            "llm_judge_coverage (fail): target path 'eval.case_results' not present in captures"
        )
        self.assertEqual(out, "Response is missing field `eval.case_results`")

    def test_literal_match_missing(self) -> None:
        out = humanize_diagnostic(
            "literal_match (fail): eval.candidate_version not found in captures"
        )
        self.assertEqual(out, "Response is missing field `eval.candidate_version`")

    def test_numeric_range_missing(self) -> None:
        out = humanize_diagnostic(
            "numeric_range (fail): eval.summary.pass_rate not found in captures"
        )
        self.assertEqual(out, "Response is missing numeric field `eval.summary.pass_rate`")

    def test_unrecognized_diagnostic_strips_rubric_prefix(self) -> None:
        """When no rule matches we keep the message (never drop signal)
        but the internal ``rubric_kind (fail):`` prefix MUST be stripped
        — learners never see rubric-kind jargon."""
        out = humanize_diagnostic("some_future_rubric (fail): a brand new message")
        self.assertEqual(out, "a brand new message")
        self.assertNotIn("some_future_rubric", out)

    def test_llm_judge_rationale_keeps_prose_drops_kind(self) -> None:
        out = humanize_diagnostic(
            "llm_judge_semantic_eq (fail): The answer says 'not yet "
            "implemented', contradicting the reference."
        )
        self.assertNotIn("llm_judge_semantic_eq", out)
        self.assertTrue(out.startswith("The answer says"))

    def test_subset_empty_target_is_actionable(self) -> None:
        out = humanize_diagnostic(
            "subset_match (fail): target is empty; cannot check subset"
        )
        self.assertNotIn("subset_match", out)
        self.assertIn("no citations", out)

    def test_empty_input(self) -> None:
        self.assertEqual(humanize_diagnostic(""), "")


if __name__ == "__main__":
    unittest.main()
