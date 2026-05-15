"""Regression test for trivial-rubric over-firing.

Bug 22 from docs/superpowers/bugs/2026-05-15-autonomous-fix-loop.md.

Live RAG/CRAG smoke (2026-05-14) flagged ``boundary_conflicting_passages``
and ``out_of_scope_insufficient_evidence`` as "uses only structural
rubrics" — yet ``malformed_input`` scenarios that legitimately ONLY
need ``numeric_range`` on the response status code SHOULD be allowed
to be structural-only.

The fix introduces a category exemption: ``malformed_input`` and
``idempotency`` scenarios are exempt from the trivial-rubric check.
Other categories still need at least one semantic rubric.
"""
from __future__ import annotations

import unittest

from app.services.oracle_validation import _scenario_only_trivial_rubrics
from app.services.scenario_loader import Scenario, RubricSpec, TraceStep


def _scenario(category: str, rubric_kinds: list[str]) -> Scenario:
    return Scenario(
        id=f"test_{category}",
        description="test",
        category=category,
        quality_bar_ids=["x"],
        trace=[TraceStep(id="s1", method="GET", path="/x")],
        rubrics=[
            RubricSpec(kind=kind, config={"target": "s1.body"})
            for kind in rubric_kinds
        ],
    )


class CategoryExemptionTests(unittest.TestCase):
    def test_malformed_input_with_numeric_range_only_is_not_trivial(self) -> None:
        s = _scenario("malformed_input", ["numeric_range"])
        self.assertFalse(_scenario_only_trivial_rubrics(s))

    def test_malformed_input_with_schema_and_numeric_range_only_is_not_trivial(
        self,
    ) -> None:
        s = _scenario("malformed_input", ["schema_match", "numeric_range"])
        self.assertFalse(_scenario_only_trivial_rubrics(s))

    def test_idempotency_with_literal_only_is_not_trivial(self) -> None:
        s = _scenario("idempotency", ["literal_match"])
        self.assertFalse(_scenario_only_trivial_rubrics(s))

    def test_happy_path_with_only_schema_match_is_still_trivial(self) -> None:
        """The exemption is targeted to malformed_input/idempotency
        only — happy_path with only structural rubrics is still a
        bug."""
        s = _scenario("happy_path", ["schema_match"])
        self.assertTrue(_scenario_only_trivial_rubrics(s))

    def test_happy_path_with_oracle_set_overlap_is_not_trivial(self) -> None:
        s = _scenario("happy_path", ["schema_match", "oracle_set_overlap"])
        self.assertFalse(_scenario_only_trivial_rubrics(s))


if __name__ == "__main__":
    unittest.main()
