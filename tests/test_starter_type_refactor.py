"""Tests pinning the post-refactor StarterType invariants.

Pass 1 of a multi-pass refactor: the only valid starter types are `empty` and
`partial`. The four legacy values (`bare_stub`, `partial_implementation`,
`working_buggy`, `working_suboptimal`) are gone. Deliverables no longer carry
their own `starter_type` -- that lives on `RuntimeDependencySpec` at the
course level.
"""
from __future__ import annotations

import unittest

from app.domain.registry import StarterType
from app.domain.task_agent import DeliverableSpec, RuntimeDependencySpec
from app.services.generated_test_harness import (
    BaselineSuiteOutcome,
    GeneratedTestBaselineVerifier,
    GeneratedTestSuiteReport,
)


class StarterTypeEnumShapeTests(unittest.TestCase):
    def test_starter_type_has_exactly_two_members_empty_and_partial(self) -> None:
        members = {member.name: member.value for member in StarterType}
        self.assertEqual(members, {"empty": "empty", "partial": "partial"})

    def test_legacy_enum_names_are_gone(self) -> None:
        for legacy in ("bare_stub", "partial_implementation", "working_buggy", "working_suboptimal"):
            self.assertNotIn(legacy, StarterType.__members__)

    def test_runtime_dependency_spec_default_starter_type_is_partial(self) -> None:
        spec = RuntimeDependencySpec(execution_surface="http_service")
        self.assertEqual(spec.starter_type, StarterType.partial)

    def test_deliverable_spec_has_no_starter_type_field(self) -> None:
        self.assertNotIn("starter_type", DeliverableSpec.model_fields)

    def test_deliverable_spec_constructs_without_starter_type(self) -> None:
        spec = DeliverableSpec(
            id="deliverable_1",
            title="Title",
            objective="Objective",
        )
        self.assertFalse(hasattr(spec, "starter_type"))


class ExpectationIssuesUnderNewEnumTests(unittest.TestCase):
    """The baseline verifier's `_expectation_issues` must flag partial-starter
    passes as errors. With the enum collapse, both `empty` and `partial`
    behave identically (must not pass starter suites); the legacy
    working_buggy/working_suboptimal branch is dead and must be removed.
    """

    def _report(self, *, suite_type: str, passed: bool) -> GeneratedTestSuiteReport:
        return GeneratedTestSuiteReport(
            suite_type=suite_type,
            command="noop",
            exit_code=0 if passed else 1,
            valid=True,
            passed=passed,
        )

    def _passing_starter_outcomes(self) -> list[BaselineSuiteOutcome]:
        return [
            BaselineSuiteOutcome(
                baseline="starter_repo",
                suite_type="visible",
                report=self._report(suite_type="visible", passed=True),
            ),
            BaselineSuiteOutcome(
                baseline="starter_repo",
                suite_type="hidden",
                report=self._report(suite_type="hidden", passed=True),
            ),
            BaselineSuiteOutcome(
                baseline="empty_repo",
                suite_type="visible",
                report=self._report(suite_type="visible", passed=False),
            ),
            BaselineSuiteOutcome(
                baseline="empty_repo",
                suite_type="hidden",
                report=self._report(suite_type="hidden", passed=False),
            ),
        ]

    def test_partial_starter_passing_visible_and_hidden_is_flagged(self) -> None:
        verifier = GeneratedTestBaselineVerifier()
        outcomes = self._passing_starter_outcomes()
        errors = verifier._expectation_issues(outcomes, StarterType.partial)
        codes = {issue.code for issue in errors}
        self.assertIn("starter_visible_tests_passed_partial_repo", codes)
        self.assertIn("starter_hidden_tests_passed_partial_repo", codes)

    def test_empty_starter_passing_visible_and_hidden_is_flagged(self) -> None:
        verifier = GeneratedTestBaselineVerifier()
        outcomes = self._passing_starter_outcomes()
        errors = verifier._expectation_issues(outcomes, StarterType.empty)
        codes = {issue.code for issue in errors}
        self.assertIn("starter_visible_tests_passed_partial_repo", codes)
        self.assertIn("starter_hidden_tests_passed_partial_repo", codes)
