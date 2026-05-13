"""ReviewerFinding must surface BundleValidationIssue.hint to the
repair LLM. Previously the 4 in-line BundleValidationIssue ->
ReviewerFinding conversions silently dropped the hint, which made the
LLM-judge's actionable revision suggestions invisible to the repair
node — causing it to fall back to "re-author everything" instead of a
targeted README rewrite, burning a full authoring cycle."""
from __future__ import annotations

import pytest

from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity
from app.services.bundle_validation import (
    BundleValidationIssue,
    BundleValidationLevel,
)


# ---------------- model: ReviewerFinding.hint ----------------


def test_reviewer_finding_accepts_hint_kwarg() -> None:
    finding = ReviewerFinding(
        category="pedagogy_review",
        severity=ReviewerFindingSeverity.error,
        title="course_readme_lacks_domain_grounding",
        detail="Course README should describe ... using concrete domain entities.",
        code="course_readme_lacks_domain_grounding",
        location="public/README.md",
        hint="Mention 'retrieval corpus' verbatim in the intro paragraph.",
    )
    assert finding.hint == "Mention 'retrieval corpus' verbatim in the intro paragraph."


def test_reviewer_finding_hint_defaults_to_none_for_back_compat() -> None:
    finding = ReviewerFinding(
        category="pedagogy_review",
        severity=ReviewerFindingSeverity.error,
        title="x",
        detail="y",
    )
    assert finding.hint is None


# ---------------- conversion helper ----------------


def test_bundle_issue_to_reviewer_finding_preserves_hint() -> None:
    from app.services.langgraph_assignment_graph import (
        _bundle_issue_to_reviewer_finding,
    )

    issue = BundleValidationIssue(
        level=BundleValidationLevel.error,
        code="starter_readme_lacks_domain_grounding",
        relative_path="public/checks/deliverable_2/README.md",
        message="Starter README should use concrete domain entities ...",
        hint="Use the word 'retrieval corpus' instead of 'API endpoint' in the intro.",
    )
    finding = _bundle_issue_to_reviewer_finding(issue, category="pedagogy_review")
    assert finding.hint == "Use the word 'retrieval corpus' instead of 'API endpoint' in the intro."
    assert finding.code == "starter_readme_lacks_domain_grounding"
    assert finding.location == "public/checks/deliverable_2/README.md"
    assert finding.severity == ReviewerFindingSeverity.error
    assert finding.category == "pedagogy_review"


def test_bundle_issue_to_reviewer_finding_maps_warning_severity() -> None:
    from app.services.langgraph_assignment_graph import (
        _bundle_issue_to_reviewer_finding,
    )

    issue = BundleValidationIssue(
        level=BundleValidationLevel.warning,
        code="x",
        relative_path="p",
        message="m",
    )
    finding = _bundle_issue_to_reviewer_finding(issue, category="code_review")
    assert finding.severity == ReviewerFindingSeverity.warning
    assert finding.hint is None  # no hint on the source, none on the finding


def test_bundle_issue_with_no_hint_yields_finding_with_none_hint() -> None:
    """Back-compat: BundleValidationIssue without a hint still converts
    cleanly; the conversion does not synthesize a hint of its own."""
    from app.services.langgraph_assignment_graph import (
        _bundle_issue_to_reviewer_finding,
    )

    issue = BundleValidationIssue(
        level=BundleValidationLevel.error,
        code="something_else",
        relative_path="some/path",
        message="something happened",
    )
    finding = _bundle_issue_to_reviewer_finding(issue, category="code_review")
    assert finding.hint is None
