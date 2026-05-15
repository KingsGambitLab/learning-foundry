"""Wiring tests: bundle_validation.validate_bundle should consult the
LLM judge for the domain-grounding check, surface its suggestions as the
finding's ``hint``, and fall back to the substring check when the judge
is unavailable."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.public_surface_quality_llm import DomainGroundingVerdict


# These tests are designed to fail loudly until the new wiring lands.


def test_validator_files_no_finding_when_llm_judge_says_grounded() -> None:
    """When the judge returns ``is_grounded=True``, no
    ``course_readme_lacks_domain_grounding`` finding should be filed —
    even if the literal entity strings don't appear via substring
    match."""
    from app.services import bundle_validation as bv

    verdict = DomainGroundingVerdict(
        is_grounded=True,
        rationale="The README uses canonical RAG vocabulary including FAISS, BM25, recall, and citation grounding.",
    )
    issues: list = []
    with patch.object(bv, "evaluate_domain_grounding", return_value=verdict):
        bv._validate_course_readme(
            issues,
            relative_path="public/README.md",
            content="Build a RAG pipeline. Index corpus chunks in FAISS, retrieve with BM25, "
            "cite the source chunks in the grounded answer. Evaluate with Recall@5.",
            spec=_make_minimal_spec(core_entities=["retrieval corpus", "grounded response"]),
        )
    assert not any(
        issue.code == "course_readme_lacks_domain_grounding" for issue in issues
    ), [i.code for i in issues]


def test_validator_files_finding_with_actionable_hint_when_llm_rejects() -> None:
    """When the judge says not-grounded, the finding must surface the
    suggested_revisions in its hint field so the repair LLM has
    something concrete to act on."""
    from app.services import bundle_validation as bv

    verdict = DomainGroundingVerdict(
        is_grounded=False,
        rationale="The README never uses the canonical singular entity 'retrieval corpus' nor the phrase 'grounded response'.",
        suggested_revisions=[
            "Replace 'API endpoints' with 'retrieval corpus endpoints' at least once in the intro.",
            "Refer to the model output as 'grounded response' alongside 'answer'.",
        ],
    )
    issues: list = []
    with patch.object(bv, "evaluate_domain_grounding", return_value=verdict):
        bv._validate_course_readme(
            issues,
            relative_path="public/README.md",
            content="Build a backend. POST /retrieval-corpuses ingests data. The endpoint returns JSON.",
            spec=_make_minimal_spec(core_entities=["retrieval corpus", "grounded response"]),
        )
    matching = [i for i in issues if i.code == "course_readme_lacks_domain_grounding"]
    assert len(matching) == 1
    issue = matching[0]
    hint = getattr(issue, "hint", None)
    assert hint, f"expected actionable hint, got {hint!r}"
    assert "retrieval corpus" in hint or "grounded response" in hint


def test_validator_falls_back_to_substring_when_llm_unavailable() -> None:
    """If the judge returns None (no router, call failed), the
    existing substring rule must still gate the finding. This keeps
    offline / no-API-key CI working."""
    from app.services import bundle_validation as bv

    # Spec entities NOT mentioned in content + a generic marker present
    # → substring rule SHOULD fire.
    issues: list = []
    with patch.object(bv, "evaluate_domain_grounding", return_value=None):
        bv._validate_course_readme(
            issues,
            relative_path="public/README.md",
            content=(
                "## Core entities\n\n"
                "We will build a service surface. The primary request shape is JSON. "
                "There is no domain vocabulary here."
            ),
            spec=_make_minimal_spec(core_entities=["retrieval corpus", "grounded response"]),
        )
    matching = [i for i in issues if i.code == "course_readme_lacks_domain_grounding"]
    assert len(matching) == 1


def test_validator_falls_back_substring_passes_when_entities_present() -> None:
    """Fallback path mirrors current behavior: if the literal entity
    appears, the substring check does NOT fire even though the LLM
    judge is unavailable."""
    from app.services import bundle_validation as bv

    issues: list = []
    with patch.object(bv, "evaluate_domain_grounding", return_value=None):
        bv._validate_course_readme(
            issues,
            relative_path="public/README.md",
            content="The retrieval corpus is ingested into chunks. We return a grounded response.",
            spec=_make_minimal_spec(core_entities=["retrieval corpus", "grounded response"]),
        )
    assert not any(
        i.code == "course_readme_lacks_domain_grounding" for i in issues
    ), [i.code for i in issues]


# ---------------- helpers ----------------


def _make_minimal_spec(*, core_entities: list[str]):
    """Build a TaskAgentServiceSpec stub with just enough surface for
    the course-README validator. Real construction would require dozens
    of nested fields; we only touch what the validator reads."""
    spec = SimpleNamespace()
    spec.project_contract = SimpleNamespace(
        core_entities=core_entities,
        system_kind="Grounded retrieval and answer service",
    )
    spec.capabilities = SimpleNamespace(
        tool_use_required=False,
        approval_flow_required=False,
        traceability_required=False,
    )
    # `_published_endpoint_identities` reads ``spec.public_endpoints``.
    spec.public_endpoints = []
    return spec
