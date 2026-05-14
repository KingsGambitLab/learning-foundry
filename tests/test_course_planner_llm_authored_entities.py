"""The course planner LLM is in a better position than a regex to pick
``system_kind`` and ``core_entities`` from a free-text brief. This
test set codifies the contract:

  - ``_CoursePlanPayload`` carries fields where the LLM writes them.
  - ``_normalize_raw_plan`` overrides the regex-extracted defaults
    with the LLM's choice when present.
  - When the LLM omits / leaves them blank, the regex fallback still
    applies (offline / no-LLM compat).
  - Per-deliverable design specs inherit the shared entities, too.

Specifically reproduces the "promptfoo course produced
``core_entities=['a small']``" failure mode — the brief contains the
colloquial phrase ``"exposes a small FastAPI service"`` which the
regex catches as a domain entity; the LLM should be authoring
``["prompt template", "promptfoo eval suite", ...]`` instead.
"""
from __future__ import annotations

import pytest

from app.domain.course import (
    CreatorCourseSetupInput,
    GenerateCourseFromBriefRequest,
)
from app.services.openai_course_planner import (
    OpenAICoursePlanner,
    _CoursePlanPayload,
)


# ---------------- _CoursePlanPayload shape ----------------


def test_course_plan_payload_accepts_llm_authored_entities() -> None:
    payload = _CoursePlanPayload(
        title="Tested, Versioned Prompt Pipelines with Promptfoo",
        summary="Move from trial-and-error prompts to a tested pipeline.",
        deliverables=[],
        system_kind="Versioned prompt pipeline harness",
        core_entities=[
            "prompt template",
            "promptfoo eval suite",
            "prompt test case",
        ],
    )
    assert payload.system_kind == "Versioned prompt pipeline harness"
    assert "prompt template" in payload.core_entities


def test_course_plan_payload_back_compat_defaults_for_legacy_callers() -> None:
    """Existing test fixtures and any cached LLM response that omits
    the new fields must still parse cleanly."""
    payload = _CoursePlanPayload(
        title="Some course",
        summary="A summary that meets the minimum.",
        deliverables=[],
    )
    assert payload.system_kind == ""
    assert payload.core_entities == []


# ---------------- end-to-end: payload → design spec ----------------


def _planner_for_tests() -> OpenAICoursePlanner:
    """OpenAICoursePlanner can be instantiated with enabled=False;
    that still exposes ``_normalize_raw_plan`` for unit testing."""
    return OpenAICoursePlanner(enabled=False)


def _basic_request(goal: str) -> GenerateCourseFromBriefRequest:
    return GenerateCourseFromBriefRequest(
        goal=goal,
        title="Test course",
        creator_setup=CreatorCourseSetupInput(
            implementation_language="Python",
            language_version="3.11",
            application_framework="FastAPI",
            package_manager="pip",
        ),
    )


def test_llm_authored_entities_override_regex_extraction_in_design_spec() -> None:
    """The exact failure mode that produced ``core_entities=['a small']``
    on the promptfoo run: the brief contains 'exposes a small FastAPI
    service'. Regex extraction would pick 'a small'. The LLM's
    authored value must win."""
    planner = _planner_for_tests()
    promptfoo_brief = (
        "Move from prompt-engineering by trial-and-error to a tested, "
        "versioned prompt pipeline using promptfoo. The learner exposes "
        "a small FastAPI service that serves prompt-driven endpoints."
    )
    raw_plan = {
        "title": "Tested, Versioned Prompt Pipelines with Promptfoo",
        "summary": "...",
        "package_type": "progressive_codebase_course",
        "deliverables": [
            {
                "title": "Versioned Prompt Template Registry",
                "summary": "...",
                "learning_outcomes": [],
            }
        ],
        "system_kind": "Versioned prompt pipeline harness",
        "core_entities": [
            "prompt template",
            "promptfoo eval suite",
        ],
    }
    plan = planner._normalize_raw_plan(_basic_request(promptfoo_brief), raw_plan)
    spec = plan.shared_design_spec
    assert spec is not None
    assert spec.project_contract.system_kind == "Versioned prompt pipeline harness"
    assert "prompt template" in spec.project_contract.core_entities
    assert "a small" not in spec.project_contract.core_entities
    # The shared spec's entities also propagate to per-deliverable specs.
    for d in plan.deliverables:
        assert d.design_spec.project_contract.system_kind == "Versioned prompt pipeline harness"
        assert "prompt template" in d.design_spec.project_contract.core_entities


def test_legacy_planner_response_without_llm_entities_keeps_regex_fallback() -> None:
    """When the LLM payload omits the new fields, regex extraction
    still populates the design spec — back-compat for offline / no-LLM
    paths and cached responses."""
    planner = _planner_for_tests()
    raw_plan = {
        "title": "Inventory Reservation Service",
        "summary": "Build a concurrency-safe inventory reservation backend.",
        "package_type": "progressive_codebase_course",
        "deliverables": [
            {
                "title": "Reservation Endpoint",
                "summary": "Implement the reservation endpoint.",
                "learning_outcomes": [],
            }
        ],
        # NOTE: no system_kind, no core_entities
    }
    inventory_brief = (
        "Build a multi-warehouse inventory reservation service with FastAPI, "
        "Postgres, and Redis. Keep reservations correct under concurrency, "
        "retries, and stock transfers."
    )
    plan = planner._normalize_raw_plan(_basic_request(inventory_brief), raw_plan)
    spec = plan.shared_design_spec
    assert spec is not None
    # Whatever the regex extraction produced is fine — the important
    # invariant is that we did NOT crash on missing fields and DID
    # populate something non-empty.
    assert isinstance(spec.project_contract.system_kind, str)
    assert spec.project_contract.system_kind != ""
    assert isinstance(spec.project_contract.core_entities, list)


def test_llm_authored_entities_filter_empties_and_strip_whitespace() -> None:
    """Defensive: model output sometimes includes blank entries or
    whitespace. The normalization should drop them and preserve
    non-empty strings."""
    planner = _planner_for_tests()
    raw_plan = {
        "title": "Promptfoo Course",
        "summary": "...",
        "package_type": "progressive_codebase_course",
        "deliverables": [
            {"title": "D1", "summary": "...", "learning_outcomes": []}
        ],
        "system_kind": "  Versioned prompt pipeline harness  ",
        "core_entities": [
            "prompt template",
            "",
            "  promptfoo eval suite  ",
            None,  # type: ignore[list-item]
        ],
    }
    plan = planner._normalize_raw_plan(
        _basic_request("Promptfoo brief that is long enough to validate."),
        raw_plan,
    )
    spec = plan.shared_design_spec
    assert spec is not None
    assert spec.project_contract.system_kind == "Versioned prompt pipeline harness"
    assert spec.project_contract.core_entities == [
        "prompt template",
        "promptfoo eval suite",
    ]
