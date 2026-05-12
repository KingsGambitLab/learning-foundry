"""Pass 10 Job A: ``build_task_agent_scaffold`` honours the planner's deliverable list.

The legacy ``_family_deliverables`` helper returned a fixed four-element list for
every ProjectFamily, dropping any extra deliverables emitted by the OpenAI
planner. These tests pin the new behaviour: the planner's list flows through
1:1, with no padding, truncation, or hardcoded family templates.
"""

from __future__ import annotations

import pytest

from app.domain.registry import PackageType
from app.domain.task_agent import DeliverableSpec
from app.services.assignment_design_inference import infer_assignment_design
from app.services.task_agent_scaffolds import build_task_agent_scaffold


def _inventory_design():
    inferred = infer_assignment_design(
        title="Inventory Reservation Service",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    assert inferred.design_spec is not None
    return inferred.design_spec


def _planner_deliverables(titles: list[str]) -> list[DeliverableSpec]:
    return [
        DeliverableSpec(
            id=f"deliverable_{index}",
            title=title,
            objective=f"Build the {title.lower()} surface.",
            learning_outcomes=[],
            overlay_ids=[],
        )
        for index, title in enumerate(titles, start=1)
    ]


def test_build_task_agent_scaffold_emits_planner_supplied_deliverable_count() -> None:
    design_spec = _inventory_design()
    planner_deliverables = _planner_deliverables(
        [
            "Reservation API and storage contract",
            "Reservation read/write correctness",
            "Reservation observability and recovery",
            "Reservation production hardening",
            "Reservation rate limiting and back-pressure",
        ]
    )

    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Concurrency-safe reservations across warehouses.",
        design_spec=design_spec,
        planner_deliverables=planner_deliverables,
    )

    assert len(spec.deliverables) == 5
    assert [deliverable.title for deliverable in spec.deliverables] == [
        "Reservation API and storage contract",
        "Reservation read/write correctness",
        "Reservation observability and recovery",
        "Reservation production hardening",
        "Reservation rate limiting and back-pressure",
    ]


def test_build_task_agent_scaffold_requires_planner_deliverables() -> None:
    design_spec = _inventory_design()
    with pytest.raises(TypeError):
        build_task_agent_scaffold(  # type: ignore[call-arg]
            title="Inventory Reservation Service",
            summary="Concurrency-safe reservations across warehouses.",
            design_spec=design_spec,
        )


def test_build_task_agent_scaffold_passes_through_two_deliverables() -> None:
    """Planner emitted exactly two deliverables - no padding to four."""
    design_spec = _inventory_design()
    planner_deliverables = _planner_deliverables(
        [
            "Reservation API contract",
            "Reservation production hardening",
        ]
    )

    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Concurrency-safe reservations across warehouses.",
        design_spec=design_spec,
        planner_deliverables=planner_deliverables,
    )

    assert len(spec.deliverables) == 2
