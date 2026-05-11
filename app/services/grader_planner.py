from __future__ import annotations

from app.domain.grader import (
    DeliverableGraderPlan,
    GraderEntryKind,
    GraderPlanEntry,
    TaskAgentGraderPlanCollection,
    TestDependencies,
)
from app.domain.task_agent import TaskAgentServiceSpec


def build_task_agent_grader_plan(spec: TaskAgentServiceSpec, deliverable_id: str) -> DeliverableGraderPlan:
    deliverable = next((item for item in spec.deliverables if item.id == deliverable_id), None)
    if deliverable is None:
        raise ValueError(f"unknown deliverable id: {deliverable_id}")
    gate = spec.gate_for(deliverable_id)
    entries = [
        GraderPlanEntry(
            test_id=check.id,
            kind=GraderEntryKind.behavior,
            test_type="public_check",
            description=check.learner_goal,
            first_required_in=deliverable.id,
            controls=[],
            dependencies=TestDependencies(endpoint_paths=[check.request_path]),
            config=check.model_dump(mode="json"),
        )
        for check in deliverable.public_checks
    ]
    return DeliverableGraderPlan(
        deliverable_id=deliverable.id,
        deliverable_title=deliverable.title,
        deliverable_objective=deliverable.objective,
        starter_type=spec.runtime_dependencies.starter_type.value,
        overlay_ids=deliverable.overlay_ids,
        cumulative_deliverables=gate.cumulative_deliverables,
        active_behavior_ids=[check.id for check in deliverable.public_checks],
        active_quality_ids=[],
        total_tests=len(entries),
        endpoint_paths=sorted({check.request_path for check in deliverable.public_checks}),
        tool_ids=[],
        controls=[],
        entries=entries,
    )


def build_task_agent_review_area_plan(spec: TaskAgentServiceSpec, deliverable_id: str) -> DeliverableGraderPlan:
    return build_task_agent_grader_plan(spec, deliverable_id)


def build_all_task_agent_grader_plans(spec: TaskAgentServiceSpec) -> TaskAgentGraderPlanCollection:
    return TaskAgentGraderPlanCollection(
        title=spec.title,
        eval_dataset_id="public_checks",
        system_profile=spec.capabilities.summary_labels(),
        deliverable_plans=[build_task_agent_grader_plan(spec, deliverable.id) for deliverable in spec.deliverables],
    )


def build_all_task_agent_review_area_plans(spec: TaskAgentServiceSpec) -> TaskAgentGraderPlanCollection:
    return build_all_task_agent_grader_plans(spec)
