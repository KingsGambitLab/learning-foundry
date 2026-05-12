from __future__ import annotations

from app.domain.task_agent import (
    AssignmentDesignSpec,
    DeliverableSpec,
    EndpointSpec,
    TaskAgentServiceSpec,
)
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs
from app.services.public_surface_quality import (
    collection_slug_for_entity,
    meaningful_domain_entities,
)


def _primary_entity_for_design(design: AssignmentDesignSpec) -> str:
    entities = meaningful_domain_entities(design.project_contract.core_entities)
    return entities[0] if entities else design.project_contract.system_kind or "resource"


def _resource_slug_for_design(title: str, design: AssignmentDesignSpec) -> str:
    del title
    return collection_slug_for_entity(_primary_entity_for_design(design))


def _public_endpoints(slug: str, design: AssignmentDesignSpec) -> list[EndpointSpec]:
    base = f"/{slug}"
    item = f"{base}/{{id}}"
    endpoints = [
        EndpointSpec(method="GET", path="/health"),
        EndpointSpec(method="POST", path=base),
        EndpointSpec(method="GET", path=item),
    ]
    if design.capabilities.approval_flow_required:
        endpoints.append(EndpointSpec(method="POST", path=f"{item}/approve"))
    if design.capabilities.traceability_required:
        endpoints.append(EndpointSpec(method="GET", path=f"{item}/trace"))
    return endpoints


def build_task_agent_scaffold(
    *,
    title: str,
    summary: str,
    design_spec: AssignmentDesignSpec,
    planner_deliverables: list[DeliverableSpec],
) -> tuple[TaskAgentServiceSpec, str]:
    """Build a TaskAgentServiceSpec from the planner-supplied deliverable list.

    Pass 10 Job A: the scaffold builder no longer invents a fixed-size
    deliverable list per ProjectFamily. The caller (planner pipeline) is the
    source of truth for how many deliverables a course has and what each one
    is about. The OpenAI customization step still refines learner-facing
    fields on top of this base.
    """
    slug = _resource_slug_for_design(title, design_spec)
    spec = TaskAgentServiceSpec(
        title=title,
        summary=summary,
        package_type=design_spec.course_structure.package_type,
        risk_class=design_spec.risk_class,
        domain_pack=design_spec.domain_pack,
        overlays=list(design_spec.overlays),
        course_structure=design_spec.course_structure.model_copy(deep=True),
        runtime_dependencies=design_spec.runtime_dependencies.model_copy(deep=True),
        capabilities=design_spec.capabilities.model_copy(deep=True),
        assessment_strategy=design_spec.assessment_strategy.model_copy(deep=True),
        project_contract=design_spec.project_contract.model_copy(deep=True),
        public_endpoints=_public_endpoints(slug, design_spec),
        deliverables=[deliverable.model_copy(deep=True) for deliverable in planner_deliverables],
    )
    spec = ensure_task_agent_deliverable_briefs(spec, overwrite=True)
    origin_template = design_spec.project_contract.family.value
    return spec, origin_template
