from __future__ import annotations

from app.domain.registry import StarterType
from app.domain.task_agent import (
    AssignmentDesignSpec,
    DeliverableSpec,
    EndpointSpec,
    ProjectFamily,
    TaskAgentServiceSpec,
)
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs
from app.services.public_surface_quality import (
    collection_slug_for_entity,
    meaningful_domain_entities,
    pluralize_phrase,
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


def _family_deliverables(design: AssignmentDesignSpec) -> list[DeliverableSpec]:
    family = design.project_contract.family
    entity = _primary_entity_for_design(design)
    entity_plural = pluralize_phrase(entity)
    if family == ProjectFamily.transactional_stateful_service:
        return [
            DeliverableSpec(
                id="deliverable_1",
                title=f"{entity.title()} contract and state model",
                objective=f"Define the public API, durable model, and key state transitions for {entity_plural}.",
                starter_type=StarterType.working_buggy,
            ),
            DeliverableSpec(
                id="deliverable_2",
                title=f"{entity.title()} read and write correctness",
                objective=f"Make the main {entity_plural} read and write paths correct under retries and concurrent requests.",
                starter_type=StarterType.partial_implementation,
            ),
            DeliverableSpec(
                id="deliverable_3",
                title=f"{entity.title()} failure handling and observability",
                objective=f"Handle failed {entity_plural} operations without hiding what happened from an operator.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            DeliverableSpec(
                id="deliverable_4",
                title=f"{entity.title()} production hardening",
                objective=f"Raise the {entity} service to a production-minded bar for reliability, latency, and diagnosability.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay", "scale_slo_overlay"],
            ),
        ]
    if family == ProjectFamily.control_plane_service:
        return [
            DeliverableSpec(
                id="deliverable_1",
                title=f"{entity.title()} control contract",
                objective=f"Define the operator-facing API and configuration model for {entity_plural}.",
                starter_type=StarterType.working_buggy,
            ),
            DeliverableSpec(
                id="deliverable_2",
                title=f"{entity.title()} read path coherence",
                objective=f"Keep {entity_plural} evaluation deterministic and coherent as configuration changes.",
                starter_type=StarterType.partial_implementation,
            ),
            DeliverableSpec(
                id="deliverable_3",
                title=f"{entity.title()} mutations and auditability",
                objective=f"Support safe {entity_plural} updates and make every change traceable.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            DeliverableSpec(
                id="deliverable_4",
                title=f"{entity.title()} production hardening",
                objective=f"Raise the {entity} control plane to a production-minded bar for latency, diagnostics, and operator trust.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay", "scale_slo_overlay"],
            ),
        ]
    if family == ProjectFamily.workflow_agent_service:
        return [
            DeliverableSpec(
                id="deliverable_1",
                title=f"{entity.title()} workflow contract",
                objective=f"Define the bounded workflow surface and the key states for {entity_plural}.",
                starter_type=StarterType.working_buggy,
            ),
            DeliverableSpec(
                id="deliverable_2",
                title=f"{entity.title()} routing and execution",
                objective=f"Implement the core {entity_plural} workflow branches without breaking the published contract.",
                starter_type=StarterType.partial_implementation,
            ),
            DeliverableSpec(
                id="deliverable_3",
                title=f"{entity.title()} fallbacks and traceability",
                objective=f"Keep the {entity} workflow explainable under edge conditions and partial failures.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            DeliverableSpec(
                id="deliverable_4",
                title=f"{entity.title()} production hardening",
                objective=f"Raise the {entity} workflow to a production-minded bar for reliability, latency, and operator trust.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay", "scale_slo_overlay"],
            ),
        ]
    return [
        DeliverableSpec(
            id="deliverable_1",
            title=f"{entity.title()} contract and public surface",
            objective=f"Define a stable public surface for {entity_plural} and the bounded behavior it must return.",
            starter_type=StarterType.working_buggy,
        ),
        DeliverableSpec(
            id="deliverable_2",
            title=f"{entity.title()} core behavior and data flow",
            objective=f"Implement the main {entity_plural} data flow without breaking the published contract.",
            starter_type=StarterType.partial_implementation,
        ),
        DeliverableSpec(
            id="deliverable_3",
            title=f"{entity.title()} observability and recovery",
            objective=f"Keep {entity_plural} failures observable enough to debug and recover without guessing.",
            starter_type=StarterType.working_buggy,
            overlay_ids=["productionization_overlay"],
        ),
        DeliverableSpec(
            id="deliverable_4",
            title=f"{entity.title()} production hardening",
            objective=f"Raise the {entity} service to a production-minded bar for reliability, latency, and diagnostics.",
            starter_type=StarterType.working_suboptimal,
            overlay_ids=["productionization_overlay", "scale_slo_overlay"],
        ),
    ]


def build_task_agent_scaffold(
    *,
    title: str,
    summary: str,
    design_spec: AssignmentDesignSpec,
) -> tuple[TaskAgentServiceSpec, str]:
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
        deliverables=_family_deliverables(design_spec),
    )
    spec = ensure_task_agent_deliverable_briefs(spec, overwrite=True)
    origin_template = design_spec.project_contract.family.value
    return spec, origin_template
