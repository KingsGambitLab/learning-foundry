from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.registry import DESIGN_CATALOG, PackageType, RiskClass
from app.domain.task_agent import DeliverableGate, TaskAgentServiceSpec
from app.services.public_surface_quality import (
    deliverable_title_lacks_domain_grounding,
    endpoint_uses_archetype_words,
    endpoint_uses_title_slug,
    starter_surface_markers,
)
from app.services.task_agent_contract_surface import is_placeholder_public_surface


class ValidationLevel(str, Enum):
    error = "error"
    warning = "warning"


class ValidationIssue(BaseModel):
    level: ValidationLevel
    code: str
    location: str
    message: str


class DeliverableGateSummary(BaseModel):
    deliverable_id: str
    active_public_check_ids: list[str]
    active_test_count: int


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    deliverable_gates: list[DeliverableGateSummary] = Field(default_factory=list)


def _gate_summaries(spec: TaskAgentServiceSpec) -> list[DeliverableGateSummary]:
    summaries: list[DeliverableGateSummary] = []
    for deliverable in spec.deliverables:
        gate = spec.gate_for(deliverable.id)
        summaries.append(
            DeliverableGateSummary(
                deliverable_id=gate.deliverable_id,
                active_public_check_ids=gate.active_public_check_ids,
                active_test_count=len(gate.active_test_ids),
            )
        )
    return summaries


def validate_task_agent_spec(spec: TaskAgentServiceSpec) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    overlay_ids = {overlay.id for overlay in DESIGN_CATALOG.overlays}
    endpoint_paths = {endpoint.path for endpoint in spec.public_endpoints}
    published_endpoints = {(endpoint.method, endpoint.path) for endpoint in spec.public_endpoints}

    def add_issue(level: ValidationLevel, code: str, location: str, message: str) -> None:
        issue = ValidationIssue(level=level, code=code, location=location, message=message)
        if level == ValidationLevel.error:
            errors.append(issue)
        else:
            warnings.append(issue)

    if spec.domain_pack:
        domain_pack = DESIGN_CATALOG.domain_pack_by_id(spec.domain_pack)
        if domain_pack is None:
            add_issue(
                ValidationLevel.error,
                "unknown_domain_pack",
                "domain_pack",
                f"Unknown domain pack '{spec.domain_pack}'.",
            )
        elif domain_pack.risk_class != spec.risk_class:
            add_issue(
                ValidationLevel.warning,
                "domain_pack_risk_mismatch",
                "risk_class",
                f"Domain pack '{spec.domain_pack}' is marked '{domain_pack.risk_class.value}'.",
            )

    if spec.assessment_strategy.public_checks_required and spec.runtime_dependencies.visible_check_command is None:
        add_issue(
            ValidationLevel.error,
            "public_checks_without_command",
            "assessment_strategy.public_checks_required",
            "Public checks are required, but the runtime dependency spec has no visible check command.",
        )
    if spec.course_structure.workspace_scope.value == "per_deliverable_workspace" and spec.course_structure.shared_codebase:
        add_issue(
            ValidationLevel.error,
            "workspace_scope_conflict",
            "course_structure.workspace_scope",
            "Per-deliverable workspaces cannot be paired with a shared codebase course structure.",
        )
    if is_placeholder_public_surface(spec.public_endpoints):
        add_issue(
            ValidationLevel.error,
            "missing_public_endpoints",
            "public_endpoints",
            "Publish at least one non-health public endpoint before review.",
        )
    if "/health" not in endpoint_paths:
        add_issue(
            ValidationLevel.error,
            "missing_health_endpoint",
            "public_endpoints",
            "Every generated service must expose `GET /health`.",
        )
    for index, endpoint in enumerate(spec.public_endpoints):
        if endpoint.path == "/health":
            continue
        if endpoint_uses_title_slug(endpoint.path, title=spec.title):
            add_issue(
                ValidationLevel.error,
                "title_slug_public_endpoint",
                f"public_endpoints[{index}].path",
                "Public endpoints should use concrete resource nouns, not the full course title as a URL slug.",
            )
        elif endpoint_uses_archetype_words(endpoint.path):
            add_issue(
                ValidationLevel.error,
                "generic_public_endpoint",
                f"public_endpoints[{index}].path",
                "Public endpoints should expose a concrete resource surface instead of archetype words like service or API.",
            )
    if spec.package_type == PackageType.survey_course and len(spec.deliverables) > 3:
        add_issue(
            ValidationLevel.warning,
            "survey_many_deliverables",
            "deliverables",
            "This looks more like a progressive codebase than a survey assignment.",
        )
    if spec.risk_class in {RiskClass.review_required, RiskClass.high_stakes}:
        add_issue(
            ValidationLevel.warning,
            "manual_review_required",
            "risk_class",
            "This spec should require human review before publish.",
        )

    for deliverable in spec.deliverables:
        if deliverable_title_lacks_domain_grounding(
            deliverable.title,
            entities=spec.project_contract.core_entities,
        ):
            add_issue(
                ValidationLevel.error,
                "generic_deliverable_title",
                f"deliverables.{deliverable.id}.title",
                "Deliverable titles should stay grounded in the concrete domain instead of generic scaffolding language.",
            )
        for overlay_id in deliverable.overlay_ids:
            if overlay_id not in overlay_ids:
                add_issue(
                    ValidationLevel.error,
                    "unknown_overlay",
                    f"deliverables.{deliverable.id}.overlay_ids",
                    f"Unknown overlay '{overlay_id}'.",
                )
        if not deliverable.learning_outcomes:
            add_issue(
                ValidationLevel.error,
                "missing_deliverable_learning_outcomes",
                f"deliverables.{deliverable.id}.learning_outcomes",
                "Each deliverable should publish concrete learning outcomes derived from the learner task.",
            )
        if deliverable.learner_starter_surface is None:
            add_issue(
                ValidationLevel.error,
                "missing_learner_starter_surface",
                f"deliverables.{deliverable.id}.learner_starter_surface",
                "Authoring must describe the real learner-owned files, endpoints, and scenarios.",
            )
        else:
            starter_surface = deliverable.learner_starter_surface
            if not starter_surface.primary_editable_paths:
                add_issue(
                    ValidationLevel.error,
                    "missing_primary_editable_paths",
                    f"deliverables.{deliverable.id}.learner_starter_surface.primary_editable_paths",
                    "Learners need a clear primary implementation surface.",
                )
            if not starter_surface.required_endpoints:
                add_issue(
                    ValidationLevel.error,
                    "missing_required_endpoints",
                    f"deliverables.{deliverable.id}.learner_starter_surface.required_endpoints",
                    "Starter surfaces should name the published endpoints they are expected to preserve.",
                )
            for index, endpoint in enumerate(starter_surface.required_endpoints):
                if (endpoint.method, endpoint.path) not in published_endpoints:
                    add_issue(
                        ValidationLevel.error,
                        "starter_required_endpoint_not_published",
                        (
                            f"deliverables.{deliverable.id}.learner_starter_surface."
                            f"required_endpoints[{index}]"
                        ),
                        "Starter surfaces must only reference endpoints from the published public surface.",
                    )
            for index, scenario in enumerate(starter_surface.domain_scenarios):
                text = " ".join([scenario.title, scenario.request_summary, scenario.expected_behavior]).lower()
                if any(marker in text for marker in ["routine case", "ambiguous or risky case", "placeholder", *starter_surface_markers()]):
                    add_issue(
                        ValidationLevel.error,
                        "placeholder_starter_scenario",
                        f"deliverables.{deliverable.id}.learner_starter_surface.domain_scenarios[{index}]",
                        "Replace placeholder starter scenarios with real domain cases.",
                    )
        if deliverable.learner_brief is None:
            add_issue(
                ValidationLevel.error,
                "missing_learner_brief",
                f"deliverables.{deliverable.id}.learner_brief",
                "Each deliverable must include a learner brief.",
            )
        elif not deliverable.learner_brief.files_to_edit or not deliverable.learner_brief.definition_of_done:
            add_issue(
                ValidationLevel.error,
                "underspecified_learner_brief",
                f"deliverables.{deliverable.id}.learner_brief",
                "Learner briefs must call out files to edit and a definition of done.",
            )
        if not deliverable.public_checks:
            add_issue(
                ValidationLevel.error,
                "missing_public_checks",
                f"deliverables.{deliverable.id}.public_checks",
                "Each deliverable should expose at least one visible learner check.",
            )
        for index, public_check in enumerate(deliverable.public_checks):
            location = f"deliverables.{deliverable.id}.public_checks[{index}]"
            if public_check.request_path not in endpoint_paths:
                add_issue(
                    ValidationLevel.error,
                    "public_check_path_not_published",
                    f"{location}.request_path",
                    "Every public check must target one of the published endpoints.",
                )
            if not public_check.learner_goal.strip():
                add_issue(
                    ValidationLevel.error,
                    "blank_public_check_goal",
                    f"{location}.learner_goal",
                    "Public checks must explain why the check matters to the learner.",
                )

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        deliverable_gates=_gate_summaries(spec),
    )


def compute_task_agent_gate(spec: TaskAgentServiceSpec, deliverable_id: str) -> DeliverableGate:
    return spec.gate_for(deliverable_id)
