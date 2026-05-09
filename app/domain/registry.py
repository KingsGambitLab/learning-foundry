from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PackageType(str, Enum):
    survey_course = "survey_course"
    progressive_codebase_course = "progressive_codebase_course"


class RiskClass(str, Enum):
    standard = "standard"
    review_required = "review_required"
    high_stakes = "high_stakes"


class CatalogStatus(str, Enum):
    active = "active"
    extended = "extended"
    later = "later"


class StarterType(str, Enum):
    bare_stub = "bare_stub"
    partial_implementation = "partial_implementation"
    working_buggy = "working_buggy"
    working_suboptimal = "working_suboptimal"


class OverlayDefinition(BaseModel):
    id: str
    summary: str
    adds_tests: list[str] = Field(default_factory=list)
    status: CatalogStatus = CatalogStatus.active


class DomainPackDefinition(BaseModel):
    id: str
    risk_class: RiskClass
    summary: str


class DesignCatalog(BaseModel):
    registry_version: str
    package_types: list[PackageType]
    domain_packs: list[DomainPackDefinition]
    overlays: list[OverlayDefinition]

    def domain_pack_by_id(self, domain_pack_id: str) -> DomainPackDefinition | None:
        return next((item for item in self.domain_packs if item.id == domain_pack_id), None)

    def overlay_by_id(self, overlay_id: str) -> OverlayDefinition | None:
        return next((item for item in self.overlays if item.id == overlay_id), None)


DESIGN_CATALOG = DesignCatalog(
    registry_version="0.2",
    package_types=[
        PackageType.survey_course,
        PackageType.progressive_codebase_course,
    ],
    domain_packs=[
        DomainPackDefinition(
            id="oncall_copilot",
            risk_class=RiskClass.standard,
            summary="Incident guidance over runbooks, logs, and operational tools.",
        ),
        DomainPackDefinition(
            id="rfp_drafter",
            risk_class=RiskClass.standard,
            summary="Evidence-grounded proposal drafting over structured inputs and internal docs.",
        ),
        DomainPackDefinition(
            id="analyst_sql",
            risk_class=RiskClass.standard,
            summary="Question-to-SQL analysis with safe execution and result interpretation.",
        ),
        DomainPackDefinition(
            id="qbr_prep",
            risk_class=RiskClass.standard,
            summary="Quarterly business review preparation with source-backed synthesis.",
        ),
        DomainPackDefinition(
            id="investment_memo",
            risk_class=RiskClass.standard,
            summary="Structured memo generation from investment data and evidence.",
        ),
        DomainPackDefinition(
            id="clinical_case_triage",
            risk_class=RiskClass.review_required,
            summary="Clinical-case triage; auto-generation requires manual review before publish.",
        ),
    ],
    overlays=[
        OverlayDefinition(
            id="productionization_overlay",
            summary="Observability, durable state, fallbacks, eval artifacts, and replayability.",
            adds_tests=[
                "observability_presence_test",
                "durable_state_test",
                "eval_artifact_test",
            ],
            status=CatalogStatus.active,
        ),
        OverlayDefinition(
            id="scale_slo_overlay",
            summary="Throughput, latency, error-rate, and cost constraints.",
            adds_tests=[
                "throughput_threshold_test",
                "p95_latency_test",
                "error_rate_threshold_test",
                "cost_threshold_test",
            ],
            status=CatalogStatus.active,
        ),
        OverlayDefinition(
            id="freshness_overlay",
            summary="Incremental indexing, freshness lag, and change-stream handling.",
            adds_tests=[
                "incremental_reindex_test",
                "freshness_lag_test",
            ],
            status=CatalogStatus.active,
        ),
        OverlayDefinition(
            id="adversarial_overlay",
            summary="Prompt injection and adversarial robustness.",
            adds_tests=[
                "prompt_injection_resistance_test",
                "adversary_fuzz_test",
            ],
            status=CatalogStatus.later,
        ),
    ],
)
