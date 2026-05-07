from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import AssignmentDesignSpec, ExecutionSurface, RetrievalMode
from app.services.assignment_design_inference import build_assignment_design


def _service_design(*, package_type: PackageType, domain_pack: str | None = None, overlays: list[str] | None = None) -> AssignmentDesignSpec:
    return build_assignment_design(
        package_type=package_type,
        risk_class=RiskClass.standard,
        domain_pack=domain_pack,
        overlays=overlays or [],
        retrieval_mode=RetrievalMode.none,
        tool_use_required=True,
        traceability_required=True,
        durable_state_required=True,
        approval_flow_required=True,
    )


def _grounded_retrieval_design(*, package_type: PackageType, overlays: list[str] | None = None) -> AssignmentDesignSpec:
    return build_assignment_design(
        package_type=package_type,
        risk_class=RiskClass.standard,
        domain_pack=None,
        overlays=overlays or [],
        retrieval_mode=RetrievalMode.grounded_answers,
        answer_synthesis_required=True,
        citations_required=True,
        abstention_required=True,
        tool_use_required=True,
        traceability_required=True,
        durable_state_required=False,
        approval_flow_required=False,
    )


def _ranked_retrieval_design(*, package_type: PackageType, overlays: list[str] | None = None) -> AssignmentDesignSpec:
    return build_assignment_design(
        package_type=package_type,
        risk_class=RiskClass.standard,
        domain_pack=None,
        overlays=overlays or [],
        retrieval_mode=RetrievalMode.ranked_results,
        tool_use_required=False,
        traceability_required=True,
        durable_state_required=False,
        approval_flow_required=False,
    )


def _stateful_design(*, package_type: PackageType, overlays: list[str] | None = None) -> AssignmentDesignSpec:
    return build_assignment_design(
        package_type=package_type,
        risk_class=RiskClass.standard,
        domain_pack=None,
        overlays=overlays or [],
        retrieval_mode=RetrievalMode.none,
        tool_use_required=False,
        traceability_required=True,
        durable_state_required=True,
        approval_flow_required=False,
    )


def _protocol_design(*, package_type: PackageType) -> AssignmentDesignSpec:
    design = build_assignment_design(
        package_type=package_type,
        risk_class=RiskClass.review_required,
        domain_pack=None,
        overlays=[],
        retrieval_mode=RetrievalMode.none,
        tool_use_required=False,
        traceability_required=True,
        durable_state_required=False,
        approval_flow_required=False,
        execution_surface=ExecutionSurface.protocol_server,
    )
    return design


class CourseModulePattern(BaseModel):
    module_slug: str
    title: str
    design_spec: AssignmentDesignSpec
    domain_pack: str | None = None
    overlays: list[str] = Field(default_factory=list)


class CoursePattern(BaseModel):
    course_slug: str
    course_title: str
    package_type: PackageType
    shared_design_spec: AssignmentDesignSpec | None = None
    modules: list[CourseModulePattern]


CATALOG_PATTERNS = [
    CoursePattern(
        course_slug="tusharbisht-system-design-hands-on",
        course_title="System Design Hands On",
        package_type=PackageType.survey_course,
        modules=[
            CourseModulePattern(
                module_slug="design/01-semantic-search",
                title="Semantic search",
                design_spec=_ranked_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="design/02-rag-qa",
                title="RAG QA",
                design_spec=_grounded_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="design/03-tinyurl",
                title="TinyURL",
                design_spec=_stateful_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="design/04-booking-concurrency",
                title="Booking concurrency",
                design_spec=_stateful_design(
                    package_type=PackageType.progressive_codebase_course,
                    overlays=["scale_slo_overlay"],
                ),
                overlays=["scale_slo_overlay"],
            ),
            CourseModulePattern(
                module_slug="design/05-langchain-agent",
                title="LangChain-style agent",
                design_spec=_service_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="design/06-vector-database",
                title="Vector database",
                design_spec=_ranked_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="design/07-mcp-server",
                title="MCP server",
                design_spec=_protocol_design(package_type=PackageType.progressive_codebase_course),
            ),
        ],
    ),
    CoursePattern(
        course_slug="tusharbisht-forward-deployed-engineering",
        course_title="Forward Deployed Engineering",
        package_type=PackageType.survey_course,
        shared_design_spec=_service_design(package_type=PackageType.progressive_codebase_course),
        modules=[
            CourseModulePattern(
                module_slug="design/01-support-triage",
                title="Support triage",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="support_triage",
                ),
                domain_pack="support_triage",
            ),
            CourseModulePattern(
                module_slug="design/02-oncall-copilot",
                title="Oncall copilot",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="oncall_copilot",
                ),
                domain_pack="oncall_copilot",
            ),
            CourseModulePattern(
                module_slug="design/03-se-rfp-drafter",
                title="SE RFP drafter",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="rfp_drafter",
                ),
                domain_pack="rfp_drafter",
            ),
            CourseModulePattern(
                module_slug="design/04-analyst-sql",
                title="Analyst SQL",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="analyst_sql",
                ),
                domain_pack="analyst_sql",
            ),
            CourseModulePattern(
                module_slug="design/05-cs-qbr-prep",
                title="CS QBR prep",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="qbr_prep",
                ),
                domain_pack="qbr_prep",
            ),
            CourseModulePattern(
                module_slug="design/06-vc-investment-memo",
                title="VC investment memo",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="investment_memo",
                ),
                domain_pack="investment_memo",
            ),
            CourseModulePattern(
                module_slug="design/07-clinical-case-triage",
                title="Clinical case triage",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="clinical_case_triage",
                ),
                domain_pack="clinical_case_triage",
            ),
        ],
    ),
    CoursePattern(
        course_slug="tusharbisht-cs-demo-agent-to-production",
        course_title="Customer Support Agent — Demo to Production",
        package_type=PackageType.progressive_codebase_course,
        shared_design_spec=_service_design(
            package_type=PackageType.progressive_codebase_course,
            domain_pack="support_triage",
        ),
        modules=[
            CourseModulePattern(
                module_slug="exercise/01-observability",
                title="Observability",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="support_triage",
                    overlays=["productionization_overlay"],
                ),
                domain_pack="support_triage",
                overlays=["productionization_overlay"],
            ),
            CourseModulePattern(
                module_slug="exercise/02-state-and-fallback",
                title="Durable state and fallback chains",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="support_triage",
                    overlays=["productionization_overlay"],
                ),
                domain_pack="support_triage",
                overlays=["productionization_overlay"],
            ),
            CourseModulePattern(
                module_slug="exercise/03-confidence-calibration",
                title="Confidence calibration",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="support_triage",
                    overlays=["productionization_overlay"],
                ),
                domain_pack="support_triage",
                overlays=["productionization_overlay"],
            ),
            CourseModulePattern(
                module_slug="exercise/04-feedback-loops",
                title="Feedback loops and prompt-as-artifact",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="support_triage",
                    overlays=["productionization_overlay"],
                ),
                domain_pack="support_triage",
                overlays=["productionization_overlay"],
            ),
            CourseModulePattern(
                module_slug="final/integrated",
                title="Integrated production-grade agent",
                design_spec=_service_design(
                    package_type=PackageType.progressive_codebase_course,
                    domain_pack="support_triage",
                    overlays=["productionization_overlay", "scale_slo_overlay"],
                ),
                domain_pack="support_triage",
                overlays=["productionization_overlay", "scale_slo_overlay"],
            ),
        ],
    ),
    CoursePattern(
        course_slug="tusharbisht-rag-on-wikipedia",
        course_title="Rag On Wikipedia",
        package_type=PackageType.progressive_codebase_course,
        shared_design_spec=_grounded_retrieval_design(
            package_type=PackageType.progressive_codebase_course,
        ),
        modules=[
            CourseModulePattern(
                module_slug="exercise/01-structure-aware-chunking",
                title="Structure-aware chunking",
                design_spec=_grounded_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="exercise/02-hybrid-retrieval",
                title="Hybrid retrieval",
                design_spec=_grounded_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="exercise/03-query-decomposition",
                title="Query rewriting and decomposition",
                design_spec=_grounded_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="exercise/04-reranking",
                title="Cross-encoder re-ranking",
                design_spec=_grounded_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="exercise/05-citation-faithfulness",
                title="Citation faithfulness and confidence calibration",
                design_spec=_grounded_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="exercise/06-disambiguation",
                title="Disambiguation and entity resolution",
                design_spec=_grounded_retrieval_design(package_type=PackageType.progressive_codebase_course),
            ),
            CourseModulePattern(
                module_slug="exercise/07-vector-scale",
                title="Vector index at production scale",
                design_spec=_grounded_retrieval_design(
                    package_type=PackageType.progressive_codebase_course,
                    overlays=["scale_slo_overlay"],
                ),
                overlays=["scale_slo_overlay"],
            ),
            CourseModulePattern(
                module_slug="exercise/08-freshness",
                title="Index freshness via change stream",
                design_spec=_grounded_retrieval_design(
                    package_type=PackageType.progressive_codebase_course,
                    overlays=["freshness_overlay"],
                ),
                overlays=["freshness_overlay"],
            ),
            CourseModulePattern(
                module_slug="exercise/09-adversarial",
                title="Prompt-injection robustness",
                design_spec=_grounded_retrieval_design(
                    package_type=PackageType.progressive_codebase_course,
                    overlays=["adversarial_overlay"],
                ),
                overlays=["adversarial_overlay"],
            ),
            CourseModulePattern(
                module_slug="exercise/10-cost-slo",
                title="Cost optimization to SLO",
                design_spec=_grounded_retrieval_design(
                    package_type=PackageType.progressive_codebase_course,
                    overlays=["scale_slo_overlay"],
                ),
                overlays=["scale_slo_overlay"],
            ),
            CourseModulePattern(
                module_slug="exercise/11-eval-driven",
                title="Eval-driven iteration",
                design_spec=_grounded_retrieval_design(
                    package_type=PackageType.progressive_codebase_course,
                    overlays=["productionization_overlay"],
                ),
                overlays=["productionization_overlay"],
            ),
            CourseModulePattern(
                module_slug="final/integrated",
                title="Production deploy at SLO",
                design_spec=_grounded_retrieval_design(
                    package_type=PackageType.progressive_codebase_course,
                    overlays=["scale_slo_overlay", "freshness_overlay"],
                ),
                overlays=["scale_slo_overlay", "freshness_overlay"],
            ),
        ],
    ),
]


def course_pattern_by_slug(course_slug: str) -> CoursePattern | None:
    return next((pattern for pattern in CATALOG_PATTERNS if pattern.course_slug == course_slug), None)
