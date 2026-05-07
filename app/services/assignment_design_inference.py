from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import (
    AssignmentDesignSpec,
    AssessmentStrategySpec,
    CapabilitySpec,
    CourseStructureSpec,
    ExecutionSurface,
    ProgressionMode,
    RetrievalMode,
    RuntimeDependencySpec,
    WorkspaceScope,
)
from app.domain.registry import StarterType


class DesignSupportStatus(str, Enum):
    supported = "supported"
    manual_review = "manual_review"
    unsupported = "unsupported"


class GenerationIntake(BaseModel):
    title: str
    problem_statement: str
    learning_outcomes: list[str] = Field(default_factory=list)
    package_type_hint: PackageType | None = None


class AssignmentDesignInference(BaseModel):
    design_spec: AssignmentDesignSpec | None = None
    package_type: PackageType
    status: DesignSupportStatus
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


OVERLAY_KEYWORDS: dict[str, list[str]] = {
    "productionization_overlay": [
        "production",
        "observability",
        "state",
        "fallback",
        "approval",
        "trace",
        "eval",
        "resume",
        "durable",
    ],
    "scale_slo_overlay": [
        "latency",
        "throughput",
        "scale",
        "slo",
        "cost",
        "p95",
        "error rate",
    ],
    "freshness_overlay": [
        "freshness",
        "change stream",
        "reindex",
        "stale",
    ],
    "adversarial_overlay": [
        "prompt injection",
        "adversarial",
        "malicious",
        "robustness",
    ],
}

DOMAIN_PACK_KEYWORDS: dict[str, list[str]] = {
    "support_triage": ["support", "ticket", "customer", "reply", "triage"],
    "oncall_copilot": ["incident", "oncall", "runbook", "alert"],
    "rfp_drafter": ["rfp", "proposal", "sales engineering"],
    "analyst_sql": ["sql", "query", "analysis", "dashboard"],
    "qbr_prep": ["qbr", "business review", "account review"],
    "investment_memo": ["investment", "memo", "venture", "vc"],
    "clinical_case_triage": ["clinical", "patient", "diagnosis", "medical"],
}

REVIEW_REQUIRED_KEYWORDS = {"clinical", "patient", "medical", "diagnosis"}
HIGH_STAKES_KEYWORDS = {"legal", "prescription", "financial advice"}
UNSUPPORTED_KEYWORDS = {
    "frontend",
    "mobile app",
    "ios",
    "android",
    "react native",
    "swiftui",
    "browser extension",
    "chrome extension",
}
PROTOCOL_KEYWORDS = {"mcp", "protocol server", "handshake", "capability discovery"}
GROUNDED_RETRIEVAL_KEYWORDS = {
    "rag",
    "citation",
    "citations",
    "grounded",
    "grounded answer",
    "faithful",
    "hallucination",
    "knowledge base",
    "answer from documents",
}
RANKED_RETRIEVAL_KEYWORDS = {
    "semantic search",
    "search",
    "retrieval",
    "vector",
    "ranking",
    "nearest neighbor",
    "metadata filter",
}
STATEFUL_KEYWORDS = {
    "booking",
    "reservation",
    "inventory",
    "wallet",
    "payment",
    "idempotent",
    "idempotency",
    "concurrency",
    "mutable state",
}
TOOL_USE_KEYWORDS = {
    "agent",
    "tool",
    "workflow",
    "triage",
    "copilot",
    "draft",
    "reply",
    "trace",
    "approval",
    "handoff",
    "support",
    "sql",
}


def infer_package_type(*, text: str, package_type_hint: PackageType | None) -> PackageType:
    if package_type_hint is not None:
        return package_type_hint
    if any(
        phrase in text
        for phrase in [
            "demo to production",
            "inherited demo",
            "progressive",
            "production ready",
            "production-ready",
        ]
    ):
        return PackageType.progressive_codebase_course
    if any(phrase in text for phrase in ["course", "catalog", "survey", "multiple assignments"]):
        return PackageType.survey_course
    return PackageType.progressive_codebase_course


def infer_risk_class(text: str) -> RiskClass:
    if any(keyword in text for keyword in HIGH_STAKES_KEYWORDS):
        return RiskClass.high_stakes
    if any(keyword in text for keyword in REVIEW_REQUIRED_KEYWORDS):
        return RiskClass.review_required
    return RiskClass.standard


def infer_overlays(text: str) -> list[str]:
    overlays: list[str] = []
    for overlay_id, keywords in OVERLAY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            overlays.append(overlay_id)
    return overlays


def infer_domain_pack(text: str) -> str | None:
    scores: list[tuple[int, str]] = []
    for domain_pack_id, keywords in DOMAIN_PACK_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            scores.append((score, domain_pack_id))
    if not scores:
        return None
    scores.sort(reverse=True)
    return scores[0][1]


def build_assignment_design(
    *,
    package_type: PackageType,
    risk_class: RiskClass,
    domain_pack: str | None,
    overlays: list[str],
    retrieval_mode: RetrievalMode = RetrievalMode.none,
    answer_synthesis_required: bool = False,
    citations_required: bool = False,
    abstention_required: bool = False,
    tool_use_required: bool = False,
    traceability_required: bool = True,
    durable_state_required: bool = False,
    approval_flow_required: bool = False,
    execution_surface: ExecutionSurface = ExecutionSurface.http_service,
    starter_type: StarterType = StarterType.partial_implementation,
    primary_database: str | None = None,
    cache_backend: str | None = None,
    tech_stack: list[str] | None = None,
) -> AssignmentDesignSpec:
    visible_fixture_files = ["data/corpus.json"] if retrieval_mode != RetrievalMode.none else []
    shared_codebase = package_type == PackageType.progressive_codebase_course
    return AssignmentDesignSpec(
        course_structure=CourseStructureSpec(
            package_type=package_type,
            workspace_scope=(
                WorkspaceScope.shared_course_workspace
                if shared_codebase
                else WorkspaceScope.per_module_workspace
            ),
            progression_mode=(
                ProgressionMode.cumulative_module_gates
                if shared_codebase
                else ProgressionMode.independent_modules
            ),
            shared_codebase=shared_codebase,
        ),
        runtime_dependencies=RuntimeDependencySpec(
            execution_surface=execution_surface,
            starter_type=starter_type,
            editable_files=["app.py"],
            visible_fixture_files=visible_fixture_files,
            primary_database=primary_database,
            cache_backend=cache_backend,
            tech_stack=list(tech_stack or []),
            local_run_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
            visible_check_command="python checks/run_visible_checks.py",
            preview_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
        ),
        capabilities=CapabilitySpec(
            retrieval_mode=retrieval_mode,
            answer_synthesis_required=answer_synthesis_required,
            citations_required=citations_required,
            abstention_required=abstention_required,
            tool_use_required=tool_use_required,
            traceability_required=traceability_required,
            durable_state_required=durable_state_required,
            approval_flow_required=approval_flow_required,
        ),
        assessment_strategy=AssessmentStrategySpec(
            public_checks_required=True,
            hidden_grader_required=True,
            cumulative_module_gates=shared_codebase,
            learner_submission_enabled=True,
        ),
        risk_class=risk_class,
        domain_pack=domain_pack,
        overlays=list(overlays),
    )


def infer_assignment_design(
    *,
    title: str,
    problem_statement: str,
    learning_outcomes: list[str],
    package_type_hint: PackageType | None = None,
) -> AssignmentDesignInference:
    text = " ".join([title, problem_statement, *learning_outcomes]).lower()
    package_type = infer_package_type(text=text, package_type_hint=package_type_hint)
    risk_class = infer_risk_class(text)
    overlays = infer_overlays(text)
    domain_pack = infer_domain_pack(text)

    reasons: list[str] = []
    warnings: list[str] = []

    if any(keyword in text for keyword in UNSUPPORTED_KEYWORDS):
        return AssignmentDesignInference(
            design_spec=None,
            package_type=package_type,
            status=DesignSupportStatus.unsupported,
            reasons=["The brief emphasizes a learner-facing UI surface that the current backend-first generator does not support."],
            warnings=["This platform currently generates backend and service assignments, not UI-first implementation projects."],
        )

    if any(keyword in text for keyword in PROTOCOL_KEYWORDS):
        return AssignmentDesignInference(
            design_spec=None,
            package_type=package_type,
            status=DesignSupportStatus.unsupported,
            reasons=["The brief depends on a protocol-specific server surface that is outside the learner-ready generator today."],
            warnings=["Protocol-oriented assignments should stay blocked until the generator can scaffold and grade them directly."],
        )

    if any(keyword in text for keyword in GROUNDED_RETRIEVAL_KEYWORDS):
        reasons.append("The brief asks for grounded answering over a visible corpus with evidence-aware behavior.")
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            retrieval_mode=RetrievalMode.grounded_answers,
            answer_synthesis_required=True,
            citations_required=True,
            abstention_required=True,
            tool_use_required=True,
            traceability_required=True,
            durable_state_required=False,
            approval_flow_required=False,
        )
    elif any(keyword in text for keyword in RANKED_RETRIEVAL_KEYWORDS):
        reasons.append("The brief centers on retrieval quality over a visible corpus.")
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            retrieval_mode=RetrievalMode.ranked_results,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=False,
            traceability_required=True,
            durable_state_required=False,
            approval_flow_required=False,
        )
    elif any(keyword in text for keyword in STATEFUL_KEYWORDS):
        reasons.append("The brief depends on correctness under persistent mutable state and concurrency.")
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            retrieval_mode=RetrievalMode.none,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=False,
            traceability_required=True,
            durable_state_required=True,
            approval_flow_required=False,
        )
    else:
        reasons.append("The brief fits the general learner-ready service pipeline with bounded workflows and observable behavior.")
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            retrieval_mode=RetrievalMode.none,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=bool(domain_pack or any(keyword in text for keyword in TOOL_USE_KEYWORDS)),
            traceability_required=True,
            durable_state_required="state" in text or "resume" in text or "durable" in text,
            approval_flow_required="approval" in text or "escalat" in text or "handoff" in text,
        )

    status = DesignSupportStatus.supported
    if risk_class != RiskClass.standard:
        status = DesignSupportStatus.manual_review
        warnings.append("The brief includes review-required or high-stakes language.")
    if "adversarial_overlay" in overlays:
        warnings.append("Adversarial robustness is still a stretch goal and may need tighter human review.")

    return AssignmentDesignInference(
        design_spec=design_spec,
        package_type=package_type,
        status=status,
        reasons=reasons,
        warnings=warnings,
    )
