from __future__ import annotations

from enum import Enum
import re

from pydantic import BaseModel, Field

from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import (
    AssignmentDesignSpec,
    AssessmentStrategySpec,
    CapabilitySpec,
    CourseStructureSpec,
    DataSourceKind,
    DataSourceSpec,
    default_project_contract,
    ExecutionSurface,
    ProjectContractSpec,
    ProjectFamily,
    ProjectRuntimeBindingSpec,
    ProjectRuntimeCommandSpec,
    ProjectRuntimePlanSpec,
    ProjectRuntimeServiceSpec,
    ProjectServiceBinding,
    ProgressionMode,
    RetrievalMode,
    RuntimeDependencySpec,
    WorkspaceScope,
)
from app.domain.registry import StarterType
from app.services.public_surface_quality import extract_project_entities, pluralize_phrase
from app.services.task_agent_starter_templates import (
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
)


class DesignSupportStatus(str, Enum):
    supported = "supported"
    manual_review = "manual_review"
    unsupported = "unsupported"


class GenerationIntake(BaseModel):
    title: str
    problem_statement: str
    learning_outcomes: list[str] = Field(default_factory=list)
    package_type_hint: PackageType | None = None
    starter_type: StarterType | None = None
    implementation_language: str | None = None
    language_version: str | None = None
    application_framework: str | None = None
    framework_version: str | None = None
    package_manager: str | None = None
    primary_database: str | None = None
    primary_database_version: str | None = None
    cache_backend: str | None = None
    cache_backend_version: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    data_sources: list[DataSourceSpec] = Field(default_factory=list)


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
    "oncall_copilot": ["incident", "oncall", "runbook", "alert"],
    "rfp_drafter": ["rfp", "proposal", "sales engineering"],
    "analyst_sql": ["sql", "query", "analysis", "dashboard"],
    "qbr_prep": ["qbr", "business review", "account review"],
    "investment_memo": ["investment", "memo", "venture", "vc"],
    "clinical_case_triage": ["clinical", "patient", "diagnosis", "medical"],
    "customer_support_agent": [
        "customer support",
        "support bot",
        "support ticket",
        "ticket",
        "refund",
        "billing",
        "outage",
        "account access",
        "suspicious login",
        "customer message",
    ],
}

REVIEW_REQUIRED_KEYWORDS = {"clinical", "patient", "medical", "diagnosis"}
HIGH_STAKES_KEYWORDS = {"legal", "prescription", "financial advice"}
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
CONTROL_PLANE_KEYWORDS = {
    "feature flag",
    "feature flags",
    "gradual rollout",
    "rollout",
    "targeting",
    "targeted",
    "kill switch",
    "environment override",
    "audit log",
    "audit trail",
    "config update",
    "configuration update",
    "flag evaluation",
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
    "copilot",
    "draft",
    "trace",
    "approval",
    "handoff",
    "sql",
}

FRAMEWORK_LANGUAGE_HINTS: dict[str, str] = {
    # Python
    "fastapi": "python",
    "flask": "python",
    "django": "python",
    # TypeScript / Node
    "express": "typescript",
    "hono": "typescript",
    "nestjs": "typescript",
    # Go
    "gin": "go",
    "fiber": "go",
    # Rust
    "actix": "rust",
    "actix-web": "rust",
    "axum": "rust",
    # Ruby — `framework` is the framework NAME ("rails"); the version
    # is captured separately in `framework_version`, so don't add
    # version-tagged keys like "rails 8" (they'd shadow "rails" on
    # length-descending sort).
    "ruby on rails": "ruby",
    "rails": "ruby",
    "sinatra": "ruby",
    "hanami": "ruby",
    # Java / Kotlin
    "spring boot": "java",
    "spring": "java",
    "quarkus": "java",
    "micronaut": "java",
    "ktor": "kotlin",
    # Elixir
    "phoenix": "elixir",
    # .NET
    "aspnet": "csharp",
    "asp.net": "csharp",
    "asp.net core": "csharp",
    # PHP
    "laravel": "php",
    "symfony": "php",
}

DEFAULT_FRAMEWORK_BY_LANGUAGE: dict[str, str] = {
    "python": "fastapi",
    "typescript": "express",
    "javascript": "express",
    "go": "gin",
    "rust": "actix-web",
    "ruby": "rails",
    "java": "spring boot",
    "kotlin": "ktor",
    "elixir": "phoenix",
    "csharp": "aspnet",
    "php": "laravel",
}

LANGUAGE_KEYWORDS: dict[str, list[str]] = {
    "python": ["python", "fastapi", "flask", "django"],
    "typescript": ["typescript", "ts", "nestjs", "hono"],
    "javascript": ["javascript", "node", "node.js", "express"],
    "go": ["go", "golang", "gin", "fiber"],
    "rust": ["rust", "actix", "axum"],
    "ruby": ["ruby", "rails", "ruby on rails", "sinatra", "hanami", "gemfile"],
    "java": ["java", "spring boot", "spring", "quarkus", "micronaut", "maven", "gradle"],
    "kotlin": ["kotlin", "ktor"],
    "elixir": ["elixir", "phoenix", "liveview"],
    "csharp": ["c#", "csharp", "aspnet", "asp.net", "dotnet", ".net"],
    "php": ["php", "laravel", "symfony"],
}


def _tokenize_phrase(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in value.lower():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _contains_keyword(text: str, keyword: str) -> bool:
    keyword_tokens = _tokenize_phrase(keyword)
    if not keyword_tokens:
        return False
    text_tokens = _tokenize_phrase(text)
    span = len(keyword_tokens)
    if span > len(text_tokens):
        return False
    for start in range(len(text_tokens) - span + 1):
        if text_tokens[start : start + span] == keyword_tokens:
            return True
    return False


def _contains_any_keyword(text: str, keywords: list[str] | set[str]) -> bool:
    return any(_contains_keyword(text, keyword) for keyword in keywords)


def _keyword_match_count(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if _contains_keyword(text, keyword))


def build_project_contract(
    *,
    family: ProjectFamily,
    title: str,
    problem_statement: str,
    implementation_language: str | None = None,
    language_version: str | None = None,
    application_framework: str | None = None,
    framework_version: str | None = None,
    package_manager: str | None = None,
    primary_database: str | None = None,
    primary_database_version: str | None = None,
    cache_backend: str | None = None,
    cache_backend_version: str | None = None,
    tech_stack: list[str] | None = None,
    data_sources: list[DataSourceSpec] | None = None,
) -> ProjectContractSpec:
    text = " ".join([title, problem_statement]).lower()
    source_specs = list(data_sources or [])
    source_titles = [source.title for source in source_specs]
    inferred_entities = extract_project_entities(title, problem_statement)
    primary_entity = inferred_entities[0] if inferred_entities else None
    runtime_binding = build_project_runtime_binding(
        family=family,
        implementation_language=implementation_language,
        application_framework=application_framework,
        primary_database=primary_database,
        cache_backend=cache_backend,
        tech_stack=tech_stack or [],
        data_sources=source_specs,
    )
    runtime_plan = build_project_runtime_plan(
        family=family,
        implementation_language=implementation_language,
        language_version=language_version,
        application_framework=application_framework,
        framework_version=framework_version,
        package_manager=package_manager,
        primary_database=primary_database,
        primary_database_version=primary_database_version,
        cache_backend=cache_backend,
        cache_backend_version=cache_backend_version,
        tech_stack=tech_stack or [],
        data_sources=source_specs,
    )

    if family == ProjectFamily.grounded_retrieval_service:
        return ProjectContractSpec(
            family=family,
            system_kind=(
                f"{primary_entity.title()} retrieval service"
                if primary_entity
                else "Grounded retrieval and answer service"
            ),
            core_entities=(
                [*inferred_entities, *source_titles]
                if inferred_entities
                else ["retrieval corpus", *source_titles] if source_titles else ["retrieval corpus", "grounded response"]
            ),
            primary_read_paths=[
                "retrieve supporting passages for a query",
                "compose a grounded answer with citations",
            ],
            primary_write_paths=[],
            invariants=[
                "Answers stay grounded in the learner-visible corpus.",
                "Unsupported questions abstain instead of guessing.",
                "Citations reference the evidence that justified the answer.",
            ],
            operational_concerns=["retrieval quality", "citation fidelity", "latency under repeated queries"],
            runtime_binding=runtime_binding,
            runtime_plan=runtime_plan,
        )
    if family == ProjectFamily.ranked_retrieval_service:
        return ProjectContractSpec(
            family=family,
            system_kind=(
                f"{primary_entity.title()} search service"
                if primary_entity
                else "Ranked retrieval service"
            ),
            core_entities=(
                [*inferred_entities, *source_titles]
                if inferred_entities
                else ["retrieval corpus", *source_titles] if source_titles else ["retrieval corpus", "search result"]
            ),
            primary_read_paths=["retrieve and rank relevant results for a query"],
            primary_write_paths=[],
            invariants=[
                "Results are ranked consistently for equivalent queries.",
                "Filters do not leak results outside the requested scope.",
            ],
            operational_concerns=["ranking quality", "metadata filtering", "read-path latency"],
            runtime_binding=runtime_binding,
            runtime_plan=runtime_plan,
        )
    if family == ProjectFamily.control_plane_service:
        entity = primary_entity or "control definition"
        entity_plural = pluralize_phrase(entity)
        return ProjectContractSpec(
            family=family,
            system_kind=f"{entity.title()} control plane",
            core_entities=inferred_entities or ["control definitions", "decision rules", "request context", "audit events"],
            primary_read_paths=[
                f"evaluate the active {entity} for a request context",
                f"serve low-latency {entity_plural} decisions for live traffic",
            ],
            primary_write_paths=[
                f"create or update {entity_plural} safely",
                f"publish {entity_plural} changes with traceable state transitions",
            ],
            invariants=[
                "Decisions are deterministic for the same context and active definition.",
                "Live reads stay coherent when control definitions change.",
                "Every mutation is auditable and attributable.",
            ],
            operational_concerns=["read-path coherence", "safe control updates", "operator-visible audit trails"],
            runtime_binding=runtime_binding,
            runtime_plan=runtime_plan,
        )
    if family == ProjectFamily.transactional_stateful_service:
        entity = primary_entity or "record"
        entity_plural = pluralize_phrase(entity)
        return ProjectContractSpec(
            family=family,
            system_kind=f"{entity.title()} service",
            core_entities=inferred_entities or ["durable records", "mutable workflow state"],
            primary_read_paths=[f"serve the current {entity} state safely under load"],
            primary_write_paths=[f"create or update {entity_plural} without violating invariants"],
            invariants=[
                "Concurrent or repeated writes do not corrupt critical state.",
                "State transitions preserve the service's core business invariants.",
            ],
            operational_concerns=["concurrency safety", "idempotency", "failure recovery"],
            runtime_binding=runtime_binding,
            runtime_plan=runtime_plan,
        )
    if family == ProjectFamily.workflow_agent_service:
        entity = primary_entity or "workflow request"
        entity_plural = pluralize_phrase(entity)
        return ProjectContractSpec(
            family=family,
            system_kind=f"{entity.title()} workflow service",
            core_entities=inferred_entities or ["requests", "tool runs", "operator decisions", "run traces"],
            primary_read_paths=[f"inspect {entity_plural} and route work through bounded workflows"],
            primary_write_paths=[f"progress {entity_plural} without breaking the published contract"],
            invariants=[
                "The service preserves a stable response contract.",
                "Operator-visible traces explain why the workflow took each step.",
            ],
            operational_concerns=["tool routing", "fallbacks", "traceability"],
            runtime_binding=runtime_binding,
            runtime_plan=runtime_plan,
        )
    entity = primary_entity or "service request"
    return ProjectContractSpec(
        family=ProjectFamily.generic_backend_service,
        system_kind=f"{entity.title()} service",
        core_entities=inferred_entities or ["service request", "service response"],
        primary_read_paths=[f"handle supported {entity} flows through a stable contract"],
        primary_write_paths=[],
        invariants=["The service preserves the published contract for supported requests."],
        operational_concerns=["error handling", "observability"],
        runtime_binding=runtime_binding,
        runtime_plan=runtime_plan,
    )


def build_project_runtime_binding(
    *,
    family: ProjectFamily,
    implementation_language: str | None,
    application_framework: str | None,
    primary_database: str | None,
    cache_backend: str | None,
    tech_stack: list[str],
    data_sources: list[DataSourceSpec],
) -> ProjectRuntimeBindingSpec:
    backing_services: list[ProjectServiceBinding] = []
    seed_artifacts: list[str] = []
    integration_points: list[str] = []

    if implementation_language:
        language_note = f"Implement the learner-facing service in {implementation_language}."
        if application_framework:
            language_note = f"Implement the learner-facing service in {implementation_language} using {application_framework}."
        integration_points.append(language_note)

    if primary_database:
        backing_services.append(
            ProjectServiceBinding(
                service_id=primary_database,
                role="durable state",
                technology=primary_database,
            )
        )
        seed_artifacts.append(f"Initialize and seed {primary_database} for learner-visible scenarios.")
        integration_points.append(f"Connect the application write path to {primary_database}.")

    if cache_backend:
        backing_services.append(
            ProjectServiceBinding(
                service_id=cache_backend,
                role="cache or fast read path",
                technology=cache_backend,
            )
        )
        integration_points.append(f"Wire {cache_backend} into the read path without breaking freshness guarantees.")

    for source in data_sources:
        if source.kind == DataSourceKind.uploaded_file:
            seed_artifacts.append(f"Materialize `{source.title}` into the learner workspace.")
        elif source.kind == DataSourceKind.seed_database:
            seed_artifacts.append(f"Load `{source.title}` into the backing state before review runs.")
        elif source.kind == DataSourceKind.mock_api:
            backing_services.append(
                ProjectServiceBinding(
                    service_id=source.id,
                    role="mock dependency",
                    technology=source.format or "http",
                )
            )
            integration_points.append(f"Bind the application to the mocked dependency `{source.title}`.")
        elif source.kind == DataSourceKind.object_store:
            backing_services.append(
                ProjectServiceBinding(
                    service_id=source.id,
                    role="object storage",
                    technology=source.format or "blob storage",
                )
            )

    if family == ProjectFamily.control_plane_service:
        integration_points.extend(
            [
                "Keep live decisions deterministic for the same request context.",
                "Publish mutations without leaving caches or derived read paths stale.",
            ]
        )
    elif family == ProjectFamily.transactional_stateful_service:
        integration_points.extend(
            [
                "Preserve write correctness under repeated or concurrent requests.",
                "Make state transitions observable enough to debug production failures.",
            ]
        )
    elif family == ProjectFamily.workflow_agent_service:
        integration_points.extend(
            [
                "Keep tool routing bounded and explainable in traces.",
                "Handle approval and fallback paths without breaking the response contract.",
            ]
        )
    elif family in {ProjectFamily.grounded_retrieval_service, ProjectFamily.ranked_retrieval_service}:
        integration_points.extend(
            [
                "Make the retrieval layer query learner-visible data consistently.",
                "Keep the answer or ranking path aligned with the visible corpus contract.",
            ]
        )
    else:
        integration_points.append("Keep the application contract stable while integrating supporting runtime pieces.")

    for tech in tech_stack:
        if tech and tech not in {primary_database, cache_backend}:
            integration_points.append(f"Use `{tech}` only where it materially supports the project contract.")

    return ProjectRuntimeBindingSpec(
        implementation_language=implementation_language,
        application_framework=application_framework,
        backing_services=backing_services,
        seed_artifacts=list(dict.fromkeys(seed_artifacts)),
        integration_points=list(dict.fromkeys(integration_points)),
    )


def _version_hint_for(*, aliases: list[str], tech_stack: list[str]) -> str | None:
    alias_set = {alias.lower() for alias in aliases if alias}
    for item in tech_stack:
        lowered = item.lower().strip()
        if not lowered:
            continue
        if not any(alias in lowered for alias in alias_set):
            continue
        match = re.search(r"\b(?:v)?(\d+(?:\.\d+)*)\b", lowered)
        if match:
            return match.group(1)
    return None


def default_language_version_for_stack(implementation_language: str | None) -> str | None:
    normalized_language = (implementation_language or "").strip().lower() or None
    defaults = {
        "python": "3.12",
        "typescript": "22",
        "javascript": "22",
        "go": "1.25",
        "rust": "1.86",
    }
    return defaults.get(normalized_language)


def default_dependency_version(technology: str | None) -> str | None:
    normalized = (technology or "").strip().lower() or None
    defaults = {
        "postgres": "16",
        "postgresql": "16",
        "mongodb": "7",
        "mongo": "7",
        "redis": "7",
        "mysql": "8",
        "mariadb": "8",
    }
    return defaults.get(normalized)


def infer_package_manager(
    *,
    implementation_language: str | None,
    tech_stack: list[str],
) -> str | None:
    lowered_stack = " ".join(tech_stack).lower()
    if "pnpm" in lowered_stack:
        return "pnpm"
    if "yarn" in lowered_stack:
        return "yarn"
    if "bun" in lowered_stack:
        return "bun"
    if "npm" in lowered_stack:
        return "npm"
    if "uv" in lowered_stack:
        return "uv"
    if "poetry" in lowered_stack:
        return "poetry"
    if implementation_language == "python":
        return "uv"
    if implementation_language in {"typescript", "javascript"}:
        return "pnpm"
    if implementation_language == "go":
        return "go"
    if implementation_language == "rust":
        return "cargo"
    return None


def runtime_container_image_for_stack(
    *,
    implementation_language: str | None,
    language_version: str | None,
    allow_inference: bool = True,
) -> str | None:
    normalized_language = (implementation_language or "").strip().lower() or None
    version = (language_version or "").strip() or (
        default_language_version_for_stack(normalized_language)
        if allow_inference
        else None
    )
    if normalized_language == "python":
        return f"python:{version}-slim" if version else None
    if normalized_language in {"typescript", "javascript"}:
        return f"node:{version}-bookworm-slim" if version else None
    if normalized_language == "go":
        return f"golang:{version}-bookworm" if version else None
    if normalized_language == "rust":
        return f"rust:{version}-bookworm" if version else None
    return None


def dependency_container_image(
    *,
    technology: str | None,
    version_hint: str | None,
    allow_inference: bool = True,
) -> str | None:
    normalized = (technology or "").strip().lower()
    version = (version_hint or "").strip() or (
        default_dependency_version(normalized)
        if allow_inference
        else None
    )
    if normalized in {"postgres", "postgresql"}:
        base_version = version or "16"
        return base_version if ":" in base_version else f"postgres:{base_version}-alpine"
    if normalized in {"mongodb", "mongo"}:
        return f"mongo:{version or '7'}"
    if normalized == "redis":
        base_version = version or "7"
        return base_version if ":" in base_version else f"redis:{base_version}-alpine"
    if normalized in {"mysql", "mariadb"}:
        return f"{normalized}:{version or '8'}"
    return None


def build_project_runtime_plan(
    *,
    family: ProjectFamily,
    implementation_language: str | None,
    language_version: str | None,
    application_framework: str | None,
    framework_version: str | None,
    package_manager: str | None,
    primary_database: str | None,
    primary_database_version: str | None,
    cache_backend: str | None,
    cache_backend_version: str | None,
    tech_stack: list[str],
    data_sources: list[DataSourceSpec],
    allow_inference: bool = True,
) -> ProjectRuntimePlanSpec:
    language_runtime = {
        "typescript": "node",
        "javascript": "node",
    }.get(implementation_language or "", implementation_language)
    resolved_language_version = (language_version or "").strip() or (
        _version_hint_for(
            aliases=[language_runtime or "", implementation_language or ""],
            tech_stack=tech_stack,
        )
        if allow_inference
        else None
    ) or (
        default_language_version_for_stack(implementation_language)
        if allow_inference
        else None
    )
    resolved_framework_version = (framework_version or "").strip() or (
        _version_hint_for(
            aliases=[application_framework or ""],
            tech_stack=tech_stack,
        )
        if allow_inference
        else None
    )
    resolved_package_manager = (package_manager or "").strip().lower() or (
        infer_package_manager(
            implementation_language=implementation_language,
            tech_stack=tech_stack,
        )
        if allow_inference
        else None
    )
    resolved_primary_database_version = (primary_database_version or "").strip() or (
        _version_hint_for(
            aliases=[primary_database or ""],
            tech_stack=tech_stack,
        )
        if allow_inference
        else None
    ) or (
        default_dependency_version(primary_database)
        if allow_inference
        else None
    )
    resolved_cache_backend_version = (cache_backend_version or "").strip() or (
        _version_hint_for(
            aliases=[cache_backend or ""],
            tech_stack=tech_stack,
        )
        if allow_inference
        else None
    ) or (
        default_dependency_version(cache_backend)
        if allow_inference
        else None
    )
    services: list[ProjectRuntimeServiceSpec] = [
        ProjectRuntimeServiceSpec(
            service_id="app",
            role="learner-facing application",
            technology=application_framework or implementation_language,
                version_hint=resolved_framework_version or resolved_language_version,
                package_manager=resolved_package_manager,
                entrypoint_path=None,
                container_image=runtime_container_image_for_stack(
                    implementation_language=implementation_language,
                    language_version=resolved_language_version,
                    allow_inference=allow_inference,
                ),
                learner_managed=True,
                healthcheck_path="/health",
                default_port=8000,
            )
    ]

    if primary_database:
        services.append(
            ProjectRuntimeServiceSpec(
                service_id=primary_database,
                role="durable state",
                technology=primary_database,
                version_hint=resolved_primary_database_version,
                container_image=dependency_container_image(
                    technology=primary_database,
                    version_hint=resolved_primary_database_version,
                    allow_inference=allow_inference,
                ),
                learner_managed=False,
            )
        )
    if cache_backend:
        services.append(
            ProjectRuntimeServiceSpec(
                service_id=cache_backend,
                role="cache or fast read path",
                technology=cache_backend,
                version_hint=resolved_cache_backend_version,
                container_image=dependency_container_image(
                    technology=cache_backend,
                    version_hint=resolved_cache_backend_version,
                    allow_inference=allow_inference,
                ),
                learner_managed=False,
            )
        )

    for source in data_sources:
        if source.kind == DataSourceKind.mock_api:
            services.append(
                ProjectRuntimeServiceSpec(
                    service_id=source.id,
                    role="mock dependency",
                    technology=source.format or "http",
                    learner_managed=False,
                )
            )
        elif source.kind == DataSourceKind.object_store:
            services.append(
                ProjectRuntimeServiceSpec(
                    service_id=source.id,
                    role="object storage",
                    technology=source.format or "blob storage",
                    learner_managed=False,
                )
            )

    seed_steps: list[ProjectRuntimeCommandSpec] = []
    if primary_database:
        seed_steps.append(
            ProjectRuntimeCommandSpec(
                phase="seed",
                command=f"Seed {primary_database} with learner-visible baseline data.",
                target_service_id=primary_database,
            )
        )
    for source in data_sources:
        if source.kind == DataSourceKind.uploaded_file and source.workspace_path:
            seed_steps.append(
                ProjectRuntimeCommandSpec(
                    phase="seed",
                    command=f"Materialize `{source.title}` at `{source.workspace_path}`.",
                target_service_id="app",
            )
        )

    notes = [
        f"Target the `{family.value}` runtime shape rather than falling back to a generic app contract.",
        "Treat runtime plan commands as advisory metadata only; authored runtime protocol files are the execution source of truth.",
    ]
    if tech_stack:
        notes.append(
            "Honor explicit runtime requirements such as "
            + ", ".join(f"`{item}`" for item in tech_stack[:5])
            + "."
        )

    return ProjectRuntimePlanSpec(
        implementation_language=implementation_language,
        language_version=resolved_language_version,
        application_framework=application_framework,
        framework_version=resolved_framework_version,
        package_manager=resolved_package_manager,
        services=services,
        setup_steps=[],
        seed_steps=seed_steps,
        verify_steps=[],
        run_steps=[],
        check_steps=[],
        notes=notes,
    )


def infer_implementation_stack(
    *,
    title: str,
    problem_statement: str,
    implementation_language: str | None,
    application_framework: str | None,
    tech_stack: list[str] | None,
) -> tuple[str | None, str | None]:
    normalized_language = (implementation_language or "").strip().lower() or None
    normalized_framework = (application_framework or "").strip().lower() or None
    text = " ".join([title, problem_statement, *(tech_stack or [])]).lower()

    if normalized_framework is None:
        for candidate_framework in sorted(FRAMEWORK_LANGUAGE_HINTS.keys(), key=len, reverse=True):
            if _contains_keyword(text, candidate_framework):
                normalized_framework = candidate_framework
                break

    if normalized_framework and not normalized_language:
        normalized_language = FRAMEWORK_LANGUAGE_HINTS.get(normalized_framework)

    if normalized_language is None:
        for candidate_language, keywords in LANGUAGE_KEYWORDS.items():
            if _contains_any_keyword(text, keywords):
                normalized_language = candidate_language
                break

    if normalized_framework is None and normalized_language is not None:
        normalized_framework = DEFAULT_FRAMEWORK_BY_LANGUAGE.get(normalized_language)

    if normalized_language is None and normalized_framework is None:
        normalized_language = "python"
        normalized_framework = "fastapi"

    return normalized_language, normalized_framework


def runtime_commands_for_stack(
    *,
    implementation_language: str | None,
    application_framework: str | None,
) -> tuple[str, str, str]:
    _ = (implementation_language, application_framework)
    local_run_command = f"sh {RUNTIME_RUN_SCRIPT_PATH}"
    visible_check_command = f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"
    preview_command = local_run_command
    return local_run_command, visible_check_command, preview_command


def infer_package_type(*, text: str, package_type_hint: PackageType | None) -> PackageType:
    if package_type_hint is not None:
        return package_type_hint
    if _contains_any_keyword(
        text,
        [
            "demo to production",
            "inherited demo",
            "progressive",
            "production ready",
            "production-ready",
        ],
    ):
        return PackageType.progressive_codebase_course
    if _contains_any_keyword(text, ["course", "catalog", "survey", "multiple assignments"]):
        return PackageType.survey_course
    return PackageType.progressive_codebase_course


def infer_risk_class(text: str) -> RiskClass:
    if _contains_any_keyword(text, HIGH_STAKES_KEYWORDS):
        return RiskClass.high_stakes
    if _contains_any_keyword(text, REVIEW_REQUIRED_KEYWORDS):
        return RiskClass.review_required
    return RiskClass.standard


def infer_overlays(text: str) -> list[str]:
    overlays: list[str] = []
    for overlay_id, keywords in OVERLAY_KEYWORDS.items():
        if _contains_any_keyword(text, keywords):
            overlays.append(overlay_id)
    return overlays


def infer_domain_pack(text: str) -> str | None:
    scores: list[tuple[int, str]] = []
    for domain_pack_id, keywords in DOMAIN_PACK_KEYWORDS.items():
        score = _keyword_match_count(text, keywords)
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
    project_contract: ProjectContractSpec | None = None,
    retrieval_mode: RetrievalMode = RetrievalMode.none,
    answer_synthesis_required: bool = False,
    citations_required: bool = False,
    abstention_required: bool = False,
    tool_use_required: bool = False,
    traceability_required: bool = True,
    durable_state_required: bool = False,
    approval_flow_required: bool = False,
    execution_surface: ExecutionSurface = ExecutionSurface.http_service,
    starter_type: StarterType = StarterType.partial,
    implementation_language: str | None = None,
    language_version: str | None = None,
    application_framework: str | None = None,
    framework_version: str | None = None,
    package_manager: str | None = None,
    primary_database: str | None = None,
    primary_database_version: str | None = None,
    cache_backend: str | None = None,
    cache_backend_version: str | None = None,
    tech_stack: list[str] | None = None,
    data_sources: list[DataSourceSpec] | None = None,
) -> AssignmentDesignSpec:
    resolved_project_contract = project_contract or default_project_contract()
    source_specs = list(data_sources or [])
    fallback_local_run_command, fallback_visible_check_command, fallback_preview_command = runtime_commands_for_stack(
        implementation_language=implementation_language,
        application_framework=application_framework,
    )
    runtime_plan = resolved_project_contract.runtime_plan
    app_service = next((service for service in runtime_plan.services if service.service_id == "app"), None)
    local_run_command = fallback_local_run_command
    preview_command = fallback_preview_command
    visible_check_command = fallback_visible_check_command
    editable_files: list[str] = []
    visible_fixture_files = [
        source.workspace_path
        for source in source_specs
        if source.learner_visible and source.workspace_path
    ]
    if not visible_fixture_files and retrieval_mode != RetrievalMode.none:
        visible_fixture_files = ["data/corpus.json"]
    shared_codebase = package_type == PackageType.progressive_codebase_course
    return AssignmentDesignSpec(
            course_structure=CourseStructureSpec(
                package_type=package_type,
                workspace_scope=(
                    WorkspaceScope.shared_course_workspace
                    if shared_codebase
                    else WorkspaceScope.per_deliverable_workspace
                ),
                progression_mode=ProgressionMode.independent_deliverables,
                shared_codebase=shared_codebase,
            ),
        runtime_dependencies=RuntimeDependencySpec(
            execution_surface=execution_surface,
            starter_type=starter_type,
            implementation_language=implementation_language,
            language_version=language_version or runtime_plan.language_version,
            application_framework=application_framework,
            framework_version=framework_version or runtime_plan.framework_version,
            package_manager=package_manager or runtime_plan.package_manager,
            editable_files=editable_files,
            visible_fixture_files=visible_fixture_files,
            data_sources=source_specs,
            primary_database=primary_database,
            primary_database_version=primary_database_version
            or next(
                (
                    service.version_hint
                    for service in runtime_plan.services
                    if service.service_id == primary_database
                ),
                None,
            ),
            cache_backend=cache_backend,
            cache_backend_version=cache_backend_version
            or next(
                (
                    service.version_hint
                    for service in runtime_plan.services
                    if service.service_id == cache_backend
                ),
                None,
            ),
            tech_stack=list(tech_stack or []),
            local_run_command=local_run_command,
            visible_check_command=visible_check_command,
            preview_command=preview_command,
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
            cumulative_deliverable_gates=False,
            learner_submission_enabled=True,
        ),
        project_contract=resolved_project_contract,
        risk_class=risk_class,
        domain_pack=domain_pack,
        overlays=list(overlays),
    )


def infer_assignment_design(
    *,
    title: str,
    problem_statement: str,
    learning_outcomes: list[str] | None = None,
    package_type_hint: PackageType | None = None,
    starter_type: StarterType | None = None,
    implementation_language: str | None = None,
    language_version: str | None = None,
    application_framework: str | None = None,
    framework_version: str | None = None,
    package_manager: str | None = None,
    primary_database: str | None = None,
    primary_database_version: str | None = None,
    cache_backend: str | None = None,
    cache_backend_version: str | None = None,
    tech_stack: list[str] | None = None,
    data_sources: list[DataSourceSpec] | None = None,
) -> AssignmentDesignInference:
    resolved_language, resolved_framework = infer_implementation_stack(
        title=title,
        problem_statement=problem_statement,
        implementation_language=implementation_language,
        application_framework=application_framework,
        tech_stack=tech_stack,
    )
    source_signal = " ".join(
        item
        for item in [
            *(source.title for source in data_sources or []),
            *(source.description or "" for source in data_sources or []),
            resolved_language or "",
            resolved_framework or "",
            primary_database or "",
            cache_backend or "",
            *(tech_stack or []),
        ]
        if item
    )
    text = " ".join([title, problem_statement, source_signal]).lower()
    package_type = infer_package_type(text=text, package_type_hint=package_type_hint)
    risk_class = infer_risk_class(text)
    overlays = infer_overlays(text)
    domain_pack = infer_domain_pack(text)

    reasons: list[str] = []
    warnings: list[str] = []

    if _contains_any_keyword(text, GROUNDED_RETRIEVAL_KEYWORDS):
        reasons.append("The brief asks for grounded answering over a visible corpus with evidence-aware behavior.")
        family = ProjectFamily.grounded_retrieval_service
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            project_contract=build_project_contract(
                family=family,
                title=title,
                problem_statement=problem_statement,
                implementation_language=resolved_language,
                language_version=language_version,
                application_framework=resolved_framework,
                framework_version=framework_version,
                package_manager=package_manager,
                primary_database=primary_database,
                primary_database_version=primary_database_version,
                cache_backend=cache_backend,
                cache_backend_version=cache_backend_version,
                tech_stack=tech_stack,
                data_sources=data_sources,
            ),
            retrieval_mode=RetrievalMode.grounded_answers,
            answer_synthesis_required=True,
            citations_required=True,
            abstention_required=True,
            tool_use_required=True,
            traceability_required=True,
            durable_state_required=False,
            approval_flow_required=False,
            starter_type=starter_type or StarterType.partial,
            implementation_language=resolved_language,
            language_version=language_version,
            application_framework=resolved_framework,
            framework_version=framework_version,
            package_manager=package_manager,
            primary_database=primary_database,
            primary_database_version=primary_database_version,
            cache_backend=cache_backend,
            cache_backend_version=cache_backend_version,
            tech_stack=tech_stack,
            data_sources=data_sources,
        )
    elif _contains_any_keyword(text, RANKED_RETRIEVAL_KEYWORDS):
        reasons.append("The brief centers on retrieval quality over a visible corpus.")
        family = ProjectFamily.ranked_retrieval_service
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            project_contract=build_project_contract(
                family=family,
                title=title,
                problem_statement=problem_statement,
                implementation_language=resolved_language,
                language_version=language_version,
                application_framework=resolved_framework,
                framework_version=framework_version,
                package_manager=package_manager,
                primary_database=primary_database,
                primary_database_version=primary_database_version,
                cache_backend=cache_backend,
                cache_backend_version=cache_backend_version,
                tech_stack=tech_stack,
                data_sources=data_sources,
            ),
            retrieval_mode=RetrievalMode.ranked_results,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=False,
            traceability_required=True,
            durable_state_required=False,
            approval_flow_required=False,
            starter_type=starter_type or StarterType.partial,
            implementation_language=resolved_language,
            language_version=language_version,
            application_framework=resolved_framework,
            framework_version=framework_version,
            package_manager=package_manager,
            primary_database=primary_database,
            primary_database_version=primary_database_version,
            cache_backend=cache_backend,
            cache_backend_version=cache_backend_version,
            tech_stack=tech_stack,
            data_sources=data_sources,
        )
    elif _contains_any_keyword(text, CONTROL_PLANE_KEYWORDS):
        reasons.append("The brief describes a control-plane service with low-latency decisions, auditable changes, and safe control updates.")
        family = ProjectFamily.control_plane_service
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            project_contract=build_project_contract(
                family=family,
                title=title,
                problem_statement=problem_statement,
                implementation_language=resolved_language,
                language_version=language_version,
                application_framework=resolved_framework,
                framework_version=framework_version,
                package_manager=package_manager,
                primary_database=primary_database,
                primary_database_version=primary_database_version,
                cache_backend=cache_backend,
                cache_backend_version=cache_backend_version,
                tech_stack=tech_stack,
                data_sources=data_sources,
            ),
            retrieval_mode=RetrievalMode.none,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=False,
            traceability_required=True,
            durable_state_required=True,
            approval_flow_required=False,
            starter_type=starter_type or StarterType.partial,
            implementation_language=resolved_language,
            language_version=language_version,
            application_framework=resolved_framework,
            framework_version=framework_version,
            package_manager=package_manager,
            primary_database=primary_database,
            primary_database_version=primary_database_version,
            cache_backend=cache_backend,
            cache_backend_version=cache_backend_version,
            tech_stack=tech_stack,
            data_sources=data_sources,
        )
    elif _contains_any_keyword(text, STATEFUL_KEYWORDS):
        reasons.append("The brief depends on correctness under persistent mutable state and concurrency.")
        family = ProjectFamily.transactional_stateful_service
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            project_contract=build_project_contract(
                family=family,
                title=title,
                problem_statement=problem_statement,
                implementation_language=resolved_language,
                language_version=language_version,
                application_framework=resolved_framework,
                framework_version=framework_version,
                package_manager=package_manager,
                primary_database=primary_database,
                primary_database_version=primary_database_version,
                cache_backend=cache_backend,
                cache_backend_version=cache_backend_version,
                tech_stack=tech_stack,
                data_sources=data_sources,
            ),
            retrieval_mode=RetrievalMode.none,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=False,
            traceability_required=True,
            durable_state_required=True,
            approval_flow_required=False,
            starter_type=starter_type or StarterType.partial,
            implementation_language=resolved_language,
            language_version=language_version,
            application_framework=resolved_framework,
            framework_version=framework_version,
            package_manager=package_manager,
            primary_database=primary_database,
            primary_database_version=primary_database_version,
            cache_backend=cache_backend,
            cache_backend_version=cache_backend_version,
            tech_stack=tech_stack,
            data_sources=data_sources,
        )
    else:
        reasons.append("The brief fits the general learner-ready service pipeline with bounded workflows and observable behavior.")
        family = (
            ProjectFamily.workflow_agent_service
            if bool(domain_pack or _contains_any_keyword(text, TOOL_USE_KEYWORDS))
            else ProjectFamily.generic_backend_service
        )
        design_spec = build_assignment_design(
            package_type=package_type,
            risk_class=risk_class,
            domain_pack=domain_pack,
            overlays=overlays,
            project_contract=build_project_contract(
                family=family,
                title=title,
                problem_statement=problem_statement,
                implementation_language=resolved_language,
                language_version=language_version,
                application_framework=resolved_framework,
                framework_version=framework_version,
                package_manager=package_manager,
                primary_database=primary_database,
                primary_database_version=primary_database_version,
                cache_backend=cache_backend,
                cache_backend_version=cache_backend_version,
                tech_stack=tech_stack,
                data_sources=data_sources,
            ),
            retrieval_mode=RetrievalMode.none,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=bool(domain_pack or _contains_any_keyword(text, TOOL_USE_KEYWORDS)),
            traceability_required=True,
            durable_state_required="state" in text or "resume" in text or "durable" in text,
            approval_flow_required="approval" in text or "escalat" in text or "handoff" in text,
            starter_type=starter_type or StarterType.partial,
            implementation_language=resolved_language,
            language_version=language_version,
            application_framework=resolved_framework,
            framework_version=framework_version,
            package_manager=package_manager,
            primary_database=primary_database,
            primary_database_version=primary_database_version,
            cache_backend=cache_backend,
            cache_backend_version=cache_backend_version,
            tech_stack=tech_stack,
            data_sources=data_sources,
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
