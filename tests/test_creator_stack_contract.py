from __future__ import annotations

import tempfile

from app.domain.course import CreatorCourseSetupChoices
from app.domain.registry import PackageType, StarterType
from app.domain.course import (
    CreateCourseDeliverableRequest,
    CreateCourseFromCreatorPlanRequest,
    CreatorCourseDeliverablePlan,
    CreatorCoursePlan,
    RecommendCreatorStackContractRequest,
    CreatorCourseSetupInput,
)
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import infer_assignment_design
from app.services.course_generation_service import CourseGenerationService
from app.services.course_workflow_service import CourseWorkflowService
from app.services.openai_task_agent_authoring import OpenAITaskAgentAuthoringService
from app.services.stack_catalog_service import StackCatalogService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _build_services(
    *,
    stack_catalog_service: StackCatalogService | None = None,
) -> tuple[tempfile.TemporaryDirectory[str], CourseGenerationService]:
    temp_dir = tempfile.TemporaryDirectory()
    store = SQLiteWorkflowStore(db_path=f"{temp_dir.name}/test.db")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{temp_dir.name}/generated"),
        task_agent_authoring_service=OpenAITaskAgentAuthoringService(enabled=False),
    )
    course_workflow_service = CourseWorkflowService(store, workflow_service)
    return temp_dir, CourseGenerationService(
        course_workflow_service,
        stack_catalog_service=stack_catalog_service,
    )


def _fake_stack_json(url: str):
    if "library/python/tags" in url:
        return {"results": [{"name": "3.13-slim"}, {"name": "3.12-slim"}]}
    if "library/postgres/tags" in url:
        return {"results": [{"name": "17-alpine"}, {"name": "16-alpine"}]}
    if "library/redis/tags" in url:
        return {"results": [{"name": "8-alpine"}, {"name": "7-alpine"}]}
    if "pypi.org/pypi/fastapi/json" in url:
        return {
            "info": {"version": "0.116.1"},
            "releases": {
                "0.116.1": [{}],
                "0.115.14": [{}],
            },
        }
    raise AssertionError(f"Unhandled URL: {url}")


def _fake_stack_text(url: str) -> str:
    raise AssertionError(f"Unhandled URL: {url}")


def test_infer_assignment_design_prefers_explicit_stack_contract_over_tech_stack() -> None:
    inferred = infer_assignment_design(
        title="Feature Flag Control Plane",
        problem_statement="Build a feature flag control plane backend with safe rollout controls and audit logs.",
        package_type_hint=PackageType.progressive_codebase_course,
        starter_type=StarterType.partial,
        implementation_language="typescript",
        language_version="24",
        application_framework="nestjs",
        framework_version="11",
        package_manager="npm",
        primary_database="mongodb",
        primary_database_version="8",
        cache_backend="redis",
        cache_backend_version="8",
        tech_stack=["Node 22", "NestJS 10", "pnpm", "MongoDB 7", "Redis 7"],
    )

    assert inferred.design_spec is not None
    runtime_plan = inferred.design_spec.project_contract.runtime_plan
    runtime_dependencies = inferred.design_spec.runtime_dependencies
    app_service = next(service for service in runtime_plan.services if service.service_id == "app")

    assert runtime_plan.language_version == "24"
    assert runtime_plan.framework_version == "11"
    assert runtime_plan.package_manager == "npm"
    assert app_service.package_manager == "npm"
    assert "node:24-bookworm-slim" in (app_service.container_image or "")
    assert runtime_dependencies.language_version == "24"
    assert runtime_dependencies.framework_version == "11"
    assert runtime_dependencies.package_manager == "npm"
    assert runtime_dependencies.primary_database_version == "8"
    assert runtime_dependencies.cache_backend_version == "8"


def test_infer_assignment_design_does_not_treat_scenarios_as_ios_ui_work() -> None:
    inferred = infer_assignment_design(
        title="Build a Correct-by-Construction Inventory Reservation Service in Rust",
        problem_statement=(
            "Implement a production-grade inventory reservation HTTP service with Axum and Postgres, "
            "focusing on correctness for reservation writes under concurrency, retries, and failure scenarios."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
        starter_type=StarterType.partial,
        implementation_language="rust",
        language_version="1.82",
        application_framework="axum",
        framework_version="0.8",
        package_manager="cargo",
        primary_database="postgres",
        primary_database_version="16",
    )

    assert inferred.status.value == "supported"
    assert inferred.design_spec is not None
    assert inferred.design_spec.project_contract.family.value == "transactional_stateful_service"


def test_infer_assignment_design_no_longer_blocks_briefs_from_ui_keywords_in_prose() -> None:
    inferred = infer_assignment_design(
        title="Build an iOS field operations app",
        problem_statement="Create an iOS mobile app for field technicians with offline sync and camera capture.",
        package_type_hint=PackageType.progressive_codebase_course,
        implementation_language="typescript",
        language_version="24",
        application_framework="express",
        framework_version="5",
        package_manager="npm",
    )

    assert inferred.status.value in {"supported", "manual_review"}
    assert inferred.design_spec is not None


def test_course_generation_service_rebuilds_runtime_plan_from_creator_contract() -> None:
    temp_dir, service = _build_services()
    try:
        inferred = infer_assignment_design(
            title="Inventory Reservation Service",
            problem_statement=(
                "Build a multi-warehouse inventory reservation service that is production ready. "
                "Keep reservations correct under concurrency, retries, and stock transfers."
            ),
            implementation_language="python",
            language_version="3.12",
            application_framework="fastapi",
            framework_version="0.115",
            package_manager="uv",
            primary_database="postgres",
            primary_database_version="16",
            cache_backend="redis",
            cache_backend_version="7",
        )
        assert inferred.design_spec is not None

        creator_choices = CreatorCourseSetupChoices(
            starter_type=StarterType.partial,
            implementation_language="go",
            language_version="1.25",
            application_framework="gin",
            framework_version="1.11",
            package_manager="go",
            primary_database="postgres",
            primary_database_version="17",
            cache_backend="redis",
            cache_backend_version="8",
            tech_stack=["Go 1.25", "Gin 1.11", "Postgres 17", "Redis 8"],
        )

        updated = service._apply_creator_choices_to_design_spec(inferred.design_spec, creator_choices)

        assert updated is not None
        runtime_dependencies = updated.runtime_dependencies
        runtime_plan = updated.project_contract.runtime_plan
        app_service = next(service for service in runtime_plan.services if service.service_id == "app")

        assert runtime_dependencies.starter_type == StarterType.partial
        assert runtime_dependencies.implementation_language == "go"
        assert runtime_dependencies.language_version == "1.25"
        assert runtime_dependencies.application_framework == "gin"
        assert runtime_dependencies.framework_version == "1.11"
        assert runtime_dependencies.package_manager == "go"
        assert runtime_dependencies.primary_database_version == "17"
        assert runtime_dependencies.cache_backend_version == "8"
        assert runtime_plan.implementation_language == "go"
        assert runtime_plan.language_version == "1.25"
        assert runtime_plan.application_framework == "gin"
        assert runtime_plan.framework_version == "1.11"
        assert runtime_plan.package_manager == "go"
        assert app_service.package_manager == "go"
        assert "golang:1.25-bookworm" in (app_service.container_image or "")
    finally:
        temp_dir.cleanup()


def test_create_course_run_from_creator_plan_keeps_creator_contract_fixed() -> None:
    temp_dir, service = _build_services()
    try:
        inferred = infer_assignment_design(
            title="Incident Commander Service",
            problem_statement="Build an incident commander service with approvals, traces, and durable state.",
            implementation_language="python",
            application_framework="fastapi",
            primary_database="postgres",
            cache_backend="redis",
        )
        assert inferred.design_spec is not None

        creator_choices = CreatorCourseSetupChoices(
            starter_type=StarterType.partial,
            implementation_language="go",
            language_version="1.25",
            application_framework="gin",
            framework_version="1.11",
            package_manager="go",
            primary_database="postgres",
            primary_database_version="17",
            cache_backend="redis",
            cache_backend_version="8",
        )
        plan = CreatorCoursePlan(
            goal="Build an incident commander service with approvals, traces, and durable state.",
            learning_outcomes=["Ship a runnable incident commander service."],
            title="Incident Commander Service",
            summary="Build an incident commander backend.",
            package_type=PackageType.progressive_codebase_course,
            creator_choices=creator_choices,
            shared_design_spec=inferred.design_spec,
            deliverables=[
                CreatorCourseDeliverablePlan(
                    deliverable_slug="service-contract",
                    title="Service contract and workflow state",
                    summary="Define the service contract and workflow states.",
                    learning_outcomes=["Model the public API and workflow state."],
                    creator_notes=[],
                    design_spec=inferred.design_spec,
                )
            ],
        )

        course_run = service.create_course_run_from_creator_plan(
            CreateCourseFromCreatorPlanRequest(plan=plan)
        )
        assert course_run.shared_design_spec is not None
        assert course_run.creator_choices is not None
        runtime_plan = course_run.shared_design_spec.project_contract.runtime_plan
        stored_course_run = service.course_workflow_service.get_run(course_run.id)

        assert stored_course_run is not None
        assert stored_course_run.creator_choices is not None

        assert runtime_plan.implementation_language == "go"
        assert runtime_plan.language_version == "1.25"
        assert runtime_plan.application_framework == "gin"
        assert runtime_plan.framework_version == "1.11"
        assert runtime_plan.package_manager == "go"
        assert course_run.creator_choices.language_version == "1.25"
        assert course_run.creator_choices.framework_version == "1.11"
        assert course_run.creator_choices.package_manager == "go"
        assert stored_course_run.creator_choices.language_version == "1.25"
        assert stored_course_run.creator_choices.framework_version == "1.11"
        assert stored_course_run.creator_choices.package_manager == "go"
    finally:
        temp_dir.cleanup()


def test_legacy_payload_normalization_uses_runtime_plan_instead_of_python_defaults() -> None:
    temp_dir = tempfile.TemporaryDirectory()
    try:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir.name}/test.db")
        normalized = store._normalize_task_agent_spec_payload(
            {
                "title": "Go Incident Commander",
                "summary": "Build a Go incident commander service.",
                "project_contract": {
                    "family": "control_plane_service",
                    "system_kind": "Incident commander backend",
                    "core_entities": [],
                    "primary_read_paths": ["/incidents/{id}"],
                    "primary_write_paths": ["/incidents"],
                    "invariants": ["Keep incident workflow state coherent."],
                    "operational_concerns": ["Boots cleanly."],
                    "runtime_binding": {
                        "implementation_language": "go",
                        "application_framework": "gin",
                        "backing_services": [],
                        "seed_artifacts": [],
                        "integration_points": [],
                    },
                    "runtime_plan": {
                        "implementation_language": "go",
                        "language_version": "1.25",
                        "application_framework": "gin",
                        "framework_version": "1.11",
                        "package_manager": "go",
                        "services": [],
                    },
                },
                "runtime_plan": {
                    "implementation_language": "go",
                    "language_version": "1.25",
                    "application_framework": "gin",
                    "framework_version": "1.11",
                    "package_manager": "go",
                    "services": [],
                },
                "public_endpoints": [{"method": "GET", "path": "/health", "required": True}],
            }
        )

        runtime_dependencies = normalized["runtime_dependencies"]
        runtime_binding = normalized["project_contract"]["runtime_binding"]

        assert runtime_dependencies["implementation_language"] == "go"
        assert runtime_dependencies["language_version"] == "1.25"
        assert runtime_dependencies["application_framework"] == "gin"
        assert runtime_dependencies["framework_version"] == "1.11"
        assert runtime_dependencies["package_manager"] == "go"
        assert runtime_binding["implementation_language"] == "go"
        assert runtime_binding["application_framework"] == "gin"
    finally:
        temp_dir.cleanup()


def test_stack_catalog_service_prefers_public_source_recommendations() -> None:
    service = StackCatalogService(
        json_fetcher=_fake_stack_json,
        text_fetcher=_fake_stack_text,
        ttl_seconds=0,
    )

    response = service.describe_choices(
        CreatorCourseSetupChoices(
            implementation_language="python",
            application_framework="fastapi",
            primary_database="postgres",
            cache_backend="redis",
        )
    )

    assert response.creator_choices.language_version is None
    assert response.creator_choices.framework_version is None
    assert response.creator_choices.package_manager is None
    assert response.creator_choices.primary_database_version is None
    assert response.creator_choices.cache_backend_version is None
    assert response.language_versions[0].recommended is True
    assert response.framework_versions[0].recommended is True
    assert response.database_versions[0].recommended is True
    assert response.cache_versions[0].recommended is True
    assert response.catalog.frameworks_by_language["python"][0].recommended is True
    assert response.catalog.package_managers_by_language["python"][0].recommended is True


def test_course_generation_service_recommends_creator_stack_contract_from_public_catalog() -> None:
    stack_catalog_service = StackCatalogService(
        json_fetcher=_fake_stack_json,
        text_fetcher=_fake_stack_text,
        ttl_seconds=0,
    )
    temp_dir, service = _build_services(stack_catalog_service=stack_catalog_service)
    try:
        response = service.recommend_creator_stack_contract(
            RecommendCreatorStackContractRequest(
                goal="Build a production-ready inventory reservation backend with durable writes and a cache.",
                creator_setup=CreatorCourseSetupInput(
                    implementation_language="python",
                    application_framework="fastapi",
                    primary_database="postgres",
                    cache_backend="redis",
                ),
            )
        )

        assert response.creator_choices.implementation_language == "python"
        assert response.creator_choices.application_framework == "fastapi"
        assert response.creator_choices.language_version is None
        assert response.creator_choices.framework_version is None
        assert response.creator_choices.package_manager is None
        assert response.creator_choices.primary_database_version is None
        assert response.creator_choices.cache_backend_version is None
        assert response.language_versions[0].recommended is True
        assert response.framework_versions[0].recommended is True
        assert response.database_versions[0].recommended is True
        assert response.cache_versions[0].recommended is True
    finally:
        temp_dir.cleanup()


def test_course_generation_service_does_not_inject_database_or_cache_from_goal() -> None:
    temp_dir, service = _build_services()
    try:
        resolved = service._resolve_creator_setup(
            "Build a cache-aware inventory reservation backend with durable writes and low-latency reads.",
            CreatorCourseSetupInput(
                implementation_language="rust",
                application_framework="axum",
            ),
        )

        assert resolved.implementation_language == "rust"
        assert resolved.application_framework == "axum"
        assert resolved.primary_database is None
        assert resolved.cache_backend is None
    finally:
        temp_dir.cleanup()
