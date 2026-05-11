from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.registry import PackageType
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import (
    dependency_container_image,
    infer_assignment_design,
)
from app.services.learner_studio_service import LearnerStudioService
from app.services.spec_validation import validate_task_agent_spec
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.services.task_agent_starter_templates import (
    build_task_agent_starter_files,
    HIDDEN_MANIFEST_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
)


def _build_spec(
    *,
    title: str,
    summary: str,
    problem_statement: str,
):
    inferred = infer_assignment_design(
        title=title,
        problem_statement=problem_statement,
        learning_outcomes=[
            "Design a production-grade backend surface.",
            "Ship a runtime that can be graded end to end.",
        ],
        package_type_hint=PackageType.progressive_codebase_course,
    )
    assert inferred.design_spec is not None
    # In production, primary_editable_paths is authored by the OpenAI task-agent
    # call (which decides based on the chosen stack). These tests bypass that
    # call, so seed a placeholder editable file so the validation downstream of
    # the scaffold builder has something to bind primary_editable_paths to.
    if not inferred.design_spec.runtime_dependencies.editable_files:
        inferred.design_spec.runtime_dependencies.editable_files = ["app.py"]
    spec, _origin = build_task_agent_scaffold(
        title=title,
        summary=summary,
        design_spec=inferred.design_spec,
    )
    return spec


def test_typescript_starter_dockerfile_follows_runtime_plan() -> None:
    spec = _build_spec(
        title="Feature Flag Control Plane",
        summary="Build a feature flag control plane service.",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, "
            "NestJS 11, Node 22, MongoDB 7, pnpm, audit logs, and safe config updates."
        ),
    )

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    dockerfile = starter_files["Dockerfile"]
    manifest = json.loads(starter_files[HIDDEN_MANIFEST_PATH])

    assert dockerfile.startswith("FROM ")
    assert "FROM debian:bookworm-slim" in dockerfile
    assert "runtime Dockerfile has not been authored yet" in dockerfile
    assert manifest["runtime_protocol_bundle"]["source"] == "starter_default"


def test_runtime_plan_prefers_lightweight_dependency_images() -> None:
    assert dependency_container_image(technology="postgres", version_hint=None) == "postgres:16-alpine"
    assert dependency_container_image(technology="redis", version_hint=None) == "redis:7-alpine"


def test_assignment_runtime_plan_prefers_explicit_stack_contract_over_tech_stack() -> None:
    inferred = infer_assignment_design(
        title="Feature Flag Control Plane",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, audit logs, and safe config updates."
        ),
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
    mongodb_service = next(service for service in runtime_plan.services if service.service_id == "mongodb")
    redis_service = next(service for service in runtime_plan.services if service.service_id == "redis")

    assert runtime_plan.language_version == "24"
    assert runtime_plan.framework_version == "11"
    assert runtime_plan.package_manager == "npm"
    assert app_service.package_manager == "npm"
    assert "node:24-bookworm-slim" in (app_service.container_image or "")
    assert runtime_plan.setup_steps == []
    assert runtime_plan.verify_steps == []
    assert runtime_plan.run_steps == []
    assert runtime_plan.check_steps == []
    assert runtime_dependencies.language_version == "24"
    assert runtime_dependencies.framework_version == "11"
    assert runtime_dependencies.package_manager == "npm"
    assert runtime_dependencies.primary_database_version == "8"
    assert runtime_dependencies.cache_backend_version == "8"
    assert mongodb_service.version_hint == "8"
    assert redis_service.version_hint == "8"


def test_assignment_runtime_dockerfile_reuses_app_base_and_adds_verifier_python() -> None:
    spec = _build_spec(
        title="Feature Flag Control Plane",
        summary="Build a feature flag control plane service.",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, "
            "NestJS 11, Node 22, MongoDB 7, pnpm, audit logs, and safe config updates."
        ),
    )

    materializer = ArtifactMaterializer()
    dockerfile = materializer._assignment_runtime_dockerfile(spec)

    assert dockerfile.startswith("FROM ")
    assert "apt-get install -y --no-install-recommends python3" in dockerfile
    assert (
        "corepack enable" in dockerfile
        or "apt-get install -y --no-install-recommends nodejs npm" in dockerfile
    )
    assert 'CMD ["python3", "runtime/verify_assignment.py"]' in dockerfile


def test_learner_runtime_launch_script_exports_corepack_prompt_override() -> None:
    spec = _build_spec(
        title="Feature Flag Control Plane",
        summary="Build a feature flag control plane service.",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, "
            "NestJS 11, Node 22, MongoDB 7, pnpm, audit logs, and safe config updates."
        ),
    )
    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    with TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        for relative_path, content in starter_files.items():
            output_path = workspace_path / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        manifest = json.loads((workspace_path / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))
        assert manifest["runtime_plan"]["package_manager"] == "pnpm"

        service = LearnerStudioService()
        launch_script = service._runtime_launch_script(
            workspace_path=workspace_path,
            spec=spec,
            include_setup=True,
        )

    assert f"sh {RUNTIME_INSTALL_SCRIPT_PATH}" in launch_script
    assert f"sh {RUNTIME_VERIFY_SCRIPT_PATH}" in launch_script
    assert f"exec sh {RUNTIME_RUN_SCRIPT_PATH}" in launch_script


def test_python_launch_script_uses_authored_runtime_protocol_before_preview() -> None:
    spec = _build_spec(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
    )

    assert spec.project_contract.runtime_plan.verify_steps == []

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    with TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        for relative_path, content in starter_files.items():
            output_path = workspace_path / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        service = LearnerStudioService()
        launch_script = service._runtime_launch_script(
            workspace_path=workspace_path,
            spec=spec,
            include_setup=False,
        )

    assert f"sh {RUNTIME_VERIFY_SCRIPT_PATH}" in launch_script
    assert launch_script.index(f"sh {RUNTIME_VERIFY_SCRIPT_PATH}") < launch_script.index(f"exec sh {RUNTIME_RUN_SCRIPT_PATH}")


def test_protocol_only_starter_surface_tracks_repo_bundle_metadata() -> None:
    spec = _build_spec(
        title="Grounded Internal Docs Assistant",
        summary="Build a grounded assistant over a visible internal docs corpus.",
        problem_statement=(
            "Build a grounded internal docs assistant that answers from a visible corpus with citations "
            "and abstains when support is weak."
        ),
    )

    validation = validate_task_agent_spec(spec)
    assert validation.valid
    starter_surface = spec.deliverables[0].learner_starter_surface
    assert starter_surface is not None
    assert starter_surface.primary_editable_paths
    assert starter_surface.required_endpoints
    assert starter_surface.domain_scenarios

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    manifest = json.loads(starter_files[HIDDEN_MANIFEST_PATH])

    assert manifest["learner_starter_surface"]["primary_editable_paths"]
    assert manifest["learner_starter_surface"]["required_endpoints"]
    assert manifest["visible_check_command"] == f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"
    assert manifest["starter_repo_bundle"]["source"] == "starter_default"
    for relative_path in starter_surface.primary_editable_paths:
        assert relative_path not in starter_files


def test_protocol_only_starter_does_not_generate_a_fake_entrypoint() -> None:
    spec = _build_spec(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
    )

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    starter_surface = spec.deliverables[0].learner_starter_surface
    assert starter_surface is not None
    for relative_path in starter_surface.primary_editable_paths:
        assert relative_path not in starter_files


def test_generic_workflow_specs_now_start_from_a_neutral_valid_contract() -> None:
    spec = _build_spec(
        title="Workflow Agent",
        summary="Build a generic workflow agent.",
        problem_statement=(
            "Build an agent that uses tools, approvals, and traceability to complete bounded workflows."
        ),
    )

    validation = validate_task_agent_spec(spec)

    assert validation.valid
    assert spec.project_contract.family.value == "workflow_agent_service"
    assert all(
        phrase not in str(check.model_dump(mode="json")).lower()
        for deliverable in spec.deliverables
        for check in deliverable.public_checks
        for phrase in ("routine case", "ambiguous or risky case")
    )
    assert any(endpoint.path.startswith("/workflows") for endpoint in spec.public_endpoints)
