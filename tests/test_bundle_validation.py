from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.domain.task_agent import EndpointSpec
from app.domain.workflow import MaterializeBundleRequest
from app.domain.registry import PackageType
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.bundle_validation import (
    inspect_materialized_starter_surface,
    validate_materialized_bundle,
    validate_seeded_learner_workspace,
)
from app.services.learner_brief_builder import (
    build_task_agent_deliverable_brief,
    ensure_task_agent_deliverable_briefs,
    render_learner_starter_readme,
)
from app.services.spec_validation import validate_task_agent_spec
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.services.task_agent_starter_templates import (
    HIDDEN_MANIFEST_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
)
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _inventory_design():
    inferred = infer_assignment_design(
        title="Inventory Reservation Service",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    assert inferred.design_spec is not None
    return inferred.design_spec


def test_transactional_scaffold_is_family_specific_not_agentic() -> None:
    design_spec = _inventory_design()

    spec, origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )

    assert origin_template == "transactional_stateful_service"
    assert spec.project_contract.family.value == "transactional_stateful_service"
    assert spec.project_contract.core_entities == ["inventory reservation"]
    assert any(endpoint.path.startswith("/inventory-reservations") for endpoint in spec.public_endpoints)
    assert all("tool" not in deliverable.title.lower() for deliverable in spec.deliverables)
    assert all("approval" not in deliverable.title.lower() for deliverable in spec.deliverables[:3])
    assert all(
        phrase not in str(check.model_dump(mode="json")).lower()
        for deliverable in spec.deliverables
        for check in deliverable.public_checks
        for phrase in ("routine case", "ambiguous or risky case")
    )


def test_materialized_bundle_readme_stays_grounded_in_backend_contract() -> None:
    design_spec = _inventory_design()
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
        )
        run = workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
            ),
            design_spec=design_spec,
            execute_nodes=False,
        )

        materialized = workflow_service.materialize_run(
            run.id,
            MaterializeBundleRequest(overwrite=True),
        )
        assert materialized.artifacts.materialized_bundle is not None
        bundle = materialized.artifacts.materialized_bundle
        result = validate_materialized_bundle(materialized.artifacts.task_agent_spec, bundle)
        assert result.valid

        readme = workflow_service.read_bundle_file(run.id, "public/README.md").content
        assert "agentic system" not in readme.lower()
        assert "tool-use policies" not in readme.lower()
        assert "Inventory Reservation service" in readme
        assert "/inventory-reservations" in readme


def test_validation_rejects_title_slug_public_endpoints() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Build a concurrency-safe multi-warehouse inventory reservation service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )
    spec.public_endpoints[1].path = "/build-a-concurrency-safe-multi-warehouse-inventory-reservation-service"

    result = validate_task_agent_spec(spec)
    assert not result.valid
    assert any(issue.code == "title_slug_public_endpoint" for issue in result.errors)


def test_validation_rejects_generic_deliverable_titles_when_entities_are_known() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )
    spec.deliverables[0].title = "Service contract and durable model"

    result = validate_task_agent_spec(spec)
    assert not result.valid
    assert any(issue.code == "generic_deliverable_title" for issue in result.errors)


def test_bundle_validation_flags_overstated_agentic_readme() -> None:
    design_spec = _inventory_design()
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
        )
        run = workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
            ),
            design_spec=design_spec,
            execute_nodes=False,
        )
        bundle = workflow_service.materializer.materialize_run(run, overwrite=True)
        readme_path = bundle.root_dir + "/public/README.md"
        with open(readme_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# Inventory Reservation Service\n\n"
                "A bounded, production-ready agentic system with stable APIs, tool-use policies, traces, approvals, and evaluation hooks.\n"
            )

        result = validate_materialized_bundle(run.artifacts.task_agent_spec, bundle)
        assert not result.valid
        assert any(issue.code == "course_readme_overstates_workflow_surface" for issue in result.errors)


def test_bundle_validation_flags_secondary_brief_reference_in_starter_readme() -> None:
    design_spec = _inventory_design()
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
        )
        run = workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
            ),
            design_spec=design_spec,
            execute_nodes=False,
        )
        bundle = workflow_service.materializer.materialize_run(run, overwrite=True)
        starter_readme_path = Path(bundle.root_dir) / "public" / "starter" / run.artifacts.task_agent_spec.deliverables[0].id / "README.md"
        starter_readme_path.write_text(
            starter_readme_path.read_text(encoding="utf-8") + "\nSee `deliverable_content.md` for more detail.\n",
            encoding="utf-8",
        )

        result = validate_materialized_bundle(run.artifacts.task_agent_spec, bundle)
        assert not result.valid
        assert any(issue.code == "starter_readme_uses_secondary_brief" for issue in result.errors)


def test_bundle_validation_flags_unpublished_endpoint_reference_in_starter_readme() -> None:
    design_spec = _inventory_design()
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
        )
        run = workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
            ),
            design_spec=design_spec,
            execute_nodes=False,
        )
        bundle = workflow_service.materializer.materialize_run(run, overwrite=True)
        starter_readme_path = Path(bundle.root_dir) / "public" / "starter" / run.artifacts.task_agent_spec.deliverables[0].id / "README.md"
        starter_readme_path.write_text(
            starter_readme_path.read_text(encoding="utf-8")
            + "\nKeep `POST /and-resolutions` stable while you work.\n",
            encoding="utf-8",
        )

        result = validate_materialized_bundle(run.artifacts.task_agent_spec, bundle)
        assert not result.valid
        assert any(issue.code == "starter_readme_unpublished_endpoint_reference" for issue in result.errors)


def test_bundle_validation_flags_generic_starter_readme_without_domain_entities() -> None:
    design_spec = _inventory_design()
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
        )
        run = workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
            ),
            design_spec=design_spec,
            execute_nodes=False,
        )
        bundle = workflow_service.materializer.materialize_run(run, overwrite=True)
        starter_readme_path = Path(bundle.root_dir) / "public" / "starter" / run.artifacts.task_agent_spec.deliverables[0].id / "README.md"
        starter_readme_path.write_text(
            "# Starter\n\n"
            "Serve the current state safely under load.\n\n"
            "## What to build\n\n"
            "Keep the service surface stable while you improve the behavior behind it.\n\n"
            "## Files to edit\n\n- `app.py`\n\n"
            "## Definition of done\n\n- Keep the service surface stable.\n\n"
            "## Helpful commands\n\n- Preview: `sh .coursegen/runtime/run.sh`\n",
            encoding="utf-8",
        )

        result = validate_materialized_bundle(run.artifacts.task_agent_spec, bundle)
        assert not result.valid
        assert any(issue.code == "starter_readme_lacks_domain_grounding" for issue in result.errors)


def test_bundle_validation_flags_runtime_protocol_bundle_marked_authored_when_placeholders_remain() -> None:
    design_spec = _inventory_design()
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
        )
        run = workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
            ),
            design_spec=design_spec,
            execute_nodes=False,
        )
        bundle = workflow_service.materializer.materialize_run(run, overwrite=True)
        starter_root = Path(bundle.root_dir) / "public" / "starter" / run.artifacts.task_agent_spec.deliverables[0].id
        manifest_path = starter_root / HIDDEN_MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_protocol_bundle"] = {
            "source": "openai_live",
            "generated_for_deliverable": run.artifacts.task_agent_spec.deliverables[0].id,
            "authored_paths": [RUNTIME_INSTALL_SCRIPT_PATH],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (starter_root / RUNTIME_INSTALL_SCRIPT_PATH).write_text(
            "#!/usr/bin/env sh\nset -eu\necho runtime install\n",
            encoding="utf-8",
        )

        result = validate_materialized_bundle(run.artifacts.task_agent_spec, bundle)
        assert not result.valid
        assert any(issue.code == "runtime_protocol_bundle_incomplete" for issue in result.errors)


def test_materialized_starter_surface_stays_honest_for_transactional_bundle() -> None:
    design_spec = _inventory_design()
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
        )
        run = workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
            ),
            design_spec=design_spec,
            execute_nodes=False,
        )
        bundle = workflow_service.materializer.materialize_run(run, overwrite=True)

        result = inspect_materialized_starter_surface(run.artifacts.task_agent_spec, bundle)
        assert result.valid


def test_ensure_task_agent_deliverable_briefs_normalizes_stale_required_endpoints() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )
    deliverable = spec.deliverables[0]
    assert deliverable.learner_starter_surface is not None
    deliverable.learner_starter_surface.required_endpoints = [
        EndpointSpec(method="POST", path="/and-resolutions"),
    ]

    normalized = ensure_task_agent_deliverable_briefs(spec, overwrite=True)
    rendered_brief = normalized.deliverables[0].learner_brief

    assert normalized.deliverables[0].learner_starter_surface is not None
    assert all(
        endpoint.path != "/and-resolutions"
        for endpoint in normalized.deliverables[0].learner_starter_surface.required_endpoints
    )
    assert any(
        endpoint.path.startswith("/inventory-reservations")
        for endpoint in normalized.deliverables[0].learner_starter_surface.required_endpoints
    )
    assert rendered_brief is not None
    assert "/and-resolutions" not in rendered_brief.task_to_build
    assert "/inventory-reservations" in rendered_brief.task_to_build


def test_ensure_task_agent_deliverable_briefs_falls_back_when_only_health_survives() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )
    deliverable = spec.deliverables[0]
    assert deliverable.learner_starter_surface is not None
    deliverable.learner_starter_surface.required_endpoints = [
        EndpointSpec(method="GET", path="/health"),
        EndpointSpec(method="POST", path="/and-resolutions"),
    ]

    normalized = ensure_task_agent_deliverable_briefs(spec, overwrite=True)
    rendered_brief = normalized.deliverables[0].learner_brief

    assert normalized.deliverables[0].learner_starter_surface is not None
    assert any(
        endpoint.path.startswith("/inventory-reservations")
        for endpoint in normalized.deliverables[0].learner_starter_surface.required_endpoints
    )
    assert rendered_brief is not None
    assert "GET /health" not in rendered_brief.task_to_build
    assert "/inventory-reservations" in rendered_brief.task_to_build


def test_validation_rejects_stale_required_endpoint_not_in_public_surface() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )
    deliverable = spec.deliverables[0]
    assert deliverable.learner_starter_surface is not None
    deliverable.learner_starter_surface.required_endpoints = [
        EndpointSpec(method="POST", path="/and-resolutions"),
    ]

    result = validate_task_agent_spec(spec)
    assert not result.valid
    assert any(issue.code == "starter_required_endpoint_not_published" for issue in result.errors)


def test_seeded_workspace_validation_rejects_secondary_brief_duplication() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )
    deliverable = spec.deliverables[0]
    brief = deliverable.learner_brief or build_task_agent_deliverable_brief(spec, deliverable)

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_root = Path(temp_dir)
        workspace_root.joinpath(".coursegen/review_areas/exercise-1").mkdir(parents=True)
        workspace_root.joinpath("README.md").write_text("# Inventory Reservation Service\n", encoding="utf-8")
        workspace_root.joinpath("project_brief.md").write_text("# Inventory Reservation Service\n", encoding="utf-8")
        workspace_root.joinpath("deliverables.md").write_text("# Project deliverables\n", encoding="utf-8")
        workspace_root.joinpath(".coursegen/review_areas/exercise-1/README.md").write_text(
            render_learner_starter_readme(
                title=deliverable.title,
                summary=deliverable.objective,
                learning_outcomes=list(deliverable.learning_outcomes),
                brief=brief,
            ),
            encoding="utf-8",
        )
        workspace_root.joinpath(".coursegen/review_areas/exercise-1/deliverable_content.md").write_text(
            "# Duplicate brief\n",
            encoding="utf-8",
        )
        assert deliverable.learner_starter_surface is not None
        primary_editable = deliverable.learner_starter_surface.primary_editable_paths[0]
        workspace_root.joinpath(primary_editable).write_text(
            "# learner-owned file\n",
            encoding="utf-8",
        )

        result = validate_seeded_learner_workspace(
            spec,
            workspace_root,
            deliverable_ids=["exercise-1"],
        )
        assert not result.valid
        assert any(issue.code == "deprecated_secondary_brief_present" for issue in result.errors)
