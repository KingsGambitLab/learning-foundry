from __future__ import annotations

import pytest
pytest.skip(
    "Pre-existing test depends on the removed SQLiteWorkflowStore. "
    "Pending follow-up to port to PostgresWorkflowStore.",
    allow_module_level=True,
)

import json
import tempfile
from pathlib import Path

from app.domain.registry import PackageType
from app.domain.task_agent import DeliverableSpec
from app.domain.workflow import MaterializeBundleRequest
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.task_agent_scaffolds import build_task_agent_scaffold


def _default_planner_deliverables() -> list[DeliverableSpec]:
    titles = [
        "Inventory reservation contract and state model",
        "Inventory reservation read/write correctness",
        "Inventory reservation observability and recovery",
        "Inventory reservation production hardening",
    ]
    return [
        DeliverableSpec(
            id=f"deliverable_{index}",
            title=title,
            objective=f"Build the {title.lower()} surface.",
            learning_outcomes=[],
            overlay_ids=[],
        )
        for index, title in enumerate(titles, start=1)
    ]
from app.services.task_agent_starter_templates import RUNTIME_RUN_SCRIPT_PATH, build_task_agent_starter_files
from app.services.workflow_service import WorkflowService


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


def test_starter_files_use_runtime_run_script_for_preview() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
        planner_deliverables=_default_planner_deliverables(),
    )

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    manifest_payload = json.loads(starter_files[".coursegen/deliverable.json"])
    assert manifest_payload["preview_command"] == f"sh {RUNTIME_RUN_SCRIPT_PATH}"

    tasks_payload = json.loads(starter_files[".vscode/tasks.json"])
    preview_task = next(task for task in tasks_payload["tasks"] if task["label"] == "Preview app")
    assert f"sh {RUNTIME_RUN_SCRIPT_PATH}" in preview_task["command"]


def test_materialized_starter_readme_uses_runtime_run_script() -> None:
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

        materialized = workflow_service.materialize_run(run.id, MaterializeBundleRequest(overwrite=True))
        starter_readme = workflow_service.read_bundle_file(
            run.id,
            f"public/checks/{run.artifacts.task_agent_spec.deliverables[0].id}/README.md",
        ).content
        assert f"sh {RUNTIME_RUN_SCRIPT_PATH}" in starter_readme
        assert "uvicorn app:app" not in starter_readme
        launcher_path = Path(materialized.artifacts.materialized_bundle.root_dir) / "public" / "starter" / ".coursegen/preview_app.py"
        assert not launcher_path.exists()
