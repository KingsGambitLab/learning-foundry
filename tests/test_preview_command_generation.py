from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.domain.registry import PackageType
from app.domain.workflow import MaterializeBundleRequest
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.services.task_agent_starter_templates import PREVIEW_LAUNCHER_PATH, build_task_agent_starter_files
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


def test_python_starter_files_use_local_preview_launcher() -> None:
    design_spec = _inventory_design()
    spec, _origin_template = build_task_agent_scaffold(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        design_spec=design_spec,
    )

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    assert PREVIEW_LAUNCHER_PATH in starter_files
    manifest_payload = json.loads(starter_files[".coursegen/deliverable.json"])
    assert manifest_payload["preview_command"] == f"python {PREVIEW_LAUNCHER_PATH} --host 0.0.0.0"

    tasks_payload = json.loads(starter_files[".vscode/tasks.json"])
    preview_task = next(task for task in tasks_payload["tasks"] if task["label"] == "Preview app")
    assert PREVIEW_LAUNCHER_PATH in preview_task["command"]


def test_materialized_starter_readme_uses_local_preview_launcher() -> None:
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
            f"public/starter/{run.artifacts.task_agent_spec.deliverables[0].id}/README.md",
        ).content
        assert f"python {PREVIEW_LAUNCHER_PATH}" in starter_readme
        assert "uvicorn app:app" not in starter_readme

        launcher_path = Path(materialized.artifacts.materialized_bundle.root_dir) / "public" / "starter" / run.artifacts.task_agent_spec.deliverables[0].id / PREVIEW_LAUNCHER_PATH
        assert launcher_path.exists()
