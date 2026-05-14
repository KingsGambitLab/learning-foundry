"""Pin: `_write_protocol_files` must NOT overwrite per-deliverable
`deliverable.json` files. The manifest is owned by the materializer
(initial creation) and `_apply_progressive_bundle` (authored
metadata); `_write_protocol_files`'s legitimate job is the harness
runtime scripts only (`run_visible_checks.py`, `run_hidden_checks.py`).

Observed today across course_07b665e9774f (TypeScript), Rails, and Go:
every `authoring_runtime` call invoked `author_workspace`, which:

    1. ensure_workspace(run)            # ok, idempotent
    2. _write_protocol_files(run)       # ← OVERWROTE deliverable.json with the
                                        #   DEFAULT TEMPLATE (starter_repo_bundle.source
                                        #   = "starter_default")
    3. author_workspace_repo(run)       # _apply_progressive_bundle wrote
                                        #   starter_repo_bundle.source = "openai_live"

Between steps 2 and 3, all manifests sat at `starter_default`. If
step 3 partially failed (OpenAI hiccup, model omitted a deliverable),
the manifest stayed `starter_default` permanently, which made
`reviewer_code` / `reviewer_tests` emit
`starter_repo_bundle_not_authored` and trigger an infinite repair
loop. This was the root cause behind:

  * Go d2 stale-manifest divergence (debugged earlier today)
  * Rails reviewer_tests stuck loop
  * TypeScript reviewer_code failure on first sandbox-clean attempt

Fix: delete the `deliverable.json` write from `_write_protocol_files`.
The materializer creates the initial file; subsequent authoring
updates it. `_write_protocol_files` no longer destroys authored state.

These tests pin:
  1. `_write_protocol_files` leaves an already-authored `deliverable.json`
     untouched (specifically: starter_repo_bundle.source stays openai_live).
  2. `_write_protocol_files` still writes the runtime scripts
     (run_visible_checks.py, run_hidden_checks.py) — those are
     legitimately part of the harness protocol.
"""

from __future__ import annotations

import pytest
pytest.skip(
    "Pre-existing test depends on the removed SQLiteWorkflowStore. "
    "Pending follow-up to port to PostgresWorkflowStore.",
    allow_module_level=True,
)

import json
import tempfile
import unittest
from pathlib import Path

from app.domain.registry import PackageType, StarterType
from app.domain.workflow import MaterializeBundleRequest
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import (
    GenerationIntake,
    infer_assignment_design,
)
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.task_agent_workspace_authoring import (
    TaskAgentWorkspaceAuthoringService,
)
from app.services.workflow_service import WorkflowService


def _materialized_run(temp_dir: str, workspace_manager: AssignmentWorkspaceManager):
    store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
    )
    intake = GenerationIntake(
        title="Test Shared Codebase Service",
        problem_statement=(
            "Build a shared-codebase service with PostgreSQL. Learners "
            "progress through deliverables that add endpoints, "
            "idempotent writes, and caching."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    inferred = infer_assignment_design(
        title=intake.title, problem_statement=intake.problem_statement,
        package_type_hint=intake.package_type_hint,
    )
    assert inferred.design_spec is not None
    inferred.design_spec.runtime_dependencies.starter_type = StarterType.partial
    inferred.design_spec.runtime_dependencies.editable_files = ["app.py", "models.py"]
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake, design_spec=inferred.design_spec, execute_nodes=False,
    )
    workflow_service.materialize_run(run.id, MaterializeBundleRequest(overwrite=True))
    run = workflow_service.get_run(run.id)
    assert run is not None
    setup_service = TaskAgentWorkspaceAuthoringService(workspace_manager=workspace_manager)
    run, _ = setup_service.author_workspace(run)
    return run


def _stamp_manifests_openai_live(run) -> None:
    """Simulate a prior `_apply_progressive_bundle` call that stamped
    every per-deliverable manifest with `starter_repo_bundle.source =
    "openai_live"`. After this, the manifests on disk represent
    successfully-authored state.
    """
    workspace = run.artifacts.workspace_snapshot
    spec = run.artifacts.task_agent_spec
    workspace_root = Path(workspace.root_dir)
    for deliverable in spec.deliverables:
        mp = workspace_root / "private" / "grader" / deliverable.id / "deliverable.json"
        if not mp.exists():
            continue
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        manifest["starter_repo_bundle"] = {
            "generated_for_deliverable": deliverable.id,
            "source": "openai_live",
            "authored_paths": ["app.py", "models.py"],
        }
        manifest["runtime_protocol_bundle"] = {
            "generated_for_deliverable": deliverable.id,
            "source": "openai_live",
            "authored_paths": ["Dockerfile", ".coursegen/runtime/install.sh"],
        }
        mp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class WriteProtocolFilesPreservesManifestsTests(unittest.TestCase):
    def test_write_protocol_files_does_not_revert_authored_deliverable_manifest(self) -> None:
        """The core invariant: an already-authored deliverable.json
        (starter_repo_bundle.source == "openai_live") survives a call
        to `_write_protocol_files`.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_manager = AssignmentWorkspaceManager(base_dir=f"{temp_dir}/workspaces")
            run = _materialized_run(temp_dir, workspace_manager)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot

            _stamp_manifests_openai_live(run)

            service = TaskAgentWorkspaceAuthoringService(workspace_manager=workspace_manager)
            # Force-rewrite protocol files (mimics what author_workspace
            # and repair_workspace do).
            service._write_protocol_files(run)

            for deliverable in spec.deliverables:
                mp = Path(workspace.root_dir) / "private" / "grader" / deliverable.id / "deliverable.json"
                manifest = json.loads(mp.read_text(encoding="utf-8"))
                self.assertEqual(
                    (manifest.get("starter_repo_bundle") or {}).get("source"),
                    "openai_live",
                    f"{deliverable.id} regressed from openai_live to "
                    f"{(manifest.get('starter_repo_bundle') or {}).get('source')!r} "
                    f"after _write_protocol_files. This is the bug behind the "
                    f"Go/Rails/TS reviewer_code/reviewer_tests loops.",
                )

    def test_write_protocol_files_force_true_still_preserves_authored_manifest(self) -> None:
        """Same invariant with force=True (the repair_workspace path).
        `force=True` historically meant "overwrite even if content
        unchanged" — but it must NOT overwrite the authored manifest
        with the default template.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_manager = AssignmentWorkspaceManager(base_dir=f"{temp_dir}/workspaces")
            run = _materialized_run(temp_dir, workspace_manager)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot

            _stamp_manifests_openai_live(run)

            service = TaskAgentWorkspaceAuthoringService(workspace_manager=workspace_manager)
            service._write_protocol_files(
                run,
                deliverable_ids=[d.id for d in spec.deliverables],
                force=True,
            )

            for deliverable in spec.deliverables:
                mp = Path(workspace.root_dir) / "private" / "grader" / deliverable.id / "deliverable.json"
                manifest = json.loads(mp.read_text(encoding="utf-8"))
                self.assertEqual(
                    (manifest.get("starter_repo_bundle") or {}).get("source"),
                    "openai_live",
                    f"force=True must NOT destroy authored manifest. "
                    f"{deliverable.id} regressed to "
                    f"{(manifest.get('starter_repo_bundle') or {}).get('source')!r}.",
                )

    def test_write_protocol_files_still_writes_runtime_scripts(self) -> None:
        """Regression guard: deleting the deliverable.json write must
        NOT also delete the run_visible_checks.py / run_hidden_checks.py
        writes. Those are legitimately part of the harness protocol
        and `_write_protocol_files` should keep producing them.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_manager = AssignmentWorkspaceManager(base_dir=f"{temp_dir}/workspaces")
            run = _materialized_run(temp_dir, workspace_manager)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot

            # Delete a known runtime script to confirm _write_protocol_files
            # restores it.
            for deliverable in spec.deliverables:
                visible_check = (
                    Path(workspace.public_dir) / "checks" / deliverable.id / "run_visible_checks.py"
                )
                if visible_check.exists():
                    visible_check.unlink()

            service = TaskAgentWorkspaceAuthoringService(workspace_manager=workspace_manager)
            service._write_protocol_files(
                run,
                deliverable_ids=[d.id for d in spec.deliverables],
                force=True,
            )

            # Runtime scripts must have been written.
            for deliverable in spec.deliverables:
                visible_check = (
                    Path(workspace.public_dir) / "checks" / deliverable.id / "run_visible_checks.py"
                )
                self.assertTrue(
                    visible_check.exists(),
                    f"_write_protocol_files must still write the visible-check "
                    f"runtime script for {deliverable.id}.",
                )


if __name__ == "__main__":
    unittest.main()
