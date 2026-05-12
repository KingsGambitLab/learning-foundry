"""Pin shared-codebase starter_repo_bundle metadata to be IDENTICAL
across every deliverable's manifest.

Shared-codebase courses keep ONE shared starter root at
`public/starter/` describing files used by every deliverable. The
per-deliverable manifest at `private/grader/<id>/deliverable.json`
carries `starter_repo_bundle` metadata (`source`, `authored_paths`)
that describes the shared bundle.

Because the bundle is shared, the metadata MUST be consistent across
deliverables: every deliverable points to the same `public/starter/`
contents, so `source` and `authored_paths` should be identical.

Today's bug (observed on course_72e0739fc3ab Go run): the loop in
`_apply_progressive_bundle` recomputes `_bundle_state` per-deliverable
using the per-deliverable manifest. Subtle variations in the
per-deliverable manifest (e.g. empty `dependency_contract.manifest_paths`)
combined with intermediate file-system states across repair cycles
caused deliverable_2 to record `source="starter_default"` while
deliverable_1/3/4/5 recorded `source="openai_live"` — even though the
disk content was identical.

Downstream, `reviewer_tests` reads the per-deliverable manifest and
emits `starter_repo_bundle_not_authored` for deliverable_2, while
ignoring that the other 4 deliverables (with the same shared files)
record the bundle as authored. The mismatch triggers an infinite
reviewer_tests → reviewer_repair loop because `_repair_generated_tests`
only re-authors test scripts; it cannot fix the bundle metadata.

This test pins the fix: for shared codebase, `_bundle_state` is
computed ONCE (against the first deliverable's manifest, which is the
canonical reference for the shared bundle), and the SAME metadata is
written to every per-deliverable manifest.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.domain.registry import PackageType, StarterType
from app.domain.workflow import MaterializeBundleRequest
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import (
    GenerationIntake,
    infer_assignment_design,
)
from app.services.openai_repo_authoring import (
    OpenAIStarterRepoAuthoringService,
    _GeneratedDependencyContract,
    _GeneratedSharedRepoBundle,
    _RepoFile,
)
from app.services.task_agent_workspace_authoring import (
    TaskAgentWorkspaceAuthoringService,
)
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _materialized_shared_run(temp_dir: str):
    store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
    )
    intake = GenerationIntake(
        title="Production URL Shortener Service in Go",
        problem_statement=(
            "Build a production-ready URL shortener service in Go using "
            "Gin and PostgreSQL. Learners progress through deliverables "
            "that add CRUD endpoints, idempotent writes, redis-backed "
            "analytics caching, and rate-limited public read paths."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    inferred = infer_assignment_design(
        title=intake.title,
        problem_statement=intake.problem_statement,
        package_type_hint=intake.package_type_hint,
    )
    assert inferred.design_spec is not None
    inferred.design_spec.runtime_dependencies.editable_files = [
        "go.mod",
        "cmd/server/main.go",
    ]
    inferred.design_spec.runtime_dependencies.starter_type = StarterType.partial
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake,
        design_spec=inferred.design_spec,
        execute_nodes=False,
    )
    workflow_service.materialize_run(
        run.id, MaterializeBundleRequest(overwrite=True)
    )
    run = workflow_service.get_run(run.id)
    assert run is not None
    run, _ = TaskAgentWorkspaceAuthoringService().author_workspace(run)
    return run


def _make_minimal_bundle() -> _GeneratedSharedRepoBundle:
    """Author a minimal but plausible shared Go bundle."""
    return _GeneratedSharedRepoBundle(
        runtime_protocol_files=[
            _RepoFile(
                path="Dockerfile",
                content="FROM golang:1.23-bookworm\nWORKDIR /app\n",
            ),
            _RepoFile(
                path=".coursegen/runtime/install.sh",
                content="#!/usr/bin/env bash\nset -euo pipefail\ngo mod download\n",
            ),
            _RepoFile(
                path=".coursegen/runtime/verify.sh",
                content="#!/usr/bin/env bash\nset -euo pipefail\ngo build ./...\n",
            ),
            _RepoFile(
                path=".coursegen/runtime/run.sh",
                content="#!/usr/bin/env bash\nset -euo pipefail\nexec ./server\n",
            ),
        ],
        files=[
            _RepoFile(
                path="go.mod",
                content="module starter\n\ngo 1.23\n",
            ),
            _RepoFile(
                path="cmd/server/main.go",
                content=(
                    "package main\n\n"
                    "import \"net/http\"\n\n"
                    "func main() {\n"
                    "    http.HandleFunc(\"/health\", func(w http.ResponseWriter, r *http.Request) {\n"
                    "        w.WriteHeader(http.StatusOK)\n"
                    "    })\n"
                    "    http.ListenAndServe(\":8000\", nil)\n"
                    "}\n"
                ),
            ),
        ],
        dependency_contract=_GeneratedDependencyContract(
            manifest_paths=["go.mod"],
            lockfile_paths=["go.sum"],
            toolchain_paths=[],
            build_support_paths=[],
            reproducibility_mode="reproducible_install",
        ),
        notes=[],
    )


class SharedCodebaseBundleMetadataConsistencyTests(unittest.TestCase):
    def test_apply_progressive_bundle_writes_identical_metadata_to_every_deliverable(self) -> None:
        """For shared codebase, every deliverable's manifest must have
        the SAME `starter_repo_bundle.source` and the SAME
        `starter_repo_bundle.authored_paths`. The bundle is shared
        — the metadata must agree on that.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_shared_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)
            self.assertTrue(spec.course_structure.shared_codebase)
            self.assertGreaterEqual(
                len(spec.deliverables),
                2,
                "Need at least two deliverables to exhibit the bug.",
            )

            service = OpenAIStarterRepoAuthoringService.__new__(
                OpenAIStarterRepoAuthoringService
            )
            bundle = _make_minimal_bundle()

            public_root = Path(workspace.public_dir)
            workspace_root = Path(workspace.root_dir)

            updated, notes = service._apply_progressive_bundle(
                run=run,
                public_root=public_root,
                workspace_root=workspace_root,
                visible_fixture_files=set(),
                deliverable_ids=[spec.deliverables[0].id],
                bundle=bundle,
            )

            sources: set[str] = set()
            authored_paths_per_deliverable: dict[str, list[str]] = {}
            for deliverable in spec.deliverables:
                manifest_path = (
                    workspace_root
                    / "private"
                    / "grader"
                    / deliverable.id
                    / "deliverable.json"
                )
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                srb = manifest.get("starter_repo_bundle") or {}
                sources.add(str(srb.get("source") or ""))
                authored_paths_per_deliverable[deliverable.id] = sorted(
                    srb.get("authored_paths") or []
                )

            self.assertEqual(
                len(sources),
                1,
                f"All deliverables in a shared-codebase course must record "
                f"the SAME starter_repo_bundle.source; got: {sources}",
            )
            self.assertEqual(
                sources.pop(),
                "openai_live",
                "After applying a fresh model bundle, every deliverable's "
                "manifest must record `source=openai_live` — not "
                "`starter_default`.",
            )

            # authored_paths must also be identical across deliverables.
            distinct_path_sets = {
                tuple(paths)
                for paths in authored_paths_per_deliverable.values()
            }
            self.assertEqual(
                len(distinct_path_sets),
                1,
                f"All deliverables must record the SAME authored_paths "
                f"for the shared bundle; got per-deliverable variation: "
                f"{authored_paths_per_deliverable}",
            )


    def test_targeted_repair_does_not_diverge_metadata_across_deliverables(self) -> None:
        """The Go bug reproduction: a SECOND call to
        `_apply_progressive_bundle` with a partial (targeted) bundle
        must not leave some deliverables marked as `openai_live` while
        others regress to `starter_default`. Either every manifest
        records the new bundle, or every manifest preserves the prior
        bundle — never half-and-half.

        Simulates the repair flow where reviewer_repair calls
        author_workspace_repo with only the failed deliverable's id.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_shared_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertGreaterEqual(len(spec.deliverables), 2)

            service = OpenAIStarterRepoAuthoringService.__new__(
                OpenAIStarterRepoAuthoringService
            )
            public_root = Path(workspace.public_dir)
            workspace_root = Path(workspace.root_dir)

            # Pass 1: full bundle, all deliverables.
            service._apply_progressive_bundle(
                run=run,
                public_root=public_root,
                workspace_root=workspace_root,
                visible_fixture_files=set(),
                deliverable_ids=[d.id for d in spec.deliverables],
                bundle=_make_minimal_bundle(),
            )

            # Pass 2: targeted repair on just the SECOND deliverable.
            # The bundle is still the same shared bundle (files don't
            # change shape for shared codebase), but the request is
            # scoped to only d2's id.
            second_deliverable_id = spec.deliverables[1].id
            service._apply_progressive_bundle(
                run=run,
                public_root=public_root,
                workspace_root=workspace_root,
                visible_fixture_files=set(),
                deliverable_ids=[second_deliverable_id],
                bundle=_make_minimal_bundle(),
            )

            # All deliverables must STILL have identical metadata.
            sources: set[str] = set()
            for deliverable in spec.deliverables:
                manifest_path = (
                    workspace_root
                    / "private"
                    / "grader"
                    / deliverable.id
                    / "deliverable.json"
                )
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                srb = manifest.get("starter_repo_bundle") or {}
                sources.add(str(srb.get("source") or ""))

            self.assertEqual(
                len(sources),
                1,
                f"After a targeted repair, deliverable metadata must NOT "
                f"diverge across shared-codebase deliverables; got "
                f"sources: {sources}",
            )
            self.assertEqual(
                sources.pop(),
                "openai_live",
                "A passing repair pass must record `openai_live` for "
                "every deliverable.",
            )


if __name__ == "__main__":
    unittest.main()
