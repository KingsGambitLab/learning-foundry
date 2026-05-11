"""Pin the workspace-seed-files logic for the shared-codebase layout.

The seed-files builder was looking under `public/starter/<deliverable_id>/`
— the legacy non-shared path that doesn't exist for shared-codebase
courses post-Pass-2. Result: learner workspaces seeded with zero source
code, just documentation files.

These tests pin the new behavior:
  - For shared-codebase, the first deliverable's seed_files include all
    shared starter content (under `public/starter/`) and all per-
    deliverable visible-check scripts (under `public/checks/<id>/`).
  - For shared-codebase, non-first deliverables return empty seed_files
    (since `_workspace_seed_source_files` for shared courses takes ONLY
    the first deliverable's; duplicating across N deliverables would
    bloat the snapshot).
  - Legacy non-shared layout still uses the per-deliverable starter
    folder.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.domain.publish import LearnerPackageFile
from app.domain.registry import PackageType, StarterType
from app.domain.task_agent import (
    AssessmentStrategySpec,
    CapabilitySpec,
    CourseStructureSpec,
    DeliverableSpec,
    ExecutionSurface,
    ProgressionMode,
    RuntimeDependencySpec,
    TaskAgentServiceSpec,
    WorkspaceScope,
)
from app.domain.workflow import (
    ArtifactVisibility,
    BundleFile,
    BundleFileContent,
    MaterializedBundle,
)
from app.services.publish_snapshot_service import PublishSnapshotService


def _make_spec(*, shared: bool, deliverable_ids: list[str]) -> TaskAgentServiceSpec:
    return TaskAgentServiceSpec(
        title="t",
        summary="s",
        package_type=PackageType.progressive_codebase_course
        if shared
        else PackageType.survey_course,
        course_structure=CourseStructureSpec(
            package_type=PackageType.progressive_codebase_course
            if shared
            else PackageType.survey_course,
            workspace_scope=WorkspaceScope.shared_course_workspace
            if shared
            else WorkspaceScope.per_deliverable_workspace,
            progression_mode=ProgressionMode.independent_deliverables,
            shared_codebase=shared,
        ),
        runtime_dependencies=RuntimeDependencySpec(
            execution_surface=ExecutionSurface.http_service,
            starter_type=StarterType.partial,
        ),
        capabilities=CapabilitySpec(),
        assessment_strategy=AssessmentStrategySpec(),
        deliverables=[
            DeliverableSpec(id=did, title=did, objective=did)
            for did in deliverable_ids
        ],
    )


def _bundle_files(paths_and_vis: list[tuple[str, str]]) -> list[BundleFile]:
    return [
        BundleFile(
            relative_path=p,
            visibility=ArtifactVisibility(vis),
            media_type="text/plain",
            size_bytes=len(p),
        )
        for p, vis in paths_and_vis
    ]


def _make_run(files: list[BundleFile]) -> MagicMock:
    """Minimal workflow_run mock — only the fields _workspace_seed_files reads."""
    bundle = MaterializedBundle(
        bundle_id="b1",
        generated_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        root_dir="/tmp",
        public_dir="/tmp/public",
        private_dir="/tmp/private",
        manifest_path="/tmp/manifest.json",
        files=files,
    )
    artifacts = MagicMock()
    artifacts.materialized_bundle = bundle
    run = MagicMock()
    run.artifacts = artifacts
    run.id = "run_test"
    return run


def _stub_service() -> PublishSnapshotService:
    """Build the service with a workflow_service that just echoes file paths."""
    service = PublishSnapshotService.__new__(PublishSnapshotService)
    workflow_service = MagicMock()

    def _read_bundle_file(workflow_run_id, relative_path):
        return BundleFileContent(
            relative_path=relative_path,
            media_type="text/plain",
            content=f"# content of {relative_path}",
        )

    workflow_service.read_bundle_file.side_effect = _read_bundle_file
    service.workflow_service = workflow_service
    service.store = MagicMock()
    return service


class SharedCodebaseSeedFilesTests(unittest.TestCase):
    def test_first_deliverable_collects_shared_starter_and_all_visible_checks(self) -> None:
        spec = _make_spec(shared=True, deliverable_ids=["d1", "d2", "d3"])
        files = _bundle_files([
            # Shared starter content (no deliverable folder).
            ("public/starter/Dockerfile", "public"),
            ("public/starter/src/main/java/App.java", "public"),
            ("public/starter/pom.xml", "public"),
            ("public/starter/.coursegen/runtime/install.sh", "public"),
            ("public/starter/README.md", "public"),  # should be excluded
            # Per-deliverable visible scripts.
            ("public/checks/d1/run_visible_checks.py", "public"),
            ("public/checks/d2/run_visible_checks.py", "public"),
            ("public/checks/d3/run_visible_checks.py", "public"),
            # Per-deliverable README (excluded — seeded via deliverable.starter_readme).
            ("public/checks/d1/README.md", "public"),
            ("public/checks/d2/README.md", "public"),
            # Private artifacts — should be excluded.
            ("private/grader/d1/run_hidden_checks.py", "private"),
            ("private/grader/d1/deliverable.json", "private"),
        ])
        run = _make_run(files)
        service = _stub_service()

        seed = service._workspace_seed_files(
            spec=spec,
            workflow_run=run,
            workflow_run_id="run_test",
            spec_deliverable_id="d1",  # first deliverable
            content_markdown="",
            starter_readme="",
        )
        paths = {f.relative_path for f in seed}

        # Shared starter content lands at workspace root (prefix stripped).
        self.assertIn("Dockerfile", paths)
        self.assertIn("src/main/java/App.java", paths)
        self.assertIn("pom.xml", paths)
        self.assertIn(".coursegen/runtime/install.sh", paths)
        # README at the starter root is excluded.
        self.assertNotIn("README.md", paths)
        # Visible check scripts land at checks/<id>/...
        self.assertIn("checks/d1/run_visible_checks.py", paths)
        self.assertIn("checks/d2/run_visible_checks.py", paths)
        self.assertIn("checks/d3/run_visible_checks.py", paths)
        # Per-deliverable READMEs are excluded (handled separately).
        self.assertNotIn("checks/d1/README.md", paths)
        # Private artifacts never leak in.
        self.assertNotIn("grader/d1/run_hidden_checks.py", paths)

    def test_non_first_deliverable_returns_empty_to_avoid_duplication(self) -> None:
        spec = _make_spec(shared=True, deliverable_ids=["d1", "d2"])
        files = _bundle_files([
            ("public/starter/Dockerfile", "public"),
            ("public/checks/d2/run_visible_checks.py", "public"),
        ])
        run = _make_run(files)
        service = _stub_service()

        seed = service._workspace_seed_files(
            spec=spec,
            workflow_run=run,
            workflow_run_id="run_test",
            spec_deliverable_id="d2",  # NOT first
            content_markdown="",
            starter_readme="",
        )
        self.assertEqual(seed, [],
                         "Non-first deliverables must return empty seed_files to "
                         "avoid duplicating the shared starter content N times.")

    def test_shared_starter_seed_excludes_private_files(self) -> None:
        spec = _make_spec(shared=True, deliverable_ids=["d1", "d2"])
        files = _bundle_files([
            ("public/starter/Dockerfile", "public"),
            ("public/checks/d1/run_visible_checks.py", "public"),
            # Private grader / manifest — must never appear in learner seed.
            ("private/grader/d1/run_hidden_checks.py", "private"),
            ("private/grader/d1/deliverable.json", "private"),
            ("public/checks/d1/run_visible_checks.py", "private"),  # mismarked
        ])
        run = _make_run(files)
        service = _stub_service()

        seed = service._workspace_seed_files(
            spec=spec, workflow_run=run, workflow_run_id="run_test",
            spec_deliverable_id="d1", content_markdown="", starter_readme="",
        )
        paths = {f.relative_path for f in seed}
        for p in paths:
            self.assertFalse(
                p.startswith("grader/") or "hidden_checks" in p,
                f"Private artifact leaked into learner workspace seed: {p}",
            )


class LegacyNonSharedSeedFilesTests(unittest.TestCase):
    def test_non_shared_layout_strips_per_deliverable_starter_prefix(self) -> None:
        spec = _make_spec(shared=False, deliverable_ids=["d1", "d2"])
        files = _bundle_files([
            ("public/starter/d1/Dockerfile", "public"),
            ("public/starter/d1/src/main/java/App.java", "public"),
            ("public/starter/d1/README.md", "public"),  # excluded
            ("public/starter/d2/Dockerfile", "public"),  # other deliverable
        ])
        run = _make_run(files)
        service = _stub_service()

        seed = service._workspace_seed_files(
            spec=spec, workflow_run=run, workflow_run_id="run_test",
            spec_deliverable_id="d1", content_markdown="", starter_readme="",
        )
        paths = {f.relative_path for f in seed}
        self.assertIn("Dockerfile", paths)
        self.assertIn("src/main/java/App.java", paths)
        self.assertNotIn("README.md", paths)
        # Other deliverable's starter content not included.
        for p in paths:
            self.assertFalse(p.startswith("d2/"), f"d2 leaked into d1's seed: {p}")


if __name__ == "__main__":
    unittest.main()
