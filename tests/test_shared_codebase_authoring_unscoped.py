"""Pin the architectural simplification: for shared-codebase progressive
courses, the authoring LLM call is NEVER scoped to a deliverable subset.

The mental model: deliverables are LEARNER CONSTRUCTS — milestones for
how the learner experiences the course. They're EVALUATION CRITERIA the
reviewer uses to assess the app. They are NOT authoring scopes.

The LLM authors ONE shared application that must satisfy ALL deliverables
simultaneously. It receives:
  - The full spec (with deliverables as evaluation criteria)
  - The current codebase (`current_files`)
  - Flat findings list from review (what's missing/broken)

It produces:
  - A complete app that satisfies every deliverable's criteria

There is no `repair_scope_deliverable_ids`, no "preserve files outside
scope," no per-deliverable file ownership tracking. The model treats the
codebase as one thing.

These tests pin the simplified contract:

  1. `_progressive_prompt_payload` does NOT carry `repair_scope_deliverable_ids`.
  2. The payload carries `shared_required_paths` — the UNION of every
     deliverable's `primary_editable_paths`.
  3. Each entry in `deliverables[]` carries that deliverable's
     `primary_editable_paths` (so the reviewer's per-deliverable
     evaluation logic has the same source of truth the model sees).
  4. `author_workspace_repo` for shared codebase calls the model with
     the FULL spec view regardless of which `deliverable_ids` the
     caller passes — deliverable_ids is purely advisory, never a
     filter.

The Go failure these tests prevent (`course_657df22958fb`): the prompt
showed only d1's required paths; the model authored only d1's files;
reviewer flagged d2-d6's missing files; repair re-authored with the
same incomplete prompt; infinite loop.
"""

from __future__ import annotations

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
from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService
from app.services.task_agent_workspace_authoring import (
    TaskAgentWorkspaceAuthoringService,
)
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _shared_run_with_disjoint_editable_paths(temp_dir: str):
    """Build a shared-codebase run whose deliverables declare DIFFERENT
    editable paths — the configuration that triggered the Go bug.
    """
    store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
    )
    intake = GenerationIntake(
        title="Production URL Shortener Service in Go",
        problem_statement=(
            "Build a production-ready URL shortener service in Go with "
            "PostgreSQL and Redis. Learners progress through deliverables "
            "that add CRUD endpoints, idempotent writes, redis-backed "
            "analytics caching, and rate-limited public reads."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    inferred = infer_assignment_design(
        title=intake.title,
        problem_statement=intake.problem_statement,
        package_type_hint=intake.package_type_hint,
    )
    assert inferred.design_spec is not None
    inferred.design_spec.runtime_dependencies.starter_type = StarterType.partial
    inferred.design_spec.runtime_dependencies.editable_files = [
        "go.mod",
        "cmd/server/main.go",
    ]
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


class SharedCodebaseAuthoringUnscopedTests(unittest.TestCase):
    def test_progressive_payload_does_not_carry_repair_scope_deliverable_ids(self) -> None:
        """The legacy `repair_scope_deliverable_ids` field was the
        signal "focus on these deliverables, preserve the others."
        That framing is what causes regressions on unrelated files
        during repair. The payload must NOT carry it — the model
        authors the whole app every time.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _shared_run_with_disjoint_editable_paths(temp_dir)
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(workspace)

            service = OpenAIStarterRepoAuthoringService.__new__(
                OpenAIStarterRepoAuthoringService
            )
            payload = service._progressive_prompt_payload(
                run=run,
                public_root=Path(workspace.public_dir),
                failure_context=None,
            )

            self.assertNotIn(
                "repair_scope_deliverable_ids",
                payload,
                "Shared-codebase authoring payload must NOT carry a "
                "deliverable-scope field. Deliverables are evaluation "
                "criteria, not authoring scopes.",
            )

    def test_progressive_payload_includes_union_of_required_paths(self) -> None:
        """The payload must carry `shared_required_paths` — the UNION of
        every deliverable's required editable paths. Without this, the
        model never learns about paths declared by deliverables 2..N.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _shared_run_with_disjoint_editable_paths(temp_dir)
            workspace = run.artifacts.workspace_snapshot
            spec = run.artifacts.task_agent_spec

            service = OpenAIStarterRepoAuthoringService.__new__(
                OpenAIStarterRepoAuthoringService
            )
            payload = service._progressive_prompt_payload(
                run=run,
                public_root=Path(workspace.public_dir),
                failure_context=None,
            )

            self.assertIn("shared_required_paths", payload)
            union_paths = set(payload["shared_required_paths"])
            self.assertGreater(
                len(union_paths),
                0,
                "shared_required_paths must contain at least one path.",
            )

            # Every primary_editable_path declared by any deliverable
            # must appear in the union.
            from app.services.task_agent_contract_surface import (
                learner_editable_paths_for_deliverable,
            )

            for deliverable in spec.deliverables:
                for path in learner_editable_paths_for_deliverable(spec, deliverable):
                    self.assertIn(
                        path,
                        union_paths,
                        f"shared_required_paths is missing path {path!r} "
                        f"declared by {deliverable.id}.",
                    )

    def test_each_deliverable_entry_carries_primary_editable_paths(self) -> None:
        """The reviewer evaluates each deliverable against its
        `primary_editable_paths`. The model needs the same view so it
        can author files that map to each milestone. Each entry in
        `deliverables[]` must carry the per-deliverable required paths.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _shared_run_with_disjoint_editable_paths(temp_dir)
            workspace = run.artifacts.workspace_snapshot
            spec = run.artifacts.task_agent_spec

            service = OpenAIStarterRepoAuthoringService.__new__(
                OpenAIStarterRepoAuthoringService
            )
            payload = service._progressive_prompt_payload(
                run=run,
                public_root=Path(workspace.public_dir),
                failure_context=None,
            )

            from app.services.task_agent_contract_surface import (
                learner_editable_paths_for_deliverable,
            )

            self.assertEqual(
                len(payload["deliverables"]),
                len(spec.deliverables),
            )
            for entry, deliverable in zip(
                payload["deliverables"], spec.deliverables
            ):
                self.assertEqual(entry["deliverable_id"], deliverable.id)
                self.assertIn(
                    "primary_editable_paths",
                    entry,
                    f"deliverables[{deliverable.id}] must carry "
                    f"primary_editable_paths.",
                )
                expected = learner_editable_paths_for_deliverable(spec, deliverable)
                self.assertEqual(
                    list(entry["primary_editable_paths"]),
                    list(expected),
                )

    def test_progressive_payload_signature_does_not_require_deliverable_ids(self) -> None:
        """The `deliverable_ids` parameter was the legacy signal for
        repair scope. After the simplification, callers must be able to
        build the payload WITHOUT providing it — the spec contains the
        deliverables already.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _shared_run_with_disjoint_editable_paths(temp_dir)
            workspace = run.artifacts.workspace_snapshot

            service = OpenAIStarterRepoAuthoringService.__new__(
                OpenAIStarterRepoAuthoringService
            )
            # No deliverable_ids passed — must succeed.
            payload = service._progressive_prompt_payload(
                run=run,
                public_root=Path(workspace.public_dir),
                failure_context=None,
            )
            self.assertIsNotNone(payload)
            self.assertIn("deliverables", payload)
            self.assertGreaterEqual(len(payload["deliverables"]), 1)


if __name__ == "__main__":
    unittest.main()
