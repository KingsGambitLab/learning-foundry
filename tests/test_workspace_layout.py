"""Pass 2 of the workspace refactor: shared-codebase courses must materialize
ONE shared starter root, with per-deliverable visible/hidden test artifacts
living outside the starter tree.

Old layout:
    public/starter/deliverable_1/  # full src tree + Dockerfile + manifests + scripts
    public/starter/deliverable_2/  # byte-identical copy
    ...

New layout (for shared_codebase=True):
    public/starter/                 # one shared root: code, Dockerfile, runtime scripts
    public/checks/<id>/             # per-deliverable: README.md + run_visible_checks.py
    private/grader/<id>/            # per-deliverable: deliverable.json + run_hidden_checks.py
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.domain.registry import PackageType
from app.domain.workflow import MaterializeBundleRequest
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _materialized_run(temp_dir: str):
    store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
    )
    intake = GenerationIntake(
        title="Inventory Reservation Service",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    inferred = infer_assignment_design(
        title=intake.title,
        problem_statement=intake.problem_statement,
        package_type_hint=intake.package_type_hint,
    )
    assert inferred.design_spec is not None
    inferred.design_spec.runtime_dependencies.editable_files = ["app.py"]
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake,
        design_spec=inferred.design_spec,
        execute_nodes=False,
    )
    workflow_service.materialize_run(run.id, MaterializeBundleRequest(overwrite=True))
    run = workflow_service.get_run(run.id)
    assert run is not None
    run, _ = TaskAgentWorkspaceAuthoringService().author_workspace(run)
    return run


class SharedCodebaseWorkspaceLayoutTests(unittest.TestCase):
    """The materialized workspace for a shared_codebase course must keep the
    starter tree as ONE shared root and split per-deliverable artifacts into
    `public/checks/<id>/` and `private/grader/<id>/`.
    """

    def test_shared_starter_root_is_singular(self) -> None:
        """`public/starter/` exists once and is NOT split into per-deliverable
        subfolders for shared_codebase courses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)
            self.assertTrue(spec.course_structure.shared_codebase)

            starter_root = Path(workspace.public_dir) / "starter"
            self.assertTrue(
                starter_root.exists(),
                f"public/starter/ must exist as the shared starter root; missing at {starter_root}",
            )

            for deliverable in spec.deliverables:
                per_deliverable_path = starter_root / deliverable.id
                self.assertFalse(
                    per_deliverable_path.exists(),
                    f"public/starter/{deliverable.id}/ must NOT exist for shared_codebase courses; "
                    f"the starter root is shared across all deliverables.",
                )

    def test_per_deliverable_visible_checks_under_public_checks(self) -> None:
        """`public/checks/<deliverable_id>/run_visible_checks.py` and README.md
        exist for each deliverable, and the old `starter/<id>/checks/` path is gone.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)

            public_root = Path(workspace.public_dir)
            for deliverable in spec.deliverables:
                checks_dir = public_root / "checks" / deliverable.id
                self.assertTrue(
                    (checks_dir / "run_visible_checks.py").exists(),
                    f"Visible check script must live at public/checks/{deliverable.id}/run_visible_checks.py",
                )
                self.assertTrue(
                    (checks_dir / "README.md").exists(),
                    f"Learner-facing brief must live at public/checks/{deliverable.id}/README.md",
                )
                old_visible = public_root / "starter" / deliverable.id / "checks" / "run_visible_checks.py"
                self.assertFalse(
                    old_visible.exists(),
                    f"Old visible check location must be gone: {old_visible}",
                )

    def test_per_deliverable_hidden_grader_under_private(self) -> None:
        """Hidden grader and per-deliverable manifest live under
        `private/grader/<id>/` and are NOT exposed under `public/`.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)

            workspace_root = Path(workspace.root_dir)
            public_root = Path(workspace.public_dir)
            for deliverable in spec.deliverables:
                grader_dir = workspace_root / "private" / "grader" / deliverable.id
                self.assertTrue(
                    (grader_dir / "deliverable.json").exists(),
                    f"Per-deliverable manifest must live at private/grader/{deliverable.id}/deliverable.json",
                )
                self.assertTrue(
                    (grader_dir / "run_hidden_checks.py").exists(),
                    f"Hidden grader script must live at private/grader/{deliverable.id}/run_hidden_checks.py",
                )
                # Hidden grader must NOT be under public/
                public_hidden = (
                    public_root
                    / "starter"
                    / deliverable.id
                    / ".coursegen"
                    / "grader"
                    / "run_hidden_checks.py"
                )
                self.assertFalse(
                    public_hidden.exists(),
                    f"Hidden grader must not be exposed under public/: {public_hidden}",
                )
                # Per-deliverable manifest must NOT be in old location
                old_manifest = (
                    public_root
                    / "starter"
                    / deliverable.id
                    / ".coursegen"
                    / "deliverable.json"
                )
                self.assertFalse(
                    old_manifest.exists(),
                    f"Per-deliverable manifest must not live in old location: {old_manifest}",
                )

    def test_shared_runtime_protocol_lives_at_starter_root(self) -> None:
        """Dockerfile and `.coursegen/runtime/*.sh` live ONCE at `public/starter/`,
        not duplicated per deliverable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)

            shared_starter = Path(workspace.public_dir) / "starter"
            self.assertTrue(
                (shared_starter / "Dockerfile").exists(),
                "Dockerfile must live once at public/starter/",
            )
            for script_name in (
                "install.sh",
                "verify.sh",
                "run.sh",
                "check_visible.sh",
                "check_hidden.sh",
            ):
                self.assertTrue(
                    (shared_starter / ".coursegen" / "runtime" / script_name).exists(),
                    f"public/starter/.coursegen/runtime/{script_name} must exist on the shared root",
                )

    def test_shared_course_manifest_lives_at_starter_root_once(self) -> None:
        """`public/starter/.coursegen/course.json` is a single shared manifest
        carrying the course-level fields that previously got duplicated into
        every `private/grader/<id>/deliverable.json`. After Pass 2 it must
        exist once at the shared starter root and carry one consistent
        `runtime_plan` for the whole course."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)

            course_manifest_path = (
                Path(workspace.public_dir) / "starter" / ".coursegen" / "course.json"
            )
            self.assertTrue(
                course_manifest_path.exists(),
                f"Shared course manifest must live at {course_manifest_path}",
            )
            course_payload = json.loads(course_manifest_path.read_text(encoding="utf-8"))
            self.assertIn(
                "runtime_plan",
                course_payload,
                "Shared course manifest must carry the course runtime_plan once.",
            )
            self.assertIn(
                "runtime_dependencies",
                course_payload,
                "Shared course manifest must carry runtime_dependencies once.",
            )
            self.assertIn(
                "course_structure",
                course_payload,
                "Shared course manifest must carry course_structure once.",
            )
            self.assertIn(
                "public_endpoints",
                course_payload,
                "Shared course manifest must carry public_endpoints once.",
            )
            self.assertEqual(
                course_payload["runtime_plan"],
                spec.project_contract.runtime_plan.model_dump(mode="json"),
                "Shared course runtime_plan must match the authoritative spec value.",
            )
            self.assertEqual(
                course_payload["runtime_dependencies"],
                spec.runtime_dependencies.model_dump(mode="json"),
                "Shared course runtime_dependencies must match the authoritative spec value.",
            )


class ProgressiveBundleSharedRootTests(unittest.TestCase):
    """The shared-codebase progressive authoring bundle must write
    `normalized_repo_files` ONCE to the shared starter root, not duplicated
    across N per-deliverable folders.
    """

    def test_progressive_bundle_writes_repo_files_to_shared_root_only(self) -> None:
        from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService
        from app.services.task_agent_starter_templates import (
            RUNTIME_INSTALL_SCRIPT_PATH,
            RUNTIME_RUN_SCRIPT_PATH,
            RUNTIME_VERIFY_SCRIPT_PATH,
        )

        class _FakeUsage:
            input_tokens = 1
            output_tokens = 1
            total_tokens = 2
            input_tokens_details = type("D", (), {"cached_tokens": 0})()
            output_tokens_details = type("D", (), {"reasoning_tokens": 0})()

        class _FakeResponse:
            def __init__(self, parsed):
                self.output_parsed = parsed
                self.usage = _FakeUsage()

        class _FakeAPI:
            def __init__(self, parsed):
                self._parsed = parsed
                self.calls = []

            def parse(self, **kwargs):
                self.calls.append(kwargs)
                return _FakeResponse(self._parsed)

        class _FakeClient:
            def __init__(self, parsed):
                self.responses = _FakeAPI(parsed)

        bundle = type(
            "SharedRepoBundle",
            (),
            {
                "runtime_protocol_files": [
                    type(
                        "RepoFile",
                        (),
                        {"path": "Dockerfile", "content": "FROM debian:bookworm-slim\n"},
                    )(),
                    type(
                        "RepoFile",
                        (),
                        {
                            "path": RUNTIME_INSTALL_SCRIPT_PATH,
                            "content": "#!/usr/bin/env sh\nset -eu\n",
                        },
                    )(),
                    type(
                        "RepoFile",
                        (),
                        {
                            "path": RUNTIME_VERIFY_SCRIPT_PATH,
                            "content": "#!/usr/bin/env sh\nset -eu\n",
                        },
                    )(),
                    type(
                        "RepoFile",
                        (),
                        {
                            "path": RUNTIME_RUN_SCRIPT_PATH,
                            "content": "#!/usr/bin/env sh\nset -eu\n",
                        },
                    )(),
                ],
                "files": [
                    type("RepoFile", (), {"path": "app.py", "content": "print('hello')\n"})(),
                ],
                "dependency_contract": {
                    "manifest_paths": [],
                    "lockfile_paths": [],
                    "toolchain_paths": [],
                    "build_support_paths": [],
                    "reproducibility_mode": None,
                },
                "notes": [],
            },
        )()

        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)

            fake_client = _FakeClient(bundle)
            service = OpenAIStarterRepoAuthoringService(
                enabled=True,
                client_factory=lambda **_: fake_client,
            )
            run, result = service.author_workspace_repo(run)
            self.assertTrue(result.available)

            shared_starter = Path(workspace.public_dir) / "starter"
            self.assertEqual(
                (shared_starter / "app.py").read_text(encoding="utf-8"),
                "print('hello')\n",
                "Progressive bundle must write authored repo files ONCE to public/starter/",
            )
            # Sanity: each per-deliverable manifest still has updated metadata
            workspace_root = Path(workspace.root_dir)
            for deliverable in spec.deliverables:
                manifest_path = (
                    workspace_root
                    / "private"
                    / "grader"
                    / deliverable.id
                    / "deliverable.json"
                )
                self.assertTrue(
                    manifest_path.exists(),
                    f"Per-deliverable manifest missing at {manifest_path}",
                )
                # No per-deliverable starter copy was created.
                per_deliverable_app = (
                    Path(workspace.public_dir) / "starter" / deliverable.id / "app.py"
                )
                self.assertFalse(
                    per_deliverable_app.exists(),
                    f"Repo files must not be copied per-deliverable: {per_deliverable_app}",
                )


class Pass5CallSiteShareCodebasePathTests(unittest.TestCase):
    """Pass 5: every caller that constructs `public/starter/<deliverable.id>` must
    branch on shared_codebase. For shared-codebase courses the visible scripts live
    at `public/checks/<id>/run_visible_checks.py` and the hidden scripts at
    `private/grader/<id>/run_hidden_checks.py`; the shared code lives at
    `public/starter/` (no deliverable.id segment).
    """

    def test_authoring_tests_node_resolves_visible_path_under_public_checks(self) -> None:
        """`_authoring_tests_node` in `langgraph_assignment_graph.py` must read
        the visible/hidden check scripts from the shared layout (public/checks/<id>/
        and private/grader/<id>/) and NOT from public/starter/<id>/.

        Verifies the positive path: with the new layout fully materialized, the
        node must pass (no missing-scripts finding). With the current code that
        reads public/starter/<id>/checks/run_visible_checks.py, this fails because
        that path never exists in the new layout.
        """
        from app.services.docker_sandbox_runner import DockerSandboxRunner
        from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
        from app.services.openai_test_script_authoring import (
            TestScriptAuthoringResult,
            TestScriptAuthoringSource,
        )

        class _NoOpTestAuthoring:
            def author_workspace_tests(self, run, **kwargs):  # noqa: ANN003
                return run, TestScriptAuthoringResult(
                    source=TestScriptAuthoringSource.unavailable,
                    updated_files=[],
                    usage=None,
                    notes=[],
                    message="left scripts in place",
                    available=False,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)
            self.assertTrue(spec.course_structure.shared_codebase)

            # Pre-condition: shared-layout scripts exist (materialized).
            for deliverable in spec.deliverables:
                shared_visible = (
                    Path(workspace.public_dir)
                    / "checks"
                    / deliverable.id
                    / "run_visible_checks.py"
                )
                shared_hidden = (
                    Path(workspace.root_dir)
                    / "private"
                    / "grader"
                    / deliverable.id
                    / "run_hidden_checks.py"
                )
                self.assertTrue(
                    shared_visible.exists(),
                    f"Pre-condition: shared visible script must exist at {shared_visible}",
                )
                self.assertTrue(
                    shared_hidden.exists(),
                    f"Pre-condition: shared hidden script must exist at {shared_hidden}",
                )

            graph = LangGraphAssignmentGraph(
                DockerSandboxRunner(),
                test_authoring_service=_NoOpTestAuthoring(),
            )
            state = {
                "run": run,
                "node_executions": [],
                "active_iteration": 1,
                "authoring_attempt": 1,
                "reviewer_attempt": 0,
                "cached_sandbox_result": None,
                "next_retry_node": None,
            }

            updated = graph._authoring_tests_node(state)
            latest = updated["node_executions"][-1]

            self.assertFalse(
                any(
                    finding.code == "generated_test_scripts_missing"
                    for finding in latest.findings
                ),
                "Authoring-tests node must NOT report a generated_test_scripts_missing "
                "finding when shared-layout scripts are in place. With the legacy "
                "code reading public/starter/<id>/checks/run_visible_checks.py, this "
                "fails because that legacy path is empty for shared-codebase courses.",
            )

    def test_reviewer_code_node_reads_app_files_from_shared_starter(self) -> None:
        """`_reviewer_code_node` in `langgraph_assignment_graph.py` must inspect
        learner-editable app files at `public/starter/<file>` for shared-codebase
        courses, NOT `public/starter/<deliverable.id>/<file>`.

        Verifies that an authored (non-placeholder, non-wrapper) app file at the
        SHARED root is recognized as real code. With the legacy buggy code at
        line 946 reading public/starter/<id>/<file>, the file is unreadable
        (OSError -> deliverable_has_placeholder = True), which produces a
        false-positive placeholder finding.
        """
        from app.services.docker_sandbox_runner import DockerSandboxRunner
        from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
        from app.services.task_agent_contract_surface import (
            learner_editable_paths_for_deliverable,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)
            self.assertTrue(spec.course_structure.shared_codebase)

            first_deliverable = spec.deliverables[0]
            editable_paths = learner_editable_paths_for_deliverable(spec, first_deliverable)
            self.assertTrue(editable_paths, "Pre-condition: editable_paths must be non-empty")

            # Author a REAL (non-placeholder, non-wrapper) body at the SHARED root.
            # A correct reviewer reads from public/starter/<file> and reports no
            # placeholder. A buggy reviewer reads public/starter/<id>/<file>,
            # which doesn't exist, marks placeholder, and reports a false positive.
            shared_starter = Path(workspace.public_dir) / "starter"
            real_path = shared_starter / editable_paths[0]
            real_path.parent.mkdir(parents=True, exist_ok=True)
            real_path.write_text(
                "from fastapi import FastAPI\n"
                "\n"
                "app = FastAPI()\n"
                "\n"
                "@app.get('/run')\n"
                "def run():\n"
                "    return {'status': 'ok'}\n",
                encoding="utf-8",
            )

            graph = LangGraphAssignmentGraph(DockerSandboxRunner())
            from datetime import UTC, datetime as _dt

            from app.domain.sandbox import (
                SandboxExecutionResult,
                SandboxExecutionStatus,
            )

            state = {
                "run": run,
                "node_executions": [],
                "active_iteration": 1,
                "authoring_attempt": 1,
                "reviewer_attempt": 0,
                "cached_sandbox_result": SandboxExecutionResult(
                    status=SandboxExecutionStatus.passed,
                    available=True,
                    build_succeeded=True,
                    run_succeeded=True,
                    generated_at=_dt.now(UTC),
                ),
                "next_retry_node": None,
            }

            updated = graph._reviewer_code_node(state)
            latest = updated["node_executions"][-1]
            self.assertFalse(
                any(
                    finding.code == "placeholder_starter_endpoints_remain"
                    for finding in latest.findings
                ),
                "Reviewer-code node must NOT report a placeholder for a real "
                "app file at the SHARED starter root. With the legacy code "
                "reading public/starter/<id>/<file>, the file is unreadable and "
                "the reviewer falsely flags it as a placeholder.",
            )

    def test_openai_test_script_authoring_writes_visible_to_public_checks_for_shared(self) -> None:
        """`OpenAITestScriptAuthoringService.author_workspace_tests` must write
        visible/hidden scripts to the SHARED layout for shared-codebase courses:
        public/checks/<id>/run_visible_checks.py
        private/grader/<id>/run_hidden_checks.py
        """
        from app.services.openai_test_script_authoring import (
            OpenAITestScriptAuthoringService,
        )

        # Fake an OpenAI client that returns known scripts.
        passing_visible = "print('visible script body')\n"
        passing_hidden = "print('hidden script body')\n"

        class _FakeUsage:
            input_tokens = 1
            output_tokens = 1
            total_tokens = 2
            input_tokens_details = type("D", (), {"cached_tokens": 0})()
            output_tokens_details = type("D", (), {"reasoning_tokens": 0})()

        class _FakeParsed:
            def __init__(self, visible: str, hidden: str) -> None:
                self.visible_script = visible
                self.hidden_script = hidden
                self.notes: list[str] = []

        class _FakeResponse:
            def __init__(self, parsed) -> None:
                self.output_parsed = parsed
                self.usage = _FakeUsage()

        class _FakeAPI:
            def __init__(self, parsed) -> None:
                self._parsed = parsed
                self.calls: list[dict] = []

            def parse(self, **kwargs):
                self.calls.append(kwargs)
                return _FakeResponse(self._parsed)

        class _FakeClient:
            def __init__(self, parsed) -> None:
                self.responses = _FakeAPI(parsed)

        parsed = _FakeParsed(passing_visible, passing_hidden)

        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            self.assertIsNotNone(spec)
            self.assertIsNotNone(workspace)
            self.assertTrue(spec.course_structure.shared_codebase)

            import os as _os

            _os.environ["OPENAI_API_KEY"] = "test-key"
            service = OpenAITestScriptAuthoringService(
                enabled=True,
                client_factory=lambda **_: _FakeClient(parsed),
                env_file=None,
            )
            run, result = service.author_workspace_tests(run)
            self.assertTrue(
                result.available,
                f"test-script authoring should succeed but got: {result.message}",
            )

            workspace_root = Path(workspace.root_dir)
            public_root = Path(workspace.public_dir)
            for deliverable in spec.deliverables:
                shared_visible = (
                    public_root / "checks" / deliverable.id / "run_visible_checks.py"
                )
                shared_hidden = (
                    workspace_root
                    / "private"
                    / "grader"
                    / deliverable.id
                    / "run_hidden_checks.py"
                )
                self.assertTrue(
                    shared_visible.exists(),
                    f"Shared-layout visible script missing at {shared_visible}",
                )
                self.assertTrue(
                    shared_hidden.exists(),
                    f"Shared-layout hidden script missing at {shared_hidden}",
                )
                self.assertIn(
                    "visible script body",
                    shared_visible.read_text(encoding="utf-8"),
                    "Visible script under public/checks/<id>/ must carry authored content.",
                )
                self.assertIn(
                    "hidden script body",
                    shared_hidden.read_text(encoding="utf-8"),
                    "Hidden script under private/grader/<id>/ must carry authored content.",
                )

                # Must NOT write to legacy public/starter/<id>/ layout for shared courses.
                legacy_visible = (
                    public_root
                    / "starter"
                    / deliverable.id
                    / "checks"
                    / "run_visible_checks.py"
                )
                legacy_hidden = (
                    public_root
                    / "starter"
                    / deliverable.id
                    / ".coursegen"
                    / "grader"
                    / "run_hidden_checks.py"
                )
                self.assertFalse(
                    legacy_visible.exists(),
                    f"Legacy public/starter/<id>/checks/run_visible_checks.py "
                    f"must NOT be written for shared courses: {legacy_visible}",
                )
                self.assertFalse(
                    legacy_hidden.exists(),
                    f"Legacy public/starter/<id>/.coursegen/grader/run_hidden_checks.py "
                    f"must NOT be written for shared courses: {legacy_hidden}",
                )


if __name__ == "__main__":
    unittest.main()
