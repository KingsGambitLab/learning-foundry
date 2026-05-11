"""Pass 3 of the workspace refactor: the baseline matrix verifier must boot
the shared starter ONCE and run per-deliverable visible+hidden suites that
live OUTSIDE the starter tree.

Old contract:
    verify_deliverable(workspace_root=public/starter/<id>, spec, starter_type)
    --> booted the per-deliverable starter, ran checks/ and .coursegen/grader/
        scripts under that workspace, mocked an empty-repo copy of the same.

New contract:
    verify_course(workspace_root=public_root, private_root=private_root, spec=spec)
    --> boots `public/starter` ONCE, runs each deliverable's
        public/checks/<id>/run_visible_checks.py and
        private/grader/<id>/run_hidden_checks.py against the live starter and
        against a single empty-repo copy. Every BaselineSuiteOutcome and
        BaselineValidationIssue carries `deliverable_id` so per-deliverable
        findings are attributable.
"""
from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

from app.domain.registry import PackageType
from app.domain.workflow import MaterializeBundleRequest
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.generated_test_harness import (
    BaselineSuiteOutcome,
    BaselineValidationIssue,
    BaselineValidationResult,
    GeneratedTestBaselineVerifier,
    GeneratedTestCaseReport,
    GeneratedTestScriptRunner,
    GeneratedTestSuiteReport,
)
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
            "Build a multi-warehouse inventory reservation service with FastAPI, "
            "Postgres, and Redis. Keep reservations correct under concurrency, "
            "retries, and stock transfers."
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


def _passing_report(suite_type: str) -> GeneratedTestSuiteReport:
    return GeneratedTestSuiteReport(
        suite_type=suite_type,
        command=f"python checks/run_{suite_type}_checks.py",
        exit_code=0,
        valid=True,
        passed=True,
        tests=[
            GeneratedTestCaseReport(
                id=f"{suite_type}_case_1",
                title=f"{suite_type} happy path",
                status="passed",
            )
        ],
        summary=f"{suite_type} passed",
    )


def _failing_report(suite_type: str) -> GeneratedTestSuiteReport:
    return GeneratedTestSuiteReport(
        suite_type=suite_type,
        command=f"python checks/run_{suite_type}_checks.py",
        exit_code=1,
        valid=True,
        passed=False,
        tests=[
            GeneratedTestCaseReport(
                id=f"{suite_type}_case_1",
                title=f"{suite_type} happy path",
                status="failed",
            )
        ],
        summary=f"{suite_type} failed",
    )


class _StubScriptRunner:
    """Records calls to run_suite and returns scripted GeneratedTestSuiteReports.

    Each call records (workspace_root, command, suite_type) so the test can
    assert which deliverable's suite ran against which baseline workspace.
    """

    def __init__(self, *, plan):
        self.plan = plan
        self.calls = []

    def run_suite(
        self,
        *,
        workspace_root: Path,
        command: str,
        base_url: str,
        suite_type: str,
    ) -> GeneratedTestSuiteReport:
        self.calls.append(
            {
                "workspace_root": Path(workspace_root),
                "command": command,
                "suite_type": suite_type,
                "base_url": base_url,
            }
        )
        return self.plan(workspace_root=workspace_root, command=command, suite_type=suite_type)


class _StubLearnerStudioService:
    """Minimal stand-in that records _running_app invocations and yields a fake URL.

    Tracks workspaces booted so tests can assert single-boot semantics.
    """

    def __init__(self):
        self.booted_workspaces = []

    @contextmanager
    def _running_app(self, *, workspace_root, spec):
        self.booted_workspaces.append(Path(workspace_root))
        yield "http://localhost:0"


class _PatchedVerifier(GeneratedTestBaselineVerifier):
    """A GeneratedTestBaselineVerifier with a deterministic _running_app shim
    so tests don't need Docker.
    """

    def __init__(self, *, script_runner, boot_log):
        super().__init__(script_runner=script_runner)
        self._boot_log = boot_log

    @contextmanager
    def _running_app(self, *, workspace_root, spec):
        self._boot_log.append(Path(workspace_root))
        yield "http://localhost:0"


class BaselineMatrixVerifyCourseTests(unittest.TestCase):
    """verify_course boots the shared starter ONCE and emits per-deliverable findings."""

    def test_verify_course_flags_starter_suite_that_passed_pre_implementation(self) -> None:
        """If a deliverable's visible suite passes against the shared starter
        (which has no business-logic implementation yet), the verifier must
        emit `starter_suite_passed_pre_implementation` attributed to that
        deliverable_id.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            public_root = Path(workspace.public_dir)
            private_root = Path(workspace.private_dir)

            first_deliverable = spec.deliverables[0]

            def plan(*, workspace_root, command, suite_type):
                # The visible suite of the FIRST deliverable passes against the
                # shared starter (wrong: starter has no implementation yet).
                # Every other suite fails properly.
                workspace_root = Path(workspace_root)
                is_starter = workspace_root == (public_root / "starter")
                if (
                    is_starter
                    and suite_type == "visible"
                    and f"checks/{first_deliverable.id}" in command
                ):
                    return _passing_report("visible")
                return _failing_report(suite_type)

            boot_log: list[Path] = []
            verifier = _PatchedVerifier(
                script_runner=_StubScriptRunner(plan=plan),
                boot_log=boot_log,
            )

            result = verifier.verify_course(
                workspace_root=public_root,
                private_root=private_root,
                spec=spec,
            )

        self.assertFalse(result.valid)
        offending = [
            issue for issue in result.errors
            if issue.code == "starter_suite_passed_pre_implementation"
        ]
        self.assertTrue(
            offending,
            f"expected starter_suite_passed_pre_implementation error; got {[i.code for i in result.errors]}",
        )
        attributed = [issue for issue in offending if issue.deliverable_id == first_deliverable.id]
        self.assertTrue(
            attributed,
            "starter_suite_passed_pre_implementation must be attributed to the offending deliverable_id",
        )
        # The other deliverable must NOT carry this error since its suite failed.
        for deliverable in spec.deliverables[1:]:
            self.assertFalse(
                any(
                    issue.code == "starter_suite_passed_pre_implementation"
                    and issue.deliverable_id == deliverable.id
                    for issue in result.errors
                ),
                f"deliverable {deliverable.id} should not be flagged when its suite failed",
            )
        # Per-deliverable BaselineSuiteOutcome attribution
        per_deliverable_outcomes = {
            (outcome.deliverable_id, outcome.baseline, outcome.suite_type)
            for outcome in result.outcomes
        }
        self.assertIn(
            (first_deliverable.id, "starter_repo", "visible"),
            per_deliverable_outcomes,
            "verify_course outcomes must carry deliverable_id for per-deliverable attribution",
        )

    def test_verify_course_flags_hidden_weaker_than_visible_against_starter(self) -> None:
        """If visible fails but hidden passes against the shared starter, emit
        `hidden_tests_weaker_than_visible` for that deliverable.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            public_root = Path(workspace.public_dir)
            private_root = Path(workspace.private_dir)

            first_deliverable = spec.deliverables[0]

            def plan(*, workspace_root, command, suite_type):
                workspace_root = Path(workspace_root)
                is_starter = workspace_root == (public_root / "starter")
                if (
                    is_starter
                    and suite_type == "hidden"
                    and f"grader/{first_deliverable.id}" in command
                ):
                    # Hidden passes but visible fails on the starter
                    return _passing_report("hidden")
                return _failing_report(suite_type)

            boot_log: list[Path] = []
            verifier = _PatchedVerifier(
                script_runner=_StubScriptRunner(plan=plan),
                boot_log=boot_log,
            )

            result = verifier.verify_course(
                workspace_root=public_root,
                private_root=private_root,
                spec=spec,
            )

        self.assertFalse(result.valid)
        offending = [
            issue for issue in result.errors
            if issue.code == "hidden_tests_weaker_than_visible"
            and issue.deliverable_id == first_deliverable.id
        ]
        self.assertTrue(
            offending,
            f"expected hidden_tests_weaker_than_visible for {first_deliverable.id}; got {[(i.code, i.deliverable_id) for i in result.errors]}",
        )

    def test_verify_course_boots_shared_starter_once(self) -> None:
        """No matter how many deliverables, the shared starter boots ONCE; the
        empty-repo copy boots ONCE; total = 2 boots, not 2*N.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            public_root = Path(workspace.public_dir)
            private_root = Path(workspace.private_dir)

            self.assertGreaterEqual(
                len(spec.deliverables), 2,
                "fixture must produce >=2 deliverables to make single-boot meaningful",
            )

            def plan(*, workspace_root, command, suite_type):
                return _failing_report(suite_type)

            boot_log: list[Path] = []
            verifier = _PatchedVerifier(
                script_runner=_StubScriptRunner(plan=plan),
                boot_log=boot_log,
            )

            verifier.verify_course(
                workspace_root=public_root,
                private_root=private_root,
                spec=spec,
            )

        # Exactly two boots: starter + empty-repo copy (NOT N starter boots).
        self.assertEqual(
            len(boot_log), 2,
            f"expected 2 boots (starter + empty-repo); got {len(boot_log)} -- one boot per deliverable suggests per-deliverable re-boot",
        )
        # Exactly one of the boots is the literal shared starter root.
        starter_boots = [path for path in boot_log if path == (public_root / "starter")]
        self.assertEqual(
            len(starter_boots), 1,
            f"shared starter must boot exactly once; got {len(starter_boots)} starter boots",
        )


if __name__ == "__main__":
    unittest.main()
