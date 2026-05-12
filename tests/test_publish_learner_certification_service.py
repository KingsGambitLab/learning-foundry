"""Pin the publish-time learner-path certification skip for partial starters.

The publish-time certification re-runs visible/hidden tests against the
seeded learner workspace and expects them to pass. For a `partial` starter
the handlers raise `NotImplementedError`-equivalents by design; the tests
are *supposed* to fail until the learner implements them. Running this
certification for partial starters is structurally contradictory — it
cannot succeed.

These tests pin two behaviors:
  - When `starter_type=partial`, certification short-circuits with a
    passed=True report and a `partial_starter_certification_skipped`
    check.
  - The decision happens AHEAD of any actual workspace seeding / editor
    launch / grade run, so no Docker / OpenAI side effects fire.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from app.domain.publish import (
    PublishCertificationCheckStatus,
    PublishSnapshot,
    PublishSnapshotProvenance,
)
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
from app.services.publish_learner_certification_service import (
    PublishLearnerCertificationService,
)


def _make_partial_spec() -> TaskAgentServiceSpec:
    return TaskAgentServiceSpec(
        title="t",
        summary="s",
        package_type=PackageType.progressive_codebase_course,
        course_structure=CourseStructureSpec(
            package_type=PackageType.progressive_codebase_course,
            workspace_scope=WorkspaceScope.shared_course_workspace,
            progression_mode=ProgressionMode.independent_deliverables,
            shared_codebase=True,
        ),
        runtime_dependencies=RuntimeDependencySpec(
            execution_surface=ExecutionSurface.http_service,
            starter_type=StarterType.partial,
            implementation_language="python",
            application_framework="fastapi",
        ),
        capabilities=CapabilitySpec(),
        assessment_strategy=AssessmentStrategySpec(),
        deliverables=[
            DeliverableSpec(id="deliverable_1", title="D1", objective="o1"),
            DeliverableSpec(id="deliverable_2", title="D2", objective="o2"),
        ],
    )


def _make_snapshot(spec: TaskAgentServiceSpec) -> PublishSnapshot:
    return PublishSnapshot(
        id="snap_test",
        course_run_id="course_test",
        course_family_id="course_test",
        created_at=datetime.now(UTC),
        version=1,
        source_hash="hash",
        shared_workflow_run_id="run_test",
        learner_package=None,
        task_agent_spec=spec,
        provenance=PublishSnapshotProvenance(
            generator_version="test",
            course_run_hash="hash",
            source_hash="hash",
            shared_workflow_run_id="run_test",
        ),
    )


class PartialStarterCertificationSkipTests(unittest.TestCase):
    def test_partial_starter_certification_returns_passed_skipped_report(self) -> None:
        spec = _make_partial_spec()
        snapshot = _make_snapshot(spec)
        service = PublishLearnerCertificationService(enabled=True)

        report = service.certify_snapshot(snapshot)

        self.assertTrue(
            report.passed,
            "Partial-starter snapshots must pass certification (the gate is "
            "structurally inapplicable to starters whose tests are designed "
            "to fail until learners implement them).",
        )
        check_keys = {check.key for check in report.checks}
        self.assertIn("partial_starter_certification_skipped", check_keys)
        skipped = next(
            c for c in report.checks
            if c.key == "partial_starter_certification_skipped"
        )
        self.assertEqual(skipped.status, PublishCertificationCheckStatus.skipped)
        self.assertFalse(skipped.blocking)

    def test_partial_starter_certification_does_not_seed_workspace_or_grade(self) -> None:
        """The short-circuit must happen BEFORE any expensive side effect
        (workspace seeding, editor launch, grade run). If certification ever
        starts re-running visible tests for partial starters, every retry
        would burn a full Docker boot + grading cycle for no possible win.
        """
        spec = _make_partial_spec()
        snapshot = _make_snapshot(spec)
        service = PublishLearnerCertificationService(enabled=True)

        with (
            patch(
                "app.services.publish_learner_certification_service.seed_workspace_from_snapshot"
            ) as seed,
            patch.object(service.learner_studio_service, "launch_editor") as launch,
            patch.object(service.learner_studio_service, "grade_assignment") as grade,
        ):
            service.certify_snapshot(snapshot)

        seed.assert_not_called()
        launch.assert_not_called()
        grade.assert_not_called()

    def test_disabled_service_still_returns_passed_skipped(self) -> None:
        """Sanity check: the existing `enabled=False` skip path still works
        and doesn't accidentally fall through to the new partial-starter
        branch.
        """
        spec = _make_partial_spec()
        snapshot = _make_snapshot(spec)
        service = PublishLearnerCertificationService(enabled=False)
        report = service.certify_snapshot(snapshot)
        self.assertTrue(report.passed)
        check_keys = {check.key for check in report.checks}
        self.assertIn("learner_certification", check_keys)


if __name__ == "__main__":
    unittest.main()
