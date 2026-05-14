"""Tests that PublishLearnerCertificationService resolves lab_tutor_enabled
from the snapshot's source CourseRun and passes it through to launch_editor.

Strategy: patch certify_snapshot's internal helpers (seed, validate, brief,
grade) with mocks, and let the actual store-lookup + launch_editor call
path run. We assert on the kwargs received by launch_editor.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from app.domain.course import CourseRun, CourseRunStage, CourseRunStatus
from app.domain.learner import LearnerWorkspaceScope, LearnerWorkspaceSession, LearnerWorkspaceSessionStatus
from app.domain.publish import PublishSnapshot, PublishSnapshotProvenance
from app.domain.registry import PackageType
from app.services.publish_learner_certification_service import (
    PublishLearnerCertificationService,
)


def _make_snapshot(course_run_id: str = "course_test") -> PublishSnapshot:
    return PublishSnapshot(
        id="snap_cert_test",
        course_run_id=course_run_id,
        course_family_id=course_run_id,
        created_at=datetime.now(UTC),
        version=1,
        source_hash="hash",
        shared_workflow_run_id="run_test",
        learner_package=None,
        task_agent_spec=None,
        provenance=PublishSnapshotProvenance(
            generator_version="test",
            course_run_hash="hash",
            source_hash="hash",
            shared_workflow_run_id="run_test",
        ),
    )


def _make_course_run(run_id: str, lab_tutor_enabled: bool) -> CourseRun:
    now = datetime.now(UTC)
    return CourseRun(
        id=run_id,
        course_family_id=run_id,
        title="Test Course",
        summary="A test course.",
        package_type=PackageType.progressive_codebase_course,
        created_at=now,
        updated_at=now,
        stage=CourseRunStage.published,
        status=CourseRunStatus.published,
        lab_tutor_enabled=lab_tutor_enabled,
    )


def _fake_session() -> LearnerWorkspaceSession:
    now = datetime.now(UTC)
    return LearnerWorkspaceSession(
        id="studio_cert",
        enrollment_id="publish_cert_snap_cert_test",
        deliverable_id="deliverable_1",
        scope=LearnerWorkspaceScope.shared_course,
        created_at=now,
        updated_at=now,
        status=LearnerWorkspaceSessionStatus.running,
        workspace_root="/tmp/cert",
    )


def _make_learner_package_mock() -> MagicMock:
    """Minimal mock of LearnerCoursePackage that satisfies certify_snapshot's checks."""
    mock_deliverable = MagicMock()
    mock_deliverable.deliverable_id = "deliverable_1"

    mock_pkg = MagicMock()
    mock_pkg.deliverables = [mock_deliverable]
    mock_pkg.workspace_scope = LearnerWorkspaceScope.shared_course
    return mock_pkg


def _make_task_agent_spec_mock() -> MagicMock:
    """Minimal mock of TaskAgentServiceSpec with non-partial starter."""
    mock_spec = MagicMock()
    mock_spec.runtime_dependencies.starter_type.value = "empty"
    return mock_spec


class PublishCertLabTutorStoreResolutionTests(unittest.TestCase):
    """Tests that the store lookup → lab_tutor_enabled forwarding works correctly.

    We patch all of the expensive side effects so certification can reach
    the launch_editor call, then assert on the kwarg value.
    """

    def _run_certify_with_store(
        self,
        *,
        course_run: CourseRun,
        store: MagicMock | None,
    ) -> dict:
        """Run certify_snapshot and return the kwargs received by launch_editor."""
        snapshot = _make_snapshot(course_run.id if course_run is not None else "course_test")
        snapshot = snapshot.model_copy(
            update={
                "learner_package": _make_learner_package_mock(),
                "task_agent_spec": _make_task_agent_spec_mock(),
            }
        )

        service = PublishLearnerCertificationService(enabled=True, store=store)

        launch_kwargs_received: dict = {}

        def fake_launch(**kwargs):
            launch_kwargs_received.update(kwargs)
            return _fake_session()

        grade_result = MagicMock()
        grade_result.assignment_report.status.value = "failed"
        grade_result.assignment_report.passed_tests = 0
        grade_result.assignment_report.total_tests = 1
        grade_result.assignment_report.review_areas = [
            MagicMock(deliverable_id="deliverable_1", grade_report=MagicMock())
        ]

        validation_result = MagicMock()
        validation_result.valid = True

        from app.domain.grading import GradeStatus
        remap_result = MagicMock()
        remap_result.status = GradeStatus.failed
        remap_result.passed_tests = 0
        remap_result.total_tests = 1
        remap_result.review_areas = [MagicMock(deliverable_id="deliverable_1")]

        with (
            patch("app.services.publish_learner_certification_service.seed_workspace_from_snapshot"),
            patch(
                "app.services.publish_learner_certification_service.validate_seeded_learner_workspace",
                return_value=validation_result,
            ),
            patch(
                "app.services.publish_learner_certification_service.project_brief_markdown",
                return_value="# Brief\n",
            ),
            patch(
                "app.services.publish_learner_certification_service.remap_assignment_report_to_deliverables",
                return_value=remap_result,
            ),
            # Patch path existence check for required_paths.
            patch(
                "app.services.publish_learner_certification_service.Path.exists",
                return_value=True,
            ),
            patch.object(service.learner_studio_service, "launch_editor", side_effect=fake_launch),
            patch.object(service.learner_studio_service, "grade_assignment", return_value=grade_result),
            patch.object(service.learner_studio_service, "stop_editor"),
        ):
            service.certify_snapshot(snapshot)

        return launch_kwargs_received

    def test_store_is_queried_and_lab_tutor_enabled_true_forwarded(self) -> None:
        """When the store returns a course run with lab_tutor_enabled=True,
        certify_snapshot must pass lab_tutor_enabled=True to launch_editor.
        """
        course_run = _make_course_run("course_tutor_on", lab_tutor_enabled=True)
        mock_store = MagicMock()
        mock_store.get_course_run.return_value = course_run

        kwargs = self._run_certify_with_store(course_run=course_run, store=mock_store)

        mock_store.get_course_run.assert_called_with("course_tutor_on")
        self.assertTrue(
            kwargs.get("lab_tutor_enabled"),
            f"Expected lab_tutor_enabled=True, got kwargs={kwargs}",
        )

    def test_no_store_defaults_lab_tutor_disabled(self) -> None:
        """Without a store, certify_snapshot defaults lab_tutor_enabled to False."""
        course_run = _make_course_run("course_tutor_off", lab_tutor_enabled=False)

        kwargs = self._run_certify_with_store(course_run=course_run, store=None)

        self.assertFalse(
            kwargs.get("lab_tutor_enabled"),
            f"Expected lab_tutor_enabled=False with no store, got kwargs={kwargs}",
        )

    def test_store_returning_none_defaults_lab_tutor_disabled(self) -> None:
        """When the store returns None (course_run not found), lab_tutor_enabled
        must default to False rather than raising an AttributeError.
        """
        course_run = _make_course_run("course_missing", lab_tutor_enabled=True)
        mock_store = MagicMock()
        mock_store.get_course_run.return_value = None  # Simulate missing course run.

        kwargs = self._run_certify_with_store(course_run=course_run, store=mock_store)

        self.assertFalse(
            kwargs.get("lab_tutor_enabled"),
            f"Expected lab_tutor_enabled=False when store returns None, got kwargs={kwargs}",
        )

    def test_store_is_queried_and_assignment_title_forwarded(self) -> None:
        """When the store returns a course run, certify_snapshot must pass
        lab_tutor_assignment_title=course_run.title to launch_editor.
        """
        course_run = _make_course_run("course_tutor_on", lab_tutor_enabled=True)
        mock_store = MagicMock()
        mock_store.get_course_run.return_value = course_run

        kwargs = self._run_certify_with_store(course_run=course_run, store=mock_store)

        self.assertEqual(
            kwargs.get("lab_tutor_assignment_title"),
            "Test Course",
            f"Expected lab_tutor_assignment_title='Test Course', got kwargs={kwargs}",
        )

    def test_no_store_defaults_assignment_title_to_none(self) -> None:
        """Without a store, certify_snapshot should pass None as lab_tutor_assignment_title."""
        course_run = _make_course_run("course_tutor_off", lab_tutor_enabled=False)

        kwargs = self._run_certify_with_store(course_run=course_run, store=None)

        self.assertIsNone(
            kwargs.get("lab_tutor_assignment_title"),
            f"Expected lab_tutor_assignment_title=None with no store, got kwargs={kwargs}",
        )


if __name__ == "__main__":
    unittest.main()
