"""Tests that LMSService.launch_workspace passes lab_tutor_enabled from the CourseRun.

For a course run with lab_tutor_enabled=True, the studio launch call must
receive lab_tutor_enabled=True; likewise for False.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from app.domain.course import (
    CourseRun,
    CourseRunStage,
    CourseRunStatus,
)
from app.domain.learner import (
    LaunchWorkspaceRequest,
    LearnerDeliverableProgress,
    LearnerDeliverableStatus,
    LearnerEnrollment,
    LearnerEnrollmentStatus,
    LearnerWorkspaceScope,
    LearnerWorkspaceSession,
    LearnerWorkspaceSessionStatus,
)
from app.domain.registry import PackageType


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


def _make_enrollment(enrollment_id: str, course_run_id: str) -> LearnerEnrollment:
    now = datetime.now(UTC)
    deliverable = LearnerDeliverableProgress(
        deliverable_id="deliverable_1",
        title="D1",
        objective="o1",
        status=LearnerDeliverableStatus.available,
        deliverable_index=0,
    )
    return LearnerEnrollment(
        id=enrollment_id,
        learner_id="local-learner",
        course_run_id=course_run_id,
        publish_snapshot_id="snap_test",
        course_title="Test Course",
        course_summary="A test course.",
        package_type=PackageType.progressive_codebase_course,
        shared_workflow_run_id="shared_run_test",
        created_at=now,
        updated_at=now,
        status=LearnerEnrollmentStatus.active,
        workspace_scope=LearnerWorkspaceScope.shared_course,
        current_deliverable_id="deliverable_1",
        deliverables=[deliverable],
    )


def _make_session(workspace_root: str) -> LearnerWorkspaceSession:
    now = datetime.now(UTC)
    return LearnerWorkspaceSession(
        id="studio_test",
        enrollment_id="enrollment_test",
        deliverable_id="deliverable_1",
        scope=LearnerWorkspaceScope.shared_course,
        created_at=now,
        updated_at=now,
        status=LearnerWorkspaceSessionStatus.running,
        workspace_root=workspace_root,
    )


class LMSServiceLabTutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_launch_workspace(
        self,
        *,
        course_run: CourseRun,
        enrollment: LearnerEnrollment,
    ) -> MagicMock:
        """Run launch_workspace with all deep dependencies patched away.

        Returns the mock_studio so callers can assert on its launch_editor call.
        """
        from app.services.lms_service import LMSService

        mock_store = MagicMock()
        mock_store.get_course_run.return_value = course_run
        mock_store.list_learner_workspace_sessions.return_value = []

        mock_studio = MagicMock()
        session = _make_session(str(self.workspace_root / enrollment.id / "workspace"))
        mock_studio.launch_editor.return_value = session

        mock_workflow = MagicMock()

        svc = LMSService(
            store=mock_store,
            workflow_service=mock_workflow,
            learner_studio_service=mock_studio,
            base_dir=self.workspace_root,
        )

        deliverable = enrollment.deliverables[0]
        workspace_path = self.workspace_root / enrollment.id / "workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(svc, "_workspace_context", return_value=(enrollment, deliverable, MagicMock(), workspace_path)),
            patch.object(svc, "get_enrollment", return_value=enrollment),
        ):
            svc.launch_workspace(enrollment.id, LaunchWorkspaceRequest())

        return mock_studio

    def test_launch_workspace_passes_lab_tutor_enabled_true(self) -> None:
        course_run = _make_course_run("course_enabled", lab_tutor_enabled=True)
        enrollment = _make_enrollment("enrollment_a", "course_enabled")

        mock_studio = self._run_launch_workspace(course_run=course_run, enrollment=enrollment)

        mock_studio.launch_editor.assert_called_once()
        call_kwargs = mock_studio.launch_editor.call_args.kwargs
        self.assertTrue(
            call_kwargs.get("lab_tutor_enabled"),
            "Expected lab_tutor_enabled=True for an enabled course run",
        )

    def test_launch_workspace_passes_lab_tutor_enabled_false(self) -> None:
        course_run = _make_course_run("course_disabled", lab_tutor_enabled=False)
        enrollment = _make_enrollment("enrollment_b", "course_disabled")

        mock_studio = self._run_launch_workspace(course_run=course_run, enrollment=enrollment)

        mock_studio.launch_editor.assert_called_once()
        call_kwargs = mock_studio.launch_editor.call_args.kwargs
        self.assertFalse(
            call_kwargs.get("lab_tutor_enabled"),
            "Expected lab_tutor_enabled=False for a disabled course run",
        )

    def test_launch_workspace_passes_assignment_title(self) -> None:
        course_run = _make_course_run("course_with_title", lab_tutor_enabled=True)
        enrollment = _make_enrollment("enrollment_c", "course_with_title")

        mock_studio = self._run_launch_workspace(course_run=course_run, enrollment=enrollment)

        mock_studio.launch_editor.assert_called_once()
        call_kwargs = mock_studio.launch_editor.call_args.kwargs
        self.assertEqual(
            call_kwargs.get("lab_tutor_assignment_title"),
            "Test Course",
            "Expected assignment title to be passed from CourseRun.title",
        )


if __name__ == "__main__":
    unittest.main()
