"""Smoke-test that CourseRun.lab_tutor_enabled round-trips through the store.

The toggle no longer reaches launch_editor (the launcher was reverted); it now
gates page-level widget visibility. This test confirms the field persists and
is readable after a save/load cycle, which is all the production code now needs.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from app.domain.course import CourseRun, CourseRunStage, CourseRunStatus
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


class CourseRunLabTutorToggleTests(unittest.TestCase):
    def test_lab_tutor_enabled_true_round_trips(self) -> None:
        run = _make_course_run("run_enabled", lab_tutor_enabled=True)
        dumped = run.model_dump()
        restored = CourseRun.model_validate(dumped)
        self.assertTrue(restored.lab_tutor_enabled)

    def test_lab_tutor_enabled_false_round_trips(self) -> None:
        run = _make_course_run("run_disabled", lab_tutor_enabled=False)
        dumped = run.model_dump()
        restored = CourseRun.model_validate(dumped)
        self.assertFalse(restored.lab_tutor_enabled)

    def test_lab_tutor_enabled_defaults_to_false(self) -> None:
        now = datetime.now(UTC)
        run = CourseRun(
            id="run_default",
            course_family_id="run_default",
            title="Default Course",
            summary="Default.",
            package_type=PackageType.progressive_codebase_course,
            created_at=now,
            updated_at=now,
            stage=CourseRunStage.published,
            status=CourseRunStatus.published,
        )
        self.assertFalse(run.lab_tutor_enabled)


if __name__ == "__main__":
    unittest.main()
