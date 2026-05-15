"""Tests that apply_generated_plan preserves lab_tutor_enabled across async rebuilds.

When a CourseRun is regenerated via apply_generated_plan (called during async
background generation), the lab_tutor_enabled flag set on the existing row must
be carried forward to the rebuilt row — not silently reset to the default False.

Strategy: patch _build_course_run so we can focus on the preservation logic
in apply_generated_plan that copies fields from the existing run.
"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from app.domain.ai import AIUsageSummary
from app.domain.course import (
    CourseAsyncOperation,
    CourseDeliverableDraft,
    CourseGenerationSource,
    CourseGenerationStatus,
    CourseRun,
    CourseRunStage,
    CourseRunStatus,
    GeneratedCoursePlan,
    CreateCourseDeliverableRequest,
)
from app.domain.registry import PackageType


def _make_course_run(run_id: str, *, lab_tutor_enabled: bool) -> CourseRun:
    now = datetime.now(UTC)
    return CourseRun(
        id=run_id,
        course_family_id=run_id,
        title="Original Title",
        summary="Original summary.",
        package_type=PackageType.survey_course,
        created_at=now,
        updated_at=now,
        stage=CourseRunStage.drafting,
        status=CourseRunStatus.active,
        lab_tutor_enabled=lab_tutor_enabled,
    )


def _make_plan() -> GeneratedCoursePlan:
    return GeneratedCoursePlan(
        title="Regenerated Title",
        summary="Regenerated summary.",
        package_type=PackageType.survey_course,
        deliverables=[
            CreateCourseDeliverableRequest(
                title="Deliverable 1",
                summary="First deliverable.",
            )
        ],
    )


def _make_generation_status() -> CourseGenerationStatus:
    return CourseGenerationStatus(
        provider="test",
        available=True,
        source=CourseGenerationSource.deterministic_fallback,
        message="Test generation.",
    )


class ApplyGeneratedPlanPreservesLabTutorTests(unittest.TestCase):
    def _make_service_and_fake_build(self, existing_run: CourseRun):
        """Return (service, fake_built_run, saved_runs_list).

        _build_course_run is patched to return a fresh CourseRun without
        lab_tutor_enabled set (default=False) so we can verify apply_generated_plan
        explicitly copies it from `existing`.
        """
        from app.services.course_workflow_service import CourseWorkflowService

        mock_store = MagicMock()
        mock_store.get_course_run.return_value = existing_run

        saved: list[CourseRun] = []
        mock_store.save_course_run.side_effect = saved.append
        mock_store.append_course_event.return_value = None

        mock_workflow_service = MagicMock()

        svc = CourseWorkflowService(
            store=mock_store,
            workflow_service=mock_workflow_service,
        )

        # A fresh CourseRun produced by _build_course_run — without lab_tutor_enabled
        # set, so it defaults to False. apply_generated_plan must overwrite this.
        built_run = _make_course_run(existing_run.id, lab_tutor_enabled=False)
        built_run.title = "Regenerated Title"
        # Add a minimal deliverable so generated_plan_from_run doesn't fail.
        built_run.deliverables = [
            CourseDeliverableDraft(
                deliverable_slug="deliverable_1",
                title="Deliverable 1",
                summary="First deliverable.",
            )
        ]

        return svc, built_run, saved

    def test_apply_generated_plan_preserves_lab_tutor_enabled_true(self) -> None:
        """When the existing CourseRun has lab_tutor_enabled=True, the rebuilt
        run after apply_generated_plan must also have lab_tutor_enabled=True.
        """
        existing = _make_course_run("course_tutor_on", lab_tutor_enabled=True)
        svc, built_run, saved = self._make_service_and_fake_build(existing)

        plan = _make_plan()
        generation_status = _make_generation_status()

        with patch.object(svc, "_build_course_run", return_value=built_run):
            rebuilt = svc.apply_generated_plan(
                existing.id,
                plan=plan,
                source=CourseGenerationSource.deterministic_fallback,
                generation_status=generation_status,
            )

        self.assertTrue(
            rebuilt.lab_tutor_enabled,
            "apply_generated_plan must preserve lab_tutor_enabled=True from the existing run",
        )
        self.assertTrue(len(saved) > 0, "Course run must have been saved")
        self.assertTrue(saved[-1].lab_tutor_enabled)

    def test_apply_generated_plan_preserves_lab_tutor_enabled_false(self) -> None:
        """When the existing CourseRun has lab_tutor_enabled=False (the default),
        the rebuilt run must also have lab_tutor_enabled=False.
        """
        existing = _make_course_run("course_tutor_off", lab_tutor_enabled=False)
        svc, built_run, saved = self._make_service_and_fake_build(existing)
        built_run = built_run.model_copy(update={"lab_tutor_enabled": False})

        plan = _make_plan()
        generation_status = _make_generation_status()

        with patch.object(svc, "_build_course_run", return_value=built_run):
            rebuilt = svc.apply_generated_plan(
                existing.id,
                plan=plan,
                source=CourseGenerationSource.deterministic_fallback,
                generation_status=generation_status,
            )

        self.assertFalse(
            rebuilt.lab_tutor_enabled,
            "apply_generated_plan must preserve lab_tutor_enabled=False from the existing run",
        )


if __name__ == "__main__":
    unittest.main()
