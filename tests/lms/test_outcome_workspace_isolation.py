"""Two outcome-mode enrollments by the same learner must not share a workspace.

Regression test for P0 #3 from the codex review: prior code defaulted
`shared_workflow_run_id` to the literal string `"shared_workflow"` when both
the snapshot and the course run had no real workflow run id, which made
every outcome-mode enrollment for a single learner collide at
`learner_workspaces/<user>/shared_workflow/workspace`. The second enrollment
would silently inherit the first course's starter and `.coursegen` metadata
because seeding skips existing files.

The fix in `lms_service.py:528` is to fall back to `course_run.id` instead
of the literal `"shared_workflow"`. This test pins that — the workspace key
is now course-unique even when the snapshot lacks a workflow run id.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.learner import (
    LearnerEnrollment,
    LearnerEnrollmentStatus,
    LearnerWorkspaceScope,
)
from app.domain.registry import PackageType
from app.services.lms_service import LMSService


def _make_enrollment(course_id: str, learner_id: str, run_id: str) -> LearnerEnrollment:
    now = datetime.now(UTC)
    return LearnerEnrollment(
        id=f"enr_{uuid4().hex[:8]}",
        learner_id=learner_id,
        course_run_id=course_id,
        publish_snapshot_id=f"snap_{course_id}",
        course_title=f"Title for {course_id}",
        course_summary=f"Summary for {course_id}",
        package_type=PackageType.progressive_codebase_course,
        shared_workflow_run_id=run_id,
        created_at=now,
        updated_at=now,
        status=LearnerEnrollmentStatus.active,
        workspace_scope=LearnerWorkspaceScope.shared_course,
        deliverables=[],
    )


def test_two_courses_for_same_learner_resolve_to_distinct_workspaces(tmp_path: Path) -> None:
    service = LMSService.__new__(LMSService)
    service.base_dir = tmp_path

    learner_id = str(uuid4())
    # Both enrollments use a course-unique run id (the post-fix behavior:
    # `lms_service.enroll` falls back to course_run.id when both snapshot
    # and course_run lack a real workflow run id).
    enr_a = _make_enrollment("course_aaa", learner_id, run_id="course_aaa")
    enr_b = _make_enrollment("course_bbb", learner_id, run_id="course_bbb")

    root_a = service._workspace_root(enr_a)
    root_b = service._workspace_root(enr_b)

    assert root_a != root_b, (
        f"Outcome-mode enrollments for the same learner must not share a workspace. "
        f"Got both at {root_a}"
    )
    # The literal `"shared_workflow"` fallback string must not appear in either path.
    assert "shared_workflow" not in str(root_a)
    assert "shared_workflow" not in str(root_b)


def test_legacy_shared_workflow_fallback_would_collide(tmp_path: Path) -> None:
    """Documents what the pre-fix code did, so the regression is unambiguous."""
    service = LMSService.__new__(LMSService)
    service.base_dir = tmp_path

    learner_id = str(uuid4())
    # If we revert to the old fallback (both enrollments share
    # `shared_workflow_run_id="shared_workflow"`), the paths collide.
    enr_a = _make_enrollment("course_aaa", learner_id, run_id="shared_workflow")
    enr_b = _make_enrollment("course_bbb", learner_id, run_id="shared_workflow")
    assert service._workspace_root(enr_a) == service._workspace_root(enr_b)
