from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.learner import LearnerEnrollment, LearnerEnrollmentStatus, LearnerWorkspaceScope
from app.domain.registry import PackageType
from app.services.lms_service import LMSService


def _make_enrollment() -> LearnerEnrollment:
    return LearnerEnrollment(
        id="enr_abc",
        learner_id=str(uuid4()),
        course_run_id="course_x",
        publish_snapshot_id="snap_x",
        course_title="Title",
        course_summary="Summary",
        package_type=PackageType.progressive_codebase_course,
        shared_workflow_run_id="wf_42",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        status=LearnerEnrollmentStatus.active,
        workspace_scope=LearnerWorkspaceScope.shared_course,
        deliverables=[],
    )


def test_workspace_root_uses_user_id_and_assignment_id(tmp_path: Path) -> None:
    service = LMSService.__new__(LMSService)
    service.base_dir = tmp_path
    enrollment = _make_enrollment()
    root = service._workspace_root(enrollment)
    assert root == tmp_path / enrollment.learner_id / "wf_42" / "workspace"
