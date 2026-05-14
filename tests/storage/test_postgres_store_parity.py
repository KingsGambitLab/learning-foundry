from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from app.domain.course import CourseRun, CourseRunStage, CourseRunStatus
from app.domain.learner import (
    LearnerEnrollment,
    LearnerEnrollmentStatus,
    LearnerWorkspaceScope,
    LearnerWorkspaceSession,
    LearnerWorkspaceSessionStatus,
)
from app.domain.registry import PackageType
from app.domain.workflow import DraftKind, WorkflowArtifacts, WorkflowRun
from app.services.assignment_design_inference import GenerationIntake
from app.storage.postgres_store import PostgresWorkflowStore

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    alembic = Path(sys.executable).parent / "alembic"
    subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": postgres_url},
        check=True,
    )


@pytest.fixture()
def store(postgres_url: str) -> PostgresWorkflowStore:
    return PostgresWorkflowStore(engine=create_engine(postgres_url))


def _make_workflow_run(run_id: str = "run_test") -> WorkflowRun:
    """Build a minimal valid WorkflowRun."""
    now = datetime.now(UTC)
    return WorkflowRun(
        id=run_id,
        title="Test run",
        stage="intake_review",
        status="active",
        created_at=now,
        updated_at=now,
        intake=GenerationIntake(
            title="Test intake",
            problem_statement="A test problem.",
        ),
        artifacts=WorkflowArtifacts(
            draft_kind=DraftKind.task_agent_spec,
        ),
    )


def test_save_and_get_run_roundtrip(store: PostgresWorkflowStore) -> None:
    run = _make_workflow_run()
    saved = store.save_run(run)
    assert saved.id == "run_test"
    fetched = store.get_run("run_test")
    assert fetched is not None
    assert fetched.title == "Test run"


def test_list_runs_orders_by_created_at_desc(store: PostgresWorkflowStore) -> None:
    older = _make_workflow_run("run_older")
    older.created_at = datetime(2026, 1, 1, tzinfo=UTC)
    newer = _make_workflow_run("run_newer")
    newer.created_at = datetime(2026, 5, 1, tzinfo=UTC)
    store.save_run(older)
    store.save_run(newer)
    summaries = store.list_runs(limit=10)
    assert [s.id for s in summaries][:2] == ["run_newer", "run_older"]


def test_append_event_assigns_monotonic_sequence(store: PostgresWorkflowStore) -> None:
    store.save_run(_make_workflow_run("run_with_events"))
    a = store.append_event("run_with_events", "stage_started", {"stage": "intake"})
    b = store.append_event("run_with_events", "stage_finished", {"stage": "intake"})
    assert (a.sequence_no, b.sequence_no) == (1, 2)


def test_list_events_returns_in_sequence_order(store: PostgresWorkflowStore) -> None:
    store.save_run(_make_workflow_run("run_for_listing"))
    store.append_event("run_for_listing", "a", {"i": 1})
    store.append_event("run_for_listing", "b", {"i": 2})
    events = store.list_events("run_for_listing")
    assert [e.event_type for e in events] == ["a", "b"]


def _make_course_run(course_run_id: str = "course_test") -> CourseRun:
    """Build a minimal valid CourseRun."""
    now = datetime.now(UTC)
    return CourseRun(
        id=course_run_id,
        course_family_id=course_run_id,
        title="Test course",
        summary="Test summary",
        package_type=PackageType.progressive_codebase_course,
        stage=CourseRunStage.drafting,
        status=CourseRunStatus.active,
        created_at=now,
        updated_at=now,
    )


def test_save_and_get_course_run_roundtrip(store: PostgresWorkflowStore) -> None:
    course = _make_course_run()
    store.save_course_run(course)
    fetched = store.get_course_run("course_test")
    assert fetched is not None
    assert fetched.title == "Test course"


def test_list_course_runs_orders_by_created_at_desc(store: PostgresWorkflowStore) -> None:
    older = _make_course_run("course_older")
    newer = _make_course_run("course_newer")
    older.created_at = datetime(2026, 1, 1, tzinfo=UTC)
    newer.created_at = datetime(2026, 5, 1, tzinfo=UTC)
    store.save_course_run(older)
    store.save_course_run(newer)
    summaries = store.list_course_runs(limit=10)
    assert [s.id for s in summaries][:2] == ["course_newer", "course_older"]


def test_append_course_event_monotonic(store: PostgresWorkflowStore) -> None:
    store.save_course_run(_make_course_run("course_with_events"))
    a = store.append_course_event("course_with_events", "draft_created", {"a": 1})
    b = store.append_course_event("course_with_events", "draft_updated", {"b": 2})
    assert (a.sequence_no, b.sequence_no) == (1, 2)


def test_list_course_events_in_order(store: PostgresWorkflowStore) -> None:
    store.save_course_run(_make_course_run("course_for_events"))
    store.append_course_event("course_for_events", "a", {"i": 1})
    store.append_course_event("course_for_events", "b", {"i": 2})
    events = store.list_course_events("course_for_events")
    assert [e.event_type for e in events] == ["a", "b"]


def test_reset_all_clears_every_table(store: PostgresWorkflowStore) -> None:
    store.save_run(_make_workflow_run("run_reset"))
    store.save_course_run(_make_course_run("course_reset"))
    counts = store.reset_all()
    # Counts dict should include every truncated table with its pre-reset row count.
    assert counts["deleted_workflow_runs"] == 1
    assert counts["deleted_course_runs"] == 1
    # After reset, queries return empty.
    assert store.get_run("run_reset") is None
    assert store.get_course_run("course_reset") is None


# ------------------------------------------------------------------ learner enrollments / submissions / sessions


def _make_enrollment(enrollment_id: str = "enr_test", learner_id: str = "learner_1") -> LearnerEnrollment:
    """Build a minimal valid LearnerEnrollment."""
    now = datetime.now(UTC)
    return LearnerEnrollment(
        id=enrollment_id,
        learner_id=learner_id,
        course_run_id="course_1",
        publish_snapshot_id="snap_1",
        course_title="Title",
        course_summary="Summary",
        package_type=PackageType.progressive_codebase_course,
        shared_workflow_run_id="wf_1",
        created_at=now,
        updated_at=now,
        status=LearnerEnrollmentStatus.active,
        workspace_scope=LearnerWorkspaceScope.shared_course,
        deliverables=[],
    )


def test_save_and_get_enrollment(store: PostgresWorkflowStore) -> None:
    enrollment = _make_enrollment()
    store.save_learner_enrollment(enrollment)
    fetched = store.get_learner_enrollment("enr_test")
    assert fetched is not None
    assert fetched.learner_id == "learner_1"


def test_find_enrollment_by_learner_and_course(store: PostgresWorkflowStore) -> None:
    store.save_learner_enrollment(_make_enrollment("enr_find", learner_id="learner_2"))
    found = store.find_learner_enrollment("learner_2", "course_1")
    assert found is not None
    assert found.id == "enr_find"


def test_list_enrollments_filters_by_learner(store: PostgresWorkflowStore) -> None:
    store.save_learner_enrollment(_make_enrollment("e1", learner_id="alpha"))
    store.save_learner_enrollment(_make_enrollment("e2", learner_id="beta"))
    alpha = store.list_learner_enrollments(learner_id="alpha")
    assert len(alpha) == 1 and alpha[0].id == "e1"


def test_save_and_list_workspace_sessions(store: PostgresWorkflowStore) -> None:
    store.save_learner_enrollment(_make_enrollment("enr_ws"))
    now = datetime.now(UTC)
    session = LearnerWorkspaceSession(
        id="ws_1",
        enrollment_id="enr_ws",
        deliverable_id="deliv_1",
        scope=LearnerWorkspaceScope.shared_course,
        created_at=now,
        updated_at=now,
        status=LearnerWorkspaceSessionStatus.running,
        workspace_root="/tmp/ws",
    )
    store.save_learner_workspace_session(session)
    sessions = store.list_learner_workspace_sessions("enr_ws")
    assert len(sessions) == 1 and sessions[0].id == "ws_1"
    all_sessions = store.list_all_learner_workspace_sessions()
    assert len(all_sessions) == 1 and all_sessions[0].id == "ws_1"


# ------------------------------------------------------------------ publish snapshots


from app.domain.publish import PublishSnapshot, PublishSnapshotProvenance  # noqa: E402
from app.domain.testing import (  # noqa: E402
    CreatorFeedbackRecord,
    LearnerCourseEvaluationReport,
    LearnerFeedbackRecord,
)
from app.domain.assets import CreatorAssetRecord  # noqa: E402
from app.domain.task_agent import DataSourceSpec, DataSourceKind, DataSourcePurpose  # noqa: E402


def _make_publish_snapshot(
    snapshot_id: str = "snap_test",
    course_run_id: str = "course_test",
    version: int = 1,
) -> PublishSnapshot:
    """Minimal valid PublishSnapshot (no learner_package, no task_agent_spec)."""
    now = datetime.now(UTC)
    return PublishSnapshot(
        id=snapshot_id,
        course_run_id=course_run_id,
        course_family_id=course_run_id,
        created_at=now,
        version=version,
        source_hash="abc123",
        provenance=PublishSnapshotProvenance(
            generator_version="0.1.0",
            course_run_hash="hash_x",
        ),
    )


def test_save_and_get_publish_snapshot(store: PostgresWorkflowStore) -> None:
    snapshot = _make_publish_snapshot()
    store.save_publish_snapshot(snapshot)
    fetched = store.get_publish_snapshot("snap_test")
    assert fetched is not None
    assert fetched.id == "snap_test"
    assert fetched.course_run_id == "course_test"


def test_get_latest_publish_snapshot_by_course_run(store: PostgresWorkflowStore) -> None:
    snap1 = _make_publish_snapshot("snap1", course_run_id="course_x", version=1)
    snap2 = _make_publish_snapshot("snap2", course_run_id="course_x", version=2)
    store.save_publish_snapshot(snap1)
    store.save_publish_snapshot(snap2)
    latest = store.get_latest_publish_snapshot(course_run_id="course_x")
    assert latest is not None
    assert latest.id == "snap2"


def test_list_publish_snapshots_filters_by_course_run(store: PostgresWorkflowStore) -> None:
    store.save_publish_snapshot(_make_publish_snapshot("snapA", course_run_id="cr_a"))
    store.save_publish_snapshot(_make_publish_snapshot("snapB", course_run_id="cr_b"))
    summaries = store.list_publish_snapshots(course_run_id="cr_a")
    assert len(summaries) == 1
    assert summaries[0].id == "snapA"


def test_list_publish_snapshots_filters_by_course_family(store: PostgresWorkflowStore) -> None:
    store.save_publish_snapshot(_make_publish_snapshot("snapC", course_run_id="cr_fam"))
    summaries = store.list_publish_snapshots(course_family_id="cr_fam")
    assert len(summaries) == 1
    assert summaries[0].id == "snapC"


# ------------------------------------------------------------------ creator feedback


def _make_creator_feedback(
    feedback_id: str = "cf_test",
    course_run_id: str = "course_test",
) -> CreatorFeedbackRecord:
    now = datetime.now(UTC)
    return CreatorFeedbackRecord(
        id=feedback_id,
        course_run_id=course_run_id,
        created_at=now,
        category="general",
        summary="Test feedback",
    )


def test_save_and_list_creator_feedback(store: PostgresWorkflowStore) -> None:
    fb = _make_creator_feedback()
    store.save_creator_feedback(fb)
    results = store.list_creator_feedback("course_test")
    assert len(results) == 1
    assert results[0].id == "cf_test"


def test_list_creator_feedback_filters_by_course_run(store: PostgresWorkflowStore) -> None:
    store.save_creator_feedback(_make_creator_feedback("cf1", course_run_id="cr_1"))
    store.save_creator_feedback(_make_creator_feedback("cf2", course_run_id="cr_2"))
    results = store.list_creator_feedback("cr_1")
    assert len(results) == 1 and results[0].id == "cf1"


# ------------------------------------------------------------------ learner feedback


def _make_learner_feedback(
    feedback_id: str = "lf_test",
    enrollment_id: str = "enr_test",
    course_run_id: str = "course_test",
) -> LearnerFeedbackRecord:
    now = datetime.now(UTC)
    return LearnerFeedbackRecord(
        id=feedback_id,
        enrollment_id=enrollment_id,
        course_run_id=course_run_id,
        publish_snapshot_id="snap_1",
        learner_id="learner_1",
        created_at=now,
        summary="Nice course",
    )


def test_save_and_list_learner_feedback(store: PostgresWorkflowStore) -> None:
    fb = _make_learner_feedback()
    store.save_learner_feedback(fb)
    results = store.list_learner_feedback("enr_test")
    assert len(results) == 1
    assert results[0].id == "lf_test"


def test_list_learner_feedback_filters_by_enrollment(store: PostgresWorkflowStore) -> None:
    store.save_learner_feedback(_make_learner_feedback("lf1", enrollment_id="enr_a"))
    store.save_learner_feedback(_make_learner_feedback("lf2", enrollment_id="enr_b"))
    results = store.list_learner_feedback("enr_a")
    assert len(results) == 1 and results[0].id == "lf1"


# ------------------------------------------------------------------ learner eval reports


def _make_eval_report(
    report_id: str = "rpt_test",
    course_run_id: str = "course_test",
) -> LearnerCourseEvaluationReport:
    now = datetime.now(UTC)
    return LearnerCourseEvaluationReport(
        id=report_id,
        course_run_id=course_run_id,
        publish_snapshot_id="snap_1",
        created_at=now,
        overall_status="passed",
        deliverable_results=[],
    )


def test_save_and_list_learner_eval_reports(store: PostgresWorkflowStore) -> None:
    report = _make_eval_report()
    store.save_learner_eval_report(report)
    results = store.list_learner_eval_reports(course_run_id="course_test")
    assert len(results) == 1
    assert results[0].id == "rpt_test"


def test_get_latest_learner_eval_report(store: PostgresWorkflowStore) -> None:
    r1 = _make_eval_report("rpt1", course_run_id="cr_eval")
    r2 = _make_eval_report("rpt2", course_run_id="cr_eval")
    store.save_learner_eval_report(r1)
    store.save_learner_eval_report(r2)
    latest = store.get_latest_learner_eval_report("cr_eval")
    # Should return one of the two (latest by created_at DESC)
    assert latest is not None
    assert latest.id in {"rpt1", "rpt2"}


def test_list_learner_eval_reports_by_snapshot(store: PostgresWorkflowStore) -> None:
    rpt = _make_eval_report("rpt_snap", course_run_id="cr_s")
    store.save_learner_eval_report(rpt)
    results = store.list_learner_eval_reports(publish_snapshot_id="snap_1")
    assert any(r.id == "rpt_snap" for r in results)


# ------------------------------------------------------------------ creator assets


def _make_creator_asset(asset_id: str = "asset_test") -> CreatorAssetRecord:
    now = datetime.now(UTC)
    return CreatorAssetRecord(
        id=asset_id,
        file_name="test.csv",
        title="Test Asset",
        size_bytes=42,
        workspace_path="/data/test.csv",
        created_at=now,
        data_source=DataSourceSpec(
            id="ds_1",
            kind=DataSourceKind.uploaded_file,
            title="Test DS",
            purpose=DataSourcePurpose.reference_data,
        ),
    )


def test_save_and_get_creator_asset(store: PostgresWorkflowStore) -> None:
    asset = _make_creator_asset()
    store.save_creator_asset(asset)
    fetched = store.get_creator_asset("asset_test")
    assert fetched is not None
    assert fetched.id == "asset_test"
    assert fetched.file_name == "test.csv"


def test_list_creator_assets(store: PostgresWorkflowStore) -> None:
    store.save_creator_asset(_make_creator_asset("asset1"))
    store.save_creator_asset(_make_creator_asset("asset2"))
    assets = store.list_creator_assets()
    ids = {a.id for a in assets}
    assert {"asset1", "asset2"}.issubset(ids)


def test_delete_creator_asset(store: PostgresWorkflowStore) -> None:
    store.save_creator_asset(_make_creator_asset("asset_del"))
    deleted = store.delete_creator_asset("asset_del")
    assert deleted is True
    assert store.get_creator_asset("asset_del") is None


def test_delete_creator_asset_missing_returns_false(store: PostgresWorkflowStore) -> None:
    result = store.delete_creator_asset("nonexistent_asset")
    assert result is False
