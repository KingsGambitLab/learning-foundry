from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.domain.assets import CreatorAssetRecord
from app.domain.course import CourseEvent, CourseRun, CourseRunSummary
from app.domain.learner import (
    LearnerEnrollment,
    LearnerEnrollmentSummary,
    LearnerSubmissionRecord,
    LearnerWorkspaceSession,
)
from app.domain.publish import PublishSnapshot, PublishSnapshotSummary
from app.domain.registry import StarterType
from app.domain.testing import (
    CreatorFeedbackRecord,
    LearnerCourseEvaluationReport,
    LearnerFeedbackRecord,
)
from app.domain.workflow import WorkflowEvent, WorkflowRun, WorkflowRunSummary


def default_db_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "course_gen.db"


class SQLiteWorkflowStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or default_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._session() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    run_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS course_runs (
                    course_run_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    package_type TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS course_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_run_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_enrollments (
                    enrollment_id TEXT PRIMARY KEY,
                    learner_id TEXT NOT NULL,
                    course_run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_submissions (
                    submission_id TEXT PRIMARY KEY,
                    enrollment_id TEXT NOT NULL,
                    deliverable_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_workspace_sessions (
                    session_id TEXT PRIMARY KEY,
                    enrollment_id TEXT NOT NULL,
                    deliverable_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS publish_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    course_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS creator_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    course_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    enrollment_id TEXT NOT NULL,
                    course_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_eval_reports (
                    report_id TEXT PRIMARY KEY,
                    course_run_id TEXT NOT NULL,
                    publish_snapshot_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS creator_assets (
                    asset_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def utcnow(self) -> datetime:
        return datetime.now(UTC)

    def save_run(self, run: WorkflowRun) -> WorkflowRun:
        payload = json.dumps(run.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    run_id, title, stage, status, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    title = excluded.title,
                    stage = excluded.stage,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    run.id,
                    run.title,
                    run.stage.value,
                    run.status.value,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return run

    def get_run(self, run_id: str) -> WorkflowRun | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT payload_json FROM workflow_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkflowRun.model_validate(self._normalize_workflow_run_payload(json.loads(row["payload_json"])))

    def list_runs(self, limit: int = 50) -> list[WorkflowRunSummary]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM workflow_runs
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            WorkflowRunSummary.from_run(
                WorkflowRun.model_validate(self._normalize_workflow_run_payload(json.loads(row["payload_json"])))
            )
            for row in rows
        ]

    def append_event(self, run_id: str, event_type: str, payload: dict) -> WorkflowEvent:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)

        with self._lock, self._session() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) AS max_sequence FROM workflow_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence_no = int(row["max_sequence"]) + 1
            event = WorkflowEvent(
                run_id=run_id,
                sequence_no=sequence_no,
                event_type=event_type,
                created_at=run.updated_at,
                payload=payload,
            )
            connection.execute(
                """
                INSERT INTO workflow_events (run_id, sequence_no, event_type, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.sequence_no,
                    event.event_type,
                    event.created_at.isoformat(),
                    json.dumps(event.model_dump(mode="json")),
                ),
            )
            connection.commit()
        return event

    def list_events(self, run_id: str) -> list[WorkflowEvent]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM workflow_events
                WHERE run_id = ?
                ORDER BY sequence_no ASC
                """,
                (run_id,),
            ).fetchall()
        return [WorkflowEvent.model_validate(json.loads(row["payload_json"])) for row in rows]

    def save_course_run(self, run: CourseRun) -> CourseRun:
        payload = json.dumps(run.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO course_runs (
                    course_run_id, title, package_type, stage, status, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(course_run_id) DO UPDATE SET
                    title = excluded.title,
                    package_type = excluded.package_type,
                    stage = excluded.stage,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    run.id,
                    run.title,
                    run.package_type.value,
                    run.stage.value,
                    run.status.value,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return run

    def get_course_run(self, course_run_id: str) -> CourseRun | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT payload_json FROM course_runs WHERE course_run_id = ?",
                (course_run_id,),
            ).fetchone()
        if row is None:
            return None
        return CourseRun.model_validate(self._normalize_course_run_payload(json.loads(row["payload_json"])))

    def list_course_runs(self, limit: int = 50) -> list[CourseRunSummary]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM course_runs
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            CourseRunSummary.from_run(CourseRun.model_validate(self._normalize_course_run_payload(json.loads(row["payload_json"]))))
            for row in rows
        ]

    def append_course_event(self, course_run_id: str, event_type: str, payload: dict) -> CourseEvent:
        run = self.get_course_run(course_run_id)
        if run is None:
            raise KeyError(course_run_id)

        with self._lock, self._session() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) AS max_sequence FROM course_events WHERE course_run_id = ?",
                (course_run_id,),
            ).fetchone()
            sequence_no = int(row["max_sequence"]) + 1
            event = CourseEvent(
                course_run_id=course_run_id,
                sequence_no=sequence_no,
                event_type=event_type,
                created_at=run.updated_at,
                payload=payload,
            )
            connection.execute(
                """
                INSERT INTO course_events (course_run_id, sequence_no, event_type, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.course_run_id,
                    event.sequence_no,
                    event.event_type,
                    event.created_at.isoformat(),
                    json.dumps(event.model_dump(mode="json")),
                ),
            )
            connection.commit()
        return event

    def list_course_events(self, course_run_id: str) -> list[CourseEvent]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM course_events
                WHERE course_run_id = ?
                ORDER BY sequence_no ASC
                """,
                (course_run_id,),
            ).fetchall()
        return [CourseEvent.model_validate(json.loads(row["payload_json"])) for row in rows]

    def reset_all(self) -> dict[str, int]:
        with self._lock, self._session() as connection:
            workflow_run_count = int(connection.execute("SELECT COUNT(*) AS count FROM workflow_runs").fetchone()["count"])
            workflow_event_count = int(connection.execute("SELECT COUNT(*) AS count FROM workflow_events").fetchone()["count"])
            course_run_count = int(connection.execute("SELECT COUNT(*) AS count FROM course_runs").fetchone()["count"])
            course_event_count = int(connection.execute("SELECT COUNT(*) AS count FROM course_events").fetchone()["count"])
            publish_snapshot_count = int(connection.execute("SELECT COUNT(*) AS count FROM publish_snapshots").fetchone()["count"])
            learner_enrollment_count = int(connection.execute("SELECT COUNT(*) AS count FROM learner_enrollments").fetchone()["count"])
            learner_submission_count = int(connection.execute("SELECT COUNT(*) AS count FROM learner_submissions").fetchone()["count"])
            learner_workspace_session_count = int(connection.execute("SELECT COUNT(*) AS count FROM learner_workspace_sessions").fetchone()["count"])
            creator_feedback_count = int(connection.execute("SELECT COUNT(*) AS count FROM creator_feedback").fetchone()["count"])
            learner_feedback_count = int(connection.execute("SELECT COUNT(*) AS count FROM learner_feedback").fetchone()["count"])
            learner_eval_report_count = int(connection.execute("SELECT COUNT(*) AS count FROM learner_eval_reports").fetchone()["count"])
            creator_asset_count = int(connection.execute("SELECT COUNT(*) AS count FROM creator_assets").fetchone()["count"])

            connection.execute("DELETE FROM workflow_events")
            connection.execute("DELETE FROM workflow_runs")
            connection.execute("DELETE FROM course_events")
            connection.execute("DELETE FROM course_runs")
            connection.execute("DELETE FROM publish_snapshots")
            connection.execute("DELETE FROM creator_feedback")
            connection.execute("DELETE FROM learner_feedback")
            connection.execute("DELETE FROM learner_eval_reports")
            connection.execute("DELETE FROM creator_assets")
            connection.execute("DELETE FROM learner_submissions")
            connection.execute("DELETE FROM learner_workspace_sessions")
            connection.execute("DELETE FROM learner_enrollments")
            connection.commit()

        return {
            "deleted_workflow_runs": workflow_run_count,
            "deleted_workflow_events": workflow_event_count,
            "deleted_course_runs": course_run_count,
            "deleted_course_events": course_event_count,
            "deleted_publish_snapshots": publish_snapshot_count,
            "deleted_learner_enrollments": learner_enrollment_count,
            "deleted_learner_submissions": learner_submission_count,
            "deleted_learner_workspace_sessions": learner_workspace_session_count,
            "deleted_creator_feedback": creator_feedback_count,
            "deleted_learner_feedback": learner_feedback_count,
            "deleted_learner_eval_reports": learner_eval_report_count,
            "deleted_creator_assets": creator_asset_count,
        }

    def save_creator_asset(self, asset: CreatorAssetRecord) -> CreatorAssetRecord:
        payload = json.dumps(asset.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO creator_assets (asset_id, created_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    payload_json = excluded.payload_json
                """,
                (asset.id, asset.created_at.isoformat(), payload),
            )
            connection.commit()
        return asset

    def get_creator_asset(self, asset_id: str) -> CreatorAssetRecord | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT payload_json FROM creator_assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
        if row is None:
            return None
        return CreatorAssetRecord.model_validate(json.loads(row["payload_json"]))

    def list_creator_assets(self, limit: int = 100) -> list[CreatorAssetRecord]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM creator_assets
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [CreatorAssetRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def delete_creator_asset(self, asset_id: str) -> bool:
        with self._lock, self._session() as connection:
            cursor = connection.execute(
                "DELETE FROM creator_assets WHERE asset_id = ?",
                (asset_id,),
            )
            connection.commit()
        return cursor.rowcount > 0

    def save_learner_enrollment(self, enrollment: LearnerEnrollment) -> LearnerEnrollment:
        payload = json.dumps(enrollment.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO learner_enrollments (
                    enrollment_id, learner_id, course_run_id, status, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(enrollment_id) DO UPDATE SET
                    learner_id = excluded.learner_id,
                    course_run_id = excluded.course_run_id,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    enrollment.id,
                    enrollment.learner_id,
                    enrollment.course_run_id,
                    enrollment.status.value,
                    enrollment.created_at.isoformat(),
                    enrollment.updated_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return enrollment

    def get_learner_enrollment(self, enrollment_id: str) -> LearnerEnrollment | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT payload_json FROM learner_enrollments WHERE enrollment_id = ?",
                (enrollment_id,),
            ).fetchone()
        if row is None:
            return None
        return LearnerEnrollment.model_validate(self._normalize_learner_enrollment_payload(json.loads(row["payload_json"])))

    def find_learner_enrollment(self, learner_id: str, course_run_id: str) -> LearnerEnrollment | None:
        with self._session() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM learner_enrollments
                WHERE learner_id = ? AND course_run_id = ?
                ORDER BY datetime(created_at) DESC
                LIMIT 1
                """,
                (learner_id, course_run_id),
            ).fetchone()
        if row is None:
            return None
        return LearnerEnrollment.model_validate(self._normalize_learner_enrollment_payload(json.loads(row["payload_json"])))

    def list_learner_enrollments(self, learner_id: str | None = None, limit: int = 50) -> list[LearnerEnrollmentSummary]:
        query = """
            SELECT payload_json
            FROM learner_enrollments
        """
        params: tuple[object, ...]
        if learner_id is not None:
            query += " WHERE learner_id = ?"
            params = (learner_id, limit)
        else:
            params = (limit,)
        query += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        with self._session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            LearnerEnrollmentSummary.from_enrollment(
                LearnerEnrollment.model_validate(self._normalize_learner_enrollment_payload(json.loads(row["payload_json"])))
            )
            for row in rows
        ]

    def save_learner_submission(self, submission: LearnerSubmissionRecord) -> LearnerSubmissionRecord:
        payload = json.dumps(submission.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO learner_submissions (
                    submission_id, enrollment_id, deliverable_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(submission_id) DO UPDATE SET
                    enrollment_id = excluded.enrollment_id,
                    deliverable_id = excluded.deliverable_id,
                    created_at = excluded.created_at,
                    payload_json = excluded.payload_json
                """,
                (
                    submission.id,
                    submission.enrollment_id,
                    submission.deliverable_id,
                    submission.created_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return submission

    def list_learner_submissions(self, enrollment_id: str, deliverable_id: str | None = None) -> list[LearnerSubmissionRecord]:
        query = """
            SELECT payload_json
            FROM learner_submissions
            WHERE enrollment_id = ?
        """
        params: tuple[object, ...]
        if deliverable_id is not None:
            query += " AND deliverable_id = ?"
            params = (enrollment_id, deliverable_id)
        else:
            params = (enrollment_id,)
        query += " ORDER BY datetime(created_at) DESC"
        with self._session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [LearnerSubmissionRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def save_learner_workspace_session(self, session: LearnerWorkspaceSession) -> LearnerWorkspaceSession:
        payload = json.dumps(session.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO learner_workspace_sessions (
                    session_id, enrollment_id, deliverable_id, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    enrollment_id = excluded.enrollment_id,
                    deliverable_id = excluded.deliverable_id,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    session.id,
                    session.enrollment_id,
                    session.deliverable_id,
                    session.updated_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return session

    def get_learner_workspace_session(self, session_id: str) -> LearnerWorkspaceSession | None:
        with self._session() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM learner_workspace_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return LearnerWorkspaceSession.model_validate(json.loads(row["payload_json"]))

    def list_learner_workspace_sessions(self, enrollment_id: str) -> list[LearnerWorkspaceSession]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM learner_workspace_sessions
                WHERE enrollment_id = ?
                ORDER BY datetime(updated_at) DESC
                """,
                (enrollment_id,),
            ).fetchall()
        return [LearnerWorkspaceSession.model_validate(json.loads(row["payload_json"])) for row in rows]

    def list_all_learner_workspace_sessions(self) -> list[LearnerWorkspaceSession]:
        """Return every workspace session across all enrollments.

        Used by `LearnerStudioService.reconcile_stale_sessions` on
        server startup to find sessions whose backing container no
        longer exists after a process restart.
        """
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM learner_workspace_sessions
                ORDER BY datetime(updated_at) DESC
                """,
            ).fetchall()
        return [LearnerWorkspaceSession.model_validate(json.loads(row["payload_json"])) for row in rows]

    def save_publish_snapshot(self, snapshot: PublishSnapshot) -> PublishSnapshot:
        payload = json.dumps(snapshot.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO publish_snapshots (
                    snapshot_id, course_run_id, created_at, version, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    course_run_id = excluded.course_run_id,
                    created_at = excluded.created_at,
                    version = excluded.version,
                    payload_json = excluded.payload_json
                """,
                (
                    snapshot.id,
                    snapshot.course_run_id,
                    snapshot.created_at.isoformat(),
                    snapshot.version,
                    payload,
                ),
            )
            connection.commit()
        return snapshot

    def get_publish_snapshot(self, snapshot_id: str) -> PublishSnapshot | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT payload_json FROM publish_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return PublishSnapshot.model_validate(self._normalize_publish_snapshot_payload(json.loads(row["payload_json"])))

    def list_publish_snapshots(
        self,
        course_run_id: str | None = None,
        course_family_id: str | None = None,
        limit: int = 50,
    ) -> list[PublishSnapshotSummary]:
        query = """
            SELECT payload_json
            FROM publish_snapshots
            ORDER BY datetime(created_at) DESC
            LIMIT ?
        """
        with self._session() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        snapshots = [
            PublishSnapshot.model_validate(self._normalize_publish_snapshot_payload(json.loads(row["payload_json"])))
            for row in rows
        ]
        if course_run_id is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.course_run_id == course_run_id]
        if course_family_id is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.course_family_id == course_family_id]
        return [PublishSnapshotSummary.from_snapshot(snapshot) for snapshot in snapshots[:limit]]

    def get_latest_publish_snapshot(
        self,
        course_run_id: str | None = None,
        course_family_id: str | None = None,
    ) -> PublishSnapshot | None:
        summaries = self.list_publish_snapshots(
            course_run_id=course_run_id,
            course_family_id=course_family_id,
            limit=500,
        )
        if not summaries:
            return None
        best = max(summaries, key=lambda item: (item.version, item.created_at))
        return self.get_publish_snapshot(best.id)

    def save_creator_feedback(self, feedback: CreatorFeedbackRecord) -> CreatorFeedbackRecord:
        payload = json.dumps(feedback.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO creator_feedback (
                    feedback_id, course_run_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(feedback_id) DO UPDATE SET
                    course_run_id = excluded.course_run_id,
                    created_at = excluded.created_at,
                    payload_json = excluded.payload_json
                """,
                (
                    feedback.id,
                    feedback.course_run_id,
                    feedback.created_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return feedback

    def list_creator_feedback(self, course_run_id: str, limit: int = 100) -> list[CreatorFeedbackRecord]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM creator_feedback
                WHERE course_run_id = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (course_run_id, limit),
            ).fetchall()
        return [CreatorFeedbackRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def save_learner_feedback(self, feedback: LearnerFeedbackRecord) -> LearnerFeedbackRecord:
        payload = json.dumps(feedback.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO learner_feedback (
                    feedback_id, enrollment_id, course_run_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(feedback_id) DO UPDATE SET
                    enrollment_id = excluded.enrollment_id,
                    course_run_id = excluded.course_run_id,
                    created_at = excluded.created_at,
                    payload_json = excluded.payload_json
                """,
                (
                    feedback.id,
                    feedback.enrollment_id,
                    feedback.course_run_id,
                    feedback.created_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return feedback

    def list_learner_feedback(self, enrollment_id: str, limit: int = 100) -> list[LearnerFeedbackRecord]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM learner_feedback
                WHERE enrollment_id = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (enrollment_id, limit),
            ).fetchall()
        return [LearnerFeedbackRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def save_learner_eval_report(self, report: LearnerCourseEvaluationReport) -> LearnerCourseEvaluationReport:
        payload = json.dumps(report.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO learner_eval_reports (
                    report_id, course_run_id, publish_snapshot_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    course_run_id = excluded.course_run_id,
                    publish_snapshot_id = excluded.publish_snapshot_id,
                    created_at = excluded.created_at,
                    payload_json = excluded.payload_json
                """,
                (
                    report.id,
                    report.course_run_id,
                    report.publish_snapshot_id,
                    report.created_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return report

    def list_learner_eval_reports(
        self,
        course_run_id: str | None = None,
        publish_snapshot_id: str | None = None,
        limit: int = 100,
    ) -> list[LearnerCourseEvaluationReport]:
        query = """
            SELECT payload_json
            FROM learner_eval_reports
        """
        clauses: list[str] = []
        params: list[object] = []
        if course_run_id is not None:
            clauses.append("course_run_id = ?")
            params.append(course_run_id)
        if publish_snapshot_id is not None:
            clauses.append("publish_snapshot_id = ?")
            params.append(publish_snapshot_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY datetime(created_at) DESC LIMIT ?"
        params.append(limit)
        with self._session() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [LearnerCourseEvaluationReport.model_validate(json.loads(row["payload_json"])) for row in rows]

    def get_latest_learner_eval_report(
        self,
        course_run_id: str,
        publish_snapshot_id: str | None = None,
    ) -> LearnerCourseEvaluationReport | None:
        reports = self.list_learner_eval_reports(
            course_run_id=course_run_id,
            publish_snapshot_id=publish_snapshot_id,
            limit=1,
        )
        return reports[0] if reports else None

    def _normalize_publish_snapshot_payload(self, payload: dict) -> dict:
        task_agent_spec = payload.get("task_agent_spec")
        if isinstance(task_agent_spec, dict):
            payload = {
                **payload,
                "task_agent_spec": self._normalize_task_agent_spec_payload(task_agent_spec),
            }
        if "course_family_id" not in payload:
            payload = {
                **payload,
                "course_family_id": payload.get("course_run_id"),
            }
        return payload

    def _normalize_workflow_run_payload(self, payload: dict) -> dict:
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, dict):
            task_agent_spec = artifacts.get("task_agent_spec")
            if isinstance(task_agent_spec, dict):
                payload = {
                    **payload,
                    "artifacts": {
                        **artifacts,
                        "task_agent_spec": self._normalize_task_agent_spec_payload(task_agent_spec),
                    },
                }
        # Pre-refactor rows may carry the legacy 4-value `starter_type` strings
        # at intake.starter_type and other nested specs. Coerce recursively.
        return self._coerce_starter_type_recursively(payload)

    @staticmethod
    def _coerce_legacy_starter_type(value: object) -> str | None:
        """Map legacy four-value starter_type strings onto the new two-value enum.

        Pre-refactor rows may carry `bare_stub`, `partial_implementation`,
        `working_buggy`, or `working_suboptimal`. Defensive read-side coerce so
        we never feed those into the new Pydantic enum.
        """
        if not isinstance(value, str):
            return None
        legacy = value.strip().lower()
        if legacy == "bare_stub":
            return "empty"
        if legacy in {"partial_implementation", "working_buggy", "working_suboptimal"}:
            return "partial"
        if legacy in {"empty", "partial"}:
            return legacy
        return None

    def _normalize_task_agent_spec_payload(self, payload: dict) -> dict:
        package_type = payload.get("package_type", "progressive_codebase_course")
        course_structure = payload.get("course_structure")
        runtime_dependencies = payload.get("runtime_dependencies")
        if isinstance(runtime_dependencies, dict):
            legacy_starter = self._coerce_legacy_starter_type(
                runtime_dependencies.get("starter_type")
            )
            if legacy_starter is not None:
                runtime_dependencies = {
                    **runtime_dependencies,
                    "starter_type": legacy_starter,
                }
        deliverables_payload = payload.get("deliverables")
        if isinstance(deliverables_payload, list):
            sanitized_deliverables = []
            for entry in deliverables_payload:
                if isinstance(entry, dict) and "starter_type" in entry:
                    entry = {key: value for key, value in entry.items() if key != "starter_type"}
                sanitized_deliverables.append(entry)
            payload = {**payload, "deliverables": sanitized_deliverables}
        capabilities = payload.get("capabilities")
        assessment_strategy = payload.get("assessment_strategy")
        project_contract = payload.get("project_contract")
        runtime_plan = payload.get("runtime_plan") or ((project_contract or {}).get("runtime_plan") or {})
        runtime_binding = ((project_contract or {}).get("runtime_binding") or {})
        legacy_production_contract = payload.get("production_contract") or {}
        editable_files = (
            (runtime_dependencies or {}).get("editable_files")
            or self._infer_editable_files(payload)
            or []
        )
        visible_fixture_files = (runtime_dependencies or {}).get("visible_fixture_files")
        if visible_fixture_files is None:
            visible_fixture_files = []
        resolved_language = (
            (runtime_dependencies or {}).get("implementation_language")
            or runtime_plan.get("implementation_language")
            or runtime_binding.get("implementation_language")
        )
        resolved_language_version = (
            (runtime_dependencies or {}).get("language_version")
            or runtime_plan.get("language_version")
        )
        resolved_framework = (
            (runtime_dependencies or {}).get("application_framework")
            or runtime_plan.get("application_framework")
            or runtime_binding.get("application_framework")
        )
        resolved_framework_version = (
            (runtime_dependencies or {}).get("framework_version")
            or runtime_plan.get("framework_version")
        )
        resolved_package_manager = (
            (runtime_dependencies or {}).get("package_manager")
            or runtime_plan.get("package_manager")
        )
        resolved_primary_database = (runtime_dependencies or {}).get("primary_database")
        resolved_primary_database_version = (runtime_dependencies or {}).get("primary_database_version")
        resolved_cache_backend = (runtime_dependencies or {}).get("cache_backend")
        resolved_cache_backend_version = (runtime_dependencies or {}).get("cache_backend_version")

        public_endpoints = payload.get("public_endpoints")
        if not isinstance(public_endpoints, list) or not public_endpoints:
            canonical_endpoints = legacy_production_contract.get("canonical_endpoints") or []
            public_endpoints = []
            for endpoint in canonical_endpoints:
                if not isinstance(endpoint, dict):
                    continue
                method = str(endpoint.get("method") or "POST").upper()
                path = str(endpoint.get("path") or "").strip()
                if not path.startswith("/"):
                    continue
                public_endpoints.append(
                    {
                        "method": method if method in {"GET", "POST", "PUT", "PATCH", "DELETE"} else "POST",
                        "path": path,
                        "required": bool(endpoint.get("required", True)),
                    }
                )
        if not any(isinstance(endpoint, dict) and endpoint.get("path") == "/health" for endpoint in public_endpoints):
            public_endpoints = [
                *public_endpoints,
                {"method": "GET", "path": "/health", "required": True},
            ]

        if project_contract is None:
            project_contract = {
                "family": "generic_backend_service",
                "system_kind": payload.get("summary") or payload.get("title") or "Generated service",
                "core_entities": [],
                "primary_read_paths": [endpoint["path"] for endpoint in public_endpoints if endpoint.get("method") == "GET"],
                "primary_write_paths": [
                    endpoint["path"]
                    for endpoint in public_endpoints
                    if endpoint.get("method") in {"POST", "PUT", "PATCH", "DELETE"}
                ],
                "invariants": ["The service keeps a stable public contract while learners implement the internals."],
                "operational_concerns": ["The generated bundle must boot, expose health, and pass visible checks."],
                "runtime_binding": {
                    "implementation_language": resolved_language,
                    "application_framework": resolved_framework,
                    "backing_services": [],
                    "seed_artifacts": [],
                    "integration_points": [],
                },
                "runtime_plan": runtime_plan,
            }

        deliverables = payload.get("deliverables")
        if isinstance(course_structure, dict) and package_type == "progressive_codebase_course":
            course_structure = {
                **course_structure,
                "shared_codebase": True,
                "workspace_scope": "shared_course_workspace",
            }

        normalized = {
            **payload,
            "course_structure": course_structure
            or {
                "package_type": package_type,
                "workspace_scope": "shared_course_workspace",
                "progression_mode": "cumulative_deliverable_gates",
                "shared_codebase": True,
            },
            "runtime_dependencies": (
                {
                    "execution_surface": "http_service",
                    "starter_type": StarterType.partial.value,
                    "implementation_language": resolved_language,
                    "language_version": resolved_language_version,
                    "application_framework": resolved_framework,
                    "framework_version": resolved_framework_version,
                    "package_manager": resolved_package_manager,
                    "editable_files": editable_files,
                    "visible_fixture_files": visible_fixture_files,
                    "primary_database": resolved_primary_database,
                    "primary_database_version": resolved_primary_database_version,
                    "cache_backend": resolved_cache_backend,
                    "cache_backend_version": resolved_cache_backend_version,
                    "tech_stack": [],
                    "local_run_command": "sh .coursegen/runtime/run.sh",
                    "visible_check_command": "sh .coursegen/runtime/check_visible.sh",
                    "preview_command": "sh .coursegen/runtime/run.sh",
                }
                if runtime_dependencies is None
                else {
                    **runtime_dependencies,
                    "starter_type": runtime_dependencies.get("starter_type") or StarterType.partial.value,
                    "implementation_language": runtime_dependencies.get("implementation_language") or resolved_language,
                    "language_version": runtime_dependencies.get("language_version") or resolved_language_version,
                    "application_framework": runtime_dependencies.get("application_framework") or resolved_framework,
                    "framework_version": runtime_dependencies.get("framework_version") or resolved_framework_version,
                    "package_manager": runtime_dependencies.get("package_manager") or resolved_package_manager,
                    "editable_files": runtime_dependencies.get("editable_files") or editable_files,
                    "visible_fixture_files": runtime_dependencies.get("visible_fixture_files") or visible_fixture_files,
                    "primary_database": runtime_dependencies.get("primary_database") or resolved_primary_database,
                    "primary_database_version": runtime_dependencies.get("primary_database_version") or resolved_primary_database_version,
                    "cache_backend": runtime_dependencies.get("cache_backend") or resolved_cache_backend,
                    "cache_backend_version": runtime_dependencies.get("cache_backend_version") or resolved_cache_backend_version,
                    "tech_stack": runtime_dependencies.get("tech_stack") or [],
                    "local_run_command": runtime_dependencies.get("local_run_command") or "sh .coursegen/runtime/run.sh",
                    "visible_check_command": runtime_dependencies.get("visible_check_command") or "sh .coursegen/runtime/check_visible.sh",
                    "preview_command": runtime_dependencies.get("preview_command") or "sh .coursegen/runtime/run.sh",
                }
            ),
            "capabilities": capabilities
            or {
                "retrieval_mode": "none",
                "answer_synthesis_required": False,
                "citations_required": False,
                "abstention_required": False,
                "tool_use_required": False,
                "traceability_required": False,
                "durable_state_required": bool(resolved_primary_database),
                "approval_flow_required": False,
            },
            "assessment_strategy": assessment_strategy
            or {
                "public_checks_required": True,
                "hidden_grader_required": True,
                "cumulative_deliverable_gates": True,
                "learner_submission_enabled": True,
            },
            "project_contract": project_contract,
            "public_endpoints": public_endpoints,
            "deliverables": deliverables or [],
        }
        return normalized

    def _infer_editable_files(self, payload: dict) -> list[str]:
        deliverables = payload.get("deliverables")
        if not isinstance(deliverables, list):
            return []
        for deliverable in deliverables:
            if not isinstance(deliverable, dict):
                continue
            starter_surface = deliverable.get("learner_starter_surface")
            if isinstance(starter_surface, dict) and starter_surface.get("primary_editable_paths"):
                return list(starter_surface["primary_editable_paths"])
            brief = deliverable.get("learner_brief")
            if isinstance(brief, dict) and brief.get("files_to_edit"):
                return list(brief["files_to_edit"])
        return []

    def _normalize_course_run_payload(self, payload: dict) -> dict:
        if "course_family_id" not in payload:
            payload = {
                **payload,
                "course_family_id": payload.get("id"),
            }
        # CourseRun payloads carry several nested specs (generated_plan,
        # shared_design_spec, design_spec, runtime_dependencies, ...) — each of
        # which may have a legacy `starter_type` string written by pre-refactor
        # rows. Walk the whole tree and coerce.
        return self._coerce_starter_type_recursively(payload)

    def _coerce_starter_type_recursively(self, payload):
        if isinstance(payload, dict):
            coerced: dict = {}
            for key, value in payload.items():
                if key == "starter_type":
                    normalized = self._coerce_legacy_starter_type(value)
                    coerced[key] = normalized if normalized is not None else value
                else:
                    coerced[key] = self._coerce_starter_type_recursively(value)
            return coerced
        if isinstance(payload, list):
            return [self._coerce_starter_type_recursively(item) for item in payload]
        return payload

    def _normalize_learner_enrollment_payload(self, payload: dict) -> dict:
        deliverables = payload.get("deliverables")
        if not isinstance(deliverables, list):
            return payload

        normalized_deliverables = []
        for deliverable in deliverables:
            if not isinstance(deliverable, dict):
                normalized_deliverables.append(deliverable)
                continue
            status = deliverable.get("status")
            normalized_deliverables.append(
                {
                    **deliverable,
                    "status": "available" if status == "locked" else status,
                }
            )

        return {
            **payload,
            "deliverables": normalized_deliverables,
        }
