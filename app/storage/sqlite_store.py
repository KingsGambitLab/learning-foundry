from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.domain.course import CourseEvent, CourseRun, CourseRunSummary
from app.domain.learner import (
    LearnerEnrollment,
    LearnerEnrollmentSummary,
    LearnerSubmissionRecord,
    LearnerWorkspaceSession,
)
from app.domain.publish import PublishSnapshot, PublishSnapshotSummary
from app.domain.task_agent import LearnerModuleBrief
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
                    module_id TEXT NOT NULL,
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
                    module_id TEXT NOT NULL,
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
            connection.commit()

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

            connection.execute("DELETE FROM workflow_events")
            connection.execute("DELETE FROM workflow_runs")
            connection.execute("DELETE FROM course_events")
            connection.execute("DELETE FROM course_runs")
            connection.execute("DELETE FROM publish_snapshots")
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
        }

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
        return LearnerEnrollment.model_validate(json.loads(row["payload_json"]))

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
        return LearnerEnrollment.model_validate(json.loads(row["payload_json"]))

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
                LearnerEnrollment.model_validate(json.loads(row["payload_json"]))
            )
            for row in rows
        ]

    def save_learner_submission(self, submission: LearnerSubmissionRecord) -> LearnerSubmissionRecord:
        payload = json.dumps(submission.model_dump(mode="json"))
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO learner_submissions (
                    submission_id, enrollment_id, module_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(submission_id) DO UPDATE SET
                    enrollment_id = excluded.enrollment_id,
                    module_id = excluded.module_id,
                    created_at = excluded.created_at,
                    payload_json = excluded.payload_json
                """,
                (
                    submission.id,
                    submission.enrollment_id,
                    submission.module_id,
                    submission.created_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return submission

    def list_learner_submissions(self, enrollment_id: str, module_id: str | None = None) -> list[LearnerSubmissionRecord]:
        query = """
            SELECT payload_json
            FROM learner_submissions
            WHERE enrollment_id = ?
        """
        params: tuple[object, ...]
        if module_id is not None:
            query += " AND module_id = ?"
            params = (enrollment_id, module_id)
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
                    session_id, enrollment_id, module_id, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    enrollment_id = excluded.enrollment_id,
                    module_id = excluded.module_id,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json
                """,
                (
                    session.id,
                    session.enrollment_id,
                    session.module_id,
                    session.updated_at.isoformat(),
                    payload,
                ),
            )
            connection.commit()
        return session

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
        learner_package = payload.get("learner_package")
        if isinstance(learner_package, dict):
            modules = learner_package.get("modules")
            if isinstance(modules, list):
                normalized_modules = []
                for module in modules:
                    if not isinstance(module, dict) or "learner_brief" in module:
                        normalized_modules.append(module)
                        continue
                    visible_files = module.get("visible_files") or []
                    files_to_edit = ["app.py"] if "app.py" in visible_files else []
                    brief = LearnerModuleBrief(
                        why_this_module_matters=module.get("objective") or module.get("title") or "Continue the learner-visible module work.",
                        task_to_build=module.get("objective") or f"Complete {module.get('title') or 'this module'}.",
                        files_to_edit=files_to_edit,
                        definition_of_done=[module.get("completion_rule") or f"Complete {module.get('title') or 'this module'}."],
                        example_scenarios=[],
                        implementation_hints=["Read `README.md` and `module_content.md` before editing the starter files."],
                        non_goals=[],
                    )
                    normalized_modules.append({**module, "learner_brief": brief.model_dump(mode="json")})
                payload = {
                    **payload,
                    "learner_package": {
                        **learner_package,
                        "modules": normalized_modules,
                    },
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
        return payload

    def _normalize_task_agent_spec_payload(self, payload: dict) -> dict:
        package_type = payload.get("package_type", "progressive_codebase_course")
        course_structure = payload.get("course_structure")
        runtime_dependencies = payload.get("runtime_dependencies")
        capabilities = payload.get("capabilities")
        assessment_strategy = payload.get("assessment_strategy")
        production_contract = payload.get("production_contract") or {}
        tool_registry = payload.get("tool_registry") or {}
        tools = tool_registry.get("tools") or []
        editable_files = (
            (runtime_dependencies or {}).get("editable_files")
            or self._infer_editable_files(payload)
            or ["app.py"]
        )
        visible_fixture_files = (runtime_dependencies or {}).get("visible_fixture_files")
        if visible_fixture_files is None:
            eval_cases = payload.get("eval_dataset", {}).get("cases", [])
            has_retrieval_fixture = any(
                isinstance(case, dict) and "retrieval" in " ".join(str(tag) for tag in case.get("tags", [])).lower()
                for case in eval_cases
            )
            visible_fixture_files = ["data/corpus.json"] if has_retrieval_fixture else []

        normalized = {
            **payload,
            "course_structure": course_structure
            or {
                "package_type": package_type,
                "workspace_scope": "shared_course_workspace",
                "progression_mode": "cumulative_module_gates",
                "shared_codebase": True,
            },
            "runtime_dependencies": runtime_dependencies
            or {
                "execution_surface": "http_service",
                "editable_files": editable_files,
                "visible_fixture_files": visible_fixture_files,
                "local_run_command": "python -m uvicorn app:app --host 127.0.0.1 --port 8000",
                "visible_check_command": "python checks/run_visible_checks.py",
                "preview_command": "python -m uvicorn app:app --host 127.0.0.1 --port 8000",
            },
            "capabilities": capabilities
            or {
                "retrieval_mode": "none",
                "answer_synthesis_required": False,
                "citations_required": False,
                "abstention_required": False,
                "tool_use_required": bool(tools),
                "traceability_required": True,
                "durable_state_required": bool(production_contract.get("supports_resume")),
                "approval_flow_required": any(tool.get("approval_required") for tool in tools),
            },
            "assessment_strategy": assessment_strategy
            or {
                "public_checks_required": True,
                "hidden_grader_required": True,
                "cumulative_module_gates": True,
                "learner_submission_enabled": True,
            },
        }
        return normalized

    def _infer_editable_files(self, payload: dict) -> list[str]:
        modules = payload.get("modules")
        if not isinstance(modules, list):
            return []
        for module in modules:
            if not isinstance(module, dict):
                continue
            brief = module.get("learner_brief")
            if isinstance(brief, dict) and brief.get("files_to_edit"):
                return list(brief["files_to_edit"])
        return []

    def _normalize_course_run_payload(self, payload: dict) -> dict:
        if "course_family_id" not in payload:
            payload = {
                **payload,
                "course_family_id": payload.get("id"),
            }
        return payload
