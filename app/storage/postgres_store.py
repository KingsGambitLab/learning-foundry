from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import Engine, text

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
from app.storage.database import build_engine


class PostgresWorkflowStore:
    """Postgres-backed WorkflowStore.

    Method signatures mirror SQLiteWorkflowStore exactly. JSON-blob shape preserved.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or build_engine()

    def utcnow(self) -> datetime:
        return datetime.now(UTC)

    # ------------------------------------------------------------------ workflow_runs

    def save_run(self, run: WorkflowRun) -> WorkflowRun:
        payload = json.dumps(run.model_dump(mode="json"))
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO workflow_runs (
                        run_id, title, stage, status, created_at, updated_at, payload
                    ) VALUES (
                        :run_id, :title, :stage, :status, :created_at, :updated_at, CAST(:payload AS JSONB)
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        stage = EXCLUDED.stage,
                        status = EXCLUDED.status,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload
                    """
                ),
                {
                    "run_id": run.id,
                    "title": run.title,
                    "stage": run.stage.value,
                    "status": run.status.value,
                    "created_at": run.created_at.isoformat(),
                    "updated_at": run.updated_at.isoformat(),
                    "payload": payload,
                },
            )
        return run

    def get_run(self, run_id: str) -> WorkflowRun | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT payload FROM workflow_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).fetchone()
        if row is None:
            return None
        return WorkflowRun.model_validate(self._normalize_workflow_run_payload(row.payload))

    def list_runs(self, limit: int = 50) -> list[WorkflowRunSummary]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT payload
                    FROM workflow_runs
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).fetchall()
        return [
            WorkflowRunSummary.from_run(
                WorkflowRun.model_validate(self._normalize_workflow_run_payload(row.payload))
            )
            for row in rows
        ]

    def append_event(self, run_id: str, event_type: str, payload: dict) -> WorkflowEvent:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)

        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT COALESCE(MAX(sequence_no), 0) AS max_sequence FROM workflow_events WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            ).fetchone()
            sequence_no = int(row.max_sequence) + 1
            event = WorkflowEvent(
                run_id=run_id,
                sequence_no=sequence_no,
                event_type=event_type,
                created_at=run.updated_at,
                payload=payload,
            )
            conn.execute(
                text(
                    """
                    INSERT INTO workflow_events (run_id, sequence_no, event_type, created_at, payload)
                    VALUES (:run_id, :sequence_no, :event_type, :created_at, CAST(:payload AS JSONB))
                    """
                ),
                {
                    "run_id": event.run_id,
                    "sequence_no": event.sequence_no,
                    "event_type": event.event_type,
                    "created_at": event.created_at.isoformat(),
                    "payload": json.dumps(event.model_dump(mode="json")),
                },
            )
        return event

    def list_events(self, run_id: str) -> list[WorkflowEvent]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT payload
                    FROM workflow_events
                    WHERE run_id = :run_id
                    ORDER BY sequence_no ASC
                    """
                ),
                {"run_id": run_id},
            ).fetchall()
        return [WorkflowEvent.model_validate(row.payload) for row in rows]

    # ------------------------------------------------------------------ private normalization helpers

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

    # ------------------------------------------------------------------ course_runs / course_events / reset_all

    def save_course_run(self, run: CourseRun) -> CourseRun:
        payload = json.dumps(run.model_dump(mode="json"))
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO course_runs (
                        course_run_id, title, package_type, stage, status, created_at, updated_at, payload
                    ) VALUES (
                        :course_run_id, :title, :package_type, :stage, :status, :created_at, :updated_at, CAST(:payload AS JSONB)
                    )
                    ON CONFLICT (course_run_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        package_type = EXCLUDED.package_type,
                        stage = EXCLUDED.stage,
                        status = EXCLUDED.status,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload
                    """
                ),
                {
                    "course_run_id": run.id,
                    "title": run.title,
                    "package_type": run.package_type.value,
                    "stage": run.stage.value,
                    "status": run.status.value,
                    "created_at": run.created_at.isoformat(),
                    "updated_at": run.updated_at.isoformat(),
                    "payload": payload,
                },
            )
        return run

    def get_course_run(self, course_run_id: str) -> CourseRun | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT payload FROM course_runs WHERE course_run_id = :course_run_id"),
                {"course_run_id": course_run_id},
            ).fetchone()
        if row is None:
            return None
        return CourseRun.model_validate(self._normalize_course_run_payload(row.payload))

    def list_course_runs(self, limit: int = 50) -> list[CourseRunSummary]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT payload
                    FROM course_runs
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).fetchall()
        return [
            CourseRunSummary.from_run(
                CourseRun.model_validate(self._normalize_course_run_payload(row.payload))
            )
            for row in rows
        ]

    def append_course_event(self, course_run_id: str, event_type: str, payload: dict) -> CourseEvent:
        run = self.get_course_run(course_run_id)
        if run is None:
            raise KeyError(course_run_id)

        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT COALESCE(MAX(sequence_no), 0) AS max_sequence FROM course_events WHERE course_run_id = :course_run_id"
                ),
                {"course_run_id": course_run_id},
            ).fetchone()
            sequence_no = int(row.max_sequence) + 1
            event = CourseEvent(
                course_run_id=course_run_id,
                sequence_no=sequence_no,
                event_type=event_type,
                created_at=run.updated_at,
                payload=payload,
            )
            conn.execute(
                text(
                    """
                    INSERT INTO course_events (course_run_id, sequence_no, event_type, created_at, payload)
                    VALUES (:course_run_id, :sequence_no, :event_type, :created_at, CAST(:payload AS JSONB))
                    """
                ),
                {
                    "course_run_id": event.course_run_id,
                    "sequence_no": event.sequence_no,
                    "event_type": event.event_type,
                    "created_at": event.created_at.isoformat(),
                    "payload": json.dumps(event.model_dump(mode="json")),
                },
            )
        return event

    def list_course_events(self, course_run_id: str) -> list[CourseEvent]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT payload
                    FROM course_events
                    WHERE course_run_id = :course_run_id
                    ORDER BY sequence_no ASC
                    """
                ),
                {"course_run_id": course_run_id},
            ).fetchall()
        return [CourseEvent.model_validate(row.payload) for row in rows]

    def reset_all(self) -> dict[str, int]:
        tables = [
            "workflow_runs",
            "workflow_events",
            "course_runs",
            "course_events",
            "publish_snapshots",
            "learner_enrollments",
            "learner_submissions",
            "learner_workspace_sessions",
            "creator_feedback",
            "learner_feedback",
            "learner_eval_reports",
            "creator_assets",
        ]
        with self.engine.begin() as conn:
            counts = {}
            for table in tables:
                row = conn.execute(text(f"SELECT COUNT(*) AS count FROM {table}")).fetchone()
                counts[table] = int(row.count)
            conn.execute(
                text(
                    "TRUNCATE TABLE "
                    + ", ".join(tables)
                    + " RESTART IDENTITY CASCADE"
                )
            )
        return counts

    def _normalize_course_run_payload(self, payload: dict) -> dict:
        if "course_family_id" not in payload:
            payload = {
                **payload,
                "course_family_id": payload.get("id"),
            }
        return self._coerce_starter_type_recursively(payload)

    def save_creator_asset(self, asset: CreatorAssetRecord) -> CreatorAssetRecord:
        raise NotImplementedError

    def get_creator_asset(self, asset_id: str) -> CreatorAssetRecord | None:
        raise NotImplementedError

    def list_creator_assets(self, limit: int = 100) -> list[CreatorAssetRecord]:
        raise NotImplementedError

    def delete_creator_asset(self, asset_id: str) -> bool:
        raise NotImplementedError

    def save_learner_enrollment(self, enrollment: LearnerEnrollment) -> LearnerEnrollment:
        raise NotImplementedError

    def get_learner_enrollment(self, enrollment_id: str) -> LearnerEnrollment | None:
        raise NotImplementedError

    def find_learner_enrollment(self, learner_id: str, course_run_id: str) -> LearnerEnrollment | None:
        raise NotImplementedError

    def list_learner_enrollments(
        self, learner_id: str | None = None, limit: int = 50
    ) -> list[LearnerEnrollmentSummary]:
        raise NotImplementedError

    def save_learner_submission(self, submission: LearnerSubmissionRecord) -> LearnerSubmissionRecord:
        raise NotImplementedError

    def list_learner_submissions(
        self, enrollment_id: str, deliverable_id: str | None = None
    ) -> list[LearnerSubmissionRecord]:
        raise NotImplementedError

    def save_learner_workspace_session(self, session: LearnerWorkspaceSession) -> LearnerWorkspaceSession:
        raise NotImplementedError

    def list_learner_workspace_sessions(self, enrollment_id: str) -> list[LearnerWorkspaceSession]:
        raise NotImplementedError

    def list_all_learner_workspace_sessions(self) -> list[LearnerWorkspaceSession]:
        raise NotImplementedError

    def save_publish_snapshot(self, snapshot: PublishSnapshot) -> PublishSnapshot:
        raise NotImplementedError

    def get_publish_snapshot(self, snapshot_id: str) -> PublishSnapshot | None:
        raise NotImplementedError

    def list_publish_snapshots(
        self,
        course_run_id: str | None = None,
        course_family_id: str | None = None,
        limit: int = 50,
    ) -> list[PublishSnapshotSummary]:
        raise NotImplementedError

    def get_latest_publish_snapshot(
        self,
        course_run_id: str | None = None,
        course_family_id: str | None = None,
    ) -> PublishSnapshot | None:
        raise NotImplementedError

    def save_creator_feedback(self, feedback: CreatorFeedbackRecord) -> CreatorFeedbackRecord:
        raise NotImplementedError

    def list_creator_feedback(self, course_run_id: str, limit: int = 100) -> list[CreatorFeedbackRecord]:
        raise NotImplementedError

    def save_learner_feedback(self, feedback: LearnerFeedbackRecord) -> LearnerFeedbackRecord:
        raise NotImplementedError

    def list_learner_feedback(self, enrollment_id: str, limit: int = 100) -> list[LearnerFeedbackRecord]:
        raise NotImplementedError

    def save_learner_eval_report(
        self, report: LearnerCourseEvaluationReport
    ) -> LearnerCourseEvaluationReport:
        raise NotImplementedError

    def list_learner_eval_reports(
        self,
        course_run_id: str | None = None,
        publish_snapshot_id: str | None = None,
        limit: int = 100,
    ) -> list[LearnerCourseEvaluationReport]:
        raise NotImplementedError

    def get_latest_learner_eval_report(
        self,
        course_run_id: str,
        publish_snapshot_id: str | None = None,
    ) -> LearnerCourseEvaluationReport | None:
        raise NotImplementedError
