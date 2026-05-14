from __future__ import annotations

import inspect

from app.storage.postgres_store import PostgresWorkflowStore
from app.storage.workflow_store import WorkflowStore


LEGACY_PUBLIC_METHOD_NAMES = {
    "utcnow",
    "save_run", "get_run", "list_runs",
    "append_event", "list_events",
    "save_course_run", "get_course_run", "list_course_runs",
    "append_course_event", "list_course_events",
    "reset_all",
    "save_creator_asset", "get_creator_asset", "list_creator_assets", "delete_creator_asset",
    "save_learner_enrollment", "get_learner_enrollment", "find_learner_enrollment", "list_learner_enrollments",
    "save_learner_submission", "list_learner_submissions",
    "save_learner_workspace_session", "list_learner_workspace_sessions", "list_all_learner_workspace_sessions",
    "save_publish_snapshot", "get_publish_snapshot", "list_publish_snapshots", "get_latest_publish_snapshot",
    "save_creator_feedback", "list_creator_feedback",
    "save_learner_feedback", "list_learner_feedback",
    "save_learner_eval_report", "list_learner_eval_reports", "get_latest_learner_eval_report",
}

AUTH_METHOD_NAMES = {
    "create_user", "get_user_by_email", "get_user_password_hash", "get_user_by_id",
    "create_user_session", "load_user_session", "revoke_user_session",
}


def test_protocol_lists_every_legacy_sqlite_method() -> None:
    protocol_methods = {
        name for name, member in inspect.getmembers(WorkflowStore, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert protocol_methods == LEGACY_PUBLIC_METHOD_NAMES | AUTH_METHOD_NAMES


def test_postgres_store_satisfies_legacy_protocol() -> None:
    store: WorkflowStore = PostgresWorkflowStore.__new__(PostgresWorkflowStore)
    for method in LEGACY_PUBLIC_METHOD_NAMES:
        assert callable(getattr(store, method, None)), f"{method} missing"
