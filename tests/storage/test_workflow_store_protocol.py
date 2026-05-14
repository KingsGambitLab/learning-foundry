from __future__ import annotations

import inspect

from app.storage.sqlite_store import SQLiteWorkflowStore
from app.storage.workflow_store import WorkflowStore


PUBLIC_METHOD_NAMES = {
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


def test_protocol_lists_every_public_sqlite_method() -> None:
    protocol_methods = {
        name for name, member in inspect.getmembers(WorkflowStore, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert protocol_methods == PUBLIC_METHOD_NAMES


def test_sqlite_store_satisfies_protocol() -> None:
    store: WorkflowStore = SQLiteWorkflowStore.__new__(SQLiteWorkflowStore)
    for method in PUBLIC_METHOD_NAMES:
        assert callable(getattr(store, method, None)), f"{method} missing"
