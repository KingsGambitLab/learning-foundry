"""Pin: on server startup, learner_workspace_sessions whose backing
container no longer exists must be marked `stopped`, NOT left as
`running`. Otherwise the web UI keeps showing the editor URL as active
and the learner gets a 404 when they click it.

Observed bug ("VS Code editor URL goes stale when app server restarts;
web UI keeps using it and 404s" — recorded in MEMORY.md as
`bug_editor_404_after_server_restart.md`):

  1. Learner launches editor → row added to learner_workspace_sessions
     with status=running, editor_url=http://...:54123
  2. Server restarts (uvicorn shutdown). All Docker editor containers
     are torn down by Docker's own session cleanup or remain orphaned.
  3. Server starts back up. The row in learner_workspace_sessions
     still says status=running. The web UI fetches the editor URL
     and the learner clicks it → 404 (container is gone).

Root cause: no reconciliation between the SQLite-persisted session
state and the actual Docker container state. The DB row outlives the
container.

This test pins the fix: `LearnerStudioService.reconcile_stale_sessions`
scans every session with status in {running, starting}, checks via
`docker inspect <container_name>` whether the container exists and is
running, and marks any that aren't as `stopped` (with a note explaining
the restart). Called from the FastAPI `lifespan` hook on startup.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from app.domain.learner import (
    LearnerWorkspaceScope,
    LearnerWorkspaceSession,
    LearnerWorkspaceSessionStatus,
)
from app.services.learner_studio_service import LearnerStudioService
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.storage.sqlite_store import SQLiteWorkflowStore


def _make_session(
    session_id: str,
    container_name: str,
    status: LearnerWorkspaceSessionStatus,
    editor_url: str = "http://127.0.0.1:54123",
) -> LearnerWorkspaceSession:
    now = datetime.now(UTC)
    return LearnerWorkspaceSession(
        id=session_id,
        enrollment_id="enroll_test",
        deliverable_id="deliverable_1",
        scope=LearnerWorkspaceScope.shared_course,
        created_at=now,
        updated_at=now,
        status=status,
        workspace_root="/tmp/test-workspace",
        container_name=container_name,
        host_port=54123,
        editor_url=editor_url,
        image_name="test-editor:latest",
        notes=[],
    )


class StaleEditorSessionReconciliationTests(unittest.TestCase):
    def test_reconcile_marks_session_stopped_when_container_missing(self) -> None:
        """Session row says status=running, but docker inspect reports
        the container no longer exists → session must be marked stopped.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
            session = _make_session(
                "session_stale_running",
                container_name="course-gen-editor-deadbeef",
                status=LearnerWorkspaceSessionStatus.running,
            )
            store.save_learner_workspace_session(session)

            service = LearnerStudioService(runner=TaskAgentBlackBoxRunner())

            # Mock `_container_running` to report "no such container."
            with patch.object(service, "_container_running", return_value=False):
                reconciled = service.reconcile_stale_sessions(store)

            self.assertIn("session_stale_running", reconciled)

            after = store.list_learner_workspace_sessions("enroll_test")
            self.assertEqual(len(after), 1)
            self.assertEqual(after[0].status, LearnerWorkspaceSessionStatus.stopped)

    def test_reconcile_marks_starting_sessions_stopped_too(self) -> None:
        """Sessions in `starting` are also stale across a restart — the
        editor launch task that was provisioning them is dead.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
            session = _make_session(
                "session_stale_starting",
                container_name="course-gen-editor-abc123",
                status=LearnerWorkspaceSessionStatus.starting,
            )
            store.save_learner_workspace_session(session)

            service = LearnerStudioService(runner=TaskAgentBlackBoxRunner())

            with patch.object(service, "_container_running", return_value=False):
                reconciled = service.reconcile_stale_sessions(store)

            self.assertIn("session_stale_starting", reconciled)
            after = store.list_learner_workspace_sessions("enroll_test")
            self.assertEqual(after[0].status, LearnerWorkspaceSessionStatus.stopped)

    def test_reconcile_leaves_already_stopped_sessions_alone(self) -> None:
        """Sessions already marked `stopped` or `failed` should not be
        re-touched; reconcile only acts on running/starting.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
            session = _make_session(
                "session_already_stopped",
                container_name="course-gen-editor-cafe42",
                status=LearnerWorkspaceSessionStatus.stopped,
            )
            store.save_learner_workspace_session(session)

            service = LearnerStudioService(runner=TaskAgentBlackBoxRunner())

            with patch.object(service, "_container_running", return_value=False):
                reconciled = service.reconcile_stale_sessions(store)

            self.assertNotIn("session_already_stopped", reconciled)

    def test_reconcile_preserves_running_sessions_when_container_alive(self) -> None:
        """If by some miracle the editor container survived the restart
        (e.g., Docker daemon kept it running), don't touch the session.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
            session = _make_session(
                "session_actually_running",
                container_name="course-gen-editor-aliveone",
                status=LearnerWorkspaceSessionStatus.running,
            )
            store.save_learner_workspace_session(session)

            service = LearnerStudioService(runner=TaskAgentBlackBoxRunner())

            with patch.object(service, "_container_running", return_value=True):
                reconciled = service.reconcile_stale_sessions(store)

            self.assertNotIn("session_actually_running", reconciled)
            after = store.list_learner_workspace_sessions("enroll_test")
            self.assertEqual(after[0].status, LearnerWorkspaceSessionStatus.running)

    def test_reconcile_adds_breadcrumb_note(self) -> None:
        """Reconciled sessions get a note explaining why their status
        changed, so operators see what happened in the timeline.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
            session = _make_session(
                "session_for_note",
                container_name="course-gen-editor-dead",
                status=LearnerWorkspaceSessionStatus.running,
            )
            store.save_learner_workspace_session(session)

            service = LearnerStudioService(runner=TaskAgentBlackBoxRunner())

            with patch.object(service, "_container_running", return_value=False):
                service.reconcile_stale_sessions(store)

            after = store.list_learner_workspace_sessions("enroll_test")
            self.assertTrue(
                any("restart" in note.lower() or "container" in note.lower()
                    for note in (after[0].notes or [])),
                f"Reconciled session must have a breadcrumb note. "
                f"Got: {after[0].notes!r}",
            )


if __name__ == "__main__":
    unittest.main()
