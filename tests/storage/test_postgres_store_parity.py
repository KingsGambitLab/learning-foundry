from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine

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
