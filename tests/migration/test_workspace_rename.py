from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from scripts.migrate_sqlite_to_postgres import (
    copy_to_postgres,
    ensure_seed_user,
    rename_workspaces,
)


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
def seeded(postgres_url: str, tmp_path: Path) -> tuple[str, Path]:
    """Seed Postgres with one enrollment for the seed learner."""
    snapshot = tmp_path / "snap.db"
    with sqlite3.connect(str(snapshot)) as conn:
        conn.executescript("""
            CREATE TABLE workflow_runs (run_id TEXT PRIMARY KEY, title TEXT, stage TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE workflow_events (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, sequence_no INTEGER, event_type TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE course_runs (course_run_id TEXT PRIMARY KEY, title TEXT, package_type TEXT, stage TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE course_events (id INTEGER PRIMARY KEY AUTOINCREMENT, course_run_id TEXT, sequence_no INTEGER, event_type TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE learner_enrollments (enrollment_id TEXT PRIMARY KEY, learner_id TEXT, course_run_id TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE learner_submissions (submission_id TEXT PRIMARY KEY, enrollment_id TEXT, deliverable_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE learner_workspace_sessions (session_id TEXT PRIMARY KEY, enrollment_id TEXT, deliverable_id TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE publish_snapshots (snapshot_id TEXT PRIMARY KEY, course_run_id TEXT, created_at TEXT, version INTEGER, payload_json TEXT);
            CREATE TABLE creator_feedback (feedback_id TEXT PRIMARY KEY, course_run_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE learner_feedback (feedback_id TEXT PRIMARY KEY, enrollment_id TEXT, course_run_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE learner_eval_reports (report_id TEXT PRIMARY KEY, course_run_id TEXT, publish_snapshot_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE creator_assets (asset_id TEXT PRIMARY KEY, created_at TEXT, payload_json TEXT);
        """)
        payload = json.dumps({
            "id": "enr_1",
            "learner_id": "local-learner",
            "course_run_id": "c1",
            "shared_workflow_run_id": "wf_999",
        })
        conn.execute(
            "INSERT INTO learner_enrollments VALUES ('enr_1','local-learner','c1','active','2026-01-01','2026-01-01',?)",
            (payload,),
        )
    engine = create_engine(postgres_url)
    seed = ensure_seed_user(engine)
    copy_to_postgres(snapshot=snapshot, engine=engine, seed_learner_id=seed)
    return str(seed), tmp_path


def test_rename_walks_enrollments_and_copies_dirs(postgres_url: str, seeded) -> None:
    seed_str, tmp_path = seeded
    old_base = tmp_path / "old_learner_workspaces"
    new_base = tmp_path / "new_learner_workspaces"

    # Create a fake old-layout workspace
    old_layout = old_base / "enr_1" / "workspace"
    old_layout.mkdir(parents=True)
    (old_layout / "marker.txt").write_text("hello")

    engine = create_engine(postgres_url)
    rename_workspaces(engine=engine, old_base=old_base, new_base=new_base)

    new_layout = new_base / seed_str / "wf_999" / "workspace"
    assert new_layout.exists()
    assert (new_layout / "marker.txt").read_text() == "hello"

    # Source must be left in place (copy, not move)
    assert old_layout.exists()


def test_rename_is_idempotent(postgres_url: str, seeded) -> None:
    seed_str, tmp_path = seeded
    old_base = tmp_path / "old"
    new_base = tmp_path / "new"

    old_layout = old_base / "enr_1" / "workspace"
    old_layout.mkdir(parents=True)
    (old_layout / "marker.txt").write_text("hello")

    engine = create_engine(postgres_url)
    rename_workspaces(engine=engine, old_base=old_base, new_base=new_base)
    # second run is a no-op
    rename_workspaces(engine=engine, old_base=old_base, new_base=new_base)

    new_layout = new_base / seed_str / "wf_999" / "workspace"
    assert (new_layout / "marker.txt").read_text() == "hello"


def test_rename_skips_missing_old_paths(postgres_url: str, seeded) -> None:
    seed_str, tmp_path = seeded
    old_base = tmp_path / "no_workspaces_here"
    new_base = tmp_path / "new"
    engine = create_engine(postgres_url)
    # Should not raise even though old_base is empty / does not exist
    rename_workspaces(engine=engine, old_base=old_base, new_base=new_base)
    assert not (new_base / seed_str).exists()
