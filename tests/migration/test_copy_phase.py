from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from scripts.migrate_sqlite_to_postgres import copy_to_postgres, ensure_seed_user

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
def snapshot_file(tmp_path: Path) -> Path:
    """Minimal SQLite snapshot: one workflow_run, one enrollment owned by local-learner."""
    path = tmp_path / "snapshot.db"
    with sqlite3.connect(str(path)) as conn:
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
        payload = json.dumps({"id": "enr_1", "learner_id": "local-learner", "course_run_id": "c1"})
        conn.execute(
            "INSERT INTO learner_enrollments VALUES ('enr_1', 'local-learner', 'c1', 'active', '2026-05-01', '2026-05-01', ?)",
            (payload,),
        )
    return path


def test_ensure_seed_user_is_idempotent(postgres_url: str) -> None:
    engine = create_engine(postgres_url)
    a = ensure_seed_user(engine)
    b = ensure_seed_user(engine)
    assert a == b


def test_copy_rewrites_local_learner(postgres_url: str, snapshot_file: Path) -> None:
    engine = create_engine(postgres_url)
    seed_id = ensure_seed_user(engine)
    copy_to_postgres(snapshot=snapshot_file, engine=engine, seed_learner_id=seed_id)
    with engine.begin() as conn:
        row = conn.execute(text("SELECT learner_id, payload FROM learner_enrollments WHERE enrollment_id = 'enr_1'")).first()
    assert row is not None
    assert row.learner_id == str(seed_id)
    assert row.payload["learner_id"] == str(seed_id)


def test_copy_is_idempotent(postgres_url: str, snapshot_file: Path) -> None:
    engine = create_engine(postgres_url)
    seed_id = ensure_seed_user(engine)
    copy_to_postgres(snapshot=snapshot_file, engine=engine, seed_learner_id=seed_id)
    copy_to_postgres(snapshot=snapshot_file, engine=engine, seed_learner_id=seed_id)
    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM learner_enrollments")).scalar_one()
    assert n == 1
