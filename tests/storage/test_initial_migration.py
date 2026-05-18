from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TABLES = {
    "workflow_runs",
    "workflow_events",
    "course_runs",
    "course_events",
    "learner_enrollments",
    "learner_submissions",
    "learner_workspace_sessions",
    "publish_snapshots",
    "creator_feedback",
    "learner_feedback",
    "learner_eval_reports",
    "creator_assets",
}


import sys
ALEMBIC = Path(sys.executable).parent / "alembic"


@pytest.fixture()
def migrated(postgres_url: str) -> None:
    import os
    subprocess.run(
        [str(ALEMBIC), "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": postgres_url},
        check=True,
    )


def test_initial_migration_creates_all_legacy_tables(postgres_url: str, migrated: None) -> None:
    engine = create_engine(postgres_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    missing = EXPECTED_TABLES - tables
    assert not missing, f"Missing tables: {missing}"
