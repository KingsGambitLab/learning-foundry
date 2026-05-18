# Auth + Postgres + Per-User Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the local single-user prototype into a real multi-user backend: replace SQLite with Postgres, add email+password auth with server-side sessions, bind enrollments to real users, and key workspaces by `<user_id, assignment_id>`. The existing server at `127.0.0.1:8010` keeps running; the new server runs in parallel on `127.0.0.1:8040` so behavior can be diffed end-to-end.

**Architecture:** Four sequenced milestones. M1 introduces a `WorkflowStore` Protocol and a `PostgresWorkflowStore` that ports the existing 35 store methods 1:1 (JSON-blob tables preserved). M2 adds `users` + `user_sessions` tables, password hashing, server-side sessions, and `/auth/*` routes. M3 wires enrollments to authenticated users, applies role guards, and runs a one-shot migrator (snapshot 8010's SQLite via `VACUUM INTO`, copy into Postgres, rewrite `local-learner` to a seed user). M4 changes the on-disk workspace path to `learner_workspaces/<user_id>/<assignment_id>/workspace/`.

**Tech Stack:** FastAPI (existing), SQLAlchemy 2.x Core + psycopg 3, Alembic, passlib[bcrypt], pytest + testcontainers-python, Docker Compose for local Postgres.

**Spec:** [docs/superpowers/specs/2026-05-14-auth-postgres-workspace-design.md](docs/superpowers/specs/2026-05-14-auth-postgres-workspace-design.md)

---

## File map

**M1 — Storage swap**

- Create: `app/storage/workflow_store.py` (Protocol), `app/storage/postgres_store.py`, `app/storage/database.py`, `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`, `alembic/versions/0001_initial.py`, `docker-compose.yml`, `.env.example`, `tests/storage/__init__.py`, `tests/storage/conftest.py`, `tests/storage/test_postgres_store_parity.py`
- Modify: `pyproject.toml`, `app/main.py`, `app/services/workflow_service.py`, `app/services/lms_service.py`, `app/services/course_workflow_service.py`, `app/services/creator_asset_service.py`, `app/services/publish_snapshot_service.py`, `app/services/failure_replay_smoke.py`
- Delete (at end of M3): `app/storage/sqlite_store.py`

**M2 — Auth**

- Create: `alembic/versions/0002_auth.py`, `app/domain/auth.py`, `app/services/auth_passwords.py`, `app/services/auth_session.py`, `app/services/auth_user_service.py`, `app/api/deps.py`, `app/api/auth_routes.py`, `app/templates/login.html`, `app/templates/register.html`, `app/templates/_auth_header.html`, `tests/auth/__init__.py`, `tests/auth/test_passwords.py`, `tests/auth/test_session.py`, `tests/auth/test_register_login_logout.py`, `tests/auth/test_role_guards.py`
- Modify: `app/storage/postgres_store.py` (add auth methods), `app/storage/workflow_store.py` (add auth methods to Protocol), `app/main.py` (mount auth router + login/register pages), `pyproject.toml` (add passlib[bcrypt])

**M3 — Enrollment + migrator**

- Create: `scripts/migrate_sqlite_to_postgres.py`, `scripts/verify_migration.py`, `tests/migration/__init__.py`, `tests/migration/test_migrator.py`
- Modify: `app/domain/learner.py` (drop `learner_id` from `CreateEnrollmentRequest`), `app/services/lms_service.py`, `app/api/routes.py` (apply role guards, drop query-param `learner_id`)
- Delete: `app/storage/sqlite_store.py`

**M4 — Workspace path**

- Modify: `app/services/lms_service.py:386` (`_workspace_root`)
- Update: `scripts/migrate_sqlite_to_postgres.py` (add Phase 3 workspace rename), `tests/migration/test_migrator.py` (workspace rename test)

---

## Milestone 1 — Storage swap

### Task 1: Add dependencies and docker-compose Postgres service

**Files:**
- Modify: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Edit `pyproject.toml`, replace the `dependencies = [...]` block with:

```toml
dependencies = [
  "fastapi>=0.136.1,<0.137.0",
  "uvicorn>=0.46.0,<0.47.0",
  "httpx>=0.28.1,<0.29.0",
  "openai>=1.79.0,<2.0.0",
  "langgraph>=0.6.0,<1.0.0",
  "sqlalchemy>=2.0.30,<2.1.0",
  "psycopg[binary]>=3.2.0,<4.0.0",
  "alembic>=1.13.0,<2.0.0",
  "passlib[bcrypt]>=1.7.4,<2.0.0",
]

[project.optional-dependencies]
test = [
  "pytest>=8.0.0,<9.0.0",
  "testcontainers[postgres]>=4.7.0,<5.0.0",
]
```

- [ ] **Step 2: Install the new dependencies**

Run: `pip install -e '.[test]'`
Expected: installs sqlalchemy, psycopg, alembic, passlib, testcontainers without error.

- [ ] **Step 3: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: course_gen_postgres
    environment:
      POSTGRES_DB: course_gen
      POSTGRES_USER: course_gen
      POSTGRES_PASSWORD: course_gen
    ports:
      - "5435:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U course_gen"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  postgres_data:
```

Host port 5435 is used because 5432 is commonly occupied by local Postgres installs.

- [ ] **Step 4: Create .env.example**

Create `.env.example`:

```dotenv
DATABASE_URL=postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen
SESSION_SECRET=change-me-in-production
SESSION_COOKIE_SECURE=false
AUTH_BCRYPT_COST=12
COURSE_GEN_PORT=8040
```

- [ ] **Step 5: Start Postgres and verify**

Run: `docker compose up -d postgres && docker compose exec postgres pg_isready -U course_gen`
Expected: container starts; `accepting connections`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example
git commit -m "M1: deps + docker-compose Postgres service for local dev"
```

---

### Task 2: Database connection module and Alembic skeleton

**Files:**
- Create: `app/storage/database.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/.gitkeep`
- Create: `tests/storage/__init__.py`
- Create: `tests/storage/conftest.py`
- Create: `tests/storage/test_database.py`

- [ ] **Step 1: Write the failing test**

Create `tests/storage/__init__.py` (empty file).
Create `tests/storage/test_database.py`:

```python
from __future__ import annotations

from sqlalchemy import text

from app.storage.database import build_engine


def test_build_engine_returns_working_postgres_engine(postgres_url: str) -> None:
    engine = build_engine(postgres_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar_one()
    assert result == 1
```

- [ ] **Step 2: Create the testcontainers-based conftest**

Create `tests/storage/conftest.py`:

```python
from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as container:
        yield container


@pytest.fixture(scope="session")
def postgres_url(postgres_container: PostgresContainer) -> str:
    url = postgres_container.get_connection_url()
    if not url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["DATABASE_URL"] = url
    return url


@pytest.fixture(autouse=True)
def _reset_tables(postgres_url: str) -> Iterator[None]:
    yield
    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        tables = conn.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname='public' AND tablename != 'alembic_version'"
            )
        ).scalars().all()
        if tables:
            joined = ", ".join(f'"{t}"' for t in tables)
            conn.execute(text(f"TRUNCATE TABLE {joined} RESTART IDENTITY CASCADE"))
    engine.dispose()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/storage/test_database.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.storage.database'`.

- [ ] **Step 4: Create app/storage/database.py**

Create `app/storage/database.py`:

```python
from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine


def build_engine(database_url: str | None = None) -> Engine:
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env or export it manually."
        )
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/storage/test_database.py -v`
Expected: PASS.

- [ ] **Step 6: Create Alembic skeleton files**

Create `alembic.ini`:

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

Create `alembic/env.py`:

```python
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

database_url = os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

Create `alembic/script.py.mako`:

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

Create `alembic/versions/.gitkeep` (empty file).

- [ ] **Step 7: Commit**

```bash
git add app/storage/database.py alembic.ini alembic/env.py alembic/script.py.mako alembic/versions/.gitkeep tests/storage/__init__.py tests/storage/conftest.py tests/storage/test_database.py
git commit -m "M1: database engine helper + Alembic skeleton + test fixtures"
```

---

### Task 3: Initial Alembic migration for the 12 legacy tables

**Files:**
- Create: `alembic/versions/0001_initial.py`
- Create: `tests/storage/test_initial_migration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/storage/test_initial_migration.py`:

```python
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


@pytest.fixture()
def migrated(postgres_url: str) -> None:
    subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "DATABASE_URL": postgres_url},
        check=True,
    )


def test_initial_migration_creates_all_legacy_tables(postgres_url: str, migrated: None) -> None:
    engine = create_engine(postgres_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    missing = EXPECTED_TABLES - tables
    assert not missing, f"Missing tables: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_initial_migration.py -v`
Expected: FAIL — `alembic upgrade head` errors because there are no revisions yet.

- [ ] **Step 3: Create the initial migration**

Create `alembic/versions/0001_initial.py`:

```python
"""Initial schema — 12 legacy tables ported from SQLiteWorkflowStore.

Revision ID: 0001
Revises:
Create Date: 2026-05-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("workflow_runs_updated_at_idx", "workflow_runs", ["updated_at"])

    op.create_table(
        "workflow_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("workflow_events_run_id_idx", "workflow_events", ["run_id", "sequence_no"])

    op.create_table(
        "course_runs",
        sa.Column("course_run_id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("package_type", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("course_runs_updated_at_idx", "course_runs", ["updated_at"])

    op.create_table(
        "course_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("course_events_run_id_idx", "course_events", ["course_run_id", "sequence_no"])

    op.create_table(
        "learner_enrollments",
        sa.Column("enrollment_id", sa.Text(), primary_key=True),
        sa.Column("learner_id", sa.Text(), nullable=False),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_enrollments_learner_idx", "learner_enrollments", ["learner_id"])
    op.create_index("learner_enrollments_course_idx", "learner_enrollments", ["course_run_id"])

    op.create_table(
        "learner_submissions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("deliverable_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_submissions_enrollment_idx", "learner_submissions", ["enrollment_id", "created_at"])

    op.create_table(
        "learner_workspace_sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("deliverable_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_workspace_sessions_enrollment_idx", "learner_workspace_sessions", ["enrollment_id", "created_at"])

    op.create_table(
        "publish_snapshots",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("publish_snapshots_course_idx", "publish_snapshots", ["course_run_id", "created_at"])

    op.create_table(
        "creator_feedback",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("creator_feedback_course_idx", "creator_feedback", ["course_run_id", "created_at"])

    op.create_table(
        "learner_feedback",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_feedback_enrollment_idx", "learner_feedback", ["enrollment_id", "created_at"])

    op.create_table(
        "learner_eval_reports",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("enrollment_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("learner_eval_reports_enrollment_idx", "learner_eval_reports", ["enrollment_id", "created_at"])

    op.create_table(
        "creator_assets",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("course_run_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
    )
    op.create_index("creator_assets_course_idx", "creator_assets", ["course_run_id", "updated_at"])


def downgrade() -> None:
    for table in [
        "creator_assets",
        "learner_eval_reports",
        "learner_feedback",
        "creator_feedback",
        "publish_snapshots",
        "learner_workspace_sessions",
        "learner_submissions",
        "learner_enrollments",
        "course_events",
        "course_runs",
        "workflow_events",
        "workflow_runs",
    ]:
        op.drop_table(table)
```

> **Note:** The exact column list above mirrors the schema in `app/storage/sqlite_store.py:_ensure_schema`. Cross-check by reading that method before merging — any column you find there that's missing here is a bug.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_initial_migration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/0001_initial.py tests/storage/test_initial_migration.py
git commit -m "M1: Alembic 0001 — port 12 legacy tables to Postgres schema"
```

---

### Task 4: WorkflowStore Protocol

**Files:**
- Create: `app/storage/workflow_store.py`
- Create: `tests/storage/test_workflow_store_protocol.py`

- [ ] **Step 1: Write the failing test**

Create `tests/storage/test_workflow_store_protocol.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_workflow_store_protocol.py -v`
Expected: FAIL — `app.storage.workflow_store` doesn't exist.

- [ ] **Step 3: Create the Protocol**

Create `app/storage/workflow_store.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from app.domain.assets import CreatorAssetRecord
from app.domain.course import CourseEvent, CourseRun, CourseRunSummary
from app.domain.learner import (
    LearnerEnrollment,
    LearnerEnrollmentSummary,
    LearnerSubmissionRecord,
    LearnerWorkspaceSession,
)
from app.domain.publish import PublishSnapshot, PublishSnapshotSummary
from app.domain.testing import (
    CreatorFeedbackRecord,
    LearnerCourseEvaluationReport,
    LearnerFeedbackRecord,
)
from app.domain.workflow import WorkflowEvent, WorkflowRun, WorkflowRunSummary


@runtime_checkable
class WorkflowStore(Protocol):
    def utcnow(self) -> datetime: ...

    def save_run(self, run: WorkflowRun) -> WorkflowRun: ...
    def get_run(self, run_id: str) -> WorkflowRun | None: ...
    def list_runs(self, limit: int = 50) -> list[WorkflowRunSummary]: ...

    def append_event(self, run_id: str, event_type: str, payload: dict) -> WorkflowEvent: ...
    def list_events(self, run_id: str) -> list[WorkflowEvent]: ...

    def save_course_run(self, run: CourseRun) -> CourseRun: ...
    def get_course_run(self, course_run_id: str) -> CourseRun | None: ...
    def list_course_runs(self, limit: int = 50) -> list[CourseRunSummary]: ...

    def append_course_event(self, course_run_id: str, event_type: str, payload: dict) -> CourseEvent: ...
    def list_course_events(self, course_run_id: str) -> list[CourseEvent]: ...

    def reset_all(self) -> dict[str, int]: ...

    def save_creator_asset(self, asset: CreatorAssetRecord) -> CreatorAssetRecord: ...
    def get_creator_asset(self, asset_id: str) -> CreatorAssetRecord | None: ...
    def list_creator_assets(self, limit: int = 100) -> list[CreatorAssetRecord]: ...
    def delete_creator_asset(self, asset_id: str) -> bool: ...

    def save_learner_enrollment(self, enrollment: LearnerEnrollment) -> LearnerEnrollment: ...
    def get_learner_enrollment(self, enrollment_id: str) -> LearnerEnrollment | None: ...
    def find_learner_enrollment(self, learner_id: str, course_run_id: str) -> LearnerEnrollment | None: ...
    def list_learner_enrollments(
        self, learner_id: str | None = None, limit: int = 50
    ) -> list[LearnerEnrollmentSummary]: ...

    def save_learner_submission(self, submission: LearnerSubmissionRecord) -> LearnerSubmissionRecord: ...
    def list_learner_submissions(
        self, enrollment_id: str, deliverable_id: str | None = None
    ) -> list[LearnerSubmissionRecord]: ...

    def save_learner_workspace_session(self, session: LearnerWorkspaceSession) -> LearnerWorkspaceSession: ...
    def list_learner_workspace_sessions(self, enrollment_id: str) -> list[LearnerWorkspaceSession]: ...
    def list_all_learner_workspace_sessions(self) -> list[LearnerWorkspaceSession]: ...

    def save_publish_snapshot(self, snapshot: PublishSnapshot) -> PublishSnapshot: ...
    def get_publish_snapshot(self, snapshot_id: str) -> PublishSnapshot | None: ...
    def list_publish_snapshots(
        self, course_run_id: str | None = None, limit: int = 50
    ) -> list[PublishSnapshotSummary]: ...
    def get_latest_publish_snapshot(self, course_run_id: str) -> PublishSnapshot | None: ...

    def save_creator_feedback(self, feedback: CreatorFeedbackRecord) -> CreatorFeedbackRecord: ...
    def list_creator_feedback(self, course_run_id: str, limit: int = 100) -> list[CreatorFeedbackRecord]: ...

    def save_learner_feedback(self, feedback: LearnerFeedbackRecord) -> LearnerFeedbackRecord: ...
    def list_learner_feedback(self, enrollment_id: str, limit: int = 100) -> list[LearnerFeedbackRecord]: ...

    def save_learner_eval_report(
        self, report: LearnerCourseEvaluationReport
    ) -> LearnerCourseEvaluationReport: ...
    def list_learner_eval_reports(
        self, enrollment_id: str | None = None, limit: int = 50
    ) -> list[LearnerCourseEvaluationReport]: ...
    def get_latest_learner_eval_report(
        self, enrollment_id: str
    ) -> LearnerCourseEvaluationReport | None: ...
```

> **Note:** Cross-check method signatures (defaults, return types) against `app/storage/sqlite_store.py`. They MUST match exactly — any drift breaks consumers.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_workflow_store_protocol.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/storage/workflow_store.py tests/storage/test_workflow_store_protocol.py
git commit -m "M1: WorkflowStore Protocol — 35 public methods from SQLiteWorkflowStore"
```

---

### Task 5: PostgresWorkflowStore — scaffold + workflow_runs/workflow_events

**Files:**
- Create: `app/storage/postgres_store.py`
- Create: `tests/storage/test_postgres_store_parity.py`

This task gets the new store working for one table family. Subsequent tasks add the remaining tables behind the same parity-test approach.

- [ ] **Step 1: Write the failing parity test**

Create `tests/storage/test_postgres_store_parity.py`:

```python
from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from app.domain.workflow import WorkflowRun
from app.storage.postgres_store import PostgresWorkflowStore

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    import os

    subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": postgres_url},
        check=True,
    )


@pytest.fixture()
def store(postgres_url: str) -> PostgresWorkflowStore:
    return PostgresWorkflowStore(engine=create_engine(postgres_url))


def _make_workflow_run(run_id: str = "run_test") -> WorkflowRun:
    now = datetime.now(UTC)
    return WorkflowRun(
        id=run_id,
        title="Test run",
        stage="intake",
        status="pending",
        created_at=now,
        updated_at=now,
    )


def test_save_and_get_run_roundtrip(store: PostgresWorkflowStore) -> None:
    run = _make_workflow_run()
    saved = store.save_run(run)
    assert saved.id == "run_test"
    fetched = store.get_run("run_test")
    assert fetched is not None
    assert fetched.title == "Test run"


def test_list_runs_orders_by_updated_at_desc(store: PostgresWorkflowStore) -> None:
    older = _make_workflow_run("run_older")
    newer = _make_workflow_run("run_newer")
    newer.updated_at = datetime.now(UTC)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_postgres_store_parity.py -v`
Expected: FAIL — `app.storage.postgres_store` doesn't exist.

- [ ] **Step 3: Create PostgresWorkflowStore with workflow_runs + workflow_events**

Create `app/storage/postgres_store.py`:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import Engine, text

from app.domain.workflow import WorkflowEvent, WorkflowRun, WorkflowRunSummary
from app.storage.database import build_engine


class PostgresWorkflowStore:
    """Postgres-backed implementation of the WorkflowStore Protocol.

    JSON-blob shape is preserved 1:1 with SQLiteWorkflowStore — every legacy
    table stores its non-key state in a JSONB `payload` column. Methods accept
    and return the same domain types as the SQLite store, so callers do not
    change.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or build_engine()

    def utcnow(self) -> datetime:
        return datetime.now(UTC)

    # ------------------------------------------------------------------ workflow_runs

    def save_run(self, run: WorkflowRun) -> WorkflowRun:
        payload = run.model_dump(mode="json")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO workflow_runs (
                        run_id, title, stage, status, created_at, updated_at, payload
                    ) VALUES (
                        :run_id, :title, :stage, :status, :created_at, :updated_at,
                        CAST(:payload AS JSONB)
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        stage = EXCLUDED.stage,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload
                    """
                ),
                {
                    "run_id": run.id,
                    "title": run.title,
                    "stage": run.stage if isinstance(run.stage, str) else run.stage.value,
                    "status": run.status if isinstance(run.status, str) else run.status.value,
                    "created_at": run.created_at.isoformat(),
                    "updated_at": run.updated_at.isoformat(),
                    "payload": json.dumps(payload),
                },
            )
        return run

    def get_run(self, run_id: str) -> WorkflowRun | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT payload FROM workflow_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).first()
        if row is None:
            return None
        return WorkflowRun.model_validate(row.payload)

    def list_runs(self, limit: int = 50) -> list[WorkflowRunSummary]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT payload FROM workflow_runs
                    ORDER BY updated_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).all()
        return [WorkflowRunSummary.model_validate(row.payload) for row in rows]

    # ------------------------------------------------------------------ workflow_events

    def append_event(self, run_id: str, event_type: str, payload: dict) -> WorkflowEvent:
        now = self.utcnow().isoformat()
        with self.engine.begin() as conn:
            seq = conn.execute(
                text(
                    """
                    SELECT COALESCE(MAX(sequence_no), 0) + 1
                    FROM workflow_events WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            ).scalar_one()
            conn.execute(
                text(
                    """
                    INSERT INTO workflow_events (
                        run_id, sequence_no, event_type, created_at, payload
                    ) VALUES (
                        :run_id, :sequence_no, :event_type, :created_at,
                        CAST(:payload AS JSONB)
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "sequence_no": seq,
                    "event_type": event_type,
                    "created_at": now,
                    "payload": json.dumps(payload),
                },
            )
        return WorkflowEvent(
            run_id=run_id,
            sequence_no=seq,
            event_type=event_type,
            created_at=datetime.fromisoformat(now),
            payload=payload,
        )

    def list_events(self, run_id: str) -> list[WorkflowEvent]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT run_id, sequence_no, event_type, created_at, payload
                    FROM workflow_events
                    WHERE run_id = :run_id
                    ORDER BY sequence_no ASC
                    """
                ),
                {"run_id": run_id},
            ).all()
        return [
            WorkflowEvent(
                run_id=row.run_id,
                sequence_no=row.sequence_no,
                event_type=row.event_type,
                created_at=datetime.fromisoformat(row.created_at),
                payload=row.payload,
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ stubs (filled in subsequent tasks)

    def save_course_run(self, run):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_course_run(self, course_run_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_course_runs(self, limit: int = 50):
        raise NotImplementedError

    def append_course_event(self, course_run_id, event_type, payload):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_course_events(self, course_run_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def reset_all(self):
        raise NotImplementedError

    def save_creator_asset(self, asset):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_creator_asset(self, asset_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_creator_assets(self, limit: int = 100):
        raise NotImplementedError

    def delete_creator_asset(self, asset_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def save_learner_enrollment(self, enrollment):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_learner_enrollment(self, enrollment_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def find_learner_enrollment(self, learner_id, course_run_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_learner_enrollments(self, learner_id=None, limit: int = 50):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def save_learner_submission(self, submission):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_learner_submissions(self, enrollment_id, deliverable_id=None):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def save_learner_workspace_session(self, session):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_learner_workspace_sessions(self, enrollment_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_all_learner_workspace_sessions(self):
        raise NotImplementedError

    def save_publish_snapshot(self, snapshot):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_publish_snapshot(self, snapshot_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_publish_snapshots(self, course_run_id=None, limit: int = 50):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_latest_publish_snapshot(self, course_run_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def save_creator_feedback(self, feedback):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_creator_feedback(self, course_run_id, limit: int = 100):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def save_learner_feedback(self, feedback):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_learner_feedback(self, enrollment_id, limit: int = 100):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def save_learner_eval_report(self, report):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def list_learner_eval_reports(self, enrollment_id=None, limit: int = 50):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_latest_learner_eval_report(self, enrollment_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_postgres_store_parity.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/storage/postgres_store.py tests/storage/test_postgres_store_parity.py
git commit -m "M1: PostgresWorkflowStore scaffold + workflow_runs/events parity"
```

---

### Task 6: PostgresWorkflowStore — course_runs / course_events / reset_all

**Files:**
- Modify: `app/storage/postgres_store.py`
- Modify: `tests/storage/test_postgres_store_parity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/storage/test_postgres_store_parity.py`:

```python
from app.domain.course import CourseRun
from app.domain.registry import PackageType


def _make_course_run(course_run_id: str = "course_test") -> CourseRun:
    now = datetime.now(UTC)
    return CourseRun(
        id=course_run_id,
        title="Test course",
        summary="Test summary",
        package_type=PackageType.coding_project,
        stage="draft",
        status="draft",
        created_at=now,
        updated_at=now,
    )


def test_save_and_get_course_run_roundtrip(store: PostgresWorkflowStore) -> None:
    course = _make_course_run()
    store.save_course_run(course)
    fetched = store.get_course_run("course_test")
    assert fetched is not None
    assert fetched.title == "Test course"


def test_append_course_event_monotonic(store: PostgresWorkflowStore) -> None:
    store.save_course_run(_make_course_run("course_with_events"))
    a = store.append_course_event("course_with_events", "draft_created", {"a": 1})
    b = store.append_course_event("course_with_events", "draft_updated", {"b": 2})
    assert (a.sequence_no, b.sequence_no) == (1, 2)


def test_reset_all_clears_every_table(store: PostgresWorkflowStore) -> None:
    store.save_run(_make_workflow_run("run_reset"))
    store.save_course_run(_make_course_run("course_reset"))
    counts = store.reset_all()
    assert counts["workflow_runs"] == 1
    assert counts["course_runs"] == 1
    assert store.get_run("run_reset") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/storage/test_postgres_store_parity.py -v`
Expected: FAIL — `NotImplementedError` on `save_course_run`.

- [ ] **Step 3: Implement the methods**

In `app/storage/postgres_store.py`, replace the `save_course_run` / `get_course_run` / `list_course_runs` / `append_course_event` / `list_course_events` / `reset_all` stubs with:

```python
    # ------------------------------------------------------------------ course_runs

    def save_course_run(self, run):
        payload = run.model_dump(mode="json")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO course_runs (
                        course_run_id, title, package_type, stage, status,
                        created_at, updated_at, payload
                    ) VALUES (
                        :course_run_id, :title, :package_type, :stage, :status,
                        :created_at, :updated_at, CAST(:payload AS JSONB)
                    )
                    ON CONFLICT (course_run_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        package_type = EXCLUDED.package_type,
                        stage = EXCLUDED.stage,
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload
                    """
                ),
                {
                    "course_run_id": run.id,
                    "title": run.title,
                    "package_type": run.package_type if isinstance(run.package_type, str) else run.package_type.value,
                    "stage": run.stage if isinstance(run.stage, str) else run.stage.value,
                    "status": run.status if isinstance(run.status, str) else run.status.value,
                    "created_at": run.created_at.isoformat(),
                    "updated_at": run.updated_at.isoformat(),
                    "payload": json.dumps(payload),
                },
            )
        return run

    def get_course_run(self, course_run_id):
        from app.domain.course import CourseRun

        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT payload FROM course_runs WHERE course_run_id = :id"),
                {"id": course_run_id},
            ).first()
        return CourseRun.model_validate(row.payload) if row is not None else None

    def list_course_runs(self, limit: int = 50):
        from app.domain.course import CourseRunSummary

        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT payload FROM course_runs ORDER BY updated_at DESC LIMIT :limit"
                ),
                {"limit": limit},
            ).all()
        return [CourseRunSummary.model_validate(row.payload) for row in rows]

    # ------------------------------------------------------------------ course_events

    def append_course_event(self, course_run_id, event_type, payload):
        from app.domain.course import CourseEvent

        now = self.utcnow().isoformat()
        with self.engine.begin() as conn:
            seq = conn.execute(
                text(
                    """
                    SELECT COALESCE(MAX(sequence_no), 0) + 1
                    FROM course_events WHERE course_run_id = :id
                    """
                ),
                {"id": course_run_id},
            ).scalar_one()
            conn.execute(
                text(
                    """
                    INSERT INTO course_events (
                        course_run_id, sequence_no, event_type, created_at, payload
                    ) VALUES (
                        :id, :seq, :event_type, :created_at, CAST(:payload AS JSONB)
                    )
                    """
                ),
                {
                    "id": course_run_id,
                    "seq": seq,
                    "event_type": event_type,
                    "created_at": now,
                    "payload": json.dumps(payload),
                },
            )
        return CourseEvent(
            course_run_id=course_run_id,
            sequence_no=seq,
            event_type=event_type,
            created_at=datetime.fromisoformat(now),
            payload=payload,
        )

    def list_course_events(self, course_run_id):
        from app.domain.course import CourseEvent

        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT course_run_id, sequence_no, event_type, created_at, payload
                    FROM course_events
                    WHERE course_run_id = :id
                    ORDER BY sequence_no ASC
                    """
                ),
                {"id": course_run_id},
            ).all()
        return [
            CourseEvent(
                course_run_id=row.course_run_id,
                sequence_no=row.sequence_no,
                event_type=row.event_type,
                created_at=datetime.fromisoformat(row.created_at),
                payload=row.payload,
            )
            for row in rows
        ]

    def reset_all(self):
        tables = [
            "workflow_runs", "workflow_events", "course_runs", "course_events",
            "learner_enrollments", "learner_submissions", "learner_workspace_sessions",
            "publish_snapshots", "creator_feedback", "learner_feedback",
            "learner_eval_reports", "creator_assets",
        ]
        counts: dict[str, int] = {}
        with self.engine.begin() as conn:
            for table in tables:
                counts[table] = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            joined = ", ".join(tables)
            conn.execute(text(f"TRUNCATE TABLE {joined} RESTART IDENTITY CASCADE"))
        return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/storage/test_postgres_store_parity.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add app/storage/postgres_store.py tests/storage/test_postgres_store_parity.py
git commit -m "M1: PostgresWorkflowStore — course runs/events + reset_all"
```

---

### Task 7: PostgresWorkflowStore — learner enrollments, submissions, workspace sessions

**Files:**
- Modify: `app/storage/postgres_store.py`
- Modify: `tests/storage/test_postgres_store_parity.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/storage/test_postgres_store_parity.py`:

```python
from app.domain.learner import LearnerEnrollment, LearnerWorkspaceScope, LearnerEnrollmentStatus, LearnerWorkspaceSession, LearnerWorkspaceSessionStatus


def _make_enrollment(enrollment_id: str = "enr_test", learner_id: str = "learner_1") -> LearnerEnrollment:
    now = datetime.now(UTC)
    return LearnerEnrollment(
        id=enrollment_id,
        learner_id=learner_id,
        course_run_id="course_1",
        publish_snapshot_id="snap_1",
        course_title="Title",
        course_summary="Summary",
        package_type=PackageType.coding_project,
        shared_workflow_run_id="wf_1",
        created_at=now,
        updated_at=now,
        status=LearnerEnrollmentStatus.active,
        workspace_scope=LearnerWorkspaceScope.shared_course,
        deliverables=[],
    )


def test_save_and_get_enrollment(store: PostgresWorkflowStore) -> None:
    enrollment = _make_enrollment()
    store.save_learner_enrollment(enrollment)
    fetched = store.get_learner_enrollment("enr_test")
    assert fetched is not None
    assert fetched.learner_id == "learner_1"


def test_find_enrollment_by_learner_and_course(store: PostgresWorkflowStore) -> None:
    store.save_learner_enrollment(_make_enrollment("enr_find", learner_id="learner_2"))
    found = store.find_learner_enrollment("learner_2", "course_1")
    assert found is not None
    assert found.id == "enr_find"


def test_list_enrollments_filters_by_learner(store: PostgresWorkflowStore) -> None:
    store.save_learner_enrollment(_make_enrollment("e1", learner_id="alpha"))
    store.save_learner_enrollment(_make_enrollment("e2", learner_id="beta"))
    alpha = store.list_learner_enrollments(learner_id="alpha")
    assert len(alpha) == 1 and alpha[0].id == "e1"


def test_save_and_list_workspace_sessions(store: PostgresWorkflowStore) -> None:
    store.save_learner_enrollment(_make_enrollment("enr_ws"))
    session = LearnerWorkspaceSession(
        id="ws_1",
        enrollment_id="enr_ws",
        deliverable_id="deliv_1",
        scope=LearnerWorkspaceScope.shared_course,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        status=LearnerWorkspaceSessionStatus.running,
        workspace_root="/tmp/ws",
    )
    store.save_learner_workspace_session(session)
    sessions = store.list_learner_workspace_sessions("enr_ws")
    assert len(sessions) == 1 and sessions[0].id == "ws_1"
    assert store.list_all_learner_workspace_sessions()[0].id == "ws_1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/storage/test_postgres_store_parity.py::test_save_and_get_enrollment -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement learner enrollment, submission, workspace session methods**

Replace the corresponding stubs in `app/storage/postgres_store.py`. The full implementation mirrors `SQLiteWorkflowStore.save_learner_enrollment`, `get_learner_enrollment`, `find_learner_enrollment`, `list_learner_enrollments`, `save_learner_submission`, `list_learner_submissions`, `save_learner_workspace_session`, `list_learner_workspace_sessions`, `list_all_learner_workspace_sessions` from `app/storage/sqlite_store.py` (lines 498–670). The translation pattern is identical to Tasks 5–6: read the SQLite SQL, change `INSERT OR REPLACE` to `INSERT ... ON CONFLICT ... DO UPDATE`, change `TEXT NOT NULL` payloads to `CAST(:payload AS JSONB)`, change `json.loads(row[col])` to `row.payload` (already a dict from JSONB), and call `_normalize_learner_enrollment_payload` (port from `sqlite_store.py` lines 1165+) before model_validate on read.

> **Note:** Port the `_normalize_learner_enrollment_payload`, `_normalize_workflow_run_payload`, `_normalize_publish_snapshot_payload`, `_normalize_task_agent_spec_payload`, `_normalize_course_run_payload`, `_coerce_starter_type_recursively`, `_coerce_legacy_starter_type`, and `_infer_editable_files` private helpers verbatim from `sqlite_store.py`. They handle legacy payload shapes and must be preserved because migrated data may contain them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/storage/test_postgres_store_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/storage/postgres_store.py tests/storage/test_postgres_store_parity.py
git commit -m "M1: PostgresWorkflowStore — enrollments / submissions / workspace sessions"
```

---

### Task 8: PostgresWorkflowStore — remaining tables (publish snapshots, feedback, eval reports, creator assets)

**Files:**
- Modify: `app/storage/postgres_store.py`
- Modify: `tests/storage/test_postgres_store_parity.py`

- [ ] **Step 1: Add failing tests**

Append minimal save+get tests for each remaining method group: `save_publish_snapshot` / `get_publish_snapshot` / `list_publish_snapshots` / `get_latest_publish_snapshot`; `save_creator_feedback` / `list_creator_feedback`; `save_learner_feedback` / `list_learner_feedback`; `save_learner_eval_report` / `list_learner_eval_reports` / `get_latest_learner_eval_report`; `save_creator_asset` / `get_creator_asset` / `list_creator_assets` / `delete_creator_asset`.

Each test should build a minimal valid instance of the relevant domain object, save it, fetch it, and assert equality of one identifying field.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/storage/test_postgres_store_parity.py -v`
Expected: FAILs on every new test.

- [ ] **Step 3: Implement the remaining methods**

Replace the remaining stubs in `app/storage/postgres_store.py`. Port from `sqlite_store.py` lines 450–880 using the same translation rules from Task 7.

- [ ] **Step 4: Run all parity tests**

Run: `pytest tests/storage/ -v`
Expected: PASS — all tests, all methods covered.

- [ ] **Step 5: Commit**

```bash
git add app/storage/postgres_store.py tests/storage/test_postgres_store_parity.py
git commit -m "M1: PostgresWorkflowStore — publish snapshots, feedback, eval reports, creator assets"
```

---

### Task 9: Swap callers from SQLiteWorkflowStore to PostgresWorkflowStore

**Files:**
- Modify: `app/main.py`
- Modify: `app/services/workflow_service.py`
- Modify: `app/services/lms_service.py`
- Modify: `app/services/course_workflow_service.py`
- Modify: `app/services/creator_asset_service.py`
- Modify: `app/services/publish_snapshot_service.py`
- Modify: `app/services/failure_replay_smoke.py`

The goal is to make every service hold a `WorkflowStore` (Protocol) and let `app/main.py` decide which concrete implementation to construct at startup.

- [ ] **Step 1: Update service type hints**

In each of the seven files listed, replace:

```python
from app.storage.sqlite_store import SQLiteWorkflowStore
```

with:

```python
from app.storage.workflow_store import WorkflowStore
```

And replace every `store: SQLiteWorkflowStore` argument and `SQLiteWorkflowStore` attribute annotation with `store: WorkflowStore`.

- [ ] **Step 2: Wire Postgres into app/main.py**

In `app/main.py`, find the existing store construction (around line 39 / 71):

```python
store = getattr(getattr(app.state, "workflow_service", None), "store", None) or SQLiteWorkflowStore()
```

Replace with:

```python
store = getattr(getattr(app.state, "workflow_service", None), "store", None) or PostgresWorkflowStore()
```

And update the import:

```python
from app.storage.postgres_store import PostgresWorkflowStore
```

Remove the `from app.storage.sqlite_store import SQLiteWorkflowStore` import.

- [ ] **Step 3: Run the full test suite**

Run: `DATABASE_URL=postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen alembic upgrade head && pytest tests/ -v`
Expected: existing tests pass; storage tests pass.

- [ ] **Step 4: Smoke-boot the new server**

Run in a separate terminal: `DATABASE_URL=postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen COURSE_GEN_PORT=8040 uvicorn app.main:app --port 8040`
Expected: server starts; `curl http://127.0.0.1:8040/v1/lms/courses` returns `{"courses": []}` (Postgres is empty).

- [ ] **Step 5: Commit**

```bash
git add app/main.py app/services/workflow_service.py app/services/lms_service.py app/services/course_workflow_service.py app/services/creator_asset_service.py app/services/publish_snapshot_service.py app/services/failure_replay_smoke.py
git commit -m "M1: swap callers to PostgresWorkflowStore behind WorkflowStore Protocol"
```

---

## Milestone 2 — Auth (users, sessions, routes, pages)

### Task 10: Alembic 0002 — users + user_sessions tables

**Files:**
- Create: `alembic/versions/0002_auth.py`
- Create: `tests/auth/__init__.py`
- Create: `tests/auth/test_auth_migration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/auth/__init__.py` (empty).
Create `tests/auth/test_auth_migration.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def migrated(postgres_url: str) -> None:
    import os
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)


def test_auth_tables_exist(postgres_url: str, migrated: None) -> None:
    inspector = inspect(create_engine(postgres_url))
    tables = set(inspector.get_table_names())
    assert {"users", "user_sessions"} <= tables


def test_users_email_unique_and_role_constrained(postgres_url: str, migrated: None) -> None:
    from sqlalchemy import text
    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO users (email, password_hash, role) VALUES ('a@b.com', 'h', 'creator')"
        ))
        import pytest as pt
        with pt.raises(Exception):
            conn.execute(text(
                "INSERT INTO users (email, password_hash, role) VALUES ('a@b.com', 'h', 'creator')"
            ))
    with engine.begin() as conn:
        with pt.raises(Exception):
            conn.execute(text(
                "INSERT INTO users (email, password_hash, role) VALUES ('c@d.com', 'h', 'invalid')"
            ))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/auth/test_auth_migration.py -v`
Expected: FAIL — tables don't exist.

- [ ] **Step 3: Create the migration**

Create `alembic/versions/0002_auth.py`:

```python
"""Auth tables: users + user_sessions.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT, INET, UUID

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", CITEXT(), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("role IN ('creator', 'learner')", name="users_role_check"),
    )

    op.create_table(
        "user_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("ip", INET(), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
    )
    op.create_index("user_sessions_expires_at_idx", "user_sessions", ["expires_at"])
    op.create_index("user_sessions_user_id_idx", "user_sessions", ["user_id"])


def downgrade() -> None:
    op.drop_table("user_sessions")
    op.drop_table("users")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/auth/test_auth_migration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/0002_auth.py tests/auth/__init__.py tests/auth/test_auth_migration.py
git commit -m "M2: Alembic 0002 — users + user_sessions tables"
```

---

### Task 11: Auth domain types

**Files:**
- Create: `app/domain/auth.py`
- Create: `tests/auth/test_domain_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/auth/test_domain_types.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.auth import RegisterRequest, Role, User


def test_role_enum() -> None:
    assert Role.creator.value == "creator"
    assert Role.learner.value == "learner"


def test_register_request_validates_role() -> None:
    req = RegisterRequest(email="a@b.com", password="abcdefgh", role="learner")
    assert req.role is Role.learner


def test_register_request_rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="abcdefgh", role="admin")


def test_register_request_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="abc", role="learner")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/auth/test_domain_types.py -v`
Expected: FAIL — `app.domain.auth` doesn't exist.

- [ ] **Step 3: Create the domain module**

Create `app/domain/auth.py`:

```python
from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class Role(str, Enum):
    creator = "creator"
    learner = "learner"


class User(BaseModel):
    id: UUID
    email: EmailStr
    role: Role
    display_name: str | None = None
    created_at: datetime
    updated_at: datetime


class UserSession(BaseModel):
    id: UUID
    user_id: UUID
    created_at: datetime
    expires_at: datetime
    last_seen_at: datetime
    ip: str | None = None
    user_agent: str | None = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    role: Role
    display_name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class AuthResponse(BaseModel):
    user_id: UUID
    role: Role
    display_name: str | None = None
```

- [ ] **Step 4: Install email-validator dependency**

`EmailStr` requires `email-validator`. Run: `pip install 'email-validator>=2.2.0'` and add `"email-validator>=2.2.0,<3.0.0"` to `pyproject.toml` `dependencies`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/auth/test_domain_types.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/domain/auth.py tests/auth/test_domain_types.py pyproject.toml
git commit -m "M2: auth domain types (User, UserSession, Role, request/response models)"
```

---

### Task 12: Password hashing module

**Files:**
- Create: `app/services/auth_passwords.py`
- Create: `tests/auth/test_passwords.py`

- [ ] **Step 1: Write the failing test**

Create `tests/auth/test_passwords.py`:

```python
from __future__ import annotations

from app.services.auth_passwords import hash_password, verify_password


def test_hash_password_is_not_plaintext() -> None:
    h = hash_password("hunter2!!")
    assert h != "hunter2!!"
    assert len(h) > 30


def test_verify_password_accepts_correct() -> None:
    h = hash_password("hunter2!!")
    assert verify_password("hunter2!!", h) is True


def test_verify_password_rejects_wrong() -> None:
    h = hash_password("hunter2!!")
    assert verify_password("wrong-password", h) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/auth/test_passwords.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create the module**

Create `app/services/auth_passwords.py`:

```python
from __future__ import annotations

import os

from passlib.context import CryptContext


_DEFAULT_ROUNDS = int(os.environ.get("AUTH_BCRYPT_COST", "12"))
_context = CryptContext(schemes=["bcrypt"], bcrypt__rounds=_DEFAULT_ROUNDS, deprecated="auto")


def hash_password(plain: str) -> str:
    return _context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _context.verify(plain, hashed)
    except ValueError:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/auth/test_passwords.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/auth_passwords.py tests/auth/test_passwords.py
git commit -m "M2: password hashing helper via passlib[bcrypt]"
```

---

### Task 13: User + session storage methods on PostgresWorkflowStore

**Files:**
- Modify: `app/storage/postgres_store.py`
- Modify: `app/storage/workflow_store.py`
- Create: `tests/auth/test_user_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/auth/test_user_store.py`:

```python
from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine

from app.domain.auth import Role
from app.storage.postgres_store import PostgresWorkflowStore

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    import os
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)


@pytest.fixture()
def store(postgres_url: str) -> PostgresWorkflowStore:
    return PostgresWorkflowStore(engine=create_engine(postgres_url))


def test_create_and_get_user_by_email(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="a@b.com", password_hash="h", role=Role.learner, display_name="Alice")
    fetched = store.get_user_by_email("a@b.com")
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.role is Role.learner


def test_create_user_rejects_duplicate_email(store: PostgresWorkflowStore) -> None:
    store.create_user(email="dup@x.com", password_hash="h", role=Role.creator)
    with pytest.raises(ValueError):
        store.create_user(email="dup@x.com", password_hash="h", role=Role.creator)


def test_get_user_by_id_returns_user(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="b@c.com", password_hash="h", role=Role.creator)
    fetched = store.get_user_by_id(user.id)
    assert fetched is not None and fetched.email == "b@c.com"


def test_create_and_load_session(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="s@s.com", password_hash="h", role=Role.learner)
    expires_at = datetime.now(UTC) + timedelta(days=14)
    sid = store.create_user_session(user_id=user.id, expires_at=expires_at, ip=None, user_agent="pytest")
    loaded = store.load_user_session(sid)
    assert loaded is not None
    assert loaded.user_id == user.id


def test_load_session_returns_none_for_expired(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="exp@e.com", password_hash="h", role=Role.learner)
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    sid = store.create_user_session(user_id=user.id, expires_at=expired_at, ip=None, user_agent=None)
    assert store.load_user_session(sid) is None


def test_revoke_session(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="r@r.com", password_hash="h", role=Role.creator)
    sid = store.create_user_session(user_id=user.id, expires_at=datetime.now(UTC) + timedelta(days=1), ip=None, user_agent=None)
    store.revoke_user_session(sid)
    assert store.load_user_session(sid) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/auth/test_user_store.py -v`
Expected: FAIL — methods don't exist.

- [ ] **Step 3: Add user/session methods on PostgresWorkflowStore**

In `app/storage/postgres_store.py`, add (top of class imports unchanged):

```python
    # ------------------------------------------------------------------ users

    def create_user(self, *, email: str, password_hash: str, role, display_name: str | None = None):
        from app.domain.auth import Role, User

        role_value = role.value if isinstance(role, Role) else role
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO users (email, password_hash, role, display_name)
                    VALUES (:email, :pw, :role, :display_name)
                    RETURNING id, email, role, display_name, created_at, updated_at
                    """
                ),
                {"email": email, "pw": password_hash, "role": role_value, "display_name": display_name},
            ).first()
        if row is None:
            raise RuntimeError("INSERT did not return a row")
        return User(
            id=row.id, email=row.email, role=Role(row.role),
            display_name=row.display_name, created_at=row.created_at, updated_at=row.updated_at,
        )

    def get_user_by_email(self, email: str):
        from app.domain.auth import Role, User
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT id, email, role, display_name, created_at, updated_at, password_hash FROM users WHERE email = :email"),
                {"email": email},
            ).first()
        if row is None:
            return None
        return User(
            id=row.id, email=row.email, role=Role(row.role),
            display_name=row.display_name, created_at=row.created_at, updated_at=row.updated_at,
        )

    def get_user_password_hash(self, email: str) -> str | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT password_hash FROM users WHERE email = :email"),
                {"email": email},
            ).first()
        return row.password_hash if row else None

    def get_user_by_id(self, user_id):
        from app.domain.auth import Role, User
        from uuid import UUID
        uid = user_id if isinstance(user_id, UUID) else UUID(str(user_id))
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT id, email, role, display_name, created_at, updated_at FROM users WHERE id = :id"),
                {"id": uid},
            ).first()
        if row is None:
            return None
        return User(
            id=row.id, email=row.email, role=Role(row.role),
            display_name=row.display_name, created_at=row.created_at, updated_at=row.updated_at,
        )

    # ------------------------------------------------------------------ user sessions

    def create_user_session(self, *, user_id, expires_at, ip, user_agent):
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO user_sessions (user_id, expires_at, ip, user_agent)
                    VALUES (:uid, :exp, :ip, :ua)
                    RETURNING id
                    """
                ),
                {"uid": user_id, "exp": expires_at, "ip": ip, "ua": user_agent},
            ).first()
        return row.id

    def load_user_session(self, session_id):
        from app.domain.auth import UserSession
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT id, user_id, created_at, expires_at, last_seen_at, host(ip) AS ip, user_agent
                    FROM user_sessions
                    WHERE id = :id AND expires_at > now()
                    """
                ),
                {"id": session_id},
            ).first()
            if row is None:
                return None
            conn.execute(
                text("UPDATE user_sessions SET last_seen_at = now() WHERE id = :id"),
                {"id": session_id},
            )
        return UserSession(
            id=row.id, user_id=row.user_id, created_at=row.created_at,
            expires_at=row.expires_at, last_seen_at=row.last_seen_at,
            ip=row.ip, user_agent=row.user_agent,
        )

    def revoke_user_session(self, session_id) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM user_sessions WHERE id = :id"),
                {"id": session_id},
            )
```

Handle duplicate-email by catching the SQLAlchemy IntegrityError and re-raising as `ValueError`:

```python
    def create_user(self, *, email, password_hash, role, display_name=None):
        from sqlalchemy.exc import IntegrityError
        from app.domain.auth import Role, User

        role_value = role.value if isinstance(role, Role) else role
        try:
            with self.engine.begin() as conn:
                row = conn.execute(
                    text("""..."""),  # same INSERT as above
                    {...},
                ).first()
        except IntegrityError as exc:
            raise ValueError(f"User with email {email!r} already exists") from exc
        ...
```

- [ ] **Step 4: Extend the WorkflowStore Protocol**

Append to `app/storage/workflow_store.py` inside the `WorkflowStore` Protocol:

```python
    # auth surface
    def create_user(self, *, email: str, password_hash: str, role, display_name: str | None = None): ...
    def get_user_by_email(self, email: str): ...
    def get_user_password_hash(self, email: str) -> str | None: ...
    def get_user_by_id(self, user_id): ...
    def create_user_session(self, *, user_id, expires_at, ip, user_agent): ...
    def load_user_session(self, session_id): ...
    def revoke_user_session(self, session_id) -> None: ...
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/auth/test_user_store.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/storage/postgres_store.py app/storage/workflow_store.py tests/auth/test_user_store.py
git commit -m "M2: user + user_session store methods on PostgresWorkflowStore"
```

---

### Task 14: Session service + FastAPI dependencies

**Files:**
- Create: `app/services/auth_session.py`
- Create: `app/api/deps.py`
- Create: `tests/auth/test_session.py`
- Create: `tests/auth/test_deps.py`

- [ ] **Step 1: Write failing tests**

Create `tests/auth/test_session.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from app.domain.auth import Role
from app.services.auth_session import SessionService
from app.storage.postgres_store import PostgresWorkflowStore

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    import os
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)


@pytest.fixture()
def store(postgres_url: str) -> PostgresWorkflowStore:
    return PostgresWorkflowStore(engine=create_engine(postgres_url))


@pytest.fixture()
def service(store: PostgresWorkflowStore) -> SessionService:
    return SessionService(store)


def test_create_and_load_session_returns_user(service: SessionService, store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="x@y.com", password_hash="h", role=Role.creator)
    sid = service.create(user_id=user.id, ip=None, user_agent=None)
    loaded = service.load(str(sid))
    assert loaded is not None
    assert loaded.user.id == user.id
    assert loaded.session.user_id == user.id


def test_load_unknown_session_returns_none(service: SessionService) -> None:
    assert service.load("00000000-0000-0000-0000-000000000000") is None


def test_revoke_removes_session(service: SessionService, store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="r@v.com", password_hash="h", role=Role.learner)
    sid = service.create(user_id=user.id, ip=None, user_agent=None)
    service.revoke(str(sid))
    assert service.load(str(sid)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/auth/test_session.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create SessionService**

Create `app/services/auth_session.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.domain.auth import User, UserSession
from app.storage.workflow_store import WorkflowStore


SESSION_TTL = timedelta(days=14)
COOKIE_NAME = "coursegen_session"


@dataclass
class LoadedSession:
    user: User
    session: UserSession


class SessionService:
    def __init__(self, store: WorkflowStore) -> None:
        self.store = store

    def create(self, *, user_id: UUID, ip: str | None, user_agent: str | None) -> UUID:
        expires_at = datetime.now(UTC) + SESSION_TTL
        return self.store.create_user_session(
            user_id=user_id, expires_at=expires_at, ip=ip, user_agent=user_agent
        )

    def load(self, session_id: str) -> LoadedSession | None:
        try:
            sid = UUID(session_id)
        except (ValueError, TypeError):
            return None
        session = self.store.load_user_session(sid)
        if session is None:
            return None
        user = self.store.get_user_by_id(session.user_id)
        if user is None:
            return None
        return LoadedSession(user=user, session=session)

    def revoke(self, session_id: str) -> None:
        try:
            sid = UUID(session_id)
        except (ValueError, TypeError):
            return
        self.store.revoke_user_session(sid)
```

- [ ] **Step 4: Create FastAPI dependencies**

Create `app/api/deps.py`:

```python
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.domain.auth import Role, User
from app.services.auth_session import COOKIE_NAME, SessionService


def _service(request: Request) -> SessionService:
    service = getattr(request.app.state, "session_service", None)
    if service is None:
        raise RuntimeError("session_service is not attached to app.state")
    return service


def current_user_optional(request: Request) -> User | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    loaded = _service(request).load(token)
    return loaded.user if loaded else None


def current_user(request: Request) -> User:
    user = current_user_optional(request)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def require_role(*roles: Role):
    allowed = {r if isinstance(r, Role) else Role(r) for r in roles}

    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Role not permitted")
        return user

    return dep
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/auth/test_session.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/auth_session.py app/api/deps.py tests/auth/test_session.py
git commit -m "M2: SessionService + FastAPI auth dependencies"
```

---

### Task 15: Auth routes (register / login / logout / me) and app wiring

**Files:**
- Create: `app/api/auth_routes.py`
- Create: `tests/auth/test_register_login_logout.py`
- Modify: `app/main.py`

- [ ] **Step 1: Write failing tests**

Create `tests/auth/test_register_login_logout.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    import os
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)


@pytest.fixture()
def client(postgres_url: str) -> TestClient:
    from app.main import app
    return TestClient(app)


def test_register_creates_user_and_session(client: TestClient) -> None:
    resp = client.post("/auth/register", json={
        "email": "alice@example.com", "password": "hunter2!!", "role": "learner",
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["role"] == "learner"
    assert "coursegen_session" in resp.cookies


def test_login_with_correct_credentials_returns_cookie(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "bob@example.com", "password": "hunter2!!", "role": "creator",
    })
    client.cookies.clear()
    resp = client.post("/auth/login", json={"email": "bob@example.com", "password": "hunter2!!"})
    assert resp.status_code == 200
    assert "coursegen_session" in resp.cookies


def test_login_with_wrong_password_returns_401(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "carol@example.com", "password": "hunter2!!", "role": "learner",
    })
    client.cookies.clear()
    resp = client.post("/auth/login", json={"email": "carol@example.com", "password": "wrong"})
    assert resp.status_code == 401


def test_logout_clears_session(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "dave@example.com", "password": "hunter2!!", "role": "learner",
    })
    resp = client.post("/auth/logout")
    assert resp.status_code == 204
    me = client.get("/auth/me")
    assert me.status_code == 401


def test_me_returns_current_user(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "eve@example.com", "password": "hunter2!!", "role": "creator", "display_name": "Eve",
    })
    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["display_name"] == "Eve"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/auth/test_register_login_logout.py -v`
Expected: FAIL — routes don't exist.

- [ ] **Step 3: Create the auth router**

Create `app/api/auth_routes.py`:

```python
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.deps import current_user
from app.domain.auth import AuthResponse, LoginRequest, RegisterRequest, User
from app.services.auth_passwords import hash_password, verify_password
from app.services.auth_session import COOKIE_NAME, SESSION_TTL, SessionService
from app.storage.workflow_store import WorkflowStore


router = APIRouter(prefix="/auth", tags=["auth"])


def _store(request: Request) -> WorkflowStore:
    return request.app.state.workflow_service.store


def _session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def _set_session_cookie(response: Response, session_id) -> None:
    secure = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key=COOKIE_NAME,
        value=str(session_id),
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        secure=secure,
        samesite="lax",
    )


@router.post("/register", status_code=201, response_model=AuthResponse)
def register(payload: RegisterRequest, request: Request, response: Response) -> AuthResponse:
    store = _store(request)
    try:
        user = store.create_user(
            email=payload.email,
            password_hash=hash_password(payload.password),
            role=payload.role,
            display_name=payload.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session_id = _session_service(request).create(
        user_id=user.id,
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    _set_session_cookie(response, session_id)
    return AuthResponse(user_id=user.id, role=user.role, display_name=user.display_name)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> AuthResponse:
    store = _store(request)
    stored_hash = store.get_user_password_hash(payload.email)
    if stored_hash is None or not verify_password(payload.password, stored_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    user = store.get_user_by_email(payload.email)
    assert user is not None  # we just verified the password
    session_id = _session_service(request).create(
        user_id=user.id,
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    _set_session_cookie(response, session_id)
    return AuthResponse(user_id=user.id, role=user.role, display_name=user.display_name)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> Response:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        _session_service(request).revoke(token)
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/me", response_model=User)
def me(user: User = Depends(current_user)) -> User:
    return user
```

- [ ] **Step 4: Wire the router and session service into app/main.py**

In `app/main.py`, after the existing store/service construction, add:

```python
from app.services.auth_session import SessionService
from app.api.auth_routes import router as auth_router

app.state.session_service = SessionService(store=app.state.workflow_service.store)
app.include_router(auth_router)
```

(Adjust the insertion point to match the file's existing construction order — session service must be created after the workflow service.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/auth/test_register_login_logout.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/api/auth_routes.py app/main.py tests/auth/test_register_login_logout.py
git commit -m "M2: /auth/register, /auth/login, /auth/logout, /auth/me routes"
```

---

### Task 16: Login + register Jinja pages and shared auth header

**Files:**
- Create: `app/templates/login.html`
- Create: `app/templates/register.html`
- Create: `app/templates/_auth_header.html`
- Modify: `app/main.py`

- [ ] **Step 1: Create login.html**

Create `app/templates/login.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Sign in — Course Gen Codex</title>
  <link rel="stylesheet" href="/static/app-shell.css" />
</head>
<body>
<main style="max-width: 420px; margin: 4rem auto;">
  <h1>Sign in</h1>
  <form id="login-form">
    <label>Email <input name="email" type="email" required /></label>
    <label>Password <input name="password" type="password" required minlength="8" /></label>
    <button type="submit">Sign in</button>
    <p id="err" style="color:#c00"></p>
  </form>
  <p>No account? <a href="/register">Register</a></p>
</main>
<script>
document.getElementById("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const data = Object.fromEntries(new FormData(ev.target).entries());
  const resp = await fetch("/auth/login", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(data),
  });
  if (resp.ok) {
    const params = new URLSearchParams(location.search);
    location.href = params.get("next") || "/";
  } else {
    document.getElementById("err").textContent = "Invalid credentials";
  }
});
</script>
</body>
</html>
```

- [ ] **Step 2: Create register.html**

Create `app/templates/register.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Register — Course Gen Codex</title>
  <link rel="stylesheet" href="/static/app-shell.css" />
</head>
<body>
<main style="max-width: 420px; margin: 4rem auto;">
  <h1>Create your account</h1>
  <form id="register-form">
    <label>Email <input name="email" type="email" required /></label>
    <label>Password <input name="password" type="password" required minlength="8" /></label>
    <label>Display name <input name="display_name" maxlength="120" /></label>
    <fieldset>
      <legend>Role</legend>
      <label><input type="radio" name="role" value="learner" required /> Learner</label>
      <label><input type="radio" name="role" value="creator" /> Creator</label>
    </fieldset>
    <button type="submit">Register</button>
    <p id="err" style="color:#c00"></p>
  </form>
  <p>Have an account? <a href="/login">Sign in</a></p>
</main>
<script>
document.getElementById("register-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const data = Object.fromEntries(new FormData(ev.target).entries());
  if (!data.display_name) delete data.display_name;
  const resp = await fetch("/auth/register", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(data),
  });
  if (resp.ok) {
    location.href = "/";
  } else {
    const body = await resp.json().catch(() => ({}));
    document.getElementById("err").textContent = body.detail || "Registration failed";
  }
});
</script>
</body>
</html>
```

- [ ] **Step 3: Create the shared header partial**

Create `app/templates/_auth_header.html`:

```html
<header class="auth-header" style="display:flex; justify-content:flex-end; padding:0.5rem 1rem; gap:0.75rem; font-size:0.9rem;">
  {% if current_user %}
    <span>{{ current_user.display_name or current_user.email }}</span>
    <form method="post" action="/auth/logout" style="display:inline">
      <button type="submit" style="background:none;border:none;color:#06f;cursor:pointer;">Log out</button>
    </form>
  {% else %}
    <a href="/login">Sign in</a>
    <a href="/register">Register</a>
  {% endif %}
</header>
```

- [ ] **Step 4: Mount the page routes**

In `app/main.py`, add (near the existing page routes):

```python
from app.api.deps import current_user_optional


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {})


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "register.html", {})
```

- [ ] **Step 5: Manually verify**

Restart the dev server: `DATABASE_URL=... uvicorn app.main:app --port 8040`
Open `http://127.0.0.1:8040/register` in a browser, register a learner, confirm the cookie is set and `/auth/me` returns the user.

- [ ] **Step 6: Commit**

```bash
git add app/templates/login.html app/templates/register.html app/templates/_auth_header.html app/main.py
git commit -m "M2: login/register Jinja pages + auth header partial"
```

---

### Task 17: Role guards on `/v1/*` routes + role-guard tests

**Files:**
- Modify: `app/api/routes.py`
- Create: `tests/auth/test_role_guards.py`

- [ ] **Step 1: Write failing role-guard tests**

Create `tests/auth/test_role_guards.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    import os
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)


@pytest.fixture()
def client(postgres_url: str) -> TestClient:
    from app.main import app
    return TestClient(app)


def _register(client: TestClient, email: str, role: str) -> None:
    client.post("/auth/register", json={"email": email, "password": "hunter2!!", "role": role})


def test_learner_cannot_hit_creator_route(client: TestClient) -> None:
    _register(client, "learner@x.com", "learner")
    resp = client.get("/v1/course-runs")
    assert resp.status_code == 403


def test_creator_cannot_hit_learner_route(client: TestClient) -> None:
    _register(client, "creator@x.com", "creator")
    resp = client.get("/v1/lms/enrollments")
    assert resp.status_code == 403


def test_unauthenticated_gets_401(client: TestClient) -> None:
    resp = client.get("/v1/lms/courses")
    assert resp.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/auth/test_role_guards.py -v`
Expected: FAIL — routes are unguarded.

- [ ] **Step 3: Apply guards in app/api/routes.py**

For every `@router.get/post/put/delete` decorator in `app/api/routes.py`, add a `dependencies=[Depends(...)]` arg. Group rules:

- `/v1/lms/*` → `Depends(require_role(Role.learner))`
- `/v1/courses/*`, `/v1/course-runs/*`, `/v1/workflow/*`, `/v1/publish/*`, `/v1/creator/*`, `/v1/task-agent-authoring/*` → `Depends(require_role(Role.creator))`
- `/v1/lms/courses` (the catalog) → `Depends(current_user)` (both roles can see it)

Example:

```python
from fastapi import Depends
from app.api.deps import current_user, require_role
from app.domain.auth import Role


@router.get(
    "/v1/lms/enrollments",
    response_model=LearnerEnrollmentList,
    tags=["lms"],
    dependencies=[Depends(require_role(Role.learner))],
)
def list_lms_enrollments(request: Request) -> LearnerEnrollmentList:
    return _lms_service(request).list_enrollments(learner_id=str(current_user(request).id))
```

> **Note:** Open `app/api/routes.py` and add a guard to every route. Cross-check by grepping `@router.` and verifying each match has `dependencies=[Depends(...)]` or accepts a user via a `Depends(current_user)` parameter.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/auth/test_role_guards.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/routes.py tests/auth/test_role_guards.py
git commit -m "M2: role guards on every /v1/* route"
```

---

## Milestone 3 — Enrollment rewiring + data migration

### Task 18: Drop `learner_id` from request shapes; route reads from session

**Files:**
- Modify: `app/domain/learner.py`
- Modify: `app/services/lms_service.py`
- Modify: `app/api/routes.py`
- Create: `tests/lms/test_enrollment_uses_session.py`

- [ ] **Step 1: Write the failing test**

Create `tests/lms/__init__.py` (empty).
Create `tests/lms/test_enrollment_uses_session.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    import os
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)


@pytest.fixture()
def client(postgres_url: str) -> TestClient:
    from app.main import app
    return TestClient(app)


def test_list_enrollments_ignores_query_param_and_uses_session(client: TestClient) -> None:
    client.post("/auth/register", json={"email": "l1@e.com", "password": "hunter2!!", "role": "learner"})
    resp = client.get("/v1/lms/enrollments")
    assert resp.status_code == 200
    assert resp.json() == {"enrollments": []}


def test_create_enrollment_body_does_not_accept_learner_id(client: TestClient) -> None:
    client.post("/auth/register", json={"email": "l2@e.com", "password": "hunter2!!", "role": "learner"})
    resp = client.post("/v1/lms/enrollments", json={
        "course_run_id": "nope",
        "learner_id": "should-be-ignored",
    })
    assert resp.status_code in (400, 404)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/lms/test_enrollment_uses_session.py -v`
Expected: FAIL — request body still accepts `learner_id`.

- [ ] **Step 3: Drop `learner_id` from CreateEnrollmentRequest**

In `app/domain/learner.py`, replace:

```python
class CreateEnrollmentRequest(BaseModel):
    course_run_id: str
    learner_id: str = "local-learner"
```

with:

```python
from pydantic import ConfigDict


class CreateEnrollmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    course_run_id: str
```

- [ ] **Step 4: Update LMSService.enroll signature**

In `app/services/lms_service.py`, change:

```python
def enroll(self, request: CreateEnrollmentRequest) -> LearnerEnrollment:
    existing = self.store.find_learner_enrollment(request.learner_id, request.course_run_id)
```

to:

```python
def enroll(self, request: CreateEnrollmentRequest, *, learner_id: str) -> LearnerEnrollment:
    existing = self.store.find_learner_enrollment(learner_id, request.course_run_id)
```

and replace `request.learner_id` with `learner_id` throughout the body of `enroll`.

- [ ] **Step 5: Update the routes**

In `app/api/routes.py`:

```python
@router.get(
    "/v1/lms/enrollments",
    response_model=LearnerEnrollmentList,
    tags=["lms"],
    dependencies=[Depends(require_role(Role.learner))],
)
def list_lms_enrollments(request: Request, user: User = Depends(current_user)) -> LearnerEnrollmentList:
    return _lms_service(request).list_enrollments(learner_id=str(user.id))


@router.post(
    "/v1/lms/enrollments",
    response_model=LearnerEnrollment,
    tags=["lms"],
    dependencies=[Depends(require_role(Role.learner))],
)
def create_lms_enrollment(
    payload: CreateEnrollmentRequest,
    request: Request,
    user: User = Depends(current_user),
) -> LearnerEnrollment:
    try:
        return _lms_service(request).enroll(payload, learner_id=str(user.id))
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/lms/test_enrollment_uses_session.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/domain/learner.py app/services/lms_service.py app/api/routes.py tests/lms/__init__.py tests/lms/test_enrollment_uses_session.py
git commit -m "M3: drop learner_id from request shapes; routes read from session"
```

---

### Task 19: SQLite snapshot phase of the migrator

**Files:**
- Create: `scripts/migrate_sqlite_to_postgres.py`
- Create: `tests/migration/__init__.py`
- Create: `tests/migration/test_snapshot_phase.py`

- [ ] **Step 1: Write the failing test**

Create `tests/migration/__init__.py` (empty).
Create `tests/migration/test_snapshot_phase.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.migrate_sqlite_to_postgres import snapshot_sqlite


def test_snapshot_creates_a_copy(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    dst = tmp_path / "snapshot.db"
    with sqlite3.connect(str(src)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO t (name) VALUES ('alice'), ('bob')")
    snapshot_sqlite(source=src, target=dst)
    assert dst.exists()
    with sqlite3.connect(str(dst)) as conn:
        names = [row[0] for row in conn.execute("SELECT name FROM t ORDER BY id")]
    assert names == ["alice", "bob"]


def test_snapshot_overwrites_existing_target(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    dst = tmp_path / "snapshot.db"
    with sqlite3.connect(str(src)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    dst.write_bytes(b"junk")
    snapshot_sqlite(source=src, target=dst)
    with sqlite3.connect(str(dst)) as conn:
        rows = list(conn.execute("SELECT * FROM t"))
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/migration/test_snapshot_phase.py -v`
Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Create the snapshot helper**

Create `scripts/__init__.py` (empty if missing).
Create `scripts/migrate_sqlite_to_postgres.py`:

```python
"""Snapshot 8010's SQLite into a local file, copy into Postgres, rename workspaces.

Usage:
    python -m scripts.migrate_sqlite_to_postgres \
        --source /Users/tushar/Desktop/codebases/course-gen-codex/data/course_gen.db \
        --snapshot data/course_gen_snapshot.db \
        --database-url $DATABASE_URL

Idempotent: re-running re-snapshots and re-applies, using ON CONFLICT DO NOTHING.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def snapshot_sqlite(*, source: Path, target: Path) -> None:
    """Open source SQLite read-only and VACUUM INTO target.

    Safe to run while another process is writing to source (WAL mode).
    """
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    src_uri = f"file:{source}?mode=ro"
    with sqlite3.connect(src_uri, uri=True) as conn:
        conn.execute(f"VACUUM INTO '{target.as_posix()}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--database-url", default=None, help="Postgres URL (defaults to DATABASE_URL env)")
    parser.add_argument("--skip-snapshot", action="store_true", help="Reuse existing snapshot file")
    args = parser.parse_args()

    if not args.skip_snapshot:
        print(f"Snapshotting {args.source} → {args.snapshot}")
        snapshot_sqlite(source=args.source, target=args.snapshot)
        print("Snapshot complete.")
    # Phase 2 and 3 implemented in subsequent tasks
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/migration/test_snapshot_phase.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/migrate_sqlite_to_postgres.py tests/migration/__init__.py tests/migration/test_snapshot_phase.py
git commit -m "M3: snapshot phase — VACUUM INTO local copy of source SQLite"
```

---

### Task 20: Migrator Phase 2 — seed user + copy tables

**Files:**
- Modify: `scripts/migrate_sqlite_to_postgres.py`
- Create: `tests/migration/test_copy_phase.py`

- [ ] **Step 1: Write failing tests**

Create `tests/migration/test_copy_phase.py`:

```python
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from scripts.migrate_sqlite_to_postgres import copy_to_postgres, ensure_seed_user

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    import os
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)


@pytest.fixture()
def snapshot_file(tmp_path: Path) -> Path:
    """A minimal SQLite snapshot with one workflow_run and one enrollment owned by local-learner."""
    path = tmp_path / "snapshot.db"
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE workflow_runs (run_id TEXT PRIMARY KEY, title TEXT, stage TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE workflow_events (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, sequence_no INTEGER, event_type TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE course_runs (course_run_id TEXT PRIMARY KEY, title TEXT, package_type TEXT, stage TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE course_events (id INTEGER PRIMARY KEY AUTOINCREMENT, course_run_id TEXT, sequence_no INTEGER, event_type TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE learner_enrollments (enrollment_id TEXT PRIMARY KEY, learner_id TEXT, course_run_id TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE learner_submissions (id TEXT PRIMARY KEY, enrollment_id TEXT, deliverable_id TEXT, created_at TEXT, status TEXT, payload_json TEXT);
            CREATE TABLE learner_workspace_sessions (id TEXT PRIMARY KEY, enrollment_id TEXT, deliverable_id TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            CREATE TABLE publish_snapshots (id TEXT PRIMARY KEY, course_run_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE creator_feedback (id TEXT PRIMARY KEY, course_run_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE learner_feedback (id TEXT PRIMARY KEY, enrollment_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE learner_eval_reports (id TEXT PRIMARY KEY, enrollment_id TEXT, created_at TEXT, payload_json TEXT);
            CREATE TABLE creator_assets (id TEXT PRIMARY KEY, course_run_id TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            """
        )
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
    # Second run is a no-op:
    copy_to_postgres(snapshot=snapshot_file, engine=engine, seed_learner_id=seed_id)
    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM learner_enrollments")).scalar_one()
    assert n == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/migration/test_copy_phase.py -v`
Expected: FAIL — functions don't exist.

- [ ] **Step 3: Implement `ensure_seed_user` and `copy_to_postgres`**

Append to `scripts/migrate_sqlite_to_postgres.py`:

```python
import json
import sqlite3
import secrets
from uuid import UUID

from sqlalchemy import Engine, text

from app.services.auth_passwords import hash_password

SEED_LEARNER_EMAIL = "legacy-local-learner@coursegen.local"

TABLES_IN_ORDER = [
    "course_runs", "workflow_runs", "publish_snapshots",
    "learner_enrollments", "learner_workspace_sessions",
    "learner_submissions", "creator_feedback", "learner_feedback",
    "learner_eval_reports", "creator_assets",
    "workflow_events", "course_events",
]

TABLE_PRIMARY_KEY = {
    "workflow_runs": "run_id",
    "course_runs": "course_run_id",
    "publish_snapshots": "id",
    "learner_enrollments": "enrollment_id",
    "learner_workspace_sessions": "id",
    "learner_submissions": "id",
    "creator_feedback": "id",
    "learner_feedback": "id",
    "learner_eval_reports": "id",
    "creator_assets": "id",
    "workflow_events": "id",
    "course_events": "id",
}


def ensure_seed_user(engine: Engine) -> UUID:
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": SEED_LEARNER_EMAIL},
        ).first()
        if existing is not None:
            return existing.id
        password = secrets.token_urlsafe(16)
        row = conn.execute(
            text(
                """
                INSERT INTO users (email, password_hash, role, display_name)
                VALUES (:email, :pw, 'learner', 'Legacy local-learner')
                RETURNING id
                """
            ),
            {"email": SEED_LEARNER_EMAIL, "pw": hash_password(password)},
        ).first()
        print(f"Seed user created. Email: {SEED_LEARNER_EMAIL}  Password: {password}")
        return row.id


def _rewrite_learner_id(payload: dict, seed_id: str) -> dict:
    """Replace 'local-learner' with seed_id everywhere in the payload."""
    if isinstance(payload, dict):
        return {k: _rewrite_learner_id(v, seed_id) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_rewrite_learner_id(v, seed_id) for v in payload]
    if payload == "local-learner":
        return seed_id
    return payload


def copy_to_postgres(*, snapshot: Path, engine: Engine, seed_learner_id: UUID) -> None:
    seed_str = str(seed_learner_id)
    with sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True) as src:
        src.row_factory = sqlite3.Row
        for table in TABLES_IN_ORDER:
            pk = TABLE_PRIMARY_KEY[table]
            rows = list(src.execute(f"SELECT * FROM {table}"))
            print(f"  {table}: {len(rows)} rows")
            if not rows:
                continue
            with engine.begin() as conn:
                for sqlite_row in rows:
                    row_dict = dict(sqlite_row)
                    payload = json.loads(row_dict.pop("payload_json"))
                    if table in ("learner_enrollments", "learner_feedback"):
                        payload = _rewrite_learner_id(payload, seed_str)
                    if table == "learner_enrollments" and row_dict.get("learner_id") == "local-learner":
                        row_dict["learner_id"] = seed_str
                    row_dict["payload"] = json.dumps(payload)
                    cols = list(row_dict.keys())
                    placeholders = ", ".join(f":{c}" for c in cols)
                    col_list = ", ".join(cols)
                    cast_payload = col_list.replace("payload", "CAST(:payload AS JSONB)")
                    conn.execute(
                        text(
                            f"""
                            INSERT INTO {table} ({col_list})
                            VALUES ({placeholders})
                            ON CONFLICT ({pk}) DO NOTHING
                            """.replace(":payload", "CAST(:payload AS JSONB)")
                        ),
                        row_dict,
                    )
    # Defensive: no 'local-learner' string should remain in any payload
    with engine.begin() as conn:
        for table in TABLES_IN_ORDER:
            offenders = conn.execute(
                text(f"SELECT {TABLE_PRIMARY_KEY[table]} FROM {table} WHERE payload::text LIKE '%local-learner%'")
            ).all()
            if offenders:
                raise RuntimeError(
                    f"local-learner string remains in {table} payload for ids: "
                    f"{[r[0] for r in offenders[:5]]}"
                )
```

Update `main()`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--skip-snapshot", action="store_true")
    args = parser.parse_args()

    if not args.skip_snapshot:
        snapshot_sqlite(source=args.source, target=args.snapshot)
        print("Snapshot complete.")

    import os
    from sqlalchemy import create_engine
    url = args.database_url or os.environ["DATABASE_URL"]
    engine = create_engine(url)
    seed_id = ensure_seed_user(engine)
    print(f"Seed learner id: {seed_id}")
    copy_to_postgres(snapshot=args.snapshot, engine=engine, seed_learner_id=seed_id)
    print("Copy complete.")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/migration/test_copy_phase.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_sqlite_to_postgres.py tests/migration/test_copy_phase.py
git commit -m "M3: migrator Phase 2 — seed user + copy + learner_id rewrite"
```

---

### Task 21: Run the migrator against 8010's live DB

- [ ] **Step 1: Confirm 8010 is running**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8010/v1/lms/courses`
Expected: 200.

- [ ] **Step 2: Ensure Postgres is at head**

Run: `DATABASE_URL=postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen alembic upgrade head`
Expected: `Running upgrade -> 0002, Auth tables …` (or "head is up to date").

- [ ] **Step 3: Run the migrator**

Run:

```bash
DATABASE_URL=postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen \
python -m scripts.migrate_sqlite_to_postgres \
  --source /Users/tushar/Desktop/codebases/course-gen-codex/data/course_gen.db \
  --snapshot data/course_gen_snapshot.db
```

Expected output: per-table row counts, seed-user creation message (only first run), `Copy complete.`

- [ ] **Step 4: Spot-check row counts**

Run:

```bash
DATABASE_URL=... python -c "
from sqlalchemy import create_engine, text
import os
engine = create_engine(os.environ['DATABASE_URL'])
with engine.begin() as conn:
    for t in ('workflow_runs','course_runs','learner_enrollments','publish_snapshots'):
        n = conn.execute(text(f'SELECT COUNT(*) FROM {t}')).scalar_one()
        print(t, n)
"
```

Compare against `sqlite3 data/course_gen_snapshot.db 'SELECT COUNT(*) FROM workflow_runs;'` etc. Counts must match.

- [ ] **Step 5: Commit any updates if needed**

If the migrator needed adjustments to handle real 8010 data shapes, commit them now:

```bash
git add scripts/migrate_sqlite_to_postgres.py
git commit -m "M3: adjust migrator for live 8010 payload shapes"
```

---

### Task 22: Side-by-side verification script (8010 vs 8040)

**Files:**
- Create: `scripts/verify_migration.py`

- [ ] **Step 1: Boot the new server on 8040**

In a separate terminal:

```bash
DATABASE_URL=postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen \
SESSION_SECRET=dev-secret \
uvicorn app.main:app --port 8040
```

Verify: `curl http://127.0.0.1:8040/v1/lms/courses` returns 401 (auth required) — the route now requires a session.

- [ ] **Step 2: Create the verification script**

Create `scripts/verify_migration.py`:

```python
"""Diff 8010 (SQLite, untouched) against 8040 (Postgres, migrated)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx


def _normalize(obj: Any, *, seed_learner_id: str) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v, seed_learner_id=seed_learner_id) for k, v in obj.items() if k not in {"last_seen_at"}}
    if isinstance(obj, list):
        return [_normalize(v, seed_learner_id=seed_learner_id) for v in obj]
    if obj == seed_learner_id or obj == "local-learner":
        return "<LEARNER_ID>"
    return obj


def _diff(name: str, a: Any, b: Any, *, seed_learner_id: str) -> bool:
    na = _normalize(a, seed_learner_id=seed_learner_id)
    nb = _normalize(b, seed_learner_id=seed_learner_id)
    if na == nb:
        print(f"  ✓ {name}")
        return True
    print(f"  ✗ {name}")
    print("    8010:", json.dumps(na, sort_keys=True)[:200])
    print("    8040:", json.dumps(nb, sort_keys=True)[:200])
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-learner-email", default="legacy-local-learner@coursegen.local")
    parser.add_argument("--seed-learner-password", required=True, help="Printed by the migrator")
    args = parser.parse_args()

    sqlite_client = httpx.Client(base_url="http://127.0.0.1:8010")
    pg_client = httpx.Client(base_url="http://127.0.0.1:8040")
    login = pg_client.post("/auth/login", json={"email": args.seed_learner_email, "password": args.seed_learner_password})
    login.raise_for_status()
    seed_learner_id = login.json()["user_id"]

    ok = True
    ok &= _diff(
        "catalog",
        sqlite_client.get("/v1/lms/courses").json(),
        pg_client.get("/v1/lms/courses").json(),
        seed_learner_id=seed_learner_id,
    )
    ok &= _diff(
        "enrollments",
        sqlite_client.get("/v1/lms/enrollments", params={"learner_id": "local-learner"}).json(),
        pg_client.get("/v1/lms/enrollments").json(),
        seed_learner_id=seed_learner_id,
    )
    ok &= _diff(
        "course runs",
        sqlite_client.get("/v1/course-runs").json(),
        pg_client.get("/v1/course-runs").json(),
        seed_learner_id=seed_learner_id,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run the verification**

Run:

```bash
python -m scripts.verify_migration --seed-learner-password '<password-from-migrator-output>'
```

Expected: all checks pass (✓). Any mismatch (✗) is investigated and the migrator updated.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_migration.py
git commit -m "M3: verify_migration script — diff 8010 vs 8040 JSON endpoints"
```

---

### Task 23: Delete `app/storage/sqlite_store.py`

**Files:**
- Delete: `app/storage/sqlite_store.py`
- Modify: `tests/storage/test_workflow_store_protocol.py` (remove the SQLite-import check)

- [ ] **Step 1: Confirm no remaining imports**

Run: `grep -rn "from app.storage.sqlite_store" app/ scripts/ tests/`
Expected: zero matches in `app/` and `scripts/`. Test file `tests/storage/test_workflow_store_protocol.py` may still reference it — remove that reference.

- [ ] **Step 2: Update the protocol test**

In `tests/storage/test_workflow_store_protocol.py`, delete the `from app.storage.sqlite_store import SQLiteWorkflowStore` import and the `test_sqlite_store_satisfies_protocol` test.

- [ ] **Step 3: Delete the file**

Run: `git rm app/storage/sqlite_store.py`

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/storage/test_workflow_store_protocol.py
git commit -m "M3: delete SQLiteWorkflowStore — all callers on Postgres"
```

---

## Milestone 4 — Workspace path change

### Task 24: Change `_workspace_root` to `<user_id>/<assignment_id>` layout

**Files:**
- Modify: `app/services/lms_service.py`
- Create: `tests/lms/test_workspace_root.py`

- [ ] **Step 1: Write the failing test**

Create `tests/lms/test_workspace_root.py`:

```python
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.domain.learner import LearnerEnrollment, LearnerEnrollmentStatus, LearnerWorkspaceScope
from app.domain.registry import PackageType
from app.services.lms_service import LMSService


def _make_enrollment() -> LearnerEnrollment:
    from datetime import UTC, datetime
    return LearnerEnrollment(
        id="enr_abc",
        learner_id=str(uuid4()),
        course_run_id="course_x",
        publish_snapshot_id="snap_x",
        course_title="Title",
        course_summary="Summary",
        package_type=PackageType.coding_project,
        shared_workflow_run_id="wf_42",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        status=LearnerEnrollmentStatus.active,
        workspace_scope=LearnerWorkspaceScope.shared_course,
        deliverables=[],
    )


def test_workspace_root_uses_user_id_and_assignment_id(tmp_path: Path) -> None:
    service = LMSService.__new__(LMSService)
    service.base_dir = tmp_path
    enrollment = _make_enrollment()
    root = service._workspace_root(enrollment)
    assert root == tmp_path / enrollment.learner_id / "wf_42" / "workspace"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/lms/test_workspace_root.py -v`
Expected: FAIL — old path scheme still in use.

- [ ] **Step 3: Update `_workspace_root`**

In `app/services/lms_service.py:386`, change:

```python
def _workspace_root(self, enrollment: LearnerEnrollment) -> Path:
    return self.base_dir / enrollment.id / "workspace"
```

to:

```python
def _workspace_root(self, enrollment: LearnerEnrollment) -> Path:
    return (
        self.base_dir
        / enrollment.learner_id
        / enrollment.shared_workflow_run_id
        / "workspace"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/lms/test_workspace_root.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full lms test suite**

Run: `pytest tests/lms/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/lms_service.py tests/lms/test_workspace_root.py
git commit -m "M4: workspace path → learner_workspaces/<user_id>/<assignment_id>/workspace"
```

---

### Task 25: Workspace rename phase in the migrator

**Files:**
- Modify: `scripts/migrate_sqlite_to_postgres.py`
- Create: `tests/migration/test_workspace_rename.py`

- [ ] **Step 1: Write the failing test**

Create `tests/migration/test_workspace_rename.py`:

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from scripts.migrate_sqlite_to_postgres import copy_to_postgres, ensure_seed_user, rename_workspaces


@pytest.fixture()
def seeded_db(postgres_url: str, tmp_path: Path) -> tuple[str, Path]:
    snapshot = tmp_path / "snap.db"
    with sqlite3.connect(str(snapshot)) as conn:
        conn.executescript(
            """
            CREATE TABLE learner_enrollments (enrollment_id TEXT PRIMARY KEY, learner_id TEXT, course_run_id TEXT, status TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
            """
        )
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
    return snapshot, tmp_path


def test_rename_walks_enrollments_and_renames_dirs(postgres_url: str, seeded_db) -> None:
    snapshot, tmp_path = seeded_db
    import subprocess, os
    REPO_ROOT = Path(__file__).resolve().parents[2]
    subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_ROOT, env={**os.environ, "DATABASE_URL": postgres_url}, check=True)

    engine = create_engine(postgres_url)
    seed = ensure_seed_user(engine)
    # We need the enrollment table in the schema we just migrated. Use the snapshot data directly:
    copy_to_postgres(snapshot=snapshot, engine=engine, seed_learner_id=seed)

    old_layout = tmp_path / "learner_workspaces" / "enr_1" / "workspace"
    old_layout.mkdir(parents=True)
    (old_layout / "marker.txt").write_text("hello")

    rename_workspaces(
        engine=engine,
        old_base=tmp_path / "learner_workspaces",
        new_base=tmp_path / "new_workspaces",
    )

    new_layout = tmp_path / "new_workspaces" / str(seed) / "wf_999" / "workspace"
    assert new_layout.exists()
    assert (new_layout / "marker.txt").read_text() == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/migration/test_workspace_rename.py -v`
Expected: FAIL — `rename_workspaces` doesn't exist.

- [ ] **Step 3: Implement `rename_workspaces`**

Append to `scripts/migrate_sqlite_to_postgres.py`:

```python
import shutil


def rename_workspaces(*, engine: Engine, old_base: Path, new_base: Path) -> None:
    """Copy <old>/<enrollment_id>/workspace to <new>/<user_id>/<assignment_id>/workspace.

    Source directories are left in place because they may still be owned by 8010.
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT enrollment_id, learner_id, payload->>'shared_workflow_run_id' AS assignment_id
                FROM learner_enrollments
                """
            )
        ).all()
    for row in rows:
        old = old_base / row.enrollment_id / "workspace"
        new = new_base / row.learner_id / row.assignment_id / "workspace"
        if not old.exists():
            continue
        if new.exists():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(old, new)
```

Wire it into `main()`:

```python
    rename_workspaces(
        engine=engine,
        old_base=args.source.parent / "learner_workspaces",
        new_base=Path("learner_workspaces"),
    )
    print("Workspace rename complete.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/migration/test_workspace_rename.py -v`
Expected: PASS.

- [ ] **Step 5: Run the migrator end-to-end again**

```bash
DATABASE_URL=... python -m scripts.migrate_sqlite_to_postgres \
  --source /Users/tushar/Desktop/codebases/course-gen-codex/data/course_gen.db \
  --snapshot data/course_gen_snapshot.db \
  --skip-snapshot
```

Expected: workspaces appear under `learner_workspaces/<seed_user_id>/<assignment_id>/workspace/` in the worktree.

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_sqlite_to_postgres.py tests/migration/test_workspace_rename.py
git commit -m "M4: workspace rename phase copies old layout to user-keyed scheme"
```

---

### Task 26: End-to-end smoke against 8040

This is a manual verification step — no automated test. Confirms the full stack works.

- [ ] **Step 1: Restart the server**

Restart the 8040 server so it picks up the new workspace path scheme:

```bash
DATABASE_URL=postgresql+psycopg://course_gen:course_gen@localhost:5435/course_gen \
SESSION_SECRET=dev-secret \
uvicorn app.main:app --port 8040
```

- [ ] **Step 2: Register a fresh learner**

```bash
curl -c /tmp/jar.txt -X POST http://127.0.0.1:8040/auth/register \
  -H 'content-type: application/json' \
  -d '{"email":"smoke@example.com","password":"hunter2!!","role":"learner","display_name":"Smoke"}'
```

Expected: 201 with `user_id` and `role: learner`.

- [ ] **Step 3: Browse the catalog and enroll**

```bash
curl -b /tmp/jar.txt http://127.0.0.1:8040/v1/lms/courses
# Pick a course_run_id from the response, then:
curl -b /tmp/jar.txt -X POST http://127.0.0.1:8040/v1/lms/enrollments \
  -H 'content-type: application/json' \
  -d '{"course_run_id":"<course_id_from_catalog>"}'
```

Expected: 200 with an enrollment object whose `learner_id` matches the registered user's UUID.

- [ ] **Step 4: Launch the workspace and inspect the disk path**

```bash
curl -b /tmp/jar.txt -X POST http://127.0.0.1:8040/v1/lms/enrollments/<enrollment_id>/workspace -d '{}'
ls learner_workspaces/<user_id>/<assignment_id>/workspace/
```

Expected: directory exists at the new path and contains the seeded starter files.

- [ ] **Step 5: Re-run verification script with the new learner**

Run the `verify_migration.py` script one more time — all checks must still pass for the seed learner.

- [ ] **Step 6: Final commit (if any tweaks)**

```bash
git add -A
git commit -m "M4: smoke validation against 8040 — workspace launch + file flow"
```

---

## Self-review checklist

- [x] **Spec coverage:** Storage swap (Tasks 1-9), auth tables/sessions/routes/pages (Tasks 10-17), enrollment rewiring + migrator (Tasks 18-23), workspace path (Tasks 24-26). All four milestones mapped.
- [x] **Placeholder scan:** No TBDs or "fill in details" — Task 7 and Task 8 reference the SQLite implementation explicitly because the port is mechanical and the source-of-truth is the SQLite file; the plan calls out the exact line ranges and the translation rule.
- [x] **Type consistency:** `WorkflowStore` Protocol method names match `PostgresWorkflowStore` method names match the calls in tests. `SESSION_TTL` and `COOKIE_NAME` are imported consistently. `seed_learner_id: UUID` is the same type from migrator → verification.
- [x] **Spec requirements covered:** snapshot via `VACUUM INTO` ✓, JSON-blob shape preserved ✓, learner_id rewrite in both column and payload ✓, defensive grep for `local-learner` ✓, idempotent migration ✓, workspace rename ✓, role-guarded routes ✓.

Plan complete and saved to [docs/superpowers/plans/2026-05-14-auth-postgres-workspace.md](docs/superpowers/plans/2026-05-14-auth-postgres-workspace.md).
