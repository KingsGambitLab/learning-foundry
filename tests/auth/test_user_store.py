from __future__ import annotations

import os
import subprocess
import sys
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


def test_create_and_get_user_by_email(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="a@b.com", password_hash="h", role=Role.learner, display_name="Alice")
    fetched = store.get_user_by_email("a@b.com")
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.role is Role.learner
    assert fetched.display_name == "Alice"


def test_get_user_by_email_case_insensitive(store: PostgresWorkflowStore) -> None:
    store.create_user(email="Bob@Example.com", password_hash="h", role=Role.creator)
    assert store.get_user_by_email("bob@example.com") is not None


def test_create_user_rejects_duplicate_email(store: PostgresWorkflowStore) -> None:
    store.create_user(email="dup@x.com", password_hash="h", role=Role.creator)
    with pytest.raises(ValueError):
        store.create_user(email="dup@x.com", password_hash="h", role=Role.creator)


def test_get_user_by_id_returns_user(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="b@c.com", password_hash="h", role=Role.creator)
    fetched = store.get_user_by_id(user.id)
    assert fetched is not None and fetched.email == "b@c.com"


def test_get_user_password_hash(store: PostgresWorkflowStore) -> None:
    store.create_user(email="ph@x.com", password_hash="secret-hash", role=Role.learner)
    assert store.get_user_password_hash("ph@x.com") == "secret-hash"
    assert store.get_user_password_hash("unknown@x.com") is None


def test_create_and_load_session(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="s@s.com", password_hash="h", role=Role.learner)
    expires_at = datetime.now(UTC) + timedelta(days=14)
    sid = store.create_user_session(user_id=user.id, expires_at=expires_at, ip=None, user_agent="pytest")
    loaded = store.load_user_session(sid)
    assert loaded is not None
    assert loaded.user_id == user.id
    assert loaded.user_agent == "pytest"


def test_load_session_returns_none_for_expired(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="exp@e.com", password_hash="h", role=Role.learner)
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    sid = store.create_user_session(user_id=user.id, expires_at=expired_at, ip=None, user_agent=None)
    assert store.load_user_session(sid) is None


def test_revoke_session(store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="r@r.com", password_hash="h", role=Role.creator)
    sid = store.create_user_session(
        user_id=user.id,
        expires_at=datetime.now(UTC) + timedelta(days=1),
        ip=None,
        user_agent=None,
    )
    store.revoke_user_session(sid)
    assert store.load_user_session(sid) is None
