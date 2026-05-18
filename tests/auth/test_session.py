from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from app.domain.auth import Role
from app.services.auth_session import SessionService
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


@pytest.fixture()
def service(store: PostgresWorkflowStore) -> SessionService:
    return SessionService(store)


def test_create_and_load_session_returns_user_and_session(service: SessionService, store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="x@y.com", password_hash="h", role=Role.creator)
    sid = service.create(user_id=user.id, ip=None, user_agent=None)
    loaded = service.load(str(sid))
    assert loaded is not None
    assert loaded.user.id == user.id
    assert loaded.session.user_id == user.id


def test_load_unknown_session_returns_none(service: SessionService) -> None:
    assert service.load("00000000-0000-0000-0000-000000000000") is None


def test_load_invalid_session_string_returns_none(service: SessionService) -> None:
    assert service.load("not-a-uuid") is None


def test_revoke_removes_session(service: SessionService, store: PostgresWorkflowStore) -> None:
    user = store.create_user(email="r@v.com", password_hash="h", role=Role.learner)
    sid = service.create(user_id=user.id, ip=None, user_agent=None)
    service.revoke(str(sid))
    assert service.load(str(sid)) is None
