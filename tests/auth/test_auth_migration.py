from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def migrated(postgres_url: str) -> None:
    alembic = Path(sys.executable).parent / "alembic"
    subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": postgres_url},
        check=True,
    )


def test_auth_tables_exist(postgres_url: str, migrated: None) -> None:
    inspector = inspect(create_engine(postgres_url))
    tables = set(inspector.get_table_names())
    assert {"users", "user_sessions"} <= tables


def test_users_email_unique(postgres_url: str, migrated: None) -> None:
    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO users (email, password_hash, role) VALUES ('a@b.com', 'h', 'creator')"
        ))
    with engine.connect() as conn:
        with pytest.raises(Exception):
            with engine.begin() as inner:
                inner.execute(text(
                    "INSERT INTO users (email, password_hash, role) VALUES ('a@b.com', 'h', 'creator')"
                ))


def test_users_role_constrained(postgres_url: str, migrated: None) -> None:
    engine = create_engine(postgres_url)
    with pytest.raises(Exception):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO users (email, password_hash, role) VALUES ('c@d.com', 'h', 'invalid_role')"
            ))


def test_email_lookup_case_insensitive(postgres_url: str, migrated: None) -> None:
    """CITEXT means SELECT by email is case-insensitive."""
    engine = create_engine(postgres_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO users (email, password_hash, role) VALUES ('MixedCase@example.com', 'h', 'learner')"
        ))
    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT email FROM users WHERE email = 'mixedcase@example.com'"
        )).first()
    assert row is not None
