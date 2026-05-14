from __future__ import annotations

from sqlalchemy import text

from app.storage.database import build_engine


def test_build_engine_returns_working_postgres_engine(postgres_url: str) -> None:
    engine = build_engine(postgres_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar_one()
    assert result == 1
