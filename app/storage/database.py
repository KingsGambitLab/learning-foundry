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
