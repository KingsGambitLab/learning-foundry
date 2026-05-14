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
