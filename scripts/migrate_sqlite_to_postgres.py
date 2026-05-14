"""Snapshot 8010's SQLite DB into a local file, copy into Postgres, rewrite local-learner.

Usage:
    python -m scripts.migrate_sqlite_to_postgres \
        --source /Users/tushar/Desktop/codebases/course-gen-codex/data/course_gen.db \
        --snapshot data/course_gen_snapshot.db \
        --database-url $DATABASE_URL

Idempotent — rerunning re-snapshots and uses ON CONFLICT DO NOTHING on Postgres.
"""
from __future__ import annotations

import argparse
import json
import secrets
import shutil
import sqlite3
from pathlib import Path
from uuid import UUID

from sqlalchemy import Engine, create_engine, text

from app.services.auth_passwords import hash_password


SEED_LEARNER_EMAIL = "legacy-local-learner@coursegen.example"

# FK-safe order: parents before children. workflow_events / course_events last
# because they may reference rows from earlier tables.
TABLES_IN_ORDER = [
    "course_runs",
    "workflow_runs",
    "publish_snapshots",
    "learner_enrollments",
    "learner_workspace_sessions",
    "learner_submissions",
    "creator_feedback",
    "learner_feedback",
    "learner_eval_reports",
    "creator_assets",
    "workflow_events",
    "course_events",
]

# Map table → primary key column name (as defined by the SQLite source schema
# and reflected in alembic/versions/0001_initial.py).
TABLE_PRIMARY_KEY = {
    "workflow_runs": "run_id",
    "course_runs": "course_run_id",
    "publish_snapshots": "snapshot_id",
    "learner_enrollments": "enrollment_id",
    "learner_workspace_sessions": "session_id",
    "learner_submissions": "submission_id",
    "creator_feedback": "feedback_id",
    "learner_feedback": "feedback_id",
    "learner_eval_reports": "report_id",
    "creator_assets": "asset_id",
    "workflow_events": "id",
    "course_events": "id",
}


def snapshot_sqlite(*, source: Path, target: Path) -> None:
    """Snapshot a SQLite DB into a new file via VACUUM INTO.

    Safe while another process is writing to source (WAL mode).
    """
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    src_uri = f"file:{source}?mode=ro"
    with sqlite3.connect(src_uri, uri=True) as conn:
        conn.execute(f"VACUUM INTO '{target.as_posix()}'")


def ensure_seed_user(engine: Engine) -> UUID:
    """Insert the seed `legacy-local-learner` row if missing. Returns its UUID. Idempotent.

    On creation, prints the temporary password to stdout once so the operator can reset it.
    """
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
        print(
            f"Seed user created. Email: {SEED_LEARNER_EMAIL}  Password: {password}\n"
            f"Save this password — it is shown only once."
        )
        return row.id


def _rewrite_learner_id(payload, seed_id: str):
    """Replace 'local-learner' with seed_id anywhere in the JSON payload tree."""
    if isinstance(payload, dict):
        return {k: _rewrite_learner_id(v, seed_id) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_rewrite_learner_id(v, seed_id) for v in payload]
    if payload == "local-learner":
        return seed_id
    return payload


def copy_to_postgres(*, snapshot: Path, engine: Engine, seed_learner_id: UUID) -> None:
    """Copy every legacy table from snapshot SQLite into Postgres.

    - ON CONFLICT (pk) DO NOTHING — safe to re-run.
    - learner_id == "local-learner" is rewritten to seed_learner_id (column + payload).
    - At end: scans every migrated table's payload for any surviving "local-learner" string
      and raises RuntimeError if found.
    """
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
                    payload_text = row_dict.pop("payload_json")
                    payload = json.loads(payload_text) if payload_text else {}

                    if table in ("learner_enrollments", "learner_feedback"):
                        payload = _rewrite_learner_id(payload, seed_str)
                    if table == "learner_enrollments" and row_dict.get("learner_id") == "local-learner":
                        row_dict["learner_id"] = seed_str

                    cols = list(row_dict.keys()) + ["payload"]
                    placeholders = []
                    for c in cols:
                        if c == "payload":
                            placeholders.append("CAST(:payload AS JSONB)")
                        else:
                            placeholders.append(f":{c}")
                    sql = (
                        f"INSERT INTO {table} ({', '.join(cols)}) "
                        f"VALUES ({', '.join(placeholders)}) "
                        f"ON CONFLICT ({pk}) DO NOTHING"
                    )
                    params = {**row_dict, "payload": json.dumps(payload)}
                    conn.execute(text(sql), params)

    # Defensive: no 'local-learner' should remain in any payload.
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


def rename_workspaces(*, engine: Engine, old_base: Path, new_base: Path) -> None:
    """Copy workspace directories from old layout to new layout.

    Old layout:  <old_base>/<enrollment_id>/workspace/
    New layout:  <new_base>/<learner_id>/<shared_workflow_run_id>/workspace/

    Behavior:
    - If the old workspace doesn't exist, skip silently.
    - If the new workspace already exists, skip (idempotent).
    - Uses shutil.copytree (copy, not move) so the source server's directory tree stays intact.
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT enrollment_id,
                       learner_id,
                       payload->>'shared_workflow_run_id' AS assignment_id
                FROM learner_enrollments
                """
            )
        ).all()
    for row in rows:
        assignment_id = row.assignment_id
        if not assignment_id:
            # No assignment id in payload — nothing to migrate for this row.
            continue
        old = old_base / row.enrollment_id / "workspace"
        new = new_base / row.learner_id / assignment_id / "workspace"
        if not old.exists():
            continue
        if new.exists():
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(old, new)
        print(f"  Copied workspace: {old} → {new}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--database-url", default=None, help="Postgres URL (defaults to DATABASE_URL env)")
    parser.add_argument("--skip-snapshot", action="store_true")
    args = parser.parse_args()

    if not args.skip_snapshot:
        print(f"Snapshotting {args.source} → {args.snapshot}")
        snapshot_sqlite(source=args.source, target=args.snapshot)
        print("Snapshot complete.")

    import os
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: --database-url or DATABASE_URL env required.")
        return 1
    engine = create_engine(url)
    seed_id = ensure_seed_user(engine)
    print(f"Seed learner id: {seed_id}")
    copy_to_postgres(snapshot=args.snapshot, engine=engine, seed_learner_id=seed_id)
    print("Copy complete.")

    # Phase 3: workspace directory rename
    # Default old_base is <source>'s sibling learner_workspaces dir.
    # Default new_base is this worktree's learner_workspaces dir.
    old_workspaces = args.source.parent / "learner_workspaces"
    new_workspaces = Path("learner_workspaces").resolve()
    print(f"Renaming workspaces: {old_workspaces} → {new_workspaces}")
    rename_workspaces(engine=engine, old_base=old_workspaces, new_base=new_workspaces)
    print("Workspace rename complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
