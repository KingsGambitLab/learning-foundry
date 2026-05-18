"""Migrate a specific set of outcome-mode courses + enrollments from a
SQLite snapshot into a target Postgres, rewriting the authoring
workspace path and the learner identity.

This is the surgical counterpart to ``migrate_sqlite_to_postgres.py``:
instead of copying the whole DB, it copies only the named course_runs
(+ their publish_snapshots) and the named enrollments, with two
rewrites required for a cross-host move:

1. ``payload_json.outcome_state.workspace_root`` — the on-disk path to
   the authoring workspace (which holds ``private/grader/``). On the
   source it points at a local dev path; on the target host the grader
   bundle lives somewhere else, so the prefix is rewritten.
2. ``learner_id`` (the row column AND the JSON payload) — the source
   uses the ``local-learner`` placeholder; the target needs a real
   ``users.id`` UUID.

Idempotent: ON CONFLICT DO UPDATE on every table.

Usage (run on the target host, DATABASE_URL pointing at its Postgres):
    python -m scripts.migrate_selected_courses \\
        --source /path/to/snapshot.db \\
        --course-id course_f918e889a33c --course-id course_wikiqa_v1 \\
        --enrollment-id enrollment_12cd1e518a94 \\
        --old-workspace-prefix /Users/.../worktrees/loving-wilson-e5e126/workspaces/outcome \\
        --new-workspace-prefix /opt/course-gen-codex/outcome_workspaces \\
        --learner-id <target-user-uuid>
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any

from sqlalchemy import create_engine, text


def _rewrite_workspace_root(obj: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(obj, dict):
        return {k: _rewrite_workspace_root(v, old_prefix, new_prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_workspace_root(v, old_prefix, new_prefix) for v in obj]
    if isinstance(obj, str) and obj.startswith(old_prefix):
        return new_prefix + obj[len(old_prefix):]
    return obj


def _rewrite_learner(obj: Any, target_uuid: str) -> Any:
    if isinstance(obj, dict):
        return {k: _rewrite_learner(v, target_uuid) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_learner(v, target_uuid) for v in obj]
    if obj == "local-learner":
        return target_uuid
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--course-id", action="append", default=[], dest="course_ids")
    ap.add_argument("--enrollment-id", action="append", default=[], dest="enrollment_ids")
    ap.add_argument("--old-workspace-prefix", required=True)
    ap.add_argument("--new-workspace-prefix", required=True)
    ap.add_argument("--learner-id", required=True, help="Target users.id UUID for migrated enrollments.")
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()

    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL or --database-url required.", file=sys.stderr)
        return 1
    engine = create_engine(url)

    src = sqlite3.connect(f"file:{args.source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row

    def _course_payload(cid: str) -> dict:
        r = src.execute(
            "SELECT payload_json FROM course_runs WHERE course_run_id=?", (cid,)
        ).fetchone()
        if r is None:
            raise SystemExit(f"course {cid} not in snapshot")
        return json.loads(r["payload_json"])

    # ---- course_runs ----
    for cid in args.course_ids:
        p = _course_payload(cid)
        p = _rewrite_workspace_root(p, args.old_workspace_prefix, args.new_workspace_prefix)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO course_runs
                      (course_run_id, title, package_type, stage, status,
                       created_at, updated_at, payload)
                    VALUES
                      (:cid, :title, :ptype, :stage, :status,
                       :created_at, :updated_at, CAST(:payload AS JSONB))
                    ON CONFLICT (course_run_id) DO UPDATE SET
                      title=EXCLUDED.title, package_type=EXCLUDED.package_type,
                      stage=EXCLUDED.stage, status=EXCLUDED.status,
                      updated_at=EXCLUDED.updated_at, payload=EXCLUDED.payload
                    """
                ),
                {
                    "cid": cid,
                    "title": p.get("title", cid),
                    "ptype": (p.get("package_type") or "progressive_codebase_course"),
                    "stage": (p.get("stage") or "published"),
                    "status": (p.get("status") or "published"),
                    "created_at": p.get("created_at"),
                    "updated_at": p.get("updated_at"),
                    "payload": json.dumps(p),
                },
            )
        print(f"course_run migrated: {cid}")

        # ---- publish_snapshots for this course ----
        snaps = src.execute(
            "SELECT snapshot_id, course_run_id, created_at, version, payload_json "
            "FROM publish_snapshots WHERE course_run_id=?",
            (cid,),
        ).fetchall()
        for s in snaps:
            sp = json.loads(s["payload_json"])
            sp = _rewrite_workspace_root(sp, args.old_workspace_prefix, args.new_workspace_prefix)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO publish_snapshots
                          (snapshot_id, course_run_id, created_at, version, payload)
                        VALUES (:sid, :cid, :created_at, :version, CAST(:payload AS JSONB))
                        ON CONFLICT (snapshot_id) DO UPDATE SET
                          payload=EXCLUDED.payload, version=EXCLUDED.version
                        """
                    ),
                    {
                        "sid": s["snapshot_id"],
                        "cid": s["course_run_id"],
                        "created_at": s["created_at"],
                        "version": s["version"],
                        "payload": json.dumps(sp),
                    },
                )
            print(f"  snapshot migrated: {s['snapshot_id']}")

    # ---- enrollments ----
    for eid in args.enrollment_ids:
        r = src.execute(
            "SELECT enrollment_id, learner_id, course_run_id, status, "
            "created_at, updated_at, payload_json FROM learner_enrollments "
            "WHERE enrollment_id=?",
            (eid,),
        ).fetchone()
        if r is None:
            raise SystemExit(f"enrollment {eid} not in snapshot")
        ep = json.loads(r["payload_json"])
        ep = _rewrite_learner(ep, args.learner_id)
        # M3 fix parity: never persist the literal "shared_workflow"
        # fallback — make the workspace key course-unique.
        if ep.get("shared_workflow_run_id") in (None, "", "shared_workflow"):
            ep["shared_workflow_run_id"] = ep["course_run_id"]
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO learner_enrollments
                      (enrollment_id, learner_id, course_run_id, status,
                       created_at, updated_at, payload)
                    VALUES
                      (:eid, :lid, :cid, :status, :created_at, :updated_at,
                       CAST(:payload AS JSONB))
                    ON CONFLICT (enrollment_id) DO UPDATE SET
                      learner_id=EXCLUDED.learner_id, status=EXCLUDED.status,
                      updated_at=EXCLUDED.updated_at, payload=EXCLUDED.payload
                    """
                ),
                {
                    "eid": r["enrollment_id"],
                    "lid": args.learner_id,
                    "cid": r["course_run_id"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "payload": json.dumps(ep),
                },
            )
        print(f"enrollment migrated: {eid} -> learner {args.learner_id}, swrid={ep['shared_workflow_run_id']}")

    # Defensive: no local-learner / old prefix should survive.
    with engine.begin() as conn:
        for tbl in ("course_runs", "publish_snapshots", "learner_enrollments"):
            bad = conn.execute(
                text(
                    f"SELECT COUNT(*) FROM {tbl} "
                    f"WHERE payload::text LIKE '%local-learner%' "
                    f"OR payload::text LIKE :oldp"
                ),
                {"oldp": f"%{args.old_workspace_prefix}%"},
            ).scalar_one()
            if bad:
                print(f"  WARNING: {tbl} still has {bad} rows with stale refs", file=sys.stderr)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
