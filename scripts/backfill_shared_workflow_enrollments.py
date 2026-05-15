"""Remediate legacy enrollments with `shared_workflow_run_id="shared_workflow"`.

Background (codex P0 follow-up):
- `lms_service.enroll` used to fall back to the literal `"shared_workflow"`
  string when both the snapshot and course run lacked a real workflow run
  id. Two outcome-mode enrollments for the same learner therefore collided
  at `learner_workspaces/<user>/shared_workflow/workspace/`.
- The lms_service fix only protects FUTURE enrollments. Existing rows
  still carry `shared_workflow_run_id="shared_workflow"`, so the second
  one's workspace continues to alias the first.

This script:
1. Finds every `learner_enrollments` row whose payload
   `shared_workflow_run_id` is the literal `"shared_workflow"`.
2. Rewrites the payload to set it to `course_run_id` (course-unique).
3. For each affected learner, copies the shared workspace directory to
   the new per-course path (`learner_workspaces/<user>/<course_run_id>/workspace/`).
4. Leaves the original directory in place so any in-flight editor
   sessions don't break mid-edit. The next workspace launch will use the
   new path.

Idempotent: re-running is safe. New rows already use `course_run_id`.

Usage:
    DATABASE_URL=postgresql+psycopg://... python -m scripts.backfill_shared_workflow_enrollments \\
        [--workspace-root /path/to/learner_workspaces] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from sqlalchemy import create_engine, text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("learner_workspaces"),
        help="Base directory where learner workspaces live on disk.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL or --database-url required.", file=sys.stderr)
        return 1
    engine = create_engine(url)
    base = args.workspace_root.resolve()

    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT enrollment_id, learner_id, course_run_id, payload "
            "FROM learner_enrollments "
            "WHERE payload->>'shared_workflow_run_id' = 'shared_workflow'"
        )).all()

    print(f"Legacy `shared_workflow` enrollments to remediate: {len(rows)}")
    if not rows:
        return 0

    db_updates = 0
    fs_copies = 0
    fs_skipped = 0

    for r in rows:
        new_run_id = r.course_run_id
        old_workspace = base / r.learner_id / "shared_workflow" / "workspace"
        new_workspace = base / r.learner_id / new_run_id / "workspace"
        action = "rewrite"
        print(f"  {r.enrollment_id}: learner={r.learner_id[:12]}... course={new_run_id}")
        print(f"    old: {old_workspace}")
        print(f"    new: {new_workspace}")

        if args.dry_run:
            continue

        # 1. Rewrite payload column.
        payload = dict(r.payload)
        payload["shared_workflow_run_id"] = new_run_id
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE learner_enrollments SET payload = CAST(:payload AS JSONB) "
                    "WHERE enrollment_id = :eid"
                ),
                {"payload": json.dumps(payload), "eid": r.enrollment_id},
            )
        db_updates += 1

        # 2. Copy workspace directory to the new per-course path.
        if old_workspace.exists() and not new_workspace.exists():
            new_workspace.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(old_workspace, new_workspace)
            fs_copies += 1
        else:
            fs_skipped += 1

    print(
        f"Done. db_updates={db_updates} fs_copies={fs_copies} fs_skipped={fs_skipped}"
        + (" (dry-run)" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
