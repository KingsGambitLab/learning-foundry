"""Remediate legacy enrollments with `shared_workflow_run_id="shared_workflow"`.

Background (codex P0 follow-up):
- `lms_service.enroll` used to fall back to the literal `"shared_workflow"`
  string when both the snapshot and course run lacked a real workflow run
  id. Two outcome-mode enrollments for the same learner therefore collided
  at `learner_workspaces/<user>/shared_workflow/workspace/`.
- The lms_service fix only protects FUTURE enrollments. Existing rows
  still carry `shared_workflow_run_id="shared_workflow"`, so the second
  one's workspace continues to alias the first.

Cutover order (codex pass 3 hardening):
1. Identify legacy rows and any half-migrated rows (already rewritten
   but missing target workspace).
2. For each row, do the **filesystem cutover first**: copy
   `<base>/<learner>/shared_workflow/workspace` to
   `<base>/<learner>/<course_run_id>/workspace` if the target is
   missing. If the source is missing too, skip.
3. **Only after the copy succeeds**, update the DB row in a single
   transaction (so an interrupted copy never leaves the DB rewritten
   ahead of the filesystem).
4. Active editor sessions on the old path keep working at the OS level
   but are out-of-band of LMS bookkeeping; operators should restart
   any live workspace launches after the cutover so the new path
   takes effect.

Idempotent: rerunning is safe. The selector matches both still-legacy
rows AND already-rewritten rows whose new workspace is missing, so
interrupted runs can be repaired.

Usage:
    DATABASE_URL=postgresql+psycopg://... python -m scripts.backfill_shared_workflow_enrollments \\
        [--workspace-root /path/to/learner_workspaces] [--dry-run] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from sqlalchemy import create_engine, text


def _select_targets(conn) -> list:
    """Return all enrollment rows that either still carry the legacy
    `"shared_workflow"` value OR were previously rewritten but whose
    new workspace can't be located on disk (the repair-after-crash case).

    The caller filters the latter using the on-disk check below.
    """
    return conn.execute(text(
        "SELECT enrollment_id, learner_id, course_run_id, payload "
        "FROM learner_enrollments "
        "WHERE payload->>'shared_workflow_run_id' = 'shared_workflow'"
    )).all()


def _needs_filesystem_repair(base: Path, learner_id: str, course_run_id: str) -> bool:
    """A row whose payload was already rewritten but whose new workspace
    is missing (interrupted prior run)."""
    new_workspace = base / learner_id / course_run_id / "workspace"
    return not new_workspace.exists()


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Just report orphans (rewritten rows whose workspace is missing) and exit.",
    )
    args = parser.parse_args()

    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL or --database-url required.", file=sys.stderr)
        return 1
    engine = create_engine(url)
    base = args.workspace_root.resolve()

    with engine.begin() as conn:
        legacy = _select_targets(conn)
        # Orphan check: rows that look correct (no longer 'shared_workflow')
        # but whose new workspace doesn't exist on disk. These are the
        # post-interrupted-run survivors.
        orphans = []
        rewritten = conn.execute(text(
            "SELECT enrollment_id, learner_id, course_run_id, payload "
            "FROM learner_enrollments "
            "WHERE payload->>'shared_workflow_run_id' != 'shared_workflow'"
        )).all()
        for r in rewritten:
            run_id = r.payload.get("shared_workflow_run_id") if isinstance(r.payload, dict) else None
            if not run_id:
                continue
            if _needs_filesystem_repair(base, r.learner_id, run_id):
                # Only flag if the legacy shared_workflow path exists for
                # this learner (otherwise this is just a non-RAG enrollment
                # with a real workflow run id and no workspace yet).
                if (base / r.learner_id / "shared_workflow" / "workspace").exists():
                    orphans.append(r)

    print(f"Legacy `shared_workflow` rows still needing rewrite: {len(legacy)}")
    print(f"Orphan rows (already rewritten, workspace missing): {len(orphans)}")
    if args.check:
        for r in legacy:
            print(f"  legacy  {r.enrollment_id} learner={r.learner_id[:12]} course={r.course_run_id}")
        for r in orphans:
            print(f"  orphan  {r.enrollment_id} learner={r.learner_id[:12]} course={r.course_run_id}")
        return 0

    fs_copies = 0
    fs_skipped = 0
    db_updates = 0
    aborted_copies = 0

    # Two-stage cutover: filesystem first, DB second.
    targets = list(legacy) + list(orphans)
    for r in targets:
        new_run_id = r.course_run_id
        old_workspace = base / r.learner_id / "shared_workflow" / "workspace"
        new_workspace = base / r.learner_id / new_run_id / "workspace"

        print(f"  {r.enrollment_id}: learner={r.learner_id[:12]}... course={new_run_id}")
        print(f"    old: {old_workspace}")
        print(f"    new: {new_workspace}")

        if args.dry_run:
            continue

        # 1. Filesystem cutover (only if needed). Atomic copy: copy into
        # a sibling temp dir first, then `os.rename` it into place once
        # the copy is fully done. A SIGKILL mid-copy leaves only the
        # temp dir behind, which subsequent runs identify and clean.
        copied_now = False
        if old_workspace.exists() and not new_workspace.exists():
            new_workspace.parent.mkdir(parents=True, exist_ok=True)
            tmp_workspace = new_workspace.with_name(
                new_workspace.name + f".tmp_{os.getpid()}"
            )
            # Sweep stale temp dirs from prior crashed runs.
            for stale in new_workspace.parent.glob(new_workspace.name + ".tmp_*"):
                shutil.rmtree(stale, ignore_errors=True)
            try:
                shutil.copytree(old_workspace, tmp_workspace)
                # Atomic publish — same filesystem, so rename is atomic.
                os.rename(tmp_workspace, new_workspace)
                copied_now = True
                fs_copies += 1
            except Exception as exc:
                aborted_copies += 1
                print(
                    f"    ERROR copying workspace, leaving DB unchanged: {exc!r}",
                    file=sys.stderr,
                )
                if tmp_workspace.exists():
                    shutil.rmtree(tmp_workspace, ignore_errors=True)
                if new_workspace.exists():
                    # Belt-and-braces: if a partial rename ever lands, clean.
                    shutil.rmtree(new_workspace, ignore_errors=True)
                continue
        else:
            fs_skipped += 1

        # 2. DB rewrite (only after a successful copy or a confirmed
        # already-present new workspace). Wrapped in a transaction so a
        # partial update can never linger.
        payload = dict(r.payload) if isinstance(r.payload, dict) else json.loads(r.payload)
        if payload.get("shared_workflow_run_id") == "shared_workflow":
            payload["shared_workflow_run_id"] = new_run_id
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE learner_enrollments "
                        "SET payload = CAST(:payload AS JSONB), updated_at = :ts "
                        "WHERE enrollment_id = :eid "
                        "AND payload->>'shared_workflow_run_id' = 'shared_workflow'"
                    ),
                    {
                        "payload": json.dumps(payload),
                        "ts": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
                        "eid": r.enrollment_id,
                    },
                )
            db_updates += 1

    print(
        f"Done. db_updates={db_updates} fs_copies={fs_copies} "
        f"fs_skipped={fs_skipped} aborted_copies={aborted_copies}"
        + (" (dry-run)" if args.dry_run else "")
    )
    return 0 if aborted_copies == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
