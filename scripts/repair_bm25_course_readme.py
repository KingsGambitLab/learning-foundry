"""Refresh the BM25 RAG course's learner-facing README so it matches
the actual contract (citations/passage_id, not cited_chunks/page_url)
and includes the step-by-step learner journey + sample_queries.json
walkthrough.

The README is shipped both as a file in the learner's workspace tree
and as ``deliverables[0].starter_readme`` / ``content_markdown`` in
the publish_snapshot. Both are updated here.
"""
from __future__ import annotations
import json, sqlite3, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SOURCE = REPO_ROOT / "learner_workspaces" / "enrollment_f7b44f8b27ce" / "workspace" / "README.md"

def main():
    new_readme = SOURCE.read_text()
    # Sync to active enrollments
    for d in (REPO_ROOT / "learner_workspaces").glob("enrollment_*"):
        target = d / "workspace" / "README.md"
        if target.exists():
            target.write_text(new_readme)
            print(f"synced -> {target}")
    # Patch publish_snapshot.
    c = sqlite3.connect(REPO_ROOT / "data/course_gen.db")
    c.row_factory = sqlite3.Row
    row = c.execute(
        "SELECT snapshot_id, payload_json FROM publish_snapshots "
        "WHERE course_run_id='course_f918e889a33c'"
    ).fetchone()
    p = json.loads(row["payload_json"])
    p["learner_package"]["deliverables"][0]["starter_readme"] = new_readme
    p["learner_package"]["deliverables"][0]["content_markdown"] = new_readme
    c.execute("UPDATE publish_snapshots SET payload_json=? WHERE snapshot_id=?",
              (json.dumps(p), row["snapshot_id"]))
    c.commit()
    print(f"patched publish_snapshot {row['snapshot_id']}")

if __name__ == "__main__":
    main()
