"""Add citation-precision rubrics to the BM25 RAG course's scenarios.

The grader already checks citation RECALL (gold passages must appear
in the learner's citations list) via ``oracle_set_overlap``. It does
NOT check PRECISION — a learner could return every passage_id in the
request OR fabricate IDs and still pass.

This patch wires ``subset_match`` rubrics into the 11 happy_path /
boundary / adversarial scenarios so every cited passage_id must
appear in the request's ``search_results``. ``min_overlap=1.0``
means strict — any fabricated citation fails the rubric.

The rubric uses the new ``call_X.request.body.search_results`` path
(commit cfe04a39 added ``request`` to every capture entry).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = REPO_ROOT / "workspaces" / "outcome" / "course_f918e889a33c" / "private" / "grader" / "scenarios"


def patch_scenario(path: Path) -> bool:
    text = path.read_text()
    # Skip categories without grounded citations to check.
    if re.search(r"category:\s*(malformed_input|out_of_scope)", text):
        return False
    # Avoid double-patching.
    if "kind: subset_match" in text:
        return False
    # Need a rubrics block; idempotency scenarios use a multi-step trace
    # but assert behavioral_equivalence, not citation correctness — skip.
    if "rubrics:" not in text:
        return False
    if re.search(r"category:\s*idempotency", text):
        return False
    # Find the call_<sid> capture id from the trace step.
    m = re.search(r"capture:\s*(call_[\w]+)", text)
    if not m:
        return False
    capture_id = m.group(1)
    precision_rubric = (
        f"- kind: subset_match\n"
        f"  target: {capture_id}.body.citations\n"
        f"  acceptable_source: {capture_id}.request.body.search_results\n"
        f"  acceptable_key: passage_id\n"
        f"  min_overlap: 1.0\n"
    )
    # Append at the end (which is already the rubric list tail).
    new_text = text.rstrip() + "\n" + precision_rubric
    path.write_text(new_text)
    return True


def main() -> None:
    patched = []
    for yaml_path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        if patch_scenario(yaml_path):
            patched.append(yaml_path.name)
    print(f"patched {len(patched)} scenarios with subset_match (citation precision):")
    for name in patched:
        print(f"  + {name}")


if __name__ == "__main__":
    main()
