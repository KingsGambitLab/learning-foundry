#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.failure_replay_smoke import FailureReplaySmokeService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the last actionable workflow failure against the current authored workspace "
            "and run a failed-deliverables-only smoke verification."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--workflow", help="Workflow run id to replay, for example run_a7c24a7d3d76.")
    target.add_argument("--course", help="Course run id to resolve to a workflow, for example course_b0307cf1cdd3.")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Invoke repo repair with the last failure packet before replaying the smoke harness.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    service = FailureReplaySmokeService()
    try:
        result = service.replay(
            workflow_run_id=args.workflow,
            course_run_id=args.course,
            repair=args.repair,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    payload = {"ok": result.smoke_passed, "result": result.model_dump(mode="json")}
    print(json.dumps(payload, indent=2))
    return 0 if result.smoke_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
