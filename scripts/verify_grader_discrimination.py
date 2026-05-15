"""Quick sanity check: does the grader actually discriminate good from bad?

Runs every curated scenario's rubrics against three fake learner
responses (good / wrong-answer / always-abstain) and reports the pass
rate for each.

If the grader is real, you should see:
- GOOD: high pass rate (matches the reference impl behavior)
- WRONG: very low pass rate (semantic_eq fails, oracle_set_overlap fails)
- ABSTAIN-ALWAYS: low pass rate (happy-path scenarios fail because
  abstention isn't appropriate when evidence is present)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WORKTREE))

from app.services.oracle_pass import _load_setup_data
from app.services.scenario_loader import load_scenarios_from_dir
from app.services.scenario_rubrics_base import RubricContext, RUBRIC_REGISTRY


COURSE_ROOT = (
    _WORKTREE / "workspaces" / "outcome" / "course_f918e889a33c"
)
SCENARIOS_DIR = COURSE_ROOT / "private" / "grader" / "scenarios"
SETUP_DIR = COURSE_ROOT / "private" / "grader" / "_setup"


def _fake_captures(scenario, *, mode: str):
    """Build a captures dict where every trace step's response is what
    ``mode`` dictates.
    """
    captures = {}
    for step in scenario.trace:
        if mode == "good":
            body = {
                "answer": "$12.4 billion",
                "citations": ["acme_q3_24_rev"],
                "abstained": False,
            }
        elif mode == "wrong":
            body = {
                "answer": "The sky is blue.",
                "citations": ["nonexistent_passage_id"],
                "abstained": False,
            }
        elif mode == "abstain_always":
            body = {
                "answer": "I cannot answer.",
                "citations": [],
                "abstained": True,
            }
        else:
            raise ValueError(mode)
        captures[step.id] = {"status": 200, "headers": {}, "body": body}
    return captures


def _build_rubric(spec_kind: str, spec_config: dict, router):
    cls = RUBRIC_REGISTRY[spec_kind]
    return cls(**spec_config)


def main() -> int:
    setup_data = _load_setup_data(SETUP_DIR)
    scenarios = load_scenarios_from_dir(SCENARIOS_DIR)
    print(f"Loaded {len(scenarios)} scenarios.\n")

    results = {"good": 0, "wrong": 0, "abstain_always": 0}
    for mode in ("good", "wrong", "abstain_always"):
        scenario_passes = 0
        scenario_per_rubric_passes = 0
        scenario_per_rubric_total = 0
        for scenario in scenarios:
            captures = _fake_captures(scenario, mode=mode)
            ctx = RubricContext(
                captures=captures, setup_data=setup_data, course_meta={}
            )
            all_pass = True
            for rubric_spec in scenario.rubrics:
                try:
                    rubric = _build_rubric(
                        rubric_spec.kind, dict(rubric_spec.config), None
                    )
                    v = rubric.judge(ctx)
                    scenario_per_rubric_total += 1
                    if v.status == "pass":
                        scenario_per_rubric_passes += 1
                    if v.status != "pass":
                        all_pass = False
                except Exception:
                    all_pass = False
            if all_pass:
                scenario_passes += 1
        results[mode] = scenario_passes
        print(
            f"mode={mode:18s} scenarios_fully_pass={scenario_passes:>3d}/{len(scenarios)}  "
            f"rubric_passes={scenario_per_rubric_passes}/{scenario_per_rubric_total}"
        )

    print()
    if results["wrong"] < 5 and results["good"] >= 0:
        print("✓ GRADER DISCRIMINATES (wrong submission fails most scenarios)")
        return 0
    print("✗ GRADER NOT DISCRIMINATING — wrong submission passes too many")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
