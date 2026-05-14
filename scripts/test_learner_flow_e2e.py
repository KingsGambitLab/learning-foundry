"""End-to-end learner flow simulation for course_f918e889a33c.

Drives the same machinery a fully-wired LMS would, without enrollment
infrastructure:

1. Copy ``public/starter/`` to a fresh learner workspace under
   ``learner_workspaces/e2e_test_learner/``.
2. Plug in the reference implementation (drop ``_reference/app/main.py``
   in as the learner's ``app.py`` — simulates a learner who got the
   answer right).
3. Boot the learner workspace via ``workspace_boot.boot_and_verify``.
4. Run every hidden scenario in ``private/grader/scenarios/`` against
   that booted learner using the same ``scenario_trace_runner`` the
   oracle pass uses.
5. Validate against the spec's quality bars via
   ``oracle_validation.validate_oracle`` and
   ``validate_curated_gold``.
6. Print per-rubric verdicts + the final publishability check.

The point is to prove the grader works against an arbitrary learner
submission — the same path a published-LMS enrollment would take once
the publish_snapshot pipeline is extended for outcome-mode courses.

Usage:
    python scripts/test_learner_flow_e2e.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WORKTREE))

COURSE_ID = "course_f918e889a33c"
COURSE_ROOT = _WORKTREE / "workspaces" / "outcome" / COURSE_ID
LEARNER_ROOT = _WORKTREE / "learner_workspaces" / "e2e_test_learner"


def _seed_learner_workspace() -> None:
    """Copy ``public/starter/`` to the learner workspace.

    Wipes any previous contents so the test is deterministic.
    """
    starter = COURSE_ROOT / "public" / "starter"
    if LEARNER_ROOT.exists():
        shutil.rmtree(LEARNER_ROOT)
    shutil.copytree(starter, LEARNER_ROOT)
    print(f"[1/6] Seeded learner workspace from starter:\n  {LEARNER_ROOT}")


def _install_working_solution() -> None:
    """Drop the reference impl into the learner workspace as ``app.py``.

    The starter is a ``partial`` starter — by itself it would fail the
    semantic_eq / oracle_set_overlap rubrics. To prove the full grading
    pipeline works we install a known-good implementation (the same one
    the oracle pass already validated).
    """
    ref_main = COURSE_ROOT / "private" / "grader" / "_reference" / "app" / "main.py"
    learner_app = LEARNER_ROOT / "app.py"
    # Reference impl lives under ``app/main.py``; the starter uses a flat
    # ``app.py`` at the workspace root (verified by inspecting
    # ``run.sh``: ``uvicorn main:app`` or ``app:app`` — check below).
    run_sh = (LEARNER_ROOT / ".coursegen" / "runtime" / "run.sh").read_text()
    if "main:app" in run_sh:
        # Need a "main" module at the workspace root.
        target = LEARNER_ROOT / "main.py"
    elif "app:app" in run_sh:
        target = LEARNER_ROOT / "app.py"
    else:
        target = learner_app
    target.write_text(ref_main.read_text())
    # httpx for the testclient dep that the materializer also injects.
    reqs = LEARNER_ROOT / "requirements.txt"
    reqs_text = reqs.read_text() if reqs.exists() else ""
    if "httpx" not in reqs_text.lower():
        reqs.write_text(reqs_text.rstrip() + "\nhttpx>=0.27\n")
    print(f"[2/6] Installed working solution at {target.name} (run.sh uses {'main:app' if 'main:app' in run_sh else 'app:app'})")


def _boot_learner_and_grade() -> int:
    """Boot the learner workspace and run all hidden scenarios."""
    from app.services.oracle_pass import OraclePass, _load_setup_data, persist_oracle_outputs
    from app.services.oracle_validation import validate_oracle, validate_curated_gold
    from app.services.scenario_loader import load_scenarios_from_dir
    from app.services.workspace_boot import WorkspaceBootSandboxAdapter
    from app.services.llm_router import get_default_router
    from app.services.course_outcome_models import CourseOutcomeSpec

    spec_path = COURSE_ROOT / "private" / "course_spec.json"
    spec = CourseOutcomeSpec.model_validate_json(spec_path.read_text())

    scenarios_dir = COURSE_ROOT / "private" / "grader" / "scenarios"
    setup_dir = COURSE_ROOT / "private" / "grader" / "_setup"
    scenarios = load_scenarios_from_dir(scenarios_dir)
    setup_data = _load_setup_data(setup_dir)
    print(
        f"[3/6] Loaded spec ({len(spec.quality_bars)} quality bars), "
        f"{len(scenarios)} hidden scenarios, "
        f"{len(setup_data)} setup_data files"
    )

    sandbox = WorkspaceBootSandboxAdapter()
    router = get_default_router()
    op = OraclePass(sandbox_runner=sandbox)
    print("[4/6] Booting learner workspace via Docker (build + /health probe)...")
    result = op.run(
        scenarios=scenarios,
        reference_impl_dir=LEARNER_ROOT,  # learner workspace IS the impl now
        setup_data_dir=setup_dir,
        router=router,
        capabilities=spec.capabilities,
    )
    print(
        f"[5/6] Graded: passed={result.passed_scenarios}/"
        f"{result.total_scenarios}, "
        f"failed={result.failed_scenarios}, "
        f"abstained={result.abstained_scenarios}"
    )
    outputs_path = LEARNER_ROOT / "_grading_outputs.json"
    persist_oracle_outputs(result, outputs_path)
    print(f"      outputs persisted: {outputs_path}")

    # Per-rubric pass-rate roll-up against the spec's quality bars.
    print()
    print("[6/6] Quality-bar roll-up:")
    bar_pass_counts: dict[str, dict[str, int]] = {}
    for s in result.scenario_outputs:
        for kind, verdict in s.verdicts:
            # Approximate bar mapping: by judge kind. Real spec ties
            # scenarios → bars via quality_bar_ids; we surface raw judge
            # outcomes here for transparency.
            d = bar_pass_counts.setdefault(kind, {"pass": 0, "fail": 0, "abstain": 0})
            d[verdict.status] += 1
    for kind, counts in sorted(bar_pass_counts.items()):
        total = sum(counts.values())
        print(
            f"  {kind:30s} pass={counts['pass']:>2d} fail={counts['fail']:>2d} "
            f"abstain={counts['abstain']:>2d} (n={total})"
        )

    run_report = validate_oracle(
        spec=spec, scenarios=scenarios, oracle_result=result
    )
    curated_report = validate_curated_gold(
        spec=spec, scenarios=scenarios, setup_data=setup_data
    )

    print()
    print(
        f"  oracle_validation:  publishable={run_report.publishable}, "
        f"blocking={len(run_report.blocking_reasons)}"
    )
    print(
        f"  curated_validation: publishable={curated_report.publishable}, "
        f"blocking={len(curated_report.blocking_reasons)}"
    )

    if run_report.publishable and curated_report.publishable:
        print()
        print("=== LEARNER WOULD PASS THE COURSE ===")
        return 0
    print()
    print("=== LEARNER WOULD NOT PASS THE COURSE ===")
    for r in (run_report.blocking_reasons + curated_report.blocking_reasons)[:10]:
        print(f"  - {r}")
    return 1


def main() -> int:
    if not COURSE_ROOT.exists():
        print(f"ERROR: course root missing: {COURSE_ROOT}")
        return 1
    _seed_learner_workspace()
    _install_working_solution()
    return _boot_learner_and_grade()


if __name__ == "__main__":
    raise SystemExit(main())
