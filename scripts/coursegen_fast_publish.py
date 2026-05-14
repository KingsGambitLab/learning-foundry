"""Drive an outcome-mode course to ``awaiting_gate_3`` using curated assets,
without going through the langgraph dispatcher's grader_repair loop.

The default graph behavior, when ``node_oracle_validation`` or
``node_oracle_curated_validation`` reports ``publishable=False``, is to
call ``node_grader_repair`` — which re-authors the oracle bundle via
``OracleAuthoring.author_oracle(spec)`` and then materializes it via
``materialize_oracle_bundle``, which WIPES ``private/grader/_setup/``,
``_reference/``, and ``scenarios/`` before re-writing.

That wipe is incompatible with a human-curated grading bundle: every
manual edit gets overwritten by the next grader_repair iteration. This
script side-steps the dispatcher entirely:

1. Runs ``scripts/curate_rag_grading_assets.py`` to populate
   ``_setup/`` + ``scenarios/`` with high-quality finance content keyed
   the way the rubrics expect.
2. Boots the reference impl via the existing
   ``WorkspaceBootSandboxAdapter``.
3. Loads scenarios from disk and executes each one against the
   reference impl via ``scenario_trace_runner.run_scenario``, producing
   the same shape ``OraclePassResult`` that ``OraclePass.run`` would.
4. Persists the oracle outputs to ``_oracle/outputs.json``.
5. Calls ``validate_oracle`` + ``validate_curated_gold`` to produce
   real validation reports against the actual scenario runs.
6. If both reports are ``publishable``, sets ``state.stage =
   awaiting_gate_3`` / ``state.status = awaiting_human`` and persists.
   The user can then POST a normal gate-3 decision to publish.
7. If either report is NOT publishable, prints the blocking reasons
   verbatim so the operator can iterate on the curated assets and re-run
   this script — no grader_repair wipe will happen.

Usage:
    python scripts/coursegen_fast_publish.py <course_run_id>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WORKTREE))


def _build_service_stack():
    """Reuse the same wiring as the FastAPI app for store + service factories."""
    from app.services.artifact_materializer import ArtifactMaterializer
    from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
    from app.services.course_artifact_materializer import CourseArtifactMaterializer
    from app.services.course_generation_service import CourseGenerationService
    from app.services.course_workflow_service import CourseWorkflowService
    from app.services.creator_asset_service import CreatorAssetService
    from app.services.learner_studio_service import LearnerStudioService
    from app.services.publish_learner_certification_service import (
        PublishLearnerCertificationService,
    )
    from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
    from app.services.workflow_service import WorkflowService
    from app.storage.sqlite_store import SQLiteWorkflowStore

    db_path = str(_WORKTREE / "data" / "course_gen.db")
    store = SQLiteWorkflowStore(db_path=db_path)
    creator_asset_service = CreatorAssetService(store)
    assignment_workspace_manager = AssignmentWorkspaceManager()
    task_agent_blackbox_runner = TaskAgentBlackBoxRunner()
    workflow_service = WorkflowService(
        store,
        ArtifactMaterializer(creator_asset_service=creator_asset_service),
        task_agent_blackbox_runner,
        assignment_workspace_manager,
    )
    learner_studio_service = LearnerStudioService(runner=task_agent_blackbox_runner)
    course_workflow_service = CourseWorkflowService(
        store,
        workflow_service,
        CourseArtifactMaterializer(),
        publish_certification_service=PublishLearnerCertificationService(
            learner_studio_service=learner_studio_service,
            enabled=True,
        ),
        creator_asset_service=creator_asset_service,
    )
    return CourseGenerationService(course_workflow_service)


def _run_curation() -> None:
    """Invoke the curation script as a subprocess so its imports stay isolated."""
    print(">>> Running curation (setup_data + 18 scenarios)…")
    result = subprocess.run(
        [
            sys.executable,
            str(_WORKTREE / "scripts" / "curate_rag_grading_assets.py"),
        ],
        cwd=str(_WORKTREE),
        check=True,
    )
    if result.returncode != 0:
        raise RuntimeError("curation script failed")


def _run_oracle_pass_in_process(state):
    """Equivalent of ``node_oracle_pass`` but with no dispatcher coupling.

    Boots the reference impl, runs each scenario against it, persists
    the outputs, and stores the result on ``state.oracle_pass_result``.
    """
    from app.services.oracle_pass import OraclePass, _load_setup_data, persist_oracle_outputs
    from app.services.scenario_loader import load_scenarios_from_dir
    from app.services.workspace_boot import WorkspaceBootSandboxAdapter
    from app.services.llm_router import get_default_router

    scenarios_dir = state.workspace_root / "private" / "grader" / "scenarios"
    ref_dir = state.workspace_root / "private" / "grader" / "_reference"
    setup_dir = state.workspace_root / "private" / "grader" / "_setup"

    scenarios = load_scenarios_from_dir(scenarios_dir)
    print(f">>> Loaded {len(scenarios)} scenarios from {scenarios_dir}")

    sandbox = WorkspaceBootSandboxAdapter()
    router = get_default_router()
    op = OraclePass(sandbox_runner=sandbox)
    pass_result = op.run(
        scenarios=scenarios,
        reference_impl_dir=ref_dir,
        setup_data_dir=setup_dir if setup_dir.exists() else None,
        router=router,
        capabilities=state.spec.capabilities,
    )
    state.oracle_pass_result = pass_result

    outputs_path = (
        state.workspace_root / "private" / "grader" / "_oracle" / "outputs.json"
    )
    persist_oracle_outputs(pass_result, outputs_path)
    print(
        f">>> oracle_pass: {pass_result.passed_scenarios}/"
        f"{pass_result.total_scenarios} passed; "
        f"failed={pass_result.failed_scenarios}, "
        f"abstained={pass_result.abstained_scenarios}"
    )
    return pass_result


def _run_validations_in_process(state):
    """Equivalent of both validation nodes, without grader_repair branching."""
    from app.services.oracle_pass import _load_setup_data
    from app.services.oracle_validation import validate_oracle, validate_curated_gold
    from app.services.scenario_loader import load_scenarios_from_dir

    scenarios_dir = state.workspace_root / "private" / "grader" / "scenarios"
    setup_dir = state.workspace_root / "private" / "grader" / "_setup"

    scenarios = load_scenarios_from_dir(scenarios_dir)
    setup_data = _load_setup_data(setup_dir if setup_dir.exists() else None)

    run_report = validate_oracle(
        spec=state.spec, scenarios=scenarios, oracle_result=state.oracle_pass_result
    )
    curated_report = validate_curated_gold(
        spec=state.spec, scenarios=scenarios, setup_data=setup_data
    )
    state.oracle_validation_report = run_report
    state.curated_validation_report = curated_report

    print(
        f">>> oracle_validation: publishable={run_report.publishable}, "
        f"blocking={len(run_report.blocking_reasons)}"
    )
    print(
        f">>> curated_validation: publishable={curated_report.publishable}, "
        f"blocking={len(curated_report.blocking_reasons)}"
    )
    return run_report, curated_report


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    course_run_id = sys.argv[1]

    # Ensure curation runs first so the rest of this script operates on
    # the high-quality bundle, not whatever the LLM author last wrote.
    _run_curation()

    service = _build_service_stack()
    state = service._load_outcome_state(course_run_id)
    if state is None:
        print(f"ERROR: outcome_state missing for '{course_run_id}'")
        return 1

    print(
        f">>> Loaded state for {course_run_id}: stage={state.stage}, "
        f"status={state.status}"
    )

    # Reset retry budgets so any internal scenario-run failures don't
    # cascade into "budget exhausted" surfaces.
    state.starter_attempt = 0
    state.grader_attempt = 0
    state.blocking_reasons = []
    state.status = "running"

    # Run the real oracle pass against the curated bundle.
    pass_result = _run_oracle_pass_in_process(state)

    # Run both validators against the curated bundle.
    run_report, curated_report = _run_validations_in_process(state)

    if run_report.publishable and curated_report.publishable:
        state.stage = "awaiting_gate_3"
        state.status = "awaiting_human"
        service._persist_outcome_state(course_run_id, state)
        print(
            "\n=== SUCCESS ===\n"
            f"Course '{course_run_id}' is now at awaiting_gate_3.\n"
            "Approve via:\n"
            f"  curl -sS -X POST http://127.0.0.1:8030/v1/course-runs/"
            f"{course_run_id}/decisions \\\n"
            "       -H 'Content-Type: application/json' \\\n"
            "       -d '{\"gate\": \"gate_3_pre_publish\", "
            '"decision": "approve", "actor": "fast-publish"}\''
        )
        return 0

    # Persist anyway so the operator can inspect via the API.
    service._persist_outcome_state(course_run_id, state)
    print("\n=== NOT YET PUBLISHABLE ===")
    print("oracle_validation blocking reasons:")
    for r in run_report.blocking_reasons[:20]:
        print(f"  - {r}")
    print("curated_validation blocking reasons:")
    for r in curated_report.blocking_reasons[:20]:
        print(f"  - {r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
