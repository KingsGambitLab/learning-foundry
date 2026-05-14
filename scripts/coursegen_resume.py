"""Resume a blocked / paused outcome-mode course run from an arbitrary stage.

Use when:
- A course is ``blocked`` because a retry budget was exhausted, but you've
  fixed the on-disk artifact (e.g., scenarios moved to the right depth,
  Dockerfile patched, requirements amended) and want the graph to
  re-evaluate the next node instead of re-authoring everything.
- A course is ``awaiting_gate_N`` and you want to roll it back to an
  earlier stage to retry it without authoring a fresh course (saves
  planner + spec-review LLM calls).

Behavior
--------
1. Loads ``outcome_state`` from ``data/course_gen.db`` for the given
   course_run_id (state lives inside ``payload_json.payload_json.outcome_state``
   — note the double nesting; this script handles both layouts).
2. Resets the requested resume point:
   - ``stage`` ← target stage you pass in
   - ``status`` ← "running" (so the graph dispatcher picks it up)
   - ``starter_attempt`` / ``grader_attempt`` ← 0 (reclaim the retry budget)
   - ``blocking_reasons`` ← []
3. Invokes ``OutcomeWorkflowGraph.execute(state, deps=production_deps)``
   directly — no HTTP round-trip, no fresh OpenAI planner call.
4. Persists the new state via the production
   ``CourseGenerationService._persist_outcome_state``.

Usage
-----
    python scripts/coursegen_resume.py <course_run_id> <stage>

    # Examples (stage names match ``OutcomeWorkflowState.stage`` values):
    #   awaiting_gate_1 / awaiting_gate_2 / awaiting_gate_3
    #   starter_authoring / starter_verify / starter_review
    #   oracle_authoring / oracle_pass / oracle_validation
    #   oracle_curated_validation / publish

Notes
-----
- The script does NOT mutate top-level ``course_runs.stage`` /
  ``course_runs.status`` directly; ``_persist_outcome_state`` derives those
  from the new ``outcome_state``.
- If you want to bypass a gate without approving it, set the target to the
  stage AFTER the gate (e.g., ``starter_authoring`` instead of
  ``awaiting_gate_1``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the worktree is on sys.path before any app imports.
_WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WORKTREE))

from app.services.langgraph_outcome_graph import OutcomeWorkflowGraph


def _build_service_stack():
    """Replicate the FastAPI startup wiring in-process.

    Mirrors ``app/main.py``'s service construction so this script gets the
    same ``CourseGenerationService`` instance the live server uses (same
    SQLite DB, same router, same materializer).
    """
    # Imports kept inside the function so a typo elsewhere doesn't break
    # ``--help`` / ``argv != 3`` paths.
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
    learner_studio_service = LearnerStudioService(
        runner=task_agent_blackbox_runner,
    )
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


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    course_run_id = sys.argv[1]
    target_stage = sys.argv[2]

    service = _build_service_stack()

    state = service._load_outcome_state(course_run_id)
    if state is None:
        print(
            f"ERROR: no outcome_state found for course_run '{course_run_id}'. "
            f"Is this a legacy (per-deliverable) course?"
        )
        return 1

    print(
        f"=== BEFORE resume ===\n"
        f"  stage           : {state.stage}\n"
        f"  status          : {state.status}\n"
        f"  starter_attempt : {state.starter_attempt}\n"
        f"  grader_attempt  : {state.grader_attempt}\n"
        f"  blocking_count  : {len(state.blocking_reasons)}\n"
    )

    state.stage = target_stage
    state.status = "running"
    state.starter_attempt = 0
    state.grader_attempt = 0
    state.blocking_reasons = []

    deps = service._build_production_outcome_deps()
    graph = OutcomeWorkflowGraph()
    state = graph.execute(state, deps=deps)
    service._persist_outcome_state(course_run_id, state)

    print(
        f"=== AFTER resume ===\n"
        f"  stage           : {state.stage}\n"
        f"  status          : {state.status}\n"
        f"  starter_attempt : {state.starter_attempt}\n"
        f"  grader_attempt  : {state.grader_attempt}\n"
        f"  blocking_count  : {len(state.blocking_reasons)}\n"
    )
    if state.blocking_reasons:
        print("  blocking_reasons (first 5):")
        for reason in state.blocking_reasons[:5]:
            print(f"    - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
