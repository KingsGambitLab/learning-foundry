from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.workflow import WorkflowFailureOwnerHint, WorkflowNodeExecution, WorkflowNodeKind, WorkflowNodeStatus, WorkflowRun
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.failure_context_builder import build_failure_context
from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.storage.postgres_store import PostgresWorkflowStore
from app.storage.workflow_store import WorkflowStore


class FailureReplaySmokeResult(BaseModel):
    workflow_run_id: str
    replay_run_id: str
    course_run_id: str | None = None
    selected_node_kind: str
    selected_node_attempt: int
    owner_hint: WorkflowFailureOwnerHint
    failure_signature: str | None = None
    target_deliverable_ids: list[str] = Field(default_factory=list)
    repaired: bool = False
    repair_summary: str | None = None
    smoke_passed: bool
    smoke_summary: str
    added_ai_requests: int = 0
    added_estimated_cost_usd: float = 0.0


class FailureReplaySmokeService:
    def __init__(
        self,
        *,
        store: WorkflowStore | None = None,
        workspace_authoring_service_factory: Callable[[Path], TaskAgentWorkspaceAuthoringService] | None = None,
    ) -> None:
        self.store = store or PostgresWorkflowStore()
        self.workspace_authoring_service_factory = (
            workspace_authoring_service_factory or self._default_workspace_authoring_service
        )

    def replay(
        self,
        *,
        workflow_run_id: str | None = None,
        course_run_id: str | None = None,
        repair: bool = False,
    ) -> FailureReplaySmokeResult:
        run, resolved_course_run_id = self._resolve_run(
            workflow_run_id=workflow_run_id,
            course_run_id=course_run_id,
        )
        latest_node = self._select_failure_node(run)
        failure_context = build_failure_context(run, latest_node)

        with TemporaryDirectory(prefix="course-gen-replay-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            replay_run = self._clone_run_into_temp_workspace(run, temp_dir)
            workspace_service = self.workspace_authoring_service_factory(temp_dir)
            target_deliverables = sorted(
                workspace_service.target_deliverable_ids(
                    replay_run,
                    latest_node=latest_node,
                    failure_context=failure_context,
                )
            )
            before_usage = replay_run.artifacts.ai_usage.model_copy(deep=True)
            repaired = False
            repair_summary: str | None = None
            if repair:
                replay_run, repaired, repair_summary = workspace_service.repair_workspace(
                    replay_run,
                    latest_node,
                    failure_context=failure_context,
                )
            smoke = workspace_service.smoke_verify_repair(
                replay_run,
                latest_node,
                failure_context=failure_context,
            )
            after_usage = replay_run.artifacts.ai_usage

        return FailureReplaySmokeResult(
            workflow_run_id=run.id,
            replay_run_id=replay_run.id,
            course_run_id=resolved_course_run_id,
            selected_node_kind=latest_node.kind.value,
            selected_node_attempt=latest_node.attempt,
            owner_hint=failure_context.owner_hint,
            failure_signature=failure_context.failure_signature,
            target_deliverable_ids=target_deliverables,
            repaired=repaired,
            repair_summary=repair_summary,
            smoke_passed=smoke.passed,
            smoke_summary=smoke.summary,
            added_ai_requests=max(after_usage.request_count - before_usage.request_count, 0),
            added_estimated_cost_usd=max(
                (after_usage.estimated_cost_usd or 0.0) - (before_usage.estimated_cost_usd or 0.0),
                0.0,
            ),
        )

    def _resolve_run(
        self,
        *,
        workflow_run_id: str | None,
        course_run_id: str | None,
    ) -> tuple[WorkflowRun, str | None]:
        if workflow_run_id:
            run = self.store.get_run(workflow_run_id)
            if run is None:
                raise ValueError(f"Workflow run '{workflow_run_id}' was not found.")
            return run, course_run_id
        if not course_run_id:
            raise ValueError("Provide either --workflow or --course.")
        course_run = self.store.get_course_run(course_run_id)
        if course_run is None:
            raise ValueError(f"Course run '{course_run_id}' was not found.")
        if not course_run.shared_workflow_run_id:
            raise ValueError(f"Course run '{course_run_id}' is not linked to a workflow run yet.")
        run = self.store.get_run(course_run.shared_workflow_run_id)
        if run is None:
            raise ValueError(
                f"Workflow run '{course_run.shared_workflow_run_id}' linked from course '{course_run_id}' was not found."
            )
        return run, course_run_id

    def _select_failure_node(self, run: WorkflowRun) -> WorkflowNodeExecution:
        failed_nodes = [
            node
            for node in run.artifacts.node_executions
            if node.status == WorkflowNodeStatus.failed
        ]
        if not failed_nodes:
            raise ValueError(f"Workflow run '{run.id}' has no failed nodes to replay.")
        actionable_nodes = [
            node
            for node in failed_nodes
            if node.kind not in {WorkflowNodeKind.authoring_repair, WorkflowNodeKind.reviewer_repair}
        ]
        candidates = actionable_nodes or failed_nodes
        return candidates[-1]

    def _clone_run_into_temp_workspace(self, run: WorkflowRun, temp_dir: Path) -> WorkflowRun:
        replay_run = run.model_copy(deep=True)
        replay_run.id = f"{run.id}-replay-{uuid4().hex[:8]}"
        replay_run.artifacts.materialized_bundle = None
        snapshot = replay_run.artifacts.workspace_snapshot
        if snapshot is None:
            workspace_manager = AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces")
            replay_run.artifacts.workspace_snapshot = workspace_manager.prepare_run_workspace(
                replay_run,
                overwrite=True,
            )
            return replay_run

        source_root = Path(snapshot.root_dir)
        target_root = temp_dir / "workspace"
        shutil.copytree(source_root, target_root)
        replay_run.artifacts.workspace_snapshot = snapshot.model_copy(
            update={
                "root_dir": str(target_root),
                "public_dir": str(target_root / "public"),
                "private_dir": str(target_root / "private"),
                "manifest_path": str(target_root / Path(snapshot.manifest_path).name),
            }
        )
        return replay_run

    def _default_workspace_authoring_service(self, temp_dir: Path) -> TaskAgentWorkspaceAuthoringService:
        return TaskAgentWorkspaceAuthoringService(
            workspace_manager=AssignmentWorkspaceManager(base_dir=temp_dir / "workspaces"),
            repo_authoring_service=OpenAIStarterRepoAuthoringService(enabled=True),
        )
