from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.workflow import FailureContext, WorkflowNodeExecution, WorkflowRun
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.task_agent_starter_templates import (
    build_task_agent_starter_files,
    render_task_agent_runtime_deliverable,
)


class WorkspaceAuthoringSource(str, Enum):
    deterministic_template = "deterministic_template"


class WorkspaceAuthoringResult(BaseModel):
    source: WorkspaceAuthoringSource = WorkspaceAuthoringSource.deterministic_template
    updated_files: list[str] = Field(default_factory=list)
    message: str


class TaskAgentWorkspaceAuthoringService:
    def __init__(self, workspace_manager: AssignmentWorkspaceManager | None = None) -> None:
        self.workspace_manager = workspace_manager or AssignmentWorkspaceManager()

    def ensure_workspace(self, run: WorkflowRun, *, overwrite: bool = False) -> WorkflowRun:
        workspace = run.artifacts.workspace_snapshot
        if overwrite or workspace is None or not Path(workspace.root_dir).exists():
            run.artifacts.workspace_snapshot = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
        return run

    def author_workspace(self, run: WorkflowRun) -> tuple[WorkflowRun, WorkspaceAuthoringResult]:
        run = self.ensure_workspace(run)
        updated_files = self._write_runtime_files(run)
        return run, WorkspaceAuthoringResult(
            updated_files=updated_files,
            message=(
                "Prepared shared runtime and starter app wrappers in the persistent workspace."
                if updated_files
                else "Persistent workspace already matched the current starter runtime templates."
            ),
        )

    def repair_workspace(
        self,
        run: WorkflowRun,
        latest_node: WorkflowNodeExecution,
        failure_context: FailureContext | None = None,
    ) -> tuple[WorkflowRun, bool, str]:
        if run.artifacts.task_agent_spec is None:
            return run, False, "No task-agent spec is available to repair the workspace."

        run = self.ensure_workspace(run)
        workspace = run.artifacts.workspace_snapshot
        if workspace is None:
            return run, False, "The workspace is missing and could not be prepared."

        failed_deliverables = {
            report.deliverable_id
            for report in (latest_node.sandbox_result.deliverable_reports if latest_node.sandbox_result else [])
            if not report.compile_succeeded or not report.runtime_succeeded
        }
        if failure_context is not None and failure_context.sandbox is not None:
            failed_deliverables.update(failure_context.sandbox.failed_deliverables)
        finding_text = " ".join(
            f"{finding.title} {finding.detail}"
            for finding in (failure_context.findings if failure_context is not None else latest_node.findings)
        ).lower()
        full_repair = (
            not failed_deliverables
            or "placeholder starter" in finding_text
            or "starter endpoints remain" in finding_text
            or (
                failure_context is not None
                and failure_context.sandbox is not None
                and bool(
                    failure_context.sandbox.build_stderr_excerpt
                    or failure_context.sandbox.run_stderr_excerpt
                )
            )
        )
        if full_repair:
            run = self.sync_workspace(run)
            reason = ""
            if failure_context is not None and failure_context.sandbox is not None:
                if failure_context.sandbox.error:
                    reason = f" Latest sandbox error: {failure_context.sandbox.error}"
                elif failure_context.sandbox.build_stderr_excerpt or failure_context.sandbox.run_stderr_excerpt:
                    reason = " Latest sandbox stderr was carried into the repair step."
            return (
                run,
                True,
                "Rematerialized the full learner workspace to resync runtime and learner-facing artifacts."
                + reason,
            )
        updated_files = self._write_runtime_files(
            run,
            deliverable_ids=sorted(failed_deliverables),
            force=True,
        )
        if updated_files:
            reason = ""
            if failure_context is not None and failure_context.sandbox is not None:
                if failure_context.sandbox.error:
                    reason = f" Latest sandbox error: {failure_context.sandbox.error}"
                elif failure_context.sandbox.build_stderr_excerpt or failure_context.sandbox.run_stderr_excerpt:
                    reason = " Latest sandbox stderr was carried into the repair step."
            return (
                run,
                True,
                (
                    "Re-rendered the shared runtime and starter wrappers for the failed workspace deliverables."
                    if not full_repair
                    else "Re-rendered the shared runtime and starter wrappers across the workspace."
                )
                + reason,
            )
        return run, False, "No workspace file changes were needed for the current sandbox failure."

    def sync_workspace(self, run: WorkflowRun) -> WorkflowRun:
        return self.ensure_workspace(run, overwrite=True)

    def _write_runtime_files(
        self,
        run: WorkflowRun,
        *,
        deliverable_ids: list[str] | None = None,
        force: bool = False,
    ) -> list[str]:
        spec = run.artifacts.task_agent_spec
        workspace = run.artifacts.workspace_snapshot
        if spec is None or workspace is None:
            return []

        updated_files: list[str] = []
        runtime_dir = Path(workspace.public_dir) / "runtime"
        runtime_path = runtime_dir / "task_agent_runtime.py"
        updated_files.extend(
            self._write_if_needed(
                runtime_path,
                render_task_agent_runtime_deliverable(),
                workspace.root_dir,
                force=force,
            )
        )

        allowed_deliverables = set(deliverable_ids or [deliverable.id for deliverable in spec.deliverables])
        for deliverable in spec.deliverables:
            if deliverable.id not in allowed_deliverables:
                continue
            for relative_path, content in build_task_agent_starter_files(spec, deliverable.id).items():
                deliverable_file = Path(workspace.public_dir) / "starter" / deliverable.id / relative_path
                updated_files.extend(
                    self._write_if_needed(
                        deliverable_file,
                        content,
                        workspace.root_dir,
                        force=force,
                    )
                )
        return updated_files

    def _write_if_needed(
        self,
        path: Path,
        content: str,
        workspace_root: str,
        *,
        force: bool,
    ) -> list[str]:
        if not force and path.exists():
            current = path.read_text(encoding="utf-8")
            if current == content:
                return []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return [str(path.relative_to(workspace_root))]
