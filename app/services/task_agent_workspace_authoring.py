from __future__ import annotations

from enum import Enum
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.sandbox import SandboxExecutionResult, SandboxExecutionStatus
from app.domain.workflow import FailureContext, WorkflowNodeExecution, WorkflowRun
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService
from app.services.task_agent_starter_templates import (
    build_task_agent_starter_files,
)


class WorkspaceAuthoringSource(str, Enum):
    deterministic_template = "deterministic_template"


class WorkspaceAuthoringResult(BaseModel):
    source: WorkspaceAuthoringSource = WorkspaceAuthoringSource.deterministic_template
    updated_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    message: str


class WorkspaceRepairSmokeResult(BaseModel):
    passed: bool
    summary: str
    sandbox_result: SandboxExecutionResult | None = None


class TaskAgentWorkspaceAuthoringService:
    def __init__(
        self,
        workspace_manager: AssignmentWorkspaceManager | None = None,
        repo_authoring_service: OpenAIStarterRepoAuthoringService | None = None,
        sandbox_runner: DockerSandboxRunner | None = None,
    ) -> None:
        self.workspace_manager = workspace_manager or AssignmentWorkspaceManager()
        self.repo_authoring_service = repo_authoring_service or OpenAIStarterRepoAuthoringService(
            enabled=False
        )
        self.sandbox_runner = sandbox_runner or DockerSandboxRunner(workspace_manager=self.workspace_manager)

    def ensure_workspace(self, run: WorkflowRun, *, overwrite: bool = False) -> WorkflowRun:
        workspace = run.artifacts.workspace_snapshot
        if overwrite or workspace is None or not Path(workspace.root_dir).exists():
            run.artifacts.workspace_snapshot = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
        return run

    def author_workspace(self, run: WorkflowRun) -> tuple[WorkflowRun, WorkspaceAuthoringResult]:
        run = self.ensure_workspace(run)
        updated_files = self._write_protocol_files(run)
        run, repo_result = self.repo_authoring_service.author_workspace_repo(run)
        updated_files.extend(repo_result.updated_files)
        return run, WorkspaceAuthoringResult(
            updated_files=updated_files,
            notes=list(repo_result.notes),
            message=(
                "Prepared the shared harness protocol and authored learner-owned repo files in the persistent workspace."
                if updated_files
                else "Persistent workspace already matched the current authored repo bundle and harness protocol."
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

        failed_deliverables = self._target_deliverable_ids(
            run=run,
            latest_node=latest_node,
            failure_context=failure_context,
        )
        full_repair = not failed_deliverables
        if full_repair:
            before_fingerprint = self._workspace_fingerprint(run, deliverable_ids=sorted(failed_deliverables))
            run = self.sync_workspace(run)
            run, repo_result = self.repo_authoring_service.author_workspace_repo(
                run,
                failure_context=failure_context,
            )
            changed = before_fingerprint != self._workspace_fingerprint(run, deliverable_ids=sorted(failed_deliverables))
            reason = ""
            if failure_context is not None and failure_context.sandbox is not None:
                if failure_context.sandbox.error:
                    reason = f" Latest sandbox error: {failure_context.sandbox.error}"
                elif failure_context.sandbox.build_stderr_excerpt or failure_context.sandbox.run_stderr_excerpt:
                    reason = " Latest sandbox stderr was carried into the repair step."
            if changed:
                reason += " Repo files were regenerated from the latest harness feedback."
            if changed:
                return (
                    run,
                    True,
                    "Rematerialized the full learner workspace to resync runtime and learner-facing artifacts."
                    + reason,
                )
            return run, False, "No workspace file changes were needed for the current sandbox failure."
        before_fingerprint = self._workspace_fingerprint(run, deliverable_ids=sorted(failed_deliverables))
        updated_files = self._write_protocol_files(
            run,
            deliverable_ids=sorted(failed_deliverables),
            force=True,
        )
        run, repo_result = self.repo_authoring_service.author_workspace_repo(
            run,
            failure_context=failure_context,
            deliverable_ids=sorted(failed_deliverables),
        )
        updated_files.extend(repo_result.updated_files)
        changed = before_fingerprint != self._workspace_fingerprint(run, deliverable_ids=sorted(failed_deliverables))
        if changed:
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

    def smoke_verify_repair(
        self,
        run: WorkflowRun,
        latest_node: WorkflowNodeExecution,
        *,
        failure_context: FailureContext | None = None,
    ) -> WorkspaceRepairSmokeResult:
        if run.artifacts.task_agent_spec is None or run.artifacts.workspace_snapshot is None:
            return WorkspaceRepairSmokeResult(
                passed=False,
                summary="Workspace repair smoke verification could not run because the spec or workspace is missing.",
            )

        target_deliverables = self._target_deliverable_ids(
            run=run,
            latest_node=latest_node,
            failure_context=failure_context,
        )
        if not target_deliverables:
            return WorkspaceRepairSmokeResult(
                passed=True,
                summary="Workspace repair smoke verification was skipped because no failed deliverables were identified.",
            )

        smoke_run = run.model_copy(deep=True)
        smoke_run.id = f"{run.id}-repair-smoke"
        smoke_run.title = f"{run.title} (repair smoke)"
        smoke_spec = smoke_run.artifacts.task_agent_spec.model_copy(deep=True)
        smoke_spec.deliverables = [
            deliverable
            for deliverable in smoke_spec.deliverables
            if deliverable.id in target_deliverables
        ]
        smoke_run.artifacts.task_agent_spec = smoke_spec

        sandbox_result = self.sandbox_runner.execute(smoke_run)
        if (
            sandbox_result.status == SandboxExecutionStatus.passed
            and sandbox_result.build_succeeded
            and sandbox_result.run_succeeded
        ):
            return WorkspaceRepairSmokeResult(
                passed=True,
                summary=(
                    "Workspace repair smoke verification passed for "
                    + ", ".join(sorted(target_deliverables))
                    + "."
                ),
                sandbox_result=sandbox_result,
            )

        failure_bits: list[str] = []
        if sandbox_result.error:
            failure_bits.append(sandbox_result.error)
        for report in sandbox_result.deliverable_reports:
            if report.compile_succeeded and report.runtime_succeeded:
                continue
            error_text = (report.error or report.stderr or "").strip()
            if not error_text:
                error_text = "deliverable smoke verification failed"
            failure_bits.append(f"{report.deliverable_id}: {error_text}")

        detail = "; ".join(failure_bits[:4]) if failure_bits else "starter smoke verification failed"
        return WorkspaceRepairSmokeResult(
            passed=False,
            summary=(
                "Workspace repair smoke verification still failed for "
                + ", ".join(sorted(target_deliverables))
                + f". {detail}"
            ),
            sandbox_result=sandbox_result,
        )

    def sync_workspace(self, run: WorkflowRun) -> WorkflowRun:
        return self.ensure_workspace(run, overwrite=True)

    def target_deliverable_ids(
        self,
        run: WorkflowRun,
        *,
        latest_node: WorkflowNodeExecution,
        failure_context: FailureContext | None = None,
    ) -> set[str]:
        return self._target_deliverable_ids(
            run=run,
            latest_node=latest_node,
            failure_context=failure_context,
        )

    def _write_protocol_files(
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
        if path.exists():
            current = path.read_text(encoding="utf-8")
            if current == content:
                return []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return [str(path.relative_to(workspace_root))]

    def _workspace_fingerprint(
        self,
        run: WorkflowRun,
        *,
        deliverable_ids: list[str] | None = None,
    ) -> dict[str, str]:
        workspace = run.artifacts.workspace_snapshot
        if workspace is None:
            return {}
        public_dir = Path(workspace.public_dir)
        starter_root = public_dir / "starter"
        allowed = set(deliverable_ids or [])
        fingerprints: dict[str, str] = {}
        if not starter_root.exists():
            return fingerprints
        for deliverable_root in sorted(path for path in starter_root.iterdir() if path.is_dir()):
            if allowed and deliverable_root.name not in allowed:
                continue
            for file_path in sorted(path for path in deliverable_root.rglob("*") if path.is_file()):
                fingerprints[str(file_path.relative_to(public_dir))] = sha256(file_path.read_bytes()).hexdigest()
        return fingerprints

    def _target_deliverable_ids(
        self,
        run: WorkflowRun,
        *,
        latest_node: WorkflowNodeExecution,
        failure_context: FailureContext | None,
    ) -> set[str]:
        spec = run.artifacts.task_agent_spec
        failed_deliverables = {
            report.deliverable_id
            for report in (latest_node.sandbox_result.deliverable_reports if latest_node.sandbox_result else [])
            if not report.compile_succeeded or not report.runtime_succeeded
        }
        if failure_context is not None and failure_context.sandbox is not None:
            failed_deliverables.update(failure_context.sandbox.failed_deliverables)
        if spec is None or not spec.course_structure.shared_codebase or not failed_deliverables:
            return failed_deliverables

        deliverable_order = [deliverable.id for deliverable in spec.deliverables]
        deliverable_positions = [
            deliverable_order.index(deliverable_id)
            for deliverable_id in failed_deliverables
            if deliverable_id in deliverable_order
        ]
        if not deliverable_positions:
            return set()

        if failure_context is not None and failure_context.phase in {
            "dependency_materialization",
            "install",
            "verify",
            "container_launch",
        }:
            return set(deliverable_order)

        earliest_failed_index = min(deliverable_positions)
        return set(deliverable_order[earliest_failed_index:])
