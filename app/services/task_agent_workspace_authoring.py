from __future__ import annotations

from enum import Enum
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.sandbox import SandboxExecutionResult, SandboxExecutionStatus
from app.domain.workflow import FailureContext, WorkflowNodeExecution, WorkflowRun
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.learner_brief_builder import (
    build_task_agent_deliverable_brief,
    render_learner_starter_readme,
)
from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService
from app.services.task_agent_starter_templates import (
    HIDDEN_GRADER_SCRIPT_PATH,
    HIDDEN_MANIFEST_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    build_task_agent_starter_files,
    default_preview_command,
)


# Reviewer finding codes that ONLY affect the deterministic README template.
# When all error findings in a reviewer failure are in this set, the repair
# can be scoped to re-rendering the README — no LLM-driven re-authoring,
# no fresh roll of the dependency-manifest dice. The codes are defined in
# `app/services/bundle_validation.py` where the reviewer emits them.
_DOCUMENTATION_ONLY_FINDING_CODES = frozenset({
    "starter_readme_missing_section",
    "starter_readme_missing_local_reference",
    "starter_readme_uses_secondary_brief",
    "starter_readme_unpublished_endpoint_reference",
    "starter_readme_lacks_domain_grounding",
})


def _is_documentation_only_failure(latest_node: WorkflowNodeExecution) -> bool:
    """True when every error finding's code is purely a README/text issue.

    `severity == "info"` findings are not gating; we only consider errors.
    A node with zero errors is not a failure we should fast-path through.
    """
    error_findings = [
        f for f in latest_node.findings if f.severity.value == "error"
    ]
    if not error_findings:
        return False
    return all(
        (f.code or "") in _DOCUMENTATION_ONLY_FINDING_CODES
        for f in error_findings
    )


def _deliverable_ids_from_findings(latest_node: WorkflowNodeExecution) -> set[str]:
    """Extract `deliverable_X` ids from finding `location` strings.

    Findings carry locations like `public/checks/deliverable_1/README.md`
    or `public/starter/deliverable_2/...`. Pull the deliverable id out.
    """
    ids: set[str] = set()
    for f in latest_node.findings:
        location = f.location or ""
        parts = [p for p in location.replace("\\", "/").split("/") if p]
        for part in parts:
            if part.startswith("deliverable_"):
                ids.add(part)
                break
    return ids


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

        # Proportional repair scope: when every error finding is a
        # documentation-only issue (README path/section/grounding), regenerate
        # the deterministic README templates only. Do NOT call
        # ``author_workspace_repo`` — that fresh LLM roll over the entire
        # workspace silently swaps dependency manifests and causes
        # "repair regression" failures (working psycopg pin → missing psycopg2
        # transitive after a README-only repair).
        if _is_documentation_only_failure(latest_node):
            return self._repair_readmes_only(run, latest_node)

        failed_deliverables = self._target_deliverable_ids(
            run=run,
            latest_node=latest_node,
            failure_context=failure_context,
        )
        full_repair = not failed_deliverables
        if full_repair:
            before_fingerprint = self._workspace_fingerprint(run, deliverable_ids=sorted(failed_deliverables))
            # DO NOT call self.sync_workspace(run) here — sync_workspace
            # chains to prepare_run_workspace(overwrite=True) which does
            # shutil.rmtree(bundle_root) and re-materializes from default
            # templates. That wipe destroys authored per-deliverable
            # manifests (every deliverable's starter_repo_bundle is reset
            # to `starter_default`). If author_workspace_repo below then
            # fails (OpenAI down, partial bundle, race), the manifests
            # stay starter_default and the next reviewer_tests fires
            # `starter_repo_bundle_not_authored` indefinitely.
            # author_workspace_repo overwrites the relevant files in place,
            # so the prior wipe was strictly destructive.
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

    def _repair_readmes_only(
        self,
        run: WorkflowRun,
        latest_node: WorkflowNodeExecution,
    ) -> tuple[WorkflowRun, bool, str]:
        """Documentation-only fast path: re-render the deterministic README
        templates for the affected deliverables. Code, dependency manifest,
        and runtime protocol files are left byte-for-byte unchanged.
        """
        spec = run.artifacts.task_agent_spec
        workspace = run.artifacts.workspace_snapshot
        assert spec is not None and workspace is not None

        affected_ids = _deliverable_ids_from_findings(latest_node)
        if not affected_ids:
            # No location-attributed findings — fall back to all deliverables.
            affected_ids = {d.id for d in spec.deliverables}

        workspace_root = Path(workspace.root_dir)
        public_root = Path(workspace.public_dir)
        updated: list[str] = []
        shared_codebase = bool(spec.course_structure.shared_codebase)
        for did in sorted(affected_ids):
            deliverable = next(
                (d for d in spec.deliverables if d.id == did), None
            )
            if deliverable is None:
                continue
            readme_content = self._render_deliverable_readme(spec, did)
            if shared_codebase:
                readme_path = public_root / "checks" / did / "README.md"
            else:
                readme_path = public_root / "starter" / did / "README.md"
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(readme_content, encoding="utf-8")
            try:
                relative = readme_path.relative_to(workspace_root)
            except ValueError:
                relative = readme_path
            updated.append(str(relative))

        if not updated:
            return run, False, "No README files were re-rendered."
        return (
            run,
            True,
            (
                "Re-rendered "
                + ", ".join(updated)
                + " from the deterministic README template. "
                "Source code and dependency manifest were not touched "
                "(documentation-only finding scope)."
            ),
        )

    def _render_deliverable_readme(
        self,
        spec,
        deliverable_id: str,
    ) -> str:
        deliverable = next(
            d for d in spec.deliverables if d.id == deliverable_id
        )
        brief = deliverable.learner_brief or build_task_agent_deliverable_brief(
            spec, deliverable
        )
        return render_learner_starter_readme(
            title=f"Starter for {deliverable.title}",
            brief=brief,
            summary=deliverable.objective,
            learning_outcomes=list(deliverable.learning_outcomes),
            visible_check_command=(
                spec.runtime_dependencies.visible_check_command
                or f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"
            ),
            preview_command=(
                spec.runtime_dependencies.preview_command
                or default_preview_command(spec, host="127.0.0.1")
            ),
            public_checks=deliverable.public_checks,
            implementation_language=spec.runtime_dependencies.implementation_language,
            language_version=spec.runtime_dependencies.language_version,
            package_manager=spec.runtime_dependencies.package_manager,
        )

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

        if spec.course_structure.shared_codebase:
            shared_starter_dir = Path(workspace.public_dir) / "starter"
            # Shared starter content from first deliverable's template.
            first_deliverable = spec.deliverables[0]
            shared_files = build_task_agent_starter_files(spec, first_deliverable.id)
            per_deliverable_only = {
                HIDDEN_MANIFEST_PATH,
                HIDDEN_GRADER_SCRIPT_PATH,
                "checks/run_visible_checks.py",
            }
            for relative_path, content in shared_files.items():
                if relative_path in per_deliverable_only:
                    continue
                if relative_path.startswith("checks/") or relative_path.startswith(".coursegen/grader/"):
                    continue
                updated_files.extend(
                    self._write_if_needed(
                        shared_starter_dir / relative_path,
                        content,
                        workspace.root_dir,
                        force=force,
                    )
                )
            # Per-deliverable artifacts in public/checks/<id>/ and private/grader/<id>/.
            #
            # IMPORTANT: deliverable.json is INTENTIONALLY not written here.
            # The manifest is owned by:
            #   - ArtifactMaterializer (initial creation, default template)
            #   - OpenAIStarterRepoAuthoringService._apply_progressive_bundle
            #     (authored metadata: starter_repo_bundle.source=openai_live,
            #      runtime_protocol_bundle, dependency_contract)
            #   - OpenAITestScriptAuthoringService.author_workspace_tests
            #     (generated_test_scripts.source)
            #
            # Writing the default template here destroys the authored
            # state on every authoring_runtime invocation, causing
            # reviewer_code/reviewer_tests to fail with
            # `starter_repo_bundle_not_authored` even when the files on
            # disk are correctly authored. This was the root cause
            # behind the Go d2 stale-manifest divergence, the Rails
            # reviewer_tests loop, and the TypeScript reviewer_code
            # failure observed today.
            for deliverable in spec.deliverables:
                if deliverable.id not in allowed_deliverables:
                    continue
                deliverable_files = build_task_agent_starter_files(spec, deliverable.id)
                checks_dir = Path(workspace.public_dir) / "checks" / deliverable.id
                grader_dir = Path(workspace.root_dir) / "private" / "grader" / deliverable.id
                updated_files.extend(
                    self._write_if_needed(
                        checks_dir / "run_visible_checks.py",
                        deliverable_files["checks/run_visible_checks.py"],
                        workspace.root_dir,
                        force=force,
                    )
                )
                updated_files.extend(
                    self._write_if_needed(
                        grader_dir / "run_hidden_checks.py",
                        deliverable_files[HIDDEN_GRADER_SCRIPT_PATH],
                        workspace.root_dir,
                        force=force,
                    )
                )
            return updated_files

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
        spec = run.artifacts.task_agent_spec
        public_dir = Path(workspace.public_dir)
        workspace_root = Path(workspace.root_dir)
        allowed = set(deliverable_ids or [])
        fingerprints: dict[str, str] = {}

        if spec is not None and spec.course_structure.shared_codebase:
            shared_starter = public_dir / "starter"
            if shared_starter.exists():
                for file_path in sorted(
                    path for path in shared_starter.rglob("*") if path.is_file()
                ):
                    fingerprints[str(file_path.relative_to(workspace_root))] = sha256(
                        file_path.read_bytes()
                    ).hexdigest()
            checks_root = public_dir / "checks"
            grader_root = workspace_root / "private" / "grader"
            for parent in (checks_root, grader_root):
                if not parent.exists():
                    continue
                for deliverable_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
                    if allowed and deliverable_dir.name not in allowed:
                        continue
                    for file_path in sorted(
                        path for path in deliverable_dir.rglob("*") if path.is_file()
                    ):
                        fingerprints[str(file_path.relative_to(workspace_root))] = sha256(
                            file_path.read_bytes()
                        ).hexdigest()
            return fingerprints

        starter_root = public_dir / "starter"
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
