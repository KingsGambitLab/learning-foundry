from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.sandbox import (
    DeliverableSandboxReport,
    SandboxAvailability,
    SandboxExecutionResult,
    SandboxFailureStage,
    SandboxExecutionStatus,
)
from app.domain.task_agent import TaskAgentServiceSpec
from app.domain.workflow import WorkflowRun
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.artifact_materializer import (
    ArtifactMaterializer,
    DELIVERABLE_MANIFEST_RELATIVE_PATH,
    VISIBLE_CHECK_SCRIPT_RELATIVE_PATH,
    deliverable_grader_dir,
    deliverable_visible_checks_dir,
)
from app.services.coursegen_logging import log_coursegen_event
from app.services.dependency_contract_materializer import DependencyContractMaterializer
from app.services.generated_test_harness import GeneratedTestScriptRunner
from app.services.learner_studio_service import LearnerStudioService, RuntimeImageBuildError


class DockerSandboxRunner:
    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        build_timeout_s: int = 600,
        run_timeout_s: int = 600,
        keep_image: bool = False,
        cache_images: bool = True,
        cache_namespace: str = "course-gen-cache",
        workspace_manager: AssignmentWorkspaceManager | None = None,
        dependency_contract_materializer: DependencyContractMaterializer | None = None,
    ) -> None:
        self.docker_binary = docker_binary
        self.build_timeout_s = build_timeout_s
        self.run_timeout_s = run_timeout_s
        self.keep_image = keep_image
        self.cache_images = cache_images
        self.cache_namespace = cache_namespace
        self.workspace_manager = workspace_manager
        # start_timeout_s gates how long `_wait_for_http` polls /health
        # before giving up. Slow installs (Rails bundle install, npm/yarn
        # with native modules, Maven first-run deps) need 3-5 min. The
        # prior 90s cap caused Rails sandboxes to fail mid-install with
        # confusing "failed during install: <last bundler progress
        # line>" messages despite the container being healthy and
        # progressing. The timeout-aware summarizer plus a 300s cap lets
        # heavy stacks complete first-run installs.
        self.runtime_harness = LearnerStudioService(
            docker_binary=docker_binary,
            build_timeout_s=build_timeout_s,
            start_timeout_s=min(run_timeout_s, 300),
            host="127.0.0.1",
        )
        self.test_script_runner = GeneratedTestScriptRunner(command_timeout_s=min(run_timeout_s, 300))
        self.dependency_contract_materializer = dependency_contract_materializer or DependencyContractMaterializer(
            docker_binary=docker_binary,
            command_timeout_s=min(build_timeout_s, 600),
        )

    def status(self) -> SandboxAvailability:
        try:
            version = subprocess.run(
                [self.docker_binary, "info", "--format", "{{json .ServerVersion}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return SandboxAvailability(
                available=False,
                message=f"Docker sandbox unavailable: {exc}",
            )

        if version.returncode != 0:
            detail = (version.stderr or version.stdout).strip() or "docker info failed"
            return SandboxAvailability(
                available=False,
                message=f"Docker sandbox unavailable: {detail}",
            )

        return SandboxAvailability(
            available=True,
            message="Docker daemon is available for assignment sandbox execution.",
            docker_version=version.stdout.strip().strip('"') or None,
        )

    def execute(self, run: WorkflowRun) -> SandboxExecutionResult:
        availability = self.status()
        started = time.perf_counter()
        now = datetime.now(UTC)
        log_coursegen_event(
            "sandbox_execute_started",
            workflow_run_id=run.id,
            title=run.title,
            docker_available=availability.available,
        )

        if run.artifacts.task_agent_spec is None:
            result = SandboxExecutionResult(
                status=SandboxExecutionStatus.unavailable,
                available=availability.available,
                generated_at=now,
                duration_ms=0,
                error="Sandbox execution only supports task-agent workflow runs.",
            )
            log_coursegen_event(
                "sandbox_execute_completed",
                workflow_run_id=run.id,
                title=run.title,
                sandbox_status=result.status.value,
                error=result.error,
                duration_ms=result.duration_ms,
            )
            return result

        if not availability.available:
            result = SandboxExecutionResult(
                status=SandboxExecutionStatus.unavailable,
                available=False,
                generated_at=now,
                duration_ms=0,
                error=availability.message,
            )
            log_coursegen_event(
                "sandbox_execute_completed",
                workflow_run_id=run.id,
                title=run.title,
                sandbox_status=result.status.value,
                error=result.error,
                duration_ms=result.duration_ms,
            )
            return result

        try:
            bundle = self._materialize_workspace(run)
            workspace_root = Path(bundle.public_dir)
            log_coursegen_event(
                "sandbox_workspace_ready",
                workflow_run_id=run.id,
                title=run.title,
                workspace_root=str(workspace_root),
            )
            starter_root = workspace_root / "starter"
            if starter_root.exists():
                result = self._execute_starter_harness(
                    workspace_root=workspace_root,
                    spec=run.artifacts.task_agent_spec,
                    workflow_run_id=run.id,
                    now=now,
                    started=started,
                )
            else:
                result = self._execute_legacy_runtime(
                    workspace_root=workspace_root,
                    now=now,
                    started=started,
                )
            log_coursegen_event(
                "sandbox_execute_completed",
                workflow_run_id=run.id,
                title=run.title,
                sandbox_status=result.status.value,
                build_succeeded=result.build_succeeded,
                run_succeeded=result.run_succeeded,
                deliverable_report_count=len(result.deliverable_reports),
                duration_ms=result.duration_ms,
                error=result.error,
            )
            return result
        except subprocess.TimeoutExpired as exc:
            result = SandboxExecutionResult(
                status=SandboxExecutionStatus.failed,
                available=True,
                build_succeeded=False,
                build_cached=False,
                run_succeeded=False,
                generated_at=now,
                duration_ms=int((time.perf_counter() - started) * 1000),
                workspace_root=None,
                image_tag=None,
                cache_key=None,
                build_command=[],
                run_command=[],
                build_stdout=self._coerce_bytes(getattr(exc, "stdout", b"")),
                build_stderr=self._coerce_bytes(getattr(exc, "stderr", b"")),
                error=f"Docker sandbox timed out: {exc}",
            )
            log_coursegen_event(
                "sandbox_execute_completed",
                workflow_run_id=run.id,
                title=run.title,
                sandbox_status=result.status.value,
                build_succeeded=result.build_succeeded,
                run_succeeded=result.run_succeeded,
                duration_ms=result.duration_ms,
                error=result.error,
            )
            return result

    def _materialize_workspace(self, run: WorkflowRun):
        if run.artifacts.workspace_snapshot is not None:
            existing = Path(run.artifacts.workspace_snapshot.root_dir)
            if existing.exists():
                return run.artifacts.workspace_snapshot
            if self.workspace_manager is not None:
                bundle = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
                run.artifacts.workspace_snapshot = bundle
                return bundle
            materializer = ArtifactMaterializer()
            return materializer.materialize_run(run, overwrite=True)
        if self.workspace_manager is not None:
            bundle = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
            run.artifacts.workspace_snapshot = bundle
            return bundle
        materializer = ArtifactMaterializer()
        return materializer.materialize_run(run, overwrite=True)

    def _execute_legacy_runtime(
        self,
        *,
        workspace_root: Path,
        now: datetime,
        started: float,
    ) -> SandboxExecutionResult:
        image_tag = f"course-gen-{workspace_root.name.lower()}-{uuid4().hex[:8]}"
        cache_key: str | None = None
        build_cached = False
        runtime_dir = workspace_root / "runtime"
        dockerfile = runtime_dir / "Dockerfile"
        if self.cache_images:
            cache_key = self._workspace_cache_key(workspace_root)
            image_tag = self._cached_image_tag(cache_key)

        build_command = [
            self.docker_binary,
            "build",
            "-f",
            str(dockerfile.relative_to(workspace_root)),
            "-t",
            image_tag,
            ".",
        ]
        if self.cache_images and self._image_exists(image_tag):
            build_cached = True
            build_result = subprocess.CompletedProcess(
                build_command,
                0,
                stdout=f"Reused cached Docker image {image_tag} for workspace hash {cache_key}.",
                stderr="",
            )
        else:
            build_result = subprocess.run(
                build_command,
                cwd=workspace_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.build_timeout_s,
            )
            if build_result.returncode != 0:
                return SandboxExecutionResult(
                    status=SandboxExecutionStatus.failed,
                    available=True,
                    build_succeeded=False,
                    build_cached=build_cached,
                    run_succeeded=False,
                    generated_at=now,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    workspace_root=str(workspace_root),
                    image_tag=image_tag,
                    cache_key=cache_key,
                    build_command=build_command,
                    build_stdout=build_result.stdout,
                    build_stderr=build_result.stderr,
                    error="Docker build failed for the generated assignment runtime.",
                )

        run_command = [self.docker_binary, "run", "--rm", image_tag]
        run_result = subprocess.run(
            run_command,
            cwd=workspace_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.run_timeout_s,
        )
        parsed = self._parse_run_output(run_result.stdout)
        deliverable_reports = [
            DeliverableSandboxReport.model_validate(item)
            for item in parsed.get("deliverable_reports", [])
        ]
        run_succeeded = run_result.returncode == 0 and bool(parsed.get("success"))
        return SandboxExecutionResult(
            status=SandboxExecutionStatus.passed if run_succeeded else SandboxExecutionStatus.failed,
            available=True,
            build_succeeded=True,
            build_cached=build_cached,
            run_succeeded=run_succeeded,
            generated_at=now,
            duration_ms=int((time.perf_counter() - started) * 1000),
            workspace_root=str(workspace_root),
            image_tag=image_tag,
            cache_key=cache_key,
            build_command=build_command,
            run_command=run_command,
            build_stdout=build_result.stdout,
            build_stderr=build_result.stderr,
            run_stdout=run_result.stdout,
            run_stderr=run_result.stderr,
            deliverable_reports=deliverable_reports,
            error=None if run_succeeded else parsed.get("error") or "Assignment sandbox verification failed.",
        )

    def _execute_starter_harness(
        self,
        *,
        workspace_root: Path,
        spec: TaskAgentServiceSpec,
        workflow_run_id: str,
        now: datetime,
        started: float,
    ) -> SandboxExecutionResult:
        build_stdout_parts: list[str] = []
        build_stderr_parts: list[str] = []
        run_stdout_parts: list[str] = []
        run_stderr_parts: list[str] = []
        deliverable_reports: list[DeliverableSandboxReport] = []
        build_command: list[str] = []
        run_command: list[str] = []
        all_builds_succeeded = True
        all_runs_succeeded = True
        any_cached = False
        fail_fast = bool(spec.course_structure.shared_codebase)
        log_coursegen_event(
            "sandbox_starter_harness_started",
            workflow_run_id=workflow_run_id,
            workspace_root=str(workspace_root),
            deliverable_count=len(spec.deliverables),
        )

        shared_codebase = bool(spec.course_structure.shared_codebase)
        shared_starter_root = workspace_root / "starter"

        if shared_codebase:
            return self._execute_shared_starter_harness(
                workspace_root=workspace_root,
                shared_starter_root=shared_starter_root,
                spec=spec,
                workflow_run_id=workflow_run_id,
                now=now,
                started=started,
                fail_fast=fail_fast,
            )

        for deliverable in spec.deliverables:
            # Non-shared (legacy) path: one starter per deliverable.
            starter_root = workspace_root / "starter" / deliverable.id
            log_coursegen_event(
                "sandbox_deliverable_started",
                workflow_run_id=workflow_run_id,
                deliverable_id=deliverable.id,
                deliverable_title=deliverable.title,
                starter_root=str(starter_root),
            )
            if not starter_root.exists():
                all_builds_succeeded = False
                log_coursegen_event(
                    "sandbox_deliverable_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    sandbox_status="failed",
                    error="Starter workspace is missing for this deliverable.",
                )
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=False,
                        runtime_succeeded=False,
                        error="Starter workspace is missing for this deliverable.",
                    )
                )
                continue

            host_port = self.runtime_harness._allocate_port()
            container_name = f"course-gen-sandbox-{deliverable.id}-{uuid4().hex[:8]}".lower()
            network_name = f"{container_name}-net"
            base_url = f"http://127.0.0.1:{host_port}"
            logs = ""
            runtime_image_ready = False
            current_runtime_workspace: Path | None = None
            fail_fast_triggered = False
            dependency_services: list[dict] = []
            try:
                manifest = self.runtime_harness._runtime_manifest(starter_root)
                materialization = self.dependency_contract_materializer.materialize(
                    starter_root=starter_root,
                    runtime_plan=spec.project_contract.runtime_plan,
                    deliverable_id=deliverable.id,
                )
                if materialization.attempted:
                    log_coursegen_event(
                        "sandbox_dependency_contract_materialized",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        image_name=materialization.image_name,
                        synced_paths=materialization.synced_paths,
                        success=materialization.succeeded,
                        error=materialization.error,
                    )
                    if materialization.stdout:
                        build_stdout_parts.append(
                            f"[{deliverable.id}] Dependency contract materialization stdout:\n{materialization.stdout}".strip()
                        )
                    if materialization.stderr:
                        build_stderr_parts.append(
                            f"[{deliverable.id}] Dependency contract materialization stderr:\n{materialization.stderr}".strip()
                        )
                if not materialization.succeeded:
                    all_builds_succeeded = False
                    log_coursegen_event(
                        "sandbox_deliverable_completed",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        sandbox_status="failed",
                        error=materialization.error,
                    )
                    deliverable_reports.append(
                        DeliverableSandboxReport(
                            deliverable_id=deliverable.id,
                            compile_succeeded=False,
                            runtime_succeeded=False,
                            failed_stage=SandboxFailureStage.dependency_materialization,
                            stage_command=list(materialization.command),
                            stage_exit_code=materialization.return_code,
                            stdout=materialization.stdout,
                            stderr=materialization.stderr,
                            error=materialization.error
                            or "Dependency contract materialization failed before runtime boot.",
                        )
                    )
                    fail_fast_triggered = fail_fast
                    continue
                with self.runtime_harness._ephemeral_runtime_workspace(starter_root) as runtime_workspace:
                    current_runtime_workspace = runtime_workspace
                    image_name = self.runtime_harness._workspace_runtime_image_name(runtime_workspace)
                    build_command = []
                    build_stdout_parts.append(
                        f"[{deliverable.id}] Using runtime image {image_name} from the authored runtime plan."
                    )
                    if materialization.synced_paths:
                        build_stdout_parts.append(
                            f"[{deliverable.id}] Materialized dependency contract paths: {', '.join(materialization.synced_paths)}"
                        )
                    self.runtime_harness._ensure_runtime_image_available(image_name)
                    runtime_image_ready = True
                    if self.runtime_harness._image_exists(image_name):
                        any_cached = True
                    dependency_services = self.runtime_harness._dependency_services(runtime_workspace) or []
                    log_coursegen_event(
                        "sandbox_deliverable_support_services_starting",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        dependency_service_count=len(dependency_services),
                        network_name=network_name,
                    )
                    self.runtime_harness._start_runtime_support_services(
                        runtime_workspace,
                        network_name=network_name,
                        container_prefix=container_name,
                    )
                    log_coursegen_event(
                        "sandbox_deliverable_support_services_started",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        dependency_service_count=len(dependency_services),
                    )
                    local_run_command = [
                        self.docker_binary,
                        "run",
                        "-d",
                        "--name",
                        container_name,
                        "-p",
                        f"{host_port}:8000",
                        "-v",
                        f"{runtime_workspace}:/workspace",
                        "-w",
                        "/workspace",
                        *(
                            [
                                "--network",
                                network_name,
                                "--network-alias",
                                "app",
                            ]
                            if dependency_services
                            else []
                        ),
                        *self.runtime_harness._docker_env_args(
                            self.runtime_harness._app_runtime_environment(runtime_workspace)
                        ),
                        image_name,
                        *self.runtime_harness._runtime_shell_command(
                            self.runtime_harness._runtime_launch_script(
                                workspace_path=runtime_workspace,
                                spec=spec,
                                include_setup=True,
                            )
                        ),
                    ]
                    run_command = local_run_command
                    log_coursegen_event(
                        "sandbox_deliverable_runtime_launching",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        image_name=image_name,
                        host_port=host_port,
                        container_name=container_name,
                    )
                    run_result = subprocess.run(
                        local_run_command,
                        cwd=starter_root,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.run_timeout_s,
                    )
                    if run_result.returncode != 0:
                        all_runs_succeeded = False
                        logs = self.runtime_harness._container_logs(container_name) or ""
                        container_stderr = self.runtime_harness._container_stderr(container_name) or ""
                        failed_stage = self._deliverable_runtime_stage(
                            logs=logs,
                            error_text="\n".join(part for part in (run_result.stderr, run_result.stdout) if part),
                            default=SandboxFailureStage.container_launch,
                        )
                        log_coursegen_event(
                            "sandbox_deliverable_runtime_launch_failed",
                            workflow_run_id=workflow_run_id,
                            deliverable_id=deliverable.id,
                            return_code=run_result.returncode,
                            error="Could not start the starter runtime container.",
                        )
                        run_stdout_parts.append(f"[{deliverable.id}] {run_result.stdout}".strip())
                        run_stderr_parts.append(f"[{deliverable.id}] {run_result.stderr}\n{logs}".strip())
                        report_stderr = "\n".join(
                            part for part in [run_result.stderr, container_stderr or logs] if part
                        )
                        (
                            app_stdout_tail,
                            app_exit_state,
                            sidecar_diagnostics,
                        ) = self._collect_failure_diagnostics(
                            app_container_name=container_name,
                            dependency_services=dependency_services,
                            sidecar_container_prefix=container_name,
                        )
                        deliverable_reports.append(
                            DeliverableSandboxReport(
                                deliverable_id=deliverable.id,
                                compile_succeeded=self._compile_succeeded_for_stage(failed_stage),
                                runtime_succeeded=False,
                                failed_stage=failed_stage,
                                stage_command=self._stage_command_for_report(
                                    workspace_path=runtime_workspace,
                                    spec=spec,
                                    failed_stage=failed_stage,
                                    fallback=local_run_command,
                                ),
                                stage_exit_code=run_result.returncode,
                                stdout=run_result.stdout,
                                stderr=report_stderr,
                                error=self._summarize_stage_failure(
                                    deliverable_id=deliverable.id,
                                    failed_stage=failed_stage,
                                    error_text=run_result.stderr,
                                    logs=container_stderr or logs,
                                    default="Could not start the starter runtime container.",
                                ),
                                stdout_tail=app_stdout_tail,
                                exit_state=app_exit_state,
                                sidecar_diagnostics=sidecar_diagnostics,
                            )
                        )
                        fail_fast_triggered = fail_fast
                        continue

                    healthcheck_path = self.runtime_harness._healthcheck_path(runtime_workspace, spec)
                    log_coursegen_event(
                        "sandbox_deliverable_healthcheck_wait_started",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        healthcheck_url=f"{base_url}{healthcheck_path}",
                    )
                    self.runtime_harness._wait_for_http(
                        f"{base_url}{healthcheck_path}",
                        container_name=container_name,
                    )
                    log_coursegen_event(
                        "sandbox_deliverable_healthcheck_wait_completed",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        healthcheck_url=f"{base_url}{healthcheck_path}",
                    )
                log_coursegen_event(
                    "sandbox_deliverable_public_checks_started",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    base_url=base_url,
                )
                contract_passed, contract_output, contract_error, contract_http_response = self._probe_contract_smoke(
                    manifest,
                    base_url,
                    starter_type=spec.runtime_dependencies.starter_type.value,
                )
                checks_passed, check_output, check_error = self._run_visible_suite(
                    starter_root=starter_root,
                    manifest=manifest,
                    base_url=base_url,
                )
                logs = self.runtime_harness._container_logs(container_name) or ""
                combined_output = "\n\n".join(
                    part
                    for part in (contract_output, check_output)
                    if part and part.strip()
                )
                run_stdout_parts.append(f"[{deliverable.id}] {combined_output}".strip())
                if logs:
                    run_stderr_parts.append(f"[{deliverable.id}] {logs}".strip())
                if not contract_passed:
                    all_runs_succeeded = False
                failed_stage: SandboxFailureStage | None = None
                if not contract_passed:
                    failed_stage = SandboxFailureStage.contract
                elif not checks_passed:
                    failed_stage = SandboxFailureStage.checks
                log_coursegen_event(
                    "sandbox_deliverable_public_checks_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    contract_passed=contract_passed,
                    checks_passed=checks_passed,
                    error=check_error,
                )
                contract_failure_diagnostics = None
                if not contract_passed or not checks_passed:
                    contract_failure_diagnostics = self._collect_failure_diagnostics(
                        app_container_name=container_name,
                        dependency_services=dependency_services,
                        sidecar_container_prefix=container_name,
                    )
                deliverable_reports.append(
                        DeliverableSandboxReport(
                            deliverable_id=deliverable.id,
                            compile_succeeded=True,
                            runtime_succeeded=contract_passed,
                            failed_stage=failed_stage,
                            stage_command=(
                                [str(manifest.get("visible_check_command") or "sh .coursegen/runtime/check_visible.sh")]
                                if failed_stage == SandboxFailureStage.checks
                                else []
                            ),
                            public_checks_passed=checks_passed,
                            health_status_code=200,
                            stdout=combined_output,
                            stderr=logs,
                            error=self._post_boot_failure_error(
                                deliverable_id=deliverable.id,
                                failed_stage=failed_stage,
                                contract_error=contract_error,
                                check_error=check_error,
                                logs=logs,
                                contract_http_response=contract_http_response,
                            ),
                            stdout_tail=(
                                contract_failure_diagnostics[0]
                                if contract_failure_diagnostics
                                else None
                            ),
                            exit_state=(
                                contract_failure_diagnostics[1]
                                if contract_failure_diagnostics
                                else None
                            ),
                            sidecar_diagnostics=(
                                contract_failure_diagnostics[2]
                                if contract_failure_diagnostics
                                else None
                            ),
                            http_response=contract_http_response,
                        )
                    )
                fail_fast_triggered = fail_fast and (not contract_passed or not checks_passed)
                log_coursegen_event(
                    "sandbox_deliverable_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    sandbox_status="passed" if contract_passed else "failed",
                    error=check_error,
                )
            except RuntimeImageBuildError as build_exc:
                all_builds_succeeded = False
                all_runs_succeeded = False
                build_stderr_tail = self._tail_lines(build_exc.stderr, max_lines=80)
                build_stdout_tail = self._tail_lines(build_exc.stdout, max_lines=40)
                combined_log = "\n".join(part for part in (build_stderr_tail, build_stdout_tail) if part)
                build_stderr_parts.append(f"[{deliverable.id}] {build_stderr_tail}".strip())
                if build_stdout_tail:
                    build_stdout_parts.append(f"[{deliverable.id}] {build_stdout_tail}".strip())
                log_coursegen_event(
                    "sandbox_deliverable_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    sandbox_status="failed",
                    error=str(build_exc),
                )
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=False,
                        runtime_succeeded=False,
                        failed_stage=SandboxFailureStage.image_build,
                        stage_command=list(build_exc.command),
                        stage_exit_code=build_exc.returncode,
                        stdout=build_stdout_tail,
                        stderr=combined_log,
                        error=self._summarize_stage_failure(
                            deliverable_id=deliverable.id,
                            failed_stage=SandboxFailureStage.image_build,
                            error_text=str(build_exc),
                            logs=combined_log,
                            default="Could not build the starter runtime image.",
                        ),
                        stdout_tail=build_stdout_tail or None,
                    )
                )
                fail_fast_triggered = fail_fast
            except Exception as exc:  # noqa: BLE001
                if not runtime_image_ready:
                    all_builds_succeeded = False
                all_runs_succeeded = False
                logs = self.runtime_harness._container_logs(container_name) or ""
                container_stderr = self.runtime_harness._container_stderr(container_name) or ""
                failed_stage = self._deliverable_runtime_stage(
                    logs=logs,
                    error_text=str(exc),
                    default=(
                        SandboxFailureStage.boot if runtime_image_ready else SandboxFailureStage.runtime
                    ),
                )
                log_coursegen_event(
                    "sandbox_deliverable_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    sandbox_status="failed",
                    error=str(exc),
                )
                run_stderr_parts.append(f"[{deliverable.id}] {exc}\n{logs}".strip())
                (
                    app_stdout_tail,
                    app_exit_state,
                    sidecar_diagnostics,
                ) = self._collect_failure_diagnostics(
                    app_container_name=container_name,
                    dependency_services=dependency_services,
                    sidecar_container_prefix=container_name,
                )
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=self._compile_succeeded_for_stage(failed_stage),
                        runtime_succeeded=False,
                        failed_stage=failed_stage,
                        stage_command=self._stage_command_for_report(
                            workspace_path=(
                                current_runtime_workspace
                                if current_runtime_workspace is not None and current_runtime_workspace.exists()
                                else starter_root
                            ),
                            spec=spec,
                            failed_stage=failed_stage,
                            fallback=run_command,
                        ),
                        stdout="",
                        stderr=container_stderr or logs,
                        error=self._summarize_stage_failure(
                            deliverable_id=deliverable.id,
                            failed_stage=failed_stage,
                            error_text=str(exc),
                            logs=container_stderr or logs,
                            default=str(exc),
                        ),
                        stdout_tail=app_stdout_tail,
                        exit_state=app_exit_state,
                        sidecar_diagnostics=sidecar_diagnostics,
                    )
                )
                fail_fast_triggered = fail_fast
            finally:
                self.runtime_harness._remove_runtime_support(
                    starter_root,
                    network_name=network_name,
                    container_prefix=container_name,
                )
            if fail_fast_triggered:
                log_coursegen_event(
                    "sandbox_fail_fast_stopped_after_deliverable",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    shared_codebase=spec.course_structure.shared_codebase,
                )
                break

        success = all_builds_succeeded and all_runs_succeeded and bool(deliverable_reports)
        result = SandboxExecutionResult(
            status=SandboxExecutionStatus.passed if success else SandboxExecutionStatus.failed,
            available=True,
            build_succeeded=all_builds_succeeded,
            build_cached=any_cached,
            run_succeeded=all_runs_succeeded,
            generated_at=now,
            duration_ms=int((time.perf_counter() - started) * 1000),
            workspace_root=str(workspace_root),
            build_command=build_command,
            run_command=run_command,
            build_stdout="\n\n".join(part for part in build_stdout_parts if part),
            build_stderr="\n\n".join(part for part in build_stderr_parts if part),
            run_stdout="\n\n".join(part for part in run_stdout_parts if part),
            run_stderr="\n\n".join(part for part in run_stderr_parts if part),
            deliverable_reports=deliverable_reports,
            error=None
            if success
            else self._summarize_failed_deliverables(deliverable_reports),
        )
        log_coursegen_event(
            "sandbox_starter_harness_completed",
            workflow_run_id=workflow_run_id,
            workspace_root=str(workspace_root),
            sandbox_status=result.status.value,
            deliverable_report_count=len(result.deliverable_reports),
            duration_ms=result.duration_ms,
            error=result.error,
        )
        return result

    def _execute_shared_starter_harness(
        self,
        *,
        workspace_root: Path,
        shared_starter_root: Path,
        spec: TaskAgentServiceSpec,
        workflow_run_id: str,
        now: datetime,
        started: float,
        fail_fast: bool,
    ) -> SandboxExecutionResult:
        """Shared-codebase variant: build the runtime image and boot the shared
        starter ONCE, then run each deliverable's visible suite + contract
        probe against the single running app."""
        build_stdout_parts: list[str] = []
        build_stderr_parts: list[str] = []
        run_stdout_parts: list[str] = []
        run_stderr_parts: list[str] = []
        deliverable_reports: list[DeliverableSandboxReport] = []
        build_command: list[str] = []
        run_command: list[str] = []
        all_builds_succeeded = True
        all_runs_succeeded = True
        any_cached = False
        private_root = workspace_root.parent / "private"

        # Sanity: shared starter must exist.
        if not shared_starter_root.exists():
            for deliverable in spec.deliverables:
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=False,
                        runtime_succeeded=False,
                        failed_stage=SandboxFailureStage.missing_workspace,
                        error="Shared starter workspace is missing.",
                    )
                )
            return self._finalize_starter_result(
                workspace_root=workspace_root,
                now=now,
                started=started,
                deliverable_reports=deliverable_reports,
                build_command=build_command,
                run_command=run_command,
                build_stdout_parts=build_stdout_parts,
                build_stderr_parts=build_stderr_parts,
                run_stdout_parts=run_stdout_parts,
                run_stderr_parts=run_stderr_parts,
                all_builds_succeeded=False,
                all_runs_succeeded=False,
                any_cached=False,
                workflow_run_id=workflow_run_id,
            )

        # Materialize the dependency contract ONCE against the shared starter.
        try:
            materialization = self.dependency_contract_materializer.materialize(
                starter_root=shared_starter_root,
                runtime_plan=spec.project_contract.runtime_plan,
                deliverable_id="shared",
            )
        except Exception as exc:  # noqa: BLE001
            for deliverable in spec.deliverables:
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=False,
                        runtime_succeeded=False,
                        failed_stage=SandboxFailureStage.dependency_materialization,
                        error=str(exc),
                    )
                )
            return self._finalize_starter_result(
                workspace_root=workspace_root,
                now=now,
                started=started,
                deliverable_reports=deliverable_reports,
                build_command=build_command,
                run_command=run_command,
                build_stdout_parts=build_stdout_parts,
                build_stderr_parts=build_stderr_parts,
                run_stdout_parts=run_stdout_parts,
                run_stderr_parts=run_stderr_parts,
                all_builds_succeeded=False,
                all_runs_succeeded=False,
                any_cached=False,
                workflow_run_id=workflow_run_id,
            )
        if materialization.attempted:
            log_coursegen_event(
                "sandbox_dependency_contract_materialized",
                workflow_run_id=workflow_run_id,
                deliverable_id="shared",
                image_name=materialization.image_name,
                synced_paths=materialization.synced_paths,
                success=materialization.succeeded,
                error=materialization.error,
            )
            if materialization.stdout:
                build_stdout_parts.append(
                    f"[shared] Dependency contract materialization stdout:\n{materialization.stdout}".strip()
                )
            if materialization.stderr:
                build_stderr_parts.append(
                    f"[shared] Dependency contract materialization stderr:\n{materialization.stderr}".strip()
                )
        if not materialization.succeeded:
            # Fail every deliverable with the same materialization error and
            # short-circuit (no per-deliverable boot).
            for deliverable in spec.deliverables:
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=False,
                        runtime_succeeded=False,
                        failed_stage=SandboxFailureStage.dependency_materialization,
                        stage_command=list(materialization.command),
                        stage_exit_code=materialization.return_code,
                        stdout=materialization.stdout,
                        stderr=materialization.stderr,
                        error=materialization.error
                        or "Dependency contract materialization failed before runtime boot.",
                    )
                )
            return self._finalize_starter_result(
                workspace_root=workspace_root,
                now=now,
                started=started,
                deliverable_reports=deliverable_reports,
                build_command=build_command,
                run_command=run_command,
                build_stdout_parts=build_stdout_parts,
                build_stderr_parts=build_stderr_parts,
                run_stdout_parts=run_stdout_parts,
                run_stderr_parts=run_stderr_parts,
                all_builds_succeeded=False,
                all_runs_succeeded=False,
                any_cached=False,
                workflow_run_id=workflow_run_id,
            )

        host_port = self.runtime_harness._allocate_port()
        container_name = f"course-gen-sandbox-shared-{uuid4().hex[:8]}".lower()
        network_name = f"{container_name}-net"
        base_url = f"http://127.0.0.1:{host_port}"
        logs = ""
        runtime_image_ready = False
        current_runtime_workspace: Path | None = None
        boot_failure_reported = False

        try:
            with self.runtime_harness._ephemeral_runtime_workspace(
                shared_starter_root
            ) as runtime_workspace:
                current_runtime_workspace = runtime_workspace
                try:
                    image_name = self.runtime_harness._workspace_runtime_image_name(
                        runtime_workspace
                    )
                except RuntimeImageBuildError as build_exc:
                    all_builds_succeeded = False
                    all_runs_succeeded = False
                    build_stderr_tail = self._tail_lines(build_exc.stderr, max_lines=80)
                    build_stdout_tail = self._tail_lines(build_exc.stdout, max_lines=40)
                    combined_log = "\n".join(
                        part for part in (build_stderr_tail, build_stdout_tail) if part
                    )
                    if build_stderr_tail:
                        build_stderr_parts.append(f"[shared] {build_stderr_tail}".strip())
                    if build_stdout_tail:
                        build_stdout_parts.append(f"[shared] {build_stdout_tail}".strip())
                    for deliverable in spec.deliverables:
                        deliverable_reports.append(
                            DeliverableSandboxReport(
                                deliverable_id=deliverable.id,
                                compile_succeeded=False,
                                runtime_succeeded=False,
                                failed_stage=SandboxFailureStage.image_build,
                                stage_command=list(build_exc.command),
                                stage_exit_code=build_exc.returncode,
                                stdout=build_stdout_tail,
                                stderr=combined_log,
                                error=self._summarize_stage_failure(
                                    deliverable_id=deliverable.id,
                                    failed_stage=SandboxFailureStage.image_build,
                                    error_text=str(build_exc),
                                    logs=combined_log,
                                    default="Could not build the starter runtime image.",
                                ),
                                stdout_tail=build_stdout_tail or None,
                            )
                        )
                    return self._finalize_starter_result(
                        workspace_root=workspace_root,
                        now=now,
                        started=started,
                        deliverable_reports=deliverable_reports,
                        build_command=build_command,
                        run_command=run_command,
                        build_stdout_parts=build_stdout_parts,
                        build_stderr_parts=build_stderr_parts,
                        run_stdout_parts=run_stdout_parts,
                        run_stderr_parts=run_stderr_parts,
                        all_builds_succeeded=all_builds_succeeded,
                        all_runs_succeeded=all_runs_succeeded,
                        any_cached=any_cached,
                        workflow_run_id=workflow_run_id,
                    )

                build_stdout_parts.append(
                    f"[shared] Using runtime image {image_name} from the authored runtime plan."
                )
                if materialization.synced_paths:
                    build_stdout_parts.append(
                        f"[shared] Materialized dependency contract paths: {', '.join(materialization.synced_paths)}"
                    )
                self.runtime_harness._ensure_runtime_image_available(image_name)
                runtime_image_ready = True
                if self.runtime_harness._image_exists(image_name):
                    any_cached = True

                dependency_services = self.runtime_harness._dependency_services(
                    runtime_workspace
                )
                log_coursegen_event(
                    "sandbox_shared_support_services_starting",
                    workflow_run_id=workflow_run_id,
                    dependency_service_count=len(dependency_services),
                    network_name=network_name,
                )
                self.runtime_harness._start_runtime_support_services(
                    runtime_workspace,
                    network_name=network_name,
                    container_prefix=container_name,
                )
                log_coursegen_event(
                    "sandbox_shared_support_services_started",
                    workflow_run_id=workflow_run_id,
                    dependency_service_count=len(dependency_services),
                )
                local_run_command = [
                    self.docker_binary,
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "-p",
                    f"{host_port}:8000",
                    "-v",
                    f"{runtime_workspace}:/workspace",
                    "-w",
                    "/workspace",
                    *(
                        [
                            "--network",
                            network_name,
                            "--network-alias",
                            "app",
                        ]
                        if dependency_services
                        else []
                    ),
                    *self.runtime_harness._docker_env_args(
                        self.runtime_harness._app_runtime_environment(runtime_workspace)
                    ),
                    image_name,
                    *self.runtime_harness._runtime_shell_command(
                        self.runtime_harness._runtime_launch_script(
                            workspace_path=runtime_workspace,
                            spec=spec,
                            include_setup=True,
                        )
                    ),
                ]
                run_command = local_run_command
                log_coursegen_event(
                    "sandbox_shared_runtime_launching",
                    workflow_run_id=workflow_run_id,
                    image_name=image_name,
                    host_port=host_port,
                    container_name=container_name,
                )
                run_result = subprocess.run(
                    local_run_command,
                    cwd=shared_starter_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.run_timeout_s,
                )
                if run_result.returncode != 0:
                    all_runs_succeeded = False
                    logs = self.runtime_harness._container_logs(container_name) or ""
                    container_stderr = self.runtime_harness._container_stderr(container_name) or ""
                    failed_stage = self._deliverable_runtime_stage(
                        logs=logs,
                        error_text="\n".join(
                            part for part in (run_result.stderr, run_result.stdout) if part
                        ),
                        default=SandboxFailureStage.container_launch,
                    )
                    run_stdout_parts.append(f"[shared] {run_result.stdout}".strip())
                    run_stderr_parts.append(
                        f"[shared] {run_result.stderr}\n{logs}".strip()
                    )
                    report_stderr = "\n".join(
                        part for part in [run_result.stderr, container_stderr or logs] if part
                    )
                    (
                        app_stdout_tail,
                        app_exit_state,
                        sidecar_diagnostics,
                    ) = self._collect_failure_diagnostics(
                        app_container_name=container_name,
                        dependency_services=dependency_services,
                        sidecar_container_prefix=container_name,
                    )
                    for deliverable in spec.deliverables:
                        deliverable_reports.append(
                            DeliverableSandboxReport(
                                deliverable_id=deliverable.id,
                                compile_succeeded=self._compile_succeeded_for_stage(
                                    failed_stage
                                ),
                                runtime_succeeded=False,
                                failed_stage=failed_stage,
                                stage_command=self._stage_command_for_report(
                                    workspace_path=runtime_workspace,
                                    spec=spec,
                                    failed_stage=failed_stage,
                                    fallback=local_run_command,
                                ),
                                stage_exit_code=run_result.returncode,
                                stdout=run_result.stdout,
                                stderr=report_stderr,
                                error=self._summarize_stage_failure(
                                    deliverable_id=deliverable.id,
                                    failed_stage=failed_stage,
                                    error_text=run_result.stderr,
                                    logs=container_stderr or logs,
                                    default="Could not start the starter runtime container.",
                                ),
                                stdout_tail=app_stdout_tail,
                                exit_state=app_exit_state,
                                sidecar_diagnostics=sidecar_diagnostics,
                            )
                        )
                    return self._finalize_starter_result(
                        workspace_root=workspace_root,
                        now=now,
                        started=started,
                        deliverable_reports=deliverable_reports,
                        build_command=build_command,
                        run_command=run_command,
                        build_stdout_parts=build_stdout_parts,
                        build_stderr_parts=build_stderr_parts,
                        run_stdout_parts=run_stdout_parts,
                        run_stderr_parts=run_stderr_parts,
                        all_builds_succeeded=all_builds_succeeded,
                        all_runs_succeeded=all_runs_succeeded,
                        any_cached=any_cached,
                        workflow_run_id=workflow_run_id,
                    )

                healthcheck_path = self.runtime_harness._healthcheck_path(
                    runtime_workspace, spec
                )
                log_coursegen_event(
                    "sandbox_shared_healthcheck_wait_started",
                    workflow_run_id=workflow_run_id,
                    healthcheck_url=f"{base_url}{healthcheck_path}",
                )
                try:
                    self.runtime_harness._wait_for_http(
                        f"{base_url}{healthcheck_path}",
                        container_name=container_name,
                    )
                except Exception as exc:  # noqa: BLE001
                    all_runs_succeeded = False
                    boot_failure_reported = True
                    logs = self.runtime_harness._container_logs(container_name) or ""
                    container_stderr = self.runtime_harness._container_stderr(container_name) or ""
                    failed_stage = self._deliverable_runtime_stage(
                        logs=logs,
                        error_text=str(exc),
                        default=SandboxFailureStage.boot,
                    )
                    run_stderr_parts.append(f"[shared] {exc}\n{logs}".strip())
                    # On boot failure, fail-fast: emit a single report for the
                    # first deliverable so the operator sees the failure
                    # without N copies of the same boot error.
                    primary = spec.deliverables[0]
                    (
                        app_stdout_tail,
                        app_exit_state,
                        sidecar_diagnostics,
                    ) = self._collect_failure_diagnostics(
                        app_container_name=container_name,
                        dependency_services=dependency_services,
                        sidecar_container_prefix=container_name,
                    )
                    deliverable_reports.append(
                        DeliverableSandboxReport(
                            deliverable_id=primary.id,
                            compile_succeeded=self._compile_succeeded_for_stage(failed_stage),
                            runtime_succeeded=False,
                            failed_stage=failed_stage,
                            stage_command=self._stage_command_for_report(
                                workspace_path=runtime_workspace,
                                spec=spec,
                                failed_stage=failed_stage,
                                fallback=run_command,
                            ),
                            stdout="",
                            stderr=container_stderr or logs,
                            error=self._summarize_stage_failure(
                                deliverable_id=primary.id,
                                failed_stage=failed_stage,
                                error_text=str(exc),
                                logs=container_stderr or logs,
                                default=str(exc),
                            ),
                            stdout_tail=app_stdout_tail,
                            exit_state=app_exit_state,
                            sidecar_diagnostics=sidecar_diagnostics,
                        )
                    )
                    return self._finalize_starter_result(
                        workspace_root=workspace_root,
                        now=now,
                        started=started,
                        deliverable_reports=deliverable_reports,
                        build_command=build_command,
                        run_command=run_command,
                        build_stdout_parts=build_stdout_parts,
                        build_stderr_parts=build_stderr_parts,
                        run_stdout_parts=run_stdout_parts,
                        run_stderr_parts=run_stderr_parts,
                        all_builds_succeeded=all_builds_succeeded,
                        all_runs_succeeded=all_runs_succeeded,
                        any_cached=any_cached,
                        workflow_run_id=workflow_run_id,
                    )
                log_coursegen_event(
                    "sandbox_shared_healthcheck_wait_completed",
                    workflow_run_id=workflow_run_id,
                    healthcheck_url=f"{base_url}{healthcheck_path}",
                )

                # The runtime is up. Walk each deliverable against the single
                # running app.
                for deliverable in spec.deliverables:
                    report, combined_output, runtime_succeeded, stop = (
                        self._run_one_deliverable_against_shared_runtime(
                            deliverable=deliverable,
                            workspace_root=workspace_root,
                            private_root=private_root,
                            shared_starter_root=shared_starter_root,
                            base_url=base_url,
                            workflow_run_id=workflow_run_id,
                            fail_fast=fail_fast,
                            app_container_name=container_name,
                            dependency_services=dependency_services,
                            starter_type=spec.runtime_dependencies.starter_type.value,
                        )
                    )
                    deliverable_reports.append(report)
                    if combined_output:
                        run_stdout_parts.append(
                            f"[{deliverable.id}] {combined_output}".strip()
                        )
                    if not runtime_succeeded:
                        all_runs_succeeded = False
                    if stop:
                        break

                # Collect container logs for forensic context.
                logs = self.runtime_harness._container_logs(container_name) or ""
                if logs:
                    run_stderr_parts.append(f"[shared] {logs}".strip())
        except Exception as exc:  # noqa: BLE001
            if not runtime_image_ready:
                all_builds_succeeded = False
            all_runs_succeeded = False
            logs = self.runtime_harness._container_logs(container_name) or ""
            container_stderr = self.runtime_harness._container_stderr(container_name) or ""
            failed_stage = self._deliverable_runtime_stage(
                logs=logs,
                error_text=str(exc),
                default=(
                    SandboxFailureStage.boot if runtime_image_ready else SandboxFailureStage.runtime
                ),
            )
            run_stderr_parts.append(f"[shared] {exc}\n{logs}".strip())
            if not deliverable_reports and not boot_failure_reported:
                primary = spec.deliverables[0]
                (
                    app_stdout_tail,
                    app_exit_state,
                    sidecar_diagnostics,
                ) = self._collect_failure_diagnostics(
                    app_container_name=container_name,
                    dependency_services=(
                        dependency_services if "dependency_services" in locals() else None
                    ),
                    sidecar_container_prefix=container_name,
                )
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=primary.id,
                        compile_succeeded=self._compile_succeeded_for_stage(failed_stage),
                        runtime_succeeded=False,
                        failed_stage=failed_stage,
                        stage_command=self._stage_command_for_report(
                            workspace_path=(
                                current_runtime_workspace
                                if current_runtime_workspace is not None
                                and current_runtime_workspace.exists()
                                else shared_starter_root
                            ),
                            spec=spec,
                            failed_stage=failed_stage,
                            fallback=run_command,
                        ),
                        stdout="",
                        stderr=container_stderr or logs,
                        error=self._summarize_stage_failure(
                            deliverable_id=primary.id,
                            failed_stage=failed_stage,
                            error_text=str(exc),
                            logs=container_stderr or logs,
                            default=str(exc),
                        ),
                        stdout_tail=app_stdout_tail,
                        exit_state=app_exit_state,
                        sidecar_diagnostics=sidecar_diagnostics,
                    )
                )
        finally:
            self.runtime_harness._remove_runtime_support(
                shared_starter_root,
                network_name=network_name,
                container_prefix=container_name,
            )

        return self._finalize_starter_result(
            workspace_root=workspace_root,
            now=now,
            started=started,
            deliverable_reports=deliverable_reports,
            build_command=build_command,
            run_command=run_command,
            build_stdout_parts=build_stdout_parts,
            build_stderr_parts=build_stderr_parts,
            run_stdout_parts=run_stdout_parts,
            run_stderr_parts=run_stderr_parts,
            all_builds_succeeded=all_builds_succeeded,
            all_runs_succeeded=all_runs_succeeded,
            any_cached=any_cached,
            workflow_run_id=workflow_run_id,
        )

    def _load_per_deliverable_manifest(
        self,
        *,
        private_root: Path,
        deliverable_id: str,
    ) -> dict:
        """Load <private>/grader/<id>/deliverable.json. Returns {} if absent."""
        manifest_path = (
            deliverable_grader_dir(private_root, deliverable_id)
            / DELIVERABLE_MANIFEST_RELATIVE_PATH
        )
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _finalize_starter_result(
        self,
        *,
        workspace_root: Path,
        now: datetime,
        started: float,
        deliverable_reports: list[DeliverableSandboxReport],
        build_command: list[str],
        run_command: list[str],
        build_stdout_parts: list[str],
        build_stderr_parts: list[str],
        run_stdout_parts: list[str],
        run_stderr_parts: list[str],
        all_builds_succeeded: bool,
        all_runs_succeeded: bool,
        any_cached: bool,
        workflow_run_id: str,
    ) -> SandboxExecutionResult:
        success = all_builds_succeeded and all_runs_succeeded and bool(deliverable_reports)
        result = SandboxExecutionResult(
            status=SandboxExecutionStatus.passed if success else SandboxExecutionStatus.failed,
            available=True,
            build_succeeded=all_builds_succeeded,
            build_cached=any_cached,
            run_succeeded=all_runs_succeeded,
            generated_at=now,
            duration_ms=int((time.perf_counter() - started) * 1000),
            workspace_root=str(workspace_root),
            build_command=build_command,
            run_command=run_command,
            build_stdout="\n\n".join(part for part in build_stdout_parts if part),
            build_stderr="\n\n".join(part for part in build_stderr_parts if part),
            run_stdout="\n\n".join(part for part in run_stdout_parts if part),
            run_stderr="\n\n".join(part for part in run_stderr_parts if part),
            deliverable_reports=deliverable_reports,
            error=None
            if success
            else self._summarize_failed_deliverables(deliverable_reports),
        )
        log_coursegen_event(
            "sandbox_starter_harness_completed",
            workflow_run_id=workflow_run_id,
            workspace_root=str(workspace_root),
            sandbox_status=result.status.value,
            deliverable_report_count=len(result.deliverable_reports),
            duration_ms=result.duration_ms,
            error=result.error,
        )
        return result

    def _run_one_deliverable_against_shared_runtime(
        self,
        *,
        deliverable,
        workspace_root: Path,
        private_root: Path,
        shared_starter_root: Path,
        base_url: str,
        workflow_run_id: str,
        fail_fast: bool,
        app_container_name: str | None = None,
        dependency_services: list[dict] | None = None,
        starter_type: str | None = None,
    ) -> tuple[DeliverableSandboxReport, str, bool, bool]:
        """Run one deliverable's contract probe + visible suite against the
        shared running app. Returns (report, combined_stdout, runtime_succeeded,
        stop_iteration). `stop_iteration` indicates the caller should break the
        loop (fail-fast semantics).
        """
        log_coursegen_event(
            "sandbox_deliverable_started",
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable.id,
            deliverable_title=deliverable.title,
            starter_root=str(shared_starter_root),
        )
        manifest = self._load_per_deliverable_manifest(
            private_root=private_root,
            deliverable_id=deliverable.id,
        )
        visible_script = (
            deliverable_visible_checks_dir(workspace_root, deliverable.id)
            / VISIBLE_CHECK_SCRIPT_RELATIVE_PATH
        )
        log_coursegen_event(
            "sandbox_deliverable_public_checks_started",
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable.id,
            base_url=base_url,
        )
        contract_passed, contract_output, contract_error, contract_http_response = self._probe_contract_smoke(
            manifest, base_url, starter_type=starter_type,
        )

        visible_command = f"python3 ../checks/{deliverable.id}/run_visible_checks.py"
        visible_script_missing = not visible_script.exists()
        if visible_script_missing:
            checks_passed = False
            check_output = ""
            check_error = (
                f"Visible check script not found: "
                f"public/checks/{deliverable.id}/run_visible_checks.py"
            )
        else:
            suite_report = self.test_script_runner.run_suite(
                workspace_root=shared_starter_root,
                command=visible_command,
                base_url=base_url,
                suite_type="visible",
            )
            lines = [f"Visible suite: {suite_report.summary}"]
            for case in suite_report.tests:
                marker = "PASS" if case.status == "passed" else "FAIL"
                lines.append(f"[{marker}] {case.title}: {case.summary}")
                for diagnostic in case.diagnostics[:3]:
                    lines.append(f"  - {diagnostic}")
            check_output = "\n".join(lines)
            # `checks_passed` reflects whether the visible script ran cleanly
            # (emitted a parseable JSON report). It does NOT reflect whether
            # individual tests passed.
            #
            # The visible suite is run for TWO purposes elsewhere in the
            # platform with opposite expectations:
            #   - The baseline matrix verifier (generated_test_harness.verify_course)
            #     EXPECTS visible tests to FAIL against partial/empty starters —
            #     handlers raise NotImplementedError by design and the test
            #     strength should detect that.
            #   - The sandbox runner just needs to confirm the harness
            #     machinery (script, JSON emission, HTTP path) is intact.
            #
            # Gating on `suite_report.passed` makes a perfectly-authored partial
            # starter fail authoring_runtime because every test fails — exactly
            # the state the baseline matrix demands. Decouple: at sandbox time
            # we only care that the script ran. Pass/fail counts still flow
            # into the report stdout for the baseline matrix to consume.
            if not suite_report.valid:
                checks_passed = False
                check_error = "Visible test script did not emit a valid report."
            else:
                checks_passed = True
                check_error = None

        combined_output = "\n\n".join(
            part for part in (contract_output, check_output) if part and part.strip()
        )
        failed_stage: SandboxFailureStage | None = None
        if not contract_passed:
            failed_stage = SandboxFailureStage.contract
        elif not checks_passed:
            failed_stage = SandboxFailureStage.checks
        log_coursegen_event(
            "sandbox_deliverable_public_checks_completed",
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable.id,
            contract_passed=contract_passed,
            checks_passed=checks_passed,
            error=check_error,
        )
        # Pass 11 Job B: on contract/checks failure the app container is
        # still running (the live container responded — just with a 500 or
        # a failing test). Capture its diagnostics the same way the legacy
        # per-deliverable path does, so the model sees the FastAPI/Spring/
        # uvicorn traceback in stdout_tail and the sidecar errors in
        # sidecar_diagnostics. Without this the model only sees the HTTP
        # response body and has to guess at the cause.
        app_stdout_tail = None
        app_exit_state = None
        sidecar_diagnostics: dict[str, dict] | None = None
        if not contract_passed or not checks_passed:
            (
                app_stdout_tail,
                app_exit_state,
                sidecar_diagnostics,
            ) = self._collect_failure_diagnostics(
                app_container_name=app_container_name,
                dependency_services=dependency_services,
                sidecar_container_prefix=app_container_name,
            )
        report = DeliverableSandboxReport(
            deliverable_id=deliverable.id,
            compile_succeeded=True,
            runtime_succeeded=contract_passed,
            failed_stage=failed_stage,
            stage_command=(
                [visible_command]
                if failed_stage == SandboxFailureStage.checks
                else []
            ),
            public_checks_passed=checks_passed,
            health_status_code=200,
            stdout=combined_output,
            stderr="",
            error=self._post_boot_failure_error(
                deliverable_id=deliverable.id,
                failed_stage=failed_stage,
                contract_error=contract_error,
                check_error=check_error,
                logs=None,
                contract_http_response=contract_http_response,
            ),
            http_response=contract_http_response,
            stdout_tail=app_stdout_tail,
            exit_state=app_exit_state,
            sidecar_diagnostics=sidecar_diagnostics,
        )
        log_coursegen_event(
            "sandbox_deliverable_completed",
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable.id,
            sandbox_status="passed" if contract_passed and checks_passed else "failed",
            error=check_error,
        )
        # Fail-fast triggers on a contract probe failure or a real
        # visible-suite failure, but NOT on a missing per-deliverable visible
        # script (an authoring gap, not a runtime regression of the shared app).
        stop = False
        if fail_fast and not contract_passed:
            stop = True
        elif fail_fast and not checks_passed and not visible_script_missing:
            stop = True
        if stop:
            log_coursegen_event(
                "sandbox_fail_fast_stopped_after_deliverable",
                workflow_run_id=workflow_run_id,
                deliverable_id=deliverable.id,
                shared_codebase=True,
            )
        runtime_succeeded = contract_passed and checks_passed
        return report, combined_output, runtime_succeeded, stop

    def _starter_runtime_image_tag(self, workspace_root: Path) -> str:
        cache_key = self._workspace_cache_key(workspace_root)
        return f"{self.cache_namespace}:{cache_key[:24]}"

    def _probe_contract_smoke(
        self,
        manifest: dict,
        base_url: str,
        *,
        starter_type: str | None = None,
    ) -> tuple[bool, str, str | None, dict | None]:
        """Run each manifest-declared public check against the booted app.

        Returns ``(passed, summary_text, error_message, http_response)``.

        Pass 11 Job A: when ``starter_type == "partial"`` the published
        contract handlers are intentionally stubbed (Pass 4 directive:
        ``raise NotImplementedError`` / language-equivalent). A reachable
        but unimplemented handler returns a non-2xx response — that's the
        author's intent for the pre-implementation state. We treat any
        non-404 HTTP response as ``endpoint reachable``: contract smoke
        passes for partial starters when the route exists, regardless of
        what the stub returns. Only 404 (route missing) still fails,
        because a missing route is a structural authoring bug, not the
        documented stub behavior. For non-partial starters (e.g. a future
        ``working`` mode), strict 2xx-or-fail semantics remain in effect.

        The fourth return element (Pass 8) captures the FIRST failed HTTP
        exchange verbatim so the LLM sees the response body — not just the
        ``HTTP 500`` status. Body / headers / status / request_method /
        request_path / request_body are all preserved so repair has the
        full canonical diagnostic for contract failures.
        """
        partial_starter = (starter_type or "").strip().lower() == "partial"
        public_checks = manifest.get("public_checks") or []
        if not public_checks:
            return (
                False,
                "No public checks were configured for this deliverable.",
                "No public checks were configured.",
                None,
            )
        contract_passed = True
        lines: list[str] = []
        first_failure: dict | None = None
        for check in public_checks:
            if not isinstance(check, dict):
                continue
            title = str(check.get("title") or check.get("request_path") or "visible check")
            method = str(check.get("request_method") or "POST").upper()
            request_path = str(check.get("request_path") or "").strip()
            request_body = check.get("request_body") or None
            if not request_path.startswith("/"):
                contract_passed = False
                lines.append(f"[FAIL] {title}: invalid request path")
                continue
            try:
                self._json_request(method, f"{base_url}{request_path}", request_body)
            except urllib.error.HTTPError as exc:
                # Pass 11 Job A: for a partial starter, any non-404 HTTP response
                # means "route exists; handler is a documented stub" — that
                # passes contract smoke (the handler body will be exercised
                # after the learner implements it).
                if partial_starter and exc.code != 404:
                    lines.append(
                        f"[PASS-PARTIAL] {title}: HTTP {exc.code} "
                        f"(endpoint reachable; stub will be exercised after learner work)"
                    )
                    continue
                contract_passed = False
                response_body = self._read_http_error_body(exc)
                response_headers = self._http_error_headers(exc)
                lines.append(f"[FAIL] {title}: HTTP {exc.code}")
                if response_body:
                    lines.append(f"  response: {response_body[:500]}")
                if first_failure is None:
                    first_failure = {
                        "request_method": method,
                        "request_path": request_path,
                        "request_body": request_body,
                        "response_status": exc.code,
                        "response_headers": response_headers,
                        "response_body_text": response_body,
                    }
                continue
            except Exception as exc:  # noqa: BLE001
                contract_passed = False
                lines.append(f"[FAIL] {title}: {exc}")
                if first_failure is None:
                    first_failure = {
                        "request_method": method,
                        "request_path": request_path,
                        "request_body": request_body,
                        "response_status": None,
                        "response_headers": None,
                        "response_body_text": str(exc),
                    }
                continue

            expected_status = int(check.get("expected_status") or 200)
            if expected_status >= 400:
                lines.append(f"[WARN] {title}")
                lines.append(f"  Non-success expected status {expected_status} is not supported by the sandbox checker yet.")
                continue
            lines.append(f"[PASS] {title}")

        if contract_passed:
            lines.append("")
            lines.append("Starter deliverable kept the public contract stable in Docker.")
            return True, "\n".join(lines), None, None
        return (
            False,
            "\n".join(lines),
            "One or more starter smoke checks could not exercise the published contract.",
            first_failure,
        )

    def _read_http_error_body(self, exc: urllib.error.HTTPError) -> str:
        """Best-effort verbatim read of the HTTP error body."""
        body: bytes | None = None
        try:
            body = exc.read() if hasattr(exc, "read") else None
        except Exception:  # noqa: BLE001
            body = None
        if body is None:
            fp = getattr(exc, "fp", None)
            if fp is not None:
                try:
                    body = fp.read()
                except Exception:  # noqa: BLE001
                    body = None
        if not body:
            return ""
        if isinstance(body, bytes):
            try:
                return body.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                return body.decode("latin-1", errors="replace")
        return str(body)

    def _http_error_headers(self, exc: urllib.error.HTTPError) -> dict | None:
        headers = getattr(exc, "headers", None)
        if headers is None:
            return None
        try:
            return {key: value for key, value in headers.items()}
        except Exception:  # noqa: BLE001
            return None

    def _run_visible_suite(
        self,
        *,
        starter_root: Path,
        manifest: dict,
        base_url: str,
    ) -> tuple[bool, str, str | None]:
        """Run the visible suite as a smoke check.

        The returned bool reflects whether the script EXECUTED CLEANLY
        (emitted a parseable JSON report). It does NOT gate on individual
        test pass/fail — that's the baseline matrix verifier's job, which
        is starter-type-aware and correctly expects visible tests to fail
        against partial/empty starters.

        See ``_run_one_deliverable_against_shared_runtime`` for the long
        rationale; both call sites share the same contract.
        """
        command = str(manifest.get("visible_check_command") or "sh .coursegen/runtime/check_visible.sh")
        report = self.test_script_runner.run_suite(
            workspace_root=starter_root,
            command=command,
            base_url=base_url,
            suite_type="visible",
        )
        lines = [
            f"Visible suite: {report.summary}",
        ]
        for case in report.tests:
            marker = "PASS" if case.status == "passed" else "FAIL"
            lines.append(f"[{marker}] {case.title}: {case.summary}")
            for diagnostic in case.diagnostics[:3]:
                lines.append(f"  - {diagnostic}")
        if not report.valid:
            return False, "\n".join(lines), "Visible test script did not emit a valid report."
        return True, "\n".join(lines), None

    def _deliverable_runtime_stage(
        self,
        *,
        logs: str | None,
        error_text: str | None,
        default: SandboxFailureStage,
    ) -> SandboxFailureStage:
        stage_name = self.runtime_harness._runtime_stage_from_logs(logs or error_text or "")
        if stage_name == "install":
            return SandboxFailureStage.install
        if stage_name == "verify":
            return SandboxFailureStage.verify
        if stage_name == "boot":
            return SandboxFailureStage.boot
        return default

    def _compile_succeeded_for_stage(self, failed_stage: SandboxFailureStage | None) -> bool:
        return failed_stage not in {
            SandboxFailureStage.dependency_materialization,
            SandboxFailureStage.image_build,
            SandboxFailureStage.install,
            SandboxFailureStage.verify,
            SandboxFailureStage.container_launch,
            SandboxFailureStage.missing_workspace,
            SandboxFailureStage.runtime,
        }

    def _stage_command_for_report(
        self,
        *,
        workspace_path: Path,
        spec: TaskAgentServiceSpec,
        failed_stage: SandboxFailureStage | None,
        fallback: list[str],
    ) -> list[str]:
        if failed_stage in {SandboxFailureStage.install, SandboxFailureStage.verify, SandboxFailureStage.boot}:
            command = self.runtime_harness._runtime_stage_command(
                workspace_path,
                spec,
                failed_stage.value,
            )
            if command:
                return command
        return list(fallback)

    def _collect_failure_diagnostics(
        self,
        *,
        app_container_name: str | None,
        dependency_services: list[dict] | None,
        sidecar_container_prefix: str | None,
    ) -> tuple[str | None, dict | None, dict[str, dict] | None]:
        """Capture the structured diagnostic surface for a failed deliverable.

        Returns ``(app_stdout_tail, app_exit_state, sidecar_diagnostics)``:

        - ``app_stdout_tail``: last 100 lines of the app container's stdout
          (framework boot logs: Spring Boot, gunicorn, structured loggers).
          The harness already captures stderr separately into
          ``report.stderr``; stdout was the previous blind spot.
        - ``app_exit_state``: ``{exit_code, oom_killed, status, error}`` from
          ``docker inspect``. Surfaces OOM kills and other non-stderr exit
          reasons.
        - ``sidecar_diagnostics``: ``{service_id: {stderr_tail, stdout_tail,
          exit_state}}`` for every sidecar started for this deliverable. The
          canonical example: app reports "connection refused", but postgres
          was OOMKilled — the real cause lives in the sidecar's diagnostics.
        """
        def _coerce_text(value: object) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                stripped = value.strip()
                return stripped or None
            return None

        def _coerce_state(value: object) -> dict | None:
            return value if isinstance(value, dict) else None

        app_stdout_tail: str | None = None
        app_exit_state: dict | None = None
        if app_container_name:
            try:
                app_stdout_tail = _coerce_text(
                    self.runtime_harness._container_stdout(app_container_name)
                )
            except Exception:  # noqa: BLE001
                app_stdout_tail = None
            try:
                app_exit_state = _coerce_state(
                    self.runtime_harness._container_exit_state(app_container_name)
                )
            except Exception:  # noqa: BLE001
                app_exit_state = None

        sidecar_diagnostics: dict[str, dict] | None = None
        if dependency_services and sidecar_container_prefix:
            sidecar_diagnostics = {}
            for service in dependency_services:
                if not isinstance(service, dict):
                    continue
                service_id = str(service.get("service_id") or "").strip()
                if not service_id:
                    continue
                try:
                    sidecar_name = self.runtime_harness._service_container_name(
                        sidecar_container_prefix, service_id
                    )
                except Exception:  # noqa: BLE001
                    sidecar_name = None
                if not isinstance(sidecar_name, str) or not sidecar_name:
                    continue
                try:
                    stderr_tail = _coerce_text(
                        self.runtime_harness._container_stderr(sidecar_name)
                    )
                except Exception:  # noqa: BLE001
                    stderr_tail = None
                try:
                    stdout_tail = _coerce_text(
                        self.runtime_harness._container_stdout(sidecar_name)
                    )
                except Exception:  # noqa: BLE001
                    stdout_tail = None
                try:
                    exit_state = _coerce_state(
                        self.runtime_harness._container_exit_state(sidecar_name)
                    )
                except Exception:  # noqa: BLE001
                    exit_state = None
                sidecar_diagnostics[service_id] = {
                    "stderr_tail": stderr_tail,
                    "stdout_tail": stdout_tail,
                    "exit_state": exit_state,
                }
            if not sidecar_diagnostics:
                sidecar_diagnostics = None

        return app_stdout_tail, app_exit_state, sidecar_diagnostics

    def _image_build_diagnostic_line(self, build_stderr: str | None) -> str | None:
        """Return the real error line from a docker buildkit failure.

        Buildkit emits structured failure blocks like::

            #10 0.5 go: ... requires go >= 1.23 (running go 1.22.4)
            #10 ERROR: process did not complete successfully: exit code: 1
            ------
             > [6/8] RUN sh .coursegen/runtime/install.sh:
            0.5 go: ... requires go >= 1.23 (running go 1.22.4)
            ------
            Dockerfile:14
            --------------------
              14 | >>> RUN sh .coursegen/runtime/install.sh
            --------------------
            ERROR: failed to build: failed to solve: process did not complete successfully: exit code: 1

        The LAST line is a generic footer; the canonical signal is the
        ``error:`` / ``ERROR:`` line BEFORE the first ``------`` (or
        ``--------------------``) separator. We walk the stream looking
        for the last ``error``-bearing line that appears before any
        separator. Falls back to the last non-blank line (Pass 7
        behaviour) when no separator is present, so non-buildkit build
        failures still get a useful headline.
        """
        if not build_stderr:
            return None
        text = build_stderr.strip()
        if not text:
            return None
        lines = text.splitlines()
        first_sep_index: int | None = None
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            if stripped and set(stripped) <= {"-"} and len(stripped) >= 3:
                first_sep_index = idx
                break
        if first_sep_index is None:
            # No buildkit-style separator: fall back to Pass 7 tail behaviour
            # — pick the last non-blank line that looks error-bearing.
            non_blank = [line for line in lines if line.strip()]
            if not non_blank:
                return None
            for candidate in reversed(non_blank):
                lowered = candidate.lower()
                if "error" in lowered or "fail" in lowered:
                    return candidate.strip()
            return non_blank[-1].strip()
        # Buildkit case: walk the lines BEFORE the first separator and pick
        # the last one that looks like a real error (contains 'error',
        # 'requires', or 'cannot'). Skip the generic 'process did not
        # complete successfully' wrapper — it's always present and useless.
        before = [line for line in lines[:first_sep_index] if line.strip()]
        for candidate in reversed(before):
            lowered = candidate.lower()
            if "did not complete successfully" in lowered:
                continue
            if (
                "error" in lowered
                or "requires" in lowered
                or "cannot" in lowered
                or "fail" in lowered
                or "not found" in lowered
                or "no such" in lowered
            ):
                return candidate.strip()
        # Nothing useful before the separator: return the last non-blank
        # line before the separator anyway (better than the footer).
        if before:
            return before[-1].strip()
        return None

    def _post_boot_failure_error(
        self,
        *,
        deliverable_id: str,
        failed_stage: SandboxFailureStage | None,
        contract_error: str | None,
        check_error: str | None,
        logs: str | None,
        contract_http_response: dict | None,
    ) -> str | None:
        """Render the headline ``error`` field for a post-boot deliverable
        report (contract or visible-checks stage).

        On success ``failed_stage`` is ``None`` and the field is the raw
        legacy fallback. On failure we route through
        :meth:`_summarize_stage_failure` so contract failures pick up the
        rich HTTP-exchange headline (Pass 9 Job A).
        """
        if failed_stage is None:
            return contract_error or check_error
        return self._summarize_stage_failure(
            deliverable_id=deliverable_id,
            failed_stage=failed_stage,
            error_text=contract_error or check_error,
            logs=logs,
            default=contract_error or check_error or "",
            http_response=contract_http_response,
        )

    def _summarize_stage_failure(
        self,
        *,
        deliverable_id: str,
        failed_stage: SandboxFailureStage | None,
        error_text: str | None,
        logs: str | None,
        default: str,
        http_response: dict | None = None,
    ) -> str:
        """Stage-agnostic failure summary: deliverable id + stage + tail of stderr.

        We trust the LLM to read the ``stderr_excerpt`` the harness ships in
        the failure context. The headline error is just a pointer: which
        deliverable failed at which stage, plus a ~3-line tail teaser. The
        full stderr is the canonical diagnostic.

        Two stages need special handling:

        * ``image_build`` — docker buildkit emits a generic
          ``failed to solve`` footer AFTER the real ``RUN`` step error.
          :meth:`_image_build_diagnostic_line` walks past the footer to
          the real cause.
        * ``contract`` (Pass 9) — the canonical diagnostic is the HTTP
          exchange captured by :meth:`_probe_contract_smoke`, NOT
          stderr. When ``http_response`` is supplied we format a
          headline like ``POST /links → 400 {"error":"…"}`` so the LLM
          doesn't have to dig into nested fields.

        Every other stage's canonical signal is at the tail of stderr,
        so we surface a 3-line teaser there.
        """
        stage_label = (
            failed_stage.value.replace("_", " ")
            if failed_stage is not None
            else "runtime"
        )
        if (
            failed_stage == SandboxFailureStage.contract
            and isinstance(http_response, dict)
        ):
            method = str(http_response.get("request_method") or "").strip() or "REQUEST"
            path = str(http_response.get("request_path") or "").strip() or "?"
            status = http_response.get("response_status")
            body = str(http_response.get("response_body_text") or "").strip()
            # Truncate body at ~400 chars so the headline stays scannable;
            # the full body lives on the http_response field.
            if len(body) > 400:
                body = body[:400] + "…"
            status_part = str(status) if status is not None else "no_response"
            tail = f"{status_part} {body}".strip()
            return (
                f"{deliverable_id} failed during {stage_label}: "
                f"{method} {path} → {tail}"
            )
        # Boot-stage HTTP healthcheck failures: when `_wait_for_http`
        # timed out with the app returning 5xx on every poll, the
        # harness emits a `Last HTTP response: <status> <body>` line
        # inside the error_text. That line is the canonical diagnostic
        # — far more useful than the last 3 stderr lines (which for
        # Python+Uvicorn is the success banner). Surface it as the
        # headline teaser regardless of what stderr looks like.
        if failed_stage in (SandboxFailureStage.boot, SandboxFailureStage.runtime):
            http_line = self._extract_last_http_response_line(error_text)
            if http_line:
                return f"{deliverable_id} failed during {stage_label}: {http_line}"
        # Stage-progress timeouts: when `_wait_for_http` deadline fires
        # while the container is still running (e.g., heavy `bundle
        # install`, `apt-get`, etc.), the canonical diagnostic is the
        # timeout itself — NOT the last 3 stderr lines, which are just
        # the progress message at the moment the harness gave up.
        # Surface "Timed out waiting for ..." as the headline regardless
        # of stage. Without this, a reader (or repair LLM) sees gem-
        # install progress lines and concludes "bundler is broken" when
        # really "harness gave up too early on a healthy container."
        timeout_line = self._extract_timeout_line(error_text)
        if timeout_line:
            return f"{deliverable_id} failed during {stage_label}: {timeout_line}"
        source = (logs or error_text or "").strip()
        if not source:
            return f"{deliverable_id} failed during {stage_label}."
        if failed_stage == SandboxFailureStage.image_build:
            diagnostic = self._image_build_diagnostic_line(source)
            if diagnostic:
                return f"{deliverable_id} failed during {stage_label}: {diagnostic}"
        # Prefer the container's stderr/logs (more focused, since the harness
        # now feeds stderr-only here) over an opaque exception string. Fall
        # back to whatever caller passed if logs are empty.
        tail_lines = [line for line in source.splitlines() if line.strip()][-3:]
        teaser = " | ".join(tail_lines)
        return f"{deliverable_id} failed during {stage_label}: {teaser}"

    @staticmethod
    def _extract_timeout_line(error_text: str | None) -> str | None:
        """Pull the `Timed out waiting for <url> during <stage>` segment
        out of a wait-for-http timeout exception message.

        Emitted by :func:`LearnerStudioService._wait_for_http` when the
        deadline fires while the container is still alive. We surface
        this verbatim so the headline says "harness gave up after Xs"
        instead of showing the last 3 stderr lines (which for slow
        installs are misleading progress messages).
        """
        if not error_text:
            return None
        marker = "Timed out waiting for"
        idx = error_text.find(marker)
        if idx == -1:
            return None
        tail = error_text[idx:]
        # Trim at the first newline so any container log dumps appended
        # to the exception message don't bleed into the headline.
        newline = tail.find("\n")
        if newline != -1:
            tail = tail[:newline]
        return tail.strip()

    @staticmethod
    def _extract_last_http_response_line(error_text: str | None) -> str | None:
        """Pull the `Last HTTP response: <status> <body>` segment out of
        the wait-for-http timeout message.

        The marker is emitted by
        :func:`LearnerStudioService._format_last_http_response`. We
        return the substring from "Last HTTP response:" to the end of
        the line / paragraph so the headline carries the status and
        body excerpt verbatim.
        """
        if not error_text:
            return None
        marker = "Last HTTP response:"
        idx = error_text.find(marker)
        if idx == -1:
            return None
        tail = error_text[idx:]
        # Trim at the first newline (so logs appended later don't bleed in).
        newline = tail.find("\n")
        if newline != -1:
            tail = tail[:newline]
        return tail.strip()

    def _summarize_failed_deliverables(
        self,
        deliverable_reports: list[DeliverableSandboxReport],
    ) -> str:
        failures = [
            report
            for report in deliverable_reports
            if not report.compile_succeeded
            or not report.runtime_succeeded
            or report.public_checks_passed is False
            or report.error
        ]
        if not failures:
            return "Starter deliverable verification failed on the authored runtime harness."
        primary = failures[0]
        return self._summarize_stage_failure(
            deliverable_id=primary.deliverable_id,
            failed_stage=primary.failed_stage,
            error_text=primary.error,
            logs=primary.stderr,
            default="Starter deliverable verification failed on the authored runtime harness.",
            http_response=primary.http_response,
        )

    def _tail_lines(self, text: str | None, *, max_lines: int) -> str:
        if not text:
            return ""
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        return "\n".join(lines[-max_lines:])

    def _json_request(self, method: str, url: str, payload: dict | None = None) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _workspace_cache_key(self, workspace_root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(p for p in workspace_root.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(workspace_root)).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _cached_image_tag(self, cache_key: str) -> str:
        return f"{self.cache_namespace}:{cache_key[:24]}"

    def _image_exists(self, image_tag: str) -> bool:
        inspect_result = subprocess.run(
            [self.docker_binary, "image", "inspect", image_tag],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return inspect_result.returncode == 0

    def _parse_run_output(self, stdout: str) -> dict:
        text = stdout.strip()
        if not text:
            return {"success": False, "deliverable_reports": [], "error": "Sandbox verification did not emit JSON output."}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"success": False, "deliverable_reports": [], "error": "Sandbox verification output was not valid JSON."}

    def _coerce_bytes(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
