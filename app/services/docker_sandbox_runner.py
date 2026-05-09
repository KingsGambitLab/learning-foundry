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
    SandboxExecutionStatus,
)
from app.domain.task_agent import TaskAgentServiceSpec
from app.domain.workflow import WorkflowRun
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.coursegen_logging import log_coursegen_event
from app.services.generated_test_harness import GeneratedTestScriptRunner
from app.services.learner_studio_service import LearnerStudioService


class DockerSandboxRunner:
    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        build_timeout_s: int = 180,
        run_timeout_s: int = 180,
        keep_image: bool = False,
        cache_images: bool = True,
        cache_namespace: str = "course-gen-cache",
        workspace_manager: AssignmentWorkspaceManager | None = None,
    ) -> None:
        self.docker_binary = docker_binary
        self.build_timeout_s = build_timeout_s
        self.run_timeout_s = run_timeout_s
        self.keep_image = keep_image
        self.cache_images = cache_images
        self.cache_namespace = cache_namespace
        self.workspace_manager = workspace_manager
        self.runtime_harness = LearnerStudioService(
            docker_binary=docker_binary,
            build_timeout_s=build_timeout_s,
            start_timeout_s=min(run_timeout_s, 90),
            host="127.0.0.1",
        )
        self.test_script_runner = GeneratedTestScriptRunner(command_timeout_s=min(run_timeout_s, 90))

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
        log_coursegen_event(
            "sandbox_starter_harness_started",
            workflow_run_id=workflow_run_id,
            workspace_root=str(workspace_root),
            deliverable_count=len(spec.deliverables),
        )

        for deliverable in spec.deliverables:
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

            manifest = self.runtime_harness._runtime_manifest(starter_root)
            image_name = self.runtime_harness._workspace_runtime_image_name(starter_root)
            build_command = []
            build_stdout_parts.append(
                f"[{deliverable.id}] Using runtime image {image_name} from the authored runtime plan."
            )
            self.runtime_harness._ensure_runtime_image_available(image_name)
            if self.runtime_harness._image_exists(image_name):
                any_cached = True

            host_port = self.runtime_harness._allocate_port()
            container_name = f"course-gen-sandbox-{deliverable.id}-{uuid4().hex[:8]}".lower()
            network_name = f"{container_name}-net"
            base_url = f"http://127.0.0.1:{host_port}"
            logs = ""
            try:
                log_coursegen_event(
                    "sandbox_deliverable_support_services_starting",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    dependency_service_count=len(self.runtime_harness._dependency_services(starter_root)),
                    network_name=network_name,
                )
                self.runtime_harness._start_runtime_support_services(
                    starter_root,
                    network_name=network_name,
                    container_prefix=container_name,
                )
                log_coursegen_event(
                    "sandbox_deliverable_support_services_started",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    dependency_service_count=len(self.runtime_harness._dependency_services(starter_root)),
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
                    f"{workspace_root}:/workspace",
                    "-w",
                    f"/workspace/starter/{deliverable.id}",
                    *(
                        [
                            "--network",
                            network_name,
                            "--network-alias",
                            "app",
                        ]
                        if self.runtime_harness._dependency_services(starter_root)
                        else []
                    ),
                    *self.runtime_harness._docker_env_args(
                        self.runtime_harness._app_runtime_environment(starter_root)
                    ),
                    image_name,
                    "sh",
                    "-lc",
                    self.runtime_harness._runtime_launch_script(
                        workspace_path=starter_root,
                        spec=spec,
                        include_setup=True,
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
                    log_coursegen_event(
                        "sandbox_deliverable_runtime_launch_failed",
                        workflow_run_id=workflow_run_id,
                        deliverable_id=deliverable.id,
                        return_code=run_result.returncode,
                        error="Could not start the starter runtime container.",
                    )
                    run_stdout_parts.append(f"[{deliverable.id}] {run_result.stdout}".strip())
                    run_stderr_parts.append(f"[{deliverable.id}] {run_result.stderr}\n{logs}".strip())
                    deliverable_reports.append(
                        DeliverableSandboxReport(
                            deliverable_id=deliverable.id,
                            compile_succeeded=True,
                            runtime_succeeded=False,
                            stdout=run_result.stdout,
                            stderr="\n".join(part for part in [run_result.stderr, logs] if part),
                            error="Could not start the starter runtime container.",
                        )
                    )
                    continue

                healthcheck_path = self.runtime_harness._healthcheck_path(starter_root, spec)
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
                contract_passed, contract_output, contract_error = self._probe_contract_smoke(manifest, base_url)
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
                log_coursegen_event(
                    "sandbox_deliverable_public_checks_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    contract_passed=contract_passed,
                    checks_passed=checks_passed,
                    error=check_error,
                )
                deliverable_reports.append(
                        DeliverableSandboxReport(
                            deliverable_id=deliverable.id,
                            compile_succeeded=True,
                            runtime_succeeded=contract_passed,
                            public_checks_passed=checks_passed,
                            health_status_code=200,
                            stdout=combined_output,
                            stderr=logs,
                            error=contract_error or check_error,
                        )
                    )
                log_coursegen_event(
                    "sandbox_deliverable_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    sandbox_status="passed" if contract_passed else "failed",
                    error=check_error,
                )
            except Exception as exc:  # noqa: BLE001
                all_runs_succeeded = False
                logs = self.runtime_harness._container_logs(container_name) or ""
                log_coursegen_event(
                    "sandbox_deliverable_completed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable.id,
                    sandbox_status="failed",
                    error=str(exc),
                )
                run_stderr_parts.append(f"[{deliverable.id}] {exc}\n{logs}".strip())
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=True,
                        runtime_succeeded=False,
                        stdout="",
                        stderr=logs,
                        error=str(exc),
                    )
                )
            finally:
                self.runtime_harness._remove_runtime_support(
                    starter_root,
                    network_name=network_name,
                    container_prefix=container_name,
                )

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
            else "Starter deliverable verification failed on the authored runtime harness.",
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

    def _starter_runtime_image_tag(self, workspace_root: Path) -> str:
        cache_key = self._workspace_cache_key(workspace_root)
        return f"{self.cache_namespace}:{cache_key[:24]}"

    def _probe_contract_smoke(self, manifest: dict, base_url: str) -> tuple[bool, str, str | None]:
        public_checks = manifest.get("public_checks") or []
        if not public_checks:
            return False, "No public checks were configured for this deliverable.", "No public checks were configured."
        contract_passed = True
        lines: list[str] = []
        for check in public_checks:
            if not isinstance(check, dict):
                continue
            title = str(check.get("title") or check.get("request_path") or "visible check")
            method = str(check.get("request_method") or "POST").upper()
            request_path = str(check.get("request_path") or "").strip()
            if not request_path.startswith("/"):
                contract_passed = False
                lines.append(f"[FAIL] {title}: invalid request path")
                continue
            try:
                response = self._json_request(method, f"{base_url}{request_path}", check.get("request_body") or None)
            except urllib.error.HTTPError as exc:
                contract_passed = False
                lines.append(f"[FAIL] {title}: HTTP {exc.code}")
                continue
            except Exception as exc:  # noqa: BLE001
                contract_passed = False
                lines.append(f"[FAIL] {title}: {exc}")
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
            return True, "\n".join(lines), None
        return False, "\n".join(lines), "One or more starter smoke checks could not exercise the published contract."

    def _run_visible_suite(
        self,
        *,
        starter_root: Path,
        manifest: dict,
        base_url: str,
    ) -> tuple[bool, str, str | None]:
        command = str(manifest.get("visible_check_command") or "python checks/run_visible_checks.py")
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
        return report.passed, "\n".join(lines), None

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
