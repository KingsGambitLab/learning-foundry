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

        if run.artifacts.task_agent_spec is None:
            return SandboxExecutionResult(
                status=SandboxExecutionStatus.unavailable,
                available=availability.available,
                generated_at=now,
                duration_ms=0,
                error="Sandbox execution only supports task-agent workflow runs.",
            )

        if not availability.available:
            return SandboxExecutionResult(
                status=SandboxExecutionStatus.unavailable,
                available=False,
                generated_at=now,
                duration_ms=0,
                error=availability.message,
            )

        try:
            bundle = self._materialize_workspace(run)
            workspace_root = Path(bundle.public_dir)
            starter_root = workspace_root / "starter"
            if starter_root.exists():
                return self._execute_starter_harness(
                    workspace_root=workspace_root,
                    spec=run.artifacts.task_agent_spec,
                    now=now,
                    started=started,
                )
            return self._execute_legacy_runtime(
                workspace_root=workspace_root,
                now=now,
                started=started,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxExecutionResult(
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

        for deliverable in spec.deliverables:
            starter_root = workspace_root / "starter" / deliverable.id
            if not starter_root.exists():
                all_builds_succeeded = False
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
                self.runtime_harness._start_runtime_support_services(
                    starter_root,
                    network_name=network_name,
                    container_prefix=container_name,
                )
                local_run_command = [
                    self.docker_binary,
                    "run",
                    "-d",
                    "--rm",
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
                self.runtime_harness._wait_for_http(
                    f"{base_url}{healthcheck_path}",
                    container_name=container_name,
                )
                checks_passed, check_output, check_error = self._run_public_checks(manifest, base_url)
                logs = self.runtime_harness._container_logs(container_name) or ""
                run_stdout_parts.append(f"[{deliverable.id}] {check_output}".strip())
                if logs:
                    run_stderr_parts.append(f"[{deliverable.id}] {logs}".strip())
                if not checks_passed:
                    all_runs_succeeded = False
                deliverable_reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=True,
                        runtime_succeeded=checks_passed,
                        health_status_code=200,
                        stdout=check_output,
                        stderr=logs,
                        error=check_error,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                all_runs_succeeded = False
                logs = self.runtime_harness._container_logs(container_name) or ""
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
        return SandboxExecutionResult(
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

    def _starter_runtime_image_tag(self, workspace_root: Path) -> str:
        cache_key = self._workspace_cache_key(workspace_root)
        return f"{self.cache_namespace}:{cache_key[:24]}"

    def _run_public_checks(self, manifest: dict, base_url: str) -> tuple[bool, str, str | None]:
        public_cases = manifest.get("public_check_cases") or []
        public_checks = manifest.get("public_checks") or []
        if not public_cases:
            return False, "No public checks were configured for this deliverable.", "No public checks were configured."

        checks_by_case = {
            check.get("case_id"): check
            for check in public_checks
            if isinstance(check, dict) and check.get("case_id")
        }
        required_fields = self._required_output_fields(manifest)
        passed = True
        lines: list[str] = []
        for case in public_cases:
            case_id = str(case.get("id") or "unnamed_case")
            check = checks_by_case.get(case_id) or {}
            title = str(check.get("title") or case_id)
            try:
                response = self._json_request("POST", f"{base_url}/run", case.get("input") or {})
            except urllib.error.HTTPError as exc:
                passed = False
                lines.append(f"[FAIL] {title}: HTTP {exc.code}")
                continue
            except Exception as exc:  # noqa: BLE001
                passed = False
                lines.append(f"[FAIL] {title}: {exc}")
                continue

            output = response.get("output") or {}
            missing = [field for field in required_fields if field not in output]
            mismatches = [
                f"{key} expected {value!r} got {output.get(key)!r}"
                for key, value in (case.get("expected_output") or {}).items()
                if output.get(key) != value
            ]
            if missing or mismatches:
                passed = False
                lines.append(f"[FAIL] {title}")
                if missing:
                    lines.append(f"  Missing output fields: {', '.join(missing)}")
                lines.extend(f"  {mismatch}" for mismatch in mismatches)
                continue
            lines.append(f"[PASS] {title}")

        if passed:
            lines.append("")
            lines.append("Starter deliverable public checks passed in Docker.")
            return True, "\n".join(lines), None
        return False, "\n".join(lines), "One or more public starter checks failed."

    def _json_request(self, method: str, url: str, payload: dict | None = None) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def _required_output_fields(self, manifest: dict) -> list[str]:
        schema = manifest.get("output_schema") or {}
        required = schema.get("required")
        if isinstance(required, list) and required:
            return [str(field) for field in required]
        properties = schema.get("properties") or {}
        return [str(field) for field in properties.keys()]

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
