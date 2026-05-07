from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.sandbox import (
    ModuleSandboxReport,
    SandboxAvailability,
    SandboxExecutionResult,
    SandboxExecutionStatus,
)
from app.domain.workflow import WorkflowRun
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.artifact_materializer import ArtifactMaterializer


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

        image_tag = f"course-gen-{run.id.lower()}-{uuid4().hex[:8]}"
        build_command: list[str] = []
        run_command: list[str] = []
        workspace_root: Path | None = None
        cache_key: str | None = None
        build_cached = False

        try:
            if run.artifacts.workspace_snapshot is not None:
                existing = Path(run.artifacts.workspace_snapshot.root_dir)
                if existing.exists():
                    bundle = run.artifacts.workspace_snapshot
                elif self.workspace_manager is not None:
                    bundle = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
                    run.artifacts.workspace_snapshot = bundle
                else:
                    materializer = ArtifactMaterializer()
                    bundle = materializer.materialize_run(run, overwrite=True)
            elif self.workspace_manager is not None:
                bundle = self.workspace_manager.prepare_run_workspace(run, overwrite=True)
                run.artifacts.workspace_snapshot = bundle
            else:
                materializer = ArtifactMaterializer()
                bundle = materializer.materialize_run(run, overwrite=True)

            workspace_root = Path(bundle.public_dir)
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
            module_reports = [
                ModuleSandboxReport.model_validate(item)
                for item in parsed.get("module_reports", [])
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
                module_reports=module_reports,
                error=None if run_succeeded else parsed.get("error") or "Assignment sandbox verification failed.",
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxExecutionResult(
                status=SandboxExecutionStatus.failed,
                available=True,
                build_succeeded=bool(build_command),
                build_cached=build_cached,
                run_succeeded=False,
                generated_at=now,
                duration_ms=int((time.perf_counter() - started) * 1000),
                workspace_root=str(workspace_root) if workspace_root is not None else None,
                image_tag=image_tag,
                cache_key=cache_key,
                build_command=build_command,
                run_command=run_command,
                build_stdout=self._coerce_bytes(getattr(exc, "stdout", b"")),
                build_stderr=self._coerce_bytes(getattr(exc, "stderr", b"")),
                error=f"Docker sandbox timed out: {exc}",
            )
        finally:
            if image_tag and not self.keep_image and not self.cache_images:
                subprocess.run(
                    [self.docker_binary, "image", "rm", "-f", image_tag],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

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
            return {"success": False, "module_reports": [], "error": "Sandbox verification did not emit JSON output."}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"success": False, "module_reports": [], "error": "Sandbox verification output was not valid JSON."}

    def _coerce_bytes(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
