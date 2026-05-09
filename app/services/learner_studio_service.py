from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx

from app.domain.grading import LiveAssignmentGradeReport, LiveGradeTaskAgentRequest
from app.domain.learner import LearnerWorkspaceScope, LearnerWorkspaceSession, LearnerWorkspaceSessionStatus
from app.domain.task_agent import TaskAgentServiceSpec
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner, TaskAgentRunnerError
from app.services.task_agent_starter_templates import HIDDEN_MANIFEST_PATH, PREVIEW_LAUNCHER_PATH


class LearnerStudioError(RuntimeError):
    """Raised when the learner workspace studio or grading runner fails."""


def default_learner_studio_image() -> str:
    return "course-gen-learner-studio:latest"


class LearnerStudioService:
    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        image_name: str | None = None,
        build_timeout_s: int = 600,
        start_timeout_s: int = 90,
        host: str = "127.0.0.1",
        runner: TaskAgentBlackBoxRunner | None = None,
    ) -> None:
        self.docker_binary = docker_binary
        self.image_name = image_name or default_learner_studio_image()
        self.build_timeout_s = build_timeout_s
        self.start_timeout_s = start_timeout_s
        self.host = host
        self.runner = runner or TaskAgentBlackBoxRunner()

    def launch_editor(
        self,
        *,
        enrollment_id: str,
        deliverable_id: str,
        workspace_root: str | Path,
        scope: LearnerWorkspaceScope,
        existing_session: LearnerWorkspaceSession | None = None,
        start_support_services: bool = True,
    ) -> LearnerWorkspaceSession:
        workspace_path = Path(workspace_root).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)

        if existing_session is not None and existing_session.container_name:
            if self._can_reuse_session(
                existing_session=existing_session,
                workspace_path=workspace_path,
            ):
                refreshed = existing_session.model_copy(deep=True)
                refreshed.deliverable_id = deliverable_id
                refreshed.status = LearnerWorkspaceSessionStatus.running
                refreshed.updated_at = self._now()
                return refreshed

        host_port = self._allocate_port()
        session_id = existing_session.id if existing_session is not None else f"studio_{uuid4().hex[:12]}"
        container_name = existing_session.container_name if existing_session and existing_session.container_name else f"course-gen-studio-{session_id.lower()}"
        network_name = f"{container_name}-net"
        dependency_services = self._dependency_services(workspace_path)
        use_support_network = start_support_services and bool(dependency_services)

        self._remove_runtime_support(workspace_path, network_name=network_name, container_prefix=container_name)
        self._ensure_image()
        if use_support_network:
            self._start_runtime_support_services(
                workspace_path,
                network_name=network_name,
                container_prefix=container_name,
            )
        command = [
            self.docker_binary,
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{host_port}:8080",
            "-v",
            f"{workspace_path}:/workspace",
            "-w",
            "/workspace",
            *(
                [
                    "--network",
                    network_name,
                    "--network-alias",
                    "editor",
                ]
                if use_support_network
                else []
            ),
            *self._docker_env_args(self._app_runtime_environment(workspace_path)),
            self.image_name,
            "code-server",
            "--bind-addr",
            "0.0.0.0:8080",
            "--auth",
            "none",
            "--user-data-dir",
            "/tmp/code-server",
            "/workspace",
        ]
        session_image_name = self.image_name
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.build_timeout_s,
        )
        if result.returncode != 0:
            raise LearnerStudioError(
                (result.stderr or result.stdout).strip() or "Could not start learner editor container."
            )

        editor_url = f"http://{self.host}:{host_port}/"
        try:
            self._wait_for_http(editor_url, container_name=container_name)
        except Exception:
            self._remove_runtime_support(workspace_path, network_name=network_name, container_prefix=container_name)
            raise
        return LearnerWorkspaceSession(
            id=session_id,
            enrollment_id=enrollment_id,
            deliverable_id=deliverable_id,
            scope=scope,
            created_at=existing_session.created_at if existing_session is not None else self._now(),
            updated_at=self._now(),
            status=LearnerWorkspaceSessionStatus.running,
            workspace_root=str(workspace_path),
            container_name=container_name,
            host_port=host_port,
            editor_url=editor_url,
            image_name=session_image_name,
            notes=["VS Code (code-server) session running in Docker."],
        )

    def stop_editor(self, session: LearnerWorkspaceSession | None) -> None:
        if session is None or not session.container_name:
            return
        self._remove_runtime_support(
            Path(session.workspace_root).resolve(),
            network_name=f"{session.container_name}-net",
            container_prefix=session.container_name,
        )

    def _can_reuse_session(
        self,
        *,
        existing_session: LearnerWorkspaceSession,
        workspace_path: Path,
    ) -> bool:
        container_name = existing_session.container_name
        if not container_name:
            return False
        if Path(existing_session.workspace_root).resolve() != workspace_path:
            return False
        if not self._container_running(container_name):
            return False
        return True

    def grade_assignment(
        self,
        *,
        workspace_root: str | Path,
        spec: TaskAgentServiceSpec,
    ) -> LiveAssignmentGradeReport:
        workspace_path = Path(workspace_root).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)

        host_port = self._allocate_port()
        container_name = f"course-gen-grade-{uuid4().hex[:12]}"
        network_name = f"{container_name}-net"
        dependency_services = self._dependency_services(workspace_path)
        try:
            image_name = self._workspace_runtime_image_name(workspace_path)
            self._ensure_runtime_image_available(image_name)
            if dependency_services:
                self._start_runtime_support_services(
                    workspace_path,
                    network_name=network_name,
                    container_prefix=container_name,
                )
            command = [
                self.docker_binary,
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{host_port}:8000",
                "-v",
                f"{workspace_path}:/workspace",
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
                *self._docker_env_args(self._app_runtime_environment(workspace_path)),
                image_name,
                "sh",
                "-lc",
                self._runtime_launch_script(
                    workspace_path=workspace_path,
                    spec=spec,
                    include_setup=True,
                ),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.build_timeout_s,
            )
            if result.returncode != 0:
                raise LearnerStudioError(
                    (result.stderr or result.stdout).strip() or "Could not start grading container."
                )

            base_url = f"http://{self.host}:{host_port}"
            self._wait_for_http(
                f"{base_url}{self._healthcheck_path(workspace_path, spec)}",
                container_name=container_name,
            )
            return self.runner.grade_assignment_live(
                spec,
                LiveGradeTaskAgentRequest(
                    base_url=base_url,
                    workspace_root=str(workspace_path),
                ),
            )
        except TaskAgentRunnerError as exc:
            raise LearnerStudioError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise LearnerStudioError(f"Unexpected learner grading failure: {exc}") from exc
        finally:
            self._remove_runtime_support(
                workspace_path,
                network_name=network_name,
                container_prefix=container_name,
            )

    def _runtime_manifest(self, workspace_path: Path) -> dict[str, object]:
        manifest_path = workspace_path / HIDDEN_MANIFEST_PATH
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _runtime_commands(self, workspace_path: Path, *, phase: str) -> list[str]:
        manifest = self._runtime_manifest(workspace_path)
        runtime_plan = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
        steps = runtime_plan.get(f"{phase}_steps") or []
        commands: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            target = step.get("target_service_id")
            command = step.get("command")
            if command and target in (None, "app"):
                commands.append(str(command))
        return commands

    def _setup_commands(self, workspace_path: Path) -> list[str]:
        return self._runtime_commands(workspace_path, phase="setup")

    def _verify_commands(self, workspace_path: Path) -> list[str]:
        return self._runtime_commands(workspace_path, phase="verify")

    def _runtime_bootstrap_commands(self, workspace_path: Path) -> list[str]:
        manifest = self._runtime_manifest(workspace_path)
        runtime_plan = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
        language = str(runtime_plan.get("implementation_language") or "").strip().lower()
        package_manager = str(runtime_plan.get("package_manager") or "").strip().lower()
        commands: list[str] = []
        if language in {"typescript", "javascript"} and package_manager in {"pnpm", "yarn"}:
            commands.append("corepack enable")
        return commands

    def _runtime_shell_exports(self, workspace_path: Path) -> list[str]:
        manifest = self._runtime_manifest(workspace_path)
        runtime_plan = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
        language = str(runtime_plan.get("implementation_language") or "").strip().lower()
        package_manager = str(runtime_plan.get("package_manager") or "").strip().lower()
        exports: list[str] = []
        if language in {"typescript", "javascript"} and package_manager in {"pnpm", "yarn"}:
            exports.append("export COREPACK_ENABLE_DOWNLOAD_PROMPT=0")
        return exports

    def _preview_command(self, workspace_path: Path, spec: TaskAgentServiceSpec) -> str:
        manifest = self._runtime_manifest(workspace_path)
        preview_command = manifest.get("preview_command")
        if isinstance(preview_command, str) and preview_command:
            return preview_command
        if spec.runtime_dependencies.preview_command:
            return spec.runtime_dependencies.preview_command
        return f"python {PREVIEW_LAUNCHER_PATH} --host 0.0.0.0"

    def _runtime_launch_script(
        self,
        *,
        workspace_path: Path,
        spec: TaskAgentServiceSpec,
        include_setup: bool = True,
    ) -> str:
        lines = [
            "set -e",
            "export PORT=8000",
            *self._runtime_shell_exports(workspace_path),
            *self._runtime_bootstrap_commands(workspace_path),
            *(self._setup_commands(workspace_path) if include_setup else []),
            *[
                line
                for index, command in enumerate(self._verify_commands(workspace_path), start=1)
                for line in (
                    f"echo '[coursegen] verify step {index} started'",
                    command,
                )
            ],
            f"exec {self._preview_command(workspace_path, spec)}",
        ]
        return "\n".join(lines)

    def _healthcheck_path(self, workspace_path: Path, spec: TaskAgentServiceSpec) -> str:
        manifest = self._runtime_manifest(workspace_path)
        runtime_plan = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
        services = runtime_plan.get("services") or []
        for service in services:
            if not isinstance(service, dict):
                continue
            if service.get("service_id") != "app":
                continue
            healthcheck_path = service.get("healthcheck_path")
            if isinstance(healthcheck_path, str) and healthcheck_path:
                return healthcheck_path
        for service in spec.project_contract.runtime_plan.services:
            if service.service_id == "app" and service.healthcheck_path:
                return service.healthcheck_path
        return "/health"

    def _runtime_services(self, workspace_path: Path) -> list[dict[str, object]]:
        manifest = self._runtime_manifest(workspace_path)
        runtime_plan = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
        services = runtime_plan.get("services") or []
        normalized: list[dict[str, object]] = []
        for service in services:
            if isinstance(service, dict) and service.get("service_id"):
                normalized.append(service)
        return normalized

    def _dependency_services(self, workspace_path: Path) -> list[dict[str, object]]:
        return [
            service
            for service in self._runtime_services(workspace_path)
            if str(service.get("service_id")) != "app" and service.get("container_image")
        ]

    def _app_runtime_environment(self, workspace_path: Path) -> dict[str, str]:
        environment: dict[str, str] = {}
        for service in self._dependency_services(workspace_path):
            service_id = str(service.get("service_id"))
            technology = str(service.get("technology") or "").strip().lower()
            if technology in {"postgres", "postgresql"}:
                environment.setdefault("DATABASE_URL", f"postgresql://postgres:postgres@{service_id}:5432/app")
                environment.setdefault("POSTGRES_HOST", service_id)
                environment.setdefault("POSTGRES_PORT", "5432")
                environment.setdefault("POSTGRES_DB", "app")
                environment.setdefault("POSTGRES_USER", "postgres")
                environment.setdefault("POSTGRES_PASSWORD", "postgres")
            elif technology in {"mongodb", "mongo"}:
                environment.setdefault("MONGODB_URL", f"mongodb://{service_id}:27017/app")
                environment.setdefault("MONGO_URL", f"mongodb://{service_id}:27017/app")
                environment.setdefault("MONGO_HOST", service_id)
            elif technology == "redis":
                environment.setdefault("REDIS_URL", f"redis://{service_id}:6379/0")
                environment.setdefault("REDIS_HOST", service_id)
                environment.setdefault("REDIS_PORT", "6379")
            elif technology in {"mysql", "mariadb"}:
                environment.setdefault("DATABASE_URL", f"mysql://root:root@{service_id}:3306/app")
                environment.setdefault("MYSQL_HOST", service_id)
                environment.setdefault("MYSQL_PORT", "3306")
                environment.setdefault("MYSQL_DATABASE", "app")
                environment.setdefault("MYSQL_ROOT_PASSWORD", "root")
            if technology:
                upper = technology.upper().replace("-", "_")
                environment.setdefault(f"{upper}_HOST", service_id)
        return environment

    def _workspace_runtime_cache_key(self, workspace_path: Path) -> str:
        digest = hashlib.sha256()
        ignored = {
            ".coursegen",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "node_modules",
        }
        for path in sorted(p for p in workspace_path.rglob("*") if p.is_file()):
            relative = path.relative_to(workspace_path)
            if any(part in ignored for part in relative.parts):
                continue
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _workspace_runtime_image_tag(self, workspace_path: Path) -> str:
        return f"course-gen-runtime:{self._workspace_runtime_cache_key(workspace_path)[:24]}"

    def _workspace_runtime_image_name(self, workspace_path: Path) -> str:
        for service in self._runtime_services(workspace_path):
            if str(service.get("service_id")) != "app":
                continue
            container_image = service.get("container_image")
            if isinstance(container_image, str) and container_image.strip():
                return container_image.strip()
        return self.image_name

    def _ensure_runtime_image_available(self, image_name: str) -> None:
        if image_name == self.image_name:
            self._ensure_image()

    def _ensure_workspace_runtime_image(self, workspace_path: Path) -> str:
        image_tag = self._workspace_runtime_image_tag(workspace_path)
        if self._image_exists(image_tag):
            return image_tag
        command = [
            self.docker_binary,
            "build",
            "-t",
            image_tag,
            ".",
        ]
        result = subprocess.run(
            command,
            cwd=workspace_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.build_timeout_s,
        )
        if result.returncode != 0:
            raise LearnerStudioError(
                (result.stderr or result.stdout).strip() or "Could not build learner runtime image."
            )
        return image_tag

    def _workspace_editor_image_tag(self, runtime_image_name: str) -> str:
        digest = hashlib.sha256(runtime_image_name.encode("utf-8")).hexdigest()
        return f"course-gen-editor:{digest[:24]}"

    def _ensure_workspace_editor_image(self, runtime_image_name: str) -> str:
        image_tag = self._workspace_editor_image_tag(runtime_image_name)
        if self._image_exists(image_tag):
            return image_tag
        with tempfile.TemporaryDirectory(prefix="course_gen_editor_image_") as temp_dir:
            dockerfile = Path(temp_dir) / "Dockerfile"
            dockerfile.write_text(
                "\n".join(
                    [
                        f"FROM {runtime_image_name}",
                        "",
                        "RUN apt-get update \\",
                        "    && apt-get install -y --no-install-recommends curl ca-certificates python3 git \\",
                        "    && curl -fsSL https://code-server.dev/install.sh | sh \\",
                        "    && rm -rf /var/lib/apt/lists/*",
                        "",
                        "WORKDIR /workspace",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    self.docker_binary,
                    "build",
                    "-t",
                    image_tag,
                    temp_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.build_timeout_s,
            )
        if result.returncode != 0:
            raise LearnerStudioError(
                (result.stderr or result.stdout).strip() or "Could not build learner editor image."
            )
        return image_tag

    def _docker_env_args(self, environment: dict[str, str]) -> list[str]:
        args: list[str] = []
        for key, value in sorted(environment.items()):
            args.extend(["-e", f"{key}={value}"])
        return args

    def _service_runtime_environment(self, service: dict[str, object]) -> dict[str, str]:
        technology = str(service.get("technology") or "").strip().lower()
        if technology in {"postgres", "postgresql"}:
            return {
                "POSTGRES_DB": "app",
                "POSTGRES_PASSWORD": "postgres",
                "POSTGRES_USER": "postgres",
            }
        if technology in {"mysql", "mariadb"}:
            return {
                "MYSQL_DATABASE": "app",
                "MYSQL_ROOT_PASSWORD": "root",
            }
        return {}

    def _create_network(self, network_name: str) -> None:
        subprocess.run(
            [self.docker_binary, "network", "create", network_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _remove_network(self, network_name: str) -> None:
        subprocess.run(
            [self.docker_binary, "network", "rm", network_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _service_container_name(self, container_prefix: str, service_id: str) -> str:
        return f"{container_prefix}-{service_id}"

    def _start_runtime_support_services(
        self,
        workspace_path: Path,
        *,
        network_name: str,
        container_prefix: str,
    ) -> None:
        dependencies = self._dependency_services(workspace_path)
        if not dependencies:
            return
        self._create_network(network_name)
        started: list[str] = []
        try:
            for service in dependencies:
                container_name = self._service_container_name(container_prefix, str(service["service_id"]))
                self._remove_container(container_name)
                command = [
                    self.docker_binary,
                    "run",
                    "-d",
                    "--name",
                    container_name,
                    "--network",
                    network_name,
                    "--network-alias",
                    str(service["service_id"]),
                    *self._docker_env_args(self._service_runtime_environment(service)),
                    str(service["container_image"]),
                ]
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.build_timeout_s,
                )
                if result.returncode != 0:
                    raise LearnerStudioError(
                        (result.stderr or result.stdout).strip()
                        or f"Could not start support service '{service['service_id']}'."
                    )
                started.append(container_name)
            time.sleep(2.0)
        except Exception:
            for container_name in started:
                self._remove_container(container_name)
            self._remove_network(network_name)
            raise

    def _remove_runtime_support(
        self,
        workspace_path: Path,
        *,
        network_name: str,
        container_prefix: str,
    ) -> None:
        self._remove_container(container_prefix)
        for service in self._dependency_services(workspace_path):
            self._remove_container(
                self._service_container_name(container_prefix, str(service["service_id"]))
            )
        self._remove_network(network_name)

    def _ensure_image(self) -> None:
        if self._image_exists():
            return
        repo_root = Path(__file__).resolve().parents[2]
        dockerfile = repo_root / "docker" / "learner-studio.Dockerfile"
        command = [
            self.docker_binary,
            "build",
            "-f",
            str(dockerfile),
            "-t",
            self.image_name,
            str(repo_root),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.build_timeout_s,
        )
        if result.returncode != 0:
            raise LearnerStudioError(
                (result.stderr or result.stdout).strip() or "Could not build learner studio image."
            )

    def _image_exists(self, image_name: str | None = None) -> bool:
        inspect = subprocess.run(
            [self.docker_binary, "image", "inspect", image_name or self.image_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return inspect.returncode == 0

    def _container_running(self, container_name: str) -> bool:
        inspect = subprocess.run(
            [self.docker_binary, "inspect", "-f", "{{.State.Running}}", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return inspect.returncode == 0 and inspect.stdout.strip() == "true"

    def _remove_container(self, container_name: str) -> None:
        subprocess.run(
            [self.docker_binary, "rm", "-f", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _container_logs(self, container_name: str) -> str | None:
        result = subprocess.run(
            [self.docker_binary, "logs", "--tail", "80", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        logs = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        return logs or None

    def _wait_for_http(self, url: str, *, container_name: str | None = None) -> None:
        deadline = time.time() + self.start_timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                response = httpx.get(url, timeout=2.0, follow_redirects=False)
                if response.status_code < 500:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            if container_name and not self._container_running(container_name):
                details = (
                    f"Container '{container_name}' stopped before '{url}' became healthy. "
                    f"Last error: {last_error}"
                )
                logs = self._container_logs(container_name)
                if logs:
                    details = f"{details}\n\nContainer logs:\n{logs}"
                raise LearnerStudioError(details)
            time.sleep(1.0)
        details = f"Timed out waiting for '{url}' to respond. Last error: {last_error}"
        if container_name:
            logs = self._container_logs(container_name)
            if logs:
                details = f"{details}\n\nContainer logs:\n{logs}"
        raise LearnerStudioError(details)

    def _allocate_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host, 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _now(self):
        from datetime import UTC, datetime

        return datetime.now(UTC)
