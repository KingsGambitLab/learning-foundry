from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx

from app.domain.grading import LiveGradeTaskAgentRequest, LiveTaskAgentGradeReport
from app.domain.learner import LearnerWorkspaceScope, LearnerWorkspaceSession, LearnerWorkspaceSessionStatus
from app.domain.task_agent import TaskAgentServiceSpec
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner, TaskAgentRunnerError


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
        module_id: str,
        workspace_root: str | Path,
        scope: LearnerWorkspaceScope,
        existing_session: LearnerWorkspaceSession | None = None,
    ) -> LearnerWorkspaceSession:
        workspace_path = Path(workspace_root).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        self._ensure_image()

        if existing_session is not None and existing_session.container_name:
            if self._can_reuse_session(
                existing_session=existing_session,
                workspace_path=workspace_path,
                module_id=module_id,
            ):
                refreshed = existing_session.model_copy(deep=True)
                refreshed.module_id = module_id
                refreshed.status = LearnerWorkspaceSessionStatus.running
                refreshed.updated_at = self._now()
                return refreshed

        host_port = self._allocate_port()
        session_id = existing_session.id if existing_session is not None else f"studio_{uuid4().hex[:12]}"
        container_name = existing_session.container_name if existing_session and existing_session.container_name else f"course-gen-studio-{session_id.lower()}"

        self._remove_container(container_name)
        command = [
            self.docker_binary,
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-p",
            f"{host_port}:8080",
            "-v",
            f"{workspace_path}:/workspace",
            "-w",
            "/workspace",
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
        self._wait_for_http(editor_url)
        return LearnerWorkspaceSession(
            id=session_id,
            enrollment_id=enrollment_id,
            module_id=module_id,
            scope=scope,
            created_at=existing_session.created_at if existing_session is not None else self._now(),
            updated_at=self._now(),
            status=LearnerWorkspaceSessionStatus.running,
            workspace_root=str(workspace_path),
            container_name=container_name,
            host_port=host_port,
            editor_url=editor_url,
            image_name=self.image_name,
            notes=["VS Code (code-server) session running in Docker."],
        )

    def _can_reuse_session(
        self,
        *,
        existing_session: LearnerWorkspaceSession,
        workspace_path: Path,
        module_id: str,
    ) -> bool:
        container_name = existing_session.container_name
        if not container_name:
            return False
        if Path(existing_session.workspace_root).resolve() != workspace_path:
            return False
        if not self._container_running(container_name):
            return False
        return self._container_current_module_id(container_name) == module_id

    def grade_workspace(
        self,
        *,
        workspace_root: str | Path,
        spec: TaskAgentServiceSpec,
        module_id: str,
    ) -> LiveTaskAgentGradeReport:
        workspace_path = Path(workspace_root).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        self._ensure_image()

        host_port = self._allocate_port()
        container_name = f"course-gen-grade-{uuid4().hex[:12]}"
        command = [
            self.docker_binary,
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-p",
            f"{host_port}:8000",
            "-v",
            f"{workspace_path}:/workspace",
            "-w",
            "/workspace",
            self.image_name,
            "python",
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
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
        try:
            self._wait_for_http(f"{base_url}/health")
            return self.runner.grade_live(
                spec,
                module_id,
                LiveGradeTaskAgentRequest(base_url=base_url),
            )
        except TaskAgentRunnerError as exc:
            raise LearnerStudioError(str(exc)) from exc
        finally:
            self._remove_container(container_name)

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

    def _image_exists(self) -> bool:
        inspect = subprocess.run(
            [self.docker_binary, "image", "inspect", self.image_name],
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

    def _container_current_module_id(self, container_name: str) -> str | None:
        inspect = subprocess.run(
            [
                self.docker_binary,
                "exec",
                container_name,
                "sh",
                "-lc",
                "cat /workspace/.coursegen/current_module.txt 2>/dev/null",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if inspect.returncode != 0:
            return None
        current_module_id = inspect.stdout.strip()
        return current_module_id or None

    def _wait_for_http(self, url: str) -> None:
        deadline = time.time() + self.start_timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                response = httpx.get(url, timeout=2.0, follow_redirects=False)
                if response.status_code < 500:
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
            time.sleep(1.0)
        raise LearnerStudioError(f"Timed out waiting for '{url}' to respond. Last error: {last_error}")

    def _allocate_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host, 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _now(self):
        from datetime import UTC, datetime

        return datetime.now(UTC)
