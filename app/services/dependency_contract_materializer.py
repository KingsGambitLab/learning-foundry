from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

from app.domain.task_agent import ProjectRuntimePlanSpec
from app.services.dependency_contract_facts import expected_dependency_contract_paths
from app.services.task_agent_starter_templates import HIDDEN_MANIFEST_PATH, RUNTIME_INSTALL_SCRIPT_PATH


class DependencyContractMaterializationResult(BaseModel):
    deliverable_id: str
    attempted: bool = False
    succeeded: bool = True
    image_name: str | None = None
    command: list[str] = Field(default_factory=list)
    synced_paths: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class DependencyContractMaterializer:
    def __init__(
        self,
        *,
        docker_binary: str = "docker",
        command_timeout_s: int = 180,
    ) -> None:
        self.docker_binary = docker_binary
        self.command_timeout_s = command_timeout_s

    def materialize(
        self,
        *,
        starter_root: Path,
        runtime_plan: ProjectRuntimePlanSpec | None,
        deliverable_id: str,
    ) -> DependencyContractMaterializationResult:
        install_script = starter_root / RUNTIME_INSTALL_SCRIPT_PATH
        if not install_script.exists():
            return DependencyContractMaterializationResult(deliverable_id=deliverable_id)

        package_manager = runtime_plan.package_manager if runtime_plan is not None else None
        expected = expected_dependency_contract_paths(package_manager)
        tracked_paths = sorted(
            {
                *expected.get("manifests", []),
                *expected.get("lockfiles", []),
                *expected.get("toolchains", []),
            }
        )
        if not tracked_paths:
            return DependencyContractMaterializationResult(deliverable_id=deliverable_id)

        image_name = self._runtime_base_image(starter_root=starter_root, runtime_plan=runtime_plan)
        if not image_name:
            return DependencyContractMaterializationResult(deliverable_id=deliverable_id)

        with TemporaryDirectory(prefix=f"course_gen_contract_{deliverable_id}_") as temp_dir:
            temp_root = Path(temp_dir) / "workspace"
            shutil.copytree(starter_root, temp_root)
            command = [
                self.docker_binary,
                "run",
                "--rm",
                "-v",
                f"{temp_root.resolve()}:/workspace",
                "-w",
                "/workspace",
                image_name,
                "sh",
                "-c",
                "set -e\nsh .coursegen/runtime/install.sh",
            ]
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                return DependencyContractMaterializationResult(
                    deliverable_id=deliverable_id,
                    attempted=True,
                    succeeded=False,
                    image_name=image_name,
                    command=command,
                    stdout=self._coerce_bytes(getattr(exc, "stdout", b"")),
                    stderr=self._coerce_bytes(getattr(exc, "stderr", b"")),
                    error=f"Dependency contract materialization timed out: {exc}",
                )

            if result.returncode != 0:
                return DependencyContractMaterializationResult(
                    deliverable_id=deliverable_id,
                    attempted=True,
                    succeeded=False,
                    image_name=image_name,
                    command=command,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error=(
                        (result.stderr or result.stdout).strip()
                        or "Dependency contract materialization failed."
                    ),
                )

            synced_paths = self._sync_dependency_contract_paths(
                source_root=temp_root,
                target_root=starter_root,
                tracked_paths=tracked_paths,
            )
            return DependencyContractMaterializationResult(
                deliverable_id=deliverable_id,
                attempted=True,
                succeeded=True,
                image_name=image_name,
                command=command,
                synced_paths=synced_paths,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    def _runtime_base_image(
        self,
        *,
        starter_root: Path,
        runtime_plan: ProjectRuntimePlanSpec | None,
    ) -> str | None:
        manifest_path = starter_root / HIDDEN_MANIFEST_PATH
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            runtime_plan_payload = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
            services = runtime_plan_payload.get("services") or []
            for service in services:
                if not isinstance(service, dict):
                    continue
                if str(service.get("service_id")) != "app":
                    continue
                container_image = service.get("container_image")
                if isinstance(container_image, str) and container_image.strip():
                    return container_image.strip()

        if runtime_plan is None:
            return None
        for service in runtime_plan.services:
            if service.service_id != "app":
                continue
            if service.container_image:
                return service.container_image
        return None

    def _sync_dependency_contract_paths(
        self,
        *,
        source_root: Path,
        target_root: Path,
        tracked_paths: list[str],
    ) -> list[str]:
        synced_paths: list[str] = []
        for relative_path in tracked_paths:
            source = source_root / relative_path
            target = target_root / relative_path
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                source_text = source.read_text(encoding="utf-8")
                if not target.exists() or target.read_text(encoding="utf-8") != source_text:
                    target.write_text(source_text, encoding="utf-8")
                    synced_paths.append(relative_path)
                continue
            if target.exists():
                target.unlink()
                synced_paths.append(relative_path)
        return synced_paths

    def _coerce_bytes(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
