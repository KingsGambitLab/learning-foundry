from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.task_agent import TaskAgentServiceSpec
from app.domain.workflow import (
    ArtifactVisibility,
    BundleFile,
    BundleFileContent,
    MaterializedBundle,
    WorkflowRun,
)
from app.services.creator_asset_service import CreatorAssetService
from app.services.learner_brief_builder import (
    build_task_agent_deliverable_brief,
    render_learner_starter_readme,
)
from app.services.runtime_contract_surface import (
    load_starter_manifest,
    starter_contract_path_sets_for_manifest,
    starter_materialization_paths,
)
from app.services.task_agent_contract_surface import (
    learner_editable_paths_for_deliverable,
    primary_submit_endpoint_for_spec,
)
from app.services.task_agent_starter_templates import (
    HIDDEN_MANIFEST_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    build_task_agent_starter_files,
    default_preview_command,
    task_agent_runtime_base_image,
    task_agent_runtime_bootstrap_commands,
    task_agent_runtime_environment_lines,
)

def default_generated_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "generated"


class ArtifactMaterializer:
    def __init__(
        self,
        base_dir: str | Path | None = None,
        creator_asset_service: CreatorAssetService | None = None,
    ) -> None:
        self.base_dir = Path(base_dir or default_generated_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.creator_asset_service = creator_asset_service

    def materialize_run(self, run: WorkflowRun, overwrite: bool = True) -> MaterializedBundle:
        bundle_root = self.base_dir / run.id
        if bundle_root.exists():
            if not overwrite:
                manifest = bundle_root / "manifest.json"
                raise FileExistsError(f"Bundle already exists at '{manifest}'.")
            shutil.rmtree(bundle_root)

        public_dir = bundle_root / "public"
        private_dir = bundle_root / "private"
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)

        files: list[BundleFile] = []
        generated_at = datetime.now(UTC)

        if run.artifacts.task_agent_spec is not None:
            self._materialize_task_agent(
                run=run,
                spec=run.artifacts.task_agent_spec,
                public_dir=public_dir,
                private_dir=private_dir,
                files=files,
            )
        else:
            self._write_text(
                private_dir / "README.txt",
                "This workflow run is blocked and does not have a materializable draft yet.\n",
                ArtifactVisibility.private,
                files,
                bundle_root,
                role="blocked_run_readme",
                audience="internal",
                semantic_source="system_rendered",
            )

        manifest_payload = {
            "bundle_id": f"{run.id}_bundle",
            "generated_at": generated_at.isoformat(),
            "root_dir": str(bundle_root),
            "public_dir": str(public_dir),
            "private_dir": str(private_dir),
            "files": [entry.model_dump(mode="json") for entry in files],
        }
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

        return MaterializedBundle(
            bundle_id=f"{run.id}_bundle",
            generated_at=generated_at,
            root_dir=str(bundle_root),
            public_dir=str(public_dir),
            private_dir=str(private_dir),
            manifest_path=str(manifest_path),
            files=files,
        )

    def read_bundle_file(self, bundle: MaterializedBundle, relative_path: str) -> BundleFileContent:
        root = Path(bundle.root_dir).resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Requested file is outside the bundle root.")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(relative_path)
        media_type = self._guess_media_type(relative_path)
        return BundleFileContent(
            relative_path=relative_path,
            media_type=media_type,
            content=target.read_text(encoding="utf-8"),
        )

    def _materialize_task_agent(
        self,
        *,
        run: WorkflowRun,
        spec: TaskAgentServiceSpec,
        public_dir: Path,
        private_dir: Path,
        files: list[BundleFile],
    ) -> None:
        bundle_root = public_dir.parent
        self._write_json(
            private_dir / "task_agent_spec.json",
            spec.model_dump(mode="json"),
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="task_agent_spec_snapshot",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "validation_summary.json",
            run.artifacts.validation_summary or {},
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="validation_summary",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "progression_preview.json",
            run.artifacts.progression_preview,
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="progression_preview",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "workflow_snapshot.json",
            {
                "run_id": run.id,
                "title": run.title,
                "stage": run.stage.value,
                "status": run.status.value,
                "pending_gate": run.pending_gate.value if run.pending_gate else None,
                "origin_template": run.artifacts.origin_template,
            },
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="workflow_snapshot",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "node_executions.json",
            [node.model_dump(mode="json") for node in run.artifacts.node_executions],
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="node_executions",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "review_summary.json",
            run.artifacts.review_summary.model_dump(mode="json") if run.artifacts.review_summary is not None else {},
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="review_summary",
            audience="internal",
            semantic_source="system_snapshot",
        )

        self._write_text(
            public_dir / "README.md",
            self._task_agent_readme(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="course_readme",
            audience="learner",
            semantic_source="spec_rendered",
        )
        self._write_text(
            public_dir / "runtime" / "Dockerfile",
            self._assignment_runtime_dockerfile(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="runtime_dockerfile",
            audience="operator",
            semantic_source="starter_compiler",
        )
        self._write_text(
            public_dir / "runtime" / "README.md",
            self._assignment_runtime_readme(),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="runtime_readme",
            audience="operator",
            semantic_source="system_rendered",
        )
        self._write_text(
            public_dir / "runtime" / "verify_assignment.py",
            self._assignment_runtime_verifier(),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="runtime_verifier",
            audience="operator",
            semantic_source="system_rendered",
        )
        self._write_text(
            public_dir / "content" / "course_outline.md",
            self._course_outline(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="course_outline",
            audience="learner",
            semantic_source="spec_rendered",
        )

        for deliverable in spec.deliverables:
            deliverable_dir = public_dir / "starter" / deliverable.id
            self._write_text(
                deliverable_dir / "README.md",
                self._starter_readme(spec, deliverable.id),
                ArtifactVisibility.public,
                files,
                bundle_root,
                role="starter_readme",
                audience="learner",
                deliverable_id=deliverable.id,
                semantic_source="spec_rendered",
            )
            workspace_starter_dir = (
                Path(run.artifacts.workspace_snapshot.public_dir) / "starter" / deliverable.id
                if run.artifacts.workspace_snapshot is not None
                and Path(run.artifacts.workspace_snapshot.root_dir).resolve() != bundle_root.resolve()
                else None
            )
            if workspace_starter_dir is not None and workspace_starter_dir.exists():
                workspace_manifest = load_starter_manifest(workspace_starter_dir)
                for relative_path in self._workspace_starter_paths(
                    workspace_starter_dir=workspace_starter_dir,
                    spec=spec,
                    deliverable_id=deliverable.id,
                ):
                    source_path = workspace_starter_dir / relative_path
                    if not source_path.exists() or not source_path.is_file():
                        continue
                    role, audience, semantic_source = self._starter_file_metadata(
                        relative_path,
                        manifest_payload=workspace_manifest,
                    )
                    try:
                        content = source_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    self._write_text(
                        deliverable_dir / relative_path,
                        content,
                        ArtifactVisibility.public,
                        files,
                        bundle_root,
                        role=role,
                        audience=audience,
                        deliverable_id=deliverable.id,
                        semantic_source=semantic_source,
                    )
            else:
                starter_files = build_task_agent_starter_files(spec, deliverable.id)
                starter_manifest = json.loads(starter_files[HIDDEN_MANIFEST_PATH])
                for relative_path, content in starter_files.items():
                    role, audience, semantic_source = self._starter_file_metadata(
                        relative_path,
                        manifest_payload=starter_manifest,
                    )
                    self._write_text(
                        deliverable_dir / relative_path,
                        content,
                        ArtifactVisibility.public,
                        files,
                        bundle_root,
                        role=role,
                        audience=audience,
                        deliverable_id=deliverable.id,
                        semantic_source=semantic_source,
                    )
                self._write_visible_fixture_files(
                    spec=spec,
                    deliverable_dir=deliverable_dir,
                    deliverable_id=deliverable.id,
                    files=files,
                    bundle_root=bundle_root,
                )

    def _task_agent_readme(self, spec: TaskAgentServiceSpec) -> str:
        runtime_plan = spec.project_contract.runtime_plan
        stack_bits = [
            runtime_plan.implementation_language,
            runtime_plan.language_version,
            runtime_plan.application_framework,
            runtime_plan.framework_version,
        ]
        stack_summary = " ".join(bit for bit in stack_bits if bit) or "not specified"
        system_profile = ", ".join(f"`{label}`" for label in spec.capabilities.summary_labels())
        visible_fixtures = ", ".join(f"`{path}`" for path in spec.runtime_dependencies.visible_fixture_files) or "`none`"
        lines = [
            f"# {spec.title}",
            "",
            spec.summary,
            "",
            f"- Package type: `{spec.package_type.value}`",
            f"- Project family: `{spec.project_contract.family.value}`",
            f"- System kind: {spec.project_contract.system_kind}",
            f"- Runtime stack: {stack_summary}",
            f"- Execution surface: `{spec.runtime_dependencies.execution_surface.value}`",
            f"- System profile: {system_profile}",
            f"- Visible fixtures: {visible_fixtures}",
            "",
            "## Public service surface",
            "",
        ]
        lines.extend(f"- `{endpoint.method} {endpoint.path}`" for endpoint in spec.public_endpoints if endpoint.required)
        lines.extend(["", "## Runtime components", ""])
        for service in runtime_plan.services:
            technology = f" using `{service.technology}`" if service.technology else ""
            lines.append(f"- `{service.service_id}` ({service.role}){technology}")
        lines.extend(["", "## Deliverable arc", ""])
        for deliverable in spec.deliverables:
            lines.append(f"- `{deliverable.id}` - {deliverable.title}: {deliverable.objective}")
        return "\n".join(lines)

    def _course_outline(self, spec: TaskAgentServiceSpec) -> str:
        lines = ["# Course Outline", ""]
        for deliverable in spec.deliverables:
            gate = spec.gate_for(deliverable.id)
            lines.extend(
                [
                    f"## {deliverable.id}: {deliverable.title}",
                    "",
                    deliverable.objective,
                    "",
                    f"- Starter type: `{spec.runtime_dependencies.starter_type.value}`",
                    f"- Active visible checks: {', '.join(f'`{item}`' for item in gate.active_public_check_ids) or 'none'}",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    def _write_visible_fixture_files(
        self,
        *,
        spec: TaskAgentServiceSpec,
        deliverable_dir: Path,
        deliverable_id: str,
        files: list[BundleFile],
        bundle_root: Path,
    ) -> None:
        visible_paths = list(dict.fromkeys(spec.runtime_dependencies.visible_fixture_files))
        sources_by_path = {
            source.workspace_path: source
            for source in spec.runtime_dependencies.data_sources
            if source.learner_visible and source.workspace_path
        }
        for relative_path in visible_paths:
            if not relative_path:
                continue
            content = self._visible_fixture_content(relative_path, sources_by_path.get(relative_path))
            self._write_text(
                deliverable_dir / relative_path,
                content,
                ArtifactVisibility.public,
                files,
                bundle_root,
                role="visible_fixture",
                audience="learner",
                deliverable_id=deliverable_id,
                semantic_source="source_materialized",
            )

    def _visible_fixture_content(self, relative_path: str, source) -> str:
        if source is not None and source.asset_id and self.creator_asset_service is not None:
            try:
                _record, content = self.creator_asset_service.read_asset_text(source.asset_id)
                return content if content.endswith("\n") else content + "\n"
            except (FileNotFoundError, KeyError):
                pass
        description = (getattr(source, "description", None) or "Visible learner fixture.").strip()
        suffix = Path(relative_path).suffix.lower()
        if suffix == ".json":
            return json.dumps(
                {"title": getattr(source, "title", "Uploaded data source"), "description": description, "items": []},
                indent=2,
            ) + "\n"
        if suffix == ".csv":
            return "id,value\n"
        if suffix in {".md", ".markdown"}:
            return f"# {getattr(source, 'title', 'Uploaded data source')}\n\n{description}\n"
        return description + "\n"

    def _starter_readme(self, spec: TaskAgentServiceSpec, deliverable_id: str) -> str:
        deliverable = next(item for item in spec.deliverables if item.id == deliverable_id)
        brief = deliverable.learner_brief or build_task_agent_deliverable_brief(spec, deliverable)
        return render_learner_starter_readme(
            title=f"Starter for {deliverable.title}",
            brief=brief,
            summary=deliverable.objective,
            learning_outcomes=list(deliverable.learning_outcomes),
            visible_check_command=spec.runtime_dependencies.visible_check_command or "sh .coursegen/runtime/check_visible.sh",
            preview_command=spec.runtime_dependencies.preview_command or default_preview_command(spec, host="127.0.0.1"),
            public_checks=deliverable.public_checks,
        )

    def _assignment_runtime_dockerfile(self, spec: TaskAgentServiceSpec) -> str:
        bootstrap_commands = task_agent_runtime_bootstrap_commands(spec, include_python=True)
        environment_lines = task_agent_runtime_environment_lines(spec)
        lines = [
            f"FROM {task_agent_runtime_base_image(spec)}",
            "",
            "ENV PYTHONDONTWRITEBYTECODE=1",
            "ENV PYTHONUNBUFFERED=1",
            *environment_lines,
            "",
            "WORKDIR /workspace",
        ]
        if bootstrap_commands:
            lines.extend(["", "RUN " + " && \\\n    ".join(bootstrap_commands)])
        lines.extend(["", "COPY . /workspace", 'CMD ["python3", "runtime/verify_assignment.py"]', ""])
        return "\n".join(lines)

    def _assignment_runtime_readme(self) -> str:
        return "\n".join(
            [
                "# Assignment Runtime Sandbox",
                "",
                "This Docker image verifies that the generated assignment starters compile and boot before author review opens.",
                "",
            ]
        )

    def _assignment_runtime_verifier(self) -> str:
        return "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import json",
                "import os",
                "import signal",
                "import subprocess",
                "import time",
                "from pathlib import Path",
                "from urllib.error import URLError",
                "from urllib.request import Request, urlopen",
                "",
                'ROOT = Path(__file__).resolve().parents[1]',
                'STARTERS = ROOT / "starter"',
                'PORT = int(os.environ.get("ASSIGNMENT_SANDBOX_PORT", "8010"))',
                "",
                "",
                "def request_json(method: str, url: str, payload=None, timeout: float = 3.0):",
                "    data = None",
                "    headers = {}",
                "    if payload is not None:",
                "        data = json.dumps(payload).encode('utf-8')",
                "        headers['content-type'] = 'application/json'",
                "    request = Request(url, data=data, headers=headers, method=method)",
                "    with urlopen(request, timeout=timeout) as response:",
                "        body = response.read().decode('utf-8', errors='replace')",
                "        return response.status, json.loads(body) if body else {}, dict(response.headers)",
                "",
                "",
                "def wait_for_health(port: int, path: str, timeout_s: float = 12.0):",
                "    deadline = time.time() + timeout_s",
                "    last_error = None",
                "    while time.time() < deadline:",
                "        try:",
                '            status, payload, _headers = request_json("GET", f"http://127.0.0.1:{port}{path}")',
                "            return status, payload",
                "        except URLError as exc:",
                "            last_error = str(exc)",
                "            time.sleep(0.25)",
                "        except Exception as exc:",
                "            last_error = str(exc)",
                "            time.sleep(0.25)",
                "    raise RuntimeError(last_error or 'health check timed out')",
                "",
                "",
                "def terminate(proc: subprocess.Popen[str]):",
                "    if proc.poll() is not None:",
                "        return proc.communicate()",
                "    try:",
                "        os.killpg(proc.pid, signal.SIGTERM)",
                "        proc.wait(timeout=5)",
                "    except subprocess.TimeoutExpired:",
                "        os.killpg(proc.pid, signal.SIGKILL)",
                "        proc.wait(timeout=5)",
                "    return proc.communicate()",
                "",
                "",
                "def manifest(deliverable_dir: Path) -> dict[str, object]:",
                f"    manifest_path = deliverable_dir / '{HIDDEN_MANIFEST_PATH}'",
                "    return json.loads(manifest_path.read_text(encoding='utf-8'))",
                "",
                "",
                "def healthcheck_path(manifest_payload: dict[str, object]) -> str:",
                "    runtime_plan = manifest_payload.get('runtime_plan') or {}",
                "    services = runtime_plan.get('services') or []",
                "    for service in services:",
                "        if service.get('service_id') == 'app' and service.get('healthcheck_path'):",
                "            return str(service['healthcheck_path'])",
                "    return '/health'",
                "",
                "",
                "def runtime_script(deliverable_dir: Path, relative_path: str) -> Path:",
                "    return deliverable_dir / relative_path",
                "",
                "",
                "def preview_command(manifest_payload: dict[str, object]) -> str:",
                "    return str(manifest_payload.get('preview_command') or 'sh .coursegen/runtime/run.sh')",
                "",
                "",
                "def primary_check(manifest_payload: dict[str, object]) -> dict[str, object] | None:",
                "    checks = manifest_payload.get('public_checks') or []",
                "    return checks[0] if checks else None",
                "",
                "",
                "def verify_deliverable(deliverable_dir: Path, port: int):",
                "    report = {'deliverable_id': deliverable_dir.name, 'compile_succeeded': False, 'runtime_succeeded': False, 'health_status_code': None, 'stdout': '', 'stderr': '', 'error': None}",
                "    environment = os.environ.copy()",
                "    environment['PORT'] = str(port)",
                "    try:",
                "        manifest_payload = manifest(deliverable_dir)",
                f"        install_script = runtime_script(deliverable_dir, '{RUNTIME_INSTALL_SCRIPT_PATH}')",
                f"        verify_script = runtime_script(deliverable_dir, '{RUNTIME_VERIFY_SCRIPT_PATH}')",
                "        if not install_script.exists():",
                "            raise FileNotFoundError(f'missing runtime install script {install_script.relative_to(deliverable_dir)}')",
                "        if not verify_script.exists():",
                "            raise FileNotFoundError(f'missing runtime verify script {verify_script.relative_to(deliverable_dir)}')",
                f"        subprocess.run('sh {RUNTIME_INSTALL_SCRIPT_PATH}', cwd=deliverable_dir, env=environment, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)",
                f"        subprocess.run('sh {RUNTIME_VERIFY_SCRIPT_PATH}', cwd=deliverable_dir, env=environment, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)",
                "        report['compile_succeeded'] = True",
                "    except Exception as exc:",
                "        report['error'] = f'compile failed: {exc}'",
                "        return report",
                "    proc = subprocess.Popen(preview_command(manifest_payload), cwd=deliverable_dir, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, start_new_session=True)",
                "    try:",
                "        status_code, _payload = wait_for_health(port, healthcheck_path(manifest_payload))",
                "        report['runtime_succeeded'] = status_code == 200",
                "        report['health_status_code'] = status_code",
                "        check = primary_check(manifest_payload)",
                "        if check:",
                "            request_json(str(check.get('request_method') or 'POST').upper(), f\"http://127.0.0.1:{port}{check.get('request_path')}\", check.get('request_body') or None)",
                "    except Exception as exc:",
                "        report['error'] = f'runtime failed: {exc}'",
                "    finally:",
                "        stdout, stderr = terminate(proc)",
                "        report['stdout'] = stdout",
                "        report['stderr'] = stderr",
                "    return report",
                "",
                "",
                "def main():",
                "    deliverable_dirs = sorted(path for path in STARTERS.iterdir() if path.is_dir())",
                "    reports = [verify_deliverable(deliverable_dir, PORT + index) for index, deliverable_dir in enumerate(deliverable_dirs)]",
                "    success = all(item['compile_succeeded'] and item['runtime_succeeded'] for item in reports)",
                "    payload = {'success': success, 'deliverable_reports': reports, 'error': None if success else 'One or more generated starters failed sandbox verification.'}",
                "    print(json.dumps(payload))",
                "    raise SystemExit(0 if success else 1)",
                "",
                "",
                "if __name__ == '__main__':",
                "    main()",
                "",
            ]
        )

    def _workspace_starter_paths(
        self,
        *,
        workspace_starter_dir: Path,
        spec: TaskAgentServiceSpec,
        deliverable_id: str,
    ) -> list[str]:
        manifest_path = workspace_starter_dir / HIDDEN_MANIFEST_PATH
        manifest_payload: dict[str, Any] | None = None
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest_payload = None
        deliverable = next(
            (candidate for candidate in spec.deliverables if candidate.id == deliverable_id),
            None,
        )
        return starter_materialization_paths(
            manifest=manifest_payload,
            editable_paths=(
                learner_editable_paths_for_deliverable(spec, deliverable)
                if deliverable is not None and manifest_payload is None
                else None
            ),
            visible_fixture_paths=(
                list(spec.runtime_dependencies.visible_fixture_files)
                if manifest_payload is None
                else None
            ),
        )

    def _starter_file_metadata(
        self,
        relative_path: str,
        *,
        manifest_payload: dict[str, Any] | None,
    ) -> tuple[str, str, str]:
        dependency_paths = starter_contract_path_sets_for_manifest(manifest_payload)
        if relative_path == HIDDEN_MANIFEST_PATH:
            return "starter_manifest", "operator", "starter_compiler"
        if relative_path == "Dockerfile":
            return "starter_dockerfile", "operator", "starter_compiler"
        if relative_path == RUNTIME_INSTALL_SCRIPT_PATH:
            return "runtime_install_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_VERIFY_SCRIPT_PATH:
            return "runtime_verify_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_RUN_SCRIPT_PATH:
            return "runtime_run_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_VISIBLE_CHECK_SCRIPT_PATH:
            return "runtime_visible_check_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_HIDDEN_CHECK_SCRIPT_PATH:
            return "runtime_hidden_check_script", "operator", "starter_compiler"
        if relative_path == "checks/run_visible_checks.py":
            return "visible_check_runner", "learner", "starter_compiler"
        if relative_path == ".vscode/tasks.json":
            return "vscode_tasks", "learner", "starter_compiler"
        if relative_path in dependency_paths["manifests"]:
            return "starter_dependency_manifest", "learner", "starter_compiler"
        if relative_path in dependency_paths["lockfiles"]:
            return "starter_dependency_lockfile", "learner", "starter_compiler"
        if relative_path in dependency_paths["toolchains"]:
            return "starter_toolchain_config", "learner", "starter_compiler"
        if relative_path in dependency_paths["build_support"]:
            return "starter_build_support", "learner", "starter_compiler"
        return "starter_entrypoint", "learner", "starter_compiler"

    def _write_text(
        self,
        path: Path,
        content: str,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
        *,
        role: str | None = None,
        audience: str | None = None,
        deliverable_id: str | None = None,
        semantic_source: str | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        files.append(
            BundleFile(
                relative_path=str(path.relative_to(bundle_root)),
                visibility=visibility,
                media_type=self._guess_media_type(path.name),
                size_bytes=path.stat().st_size,
                role=role,
                audience=audience,
                deliverable_id=deliverable_id,
                semantic_source=semantic_source,
            )
        )

    def _write_json(
        self,
        path: Path,
        payload: Any,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
        *,
        role: str | None = None,
        audience: str | None = None,
        deliverable_id: str | None = None,
        semantic_source: str | None = None,
    ) -> None:
        self._write_text(
            path,
            json.dumps(payload, indent=2) + "\n",
            visibility,
            files,
            bundle_root,
            role=role,
            audience=audience,
            deliverable_id=deliverable_id,
            semantic_source=semantic_source,
        )

    def _guess_media_type(self, filename: str) -> str:
        if filename.endswith(".json"):
            return "application/json"
        if filename.endswith(".md"):
            return "text/markdown"
        if filename.endswith(".py"):
            return "text/x-python"
        if filename.endswith(".ts"):
            return "application/typescript"
        if filename.endswith(".js"):
            return "application/javascript"
        if filename.endswith(".txt"):
            return "text/plain"
        return "text/plain"
