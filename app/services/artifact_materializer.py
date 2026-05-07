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
from app.services.grader_planner import build_all_task_agent_grader_plans
from app.services.learner_brief_builder import (
    build_task_agent_module_brief,
    render_learner_module_markdown,
    render_learner_starter_readme,
)
from app.services.task_agent_starter_templates import (
    build_task_agent_starter_manifest,
    render_task_agent_module_app,
    render_task_agent_runtime_module,
    render_task_agent_visible_checks_script,
    render_task_agent_vscode_tasks,
)


def default_generated_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "generated"


class ArtifactMaterializer:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or default_generated_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

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

        bundle = MaterializedBundle(
            bundle_id=f"{run.id}_bundle",
            generated_at=generated_at,
            root_dir=str(bundle_root),
            public_dir=str(public_dir),
            private_dir=str(private_dir),
            manifest_path=str(manifest_path),
            files=files,
        )
        return bundle

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
        grader_plans = build_all_task_agent_grader_plans(spec)
        self._write_json(private_dir / "task_agent_spec.json", spec.model_dump(mode="json"), ArtifactVisibility.private, files, bundle_root)
        self._write_json(
            private_dir / "validation_summary.json",
            run.artifacts.validation_summary or {},
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        self._write_json(
            private_dir / "progression_preview.json",
            run.artifacts.progression_preview,
            ArtifactVisibility.private,
            files,
            bundle_root,
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
        )
        self._write_json(
            private_dir / "node_executions.json",
            [node.model_dump(mode="json") for node in run.artifacts.node_executions],
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        self._write_json(
            private_dir / "review_summary.json",
            run.artifacts.review_summary.model_dump(mode="json") if run.artifacts.review_summary is not None else {},
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        self._write_json(
            private_dir / "grader_bindings.json",
            {
                "behaviors": [
                    {
                        "id": behavior.id,
                        "description": behavior.description,
                        "first_required_in": behavior.first_required_in,
                        "test": behavior.test.model_dump(mode="json"),
                    }
                    for behavior in spec.behaviors
                ],
                "qualities": [
                    {
                        "id": quality.id,
                        "description": quality.description,
                        "first_required_in": quality.first_required_in,
                        "test": quality.test.model_dump(mode="json"),
                    }
                    for quality in spec.qualities
                ],
            },
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        self._write_json(
            private_dir / "grader_plans" / "index.json",
            grader_plans.model_dump(mode="json"),
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        for module_plan in grader_plans.module_plans:
            self._write_json(
                private_dir / "grader_plans" / f"{module_plan.module_id}.json",
                module_plan.model_dump(mode="json"),
                ArtifactVisibility.private,
                files,
                bundle_root,
            )

        self._write_text(
            public_dir / "README.md",
            self._task_agent_readme(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "runtime" / "Dockerfile",
            self._assignment_runtime_dockerfile(),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "runtime" / "requirements.txt",
            self._assignment_runtime_requirements(),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "runtime" / "__init__.py",
            '"""Shared runtime helpers for generated task-agent starters."""\n',
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "runtime" / "task_agent_runtime.py",
            render_task_agent_runtime_module(),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "runtime" / "verify_assignment.py",
            self._assignment_runtime_verifier(),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "runtime" / "README.md",
            self._assignment_runtime_readme(),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "content" / "course_outline.md",
            self._course_outline(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )

        for module in spec.modules:
            module_dir = public_dir / "starter" / module.id
            manifest_payload = build_task_agent_starter_manifest(spec, module.id)
            self._write_json(
                module_dir / "starter_manifest.json",
                manifest_payload,
                ArtifactVisibility.public,
                files,
                bundle_root,
            )
            self._write_text(
                public_dir / "content" / f"{module.id}.md",
                self._module_content(spec, module.id),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )
            plan = next(item for item in grader_plans.module_plans if item.module_id == module.id)
            self._write_text(
                public_dir / "content" / f"{module.id}_grading.md",
                self._module_grading_guide(plan),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )
            self._write_text(
                module_dir / "README.md",
                self._starter_readme(spec, module.id),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )
            self._write_text(
                module_dir / "app.py",
                render_task_agent_module_app(),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )
            self._write_text(
                module_dir / "checks" / "run_visible_checks.py",
                render_task_agent_visible_checks_script(),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )
            self._write_text(
                module_dir / ".vscode" / "tasks.json",
                render_task_agent_vscode_tasks(),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )

    def _task_agent_readme(self, spec: TaskAgentServiceSpec) -> str:
        tool_lines = "\n".join(
            f"- `{tool.id}` ({tool.safety.value}) - {tool.description}"
            for tool in spec.tool_registry.tools
        )
        system_profile = ", ".join(f"`{label}`" for label in spec.capabilities.summary_labels())
        visible_fixtures = ", ".join(f"`{path}`" for path in spec.runtime_dependencies.visible_fixture_files) or "`none`"
        return "\n".join(
            [
                f"# {spec.title}",
                "",
                spec.summary,
                "",
                f"- Package type: `{spec.package_type.value}`",
                f"- Domain pack: `{spec.domain_pack or 'generic'}`",
                f"- System profile: {system_profile}",
                f"- Execution surface: `{spec.runtime_dependencies.execution_surface.value}`",
                f"- Visible fixtures: {visible_fixtures}",
                "",
                "## What learners build",
                "",
                "A bounded, production-ready agentic system with stable APIs, tool-use policies, traces, approvals, and evaluation hooks.",
                "",
                "## Tooling surface",
                "",
                tool_lines,
                "",
                "## Production contract",
                "",
                f"- Async runs: `{spec.production_contract.supports_async_runs}`",
                f"- Resume support: `{spec.production_contract.supports_resume}`",
                f"- Dry-run support: `{spec.production_contract.supports_dry_run}`",
                f"- State backend target: `{spec.production_contract.state_backend}`",
                "",
                "## Docker verification",
                "",
                "The bundle ships with `public/runtime/Dockerfile` and `public/runtime/verify_assignment.py` so the backend can compile and boot every starter in an isolated sandbox before review.",
            ]
        ) + "\n"

    def _course_outline(self, spec: TaskAgentServiceSpec) -> str:
        lines = ["# Course Outline", ""]
        for module in spec.modules:
            gate = spec.gate_for(module.id)
            lines.extend(
                [
                    f"## {module.id}: {module.title}",
                    "",
                    module.objective,
                    "",
                    f"- Starter type: `{module.starter_type.value}`",
                    f"- Active behaviors: {', '.join(f'`{item}`' for item in gate.active_behavior_ids) or 'none'}",
                    f"- Active qualities: {', '.join(f'`{item}`' for item in gate.active_quality_ids) or 'none'}",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    def _module_content(self, spec: TaskAgentServiceSpec, module_id: str) -> str:
        module = next(item for item in spec.modules if item.id == module_id)
        brief = module.learner_brief or build_task_agent_module_brief(spec, module)
        return render_learner_module_markdown(
            module_index=spec.module_order[module.id] + 1,
            title=module.title,
            summary=module.objective,
            learning_outcomes=[],
            brief=brief,
            public_checks=module.public_checks,
        )

    def _module_grading_guide(self, plan) -> str:
        lines = [
            f"# Grading Guide: {plan.module_title}",
            "",
            plan.module_objective,
            "",
            f"- Total active tests: `{plan.total_tests}`",
            f"- Endpoints touched: {', '.join(f'`{item}`' for item in plan.endpoint_paths) or 'none'}",
            f"- Tools touched: {', '.join(f'`{item}`' for item in plan.tool_ids) or 'none'}",
            f"- Controls exercised: {', '.join(f'`{item.value}`' for item in plan.controls) or 'none'}",
            "",
            "## Active tests",
            "",
        ]
        for entry in plan.entries:
            lines.extend(
                [
                    f"### {entry.test_id}",
                    "",
                    f"- Kind: `{entry.kind.value}`",
                    f"- Dispatcher: `{entry.test_type}`",
                    f"- First required in: `{entry.first_required_in}`",
                    f"- Description: {entry.description}",
                ]
            )
            if entry.dependencies.eval_case_ids:
                lines.append(
                    f"- Eval cases: {', '.join(f'`{item}`' for item in entry.dependencies.eval_case_ids)}"
                )
            if entry.dependencies.dataset_id:
                lines.append(f"- Dataset: `{entry.dependencies.dataset_id}`")
            if entry.dependencies.tool_ids:
                lines.append(
                    f"- Tool refs: {', '.join(f'`{item}`' for item in entry.dependencies.tool_ids)}"
                )
            if entry.dependencies.required_events:
                lines.append(
                    f"- Trace events: {', '.join(f'`{item}`' for item in entry.dependencies.required_events)}"
                )
            if entry.dependencies.injected_failures:
                lines.append(
                    f"- Fault injections: {', '.join(f'`{item}`' for item in entry.dependencies.injected_failures)}"
                )
            lines.append("")
        return "\n".join(lines)

    def _starter_readme(self, spec: TaskAgentServiceSpec, module_id: str) -> str:
        module = next(item for item in spec.modules if item.id == module_id)
        brief = module.learner_brief or build_task_agent_module_brief(spec, module)
        return render_learner_starter_readme(
            title=f"Starter for {module.title}",
            brief=brief,
            public_checks=module.public_checks,
        )

    def _assignment_runtime_dockerfile(self) -> str:
        return "\n".join(
            [
                "FROM python:3.12-slim",
                "",
                "ENV PYTHONDONTWRITEBYTECODE=1",
                "ENV PYTHONUNBUFFERED=1",
                "",
                "WORKDIR /workspace",
                "COPY runtime/requirements.txt /tmp/requirements.txt",
                "RUN pip install --no-cache-dir -r /tmp/requirements.txt",
                "COPY . /workspace",
                'CMD ["python", "runtime/verify_assignment.py"]',
                "",
            ]
        )

    def _assignment_runtime_requirements(self) -> str:
        return "\n".join(
            [
                "fastapi>=0.136.1,<0.137.0",
                "uvicorn>=0.46.0,<0.47.0",
                "",
            ]
        )

    def _assignment_runtime_readme(self) -> str:
        return "\n".join(
            [
                "# Assignment Runtime Sandbox",
                "",
                "This Docker image verifies that the generated assignment starters compile and boot before author review opens.",
                "",
                "## Commands",
                "",
                "```bash",
                "docker build -f runtime/Dockerfile -t assignment-runtime .",
                "docker run --rm assignment-runtime",
                "```",
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
                "import py_compile",
                "import subprocess",
                "import sys",
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
                "        return response.status, json.loads(body) if body else {}",
                "",
                "",
                "def wait_for_health(port: int, timeout_s: float = 12.0):",
                "    deadline = time.time() + timeout_s",
                "    last_error = None",
                "    while time.time() < deadline:",
                "        try:",
                '            status, payload = request_json("GET", f"http://127.0.0.1:{port}/health")',
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
                "    proc.terminate()",
                "    try:",
                "        return proc.communicate(timeout=5)",
                "    except subprocess.TimeoutExpired:",
                "        proc.kill()",
                "        return proc.communicate(timeout=5)",
                "",
                "",
                "def verify_module(module_dir: Path, port: int):",
                '    app_path = module_dir / "app.py"',
                "    report = {",
                '        "module_id": module_dir.name,',
                '        "compile_succeeded": False,',
                '        "runtime_succeeded": False,',
                '        "health_status_code": None,',
                '        "stdout": "",',
                '        "stderr": "",',
                '        "error": None,',
                "    }",
                "    manifest_path = module_dir / 'starter_manifest.json'",
                "    try:",
                "        py_compile.compile(str(app_path), doraise=True)",
                '        report["compile_succeeded"] = True',
                "    except Exception as exc:",
                '        report["error"] = f"compile failed: {exc}"',
                "        return report",
                "",
                "    proc = subprocess.Popen(",
                "        [sys.executable, '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', str(port), '--log-level', 'warning'],",
                "        cwd=module_dir,",
                "        stdout=subprocess.PIPE,",
                "        stderr=subprocess.PIPE,",
                "        text=True,",
                "    )",
                "    try:",
                "        status_code, _payload = wait_for_health(port)",
                '        report["runtime_succeeded"] = status_code == 200',
                '        report["health_status_code"] = status_code',
                "        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))",
                "        sample_input = manifest.get('sample_input') or {}",
                "        run_status, run_payload = request_json('POST', f'http://127.0.0.1:{port}/run', sample_input)",
                "        if run_status not in (200, 201, 202):",
                '            raise RuntimeError(f"/run returned unexpected status {run_status}")',
                "        run_id = run_payload.get('run_id')",
                "        if not run_id:",
                "            raise RuntimeError('/run did not return a run_id')",
                "        runs_status, _runs_payload = request_json('GET', f'http://127.0.0.1:{port}/runs/{run_id}')",
                "        if runs_status != 200:",
                '            raise RuntimeError(f"/runs/{{run_id}} returned unexpected status {runs_status}")',
                "        trace_status, _trace_payload = request_json('GET', f'http://127.0.0.1:{port}/trace/{run_id}')",
                "        if trace_status != 200:",
                '            raise RuntimeError(f"/trace/{{run_id}} returned unexpected status {trace_status}")',
                "        if run_payload.get('status') == 'awaiting_approval':",
                "            approve_status, _approve_payload = request_json(",
                "                'POST',",
                "                f'http://127.0.0.1:{port}/approve/{run_id}',",
                "                {'note': 'sandbox approval'},",
                "            )",
                "            if approve_status != 200:",
                '                raise RuntimeError(f"/approve/{{run_id}} returned unexpected status {approve_status}")',
                "        eval_status, _eval_payload = request_json('POST', f'http://127.0.0.1:{port}/eval', {'source': 'sandbox'})",
                "        if eval_status != 200:",
                '            raise RuntimeError(f"/eval returned unexpected status {eval_status}")',
                "    except Exception as exc:",
                '        report["error"] = f"runtime failed: {exc}"',
                "    finally:",
                "        stdout, stderr = terminate(proc)",
                '        report["stdout"] = stdout',
                '        report["stderr"] = stderr',
                "    return report",
                "",
                "",
                "def main():",
                "    module_dirs = sorted(path for path in STARTERS.iterdir() if path.is_dir())",
                "    reports = []",
                "    for index, module_dir in enumerate(module_dirs):",
                "        reports.append(verify_module(module_dir, PORT + index))",
                '    success = all(item["compile_succeeded"] and item["runtime_succeeded"] for item in reports)',
                "    payload = {",
                '        "success": success,',
                '        "module_reports": reports,',
                '        "error": None if success else "One or more generated starters failed sandbox verification.",',
                "    }",
                "    print(json.dumps(payload))",
                "    raise SystemExit(0 if success else 1)",
                "",
                "",
                "if __name__ == '__main__':",
                "    main()",
                "",
            ]
        )

    def _write_text(
        self,
        path: Path,
        content: str,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        files.append(
            BundleFile(
                relative_path=str(path.relative_to(bundle_root)),
                visibility=visibility,
                media_type=self._guess_media_type(path.name),
                size_bytes=path.stat().st_size,
            )
        )

    def _write_json(
        self,
        path: Path,
        payload: Any,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
    ) -> None:
        self._write_text(path, json.dumps(payload, indent=2) + "\n", visibility, files, bundle_root)

    def _guess_media_type(self, filename: str) -> str:
        if filename.endswith(".json"):
            return "application/json"
        if filename.endswith(".md"):
            return "text/markdown"
        if filename.endswith(".py"):
            return "text/x-python"
        return "text/plain"
