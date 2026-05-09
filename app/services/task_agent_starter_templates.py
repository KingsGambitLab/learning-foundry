from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.domain.task_agent import TaskAgentServiceSpec
from app.services.task_agent_contract_surface import primary_submit_endpoint_for_spec

HIDDEN_MANIFEST_PATH = ".coursegen/deliverable.json"
PREVIEW_LAUNCHER_PATH = ".coursegen/preview_app.py"
HIDDEN_GRADER_SCRIPT_PATH = ".coursegen/grader/run_hidden_checks.py"


def _runtime_language(spec: TaskAgentServiceSpec) -> str:
    runtime_plan = spec.project_contract.runtime_plan
    return (
        runtime_plan.implementation_language
        or spec.runtime_dependencies.implementation_language
        or "python"
    ).strip().lower()


def _runtime_framework(spec: TaskAgentServiceSpec) -> str:
    runtime_plan = spec.project_contract.runtime_plan
    return (
        runtime_plan.application_framework
        or spec.runtime_dependencies.application_framework
        or ("fastapi" if _runtime_language(spec) == "python" else "express")
    ).strip().lower()


def _runtime_package_manager(spec: TaskAgentServiceSpec) -> str:
    runtime_plan = spec.project_contract.runtime_plan
    return (runtime_plan.package_manager or "npm").strip().lower()


def _runtime_app_service(spec: TaskAgentServiceSpec):
    for service in spec.project_contract.runtime_plan.services:
        if service.service_id == "app":
            return service
    return None


def task_agent_runtime_base_image(spec: TaskAgentServiceSpec) -> str:
    app_service = _runtime_app_service(spec)
    if app_service is not None and app_service.container_image:
        return app_service.container_image
    language = _runtime_language(spec)
    if language == "python":
        return "python:3.12-slim"
    if language in {"typescript", "javascript"}:
        return "node:22-bookworm-slim"
    if language == "go":
        return "golang:1.23"
    if language == "rust":
        return "rust:1.82-bookworm"
    return "python:3.12-slim"


def task_agent_runtime_bootstrap_commands(
    spec: TaskAgentServiceSpec,
    *,
    include_python: bool = False,
) -> list[str]:
    language = _runtime_language(spec)
    package_manager = _runtime_package_manager(spec)
    commands: list[str] = []
    if language in {"typescript", "javascript"} and package_manager in {"pnpm", "yarn"}:
        commands.append("corepack enable")
    if include_python and language != "python":
        commands.extend(
            [
                "apt-get update",
                "apt-get install -y --no-install-recommends python3",
                "rm -rf /var/lib/apt/lists/*",
            ]
        )
    return commands


def task_agent_runtime_environment_lines(spec: TaskAgentServiceSpec) -> list[str]:
    language = _runtime_language(spec)
    package_manager = _runtime_package_manager(spec)
    if language in {"typescript", "javascript"} and package_manager in {"pnpm", "yarn"}:
        return ["ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0"]
    return []


def task_agent_entrypoint_path(spec: TaskAgentServiceSpec) -> str:
    app_service = _runtime_app_service(spec)
    if app_service is not None and app_service.entrypoint_path:
        return app_service.entrypoint_path
    language = _runtime_language(spec)
    framework = _runtime_framework(spec)
    if language == "python":
        return "app.py"
    if framework == "nestjs":
        return "src/main.ts"
    if language == "typescript":
        return "src/main.ts"
    if language == "javascript":
        return "src/main.js"
    return "app.py"


def task_agent_starter_relative_paths(spec: TaskAgentServiceSpec) -> list[str]:
    paths = [
        HIDDEN_MANIFEST_PATH,
        "Dockerfile",
        task_agent_entrypoint_path(spec),
        "checks/run_visible_checks.py",
        HIDDEN_GRADER_SCRIPT_PATH,
        ".vscode/tasks.json",
    ]
    language = _runtime_language(spec)
    if language == "python":
        paths.append("requirements.txt")
        paths.append(PREVIEW_LAUNCHER_PATH)
    if language in {"typescript", "javascript"}:
        paths.append("package.json")
        if language == "typescript":
            paths.append("tsconfig.json")
    return paths


def build_task_agent_starter_files(spec: TaskAgentServiceSpec, deliverable_id: str) -> dict[str, str]:
    deliverable = next(item for item in spec.deliverables if item.id == deliverable_id)
    starter_surface = deliverable.learner_starter_surface
    manifest_payload = {
        "title": spec.title,
        "summary": spec.summary,
        "deliverable_id": deliverable.id,
        "deliverable_title": deliverable.title,
        "deliverable_objective": deliverable.objective,
        "runtime_plan": spec.project_contract.runtime_plan.model_dump(mode="json"),
        "runtime_dependencies": spec.runtime_dependencies.model_dump(mode="json"),
        "public_endpoints": [endpoint.model_dump(mode="json") for endpoint in spec.public_endpoints],
        "public_checks": [check.model_dump(mode="json") for check in deliverable.public_checks],
        "visible_check_command": spec.runtime_dependencies.visible_check_command or "python checks/run_visible_checks.py",
        "hidden_check_command": "python .coursegen/grader/run_hidden_checks.py",
        "preview_command": spec.runtime_dependencies.preview_command or default_preview_command(spec),
        "entrypoint_path": task_agent_entrypoint_path(spec),
        "generated_test_scripts": {
            "source": "starter_default",
            "generated_for_deliverable": deliverable.id,
        },
        "learner_starter_surface": (
            starter_surface.model_dump(mode="json")
            if starter_surface is not None
            else None
        ),
    }
    files: dict[str, str] = {
        HIDDEN_MANIFEST_PATH: json.dumps(manifest_payload, indent=2) + "\n",
        "Dockerfile": render_task_agent_starter_dockerfile(spec),
        task_agent_entrypoint_path(spec): render_task_agent_runtime_entrypoint(spec, for_root_workspace=False),
        "checks/run_visible_checks.py": render_task_agent_visible_checks_script(),
        HIDDEN_GRADER_SCRIPT_PATH: render_task_agent_hidden_checks_script(),
        ".vscode/tasks.json": render_task_agent_vscode_tasks(),
    }
    language = _runtime_language(spec)
    if language == "python":
        files["requirements.txt"] = render_task_agent_python_requirements(spec)
        files[PREVIEW_LAUNCHER_PATH] = render_task_agent_python_preview_launcher()
    if language in {"typescript", "javascript"}:
        files["package.json"] = render_task_agent_node_package_json(spec)
        if language == "typescript":
            files["tsconfig.json"] = render_task_agent_tsconfig(spec)
    return files


def default_preview_command(spec: TaskAgentServiceSpec, *, host: str = "0.0.0.0") -> str:
    language = _runtime_language(spec)
    framework = _runtime_framework(spec)
    entrypoint = task_agent_entrypoint_path(spec)
    if language == "python":
        return f"python {PREVIEW_LAUNCHER_PATH} --host {host}"
    if framework == "nestjs":
        return "pnpm start:dev" if _runtime_package_manager(spec) == "pnpm" else "npm run start:dev"
    if language == "typescript":
        return f"{'pnpm' if _runtime_package_manager(spec) == 'pnpm' else 'npm run'} start:dev"
    if language == "javascript":
        return f"node {entrypoint}"
    return f"python {PREVIEW_LAUNCHER_PATH} --host {host}"


def render_task_agent_python_requirements(spec: TaskAgentServiceSpec) -> str:
    return "\n".join(
        [
            "fastapi>=0.116.0,<1.0.0",
            "uvicorn[standard]>=0.35.0,<1.0.0",
            "",
        ]
    )


def render_task_agent_python_preview_launcher() -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import argparse",
            "import importlib.util",
            "import json",
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "import uvicorn",
            "",
            "ROOT = Path(__file__).resolve().parents[1]",
            f"MANIFEST_PATH = ROOT / '{HIDDEN_MANIFEST_PATH}'",
            "",
            "",
            "def _entrypoint_path() -> Path:",
            "    payload = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))",
            "    relative_path = str(payload.get('entrypoint_path') or 'app.py')",
            "    return (ROOT / relative_path).resolve()",
            "",
            "",
            "def _load_app(entrypoint_path: Path):",
            "    sys.path.insert(0, str(ROOT))",
            "    if str(entrypoint_path.parent) != str(ROOT):",
            "        sys.path.insert(0, str(entrypoint_path.parent))",
            "    spec = importlib.util.spec_from_file_location('coursegen_student_app', entrypoint_path)",
            "    if spec is None or spec.loader is None:",
            "        raise SystemExit(f'Could not load preview entrypoint from {entrypoint_path}')",
            "    module = importlib.util.module_from_spec(spec)",
            "    spec.loader.exec_module(module)",
            "    app = getattr(module, 'app', None)",
            "    if app is None:",
            "        raise SystemExit(f'{entrypoint_path.name} must define a global `app` object for preview.')",
            "    return app",
            "",
            "",
            "def main() -> None:",
            "    parser = argparse.ArgumentParser(description='Run the learner preview app from the local starter workspace.')",
            "    parser.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'))",
            "    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8000')))",
            "    args = parser.parse_args()",
            "    entrypoint_path = _entrypoint_path()",
            "    app = _load_app(entrypoint_path)",
            "    uvicorn.run(app, host=args.host, port=args.port)",
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )


def render_task_agent_starter_dockerfile(spec: TaskAgentServiceSpec) -> str:
    bootstrap_commands = task_agent_runtime_bootstrap_commands(spec)
    environment_lines = task_agent_runtime_environment_lines(spec)
    language = _runtime_language(spec)
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
    if language == "python":
        lines.extend(
            [
                "",
                "COPY requirements.txt /workspace/requirements.txt",
                "RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt",
            ]
        )
    elif language in {"typescript", "javascript"}:
        lines.extend(
            [
                "",
                "COPY package.json /workspace/package.json",
                "RUN "
                + (
                    "pnpm install --no-frozen-lockfile"
                    if _runtime_package_manager(spec) == "pnpm"
                    else "npm install"
                ),
            ]
        )
    lines.extend(["", "COPY . /workspace", ""])
    return "\n".join(lines)


def render_task_agent_runtime_deliverable() -> str:
    return "# Runtime helper deleted in favor of real learner-owned entrypoints.\n"


def render_legacy_task_agent_root_app() -> str:
    return render_task_agent_root_app()


def render_task_agent_root_app() -> str:
    return render_task_agent_python_entrypoint()


def render_task_agent_deliverable_app() -> str:
    return render_task_agent_python_entrypoint()


def render_task_agent_python_entrypoint() -> str:
    return render_task_agent_python_starter_entrypoint(None)


def _route_stub_response(method: str, path: str) -> str:
    return (
        "{\n"
        f'    "status": "starter",\n'
        f'    "summary": "Implement {method} {path} in learner-owned code."\n'
        "}"
    )


def render_task_agent_python_starter_entrypoint(spec: TaskAgentServiceSpec | None) -> str:
    endpoints = spec.public_endpoints if spec is not None else [
        type("Endpoint", (), {"method": "GET", "path": "/health", "required": True})()
    ]
    route_blocks: list[str] = []
    for endpoint in endpoints:
        if endpoint.path == "/health":
            route_blocks.append(
                "\n\n@app.get('/health')\ndef health() -> dict[str, str]:\n"
                "    return {'status': 'ok'}\n"
            )
            continue
        decorator = endpoint.method.lower()
        fn_name = _python_function_name(endpoint.method, endpoint.path)
        params = "payload: dict[str, Any] | None = None"
        route_blocks.append(
            f"\n\n@app.{decorator}('{endpoint.path}')\n"
            f"def {fn_name}({params}) -> dict[str, Any]:\n"
            f"    return {_route_stub_response(endpoint.method, endpoint.path)}\n"
        )
    return (
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "from fastapi import FastAPI\n\n"
        "app = FastAPI(title='CourseGen Starter')\n"
        + "".join(route_blocks)
        + "\n"
    )


def _python_function_name(method: str, path: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", path.strip("/").replace("/", "_"))
    cleaned = cleaned.replace("{", "").replace("}", "").strip("_") or "root"
    if cleaned[0].isdigit():
        cleaned = f"route_{cleaned}"
    return f"{method.lower()}_{cleaned}"


def _task_agent_node_runtime_helpers() -> str:
    return ""


def render_task_agent_node_express_entrypoint(*, language: str) -> str:
    ext = "ts" if language == "typescript" else "js"
    if language == "typescript":
        return (
            "import express from 'express';\n\n"
            "const app = express();\n"
            "app.use(express.json());\n"
            "app.get('/health', (_req, res) => res.json({ status: 'ok' }));\n"
            "app.post('/service', (_req, res) => res.status(200).json({ status: 'starter', summary: 'Implement the service route in learner-owned code.' }));\n"
            "const port = Number(process.env.PORT || 8000);\n"
            "app.listen(port, '0.0.0.0', () => console.log(`starter listening on ${port}`));\n"
        )
    return (
        "const express = require('express');\n\n"
        "const app = express();\n"
        "app.use(express.json());\n"
        "app.get('/health', (_req, res) => res.json({ status: 'ok' }));\n"
        "app.post('/service', (_req, res) => res.status(200).json({ status: 'starter', summary: 'Implement the service route in learner-owned code.' }));\n"
        "const port = Number(process.env.PORT || 8000);\n"
        "app.listen(port, '0.0.0.0', () => console.log(`starter listening on ${port}`));\n"
    )


def render_task_agent_node_nest_entrypoint(spec: TaskAgentServiceSpec) -> str:
    route_blocks: list[str] = []
    for endpoint in spec.public_endpoints:
        method_name = _typescript_function_name(endpoint.method, endpoint.path)
        decorator = endpoint.method.title().lower().capitalize()
        route_path = endpoint.path.replace("{", ":").replace("}", "")
        if endpoint.path == "/health":
            route_blocks.append(
                "  @Get('/health')\n"
                "  health() {\n"
                "    return { status: 'ok' };\n"
                "  }\n"
            )
            continue
        params = ""
        if endpoint.method in {"POST", "PUT", "PATCH"}:
            params = "@Body() _payload: Record<string, unknown>"
        route_blocks.append(
            f"  @{decorator}('{route_path}')\n"
            f"  {method_name}({params}) {{\n"
            f"    return {{ status: 'starter', summary: 'Implement {endpoint.method} {endpoint.path} in learner-owned code.' }};\n"
            "  }\n"
        )
    return (
        "import { Body, Controller, Delete, Get, Module, Patch, Post, Put } from '@nestjs/common';\n"
        "import { NestFactory } from '@nestjs/core';\n\n"
        "@Controller()\n"
        "class AppController {\n"
        + "\n".join(route_blocks)
        + "\n}\n\n"
        "@Module({ controllers: [AppController] })\n"
        "class AppModule {}\n\n"
        "async function bootstrap() {\n"
        "  const app = await NestFactory.create(AppModule);\n"
        "  await app.listen(Number(process.env.PORT || 8000), '0.0.0.0');\n"
        "}\n"
        "void bootstrap();\n"
    )


def render_task_agent_node_package_json(spec: TaskAgentServiceSpec) -> str:
    framework = _runtime_framework(spec)
    package_manager = _runtime_package_manager(spec)
    use_pnpm = package_manager == "pnpm"
    if framework == "nestjs":
        payload = {
            "name": "coursegen-starter",
            "private": True,
            "scripts": {
                "start:dev": "tsx watch src/main.ts",
                "check": "python checks/run_visible_checks.py",
            },
            "dependencies": {
                "@nestjs/common": "^11.0.0",
                "@nestjs/core": "^11.0.0",
                "reflect-metadata": "^0.2.2",
                "rxjs": "^7.8.1",
            },
            "devDependencies": {"tsx": "^4.19.0", "typescript": "^5.8.0"},
            "packageManager": "pnpm@10.11.0" if use_pnpm else None,
        }
    else:
        payload = {
            "name": "coursegen-starter",
            "private": True,
            "scripts": {
                "start:dev": "tsx watch src/main.ts" if _runtime_language(spec) == "typescript" else f"node {task_agent_entrypoint_path(spec)}",
                "check": "python checks/run_visible_checks.py",
            },
            "dependencies": {"express": "^5.1.0"},
            "devDependencies": {"tsx": "^4.19.0", "typescript": "^5.8.0"} if _runtime_language(spec) == "typescript" else {},
            "packageManager": "pnpm@10.11.0" if use_pnpm else None,
        }
    payload = {key: value for key, value in payload.items() if value is not None}
    return json.dumps(payload, indent=2) + "\n"


def render_task_agent_tsconfig(spec: TaskAgentServiceSpec) -> str:
    payload = {
        "compilerOptions": {
            "target": "ES2022",
            "module": "NodeNext" if _runtime_framework(spec) == "nestjs" else "ESNext",
            "moduleResolution": "NodeNext" if _runtime_framework(spec) == "nestjs" else "Bundler",
            "esModuleInterop": True,
            "strict": True,
            "skipLibCheck": True,
            "outDir": "dist",
        },
        "include": ["src/**/*.ts"],
    }
    return json.dumps(payload, indent=2) + "\n"


def render_task_agent_runtime_entrypoint(
    spec: TaskAgentServiceSpec,
    *,
    for_root_workspace: bool = False,
) -> str:
    language = _runtime_language(spec)
    framework = _runtime_framework(spec)
    if language == "python":
        return render_task_agent_python_starter_entrypoint(spec)
    if framework == "nestjs":
        return render_task_agent_node_nest_starter_entrypoint(spec)
    if language in {"typescript", "javascript"}:
        return render_task_agent_node_express_starter_entrypoint(spec=spec, language=language)
    return render_task_agent_python_starter_entrypoint(spec)


def render_task_agent_node_express_starter_entrypoint(*, spec: TaskAgentServiceSpec, language: str) -> str:
    body_annotation = ": any" if language == "typescript" else ""
    import_line = "import express from 'express';" if language == "typescript" else "const express = require('express');"
    route_blocks: list[str] = []
    for endpoint in spec.public_endpoints:
        if endpoint.path == "/health":
            route_blocks.append("app.get('/health', (_req, res) => res.json({ status: 'ok' }));")
            continue
        handler = (
            "(_req, res) => res.status(200).json({ status: 'starter', summary: 'Implement this route in learner-owned code.' })"
        )
        route_blocks.append(f"app.{endpoint.method.lower()}('{endpoint.path}', {handler});")
    return "\n".join(
        [
            import_line,
            "",
            "const app = express();",
            "app.use(express.json());",
            *route_blocks,
            "const port = Number(process.env.PORT || 8000);",
            "app.listen(port, '0.0.0.0', () => console.log(`starter listening on ${port}`));",
            "",
        ]
    )


def render_task_agent_node_nest_starter_entrypoint(spec: TaskAgentServiceSpec) -> str:
    return render_task_agent_node_nest_entrypoint(spec)


def _typescript_function_name(method: str, path: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", path.strip("/").replace("/", "_"))
    cleaned = cleaned.replace("{", "").replace("}", "").strip("_") or "root"
    if cleaned[0].isdigit():
        cleaned = f"route_{cleaned}"
    return f"{method.lower()}_{cleaned}"


def render_task_agent_visible_checks_script() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from urllib.error import HTTPError, URLError\n"
        "from urllib.request import Request, urlopen\n\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        f"MANIFEST_PATH = ROOT / '{HIDDEN_MANIFEST_PATH}'\n\n"
        "BASE_URL = os.environ.get('BASE_URL', 'http://127.0.0.1:8000').rstrip('/')\n"
        "REPORT_PATH = os.environ.get('REPORT_PATH')\n\n"
        "def request(method: str, path: str, body=None):\n"
        "    data = None\n"
        "    headers = {}\n"
        "    if body is not None:\n"
        "        data = json.dumps(body).encode('utf-8')\n"
        "        headers['content-type'] = 'application/json'\n"
        "    req = Request(f\"{BASE_URL}{path}\", data=data, headers=headers, method=method)\n"
        "    with urlopen(req, timeout=3.0) as response:\n"
        "        text = response.read().decode('utf-8', errors='replace')\n"
        "        try:\n"
        "            payload = json.loads(text) if text else {}\n"
        "        except json.JSONDecodeError:\n"
        "            payload = {'raw': text}\n"
        "        return response.status, payload\n\n"
        "def emit(report: dict[str, object], exit_code: int) -> None:\n"
        "    payload = json.dumps(report, indent=2)\n"
        "    if REPORT_PATH:\n"
        "        Path(REPORT_PATH).write_text(payload, encoding='utf-8')\n"
        "    else:\n"
        "        print(payload)\n"
        "    raise SystemExit(exit_code)\n\n"
        "def main():\n"
        "    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))\n"
        "    cases = []\n"
        "    health_status, _payload = request('GET', '/health')\n"
        "    cases.append({\n"
        "        'id': 'health',\n"
        "        'title': 'Health endpoint stays up',\n"
        "        'status': 'passed' if health_status == 200 else 'failed',\n"
        "        'summary': 'Health endpoint responded.' if health_status == 200 else f'/health returned {health_status}',\n"
        "        'diagnostics': [] if health_status == 200 else [f'/health returned {health_status}'],\n"
        "    })\n"
        "    for check in manifest.get('public_checks') or []:\n"
        "        method = str(check.get('request_method') or 'POST').upper()\n"
        "        path = str(check.get('request_path') or '')\n"
        "        title = str(check.get('title') or path or 'visible check')\n"
        "        if not path.startswith('/'):\n"
        "            cases.append({'id': str(check.get('id') or title), 'title': title, 'status': 'failed', 'summary': 'Invalid request path.', 'diagnostics': ['invalid request path']})\n"
        "            continue\n"
        "        try:\n"
        "            status, payload = request(method, path, check.get('request_body') or None)\n"
        "        except (HTTPError, URLError) as exc:\n"
        "            cases.append({'id': str(check.get('id') or title), 'title': title, 'status': 'failed', 'summary': 'Request failed.', 'diagnostics': [str(exc)]})\n"
        "            continue\n"
        "        expected_status = int(check.get('expected_status') or 200)\n"
        "        diagnostics = []\n"
        "        if status != expected_status:\n"
        "            diagnostics.append(f'expected HTTP {expected_status}, got {status}')\n"
        "        haystack = json.dumps(payload, sort_keys=True).lower()\n"
        "        for token in check.get('expected_response_contains') or []:\n"
        "            if str(token).lower() not in haystack:\n"
        "                diagnostics.append(f\"response did not mention {token!r}\")\n"
        "        cases.append({\n"
        "            'id': str(check.get('id') or title),\n"
        "            'title': title,\n"
        "            'status': 'passed' if not diagnostics else 'failed',\n"
        "            'summary': 'Visible check passed.' if not diagnostics else 'Visible check failed.',\n"
        "            'diagnostics': diagnostics,\n"
        "        })\n"
        "    failed = [case for case in cases if case['status'] != 'passed']\n"
        "    emit({'summary': 'Visible checks passed.' if not failed else 'Visible checks failed.', 'tests': cases}, 0 if not failed else 1)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def render_task_agent_hidden_checks_script() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parents[2]\n"
        "REPORT_PATH = os.environ.get('REPORT_PATH')\n\n"
        "def emit(report: dict[str, object], exit_code: int) -> None:\n"
        "    payload = json.dumps(report, indent=2)\n"
        "    if REPORT_PATH:\n"
        "        Path(REPORT_PATH).write_text(payload, encoding='utf-8')\n"
        "    else:\n"
        "        print(payload)\n"
        "    raise SystemExit(exit_code)\n\n"
        "def main() -> None:\n"
        "    emit(\n"
        "        {\n"
        "            'summary': 'Hidden tests have not been authored for this starter yet.',\n"
        "            'tests': [\n"
        "                {\n"
        "                    'id': 'hidden_tests_not_authored',\n"
        "                    'title': 'Hidden tests not authored',\n"
        "                    'status': 'failed',\n"
        "                    'summary': 'Author a real hidden grader before using this starter.',\n"
        "                    'diagnostics': ['The default hidden test script is only a placeholder.'],\n"
        "                }\n"
        "            ],\n"
        "        },\n"
        "        1,\n"
        "    )\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def render_task_agent_vscode_tasks() -> str:
    payload = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Visible checks",
                "type": "shell",
                "command": "python checks/run_visible_checks.py",
                "problemMatcher": [],
            },
            {
                "label": "Preview app",
                "type": "shell",
                "command": (
                    "python -c \"import json, os, subprocess; from pathlib import Path; "
                    f"manifest = json.loads(Path('{HIDDEN_MANIFEST_PATH}').read_text()); "
                    "env = os.environ.copy(); env.setdefault('PORT', '8000'); "
                    f"command = manifest.get('preview_command') or 'python {PREVIEW_LAUNCHER_PATH} --host 127.0.0.1'; "
                    "raise SystemExit(subprocess.run(command, shell=True, env=env).returncode)\""
                ),
                "problemMatcher": [],
            },
        ],
    }
    return json.dumps(payload, indent=2) + "\n"
