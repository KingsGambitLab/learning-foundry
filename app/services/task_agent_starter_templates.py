from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.task_agent import TaskAgentServiceSpec
from app.services.task_agent_contract_surface import primary_submit_endpoint_for_spec

HIDDEN_MANIFEST_PATH = ".coursegen/deliverable.json"
HIDDEN_GRADER_SCRIPT_PATH = ".coursegen/grader/run_hidden_checks.py"
RUNTIME_INSTALL_SCRIPT_PATH = ".coursegen/runtime/install.sh"
RUNTIME_VERIFY_SCRIPT_PATH = ".coursegen/runtime/verify.sh"
RUNTIME_RUN_SCRIPT_PATH = ".coursegen/runtime/run.sh"
RUNTIME_VISIBLE_CHECK_SCRIPT_PATH = ".coursegen/runtime/check_visible.sh"
RUNTIME_HIDDEN_CHECK_SCRIPT_PATH = ".coursegen/runtime/check_hidden.sh"
RUNTIME_PROTOCOL_PATHS = (
    "Dockerfile",
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
)


def _runtime_language(spec: TaskAgentServiceSpec) -> str:
    runtime_plan = spec.project_contract.runtime_plan
    return (
        runtime_plan.implementation_language
        or spec.runtime_dependencies.implementation_language
        or ""
    ).strip().lower()

def _runtime_package_manager(spec: TaskAgentServiceSpec) -> str:
    runtime_plan = spec.project_contract.runtime_plan
    return (
        runtime_plan.package_manager
        or spec.runtime_dependencies.package_manager
        or ""
    ).strip().lower()


def _runtime_app_service(spec: TaskAgentServiceSpec):
    for service in spec.project_contract.runtime_plan.services:
        if service.service_id == "app":
            return service
    return None


def task_agent_runtime_base_image(spec: TaskAgentServiceSpec) -> str:
    app_service = _runtime_app_service(spec)
    if app_service is not None and app_service.container_image:
        return app_service.container_image
    return "debian:bookworm-slim"


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
    if spec.runtime_dependencies.editable_files:
        return spec.runtime_dependencies.editable_files[0]
    return "main"


def task_agent_starter_relative_paths(spec: TaskAgentServiceSpec) -> list[str]:
    paths = [
        HIDDEN_MANIFEST_PATH,
        "Dockerfile",
        RUNTIME_INSTALL_SCRIPT_PATH,
        RUNTIME_VERIFY_SCRIPT_PATH,
        RUNTIME_RUN_SCRIPT_PATH,
        RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
        RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
        "checks/run_visible_checks.py",
        HIDDEN_GRADER_SCRIPT_PATH,
        ".vscode/tasks.json",
        *list(spec.runtime_dependencies.visible_fixture_files),
    ]
    return list(dict.fromkeys(path for path in paths if path))


def build_task_agent_starter_files(
    spec: TaskAgentServiceSpec,
    deliverable_id: str,
    *,
    authored_files: dict[str, str] | None = None,
) -> dict[str, str]:
    deliverable = next(item for item in spec.deliverables if item.id == deliverable_id)
    starter_surface = deliverable.learner_starter_surface
    authored_bundle_files = {
        path: content
        for path, content in (authored_files or {}).items()
        if path
        and not path.startswith("checks/")
        and path != "README.md"
        and path != HIDDEN_MANIFEST_PATH
        and path != HIDDEN_GRADER_SCRIPT_PATH
        and path != RUNTIME_VISIBLE_CHECK_SCRIPT_PATH
        and path != RUNTIME_HIDDEN_CHECK_SCRIPT_PATH
        and path != ".vscode/tasks.json"
    }
    runtime_protocol_files = {
        path: content
        for path, content in authored_bundle_files.items()
        if path in RUNTIME_PROTOCOL_PATHS
    }
    repo_files = {
        path: content
        for path, content in authored_bundle_files.items()
        if path not in RUNTIME_PROTOCOL_PATHS
    }
    manifest_payload = {
        "title": spec.title,
        "summary": spec.summary,
        "deliverable_id": deliverable.id,
        "deliverable_title": deliverable.title,
        "deliverable_objective": deliverable.objective,
        "course_structure": spec.course_structure.model_dump(mode="json"),
        "runtime_plan": spec.project_contract.runtime_plan.model_dump(mode="json"),
        "runtime_dependencies": spec.runtime_dependencies.model_dump(mode="json"),
        "public_endpoints": [endpoint.model_dump(mode="json") for endpoint in spec.public_endpoints],
        "public_checks": [check.model_dump(mode="json") for check in deliverable.public_checks],
        "visible_check_command": spec.runtime_dependencies.visible_check_command or default_visible_check_command(),
        "hidden_check_command": default_hidden_check_command(),
        "preview_command": spec.runtime_dependencies.preview_command or default_preview_command(spec),
        "entrypoint_path": task_agent_entrypoint_path(spec),
        "generated_test_scripts": {
            "source": "starter_default",
            "generated_for_deliverable": deliverable.id,
        },
        "starter_repo_bundle": {
            "source": "starter_default",
            "generated_for_deliverable": deliverable.id,
            "authored_paths": sorted(repo_files),
        },
        "runtime_protocol_bundle": {
            "source": "starter_default",
            "generated_for_deliverable": deliverable.id,
            "authored_paths": sorted(runtime_protocol_files),
        },
        "dependency_contract": {
            "manifest_paths": [],
            "lockfile_paths": [],
            "toolchain_paths": [],
            "build_support_paths": [],
            "reproducibility_mode": None,
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
        RUNTIME_INSTALL_SCRIPT_PATH: render_task_agent_runtime_install_script(spec),
        RUNTIME_VERIFY_SCRIPT_PATH: render_task_agent_runtime_verify_script(spec),
        RUNTIME_RUN_SCRIPT_PATH: render_task_agent_runtime_run_script(spec),
        RUNTIME_VISIBLE_CHECK_SCRIPT_PATH: render_task_agent_runtime_visible_check_script(),
        RUNTIME_HIDDEN_CHECK_SCRIPT_PATH: render_task_agent_runtime_hidden_check_script(),
        "checks/run_visible_checks.py": render_task_agent_visible_checks_script(),
        HIDDEN_GRADER_SCRIPT_PATH: render_task_agent_hidden_checks_script(),
        ".vscode/tasks.json": render_task_agent_vscode_tasks(),
    }
    files.update(runtime_protocol_files)
    files.update(repo_files)
    return files


def default_preview_command(spec: TaskAgentServiceSpec, *, host: str = "0.0.0.0") -> str:
    _ = (spec, host)
    return f"sh {RUNTIME_RUN_SCRIPT_PATH}"


def default_visible_check_command() -> str:
    return f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"


def default_hidden_check_command() -> str:
    return f"sh {RUNTIME_HIDDEN_CHECK_SCRIPT_PATH}"


def _runtime_protocol_placeholder(stage: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env sh",
            "set -eu",
            f"echo '[coursegen] {stage} has not been authored yet.' >&2",
            "exit 1",
            "",
        ]
    )


def render_task_agent_runtime_install_script(spec: TaskAgentServiceSpec) -> str:
    _ = spec
    return _runtime_protocol_placeholder("runtime install script")


def render_task_agent_runtime_verify_script(spec: TaskAgentServiceSpec) -> str:
    _ = spec
    return _runtime_protocol_placeholder("runtime verify script")


def render_task_agent_runtime_run_script(spec: TaskAgentServiceSpec) -> str:
    _ = spec
    return _runtime_protocol_placeholder("runtime run script")


def render_task_agent_runtime_visible_check_script() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env sh",
            "set -eu",
            "exec python3 checks/run_visible_checks.py",
            "",
        ]
    )


def render_task_agent_runtime_hidden_check_script() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env sh",
            "set -eu",
            "exec python3 .coursegen/grader/run_hidden_checks.py",
            "",
        ]
    )


def render_task_agent_starter_dockerfile(spec: TaskAgentServiceSpec) -> str:
    _ = spec
    return "\n".join(
        [
            "FROM debian:bookworm-slim",
            "",
            "RUN echo '[coursegen] runtime Dockerfile has not been authored yet.' >&2 && exit 1",
            "",
        ]
    )


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
                "command": f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}",
                "problemMatcher": [],
            },
            {
                "label": "Preview app",
                "type": "shell",
                "command": (
                    "python -c \"import json, os, subprocess; from pathlib import Path; "
                    f"manifest = json.loads(Path('{HIDDEN_MANIFEST_PATH}').read_text()); "
                    "env = os.environ.copy(); env.setdefault('PORT', '8000'); "
                    f"command = manifest.get('preview_command') or 'sh {RUNTIME_RUN_SCRIPT_PATH}'; "
                    "raise SystemExit(subprocess.run(command, shell=True, env=env).returncode)\""
                ),
                "problemMatcher": [],
            },
        ],
    }
    return json.dumps(payload, indent=2) + "\n"
