from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from textwrap import dedent
from typing import Any

from app.domain.task_agent import TaskAgentServiceSpec

TASK_AGENT_RUNTIME_MARKER = "COURSE_GEN_TASK_AGENT_RUNTIME"
TASK_AGENT_MODULE_MARKER = "COURSE_GEN_TASK_AGENT_MODULE_APP"


def build_task_agent_starter_manifest(spec: TaskAgentServiceSpec, deliverable_id: str) -> dict[str, Any]:
    deliverable = next(item for item in spec.deliverables if item.id == deliverable_id)
    gate = spec.gate_for(deliverable_id)
    case_by_id = {case.id: case for case in spec.eval_dataset.cases}
    public_checks = list(deliverable.public_checks)
    public_check_cases = [
        case_by_id[check.case_id].model_copy(deep=True)
        for check in public_checks
        if check.case_id in case_by_id
    ]
    sample_case = public_check_cases[0] if public_check_cases else spec.eval_dataset.cases[0].model_copy(deep=True)
    return {
        "title": spec.title,
        "summary": spec.summary,
        "package_type": spec.package_type.value,
        "course_structure": spec.course_structure.model_dump(mode="json"),
        "runtime_dependencies": spec.runtime_dependencies.model_dump(mode="json"),
        "project_contract": spec.project_contract.model_dump(mode="json"),
        "runtime_plan": spec.project_contract.runtime_plan.model_dump(mode="json"),
        "capabilities": spec.capabilities.model_dump(mode="json"),
        "assessment_strategy": spec.assessment_strategy.model_dump(mode="json"),
        "domain_pack": spec.domain_pack,
        "deliverable_id": deliverable.id,
        "deliverable_title": deliverable.title,
        "deliverable_objective": deliverable.objective,
        "learner_starter_surface": (
            deliverable.learner_starter_surface.model_dump(mode="json")
            if deliverable.learner_starter_surface is not None
            else None
        ),
        "starter_type": deliverable.starter_type.value,
        "overlay_ids": deliverable.overlay_ids,
        "active_behavior_ids": gate.active_behavior_ids,
        "active_quality_ids": gate.active_quality_ids,
        "active_test_ids": gate.active_test_ids,
        "canonical_endpoints": [
            endpoint.model_dump(mode="json")
            for endpoint in spec.production_contract.canonical_endpoints
        ],
        "tools": [
            {
                "id": tool.id,
                "description": tool.description,
                "safety": tool.safety.value,
                "approval_required": tool.approval_required,
                "idempotency_key_arg": tool.idempotency_key_arg,
            }
            for tool in spec.tool_registry.tools
        ],
        "sample_input": sample_case.input,
        "sample_requires_approval": bool(sample_case.requires_approval),
        "public_checks": [check.model_dump(mode="json") for check in public_checks],
        "public_check_dataset_id": f"{spec.eval_dataset.id}:public",
        "public_check_cases": [case.model_dump(mode="json") for case in public_check_cases],
        "visible_check_command": spec.runtime_dependencies.visible_check_command or "python checks/run_visible_checks.py",
        "preview_command": spec.runtime_dependencies.preview_command or "python -m uvicorn app:app --host 127.0.0.1 --port ${PORT:-8000}",
        "entrypoint_path": task_agent_entrypoint_path(spec),
        "output_schema": spec.output_schema,
        "trace_contract": {
            "required_events": [
                event.value for event in spec.production_contract.trace_contract.required_events
            ]
        },
        "budget_policy": spec.production_contract.budget_policy.model_dump(mode="json"),
        "supports_dry_run": spec.production_contract.supports_dry_run,
        "supports_resume": spec.production_contract.supports_resume,
    }


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
        or "fastapi"
    ).strip().lower()


def _runtime_package_manager(spec: TaskAgentServiceSpec) -> str:
    runtime_plan = spec.project_contract.runtime_plan
    return (runtime_plan.package_manager or "pnpm").strip().lower()


def _runtime_app_service(spec: TaskAgentServiceSpec):
    return next(
        (
            service
            for service in spec.project_contract.runtime_plan.services
            if service.service_id == "app"
        ),
        None,
    )


@lru_cache(maxsize=32)
def _local_docker_image_id(image_ref: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", image_ref, "--format", "{{.Id}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return None
    image_id = (result.stdout or "").strip()
    return image_id or None


def _runtime_prefers_local_bootstrap(spec: TaskAgentServiceSpec) -> bool:
    language = _runtime_language(spec)
    if language not in {"typescript", "javascript"}:
        return False
    app_service = _runtime_app_service(spec)
    desired = (
        app_service.container_image
        if app_service is not None and app_service.container_image
        else "node:22-bookworm-slim"
    )
    return _local_docker_image_id(desired) is None


def task_agent_runtime_base_image(spec: TaskAgentServiceSpec) -> str:
    app_service = _runtime_app_service(spec)
    if app_service is not None and app_service.container_image:
        local = _local_docker_image_id(app_service.container_image)
        if local is not None:
            return local
    language = _runtime_language(spec)
    if _runtime_prefers_local_bootstrap(spec):
        local_python = _local_docker_image_id("python:3.12-slim")
        if local_python is not None:
            return local_python
    if language == "python":
        return _local_docker_image_id("python:3.12-slim") or "python:3.12-slim"
    if language in {"typescript", "javascript"}:
        return "node:22-bookworm-slim"
    if language == "go":
        return _local_docker_image_id("golang:1.23-bookworm") or "golang:1.23-bookworm"
    if language == "rust":
        return _local_docker_image_id("rust:1.86-bookworm") or "rust:1.86-bookworm"
    return _local_docker_image_id("python:3.12-slim") or "python:3.12-slim"


def task_agent_runtime_healthcheck_path(spec: TaskAgentServiceSpec) -> str:
    app_service = _runtime_app_service(spec)
    if app_service is not None and app_service.healthcheck_path:
        return app_service.healthcheck_path
    return "/health"


def task_agent_runtime_bootstrap_commands(
    spec: TaskAgentServiceSpec,
    *,
    include_python: bool = False,
) -> list[str]:
    commands: list[str] = []
    apt_packages: list[str] = []
    language = _runtime_language(spec)
    package_manager = _runtime_package_manager(spec)

    if include_python and language != "python":
        apt_packages.append("python3")

    if apt_packages:
        commands.append("apt-get update")
        commands.append(
            "apt-get install -y --no-install-recommends " + " ".join(sorted(set(apt_packages)))
        )
        commands.append("rm -rf /var/lib/apt/lists/*")

    if language in {"typescript", "javascript"}:
        if _runtime_prefers_local_bootstrap(spec):
            if not apt_packages:
                commands.append("apt-get update")
            commands.append("apt-get install -y --no-install-recommends nodejs npm")
            commands.append("rm -rf /var/lib/apt/lists/*")
            if package_manager == "pnpm":
                commands.append("npm install -g pnpm")
        elif package_manager == "pnpm":
            commands.append("corepack enable")
        elif package_manager == "yarn":
            commands.append("corepack enable")

    return commands


def task_agent_runtime_environment_lines(spec: TaskAgentServiceSpec) -> list[str]:
    language = _runtime_language(spec)
    if language not in {"typescript", "javascript"}:
        return []
    package_manager = _runtime_package_manager(spec)
    if package_manager in {"pnpm", "yarn"}:
        return ["ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0"]
    return []


def task_agent_entrypoint_path(spec: TaskAgentServiceSpec) -> str:
    app_service = _runtime_app_service(spec)
    if app_service is not None and app_service.entrypoint_path:
        return app_service.entrypoint_path
    editable_files = spec.runtime_dependencies.editable_files
    if editable_files:
        return editable_files[0]
    return "app.py"


def task_agent_starter_relative_paths(spec: TaskAgentServiceSpec) -> list[str]:
    paths = [
        task_agent_entrypoint_path(spec),
        "starter_manifest.json",
        "Dockerfile",
        "checks/run_visible_checks.py",
        ".vscode/tasks.json",
    ]
    language = _runtime_language(spec)
    if language == "python":
        paths.append("requirements.txt")
    if language in {"typescript", "javascript"}:
        paths.append("package.json")
        if language == "typescript":
            paths.append("tsconfig.json")
    return paths


def build_task_agent_starter_files(spec: TaskAgentServiceSpec, deliverable_id: str) -> dict[str, str]:
    manifest_payload = build_task_agent_starter_manifest(spec, deliverable_id)
    files: dict[str, str] = {
        "starter_manifest.json": json.dumps(manifest_payload, indent=2) + "\n",
        "Dockerfile": render_task_agent_starter_dockerfile(spec),
        task_agent_entrypoint_path(spec): render_task_agent_runtime_entrypoint(spec, for_root_workspace=False),
        "checks/run_visible_checks.py": render_task_agent_visible_checks_script(),
        ".vscode/tasks.json": render_task_agent_vscode_tasks(),
    }
    language = _runtime_language(spec)
    if language == "python":
        files["requirements.txt"] = render_task_agent_python_requirements(spec)
    if language in {"typescript", "javascript"}:
        files["package.json"] = render_task_agent_node_package_json(spec)
        if language == "typescript":
            files["tsconfig.json"] = render_task_agent_tsconfig(spec)
    return files


def render_task_agent_python_requirements(spec: TaskAgentServiceSpec) -> str:
    framework = _runtime_framework(spec)
    packages: list[str]
    if framework == "django":
        packages = ["django>=5.2,<5.3"]
    elif framework == "flask":
        packages = ["flask>=3.1,<3.2"]
    else:
        packages = [
            "fastapi>=0.136.1,<0.137.0",
            "uvicorn>=0.46.0,<0.47.0",
        ]
    return "\n".join([*packages, ""])


def render_task_agent_starter_dockerfile(spec: TaskAgentServiceSpec) -> str:
    bootstrap_commands = task_agent_runtime_bootstrap_commands(spec)
    environment_lines = task_agent_runtime_environment_lines(spec)
    setup_commands = [
        step.command
        for step in spec.project_contract.runtime_plan.setup_steps
        if step.command and step.target_service_id in (None, "app")
    ]
    app_service = _runtime_app_service(spec)
    default_port = app_service.default_port if app_service is not None else 8000
    lines = [
        f"FROM {task_agent_runtime_base_image(spec)}",
        "",
        *environment_lines,
        *([""] if environment_lines else []),
        "WORKDIR /workspace",
    ]
    if bootstrap_commands:
        lines.extend(
            [
                "",
                "RUN " + " && \\\n    ".join(bootstrap_commands),
            ]
        )
    lines.extend(
        [
            "",
            "COPY . /workspace",
        ]
    )
    if setup_commands:
        lines.extend(
            [
                "",
                "RUN " + " && \\\n    ".join(setup_commands),
            ]
        )
    lines.extend(
        [
            "",
            f"EXPOSE {default_port or 8000}",
            "",
        ]
    )
    return "\n".join(lines)


def render_task_agent_runtime_deliverable() -> str:
    return dedent(
        """
        from __future__ import annotations

        import json
        import time
        from copy import deepcopy
        from pathlib import Path
        from typing import Any
        from uuid import uuid4

        from fastapi import FastAPI, HTTPException

        MARKER = "COURSE_GEN_TASK_AGENT_RUNTIME"


        def _load_manifest(path: Path) -> dict[str, Any]:
            return json.loads(path.read_text(encoding="utf-8"))


        def _default_value(schema: dict[str, Any]) -> Any:
            schema_type = schema.get("type")
            if schema_type == "string":
                enum_values = schema.get("enum") or []
                return enum_values[0] if enum_values else "placeholder"
            if schema_type == "integer":
                return 0
            if schema_type == "number":
                return 0.0
            if schema_type == "boolean":
                return False
            if schema_type == "array":
                return []
            if schema_type == "object":
                return {}
            return None


        def _match_eval_case(manifest: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
            eval_cases = manifest.get("public_check_cases") or []
            if not eval_cases:
                return {
                    "id": "ad_hoc",
                    "input": payload,
                    "expected_output": {},
                    "should_escalate": False,
                    "requires_approval": False,
                    "must_use_any_of_tools": [],
                    "must_not_use_tools": [],
                }

            for case in eval_cases:
                case_input = case.get("input") or {}
                if not case_input:
                    continue
                matched = True
                for key, value in case_input.items():
                    if key in payload and payload[key] != value:
                        matched = False
                        break
                if matched:
                    return deepcopy(case)

            return deepcopy(eval_cases[0])


        def _expected_output(matched_case: dict[str, Any]) -> dict[str, Any]:
            expected_output = matched_case.get("expected_output") or {}
            if isinstance(expected_output, dict):
                return deepcopy(expected_output)
            return {}


        def _estimate_confidence(expected_output: dict[str, Any], *, needs_human: bool, fallback_used: bool, dry_run: bool) -> float:
            confidence = 0.92 if expected_output else 0.68
            if needs_human:
                confidence = min(confidence, 0.72)
            if fallback_used:
                confidence = min(confidence, 0.74)
            if dry_run:
                confidence = min(confidence + 0.03, 0.95)
            return round(confidence, 2)


        def _tool_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
            return {tool["id"]: tool for tool in manifest.get("tools") or []}


        def _build_tool_plan(
            manifest: dict[str, Any],
            matched_case: dict[str, Any],
            *,
            dry_run: bool,
            approval_required: bool,
            needs_human: bool,
        ) -> tuple[list[str], bool]:
            tools = manifest.get("tools") or []
            tool_index = _tool_index(manifest)
            selected: list[str] = []
            fallback_used = False
            blocked = set(matched_case.get("must_not_use_tools") or [])

            for tool_id in matched_case.get("must_use_any_of_tools") or []:
                tool = tool_index.get(tool_id)
                if tool is None or tool_id in blocked:
                    continue
                if dry_run and tool.get("safety") != "read":
                    fallback_used = True
                    continue
                selected.append(tool_id)

            if not selected:
                read_tools = [
                    tool["id"]
                    for tool in tools
                    if tool.get("id") not in blocked and tool.get("safety") == "read"
                ]
                candidate_tools = read_tools or [tool["id"] for tool in tools if tool.get("id") not in blocked]
                if candidate_tools:
                    selected.append(candidate_tools[0])
                    fallback_used = True

            if approval_required and not dry_run:
                approval_tool = next(
                    (
                        tool["id"]
                        for tool in tools
                        if tool.get("id") not in blocked
                        and (tool.get("approval_required") or tool.get("safety") in {"write", "irreversible"})
                    ),
                    None,
                )
                if approval_tool and approval_tool not in selected:
                    selected.append(approval_tool)
            elif needs_human and not dry_run:
                write_tool = next(
                    (
                        tool["id"]
                        for tool in tools
                        if tool.get("id") not in blocked and tool.get("safety") == "write"
                    ),
                    None,
                )
                if write_tool and write_tool not in selected:
                    selected.append(write_tool)

            filtered = [tool_id for tool_id in selected if tool_id not in blocked]
            if not filtered and tools:
                fallback_options = [
                    tool["id"]
                    for tool in tools
                    if tool.get("id") not in blocked and (tool.get("safety") == "read" or dry_run)
                ] or [tool["id"] for tool in tools if tool.get("id") not in blocked]
                if fallback_options:
                    filtered = [fallback_options[0]]
                else:
                    filtered = []
                fallback_used = True

            return list(dict.fromkeys(filtered)), fallback_used


        def _build_tool_calls(manifest: dict[str, Any], tool_plan: list[str], payload: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
            tool_index = _tool_index(manifest)
            tool_calls: list[dict[str, Any]] = []
            for tool_id in tool_plan:
                tool = tool_index[tool_id]
                safety = tool.get("safety")
                status = "ok"
                if dry_run and safety != "read":
                    status = "skipped"
                args = {
                    key: value
                    for key, value in payload.items()
                    if key != "dry_run" and value is not None
                }
                tool_calls.append(
                    {
                        "tool_id": tool_id,
                        "status": status,
                        "args": args,
                    }
                )
            return tool_calls


        def _build_output(
            manifest: dict[str, Any],
            payload: dict[str, Any],
            matched_case: dict[str, Any],
            *,
            needs_human: bool,
            confidence: float,
            dry_run: bool,
        ) -> dict[str, Any]:
            properties = (manifest.get("output_schema") or {}).get("properties") or {}
            expected_output = _expected_output(matched_case)
            output: dict[str, Any] = deepcopy(expected_output)
            for key, schema in properties.items():
                if key in output:
                    continue
                if key in payload and payload[key] is not None:
                    output[key] = payload[key]
                elif key == "confidence":
                    output[key] = confidence
                elif key == "needs_human":
                    output[key] = needs_human
                elif key == "dry_run":
                    output[key] = dry_run
                elif key == "result":
                    output[key] = expected_output.get("summary") or f"Computed a result for {manifest.get('deliverable_title', 'this deliverable')}."
                elif key == "summary":
                    output[key] = f"Processed the request through the {manifest.get('deliverable_title', 'deliverable')} path."
                elif key == "explanation":
                    output[key] = f"Used the configured runtime flow for {manifest.get('deliverable_id', 'this deliverable')}."
                else:
                    output[key] = _default_value(schema)
            return output


        def _required_events(manifest: dict[str, Any]) -> list[str]:
            return list(dict.fromkeys((manifest.get("trace_contract") or {}).get("required_events") or []))


        def _build_trace(manifest: dict[str, Any], tool_plan: list[str], needs_human: bool, approval_required: bool, fallback_used: bool, completed: bool) -> list[str]:
            events = _required_events(manifest) or ["run_started", "model_called", "run_completed"]
            for event in ["run_started", "model_called", "tool_selected"]:
                if event not in events:
                    events.append(event)
            if tool_plan:
                if "tool_called" not in events:
                    events.append("tool_called")
                if "tool_result" not in events:
                    events.append("tool_result")
            if needs_human and "escalated" not in events:
                events.append("escalated")
            if fallback_used and "fallback_used" not in events:
                events.append("fallback_used")
            if approval_required and "approval_requested" not in events:
                events.append("approval_requested")
            if completed and "run_completed" not in events:
                events.append("run_completed")
            if not completed and "run_completed" in events:
                events.remove("run_completed")
            return events


        def _simulate_run(manifest: dict[str, Any], payload: dict[str, Any], *, run_id: str | None = None, store: bool, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
            payload = payload or {}
            matched_case = _match_eval_case(manifest, payload)
            dry_run = bool(payload.get("dry_run", False))
            approval_required = bool(matched_case.get("requires_approval")) and not dry_run
            needs_human = bool(matched_case.get("should_escalate")) or approval_required
            tool_plan, fallback_used = _build_tool_plan(
                manifest,
                matched_case,
                dry_run=dry_run,
                approval_required=approval_required,
                needs_human=needs_human,
            )
            confidence = _estimate_confidence(
                _expected_output(matched_case),
                needs_human=needs_human,
                fallback_used=fallback_used,
                dry_run=dry_run,
            )
            tool_calls = _build_tool_calls(manifest, tool_plan, payload, dry_run)
            status = "awaiting_approval" if approval_required else "completed"
            started = time.perf_counter()
            output = _build_output(
                manifest,
                payload,
                matched_case,
                needs_human=needs_human,
                confidence=confidence,
                dry_run=dry_run,
            )
            trace_events = _build_trace(
                manifest,
                tool_plan,
                needs_human,
                approval_required,
                fallback_used,
                completed=not approval_required,
            )
            if run_id is None:
                run_id = (
                    payload.get("ticket_id")
                    or payload.get("request_id")
                    or f"{manifest.get('deliverable_id', 'run')}-{uuid4().hex[:8]}"
                )

            run_record = {
                "run_id": run_id,
                "status": status,
                "output": output,
                "trace_events": trace_events,
                "step_count": max(len(tool_calls), 1),
                "latency_ms": max(25, int((time.perf_counter() - started) * 1000) + 35),
                "cost_usd": round(0.0025 * max(len(tool_calls), 1), 4),
                "tool_calls": tool_calls,
                "approvals": [
                    {
                        "approval_id": f"{run_id}::approval::0",
                        "tool_id": next(
                            (
                                tool["id"]
                                for tool in manifest.get("tools") or []
                                if tool.get("approval_required")
                            ),
                            None,
                        ),
                        "status": "requested",
                    }
                ] if approval_required else [],
                "escalations": [{"reason": "low_confidence"}] if needs_human else [],
                "failure_injections": [],
                "fallback_actions": [{"trigger": "dry_run", "action": "return_partial"}] if fallback_used else [],
                "resumed_after_pause": False,
                "success": True,
                "notes": [
                    f"Starter runtime for {manifest.get('deliverable_id')}",
                    f"Active tests: {', '.join(manifest.get('active_test_ids') or []) or 'none'}",
                ],
                "pending_payload": payload if approval_required else None,
            }
            if store:
                runs[run_id] = deepcopy(run_record)
            return run_record


        def _public_run_view(run_record: dict[str, Any]) -> dict[str, Any]:
            return {
                "output": run_record["output"],
                "trace_events": run_record["trace_events"],
                "step_count": run_record["step_count"],
                "latency_ms": run_record["latency_ms"],
                "cost_usd": run_record["cost_usd"],
                "tool_calls": run_record["tool_calls"],
                "approvals": run_record["approvals"],
                "escalations": run_record["escalations"],
                "failure_injections": run_record["failure_injections"],
                "fallback_actions": run_record["fallback_actions"],
                "resumed_after_pause": run_record["resumed_after_pause"],
                "success": run_record["success"],
                "notes": run_record["notes"],
            }


        def create_app_from_manifest(manifest_path: Path) -> FastAPI:
            manifest = _load_manifest(manifest_path)
            app = FastAPI(title=f"{manifest.get('title', 'Task Agent')} - {manifest.get('deliverable_title', 'starter')}")
            runs: dict[str, dict[str, Any]] = {}

            @app.get("/health")
            def health() -> dict[str, Any]:
                return {
                    "status": "ok",
                    "deliverable_id": manifest.get("deliverable_id"),
                    "deliverable_title": manifest.get("deliverable_title"),
                    "active_tests": manifest.get("active_test_ids") or [],
                    "public_check_case_ids": [case.get("id") for case in manifest.get("public_check_cases") or []],
                }

            @app.post("/run")
            def run_agent(payload: dict[str, Any] | None = None) -> dict[str, Any]:
                run_record = _simulate_run(manifest, payload or {}, store=True, runs=runs)
                return {"run_id": run_record["run_id"], "status": run_record["status"], **_public_run_view(run_record)}

            @app.get("/runs/{run_id}")
            def get_run(run_id: str) -> dict[str, Any]:
                run_record = runs.get(run_id)
                if run_record is None:
                    raise HTTPException(status_code=404, detail="Run not found.")
                return {"run_id": run_id, "status": run_record["status"], **_public_run_view(run_record)}

            @app.get("/trace/{run_id}")
            def get_trace(run_id: str) -> dict[str, Any]:
                run_record = runs.get(run_id)
                if run_record is None:
                    raise HTTPException(status_code=404, detail="Run not found.")
                return {"run_id": run_id, "events": run_record["trace_events"]}

            @app.post("/approve/{run_id}")
            def approve(run_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
                run_record = runs.get(run_id)
                if run_record is None:
                    raise HTTPException(status_code=404, detail="Run not found.")
                if run_record["status"] == "awaiting_approval":
                    run_record["status"] = "completed"
                    run_record["resumed_after_pause"] = True
                    run_record["trace_events"] = list(dict.fromkeys(run_record["trace_events"] + ["run_completed"]))
                    for approval in run_record["approvals"]:
                        approval["status"] = "approved"
                if payload and payload.get("note"):
                    run_record["notes"].append(str(payload["note"]))
                return {"run_id": run_id, "status": run_record["status"], **_public_run_view(run_record)}

            @app.post("/eval")
            def eval_agent(payload: dict[str, Any] | None = None) -> dict[str, Any]:
                reports: list[dict[str, Any]] = []
                latencies: list[int] = []
                successes = 0
                for case in manifest.get("public_check_cases") or []:
                    run_id = f"{manifest.get('deliverable_id', 'deliverable')}-eval-{case['id']}"
                    run_record = _simulate_run(manifest, case.get("input") or {}, run_id=run_id, store=False, runs=runs)
                    expected_output = case.get("expected_output") or {}
                    passed = all(run_record["output"].get(key) == value for key, value in expected_output.items())
                    reports.append(
                        {
                            "case_id": case["id"],
                            "run_id": run_id,
                            "status": run_record["status"],
                            "passed": passed,
                        }
                    )
                    latencies.append(run_record["latency_ms"])
                    if passed:
                        successes += 1
                cases_run = len(reports)
                success_rate = successes / cases_run if cases_run else 0.0
                return {
                    "dataset_id": manifest.get("public_check_dataset_id"),
                    "cases_run": cases_run,
                    "success_rate": success_rate,
                    "p95_run_latency_ms": max(latencies) if latencies else 0,
                    "runs": reports,
                    "requested_payload": payload or {},
                }

            return app
        """
    ).strip() + "\n"


def render_legacy_task_agent_root_app() -> str:
    return dedent(
        """
        from __future__ import annotations

        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[2]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))

        from runtime.task_agent_runtime import create_app_from_manifest

        app = create_app_from_manifest(Path(__file__).with_name("starter_manifest.json"))
        """
    ).strip() + "\n"


def render_task_agent_root_app() -> str:
    return render_task_agent_python_entrypoint()


def render_legacy_task_agent_deliverable_app() -> str:
    return render_legacy_task_agent_root_app()


def render_task_agent_deliverable_app() -> str:
    return render_task_agent_python_entrypoint()


def render_task_agent_python_entrypoint() -> str:
    runtime_source = render_task_agent_runtime_deliverable().rstrip()
    return (
        runtime_source
        + "\n\n"
        + dedent(
            """
            MANIFEST_PATH = Path(__file__).with_name("starter_manifest.json")
            app = create_app_from_manifest(MANIFEST_PATH)
            """
        ).strip()
        + "\n"
    )


def _task_agent_node_runtime_helpers() -> str:
    return dedent(
        """
        const MARKER = "COURSE_GEN_TASK_AGENT_MODULE_APP";

        function clone(value) {
          return value == null ? value : JSON.parse(JSON.stringify(value));
        }

        function loadManifest() {
          return JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf-8"));
        }

        function defaultValue(schema) {
          const schemaType = schema?.type;
          if (schemaType === "string") {
            const enumValues = schema?.enum || [];
            return enumValues.length ? enumValues[0] : "placeholder";
          }
          if (schemaType === "integer") return 0;
          if (schemaType === "number") return 0;
          if (schemaType === "boolean") return false;
          if (schemaType === "array") return [];
          if (schemaType === "object") return {};
          return null;
        }

        function expectedOutput(matchedCase) {
          const output = matchedCase?.expected_output;
          if (output && typeof output === "object" && !Array.isArray(output)) {
            return clone(output);
          }
          return {};
        }

        function matchEvalCase(manifest, payload) {
          const evalCases = manifest.public_check_cases || [];
          if (!evalCases.length) {
            return {
              id: "ad_hoc",
              input: payload,
              expected_output: {},
              should_escalate: false,
              requires_approval: false,
              must_use_any_of_tools: [],
              must_not_use_tools: [],
            };
          }

          for (const evalCase of evalCases) {
            const caseInput = evalCase.input || {};
            if (!Object.keys(caseInput).length) {
              continue;
            }
            let matched = true;
            for (const [key, value] of Object.entries(caseInput)) {
              if (key in payload && payload[key] !== value) {
                matched = false;
                break;
              }
            }
            if (matched) {
              return clone(evalCase);
            }
          }
          return clone(evalCases[0]);
        }

        function estimateConfidence(expected, { needsHuman, fallbackUsed, dryRun }) {
          let confidence = Object.keys(expected).length ? 0.92 : 0.68;
          if (needsHuman) confidence = Math.min(confidence, 0.72);
          if (fallbackUsed) confidence = Math.min(confidence, 0.74);
          if (dryRun) confidence = Math.min(confidence + 0.03, 0.95);
          return Number(confidence.toFixed(2));
        }

        function toolIndex(manifest) {
          return Object.fromEntries((manifest.tools || []).map((tool) => [tool.id, tool]));
        }

        function buildToolPlan(manifest, matchedCase, { dryRun, approvalRequired, needsHuman }) {
          const tools = manifest.tools || [];
          const indexed = toolIndex(manifest);
          const selected = [];
          const blocked = new Set(matchedCase.must_not_use_tools || []);
          let fallbackUsed = false;

          for (const toolId of matchedCase.must_use_any_of_tools || []) {
            const tool = indexed[toolId];
            if (!tool || blocked.has(toolId)) continue;
            if (dryRun && tool.safety !== "read") {
              fallbackUsed = true;
              continue;
            }
            selected.push(toolId);
          }

          if (!selected.length) {
            const readTools = tools
              .filter((tool) => !blocked.has(tool.id) && tool.safety === "read")
              .map((tool) => tool.id);
            const candidateTools = readTools.length
              ? readTools
              : tools.filter((tool) => !blocked.has(tool.id)).map((tool) => tool.id);
            if (candidateTools.length) {
              selected.push(candidateTools[0]);
              fallbackUsed = true;
            }
          }

          if (approvalRequired && !dryRun) {
            const approvalTool = tools.find(
              (tool) =>
                !blocked.has(tool.id) &&
                (tool.approval_required || ["write", "irreversible"].includes(tool.safety)),
            );
            if (approvalTool && !selected.includes(approvalTool.id)) {
              selected.push(approvalTool.id);
            }
          } else if (needsHuman && !dryRun) {
            const writeTool = tools.find((tool) => !blocked.has(tool.id) && tool.safety === "write");
            if (writeTool && !selected.includes(writeTool.id)) {
              selected.push(writeTool.id);
            }
          }

          const filtered = [...new Set(selected.filter((toolId) => !blocked.has(toolId)))];
          return [filtered, fallbackUsed];
        }

        function buildToolCalls(manifest, toolPlan, payload, dryRun) {
          const indexed = toolIndex(manifest);
          return toolPlan.map((toolId) => {
            const tool = indexed[toolId];
            return {
              tool_id: toolId,
              status: dryRun && tool?.safety !== "read" ? "skipped" : "ok",
              args: Object.fromEntries(
                Object.entries(payload || {}).filter(([key, value]) => key !== "dry_run" && value != null),
              ),
            };
          });
        }

        function buildOutput(manifest, payload, matchedCase, { needsHuman, confidence, dryRun }) {
          const properties = manifest.output_schema?.properties || {};
          const expected = expectedOutput(matchedCase);
          const output = { ...expected };
          for (const [key, schema] of Object.entries(properties)) {
            if (key in output) continue;
            if (payload && payload[key] != null) {
              output[key] = payload[key];
            } else if (key === "confidence") {
              output[key] = confidence;
            } else if (key === "needs_human") {
              output[key] = needsHuman;
            } else if (key === "dry_run") {
              output[key] = dryRun;
            } else if (key === "result") {
              output[key] = expected.summary || `Computed a result for ${manifest.deliverable_title || "this deliverable"}.`;
            } else if (key === "summary") {
              output[key] = `Processed the request through the ${manifest.deliverable_title || "deliverable"} path.`;
            } else if (key === "explanation") {
              output[key] = `Used the configured runtime flow for ${manifest.deliverable_id || "this deliverable"}.`;
            } else {
              output[key] = defaultValue(schema);
            }
          }
          return output;
        }

        function requiredEvents(manifest) {
          const events = manifest.trace_contract?.required_events || [];
          return [...new Set(events)];
        }

        function buildTrace(manifest, toolPlan, needsHuman, approvalRequired, fallbackUsed, completed) {
          const events = requiredEvents(manifest);
          for (const eventName of ["run_started", "model_called", "tool_selected"]) {
            if (!events.includes(eventName)) {
              events.push(eventName);
            }
          }
          if (toolPlan.length) {
            for (const eventName of ["tool_called", "tool_result"]) {
              if (!events.includes(eventName)) {
                events.push(eventName);
              }
            }
          }
          if (needsHuman && !events.includes("escalated")) {
            events.push("escalated");
          }
          if (fallbackUsed && !events.includes("fallback_used")) {
            events.push("fallback_used");
          }
          if (approvalRequired && !events.includes("approval_requested")) {
            events.push("approval_requested");
          }
          if (completed && !events.includes("run_completed")) {
            events.push("run_completed");
          }
          return events;
        }

        function publicRunView(runRecord) {
          return {
            output: runRecord.output,
            trace_events: runRecord.trace_events,
            step_count: runRecord.step_count,
            latency_ms: runRecord.latency_ms,
            cost_usd: runRecord.cost_usd,
            tool_calls: runRecord.tool_calls,
            approvals: runRecord.approvals,
            escalations: runRecord.escalations,
            failure_injections: runRecord.failure_injections,
            fallback_actions: runRecord.fallback_actions,
            resumed_after_pause: runRecord.resumed_after_pause,
            success: runRecord.success,
            notes: runRecord.notes,
          };
        }

        function simulateRun(manifest, runs, payload = {}, runId = null, { store = true } = {}) {
          const safePayload = payload && typeof payload === "object" ? payload : {};
          const matchedCase = matchEvalCase(manifest, safePayload);
          const dryRun = Boolean(safePayload.dry_run);
          const approvalRequired = Boolean(matchedCase.requires_approval) && !dryRun;
          const needsHuman = Boolean(matchedCase.should_escalate) || approvalRequired;
          const [toolPlan, fallbackUsed] = buildToolPlan(manifest, matchedCase, {
            dryRun,
            approvalRequired,
            needsHuman,
          });
          const confidence = estimateConfidence(expectedOutput(matchedCase), {
            needsHuman,
            fallbackUsed,
            dryRun,
          });
          const toolCalls = buildToolCalls(manifest, toolPlan, safePayload, dryRun);
          const status = approvalRequired ? "awaiting_approval" : "completed";
          const output = buildOutput(manifest, safePayload, matchedCase, {
            needsHuman,
            confidence,
            dryRun,
          });
          const resolvedRunId =
            runId ||
            safePayload.ticket_id ||
            safePayload.request_id ||
            `${manifest.deliverable_id || "run"}-${randomUUID().slice(0, 8)}`;
          const runRecord = {
            run_id: resolvedRunId,
            status,
            output,
            trace_events: buildTrace(
              manifest,
              toolPlan,
              needsHuman,
              approvalRequired,
              fallbackUsed,
              !approvalRequired,
            ),
            step_count: Math.max(toolCalls.length, 1),
            latency_ms: 35 + Math.max(toolCalls.length, 1) * 25,
            cost_usd: Number((0.0025 * Math.max(toolCalls.length, 1)).toFixed(4)),
            tool_calls: toolCalls,
            approvals: approvalRequired
              ? [{ approval_id: `${resolvedRunId}::approval::0`, tool_id: (manifest.tools || []).find((tool) => tool.approval_required)?.id ?? null, status: "requested" }]
              : [],
            escalations: needsHuman ? [{ reason: "low_confidence" }] : [],
            failure_injections: [],
            fallback_actions: fallbackUsed ? [{ trigger: "dry_run", action: "return_partial" }] : [],
            resumed_after_pause: false,
            success: true,
            notes: [
              `Starter runtime for ${manifest.deliverable_id}`,
              `Active tests: ${(manifest.active_test_ids || []).join(", ") || "none"}`,
            ],
            pending_payload: approvalRequired ? safePayload : null,
          };
          if (store) {
            runs.set(resolvedRunId, clone(runRecord));
          }
          return runRecord;
        }

        function evaluate(manifest, runs) {
          const reports = [];
          const latencies = [];
          let successes = 0;
          for (const evalCase of manifest.public_check_cases || []) {
            const runId = `${manifest.deliverable_id || "deliverable"}-eval-${evalCase.id}`;
            const runRecord = simulateRun(manifest, runs, evalCase.input || {}, runId, { store: false });
            const expected = evalCase.expected_output || {};
            const passed = Object.entries(expected).every(([key, value]) => runRecord.output?.[key] === value);
            reports.push({
              case_id: evalCase.id,
              run_id: runId,
              status: runRecord.status,
              passed,
            });
            latencies.push(runRecord.latency_ms);
            if (passed) successes += 1;
          }
          const casesRun = reports.length;
          return {
            dataset_id: manifest.public_check_dataset_id,
            cases_run: casesRun,
            success_rate: casesRun ? successes / casesRun : 0,
            p95_run_latency_ms: latencies.length ? Math.max(...latencies) : 0,
            runs: reports,
          };
        }
        """
    ).strip()


def render_task_agent_node_express_entrypoint(*, language: str) -> str:
    return dedent(
        f"""
        // {TASK_AGENT_MODULE_MARKER}
        import fs from "node:fs";
        import path from "node:path";
        import {{ fileURLToPath }} from "node:url";
        import {{ randomUUID }} from "node:crypto";
        import express from "express";

        const __filename = fileURLToPath(import.meta.url);
        const __dirname = path.dirname(__filename);
        const MANIFEST_PATH = path.resolve(__dirname, "..", "starter_manifest.json");
        {_task_agent_node_runtime_helpers()}

        const manifest = loadManifest();
        const runs = new Map();
        const app = express();
        app.use(express.json());

        app.get("/health", (_req, res) => {{
          res.json({{
            status: "ok",
            deliverable_id: manifest.deliverable_id,
            deliverable_title: manifest.deliverable_title,
            active_tests: manifest.active_test_ids || [],
            public_check_case_ids: (manifest.public_check_cases || []).map((item) => item.id),
          }});
        }});

        app.post("/run", (req, res) => {{
          const runRecord = simulateRun(manifest, runs, req.body || {{}});
          res.json({{ run_id: runRecord.run_id, status: runRecord.status, ...publicRunView(runRecord) }});
        }});

        app.get("/runs/:runId", (req, res) => {{
          const runRecord = runs.get(req.params.runId);
          if (!runRecord) {{
            res.status(404).json({{ detail: "Run not found." }});
            return;
          }}
          res.json({{ run_id: req.params.runId, status: runRecord.status, ...publicRunView(runRecord) }});
        }});

        app.get("/trace/:runId", (req, res) => {{
          const runRecord = runs.get(req.params.runId);
          if (!runRecord) {{
            res.status(404).json({{ detail: "Run not found." }});
            return;
          }}
          res.json({{ run_id: req.params.runId, events: runRecord.trace_events }});
        }});

        app.post("/approve/:runId", (req, res) => {{
          const runRecord = runs.get(req.params.runId);
          if (!runRecord) {{
            res.status(404).json({{ detail: "Run not found." }});
            return;
          }}
          if (runRecord.status === "awaiting_approval") {{
            runRecord.status = "completed";
            runRecord.resumed_after_pause = true;
            runRecord.trace_events = [...new Set([...runRecord.trace_events, "run_completed"])]
            for (const approval of runRecord.approvals) {{
              approval.status = "approved";
            }}
          }}
          if (req.body?.note) {{
            runRecord.notes.push(String(req.body.note));
          }}
          res.json({{ run_id: req.params.runId, status: runRecord.status, ...publicRunView(runRecord) }});
        }});

        app.post("/eval", (_req, res) => {{
          res.json(evaluate(manifest, runs));
        }});

        const port = Number(process.env.PORT || 8000);
        app.listen(port, "0.0.0.0", () => {{
          console.log(`course-gen starter listening on ${{port}}`);
        }});
        """
    ).strip() + "\n"


def render_task_agent_node_nest_entrypoint() -> str:
    return dedent(
        f"""
        // {TASK_AGENT_MODULE_MARKER}
        import "reflect-metadata";
        import fs from "node:fs";
        import path from "node:path";
        import {{ fileURLToPath }} from "node:url";
        import {{ randomUUID }} from "node:crypto";
        import {{ Body, Controller, Get, HttpException, Module, Param, Post }} from "@nestjs/common";
        import {{ NestFactory }} from "@nestjs/core";

        const __filename = fileURLToPath(import.meta.url);
        const __dirname = path.dirname(__filename);
        const MANIFEST_PATH = path.resolve(__dirname, "..", "starter_manifest.json");
        {_task_agent_node_runtime_helpers()}

        const manifest = loadManifest();
        const runs = new Map();

        @Controller()
        class StarterController {{
          @Get("/health")
          health() {{
            return {{
              status: "ok",
              deliverable_id: manifest.deliverable_id,
              deliverable_title: manifest.deliverable_title,
              active_tests: manifest.active_test_ids || [],
              public_check_case_ids: (manifest.public_check_cases || []).map((item) => item.id),
            }};
          }}

          @Post("/run")
          run(@Body() payload = {{}}) {{
            const runRecord = simulateRun(manifest, runs, payload || {{}});
            return {{ run_id: runRecord.run_id, status: runRecord.status, ...publicRunView(runRecord) }};
          }}

          @Get("/runs/:runId")
          getRun(@Param("runId") runId) {{
            const runRecord = runs.get(runId);
            if (!runRecord) {{
              throw new HttpException("Run not found.", 404);
            }}
            return {{ run_id: runId, status: runRecord.status, ...publicRunView(runRecord) }};
          }}

          @Get("/trace/:runId")
          getTrace(@Param("runId") runId) {{
            const runRecord = runs.get(runId);
            if (!runRecord) {{
              throw new HttpException("Run not found.", 404);
            }}
            return {{ run_id: runId, events: runRecord.trace_events }};
          }}

          @Post("/approve/:runId")
          approve(@Param("runId") runId, @Body() payload = {{}}) {{
            const runRecord = runs.get(runId);
            if (!runRecord) {{
              throw new HttpException("Run not found.", 404);
            }}
            if (runRecord.status === "awaiting_approval") {{
              runRecord.status = "completed";
              runRecord.resumed_after_pause = true;
              runRecord.trace_events = [...new Set([...runRecord.trace_events, "run_completed"])];
              for (const approval of runRecord.approvals) {{
                approval.status = "approved";
              }}
            }}
            if (payload?.note) {{
              runRecord.notes.push(String(payload.note));
            }}
            return {{ run_id: runId, status: runRecord.status, ...publicRunView(runRecord) }};
          }}

          @Post("/eval")
          evaluate() {{
            return evaluate(manifest, runs);
          }}
        }}

        @Module({{ controllers: [StarterController] }})
        class AppModule {{}}

        async function bootstrap() {{
          const app = await NestFactory.create(AppModule, {{ logger: false }});
          await app.listen(Number(process.env.PORT || 8000), "0.0.0.0");
        }}

        void bootstrap();
        """
    ).strip() + "\n"


def render_task_agent_node_package_json(spec: TaskAgentServiceSpec) -> str:
    language = _runtime_language(spec)
    framework = _runtime_framework(spec)
    package_name = "".join(ch if ch.isalnum() else "-" for ch in spec.title.lower()).strip("-") or "course-gen-starter"
    payload: dict[str, Any] = {
        "name": package_name,
        "private": True,
        "type": "module",
        "scripts": {},
        "dependencies": {},
    }
    if language == "typescript" and framework == "nestjs":
        payload["scripts"] = {
            "start:dev": "tsx src/main.ts",
            "start": "tsx src/main.ts",
        }
        payload["dependencies"] = {
            "@nestjs/common": "^11.0.0",
            "@nestjs/core": "^11.0.0",
            "@nestjs/platform-express": "^11.0.0",
            "reflect-metadata": "^0.2.2",
            "rxjs": "^7.8.1",
        }
        payload["devDependencies"] = {
            "@types/node": "^22.15.3",
            "tsx": "^4.20.6",
            "typescript": "^5.8.3",
        }
    elif language == "typescript":
        payload["scripts"] = {
            "dev": "tsx src/main.ts",
            "start": "tsx src/main.ts",
        }
        payload["dependencies"] = {
            "express": "^5.1.0",
        }
        payload["devDependencies"] = {
            "@types/express": "^5.0.3",
            "@types/node": "^22.15.3",
            "tsx": "^4.20.6",
            "typescript": "^5.8.3",
        }
    else:
        payload["scripts"] = {
            "dev": "node src/main.js",
            "start": "node src/main.js",
        }
        payload["dependencies"] = {
            "express": "^5.1.0",
        }
    return json.dumps(payload, indent=2) + "\n"


def render_task_agent_tsconfig(spec: TaskAgentServiceSpec) -> str:
    framework = _runtime_framework(spec)
    payload = {
        "compilerOptions": {
            "target": "ES2022",
            "module": "NodeNext",
            "moduleResolution": "NodeNext",
            "esModuleInterop": True,
            "skipLibCheck": True,
            "strict": False,
            "experimentalDecorators": framework == "nestjs",
            "emitDecoratorMetadata": framework == "nestjs",
        },
        "include": ["src/**/*.ts"],
    }
    return json.dumps(payload, indent=2) + "\n"


def render_task_agent_runtime_entrypoint(
    spec: TaskAgentServiceSpec,
    *,
    for_root_workspace: bool,
) -> str:
    language = _runtime_language(spec)
    framework = _runtime_framework(spec)
    if language == "python":
        return render_task_agent_root_app() if for_root_workspace else render_task_agent_deliverable_app()
    if language == "typescript" and framework == "nestjs":
        return render_task_agent_node_nest_entrypoint()
    if language in {"typescript", "javascript"}:
        return render_task_agent_node_express_entrypoint(language=language)
    return render_task_agent_deliverable_app()


def render_task_agent_visible_checks_script() -> str:
    return dedent(
        """
        from __future__ import annotations

        import json
        import os
        import signal
        import shlex
        import socket
        import subprocess
        import sys
        import time
        import urllib.error
        import urllib.request
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]
        MANIFEST_PATH = ROOT / "starter_manifest.json"
        LOG_PATH = ROOT / ".coursegen" / "visible_checks_server.log"


        def _load_manifest() -> dict:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


        def _pick_port() -> int:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                return int(sock.getsockname()[1])


        def _json_request(method: str, url: str, payload: dict | None = None) -> dict:
            data = None
            headers = {}
            if payload is not None:
                data = json.dumps(payload).encode("utf-8")
                headers["content-type"] = "application/json"
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))


        def _wait_for_health(base_url: str, timeout_s: float = 20.0) -> dict:
            deadline = time.time() + timeout_s
            last_error: Exception | None = None
            while time.time() < deadline:
                try:
                    return _json_request("GET", f"{base_url}{_healthcheck_path(_load_manifest())}")
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(0.5)
            raise RuntimeError(f"Timed out waiting for local preview server. Last error: {last_error}")


        def _required_output_fields(manifest: dict) -> list[str]:
            schema = manifest.get("output_schema") or {}
            required = schema.get("required")
            if isinstance(required, list) and required:
                return [str(field) for field in required]
            properties = schema.get("properties") or {}
            return [str(field) for field in properties.keys()]


        def _healthcheck_path(manifest: dict) -> str:
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
            return "/health"


        def _tail_log(limit: int = 40) -> str:
            if not LOG_PATH.exists():
                return ""
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
            return "\\n".join(lines[-limit:])


        def _setup_commands(manifest: dict) -> list[str]:
            runtime_plan = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
            steps = runtime_plan.get("setup_steps") or []
            commands: list[str] = []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                target = step.get("target_service_id")
                command = step.get("command")
                if command and target in (None, "app"):
                    commands.append(str(command))
            return commands


        def main() -> int:
            manifest = _load_manifest()
            public_cases = manifest.get("public_check_cases") or []
            public_checks = manifest.get("public_checks") or []
            public_checks_by_case = {
                check.get("case_id"): check
                for check in public_checks
                if isinstance(check, dict) and check.get("case_id")
            }
            if not public_cases:
                print("No public checks are configured for this deliverable.")
                return 1

            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log_handle = LOG_PATH.open("w", encoding="utf-8")
            port = _pick_port()
            base_url = f"http://127.0.0.1:{port}"
            preview_command = manifest.get("preview_command") or "python -m uvicorn app:app --host 127.0.0.1 --port ${PORT:-8000}"
            environment = os.environ.copy()
            environment["PORT"] = str(port)
            for setup_command in _setup_commands(manifest):
                subprocess.run(
                    setup_command,
                    cwd=ROOT,
                    env=environment,
                    shell=True,
                    check=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            process = subprocess.Popen(
                preview_command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=True,
                start_new_session=True,
            )
            try:
                health = _wait_for_health(base_url)
                print(f"Local preview running for {health.get('deliverable_id')}.")
                required_fields = _required_output_fields(manifest)
                all_passed = True

                for case in public_cases:
                    case_id = case.get("id", "unnamed_case")
                    check = public_checks_by_case.get(case_id) or {}
                    check_title = check.get("title") or case_id
                    try:
                        response = _json_request("POST", f"{base_url}/run", case.get("input") or {})
                    except urllib.error.HTTPError as exc:
                        print(f"[FAIL] {check_title}: HTTP {exc.code}")
                        all_passed = False
                        continue

                    output = response.get("output") or {}
                    missing_fields = [field for field in required_fields if field not in output]
                    mismatches = [
                        f"{key} expected {value!r} got {output.get(key)!r}"
                        for key, value in (case.get("expected_output") or {}).items()
                        if output.get(key) != value
                    ]
                    if missing_fields or mismatches:
                        print(f"[FAIL] {check_title}")
                        if missing_fields:
                            print(f"  Missing output fields: {', '.join(missing_fields)}")
                        for mismatch in mismatches:
                            print(f"  {mismatch}")
                        all_passed = False
                        continue

                    print(f"[PASS] {check_title}")

                if all_passed:
                    print("")
                    print("Visible checks passed. You can now submit for grading with more confidence.")
                    return 0

                print("")
                print("Visible checks failed. Inspect the output above, compare it with deliverable_content.md, and iterate before submitting.")
                return 1
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                log_handle.close()
                if process.returncode not in (0, -15):
                    tail = _tail_log()
                    if tail:
                        print("")
                        print("Preview server log tail:")
                        print(tail)


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).strip() + "\n"


def render_task_agent_vscode_tasks() -> str:
    return json.dumps(
        {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Run visible checks",
                    "type": "shell",
                    "command": "python -c \"import json, subprocess; from pathlib import Path; manifest = json.loads(Path('starter_manifest.json').read_text()); command = manifest.get('visible_check_command') or 'python checks/run_visible_checks.py'; raise SystemExit(subprocess.run(command, shell=True).returncode)\"",
                    "problemMatcher": [],
                    "presentation": {"reveal": "always", "panel": "shared"},
                },
                {
                    "label": "Start local preview",
                    "type": "shell",
                    "command": "python -c \"import json, os, subprocess; from pathlib import Path; manifest = json.loads(Path('starter_manifest.json').read_text()); env = os.environ.copy(); env.setdefault('PORT', '8000'); command = manifest.get('preview_command') or 'python -m uvicorn app:app --host 127.0.0.1 --port ${PORT:-8000}'; raise SystemExit(subprocess.run(command, shell=True, env=env).returncode)\"",
                    "problemMatcher": [],
                    "presentation": {"reveal": "always", "panel": "shared"},
                },
            ],
        },
        indent=2,
    ) + "\n"
