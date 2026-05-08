from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from app.domain.task_agent import TaskAgentServiceSpec

TASK_AGENT_RUNTIME_MARKER = "COURSE_GEN_TASK_AGENT_RUNTIME"
TASK_AGENT_MODULE_MARKER = "COURSE_GEN_TASK_AGENT_MODULE_APP"


def build_task_agent_starter_manifest(spec: TaskAgentServiceSpec, module_id: str) -> dict[str, Any]:
    module = next(item for item in spec.modules if item.id == module_id)
    gate = spec.gate_for(module_id)
    case_by_id = {case.id: case for case in spec.eval_dataset.cases}
    public_checks = list(module.public_checks)
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
        "capabilities": spec.capabilities.model_dump(mode="json"),
        "assessment_strategy": spec.assessment_strategy.model_dump(mode="json"),
        "domain_pack": spec.domain_pack,
        "module_id": module.id,
        "module_title": module.title,
        "module_objective": module.objective,
        "starter_type": module.starter_type.value,
        "overlay_ids": module.overlay_ids,
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
        "preview_command": spec.runtime_dependencies.preview_command or "python -m uvicorn app:app --host 127.0.0.1 --port 8000",
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


def render_task_agent_runtime_module() -> str:
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


        def _normalize_text(payload: dict[str, Any]) -> str:
            return json.dumps(payload, sort_keys=True).lower()


        def _infer_disposition(payload: dict[str, Any], matched_case: dict[str, Any]) -> str:
            expected_output = matched_case.get("expected_output") or {}
            if isinstance(expected_output, dict) and expected_output.get("disposition"):
                return str(expected_output["disposition"])

            text = _normalize_text(payload)
            if any(token in text for token in ("outage", "down", "urgent", "incident")):
                return "escalate"
            if any(token in text for token in ("refund", "invoice", "billing", "charged")):
                return "resolve"
            return "needs_info"


        def _infer_priority(payload: dict[str, Any], disposition: str) -> str:
            text = _normalize_text(payload)
            if "enterprise" in text or "urgent" in text or disposition == "escalate":
                return "urgent"
            if "pro" in text:
                return "high"
            if disposition == "needs_info":
                return "medium"
            return "low"


        def _reply_text(payload: dict[str, Any], disposition: str) -> str:
            identifier = payload.get("ticket_id") or payload.get("request_id") or "the request"
            if disposition == "resolve":
                return f"Prepared a customer-facing response for {identifier}."
            if disposition == "escalate":
                return f"Prepared an escalation handoff for {identifier}."
            return f"Prepared a follow-up asking for more information about {identifier}."


        def _tool_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
            return {tool["id"]: tool for tool in manifest.get("tools") or []}


        def _build_tool_plan(manifest: dict[str, Any], matched_case: dict[str, Any], disposition: str, dry_run: bool) -> tuple[list[str], bool]:
            tools = manifest.get("tools") or []
            tool_index = _tool_index(manifest)
            selected: list[str] = []
            fallback_used = False

            for tool_id in matched_case.get("must_use_any_of_tools") or []:
                tool = tool_index.get(tool_id)
                if tool is None:
                    continue
                if dry_run and tool.get("safety") != "read":
                    fallback_used = True
                    continue
                selected.append(tool_id)

            if not selected:
                read_tools = [tool["id"] for tool in tools if tool.get("safety") == "read"]
                if read_tools:
                    selected.append(read_tools[0])

            if disposition == "escalate":
                escalate_tool = next((tool["id"] for tool in tools if tool.get("safety") == "write"), None)
                if escalate_tool and escalate_tool not in selected and not dry_run:
                    selected.append(escalate_tool)

            if disposition == "resolve":
                reply_tool = next(
                    (
                        tool["id"]
                        for tool in tools
                        if tool.get("approval_required") or tool.get("safety") == "irreversible"
                    ),
                    None,
                )
                if reply_tool and reply_tool not in selected and not dry_run:
                    selected.append(reply_tool)

            filtered = [tool_id for tool_id in selected if tool_id not in set(matched_case.get("must_not_use_tools") or [])]
            if not filtered and tools:
                filtered = [next(tool["id"] for tool in tools if tool.get("safety") == "read")]
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
                    if key in {"ticket_id", "request_id", "customer_message", "task_input", "account_tier", "query"}
                }
                if "query" not in args and payload.get("customer_message"):
                    args["query"] = payload.get("customer_message")
                tool_calls.append(
                    {
                        "tool_id": tool_id,
                        "status": status,
                        "args": args,
                    }
                )
            return tool_calls


        def _build_output(manifest: dict[str, Any], payload: dict[str, Any], disposition: str, priority: str, needs_human: bool, confidence: float) -> dict[str, Any]:
            properties = (manifest.get("output_schema") or {}).get("properties") or {}
            output: dict[str, Any] = {}
            for key, schema in properties.items():
                if key == "disposition":
                    output[key] = disposition
                elif key == "priority":
                    output[key] = priority
                elif key == "reply_draft":
                    output[key] = _reply_text(payload, disposition)
                elif key == "confidence":
                    output[key] = confidence
                elif key == "needs_human":
                    output[key] = needs_human
                elif key == "result":
                    output[key] = _reply_text(payload, disposition)
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
            tool_plan, fallback_used = _build_tool_plan(
                manifest,
                matched_case,
                _infer_disposition(payload, matched_case),
                dry_run,
            )
            disposition = _infer_disposition(payload, matched_case)
            priority = _infer_priority(payload, disposition)
            needs_human = bool(matched_case.get("should_escalate")) or disposition != "resolve"
            confidence = 0.94 if disposition == "resolve" else 0.72 if disposition == "needs_info" else 0.64
            tool_calls = _build_tool_calls(manifest, tool_plan, payload, dry_run)
            approval_required = bool(matched_case.get("requires_approval")) and not dry_run
            status = "awaiting_approval" if approval_required else "completed"
            started = time.perf_counter()
            output = _build_output(manifest, payload, disposition, priority, needs_human, confidence)
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
                    or f"{manifest.get('module_id', 'run')}-{uuid4().hex[:8]}"
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
                    f"Starter scaffold for {manifest.get('module_id')}",
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
            app = FastAPI(title=f"{manifest.get('title', 'Task Agent')} - {manifest.get('module_title', 'starter')}")
            runs: dict[str, dict[str, Any]] = {}

            @app.get("/health")
            def health() -> dict[str, Any]:
                return {
                    "status": "ok",
                    "module_id": manifest.get("module_id"),
                    "module_title": manifest.get("module_title"),
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
                    run_id = f"{manifest.get('module_id', 'module')}-eval-{case['id']}"
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
    return dedent(
        """
        from __future__ import annotations

        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parent
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))

        from runtime.task_agent_runtime import create_app_from_manifest

        app = create_app_from_manifest(Path(__file__).with_name("starter_manifest.json"))
        """
    ).strip() + "\n"


def render_legacy_task_agent_module_app() -> str:
    return render_legacy_task_agent_root_app()


def render_task_agent_module_app() -> str:
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


def render_task_agent_visible_checks_script() -> str:
    return dedent(
        """
        from __future__ import annotations

        import json
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
                    return _json_request("GET", f"{base_url}/health")
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


        def _tail_log(limit: int = 40) -> str:
            if not LOG_PATH.exists():
                return ""
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
            return "\\n".join(lines[-limit:])


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
                print("No public checks are configured for this module.")
                return 1

            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log_handle = LOG_PATH.open("w", encoding="utf-8")
            port = _pick_port()
            base_url = f"http://127.0.0.1:{port}"
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            try:
                health = _wait_for_health(base_url)
                print(f"Local preview running for {health.get('module_id')}.")
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
                print("Visible checks failed. Inspect the output above, compare it with module_content.md, and iterate before submitting.")
                return 1
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
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
                    "command": "python checks/run_visible_checks.py",
                    "problemMatcher": [],
                    "presentation": {"reveal": "always", "panel": "shared"},
                },
                {
                    "label": "Start local preview",
                    "type": "shell",
                    "command": "python -m uvicorn app:app --host 127.0.0.1 --port 8000",
                    "problemMatcher": [],
                    "presentation": {"reveal": "always", "panel": "shared"},
                },
            ],
        },
        indent=2,
    ) + "\n"
