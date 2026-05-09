from __future__ import annotations

import json
import os
import queue
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.domain.ai import AIUsageSummary, merge_ai_usage
from app.domain.workflow import FailureContext, WorkflowRun
from app.services.coursegen_logging import log_coursegen_event
from app.services.openai_runtime_support import (
    extract_openai_usage,
    load_openai_env_file,
    resolve_openai_env_file,
)
from app.services.task_agent_starter_templates import HIDDEN_MANIFEST_PATH

VISIBLE_TEST_SCRIPT_PATH = "checks/run_visible_checks.py"
HIDDEN_TEST_SCRIPT_PATH = ".coursegen/grader/run_hidden_checks.py"


class TestScriptAuthoringSource(str, Enum):
    openai_live = "openai_live"
    unavailable = "unavailable"


class TestScriptAuthoringResult(BaseModel):
    source: TestScriptAuthoringSource
    updated_files: list[str] = Field(default_factory=list)
    usage: AIUsageSummary | None = None
    notes: list[str] = Field(default_factory=list)
    message: str
    available: bool = False


class _GeneratedScripts(BaseModel):
    visible_script: str
    hidden_script: str
    notes: list[str] = Field(default_factory=list)


class OpenAITestScriptAuthoringService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        env_file: str | None = None,
        model: str | None = None,
        client_factory=None,
        request_timeout_s: float = 90.0,
        max_request_retries: int = 2,
    ) -> None:
        self.enabled = enabled
        self.env_file = resolve_openai_env_file(env_file)
        self.model = model
        self.client_factory = client_factory
        self.request_timeout_s = request_timeout_s
        self.max_request_retries = max(0, max_request_retries)

    def author_workspace_tests(
        self,
        run: WorkflowRun,
        *,
        failure_context: FailureContext | None = None,
        deliverable_ids: list[str] | None = None,
    ) -> tuple[WorkflowRun, TestScriptAuthoringResult]:
        spec = run.artifacts.task_agent_spec
        workspace = run.artifacts.workspace_snapshot
        if spec is None or workspace is None:
            return run, TestScriptAuthoringResult(
                source=TestScriptAuthoringSource.unavailable,
                updated_files=[],
                usage=None,
                notes=[],
                message="Workspace test authoring skipped because the spec or workspace is missing.",
                available=False,
            )

        config = self._config()
        if not self.enabled or not self._openai_sdk_available() or not config.get("OPENAI_API_KEY"):
            return run, TestScriptAuthoringResult(
                source=TestScriptAuthoringSource.unavailable,
                updated_files=[],
                usage=None,
                notes=[],
                message="OpenAI test authoring is unavailable, so the existing learner test scripts were left in place.",
                available=False,
            )

        requested_ids = set(deliverable_ids or [deliverable.id for deliverable in spec.deliverables])
        updated_files: list[str] = []
        usage = AIUsageSummary()
        notes: list[str] = []
        workspace_root = Path(workspace.root_dir)
        public_root = Path(workspace.public_dir)
        client = self._client(
            api_key=config["OPENAI_API_KEY"],
            base_url=config.get("OPENAI_BASE_URL"),
        )
        model_id = config.get("OPENAI_MODEL") or self.model or "gpt-5.4"

        for deliverable in spec.deliverables:
            if deliverable.id not in requested_ids:
                continue
            starter_root = public_root / "starter" / deliverable.id
            manifest_path = starter_root / HIDDEN_MANIFEST_PATH
            if not starter_root.exists() or not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = self._prompt_payload(run, starter_root=starter_root, manifest=manifest, failure_context=failure_context)
            scripts, response_usage = self._generate_scripts(
                client,
                model_id=model_id,
                payload=payload,
                workflow_run_id=run.id,
                deliverable_id=deliverable.id,
            )
            compile(scripts.visible_script, f"{deliverable.id}:{VISIBLE_TEST_SCRIPT_PATH}", "exec")
            compile(scripts.hidden_script, f"{deliverable.id}:{HIDDEN_TEST_SCRIPT_PATH}", "exec")

            visible_path = starter_root / VISIBLE_TEST_SCRIPT_PATH
            hidden_path = starter_root / HIDDEN_TEST_SCRIPT_PATH
            hidden_path.parent.mkdir(parents=True, exist_ok=True)
            updated_files.extend(self._write_if_changed(visible_path, scripts.visible_script, workspace_root))
            updated_files.extend(self._write_if_changed(hidden_path, scripts.hidden_script, workspace_root))

            manifest["visible_check_command"] = "python checks/run_visible_checks.py"
            manifest["hidden_check_command"] = "python .coursegen/grader/run_hidden_checks.py"
            manifest["generated_test_scripts"] = {
                "source": "openai_live",
                "generated_for_deliverable": deliverable.id,
            }
            updated_files.extend(
                self._write_if_changed(
                    manifest_path,
                    json.dumps(manifest, indent=2) + "\n",
                    workspace_root,
                )
            )
            usage = merge_ai_usage(usage, response_usage)
            notes.extend(scripts.notes)

        if usage.request_count:
            run.artifacts.ai_usage = merge_ai_usage(run.artifacts.ai_usage, usage)

        message = (
            "Generated learner-visible and hidden test scripts against the materialized starter workspace."
            if updated_files
            else "Generated test scripts matched the current workspace."
        )
        return run, TestScriptAuthoringResult(
            source=TestScriptAuthoringSource.openai_live,
            updated_files=updated_files,
            usage=usage if usage.request_count else None,
            notes=notes,
            message=message,
            available=True,
        )

    def _prompt_payload(
        self,
        run: WorkflowRun,
        *,
        starter_root: Path,
        manifest: dict[str, Any],
        failure_context: FailureContext | None,
    ) -> dict[str, Any]:
        editable_paths = (
            (manifest.get("learner_starter_surface") or {}).get("primary_editable_paths")
            or (manifest.get("runtime_dependencies") or {}).get("editable_files")
            or ["app.py"]
        )
        support_paths = (
            (manifest.get("learner_starter_surface") or {}).get("support_paths")
            or ["checks/run_visible_checks.py"]
        )
        file_payload: dict[str, str] = {}
        for relative_path in [*editable_paths, *support_paths, "README.md", HIDDEN_MANIFEST_PATH]:
            target = starter_root / str(relative_path)
            if not target.exists() or target.is_dir():
                continue
            try:
                file_payload[str(relative_path)] = target.read_text(encoding="utf-8")
            except OSError:
                continue
        return {
            "workflow_title": run.title,
            "problem_statement": run.intake.problem_statement,
            "starter_root": starter_root.name,
            "manifest": manifest,
            "files": file_payload,
            "failure_context": failure_context.model_dump(mode="json") if failure_context is not None else None,
        }

    def _generate_scripts(
        self,
        client,
        *,
        model_id: str,
        payload: dict[str, Any],
        workflow_run_id: str,
        deliverable_id: str,
    ) -> tuple[_GeneratedScripts, AIUsageSummary | None]:
        response = self._create_response_with_retries(
            client,
            model=model_id,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are writing real Python test scripts for a learner starter workspace. "
                        "Return JSON only with keys `visible_script`, `hidden_script`, and optional `notes`. "
                        "Do not invent a test DSL. Write complete executable Python scripts. "
                        "Scripts must use only the Python standard library. "
                        "Scripts must black-box test the running app over HTTP using the `BASE_URL` environment variable. "
                        "If `REPORT_PATH` is set, write a JSON report there. Otherwise print the same JSON to stdout. "
                        "Report shape: {\"summary\": str, \"tests\": [{\"id\": str, \"title\": str, \"status\": \"passed\"|\"failed\", \"summary\": str, \"diagnostics\": [str]}]}. "
                        "Exit 0 only when every test passes. Exit non-zero when any test fails. "
                        "Visible tests should be learner-friendly and basic. Hidden tests should be materially stronger. "
                        "For `partial_implementation` starters, visible tests should still fail the untouched starter when core behavior is missing. "
                        "For `working_buggy` starters, visible tests may pass but hidden tests should expose the deeper bug. "
                        "Use only the published endpoints and the actual learner files in the prompt. "
                        "Do not import the learner application directly."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            temperature=0.1,
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable_id,
        )
        raw = getattr(response, "output_text", "")
        data = self._extract_json(raw)
        scripts = _GeneratedScripts.model_validate(data)
        log_coursegen_event(
            "workspace_test_authoring_deliverable_completed",
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable_id,
            model_id=model_id,
        )
        return scripts, extract_openai_usage(response, model_id)

    def _create_response_with_retries(
        self,
        client,
        *,
        model: str,
        input: list[dict[str, Any]],
        temperature: float,
        workflow_run_id: str,
        deliverable_id: str,
    ):
        last_error: Exception | None = None
        for attempt in range(1, self.max_request_retries + 2):
            log_coursegen_event(
                "workspace_test_authoring_attempt_started",
                workflow_run_id=workflow_run_id,
                deliverable_id=deliverable_id,
                model_id=model,
                attempt=attempt,
            )
            try:
                return self._run_with_timeout(
                    lambda: client.responses.create(
                        model=model,
                        input=input,
                        temperature=temperature,
                    )
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                log_coursegen_event(
                    "workspace_test_authoring_attempt_failed",
                    workflow_run_id=workflow_run_id,
                    deliverable_id=deliverable_id,
                    model_id=model,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt > self.max_request_retries:
                    break
                time.sleep(min(2**attempt, 4))
        assert last_error is not None
        raise last_error

    def _run_with_timeout(self, fn):
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def target() -> None:
            try:
                result_queue.put(("ok", fn()))
            except Exception as exc:  # pragma: no cover
                result_queue.put(("error", exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            status, payload = result_queue.get(timeout=self.request_timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(
                f"OpenAI test authoring request exceeded {self.request_timeout_s:.0f}s."
            ) from exc
        if status == "error":
            raise payload
        return payload

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            while lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise
            return json.loads(text[start : end + 1])

    def _config(self) -> dict[str, str]:
        config: dict[str, str] = {}
        if self.env_file:
            config.update(load_openai_env_file(self.env_file))
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_BASE_URL",
            "COURSE_GEN_OPENAI_TEST_AUTHORING_MODEL",
        ):
            value = os.environ.get(key)
            if value:
                config[key] = value
        if "OPENAI_MODEL" not in config:
            config["OPENAI_MODEL"] = config.get("COURSE_GEN_OPENAI_TEST_AUTHORING_MODEL") or self.model or "gpt-5.4"
        return config

    def _client(self, *, api_key: str, base_url: str | None):
        if self.client_factory is not None:
            return self.client_factory(api_key=api_key, base_url=base_url)
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def _openai_sdk_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _write_if_changed(self, path: Path, content: str, workspace_root: Path) -> list[str]:
        if path.exists():
            current = path.read_text(encoding="utf-8")
            if current == content:
                return []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return [str(path.relative_to(workspace_root))]
