from __future__ import annotations

import json
import os
import posixpath
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
    parse_structured_openai_response_with_hard_timeout,
    resolve_openai_env_file,
)
from app.services.starter_authoring_payload import build_starter_authoring_payload
from app.services.task_agent_contract_surface import learner_editable_paths_for_manifest
from app.services.task_agent_starter_templates import (
    HIDDEN_GRADER_SCRIPT_PATH,
    HIDDEN_MANIFEST_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    build_task_agent_starter_files,
)


class RepoAuthoringSource(str, Enum):
    openai_live = "openai_live"
    unavailable = "unavailable"


class RepoAuthoringResult(BaseModel):
    source: RepoAuthoringSource
    updated_files: list[str] = Field(default_factory=list)
    usage: AIUsageSummary | None = None
    notes: list[str] = Field(default_factory=list)
    message: str
    available: bool = False


class _RepoFile(BaseModel):
    path: str
    content: str


class _GeneratedRepoBundle(BaseModel):
    files: list[_RepoFile] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


_RESERVED_PATHS = {
    "README.md",
    ".vscode/tasks.json",
    HIDDEN_MANIFEST_PATH,
    HIDDEN_GRADER_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
    "checks/run_visible_checks.py",
}
_RESERVED_PREFIXES = (
    "checks/",
    ".vscode/",
    ".coursegen/grader/",
)
_RUNTIME_PROTOCOL_PATHS = {
    "Dockerfile",
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
}


class OpenAIStarterRepoAuthoringService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        env_file: str | None = None,
        model: str | None = None,
        client_factory=None,
        request_timeout_s: float = 300.0,
        max_request_retries: int = 2,
    ) -> None:
        self.enabled = enabled
        self.env_file = resolve_openai_env_file(env_file)
        self.model = model
        self.client_factory = client_factory
        self.request_timeout_s = request_timeout_s
        self.max_request_retries = max(0, max_request_retries)

    def author_workspace_repo(
        self,
        run: WorkflowRun,
        *,
        failure_context: FailureContext | None = None,
        deliverable_ids: list[str] | None = None,
    ) -> tuple[WorkflowRun, RepoAuthoringResult]:
        spec = run.artifacts.task_agent_spec
        workspace = run.artifacts.workspace_snapshot
        if spec is None or workspace is None:
            return run, RepoAuthoringResult(
                source=RepoAuthoringSource.unavailable,
                updated_files=[],
                usage=None,
                notes=[],
                message="Repo authoring skipped because the spec or workspace is missing.",
                available=False,
            )

        config = self._config()
        if not self.enabled or not self._openai_sdk_available() or not config.get("OPENAI_API_KEY"):
            return run, RepoAuthoringResult(
                source=RepoAuthoringSource.unavailable,
                updated_files=[],
                usage=None,
                notes=[],
                message="OpenAI repo authoring is unavailable, so the starter repo was left as protocol-only scaffolding.",
                available=False,
            )

        requested_ids = set(deliverable_ids or [deliverable.id for deliverable in spec.deliverables])
        updated_files: list[str] = []
        usage = AIUsageSummary()
        notes: list[str] = []
        workspace_root = Path(workspace.root_dir)
        public_root = Path(workspace.public_dir)
        client = (
            self._client(
                api_key=config["OPENAI_API_KEY"],
                base_url=config.get("OPENAI_BASE_URL"),
            )
            if self.client_factory is not None
            else None
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
            payload = self._prompt_payload(
                run,
                deliverable_id=deliverable.id,
                starter_root=starter_root,
                manifest=manifest,
                failure_context=failure_context,
            )
            bundle, response_usage = self._generate_bundle(
                client,
                model_id=model_id,
                api_key=config["OPENAI_API_KEY"],
                base_url=config.get("OPENAI_BASE_URL"),
                payload=payload,
                workflow_run_id=run.id,
                deliverable_id=deliverable.id,
            )
            normalized_files = self._normalize_repo_files(bundle.files)
            updated_files.extend(
                self._replace_repo_files(
                    starter_root=starter_root,
                    files=normalized_files,
                    workspace_root=workspace_root,
                    visible_fixture_files=set(spec.runtime_dependencies.visible_fixture_files),
                )
            )
            default_starter_files = build_task_agent_starter_files(spec, deliverable.id)
            starter_repo_bundle, runtime_protocol_bundle = self._bundle_state(
                starter_root=starter_root,
                manifest=manifest,
                default_starter_files=default_starter_files,
                visible_fixture_files=set(spec.runtime_dependencies.visible_fixture_files),
            )
            manifest["starter_repo_bundle"] = {
                "generated_for_deliverable": deliverable.id,
                **starter_repo_bundle,
            }
            manifest["runtime_protocol_bundle"] = {
                "generated_for_deliverable": deliverable.id,
                **runtime_protocol_bundle,
            }
            updated_files.extend(
                self._write_if_changed(
                    manifest_path,
                    json.dumps(manifest, indent=2) + "\n",
                    workspace_root,
                )
            )
            usage = merge_ai_usage(usage, response_usage)
            notes.extend(bundle.notes)

        if usage.request_count:
            run.artifacts.ai_usage = merge_ai_usage(run.artifacts.ai_usage, usage)

        message = (
            "Authored learner-owned starter repo files against the creator-owned stack contract."
            if updated_files
            else "Starter repo files already matched the current authored bundle."
        )
        return run, RepoAuthoringResult(
            source=RepoAuthoringSource.openai_live,
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
        deliverable_id: str,
        starter_root: Path,
        manifest: dict[str, Any],
        failure_context: FailureContext | None,
    ) -> dict[str, Any]:
        prompt_files = build_starter_authoring_payload(
            starter_root=starter_root,
            manifest=manifest,
        )
        return {
            "workflow_title": run.title,
            "problem_statement": run.intake.problem_statement,
            "deliverable_id": deliverable_id,
            "starter_root": starter_root.name,
            "manifest": manifest,
            "current_files": prompt_files["learner_files"],
            "dependency_contract_files": prompt_files["dependency_contract_files"],
            "runtime_protocol_files": prompt_files["runtime_protocol_files"],
            "public_endpoints": prompt_files["public_endpoints"],
            "failure_context": failure_context.model_dump(mode="json") if failure_context is not None else None,
        }

    def _generate_bundle(
        self,
        client,
        *,
        model_id: str,
        api_key: str,
        base_url: str | None,
        payload: dict[str, Any],
        workflow_run_id: str,
        deliverable_id: str,
    ) -> tuple[_GeneratedRepoBundle, AIUsageSummary | None]:
        response = self._create_response_with_retries(
                client,
                model=model_id,
                api_key=api_key,
                base_url=base_url,
                input=[
                {
                    "role": "system",
                    "content": (
                        "You are authoring the actual learner-owned repo files for one starter workspace. "
                        "Return JSON only with keys `files` and optional `notes`. "
                        "Each file must have `path` and `content`. "
                        "Return the complete current snapshot for every learner-owned file, dependency-contract file, and runtime protocol file that belongs in the starter workspace, "
                        "not just the files you changed in this attempt. "
                        "Author the real repo files needed to boot under the creator-owned stack contract, including "
                        "`Dockerfile` and `.coursegen/runtime/*.sh` when needed. "
                        "Do not write `README.md`, `.coursegen/grader/*`, `checks/*`, or `.vscode/*`; those belong to the harness protocol. "
                        "Use `current_files` as the learner-owned editable baseline, `dependency_contract_files` for manifests/toolchain files, "
                        "and `runtime_protocol_files` for the authored Docker/install/verify/run bundle during retries; preserve or revise them intentionally rather than starting over blindly. "
                        "Lockfiles, build artifacts, generated tests, and other harness-managed outputs are intentionally omitted from the prompt and should not be treated as learner-owned source. "
                        "Write a believable partial implementation, not a hidden simulator. "
                        "Use the exact stack contract and public endpoints from the prompt. "
                        "When `failure_context.dependency_contracts` is present, treat those repo/runtime facts as authoritative for the failed deliverables and repair the dependency contract coherently instead of guessing from stderr alone. "
                        "Dependency manifests must be coherent with the chosen language, framework, package manager, and versions. "
                        "Do not rely on unbounded latest dependency resolution; pin dependency versions and editions that the chosen toolchain can build today. "
                        "When the ecosystem supports a lockfile, author the install script so it can generate or refresh that lockfile deterministically inside the creator-selected base image, "
                        "and keep the checked-in dependency contract consistent with that install step so transitive dependency resolution stays reproducible under retries and fresh builds. "
                        "Do not hand-write fragile lockfile bodies that only work in one snapshot; prefer manifests plus install/build steps that can materialize the dependency contract repeatably. "
                        "If you author any runtime protocol file, author the full runtime bundle coherently: `Dockerfile`, install script, verify script, and run script. "
                        "The authored runtime bundle must be self-consistent: every command used by `.coursegen/runtime/*.sh` must be available from the authored Dockerfile and install script without relying on shell profile side effects. "
                        "Use `install.sh` for dependency setup and dependency-contract materialization. "
                        "Use `verify.sh` only for essential dependency/build/runtime sanity checks needed after install and before boot. "
                        "Do not use `verify.sh` for formatter, linter, or style-only gates unless the creator contract explicitly requires them and the authored runtime bundle installs those tools. "
                        "Keep the runtime protocol minimal and deterministic so the harness can repair it from sandbox failures. "
                        "Return only relative file paths inside the starter workspace. "
                        "Do not invent internal platform hooks or manifest-driven runtime behavior."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            temperature=0.1,
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable_id,
            text_format=_GeneratedRepoBundle,
        )
        bundle = response.output_parsed
        if bundle is None:
            raise ValueError("OpenAI repo authoring returned no parsed bundle.")
        log_coursegen_event(
            "workspace_repo_authoring_deliverable_completed",
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable_id,
            model_id=model_id,
            file_count=len(bundle.files),
        )
        return bundle, extract_openai_usage(response, model_id)

    def _normalize_repo_files(self, files: list[_RepoFile]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for file in files:
            relative_path = self._normalize_relative_path(file.path)
            if relative_path is None:
                continue
            normalized[relative_path] = file.content
        return normalized

    def _normalize_relative_path(self, raw_path: str) -> str | None:
        candidate = str(raw_path or "").strip().replace("\\", "/")
        if not candidate:
            return None
        normalized = posixpath.normpath(candidate)
        if normalized in {".", ""} or normalized.startswith("../") or normalized.startswith("/"):
            return None
        if normalized in _RESERVED_PATHS or normalized.startswith(_RESERVED_PREFIXES):
            return None
        return normalized

    def _replace_repo_files(
        self,
        *,
        starter_root: Path,
        files: dict[str, str],
        workspace_root: Path,
        visible_fixture_files: set[str],
    ) -> list[str]:
        updated_files: list[str] = []
        existing_paths: set[str] = set()
        for path in starter_root.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(starter_root).as_posix()
            if relative_path in visible_fixture_files:
                continue
            if relative_path in _RESERVED_PATHS or relative_path.startswith(_RESERVED_PREFIXES):
                continue
            existing_paths.add(relative_path)

        for obsolete_path in sorted(existing_paths - set(files)):
            target = starter_root / obsolete_path
            target.unlink(missing_ok=True)
            updated_files.append(str(target.relative_to(workspace_root)))

        for relative_path, content in files.items():
            target = starter_root / relative_path
            updated_files.extend(self._write_if_changed(target, content, workspace_root))
        return updated_files

    def _bundle_state(
        self,
        *,
        starter_root: Path,
        manifest: dict[str, Any],
        default_starter_files: dict[str, str],
        visible_fixture_files: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt_files = build_starter_authoring_payload(
            starter_root=starter_root,
            manifest=manifest,
        )
        existing_files = {
            **prompt_files["learner_files"],
            **prompt_files["dependency_contract_files"],
            **prompt_files["runtime_protocol_files"],
        }
        repo_files = sorted(
            relative_path
            for relative_path in existing_files
            if relative_path not in visible_fixture_files
            and relative_path not in _RESERVED_PATHS
            and not relative_path.startswith(_RESERVED_PREFIXES)
            and relative_path not in _RUNTIME_PROTOCOL_PATHS
        )
        runtime_authored_paths = sorted(
            relative_path
            for relative_path in _RUNTIME_PROTOCOL_PATHS
            if relative_path in existing_files
            and existing_files[relative_path] != default_starter_files.get(relative_path, "")
        )
        editable_paths = learner_editable_paths_for_manifest(manifest)
        repo_complete = (
            bool(editable_paths)
            and all((starter_root / relative_path).exists() for relative_path in editable_paths)
        ) or bool(repo_files)
        runtime_complete = len(runtime_authored_paths) == len(_RUNTIME_PROTOCOL_PATHS)
        return (
            {
                "source": "openai_live" if repo_complete else "starter_default",
                "authored_paths": repo_files,
            },
            {
                "source": "openai_live" if runtime_complete else "starter_default",
                "authored_paths": runtime_authored_paths,
            },
        )

    def _write_if_changed(
        self,
        path: Path,
        content: str,
        workspace_root: Path,
    ) -> list[str]:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return [str(path.relative_to(workspace_root))]

    def _create_response_with_retries(
        self,
        client,
        *,
        model: str,
        api_key: str,
        base_url: str | None,
        input: list[dict[str, Any]],
        temperature: float,
        workflow_run_id: str,
        deliverable_id: str,
        text_format: type[BaseModel],
    ):
        last_error: Exception | None = None
        for attempt in range(1, self.max_request_retries + 2):
            log_coursegen_event(
                "workspace_repo_authoring_attempt_started",
                workflow_run_id=workflow_run_id,
                deliverable_id=deliverable_id,
                model_id=model,
                attempt=attempt,
            )
            try:
                if self.client_factory is not None:
                    return client.responses.parse(
                        model=model,
                        input=input,
                        temperature=temperature,
                        text_format=text_format,
                        timeout=self.request_timeout_s,
                    )
                return parse_structured_openai_response_with_hard_timeout(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    input=input,
                    text_format=text_format,
                    request_timeout_s=self.request_timeout_s,
                    extra_request_kwargs={"temperature": temperature},
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                log_coursegen_event(
                    "workspace_repo_authoring_attempt_failed",
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

    def _config(self) -> dict[str, str]:
        config = load_openai_env_file(self.env_file)
        env_values = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL"),
        }
        for key, value in env_values.items():
            if value:
                config[key] = value
        return config

    def _client(self, *, api_key: str, base_url: str | None = None):
        if self.client_factory is not None:
            return self.client_factory(api_key=api_key, base_url=base_url)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The OpenAI Python SDK is not installed.") from exc
        kwargs = {"api_key": api_key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def _openai_sdk_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True
