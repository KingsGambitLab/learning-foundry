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
from app.services.runtime_contract_surface import (
    dependency_contract_from_manifest,
    is_repo_contract_path,
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


class _GeneratedDependencyContract(BaseModel):
    manifest_paths: list[str] = Field(default_factory=list)
    lockfile_paths: list[str] = Field(default_factory=list)
    toolchain_paths: list[str] = Field(default_factory=list)
    build_support_paths: list[str] = Field(default_factory=list)
    reproducibility_mode: str | None = None


class _GeneratedRepoBundle(BaseModel):
    files: list[_RepoFile] = Field(default_factory=list)
    dependency_contract: _GeneratedDependencyContract
    notes: list[str] = Field(default_factory=list)


class _GeneratedSharedRepoBundle(BaseModel):
    runtime_protocol_files: list[_RepoFile] = Field(default_factory=list)
    files: list[_RepoFile] = Field(default_factory=list)
    dependency_contract: _GeneratedDependencyContract
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
        visible_fixture_files = set(spec.runtime_dependencies.visible_fixture_files)
        client = (
            self._client(
                api_key=config["OPENAI_API_KEY"],
                base_url=config.get("OPENAI_BASE_URL"),
            )
            if self.client_factory is not None
            else None
        )
        model_id = config.get("OPENAI_MODEL") or self.model or "gpt-5.4"
        ordered_requested_ids = [
            deliverable.id
            for deliverable in spec.deliverables
            if deliverable.id in requested_ids
        ]

        if spec.course_structure.shared_codebase and ordered_requested_ids:
            payload = self._progressive_prompt_payload(
                run=run,
                public_root=public_root,
                deliverable_ids=ordered_requested_ids,
                failure_context=failure_context,
            )
            bundle, response_usage = self._generate_progressive_bundle(
                client,
                model_id=model_id,
                api_key=config["OPENAI_API_KEY"],
                base_url=config.get("OPENAI_BASE_URL"),
                payload=payload,
                workflow_run_id=run.id,
                deliverable_ids=ordered_requested_ids,
            )
            progressive_updates, bundle_notes = self._apply_progressive_bundle(
                run=run,
                public_root=public_root,
                workspace_root=workspace_root,
                visible_fixture_files=visible_fixture_files,
                deliverable_ids=ordered_requested_ids,
                bundle=bundle,
            )
            updated_files.extend(progressive_updates)
            notes.extend(bundle_notes)
            usage = merge_ai_usage(usage, response_usage)
        else:
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
                normalized_contract = self._normalize_dependency_contract(
                    bundle.dependency_contract,
                    current_manifest=manifest,
                )
                updated_files.extend(
                    self._replace_repo_files(
                        starter_root=starter_root,
                        manifest=manifest,
                        files=normalized_files,
                        workspace_root=workspace_root,
                        visible_fixture_files=visible_fixture_files,
                    )
                )
                default_starter_files = build_task_agent_starter_files(spec, deliverable.id)
                starter_repo_bundle, runtime_protocol_bundle = self._bundle_state(
                    starter_root=starter_root,
                    manifest=manifest,
                    default_starter_files=default_starter_files,
                    visible_fixture_files=visible_fixture_files,
                )
                manifest["starter_repo_bundle"] = {
                    "generated_for_deliverable": deliverable.id,
                    **starter_repo_bundle,
                }
                manifest["runtime_protocol_bundle"] = {
                    "generated_for_deliverable": deliverable.id,
                    **runtime_protocol_bundle,
                }
                manifest["dependency_contract"] = normalized_contract
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

    def _progressive_prompt_payload(
        self,
        *,
        run: WorkflowRun,
        public_root: Path,
        deliverable_ids: list[str],
        failure_context: FailureContext | None,
    ) -> dict[str, Any]:
        spec = run.artifacts.task_agent_spec
        if spec is None:
            raise ValueError("Task-agent spec is required for progressive repo authoring.")
        if not spec.deliverables:
            raise ValueError("At least one deliverable is required for progressive repo authoring.")
        # Shared starter is now a single root at public/starter/. The hidden
        # manifest lives under private/grader/<first_deliverable_id>/deliverable.json.
        shared_root = public_root / "starter"
        first_deliverable_id = spec.deliverables[0].id
        workspace = run.artifacts.workspace_snapshot
        if workspace is not None:
            shared_manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / first_deliverable_id
                / "deliverable.json"
            )
        else:
            shared_manifest_path = (
                public_root.parent
                / "private"
                / "grader"
                / first_deliverable_id
                / "deliverable.json"
            )
        shared_manifest = json.loads(shared_manifest_path.read_text(encoding="utf-8"))
        prompt_files = build_starter_authoring_payload(
            starter_root=shared_root,
            manifest=shared_manifest,
        )
        deliverable_payloads: list[dict[str, Any]] = []
        for deliverable in spec.deliverables:
            deliverable_payloads.append(
                {
                    "deliverable_id": deliverable.id,
                    "title": deliverable.title,
                    "objective": deliverable.objective,
                    "learning_outcomes": list(deliverable.learning_outcomes),
                    "learner_brief": (
                        deliverable.learner_brief.model_dump(mode="json")
                        if deliverable.learner_brief is not None
                        else None
                    ),
                    "public_checks": [check.model_dump(mode="json") for check in deliverable.public_checks],
                }
            )

        return {
            "workflow_title": run.title,
            "problem_statement": run.intake.problem_statement,
            "shared_codebase": True,
            "course_starter_type": spec.runtime_dependencies.starter_type.value,
            "repair_scope_deliverable_ids": deliverable_ids,
            "shared_repo_root": shared_root.name,
            "manifest": shared_manifest,
            "current_files": prompt_files["learner_files"],
            "dependency_contract_files": prompt_files["dependency_contract_files"],
            "shared_runtime_protocol_files": prompt_files["runtime_protocol_files"],
            "public_endpoints": prompt_files["public_endpoints"],
            "deliverables": deliverable_payloads,
            "failure_context": failure_context.model_dump(mode="json") if failure_context is not None else None,
        }

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

    def _normalize_dependency_contract(
        self,
        dependency_contract: _GeneratedDependencyContract | dict[str, Any],
        *,
        current_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(dependency_contract, dict):
            dependency_contract = _GeneratedDependencyContract.model_validate(dependency_contract)
        current = dependency_contract_from_manifest(current_manifest)
        normalized: dict[str, Any] = {
            "manifest_paths": self._normalize_contract_paths(
                dependency_contract.manifest_paths,
            ),
            "lockfile_paths": self._normalize_contract_paths(
                dependency_contract.lockfile_paths,
            ),
            "toolchain_paths": self._normalize_contract_paths(
                dependency_contract.toolchain_paths,
            ),
            "build_support_paths": self._normalize_contract_paths(
                dependency_contract.build_support_paths,
            ),
            "reproducibility_mode": (
                dependency_contract.reproducibility_mode.strip()
                if isinstance(dependency_contract.reproducibility_mode, str)
                and dependency_contract.reproducibility_mode.strip()
                else current.get("reproducibility_mode")
            ),
        }
        for key in ("manifest_paths", "lockfile_paths", "toolchain_paths", "build_support_paths"):
            if not normalized[key]:
                normalized[key] = list(current.get(key, []))
        return normalized

    def _normalize_contract_paths(
        self,
        paths: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            relative_path = self._normalize_relative_path(raw_path)
            if relative_path is None:
                continue
            if relative_path in seen:
                continue
            seen.add(relative_path)
            normalized.append(relative_path)
        return normalized

    def _normalize_runtime_protocol_files(self, files: list[_RepoFile]) -> dict[str, str]:
        normalized = self._normalize_repo_files(files)
        return {
            relative_path: content
            for relative_path, content in normalized.items()
            if relative_path in _RUNTIME_PROTOCOL_PATHS
        }

    def _generate_progressive_bundle(
        self,
        client,
        *,
        model_id: str,
        api_key: str,
        base_url: str | None,
        payload: dict[str, Any],
        workflow_run_id: str,
        deliverable_ids: list[str],
    ) -> tuple[_GeneratedSharedRepoBundle, AIUsageSummary | None]:
        deliverable_label = (
            f"{deliverable_ids[0]}..{deliverable_ids[-1]}"
            if deliverable_ids
            else "shared_course_repo"
        )
        response = self._create_response_with_retries(
            client,
            model=model_id,
            api_key=api_key,
            base_url=base_url,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are authoring the single shared repo for a progressive shared-codebase course. "
                        "Return JSON only with keys `runtime_protocol_files`, `files`, `dependency_contract`, and optional `notes`. "
                        "`runtime_protocol_files` must contain the complete shared course-level runtime bundle with "
                        "`Dockerfile`, `.coursegen/runtime/install.sh`, `.coursegen/runtime/verify.sh`, and `.coursegen/runtime/run.sh`. "
                        "`files` must contain the complete learner-owned repo and dependency-contract snapshot for the shared course repo, "
                        "but must not repeat the shared runtime protocol files. "
                        "`dependency_contract` must describe the dependency/build contract for that shared repo. "
                        "Treat deliverables only as milestone briefs/tests/gates over the same app, not as separate repo states. "
                        "The course-level `course_starter_type` field in the payload is either `empty` or `partial`. Default to `partial` unless the payload explicitly says `empty`. "
                        "For a `partial` starter: implement the full scaffold — project skeleton, framework wiring, dependency manifests, data schema, repository layer, type definitions, configuration, and any boilerplate the framework needs to boot. "
                        "Leave every API handler, primary service method, and route body as an explicit unimplemented stub that throws or raises a language-appropriate not-implemented exception (`UnsupportedOperationException` in Java/Kotlin, `NotImplementedError` in Python, `errors.New(\"not implemented\")` in Go, `throw new Error('not implemented')` in TypeScript, etc.). "
                        "Do not return placeholder data, default fixtures, hardcoded success responses, or `Optional.empty()` 200s. The starter MUST boot and stay up (health endpoint returns 200), but every business endpoint must throw the not-implemented exception when called. "
                        "The visible and hidden tests are authored separately and are EXPECTED to fail against this starter — that is the whole point. If your starter accidentally implements deliverable logic, the test-strength baseline will catch it. "
                        "For an `empty` starter: author only the minimum project skeleton needed to boot — framework wiring, dependency manifest, application entry, and a single health endpoint. No business code, no schema, no repository layer, no stubs. "
                        "Preserve one package root, one build identity, and one shared module structure across the whole course. "
                        "Do not write `README.md`, `.coursegen/grader/*`, `checks/*`, or `.vscode/*`; those belong to the harness protocol. "
                        "Lockfiles, build artifacts, logs, generated tests, and other harness-managed outputs are intentionally omitted from the prompt and should not be treated as learner-owned source. "
                        "Structured outputs can only carry text files. You cannot bundle binary assets such as `.mvn/wrapper/maven-wrapper.jar`, `gradle/wrapper/gradle-wrapper.jar`, JAR distributions, fonts, images, or compiled artifacts. "
                        "If a build wrapper depends on a binary (e.g. `./mvnw` needs `maven-wrapper.jar`, `./gradlew` needs `gradle-wrapper.jar`), either have `install.sh` download or regenerate that binary before invoking the wrapper, OR drop the wrapper and use the system-installed tool from the creator-selected base image (e.g. `mvn` directly when the base image already provides Maven). Do not write an `install.sh` that invokes a wrapper script whose required binary is not present and not generated. "
                        "The harness provides every dependency service in the runtime plan (e.g. `postgres`, `redis`, `mongodb`) as a separate sidecar container on a shared Docker network. Each service is reachable from the app container by its `service_id` as the hostname using the service's default port (for example `postgres:5432`, `redis:6379`, `mongodb:27017`). The app container has neither docker nor the dependency service binaries installed. "
                        "Do not install, start, or initialize those services inside the app's `Dockerfile`, `install.sh`, `verify.sh`, or `run.sh` — that means no `initdb`, no `pg_ctl start`, no `redis-server`, no `docker run`, and no `apt-get install postgresql-server`. Configure the app to connect to the sidecar hostnames instead. "
                        "Treat the manifest `dependency_contract` as the current contract source of truth and update it when the authored repo changes. "
                        "The shared runtime bundle must stay coherent with the creator-owned stack contract and with the shared repo you return. "
                        "When `failure_context.dependency_contracts` is present, treat those repo/runtime facts as authoritative for the failing deliverables and repair the shared dependency contract coherently instead of guessing from stderr alone. "
                        "When inspecting `failure_context.sandbox.deliverable_reports[*]`, the structured fields below are the canonical diagnostic — the headline `error` field is just a label, not the source of truth: "
                        "`stderr_excerpt` (8KB tail) is the primary error source for install/verify/boot failures; "
                        "`stdout_excerpt` (8KB tail) is the framework boot log (Spring Boot, Logback, gunicorn, structured loggers) — read this when the failure is post-boot and stderr is sparse; "
                        "`exit_state` carries the structured container exit reason — `oom_killed=true` means raise the container memory cap or trim resource use, `exit_code=137` usually pairs with OOM; "
                        "`sidecar_diagnostics` carries every dependency service's stderr/stdout/exit_state keyed by service_id — check these first when the app shows 'connection refused' or 'no such host', because the real cause is often a postgres/redis sidecar that crashed or OOM-killed; "
                        "`http_response` is only present for contract/checks failures and carries the verbatim response body+status+headers — the response_body_text is the canonical diagnostic for those stages. "
                        "When `failure_context.previously_verified_runtime` is present, those files already passed sandbox verification for the listed deliverables. "
                        "For every path listed in `failure_context.previously_verified_runtime.verified_files`, use the provided verified content as the preservation source of truth and emit that file verbatim in your response unless the current failure packet explicitly proves it caused the new failure. "
                        "When `failure_context.last_attempted_runtime` is present, its `stage_outcomes` shows which harness stages (`image_build`, `install`, `verify`, `boot`, `contract`, `checks`) passed or failed in the most recent attempt, even if the overall sandbox failed. "
                        "Treat every stage marked `passed` as verified by the harness: do not change files that contributed to that stage unless the current failure packet proves they caused the new failure. "
                        "Specifically: if `boot` passed, preserve the runtime protocol bundle (`Dockerfile`, install/verify/run scripts) verbatim from `last_attempted_runtime.verified_files`; if `install` passed, preserve dependency-contract files. Use those `verified_files` entries (with `preserve_verbatim=true`) as the byte-for-byte source for those paths. "
                        "If the only failed stage is `contract` or `checks`, do not touch the runtime protocol bundle or the dependency contract — fix the learner-owned source so the published smoke check can exercise it instead. "
                        "Prefer preserving the exact matching content already present in `shared_runtime_protocol_files`, `dependency_contract_files`, and `current_files` when those paths overlap. "
                        "Do not rewrite known-good runtime wiring or shared config to fix reviewer-only findings. "
                        "Dependency manifests must be coherent with the chosen language, framework, package manager, and versions. "
                        "Do not rely on unbounded latest dependency resolution; pin dependency versions and editions that the chosen toolchain can build today. "
                        "TOOLCHAIN VERSION MISMATCH (language-agnostic): if a sandbox stage fails with a message of the form `running X, requires Y` where Y > X (e.g. Go `requires go >= 1.23 (running go 1.22.4)`, Python `requires-python >= 3.12` against a `python:3.11` base, Node `engines.node >= 20` against `FROM node:18`, Java `requires JDK 21` against `eclipse-temurin:17`, Rust `requires rustc >= 1.78` against `FROM rust:1.75`, .NET `requires net8.0` against `mcr.microsoft.com/dotnet/sdk:7.0`), the CANONICAL FIX is to (1) bump the Dockerfile `FROM` line to a base image that ships version Y or higher, AND (2) raise the in-manifest version constraint (`go.mod` `go` directive, `pyproject.toml` `requires-python`, `package.json` `engines`, `pom.xml`/`build.gradle` `<release>` or `sourceCompatibility`, `Cargo.toml` `rust-version`, `*.csproj` `TargetFramework`) to Y. Do not attempt to pin transitive dependencies to lower versions to work around it — toolchain-version requirements are non-negotiable upstream signals. Do not downgrade or constrain the dependency that surfaced the requirement; bump the toolchain. "
                        "LOCKFILE INTEGRITY MISMATCH (language-agnostic): if a sandbox stage fails with a hash/checksum mismatch against a lockfile (Go `checksum mismatch` / `bits may have been replaced on the origin server`, npm `EINTEGRITY: sha512-... integrity checksum failed` from `npm ci`, pnpm `ERR_PNPM_LOCKFILE_BREAKING_CHANGE` / lockfile integrity errors, Yarn `Hash mismatch detected`, Cargo `the lockfile ... is corrupt`, Poetry `Lock file is not compatible`, pip `THESE PACKAGES DO NOT MATCH THE HASHES`), DO NOT author lockfile entries by hand with synthesized hashes. Lockfile content hashes are produced by the package registry and cannot be authored — they must come from a real download. The CANONICAL FIX is to (1) author `install.sh` so it REGENERATES the lockfile from the manifest using the language's native refresh command (`go mod tidy && go mod download` for Go — never hand-write `go.sum`; `npm install` (NOT `npm ci`) when lockfile is stale; `pnpm install --no-frozen-lockfile`; `cargo generate-lockfile`; `poetry lock --no-update`; `pip-compile` for pip-tools), and (2) if a stale lockfile keeps causing integrity failures, DELETE it inside the install script and let the install command rematerialize it from scratch. Lockfile-integrity errors are NEVER fixed by editing the lockfile — only by regenerating it from the manifest. "
                        "BEFORE returning the bundle, perform an import-vs-manifest audit on every code file you author: walk every source file, collect every `import X` / `from X import ...` (Python), `require('X')` / `import ... from 'X'` (Node), `import \"X\"` (Go), `use X::...` / `extern crate X` (Rust), `import X.Y.Z` (Java/Kotlin) of a non-stdlib package, and confirm each one has a matching pinned entry in the dependency manifest (`requirements.txt`, `package.json` + `package-lock.json`, `go.mod` + `go.sum`, `Cargo.toml` + `Cargo.lock`, `pom.xml` / `build.gradle`). Common omissions that cost a full retry cycle: `pydantic_settings`, `psycopg2`/`psycopg`, `redis`, `httpx`, `sqlalchemy`, `alembic`, `uvicorn[standard]` extras, `python-jose`, `passlib` — if your code imports it, the manifest must list it. Do NOT ship code that imports a package you haven't pinned. "
                        "When the ecosystem supports a lockfile, author the install script so it can generate or refresh that lockfile deterministically inside the creator-selected base image, "
                        "and keep the checked-in dependency contract consistent with that install step so transitive dependency resolution stays reproducible under retries and fresh builds. "
                        "Use `install.sh` for dependency setup and dependency-contract materialization. "
                        "Use `verify.sh` only for essential dependency/build/runtime sanity checks needed after install and before boot. "
                        "PATH CONVENTION: every `files[].path` you return MUST be relative to the shared starter root (the directory described by `shared_repo_root`), NOT prefixed with the repo name. The shared starter root is already named `starter` — prefixing your paths with `starter/` will create a duplicate `starter/starter/` directory and the reviewer will report your editable files as missing. Examples: emit `cmd/server/main.go`, NOT `starter/cmd/server/main.go`; emit `pom.xml`, NOT `starter/pom.xml`; emit `Dockerfile`, NOT `starter/Dockerfile`. If your language's module name happens to be `starter` (common for Go: `module starter`), keep that module name in import statements but DO NOT mirror it in the file layout — files live at the starter root. "
                        "Do not invent internal platform hooks or manifest-driven runtime behavior."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            temperature=0.1,
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable_label,
            text_format=_GeneratedSharedRepoBundle,
        )
        bundle = response.output_parsed
        if bundle is None:
            raise ValueError("OpenAI shared repo authoring returned no parsed bundle.")
        log_coursegen_event(
            "workspace_repo_authoring_shared_completed",
            workflow_run_id=workflow_run_id,
            deliverable_id=deliverable_label,
            model_id=model_id,
            repo_file_count=len(bundle.files),
            runtime_file_count=len(bundle.runtime_protocol_files),
        )
        return bundle, extract_openai_usage(response, model_id)

    def _apply_progressive_bundle(
        self,
        *,
        run: WorkflowRun,
        public_root: Path,
        workspace_root: Path,
        visible_fixture_files: set[str],
        deliverable_ids: list[str],
        bundle: _GeneratedSharedRepoBundle,
    ) -> tuple[list[str], list[str]]:
        spec = run.artifacts.task_agent_spec
        if spec is None:
            raise ValueError("Task-agent spec is required for progressive repo authoring.")

        normalized_runtime_files = self._normalize_runtime_protocol_files(bundle.runtime_protocol_files)
        normalized_repo_files = {
            relative_path: content
            for relative_path, content in self._normalize_repo_files(bundle.files).items()
            if relative_path not in _RUNTIME_PROTOCOL_PATHS
        }
        updated_files: list[str] = []
        notes = list(bundle.notes)

        if not spec.deliverables:
            return updated_files, notes

        # Write the authored files ONCE to the shared starter root.
        shared_starter_root = public_root / "starter"
        first_deliverable = spec.deliverables[0]
        default_starter_files = build_task_agent_starter_files(spec, first_deliverable.id)
        first_manifest_path = (
            workspace_root
            / "private"
            / "grader"
            / first_deliverable.id
            / "deliverable.json"
        )
        first_manifest: dict[str, Any] = {}
        if first_manifest_path.exists():
            try:
                first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                first_manifest = {}

        files_to_apply = {
            **normalized_repo_files,
            **normalized_runtime_files,
        }
        if files_to_apply or shared_starter_root.exists():
            updated_files.extend(
                self._replace_repo_files(
                    starter_root=shared_starter_root,
                    manifest=first_manifest,
                    files=files_to_apply,
                    workspace_root=workspace_root,
                    visible_fixture_files=visible_fixture_files,
                )
            )

        # Update every per-deliverable manifest with the new dependency contract
        # and bundle-state metadata. Manifests now live at
        # private/grader/<id>/deliverable.json.
        for deliverable in spec.deliverables:
            manifest_path = (
                workspace_root
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            normalized_contract = self._normalize_dependency_contract(
                bundle.dependency_contract,
                current_manifest=manifest,
            )
            starter_repo_bundle, _ = self._bundle_state(
                starter_root=shared_starter_root,
                manifest=manifest,
                default_starter_files=default_starter_files,
                visible_fixture_files=visible_fixture_files,
            )
            manifest["starter_repo_bundle"] = {
                "generated_for_deliverable": deliverable.id,
                **starter_repo_bundle,
            }
            manifest["dependency_contract"] = normalized_contract
            if normalized_runtime_files:
                manifest["runtime_protocol_bundle"] = {
                    "generated_for_deliverable": deliverable.id,
                    **self._runtime_bundle_state(
                        default_starter_files=default_starter_files,
                        authored_runtime_files=normalized_runtime_files,
                    ),
                }
            updated_files.extend(
                self._write_if_changed(
                    manifest_path,
                    json.dumps(manifest, indent=2) + "\n",
                    workspace_root,
                )
            )

        return updated_files, notes

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
                        "Return JSON only with keys `files`, `dependency_contract`, and optional `notes`. "
                        "Each file must have `path` and `content`. "
                        "The `dependency_contract` must explicitly list the relative paths that define dependency resolution and build invocation for this starter: "
                        "`manifest_paths`, `lockfile_paths`, `toolchain_paths`, `build_support_paths`, and optional `reproducibility_mode`. "
                        "Return the complete current snapshot for every learner-owned file, dependency-contract file, and runtime protocol file that belongs in the starter workspace, "
                        "not just the files you changed in this attempt. "
                        "Author the real repo files needed to boot under the creator-owned stack contract, including "
                        "`Dockerfile` and `.coursegen/runtime/*.sh` when needed. "
                        "Do not write `README.md`, `.coursegen/grader/*`, `checks/*`, or `.vscode/*`; those belong to the harness protocol. "
                        "Use `current_files` as the learner-owned editable baseline, `dependency_contract_files` for manifests/toolchain files, "
                        "and `runtime_protocol_files` for the authored Docker/install/verify/run bundle during retries; preserve or revise them intentionally rather than starting over blindly. "
                        "Treat the existing manifest `dependency_contract` as the current contract source of truth and update it when the authored repo changes. "
                        "For shared progressive codebases, treat later deliverables as the next milestone of the same repo lineage rather than a fresh app: preserve package roots, build identity, and core repo structure unless the prompt explicitly changes them. "
                        "Lockfiles, build artifacts, generated tests, and other harness-managed outputs are intentionally omitted from the prompt and should not be treated as learner-owned source. "
                        "Write a believable partial implementation, not a hidden simulator. "
                        "Use the exact stack contract and public endpoints from the prompt. "
                        "When `failure_context.dependency_contracts` is present, treat those repo/runtime facts as authoritative for the failed deliverables and repair the dependency contract coherently instead of guessing from stderr alone. "
                        "When inspecting `failure_context.sandbox.deliverable_reports[*]`, the structured fields below are the canonical diagnostic — the headline `error` field is just a label, not the source of truth: "
                        "`stderr_excerpt` (8KB tail) is the primary error source for install/verify/boot failures; "
                        "`stdout_excerpt` (8KB tail) is the framework boot log (Spring Boot, Logback, gunicorn, structured loggers) — read this when the failure is post-boot and stderr is sparse; "
                        "`exit_state` carries the structured container exit reason — `oom_killed=true` means raise the container memory cap or trim resource use, `exit_code=137` usually pairs with OOM; "
                        "`sidecar_diagnostics` carries every dependency service's stderr/stdout/exit_state keyed by service_id — check these first when the app shows 'connection refused' or 'no such host', because the real cause is often a postgres/redis sidecar that crashed or OOM-killed; "
                        "`http_response` is only present for contract/checks failures and carries the verbatim response body+status+headers — the response_body_text is the canonical diagnostic for those stages. "
                        "When `failure_context.previously_verified_runtime` is present, those files already passed sandbox verification for the listed deliverables. "
                        "For every path listed in `failure_context.previously_verified_runtime.verified_files`, use the provided verified content as the preservation source of truth and emit that file verbatim in your response unless the current failure packet explicitly proves it caused the new failure. "
                        "When `failure_context.last_attempted_runtime` is present, its `stage_outcomes` shows which harness stages (`image_build`, `install`, `verify`, `boot`, `contract`, `checks`) passed or failed in the most recent attempt, even if the overall sandbox failed. "
                        "Treat every stage marked `passed` as verified by the harness: do not change files that contributed to that stage unless the current failure packet proves they caused the new failure. "
                        "Specifically: if `boot` passed, preserve the runtime protocol bundle (`Dockerfile`, install/verify/run scripts) verbatim from `last_attempted_runtime.verified_files`; if `install` passed, preserve dependency-contract files. Use those `verified_files` entries (with `preserve_verbatim=true`) as the byte-for-byte source for those paths. "
                        "If the only failed stage is `contract` or `checks`, do not touch the runtime protocol bundle or the dependency contract — fix the learner-owned source so the published smoke check can exercise it instead. "
                        "Prefer preserving the exact matching content already present in `runtime_protocol_files`, `dependency_contract_files`, and `current_files` when those paths overlap. "
                        "Do not rewrite known-good runtime wiring or shared config to fix reviewer-only findings. "
                        "Dependency manifests must be coherent with the chosen language, framework, package manager, and versions. "
                        "Do not rely on unbounded latest dependency resolution; pin dependency versions and editions that the chosen toolchain can build today. "
                        "TOOLCHAIN VERSION MISMATCH (language-agnostic): if a sandbox stage fails with a message of the form `running X, requires Y` where Y > X (e.g. Go `requires go >= 1.23 (running go 1.22.4)`, Python `requires-python >= 3.12` against a `python:3.11` base, Node `engines.node >= 20` against `FROM node:18`, Java `requires JDK 21` against `eclipse-temurin:17`, Rust `requires rustc >= 1.78` against `FROM rust:1.75`, .NET `requires net8.0` against `mcr.microsoft.com/dotnet/sdk:7.0`), the CANONICAL FIX is to (1) bump the Dockerfile `FROM` line to a base image that ships version Y or higher, AND (2) raise the in-manifest version constraint (`go.mod` `go` directive, `pyproject.toml` `requires-python`, `package.json` `engines`, `pom.xml`/`build.gradle` `<release>` or `sourceCompatibility`, `Cargo.toml` `rust-version`, `*.csproj` `TargetFramework`) to Y. Do not attempt to pin transitive dependencies to lower versions to work around it — toolchain-version requirements are non-negotiable upstream signals. Do not downgrade or constrain the dependency that surfaced the requirement; bump the toolchain. "
                        "LOCKFILE INTEGRITY MISMATCH (language-agnostic): if a sandbox stage fails with a hash/checksum mismatch against a lockfile (Go `checksum mismatch` / `bits may have been replaced on the origin server`, npm `EINTEGRITY: sha512-... integrity checksum failed` from `npm ci`, pnpm `ERR_PNPM_LOCKFILE_BREAKING_CHANGE` / lockfile integrity errors, Yarn `Hash mismatch detected`, Cargo `the lockfile ... is corrupt`, Poetry `Lock file is not compatible`, pip `THESE PACKAGES DO NOT MATCH THE HASHES`), DO NOT author lockfile entries by hand with synthesized hashes. Lockfile content hashes are produced by the package registry and cannot be authored — they must come from a real download. The CANONICAL FIX is to (1) author `install.sh` so it REGENERATES the lockfile from the manifest using the language's native refresh command (`go mod tidy && go mod download` for Go — never hand-write `go.sum`; `npm install` (NOT `npm ci`) when lockfile is stale; `pnpm install --no-frozen-lockfile`; `cargo generate-lockfile`; `poetry lock --no-update`; `pip-compile` for pip-tools), and (2) if a stale lockfile keeps causing integrity failures, DELETE it inside the install script and let the install command rematerialize it from scratch. Lockfile-integrity errors are NEVER fixed by editing the lockfile — only by regenerating it from the manifest. "
                        "BEFORE returning the bundle, perform an import-vs-manifest audit on every code file you author: walk every source file, collect every `import X` / `from X import ...` (Python), `require('X')` / `import ... from 'X'` (Node), `import \"X\"` (Go), `use X::...` / `extern crate X` (Rust), `import X.Y.Z` (Java/Kotlin) of a non-stdlib package, and confirm each one has a matching pinned entry in the dependency manifest (`requirements.txt`, `package.json` + `package-lock.json`, `go.mod` + `go.sum`, `Cargo.toml` + `Cargo.lock`, `pom.xml` / `build.gradle`). Common omissions that cost a full retry cycle: `pydantic_settings`, `psycopg2`/`psycopg`, `redis`, `httpx`, `sqlalchemy`, `alembic`, `uvicorn[standard]` extras, `python-jose`, `passlib` — if your code imports it, the manifest must list it. Do NOT ship code that imports a package you haven't pinned. "
                        "When the ecosystem supports a lockfile, author the install script so it can generate or refresh that lockfile deterministically inside the creator-selected base image, "
                        "and keep the checked-in dependency contract consistent with that install step so transitive dependency resolution stays reproducible under retries and fresh builds. "
                        "Do not hand-write fragile lockfile bodies that only work in one snapshot; prefer manifests plus install/build steps that can materialize the dependency contract repeatably. "
                        "If you author any runtime protocol file, author the full runtime bundle coherently: `Dockerfile`, install script, verify script, and run script. "
                        "The authored runtime bundle must be self-consistent: every command used by `.coursegen/runtime/*.sh` must be available from the authored Dockerfile and install script without relying on shell profile side effects. "
                        "Use `install.sh` for dependency setup and dependency-contract materialization. "
                        "Use `verify.sh` only for essential dependency/build/runtime sanity checks needed after install and before boot. "
                        "PATH CONVENTION: every `files[].path` you return MUST be relative to the shared starter root (the directory described by `shared_repo_root`), NOT prefixed with the repo name. The shared starter root is already named `starter` — prefixing your paths with `starter/` will create a duplicate `starter/starter/` directory and the reviewer will report your editable files as missing. Examples: emit `cmd/server/main.go`, NOT `starter/cmd/server/main.go`; emit `pom.xml`, NOT `starter/pom.xml`; emit `Dockerfile`, NOT `starter/Dockerfile`. If your language's module name happens to be `starter` (common for Go: `module starter`), keep that module name in import statements but DO NOT mirror it in the file layout — files live at the starter root. "
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
        # Defensive strip: the model is told `shared_repo_root: "starter"`
        # in the prompt payload. For some stacks (notably Go, where the
        # natural module name `starter` mirrors the repo root name), the
        # model prefixes every authored path with `starter/` thinking
        # they're workspace-relative. The writer expects them
        # starter-root-relative, so the prefix would yield
        # `public/starter/starter/cmd/server/main.go` — one directory
        # too deep, and reviewer_code reports the files as missing.
        # Strip the literal prefix (only with the trailing slash, to
        # avoid false-stripping lookalikes like `starters_helper/`).
        if normalized.startswith("starter/"):
            normalized = normalized[len("starter/"):]
        if normalized in {".", "", "starter"} or normalized.startswith("../") or normalized.startswith("/"):
            return None
        if normalized in _RESERVED_PATHS or normalized.startswith(_RESERVED_PREFIXES):
            return None
        if not is_repo_contract_path(normalized) and normalized not in _RUNTIME_PROTOCOL_PATHS:
            return None
        return normalized

    def _replace_repo_files(
        self,
        *,
        starter_root: Path,
        manifest: dict[str, Any],
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

        prompt_files = build_starter_authoring_payload(
            starter_root=starter_root,
            manifest=manifest,
        )
        managed_paths = {
            *prompt_files["learner_files"],
            *prompt_files["dependency_contract_files"],
            *prompt_files["runtime_protocol_files"],
        }
        managed_paths.update(files)
        obsolete_paths = [
            path
            for path in sorted(existing_paths - set(files))
            if path in managed_paths or not is_repo_contract_path(path)
        ]

        for obsolete_path in obsolete_paths:
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
        authored_files: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        existing_files = dict(authored_files or {})
        if not existing_files:
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
        runtime_bundle = self._runtime_bundle_state(
            default_starter_files=default_starter_files,
            authored_runtime_files={
                relative_path: existing_files[relative_path]
                for relative_path in runtime_authored_paths
            },
        )
        return (
            {
                "source": "openai_live" if repo_complete else "starter_default",
                "authored_paths": repo_files,
            },
            runtime_bundle,
        )

    def _runtime_bundle_state(
        self,
        *,
        default_starter_files: dict[str, str],
        authored_runtime_files: dict[str, str],
    ) -> dict[str, Any]:
        runtime_authored_paths = sorted(
            relative_path
            for relative_path in _RUNTIME_PROTOCOL_PATHS
            if relative_path in authored_runtime_files
            and authored_runtime_files[relative_path] != default_starter_files.get(relative_path, "")
        )
        runtime_complete = len(runtime_authored_paths) == len(_RUNTIME_PROTOCOL_PATHS)
        return {
            "source": "openai_live" if runtime_complete else "starter_default",
            "authored_paths": runtime_authored_paths,
        }

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
        if content.startswith("#!"):
            current_mode = path.stat().st_mode
            path.chmod(current_mode | 0o111)
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
