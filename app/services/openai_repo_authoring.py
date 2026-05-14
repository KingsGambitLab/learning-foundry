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
from app.services.runtime_normalization import normalize_runtime_protocol_dict
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
        # Bumped from 300s — repo authoring on Anthropic Sonnet 4.6 with
        # max_tokens=16000 (Dockerfile + install.sh + full app source +
        # dependency contract) can exceed 5 min wall-clock on cold paths.
        request_timeout_s: float = 600.0,
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
                        "NATIVE BUILD TOOLCHAIN DEPENDENCIES (language-agnostic): Some package installs trigger native code compilation that requires build tools INSIDE the Docker image. The package manager itself is NOT enough — the C toolchain and language-specific build prerequisites must be present BEFORE the install step, either baked into the base image or added in `install.sh`. PROACTIVELY include the relevant build toolchain on the FIRST authoring pass whenever the manifest pulls in a historically native-compiling dependency; do not wait for a failed attempt. Canonical signals and fixes by ecosystem: "
                        "(1) npm/yarn/pnpm: `gyp ERR! stack Error: \\`gyp\\` failed with exit code: 1`, `ModuleNotFoundError: No module named 'gyp'`, `g++: not found`, `make: not found`. Common triggers are `better-sqlite3`, `bcrypt`, `node-pty`, `sharp`, `canvas`, `node-sass`, `node-gyp`, and ANY transitive dep pulled in by `promptfoo`, `playwright`, or `puppeteer`. FIX: either choose a Debian-based Node base image (`node:20-bookworm` / `node:20`, NOT `node:20-alpine` / `node:20-slim`), OR add `apt-get update && apt-get install -y --no-install-recommends python3 make g++` to `install.sh` BEFORE `npm install`. "
                        "(2) pip C-extension wheels: `error: command 'gcc' failed`, `Failed building wheel for {Pillow,lxml,psycopg2,cryptography,bcrypt}`, `fatal error: Python.h: No such file or directory`, `pg_config executable not found`. Triggers: `Pillow`, `lxml`, `psycopg2` (not `psycopg2-binary`), `cryptography` on non-glibc bases, `bcrypt`, `numpy`/`pandas`/`scipy` on minimal images. FIX: add `apt-get install -y build-essential python3-dev libjpeg-dev zlib1g-dev libxml2-dev libxslt1-dev libpq-dev libssl-dev libffi-dev` to `install.sh`, OR pin to versions that ship binary wheels for the base image's CPU arch, OR substitute `psycopg2-binary` for `psycopg2`. "
                        "(3) cargo `*-sys` crates (e.g. `openssl-sys`, `libsqlite3-sys`, `libgit2-sys`): `linking with cc failed`, `pkg-config exited with status code 1`, `could not find {openssl, zlib, sqlite3} in pkg-config`. FIX: `apt-get install -y pkg-config libssl-dev libsqlite3-dev` (plus the relevant `-dev` package for the missing sys lib). "
                        "(4) go cgo: `cgo: C compiler \"gcc\" not found`, `exec: \"gcc\": executable file not found in $PATH`. FIX: switch off `golang:*-alpine` to `golang:*-bookworm`, OR `apk add --no-cache gcc musl-dev` for alpine. "
                        "Treat `gyp failed with exit code: 1` (npm), `Failed building wheel` (pip), `linking with cc failed` (cargo), and `cgo: C compiler ... not found` (go) as missing-build-toolchain signals — NOT as dependency-version problems. Do not respond by downgrading the dependency; respond by adding the toolchain. "
                        "DOCKERFILE FILE AVAILABILITY (language-agnostic): a Dockerfile build only sees files that have been brought into the image via `COPY` or `ADD`. `WORKDIR` sets the working directory but does NOT copy any files into the image. Before ANY `RUN` step that touches a relative path (chmod, cat, ./<script>, sh <script>, npm install, pip install, mvn, cargo, etc.), there MUST be a `COPY` (or `ADD`) step earlier in the Dockerfile that brings that path into the image. The canonical failing pattern is: `FROM python:3.11-slim / WORKDIR /app / RUN chmod +x .coursegen/runtime/install.sh` — this fails because no `COPY` has happened yet, so install.sh is not in /app inside the container even though it exists in the build context. The canonical FIX is: `FROM python:3.11-slim / WORKDIR /app / COPY . . / RUN chmod +x .coursegen/runtime/install.sh / RUN ./.coursegen/runtime/install.sh`. If your Dockerfile references `.coursegen/runtime/*.sh`, the application source, the dependency manifest, or any other build-context file, you MUST have a `COPY . .` (or a more selective `COPY <src> <dst>`) step BEFORE the first reference. The signal `chmod: cannot access '<path>': No such file or directory`, `cp: cannot stat '<path>'`, or `./<script>: No such file or directory` DURING the Docker BUILD is missing-COPY, NOT missing-file — do NOT respond by re-authoring the script (it is already present in your `runtime_protocol_files` / `files` output); add the missing `COPY` step to the Dockerfile and keep the script content untouched. "
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
        from app.services.llm_router import usage_summary_from_response

        return bundle, usage_summary_from_response(response, model_id=model_id)

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
                from app.services.llm_router import (
                    LLMTier,
                    get_default_router,
                    messages_to_system_user,
                )

                router = get_default_router()
                system, user = messages_to_system_user(input)
                return router.parse_structured(
                    tier=LLMTier.sonnet,
                    system=system,
                    user=user,
                    text_format=text_format,
                    request_timeout_s=self.request_timeout_s,
                    # Repo authoring emits the full starter bundle —
                    # source files + Dockerfile + install.sh + manifest —
                    # so give the response room.
                    max_tokens=16_000,
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
