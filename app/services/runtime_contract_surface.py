from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.task_agent import ProjectRuntimePlanSpec
from app.domain.workflow import FailureContextDependencyContract
from app.services.task_agent_contract_surface import learner_editable_paths_for_manifest
from app.services.task_agent_starter_templates import (
    HIDDEN_MANIFEST_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
)

STARTER_RUNTIME_PROTOCOL_PATHS = [
    "Dockerfile",
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
]

STARTER_RUNTIME_CHECK_SCRIPT_PATHS = [
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
]

STARTER_SUPPORT_PATHS = [
    HIDDEN_MANIFEST_PATH,
    "checks/run_visible_checks.py",
    ".coursegen/grader/run_hidden_checks.py",
    ".vscode/tasks.json",
]

_DEPENDENCY_CONTRACT_KEYS = (
    "manifest_paths",
    "lockfile_paths",
    "toolchain_paths",
    "build_support_paths",
)

_AUTHORED_REPO_EXCLUDED_PREFIXES = (
    ".git/",
    "logs/",
    "target/",
    "node_modules/",
    "dist/",
    "build/",
    ".next/",
    ".pytest_cache/",
    ".mypy_cache/",
    "__pycache__/",
)
_AUTHORED_REPO_EXCLUDED_NAMES = {
    ".DS_Store",
}


def empty_dependency_contract() -> dict[str, Any]:
    return {
        "manifest_paths": [],
        "lockfile_paths": [],
        "toolchain_paths": [],
        "build_support_paths": [],
        "reproducibility_mode": None,
    }


def dependency_contract_from_manifest(manifest: dict[str, Any] | None) -> dict[str, Any]:
    payload = manifest.get("dependency_contract") if isinstance(manifest, dict) else None
    if not isinstance(payload, dict):
        return empty_dependency_contract()

    contract = empty_dependency_contract()
    for key in _DEPENDENCY_CONTRACT_KEYS:
        contract[key] = _dedupe_paths(
            _normalize_contract_path(path)
            for path in payload.get(key, []) or []
        )
    reproducibility_mode = payload.get("reproducibility_mode")
    if isinstance(reproducibility_mode, str) and reproducibility_mode.strip():
        contract["reproducibility_mode"] = reproducibility_mode.strip()
    return contract


def starter_dependency_contract_paths(
    *,
    manifest: dict[str, Any] | None = None,
    include_lockfiles: bool = False,
    include_build_support: bool = True,
) -> list[str]:
    contract = dependency_contract_from_manifest(manifest)
    groups = [contract.get("manifest_paths", []), contract.get("toolchain_paths", [])]
    if include_lockfiles:
        groups.append(contract.get("lockfile_paths", []))
    if include_build_support:
        groups.append(contract.get("build_support_paths", []))
    return _dedupe_paths(path for group in groups for path in group)


def starter_contract_path_sets_for_manifest(
    manifest: dict[str, Any] | None,
) -> dict[str, set[str]]:
    contract = dependency_contract_from_manifest(manifest)
    return {
        "manifests": set(contract.get("manifest_paths", [])),
        "lockfiles": set(contract.get("lockfile_paths", [])),
        "toolchains": set(contract.get("toolchain_paths", [])),
        "build_support": set(contract.get("build_support_paths", [])),
    }


def starter_repo_authoring_paths(
    *,
    starter_root: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    if manifest is None:
        return []
    starter_repo_bundle = manifest.get("starter_repo_bundle") or {}
    authored_paths = _authored_paths(starter_repo_bundle)
    discovered_paths = _scan_repo_contract_paths(starter_root, manifest)
    if authored_paths or discovered_paths:
        return _dedupe_paths([*authored_paths, *discovered_paths])
    return learner_editable_paths_for_manifest(manifest)


def starter_materialization_paths(
    *,
    manifest: dict[str, Any] | None = None,
    editable_paths: list[str] | None = None,
    visible_fixture_paths: list[str] | None = None,
) -> list[str]:
    learner_paths = editable_paths
    runtime_paths = list(STARTER_RUNTIME_PROTOCOL_PATHS)
    if manifest is not None:
        starter_repo_bundle = manifest.get("starter_repo_bundle") or {}
        runtime_protocol_bundle = manifest.get("runtime_protocol_bundle") or {}
        authored_repo_paths = _authored_paths(starter_repo_bundle)
        authored_runtime_paths = _authored_paths(runtime_protocol_bundle)
        if authored_repo_paths:
            learner_paths = authored_repo_paths
        elif learner_paths is None:
            learner_paths = starter_repo_authoring_paths(manifest=manifest)
        if authored_runtime_paths:
            runtime_paths = authored_runtime_paths
    fixture_paths = visible_fixture_paths
    if fixture_paths is None and manifest is not None:
        runtime_dependencies = manifest.get("runtime_dependencies") or {}
        fixture_paths = list(runtime_dependencies.get("visible_fixture_files") or [])
    return _dedupe_paths(
        [
            *(learner_paths or []),
            *(fixture_paths or []),
            *starter_dependency_contract_paths(
                manifest=manifest,
                include_lockfiles=True,
                include_build_support=True,
            ),
            *runtime_paths,
            *STARTER_RUNTIME_CHECK_SCRIPT_PATHS,
            *STARTER_SUPPORT_PATHS,
        ]
    )


def read_text_files_for_paths(
    starter_root: Path,
    relative_paths: list[str],
) -> dict[str, str]:
    files: dict[str, str] = {}
    for relative_path in _dedupe_paths(relative_paths):
        path = starter_root / relative_path
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not _looks_like_text(content):
            continue
        files[relative_path] = content
    return files


def load_starter_manifest(starter_root: Path) -> dict[str, Any] | None:
    manifest_path = starter_root / HIDDEN_MANIFEST_PATH
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dependency_contract_facts_for_deliverables(
    *,
    public_root: str | None,
    runtime_plan: ProjectRuntimePlanSpec | None,
    deliverable_ids: list[str],
) -> list[FailureContextDependencyContract]:
    if not public_root or not deliverable_ids:
        return []

    public_dir = Path(public_root)
    container_image = _primary_container_image(runtime_plan)
    facts: list[FailureContextDependencyContract] = []
    for deliverable_id in deliverable_ids:
        starter_root = public_dir / "starter" / deliverable_id
        if not starter_root.exists():
            continue
        manifest = load_starter_manifest(starter_root)
        contract = dependency_contract_from_manifest(manifest)
        manifest_paths = list(contract.get("manifest_paths", []))
        lockfile_paths = list(contract.get("lockfile_paths", []))
        toolchain_paths = list(contract.get("toolchain_paths", []))
        build_support_paths = list(contract.get("build_support_paths", []))
        present_manifests = [path for path in manifest_paths if (starter_root / path).exists()]
        present_lockfiles = [path for path in lockfile_paths if (starter_root / path).exists()]
        present_toolchains = [path for path in toolchain_paths if (starter_root / path).exists()]
        present_build_support = [path for path in build_support_paths if (starter_root / path).exists()]
        runtime_paths_present = [path for path in STARTER_RUNTIME_PROTOCOL_PATHS if (starter_root / path).exists()]
        root_files = sorted(
            {
                *present_manifests,
                *present_toolchains,
                *present_build_support,
                *runtime_paths_present,
            }
        )
        facts.append(
            FailureContextDependencyContract(
                deliverable_id=deliverable_id,
                starter_root=str(starter_root),
                implementation_language=runtime_plan.implementation_language if runtime_plan else None,
                language_version=runtime_plan.language_version if runtime_plan else None,
                application_framework=runtime_plan.application_framework if runtime_plan else None,
                framework_version=runtime_plan.framework_version if runtime_plan else None,
                package_manager=runtime_plan.package_manager if runtime_plan else None,
                container_image=container_image,
                root_files=root_files,
                expected_manifest_paths=manifest_paths,
                present_manifest_paths=present_manifests,
                expected_lockfile_paths=lockfile_paths,
                present_lockfile_paths=present_lockfiles,
                expected_toolchain_paths=toolchain_paths,
                present_toolchain_paths=present_toolchains,
                expected_build_support_paths=build_support_paths,
                present_build_support_paths=present_build_support,
                runtime_protocol_paths_present=runtime_paths_present,
                runtime_bundle_complete=len(runtime_paths_present) == len(STARTER_RUNTIME_PROTOCOL_PATHS),
            )
        )
    return facts


def is_repo_contract_path(relative_path: str) -> bool:
    normalized = str(relative_path or "").strip().replace("\\", "/")
    if not normalized:
        return False
    if normalized in _AUTHORED_REPO_EXCLUDED_NAMES:
        return False
    if any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in _AUTHORED_REPO_EXCLUDED_PREFIXES
    ):
        return False
    if normalized.endswith(".log"):
        return False
    return True


def _primary_container_image(runtime_plan: ProjectRuntimePlanSpec | None) -> str | None:
    if runtime_plan is None:
        return None
    learner_managed = [
        service.container_image
        for service in runtime_plan.services
        if service.learner_managed and service.container_image
    ]
    if learner_managed:
        return learner_managed[0]
    for service in runtime_plan.services:
        if service.container_image:
            return service.container_image
    return None


def _authored_paths(payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return _dedupe_paths(
        _normalize_contract_path(path)
        for path in (payload.get("authored_paths") or [])
        if is_repo_contract_path(path)
    )


def _normalize_contract_path(path: object) -> str | None:
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        return None
    normalized = str(Path(normalized).as_posix())
    if normalized in {".", ""} or normalized.startswith("../") or normalized.startswith("/"):
        return None
    if not is_repo_contract_path(normalized):
        return None
    return normalized


def _scan_repo_contract_paths(
    starter_root: Path | None,
    manifest: dict[str, Any],
) -> list[str]:
    if starter_root is None or not starter_root.exists():
        return []
    visible_fixture_paths = set((manifest.get("runtime_dependencies") or {}).get("visible_fixture_files") or [])
    dependency_paths = set(
        starter_dependency_contract_paths(
            manifest=manifest,
            include_lockfiles=True,
            include_build_support=True,
        )
    )
    excluded_paths = {
        "README.md",
        *STARTER_RUNTIME_PROTOCOL_PATHS,
        *STARTER_RUNTIME_CHECK_SCRIPT_PATHS,
        *STARTER_SUPPORT_PATHS,
        *dependency_paths,
        *visible_fixture_paths,
    }
    discovered: list[str] = []
    for path in sorted(item for item in starter_root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(starter_root).as_posix()
        if relative_path in excluded_paths:
            continue
        if relative_path.startswith(("checks/", ".coursegen/", ".vscode/")):
            continue
        if not is_repo_contract_path(relative_path):
            continue
        discovered.append(relative_path)
    return discovered


def _dedupe_paths(relative_paths: list[str] | tuple[str, ...] | set[str] | Any) -> list[str]:
    seen: set[str] = set()
    resolved: list[str] = []
    for relative_path in relative_paths:
        if not relative_path:
            continue
        normalized = str(relative_path).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def _looks_like_text(content: str) -> bool:
    for character in content:
        codepoint = ord(character)
        if codepoint < 32 and character not in {"\n", "\r", "\t"}:
            return False
    return True
