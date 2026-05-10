from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.dependency_contract_facts import expected_dependency_contract_paths
from app.services.task_agent_contract_surface import (
    learner_editable_paths_for_manifest,
    required_public_endpoints_for_manifest,
)
from app.services.task_agent_starter_templates import (
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
)

_RUNTIME_PROTOCOL_PATHS = [
    "Dockerfile",
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
]


def build_starter_authoring_payload(
    *,
    starter_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "learner_files": _read_text_files(
            starter_root,
            learner_editable_paths_for_manifest(manifest),
        ),
        "dependency_contract_files": _read_text_files(
            starter_root,
            _dependency_contract_paths(manifest),
        ),
        "runtime_protocol_files": _read_text_files(
            starter_root,
            _RUNTIME_PROTOCOL_PATHS,
        ),
        "public_endpoints": [
            endpoint.model_dump(mode="json")
            for endpoint in required_public_endpoints_for_manifest(manifest)
        ],
    }


def _dependency_contract_paths(manifest: dict[str, Any]) -> list[str]:
    runtime_plan = manifest.get("runtime_plan") or (manifest.get("project_contract") or {}).get("runtime_plan") or {}
    expected = expected_dependency_contract_paths(runtime_plan.get("package_manager"))
    seen: set[str] = set()
    resolved: list[str] = []
    for relative_path in [*expected.get("manifests", []), *expected.get("toolchains", [])]:
        normalized = str(relative_path or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def _read_text_files(starter_root: Path, relative_paths: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    seen: set[str] = set()
    for relative_path in relative_paths:
        normalized = str(relative_path or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        path = starter_root / normalized
        if not path.exists() or not path.is_file():
            continue
        try:
            files[normalized] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return files
