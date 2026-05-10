from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.runtime_contract_surface import (
    STARTER_RUNTIME_PROTOCOL_PATHS,
    read_text_files_for_paths,
    starter_dependency_contract_paths,
    starter_repo_authoring_paths,
)
from app.services.task_agent_contract_surface import (
    required_public_endpoints_for_manifest,
)


def build_starter_authoring_payload(
    *,
    starter_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "learner_files": read_text_files_for_paths(
            starter_root,
            starter_repo_authoring_paths(
                starter_root=starter_root,
                manifest=manifest,
            ),
        ),
        "dependency_contract_files": read_text_files_for_paths(
            starter_root,
            starter_dependency_contract_paths(manifest=manifest, include_lockfiles=False),
        ),
        "runtime_protocol_files": read_text_files_for_paths(
            starter_root,
            STARTER_RUNTIME_PROTOCOL_PATHS,
        ),
        "public_endpoints": [
            endpoint.model_dump(mode="json")
            for endpoint in required_public_endpoints_for_manifest(manifest)
        ],
    }
