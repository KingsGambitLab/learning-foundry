from __future__ import annotations

from pathlib import Path

from app.domain.task_agent import ProjectRuntimePlanSpec
from app.domain.workflow import FailureContextDependencyContract
from app.services.task_agent_starter_templates import (
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
)

_PACKAGE_MANAGER_CONTRACTS: dict[str, dict[str, list[str]]] = {
    "cargo": {
        "manifests": ["Cargo.toml"],
        "lockfiles": ["Cargo.lock"],
        "toolchains": ["rust-toolchain.toml", "rust-toolchain"],
    },
    "npm": {
        "manifests": ["package.json"],
        "lockfiles": ["package-lock.json"],
        "toolchains": [".nvmrc", ".node-version"],
    },
    "pnpm": {
        "manifests": ["package.json"],
        "lockfiles": ["pnpm-lock.yaml"],
        "toolchains": [".nvmrc", ".node-version"],
    },
    "yarn": {
        "manifests": ["package.json"],
        "lockfiles": ["yarn.lock"],
        "toolchains": [".nvmrc", ".node-version"],
    },
    "uv": {
        "manifests": ["pyproject.toml"],
        "lockfiles": ["uv.lock"],
        "toolchains": [".python-version"],
    },
    "poetry": {
        "manifests": ["pyproject.toml"],
        "lockfiles": ["poetry.lock"],
        "toolchains": [".python-version"],
    },
    "pip": {
        "manifests": ["pyproject.toml", "requirements.txt"],
        "lockfiles": ["requirements.lock", "constraints.txt"],
        "toolchains": [".python-version"],
    },
    "go": {
        "manifests": ["go.mod"],
        "lockfiles": ["go.sum"],
        "toolchains": ["go.work"],
    },
    "maven": {
        "manifests": ["pom.xml"],
        "lockfiles": [],
        "toolchains": [".mvn/wrapper/maven-wrapper.properties"],
    },
    "gradle": {
        "manifests": ["build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"],
        "lockfiles": ["gradle.lockfile"],
        "toolchains": ["gradle/wrapper/gradle-wrapper.properties"],
    },
}

_RUNTIME_PROTOCOL_PATHS = [
    "Dockerfile",
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
]


def expected_dependency_contract_paths(package_manager: str | None) -> dict[str, list[str]]:
    normalized = (package_manager or "").strip().lower()
    expected = _PACKAGE_MANAGER_CONTRACTS.get(normalized, {})
    return {
        "manifests": list(expected.get("manifests", [])),
        "lockfiles": list(expected.get("lockfiles", [])),
        "toolchains": list(expected.get("toolchains", [])),
    }


def dependency_contract_facts_for_deliverables(
    *,
    public_root: str | None,
    runtime_plan: ProjectRuntimePlanSpec | None,
    deliverable_ids: list[str],
) -> list[FailureContextDependencyContract]:
    if not public_root or not deliverable_ids:
        return []

    public_dir = Path(public_root)
    package_manager = (runtime_plan.package_manager or "").strip().lower() if runtime_plan else ""
    expected = expected_dependency_contract_paths(package_manager)
    container_image = _primary_container_image(runtime_plan)
    facts: list[FailureContextDependencyContract] = []
    for deliverable_id in deliverable_ids:
        starter_root = public_dir / "starter" / deliverable_id
        if not starter_root.exists():
            continue
        root_files = sorted(
            path.relative_to(starter_root).as_posix()
            for path in starter_root.rglob("*")
            if path.is_file()
        )
        present_manifests = [path for path in expected.get("manifests", []) if (starter_root / path).exists()]
        present_lockfiles = [path for path in expected.get("lockfiles", []) if (starter_root / path).exists()]
        present_toolchains = [path for path in expected.get("toolchains", []) if (starter_root / path).exists()]
        runtime_paths_present = [path for path in _RUNTIME_PROTOCOL_PATHS if (starter_root / path).exists()]
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
                expected_manifest_paths=expected.get("manifests", []),
                present_manifest_paths=present_manifests,
                expected_lockfile_paths=expected.get("lockfiles", []),
                present_lockfile_paths=present_lockfiles,
                expected_toolchain_paths=expected.get("toolchains", []),
                present_toolchain_paths=present_toolchains,
                runtime_protocol_paths_present=runtime_paths_present,
                runtime_bundle_complete=len(runtime_paths_present) == len(_RUNTIME_PROTOCOL_PATHS),
            )
        )
    return facts


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
