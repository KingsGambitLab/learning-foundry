from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.task_agent import DeliverableSpec, TaskAgentServiceSpec
from app.domain.workflow import ArtifactVisibility, MaterializedBundle
from app.services.public_surface_quality import (
    content_lacks_domain_grounding,
    endpoint_uses_archetype_words,
    endpoint_uses_title_slug,
)
from app.services.task_agent_contract_surface import learner_editable_paths_for_deliverable
from app.services.task_agent_starter_templates import (
    HIDDEN_MANIFEST_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    build_task_agent_starter_files,
)


class BundleValidationLevel(str, Enum):
    error = "error"
    warning = "warning"


class BundleValidationIssue(BaseModel):
    level: BundleValidationLevel
    code: str
    relative_path: str
    message: str


class BundleValidationResult(BaseModel):
    valid: bool
    errors: list[BundleValidationIssue] = Field(default_factory=list)
    warnings: list[BundleValidationIssue] = Field(default_factory=list)


_STARTER_README_REQUIRED_SECTIONS = (
    "## What to build",
    "## Files to edit",
    "## Definition of done",
    "## Helpful commands",
)
_SECONDARY_BRIEF_FILENAMES = {
    "deliverable_content.md",
    "project_brief.md",
}
_LOCAL_FILE_REFERENCE_PATTERN = re.compile(
    r"`(?P<path>[A-Za-z0-9_./-]+\.(?:md|py|ts|js|json|txt|yaml|yml|csv|sql|toml))`"
)
_ENDPOINT_REFERENCE_PATTERN = re.compile(
    r"`(?P<method>GET|POST|PUT|PATCH|DELETE)\s+(?P<path>/[A-Za-z0-9_./{}-]*)`"
)
_INTERNAL_RUNTIME_MARKERS = (
    "COURSE_GEN_TASK_AGENT_RUNTIME",
    "starter_manifest.json",
)
_MANIFEST_SIMULATION_MARKERS = (
    "public_check_cases",
    "_simulate_run(",
    "_match_eval_case(",
)
_OVERSTATED_WORKFLOW_MARKERS = (
    ("agentic system", "course_readme_overstates_workflow_surface", "Course README should describe the shipped service honestly, not as a generic agent runtime."),
    ("tool-use policies", "course_readme_overstates_workflow_surface", "Course README should not promise tool-use semantics unless the project actually requires them."),
    ("tool use policies", "course_readme_overstates_workflow_surface", "Course README should not promise tool-use semantics unless the project actually requires them."),
    ("evaluation hooks", "course_readme_overstates_workflow_surface", "Course README should not promise hidden runtime hooks in learner-facing packaging."),
)


def _expected_public_artifacts(spec: TaskAgentServiceSpec) -> dict[str, tuple[str, str, str | None]]:
    expected: dict[str, tuple[str, str, str | None]] = {
        "public/README.md": ("course_readme", "learner", None),
        "public/content/course_outline.md": ("course_outline", "learner", None),
    }
    for deliverable in spec.deliverables:
        deliverable_root = f"public/starter/{deliverable.id}"
        expected[f"{deliverable_root}/README.md"] = ("starter_readme", "learner", deliverable.id)
        expected[f"{deliverable_root}/checks/run_visible_checks.py"] = (
            "visible_check_runner",
            "learner",
            deliverable.id,
        )
        expected[f"{deliverable_root}/{RUNTIME_INSTALL_SCRIPT_PATH}"] = (
            "runtime_install_script",
            "operator",
            deliverable.id,
        )
        expected[f"{deliverable_root}/{RUNTIME_VERIFY_SCRIPT_PATH}"] = (
            "runtime_verify_script",
            "operator",
            deliverable.id,
        )
        expected[f"{deliverable_root}/{RUNTIME_RUN_SCRIPT_PATH}"] = (
            "runtime_run_script",
            "operator",
            deliverable.id,
        )
        expected[f"{deliverable_root}/{RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"] = (
            "runtime_visible_check_script",
            "operator",
            deliverable.id,
        )
        expected[f"{deliverable_root}/{RUNTIME_HIDDEN_CHECK_SCRIPT_PATH}"] = (
            "runtime_hidden_check_script",
            "operator",
            deliverable.id,
        )
        expected[f"{deliverable_root}/.vscode/tasks.json"] = ("vscode_tasks", "learner", deliverable.id)
        expected[f"{deliverable_root}/{HIDDEN_MANIFEST_PATH}"] = (
            "starter_manifest",
            "operator",
            deliverable.id,
        )
    return expected


def _add_issue(
    issues: list[BundleValidationIssue],
    *,
    level: BundleValidationLevel,
    code: str,
    relative_path: str,
    message: str,
) -> None:
    issues.append(
        BundleValidationIssue(
            level=level,
            code=code,
            relative_path=relative_path,
            message=message,
        )
    )


def _read_text(bundle: MaterializedBundle, relative_path: str) -> str:
    root = Path(bundle.root_dir)
    target = root / relative_path
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8")


def _primary_editable_paths(spec: TaskAgentServiceSpec, deliverable: DeliverableSpec) -> list[str]:
    return learner_editable_paths_for_deliverable(spec, deliverable)


def _iter_local_file_references(content: str) -> list[str]:
    seen: set[str] = set()
    refs: list[str] = []
    for match in _LOCAL_FILE_REFERENCE_PATTERN.finditer(content):
        candidate = match.group("path").strip()
        if candidate.startswith(("http://", "https://")) or candidate in seen:
            continue
        refs.append(candidate)
        seen.add(candidate)
    return refs


def _iter_endpoint_references(content: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    refs: list[tuple[str, str]] = []
    for match in _ENDPOINT_REFERENCE_PATTERN.finditer(content):
        candidate = (match.group("method").upper(), match.group("path").strip())
        if candidate in seen:
            continue
        refs.append(candidate)
        seen.add(candidate)
    return refs


def _published_endpoint_identities(spec: TaskAgentServiceSpec) -> set[tuple[str, str]]:
    return {(endpoint.method, endpoint.path) for endpoint in spec.public_endpoints}


def _validate_starter_readme(
    issues: list[BundleValidationIssue],
    *,
    relative_path: str,
    content: str,
    reference_root: Path,
    spec: TaskAgentServiceSpec,
    check_local_refs: bool = True,
) -> None:
    published_endpoints = _published_endpoint_identities(spec)
    for section in _STARTER_README_REQUIRED_SECTIONS:
        if section not in content:
            _add_issue(
                issues,
                level=BundleValidationLevel.error,
                code="starter_readme_missing_section",
                relative_path=relative_path,
                message=f"Starter README is missing the required section `{section}`.",
            )
    for reference in _iter_local_file_references(content):
        if check_local_refs and not (reference_root / reference).exists():
            _add_issue(
                issues,
                level=BundleValidationLevel.error,
                code="starter_readme_missing_local_reference",
                relative_path=relative_path,
                message=f"Starter README references `{reference}`, but that file is not present in the learner workspace.",
            )
        if Path(reference).name in _SECONDARY_BRIEF_FILENAMES:
            _add_issue(
                issues,
                level=BundleValidationLevel.error,
                code="starter_readme_uses_secondary_brief",
                relative_path=relative_path,
                message="Starter README must be self-contained and must not send learners to a secondary brief file.",
            )
    for method, path in _iter_endpoint_references(content):
        if (method, path) not in published_endpoints:
            _add_issue(
                issues,
                level=BundleValidationLevel.error,
                code="starter_readme_unpublished_endpoint_reference",
                relative_path=relative_path,
                message=(
                    f"Starter README references `{method} {path}`, but that route is not part of the "
                    "published public surface."
                ),
            )
    if content_lacks_domain_grounding(content, entities=spec.project_contract.core_entities):
        _add_issue(
            issues,
            level=BundleValidationLevel.error,
            code="starter_readme_lacks_domain_grounding",
            relative_path=relative_path,
            message="Starter README should use concrete domain entities from the project brief instead of generic service wording.",
        )


def _validate_course_readme(
    issues: list[BundleValidationIssue],
    *,
    relative_path: str,
    content: str,
    spec: TaskAgentServiceSpec,
) -> None:
    lowered = content.lower()
    published_endpoints = _published_endpoint_identities(spec)
    for marker, code, message in _OVERSTATED_WORKFLOW_MARKERS:
        if marker in lowered:
            _add_issue(
                issues,
                level=BundleValidationLevel.error,
                code=code,
                relative_path=relative_path,
                message=message,
            )
    if not spec.capabilities.tool_use_required and ("tool-use" in lowered or "tool use" in lowered):
        _add_issue(
            issues,
            level=BundleValidationLevel.error,
            code="course_readme_unbacked_tooling_claim",
            relative_path=relative_path,
            message="Course README mentions tool-use semantics that are not part of this project contract.",
        )
    if not spec.capabilities.approval_flow_required and "approval" in lowered:
        _add_issue(
            issues,
            level=BundleValidationLevel.error,
            code="course_readme_unbacked_approval_claim",
            relative_path=relative_path,
            message="Course README mentions approval flow semantics that are not part of this project contract.",
        )
    if not spec.capabilities.traceability_required and "trace" in lowered:
        _add_issue(
            issues,
            level=BundleValidationLevel.error,
            code="course_readme_unbacked_trace_claim",
            relative_path=relative_path,
            message="Course README mentions trace semantics that are not part of this project contract.",
        )
    if content_lacks_domain_grounding(content, entities=spec.project_contract.core_entities):
        _add_issue(
            issues,
            level=BundleValidationLevel.error,
            code="course_readme_lacks_domain_grounding",
            relative_path=relative_path,
            message="Course README should describe the project using concrete domain entities, not only generic backend wording.",
        )
    for method, path in _iter_endpoint_references(content):
        if (method, path) not in published_endpoints:
            _add_issue(
                issues,
                level=BundleValidationLevel.error,
                code="course_readme_unpublished_endpoint_reference",
                relative_path=relative_path,
                message=(
                    f"Course README references `{method} {path}`, but that route is not part of the "
                    "published public surface."
                ),
            )


def _validate_starter_entrypoint_honesty(
    issues: list[BundleValidationIssue],
    *,
    relative_path: str,
    content: str,
) -> None:
    lowered = content.lower()
    if any(marker in content for marker in _INTERNAL_RUNTIME_MARKERS):
        _add_issue(
            issues,
            level=BundleValidationLevel.error,
            code="starter_entrypoint_embeds_internal_runtime",
            relative_path=relative_path,
            message="Primary learner-owned files should contain the real app surface, not internal CourseGen runtime glue.",
        )
    if any(marker in lowered for marker in _MANIFEST_SIMULATION_MARKERS):
        _add_issue(
            issues,
            level=BundleValidationLevel.error,
            code="starter_entrypoint_simulates_from_manifest",
            relative_path=relative_path,
            message="Primary learner-owned files should not derive business behavior from hidden harness manifests.",
        )


def validate_materialized_bundle(
    spec: TaskAgentServiceSpec,
    bundle: MaterializedBundle,
) -> BundleValidationResult:
    errors: list[BundleValidationIssue] = []
    warnings: list[BundleValidationIssue] = []
    files_by_path = {entry.relative_path: entry for entry in bundle.files}
    expected = _expected_public_artifacts(spec)
    for index, endpoint in enumerate(spec.public_endpoints):
        if endpoint.path == "/health":
            continue
        if endpoint_uses_title_slug(endpoint.path, title=spec.title):
            _add_issue(
                errors,
                level=BundleValidationLevel.error,
                code="title_slug_public_endpoint",
                relative_path=f"public_endpoints[{index}]",
                message="Public endpoints should use concrete resource nouns, not the full course title as a URL slug.",
            )
        elif endpoint_uses_archetype_words(endpoint.path):
            _add_issue(
                errors,
                level=BundleValidationLevel.error,
                code="generic_public_endpoint",
                relative_path=f"public_endpoints[{index}]",
                message="Public endpoints should expose a concrete resource surface instead of archetype words like service or API.",
            )

    for entry in bundle.files:
        if entry.visibility != ArtifactVisibility.public:
            continue
        if not entry.role or not entry.audience or not entry.semantic_source:
            _add_issue(
                errors,
                level=BundleValidationLevel.error,
                code="missing_bundle_file_metadata",
                relative_path=entry.relative_path,
                message="Every public artifact must declare a role, audience, and semantic source.",
            )

    for relative_path, (role, audience, deliverable_id) in expected.items():
        entry = files_by_path.get(relative_path)
        if entry is None:
            _add_issue(
                errors,
                level=BundleValidationLevel.error,
                code="missing_expected_public_artifact",
                relative_path=relative_path,
                message="The materialized bundle is missing a required public artifact.",
            )
            continue
        if entry.role != role or entry.audience != audience:
            _add_issue(
                errors,
                level=BundleValidationLevel.error,
                code="public_artifact_role_mismatch",
                relative_path=relative_path,
                message=(
                    f"Expected role `{role}` for audience `{audience}`, "
                    f"but found role `{entry.role}` and audience `{entry.audience}`."
                ),
            )
        if entry.deliverable_id != deliverable_id:
            _add_issue(
                errors,
                level=BundleValidationLevel.error,
                code="public_artifact_deliverable_mismatch",
                relative_path=relative_path,
                message=(
                    f"Expected deliverable `{deliverable_id or 'none'}` but found "
                    f"`{entry.deliverable_id or 'none'}`."
                ),
            )

    for deliverable in spec.deliverables:
        starter_root = Path(bundle.root_dir) / "public" / "starter" / deliverable.id
        manifest_path = starter_root / HIDDEN_MANIFEST_PATH
        manifest_payload: dict[str, object] = {}
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                _add_issue(
                    errors,
                    level=BundleValidationLevel.error,
                    code="starter_manifest_invalid_json",
                    relative_path=str(manifest_path.relative_to(bundle.root_dir)),
                    message="Starter manifest must contain valid JSON.",
                )
        starter_repo_bundle = (
            manifest_payload.get("starter_repo_bundle")
            if isinstance(manifest_payload, dict)
            else {}
        ) or {}
        runtime_protocol_bundle = (
            manifest_payload.get("runtime_protocol_bundle")
            if isinstance(manifest_payload, dict)
            else {}
        ) or {}
        starter_repo_source = str(
            starter_repo_bundle.get("source")
            if isinstance(starter_repo_bundle, dict)
            else ""
        ).strip().lower()
        starter_repo_authored_paths = sorted(
            str(path).strip()
            for path in (
                starter_repo_bundle.get("authored_paths")
                if isinstance(starter_repo_bundle, dict)
                else []
            )
            if str(path).strip()
        )
        runtime_protocol_source = str(
            runtime_protocol_bundle.get("source")
            if isinstance(runtime_protocol_bundle, dict)
            else ""
        ).strip().lower()
        runtime_protocol_authored_paths = sorted(
            str(path).strip()
            for path in (
                runtime_protocol_bundle.get("authored_paths")
                if isinstance(runtime_protocol_bundle, dict)
                else []
            )
            if str(path).strip()
        )
        default_starter_files = build_task_agent_starter_files(spec, deliverable.id)
        readme_path = starter_root / "README.md"
        if readme_path.exists():
            _validate_starter_readme(
                errors,
                relative_path=str(readme_path.relative_to(bundle.root_dir)),
                content=readme_path.read_text(encoding="utf-8"),
                reference_root=starter_root,
                spec=spec,
                check_local_refs=starter_repo_source not in {"", "starter_default"},
            )
        if starter_repo_source in {"", "starter_default"}:
            _add_issue(
                warnings,
                level=BundleValidationLevel.warning,
                code="starter_repo_bundle_not_authored",
                relative_path=str(manifest_path.relative_to(bundle.root_dir)),
                message="Starter repo files were not authored from the real starter workspace.",
            )
        if runtime_protocol_source in {"", "starter_default"}:
            _add_issue(
                warnings,
                level=BundleValidationLevel.warning,
                code="runtime_protocol_bundle_not_authored",
                relative_path=str(manifest_path.relative_to(bundle.root_dir)),
                message="Runtime install/verify/run files were not authored from the real starter workspace.",
            )
        else:
            missing_runtime_paths = [
                relative_path
                for relative_path in (
                    "Dockerfile",
                    RUNTIME_INSTALL_SCRIPT_PATH,
                    RUNTIME_VERIFY_SCRIPT_PATH,
                    RUNTIME_RUN_SCRIPT_PATH,
                )
                if not (starter_root / relative_path).exists()
                or relative_path not in runtime_protocol_authored_paths
            ]
            default_runtime_paths = [
                relative_path
                for relative_path in (
                    "Dockerfile",
                    RUNTIME_INSTALL_SCRIPT_PATH,
                    RUNTIME_VERIFY_SCRIPT_PATH,
                    RUNTIME_RUN_SCRIPT_PATH,
                )
                if (starter_root / relative_path).exists()
                and (starter_root / relative_path).read_text(encoding="utf-8")
                == default_starter_files.get(relative_path, "")
            ]
            if missing_runtime_paths or default_runtime_paths:
                _add_issue(
                    errors,
                    level=BundleValidationLevel.error,
                    code="runtime_protocol_bundle_incomplete",
                    relative_path=str(manifest_path.relative_to(bundle.root_dir)),
                    message=(
                        "Runtime protocol must be authored as a complete bundle. "
                        f"Missing declared runtime files: {', '.join(missing_runtime_paths) or 'none'}. "
                        f"Default placeholders still present: {', '.join(default_runtime_paths) or 'none'}."
                    ),
                )
        for relative_path in _primary_editable_paths(spec, deliverable):
            entrypoint_path = starter_root / relative_path
            if not entrypoint_path.exists():
                if starter_repo_source not in {"", "starter_default"}:
                    _add_issue(
                        errors,
                        level=BundleValidationLevel.error,
                        code="starter_primary_editable_missing",
                        relative_path=str((starter_root / relative_path).relative_to(bundle.root_dir)),
                        message="Primary learner-owned files must exist in the published starter workspace.",
                    )
                continue
            _validate_starter_entrypoint_honesty(
                errors,
                relative_path=str(entrypoint_path.relative_to(bundle.root_dir)),
                content=entrypoint_path.read_text(encoding="utf-8"),
            )
        if starter_repo_source not in {"", "starter_default"}:
            required_editable_paths = _primary_editable_paths(spec, deliverable)
            missing_repo_paths = [
                relative_path
                for relative_path in required_editable_paths
                if not (starter_root / relative_path).exists()
                or relative_path not in starter_repo_authored_paths
            ]
            if missing_repo_paths:
                _add_issue(
                    errors,
                    level=BundleValidationLevel.error,
                    code="starter_repo_bundle_incomplete",
                    relative_path=str(manifest_path.relative_to(bundle.root_dir)),
                    message=(
                        "Starter repo bundle is marked authored, but primary learner-owned files are still missing "
                        f"or undeclared: {', '.join(missing_repo_paths)}."
                    ),
                )

    course_readme_path = Path(bundle.root_dir) / "public" / "README.md"
    if course_readme_path.exists():
        _validate_course_readme(
            errors,
            relative_path=str(course_readme_path.relative_to(bundle.root_dir)),
            content=course_readme_path.read_text(encoding="utf-8"),
            spec=spec,
        )

    return BundleValidationResult(valid=not errors, errors=errors, warnings=warnings)


def inspect_materialized_starter_surface(
    spec: TaskAgentServiceSpec,
    bundle: MaterializedBundle,
) -> BundleValidationResult:
    return validate_materialized_bundle(spec, bundle)


def validate_seeded_learner_workspace(
    spec: TaskAgentServiceSpec,
    workspace_root: str | Path,
    *,
    deliverable_ids: list[str] | None = None,
) -> BundleValidationResult:
    root = Path(workspace_root)
    errors: list[BundleValidationIssue] = []
    target_ids = deliverable_ids or [deliverable.id for deliverable in spec.deliverables]
    for deliverable_id in target_ids:
        readme_path = root / ".coursegen" / "review_areas" / deliverable_id / "README.md"
        if not readme_path.exists():
            _add_issue(
                errors,
                level=BundleValidationLevel.error,
                code="seeded_workspace_missing_review_area_readme",
                relative_path=str(readme_path.relative_to(root)),
                message="The seeded learner workspace is missing the review-area README.",
            )
            continue
        _validate_starter_readme(
            errors,
            relative_path=str(readme_path.relative_to(root)),
            content=readme_path.read_text(encoding="utf-8"),
            reference_root=readme_path.parent,
            spec=spec,
        )
        for legacy_name in _SECONDARY_BRIEF_FILENAMES:
            legacy_brief = readme_path.parent / legacy_name
            if legacy_brief.exists():
                _add_issue(
                    errors,
                    level=BundleValidationLevel.error,
                    code="deprecated_secondary_brief_present",
                    relative_path=str(legacy_brief.relative_to(root)),
                    message="Seeded learner workspaces should not contain duplicate secondary deliverable brief files.",
                )
    return BundleValidationResult(valid=not errors, errors=errors, warnings=[])
