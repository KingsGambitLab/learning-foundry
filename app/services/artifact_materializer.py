from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.task_agent import TaskAgentServiceSpec
from app.services.coursegen_logging import log_coursegen_event
from app.domain.workflow import (
    ArtifactVisibility,
    BundleFile,
    BundleFileContent,
    MaterializedBundle,
    WorkflowRun,
)
from app.services.creator_asset_service import CreatorAssetService
from app.services.learner_brief_builder import (
    build_task_agent_deliverable_brief,
    render_learner_starter_readme,
)
from app.services.runtime_contract_surface import (
    load_starter_manifest,
    starter_contract_path_sets_for_manifest,
    starter_materialization_paths,
)
from app.services.task_agent_contract_surface import (
    learner_editable_paths_for_deliverable,
    primary_submit_endpoint_for_spec,
)
from app.services.task_agent_starter_templates import (
    HIDDEN_GRADER_SCRIPT_PATH,
    HIDDEN_MANIFEST_PATH,
    RUNTIME_HIDDEN_CHECK_SCRIPT_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    build_task_agent_starter_files,
    default_preview_command,
    task_agent_runtime_base_image,
    task_agent_runtime_bootstrap_commands,
    task_agent_runtime_environment_lines,
)

# Per-deliverable artifacts that live OUTSIDE the shared starter tree.
VISIBLE_CHECK_SCRIPT_RELATIVE_PATH = "run_visible_checks.py"
HIDDEN_GRADER_SCRIPT_RELATIVE_PATH = "run_hidden_checks.py"
DELIVERABLE_MANIFEST_RELATIVE_PATH = "deliverable.json"
# Shared course-level manifest at the shared starter root. Carries the
# course-wide fields (runtime_plan, runtime_dependencies, course_structure,
# public_endpoints, dependency_contract) once, instead of duplicating them
# inside every per-deliverable manifest.
SHARED_COURSE_MANIFEST_RELATIVE_PATH = ".coursegen/course.json"


def shared_starter_workspace_path(public_dir: Path) -> Path:
    """For shared_codebase courses: the single shared starter root."""
    return public_dir / "starter"


def deliverable_visible_checks_dir(public_dir: Path, deliverable_id: str) -> Path:
    """For shared_codebase courses: per-deliverable learner-facing brief + visible script."""
    return public_dir / "checks" / deliverable_id


def deliverable_grader_dir(private_dir: Path, deliverable_id: str) -> Path:
    """For shared_codebase courses: per-deliverable hidden grader + manifest."""
    return private_dir / "grader" / deliverable_id


def build_shared_course_manifest_payload(
    spec: TaskAgentServiceSpec,
    *,
    dependency_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Course-level fields that previously got copied into every per-deliverable
    manifest. Living once at public/starter/.coursegen/course.json keeps the
    shared starter authoritative for the whole course."""
    payload: dict[str, Any] = {
        "title": spec.title,
        "summary": spec.summary,
        "course_structure": spec.course_structure.model_dump(mode="json"),
        "runtime_plan": spec.project_contract.runtime_plan.model_dump(mode="json"),
        "runtime_dependencies": spec.runtime_dependencies.model_dump(mode="json"),
        "public_endpoints": [endpoint.model_dump(mode="json") for endpoint in spec.public_endpoints],
    }
    if dependency_contract is not None:
        payload["dependency_contract"] = dependency_contract
    return payload

def default_generated_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "generated"


class ArtifactMaterializer:
    def __init__(
        self,
        base_dir: str | Path | None = None,
        creator_asset_service: CreatorAssetService | None = None,
    ) -> None:
        self.base_dir = Path(base_dir or default_generated_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.creator_asset_service = creator_asset_service

    def materialize_run(self, run: WorkflowRun, overwrite: bool = True) -> MaterializedBundle:
        bundle_root = self.base_dir / run.id
        existed = bundle_root.exists()
        wiped = False
        if existed:
            if not overwrite:
                manifest = bundle_root / "manifest.json"
                raise FileExistsError(f"Bundle already exists at '{manifest}'.")
            shutil.rmtree(bundle_root)
            wiped = True
        log_coursegen_event(
            "materialize_run_invoked",
            workflow_run_id=run.id,
            bundle_root=str(bundle_root),
            existed=existed,
            wiped=wiped,
            overwrite=overwrite,
        )

        public_dir = bundle_root / "public"
        private_dir = bundle_root / "private"
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)

        files: list[BundleFile] = []
        generated_at = datetime.now(UTC)

        if run.artifacts.task_agent_spec is not None:
            self._materialize_task_agent(
                run=run,
                spec=run.artifacts.task_agent_spec,
                public_dir=public_dir,
                private_dir=private_dir,
                files=files,
            )
        else:
            self._write_text(
                private_dir / "README.txt",
                "This workflow run is blocked and does not have a materializable draft yet.\n",
                ArtifactVisibility.private,
                files,
                bundle_root,
                role="blocked_run_readme",
                audience="internal",
                semantic_source="system_rendered",
            )

        manifest_payload = {
            "bundle_id": f"{run.id}_bundle",
            "generated_at": generated_at.isoformat(),
            "root_dir": str(bundle_root),
            "public_dir": str(public_dir),
            "private_dir": str(private_dir),
            "files": [entry.model_dump(mode="json") for entry in files],
        }
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

        return MaterializedBundle(
            bundle_id=f"{run.id}_bundle",
            generated_at=generated_at,
            root_dir=str(bundle_root),
            public_dir=str(public_dir),
            private_dir=str(private_dir),
            manifest_path=str(manifest_path),
            files=files,
        )

    def read_bundle_file(self, bundle: MaterializedBundle, relative_path: str) -> BundleFileContent:
        root = Path(bundle.root_dir).resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Requested file is outside the bundle root.")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(relative_path)
        media_type = self._guess_media_type(relative_path)
        return BundleFileContent(
            relative_path=relative_path,
            media_type=media_type,
            content=target.read_text(encoding="utf-8"),
        )

    def _materialize_task_agent(
        self,
        *,
        run: WorkflowRun,
        spec: TaskAgentServiceSpec,
        public_dir: Path,
        private_dir: Path,
        files: list[BundleFile],
    ) -> None:
        bundle_root = public_dir.parent
        self._write_json(
            private_dir / "task_agent_spec.json",
            spec.model_dump(mode="json"),
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="task_agent_spec_snapshot",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "validation_summary.json",
            run.artifacts.validation_summary or {},
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="validation_summary",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "progression_preview.json",
            run.artifacts.progression_preview,
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="progression_preview",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "workflow_snapshot.json",
            {
                "run_id": run.id,
                "title": run.title,
                "stage": run.stage.value,
                "status": run.status.value,
                "pending_gate": run.pending_gate.value if run.pending_gate else None,
                "origin_template": run.artifacts.origin_template,
            },
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="workflow_snapshot",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "node_executions.json",
            [node.model_dump(mode="json") for node in run.artifacts.node_executions],
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="node_executions",
            audience="internal",
            semantic_source="system_snapshot",
        )
        self._write_json(
            private_dir / "review_summary.json",
            run.artifacts.review_summary.model_dump(mode="json") if run.artifacts.review_summary is not None else {},
            ArtifactVisibility.private,
            files,
            bundle_root,
            role="review_summary",
            audience="internal",
            semantic_source="system_snapshot",
        )

        self._write_text(
            public_dir / "README.md",
            self._task_agent_readme(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="course_readme",
            audience="learner",
            semantic_source="spec_rendered",
        )
        self._write_text(
            public_dir / "runtime" / "Dockerfile",
            self._assignment_runtime_dockerfile(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="runtime_dockerfile",
            audience="operator",
            semantic_source="starter_compiler",
        )
        self._write_text(
            public_dir / "runtime" / "README.md",
            self._assignment_runtime_readme(),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="runtime_readme",
            audience="operator",
            semantic_source="system_rendered",
        )
        self._write_text(
            public_dir / "runtime" / "verify_assignment.py",
            self._assignment_runtime_verifier(),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="runtime_verifier",
            audience="operator",
            semantic_source="system_rendered",
        )
        self._write_text(
            public_dir / "content" / "course_outline.md",
            self._course_outline(spec),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="course_outline",
            audience="learner",
            semantic_source="spec_rendered",
        )

        if spec.course_structure.shared_codebase:
            self._materialize_shared_codebase(
                run=run,
                spec=spec,
                public_dir=public_dir,
                private_dir=private_dir,
                bundle_root=bundle_root,
                files=files,
            )
        else:
            self._materialize_per_deliverable_starters(
                run=run,
                spec=spec,
                public_dir=public_dir,
                bundle_root=bundle_root,
                files=files,
            )

    def _materialize_shared_codebase(
        self,
        *,
        run: WorkflowRun,
        spec: TaskAgentServiceSpec,
        public_dir: Path,
        private_dir: Path,
        bundle_root: Path,
        files: list[BundleFile],
    ) -> None:
        """Write the new shared-codebase workspace layout:

            public/starter/             # ONE shared root
            public/checks/<id>/         # per-deliverable README + visible script
            private/grader/<id>/        # per-deliverable manifest + hidden grader
        """
        shared_starter_dir = shared_starter_workspace_path(public_dir)

        # Decide source of shared starter content: workspace snapshot (already authored)
        # or fresh default templates.
        workspace_shared_dir = (
            Path(run.artifacts.workspace_snapshot.public_dir) / "starter"
            if run.artifacts.workspace_snapshot is not None
            and Path(run.artifacts.workspace_snapshot.root_dir).resolve() != bundle_root.resolve()
            else None
        )

        first_deliverable = spec.deliverables[0]
        default_starter_files = build_task_agent_starter_files(spec, first_deliverable.id)
        # Drop per-deliverable artifacts from the shared starter scaffolding.
        default_shared_files = {
            relative_path: content
            for relative_path, content in default_starter_files.items()
            if relative_path
            not in {HIDDEN_MANIFEST_PATH, HIDDEN_GRADER_SCRIPT_PATH, "checks/run_visible_checks.py"}
            and not relative_path.startswith("checks/")
            and not relative_path.startswith(".coursegen/grader/")
        }

        shared_manifest_payload: dict[str, Any] | None = None
        if workspace_shared_dir is not None and workspace_shared_dir.exists():
            # Mirror the authored shared workspace into the bundle's public/starter/.
            workspace_grader_root = (
                Path(run.artifacts.workspace_snapshot.root_dir) / "private" / "grader"
                if run.artifacts.workspace_snapshot is not None
                else None
            )
            if workspace_grader_root is not None:
                first_manifest_path = (
                    workspace_grader_root / first_deliverable.id / DELIVERABLE_MANIFEST_RELATIVE_PATH
                )
                if first_manifest_path.exists():
                    try:
                        shared_manifest_payload = json.loads(first_manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        shared_manifest_payload = None
            for relative_path in self._shared_workspace_paths(
                workspace_shared_dir=workspace_shared_dir,
                spec=spec,
                manifest=shared_manifest_payload,
            ):
                source_path = workspace_shared_dir / relative_path
                if not source_path.exists() or not source_path.is_file():
                    continue
                role, audience, semantic_source = self._starter_file_metadata(
                    relative_path,
                    manifest_payload=shared_manifest_payload,
                )
                try:
                    content = source_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                self._write_text(
                    shared_starter_dir / relative_path,
                    content,
                    ArtifactVisibility.public,
                    files,
                    bundle_root,
                    role=role,
                    audience=audience,
                    deliverable_id=None,
                    semantic_source=semantic_source,
                )
        else:
            for relative_path, content in default_shared_files.items():
                role, audience, semantic_source = self._starter_file_metadata(
                    relative_path,
                    manifest_payload=None,
                )
                self._write_text(
                    shared_starter_dir / relative_path,
                    content,
                    ArtifactVisibility.public,
                    files,
                    bundle_root,
                    role=role,
                    audience=audience,
                    deliverable_id=None,
                    semantic_source=semantic_source,
                )
            self._write_visible_fixture_files(
                spec=spec,
                deliverable_dir=shared_starter_dir,
                deliverable_id=None,
                files=files,
                bundle_root=bundle_root,
            )

        # Shared course-level manifest at public/starter/.coursegen/course.json.
        # Sourced from the authored per-deliverable manifest's dependency_contract
        # (if present) so the shared course manifest stays coherent with what the
        # bundle authoring loop produced.
        course_dependency_contract: dict[str, Any] | None = None
        if isinstance(shared_manifest_payload, dict):
            course_dependency_contract = shared_manifest_payload.get("dependency_contract")
        self._write_json(
            shared_starter_dir / SHARED_COURSE_MANIFEST_RELATIVE_PATH,
            build_shared_course_manifest_payload(
                spec,
                dependency_contract=course_dependency_contract,
            ),
            ArtifactVisibility.public,
            files,
            bundle_root,
            role="shared_course_manifest",
            audience="operator",
            semantic_source="starter_compiler",
        )

        # Per-deliverable: public/checks/<id>/ and private/grader/<id>/.
        for deliverable in spec.deliverables:
            checks_dir = deliverable_visible_checks_dir(public_dir, deliverable.id)
            grader_dir = deliverable_grader_dir(private_dir, deliverable.id)

            # public/checks/<id>/README.md
            self._write_text(
                checks_dir / "README.md",
                self._starter_readme(spec, deliverable.id),
                ArtifactVisibility.public,
                files,
                bundle_root,
                role="starter_readme",
                audience="learner",
                deliverable_id=deliverable.id,
                semantic_source="spec_rendered",
            )

            workspace_root_for_run = (
                Path(run.artifacts.workspace_snapshot.root_dir)
                if run.artifacts.workspace_snapshot is not None
                else None
            )

            # Source for visible/hidden scripts + manifest: prefer the authored workspace
            # (under public/checks/<id>/ and private/grader/<id>/), else default templates.
            authored_visible = None
            authored_hidden = None
            authored_manifest_text = None
            if (
                workspace_root_for_run is not None
                and workspace_root_for_run.resolve() != bundle_root.resolve()
            ):
                ws_visible = (
                    Path(run.artifacts.workspace_snapshot.public_dir)
                    / "checks"
                    / deliverable.id
                    / VISIBLE_CHECK_SCRIPT_RELATIVE_PATH
                )
                ws_hidden = (
                    workspace_root_for_run
                    / "private"
                    / "grader"
                    / deliverable.id
                    / HIDDEN_GRADER_SCRIPT_RELATIVE_PATH
                )
                ws_manifest = (
                    workspace_root_for_run
                    / "private"
                    / "grader"
                    / deliverable.id
                    / DELIVERABLE_MANIFEST_RELATIVE_PATH
                )
                if ws_visible.exists():
                    try:
                        authored_visible = ws_visible.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        authored_visible = None
                if ws_hidden.exists():
                    try:
                        authored_hidden = ws_hidden.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        authored_hidden = None
                if ws_manifest.exists():
                    try:
                        authored_manifest_text = ws_manifest.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        authored_manifest_text = None

            visible_content = (
                authored_visible
                if authored_visible is not None
                else default_starter_files.get("checks/run_visible_checks.py", "")
            )
            hidden_content = (
                authored_hidden
                if authored_hidden is not None
                else default_starter_files.get(HIDDEN_GRADER_SCRIPT_PATH, "")
            )
            if deliverable.id == first_deliverable.id or authored_manifest_text is None:
                # Default templates produce the same manifest content per deliverable.
                manifest_for_this_deliverable = build_task_agent_starter_files(spec, deliverable.id)
                manifest_text = (
                    authored_manifest_text
                    if authored_manifest_text is not None
                    else manifest_for_this_deliverable[HIDDEN_MANIFEST_PATH]
                )
            else:
                manifest_text = authored_manifest_text
            # Log which path produced the manifest. `default_template`
            # means we lost any prior authored state (starter_repo_bundle
            # reverts to `starter_default`).
            try:
                preview = json.loads(manifest_text)
                preview_source = (preview.get("starter_repo_bundle") or {}).get("source")
            except (json.JSONDecodeError, AttributeError):
                preview_source = None
            log_coursegen_event(
                "materializer_manifest_resolved",
                workflow_run_id=run.id,
                deliverable_id=deliverable.id,
                is_first_deliverable=(deliverable.id == first_deliverable.id),
                authored_manifest_text_present=authored_manifest_text is not None,
                source_path=(
                    "authored_workspace"
                    if authored_manifest_text is not None
                    else "default_template"
                ),
                starter_repo_bundle_source=preview_source,
            )

            # public/checks/<id>/run_visible_checks.py
            self._write_text(
                checks_dir / VISIBLE_CHECK_SCRIPT_RELATIVE_PATH,
                visible_content,
                ArtifactVisibility.public,
                files,
                bundle_root,
                role="visible_check_runner",
                audience="learner",
                deliverable_id=deliverable.id,
                semantic_source="starter_compiler",
            )

            # private/grader/<id>/deliverable.json
            self._write_text(
                grader_dir / DELIVERABLE_MANIFEST_RELATIVE_PATH,
                manifest_text,
                ArtifactVisibility.private,
                files,
                bundle_root,
                role="starter_manifest",
                audience="operator",
                deliverable_id=deliverable.id,
                semantic_source="starter_compiler",
            )
            # private/grader/<id>/run_hidden_checks.py
            self._write_text(
                grader_dir / HIDDEN_GRADER_SCRIPT_RELATIVE_PATH,
                hidden_content,
                ArtifactVisibility.private,
                files,
                bundle_root,
                role="runtime_hidden_check_script",
                audience="operator",
                deliverable_id=deliverable.id,
                semantic_source="starter_compiler",
            )

    def _materialize_per_deliverable_starters(
        self,
        *,
        run: WorkflowRun,
        spec: TaskAgentServiceSpec,
        public_dir: Path,
        bundle_root: Path,
        files: list[BundleFile],
    ) -> None:
        """Legacy materialization path for non-shared-codebase courses."""
        for deliverable in spec.deliverables:
            deliverable_dir = public_dir / "starter" / deliverable.id
            self._write_text(
                deliverable_dir / "README.md",
                self._starter_readme(spec, deliverable.id),
                ArtifactVisibility.public,
                files,
                bundle_root,
                role="starter_readme",
                audience="learner",
                deliverable_id=deliverable.id,
                semantic_source="spec_rendered",
            )
            workspace_starter_dir = (
                Path(run.artifacts.workspace_snapshot.public_dir) / "starter" / deliverable.id
                if run.artifacts.workspace_snapshot is not None
                and Path(run.artifacts.workspace_snapshot.root_dir).resolve() != bundle_root.resolve()
                else None
            )
            if workspace_starter_dir is not None and workspace_starter_dir.exists():
                workspace_manifest = load_starter_manifest(workspace_starter_dir)
                for relative_path in self._workspace_starter_paths(
                    workspace_starter_dir=workspace_starter_dir,
                    spec=spec,
                    deliverable_id=deliverable.id,
                ):
                    source_path = workspace_starter_dir / relative_path
                    if not source_path.exists() or not source_path.is_file():
                        continue
                    role, audience, semantic_source = self._starter_file_metadata(
                        relative_path,
                        manifest_payload=workspace_manifest,
                    )
                    try:
                        content = source_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    self._write_text(
                        deliverable_dir / relative_path,
                        content,
                        ArtifactVisibility.public,
                        files,
                        bundle_root,
                        role=role,
                        audience=audience,
                        deliverable_id=deliverable.id,
                        semantic_source=semantic_source,
                    )
            else:
                starter_files = build_task_agent_starter_files(spec, deliverable.id)
                starter_manifest = json.loads(starter_files[HIDDEN_MANIFEST_PATH])
                for relative_path, content in starter_files.items():
                    role, audience, semantic_source = self._starter_file_metadata(
                        relative_path,
                        manifest_payload=starter_manifest,
                    )
                    self._write_text(
                        deliverable_dir / relative_path,
                        content,
                        ArtifactVisibility.public,
                        files,
                        bundle_root,
                        role=role,
                        audience=audience,
                        deliverable_id=deliverable.id,
                        semantic_source=semantic_source,
                    )
                self._write_visible_fixture_files(
                    spec=spec,
                    deliverable_dir=deliverable_dir,
                    deliverable_id=deliverable.id,
                    files=files,
                    bundle_root=bundle_root,
                )

    def _task_agent_readme(self, spec: TaskAgentServiceSpec) -> str:
        runtime_plan = spec.project_contract.runtime_plan
        stack_bits = [
            runtime_plan.implementation_language,
            runtime_plan.language_version,
            runtime_plan.application_framework,
            runtime_plan.framework_version,
        ]
        stack_summary = " ".join(bit for bit in stack_bits if bit) or "not specified"
        system_profile = ", ".join(f"`{label}`" for label in spec.capabilities.summary_labels())
        visible_fixtures = ", ".join(f"`{path}`" for path in spec.runtime_dependencies.visible_fixture_files) or "`none`"
        lines = [
            f"# {spec.title}",
            "",
            spec.summary,
            "",
            f"- Package type: `{spec.package_type.value}`",
            f"- Project family: `{spec.project_contract.family.value}`",
            f"- System kind: {spec.project_contract.system_kind}",
            f"- Runtime stack: {stack_summary}",
            f"- Execution surface: `{spec.runtime_dependencies.execution_surface.value}`",
            f"- System profile: {system_profile}",
            f"- Visible fixtures: {visible_fixtures}",
            "",
            "## Public service surface",
            "",
        ]
        lines.extend(f"- `{endpoint.method} {endpoint.path}`" for endpoint in spec.public_endpoints if endpoint.required)
        lines.extend(["", "## Runtime components", ""])
        for service in runtime_plan.services:
            technology = f" using `{service.technology}`" if service.technology else ""
            lines.append(f"- `{service.service_id}` ({service.role}){technology}")
        lines.extend(["", "## Deliverable arc", ""])
        for deliverable in spec.deliverables:
            lines.append(f"- `{deliverable.id}` - {deliverable.title}: {deliverable.objective}")
        return "\n".join(lines)

    def _course_outline(self, spec: TaskAgentServiceSpec) -> str:
        lines = ["# Course Outline", ""]
        for deliverable in spec.deliverables:
            gate = spec.gate_for(deliverable.id)
            lines.extend(
                [
                    f"## {deliverable.id}: {deliverable.title}",
                    "",
                    deliverable.objective,
                    "",
                    f"- Starter type: `{spec.runtime_dependencies.starter_type.value}`",
                    f"- Active visible checks: {', '.join(f'`{item}`' for item in gate.active_public_check_ids) or 'none'}",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    def _write_visible_fixture_files(
        self,
        *,
        spec: TaskAgentServiceSpec,
        deliverable_dir: Path,
        deliverable_id: str | None,
        files: list[BundleFile],
        bundle_root: Path,
    ) -> None:
        visible_paths = list(dict.fromkeys(spec.runtime_dependencies.visible_fixture_files))
        sources_by_path = {
            source.workspace_path: source
            for source in spec.runtime_dependencies.data_sources
            if source.learner_visible and source.workspace_path
        }
        for relative_path in visible_paths:
            if not relative_path:
                continue
            content = self._visible_fixture_content(relative_path, sources_by_path.get(relative_path))
            self._write_text(
                deliverable_dir / relative_path,
                content,
                ArtifactVisibility.public,
                files,
                bundle_root,
                role="visible_fixture",
                audience="learner",
                deliverable_id=deliverable_id,
                semantic_source="source_materialized",
            )

    def _visible_fixture_content(self, relative_path: str, source) -> str:
        if source is not None and source.asset_id and self.creator_asset_service is not None:
            try:
                _record, content = self.creator_asset_service.read_asset_text(source.asset_id)
                return content if content.endswith("\n") else content + "\n"
            except (FileNotFoundError, KeyError):
                pass
        description = (getattr(source, "description", None) or "Visible learner fixture.").strip()
        suffix = Path(relative_path).suffix.lower()
        if suffix == ".json":
            return json.dumps(
                {"title": getattr(source, "title", "Uploaded data source"), "description": description, "items": []},
                indent=2,
            ) + "\n"
        if suffix == ".csv":
            return "id,value\n"
        if suffix in {".md", ".markdown"}:
            return f"# {getattr(source, 'title', 'Uploaded data source')}\n\n{description}\n"
        return description + "\n"

    def _starter_readme(self, spec: TaskAgentServiceSpec, deliverable_id: str) -> str:
        deliverable = next(item for item in spec.deliverables if item.id == deliverable_id)
        brief = deliverable.learner_brief or build_task_agent_deliverable_brief(spec, deliverable)
        return render_learner_starter_readme(
            title=f"Starter for {deliverable.title}",
            brief=brief,
            summary=deliverable.objective,
            learning_outcomes=list(deliverable.learning_outcomes),
            visible_check_command=spec.runtime_dependencies.visible_check_command or "sh .coursegen/runtime/check_visible.sh",
            preview_command=spec.runtime_dependencies.preview_command or default_preview_command(spec, host="127.0.0.1"),
            public_checks=deliverable.public_checks,
            implementation_language=spec.runtime_dependencies.implementation_language,
            language_version=spec.runtime_dependencies.language_version,
            package_manager=spec.runtime_dependencies.package_manager,
        )

    def _assignment_runtime_dockerfile(self, spec: TaskAgentServiceSpec) -> str:
        bootstrap_commands = task_agent_runtime_bootstrap_commands(spec, include_python=True)
        environment_lines = task_agent_runtime_environment_lines(spec)
        lines = [
            f"FROM {task_agent_runtime_base_image(spec)}",
            "",
            "ENV PYTHONDONTWRITEBYTECODE=1",
            "ENV PYTHONUNBUFFERED=1",
            *environment_lines,
            "",
            "WORKDIR /workspace",
        ]
        if bootstrap_commands:
            lines.extend(["", "RUN " + " && \\\n    ".join(bootstrap_commands)])
        lines.extend(["", "COPY . /workspace", 'CMD ["python3", "runtime/verify_assignment.py"]', ""])
        return "\n".join(lines)

    def _assignment_runtime_readme(self) -> str:
        return "\n".join(
            [
                "# Assignment Runtime Sandbox",
                "",
                "This Docker image verifies that the generated assignment starters compile and boot before author review opens.",
                "",
            ]
        )

    def _assignment_runtime_verifier(self) -> str:
        return "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import json",
                "import os",
                "import signal",
                "import subprocess",
                "import time",
                "from pathlib import Path",
                "from urllib.error import URLError",
                "from urllib.request import Request, urlopen",
                "",
                'ROOT = Path(__file__).resolve().parents[1]',
                'STARTERS = ROOT / "starter"',
                'PORT = int(os.environ.get("ASSIGNMENT_SANDBOX_PORT", "8010"))',
                "",
                "",
                "def request_json(method: str, url: str, payload=None, timeout: float = 3.0):",
                "    data = None",
                "    headers = {}",
                "    if payload is not None:",
                "        data = json.dumps(payload).encode('utf-8')",
                "        headers['content-type'] = 'application/json'",
                "    request = Request(url, data=data, headers=headers, method=method)",
                "    with urlopen(request, timeout=timeout) as response:",
                "        body = response.read().decode('utf-8', errors='replace')",
                "        return response.status, json.loads(body) if body else {}, dict(response.headers)",
                "",
                "",
                "def wait_for_health(port: int, path: str, timeout_s: float = 12.0):",
                "    deadline = time.time() + timeout_s",
                "    last_error = None",
                "    while time.time() < deadline:",
                "        try:",
                '            status, payload, _headers = request_json("GET", f"http://127.0.0.1:{port}{path}")',
                "            return status, payload",
                "        except URLError as exc:",
                "            last_error = str(exc)",
                "            time.sleep(0.25)",
                "        except Exception as exc:",
                "            last_error = str(exc)",
                "            time.sleep(0.25)",
                "    raise RuntimeError(last_error or 'health check timed out')",
                "",
                "",
                "def terminate(proc: subprocess.Popen[str]):",
                "    if proc.poll() is not None:",
                "        return proc.communicate()",
                "    try:",
                "        os.killpg(proc.pid, signal.SIGTERM)",
                "        proc.wait(timeout=5)",
                "    except subprocess.TimeoutExpired:",
                "        os.killpg(proc.pid, signal.SIGKILL)",
                "        proc.wait(timeout=5)",
                "    return proc.communicate()",
                "",
                "",
                "def manifest(deliverable_dir: Path) -> dict[str, object]:",
                f"    manifest_path = deliverable_dir / '{HIDDEN_MANIFEST_PATH}'",
                "    return json.loads(manifest_path.read_text(encoding='utf-8'))",
                "",
                "",
                "def healthcheck_path(manifest_payload: dict[str, object]) -> str:",
                "    runtime_plan = manifest_payload.get('runtime_plan') or {}",
                "    services = runtime_plan.get('services') or []",
                "    for service in services:",
                "        if service.get('service_id') == 'app' and service.get('healthcheck_path'):",
                "            return str(service['healthcheck_path'])",
                "    return '/health'",
                "",
                "",
                "def runtime_script(deliverable_dir: Path, relative_path: str) -> Path:",
                "    return deliverable_dir / relative_path",
                "",
                "",
                "def preview_command(manifest_payload: dict[str, object]) -> str:",
                "    return str(manifest_payload.get('preview_command') or 'sh .coursegen/runtime/run.sh')",
                "",
                "",
                "def primary_check(manifest_payload: dict[str, object]) -> dict[str, object] | None:",
                "    checks = manifest_payload.get('public_checks') or []",
                "    return checks[0] if checks else None",
                "",
                "",
                "def verify_deliverable(deliverable_dir: Path, port: int):",
                "    report = {'deliverable_id': deliverable_dir.name, 'compile_succeeded': False, 'runtime_succeeded': False, 'health_status_code': None, 'stdout': '', 'stderr': '', 'error': None}",
                "    environment = os.environ.copy()",
                "    environment['PORT'] = str(port)",
                "    try:",
                "        manifest_payload = manifest(deliverable_dir)",
                f"        install_script = runtime_script(deliverable_dir, '{RUNTIME_INSTALL_SCRIPT_PATH}')",
                f"        verify_script = runtime_script(deliverable_dir, '{RUNTIME_VERIFY_SCRIPT_PATH}')",
                "        if not install_script.exists():",
                "            raise FileNotFoundError(f'missing runtime install script {install_script.relative_to(deliverable_dir)}')",
                "        if not verify_script.exists():",
                "            raise FileNotFoundError(f'missing runtime verify script {verify_script.relative_to(deliverable_dir)}')",
                f"        subprocess.run('sh {RUNTIME_INSTALL_SCRIPT_PATH}', cwd=deliverable_dir, env=environment, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)",
                f"        subprocess.run('sh {RUNTIME_VERIFY_SCRIPT_PATH}', cwd=deliverable_dir, env=environment, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)",
                "        report['compile_succeeded'] = True",
                "    except Exception as exc:",
                "        report['error'] = f'compile failed: {exc}'",
                "        return report",
                "    proc = subprocess.Popen(preview_command(manifest_payload), cwd=deliverable_dir, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, start_new_session=True)",
                "    try:",
                "        status_code, _payload = wait_for_health(port, healthcheck_path(manifest_payload))",
                "        report['runtime_succeeded'] = status_code == 200",
                "        report['health_status_code'] = status_code",
                "        check = primary_check(manifest_payload)",
                "        if check:",
                "            request_json(str(check.get('request_method') or 'POST').upper(), f\"http://127.0.0.1:{port}{check.get('request_path')}\", check.get('request_body') or None)",
                "    except Exception as exc:",
                "        report['error'] = f'runtime failed: {exc}'",
                "    finally:",
                "        stdout, stderr = terminate(proc)",
                "        report['stdout'] = stdout",
                "        report['stderr'] = stderr",
                "    return report",
                "",
                "",
                "def main():",
                "    deliverable_dirs = sorted(path for path in STARTERS.iterdir() if path.is_dir())",
                "    reports = [verify_deliverable(deliverable_dir, PORT + index) for index, deliverable_dir in enumerate(deliverable_dirs)]",
                "    success = all(item['compile_succeeded'] and item['runtime_succeeded'] for item in reports)",
                "    payload = {'success': success, 'deliverable_reports': reports, 'error': None if success else 'One or more generated starters failed sandbox verification.'}",
                "    print(json.dumps(payload))",
                "    raise SystemExit(0 if success else 1)",
                "",
                "",
                "if __name__ == '__main__':",
                "    main()",
                "",
            ]
        )

    def _shared_workspace_paths(
        self,
        *,
        workspace_shared_dir: Path,
        spec: TaskAgentServiceSpec,
        manifest: dict[str, Any] | None,
    ) -> list[str]:
        """Files to mirror from the workspace's shared starter root into the bundle.

        Excludes per-deliverable artifacts (manifest, hidden grader, visible script,
        their containing folders) — those live outside the shared starter now.
        """
        first_deliverable = spec.deliverables[0] if spec.deliverables else None
        editable_paths = (
            learner_editable_paths_for_deliverable(spec, first_deliverable)
            if first_deliverable is not None and manifest is None
            else None
        )
        paths = starter_materialization_paths(
            manifest=manifest,
            editable_paths=editable_paths,
            visible_fixture_paths=(
                list(spec.runtime_dependencies.visible_fixture_files)
                if manifest is None
                else None
            ),
        )
        # Strip per-deliverable artifacts; they live in public/checks/<id> and private/grader/<id>.
        return [
            relative_path
            for relative_path in paths
            if relative_path
            not in {
                HIDDEN_MANIFEST_PATH,
                HIDDEN_GRADER_SCRIPT_PATH,
                "checks/run_visible_checks.py",
            }
            and not relative_path.startswith("checks/")
            and not relative_path.startswith(".coursegen/grader/")
        ]

    def _workspace_starter_paths(
        self,
        *,
        workspace_starter_dir: Path,
        spec: TaskAgentServiceSpec,
        deliverable_id: str,
    ) -> list[str]:
        manifest_path = workspace_starter_dir / HIDDEN_MANIFEST_PATH
        manifest_payload: dict[str, Any] | None = None
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest_payload = None
        deliverable = next(
            (candidate for candidate in spec.deliverables if candidate.id == deliverable_id),
            None,
        )
        return starter_materialization_paths(
            manifest=manifest_payload,
            editable_paths=(
                learner_editable_paths_for_deliverable(spec, deliverable)
                if deliverable is not None and manifest_payload is None
                else None
            ),
            visible_fixture_paths=(
                list(spec.runtime_dependencies.visible_fixture_files)
                if manifest_payload is None
                else None
            ),
        )

    def _starter_file_metadata(
        self,
        relative_path: str,
        *,
        manifest_payload: dict[str, Any] | None,
    ) -> tuple[str, str, str]:
        dependency_paths = starter_contract_path_sets_for_manifest(manifest_payload)
        if relative_path == HIDDEN_MANIFEST_PATH:
            return "starter_manifest", "operator", "starter_compiler"
        if relative_path == "Dockerfile":
            return "starter_dockerfile", "operator", "starter_compiler"
        if relative_path == RUNTIME_INSTALL_SCRIPT_PATH:
            return "runtime_install_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_VERIFY_SCRIPT_PATH:
            return "runtime_verify_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_RUN_SCRIPT_PATH:
            return "runtime_run_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_VISIBLE_CHECK_SCRIPT_PATH:
            return "runtime_visible_check_script", "operator", "starter_compiler"
        if relative_path == RUNTIME_HIDDEN_CHECK_SCRIPT_PATH:
            return "runtime_hidden_check_script", "operator", "starter_compiler"
        if relative_path == "checks/run_visible_checks.py":
            return "visible_check_runner", "learner", "starter_compiler"
        if relative_path == ".vscode/tasks.json":
            return "vscode_tasks", "learner", "starter_compiler"
        if relative_path in dependency_paths["manifests"]:
            return "starter_dependency_manifest", "learner", "starter_compiler"
        if relative_path in dependency_paths["lockfiles"]:
            return "starter_dependency_lockfile", "learner", "starter_compiler"
        if relative_path in dependency_paths["toolchains"]:
            return "starter_toolchain_config", "learner", "starter_compiler"
        if relative_path in dependency_paths["build_support"]:
            return "starter_build_support", "learner", "starter_compiler"
        return "starter_entrypoint", "learner", "starter_compiler"

    def _write_text(
        self,
        path: Path,
        content: str,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
        *,
        role: str | None = None,
        audience: str | None = None,
        deliverable_id: str | None = None,
        semantic_source: str | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        files.append(
            BundleFile(
                relative_path=str(path.relative_to(bundle_root)),
                visibility=visibility,
                media_type=self._guess_media_type(path.name),
                size_bytes=path.stat().st_size,
                role=role,
                audience=audience,
                deliverable_id=deliverable_id,
                semantic_source=semantic_source,
            )
        )

    def _write_json(
        self,
        path: Path,
        payload: Any,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
        *,
        role: str | None = None,
        audience: str | None = None,
        deliverable_id: str | None = None,
        semantic_source: str | None = None,
    ) -> None:
        self._write_text(
            path,
            json.dumps(payload, indent=2) + "\n",
            visibility,
            files,
            bundle_root,
            role=role,
            audience=audience,
            deliverable_id=deliverable_id,
            semantic_source=semantic_source,
        )

    def _guess_media_type(self, filename: str) -> str:
        if filename.endswith(".json"):
            return "application/json"
        if filename.endswith(".md"):
            return "text/markdown"
        if filename.endswith(".py"):
            return "text/x-python"
        if filename.endswith(".ts"):
            return "application/typescript"
        if filename.endswith(".js"):
            return "application/javascript"
        if filename.endswith(".txt"):
            return "text/plain"
        return "text/plain"
