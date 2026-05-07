from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.course import CourseReviewReport, CourseRun
from app.domain.workflow import (
    ArtifactVisibility,
    BundleFile,
    BundleFileContent,
    MaterializedBundle,
    WorkflowRun,
)
from app.services.artifact_materializer import default_generated_dir


class CourseArtifactMaterializer:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or default_generated_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def materialize_course_run(
        self,
        course_run: CourseRun,
        linked_runs: dict[str, WorkflowRun],
        review_report: CourseReviewReport | None = None,
        overwrite: bool = True,
    ) -> MaterializedBundle:
        bundle_root = self.base_dir / course_run.id
        if bundle_root.exists():
            if not overwrite:
                manifest = bundle_root / "manifest.json"
                raise FileExistsError(f"Bundle already exists at '{manifest}'.")
            shutil.rmtree(bundle_root)

        public_dir = bundle_root / "public"
        private_dir = bundle_root / "private"
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)

        files: list[BundleFile] = []
        generated_at = datetime.now(UTC)

        self._write_text(
            public_dir / "README.md",
            self._course_readme(course_run),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "content" / "syllabus.md",
            self._course_syllabus(course_run),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        self._write_text(
            public_dir / "content" / "module_sequence.md",
            self._module_sequence(course_run),
            ArtifactVisibility.public,
            files,
            bundle_root,
        )
        if review_report is not None:
            self._write_text(
                public_dir / "content" / "review.md",
                self._review_markdown(review_report),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )
        for index, module in enumerate(course_run.modules, start=1):
            linked_run = linked_runs.get(module.workflow_run_id) if module.workflow_run_id else None
            self._write_text(
                public_dir / "content" / "modules" / f"{module.module_slug}.md",
                self._module_doc(index, module, linked_run),
                ArtifactVisibility.public,
                files,
                bundle_root,
            )

        self._write_json(
            private_dir / "course_snapshot.json",
            course_run.model_dump(mode="json"),
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        self._write_json(
            private_dir / "module_index.json",
            [
                {
                    "position": index,
                    "module_slug": module.module_slug,
                    "title": module.title,
                    "design_spec": module.design_spec.model_dump(mode="json") if module.design_spec is not None else None,
                    "workflow_run_id": module.workflow_run_id,
                    "workflow_stage": module.workflow_stage,
                    "workflow_status": module.workflow_status,
                }
                for index, module in enumerate(course_run.modules, start=1)
            ],
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        self._write_json(
            private_dir / "linked_workflow_runs.json",
            {
                run_id: self._linked_run_snapshot(run)
                for run_id, run in linked_runs.items()
            },
            ArtifactVisibility.private,
            files,
            bundle_root,
        )
        if review_report is not None:
            self._write_json(
                private_dir / "review_report.json",
                review_report.model_dump(mode="json"),
                ArtifactVisibility.private,
                files,
                bundle_root,
            )

        manifest_payload = {
            "bundle_id": f"{course_run.id}_bundle",
            "generated_at": generated_at.isoformat(),
            "root_dir": str(bundle_root),
            "public_dir": str(public_dir),
            "private_dir": str(private_dir),
            "files": [entry.model_dump(mode="json") for entry in files],
        }
        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

        return MaterializedBundle(
            bundle_id=f"{course_run.id}_bundle",
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
        return BundleFileContent(
            relative_path=relative_path,
            media_type=self._guess_media_type(relative_path),
            content=target.read_text(encoding="utf-8"),
        )

    def _course_readme(self, course_run: CourseRun) -> str:
        return "\n".join(
            [
                f"# {course_run.title}",
                "",
                course_run.summary,
                "",
                f"- Package type: `{course_run.package_type.value}`",
                f"- Stage: `{course_run.stage.value}`",
                f"- Status: `{course_run.status.value}`",
                f"- Shared design: {self._design_summary(course_run.shared_design_spec)}",
                f"- Shared assignment workflow: `{course_run.shared_workflow_run_id or 'none'}`",
                "",
                "## What this bundle contains",
                "",
                "- Public syllabus and module sequencing notes",
                "- Public review report with module-by-module course readiness",
                "- Per-module author-facing docs with linked assignment workflow state",
                "- Private snapshots of course/workflow linkage",
                "",
            ]
        ) + "\n"

    def _course_syllabus(self, course_run: CourseRun) -> str:
        lines = ["# Syllabus", ""]
        for index, module in enumerate(course_run.modules, start=1):
            lines.extend(
                [
                    f"## Module {index}: {module.title}",
                    "",
                    module.summary,
                    "",
                    f"- Design: {self._design_summary(module.design_spec)}",
                    f"- Domain pack: `{module.domain_pack or 'generic'}`",
                    f"- Overlays: {', '.join(f'`{item}`' for item in module.overlays) or 'none'}",
                    f"- Assignment workflow: `{module.workflow_run_id or 'none'}`",
                    f"- Assignment status: `{module.workflow_status or 'unknown'}`",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    def _module_sequence(self, course_run: CourseRun) -> str:
        lines = ["# Module Sequence", ""]
        for index, module in enumerate(course_run.modules, start=1):
            lines.append(f"{index}. `{module.module_slug}` - {module.title}")
        lines.append("")
        return "\n".join(lines)

    def _review_markdown(self, review_report: CourseReviewReport) -> str:
        lines = [
            "# Course Review",
            "",
            f"- Course run: `{review_report.course_run_id}`",
            f"- Stage: `{review_report.stage.value}`",
            f"- Status: `{review_report.status.value}`",
            f"- Shared workflow run: `{review_report.shared_workflow_run_id or 'none'}`",
            "",
            "## Counts",
            "",
            f"- Total modules: `{review_report.counts.total_modules}`",
            f"- Ready modules: `{review_report.counts.ready_modules}`",
            f"- Modules with blockers: `{review_report.counts.modules_with_blockers}`",
            f"- Modules with linked bundles: `{review_report.counts.modules_with_bundle}`",
            f"- Linked workflow runs: `{review_report.counts.linked_workflow_runs}`",
            f"- Published workflow runs: `{review_report.counts.published_workflow_runs}`",
            f"- Workflow runs with bundles: `{review_report.counts.workflow_runs_with_bundle}`",
            "",
            "## Next Actions",
            "",
        ]
        if review_report.next_actions:
            lines.extend(f"- {action}" for action in review_report.next_actions)
        else:
            lines.append("- No immediate actions recorded.")

        lines.extend(["", "## Module Review", ""])
        for module in review_report.modules:
            lines.extend(
                [
                    f"### {module.position}. {module.title}",
                    "",
                    f"- Module slug: `{module.module_slug}`",
                    f"- Workflow run: `{module.workflow_run_id or 'none'}`",
                    f"- Ready for publish: `{module.ready_for_publish}`",
                    f"- Linked bundle available: `{module.bundle_available}`",
                ]
            )
            if module.blockers:
                lines.append("- Blockers:")
                lines.extend(f"  - {blocker}" for blocker in module.blockers)
            if module.linked_workflow is not None and module.linked_workflow.bundle is not None:
                lines.append("- Linked assignment public files:")
                lines.extend(
                    f"  - `{path}`"
                    for path in module.linked_workflow.bundle.public_files
                )
            lines.append("")
        return "\n".join(lines)

    def _module_doc(self, index: int, module, linked_run: WorkflowRun | None) -> str:
        lines = [
            f"# Module {index}: {module.title}",
            "",
            module.summary,
            "",
            f"- Module slug: `{module.module_slug}`",
            f"- Design: {self._design_summary(module.design_spec)}",
            f"- Domain pack: `{module.domain_pack or 'generic'}`",
            f"- Overlays: {', '.join(f'`{item}`' for item in module.overlays) or 'none'}",
            "",
            "## Learning outcomes",
            "",
        ]
        if module.learning_outcomes:
            lines.extend(f"- {outcome}" for outcome in module.learning_outcomes)
        else:
            lines.append("- No explicit learning outcomes recorded.")
        lines.extend(
            [
                "",
                "## Linked assignment workflow",
                "",
                f"- Workflow run: `{module.workflow_run_id or 'none'}`",
                f"- Stage: `{module.workflow_stage or 'unknown'}`",
                f"- Status: `{module.workflow_status or 'unknown'}`",
                f"- Draft kind: `{module.draft_kind or 'unknown'}`",
                f"- Recommendation status: `{module.recommendation_status or 'unknown'}`",
            ]
        )
        if linked_run is not None:
            lines.extend(
                [
                    f"- Pending gate: `{linked_run.pending_gate.value if linked_run.pending_gate else 'none'}`",
                    f"- Bundle available: `{bool(linked_run.artifacts.materialized_bundle)}`",
                ]
            )
            if linked_run.artifacts.materialized_bundle is not None:
                lines.append(
                    f"- Assignment bundle root: `{linked_run.artifacts.materialized_bundle.root_dir}`"
                )
                public_files = [
                    bundle_file.relative_path
                    for bundle_file in linked_run.artifacts.materialized_bundle.files
                    if bundle_file.visibility == ArtifactVisibility.public
                ]
                if public_files:
                    lines.append("- Assignment bundle public files:")
                    lines.extend(f"  - `{path}`" for path in public_files)
        if module.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in module.notes)
        lines.append("")
        return "\n".join(lines)

    def _linked_run_snapshot(self, run: WorkflowRun) -> dict[str, Any]:
        return {
            "run_id": run.id,
            "title": run.title,
            "stage": run.stage.value,
            "status": run.status.value,
            "pending_gate": run.pending_gate.value if run.pending_gate else None,
            "draft_kind": run.artifacts.draft_kind.value,
            "materialized_bundle": (
                run.artifacts.materialized_bundle.model_dump(mode="json")
                if run.artifacts.materialized_bundle is not None
                else None
            ),
        }

    def _design_summary(self, design_spec) -> str:
        if design_spec is None:
            return "`not specified`"
        labels = ", ".join(f"`{label}`" for label in design_spec.capabilities.summary_labels())
        return labels or "`general service workflow`"

    def _write_text(
        self,
        path: Path,
        content: str,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        files.append(
            BundleFile(
                relative_path=str(path.relative_to(bundle_root)),
                visibility=visibility,
                media_type=self._guess_media_type(path.name),
                size_bytes=path.stat().st_size,
            )
        )

    def _write_json(
        self,
        path: Path,
        payload: Any,
        visibility: ArtifactVisibility,
        files: list[BundleFile],
        bundle_root: Path,
    ) -> None:
        self._write_text(path, json.dumps(payload, indent=2) + "\n", visibility, files, bundle_root)

    def _guess_media_type(self, filename: str) -> str:
        if filename.endswith(".json"):
            return "application/json"
        if filename.endswith(".md"):
            return "text/markdown"
        if filename.endswith(".py"):
            return "text/x-python"
        return "text/plain"
