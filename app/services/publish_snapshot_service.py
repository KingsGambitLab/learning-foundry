from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from app.domain.course import CourseRun
from app.domain.learner import LearnerWorkspaceScope
from app.domain.publish import (
    LearnerCoursePackage,
    LearnerModulePackage,
    LearnerPackageFile,
    PublishSnapshot,
    PublishSnapshotProvenance,
)
from app.domain.workflow import MaterializeBundleRequest, WorkflowRun
from app.services.learner_brief_builder import (
    combine_public_checks,
    combine_learner_module_briefs,
    render_learner_module_markdown,
    render_learner_starter_readme,
)
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class PublishSnapshotService:
    def __init__(self, store: SQLiteWorkflowStore, workflow_service: WorkflowService) -> None:
        self.store = store
        self.workflow_service = workflow_service

    def create_snapshot(self, course_run: CourseRun, linked_runs: dict[str, WorkflowRun]) -> PublishSnapshot | None:
        shared_workflow_id = course_run.shared_workflow_run_id
        if not shared_workflow_id:
            return None

        shared_run = linked_runs.get(shared_workflow_id) or self.workflow_service.get_run(shared_workflow_id)
        if shared_run is None or shared_run.artifacts.task_agent_spec is None:
            return None
        if shared_run.artifacts.materialized_bundle is None:
            shared_run = self.workflow_service.materialize_run(
                shared_run.id,
                MaterializeBundleRequest(overwrite=True),
            )

        spec = shared_run.artifacts.task_agent_spec.model_copy(deep=True)
        latest_snapshot = self.store.get_latest_publish_snapshot(course_family_id=course_run.course_family_id)
        version = (latest_snapshot.version + 1) if latest_snapshot is not None else 1
        learner_package = self._build_learner_package(course_run, shared_run)
        provenance = PublishSnapshotProvenance(
            generator_version="publish-snapshot-v1",
            course_run_hash=self._stable_hash(course_run.model_dump(mode="json")),
            workflow_run_hashes={
                run_id: self._stable_hash(run.model_dump(mode="json"))
                for run_id, run in sorted(linked_runs.items())
            },
            workflow_bundle_ids={
                run_id: run.artifacts.materialized_bundle.bundle_id
                for run_id, run in sorted(linked_runs.items())
                if run.artifacts.materialized_bundle is not None
            },
            course_bundle_id=course_run.materialized_bundle.bundle_id if course_run.materialized_bundle is not None else None,
        )
        payload_hash = self._stable_hash(
            {
                "course_run_id": course_run.id,
                "learner_package": learner_package.model_dump(mode="json"),
                "task_agent_spec": spec.model_dump(mode="json"),
                "provenance": provenance.model_dump(mode="json"),
            }
        )
        snapshot = PublishSnapshot(
            id=f"publish_{uuid4().hex[:12]}",
            course_run_id=course_run.id,
            course_family_id=course_run.course_family_id,
            created_at=datetime.now(UTC),
            version=version,
            source_hash=payload_hash,
            shared_workflow_run_id=shared_workflow_id,
            learner_package=learner_package,
            task_agent_spec=spec,
            provenance=provenance,
            notes=[
                "Immutable learner-facing publish snapshot derived from the approved course and workflow state.",
                "Regenerate this snapshot from canonical course/workflow state instead of editing it directly.",
            ],
        )
        self.store.save_publish_snapshot(snapshot)
        return snapshot

    def _build_learner_package(self, course_run: CourseRun, workflow_run: WorkflowRun) -> LearnerCoursePackage:
        spec = workflow_run.artifacts.task_agent_spec
        assert spec is not None

        deliverable_packages: list[LearnerModulePackage] = []
        spec_modules = list(spec.modules)
        for index, course_module in enumerate(course_run.modules, start=1):
            spec_module = spec_modules[index - 1] if index - 1 < len(spec_modules) else None
            gate = spec.gate_for(spec_module.id) if spec_module is not None else None
            learner_brief = combine_learner_module_briefs(
                fallback_task=(
                    f"Extend the learner-visible starter so it satisfies {course_module.summary.rstrip('.').lower()}."
                ),
                fallback_why=course_module.summary,
                briefs=[
                    aligned_module.learner_brief
                    for aligned_module in [spec_module]
                    if aligned_module is not None and aligned_module.learner_brief is not None
                ],
            )
            public_checks = combine_public_checks(
                check
                for aligned_module in [spec_module]
                if aligned_module is not None
                for check in aligned_module.public_checks
            )
            content_markdown = self._learner_module_markdown(
                course_module=course_module,
                module_index=index,
                learner_brief=learner_brief,
                public_checks=public_checks,
            )
            starter_readme = self._learner_starter_readme(
                course_module_title=course_module.title,
                learner_brief=learner_brief,
                public_checks=public_checks,
            )
            seed_files = self._workspace_seed_files(
                spec=spec,
                workflow_run_id=workflow_run.id,
                spec_module_id=(spec_module.id if spec_module is not None else None),
                content_markdown=content_markdown,
                starter_readme=starter_readme,
            )
            deliverable_packages.append(
                LearnerModulePackage(
                    deliverable_id=course_module.deliverable_slug,
                    course_deliverable_slug=course_module.deliverable_slug,
                    title=course_module.title,
                    objective=course_module.summary,
                    deliverable_index=index,
                    learner_brief=learner_brief,
                    public_checks=public_checks,
                    content_markdown=content_markdown,
                    starter_readme=starter_readme,
                    learning_outcomes=list(course_module.learning_outcomes),
                    active_test_ids=list(gate.active_test_ids) if gate is not None else [],
                    completion_rule=(
                        learner_brief.definition_of_done[0]
                        if learner_brief.definition_of_done
                        else f"Complete {course_module.title}."
                    ),
                    visible_files=[file.relative_path for file in seed_files],
                    workspace_seed_files=seed_files,
                )
            )

        return LearnerCoursePackage(
            course_run_id=course_run.id,
            title=course_run.title,
            summary=course_run.summary,
            package_type=course_run.package_type,
            published_at=datetime.now(UTC),
            workspace_scope=LearnerWorkspaceScope.shared_course,
            project_brief_markdown=self._project_brief_markdown(course_run, deliverable_packages),
            deliverables=deliverable_packages,
            notes=[
                "Learner-visible package derived from the published project deliverables.",
                "Each deliverable is backed by the matching review area in the shared assignment spec.",
                "Active learners stay pinned to this snapshot while authors keep iterating on future drafts.",
            ],
        )

    def _project_brief_markdown(
        self,
        course_run: CourseRun,
        deliverables: list[LearnerModulePackage],
    ) -> str:
        lines = [
            f"# {course_run.title}",
            "",
            course_run.goal or course_run.summary,
            "",
            "## What we are building",
            "",
            course_run.summary or "Build the shared project in the workspace.",
            "",
            "## What review will look at",
            "",
        ]
        for index, deliverable in enumerate(deliverables, start=1):
            lines.append(f"{index}. **{deliverable.title}** - {deliverable.objective}")
        lines.extend(
            [
                "",
                "## How to work",
                "",
                "- Open the shared VS Code workspace.",
                "- Run the visible checks while you iterate.",
                "- Submit the whole project for review.",
                "- Use the deliverable scorecard to see which areas still need work.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def _learner_module_markdown(
        self,
        *,
        course_module,
        module_index: int,
        learner_brief,
        public_checks,
    ) -> str:
        return render_learner_module_markdown(
            module_index=module_index,
            title=course_module.title,
            summary=course_module.summary,
            learning_outcomes=list(course_module.learning_outcomes),
            brief=learner_brief,
            public_checks=public_checks,
        )

    def _learner_starter_readme(
        self,
        *,
        course_module_title: str,
        learner_brief,
        public_checks,
    ) -> str:
        return render_learner_starter_readme(
            title=course_module_title,
            brief=learner_brief,
            public_checks=public_checks,
        )

    def _workspace_seed_files(
        self,
        *,
        spec,
        workflow_run_id: str,
        spec_module_id: str | None,
        content_markdown: str,
        starter_readme: str,
    ) -> list[LearnerPackageFile]:
        seed_files: list[LearnerPackageFile] = [
            self._read_seed_file(workflow_run_id, "runtime/__init__.py", "public/runtime/__init__.py"),
            self._read_seed_file(workflow_run_id, "runtime/task_agent_runtime.py", "public/runtime/task_agent_runtime.py"),
            self._read_seed_file(workflow_run_id, "runtime/requirements.txt", "public/runtime/requirements.txt"),
        ]
        if spec_module_id is None:
            return seed_files
        seed_files[:0] = [
            self._read_seed_file(workflow_run_id, "app.py", f"public/starter/{spec_module_id}/app.py"),
            self._read_seed_file(
                workflow_run_id,
                "starter_manifest.json",
                f"public/starter/{spec_module_id}/starter_manifest.json",
            ),
            self._read_seed_file(
                workflow_run_id,
                "checks/run_visible_checks.py",
                f"public/starter/{spec_module_id}/checks/run_visible_checks.py",
            ),
            self._read_seed_file(
                workflow_run_id,
                ".vscode/tasks.json",
                f"public/starter/{spec_module_id}/.vscode/tasks.json",
            ),
        ]
        seen_paths = {file.relative_path for file in seed_files}
        for relative_path in spec.runtime_dependencies.visible_fixture_files:
            if not relative_path or relative_path in seen_paths:
                continue
            try:
                seed_files.append(
                    self._read_seed_file(
                        workflow_run_id,
                        relative_path,
                        f"public/starter/{spec_module_id}/{relative_path}",
                    )
                )
                seen_paths.add(relative_path)
            except FileNotFoundError:
                continue
        return seed_files

    def _read_seed_file(
        self,
        workflow_run_id: str,
        relative_path: str,
        bundle_relative_path: str,
    ) -> LearnerPackageFile:
        file = self.workflow_service.read_bundle_file(workflow_run_id, bundle_relative_path)
        return LearnerPackageFile(
            relative_path=relative_path,
            media_type=file.media_type,
            content=file.content,
        )

    def _stable_hash(self, payload: object) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
