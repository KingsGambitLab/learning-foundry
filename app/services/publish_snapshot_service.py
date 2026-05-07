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

        module_packages: list[LearnerModulePackage] = []
        checkpoint_groups = self._checkpoint_groups(course_run, spec)
        spec_modules_by_id = {module.id: module for module in spec.modules}
        for index, course_module in enumerate(course_run.modules, start=1):
            checkpoint_ids = checkpoint_groups[index - 1]
            checkpoint_modules = [
                spec_modules_by_id[checkpoint_id]
                for checkpoint_id in checkpoint_ids
                if checkpoint_id in spec_modules_by_id
            ]
            checkpoint_titles = [module.title for module in checkpoint_modules]
            entry_checkpoint_id = checkpoint_ids[0] if checkpoint_ids else None
            completion_checkpoint_id = checkpoint_ids[-1] if checkpoint_ids else None
            gate = spec.gate_for(completion_checkpoint_id) if completion_checkpoint_id else None
            learner_brief = combine_learner_module_briefs(
                fallback_task=(
                    f"Extend the learner-visible starter so it satisfies {course_module.summary.rstrip('.').lower()}."
                ),
                fallback_why=course_module.summary,
                briefs=[
                    module.learner_brief
                    for module in checkpoint_modules
                    if module.learner_brief is not None
                ],
            )
            public_checks = combine_public_checks(
                check
                for module in checkpoint_modules
                for check in module.public_checks
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
                entry_checkpoint_id=entry_checkpoint_id,
                content_markdown=content_markdown,
                starter_readme=starter_readme,
            )
            module_packages.append(
                LearnerModulePackage(
                    module_id=course_module.module_slug,
                    course_module_slug=course_module.module_slug,
                    title=course_module.title,
                    objective=course_module.summary,
                    module_index=index,
                    learner_brief=learner_brief,
                    public_checks=public_checks,
                    content_markdown=content_markdown,
                    starter_readme=starter_readme,
                    learning_outcomes=list(course_module.learning_outcomes),
                    checkpoint_module_ids=checkpoint_ids,
                    checkpoint_titles=checkpoint_titles,
                    entry_checkpoint_id=entry_checkpoint_id,
                    completion_checkpoint_id=completion_checkpoint_id,
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
            modules=module_packages,
            notes=[
                "Learner-visible package derived from the published course module ladder.",
                "Each learner module is pinned to one or more hidden assignment checkpoints so grading can evolve without changing the learner-facing course structure.",
                "Module progression is pinned to this snapshot so authors can keep iterating on future drafts without affecting active learners.",
            ],
        )

    def _checkpoint_groups(self, course_run: CourseRun, spec) -> list[list[str]]:
        course_modules = course_run.modules
        checkpoint_ids = [module.id for module in spec.modules]
        if not course_modules:
            return []
        if not checkpoint_ids:
            return [[] for _ in course_modules]

        explicit_groups = [list(module.checkpoint_module_ids) for module in course_modules]
        if all(group for group in explicit_groups):
            ordered_groups: list[list[str]] = []
            assigned: list[str] = []
            for group in explicit_groups:
                ordered = [checkpoint_id for checkpoint_id in checkpoint_ids if checkpoint_id in group]
                ordered_groups.append(ordered)
                assigned.extend(ordered)
            if len(assigned) == len(set(assigned)) and assigned == checkpoint_ids:
                return ordered_groups

        total_course_modules = len(course_modules)
        total_checkpoints = len(checkpoint_ids)
        groups: list[list[str]] = []
        for index in range(total_course_modules):
            start = (index * total_checkpoints) // total_course_modules
            end = ((index + 1) * total_checkpoints) // total_course_modules
            groups.append(checkpoint_ids[start:end])
        return groups

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
        entry_checkpoint_id: str | None,
        content_markdown: str,
        starter_readme: str,
    ) -> list[LearnerPackageFile]:
        seed_files: list[LearnerPackageFile] = [
            LearnerPackageFile(
                relative_path="README.md",
                media_type="text/markdown",
                content=starter_readme,
            ),
            LearnerPackageFile(
                relative_path="module_content.md",
                media_type="text/markdown",
                content=content_markdown,
            ),
            self._read_seed_file(workflow_run_id, "runtime/__init__.py", "public/runtime/__init__.py"),
            self._read_seed_file(workflow_run_id, "runtime/task_agent_runtime.py", "public/runtime/task_agent_runtime.py"),
            self._read_seed_file(workflow_run_id, "runtime/requirements.txt", "public/runtime/requirements.txt"),
        ]
        if entry_checkpoint_id is None:
            return seed_files
        seed_files[:0] = [
            self._read_seed_file(workflow_run_id, "app.py", f"public/starter/{entry_checkpoint_id}/app.py"),
            self._read_seed_file(
                workflow_run_id,
                "starter_manifest.json",
                f"public/starter/{entry_checkpoint_id}/starter_manifest.json",
            ),
            self._read_seed_file(
                workflow_run_id,
                "checks/run_visible_checks.py",
                f"public/starter/{entry_checkpoint_id}/checks/run_visible_checks.py",
            ),
            self._read_seed_file(
                workflow_run_id,
                ".vscode/tasks.json",
                f"public/starter/{entry_checkpoint_id}/.vscode/tasks.json",
            ),
        ]
        if "data/corpus.json" in set(spec.runtime_dependencies.visible_fixture_files):
            seed_files.append(
                LearnerPackageFile(
                    relative_path="data/corpus.json",
                    media_type="application/json",
                    content=json.dumps(
                        [
                            {
                                "doc_id": "doc:ada_lovelace",
                                "title": "Ada Lovelace",
                                "content": "Ada Lovelace was born in London, England.",
                            },
                            {
                                "doc_id": "doc:alan_turing",
                                "title": "Alan Turing",
                                "content": "Alan Turing was an English mathematician and computer scientist.",
                            },
                            {
                                "doc_id": "doc:grounding_policy",
                                "title": "Grounding policy",
                                "content": "Answer only from the visible corpus and abstain when support is missing.",
                            },
                        ],
                        indent=2,
                    )
                    + "\n",
                )
            )
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
