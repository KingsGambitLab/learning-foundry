from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from app.domain.course import CourseRun
from app.domain.learner import LearnerWorkspaceScope
from app.domain.publish import (
    LearnerCoursePackage,
    LearnerDeliverablePackage,
    LearnerPackageFile,
    PublishSnapshot,
    PublishSnapshotProvenance,
)
from app.domain.workflow import MaterializeBundleRequest, WorkflowRun
from app.services.learner_brief_builder import (
    combine_public_checks,
    combine_learner_deliverable_briefs,
    render_learner_deliverable_markdown,
    render_learner_starter_readme,
)
from app.services.task_agent_starter_templates import task_agent_starter_relative_paths
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class PublishSnapshotService:
    def __init__(self, store: SQLiteWorkflowStore, workflow_service: WorkflowService) -> None:
        self.store = store
        self.workflow_service = workflow_service

    def create_snapshot(
        self,
        course_run: CourseRun,
        linked_runs: dict[str, WorkflowRun],
        *,
        persist: bool = True,
    ) -> PublishSnapshot | None:
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
        if persist:
            self.store.save_publish_snapshot(snapshot)
        return snapshot

    def _build_learner_package(self, course_run: CourseRun, workflow_run: WorkflowRun) -> LearnerCoursePackage:
        spec = workflow_run.artifacts.task_agent_spec
        assert spec is not None

        deliverable_packages: list[LearnerDeliverablePackage] = []
        spec_deliverables = list(spec.deliverables)
        for index, course_deliverable in enumerate(course_run.deliverables, start=1):
            spec_deliverable = spec_deliverables[index - 1] if index - 1 < len(spec_deliverables) else None
            gate = spec.gate_for(spec_deliverable.id) if spec_deliverable is not None else None
            learner_brief = combine_learner_deliverable_briefs(
                fallback_task=(
                    f"Extend the learner-visible starter so it satisfies {course_deliverable.summary.rstrip('.').lower()}."
                ),
                fallback_why=course_deliverable.summary,
                briefs=[
                    aligned_deliverable.learner_brief
                    for aligned_deliverable in [spec_deliverable]
                    if aligned_deliverable is not None and aligned_deliverable.learner_brief is not None
                ],
            )
            public_checks = combine_public_checks(
                check
                for aligned_deliverable in [spec_deliverable]
                if aligned_deliverable is not None
                for check in aligned_deliverable.public_checks
            )
            content_markdown = self._learner_deliverable_markdown(
                course_deliverable=course_deliverable,
                deliverable_index=index,
                learner_brief=learner_brief,
                public_checks=public_checks,
            )
            starter_readme = self._learner_starter_readme(
                spec=spec,
                course_deliverable_title=course_deliverable.title,
                learner_brief=learner_brief,
                public_checks=public_checks,
            )
            seed_files = self._workspace_seed_files(
                spec=spec,
                workflow_run_id=workflow_run.id,
                spec_deliverable_id=(spec_deliverable.id if spec_deliverable is not None else None),
                content_markdown=content_markdown,
                starter_readme=starter_readme,
            )
            deliverable_packages.append(
                LearnerDeliverablePackage(
                    deliverable_id=course_deliverable.deliverable_slug,
                    course_deliverable_slug=course_deliverable.deliverable_slug,
                    title=course_deliverable.title,
                    objective=course_deliverable.summary,
                    deliverable_index=index,
                    learner_brief=learner_brief,
                    public_checks=public_checks,
                    content_markdown=content_markdown,
                    starter_readme=starter_readme,
                    learning_outcomes=list(course_deliverable.learning_outcomes),
                    active_test_ids=list(gate.active_test_ids) if gate is not None else [],
                    completion_rule=(
                        learner_brief.definition_of_done[0]
                        if learner_brief.definition_of_done
                        else f"Complete {course_deliverable.title}."
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
        deliverables: list[LearnerDeliverablePackage],
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

    def _learner_deliverable_markdown(
        self,
        *,
        course_deliverable,
        deliverable_index: int,
        learner_brief,
        public_checks,
    ) -> str:
        return render_learner_deliverable_markdown(
            deliverable_index=deliverable_index,
            title=course_deliverable.title,
            summary=course_deliverable.summary,
            learning_outcomes=list(course_deliverable.learning_outcomes),
            brief=learner_brief,
            public_checks=public_checks,
        )

    def _learner_starter_readme(
        self,
        *,
        spec,
        course_deliverable_title: str,
        learner_brief,
        public_checks,
    ) -> str:
        return render_learner_starter_readme(
            title=course_deliverable_title,
            brief=learner_brief,
            visible_check_command=spec.runtime_dependencies.visible_check_command or "python checks/run_visible_checks.py",
            preview_command=spec.runtime_dependencies.preview_command or "python -m uvicorn app:app --host 127.0.0.1 --port ${PORT:-8000}",
            public_checks=public_checks,
        )

    def _workspace_seed_files(
        self,
        *,
        spec,
        workflow_run_id: str,
        spec_deliverable_id: str | None,
        content_markdown: str,
        starter_readme: str,
    ) -> list[LearnerPackageFile]:
        seed_files: list[LearnerPackageFile] = [
            self._read_seed_file(workflow_run_id, "runtime/__init__.py", "public/runtime/__init__.py"),
            self._read_seed_file(workflow_run_id, "runtime/task_agent_runtime.py", "public/runtime/task_agent_runtime.py"),
            self._read_seed_file(workflow_run_id, "runtime/requirements.txt", "public/runtime/requirements.txt"),
        ]
        if spec_deliverable_id is None:
            return seed_files
        seed_files[:0] = [
            self._read_seed_file(
                workflow_run_id,
                relative_path,
                f"public/starter/{spec_deliverable_id}/{relative_path}",
            )
            for relative_path in task_agent_starter_relative_paths(spec)
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
                        f"public/starter/{spec_deliverable_id}/{relative_path}",
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
