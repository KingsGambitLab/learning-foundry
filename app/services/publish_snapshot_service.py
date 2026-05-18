from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace
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
from app.services.task_agent_starter_templates import default_preview_command
from app.services.workflow_service import WorkflowService
from app.storage.workflow_store import WorkflowStore


class PublishSnapshotService:
    def __init__(self, store: WorkflowStore, workflow_service: WorkflowService) -> None:
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
        course_deliverables = list(course_run.deliverables)
        for index, spec_deliverable in enumerate(spec_deliverables, start=1):
            course_deliverable = (
                course_deliverables[index - 1]
                if index - 1 < len(course_deliverables)
                else None
            )
            deliverable_title = (
                course_deliverable.title if course_deliverable is not None else spec_deliverable.title
            )
            deliverable_summary = (
                course_deliverable.summary if course_deliverable is not None else spec_deliverable.objective
            )
            deliverable_slug = (
                course_deliverable.deliverable_slug
                if course_deliverable is not None
                else spec_deliverable.id.replace("_", "-")
            )
            learning_outcomes = list(
                course_deliverable.learning_outcomes
                if course_deliverable is not None
                else spec_deliverable.learning_outcomes
            )
            course_deliverable_view = (
                course_deliverable
                if course_deliverable is not None
                else SimpleNamespace(
                    title=deliverable_title,
                    summary=deliverable_summary,
                    learning_outcomes=learning_outcomes,
                )
            )
            gate = spec.gate_for(spec_deliverable.id)
            learner_brief = combine_learner_deliverable_briefs(
                fallback_task=(
                    "Extend the learner-visible starter so it satisfies "
                    + deliverable_summary.rstrip(".").lower()
                    + "."
                ),
                fallback_why=deliverable_summary,
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
                course_deliverable=course_deliverable_view,
                deliverable_index=index,
                learner_brief=learner_brief,
                public_checks=public_checks,
            )
            starter_readme = self._learner_starter_readme(
                spec=spec,
                course_deliverable_title=deliverable_title,
                course_deliverable_summary=deliverable_summary,
                learning_outcomes=learning_outcomes,
                learner_brief=learner_brief,
                public_checks=public_checks,
            )
            seed_files = self._workspace_seed_files(
                spec=spec,
                workflow_run=workflow_run,
                workflow_run_id=workflow_run.id,
                spec_deliverable_id=(spec_deliverable.id if spec_deliverable is not None else None),
                content_markdown=content_markdown,
                starter_readme=starter_readme,
            )
            deliverable_packages.append(
                LearnerDeliverablePackage(
                    deliverable_id=deliverable_slug,
                    course_deliverable_slug=deliverable_slug,
                    title=deliverable_title,
                    objective=deliverable_summary,
                    deliverable_index=index,
                    learner_brief=learner_brief,
                    public_checks=public_checks,
                    content_markdown=content_markdown,
                    starter_readme=starter_readme,
                    learning_outcomes=learning_outcomes,
                    active_test_ids=list(gate.active_test_ids) if gate is not None else [],
                    completion_rule=(
                        learner_brief.definition_of_done[0]
                        if learner_brief.definition_of_done
                        else f"Complete {deliverable_title}."
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
        course_deliverable_summary: str,
        learning_outcomes: list[str],
        learner_brief,
        public_checks,
    ) -> str:
        return render_learner_starter_readme(
            title=course_deliverable_title,
            brief=learner_brief,
            summary=course_deliverable_summary,
            learning_outcomes=learning_outcomes,
            visible_check_command=spec.runtime_dependencies.visible_check_command or "sh .coursegen/runtime/check_visible.sh",
            preview_command=spec.runtime_dependencies.preview_command or default_preview_command(spec, host="127.0.0.1"),
            public_checks=public_checks,
            implementation_language=spec.runtime_dependencies.implementation_language,
            language_version=spec.runtime_dependencies.language_version,
            package_manager=spec.runtime_dependencies.package_manager,
        )

    def _workspace_seed_files(
        self,
        *,
        spec,
        workflow_run,
        workflow_run_id: str,
        spec_deliverable_id: str | None,
        content_markdown: str,
        starter_readme: str,
    ) -> list[LearnerPackageFile]:
        seed_files: list[LearnerPackageFile] = []
        if spec_deliverable_id is None or workflow_run.artifacts.materialized_bundle is None:
            return seed_files

        shared_codebase = bool(
            spec is not None
            and spec.course_structure is not None
            and spec.course_structure.shared_codebase
        )

        if shared_codebase:
            # Shared-codebase layout: source code lives at `public/starter/`
            # (no deliverable folder) and is shared across all deliverables;
            # per-deliverable visible-check artifacts live at
            # `public/checks/<id>/`.
            #
            # `learner_package_runtime._workspace_seed_source_files` takes only
            # the FIRST deliverable's seed files for shared-codebase courses,
            # so we pack everything the learner needs — shared starter content
            # + ALL per-deliverable visible check scripts — into the first
            # deliverable's seed_files. Non-first deliverables get nothing
            # (avoiding N× duplication of the source tree in the snapshot).
            first_deliverable_id = spec.deliverables[0].id if spec.deliverables else None
            if spec_deliverable_id != first_deliverable_id:
                return seed_files
            for entry in workflow_run.artifacts.materialized_bundle.files:
                if entry.visibility.value != "public":
                    continue
                path = entry.relative_path
                if path.startswith("public/starter/"):
                    # Shared starter content (source, Dockerfile, runtime
                    # scripts). Strip the prefix so it lands at the
                    # learner-workspace root.
                    stripped = path.removeprefix("public/starter/")
                    if not stripped or stripped == "README.md":
                        continue
                    seed_files.append(
                        self._read_seed_file(workflow_run_id, stripped, path)
                    )
                elif path.startswith("public/checks/"):
                    # Per-deliverable visible-check artifacts. Strip "public/"
                    # so they land at `checks/<id>/...` in the workspace. The
                    # per-deliverable README is materialized separately by
                    # `seed_workspace_from_snapshot` via
                    # `deliverable.starter_readme`, so skip it here.
                    stripped = path.removeprefix("public/")
                    if stripped.endswith("/README.md"):
                        continue
                    seed_files.append(
                        self._read_seed_file(workflow_run_id, stripped, path)
                    )
            return seed_files

        # Legacy non-shared layout: per-deliverable starter folder.
        starter_prefix = f"public/starter/{spec_deliverable_id}/"
        for entry in workflow_run.artifacts.materialized_bundle.files:
            if entry.visibility.value != "public":
                continue
            if not entry.relative_path.startswith(starter_prefix):
                continue
            relative_path = entry.relative_path.removeprefix(starter_prefix)
            if not relative_path or relative_path == "README.md":
                continue
            seed_files.append(
                self._read_seed_file(
                    workflow_run_id,
                    relative_path,
                    entry.relative_path,
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
