"""Synthesize a ``PublishSnapshot`` for an outcome-mode course.

The legacy publish flow goes through ``CourseWorkflowService.publish_course_run``
which calls ``PublishSnapshotService.create_snapshot``. That service
demands a ``shared_workflow_run_id`` and a ``TaskAgentServiceSpec``
embedded in the workflow run's artifacts — neither exists for an
outcome-mode course.

Without a snapshot, the LMS catalog correctly marks the course as
"Not learner-ready": ``_lms_support`` requires a non-null
``latest_publish_snapshot_id`` whose ``learner_package`` carries at
least one deliverable.

This module produces that snapshot from the artifacts on disk inside
``state.workspace_root``: it gathers ``public/starter/``,
``public/checks/``, ``public/examples/``, and ``public/README.md`` into
a single ``LearnerDeliverablePackage`` wrapping the whole course bundle,
then writes a ``PublishSnapshot`` row. ``task_agent_spec`` stays
``None`` — outcome-mode courses don't carry the legacy spec shape — and
``LMSService._lms_support`` is amended in parallel to skip the
``task_agent_spec`` requirement when an outcome snapshot is present.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain.course import CourseRun
from app.domain.learner import LearnerWorkspaceScope
from app.domain.publish import (
    LearnerCoursePackage,
    LearnerDeliverablePackage,
    LearnerPackageFile,
    PublishSnapshot,
    PublishSnapshotProvenance,
)
from app.domain.task_agent import LearnerDeliverableBrief


UTC = timezone.utc


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return ""


def _media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".md": "text/markdown",
        ".py": "text/x-python",
        ".json": "application/json",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".txt": "text/plain",
        ".sh": "text/x-shellscript",
    }.get(suffix, "text/plain")


def _gather_seed_files(workspace_root: Path) -> list[LearnerPackageFile]:
    """Collect ``public/`` files into a flat seed list.

    The learner's workspace gets seeded from these on enroll. We include
    ``starter/``, ``checks/``, and ``examples/`` so the learner can run
    visible self-tests without leaving the workspace. ``README.md`` at
    the public root rides along too.

    Skips dotfiles inside ``starter/.coursegen/`` are NOT skipped on
    purpose — the runtime scripts (install/run/verify) need to be
    present so the workspace can boot.
    """
    public_dir = workspace_root / "public"
    if not public_dir.exists():
        return []
    seed_files: list[LearnerPackageFile] = []
    for path in sorted(public_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace_root)
        # Skip Python bytecode and lock artifacts.
        if path.suffix in (".pyc", ".pyo") or "__pycache__" in path.parts:
            continue
        seed_files.append(
            LearnerPackageFile(
                relative_path=str(rel),
                media_type=_media_type_for(path),
                content=_read_text_safe(path),
            )
        )
    return seed_files


def _learner_brief_from_spec(spec: Any) -> LearnerDeliverableBrief:
    """Map a ``CourseOutcomeSpec`` into the legacy ``LearnerDeliverableBrief``."""
    files_to_edit: list[str] = []
    # Best effort: surface ``app.py`` / ``main.py`` if present in the
    # starter bundle. We don't have access to disk here — leave the
    # list empty and let the README guide the learner.
    definition_of_done = [
        f"Quality bar `{bar.id}` ({bar.threshold}) passes"
        for bar in (spec.quality_bars or [])
    ]
    example_scenarios = [bar.metric_description for bar in (spec.quality_bars or [])]
    implementation_hints = [
        hint.hint for hint in (spec.learning_path or []) if hint.hint
    ]
    return LearnerDeliverableBrief(
        why_this_deliverable_matters=spec.goal,
        task_to_build=spec.goal,
        files_to_edit=files_to_edit,
        definition_of_done=definition_of_done,
        example_scenarios=example_scenarios,
        implementation_hints=implementation_hints,
        non_goals=[],
    )


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_outcome_publish_snapshot(
    course_run: CourseRun, state: Any
) -> PublishSnapshot:
    """Build (but DON'T save) a ``PublishSnapshot`` for an outcome course.

    The caller persists it via ``store.save_publish_snapshot`` and then
    updates ``course_run.latest_publish_snapshot_id``.
    """
    workspace_root = Path(state.workspace_root)
    public_readme = workspace_root / "public" / "README.md"
    readme_text = _read_text_safe(public_readme) if public_readme.exists() else ""
    seed_files = _gather_seed_files(workspace_root)
    spec = state.spec
    deliverable = LearnerDeliverablePackage(
        deliverable_id="outcome_main",
        course_deliverable_slug="outcome-main",
        title=spec.title,
        objective=spec.goal,
        deliverable_index=1,
        learner_brief=_learner_brief_from_spec(spec),
        public_checks=[],
        content_markdown=readme_text,
        starter_readme=readme_text,
        learning_outcomes=[hint.on_metric_fail for hint in (spec.learning_path or [])],
        active_test_ids=[bar.id for bar in (spec.quality_bars or [])],
        completion_rule="all_quality_bars_pass",
        visible_files=[f.relative_path for f in seed_files if f.relative_path.startswith("public/starter/")],
        workspace_seed_files=seed_files,
    )
    learner_package = LearnerCoursePackage(
        course_run_id=course_run.id,
        title=spec.title,
        summary=spec.goal,
        package_type=course_run.package_type,
        published_at=datetime.now(UTC),
        workspace_scope=LearnerWorkspaceScope.shared_course,
        project_brief_markdown=readme_text,
        deliverables=[deliverable],
        notes=["Synthesized from outcome-mode publish."],
    )
    provenance = PublishSnapshotProvenance(
        generator_version="outcome-publish-snapshot-v1",
        course_run_hash=_stable_hash(course_run.model_dump(mode="json")),
        workflow_run_hashes={},
        workflow_bundle_ids={},
        course_bundle_id=None,
    )
    snapshot_id = f"publish_outcome_{course_run.id[-12:]}_{int(time.time())}"
    source_hash = _stable_hash(
        {
            "course_run_id": course_run.id,
            "learner_package": learner_package.model_dump(mode="json"),
            "spec_title": spec.title,
        }
    )
    return PublishSnapshot(
        id=snapshot_id,
        course_run_id=course_run.id,
        course_family_id=course_run.course_family_id,
        created_at=datetime.now(UTC),
        version=1,
        source_hash=source_hash,
        shared_workflow_run_id=None,  # outcome-mode courses have no workflow run
        learner_package=learner_package,
        task_agent_spec=None,  # legacy field; not applicable for outcome courses
        learner_certification=None,
        provenance=provenance,
        notes=["outcome-mode publish; task_agent_spec intentionally null"],
    )
