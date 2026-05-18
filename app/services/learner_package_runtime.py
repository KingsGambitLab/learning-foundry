from __future__ import annotations

import json
from pathlib import Path

from app.domain.grading import AssignmentGradeReport, GradeStatus, DeliverableGradeReport, ReviewAreaGradeReport
from app.domain.registry import PackageType
from app.domain.learner import LearnerDeliverableProgress
from app.domain.publish import LearnerDeliverablePackage, LearnerPackageFile, PublishSnapshot


def project_brief_markdown(snapshot: PublishSnapshot) -> str:
    learner_package = snapshot.learner_package
    if learner_package is None:
        return ""
    brief = (learner_package.project_brief_markdown or "").strip()
    if brief:
        return brief
    lines = [
        f"# {learner_package.title}",
        "",
        learner_package.summary or "Build the shared project in the workspace.",
        "",
        "## What review will look at",
        "",
    ]
    for index, deliverable in enumerate(learner_package.deliverables, start=1):
        objective = (deliverable.objective or deliverable.title).strip()
        lines.append(f"{index}. **{deliverable.title}** - {objective}")
    lines.extend(
        [
            "",
            "## How to work",
            "",
            "- Open the shared VS Code workspace.",
            "- Run the visible checks while you iterate.",
            "- Submit the full project for review.",
            "- Use the deliverable scorecard to see which areas still need work.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def review_area_index_json(deliverables: list[LearnerDeliverablePackage]) -> str:
    rows = [
        {
            "deliverable_id": deliverable.deliverable_id,
            "title": deliverable.title,
            "objective": deliverable.objective,
            "deliverable_index": deliverable.deliverable_index,
        }
        for deliverable in deliverables
    ]
    return json.dumps({"review_areas": rows, "deliverables": rows}, indent=2, ensure_ascii=True) + "\n"


def deliverables_markdown(deliverables: list[LearnerDeliverablePackage]) -> str:
    lines = [
        "# Project deliverables",
        "",
        "Use this as the checklist for what review will look at on submission.",
        "",
    ]
    for deliverable in deliverables:
        lines.extend(
            [
                f"## {deliverable.deliverable_index}. {deliverable.title}",
                "",
                deliverable.objective,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def readme_markdown(snapshot: PublishSnapshot) -> str:
    """The single learner-facing doc: the project brief followed by a
    Deliverables section. (We used to seed three files — README.md,
    project_brief.md, deliverables.md — where the first two were
    byte-identical; one well-structured README is clearer.)

    Heading hygiene: the brief owns the only H1 (the course title).
    The Deliverables block is an H2 section with H3 per deliverable —
    NOT ``deliverables_markdown`` (that emits its own H1, kept intact
    for the tutor's context builder).
    """
    learner_package = snapshot.learner_package
    brief = project_brief_markdown(snapshot).rstrip()
    parts = [
        brief,
        "",
        "---",
        "",
        "## Deliverables",
        "",
        "Checklist of what the review scores when you submit:",
        "",
    ]
    deliverables = learner_package.deliverables if learner_package is not None else []
    for index, deliverable in enumerate(deliverables, start=1):
        parts.append(f"### {index}. {deliverable.title}")
        parts.append("")
        objective = (deliverable.objective or "").strip()
        if objective:
            parts.append(objective)
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# Platform-managed dirs the learner should never have to look at. We
# can't delete them (`.coursegen` is the runtime/grader protocol; see
# course-gen-backlog §26) — but we can hide them from the code-server
# Explorer/search via a workspace settings file.
_HARNESS_HIDE_GLOBS = {"**/.coursegen": True, "**/.coursegen_data": True}


def vscode_settings_hiding_harness(existing: str | None = None) -> str:
    """Return `.vscode/settings.json` content that hides the
    platform-managed harness dirs from the Explorer + search. MERGES
    into any course-provided settings (never clobbers other keys)."""
    data: dict = {}
    if existing:
        try:
            loaded = json.loads(existing)
            if isinstance(loaded, dict):
                data = loaded
        except (ValueError, TypeError):
            data = {}
    for key in ("files.exclude", "search.exclude"):
        merged = dict(data.get(key) or {})
        merged.update(_HARNESS_HIDE_GLOBS)
        data[key] = merged
    return json.dumps(data, indent=2) + "\n"


def seed_workspace_from_snapshot(workspace_root: str | Path, snapshot: PublishSnapshot) -> Path:
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    learner_package = snapshot.learner_package
    if learner_package is None or not learner_package.deliverables:
        raise ValueError("This publish snapshot is missing learner workspace seed files.")

    files_to_write: dict[str, str] = {}
    for file in _workspace_seed_source_files(learner_package.deliverables, learner_package.package_type):
        files_to_write[file.relative_path] = file.content
    # One consolidated learner doc. (Previously also wrote
    # project_brief.md — an exact dup of README — and a standalone
    # deliverables.md; folded into README for a single source.)
    files_to_write["README.md"] = readme_markdown(snapshot)
    files_to_write[".coursegen/workspace_seeded.txt"] = snapshot.id + "\n"
    # NOTE: review_areas/index.json + deliverables/index.json are no
    # longer seeded — nothing reads them at runtime or in validation;
    # they were learner-visible clutter. The per-deliverable
    # review-area README is still required by
    # validate_seeded_learner_workspace (publish certification gate), so
    # it stays for now (see backlog: retire it + retarget the validator).
    for deliverable in learner_package.deliverables:
        files_to_write[f".coursegen/review_areas/{deliverable.deliverable_id}/README.md"] = deliverable.starter_readme
    # Hide the platform-managed harness dirs from the learner's editor
    # (merges with a course-provided .vscode/settings.json if present).
    files_to_write[".vscode/settings.json"] = vscode_settings_hiding_harness(
        files_to_write.get(".vscode/settings.json")
    )

    for relative_path, content in files_to_write.items():
        target = root / relative_path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _workspace_seed_source_files(
    deliverables: list[LearnerDeliverablePackage],
    package_type: PackageType,
) -> list[LearnerPackageFile]:
    if not deliverables:
        return []
    if package_type == PackageType.progressive_codebase_course:
        first_deliverable = min(deliverables, key=lambda deliverable: deliverable.deliverable_index)
        return list(first_deliverable.workspace_seed_files)
    files: list[LearnerPackageFile] = []
    for deliverable in deliverables:
        files.extend(deliverable.workspace_seed_files)
    return files


def remap_assignment_report_to_deliverables(
    snapshot: PublishSnapshot,
    assignment_report: AssignmentGradeReport,
) -> AssignmentGradeReport:
    learner_package = snapshot.learner_package
    if learner_package is None:
        return assignment_report

    spec_deliverable_order = (
        {
            deliverable.id: index
            for index, deliverable in enumerate(snapshot.task_agent_spec.deliverables)
        }
        if snapshot.task_agent_spec is not None
        else {}
    )
    deliverable_positions = {
        deliverable.deliverable_id: position
        for position, deliverable in enumerate(learner_package.deliverables)
    }
    aggregated: dict[str, ReviewAreaGradeReport] = {}
    for review_area in assignment_report.review_areas:
        learner_deliverable = resolve_review_area_deliverable(
            learner_package.deliverables,
            review_area.deliverable_id,
            spec_deliverable_order,
        )
        learner_deliverable_id = (
            learner_deliverable.deliverable_id if learner_deliverable is not None else review_area.deliverable_id
        )
        learner_title = learner_deliverable.title if learner_deliverable is not None else review_area.title
        learner_objective = learner_deliverable.objective if learner_deliverable is not None else review_area.objective
        learner_index = (
            learner_deliverable.deliverable_index
            if learner_deliverable is not None
            else review_area.deliverable_index
        )

        existing = aggregated.get(learner_deliverable_id)
        if existing is None:
            aggregated[learner_deliverable_id] = ReviewAreaGradeReport(
                deliverable_id=learner_deliverable_id,
                title=learner_title,
                objective=learner_objective,
                deliverable_index=learner_index,
                grade_report=review_area.grade_report.model_copy(update={"deliverable_id": learner_deliverable_id}),
            )
            continue

        merged_results = [*existing.grade_report.results, *review_area.grade_report.results]
        merged_warnings = list(dict.fromkeys([
            *existing.grade_report.submission_warnings,
            *review_area.grade_report.submission_warnings,
        ]))
        passed_tests = existing.grade_report.passed_tests + review_area.grade_report.passed_tests
        total_tests = existing.grade_report.total_tests + review_area.grade_report.total_tests
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests else 0.0
        aggregated[learner_deliverable_id] = ReviewAreaGradeReport(
            deliverable_id=learner_deliverable_id,
            title=learner_title,
            objective=learner_objective,
            deliverable_index=learner_index,
            grade_report=DeliverableGradeReport(
                deliverable_id=learner_deliverable_id,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                pass_rate=pass_rate,
                status=GradeStatus.passed if failed_tests == 0 else GradeStatus.failed,
                results=merged_results,
                submission_warnings=merged_warnings,
            ),
        )

    ordered_review_areas = sorted(
        aggregated.values(),
        key=lambda item: (
            deliverable_positions.get(item.deliverable_id, item.deliverable_index - 1),
            item.deliverable_index,
            item.deliverable_id,
        ),
    )
    passed_tests = sum(item.grade_report.passed_tests for item in ordered_review_areas)
    total_tests = sum(item.grade_report.total_tests for item in ordered_review_areas)
    failed_tests = total_tests - passed_tests
    pass_rate = passed_tests / total_tests if total_tests else 0.0
    return AssignmentGradeReport(
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        pass_rate=pass_rate,
        status=GradeStatus.passed if failed_tests == 0 else GradeStatus.failed,
        review_areas=ordered_review_areas,
        submission_warnings=assignment_report.submission_warnings,
    )


def resolve_review_area_deliverable(
    deliverables: list[LearnerDeliverablePackage] | list[LearnerDeliverableProgress],
    spec_deliverable_id: str,
    spec_deliverable_order: dict[str, int] | None = None,
) -> LearnerDeliverablePackage | LearnerDeliverableProgress | None:
    for deliverable in deliverables:
        if deliverable.deliverable_id == spec_deliverable_id:
            return deliverable
    if spec_deliverable_order is not None:
        deliverable_position = spec_deliverable_order.get(spec_deliverable_id)
        if deliverable_position is not None and 0 <= deliverable_position < len(deliverables):
            return deliverables[deliverable_position]
    return None
