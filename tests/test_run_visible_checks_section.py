"""Every deliverable README must explain how to run the visible test
script. The platform's intended runtime path is Docker (the harness
authors a Dockerfile + install.sh that pin the language toolchain), so
the section needs to surface the Docker one-liner — plus an opt-in
local-run path that names the toolchain pre-requisites for stacks
where ``language: command not found`` is the common failure mode.
"""
from __future__ import annotations

from app.services.bundle_validation import _STARTER_README_REQUIRED_SECTIONS
from app.services.learner_brief_builder import render_learner_starter_readme
from app.domain.task_agent import LearnerDeliverableBrief


def _basic_brief() -> LearnerDeliverableBrief:
    return LearnerDeliverableBrief(
        task_to_build="Implement the corpus ingestion endpoint.",
        files_to_edit=["app/retrieval.py"],
        definition_of_done=["POST /retrieval-corpuses returns 201"],
        example_scenarios=["A small text file is uploaded; chunks land in the store."],
        why_this_deliverable_matters="Learners need a working ingest path before retrieval makes sense.",
    )


# ---------------- required-section list ----------------


def test_run_visible_checks_is_in_required_sections_list() -> None:
    """``validate_starter_readme`` consults this tuple to gate README
    completeness. Adding the new section here is what turns its
    omission into a starter_readme_missing_section finding so a
    regressing future edit fails the reviewer."""
    assert "## Run visible checks" in _STARTER_README_REQUIRED_SECTIONS


# ---------------- README content ----------------


def test_render_emits_run_visible_checks_section() -> None:
    md = render_learner_starter_readme(
        title="Deliverable 1",
        brief=_basic_brief(),
        summary="Build the corpus ingestion endpoint.",
        visible_check_command="sh .coursegen/runtime/check_visible.sh",
        preview_command="sh .coursegen/runtime/run.sh",
    )
    assert "## Run visible checks" in md
    # Generic Docker command — works without knowing the stack
    assert "docker build" in md
    assert "docker run" in md
    # Must reference the visible-check entry point
    assert "check_visible.sh" in md or "run_visible_checks.py" in md


def test_render_includes_local_path_when_language_known() -> None:
    md = render_learner_starter_readme(
        title="Deliverable 1",
        brief=_basic_brief(),
        summary="Build the corpus ingestion endpoint.",
        visible_check_command="sh .coursegen/runtime/check_visible.sh",
        preview_command="sh .coursegen/runtime/run.sh",
        implementation_language="python",
        language_version="3.11",
        package_manager="pip",
    )
    # Calling out the local-install path lets learners use a faster
    # inner loop when their toolchain is already on the host
    assert "install.sh" in md
    # And it should name the toolchain so the learner knows what's
    # missing if a command fails ("python: command not found")
    assert "python" in md.lower()
    assert "3.11" in md or "python 3.11" in md.lower()


def test_render_warns_about_missing_toolchain_for_non_python_stacks() -> None:
    """For Go / Rust / Java etc., the learner's machine almost certainly
    doesn't have the toolchain — call that out explicitly so a learner
    who sees ``go: command not found`` knows to fall back to Docker."""
    md = render_learner_starter_readme(
        title="Deliverable 1",
        brief=_basic_brief(),
        summary="Build the ingest endpoint.",
        visible_check_command="sh .coursegen/runtime/check_visible.sh",
        preview_command="sh .coursegen/runtime/run.sh",
        implementation_language="go",
        language_version="1.22",
        package_manager="gomod",
    )
    # Explicit "command not found" mention so the failure mode is
    # discoverable in the README itself
    assert "command not found" in md.lower() or "fall back to docker" in md.lower()
    assert "go" in md.lower()


def test_render_falls_back_to_generic_docker_only_when_language_unknown() -> None:
    """When the spec doesn't pin a language, do NOT invent a fake
    local-run path — only document the Docker-recommended path."""
    md = render_learner_starter_readme(
        title="Deliverable 1",
        brief=_basic_brief(),
        summary="Build the ingest endpoint.",
        visible_check_command="sh .coursegen/runtime/check_visible.sh",
        preview_command="sh .coursegen/runtime/run.sh",
    )
    assert "## Run visible checks" in md
    assert "docker run" in md


def test_run_visible_checks_block_uses_bind_mount_for_dev_loop() -> None:
    """The Docker command needs to bind-mount the workspace so the
    learner's edits flow into the container without rebuilding."""
    md = render_learner_starter_readme(
        title="Deliverable 1",
        brief=_basic_brief(),
        summary="Build the ingest endpoint.",
        visible_check_command="sh .coursegen/runtime/check_visible.sh",
        preview_command="sh .coursegen/runtime/run.sh",
    )
    # The bind-mount pattern is what makes "edit on host, run in
    # container" work without baking source into the image
    assert "-v" in md
    assert "/workspace" in md
