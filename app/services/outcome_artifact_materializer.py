"""File materializer for the single-outcome course pipeline (Wave 4).

The outcome graph's nodes produce in-memory artifacts (starter files,
oracle authoring result, course spec). This module owns turning those
in-memory artifacts into on-disk files at the canonical paths the rest
of the pipeline expects:

    <workspace_root>/
    ├── public/
    │   ├── README.md                   # (legacy materializer's job)
    │   └── starter/
    │       ├── Dockerfile
    │       ├── (app source)
    │       └── ...
    └── private/
        ├── course_spec.json
        └── grader/
            ├── runner.py               # GRADER_RUNNER_SCRIPT_SOURCE
            ├── scenarios/*.yaml
            ├── _reference/
            ├── _setup/
            └── _oracle/outputs.json    # written by oracle_pass node

Every function is a pure file-write helper:

  - parents are created (parents=True, exist_ok=True),
  - existing files are overwritten,
  - relative_path entries are validated to live inside the destination
    directory (no ``..`` traversal, no absolute paths).

Tests use ``tmp_path`` fixtures and never touch real workspaces.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath

from app.services.course_outcome_models import CourseOutcomeSpec
from app.services.grader_runner_script_template import GRADER_RUNNER_SCRIPT_SOURCE
from app.services.oracle_authoring import OracleAuthoringResult
from app.services.readme_templater import render_outcome_readme
from app.services.visible_checks_script_template import (
    VISIBLE_CHECKS_SCRIPT_SOURCE,
)

__all__ = [
    "materialize_starter",
    "materialize_oracle_bundle",
    "materialize_grader_runner",
    "materialize_course_spec",
    "materialize_readme",
    "materialize_visible_samples",
]


# ---------------- helpers ----------------


def _safe_write(dest_root: Path, relative_path: str, content: str) -> Path:
    """Write ``content`` to ``dest_root / relative_path`` after validating
    the path stays inside ``dest_root``.

    Raises ValueError for absolute paths or ``..`` traversal.
    """
    if not relative_path:
        raise ValueError("relative_path must be non-empty")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute():
        raise ValueError(f"relative_path must not be absolute: {relative_path!r}")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"relative_path must not contain '..': {relative_path!r}")

    target = dest_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


# ---------------- starter ----------------


def materialize_starter(
    workspace_root: Path, starter_files: list[tuple[str, str]]
) -> None:
    """Write every ``(relative_path, content)`` entry to
    ``<workspace_root>/public/starter/<relative_path>``.

    This function is **idempotent** and treats ``public/starter/`` as
    exclusively owned by the most recent call: the entire subtree is
    deleted before writing so files present in a previous attempt that
    do not appear in ``starter_files`` are removed. Files outside
    ``public/starter/`` (e.g. ``public/README.md`` from the legacy
    materializer) are left untouched.

    Raises ValueError if any entry's path is absolute or escapes
    ``public/starter/`` via ``..``. Disk errors during the pre-write
    cleanup surface as-is — we deliberately do not pass
    ``ignore_errors=True`` so genuine permission / I/O problems are
    visible at the caller.
    """
    starter_root = Path(workspace_root) / "public" / "starter"
    # Atomic subtree replacement (Finding G): wipe the previous attempt's
    # contents before writing this attempt's so stale files cannot leak
    # into the published bundle.
    if starter_root.exists():
        shutil.rmtree(starter_root, ignore_errors=False)
    starter_root.mkdir(parents=True, exist_ok=True)
    # HARDCODED STARTER REPAIR (2026-05-14 live-run finding):
    # the OpenAI starter author frequently:
    #   (a) emits a Dockerfile with ``RUN ./.coursegen/runtime/verify.sh``
    #       at BUILD time, where verify.sh imports ``fastapi.testclient``
    #       (which requires ``httpx``);
    #   (b) omits ``httpx`` from requirements.txt.
    # This combination crashes ``docker build`` before the container ever
    # boots, and the starter_verify retry loop exhausts its 3-attempt
    # budget on the same defect. Patch both at materialize time so the
    # smoke advances. Real fix belongs in the starter-author prompt.
    patched: list[tuple[str, str]] = []
    for relative_path, content in starter_files:
        if relative_path == "Dockerfile":
            content = "\n".join(
                line
                for line in content.splitlines()
                if "verify.sh" not in line or line.lstrip().startswith("RUN chmod")
            ) + ("\n" if content.endswith("\n") else "")
        elif relative_path == "requirements.txt":
            if "httpx" not in content.lower():
                content = content.rstrip() + "\nhttpx>=0.27\n"
        patched.append((relative_path, content))
    for relative_path, content in patched:
        _safe_write(starter_root, relative_path, content)


# ---------------- oracle bundle ----------------


def materialize_oracle_bundle(
    workspace_root: Path, result: OracleAuthoringResult
) -> None:
    """Write the oracle authoring result to its canonical sub-trees:

    - scenarios → ``private/grader/scenarios/<filename>``
    - reference_files → ``private/grader/_reference/<relative_path>``
    - setup_files → ``private/grader/_setup/<relative_path>``

    This function is **idempotent** and treats each of those three
    subtrees as exclusively owned by the most recent call: they are
    deleted in full before writing so files present in a previous
    attempt that do not appear in ``result`` are removed (Finding G).

    Sibling subtrees under ``private/grader/`` are deliberately
    preserved:

    - ``_oracle/`` (written by :func:`oracle_pass.persist_oracle_outputs`
      *after* this materializer runs — wiping it on a re-author would
      drop the captured ground truth);
    - ``runner.py`` (written by :func:`materialize_grader_runner`
      independently of this function).

    Disk errors during pre-write cleanup surface as-is
    (``ignore_errors=False``) so a genuine permission / I/O problem is
    not silently masked.
    """
    grader_root = Path(workspace_root) / "private" / "grader"

    # Atomic subtree replacement (Finding G). We wipe only the three
    # subtrees this function owns; ``_oracle/`` and ``runner.py`` are
    # written by other steps and must persist across re-author retries.
    for owned in ("scenarios", "_reference", "_setup"):
        target = grader_root / owned
        if target.exists():
            shutil.rmtree(target, ignore_errors=False)

    scenarios_dir = grader_root / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    for sf in result.scenarios:
        # HARDCODED PATH NORMALIZATION (2026-05-14 live-run finding):
        # the LLM emits filenames with a redundant "scenarios/" prefix
        # (e.g. "scenarios/001_happy.yaml"), which the materializer would
        # otherwise expand to "private/grader/scenarios/scenarios/001_happy.yaml"
        # — and ``load_scenarios_from_dir(private/grader/scenarios)`` then
        # finds zero YAMLs, blocking publish via "Required scenario category
        # 'X' has no scenarios". Strip the LLM's leading "scenarios/"
        # segment(s) so files land at the loader's expected depth.
        filename = sf.filename
        while filename.startswith("scenarios/"):
            filename = filename[len("scenarios/") :]
        _safe_write(scenarios_dir, filename, sf.yaml_content)

    ref_dir = grader_root / "_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for rf in result.reference_files:
        _safe_write(ref_dir, rf.relative_path, rf.content)

    if result.setup_files:
        setup_dir = grader_root / "_setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        for sf in result.setup_files:
            _safe_write(setup_dir, sf.relative_path, sf.content)

    # Benchmark-backed courses carry a learner-visible sample payload
    # alongside the hidden setup files. Land it under ``public/`` plus
    # the visible-check runner script so a learner can self-test
    # before submitting.
    if result.visible_sample_queries_json is not None:
        materialize_visible_samples(
            workspace_root,
            sample_queries_json=result.visible_sample_queries_json,
        )


# ---------------- grader runner ----------------


def materialize_grader_runner(workspace_root: Path) -> None:
    """Write ``GRADER_RUNNER_SCRIPT_SOURCE`` to
    ``<workspace_root>/private/grader/runner.py``."""
    runner_path = Path(workspace_root) / "private" / "grader" / "runner.py"
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(GRADER_RUNNER_SCRIPT_SOURCE)


# ---------------- course spec ----------------


# ---------------- visible samples ----------------


def materialize_visible_samples(
    workspace_root: Path, *, sample_queries_json: str
) -> None:
    """Land the learner-visible sample-query payload + the visible-check
    runner script under ``public/`` so a learner can self-test before
    submitting.

    Writes:

    - ``<workspace_root>/public/examples/sample_queries.json``
      (verbatim from ``sample_queries_json``). The caller is responsible
      for shaping the payload — see
      :func:`benchmark_loader.crag_bundle_to_visible_payload` and
      :func:`benchmark_loader.beir_bundle_to_visible_payload`.
    - ``<workspace_root>/public/checks/run_visible_checks.py``
      (verbatim from ``VISIBLE_CHECKS_SCRIPT_SOURCE``).

    Idempotent: existing files at those paths are overwritten. Parent
    directories are created as needed.

    This function is intentionally a thin file-write helper — same
    style as ``materialize_grader_runner`` — so the oracle-authoring
    path can call it without the materializer needing knowledge of the
    benchmark bundle shape.
    """
    samples_path = Path(workspace_root) / "public" / "examples" / "sample_queries.json"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(sample_queries_json)

    script_path = Path(workspace_root) / "public" / "checks" / "run_visible_checks.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(VISIBLE_CHECKS_SCRIPT_SOURCE)


def materialize_course_spec(
    workspace_root: Path, spec: CourseOutcomeSpec
) -> None:
    """Write the spec to ``<workspace_root>/private/course_spec.json``
    as pretty-printed JSON (model_dump(mode='json'))."""
    target = Path(workspace_root) / "private" / "course_spec.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = spec.model_dump(mode="json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------- learner README ----------------


def _gather_scaffold_blocks(spec: CourseOutcomeSpec) -> list[str]:
    """Ask each known sibling-scaffold module for a README block.

    The materializer doesn't depend on the scaffold modules statically:
    they ship independently (Wave 6.7b is building the RAG one in
    parallel). We try-import each known module and call its README hook
    if it exists. A missing scaffold is silently skipped — the README
    still renders, just without that scaffold's content.
    """
    blocks: list[str] = []
    # RAG family scaffold (Wave 6.7b). Only attempt the import when the
    # spec carries a benchmark source — benchmark-less specs don't need
    # RAG-specific guidance.
    if spec.benchmark is not None:
        try:
            from app.services.rag_scaffold import rag_readme_block
        except ImportError:
            # Scaffold not yet shipped — skip.
            pass
        else:
            try:
                block = rag_readme_block(spec)
            except Exception:
                # A scaffold that raises must NOT take down the README.
                # The materializer's contract is "the README always
                # writes"; scaffold bugs are a follow-up.
                block = ""
            if block:
                blocks.append(block)
    return blocks


def materialize_readme(workspace_root: Path, spec: CourseOutcomeSpec) -> None:
    """Render ``public/README.md`` from the spec via the capability-gated
    templater.

    Sibling family scaffolds (e.g. ``app.services.rag_scaffold``) are
    consulted through :func:`_gather_scaffold_blocks` so a scaffold can
    contribute a markdown block without the materializer needing a
    static dependency on it. Missing scaffolds are silently skipped.

    Idempotent: an existing ``public/README.md`` is overwritten.
    """
    blocks = _gather_scaffold_blocks(spec)
    readme_text = render_outcome_readme(spec, scaffold_blocks=blocks)
    target = Path(workspace_root) / "public" / "README.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(readme_text)
