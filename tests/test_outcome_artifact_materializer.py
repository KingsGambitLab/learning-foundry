"""Tests for the outcome-bundle file materializer.

Covers the four entrypoints (materialize_starter, materialize_oracle_bundle,
materialize_grader_runner, materialize_course_spec) and the edge cases
that matter for downstream nodes (idempotency, nested paths, JSON
round-trip for the spec, runner script byte-equality).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CapabilityFlags,
    CourseOutcomeSpec,
    EndpointContract,
    HFBenchmarkSource,
    HttpMethod,
    JudgeKind,
    LearningHint,
    QualityBar,
    StarterType,
)
from app.services.grader_runner_script_template import GRADER_RUNNER_SCRIPT_SOURCE
from app.services.oracle_authoring import (
    GeneratedReferenceFile,
    GeneratedScenarioFile,
    GeneratedSetupFile,
    OracleAuthoringResult,
)
from app.services.outcome_artifact_materializer import (
    materialize_course_spec,
    materialize_grader_runner,
    materialize_oracle_bundle,
    materialize_readme,
    materialize_starter,
    materialize_visible_samples,
)
from app.services.visible_checks_script_template import (
    VISIBLE_CHECKS_SCRIPT_SOURCE,
)


# ---------------- Fixtures ----------------


@pytest.fixture
def minimal_spec() -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title="Build a Grounded RAG Service",
        goal=(
            "Build a small HTTP service that ingests documents, retrieves "
            "passages for a question, and returns a grounded answer."
        ),
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str"},
                description="Answer the question.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite supporting passages.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
        ],
        learning_path=[
            LearningHint(
                on_metric_fail="faithfulness",
                hint="Add an explicit citation field.",
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
    )


@pytest.fixture
def sample_oracle_result() -> OracleAuthoringResult:
    return OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(
                filename="happy_path_basic.yaml",
                yaml_content="id: hp1\ndescription: ok\ncategory: happy_path\n",
            ),
            GeneratedScenarioFile(
                filename="boundary_empty.yaml",
                yaml_content="id: b1\ndescription: boundary\ncategory: boundary\n",
            ),
        ],
        reference_files=[
            GeneratedReferenceFile(
                relative_path="Dockerfile",
                content="FROM python:3.12-slim\n",
            ),
            GeneratedReferenceFile(
                relative_path="requirements.txt",
                content="fastapi\n",
            ),
            GeneratedReferenceFile(
                relative_path="app/main.py",
                content="print('hi')\n",
            ),
        ],
        setup_files=[
            GeneratedSetupFile(
                relative_path="gold.json",
                content='{"q1": "a1"}\n',
            ),
        ],
        notes=["bundle ready"],
        cost_usd=0.12,
        model_id="claude-sonnet-4-6",
    )


# ---------------- materialize_starter ----------------


def test_materialize_starter_writes_files_under_public_starter(tmp_path: Path) -> None:
    starter_files = [
        ("Dockerfile", "FROM python:3.12-slim\n"),
        ("app/main.py", "x = 1\n"),
        ("requirements.txt", "fastapi\n"),
    ]
    materialize_starter(tmp_path, starter_files)
    assert (tmp_path / "public/starter/Dockerfile").read_text() == "FROM python:3.12-slim\n"
    assert (tmp_path / "public/starter/app/main.py").read_text() == "x = 1\n"
    assert (tmp_path / "public/starter/requirements.txt").read_text() == "fastapi\n"


def test_materialize_starter_creates_intermediate_dirs(tmp_path: Path) -> None:
    materialize_starter(tmp_path, [("a/b/c/deep.txt", "deep")])
    assert (tmp_path / "public/starter/a/b/c/deep.txt").read_text() == "deep"


def test_materialize_starter_overwrites_existing_files(tmp_path: Path) -> None:
    materialize_starter(tmp_path, [("Dockerfile", "OLD\n")])
    materialize_starter(tmp_path, [("Dockerfile", "NEW\n")])
    assert (tmp_path / "public/starter/Dockerfile").read_text() == "NEW\n"


def test_materialize_starter_rejects_absolute_or_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        materialize_starter(tmp_path, [("/etc/passwd", "x")])
    with pytest.raises(ValueError):
        materialize_starter(tmp_path, [("../escape.txt", "x")])


# ---------------- materialize_oracle_bundle ----------------


def test_materialize_oracle_bundle_lays_out_subdirs(
    tmp_path: Path, sample_oracle_result: OracleAuthoringResult
) -> None:
    materialize_oracle_bundle(tmp_path, sample_oracle_result)

    scenarios = tmp_path / "private/grader/scenarios"
    assert (scenarios / "happy_path_basic.yaml").exists()
    assert (scenarios / "boundary_empty.yaml").exists()

    ref = tmp_path / "private/grader/_reference"
    assert (ref / "Dockerfile").read_text().startswith("FROM python")
    assert (ref / "requirements.txt").exists()
    assert (ref / "app/main.py").read_text() == "print('hi')\n"

    setup = tmp_path / "private/grader/_setup"
    assert (setup / "gold.json").exists()


def test_materialize_oracle_bundle_handles_empty_setup_files(
    tmp_path: Path,
) -> None:
    result = OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(
                filename="h.yaml",
                yaml_content="id: h\ncategory: happy_path\n",
            ),
        ],
        reference_files=[
            GeneratedReferenceFile(relative_path="Dockerfile", content="FROM x"),
        ],
        setup_files=[],
    )
    materialize_oracle_bundle(tmp_path, result)
    assert (tmp_path / "private/grader/scenarios/h.yaml").exists()
    assert (tmp_path / "private/grader/_reference/Dockerfile").exists()
    # _setup should NOT be created if empty (or if created, just empty)
    setup_dir = tmp_path / "private/grader/_setup"
    if setup_dir.exists():
        assert not list(setup_dir.iterdir())


# ---------------- materialize_grader_runner ----------------


def test_materialize_grader_runner_writes_source(tmp_path: Path) -> None:
    materialize_grader_runner(tmp_path)
    runner = tmp_path / "private/grader/runner.py"
    assert runner.exists()
    assert runner.read_text() == GRADER_RUNNER_SCRIPT_SOURCE


# ---------------- materialize_course_spec ----------------


def test_materialize_course_spec_roundtrips_via_json(
    tmp_path: Path, minimal_spec: CourseOutcomeSpec
) -> None:
    materialize_course_spec(tmp_path, minimal_spec)
    spec_path = tmp_path / "private/course_spec.json"
    assert spec_path.exists()
    data = json.loads(spec_path.read_text())
    assert data["title"] == minimal_spec.title
    assert data["goal"] == minimal_spec.goal
    # The serialized form must round-trip cleanly through pydantic.
    restored = CourseOutcomeSpec.model_validate(data)
    assert restored.title == minimal_spec.title
    assert len(restored.quality_bars) == 1
    assert restored.quality_bars[0].id == "faithfulness"


def test_materialize_course_spec_overwrites_existing(
    tmp_path: Path, minimal_spec: CourseOutcomeSpec
) -> None:
    spec_path = tmp_path / "private/course_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("{}")
    materialize_course_spec(tmp_path, minimal_spec)
    data = json.loads(spec_path.read_text())
    assert data["title"] == minimal_spec.title


# ---------------- Finding G: atomic subtree replacement ----------------


def test_materialize_starter_removes_stale_file_from_previous_attempt(
    tmp_path: Path,
) -> None:
    """A file written by attempt 1 that no longer appears in attempt 2's
    bundle must be removed from disk. Without this, repair retries
    accumulate ghost files that contaminate later runs.
    """
    # Attempt 1: write two files.
    materialize_starter(
        tmp_path,
        [
            ("Dockerfile", "FROM python:3.12-slim\n"),
            ("app/old_module.py", "# stale\n"),
        ],
    )
    assert (tmp_path / "public/starter/app/old_module.py").exists()

    # Attempt 2: rewrite without old_module.py.
    materialize_starter(
        tmp_path,
        [
            ("Dockerfile", "FROM python:3.13-slim\n"),
            ("app/new_module.py", "# fresh\n"),
        ],
    )

    assert not (tmp_path / "public/starter/app/old_module.py").exists()
    assert (tmp_path / "public/starter/app/new_module.py").exists()
    assert (
        tmp_path / "public/starter/Dockerfile"
    ).read_text() == "FROM python:3.13-slim\n"


def test_materialize_starter_first_call_on_empty_workspace_works(
    tmp_path: Path,
) -> None:
    """No subtree to delete on a fresh workspace — must not raise."""
    # public/starter does not exist yet
    materialize_starter(tmp_path, [("Dockerfile", "FROM x\n")])
    assert (tmp_path / "public/starter/Dockerfile").read_text() == "FROM x\n"


def test_materialize_starter_does_not_touch_public_readme(
    tmp_path: Path,
) -> None:
    """``public/README.md`` is owned by a different materializer (legacy
    starter pipeline). Wiping ``public/starter/`` must not affect it.
    """
    readme = tmp_path / "public" / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("# Course README\n")
    materialize_starter(tmp_path, [("Dockerfile", "FROM x\n")])
    assert readme.read_text() == "# Course README\n"


def test_materialize_oracle_bundle_removes_stale_scenario_from_previous_attempt(
    tmp_path: Path,
) -> None:
    """A scenario YAML written by attempt 1 that no longer appears in
    attempt 2's bundle must be removed from ``scenarios/``."""
    attempt_1 = OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(
                filename="happy.yaml",
                yaml_content="id: h\ncategory: happy_path\n",
            ),
            GeneratedScenarioFile(
                filename="stale.yaml",
                yaml_content="id: stale\ncategory: boundary\n",
            ),
        ],
        reference_files=[
            GeneratedReferenceFile(
                relative_path="Dockerfile", content="FROM old\n"
            ),
        ],
        setup_files=[
            GeneratedSetupFile(relative_path="gold.json", content='{"v":1}\n'),
        ],
    )
    materialize_oracle_bundle(tmp_path, attempt_1)
    assert (tmp_path / "private/grader/scenarios/stale.yaml").exists()

    attempt_2 = OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(
                filename="happy.yaml",
                yaml_content="id: h\ncategory: happy_path\n",
            ),
            # ``stale.yaml`` no longer present in this attempt
        ],
        reference_files=[
            GeneratedReferenceFile(
                relative_path="Dockerfile", content="FROM new\n"
            ),
        ],
        setup_files=[
            GeneratedSetupFile(relative_path="gold.json", content='{"v":2}\n'),
        ],
    )
    materialize_oracle_bundle(tmp_path, attempt_2)

    assert not (tmp_path / "private/grader/scenarios/stale.yaml").exists()
    assert (tmp_path / "private/grader/scenarios/happy.yaml").exists()
    assert (
        tmp_path / "private/grader/_reference/Dockerfile"
    ).read_text() == "FROM new\n"
    assert (
        tmp_path / "private/grader/_setup/gold.json"
    ).read_text() == '{"v":2}\n'


def test_materialize_oracle_bundle_preserves_oracle_outputs_subtree(
    tmp_path: Path,
    sample_oracle_result: OracleAuthoringResult,
) -> None:
    """``private/grader/_oracle/`` is owned by ``oracle_pass``, written
    AFTER materialization. The bundle materializer must NOT wipe it on
    retry — otherwise a re-author followed by no fresh oracle pass would
    leave the grader without ground truth.
    """
    materialize_oracle_bundle(tmp_path, sample_oracle_result)
    # Simulate oracle_pass having written its outputs after a prior
    # materialization.
    oracle_dir = tmp_path / "private" / "grader" / "_oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    outputs = oracle_dir / "outputs.json"
    outputs.write_text('{"scenario_outputs": []}')

    # Re-materialize (e.g. repair retry).
    materialize_oracle_bundle(tmp_path, sample_oracle_result)

    assert outputs.exists(), "_oracle/outputs.json must survive a re-materialize"
    assert outputs.read_text() == '{"scenario_outputs": []}'


def test_materialize_oracle_bundle_preserves_runner_py(
    tmp_path: Path,
    sample_oracle_result: OracleAuthoringResult,
) -> None:
    """``private/grader/runner.py`` is written by
    :func:`materialize_grader_runner`, not by us. The bundle materializer
    wipes ``scenarios/``, ``_reference/``, and ``_setup/`` but must leave
    ``runner.py`` alone.
    """
    materialize_grader_runner(tmp_path)
    runner = tmp_path / "private/grader/runner.py"
    assert runner.exists()

    materialize_oracle_bundle(tmp_path, sample_oracle_result)
    assert runner.exists()
    assert runner.read_text() == GRADER_RUNNER_SCRIPT_SOURCE


def test_materialize_oracle_bundle_first_call_on_empty_workspace_works(
    tmp_path: Path,
    sample_oracle_result: OracleAuthoringResult,
) -> None:
    """Subtrees don't exist yet on a fresh workspace — must not raise."""
    materialize_oracle_bundle(tmp_path, sample_oracle_result)
    assert (tmp_path / "private/grader/scenarios/happy_path_basic.yaml").exists()
    assert (tmp_path / "private/grader/_reference/Dockerfile").exists()
    assert (tmp_path / "private/grader/_setup/gold.json").exists()


# ---------------- materialize_visible_samples ----------------


def test_materialize_visible_samples_writes_sample_queries_json(
    tmp_path: Path,
) -> None:
    """Benchmark-backed courses ship learner-visible sample data alongside
    the hidden grader bundle. The materializer accepts a pre-serialized
    JSON string (the visible-sample payload) and lands it at
    ``public/examples/sample_queries.json``."""
    payload = '[{"query_id": "q1", "question": "hi?", "expected_answer": "yo"}]'
    materialize_visible_samples(tmp_path, sample_queries_json=payload)
    target = tmp_path / "public" / "examples" / "sample_queries.json"
    assert target.exists()
    assert target.read_text() == payload


def test_materialize_visible_samples_writes_run_visible_checks_script(
    tmp_path: Path,
) -> None:
    """The materializer also writes the visible self-test script verbatim
    so the learner has a runnable entry point that consumes the JSON."""
    payload = '[]'
    materialize_visible_samples(tmp_path, sample_queries_json=payload)
    script = tmp_path / "public" / "checks" / "run_visible_checks.py"
    assert script.exists()
    assert script.read_text() == VISIBLE_CHECKS_SCRIPT_SOURCE


def test_materialize_oracle_bundle_writes_visible_samples_when_present(
    tmp_path: Path,
) -> None:
    """When an :class:`OracleAuthoringResult` carries a non-None
    ``visible_sample_queries_json`` (benchmark-backed course), the bundle
    materializer must drop it into ``public/examples/`` AND ship the
    visible-checks runner script. Non-benchmark results (None) leave
    ``public/`` untouched by this materializer."""
    result = OracleAuthoringResult(
        scenarios=[
            GeneratedScenarioFile(
                filename="h.yaml",
                yaml_content="id: h\ncategory: happy_path\n",
            ),
        ],
        reference_files=[
            GeneratedReferenceFile(relative_path="Dockerfile", content="FROM x"),
        ],
        setup_files=[],
        visible_sample_queries_json='[{"query_id":"q1"}]',
    )
    materialize_oracle_bundle(tmp_path, result)
    assert (tmp_path / "public/examples/sample_queries.json").read_text() == (
        '[{"query_id":"q1"}]'
    )
    assert (tmp_path / "public/checks/run_visible_checks.py").exists()


def test_materialize_oracle_bundle_skips_visible_samples_when_none(
    tmp_path: Path,
    sample_oracle_result: OracleAuthoringResult,
) -> None:
    """The default (non-benchmark) result has
    ``visible_sample_queries_json = None``. The materializer must NOT
    write ``public/examples/`` or ``public/checks/`` in that case."""
    assert sample_oracle_result.visible_sample_queries_json is None
    materialize_oracle_bundle(tmp_path, sample_oracle_result)
    assert not (tmp_path / "public/examples/sample_queries.json").exists()
    assert not (tmp_path / "public/checks/run_visible_checks.py").exists()


# ---------------- materialize_readme ----------------


def test_materialize_readme_writes_through_templater(
    tmp_path: Path, minimal_spec: CourseOutcomeSpec
) -> None:
    """The materializer writes ``public/README.md`` by running the spec
    through :func:`render_outcome_readme`. This means README content is
    fully derivable from the spec (no opaque LLM-authored blob) and
    sections like the endpoint table appear automatically.
    """
    materialize_readme(tmp_path, minimal_spec)
    readme = tmp_path / "public" / "README.md"
    assert readme.exists()
    content = readme.read_text()
    # Title + goal from the spec
    assert minimal_spec.title in content
    # Endpoint surfaces from the templater
    assert "POST" in content
    assert "/answer" in content
    # Quality bar id surfaces from the templater
    assert "faithfulness" in content


def test_materialize_readme_calls_rag_scaffold_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    """When the sibling RAG scaffold module (Wave 6.7b) is importable
    and the spec carries a benchmark source, the materializer calls
    ``rag_readme_block(spec)`` and includes its output in the README.
    When the scaffold ships independently it must show up without
    a code change on the materializer side.
    """
    import sys
    import types

    # Build an in-process stand-in for ``app.services.rag_scaffold``.
    fake_module = types.ModuleType("app.services.rag_scaffold")

    def fake_rag_readme_block(spec: CourseOutcomeSpec) -> str:
        return "## RAG-specific guidance\n\nFancy chunking notes here."

    fake_module.rag_readme_block = fake_rag_readme_block  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.services.rag_scaffold", fake_module)

    spec = CourseOutcomeSpec(
        title="Build a Grounded RAG Service",
        goal=(
            "Build a small HTTP service that ingests documents, retrieves "
            "passages for a question, and returns a grounded answer."
        ),
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str"},
                description="Answer the question.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite supporting passages.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
        capabilities=CapabilityFlags(runtime_llm_required=True),
        benchmark=HFBenchmarkSource(
            corpus_dataset="BeIR/scifact",
            qrels_dataset="BeIR/scifact-qrels",
        ),
    )
    materialize_readme(tmp_path, spec)
    readme = (tmp_path / "public" / "README.md").read_text()
    assert "RAG-specific guidance" in readme
    assert "Fancy chunking notes here." in readme
