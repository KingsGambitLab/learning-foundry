"""Tests for ``app.services.rag_scaffold``.

The RAG scaffold module supplies the RAG-specific artifacts (README
block, starter helper file, default citation rubric) for courses whose
benchmark is BeIR-shaped (``HFBenchmarkSource``) or CRAG-shaped
(``CRAGBenchmarkSource``). Tests below cover each public function for
both benchmark families and the "no benchmark" defensive branch.

Tests 7-9 exec the generated Python file in a controlled namespace and
call the functions; that catches a syntax / runtime regression in the
emitted helper code, not just byte-equality.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    CRAGBenchmarkSource,
    EndpointContract,
    HFBenchmarkSource,
    HttpMethod,
    JudgeKind,
    LearningHint,
    QualityBar,
    StarterType,
)
from app.services.rag_scaffold import (
    apply_rag_scaffold,
    default_rag_citation_rubric,
    rag_readme_block,
)
from app.services.scenario_loader import RubricSpec


# ---------------- spec helper ----------------


def _spec(**overrides: Any) -> CourseOutcomeSpec:
    payload: dict[str, Any] = {
        "title": "Retrieval QA service",
        "goal": (
            "Build a small retrieval-augmented QA service that answers "
            "user questions grounded in a corpus of internal docs."
        ),
        "starter_type": StarterType.partial,
        "endpoints": [
            EndpointContract(
                method=HttpMethod.POST,
                path="/answers",
                request_schema={"query": "str"},
                response_schema={"answer": "str", "cited_chunks": "list[str]"},
                description="Answer the user's question with citations.",
            )
        ],
        "quality_bars": [
            QualityBar(
                id="groundedness",
                metric_description="Every cited chunk exists.",
                threshold=">= 0.9",
                judged_by=JudgeKind.oracle_set_overlap,
                sample_size=20,
            )
        ],
        "learning_path": [
            LearningHint(
                on_metric_fail="groundedness",
                hint="Only cite chunks you actually retrieved.",
            )
        ],
        "package_type": PackageType.progressive_codebase_course,
    }
    payload.update(overrides)
    return CourseOutcomeSpec(**payload)


def _crag_spec() -> CourseOutcomeSpec:
    return _spec(benchmark=CRAGBenchmarkSource())


def _beir_spec() -> CourseOutcomeSpec:
    return _spec(
        benchmark=HFBenchmarkSource(
            corpus_dataset="BeIR/scifact",
            qrels_dataset="BeIR/scifact-qrels",
        )
    )


def _no_benchmark_spec() -> CourseOutcomeSpec:
    return _spec()


# ---------------- rag_readme_block ----------------


def test_rag_readme_block_returns_crag_specific_block_for_crag_benchmark() -> None:
    block = rag_readme_block(_crag_spec())
    assert "RAG-specific notes (CRAG)" in block
    assert "search_results" in block
    assert "html_parsing.py" in block
    assert "cited_chunks" in block
    # Sanity: must NOT leak the BeIR-only language into the CRAG block.
    assert "data/corpus.json" not in block
    assert "recall_at_k" not in block


def test_rag_readme_block_returns_beir_specific_block_for_hf_benchmark() -> None:
    block = rag_readme_block(_beir_spec())
    assert "RAG-specific notes (BeIR)" in block
    assert "data/corpus.json" in block
    assert "cited_doc_ids" in block
    assert "recall_at_k" in block
    # Sanity: must NOT leak the CRAG-only language into the BeIR block.
    assert "html_parsing.py" not in block
    assert "search_results" not in block


def test_rag_readme_block_returns_empty_string_when_no_benchmark() -> None:
    assert rag_readme_block(_no_benchmark_spec()) == ""


# ---------------- apply_rag_scaffold ----------------


def test_apply_rag_scaffold_for_crag_writes_html_parsing_helper(
    tmp_path: Path,
) -> None:
    written = apply_rag_scaffold(tmp_path, _crag_spec())

    expected = tmp_path / "public" / "starter" / "app" / "utils" / "html_parsing.py"
    assert expected.exists()
    assert expected in written
    body = expected.read_text(encoding="utf-8")
    # Stdlib-only constraint: must NOT import BeautifulSoup / bs4 / lxml.
    assert "bs4" not in body
    assert "beautifulsoup" not in body.lower()
    assert "lxml" not in body
    # Module exposes the documented surface.
    assert "def extract_text" in body
    assert "def extract_snippets" in body


def test_apply_rag_scaffold_for_beir_writes_corpus_loader(tmp_path: Path) -> None:
    written = apply_rag_scaffold(tmp_path, _beir_spec())

    expected = tmp_path / "public" / "starter" / "app" / "utils" / "corpus_loader.py"
    assert expected.exists()
    assert expected in written
    body = expected.read_text(encoding="utf-8")
    assert "def load_corpus" in body


def test_apply_rag_scaffold_returns_empty_list_when_no_benchmark(
    tmp_path: Path,
) -> None:
    written = apply_rag_scaffold(tmp_path, _no_benchmark_spec())
    assert written == []
    # And critically: nothing was written under public/starter/app/utils.
    utils_dir = tmp_path / "public" / "starter" / "app" / "utils"
    assert not utils_dir.exists()


# ---------------- generated helper behavior ----------------


def _exec_module(path: Path) -> dict[str, Any]:
    """Exec a generated Python file into a fresh namespace and return it."""
    source = path.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "_generated_helper"}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def test_generated_extract_text_strips_html_and_returns_visible_text(
    tmp_path: Path,
) -> None:
    apply_rag_scaffold(tmp_path, _crag_spec())
    ns = _exec_module(
        tmp_path / "public" / "starter" / "app" / "utils" / "html_parsing.py"
    )
    extract_text = ns["extract_text"]
    html = "<html><body><p>Hello <b>world</b>.</p><p>Second line.</p></body></html>"
    out = extract_text(html)
    assert "Hello" in out
    assert "world" in out
    assert "Second line." in out
    # No raw tags should leak through.
    assert "<p>" not in out
    assert "<b>" not in out


def test_generated_extract_text_ignores_script_and_style(tmp_path: Path) -> None:
    apply_rag_scaffold(tmp_path, _crag_spec())
    ns = _exec_module(
        tmp_path / "public" / "starter" / "app" / "utils" / "html_parsing.py"
    )
    extract_text = ns["extract_text"]
    html = (
        "<html><head><style>.x { color: red; }</style>"
        "<script>var leak = 'should-not-appear';</script></head>"
        "<body><p>Visible body text.</p></body></html>"
    )
    out = extract_text(html)
    assert "Visible body text." in out
    assert "should-not-appear" not in out
    assert "color: red" not in out


def test_generated_extract_snippets_filters_short_chunks(tmp_path: Path) -> None:
    apply_rag_scaffold(tmp_path, _crag_spec())
    ns = _exec_module(
        tmp_path / "public" / "starter" / "app" / "utils" / "html_parsing.py"
    )
    extract_snippets = ns["extract_snippets"]
    html = (
        "<p>This is the first long enough sentence to keep.</p>"
        "<p>Hi.</p>"
        "<p>Another sufficiently long sentence to retain.</p>"
    )
    snippets = extract_snippets(html, max_snippets=5)
    # The two long sentences survive; the short "Hi." is filtered.
    joined = " | ".join(snippets)
    assert "first long enough sentence" in joined
    assert "sufficiently long sentence" in joined
    assert all(len(s.strip()) > 20 for s in snippets)


# ---------------- default_rag_citation_rubric ----------------


def test_default_rag_citation_rubric_returns_crag_shaped_config_for_crag() -> None:
    rubric = default_rag_citation_rubric(_crag_spec())
    assert isinstance(rubric, RubricSpec)
    assert rubric.kind == "subset_match"
    assert rubric.config["target"] == "ans.body.cited_chunks"
    assert (
        rubric.config["acceptable_source"]
        == "setup_data.search_results_index.${query_id}"
    )
    assert rubric.config["acceptable_key"] == "page_url"
    assert rubric.config["min_overlap"] == 1.0


def test_default_rag_citation_rubric_returns_beir_shaped_config_for_hf() -> None:
    rubric = default_rag_citation_rubric(_beir_spec())
    assert isinstance(rubric, RubricSpec)
    assert rubric.kind == "subset_match"
    assert rubric.config["target"] == "ans.body.cited_doc_ids"
    assert rubric.config["acceptable_source"] == "setup_data.corpus"
    assert rubric.config["acceptable_key"] == "_id"
    assert rubric.config["min_overlap"] == 1.0


def test_default_rag_citation_rubric_returns_none_when_no_benchmark() -> None:
    assert default_rag_citation_rubric(_no_benchmark_spec()) is None
