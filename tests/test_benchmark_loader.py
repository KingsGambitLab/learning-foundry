"""Tests for the Hugging Face benchmark loader.

The loader downloads a published benchmark (BeIR-family layouts in v1)
and turns it into a :class:`BenchmarkBundle` that the oracle authoring
node serializes directly into ``private/grader/_setup/``. All tests
patch ``datasets.load_dataset`` — no network access.

The ``gold_qa.json`` schema written by ``serialize_benchmark_to_setup``
must match the shape ``OracleSetOverlap`` reads via
``setup_data.gold_qa.<query_id>.expected_doc_ids``. Specifically:

    {
      "q1": {"expected_doc_ids": ["doc_001", "doc_005"]},
      "q2": {"expected_doc_ids": ["doc_010"]},
      ...
    }

Only qrels rows with score > 0 contribute to a query's expected doc ids;
score=0 rows (explicitly judged irrelevant) are excluded.
"""
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.benchmark_loader import (
    BenchmarkBundle,
    BenchmarkDocument,
    BenchmarkLoadError,
    BenchmarkQuery,
    CRAGBenchmarkBundle,
    CRAGQuery,
    load_benchmark,
    load_crag_benchmark,
    serialize_benchmark_to_setup,
    serialize_crag_to_setup,
)
from app.services.course_outcome_models import (
    CRAGBenchmarkSource,
    HFBenchmarkSource,
)


# ---------------- fake datasets fixtures ----------------


class _FakeIterable:
    """Tiny stand-in for a ``datasets.Dataset`` row iterator.

    The real ``datasets.Dataset`` object is iterable and yields dict rows;
    that's all the loader uses, so we model just that surface here.
    """

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _beir_corpus_rows() -> list[dict]:
    return [
        {"_id": "doc_001", "title": "T1", "text": "First doc."},
        {"_id": "doc_002", "title": "T2", "text": "Second doc."},
        {"_id": "doc_003", "title": "T3", "text": "Third doc."},
        {"_id": "doc_004", "title": "T4", "text": "Fourth doc."},
    ]


def _beir_queries_rows() -> list[dict]:
    return [
        {"_id": "q1", "text": "Tell me about T1."},
        {"_id": "q2", "text": "Tell me about T2."},
        {"_id": "q3", "text": "Tell me about T3."},
    ]


def _beir_qrels_rows() -> list[dict]:
    # Note: q1 has two positive judgments, q2 has one positive plus one
    # explicit zero (which must be excluded), q3 has only a zero (so it
    # contributes nothing to gold_qa.json — but the loader keeps q3 in
    # the queries list regardless).
    return [
        {"query-id": "q1", "corpus-id": "doc_001", "score": 1},
        {"query-id": "q1", "corpus-id": "doc_002", "score": 2},
        {"query-id": "q2", "corpus-id": "doc_003", "score": 1},
        {"query-id": "q2", "corpus-id": "doc_004", "score": 0},
        {"query-id": "q3", "corpus-id": "doc_001", "score": 0},
    ]


def _beir_source(**overrides) -> HFBenchmarkSource:
    payload = {
        "corpus_dataset": "BeIR/scifact",
        "queries_dataset": "BeIR/scifact-queries",
        "qrels_dataset": "BeIR/scifact-qrels",
    }
    payload.update(overrides)
    return HFBenchmarkSource(**payload)


def _fake_load_dataset(corpus_rows, queries_rows, qrels_rows):
    """Build a ``load_dataset`` stand-in routed by dataset name.

    The loader is expected to call ``load_dataset(name, split=...)``;
    we route by the prefix of the dataset name to the right row set.
    """

    def _impl(name: str, split: str = "test"):
        if name.endswith("-qrels"):
            return _FakeIterable(qrels_rows)
        if name.endswith("-queries"):
            return _FakeIterable(queries_rows)
        # The corpus dataset (the base BeIR name).
        return _FakeIterable(corpus_rows)

    return _impl


# ---------------- 1. import error ----------------


def test_load_benchmark_raises_when_datasets_not_installed() -> None:
    """If ``datasets`` is not importable, the loader must fail fast with a
    clear install-this-library message — not a cryptic ImportError that
    surfaces deep inside the call.
    """
    src = _beir_source()
    real_import = builtins.__import__

    def _import_blocker(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("No module named 'datasets'")
        return real_import(name, globals, locals, fromlist, level)

    # Drop any cached "datasets" module so the loader's lazy import
    # actually re-runs the patched import.
    saved = {k: v for k, v in sys.modules.items() if k.startswith("datasets")}
    for k in list(saved):
        del sys.modules[k]
    try:
        with patch("builtins.__import__", side_effect=_import_blocker):
            with pytest.raises(BenchmarkLoadError) as excinfo:
                load_benchmark(src)
    finally:
        sys.modules.update(saved)

    msg = str(excinfo.value).lower()
    assert "datasets" in msg
    assert "install" in msg


# ---------------- 2. missing dataset ----------------


def test_load_benchmark_raises_when_hf_dataset_not_found() -> None:
    src = _beir_source()

    def _boom(*args, **kwargs):
        raise FileNotFoundError("Dataset 'BeIR/does-not-exist' not found")

    with patch("app.services.benchmark_loader.load_dataset", _boom, create=True):
        with pytest.raises(BenchmarkLoadError) as excinfo:
            load_benchmark(src)

    msg = str(excinfo.value)
    assert "BeIR" in msg or "not found" in msg.lower()


# ---------------- 3. field mapping mismatch ----------------


def test_load_benchmark_raises_on_field_mapping_mismatch() -> None:
    """If the configured ``corpus_text_field`` (or any other) is missing
    from the dataset rows, the loader should fail with a clear message
    naming the offending field — not silently produce empty text.
    """
    src = _beir_source(corpus_text_field="content")  # rows actually have ``text``
    fake = _fake_load_dataset(
        _beir_corpus_rows(), _beir_queries_rows(), _beir_qrels_rows()
    )

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        with pytest.raises(BenchmarkLoadError) as excinfo:
            load_benchmark(src)

    assert "content" in str(excinfo.value)


# ---------------- 4. happy path ----------------


def test_load_benchmark_happy_path_beir_shape() -> None:
    src = _beir_source()
    fake = _fake_load_dataset(
        _beir_corpus_rows(), _beir_queries_rows(), _beir_qrels_rows()
    )

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_benchmark(src)

    assert isinstance(bundle, BenchmarkBundle)
    assert bundle.source is src
    assert len(bundle.corpus) == 4
    assert all(isinstance(d, BenchmarkDocument) for d in bundle.corpus)
    assert bundle.corpus[0].doc_id == "doc_001"
    assert bundle.corpus[0].title == "T1"
    assert bundle.corpus[0].text == "First doc."

    assert len(bundle.queries) == 3
    assert all(isinstance(q, BenchmarkQuery) for q in bundle.queries)
    assert bundle.queries[0].query_id == "q1"
    assert bundle.queries[0].text == "Tell me about T1."

    # qrels: only positive scores count, and they're keyed by query-id.
    assert "q1" in bundle.qrels
    assert bundle.qrels["q1"] == {"doc_001": 1, "doc_002": 2}
    assert bundle.qrels["q2"] == {"doc_003": 1}
    # q3 only had a zero, so it should not appear in qrels at all (no
    # positive judgments).
    assert "q3" not in bundle.qrels


# ---------------- 5. max_corpus_docs ----------------


def test_load_benchmark_truncates_corpus_to_max_corpus_docs() -> None:
    src = _beir_source(max_corpus_docs=2)
    fake = _fake_load_dataset(
        _beir_corpus_rows(), _beir_queries_rows(), _beir_qrels_rows()
    )

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_benchmark(src)

    assert len(bundle.corpus) == 2
    assert [d.doc_id for d in bundle.corpus] == ["doc_001", "doc_002"]


# ---------------- 6. max_queries ----------------


def test_load_benchmark_truncates_queries_to_max_queries() -> None:
    src = _beir_source(max_queries=2)
    fake = _fake_load_dataset(
        _beir_corpus_rows(), _beir_queries_rows(), _beir_qrels_rows()
    )

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_benchmark(src)

    assert len(bundle.queries) == 2
    assert [q.query_id for q in bundle.queries] == ["q1", "q2"]
    # qrels are filtered to only the retained query ids.
    assert set(bundle.qrels.keys()) <= {"q1", "q2"}


# ---------------- 7. zero-score qrels excluded ----------------


def test_load_benchmark_excludes_zero_score_qrels_from_gold() -> None:
    src = _beir_source()
    qrels = [
        {"query-id": "qx", "corpus-id": "d1", "score": 0},
        {"query-id": "qx", "corpus-id": "d2", "score": 0},
        {"query-id": "qy", "corpus-id": "d3", "score": 1},
    ]
    corpus = [
        {"_id": "d1", "title": "t", "text": "x"},
        {"_id": "d2", "title": "t", "text": "y"},
        {"_id": "d3", "title": "t", "text": "z"},
    ]
    queries = [
        {"_id": "qx", "text": "?"},
        {"_id": "qy", "text": "?"},
    ]
    fake = _fake_load_dataset(corpus, queries, qrels)

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_benchmark(src)

    # qx only had zero-score rows: it MUST NOT appear in qrels.
    assert "qx" not in bundle.qrels
    # qy had one positive: it MUST appear.
    assert bundle.qrels["qy"] == {"d3": 1}


# ---------------- 8. multiple relevant docs per query ----------------


def test_load_benchmark_collects_multiple_relevant_docs_per_query() -> None:
    src = _beir_source()
    qrels = [
        {"query-id": "q1", "corpus-id": "d1", "score": 1},
        {"query-id": "q1", "corpus-id": "d2", "score": 2},
        {"query-id": "q1", "corpus-id": "d3", "score": 1},
    ]
    corpus = [
        {"_id": "d1", "title": "t", "text": "x"},
        {"_id": "d2", "title": "t", "text": "y"},
        {"_id": "d3", "title": "t", "text": "z"},
    ]
    queries = [{"_id": "q1", "text": "?"}]
    fake = _fake_load_dataset(corpus, queries, qrels)

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_benchmark(src)

    assert set(bundle.qrels["q1"].keys()) == {"d1", "d2", "d3"}


# ---------------- 9. serialize corpus.jsonl ----------------


def test_serialize_benchmark_writes_corpus_jsonl(tmp_path: Path) -> None:
    bundle = BenchmarkBundle(
        corpus=[
            BenchmarkDocument(doc_id="d1", title="T1", text="Hello"),
            BenchmarkDocument(doc_id="d2", title="T2", text="World"),
        ],
        queries=[BenchmarkQuery(query_id="q1", text="Hi?")],
        qrels={"q1": {"d1": 1}},
        source=_beir_source(),
    )

    serialize_benchmark_to_setup(bundle, tmp_path)
    corpus_path = tmp_path / "corpus.jsonl"
    assert corpus_path.exists()
    lines = corpus_path.read_text().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["doc_id"] == "d1"
    assert row0["title"] == "T1"
    assert row0["text"] == "Hello"


# ---------------- 10. gold_qa.json matches OracleSetOverlap shape ----------------


def test_serialize_benchmark_writes_gold_qa_in_oracle_set_overlap_shape(
    tmp_path: Path,
) -> None:
    """``OracleSetOverlap`` resolves ``setup_data.gold_qa.<qid>.expected_doc_ids``;
    the file we write must match that shape exactly when the runner loads
    it as ``setup_data["gold_qa"]``.
    """
    bundle = BenchmarkBundle(
        corpus=[BenchmarkDocument(doc_id="d1", text="x"),
                BenchmarkDocument(doc_id="d2", text="y")],
        queries=[
            BenchmarkQuery(query_id="q1", text="?"),
            BenchmarkQuery(query_id="q2", text="?"),
        ],
        qrels={"q1": {"d1": 1, "d2": 2}, "q2": {"d2": 1}},
        source=_beir_source(),
    )

    serialize_benchmark_to_setup(bundle, tmp_path)
    gold_path = tmp_path / "gold_qa.json"
    assert gold_path.exists()
    data = json.loads(gold_path.read_text())
    assert isinstance(data, dict)
    assert "q1" in data and "q2" in data
    assert set(data["q1"]["expected_doc_ids"]) == {"d1", "d2"}
    assert data["q2"]["expected_doc_ids"] == ["d2"]


# ---------------- 11. serialize creates the target dir ----------------


def test_serialize_benchmark_creates_target_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "_setup"
    assert not target.exists()
    bundle = BenchmarkBundle(
        corpus=[BenchmarkDocument(doc_id="d1", text="x")],
        queries=[BenchmarkQuery(query_id="q1", text="?")],
        qrels={"q1": {"d1": 1}},
        source=_beir_source(),
    )

    serialize_benchmark_to_setup(bundle, target)

    assert target.is_dir()
    assert (target / "corpus.jsonl").exists()
    assert (target / "queries.jsonl").exists()
    assert (target / "gold_qa.json").exists()


# ---------------- 12. corpus_title_field=None skips title ----------------


def test_load_benchmark_omits_title_when_corpus_title_field_is_none() -> None:
    """Some datasets have no title column. When configured to skip
    title, the loader must not raise and the resulting documents must
    have ``title is None``.
    """
    src = _beir_source(corpus_title_field=None)
    corpus = [
        {"_id": "d1", "text": "first"},
        {"_id": "d2", "text": "second"},
    ]
    queries = [{"_id": "q1", "text": "?"}]
    qrels = [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
    fake = _fake_load_dataset(corpus, queries, qrels)

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_benchmark(src)

    assert all(d.title is None for d in bundle.corpus)


def test_serialize_benchmark_writes_queries_jsonl(tmp_path: Path) -> None:
    """One BenchmarkQuery per line; round-trip sanity."""
    bundle = BenchmarkBundle(
        corpus=[BenchmarkDocument(doc_id="d1", text="x")],
        queries=[
            BenchmarkQuery(query_id="q1", text="first?"),
            BenchmarkQuery(query_id="q2", text="second?"),
        ],
        qrels={"q1": {"d1": 1}},
        source=_beir_source(),
    )

    serialize_benchmark_to_setup(bundle, tmp_path)
    lines = (tmp_path / "queries.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["query_id"] == "q1"
    assert json.loads(lines[1])["text"] == "second?"


# ---------------- CRAG fixtures ----------------


def _crag_rows() -> list[dict]:
    """A small CRAG-shaped sample covering every filter dimension.

    Mirrors the Quivr/CRAG schema: per-row query + answer + per-query
    retrieval pool (``search_results``). Splits 0 (validation) and 1
    (test) coexist in the same dataset. Domains, question types, and
    answer types are varied so the filters can be exercised.
    """
    return [
        {
            "interaction_id": "fin_simple_valid_v",
            "query": "What was AAPL's revenue in Q1 2023?",
            "answer": "AAPL Q1 2023 revenue was $117.2B.",
            "alt_ans": ["$117.2 billion", "117.2B"],
            "search_results": [
                {
                    "page_url": "https://example.com/aapl-q1-2023",
                    "page_snippet": "Apple Q1 2023 revenue 117.2B...",
                    "page_result": "<html>full</html>",
                }
            ],
            "domain": "finance",
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "music_simple_valid_v",
            "query": "Who composed Boléro?",
            "answer": "Maurice Ravel.",
            "alt_ans": ["Ravel"],
            "search_results": [
                {
                    "page_url": "https://example.com/bolero",
                    "page_snippet": "Boléro by Maurice Ravel...",
                    "page_result": "<html>full</html>",
                }
            ],
            "domain": "music",
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "fin_false_premise_v",
            "query": "Why did MSFT acquire OpenAI in 2024?",
            "answer": "Microsoft did not acquire OpenAI in 2024.",
            "alt_ans": ["MSFT did not acquire OpenAI", "false premise"],
            "search_results": [
                {
                    "page_url": "https://example.com/msft",
                    "page_snippet": "Microsoft invested in OpenAI...",
                    "page_result": "<html>full</html>",
                }
            ],
            "domain": "finance",
            "question_type": "false_premise",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "movie_no_answer_v",
            "query": "What's tomorrow's movie release schedule?",
            "answer": "I don't know.",
            "alt_ans": [],
            "search_results": [],
            "domain": "movie",
            "question_type": "simple",
            "static_or_dynamic": "dynamic",
            "answer_type": "no_answer",
            "split": 0,
        },
        {
            "interaction_id": "sports_simple_valid_t",
            "query": "Which team won the 2022 World Cup?",
            "answer": "Argentina.",
            "alt_ans": ["Argentina national team"],
            "search_results": [
                {
                    "page_url": "https://example.com/wc2022",
                    "page_snippet": "Argentina beat France...",
                    "page_result": "<html>full</html>",
                }
            ],
            "domain": "sports",
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 1,
        },
    ]


def _fake_crag_load_dataset(rows: list[dict]):
    """Build a ``load_dataset`` stand-in for Quivr/CRAG.

    The real Quivr/CRAG has a single ``train`` HF split. The loader is
    expected to call ``load_dataset(name, split="train")`` (or whatever
    the source dictates) and then filter by the internal ``split``
    column. We return every row regardless of HF split so the loader's
    own split-column filter is the thing under test.
    """

    def _impl(name: str, split: str = "train"):
        return _FakeIterable(rows)

    return _impl


# ---------------- 13. load_crag_benchmark happy path ----------------


def test_load_crag_benchmark_happy_path() -> None:
    """Default source: validation split + ``answer_type_filter=['valid']``
    + no other filter. Should keep the three valid+validation rows and
    drop the no_answer row and the test-split row."""
    src = CRAGBenchmarkSource()
    fake = _fake_crag_load_dataset(_crag_rows())

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_crag_benchmark(src)

    assert isinstance(bundle, CRAGBenchmarkBundle)
    assert bundle.source is src
    # Three rows have split=0 AND answer_type=valid.
    ids = [q.query_id for q in bundle.queries]
    assert set(ids) == {
        "fin_simple_valid_v",
        "music_simple_valid_v",
        "fin_false_premise_v",
    }
    # Each query carries its embedded retrieval pool + metadata.
    fin = next(q for q in bundle.queries if q.query_id == "fin_simple_valid_v")
    assert isinstance(fin, CRAGQuery)
    assert fin.answer.startswith("AAPL Q1 2023")
    assert fin.alt_ans == ["$117.2 billion", "117.2B"]
    assert fin.domain == "finance"
    assert fin.question_type == "simple"
    assert fin.answer_type == "valid"
    assert len(fin.search_results) == 1
    assert fin.search_results[0]["page_snippet"].startswith("Apple Q1 2023")


# ---------------- 14. use_split=test ----------------


def test_load_crag_benchmark_use_split_test_filters_to_split_one() -> None:
    src = CRAGBenchmarkSource(use_split="test")
    fake = _fake_crag_load_dataset(_crag_rows())

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_crag_benchmark(src)

    ids = {q.query_id for q in bundle.queries}
    assert ids == {"sports_simple_valid_t"}


# ---------------- 15. domain_filter ----------------


def test_load_crag_benchmark_domain_filter_restricts_to_named_domains() -> None:
    src = CRAGBenchmarkSource(domain_filter=["finance"])
    fake = _fake_crag_load_dataset(_crag_rows())

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_crag_benchmark(src)

    domains = {q.domain for q in bundle.queries}
    assert domains == {"finance"}
    ids = {q.query_id for q in bundle.queries}
    # Both finance+valid+validation rows: simple + false_premise.
    assert ids == {"fin_simple_valid_v", "fin_false_premise_v"}


# ---------------- 16. question_type_filter ----------------


def test_load_crag_benchmark_question_type_filter_keeps_only_listed_types() -> None:
    src = CRAGBenchmarkSource(question_type_filter=["false_premise"])
    fake = _fake_crag_load_dataset(_crag_rows())

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_crag_benchmark(src)

    types = {q.question_type for q in bundle.queries}
    assert types == {"false_premise"}
    assert {q.query_id for q in bundle.queries} == {"fin_false_premise_v"}


# ---------------- 17. answer_type_filter default excludes no_answer ----------------


def test_load_crag_benchmark_answer_type_default_excludes_no_answer() -> None:
    """The default ``answer_type_filter=['valid']`` should drop rows where
    ``answer_type`` is ``no_answer`` or ``invalid``."""
    src = CRAGBenchmarkSource()
    fake = _fake_crag_load_dataset(_crag_rows())

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_crag_benchmark(src)

    answer_types = {q.answer_type for q in bundle.queries}
    assert "no_answer" not in answer_types
    assert answer_types == {"valid"}


def test_load_crag_benchmark_answer_type_filter_none_keeps_everything() -> None:
    """Passing ``answer_type_filter=None`` explicitly disables the default
    filter and retains every row regardless of ``answer_type``."""
    src = CRAGBenchmarkSource(answer_type_filter=None)
    fake = _fake_crag_load_dataset(_crag_rows())

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_crag_benchmark(src)

    answer_types = {q.answer_type for q in bundle.queries}
    assert "no_answer" in answer_types
    assert {q.query_id for q in bundle.queries} == {
        "fin_simple_valid_v",
        "music_simple_valid_v",
        "fin_false_premise_v",
        "movie_no_answer_v",
    }


# ---------------- 18. max_queries truncation after filters ----------------


def test_load_crag_benchmark_max_queries_truncates_after_filters() -> None:
    """``max_queries`` applies AFTER the split + filter passes so the cap
    reflects "this many filtered rows" rather than "this many raw rows"."""
    src = CRAGBenchmarkSource(max_queries=2)
    fake = _fake_crag_load_dataset(_crag_rows())

    with patch("app.services.benchmark_loader.load_dataset", fake, create=True):
        bundle = load_crag_benchmark(src)

    assert len(bundle.queries) == 2


# ---------------- 19. serialize_crag_to_setup writes three files ----------------


def test_serialize_crag_to_setup_writes_three_files(tmp_path: Path) -> None:
    bundle = CRAGBenchmarkBundle(
        queries=[
            CRAGQuery(
                query_id="q1",
                query="What is X?",
                answer="X is a thing.",
                alt_ans=["a thing"],
                search_results=[
                    {
                        "page_url": "https://example.com/x",
                        "page_snippet": "X is a thing...",
                        "page_result": "<html>X</html>",
                    }
                ],
                domain="finance",
                question_type="simple",
                answer_type="valid",
            ),
        ],
        source=CRAGBenchmarkSource(),
    )
    serialize_crag_to_setup(bundle, tmp_path)
    assert (tmp_path / "queries.jsonl").exists()
    assert (tmp_path / "gold_answers.json").exists()
    assert (tmp_path / "search_results_index.json").exists()
    # No CRAG-shape global corpus file — retrieval pool is per-query.
    assert not (tmp_path / "corpus.jsonl").exists()


# ---------------- 20. gold_answers.json schema ----------------


def test_serialize_crag_gold_answers_schema_carries_answer_alt_and_type(
    tmp_path: Path,
) -> None:
    """The grader uses ``gold_answers.json`` for semantic-equivalence
    judgments — so each entry must carry the primary answer, the alts,
    AND the answer_type (so the grader can dispatch to the false_premise
    rubric when applicable)."""
    bundle = CRAGBenchmarkBundle(
        queries=[
            CRAGQuery(
                query_id="qa",
                query="?",
                answer="Yes.",
                alt_ans=["yes"],
                search_results=[],
                domain="open",
                question_type="simple",
                answer_type="valid",
            ),
            CRAGQuery(
                query_id="qb",
                query="?",
                answer="That premise is false.",
                alt_ans=[],
                search_results=[],
                domain="open",
                question_type="false_premise",
                answer_type="valid",
            ),
        ],
        source=CRAGBenchmarkSource(),
    )
    serialize_crag_to_setup(bundle, tmp_path)
    data = json.loads((tmp_path / "gold_answers.json").read_text())
    assert set(data.keys()) == {"qa", "qb"}
    assert data["qa"] == {
        "answer": "Yes.",
        "alt_ans": ["yes"],
        "answer_type": "valid",
        "question_type": "simple",
    }
    assert data["qb"]["question_type"] == "false_premise"


# ---------------- 21. search_results_index.json keyed by query_id ----------------


def test_serialize_crag_search_results_index_keyed_by_query_id(
    tmp_path: Path,
) -> None:
    pool_q1 = [
        {
            "page_url": "https://example.com/a",
            "page_snippet": "A",
            "page_result": "<a>",
        },
        {
            "page_url": "https://example.com/b",
            "page_snippet": "B",
            "page_result": "<b>",
        },
    ]
    pool_q2 = [
        {
            "page_url": "https://example.com/c",
            "page_snippet": "C",
            "page_result": "<c>",
        },
    ]
    bundle = CRAGBenchmarkBundle(
        queries=[
            CRAGQuery(
                query_id="q1",
                query="?",
                answer="x",
                alt_ans=[],
                search_results=pool_q1,
                domain="open",
                question_type="simple",
                answer_type="valid",
            ),
            CRAGQuery(
                query_id="q2",
                query="?",
                answer="y",
                alt_ans=[],
                search_results=pool_q2,
                domain="open",
                question_type="simple",
                answer_type="valid",
            ),
        ],
        source=CRAGBenchmarkSource(),
    )
    serialize_crag_to_setup(bundle, tmp_path)
    index = json.loads(
        (tmp_path / "search_results_index.json").read_text()
    )
    assert set(index.keys()) == {"q1", "q2"}
    assert index["q1"] == pool_q1
    assert index["q2"] == pool_q2


# ---------------- 22. CRAG without false_premise still serializes ----------------


def test_serialize_crag_without_false_premise_still_clean(
    tmp_path: Path,
) -> None:
    """A bundle that contains only ``simple`` questions still serializes
    cleanly — no false_premise rows is a valid state, not an error."""
    bundle = CRAGBenchmarkBundle(
        queries=[
            CRAGQuery(
                query_id="qa",
                query="?",
                answer="answer",
                alt_ans=[],
                search_results=[],
                domain="open",
                question_type="simple",
                answer_type="valid",
            ),
        ],
        source=CRAGBenchmarkSource(),
    )
    # Must not raise.
    serialize_crag_to_setup(bundle, tmp_path)
    queries_path = tmp_path / "queries.jsonl"
    assert queries_path.exists()
    rows = [
        json.loads(line)
        for line in queries_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["query_id"] == "qa"
    assert rows[0]["question_type"] == "simple"
