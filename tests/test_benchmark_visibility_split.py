"""Tests for visibility split helpers + visible-sample serialization.

Benchmark-backed courses (CRAG + BeIR) need a small learner-visible sample
sliced off the full benchmark and written under ``public/examples/``. The
split is deterministic (seeded) and stratified so the visible sample
covers the framework's category taxonomy where present.

The full hidden bundle keeps everything OTHER than the visible sample —
no query appears in both visible and hidden sets.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.services.benchmark_loader import (
    BenchmarkBundle,
    BenchmarkDocument,
    BenchmarkQuery,
    CRAGBenchmarkBundle,
    CRAGQuery,
    serialize_benchmark_to_setup,
    serialize_crag_to_setup,
    split_beir_for_visibility,
    split_crag_for_visibility,
)
from app.services.course_outcome_models import (
    CRAGBenchmarkSource,
    HFBenchmarkSource,
)


# ---------------- fixtures ----------------


def _beir_source() -> HFBenchmarkSource:
    return HFBenchmarkSource(
        corpus_dataset="BeIR/test",
        queries_dataset="BeIR/test-queries",
        qrels_dataset="BeIR/test-qrels",
    )


def _make_crag_bundle(n: int = 10) -> CRAGBenchmarkBundle:
    """A CRAG bundle covering every category dimension we stratify on.

    Includes ``answer_type='valid'``, ``question_type='false_premise'``,
    ``question_type='simple'``, ``question_type='multi-hop'`` plus a few
    plain extras so ``sample_size`` can range over them.
    """
    queries: list[CRAGQuery] = []
    # one false_premise
    queries.append(
        CRAGQuery(
            query_id="fp_1",
            query="Why did MSFT acquire OpenAI?",
            answer="Microsoft did not acquire OpenAI.",
            alt_ans=["false premise"],
            search_results=[
                {"page_url": "u/fp", "page_snippet": "no acquisition", "page_result": "<>"}
            ],
            domain="finance",
            question_type="false_premise",
            answer_type="valid",
        )
    )
    # one multi-hop
    queries.append(
        CRAGQuery(
            query_id="mh_1",
            query="Who composed Boléro and what year?",
            answer="Ravel, 1928.",
            alt_ans=[],
            search_results=[
                {"page_url": "u/mh", "page_snippet": "Boléro Ravel 1928", "page_result": "<>"}
            ],
            domain="music",
            question_type="multi-hop",
            answer_type="valid",
        )
    )
    # one simple
    queries.append(
        CRAGQuery(
            query_id="s_1",
            query="Capital of France?",
            answer="Paris.",
            alt_ans=[],
            search_results=[
                {"page_url": "u/s1", "page_snippet": "Paris", "page_result": "<>"}
            ],
            domain="open",
            question_type="simple",
            answer_type="valid",
        )
    )
    # fill the rest with simple+valid rows
    for i in range(n - 3):
        queries.append(
            CRAGQuery(
                query_id=f"x_{i:02d}",
                query=f"Filler question {i}",
                answer=f"Filler answer {i}",
                alt_ans=[],
                search_results=[
                    {
                        "page_url": f"u/x{i}",
                        "page_snippet": f"snippet {i}",
                        "page_result": "<>",
                    }
                ],
                domain="open",
                question_type="simple",
                answer_type="valid",
            )
        )
    return CRAGBenchmarkBundle(queries=queries, source=CRAGBenchmarkSource())


def _make_beir_bundle(
    *, num_queries: int = 10, with_zero_only_queries: bool = True
) -> BenchmarkBundle:
    """Bundle with positives-bearing queries plus optionally some with no
    positive qrels (so the visible split has to filter them out)."""
    corpus = [
        BenchmarkDocument(doc_id=f"doc_{i:03d}", title=f"T{i}", text=f"text {i}")
        for i in range(20)
    ]
    queries = [
        BenchmarkQuery(query_id=f"q{i:02d}", text=f"question {i}")
        for i in range(num_queries)
    ]
    qrels: dict[str, dict[str, int]] = {}
    # First ``num_queries-1`` queries get at least one positive
    for i in range(num_queries - (1 if with_zero_only_queries else 0)):
        qid = f"q{i:02d}"
        qrels[qid] = {f"doc_{i:03d}": 1, f"doc_{(i + 1) % 20:03d}": 2}
    # The last query has no positive qrels (omitted from qrels entirely
    # since zero-score rows are excluded upstream)
    return BenchmarkBundle(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        source=_beir_source(),
    )


# ---------------- CRAG split helpers ----------------


def test_split_crag_returns_visible_plus_hidden_summing_to_total() -> None:
    bundle = _make_crag_bundle(n=10)
    visible, hidden = split_crag_for_visibility(bundle, sample_size=5)
    assert len(visible.queries) == 5
    assert len(hidden.queries) == 5
    # disjoint
    vids = {q.query_id for q in visible.queries}
    hids = {q.query_id for q in hidden.queries}
    assert vids.isdisjoint(hids)
    assert vids | hids == {q.query_id for q in bundle.queries}


def test_split_crag_visible_covers_taxonomy_when_present() -> None:
    """Visible samples MUST include false_premise + multi-hop + simple +
    answer_type=valid when those categories exist in the source bundle."""
    bundle = _make_crag_bundle(n=10)
    visible, _ = split_crag_for_visibility(bundle, sample_size=5)
    question_types = {q.question_type for q in visible.queries}
    answer_types = {q.answer_type for q in visible.queries}
    assert "false_premise" in question_types
    assert "multi-hop" in question_types
    assert "simple" in question_types
    assert "valid" in answer_types


def test_split_crag_is_deterministic_for_same_seed() -> None:
    bundle = _make_crag_bundle(n=10)
    v1, h1 = split_crag_for_visibility(bundle, sample_size=5, seed=42)
    v2, h2 = split_crag_for_visibility(bundle, sample_size=5, seed=42)
    assert [q.query_id for q in v1.queries] == [q.query_id for q in v2.queries]
    assert [q.query_id for q in h1.queries] == [q.query_id for q in h2.queries]


def test_split_crag_sample_size_exceeds_total_returns_all_visible() -> None:
    bundle = _make_crag_bundle(n=3)
    visible, hidden = split_crag_for_visibility(bundle, sample_size=10)
    assert len(visible.queries) == 3
    assert len(hidden.queries) == 0


def test_split_crag_falls_back_gracefully_when_category_missing() -> None:
    """If the bundle has no ``multi-hop`` rows the split must still return
    ``sample_size`` items — just drawn from whatever IS present."""
    # Build a bundle with NO multi-hop rows.
    bundle = CRAGBenchmarkBundle(
        queries=[
            CRAGQuery(
                query_id=f"s_{i}",
                query="q",
                answer="a",
                alt_ans=[],
                search_results=[],
                domain="open",
                question_type="simple",
                answer_type="valid",
            )
            for i in range(6)
        ],
        source=CRAGBenchmarkSource(),
    )
    visible, hidden = split_crag_for_visibility(bundle, sample_size=3)
    assert len(visible.queries) == 3
    assert len(hidden.queries) == 3


# ---------------- BeIR split helpers ----------------


def test_split_beir_picks_only_queries_with_positive_qrels() -> None:
    """Every visible query MUST have at least one positive-relevance qrel
    in the visible bundle's qrels — otherwise the sample is useless for
    self-test."""
    bundle = _make_beir_bundle(num_queries=10)
    visible, hidden = split_beir_for_visibility(bundle, sample_size=5)
    for q in visible.queries:
        assert q.query_id in visible.qrels
        assert len(visible.qrels[q.query_id]) > 0
    # disjoint query-id sets
    vids = {q.query_id for q in visible.queries}
    hids = {q.query_id for q in hidden.queries}
    assert vids.isdisjoint(hids)


def test_split_beir_includes_corpus_slice_for_acceptable_doc_ids() -> None:
    """The visible bundle must carry a corpus slice with at least one doc
    per visible query (the one referenced by its positive qrels)."""
    bundle = _make_beir_bundle(num_queries=10)
    visible, _ = split_beir_for_visibility(bundle, sample_size=5)
    visible_doc_ids = {d.doc_id for d in visible.corpus}
    # every acceptable_doc_id in the visible qrels appears in the corpus slice
    for qid, doc_scores in visible.qrels.items():
        for doc_id in doc_scores:
            assert doc_id in visible_doc_ids, (
                f"corpus slice missing doc {doc_id} for query {qid}"
            )
    # The slice is not the entire 20-doc corpus.
    assert len(visible.corpus) < len(bundle.corpus)


def test_split_beir_is_deterministic_for_same_seed() -> None:
    bundle = _make_beir_bundle(num_queries=10)
    v1, h1 = split_beir_for_visibility(bundle, sample_size=5, seed=7)
    v2, h2 = split_beir_for_visibility(bundle, sample_size=5, seed=7)
    assert [q.query_id for q in v1.queries] == [q.query_id for q in v2.queries]
    assert [q.query_id for q in h1.queries] == [q.query_id for q in h2.queries]


# ---------------- Serializer extension (CRAG) ----------------


def test_serialize_crag_with_visible_dir_writes_sample_queries(
    tmp_path: Path,
) -> None:
    hidden_dir = tmp_path / "_setup"
    visible_dir = tmp_path / "examples"
    bundle = _make_crag_bundle(n=10)
    serialize_crag_to_setup(
        bundle, hidden_dir, visible_dir=visible_dir, sample_size=5
    )
    sample_path = visible_dir / "sample_queries.json"
    assert sample_path.exists()
    data = json.loads(sample_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 5
    sample0 = data[0]
    # Learner-friendly schema. Every field required.
    for field in (
        "query_id",
        "question",
        "search_results",
        "expected_answer",
        "alt_acceptable_answers",
        "answer_type",
        "question_type",
        "domain",
    ):
        assert field in sample0, f"missing field '{field}'"


def test_serialize_crag_visible_carries_expected_answer_for_self_test(
    tmp_path: Path,
) -> None:
    """The visible sample must include the gold answer so a learner can
    compare their own service output against it locally."""
    hidden_dir = tmp_path / "_setup"
    visible_dir = tmp_path / "examples"
    bundle = _make_crag_bundle(n=10)
    serialize_crag_to_setup(
        bundle, hidden_dir, visible_dir=visible_dir, sample_size=5
    )
    data = json.loads((visible_dir / "sample_queries.json").read_text())
    for sample in data:
        assert sample["expected_answer"], (
            f"visible sample {sample['query_id']} missing expected_answer"
        )


def test_serialize_crag_hidden_and_visible_are_disjoint(tmp_path: Path) -> None:
    hidden_dir = tmp_path / "_setup"
    visible_dir = tmp_path / "examples"
    bundle = _make_crag_bundle(n=10)
    serialize_crag_to_setup(
        bundle, hidden_dir, visible_dir=visible_dir, sample_size=5
    )
    # Hidden queries.jsonl
    hidden_queries = [
        json.loads(line)
        for line in (hidden_dir / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    hidden_ids = {q["query_id"] for q in hidden_queries}
    # Visible sample_queries.json
    visible_data = json.loads((visible_dir / "sample_queries.json").read_text())
    visible_ids = {s["query_id"] for s in visible_data}
    assert hidden_ids.isdisjoint(visible_ids), (
        "no query may appear in both hidden and visible bundles"
    )


def test_serialize_crag_without_visible_dir_preserves_legacy_behavior(
    tmp_path: Path,
) -> None:
    """When ``visible_dir`` is omitted (None) the old behavior holds: all
    queries land in the hidden dir; no visible side-output."""
    hidden_dir = tmp_path / "_setup"
    bundle = _make_crag_bundle(n=5)
    serialize_crag_to_setup(bundle, hidden_dir)
    hidden_queries = [
        json.loads(line)
        for line in (hidden_dir / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(hidden_queries) == 5


# ---------------- Serializer extension (BeIR) ----------------


def test_serialize_benchmark_with_visible_dir_writes_sample_queries(
    tmp_path: Path,
) -> None:
    hidden_dir = tmp_path / "_setup"
    visible_dir = tmp_path / "examples"
    bundle = _make_beir_bundle(num_queries=10)
    serialize_benchmark_to_setup(
        bundle, hidden_dir, visible_dir=visible_dir, sample_size=5
    )
    sample_path = visible_dir / "sample_queries.json"
    assert sample_path.exists()
    data = json.loads(sample_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 5
    sample0 = data[0]
    # BeIR-shape visible schema.
    for field in (
        "query_id",
        "question",
        "acceptable_doc_ids",
        "min_relevance_score",
        "corpus_sample",
    ):
        assert field in sample0, f"missing field '{field}'"
    # Each acceptable_doc_id should appear in the corpus_sample list.
    sample_doc_ids = {d["doc_id"] for d in sample0["corpus_sample"]}
    for accepted in sample0["acceptable_doc_ids"]:
        assert accepted in sample_doc_ids


def test_serialize_benchmark_hidden_and_visible_are_disjoint(tmp_path: Path) -> None:
    hidden_dir = tmp_path / "_setup"
    visible_dir = tmp_path / "examples"
    bundle = _make_beir_bundle(num_queries=10)
    serialize_benchmark_to_setup(
        bundle, hidden_dir, visible_dir=visible_dir, sample_size=5
    )
    hidden_queries = [
        json.loads(line)
        for line in (hidden_dir / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    hidden_ids = {q["query_id"] for q in hidden_queries}
    visible_data = json.loads((visible_dir / "sample_queries.json").read_text())
    visible_ids = {s["query_id"] for s in visible_data}
    assert hidden_ids.isdisjoint(visible_ids)
