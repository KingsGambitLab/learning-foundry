"""Hugging Face benchmark loader for the single-outcome course pipeline.

When a :class:`CourseOutcomeSpec` declares a ``benchmark`` (an
:class:`HFBenchmarkSource`), the oracle authoring node uses this module
to materialize the corpus, queries, and relevance judgments from a
published Hugging Face dataset instead of asking the LLM to synthesize
them.

The two public entry points are:

- :func:`load_benchmark` — pull rows from the configured HF datasets,
  apply per-field name mapping, truncate to the sandbox-cost caps, and
  return a :class:`BenchmarkBundle`.
- :func:`serialize_benchmark_to_setup` — write the bundle's three files
  into a target directory (``private/grader/_setup/``):

    - ``corpus.jsonl`` — one :class:`BenchmarkDocument` per line.
    - ``queries.jsonl`` — one :class:`BenchmarkQuery` per line.
    - ``gold_qa.json`` — ``{query_id: {"expected_doc_ids": [...]}}``,
      the exact shape :class:`OracleSetOverlap` expects via
      ``setup_data.gold_qa.<query_id>.expected_doc_ids``. Only qrels
      rows with ``score > 0`` are surfaced; zero-score rows (which are
      explicit "not relevant" judgments in the BeIR convention) are
      excluded.

Dependency: this module uses the `datasets`_ library lazily — the import
happens inside :func:`load_benchmark` so test environments and other
callers can ``import app.services.benchmark_loader`` without the package
installed. If ``datasets`` is missing at call time the loader raises
:class:`BenchmarkLoadError` with an "install datasets" hint.

.. _datasets: https://huggingface.co/docs/datasets
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from app.services.course_outcome_models import (
    CRAGBenchmarkSource,
    HFBenchmarkSource,
)


__all__ = [
    "BenchmarkDocument",
    "BenchmarkQuery",
    "BenchmarkBundle",
    "BenchmarkLoadError",
    "CRAGQuery",
    "CRAGBenchmarkBundle",
    "load_benchmark",
    "load_crag_benchmark",
    "serialize_benchmark_to_setup",
    "serialize_crag_to_setup",
    "split_crag_for_visibility",
    "split_beir_for_visibility",
    "crag_bundle_to_visible_payload",
    "beir_bundle_to_visible_payload",
]


class BenchmarkDocument(BaseModel):
    """One corpus document extracted from the HF dataset."""

    doc_id: str
    text: str
    title: str | None = None


class BenchmarkQuery(BaseModel):
    """One query extracted from the HF dataset."""

    query_id: str
    text: str


class BenchmarkBundle(BaseModel):
    """Materialized form of an HF benchmark, ready for serialization
    into ``private/grader/_setup/`` files.

    ``qrels`` maps each query id to a dict of ``corpus_id -> relevance``
    for that query's *positive* judgments (score > 0). Zero-score
    judgments are dropped; they are explicit "not relevant" markers in
    the BeIR convention and ``OracleSetOverlap`` only consumes positives.
    """

    corpus: list[BenchmarkDocument]
    queries: list[BenchmarkQuery]
    qrels: dict[str, dict[str, int]]
    source: HFBenchmarkSource


class BenchmarkLoadError(RuntimeError):
    """Raised when the benchmark cannot be loaded.

    Wraps every failure mode the loader knows about:

    - the ``datasets`` library is not installed,
    - an HF dataset name doesn't resolve,
    - a configured field-name doesn't exist in the dataset rows.

    The message is always human-readable and names the offending
    dataset / field whenever possible.
    """


# ---------------- loader ----------------


def _import_load_dataset() -> Any:
    """Lazy import of ``datasets.load_dataset``.

    Kept inside a function so tests can patch
    ``app.services.benchmark_loader.load_dataset`` directly (the
    attribute is hydrated on first call) and so ``import
    app.services.benchmark_loader`` works without ``datasets``
    installed.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BenchmarkLoadError(
            "the 'datasets' library is required to load HF benchmarks; "
            "install it: pip install datasets"
        ) from exc
    return load_dataset


# Module-level name; tests patch this directly via
# ``patch("app.services.benchmark_loader.load_dataset", ..., create=True)``.
# We attach a sentinel ``None`` so the attribute exists for patching even
# before the first real import.
load_dataset: Any = None


def _resolve_field(row: dict, field: str, *, dataset_name: str) -> Any:
    """Extract ``field`` from ``row`` or raise :class:`BenchmarkLoadError`.

    The error names the dataset and the missing field so the operator can
    fix the field-name mapping without reading the loader source.
    """
    if field not in row:
        available = sorted(row.keys())
        raise BenchmarkLoadError(
            f"field '{field}' not found in dataset '{dataset_name}' rows; "
            f"available fields: {available}"
        )
    return row[field]


def _load_one_dataset(name: str, split: str) -> Iterable[dict]:
    """Resolve ``load_dataset`` (lazy) and pull rows.

    Wraps every exception below ``BenchmarkLoadError`` into the same
    error class so the oracle author has one failure surface to catch.
    """
    global load_dataset
    if load_dataset is None:
        load_dataset = _import_load_dataset()
    # HARDCODED CONFIG (2026-05-14 live-run finding):
    # Quivr/CRAG is published as 20+ HF configs (``crag_task_1_and_2``,
    # ``crag_task_1_and_2_subset_1``, ..., ``_subset_19``); calling
    # ``load_dataset("Quivr/CRAG", split="train")`` without a config
    # raises ``Config name is missing``. Until ``CRAGBenchmarkSource``
    # grows a ``config_name`` field, default to the base ``crag_task_1_and_2``
    # config so the loader works out-of-the-box for the live RAG smoke.
    hf_config: str | None = None
    if name == "Quivr/CRAG":
        hf_config = "crag_task_1_and_2"
    try:
        if hf_config is not None:
            # HF's ``load_dataset`` signature is ``(path, name=None, ...)``
            # where the second positional is the CONFIG name (confusingly
            # also called ``name``). Call by keyword to dodge positional
            # ambiguity with test fakes that only declare ``(name, split)``.
            try:
                return load_dataset(path=name, name=hf_config, split=split)
            except TypeError:
                # Fallback for test fakes whose first positional is
                # called ``name`` (legacy HF signature before 4.x).
                return load_dataset(name, hf_config, split=split)
        return load_dataset(name, split=split)
    except BenchmarkLoadError:
        raise
    except Exception as exc:  # network / not-found / etc.
        raise BenchmarkLoadError(
            f"failed to load HF dataset '{name}' (split={split!r}): {exc}"
        ) from exc


def load_benchmark(source: HFBenchmarkSource) -> BenchmarkBundle:
    """Download the configured HF datasets and return a
    :class:`BenchmarkBundle`.

    Truncation honours ``source.max_corpus_docs`` and
    ``source.max_queries`` so test runs don't pay for the full corpus.
    When ``max_queries`` truncates the query list, qrels are also
    filtered to only the retained query ids.
    """
    # ---- corpus ----
    corpus_rows = _load_one_dataset(source.corpus_dataset, source.split)
    corpus: list[BenchmarkDocument] = []
    for idx, row in enumerate(corpus_rows):
        if (
            source.max_corpus_docs is not None
            and idx >= source.max_corpus_docs
        ):
            break
        doc_id = _resolve_field(
            row, source.corpus_id_field, dataset_name=source.corpus_dataset
        )
        text = _resolve_field(
            row, source.corpus_text_field, dataset_name=source.corpus_dataset
        )
        title: str | None = None
        if source.corpus_title_field is not None:
            title = row.get(source.corpus_title_field)
        corpus.append(
            BenchmarkDocument(doc_id=str(doc_id), text=str(text), title=title)
        )

    # ---- queries ----
    # BeIR sometimes ships queries as a separate dataset (``-queries``
    # suffix) and sometimes under the corpus dataset's ``queries``
    # split. Configurable per-spec; if no ``queries_dataset`` is set,
    # fall back to the corpus dataset name (with the same split).
    queries_name = source.queries_dataset or source.corpus_dataset
    query_rows = _load_one_dataset(queries_name, source.split)
    queries: list[BenchmarkQuery] = []
    for idx, row in enumerate(query_rows):
        if source.max_queries is not None and idx >= source.max_queries:
            break
        query_id = _resolve_field(
            row, source.query_id_field, dataset_name=queries_name
        )
        text = _resolve_field(
            row, source.query_text_field, dataset_name=queries_name
        )
        queries.append(BenchmarkQuery(query_id=str(query_id), text=str(text)))

    retained_qids = {q.query_id for q in queries}

    # ---- qrels ----
    qrels_rows = _load_one_dataset(source.qrels_dataset, source.split)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qid = _resolve_field(
            row, source.qrels_query_field, dataset_name=source.qrels_dataset
        )
        cid = _resolve_field(
            row, source.qrels_corpus_field, dataset_name=source.qrels_dataset
        )
        score_raw = _resolve_field(
            row, source.qrels_score_field, dataset_name=source.qrels_dataset
        )
        try:
            score = int(score_raw)
        except (TypeError, ValueError) as exc:
            raise BenchmarkLoadError(
                f"qrels row score is not coercible to int in "
                f"'{source.qrels_dataset}': {score_raw!r}"
            ) from exc
        qid_s = str(qid)
        # When max_queries truncates the queries list, drop qrels
        # rows whose query id is no longer in the retained set so the
        # downstream gold_qa.json only references queries we actually
        # carry.
        if source.max_queries is not None and qid_s not in retained_qids:
            continue
        # Zero-score rows are explicit "not relevant" judgments in the
        # BeIR convention. Drop them — only positive relevance survives
        # into ``gold_qa.json``.
        if score <= 0:
            continue
        qrels.setdefault(qid_s, {})[str(cid)] = score

    return BenchmarkBundle(
        corpus=corpus, queries=queries, qrels=qrels, source=source
    )


# ---------------- serialization ----------------


def serialize_benchmark_to_setup(
    bundle: BenchmarkBundle,
    target_dir: Path,
    *,
    visible_dir: Path | None = None,
    sample_size: int = 5,
) -> None:
    """Write the bundle's three files into ``target_dir``.

    The directory is created (with parents) if it does not exist. Files
    are overwritten if present.

    Output schema:

    - ``corpus.jsonl`` — one JSON object per line, fields:
      ``doc_id`` (str), ``text`` (str), ``title`` (str | null).
    - ``queries.jsonl`` — one JSON object per line, fields:
      ``query_id`` (str), ``text`` (str).
    - ``gold_qa.json`` — top-level dict keyed by query id; each value is
      ``{"expected_doc_ids": [<corpus_id>, ...]}``. Only qrels rows
      with positive relevance contribute (zero scores were already
      filtered upstream in :func:`load_benchmark`). The doc-id list
      ordering is stable (sorted) so two serializations of the same
      bundle produce identical files.

    Why this schema. The :class:`OracleSetOverlap` rubric reads its
    gold set via a dotted path resolved against ``setup_data`` — and
    the oracle-pass runner loads files in
    ``private/grader/_setup/`` keyed by file stem. So
    ``setup_data["gold_qa"]["q1"]["expected_doc_ids"]`` is exactly
    what scenarios configure ``gold_set_path:
    "gold_qa.q1.expected_doc_ids"`` to read.

    When ``visible_dir`` is provided, the bundle is split via
    :func:`split_beir_for_visibility`: the ``sample_size`` visible
    samples land in ``visible_dir/sample_queries.json`` (a learner-
    friendly JSON array — see :func:`beir_bundle_to_visible_payload`)
    and the *remainder* (the "kept" set) is what gets written to the
    hidden ``target_dir``. No query appears in both directories.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if visible_dir is not None:
        visible_bundle, bundle = split_beir_for_visibility(
            bundle, sample_size=sample_size
        )
        visible_dir = Path(visible_dir)
        visible_dir.mkdir(parents=True, exist_ok=True)
        payload = beir_bundle_to_visible_payload(visible_bundle)
        (visible_dir / "sample_queries.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )

    # corpus.jsonl
    corpus_path = target_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for doc in bundle.corpus:
            fh.write(json.dumps(doc.model_dump(), sort_keys=True))
            fh.write("\n")

    # queries.jsonl
    queries_path = target_dir / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as fh:
        for query in bundle.queries:
            fh.write(json.dumps(query.model_dump(), sort_keys=True))
            fh.write("\n")

    # gold_qa.json — the OracleSetOverlap-compatible shape.
    gold: dict[str, dict[str, list[str]]] = {}
    for query_id, doc_scores in bundle.qrels.items():
        # Deterministic ordering: sort doc ids alphabetically.
        gold[query_id] = {"expected_doc_ids": sorted(doc_scores.keys())}
    gold_path = target_dir / "gold_qa.json"
    gold_path.write_text(json.dumps(gold, indent=2, sort_keys=True))


# ============================================================
# CRAG (Quivr/CRAG) — generative RAG benchmark
# ============================================================


class CRAGQuery(BaseModel):
    """One CRAG row materialized for the grader.

    Distinct from :class:`BenchmarkQuery` because CRAG embeds everything
    per-query: the gold answer text, the alt-answer list, the retrieval
    pool (``search_results``), and the metadata that drives rubric
    selection (``domain`` / ``question_type`` / ``answer_type``).
    """

    query_id: str
    query: str
    answer: str
    alt_ans: list[str] = Field(default_factory=list)
    # CRAG ships ``search_results`` as a list of dicts with fields like
    # ``page_url`` / ``page_snippet`` / ``page_result``. We keep the raw
    # shape (``dict[str, Any]``) so we don't lose fields the grader
    # might want — Quivr's row schema can evolve and we don't want a
    # rigid model breaking the loader on new columns.
    search_results: list[dict[str, Any]]
    domain: str
    question_type: str
    answer_type: str


class CRAGBenchmarkBundle(BaseModel):
    """Materialized form of a CRAG benchmark.

    Unlike :class:`BenchmarkBundle`, there is no global corpus — the
    retrieval pool is per-query (see :class:`CRAGQuery.search_results`).
    The serializer writes a ``search_results_index.json`` keyed by
    ``query_id`` so the learner's service can look up its per-query
    retrieval pool the same way the benchmark does.
    """

    queries: list[CRAGQuery]
    source: CRAGBenchmarkSource


def _row_passes_crag_filters(
    row: dict, source: CRAGBenchmarkSource
) -> bool:
    """Apply the CRAG row filters from a configured source.

    The split filter is mandatory (``use_split`` is non-optional on the
    source); the rest are no-ops when their lists are ``None``.
    """
    split_value = row.get(source.split_field)
    # Map use_split string to the int64 column value (0 = validation,
    # 1 = test). The mapping is fixed by the dataset.
    want_split = 0 if source.use_split == "validation" else 1
    try:
        if int(split_value) != want_split:
            return False
    except (TypeError, ValueError):
        return False

    if source.domain_filter is not None:
        if row.get(source.domain_field) not in source.domain_filter:
            return False
    if source.question_type_filter is not None:
        if row.get(source.question_type_field) not in source.question_type_filter:
            return False
    if source.answer_type_filter is not None:
        if row.get(source.answer_type_field) not in source.answer_type_filter:
            return False
    return True


def load_crag_benchmark(
    source: CRAGBenchmarkSource,
) -> CRAGBenchmarkBundle:
    """Download the configured CRAG dataset and return a
    :class:`CRAGBenchmarkBundle`.

    Quivr/CRAG ships everything under a single ``train`` HF split, with
    the dataset's own ``split`` column (0 = validation, 1 = test)
    selecting between sub-splits. We always pull the ``train`` HF split
    and let ``use_split`` filter the rows by their ``split`` column.

    Filters are applied in order: split → domain → question_type →
    answer_type. ``max_queries`` truncates AFTER all filters so the
    cap reflects "this many course-relevant rows" not "this many raw
    rows" (otherwise a domain_filter could easily yield zero
    surviving rows when max_queries is small).
    """
    rows = _load_one_dataset(source.dataset, split="train")
    queries: list[CRAGQuery] = []
    for row in rows:
        if not _row_passes_crag_filters(row, source):
            continue
        if source.max_queries is not None and len(queries) >= source.max_queries:
            break
        try:
            query_id = _resolve_field(
                row, source.query_id_field, dataset_name=source.dataset
            )
            query_text = _resolve_field(
                row, source.query_text_field, dataset_name=source.dataset
            )
            answer = _resolve_field(
                row, source.answer_field, dataset_name=source.dataset
            )
        except BenchmarkLoadError:
            raise
        alt_raw = row.get(source.alt_ans_field) or []
        search_raw = row.get(source.search_results_field) or []
        domain = row.get(source.domain_field, "") or ""
        question_type = row.get(source.question_type_field, "") or ""
        answer_type = row.get(source.answer_type_field, "") or ""
        queries.append(
            CRAGQuery(
                query_id=str(query_id),
                query=str(query_text),
                answer=str(answer),
                alt_ans=[str(x) for x in alt_raw],
                search_results=list(search_raw),
                domain=str(domain),
                question_type=str(question_type),
                answer_type=str(answer_type),
            )
        )

    return CRAGBenchmarkBundle(queries=queries, source=source)


def serialize_crag_to_setup(
    bundle: CRAGBenchmarkBundle,
    target_dir: Path,
    *,
    visible_dir: Path | None = None,
    sample_size: int = 5,
) -> None:
    """Write the three CRAG setup files into ``target_dir``.

    Output schema:

    - ``queries.jsonl`` — one :class:`CRAGQuery` per line (full record
      including the embedded retrieval pool, so a single file
      round-trips the bundle).
    - ``gold_answers.json`` — ``{query_id: {answer, alt_ans, answer_type,
      question_type}}``. The grader reads this for semantic-equivalence
      and false-premise rubrics; ``question_type`` is carried so the
      grader / scenario author can dispatch rubrics by question type.
    - ``search_results_index.json`` — ``{query_id: [search_result, ...]}``
      so the learner's service can look up its per-query retrieval pool
      the same way the benchmark does. There is no global ``corpus.jsonl``
      — CRAG's retrieval pool is per-query, not global.

    When ``visible_dir`` is provided, the bundle is split via
    :func:`split_crag_for_visibility`: ``sample_size`` learner-visible
    samples land in ``visible_dir/sample_queries.json`` (a flat JSON
    array, see :func:`crag_bundle_to_visible_payload`) and the
    *remainder* is what flows into the hidden ``target_dir``. No
    query appears in both directories.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if visible_dir is not None:
        visible_bundle, bundle = split_crag_for_visibility(
            bundle, sample_size=sample_size
        )
        visible_dir = Path(visible_dir)
        visible_dir.mkdir(parents=True, exist_ok=True)
        payload = crag_bundle_to_visible_payload(visible_bundle)
        (visible_dir / "sample_queries.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True)
        )

    # queries.jsonl — full per-row record for round-trip.
    queries_path = target_dir / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as fh:
        for query in bundle.queries:
            fh.write(json.dumps(query.model_dump(), sort_keys=True))
            fh.write("\n")

    # gold_answers.json — the grader's semantic-equivalence reference.
    gold: dict[str, dict[str, Any]] = {}
    for q in bundle.queries:
        gold[q.query_id] = {
            "answer": q.answer,
            "alt_ans": list(q.alt_ans),
            "answer_type": q.answer_type,
            "question_type": q.question_type,
        }
    (target_dir / "gold_answers.json").write_text(
        json.dumps(gold, indent=2, sort_keys=True)
    )

    # search_results_index.json — per-query retrieval pool.
    index: dict[str, list[dict[str, Any]]] = {
        q.query_id: list(q.search_results) for q in bundle.queries
    }
    (target_dir / "search_results_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True)
    )


# ============================================================
# Visibility split + visible-payload helpers
# ============================================================


# Categories the CRAG visibility split *prefers* to include when at least
# one query in the source bundle carries them. The stratification picks
# one query per category (in priority order) up to ``sample_size``, then
# fills the remaining slots from whatever's left so the final visible
# count equals ``min(sample_size, len(bundle.queries))``.
#
# Each tuple is ``(attribute_name, value)``:
_CRAG_STRATIFIED_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("answer_type", "valid"),
    ("question_type", "false_premise"),
    ("question_type", "simple"),
    ("question_type", "multi-hop"),
)


def split_crag_for_visibility(
    bundle: CRAGBenchmarkBundle,
    *,
    sample_size: int = 5,
    seed: int = 42,
) -> tuple[CRAGBenchmarkBundle, CRAGBenchmarkBundle]:
    """Return ``(visible_samples, hidden_graded)``.

    The visible bundle holds a stratified subsample of ``sample_size``
    queries chosen to cover the CRAG framework's category taxonomy:

      - at least one ``answer_type='valid'`` row,
      - at least one ``question_type='false_premise'``,
      - at least one ``question_type='simple'``,
      - at least one ``question_type='multi-hop'``,

    when each is present in the source. Categories missing from the
    source bundle are skipped silently (the slot is filled by the
    next-priority category or by the random fill pass). The split is
    deterministic given the seed.

    When ``sample_size`` exceeds ``len(bundle.queries)`` every query is
    visible and the hidden bundle is empty.
    """
    queries = list(bundle.queries)
    if sample_size <= 0:
        return (
            CRAGBenchmarkBundle(queries=[], source=bundle.source),
            CRAGBenchmarkBundle(queries=queries, source=bundle.source),
        )
    if sample_size >= len(queries):
        return (
            CRAGBenchmarkBundle(queries=queries, source=bundle.source),
            CRAGBenchmarkBundle(queries=[], source=bundle.source),
        )

    rng = random.Random(seed)
    by_id = {q.query_id: q for q in queries}
    remaining_ids = [q.query_id for q in queries]
    rng.shuffle(remaining_ids)

    picked: list[str] = []
    picked_set: set[str] = set()

    # Pass 1: stratified — pick one query per category (in priority
    # order) when at least one exists in the remaining pool.
    for attr, value in _CRAG_STRATIFIED_CATEGORIES:
        if len(picked) >= sample_size:
            break
        for qid in remaining_ids:
            if qid in picked_set:
                continue
            q = by_id[qid]
            if getattr(q, attr, None) == value:
                picked.append(qid)
                picked_set.add(qid)
                break

    # Pass 2: random fill — fill the remaining slots from the leftover
    # pool (already shuffled deterministically).
    for qid in remaining_ids:
        if len(picked) >= sample_size:
            break
        if qid in picked_set:
            continue
        picked.append(qid)
        picked_set.add(qid)

    visible_queries = [by_id[qid] for qid in picked]
    hidden_queries = [q for q in queries if q.query_id not in picked_set]
    return (
        CRAGBenchmarkBundle(queries=visible_queries, source=bundle.source),
        CRAGBenchmarkBundle(queries=hidden_queries, source=bundle.source),
    )


def split_beir_for_visibility(
    bundle: BenchmarkBundle,
    *,
    sample_size: int = 5,
    seed: int = 42,
) -> tuple[BenchmarkBundle, BenchmarkBundle]:
    """Return ``(visible_samples, hidden_graded)`` for a BeIR-shape bundle.

    Picks ``sample_size`` queries that each have at least one positive-
    relevance qrel — a query with no positives is useless for learner
    self-test because there is no acceptable doc id to compare against.
    The split also slices the qrels and the corpus so the visible
    bundle is self-contained: only qrels for visible queries land in
    ``visible.qrels``, and only the documents referenced by those
    qrels (the acceptable docs) land in ``visible.corpus``.

    Determinism follows from a fixed ``seed`` to ``random.Random``.
    """
    candidate_ids = [q.query_id for q in bundle.queries if q.query_id in bundle.qrels]
    if sample_size <= 0 or not candidate_ids:
        return (
            BenchmarkBundle(
                corpus=[],
                queries=[],
                qrels={},
                source=bundle.source,
            ),
            bundle.model_copy(deep=True),
        )

    rng = random.Random(seed)
    shuffled = list(candidate_ids)
    rng.shuffle(shuffled)
    picked_ids = shuffled[: min(sample_size, len(shuffled))]
    picked_set = set(picked_ids)

    visible_queries = [q for q in bundle.queries if q.query_id in picked_set]
    visible_qrels = {qid: dict(bundle.qrels[qid]) for qid in picked_ids}

    # Corpus slice: include every doc referenced by a visible qrel. This
    # keeps the slice tight and learner-friendly — no need to bundle
    # the entire corpus when only a handful of acceptable docs matter
    # for the visible self-test.
    visible_doc_ids: set[str] = set()
    for qid, doc_scores in visible_qrels.items():
        visible_doc_ids.update(doc_scores.keys())
    visible_corpus = [d for d in bundle.corpus if d.doc_id in visible_doc_ids]

    # Hidden bundle: keep every NON-visible query, plus the remaining
    # qrels and the full corpus (the hidden grader needs the entire
    # corpus, not the trimmed slice).
    hidden_queries = [q for q in bundle.queries if q.query_id not in picked_set]
    hidden_qrels = {
        qid: dict(scores) for qid, scores in bundle.qrels.items() if qid not in picked_set
    }
    hidden_corpus = list(bundle.corpus)

    return (
        BenchmarkBundle(
            corpus=visible_corpus,
            queries=visible_queries,
            qrels=visible_qrels,
            source=bundle.source,
        ),
        BenchmarkBundle(
            corpus=hidden_corpus,
            queries=hidden_queries,
            qrels=hidden_qrels,
            source=bundle.source,
        ),
    )


def crag_bundle_to_visible_payload(
    bundle: CRAGBenchmarkBundle,
) -> list[dict[str, Any]]:
    """Render a (visible) CRAG bundle as the JSON-array payload that
    lands at ``public/examples/sample_queries.json``.

    Schema per entry::

        {
          "query_id":              str,
          "question":              str,
          "search_results":        list[dict],  # per-query retrieval pool
          "expected_answer":       str,         # gold (visible for self-test)
          "alt_acceptable_answers": list[str],  # gold alts
          "answer_type":           str,         # "valid"/"invalid"/"no_answer"
          "question_type":         str,         # CRAG metadata
          "domain":                str,
        }
    """
    return [
        {
            "query_id": q.query_id,
            "question": q.query,
            "search_results": list(q.search_results),
            "expected_answer": q.answer,
            "alt_acceptable_answers": list(q.alt_ans),
            "answer_type": q.answer_type,
            "question_type": q.question_type,
            "domain": q.domain,
        }
        for q in bundle.queries
    ]


def beir_bundle_to_visible_payload(
    bundle: BenchmarkBundle,
) -> list[dict[str, Any]]:
    """Render a (visible) BeIR bundle as the JSON-array payload that
    lands at ``public/examples/sample_queries.json``.

    Schema per entry::

        {
          "query_id":            str,
          "question":            str,
          "acceptable_doc_ids":  list[str],          # gold from qrels
          "min_relevance_score": int,                # min positive score
          "corpus_sample":       list[{doc_id,
                                       title,
                                       text}],       # tight slice of corpus
        }

    ``corpus_sample`` is the subset of the bundle's corpus that's
    referenced by this query's acceptable_doc_ids — the learner sees
    exactly the documents they'd need to retrieve, no more. The full
    corpus stays hidden.
    """
    corpus_by_id = {d.doc_id: d for d in bundle.corpus}
    payload: list[dict[str, Any]] = []
    for q in bundle.queries:
        scores = bundle.qrels.get(q.query_id, {})
        accepted = sorted(scores.keys())
        min_score = min(scores.values()) if scores else 0
        slice_docs = []
        for doc_id in accepted:
            doc = corpus_by_id.get(doc_id)
            if doc is None:
                continue
            slice_docs.append(
                {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}
            )
        payload.append(
            {
                "query_id": q.query_id,
                "question": q.text,
                "acceptable_doc_ids": accepted,
                "min_relevance_score": min_score,
                "corpus_sample": slice_docs,
            }
        )
    return payload
