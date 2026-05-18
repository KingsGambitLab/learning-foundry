"""RAG family scaffold: per-benchmark README block, starter helpers, and
default citation rubric.

The framework is course-family-agnostic; this module supplies the
RAG-specific bits a CRAG or BeIR course needs:

- :func:`rag_readme_block` — a markdown block to inject into the
  learner README, describing benchmark-specific conventions
  (per-query retrieval and HTML parsing for CRAG; global corpus and
  doc-ID citations for BeIR).
- :func:`apply_rag_scaffold` — writes RAG-specific helper files into
  the materialized starter bundle (``app/utils/html_parsing.py`` for
  CRAG, ``app/utils/corpus_loader.py`` for BeIR).
- :func:`default_rag_citation_rubric` — returns a default
  ``subset_match`` rubric that validates citations against the
  appropriate source (per-query ``search_results_index`` for CRAG,
  global ``corpus`` for BeIR).

The CRAG HTML parsing helper is stdlib-only (``html.parser`` +
``re``); no BeautifulSoup / lxml dependency is added. This keeps the
starter bundle's runtime requirements minimal and consistent across
courses.

This module is self-contained: it depends on the existing
``course_outcome_models`` and ``scenario_loader`` modules only. No
imports from Wave 6.7a's new files.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.services.course_outcome_models import (
    CRAGBenchmarkSource,
    HFBenchmarkSource,
)
from app.services.scenario_loader import RubricSpec

if TYPE_CHECKING:
    from app.services.course_outcome_models import CourseOutcomeSpec

__all__ = [
    "rag_readme_block",
    "apply_rag_scaffold",
    "default_rag_citation_rubric",
]


# ---------------- README blocks ----------------


_CRAG_README_BLOCK = """## RAG-specific notes (CRAG)

### Retrieval is provided per-query

Your /answers endpoint receives `search_results` in the request body \
— an array of web-page snippets with `page_url`, `page_snippet`, \
`page_result` (HTML). You do NOT need to implement a corpus index; \
use the provided search_results directly.

### HTML parsing

A helper `app/utils/html_parsing.py` is included in the starter:
- `extract_text(html: str) -> str` — strips HTML, returns plain text
- `extract_snippets(html: str, max_snippets: int = 5) -> list[str]` \
— paragraph-level chunks

### Citation tracking

Your response includes `cited_chunks: list[str]`. Each element should \
be the `page_url` of the search_result you used to ground a claim. \
The hidden grader checks every cited URL appears in the question's \
search_results.

### False-premise refusal

Some questions contain falsifiable premises (e.g., "Why did Tesla \
move HQ to Mars?"). Your service should set `abstained=true` and \
provide a brief rationale rather than fabricate an answer.
"""


_BEIR_README_BLOCK = """## RAG-specific notes (BeIR)

### Global corpus

The corpus lives under `data/corpus.json` (a list of documents with \
`_id`, `title`, `text`). Your service should ingest it at startup \
(in `app.startup` or equivalent) and build whatever index your \
retriever needs.

### Citation tracking

Your response includes `cited_doc_ids: list[str]`. Each element must \
be a real `_id` from the corpus. The hidden grader checks \
`cited_doc_ids ⊆ corpus_ids`.

### Recall metric

Quality bars include `recall_at_k`: how often the gold doc IDs \
appear in your top-k retrieval. The grader has gold qrels per query; \
you don't see them.
"""


def rag_readme_block(spec: "CourseOutcomeSpec") -> str:
    """Return a markdown block describing the RAG conventions for ``spec``.

    The content varies by benchmark family:

    - CRAG (``CRAGBenchmarkSource``): per-query retrieval, HTML
      parsing helper, ``cited_chunks`` convention, false-premise
      refusal.
    - BeIR (``HFBenchmarkSource``): global corpus, ``cited_doc_ids``
      convention, recall metric.
    - Neither: empty string (defensive — caller decides whether to
      include this section at all).
    """
    benchmark = getattr(spec, "benchmark", None)
    if isinstance(benchmark, CRAGBenchmarkSource):
        return _CRAG_README_BLOCK
    if isinstance(benchmark, HFBenchmarkSource):
        return _BEIR_README_BLOCK
    return ""


# ---------------- Generated starter helpers ----------------


_HTML_PARSING_HELPER = '''"""HTML parsing helpers for CRAG search results."""
from __future__ import annotations

import re
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self._skip = False

    def handle_data(self, data):
        if not self._skip and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self._parts)


def extract_text(html: str) -> str:
    """Return the visible text content of an HTML string."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return ""
    return re.sub(r"\\s+", " ", parser.text()).strip()


def extract_snippets(html: str, max_snippets: int = 5) -> list[str]:
    """Return up to max_snippets paragraph-level chunks from the HTML."""
    text = extract_text(html)
    # Split on sentence-like boundaries.
    parts = re.split(r"(?<=[.!?])\\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    return parts[:max_snippets]
'''


_CORPUS_LOADER_HELPER = '''"""Corpus loader for BeIR-shaped data."""
from __future__ import annotations

import json
from pathlib import Path


def load_corpus(path: str | Path = "data/corpus.json") -> list[dict]:
    """Load BeIR corpus as a list of {_id, title, text} dicts."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
'''


def apply_rag_scaffold(
    workspace_root: Path, spec: "CourseOutcomeSpec"
) -> list[Path]:
    """Write RAG-specific helper files into the starter bundle.

    For a CRAG benchmark: writes ``public/starter/app/utils/html_parsing.py``.
    For a BeIR benchmark: writes ``public/starter/app/utils/corpus_loader.py``.
    For no benchmark: no-op; returns ``[]``.

    Returns the list of paths written so the caller can track which
    files this scaffold touched.
    """
    benchmark = getattr(spec, "benchmark", None)
    if benchmark is None:
        return []

    starter_utils = workspace_root / "public" / "starter" / "app" / "utils"

    if isinstance(benchmark, CRAGBenchmarkSource):
        starter_utils.mkdir(parents=True, exist_ok=True)
        target = starter_utils / "html_parsing.py"
        target.write_text(_HTML_PARSING_HELPER, encoding="utf-8")
        return [target]

    if isinstance(benchmark, HFBenchmarkSource):
        starter_utils.mkdir(parents=True, exist_ok=True)
        target = starter_utils / "corpus_loader.py"
        target.write_text(_CORPUS_LOADER_HELPER, encoding="utf-8")
        return [target]

    return []


# ---------------- Default citation rubric ----------------


def default_rag_citation_rubric(
    spec: "CourseOutcomeSpec",
) -> RubricSpec | None:
    """Return the default ``subset_match`` rubric for the RAG family.

    CRAG: every ``cited_chunks`` URL must appear in the request's
    ``search_results`` for that ``query_id``.

    BeIR: every ``cited_doc_ids`` value must be a real corpus ``_id``.

    Returns ``None`` when the spec has no benchmark — there's nothing
    RAG-specific to enforce in that case.

    This is a HINT: the LLM author can include it verbatim, override it,
    or omit it; the materializer can also auto-inject it when no
    citation rubric is present.
    """
    benchmark = getattr(spec, "benchmark", None)
    if isinstance(benchmark, CRAGBenchmarkSource):
        return RubricSpec(
            kind="subset_match",
            config={
                "target": "ans.body.cited_chunks",
                "acceptable_source": (
                    "setup_data.search_results_index.${query_id}"
                ),
                "acceptable_key": "page_url",
                "min_overlap": 1.0,
            },
        )
    if isinstance(benchmark, HFBenchmarkSource):
        return RubricSpec(
            kind="subset_match",
            config={
                "target": "ans.body.cited_doc_ids",
                "acceptable_source": "setup_data.corpus",
                "acceptable_key": "_id",
                "min_overlap": 1.0,
            },
        )
    return None
