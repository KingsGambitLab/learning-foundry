"""Finance QA API — V5: dense embeddings + FAISS retrieval.

Replaces V4's BM25 lexical scoring with semantic similarity via
``sentence-transformers/all-MiniLM-L6-v2`` (a 22M-parameter
distilled encoder, ~80MB). FAISS provides nearest-neighbor
search over the per-request passage pool.

Why dense beats BM25 here: BeIR/fiqa distractors share lots of
finance-topic tokens with the query. BM25's IDF helps but
paraphrased queries ("How do I finance a small business?" vs
"investing using other people's money") still trip lexical
retrieval. A semantic embedder maps both phrasings to nearby
vectors and routes around the lexical-overlap trap.

Boot cost: the model is downloaded on first container start
(~10s on a fast connection, cached afterwards). Inference is
fast for 10-passage pools (~50ms per request).
"""
from __future__ import annotations

import re
from typing import List, Optional

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


app = FastAPI(title="Finance QA API")

# Lazy-load the encoder so the test client doesn't pay the model load
# cost on import. The first request inside the container takes the hit.
_ENCODER: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ENCODER


class SearchResult(BaseModel):
    passage_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    title: Optional[str] = None
    source: Optional[str] = None


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    search_results: List[SearchResult] = Field(min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    citations: List[str]
    abstained: bool


@app.post("/finance/answer", response_model=AnswerResponse)
def finance_answer(payload: AnswerRequest) -> AnswerResponse:
    encoder = _get_encoder()

    # Embed query + all passages.
    passages = list(payload.search_results)
    query_vec = encoder.encode([payload.question], normalize_embeddings=True)
    passage_vecs = encoder.encode(
        [p.text for p in passages], normalize_embeddings=True
    )

    # FAISS inner-product index (cosine since vectors are L2-normalized).
    dim = int(passage_vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(passage_vecs.astype(np.float32))

    # Search: rank all passages by similarity.
    scores, indices = index.search(query_vec.astype(np.float32), k=len(passages))
    top_score = float(scores[0][0])
    top_idx = int(indices[0][0])

    # Abstention threshold: cosine similarity below 0.3 means the query
    # is semantically far from every passage. Empirically chosen for
    # the all-MiniLM-L6-v2 encoder on financial QA data.
    if top_score < 0.30:
        return AnswerResponse(answer="", citations=[], abstained=True)

    best = passages[top_idx]
    answer_text = best.text.strip()
    sents = re.split(r"(?<=[.!?])\s+", answer_text)
    answer = " ".join(sents[:2]).rstrip(".")

    # Citations: passages within 80% of the top score.
    citation_threshold = top_score * 0.80
    citations = [
        passages[int(indices[0][k])].passage_id
        for k in range(len(passages))
        if float(scores[0][k]) >= citation_threshold
    ]

    return AnswerResponse(answer=answer, citations=citations, abstained=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
