"""Customer Support Bot — STARTER. Replace the TODOs.

Contract:  POST /support/answer
  request : {message, conversation_id?, history?:[{role,content}],
             kb_articles:[{article_id,title,text}]}
  response: {reply, action, citations:[article_id], redactions:int,
             abstained:bool, escalation_reason?}

You are graded on the skills in project_brief.md. The green bar is
reachable with the free core libs (no LLM needed for the decisions);
the LLM (via the platform proxy at env LAB_LLM_BASE_URL, token
LAB_LLM_TOKEN) only polishes `reply` and is a non-gating bonus.

This starter returns a fixed baseline so every behavioral scenario
fails until you implement the skills.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Customer Support Bot (starter)")


class Article(BaseModel):
    article_id: str
    title: str = ""
    text: str = ""


class Turn(BaseModel):
    role: str
    content: str


class AnswerRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    history: list[Turn] = Field(default_factory=list)
    kb_articles: list[Article] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    reply: str
    action: str
    citations: list[str]
    redactions: int
    abstained: bool
    escalation_reason: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Optional: the platform LLM proxy (small fast model, budget-capped at
# ~60k tokens/submission — no API key needed). S8 ONLY. S1–S7 need no LLM.
LLM_BASE = os.environ.get("LAB_LLM_BASE_URL")
LLM_TOKEN = os.environ.get("LAB_LLM_TOKEN")


def call_llm(system: str, user: str, max_tokens: int = 320) -> str | None:
    """S8 helper — already written, just call it. Returns the model's
    text, or None if the LLM is unset/slow/failing (then fall back to a
    plain templated reply; never block a decision on this)."""
    if not LLM_BASE or not LLM_TOKEN:
        return None
    import json
    import urllib.request

    body = json.dumps({
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        LLM_BASE.rstrip("/") + "/llm/complete",
        data=body, method="POST",
        headers={"content-type": "application/json", "x-lab-llm-token": LLM_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return (json.loads(r.read().decode()).get("content") or "").strip() or None
    except Exception:
        return None  # graceful degradation


@app.post("/support/answer", response_model=AnswerResponse)
def support_answer(req: AnswerRequest) -> AnswerResponse:
    # TODO S4  PII redaction: detect+redact email/phone/card/SSN, count them.
    # TODO S5  Strip prompt-injection / override text before deciding policy.
    # TODO S6  Multi-turn: resolve anaphora against `history`.
    # TODO S1  Retrieve relevant kb_articles (rank_bm25 / sentence-transformers+faiss).
    # TODO S2  Policy-as-code: answer | clarify | escalate | refuse (+ thresholds).
    # TODO S3  Refuse / abstain when nothing in the KB supports the question.
    # TODO S8  (bonus) optionally call_llm(system, user) to phrase a
    #          grounded reply; fall back to a template if it returns None.
    return AnswerResponse(
        reply="not yet implemented",
        action="answer",
        citations=[],
        redactions=0,
        abstained=False,
    )
