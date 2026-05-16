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


# Optional: the platform LLM proxy (Haiku, budget-capped). No key needed.
LLM_BASE = os.environ.get("LAB_LLM_BASE_URL")
LLM_TOKEN = os.environ.get("LAB_LLM_TOKEN")


@app.post("/support/answer", response_model=AnswerResponse)
def support_answer(req: AnswerRequest) -> AnswerResponse:
    # TODO S4  PII redaction: detect+redact email/phone/card/SSN, count them.
    # TODO S5  Strip prompt-injection / override text before deciding policy.
    # TODO S6  Multi-turn: resolve anaphora against `history`.
    # TODO S1  Retrieve relevant kb_articles (rank_bm25 / sentence-transformers+faiss).
    # TODO S2  Policy-as-code: answer | clarify | escalate | refuse (+ thresholds).
    # TODO S3  Refuse / abstain when nothing in the KB supports the question.
    # TODO S8  (bonus) Call the LLM proxy to phrase a grounded reply.
    return AnswerResponse(
        reply="not yet implemented",
        action="answer",
        citations=[],
        redactions=0,
        abstained=False,
    )
