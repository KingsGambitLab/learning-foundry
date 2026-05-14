from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TutorChatRequest(BaseModel):
    session_id: str
    message: str


class TutorChatResponse(BaseModel):
    reply: str
    hint_tier: int | None = None


class TutorSubmitRequest(BaseModel):
    session_id: str
    code_snapshot: str


class TutorVivaQuestion(BaseModel):
    prompt: str


class TutorSubmitResponse(BaseModel):
    test_results: dict[str, Any] = Field(default_factory=dict)
    viva_questions: list[TutorVivaQuestion] = Field(default_factory=list)
