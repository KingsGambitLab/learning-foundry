from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TutorChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    assignment_title: str | None = None


class TutorChatResponse(BaseModel):
    reply: str
    hint_tier: int | None = None


class TutorSubmitRequest(BaseModel):
    session_id: str = Field(min_length=1)
    code_snapshot: str


class TutorVivaQuestion(BaseModel):
    prompt: str


class TutorSubmitResponse(BaseModel):
    test_results: dict[str, Any] = Field(default_factory=dict)
    viva_questions: list[TutorVivaQuestion] = Field(default_factory=list)


class TutorTriageRequest(BaseModel):
    session_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    assignment_title: str | None = None


class TutorTriageResponse(BaseModel):
    action: Literal["tutor", "agent"]
    reason: str
    original_prompt: str


class TutorEditorContext(BaseModel):
    """Resolved tutor context for an embedded code-server editor.

    nginx only knows the dynamic editor port; this maps that port to
    the owning learner's enrollment so the in-editor widget gets the
    same ``session_id`` / ``assignment_title`` as the LMS-page widget
    (shared history). Falls back to a generic context on any miss so
    the bubble still mounts.
    """

    assignment_title: str
    session_id: str
