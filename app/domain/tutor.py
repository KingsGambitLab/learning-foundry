from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TutorChatRequest:
    session_id: str
    message: str


@dataclass(frozen=True)
class TutorChatResponse:
    reply: str
    hint_tier: int | None = None


@dataclass(frozen=True)
class TutorSubmitRequest:
    session_id: str
    code_snapshot: str


@dataclass(frozen=True)
class TutorVivaQuestion:
    prompt: str


@dataclass(frozen=True)
class TutorSubmitResponse:
    test_results: dict[str, Any] = field(default_factory=dict)
    viva_questions: list[TutorVivaQuestion] = field(default_factory=list)
