from __future__ import annotations

from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorSubmitRequest,
    TutorSubmitResponse,
    TutorVivaQuestion,
)

_CHAT_PREVIEW_LIMIT = 80


class TutorService:
    """Phase 1 stub. Returns canned responses; real RAG/judge wiring is Phase 2."""

    def chat(self, req: TutorChatRequest) -> TutorChatResponse:
        preview = req.message[:_CHAT_PREVIEW_LIMIT]
        return TutorChatResponse(reply=f"(stub) Got: {preview}", hint_tier=None)

    def submit(self, req: TutorSubmitRequest) -> TutorSubmitResponse:
        return TutorSubmitResponse(
            test_results={"passed": True, "details": "stub"},
            viva_questions=[
                TutorVivaQuestion(prompt="Explain why you chose this data structure."),
                TutorVivaQuestion(prompt="Walk through your error handling."),
            ],
        )
