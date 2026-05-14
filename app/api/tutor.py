from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.domain.tutor import (
    TutorChatRequest,
    TutorSubmitRequest,
)
from app.services.tutor_service import TutorService


class ChatRequestSchema(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponseSchema(BaseModel):
    reply: str
    hint_tier: int | None


class SubmitRequestSchema(BaseModel):
    session_id: str = Field(min_length=1)
    code_snapshot: str


class VivaQuestionSchema(BaseModel):
    prompt: str


class SubmitResponseSchema(BaseModel):
    test_results: dict
    viva_questions: list[VivaQuestionSchema]


_service = TutorService()


def get_tutor_service() -> TutorService:
    return _service


router = APIRouter(prefix="/v1/tutor", tags=["tutor"])


@router.post("/chat", response_model=ChatResponseSchema)
def chat(
    req: ChatRequestSchema,
    svc: TutorService = Depends(get_tutor_service),
) -> ChatResponseSchema:
    reply = svc.chat(
        TutorChatRequest(session_id=req.session_id, message=req.message)
    )
    return ChatResponseSchema(reply=reply.reply, hint_tier=reply.hint_tier)


@router.post("/submit", response_model=SubmitResponseSchema)
def submit(
    req: SubmitRequestSchema,
    svc: TutorService = Depends(get_tutor_service),
) -> SubmitResponseSchema:
    result = svc.submit(
        TutorSubmitRequest(
            session_id=req.session_id,
            code_snapshot=req.code_snapshot,
        )
    )
    return SubmitResponseSchema(
        test_results=result.test_results,
        viva_questions=[VivaQuestionSchema(prompt=q.prompt) for q in result.viva_questions],
    )
