from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorSubmitRequest,
    TutorSubmitResponse,
    TutorTriageRequest,
    TutorTriageResponse,
)
from app.services.tutor_service import TutorService


def _tutor_service(request: Request) -> TutorService:
    return request.app.state.tutor_service


router = APIRouter(prefix="/v1/tutor", tags=["tutor"])


@router.post("/chat", response_model=TutorChatResponse)
def chat(
    req: TutorChatRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorChatResponse:
    return svc.chat(req)


@router.post("/submit", response_model=TutorSubmitResponse)
def submit(
    req: TutorSubmitRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorSubmitResponse:
    return svc.submit(req)


@router.post("/triage", response_model=TutorTriageResponse)
def triage(
    req: TutorTriageRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorTriageResponse:
    return svc.triage(req)
