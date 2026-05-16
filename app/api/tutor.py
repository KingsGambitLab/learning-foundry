from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import current_user, require_role
from app.domain.auth import Role, User
from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorEditorContext,
    TutorSubmitRequest,
    TutorSubmitResponse,
    TutorTriageRequest,
    TutorTriageResponse,
)
from app.services.tutor_service import TutorService


def _tutor_service(request: Request) -> TutorService:
    return request.app.state.tutor_service


def _store(request: Request):
    svc = getattr(request.app.state, "workflow_service", None)
    return getattr(svc, "store", None)


router = APIRouter(prefix="/v1/tutor", tags=["tutor"])


@router.post(
    "/chat",
    response_model=TutorChatResponse,
    dependencies=[Depends(require_role(Role.learner))],
)
def chat(
    req: TutorChatRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorChatResponse:
    return svc.chat(req)


@router.post(
    "/submit",
    response_model=TutorSubmitResponse,
    dependencies=[Depends(require_role(Role.learner))],
)
def submit(
    req: TutorSubmitRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorSubmitResponse:
    return svc.submit(req)


@router.post(
    "/triage",
    response_model=TutorTriageResponse,
    dependencies=[Depends(require_role(Role.learner))],
)
def triage(
    req: TutorTriageRequest,
    svc: TutorService = Depends(_tutor_service),
) -> TutorTriageResponse:
    return svc.triage(req)


@router.get(
    "/editor-context",
    response_model=TutorEditorContext,
    dependencies=[Depends(require_role(Role.learner))],
)
def editor_context(
    port: int,
    request: Request,
    user: User = Depends(current_user),
) -> TutorEditorContext:
    """Map an embedded editor's dynamic port to the owning learner's
    enrollment so the in-editor tutor shares session/title with the
    LMS-page widget. Owner-checked; generic fallback on any miss so
    the widget always mounts and never leaks another learner's title.
    """
    fallback = TutorEditorContext(
        assignment_title="Lab workspace", session_id=f"editor-{port}"
    )
    store = _store(request)
    if store is None:
        return fallback
    try:
        sessions = [
            s
            for s in store.list_all_learner_workspace_sessions()
            if s.host_port == port
        ]
        # Prefer a running session; else most recently updated.
        sessions.sort(key=lambda s: (s.status.value == "running", s.updated_at), reverse=True)
        session = sessions[0] if sessions else None
        if session is None:
            return fallback
        enrollment = store.get_learner_enrollment(session.enrollment_id)
        if enrollment is None or enrollment.learner_id != str(user.id):
            return fallback
        return TutorEditorContext(
            assignment_title=enrollment.course_title or "Lab workspace",
            session_id=f"lms-{enrollment.id}",
        )
    except Exception:
        return fallback
