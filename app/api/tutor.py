from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import current_user, require_role
from app.domain.auth import Role, User
from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorEditorContext,
    TutorHistoryResponse,
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
    user: User = Depends(current_user),
) -> TutorChatResponse:
    return svc.chat(req, user_id=str(user.id))


@router.get(
    "/history",
    response_model=TutorHistoryResponse,
    dependencies=[Depends(require_role(Role.learner))],
)
def history(
    session_id: str,
    request: Request,
    user: User = Depends(current_user),
) -> TutorHistoryResponse:
    """Durable transcript for this learner + session. The widget
    hydrates from here on open; browser localStorage is only an offline
    cache. Scoped to the authenticated user so one learner can never
    read another's conversation."""
    store = _store(request)
    if store is None:
        return TutorHistoryResponse(messages=[])
    try:
        messages = store.list_tutor_chat_messages(str(user.id), session_id)
    except Exception:
        messages = []
    return TutorHistoryResponse(messages=messages)


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
    store = _store(request)

    def _fallback() -> TutorEditorContext:
        # NEVER key on the ephemeral code-server port: it changes on
        # every container restart, so an `editor-<port>` session_id
        # silently fragments/orphans the learner's tutor history. Fall
        # back to the learner's own enrollment (the SAME stable
        # `lms-<enrollment.id>` the LMS-page widget uses — shared
        # history); only if they have no enrollment, a stable per-user
        # key. Both survive editor restarts.
        try:
            if store is not None:
                enrs = store.list_learner_enrollments(
                    learner_id=str(user.id), limit=50
                )
                if enrs:  # store returns most-recently-updated first
                    e = enrs[0]
                    return TutorEditorContext(
                        assignment_title=getattr(e, "course_title", None)
                        or "Lab workspace",
                        session_id=f"lms-{e.id}",
                    )
        except Exception:
            pass
        return TutorEditorContext(
            assignment_title="Lab workspace",
            session_id=f"editor-user-{user.id}",
        )

    if store is None:
        return _fallback()
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
            return _fallback()
        enrollment = store.get_learner_enrollment(session.enrollment_id)
        if enrollment is None or enrollment.learner_id != str(user.id):
            return _fallback()
        return TutorEditorContext(
            assignment_title=enrollment.course_title or "Lab workspace",
            session_id=f"lms-{enrollment.id}",
        )
    except Exception:
        return _fallback()
