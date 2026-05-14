from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from app.domain.auth import User, UserSession
from app.storage.workflow_store import WorkflowStore


SESSION_TTL = timedelta(days=14)
COOKIE_NAME = "coursegen_session"


@dataclass
class LoadedSession:
    user: User
    session: UserSession


class SessionService:
    def __init__(self, store: WorkflowStore) -> None:
        self.store = store

    def create(self, *, user_id: UUID, ip: str | None, user_agent: str | None) -> UUID:
        expires_at = datetime.now(UTC) + SESSION_TTL
        return self.store.create_user_session(
            user_id=user_id, expires_at=expires_at, ip=ip, user_agent=user_agent
        )

    def load(self, session_id: str) -> LoadedSession | None:
        try:
            sid = UUID(session_id)
        except (ValueError, TypeError):
            return None
        session = self.store.load_user_session(sid)
        if session is None:
            return None
        user = self.store.get_user_by_id(session.user_id)
        if user is None:
            return None
        return LoadedSession(user=user, session=session)

    def revoke(self, session_id: str) -> None:
        try:
            sid = UUID(session_id)
        except (ValueError, TypeError):
            return
        self.store.revoke_user_session(sid)
