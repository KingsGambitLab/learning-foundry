from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.domain.auth import Role, User
from app.services.auth_session import COOKIE_NAME, SessionService


def _service(request: Request) -> SessionService:
    service = getattr(request.app.state, "session_service", None)
    if service is None:
        raise RuntimeError("session_service is not attached to app.state")
    return service


def current_user_optional(request: Request) -> User | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    loaded = _service(request).load(token)
    return loaded.user if loaded else None


def current_user(request: Request) -> User:
    user = current_user_optional(request)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def require_role(*roles: Role):
    allowed = {r if isinstance(r, Role) else Role(r) for r in roles}

    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Role not permitted")
        return user

    return dep
