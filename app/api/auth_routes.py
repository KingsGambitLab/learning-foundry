from __future__ import annotations

import ipaddress
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.api.deps import current_user
from app.domain.auth import AuthResponse, LoginRequest, RegisterRequest, User
from app.services.auth_passwords import hash_password, verify_password
from app.services.auth_session import COOKIE_NAME, SESSION_TTL, SessionService
from app.storage.workflow_store import WorkflowStore


router = APIRouter(prefix="/auth", tags=["auth"])


def _store(request: Request) -> WorkflowStore:
    return request.app.state.workflow_service.store


def _session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def _safe_ip(host: str | None) -> str | None:
    """Return host only if it is a valid IP address; discard non-IP strings (e.g. 'testclient')."""
    if host is None:
        return None
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return None


def _set_session_cookie(response: Response, session_id) -> None:
    secure = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key=COOKIE_NAME,
        value=str(session_id),
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        secure=secure,
        samesite="lax",
    )


@router.post("/register", status_code=201, response_model=AuthResponse)
def register(payload: RegisterRequest, request: Request, response: Response) -> AuthResponse:
    store = _store(request)
    try:
        user = store.create_user(
            email=payload.email,
            password_hash=hash_password(payload.password),
            role=payload.role,
            display_name=payload.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session_id = _session_service(request).create(
        user_id=user.id,
        ip=_safe_ip(request.client.host if request.client else None),
        user_agent=request.headers.get("user-agent"),
    )
    _set_session_cookie(response, session_id)
    return AuthResponse(user_id=user.id, role=user.role, display_name=user.display_name)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> AuthResponse:
    store = _store(request)
    stored_hash = store.get_user_password_hash(payload.email)
    if stored_hash is None or not verify_password(payload.password, stored_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    user = store.get_user_by_email(payload.email)
    assert user is not None
    session_id = _session_service(request).create(
        user_id=user.id,
        ip=_safe_ip(request.client.host if request.client else None),
        user_agent=request.headers.get("user-agent"),
    )
    _set_session_cookie(response, session_id)
    return AuthResponse(user_id=user.id, role=user.role, display_name=user.display_name)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        _session_service(request).revoke(token)
    response.delete_cookie(COOKIE_NAME)


@router.get("/me", response_model=User)
def me(user: User = Depends(current_user)) -> User:
    return user
