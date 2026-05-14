from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class Role(str, Enum):
    creator = "creator"
    learner = "learner"


class User(BaseModel):
    id: UUID
    email: EmailStr
    role: Role
    display_name: str | None = None
    created_at: datetime
    updated_at: datetime


class UserSession(BaseModel):
    id: UUID
    user_id: UUID
    created_at: datetime
    expires_at: datetime
    last_seen_at: datetime
    ip: str | None = None
    user_agent: str | None = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    role: Role
    display_name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class AuthResponse(BaseModel):
    user_id: UUID
    role: Role
    display_name: str | None = None
