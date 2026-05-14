from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.auth import RegisterRequest, Role, User


def test_role_enum() -> None:
    assert Role.creator.value == "creator"
    assert Role.learner.value == "learner"


def test_register_request_validates_role() -> None:
    req = RegisterRequest(email="a@b.com", password="abcdefgh", role="learner")
    assert req.role is Role.learner


def test_register_request_rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="abcdefgh", role="admin")


def test_register_request_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="abc", role="learner")
