from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.auth import RegisterRequest, Role


def test_role_enum() -> None:
    assert Role.creator.value == "creator"
    assert Role.learner.value == "learner"


def test_register_request_rejects_role_in_body() -> None:
    """Public signup is locked to learner — role must not be acceptable in the body.

    This is the schema-level pin of the P0 #1 fix: callers cannot grant
    themselves a non-default role via the public registration endpoint.
    """
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="abcdefgh", role="creator")  # type: ignore[call-arg]


def test_register_request_accepts_required_fields() -> None:
    req = RegisterRequest(email="a@b.com", password="abcdefgh")
    assert req.email == "a@b.com"
    assert req.password == "abcdefgh"
    assert req.display_name is None


def test_register_request_accepts_display_name() -> None:
    req = RegisterRequest(email="a@b.com", password="abcdefgh", display_name="Alice")
    assert req.display_name == "Alice"


def test_register_request_rejects_short_password() -> None:
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.com", password="abc")
