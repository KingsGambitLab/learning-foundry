from __future__ import annotations

from app.services.auth_passwords import hash_password, verify_password


def test_hash_password_is_not_plaintext() -> None:
    h = hash_password("hunter2!!")
    assert h != "hunter2!!"
    assert len(h) > 30


def test_verify_password_accepts_correct() -> None:
    h = hash_password("hunter2!!")
    assert verify_password("hunter2!!", h) is True


def test_verify_password_rejects_wrong() -> None:
    h = hash_password("hunter2!!")
    assert verify_password("wrong-password", h) is False


def test_verify_password_rejects_invalid_hash() -> None:
    """Defensive: a malformed hash should return False, not raise."""
    assert verify_password("anything", "not-a-real-hash") is False
