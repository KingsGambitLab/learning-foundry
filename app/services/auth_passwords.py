from __future__ import annotations

import os

import bcrypt


def _rounds() -> int:
    return int(os.environ.get("AUTH_BCRYPT_COST", "12"))


def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt. Returns the hash as a UTF-8 string."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=_rounds())).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time verify. Returns False on malformed hash rather than raising."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False
