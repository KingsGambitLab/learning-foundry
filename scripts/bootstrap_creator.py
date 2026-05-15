"""Bootstrap a creator account.

Public /auth/register is locked to learner role; this is the only supported
path to mint a creator. Run on the box that the API server runs against,
with DATABASE_URL pointing at the same Postgres.

Usage:
    DATABASE_URL=postgresql+psycopg://... python -m scripts.bootstrap_creator \\
        --email founder@example.com [--password '<plain>'] [--display-name "Founder"]

Behavior:
- If --password is omitted, a random 20-char URL-safe token is generated and
  printed once to stdout.
- If the email already exists with role=creator, exits 0 (idempotent).
- If the email already exists with role=learner, exits 2 with a clear message
  (we do not silently elevate existing learner accounts).
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
from sqlalchemy import create_engine, text

from app.domain.auth import Role
from app.services.auth_passwords import hash_password


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", default=None, help="If omitted, a random password is generated.")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: --database-url or DATABASE_URL env required.", file=sys.stderr)
        return 1
    engine = create_engine(url)
    password = args.password or secrets.token_urlsafe(15)

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id, role FROM users WHERE email = :email"),
            {"email": args.email},
        ).first()
        if existing is not None:
            if existing.role == Role.creator.value:
                print(f"Already a creator: {args.email} (id={existing.id})")
                return 0
            print(
                f"ERROR: user {args.email!r} already exists with role={existing.role!r}. "
                "Refusing to elevate. Create with a different email or remove the existing row.",
                file=sys.stderr,
            )
            return 2
        row = conn.execute(
            text(
                """
                INSERT INTO users (email, password_hash, role, display_name)
                VALUES (:email, :pw, 'creator', :display_name)
                RETURNING id
                """
            ),
            {
                "email": args.email,
                "pw": hash_password(password),
                "display_name": args.display_name,
            },
        ).first()
    print(f"Creator created: {args.email}  id={row.id}")
    if args.password is None:
        print(f"Generated password (shown once): {password}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
