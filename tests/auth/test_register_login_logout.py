from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _migrate(postgres_url: str) -> None:
    alembic = Path(sys.executable).parent / "alembic"
    subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": postgres_url},
        check=True,
    )


@pytest.fixture()
def client(postgres_url: str):
    # DATABASE_URL must be set BEFORE app/main.py is imported because that's when the engine is built.
    os.environ["DATABASE_URL"] = postgres_url
    # Re-import the app to pick up the fresh DATABASE_URL each test if needed:
    import importlib
    import app.main as main_module
    importlib.reload(main_module)
    # Use TestClient as a context manager so the lifespan (which wires app.state) runs.
    with TestClient(main_module.app, raise_server_exceptions=True) as c:
        yield c


def test_register_creates_user_and_sets_cookie(client: TestClient) -> None:
    resp = client.post("/auth/register", json={
        "email": "alice@example.com", "password": "hunter2!!", "role": "learner",
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["role"] == "learner"
    assert "user_id" in body
    assert "coursegen_session" in resp.cookies


def test_login_with_correct_credentials_returns_cookie(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "bob@example.com", "password": "hunter2!!", "role": "creator",
    })
    client.cookies.clear()
    resp = client.post("/auth/login", json={"email": "bob@example.com", "password": "hunter2!!"})
    assert resp.status_code == 200
    assert "coursegen_session" in resp.cookies


def test_login_with_wrong_password_returns_401(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "carol@example.com", "password": "hunter2!!", "role": "learner",
    })
    client.cookies.clear()
    resp = client.post("/auth/login", json={"email": "carol@example.com", "password": "wrong-password"})
    assert resp.status_code == 401


def test_login_with_unknown_email_returns_401(client: TestClient) -> None:
    resp = client.post("/auth/login", json={"email": "nobody@example.com", "password": "anything"})
    assert resp.status_code == 401


def test_register_with_duplicate_email_returns_409(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "dup@example.com", "password": "hunter2!!", "role": "learner",
    })
    client.cookies.clear()
    resp = client.post("/auth/register", json={
        "email": "dup@example.com", "password": "hunter2!!", "role": "creator",
    })
    assert resp.status_code == 409


def test_logout_revokes_session(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "dave@example.com", "password": "hunter2!!", "role": "learner",
    })
    resp = client.post("/auth/logout")
    assert resp.status_code == 204
    # cookie cleared via Set-Cookie; subsequent /auth/me must 401
    me = client.get("/auth/me")
    assert me.status_code == 401


def test_me_returns_current_user(client: TestClient) -> None:
    client.post("/auth/register", json={
        "email": "eve@example.com", "password": "hunter2!!", "role": "creator", "display_name": "Eve",
    })
    me = client.get("/auth/me")
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "eve@example.com"
    assert body["display_name"] == "Eve"
    assert body["role"] == "creator"
