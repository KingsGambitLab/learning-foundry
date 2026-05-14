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
def client(postgres_url: str) -> TestClient:
    os.environ["DATABASE_URL"] = postgres_url
    import importlib
    import app.main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app, raise_server_exceptions=True) as c:
        yield c


def _register(client: TestClient, email: str, role: str) -> None:
    resp = client.post("/auth/register", json={
        "email": email, "password": "hunter2!!", "role": role,
    })
    assert resp.status_code == 201, resp.text


def test_unauthenticated_creator_route_returns_401(client: TestClient) -> None:
    resp = client.get("/v1/course-runs")
    assert resp.status_code == 401


def test_unauthenticated_learner_route_returns_401(client: TestClient) -> None:
    resp = client.get("/v1/lms/catalog")
    assert resp.status_code == 401


def test_learner_cannot_hit_creator_route(client: TestClient) -> None:
    _register(client, "learner@x.com", "learner")
    resp = client.get("/v1/course-runs")
    assert resp.status_code == 403


def test_creator_cannot_hit_learner_route(client: TestClient) -> None:
    _register(client, "creator@x.com", "creator")
    resp = client.get("/v1/lms/enrollments")
    assert resp.status_code == 403


def test_authorized_learner_can_hit_learner_route(client: TestClient) -> None:
    _register(client, "learner2@x.com", "learner")
    resp = client.get("/v1/lms/enrollments")
    assert resp.status_code == 200


def test_authorized_creator_can_hit_creator_route(client: TestClient) -> None:
    _register(client, "creator2@x.com", "creator")
    resp = client.get("/v1/course-runs")
    assert resp.status_code == 200
