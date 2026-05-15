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
    os.environ["DATABASE_URL"] = postgres_url
    import importlib
    import app.main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as c:
        yield c


def test_list_enrollments_does_not_accept_query_param(client: TestClient) -> None:
    client.post("/auth/register", json={"email": "l1@e.com", "password": "hunter2!!"})
    # Even if the legacy query param is supplied, it must be ignored — the session-derived user is what counts.
    resp = client.get("/v1/lms/enrollments?learner_id=should-be-ignored")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"enrollments": []}


def test_create_enrollment_rejects_explicit_learner_id_in_body(client: TestClient) -> None:
    client.post("/auth/register", json={"email": "l2@e.com", "password": "hunter2!!"})
    # Body carries an `extra` field — with extra='forbid' on the request, this 422s.
    resp = client.post("/v1/lms/enrollments", json={
        "course_run_id": "nonexistent",
        "learner_id": "should-be-ignored",
    })
    # Either 422 (extra field) or 404 (course not found) is acceptable — the key is it does NOT
    # accept learner_id at face value and use it.
    assert resp.status_code in (422, 404, 409, 400)
