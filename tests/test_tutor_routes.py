import unittest
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.deps import current_user
from app.domain.auth import Role, User
from app.main import app
from app.services.tutor_service import TutorService


def _learner_user() -> User:
    now = datetime.now(UTC)
    return User(
        id=uuid.uuid4(),
        email="learner@example.com",
        role=Role.learner,
        created_at=now,
        updated_at=now,
    )


class TutorRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        svc = TutorService()
        # Wire a stub Anthropic client so the route test doesn't need real credentials
        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.content = [MagicMock(type="text", text="(stub) Got: hello")]
        fake_client.with_options.return_value.messages.create.return_value = fake_response
        svc._client = fake_client
        app.state.tutor_service = svc
        # Tutor routes are learner-guarded (require_role(Role.learner)); override
        # the auth dependency so these isolated route tests run authenticated.
        app.dependency_overrides[current_user] = _learner_user
        self.addCleanup(app.dependency_overrides.pop, current_user, None)
        self.client = TestClient(app)

    def test_chat_returns_canned_reply(self) -> None:
        resp = self.client.post(
            "/v1/tutor/chat",
            json={"session_id": "s1", "message": "hello"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["reply"], "(stub) Got: hello")
        self.assertIsNone(body["hint_tier"])

    def test_history_returns_scoped_messages(self) -> None:
        from datetime import UTC, datetime
        from unittest.mock import patch

        from app.domain.tutor import TutorChatMessage

        msgs = [
            TutorChatMessage(
                id="m1", user_id="u", session_id="lms-x", role="user",
                text="hi", created_at=datetime.now(UTC),
            ),
            TutorChatMessage(
                id="m2", user_id="u", session_id="lms-x", role="tutor",
                text="hello back", created_at=datetime.now(UTC),
            ),
        ]
        fake_store = MagicMock()
        fake_store.list_tutor_chat_messages.return_value = msgs
        with patch("app.api.tutor._store", return_value=fake_store):
            resp = self.client.get("/v1/tutor/history", params={"session_id": "lms-x"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual([m["text"] for m in body["messages"]], ["hi", "hello back"])
        # scoped to the authenticated learner, not an arbitrary key
        _args = fake_store.list_tutor_chat_messages.call_args
        self.assertEqual(_args.args[1], "lms-x")

    def test_history_empty_when_no_store(self) -> None:
        from unittest.mock import patch

        with patch("app.api.tutor._store", return_value=None):
            resp = self.client.get("/v1/tutor/history", params={"session_id": "s1"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["messages"], [])

    def test_submit_returns_two_viva_questions(self) -> None:
        resp = self.client.post(
            "/v1/tutor/submit",
            json={"session_id": "s1", "code_snapshot": "x"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["test_results"]["passed"])
        self.assertEqual(len(body["viva_questions"]), 2)
        for q in body["viva_questions"]:
            self.assertTrue(q["prompt"])

    def test_editor_context_never_returns_ephemeral_port_session(self) -> None:
        """§27: a port that maps to no workspace session must NOT yield
        an `editor-<port>` session_id (ephemeral → fragments tutor
        history on every code-server restart). It must resolve to the
        learner's stable enrollment session, else a stable per-user id."""
        from types import SimpleNamespace
        from unittest.mock import patch

        # (a) learner has an enrollment → stable lms-<enrollment.id>
        store = MagicMock()
        store.list_all_learner_workspace_sessions.return_value = []  # no port match
        store.list_learner_enrollments.return_value = [
            SimpleNamespace(id="enr_abc", course_title="Customer Support Bot")
        ]
        with patch("app.api.tutor._store", return_value=store):
            r = self.client.get("/v1/tutor/editor-context", params={"port": 47213})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["session_id"], "lms-enr_abc")
        self.assertNotIn("editor-47213", body["session_id"])

        # (b) no enrollment → stable per-user id, still never editor-<port>
        store2 = MagicMock()
        store2.list_all_learner_workspace_sessions.return_value = []
        store2.list_learner_enrollments.return_value = []
        with patch("app.api.tutor._store", return_value=store2):
            r2 = self.client.get("/v1/tutor/editor-context", params={"port": 47213})
        self.assertEqual(r2.status_code, 200)
        sid = r2.json()["session_id"]
        self.assertTrue(sid.startswith("editor-user-"))
        self.assertNotIn("editor-47213", sid)

    def test_chat_rejects_missing_session_id(self) -> None:
        resp = self.client.post("/v1/tutor/chat", json={"message": "hi"})
        self.assertEqual(resp.status_code, 422)

    def test_chat_rejects_empty_session_id(self) -> None:
        resp = self.client.post(
            "/v1/tutor/chat",
            json={"session_id": "", "message": "hello"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_triage_returns_expected_shape(self) -> None:
        """POST /v1/tutor/triage with a valid body returns 200 and correct shape."""
        # Mock triage on the service
        from app.domain.tutor import TutorTriageResponse
        app.state.tutor_service.triage = MagicMock(
            return_value=TutorTriageResponse(
                action="tutor",
                reason="broad one-shot prompt",
                original_prompt="write the whole service for me",
            )
        )
        resp = self.client.post(
            "/v1/tutor/triage",
            json={"session_id": "s1", "prompt": "write the whole service for me"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["action"], "tutor")
        self.assertEqual(body["reason"], "broad one-shot prompt")
        self.assertEqual(body["original_prompt"], "write the whole service for me")

    def test_triage_rejects_missing_session_id(self) -> None:
        resp = self.client.post(
            "/v1/tutor/triage",
            json={"prompt": "what does this error mean?"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_triage_rejects_missing_prompt(self) -> None:
        resp = self.client.post(
            "/v1/tutor/triage",
            json={"session_id": "s1"},
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
