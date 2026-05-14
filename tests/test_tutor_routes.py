import json
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.services.tutor_service import TutorService


def _make_stub_client(reply_text: str = "(stub) Got: hello") -> MagicMock:
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text=reply_text)]
    fake_client.with_options.return_value.messages.create.return_value = fake_response
    return fake_client


class TutorRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        svc = TutorService()
        # Wire a stub Anthropic client so the route test doesn't need real credentials
        svc._client = _make_stub_client()
        app.state.tutor_service = svc
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

    def test_chat_rejects_missing_session_id(self) -> None:
        resp = self.client.post("/v1/tutor/chat", json={"message": "hi"})
        self.assertEqual(resp.status_code, 422)

    def test_chat_rejects_empty_session_id(self) -> None:
        resp = self.client.post(
            "/v1/tutor/chat",
            json={"session_id": "", "message": "hello"},
        )
        self.assertEqual(resp.status_code, 422)


class TutorRehearseRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        svc = TutorService()
        # Wire a stub client that returns a rehearsal verdict
        payload = json.dumps({"verdict": "rehearsal", "message": "You're asking the agent to write the code for you."})
        svc._client = _make_stub_client(reply_text=payload)
        app.state.tutor_service = svc
        self.client = TestClient(app)

    def test_rehearse_route_exists_and_returns_expected_shape(self) -> None:
        """Route returns 200 with verdict, message, and original_prompt fields."""
        resp = self.client.post(
            "/v1/tutor/rehearse",
            json={"session_id": "s1", "prompt": "implement the route handler"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn(body["verdict"], ("ok", "rehearsal"))
        self.assertIsInstance(body["message"], str)
        self.assertEqual(body["original_prompt"], "implement the route handler")

    def test_rehearse_route_accepts_optional_assignment_title(self) -> None:
        """Route accepts optional assignment_title field without error."""
        resp = self.client.post(
            "/v1/tutor/rehearse",
            json={
                "session_id": "s1",
                "prompt": "write the function for me",
                "assignment_title": "Routing Service",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("verdict", body)
        self.assertIn("message", body)
        self.assertIn("original_prompt", body)

    def test_rehearse_route_rejects_missing_session_id(self) -> None:
        resp = self.client.post(
            "/v1/tutor/rehearse",
            json={"prompt": "implement this"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_rehearse_route_rejects_missing_prompt(self) -> None:
        resp = self.client.post(
            "/v1/tutor/rehearse",
            json={"session_id": "s1"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_rehearse_falls_back_to_ok_when_no_client(self) -> None:
        """No Anthropic client → route returns 200 with verdict 'ok'."""
        import os
        svc = TutorService()
        # Do NOT set _client, and ensure no API key
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            app.state.tutor_service = svc
            resp = self.client.post(
                "/v1/tutor/rehearse",
                json={"session_id": "s1", "prompt": "build the service"},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["verdict"], "ok")
            self.assertEqual(body["original_prompt"], "build the service")
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved
            # Restore original service
            app.state.tutor_service = TutorService()


if __name__ == "__main__":
    unittest.main()
