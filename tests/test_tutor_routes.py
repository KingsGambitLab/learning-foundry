import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.services.tutor_service import TutorService


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
