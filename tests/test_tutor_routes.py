import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.services.tutor_service import TutorService


class TutorRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        app.state.tutor_service = TutorService()
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


if __name__ == "__main__":
    unittest.main()
