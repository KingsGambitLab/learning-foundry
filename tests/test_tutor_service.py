import unittest

from app.domain.tutor import TutorChatRequest, TutorSubmitRequest
from app.services.tutor_service import TutorService


class TutorServiceTest(unittest.TestCase):
    def test_chat_calls_anthropic_and_returns_reply(self) -> None:
        svc = TutorService()
        # Force client construction with a stub instead of real network
        from unittest.mock import MagicMock, patch
        fake_client = MagicMock()
        fake_response = MagicMock()
        fake_response.content = [MagicMock(type="text", text="hello back")]
        fake_client.with_options.return_value.messages.create.return_value = fake_response
        svc._client = fake_client
        reply = svc.chat(TutorChatRequest(session_id="s1", message="hello"))
        self.assertEqual(reply.reply, "hello back")
        self.assertIsNone(reply.hint_tier)
        fake_client.with_options.assert_called_once_with(timeout=30.0)

    def test_chat_returns_config_error_when_no_api_key(self) -> None:
        import os
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            svc = TutorService()
            reply = svc.chat(TutorChatRequest(session_id="s1", message="hello"))
            self.assertIn("not configured", reply.reply.lower())
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_submit_returns_two_viva_questions(self) -> None:
        svc = TutorService()
        result = svc.submit(
            TutorSubmitRequest(session_id="s1", code_snapshot="code")
        )
        self.assertEqual(result.test_results["passed"], True)
        self.assertEqual(len(result.viva_questions), 2)
        for q in result.viva_questions:
            self.assertTrue(q.prompt)


if __name__ == "__main__":
    unittest.main()
