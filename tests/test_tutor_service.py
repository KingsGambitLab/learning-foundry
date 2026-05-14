import unittest

from app.domain.tutor import TutorChatRequest, TutorSubmitRequest
from app.services.tutor_service import TutorService


class TutorServiceTest(unittest.TestCase):
    def test_chat_echoes_truncated_message(self) -> None:
        svc = TutorService()
        reply = svc.chat(TutorChatRequest(session_id="s1", message="hello"))
        self.assertIn("hello", reply.reply)
        self.assertEqual(reply.hint_tier, None)

    def test_chat_truncates_long_message(self) -> None:
        svc = TutorService()
        msg = "x" * 200
        reply = svc.chat(TutorChatRequest(session_id="s1", message=msg))
        self.assertLessEqual(len(reply.reply), 120)

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
