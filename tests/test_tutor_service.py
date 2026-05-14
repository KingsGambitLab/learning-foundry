from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.domain.tutor import TutorChatRequest, TutorSubmitRequest, TutorTriageRequest
from app.services.tutor_service import TutorService


def _make_fake_client(reply_text: str = "hello back") -> MagicMock:
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text=reply_text)]
    fake_client.with_options.return_value.messages.create.return_value = fake_response
    return fake_client


def _make_fake_store(session=None, submissions=None) -> MagicMock:
    store = MagicMock()
    store.get_learner_workspace_session.return_value = session
    store.list_learner_submissions.return_value = submissions or []
    return store


def _make_session(workspace_root: str, enrollment_id: str = "enroll_1") -> MagicMock:
    session = MagicMock()
    session.id = "studio_abc123"
    session.enrollment_id = enrollment_id
    session.workspace_root = workspace_root
    return session


def _make_submission(status: str, passed: int = 0, total: int = 3) -> MagicMock:
    sub = MagicMock()
    sub.status = status
    sub.passed_tests = passed
    sub.total_tests = total
    sub.created_at = "2026-05-14T10:00:00"

    grade_report = MagicMock()
    if status == "failed":
        failed_result = MagicMock()
        failed_result.status = "failed"
        failed_result.summary = "test_routing_priority FAILED: expected 'urgent' got 'normal'"
        failed_result.diagnostics = ["thread 0 panicked at src/main.rs:42"]
        grade_report.results = [failed_result]
    else:
        grade_report.results = []
    sub.grade_report = grade_report
    return sub


class TutorServiceTest(unittest.TestCase):
    def test_chat_calls_anthropic_and_returns_reply(self) -> None:
        svc = TutorService()
        fake_client = _make_fake_client()
        svc._client = fake_client
        reply = svc.chat(TutorChatRequest(session_id="s1", message="hello"))
        self.assertEqual(reply.reply, "hello back")
        self.assertIsNone(reply.hint_tier)
        fake_client.with_options.assert_called_once_with(timeout=30.0)

    def test_chat_returns_config_error_when_no_api_key(self) -> None:
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

    def test_chat_without_store_uses_basic_system_prompt(self) -> None:
        """No store → falls back to context-free prompt, user content is just the message."""
        svc = TutorService()
        fake_client = _make_fake_client("think harder")
        svc._client = fake_client

        svc.chat(TutorChatRequest(session_id="lms-random123", message="help me"))

        call_kwargs = fake_client.with_options.return_value.messages.create.call_args[1]
        system_text = call_kwargs["system"][0]["text"]
        user_content = call_kwargs["messages"][0]["content"]

        # System prompt should be plain persona with no project_brief
        self.assertNotIn("<project_brief>", system_text)
        self.assertNotIn("<deliverables>", system_text)
        # User content should be the raw message
        self.assertEqual(user_content, "help me")

    def test_chat_with_store_but_no_matching_session_uses_basic_prompt(self) -> None:
        """Store present but session_id doesn't resolve → context-free path."""
        store = _make_fake_store(session=None)
        svc = TutorService(store=store)
        fake_client = _make_fake_client("fallback reply")
        svc._client = fake_client

        svc.chat(TutorChatRequest(session_id="lms-unknown", message="what do I do?"))

        store.get_learner_workspace_session.assert_called_once_with("lms-unknown")
        call_kwargs = fake_client.with_options.return_value.messages.create.call_args[1]
        system_text = call_kwargs["system"][0]["text"]
        user_content = call_kwargs["messages"][0]["content"]

        self.assertNotIn("<project_brief>", system_text)
        self.assertEqual(user_content, "what do I do?")

    def test_chat_with_store_and_matching_session_includes_context(self) -> None:
        """Store resolves session → system prompt has brief; user turn has file tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "project_brief.md").write_text("# Routing Service\nBuild a routing engine.")
            (ws / "deliverables.md").write_text("## Deliverable 1\nImplement priority routing.")
            (ws / "src").mkdir()
            (ws / "src" / "main.rs").write_text('fn main() { println!("hello"); }')

            session = _make_session(workspace_root=str(ws))
            store = _make_fake_store(session=session, submissions=[])
            svc = TutorService(store=store)
            fake_client = _make_fake_client("what have you tried?")
            svc._client = fake_client

            svc.chat(TutorChatRequest(
                session_id="studio_abc123",
                message="Where do I start?",
                assignment_title="Routing Service",
            ))

            call_kwargs = fake_client.with_options.return_value.messages.create.call_args[1]
            system_text = call_kwargs["system"][0]["text"]
            user_content = call_kwargs["messages"][0]["content"]

            # System prompt should embed the project brief and deliverables
            self.assertIn("<project_brief>", system_text)
            self.assertIn("Routing Service", system_text)
            self.assertIn("<deliverables>", system_text)
            self.assertIn("priority routing", system_text)

            # User content should contain the file tree and the learner's question
            self.assertIn("<workspace>", user_content)
            self.assertIn("File tree:", user_content)
            self.assertIn("main.rs", user_content)
            self.assertIn("<learner_question>", user_content)
            self.assertIn("Where do I start?", user_content)

    def test_chat_with_failed_submission_includes_failure_details(self) -> None:
        """Store + session + failed submission → user turn includes recent_failure section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "project_brief.md").write_text("# Escalation Service")
            (ws / "deliverables.md").write_text("## Deliverable 1\nHandle escalation.")
            (ws / "src").mkdir()
            (ws / "src" / "lib.rs").write_text("pub fn escalate() {}")

            session = _make_session(workspace_root=str(ws))
            failed_sub = _make_submission(status="failed", passed=0, total=3)
            store = _make_fake_store(session=session, submissions=[failed_sub])
            svc = TutorService(store=store)
            fake_client = _make_fake_client("look at line 42")
            svc._client = fake_client

            svc.chat(TutorChatRequest(
                session_id="studio_abc123",
                message="My tests are failing",
            ))

            call_kwargs = fake_client.with_options.return_value.messages.create.call_args[1]
            user_content = call_kwargs["messages"][0]["content"]

            self.assertIn("<recent_failure>", user_content)
            self.assertIn("FAILED", user_content)
            self.assertIn("0/3", user_content)
            # The specific test summary from our fake submission
            self.assertIn("test_routing_priority", user_content)


class TutorServiceTriageTest(unittest.TestCase):
    def test_triage_returns_agent_when_no_api_key(self) -> None:
        """No API key → client is None → default to 'agent'."""
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            svc = TutorService()
            result = svc.triage(TutorTriageRequest(session_id="s1", prompt="write the whole service"))
            self.assertEqual(result.action, "agent")
            self.assertIn("not configured", result.reason)
            self.assertEqual(result.original_prompt, "write the whole service")
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_triage_returns_agent_when_api_call_raises(self) -> None:
        """API call raises → default to 'agent' (fail-open)."""
        svc = TutorService()
        fake_client = MagicMock()
        fake_client.with_options.return_value.messages.create.side_effect = RuntimeError("network error")
        svc._client = fake_client

        result = svc.triage(TutorTriageRequest(session_id="s1", prompt="implement everything"))
        self.assertEqual(result.action, "agent")
        self.assertIn("triage call failed", result.reason)

    def test_triage_returns_agent_when_response_is_unparsable(self) -> None:
        """Response JSON is garbage → default to 'agent'."""
        svc = TutorService()
        fake_client = _make_fake_client("NOT JSON AT ALL !!!")
        svc._client = fake_client

        result = svc.triage(TutorTriageRequest(session_id="s1", prompt="fix all the tests"))
        self.assertEqual(result.action, "agent")
        self.assertIn("parse failed", result.reason)

    def test_triage_returns_tutor_for_broad_prompt(self) -> None:
        """Mocked response sets action: 'tutor' → returns tutor verdict."""
        svc = TutorService()
        import json
        payload = json.dumps({"action": "tutor", "reason": "learner asked agent to do whole assignment"})
        fake_client = _make_fake_client(payload)
        svc._client = fake_client

        result = svc.triage(TutorTriageRequest(
            session_id="s1",
            prompt="write the whole service for me",
            assignment_title="Routing Service",
        ))
        self.assertEqual(result.action, "tutor")
        self.assertEqual(result.reason, "learner asked agent to do whole assignment")
        self.assertEqual(result.original_prompt, "write the whole service for me")

    def test_triage_returns_agent_for_focused_prompt(self) -> None:
        """Mocked response sets action: 'agent' → returns agent verdict."""
        svc = TutorService()
        import json
        payload = json.dumps({"action": "agent", "reason": "specific error explanation request"})
        fake_client = _make_fake_client(payload)
        svc._client = fake_client

        result = svc.triage(TutorTriageRequest(
            session_id="s1",
            prompt="what does 'cannot borrow as mutable' mean?",
        ))
        self.assertEqual(result.action, "agent")
        self.assertEqual(result.reason, "specific error explanation request")

    def test_triage_returns_agent_for_unknown_action_value(self) -> None:
        """Mocked response has unrecognised action → sanitised to 'agent'."""
        svc = TutorService()
        import json
        payload = json.dumps({"action": "unknown_value", "reason": "something"})
        fake_client = _make_fake_client(payload)
        svc._client = fake_client

        result = svc.triage(TutorTriageRequest(session_id="s1", prompt="some prompt"))
        self.assertEqual(result.action, "agent")

    def test_triage_uses_15s_timeout(self) -> None:
        """Triage uses a 15-second timeout (not the 30s chat timeout)."""
        svc = TutorService()
        import json
        payload = json.dumps({"action": "agent", "reason": "ok"})
        fake_client = _make_fake_client(payload)
        svc._client = fake_client

        svc.triage(TutorTriageRequest(session_id="s1", prompt="what is a trait?"))
        fake_client.with_options.assert_called_once_with(timeout=15.0)


if __name__ == "__main__":
    unittest.main()
