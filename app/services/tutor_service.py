from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIError

from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorSubmitRequest,
    TutorSubmitResponse,
    TutorVivaQuestion,
)

_TUTOR_SYSTEM_PROMPT = """You are a Lab Tutor for a learner working on a graded coding assignment.

Your job is to coach, not to solve. Use short, Socratic questions. Help the learner think through the problem themselves. Never write more than 3-5 lines of code for them. If they ask for the full solution, gently redirect them to decompose the problem.

Keep responses concise — 2-4 sentences in most cases. No preamble, no "Great question!", no fluff. Just the helpful nudge.

If the learner is stuck, offer a single concrete hint at a time. Escalate only if they're still stuck after the hint.

You don't have access to their code or the assignment text — work from what they tell you."""

_CHAT_PREVIEW_LIMIT = 80


def _load_env_file(path: str | None) -> None:
    """Parse a KEY=VALUE env file and setdefault each entry into os.environ."""
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            # Treat empty-string env vars the same as unset so a stale `KEY=`
            # exported by the parent shell doesn't shadow the file.
            os.environ[key] = value


class TutorService:
    """Phase 1 tutor backend.

    Chat uses Claude Haiku 4.5 with a Socratic-coach system prompt.
    Submit/viva grading is still a canned stub.
    """

    def __init__(
        self,
        *,
        anthropic_env_file: str | None = None,
        model: str = "claude-haiku-4-5",
    ) -> None:
        _load_env_file(anthropic_env_file)
        self._model = model
        # Lazy client init — if the API key is missing we want a clean runtime error
        # surfaced to the learner, not a startup crash.
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic | None:
        if self._client is not None:
            return self._client
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        self._client = Anthropic()
        return self._client

    def chat(self, req: TutorChatRequest) -> TutorChatResponse:
        client = self._get_client()
        if client is None:
            return TutorChatResponse(
                reply=(
                    "Tutor backend not configured — set "
                    "COURSE_GEN_ANTHROPIC_ENV_FILE to an env file containing "
                    "ANTHROPIC_API_KEY."
                ),
                hint_tier=None,
            )

        try:
            response = client.with_options(timeout=30.0).messages.create(
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": _TUTOR_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": req.message}],
            )
        except APIError as exc:
            return TutorChatResponse(
                reply=f"Tutor backend error: {exc.message if hasattr(exc, 'message') else exc!s}",
                hint_tier=None,
            )

        text = next(
            (b.text for b in response.content if b.type == "text"),
            "",
        ).strip()
        if not text:
            text = "(no response)"
        return TutorChatResponse(reply=text, hint_tier=None)

    def submit(self, req: TutorSubmitRequest) -> TutorSubmitResponse:
        # Phase 1 stub — real grading lands later.
        return TutorSubmitResponse(
            test_results={"passed": True, "details": "stub"},
            viva_questions=[
                TutorVivaQuestion(prompt="Explain why you chose this data structure."),
                TutorVivaQuestion(prompt="Walk through your error handling."),
            ],
        )
