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

_TUTOR_SYSTEM_BASE = """You are a Lab Tutor — a sharp, direct coach sitting next to a learner doing a graded coding assignment.

Your job is to make them think, not to think for them. Be confident. Be specific. One pointed question or one concrete hint per turn — then stop. Do not pad responses with "Great question!" or hedge with "you might want to consider". Get to the point.

Hard rules:
- Never write more than 3-5 lines of code in a single response. If a longer block is the only way, refuse and ask them what they've tried.
- Never reveal the full solution or post-test answer keys.
- If they ask for the answer, push back: say "I won't hand that over. Walk me through what you've tried." Keep it short.
- Assume they're capable. Don't over-explain. Don't soften with "this might be tricky, but...".

Style: 2-4 sentences. Treat them like a peer, not a beginner. The learner sees your responses in a sidebar next to their code editor."""

_CHAT_PREVIEW_LIMIT = 80


def _build_system_prompt(assignment_title: str | None) -> str:
    if not assignment_title:
        return _TUTOR_SYSTEM_BASE
    return (
        _TUTOR_SYSTEM_BASE
        + f"\n\nThe learner is currently working on: **{assignment_title}**. "
          "When they ask something context-free, assume it's about this assignment. "
          "Do not ask 'what are you working on' — you already know."
    )


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

        system_prompt = _build_system_prompt(req.assignment_title)
        try:
            response = client.with_options(timeout=30.0).messages.create(
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
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
