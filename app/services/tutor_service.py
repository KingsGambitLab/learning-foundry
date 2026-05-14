from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from anthropic import Anthropic, APIError

from app.domain.tutor import (
    TutorChatRequest,
    TutorChatResponse,
    TutorSubmitRequest,
    TutorSubmitResponse,
    TutorVivaQuestion,
)

if TYPE_CHECKING:
    from app.storage.sqlite_store import SQLiteWorkflowStore

logger = logging.getLogger(__name__)

_TUTOR_PERSONA = """You are a sharp, direct lab tutor sitting next to a learner working on a graded coding assignment.

Your job is to make the learner think. You do not solve for them, ever. When they ask for the answer, refuse plainly and ask what they have tried.

Style:
- Be confident and concrete. No apologies, no "Great question!", no preamble.
- One pointed question or one specific hint per reply. Then stop.
- Maximum 2-3 short sentences. No lists unless the learner explicitly asks for one.
- Never write more than 3-5 lines of code, and only when illustrating a technique they could not look up.
- Assume the learner is intelligent and capable — talk to them like a peer who has read more, not a lecturer."""

# Workspace files we never include (build artefacts, VCS, lockfiles, vendored deps).
_SKIP_DIRS = {
    ".git", ".github", ".vscode", ".idea", ".cache",
    "node_modules", "target", "dist", "build", "out",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".coursegen",  # platform-managed metadata
    ".venv", "venv",
}
_SKIP_SUFFIXES = {".lock", ".pyc", ".pyo", ".so", ".o", ".a", ".class", ".jar"}
_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".tar", ".gz", ".bin"}

# Hard size limits to keep prompt cost bounded.
_PER_FILE_MAX_BYTES = 8 * 1024   # skip a file if it exceeds this on its own
_TOTAL_CODE_MAX_BYTES = 40 * 1024  # truncate code-snapshot collection at this aggregate

# Failure detail caps.
_FAILURE_MAX_BYTES = 4 * 1024


def _load_env_file(path: str | None) -> None:
    """Parse a KEY=VALUE env file. Treat empty env vars as unset."""
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
            os.environ[key] = value


def _read_text_safely(path: Path, max_bytes: int) -> str | None:
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("[tutor] could not read %s: %s", path, exc)
        return None


def _iter_workspace_files(root: Path):
    """Walk the workspace, yielding (relative_path, absolute_path) for text files."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = set(rel.parts)
        if parts & _SKIP_DIRS:
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES or path.suffix.lower() in _BINARY_SUFFIXES:
            continue
        yield rel, path


def _build_code_snapshot(root: Path) -> str:
    """Compact tree + bounded file contents."""
    files = list(_iter_workspace_files(root))
    if not files:
        return ""

    # Sort by modification time (newest first) so the learner's recent edits win the budget.
    files.sort(key=lambda pair: pair[1].stat().st_mtime, reverse=True)

    tree_lines = [str(rel) for rel, _ in files[:80]]
    body_lines: list[str] = []
    budget = _TOTAL_CODE_MAX_BYTES
    for rel, abs_path in files:
        if budget <= 0:
            break
        text = _read_text_safely(abs_path, _PER_FILE_MAX_BYTES)
        if text is None:
            continue
        chunk = f"\n--- {rel} ---\n{text}\n"
        chunk_bytes = len(chunk.encode("utf-8"))
        if chunk_bytes > budget:
            # Truncate this file to fit
            keep = max(0, budget - len(f"\n--- {rel} ---\n... [truncated]\n".encode("utf-8")))
            text = text.encode("utf-8")[:keep].decode("utf-8", errors="ignore")
            chunk = f"\n--- {rel} ---\n{text}\n... [truncated]\n"
        body_lines.append(chunk)
        budget -= len(chunk.encode("utf-8"))

    return "File tree:\n" + "\n".join(f"  {ln}" for ln in tree_lines) + "\n\nFile contents:\n" + "".join(body_lines)


def _read_brief_files(root: Path) -> tuple[str, str]:
    """Return (project_brief, deliverables) as strings."""
    brief = _read_text_safely(root / "project_brief.md", 32 * 1024) or ""
    delivs = _read_text_safely(root / "deliverables.md", 32 * 1024) or ""
    return brief.strip(), delivs.strip()


def _format_failure(submission) -> str | None:
    """Pull a compact failure summary out of a LearnerSubmissionRecord.

    The record has typed fields: status (str), passed_tests, total_tests,
    pass_rate, grade_report (DeliverableGradeReport with results list).
    """
    try:
        status = getattr(submission, "status", None)
        # status is a plain str — check for "failed"
        if status != "failed":
            return None

        grade_report = getattr(submission, "grade_report", None)
        if grade_report is None:
            return None

        created_at = getattr(submission, "created_at", "?")
        passed = getattr(submission, "passed_tests", "?")
        total = getattr(submission, "total_tests", "?")

        # Collect failure details from individual test results
        failed_details: list[str] = []
        results = getattr(grade_report, "results", [])
        for r in results:
            r_status = getattr(r, "status", None)
            # GradeStatus.failed == "failed"
            if str(r_status) == "failed":
                summary = getattr(r, "summary", "")
                diagnostics = getattr(r, "diagnostics", [])
                entry = f"- {summary}"
                if diagnostics:
                    entry += "\n  " + "\n  ".join(str(d) for d in diagnostics[:3])
                failed_details.append(entry)

        if not failed_details:
            # Still report the failure even without per-test details
            return (
                f"Most recent submission at {created_at} FAILED "
                f"({passed}/{total} tests passed). No detailed diagnostics available."
            )

        details_text = "\n".join(failed_details)[:_FAILURE_MAX_BYTES]
        return (
            f"Most recent submission at {created_at} FAILED "
            f"({passed}/{total} tests passed).\nFailed tests:\n{details_text}"
        )
    except Exception as exc:
        logger.debug("[tutor] _format_failure error: %s", exc)
        return None


class TutorService:
    """Phase 1 tutor backend. Reads context from disk + DB on every chat."""

    def __init__(
        self,
        *,
        anthropic_env_file: str | None = None,
        model: str = "claude-haiku-4-5",
        store: "SQLiteWorkflowStore | None" = None,
    ) -> None:
        _load_env_file(anthropic_env_file)
        self._model = model
        self._store = store
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic | None:
        if self._client is not None:
            return self._client
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        self._client = Anthropic()
        return self._client

    def _resolve_session_context(self, session_id: str) -> tuple[str, str, str, str | None] | None:
        """Return (project_brief, deliverables, code_snapshot, failure_summary) or None.

        None means we couldn't resolve the session (widget on LMS page, missing DB row, etc.)
        and the chat should proceed without filesystem context.
        """
        if self._store is None:
            return None
        try:
            session = self._store.get_learner_workspace_session(session_id)
        except Exception as exc:
            logger.debug("[tutor] session lookup failed for %s: %s", session_id, exc)
            return None
        if session is None:
            return None

        workspace_root = Path(getattr(session, "workspace_root", ""))
        if not workspace_root.exists():
            logger.debug("[tutor] workspace_root missing for %s: %s", session_id, workspace_root)
            return None

        brief, delivs = _read_brief_files(workspace_root)
        code = _build_code_snapshot(workspace_root)

        failure: str | None = None
        enrollment_id = getattr(session, "enrollment_id", None)
        if enrollment_id:
            try:
                submissions = self._store.list_learner_submissions(enrollment_id)
                # Already ordered DESC by created_at from the store query; pick first failure
                for sub in submissions:
                    summary = _format_failure(sub)
                    if summary:
                        failure = summary
                        break
            except Exception as exc:
                logger.debug("[tutor] failure lookup error: %s", exc)

        return brief, delivs, code, failure

    def _build_system_prompt(
        self,
        assignment_title: str | None,
        project_brief: str,
        deliverables: str,
    ) -> str:
        parts = [_TUTOR_PERSONA]
        if assignment_title:
            parts.append(f"\nThe learner is working on: {assignment_title}.")
        if project_brief:
            parts.append("\n\n<project_brief>\n" + project_brief + "\n</project_brief>")
        if deliverables:
            parts.append("\n\n<deliverables>\n" + deliverables + "\n</deliverables>")
        return "".join(parts)

    def _build_user_message(
        self,
        message: str,
        code_snapshot: str,
        failure_summary: str | None,
    ) -> str:
        sections: list[str] = []
        if code_snapshot:
            sections.append("<workspace>\n" + code_snapshot + "\n</workspace>")
        if failure_summary:
            sections.append("<recent_failure>\n" + failure_summary + "\n</recent_failure>")
        if sections:
            return "\n\n".join(sections) + "\n\n<learner_question>\n" + message + "\n</learner_question>"
        return message

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

        ctx = self._resolve_session_context(req.session_id)
        if ctx is not None:
            project_brief, deliverables, code_snapshot, failure_summary = ctx
        else:
            project_brief, deliverables, code_snapshot, failure_summary = "", "", "", None

        system_text = self._build_system_prompt(req.assignment_title, project_brief, deliverables)
        user_text = self._build_user_message(req.message, code_snapshot, failure_summary)

        try:
            response = client.with_options(timeout=30.0).messages.create(
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_text}],
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
