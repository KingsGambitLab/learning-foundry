from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from anthropic import Anthropic, APIError

from app.domain.tutor import (
    TutorChatMessage,
    TutorChatRequest,
    TutorChatResponse,
    TutorSubmitRequest,
    TutorSubmitResponse,
    TutorTriageRequest,
    TutorTriageResponse,
    TutorVivaQuestion,
)
from app.storage.workflow_store import WorkflowStore

logger = logging.getLogger(__name__)

_TUTOR_PERSONA = """You are a tutor who cares about the learner's growth, sitting next to them while they work through a graded coding assignment. You're rooting for them.

Your job is to help them think, not to solve the problem for them. When they ask for the full answer, gently explain they'll learn more by working through it themselves, and invite them into the next small step — what have they tried, what's the part they're stuck on, what does the spec actually say.

Style:
- Warm but direct. Acknowledge the question or what they're trying, then move forward. No flattery ("Great question!"), no over-hedging.
- One specific hint or one pointed question per reply. Don't pile on.
- Keep replies short — usually 2-4 sentences. Bullet lists only if they ask.
- If you write code, keep it tiny (3-5 lines max) and only to illustrate a technique they couldn't easily look up.
- Talk to them like a peer who's worked through this kind of problem before — not like a lecturer or a drill sergeant.
- When you have to push back (e.g. "just write it for me"), be kind about it. Explain why briefly, then offer the next thing they can try.
- Diagram-first: include a Mermaid diagram in your reply WHENEVER the content can be visualized — for this kind of work that is almost always. Default to drawing; only skip for a pure yes/no or a single-token syntax fact. Pick the SINGLE best diagram type for what you're explaining:
  - breaking a problem/spec into parts → `mindmap`
  - a pipeline, control flow, or decision logic → `flowchart` / `graph LR`
  - request lifecycle / who-calls-whom over time → `sequenceDiagram`
  - retry/lifecycle/escalation states → `stateDiagram-v2`
  - data model / schema → `erDiagram`
  - comparing options on tradeoffs → `quadrantChart`
  Use a fenced block with the mermaid language tag, e.g.:
  ```mermaid
  graph LR
    A[Query] --> B[Embed]
    B --> C[Search]
    C --> D[Rerank]
  ```
  Keep diagrams small (5-10 nodes). Always pair the diagram with ONE short follow-up question or a single specific hint — the diagram is a teaching aid, not a replacement for the conversation.
- You ALWAYS know the assignment from the title and (when present) the <project_brief> / <deliverables> below. NEVER ask the learner to paste the assignment, the spec, or "the question" — reason from the context you already have and, if the brief is thin, make a reasonable best-effort diagram from the title and say what you assumed. Asking them to paste what you can already see is the one thing that breaks their trust."""

_TRIAGE_JUDGE_PROMPT = """You sit between a learner working on a graded coding assignment and an AI coding agent the learner can talk to. Your one job: decide which side handles the learner's next prompt.

Route to AGENT (action: "agent") when the learner is using the agent as a TOOL for a focused, bounded task:
- Asking a conceptual question — "what does this trait bound mean", "explain async/await", "what is cargo"
- Looking up syntax, naming, or library docs
- Debugging a specific error or test failure they're already looking at — "what does this error mean: cannot borrow as mutable"
- Renaming, formatting, or refactoring a specific symbol they already wrote
- Asking the agent to read/summarise a piece of code that already exists
- Asking "is this approach right" with a concrete sketch attached

Route to TUTOR (action: "tutor") when the learner is asking the agent to do the assignment FOR them — to write the implementation, build the system, complete the deliverable, generate the solution end-to-end. This is the case where, if the agent answered, the learner would skip the actual problem-solving:
- "implement the route handler", "write the escalation logic", "build the service"
- "write all the code", "do the assignment", "complete this"
- "fix all the failing tests" (without any diagnosis attached)
- "implement the README" (meaning: build everything described in it)
- "generate the solution"
- Prompts vague enough that the agent would have to invent the spec ("just make it work", "do whatever you think is best")

The bar: would a senior reviewer watching over the learner's shoulder say "wait, you should think this one through first"? If yes → tutor. If they'd let the prompt fly to the agent without intervention → agent.

When in doubt, lean toward AGENT — the learner can ALWAYS choose to ask the tutor directly through the tutor widget; we only intercept when the agent answering would replace the learner's own problem-solving.

Output strictly JSON: {"action": "agent" | "tutor", "reason": "<one short sentence>"}"""


_TRIAGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["agent", "tutor"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}

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


# Learner workspaces live at <repo>/learner_workspaces/<learner_id>/
# <shared_workflow_run_id>/workspace (mirrors lms_service._workspace_root).
# tutor_service is app/services/, so parents[2] == repo root.
_LEARNER_WS_BASE = Path(__file__).resolve().parents[2] / "learner_workspaces"


def _read_brief_files(root: Path) -> tuple[str, str]:
    """Return (project_brief, deliverables) as strings.

    Workspaces now seed a single consolidated ``README.md`` (the brief
    + a Deliverables section); ``project_brief.md`` / ``deliverables.md``
    are legacy and no longer written. Prefer README; fall back to the
    legacy split files for older workspaces."""
    readme = _read_text_safely(root / "README.md", 48 * 1024) or ""
    if readme.strip():
        return readme.strip(), ""
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
        model: str = "claude-sonnet-4-6",
        store: WorkflowStore | None = None,
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

        # CONTRACT (do not break — see docs/lab-tutor.md "Context wiring"):
        # both the LMS-page widget (lms.js) and the in-editor widget
        # (lab-tutor-editor-boot.js via /v1/tutor/editor-context) send
        # session_id = "lms-<enrollmentId>". A LearnerWorkspaceSession.id
        # is NOT that value, so the legacy exact-id match alone always
        # missed -> the tutor never got the brief and asked the learner
        # to paste the spec. Resolve the "lms-" namespace too, and fall
        # back to the publish-snapshot brief when no workspace exists yet.
        try:
            sessions = list(self._store.list_all_learner_workspace_sessions())
        except Exception as exc:
            logger.debug("[tutor] session lookup failed for %s: %s", session_id, exc)
            sessions = []

        session = next((s for s in sessions if getattr(s, "id", None) == session_id), None)
        enrollment_id: str | None = None
        if session is None and session_id.startswith("lms-"):
            enrollment_id = session_id[len("lms-"):]
            cand = [s for s in sessions if getattr(s, "enrollment_id", None) == enrollment_id]
            cand.sort(
                key=lambda s: (
                    getattr(getattr(s, "status", None), "value", "") == "running",
                    getattr(s, "updated_at", None) or "",
                ),
                reverse=True,
            )
            session = cand[0] if cand else None
        if enrollment_id is None and session is not None:
            enrollment_id = getattr(session, "enrollment_id", None)

        # Resolve the enrollment once — used both to locate the live
        # learner workspace and for the snapshot-brief fallback.
        enr = None
        if enrollment_id:
            try:
                enr = self._store.get_learner_enrollment(enrollment_id)
            except Exception as exc:
                logger.debug("[tutor] enrollment lookup failed for %s: %s", session_id, exc)
                enr = None

        # Find the learner's live workspace dir so the tutor can READ
        # their code (the #1 thing it needs to actually help). Prefer the
        # session's workspace_root; it is frequently unset on editor
        # sessions, so fall back to the deterministic LMS path
        # learner_workspaces/<learner_id>/<shared_workflow_run_id>/workspace.
        brief, delivs, code = "", "", ""
        ws_dir: Path | None = None
        if session is not None:
            sroot = Path(getattr(session, "workspace_root", "") or "")
            if str(sroot) and sroot.exists():
                ws_dir = sroot
        if ws_dir is None and enr is not None:
            cand = (
                _LEARNER_WS_BASE
                / str(enr.learner_id)
                / str(enr.shared_workflow_run_id)
                / "workspace"
            )
            if cand.exists():
                ws_dir = cand
        if ws_dir is not None:
            brief, delivs = _read_brief_files(ws_dir)
            code = _build_code_snapshot(ws_dir)

        # No materialized workspace yet (learner hasn't opened the editor)
        # — use the publish snapshot's brief so the tutor still has the
        # full spec. The tutor must NEVER ask the learner to paste it.
        if not brief and enr is not None:
            try:
                snap = self._store.get_publish_snapshot(enr.publish_snapshot_id)
                if snap is not None:
                        from app.services.learner_package_runtime import (
                            deliverables_markdown,
                            project_brief_markdown,
                        )
                        # Only accept real, non-empty strings — a
                        # malformed snapshot (or a test double) must not
                        # let a non-str flow into _build_system_prompt's
                        # "".join (TypeError).
                        _b = project_brief_markdown(snap)
                        if isinstance(_b, str) and _b.strip():
                            brief = _b
                        lp = getattr(snap, "learner_package", None)
                        if lp is not None and getattr(lp, "deliverables", None):
                            _d = deliverables_markdown(lp.deliverables)
                            if isinstance(_d, str) and _d.strip():
                                delivs = _d
            except Exception as exc:
                logger.debug("[tutor] snapshot brief fallback failed for %s: %s", session_id, exc)

        if not brief and not delivs and session is None:
            return None

        failure: str | None = None
        if enrollment_id:
            try:
                for sub in self._store.list_learner_submissions(enrollment_id):
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

    def _persist_turn(
        self, user_id: str, session_id: str, user_msg: str, tutor_reply: str
    ) -> None:
        """Append the user turn + tutor reply to durable storage. Best
        effort: a persistence failure must never break the chat reply
        (consistent with the rest of the tutor's graceful degradation)."""
        if self._store is None or not user_id:
            return
        try:
            now = datetime.now(UTC)
            # The tutor reply must sort strictly AFTER the user turn —
            # identical timestamps tie and the transcript can render
            # reversed (history is ordered by created_at).
            self._store.append_tutor_chat_message(
                TutorChatMessage(
                    id=f"tcm_{uuid.uuid4().hex}",
                    user_id=user_id,
                    session_id=session_id,
                    role="user",
                    text=user_msg,
                    created_at=now,
                )
            )
            self._store.append_tutor_chat_message(
                TutorChatMessage(
                    id=f"tcm_{uuid.uuid4().hex}",
                    user_id=user_id,
                    session_id=session_id,
                    role="tutor",
                    text=tutor_reply,
                    created_at=now + timedelta(milliseconds=1),
                )
            )
        except Exception as exc:  # noqa: BLE001 — never block the reply
            logger.warning("[tutor] chat persistence failed: %s", exc)

    def chat(
        self, req: TutorChatRequest, user_id: str | None = None
    ) -> TutorChatResponse:
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
        if user_id:
            self._persist_turn(user_id, req.session_id, req.message, text)
        return TutorChatResponse(reply=text, hint_tier=None)

    def triage(self, req: TutorTriageRequest) -> TutorTriageResponse:
        """Decide whether the agent or the tutor handles the learner's prompt.

        Defaults to 'agent' on any error so the learner is never blocked by a
        flaky backend.
        """
        client = self._get_client()
        if client is None:
            return TutorTriageResponse(
                action="agent",
                reason="tutor backend not configured",
                original_prompt=req.prompt,
            )

        # Build the system prompt: judge rubric + (optional) project brief +
        # deliverables. NO workspace code, NO failure summary — the triage
        # decision is about the *prompt*, not the code state.
        parts = [_TRIAGE_JUDGE_PROMPT]
        ctx = self._resolve_session_context(req.session_id)
        if ctx is not None:
            project_brief, deliverables, _code, _failure = ctx
            if project_brief:
                parts.append("\n\n<project_brief>\n" + project_brief + "\n</project_brief>")
            if deliverables:
                parts.append("\n\n<deliverables>\n" + deliverables + "\n</deliverables>")
        elif req.assignment_title:
            parts.append(f"\n\nThe learner is working on: {req.assignment_title}.")
        system_text = "".join(parts)

        user_text = f"<learner_prompt>\n{req.prompt}\n</learner_prompt>"

        try:
            response = client.with_options(timeout=15.0).messages.create(
                model=self._model,
                max_tokens=256,
                system=[
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_text}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _TRIAGE_JSON_SCHEMA,
                    }
                },
            )
        except Exception as exc:
            logger.warning("[tutor] triage API call failed: %s", exc)
            return TutorTriageResponse(
                action="agent",
                reason=f"triage call failed: {exc!s}",
                original_prompt=req.prompt,
            )

        try:
            import json as _json
            raw_text = next(
                (b.text for b in response.content if b.type == "text"),
                "{}",
            )
            parsed = _json.loads(raw_text)
            action = str(parsed.get("action", "agent"))
            if action not in ("agent", "tutor"):
                action = "agent"
            reason = str(parsed.get("reason", ""))
        except Exception as exc:
            logger.warning("[tutor] triage response parse failed: %s", exc)
            return TutorTriageResponse(
                action="agent",
                reason="triage response parse failed",
                original_prompt=req.prompt,
            )

        return TutorTriageResponse(
            action=action,
            reason=reason,
            original_prompt=req.prompt,
        )

    def submit(self, req: TutorSubmitRequest) -> TutorSubmitResponse:
        # Phase 1 stub — real grading lands later.
        return TutorSubmitResponse(
            test_results={"passed": True, "details": "stub"},
            viva_questions=[
                TutorVivaQuestion(prompt="Explain why you chose this data structure."),
                TutorVivaQuestion(prompt="Walk through your error handling."),
            ],
        )
