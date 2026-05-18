"""Platform LLM proxy for the Customer Support Bot course.

The grader sandbox blocks learner-container egress (M13). To let a
learner service make runtime LLM calls *safely and cost-bounded*, the
grader opens egress ONLY to this internal proxy and injects a
short-lived, per-submission scoped token. The proxy:

  - validates the scoped token (exists, not expired, budget remaining);
  - forwards to Anthropic **Haiku** only (cheap), reusing the platform
    key file;
  - charges the submission's token budget by real usage;
  - logs spend and enforces a GLOBAL USD hard-stop so a runaway course
    can never exceed the cap.

Runs as its own process (systemd unit) so it can bind the docker-bridge
address reachable from learner containers while the main app stays
loopback-only. Token mint/revoke is done by the grader via
``issue_token`` / ``revoke_token`` (shared JSON store on disk).
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

HAIKU_MODEL = "claude-haiku-4-5"
_PRICE_IN_PER_TOK = 1.00 / 1_000_000   # $/input token  (claude-haiku-4-5)
_PRICE_OUT_PER_TOK = 5.00 / 1_000_000  # $/output token

DEFAULT_PER_SUBMISSION_TOKENS = int(os.environ.get("LAB_LLM_SUBMISSION_TOKEN_CAP", "60000"))
GLOBAL_USD_CAP = float(os.environ.get("LAB_LLM_GLOBAL_USD_CAP", "5.0"))
TOKENS_PATH = Path(os.environ.get("LAB_LLM_TOKENS_PATH", "/opt/course-gen-codex/tmp/lab_llm_tokens.json"))
SPEND_LOG = Path(os.environ.get("LAB_LLM_SPEND_LOG", "/opt/course-gen-codex/logs/lab-llm-spend.log"))

_LOCK = threading.Lock()


# ---------------- shared token store (grader writes, proxy charges) ----------------


def _load_store() -> dict[str, Any]:
    if not TOKENS_PATH.exists():
        return {"tokens": {}, "cumulative_usd": 0.0}
    try:
        return json.loads(TOKENS_PATH.read_text())
    except Exception:
        return {"tokens": {}, "cumulative_usd": 0.0}


def _save_store(store: dict[str, Any]) -> None:
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKENS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(store))
    tmp.replace(TOKENS_PATH)  # atomic


def issue_token(submission_id: str, budget_tokens: int | None = None, ttl_s: int = 1800) -> str:
    """Mint a scoped token for one graded submission. Called by the grader."""
    tok = "labllm_" + secrets.token_urlsafe(24)
    with _LOCK:
        store = _load_store()
        store.setdefault("tokens", {})[tok] = {
            "submission_id": submission_id,
            "remaining": int(budget_tokens or DEFAULT_PER_SUBMISSION_TOKENS),
            "expires_at": time.time() + ttl_s,
        }
        _save_store(store)
    return tok


def revoke_token(tok: str) -> None:
    with _LOCK:
        store = _load_store()
        if store.get("tokens", {}).pop(tok, None) is not None:
            _save_store(store)


def _log_spend(submission_id: str, in_tok: int, out_tok: int, cumulative: float) -> None:
    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submission_id": submission_id,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "call_usd": round(in_tok * _PRICE_IN_PER_TOK + out_tok * _PRICE_OUT_PER_TOK, 6),
        "cumulative_usd": round(cumulative, 6),
    }
    with SPEND_LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---------------- Anthropic key bootstrap (reuse platform plumbing) ----------------

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    from app.services.anthropic_runtime_support import resolve_anthropic_env_file

    env_file = resolve_anthropic_env_file()
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() and v.strip():
                os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from anthropic import Anthropic

    _client = Anthropic()
    return _client


# ---------------- proxy app ----------------

# Model MUST be module-level: a Pydantic model defined inside create_app()
# is a closure ForwardRef that FastAPI/pydantic cannot resolve at request
# time. Guarded import so the token helpers stay importable without
# pydantic (local $0 tests).
try:
    from pydantic import BaseModel, Field

    class CompleteRequest(BaseModel):
        messages: list[dict] = Field(default_factory=list)
        system: str | None = None
        prompt: str | None = None
        max_tokens: int = Field(default=512, ge=1, le=2048)
except Exception:  # pragma: no cover - pydantic absent in local token-only tests
    CompleteRequest = None  # type: ignore


def create_app():
    from fastapi import Body, FastAPI, Header, HTTPException

    app = FastAPI(title="Lab LLM Proxy", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/llm/complete")
    def complete(
        req: CompleteRequest = Body(...),
        x_lab_llm_token: str = Header(default=""),
    ) -> dict:
        with _LOCK:
            store = _load_store()
            if store.get("cumulative_usd", 0.0) >= GLOBAL_USD_CAP:
                raise HTTPException(503, f"Global LLM budget (${GLOBAL_USD_CAP}) reached — refusing all calls.")
            entry = store.get("tokens", {}).get(x_lab_llm_token)
            if entry is None:
                raise HTTPException(401, "Invalid or unknown lab LLM token.")
            if time.time() > entry["expires_at"]:
                raise HTTPException(401, "Lab LLM token expired.")
            if entry["remaining"] <= 0:
                raise HTTPException(429, "Per-submission LLM token budget exhausted.")
            submission_id = entry["submission_id"]

        client = _get_client()
        if client is None:
            raise HTTPException(503, "LLM key not configured on the proxy host.")

        messages = req.messages or ([{"role": "user", "content": req.prompt}] if req.prompt else [])
        if not messages:
            raise HTTPException(422, "Provide `messages` or `prompt`.")
        kwargs: dict[str, Any] = {
            "model": HAIKU_MODEL,
            "max_tokens": req.max_tokens,
            "messages": messages,
        }
        if req.system:
            kwargs["system"] = req.system
        try:
            resp = client.with_options(timeout=30.0).messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Upstream LLM error: {exc!s}") from exc

        in_tok = int(getattr(resp.usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(resp.usage, "output_tokens", 0) or 0)
        text = "".join(
            getattr(b, "text", "") for b in (resp.content or []) if getattr(b, "type", "") == "text"
        )

        with _LOCK:
            store = _load_store()
            entry = store.get("tokens", {}).get(x_lab_llm_token)
            if entry is not None:
                entry["remaining"] = max(0, entry["remaining"] - (in_tok + out_tok))
            call_usd = in_tok * _PRICE_IN_PER_TOK + out_tok * _PRICE_OUT_PER_TOK
            store["cumulative_usd"] = round(store.get("cumulative_usd", 0.0) + call_usd, 6)
            _save_store(store)
            cumulative = store["cumulative_usd"]
        _log_spend(submission_id, in_tok, out_tok, cumulative)

        return {
            "content": text,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            "model": HAIKU_MODEL,
        }

    return app


# NB: the app is created lazily (only when run as the proxy process) so
# the grader can `from app.services.lab_llm_proxy import issue_token,
# revoke_token` without importing FastAPI or standing up a server.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("LAB_LLM_BIND", "127.0.0.1"),
        port=int(os.environ.get("LAB_LLM_PORT", "8055")),
    )
