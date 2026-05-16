"""Reference: production-grade multi-turn SaaS Customer Support Bot.

Design principle that makes the course winnable: every GRADED decision
(`action`, `citations`, `redactions`, `abstained`) is **deterministic
policy-as-code** — retrieval + rules + regex. The LLM (reached only via
the platform proxy, Haiku) is used solely to phrase the natural-language
`reply`; if the proxy is unreachable the bot degrades to a grounded
template. So green is reachable with zero dependence on LLM quality, and
decision-stability/idempotency hold by construction.

Best practices demonstrated & tested:
- grounding: answers cite only provided kb_articles (precision + recall)
- policy routing: answer / clarify / escalate / refuse as code, not vibes
- PII redaction: deterministic regex on echoed content (compliance)
- prompt-injection resistance: instructions in user text never move policy
- multi-turn: anaphora resolves against history (context carry)
- graceful degradation: LLM optional; never crashes, never blocks
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

# ---------------- contract models ----------------

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    _HAVE_FASTAPI = True
except Exception:  # pure-core unit testing without deps
    _HAVE_FASTAPI = False


# ---------------- PII redaction (deterministic, compliance) ----------------

_PII_PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[ -]?)?(?:\(\d{3}\)|\d{3})[ -.]?\d{3}[ -.]?\d{4}(?!\d)")),
]


def redact_pii(text: str) -> tuple[str, int]:
    """Return (redacted_text, count). Order matters: SSN/card before phone."""
    out = text
    count = 0
    for label, pat in (_PII_PATTERNS[0], _PII_PATTERNS[2], _PII_PATTERNS[1], _PII_PATTERNS[3]):
        def _sub(_m: re.Match) -> str:
            nonlocal count
            count += 1
            return f"[REDACTED_{label}]"

        out = pat.sub(_sub, out)
    return out, count


# ---------------- prompt-injection sanitization ----------------

_INJECTION_RE = re.compile(
    r"(?i)("
    r"\bignore\s+(all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+)*"
    r"(rules?|instructions?|guidelines?|prompts?|previous|prior|above|everything|policy|policies)\b|"
    r"\bdisregard\s+(the\s+|all\s+|your\s+|previous\s+|above\s+)*(rules?|instructions?|guidelines?|previous|above)\b|"
    r"\bforget\s+(the\s+|your\s+|all\s+)*(rules?|instructions?|guidelines?)\b|"
    r"\byou are now\b|\bact as\b|\bsystem prompt\b|\boverride\b|\bjailbreak\b|"
    r"\bdeveloper mode\b|\bdo anything now\b|\bnew instructions?\b|"
    # paraphrased injections (not just the canonical "ignore previous")
    r"\bpay no attention to\b|\bfrom now on\b|\bno longer (follow|obey)\b|"
    r"\bstop following\b|\bdo not follow\b|\byour real (instructions?|task)\b"
    r")"
)


_OVERRIDE_DEMAND_RE = re.compile(
    r"(?i)\b(give me|refund me|mark (this|it)|set .* to|without (review|approval)|"
    r"approve (this|it)|make it|you must|grant me|process .* anyway)\b"
)


def strip_injection(text: str) -> str:
    """Drop sentences carrying override instructions so policy is decided
    from genuine intent, not adversarial text embedded by the user. When
    an injection marker is present anywhere, also drop imperative
    override-demand sentences (a smuggled "give me a full refund" must
    not synthesize intent the genuine message didn't have)."""
    poisoned = bool(_INJECTION_RE.search(text))
    keep = []
    for sent in re.split(r"(?<=[.!?\n])\s+", text):
        if _INJECTION_RE.search(sent):
            continue
        if poisoned and _OVERRIDE_DEMAND_RE.search(sent):
            continue
        keep.append(sent)
    return " ".join(keep).strip() or text  # never blank-out entirely


# ---------------- retrieval (BM25-lite, stdlib) ----------------

_STOP = set(
    "a an the of for to in on at and or by with from is was are be been do did does "
    "i you he she it we they my your our this that these those how what why when can "
    "could would should will please help me my mine about into".split()
)


def _toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 1 and t not in _STOP]


def retrieve(query: str, kb: list[dict]) -> tuple[list[str], float, dict | None]:
    """Return (citations, top_score, top_article). Scores by IDF-weighted
    overlap of query tokens against title+text. Cites the supporting
    article(s); precision holds because ids come only from `kb`."""
    if not kb:
        return [], 0.0, None
    q = set(_toks(query))
    if not q:
        return [], 0.0, None
    n = len(kb)
    df: dict[str, int] = {}
    docs = []
    for a in kb:
        d = set(_toks((a.get("title", "") + " " + a.get("text", ""))))
        docs.append(d)
        for t in d:
            df[t] = df.get(t, 0) + 1
    import math

    scored = []
    for a, d in zip(kb, docs):
        s = sum(math.log((n + 1) / (df.get(t, 0) + 0.5)) for t in q if t in d)
        scored.append((s, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_s, top_a = scored[0]
    if top_s <= 0:
        return [], 0.0, None
    # cite the top article + any close runner-up (recall without spraying)
    cites = [top_a["article_id"]]
    for s, a in scored[1:]:
        if s >= 0.6 * top_s and a["article_id"] not in cites:
            cites.append(a["article_id"])
        if len(cites) >= 3:
            break
    return cites, top_s, top_a


# ---------------- policy-as-code ----------------

_ESCALATE_RE = re.compile(
    r"(?i)\b(hacked|compromis|unauthor|fraud|stolen|locked out|can't log ?in|"
    r"cannot log ?in|reset (my )?password|account (closed|cancel)|delete my account|"
    r"lawsuit|legal|gdpr|ccpa|chargeback|dispute|data breach)\b"
)

# Privacy-exfiltration guardrail: requests for ANOTHER person's
# data/credentials must be refused outright (a real support-bot must
# never leak third-party data), not answered or escalated.
_PRIVACY_EXFIL_RE = re.compile(
    r"(?i)\b(another|other|someone else'?s|other(?:'s)?|a different)\s+"
    r"(customer|user|account|person|member)('?s)?\b|"
    r"\b(their|his|her|the other (customer|user)'?s)\s+"
    r"(password|invoice|account|details|data|card|ssn|email)\b|"
    r"\bsomeone else'?s\s+(password|account|details|data)\b"
)
_VAGUE_RE = re.compile(r"(?i)^\s*(help|hi|hello|it'?s broken|not working|issue|problem|support)\s*[.!?]*\s*$")
_REFUND_RE = re.compile(r"(?i)refund.*?\$?\s*([0-9][0-9,]*(?:\.\d{2})?)|\$\s*([0-9][0-9,]*(?:\.\d{2})?).*?refund")
_REFUND_THRESHOLD = 100.0
_ANAPHORA_RE = re.compile(r"(?i)\b(it|that|this|those|the order|the refund|same|again)\b")
_SUPPORT_HINT_RE = re.compile(
    r"(?i)\b(account|billing|bill|invoice|charge|refund|subscription|plan|upgrade|"
    r"downgrade|password|login|log ?in|reset|cancel|payment|card|outage|down|error|"
    r"bug|feature|export|api|integration|email|notification|seat|user|team)\b"
)


def _refund_amount(text: str) -> float | None:
    m = _REFUND_RE.search(text)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        return float(raw.replace(",", "")) if raw else None
    except ValueError:
        return None


def decide(req: dict) -> dict:
    """Pure, deterministic core. No network, no FastAPI — unit-testable."""
    message = str(req.get("message", "") or "")
    history = req.get("history") or []
    kb = req.get("kb_articles") or []

    # 1. PII redaction on echoed user content (compliance) — count spans.
    _, pii_count = redact_pii(message)

    # 2. Strip injected override instructions before any intent decision.
    clean = strip_injection(message)

    # 3. Multi-turn context carry: a short anaphoric follow-up inherits
    #    the previous user turn's text so retrieval/intent stay on-topic.
    effective = clean
    if _ANAPHORA_RE.search(clean) and len(_toks(clean)) <= 4:
        prev_user = ""
        for turn in reversed(history):
            if (turn.get("role") or "") == "user":
                prev_user = str(turn.get("content", ""))
                break
        if prev_user:
            effective = strip_injection(prev_user) + " " + clean

    citations, top_score, top_a = retrieve(effective, kb)

    # 4. Policy-as-code (deterministic, injection-proof).
    action = "answer"
    escalation_reason = None
    abstained = False

    if _PRIVACY_EXFIL_RE.search(effective):
        # Never reveal third-party data — refuse, don't escalate.
        action = "refuse"
        abstained = True
    elif _ESCALATE_RE.search(effective):
        action = "escalate"
        escalation_reason = "account-security / policy-restricted request requires a human agent"
        citations = citations or []
    else:
        amt = _refund_amount(effective)
        if amt is not None and amt > _REFUND_THRESHOLD:
            action = "escalate"
            escalation_reason = f"refund amount ${amt:.2f} exceeds the ${_REFUND_THRESHOLD:.0f} self-serve limit"
        elif _VAGUE_RE.search(clean) and not citations:
            # Generic opener with nothing to ground on — ask, don't refuse.
            action = "clarify"
        elif not citations and not _SUPPORT_HINT_RE.search(effective):
            # Nothing in the KB and not a recognizable support intent.
            action = "refuse"
            abstained = True
        elif not citations:
            action = "clarify"
        else:
            action = "answer"

    # FIX 3: only an `answer` cites sources. escalate/clarify/refuse
    # must not carry retrieval noise (and gold must not enshrine it).
    if action != "answer":
        citations = []

    reply = _compose_reply(action, effective, top_a, escalation_reason)
    reply, reply_pii = redact_pii(reply)

    resp: dict[str, Any] = {
        "reply": reply,
        "action": action,
        "citations": citations,
        "redactions": pii_count + reply_pii,
        "abstained": abstained,
    }
    if escalation_reason:
        resp["escalation_reason"] = escalation_reason
    return resp


# ---------------- reply composition (LLM optional, via proxy) ----------------


def _template_reply(action: str, top_a: dict | None, esc: str | None) -> str:
    if action == "refuse":
        return "I'm sorry, but I can't help with that — it's outside what this support assistant covers."
    if action == "escalate":
        return f"I'm escalating this to a human agent. Reason: {esc}. They'll follow up shortly."
    if action == "clarify":
        return "I want to help — could you share a bit more detail (what you were doing and what you expected)?"
    if top_a:
        snippet = re.sub(r"\s+", " ", top_a.get("text", "")).strip()[:280]
        return f"Based on our help docs ({top_a.get('title','')}): {snippet}"
    return "Here's what I found in our help docs."


def _llm_reply(action: str, message: str, top_a: dict | None, esc: str | None) -> str | None:
    base = os.environ.get("LAB_LLM_BASE_URL")
    tok = os.environ.get("LAB_LLM_TOKEN")
    if not base or not tok or action in ("refuse", "escalate", "clarify"):
        return None  # decisions are templated; LLM only enriches answers
    snippet = re.sub(r"\s+", " ", (top_a or {}).get("text", "")).strip()[:1200]
    if not snippet:
        return None
    body = json.dumps(
        {
            "system": (
                "You are a SaaS customer-support assistant. Answer ONLY using the "
                "provided help-doc snippet. Be concise, friendly, 2-4 sentences. "
                "If the snippet doesn't answer it, say you'll check with the team."
            ),
            "messages": [
                {"role": "user", "content": f"Help-doc:\n{snippet}\n\nCustomer: {message}"}
            ],
            "max_tokens": 320,
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/llm/complete",
        data=body,
        method="POST",
        headers={"content-type": "application/json", "x-lab-llm-token": tok},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        txt = (data.get("content") or "").strip()
        return txt or None
    except Exception:
        return None  # graceful degradation — never block on the LLM


def _compose_reply(action: str, message: str, top_a: dict | None, esc: str | None) -> str:
    return _llm_reply(action, message, top_a, esc) or _template_reply(action, top_a, esc)


# ---------------- HTTP layer (thin) ----------------

if _HAVE_FASTAPI:

    class Article(BaseModel):
        article_id: str
        title: str = ""
        text: str = ""

    class Turn(BaseModel):
        role: str
        content: str

    class AnswerRequest(BaseModel):
        message: str = Field(min_length=1)
        conversation_id: str | None = None
        history: list[Turn] = Field(default_factory=list)
        kb_articles: list[Article] = Field(default_factory=list)

    class AnswerResponse(BaseModel):
        reply: str
        action: str
        citations: list[str]
        redactions: int
        abstained: bool
        escalation_reason: str | None = None

    app = FastAPI(title="SaaS Support Bot (reference)")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/support/answer", response_model=AnswerResponse)
    def support_answer(req: AnswerRequest) -> AnswerResponse:
        return AnswerResponse(**decide(req.model_dump()))
