"""Reference: production-grade multi-turn SaaS Customer Support Bot.

Design principle that makes the course winnable: the routing decisions
(`action`, `redactions`, `abstained`) are **deterministic policy-as-code**
— rules + regex; `citations` are grounded on **dense semantic retrieval**
(a pinned MiniLM embedding model + faiss) because keyword matching cannot
ground the vocabulary-mismatch scenarios. The LLM (reached only via the
platform proxy, Haiku) is used solely to phrase the natural-language
`reply`; if the proxy is unreachable the bot degrades to a grounded
template. So the pass bar is reachable with zero dependence on LLM
quality, and decision-stability/idempotency hold by construction.

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


# ---------------- retrieval (dense: MiniLM embeddings + faiss) ----------------
#
# Retrieval is SEMANTIC, not lexical. Many real support questions share
# almost no surface words with the help doc that actually answers them
# ("teammates aren't getting alerts when the site goes down" -> the
# *outage notifications* article, not the *team seats* article that
# happens to contain the word "teammates"). A keyword/BM25 ranker is
# fooled by a lexical decoy; a sentence-embedding ranker is not. So the
# reference grounds citations on cosine similarity of pinned-model
# embeddings indexed in faiss. Everything downstream of `retrieve()`
# (policy, PII, injection, abstention, multi-turn) stays deterministic.

_STOP = set(
    "a an the of for to in on at and or by with from is was are be been do did does "
    "i you he she it we they my your our this that these those how what why when can "
    "could would should will please help me my mine about into".split()
)


def _toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 1 and t not in _STOP]


# Pinned embedding model — the brief requires this exact model so the
# semantic winner is reproducible and gold is not model-dependent.
_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL = None  # lazy singleton; pure-core policy tests never load it

# cosine thresholds (tuned against this KB; verified server-side):
#  - below ABS_FLOOR  -> nothing in the KB supports the query -> abstain
#  - a runner-up is co-cited only if it is genuinely close to the top
#    AND independently relevant (keeps precision: single-winner queries
#    cite exactly one id, so subset_match against the gold set holds)
_ABS_FLOOR = 0.30
_REL_FLOOR = 0.42
_MARGIN = 0.94


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(_EMBED_MODEL_ID)
    return _MODEL


def retrieve(query: str, kb: list[dict]) -> tuple[list[str], float, dict | None]:
    """Return (citations, top_score, top_article). Dense semantic ranking:
    embed the query and each article with a pinned MiniLM model, index
    with faiss IndexFlatIP over L2-normalized vectors (inner product =
    cosine). Cites the semantically-supporting article(s); precision
    holds because ids come only from `kb` and runner-ups must be close
    AND independently relevant. A purely lexical ranker mis-cites the
    vocabulary-mismatch scenarios; this does not."""
    if not kb or not query.strip():
        return [], 0.0, None
    import faiss
    import numpy as np

    model = _model()
    docs = [(a.get("title", "") + ". " + a.get("text", "")).strip() for a in kb]
    doc_vecs = np.asarray(
        model.encode(docs, normalize_embeddings=True), dtype="float32"
    )
    q_vec = np.asarray(
        model.encode([query], normalize_embeddings=True), dtype="float32"
    )
    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    scores, idxs = index.search(q_vec, min(len(kb), 5))
    scores, idxs = scores[0], idxs[0]

    top_s = float(scores[0])
    top_a = kb[int(idxs[0])]
    if top_s < _ABS_FLOOR:
        return [], top_s, None
    cites = [top_a["article_id"]]
    for s, i in zip(scores[1:], idxs[1:]):
        s = float(s)
        a = kb[int(i)]
        if s >= _MARGIN * top_s and s >= _REL_FLOOR and a["article_id"] not in cites:
            cites.append(a["article_id"])
        if len(cites) >= 3:
            break
    return cites, top_s, top_a


# ---------------- policy-as-code ----------------

# Genuine account-security / fraud / legal incidents -> human agent.
# A plain "how do I reset my password" is NOT a security incident — it
# is a normal grounded answer (cite kb_password); only lockout /
# unauthorized-access / fraud / legal escalate. Keeps the routing spec
# unambiguous and ungameable.
_ESCALATE_RE = re.compile(
    r"(?i)\b(hacked|compromis|unauthor|fraud|stolen|locked out|can'?t log ?in|"
    r"cannot log ?in|lawsuit|legal|gdpr|ccpa|chargeback|dispute|data breach)\b"
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
