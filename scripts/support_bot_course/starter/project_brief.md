# Build a production Customer Support Bot (multi-turn, SaaS)

Build one HTTP service: `POST /support/answer`. It answers SaaS support
questions **grounded in the knowledge base supplied in each request**,
routes by **policy-as-code**, redacts **PII**, resists **prompt
injection**, and holds **context across turns** — graded against hidden
multi-turn scenarios.

## Endpoint contract

`POST /support/answer`

Request:
```json
{
  "message": "string (required)",
  "conversation_id": "string (optional)",
  "history": [{"role": "user|assistant", "content": "string"}],
  "kb_articles": [{"article_id": "string", "title": "string", "text": "string"}]
}
```
Response:
```json
{
  "reply": "string",
  "action": "answer | clarify | escalate | refuse",
  "citations": ["article_id"],
  "redactions": 0,
  "abstained": false,
  "escalation_reason": "string (only when action=escalate)"
}
```

## Skills you are graded on

All skills below are **gating** (must pass to score) except S8.

- **S1 — Retrieval-grounded answering:** answer only from `kb_articles`; cite the supporting `article_id`s (and only those).
- **S2 — Policy-as-code routing:** choose `answer` / `clarify` / `escalate` / `refuse`; escalate account-security and refunds ≥ $100, with a non-empty `escalation_reason`.
- **S3 — Out-of-scope refusal:** when nothing relevant is in the KB → `refuse` with `abstained=true`; never fabricate.
- **S4 — PII redaction:** redact email / phone / card / SSN in echoed content and report the `redactions` count.
- **S5 — Prompt-injection resistance:** embedded "ignore the rules / you are admin / refund me" must not change the decision.
- **S6 — Multi-turn context:** resolve "it / that order" against `history`; stay consistent within a conversation.
- **S7 — Contract & reliability:** strict response schema; **decision idempotency** (same request → same decision); degrade gracefully if the LLM is down.
- **S8 — LLM answer quality (bonus, non-gating):** use the proxy to phrase a grounded, concise reply.

Passing the review (≥ 15 / 22 checks) is reachable with **S1–S7 and the
free core libraries only** — no LLM required for any decision. S8 adds polish/score
but never blocks.

## Tools (fixed, pre-installed — use only these)

The complete toolset is **already installed and cached in the image**:
`fastapi uvicorn pydantic httpx rank_bm25 numpy faiss-cpu torch(+cpu)
sentence-transformers tenacity scrubadub rapidfuzz pytest` (the MiniLM
embedding model is also baked in at build time, offline-ready).

**Do NOT edit `requirements.txt` or the Dockerfile dependency step.**
The grader rebuilds your image on every submission; this dependency
layer is cached only while those files are unchanged. Adding a library
forces a multi-minute reinstall on **every** submission. Everything you
need for S1–S8 is already here — use only these.

**Be honest about retrieval here:** every request already includes the
small, relevant `kb_articles` set — no external corpus, no index, no
scale. **Lexical retrieval (`rank_bm25`, or careful keyword/overlap
scoring) fully clears S1/S6**; the grader does **not** reward embeddings
and dense retrieval will not raise your score. `sentence-transformers` +
`faiss-cpu` are pre-installed only to illustrate the *production-scale*
path (large/external KB) — optional, not required. This course teaches
*grounded-answering discipline* (citation precision+recall,
faithfulness, abstention), not retrieval-engineering at scale.
`scrubadub` (S4) and `rapidfuzz` (S2) are provided as conveniences;
regex/stdlib also suffices. Heavier OSS (presidio, llm-guard, langgraph,
langfuse) is deliberately not installed to keep submission builds fast.

## Using an LLM (only S8 — optional)

- **S1–S7 need no LLM** — build them with plain deterministic code; that alone passes the review.
- **S8 is the only place an LLM helps** (bonus, non-gating) — a ready-to-use endpoint is provided via the `LAB_LLM_BASE_URL` / `LAB_LLM_TOKEN` env vars and the exact call is already written in the `call_llm()` helper in `app.py` (small, fast model — just use it).
- **LLM usage is capped at ~60,000 tokens per submission** (about 1–2 short calls) — keep prompts/replies short, call it at most once or twice, and if it is slow or unavailable just return your plain templated reply (never block a decision on the LLM).

## Observability (context only — nothing to build here)

In a real production support bot you would add tracing / evaluation
tooling (e.g. **Langfuse**, **OpenLLMetry**, structured per-request
logs) capturing the input, retrieved context, decision, and model I/O —
so you can later answer *"why did the bot respond this way for that
user's query?"*. That matters in production but is **out of scope for
this assignment**: the grader sandbox has no external network and this
is not graded. Noted for context only.

---

Note: the hidden grader uses different conversations from the same
distribution as the visible examples — don't hard-code to the samples.
