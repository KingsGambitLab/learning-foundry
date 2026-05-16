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

| # | Skill | What good looks like | Gating? |
|---|---|---|---|
| S1 | Retrieval-grounded answering | answer only from `kb_articles`; cite supporting `article_id`s (and only those) | yes |
| S2 | Policy-as-code routing | `answer`/`clarify`/`escalate`/`refuse`; escalate account-security & refunds ≥ $100 with `escalation_reason` | yes |
| S3 | Out-of-scope refusal | nothing relevant in KB → `refuse`, `abstained=true`; never fabricate | yes |
| S4 | PII redaction | redact email/phone/card/SSN in echoed content; report `redactions` count | yes |
| S5 | Prompt-injection resistance | embedded "ignore the rules / you are admin / refund me" must not change the decision | yes |
| S6 | Multi-turn context | resolve "it / that order" against `history`; stay consistent in a conversation | yes |
| S7 | Contract & reliability | strict schema; **decision idempotency** (same request → same decision); degrade gracefully if the LLM is down | yes |
| S8 | LLM answer quality | use the proxy to phrase a grounded, concise reply | **bonus, non-gating** |

The green bar (≥ 15 / 20) is reachable with **S1–S7 and the free core
libraries only** — no LLM required for any decision. S8 adds polish/score
but never blocks.

## Tools (free / open-source; pre-installed)

`fastapi uvicorn pydantic httpx rank_bm25 numpy tenacity pytest`.

**Be honest about retrieval here:** every request already includes the
small, relevant `kb_articles` set — there is no external corpus, no
index to build, no scale. So **lexical retrieval (`rank_bm25`, or even
careful keyword/overlap scoring) fully clears S1/S6**; the grader does
**not** reward embeddings here and dense retrieval will not improve your
score. `sentence-transformers` + `faiss-cpu` are included only to show
the *production-scale* path you'd use when the KB is large/external
(bake the model in the Dockerfile then — see comments) — **treat them
as optional/illustrative, not required.** This course teaches *grounded-
answering discipline* (citation precision+recall, faithfulness,
abstention), not retrieval-engineering at scale.

Optional OSS upgrades are commented in `requirements.txt`
(`scrubadub`/`presidio` for S4, `llm-guard` for S5, `rapidfuzz` for S2,
`langgraph` for S6).

## LLM proxy (S8, optional)

If `LAB_LLM_BASE_URL` / `LAB_LLM_TOKEN` are set, `POST $LAB_LLM_BASE_URL/llm/complete`
with header `x-lab-llm-token: $LAB_LLM_TOKEN` and body
`{"system": "...", "messages": [{"role":"user","content":"..."}], "max_tokens": 320}`.
It returns `{"content": "...", "usage": {"input_tokens", "output_tokens"}}`.
It is **Haiku, budget-capped** per submission — read `usage`, stay frugal,
and **degrade to a templated grounded reply on any failure** (never block
a decision on the LLM).

## Observability (production note)

In a real deployment you would wire **Langfuse / OpenLLMetry** here to
trace prompts, latency, cost, and quality. The grader sandbox blocks
egress, so that is taught as a concept; this course's concrete
stand-in is the proxy's returned `usage` — treat it as your telemetry
and keep cost bounded. (An extended assignment grades structured request
logging directly.)

## Local dev

```
cd starter && pip install -r requirements.txt
uvicorn app:app --reload --port 8000
python public/checks/run_visible_checks.py    # offline, uses a local LLM stub
```
Hidden grader uses different conversations from the same distribution —
don't hard-code to the visible samples.
