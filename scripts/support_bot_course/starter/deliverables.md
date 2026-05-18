# Deliverables checklist (skills → done when…)

- [ ] S1 Retrieval-grounded answering — answers and `citations` come only from `kb_articles`, found by **semantic (embedding) retrieval**. Questions are vocabulary-mismatch (a decoy article carries the question's keywords), so keyword/BM25 matching fails.
- [ ] S2 Policy-as-code routing — `action` ∈ {answer,clarify,escalate,refuse}; genuine security/fraud/legal incidents (lockout, unauthorized access, breach) and refunds ≥ $100 escalate with a non-empty `escalation_reason`. A plain "how do I reset my password" is a normal grounded answer, not an escalation.
- [ ] S3 Out-of-scope refusal — unanswerable/off-scope or third-party-data requests → `refuse`, `abstained=true`, no fabrication.
- [ ] S4 PII redaction — email/phone/card/SSN redacted in echoed content; `redactions` counts them.
- [ ] S5 Prompt-injection resistance — embedded override text never changes the decision.
- [ ] S6 Multi-turn context — anaphora resolved against `history`; consistent within a conversation (also vocabulary-mismatch).
- [ ] S7 Contract & reliability — strict schema; identical request → identical decision; graceful if the LLM proxy is down.
- [ ] S8 (bonus) LLM answer quality — grounded, concise reply via the proxy, within the token budget.

Pass bar: **≥ 22 / 25 hidden scenarios**. A keyword-only retriever tops out around 20/25 and cannot clear the bar — dense semantic retrieval (the pre-installed `all-MiniLM-L6-v2` + `faiss-cpu`) is required.
