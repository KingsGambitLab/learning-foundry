# Deliverables checklist (skills → done when…)

- [ ] S1 Retrieval-grounded answering — answers come only from `kb_articles`; `citations` are supporting `article_id`s and nothing else.
- [ ] S2 Policy-as-code routing — `action` ∈ {answer,clarify,escalate,refuse}; account-security & refunds ≥ $100 escalate with `escalation_reason`.
- [ ] S3 Out-of-scope refusal — unanswerable/off-scope → `refuse`, `abstained=true`, no fabrication.
- [ ] S4 PII redaction — email/phone/card/SSN redacted in echoed content; `redactions` counts them.
- [ ] S5 Prompt-injection resistance — embedded override text never changes the decision.
- [ ] S6 Multi-turn context — anaphora resolved against `history`; consistent within a conversation.
- [ ] S7 Contract & reliability — strict schema; identical request → identical decision; graceful if the LLM proxy is down.
- [ ] S8 (bonus) LLM answer quality — grounded, concise reply via the proxy, within the token budget.
