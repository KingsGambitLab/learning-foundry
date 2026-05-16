# Customer Support Bot — starter

Implement `POST /support/answer` per `project_brief.md`. Read the brief +
`deliverables.md` first.

Quickstart:
```
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
python public/checks/run_visible_checks.py   # offline self-check (+ local LLM stub)
```
Green ≥15/20 is reachable with the free core libs and **no LLM**. The LLM
proxy (S8) only polishes the reply and is non-gating. Submit from the LMS;
the hidden grader uses different conversations from the same distribution.
