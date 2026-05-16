# Customer Support Bot — starter

Implement `POST /support/answer` as described in `project_brief.md`.
Read the brief and `deliverables.md` first.

## Iterate in this workspace with the visible checks

Run the visible checks inside this workspace and iterate locally before
you submit:

- `pip install -r requirements.txt`
- `uvicorn app:app --reload --port 8000` — run your service
- `python public/checks/run_visible_checks.py` — offline self-check (uses a local LLM stub)

The visible checks are a **small subset of the real review run**. The
hidden grader uses different conversations from the same distribution —
make your solution general; do not hard-code to the visible samples.

The green bar (≥ 15 / 22) is reachable with the pre-installed core
libraries and **no LLM**. The LLM (S8) only polishes the reply and is a
non-gating bonus — see `project_brief.md`.
