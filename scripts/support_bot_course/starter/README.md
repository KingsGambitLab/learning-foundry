# Customer Support Bot — starter

Implement `POST /support/answer` as described in `project_brief.md`.
Read the brief and `deliverables.md` first.

## :vscode: Iterate in this workspace with the visible checks

Your code lives in `public/starter/`. Run everything from there:

- `cd public/starter`
- `pip install -r requirements.txt`
- `uvicorn app:app --reload --port 8000` — run your service
- `python public/checks/run_visible_checks.py` — offline self-check (uses a local LLM stub)

The visible checks are a **small subset of the real review run**. The
hidden grader uses different conversations from the same distribution —
make your solution general; do not hard-code to the visible samples.

## Submit

Submit the whole project to run the full learner review checks. You get
feedback on **what works and what to improve**, useful to iterate — and
you can **submit as many times as you need** to solve the assignment.
Use the visible checks above to iterate fast locally between submissions.

The green bar (≥ 15 / 22) is reachable with the pre-installed core
libraries and **no LLM**. The LLM (S8) only polishes the reply and is a
non-gating bonus — see `project_brief.md`.
