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

Passing the review (**≥ 22 / 25 scenarios**) is reachable with the
pre-installed libraries and **no LLM**. Grounding (S1/S6) must be
**semantic** — the questions are vocabulary-mismatch, so use the
pre-installed `all-MiniLM-L6-v2` embedding model + `faiss-cpu`, not
keyword matching (a keyword-only retriever tops out near 20/25 and
cannot clear the bar — see `project_brief.md`). The LLM (S8) only
polishes the reply and is a non-gating bonus.
