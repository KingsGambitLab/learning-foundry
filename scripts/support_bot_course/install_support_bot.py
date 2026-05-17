"""Install / re-publish the Customer Support Bot outcome course on staging.

Clones the known-good course_wikiqa_v1 schema and mutates it. Grader
bundle already on disk at outcome_workspaces/course_support_bot.

Idempotent + safe to re-run: it also REPINS every active enrollment to
the freshly published snapshot (and refreshes the non-learner docs in
any already-seeded workspace) so a re-publish never strands learners on
a stale snapshot — that bug ("README not updated") is what this block
prevents. Learner-authored files (app.py etc.) are never touched.
"""
import json, mimetypes, pathlib, time
from app.storage.postgres_store import PostgresWorkflowStore
from app.domain.publish import LearnerPackageFile
from app.domain.task_agent import LearnerDeliverableBrief
from app.domain.course import CourseRunStatus

st = PostgresWorkflowStore()
CID = "course_support_bot"
ROOT = "/opt/course-gen-codex/outcome_workspaces/course_support_bot"
STARTER = pathlib.Path(ROOT) / "public" / "starter"
TITLE = "Customer Support Bot: Grounded Answers, Policy Routing, PII & Injection Defense"
SUMMARY = ("Build a production multi-turn SaaS customer-support bot: dense-semantic "
           "retrieval-grounded answers with citations, policy-as-code routing, PII "
           "redaction, prompt-injection resistance, multi-turn context, and a strict "
           "API contract. Graded on 25 hidden scenarios (pass ≥ 22/25); keyword "
           "retrieval cannot clear the bar — semantic embedding retrieval is "
           "required. An optional budget-capped LLM proxy polishes replies.")

LEARNER_BRIEF = LearnerDeliverableBrief(
    why_this_deliverable_matters=(
        "Production support bots must answer ONLY from the supplied help docs, "
        "route by policy, redact PII, resist prompt injection, and hold multi-turn "
        "context — and still find the right doc when the user's words don't "
        "match it. This builds exactly that, end to end."
    ),
    task_to_build=(
        "Implement POST /support/answer (FastAPI) returning "
        "{reply, action, citations, redactions, abstained, escalation_reason?}, "
        "grounding answers via dense semantic retrieval over the per-request "
        "kb_articles."
    ),
    files_to_edit=["public/starter/app.py"],
    definition_of_done=[
        "≥ 22 / 25 hidden scenarios pass.",
        "citations come only from kb_articles and name the semantically-supporting "
        "article — keyword/BM25 fails the vocabulary-mismatch scenarios.",
        "action ∈ {answer,clarify,escalate,refuse}; security/fraud/legal "
        "incidents and refunds ≥ $100 escalate with a non-empty "
        "escalation_reason; a plain password-reset how-to is a normal answer.",
        "Out-of-scope / unsupported / third-party-data → refuse with "
        "abstained=true; never fabricate.",
        "email/phone/card/SSN redacted in echoed content; redactions counts them.",
        "Embedded injection/override text never changes the decision.",
        "Anaphora resolved against history; identical request → identical "
        "decision; degrades gracefully if the LLM proxy is down.",
    ],
    example_scenarios=[
        "A paraphrased question whose answering article shares almost no words "
        "with it (a decoy article carries the keywords) → must cite the "
        "semantically-correct article.",
        "Refund of $480 → escalate with reason; refund under $100 → a "
        "grounded answer.",
        "Off-scope (weather/recipe) or 'send me another customer's data' → "
        "refuse, abstained=true.",
    ],
    implementation_hints=[
        "Retrieve with the pre-installed, pinned "
        "sentence-transformers/all-MiniLM-L6-v2 + faiss-cpu (IndexFlatIP over "
        "L2-normalized vectors); use a cosine floor to drive abstention. A "
        "keyword/BM25 ranker cannot clear the pass bar.",
        "Keep routing / PII / injection / abstention deterministic; the LLM (S8) "
        "only polishes `reply` and is optional and non-gating.",
    ],
    non_goals=[
        "No external corpus or retrieval-at-scale — kb_articles is small and "
        "supplied per request.",
        "No LLM in any graded decision; observability tooling is context-only.",
    ],
)

tmpl_run = st.get_course_run("course_wikiqa_v1")
tmpl_snap = st.get_publish_snapshot(tmpl_run.latest_publish_snapshot_id)

# ---- build seed files from the starter tree ----
seed = []
for p in sorted(STARTER.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(STARTER).as_posix()
    if rel in ("README.md", "project_brief.md", "deliverables.md"):
        continue  # written from snapshot markdown by seed_workspace_from_snapshot
    seed.append(LearnerPackageFile(
        relative_path=f"public/starter/{rel}",
        media_type=mimetypes.guess_type(rel)[0] or "text/plain",
        content=p.read_text(),
    ))
brief = (STARTER / "project_brief.md").read_text()
readme = (STARTER / "README.md").read_text()

# ---- mutate the cloned snapshot ----
snap = tmpl_snap.model_copy(deep=True)
snap.id = f"publish_support_bot_{int(time.time())}"
snap.course_run_id = CID
snap.task_agent_spec = None
lp = snap.learner_package
lp.course_run_id = CID
lp.title = TITLE
lp.summary = SUMMARY
lp.project_brief_markdown = brief
lp.deliverables = [lp.deliverables[0]]
d = lp.deliverables[0]
d.title = TITLE
d.objective = SUMMARY
d.content_markdown = brief
d.starter_readme = readme
d.learner_brief = LEARNER_BRIEF  # was the stale Wikipedia-QA template brief
d.workspace_seed_files = seed
d.visible_files = [
    "public/starter/app.py", "public/starter/requirements.txt",
    "public/starter/Dockerfile", "public/starter/public/examples/sample_conversations.json",
    "public/starter/public/checks/run_visible_checks.py",
]
st.save_publish_snapshot(snap)

# ---- mutate the cloned course run ----
run = tmpl_run.model_copy(deep=True)
run.id = CID
run.title = TITLE
run.summary = SUMMARY
run.shared_workflow_run_id = CID
run.course_family_id = CID
run.lab_tutor_enabled = True
# Force published — the template (course_wikiqa_v1) is hidden
# (status=active), so a deep-copy would otherwise un-publish the
# Support Bot and drop it from the catalog.
run.status = CourseRunStatus.published
run.latest_publish_snapshot_id = snap.id
pj = dict(run.payload_json or {})
pj.setdefault("outcome_state", {})
pj["outcome_state"] = dict(pj.get("outcome_state") or {})
pj["outcome_state"]["workspace_root"] = ROOT
run.payload_json = pj
st.save_course_run(run)

# ---- repin every active enrollment to the new snapshot ----
# Without this, learners stay pinned to whatever snapshot they enrolled
# under and see the OLD brief/README while being graded on the NEW
# dataset. Back up each enrollment row before mutating; refresh the
# non-learner docs in any already-seeded workspace (never touch
# learner-authored files).
from app.services.learner_package_runtime import readme_markdown

BACKUP = pathlib.Path("/opt/course-gen-codex/tmp") / f"support_bot_enrollments.bak.{int(time.time())}.json"
WS_BASE = pathlib.Path("/opt/course-gen-codex/learner_workspaces")
summaries = st.list_learner_enrollments(limit=500)
repinned, refreshed, backup_rows = [], [], []
readme_md = readme_markdown(snap)  # single consolidated learner doc
for s in summaries:
    if getattr(s, "course_run_id", None) != CID:
        continue
    e = st.get_learner_enrollment(s.id)
    if e is None or str(getattr(e, "status", "")).endswith("archived"):
        continue
    backup_rows.append(json.loads(e.model_dump_json()))
    if e.publish_snapshot_id != snap.id:
        e.publish_snapshot_id = snap.id
        st.save_learner_enrollment(e)
        repinned.append(e.id)
    # refresh non-learner docs if this workspace was already seeded
    try:
        ws = WS_BASE / e.learner_id / e.shared_workflow_run_id / "workspace"
        if (ws / ".coursegen" / "workspace_seeded.txt").exists():
            (ws / "README.md").write_text(readme_md, encoding="utf-8")
            # Single-README consolidation: remove the now-retired dup
            # files from already-seeded workspaces (learner code, e.g.
            # public/starter/app.py, is never touched).
            for stale in ("project_brief.md", "deliverables.md"):
                try:
                    (ws / stale).unlink()
                except FileNotFoundError:
                    pass
            (ws / ".coursegen" / "workspace_seeded.txt").write_text(snap.id + "\n", encoding="utf-8")
            refreshed.append(e.id)
    except Exception as exc:  # best-effort; never fail the publish on this
        print("  ws refresh skipped for", e.id, ":", exc)
if backup_rows:
    BACKUP.write_text(json.dumps(backup_rows, indent=2))

# ---- verify ----
chk = st.get_course_run(CID)
csnap = st.get_publish_snapshot(chk.latest_publish_snapshot_id)
cd = csnap.learner_package.deliverables[0]
print("course:", chk.id, "| status:", chk.status, "| snap:", csnap.id)
print("summary 25-scenario:", "25 hidden" in (csnap.learner_package.summary or ""))
print("brief 22/25:", "22 / 25" in (csnap.learner_package.project_brief_markdown or ""))
print("learner_brief task:", (cd.learner_brief.task_to_build or "")[:60] if cd.learner_brief else None)
print("seed files:", len(cd.workspace_seed_files), "| enrollments backed up:", len(backup_rows), "->", BACKUP.name)
print("repinned:", repinned or "none", "| ws-docs refreshed:", refreshed or "none")
print("grader scenarios on disk:",
      len(list(pathlib.Path(ROOT, "private/grader/scenarios").glob("*.yaml"))))
