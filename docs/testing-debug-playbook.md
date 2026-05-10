# Testing And Debug Playbook

This playbook is for fast local debugging of the course-generation pipeline without losing the architectural rules we just fought for.

Core rule:

> strong harness, dumb everything else

That means:

- creator owns the exact stack contract
- authoring owns the spec, repo files, and runtime protocol
- harness executes and judges
- retries must use grounded failure facts, not prose guessing

And just as important:

- do **not** add language-specific starter/compiler logic
- do **not** use string matching or regex heuristics to decide support or execution
- do **not** parse raw JSON text from LLM output; use structured outputs at the boundary

## 1. Quick start

Start the app:

```bash
cd /Users/tushar/Desktop/codebases/course-gen-codex
source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

App URL:

- [http://127.0.0.1:8010](http://127.0.0.1:8010)

Tail the main log:

```bash
cd /Users/tushar/Desktop/codebases/course-gen-codex
tail -f logs/course_generation.log
```

Useful grep:

```bash
rg -n "run_<WORKFLOW_ID>|course_<COURSE_ID>" logs/course_generation.log
```

## 2. Use the cheapest loop first

When debugging, use this order:

1. focused tests
2. replay smoke on the last failure
3. full course rerun

Do **not** jump straight to a fresh full run unless the first two are already green or unhelpful.

### Focused tests

Common slices:

```bash
uv run --with pytest pytest -q tests/test_authoring_resilience.py tests/test_authoring_payloads.py
```

```bash
uv run --with pytest pytest -q tests/test_task_agent_retry_service.py tests/test_generated_test_loop.py
```

```bash
uv run --with pytest pytest -q tests/test_creator_stack_contract.py tests/test_course_workflow_runtime_context.py
```

### Replay smoke

Replay the last actionable failure against the current code, without mutating real workflow state:

```bash
source .venv/bin/activate
python scripts/replay_failure_smoke.py --workflow run_<WORKFLOW_ID>
```

If you want repo repair included before smoke:

```bash
source .venv/bin/activate
python scripts/replay_failure_smoke.py --workflow run_<WORKFLOW_ID> --repair
```

You can also resolve from a course id:

```bash
source .venv/bin/activate
python scripts/replay_failure_smoke.py --course course_<COURSE_ID> --repair
```

What replay smoke is good for:

- verifying a harness fix cheaply
- checking whether a new failure packet is actually better
- avoiding another full expensive course run

What it is **not**:

- a production-loop stop policy
- a substitute for full end-to-end verification

## 3. When to run a full course

Run a full course when one of these is true:

- replay smoke passes
- the fix changed early authoring behavior and replay smoke cannot cover it
- you need to verify creator flow, timeline, and linked workflow behavior together

For stack-contract-driven reruns, prefer the creator-plan path over ad hoc DB edits.

## 4. Rust rerun recipe

This is the exact contract that got us the most useful recent signal:

- language: `rust`
- language version: `1.95`
- framework: `axum`
- framework version: `0.8.9`
- package manager: `cargo`
- primary database: `postgres`
- primary database version: `18`
- cache backend: `redis`
- cache backend version: `8`

Quick local launch through the API:

```bash
source .venv/bin/activate
python - <<'PY'
import httpx

base = "http://127.0.0.1:8010"

plan_payload = {
    "goal": "Build a production routing and escalation service in Rust with Axum, Postgres, and Redis.",
    "learning_outcomes": [
        "Implement a durable routing and escalation flow with explicit workflow state.",
        "Keep reads coherent while using Postgres for source of truth and Redis for fast paths.",
        "Add operational hardening with health, observability, and recovery-oriented checks.",
    ],
    "creator_choices": {
        "starter_type": "partial_implementation",
        "implementation_language": "rust",
        "language_version": "1.95",
        "application_framework": "axum",
        "framework_version": "0.8.9",
        "package_manager": "cargo",
        "primary_database": "postgres",
        "primary_database_version": "18",
        "cache_backend": "redis",
        "cache_backend_version": "8",
        "tech_stack": [],
        "data_sources": [],
    },
}

client = httpx.Client(timeout=120)
planned = client.post(f"{base}/v1/course-generation/creator-plan", json=plan_payload).json()
created = client.post(
    f"{base}/v1/course-runs/from-creator-plan-async",
    json={"plan": planned["plan"]},
).json()
print(created["course_run"]["id"])
PY
```

Then poll:

```bash
source .venv/bin/activate
python - <<'PY'
import httpx, json, time

base = "http://127.0.0.1:8010"
course_run_id = "course_<REPLACE_ME>"
client = httpx.Client(timeout=60)

while True:
    body = client.get(f"{base}/v1/course-runs/{course_run_id}/creator-view").json()
    print(json.dumps({
        "stage": body["course_run"]["stage"],
        "status": body["course_run"]["status"],
        "shared_workflow_run_id": body["course_run"].get("shared_workflow_run_id"),
        "diagnostic_codes": [d["code"] for d in body.get("diagnostics", [])],
    }, indent=2))
    time.sleep(5)
PY
```

## 5. How to read the log

### Spec customization / authoring

Look for:

- `task_agent_authoring_generate_started`
- `task_agent_authoring_customization_rejected`
- `task_agent_authoring_customization_preflight_retry`
- `workflow_authoring_scaffold_generated`

Interpretation:

- `customization_rejected` + no retry means old bad behavior
- `customization_rejected` + `preflight_retry` means the cheap loop is working
- `origin_template: generic_backend_service` with `source: deterministic_fallback` means authoring gave up and fell back
- `origin_template: openai_customized:...` means we stayed on the authored path

### Repo authoring

Look for:

- `workspace_repo_authoring_attempt_started`
- `workspace_repo_authoring_attempt_failed`
- `workspace_repo_authoring_deliverable_completed`

Interpretation:

- one failed attempt followed by a successful second attempt is acceptable if it converges quickly
- repeated timeout failures are a boundary/tooling issue, not a “just rerun it” issue

### Runtime

Look for:

- `sandbox_dependency_contract_materialized`
- `sandbox_deliverable_healthcheck_wait_started`
- `sandbox_deliverable_completed`
- `authoring_runtime_sandbox_completed`

Interpretation:

- if `sandbox_dependency_contract_materialized` fails, the issue is in dependency contract / install protocol
- if build succeeds but health never comes up, the issue is boot/runtime protocol or app correctness
- if public checks fail after healthy boot, the issue is app behavior or authored visible checks

### Tests

Look for:

- `workspace_test_authoring_attempt_started`
- `workspace_test_authoring_deliverable_completed`
- `reviewer_tests`

Interpretation:

- if runtime is green but spend explodes here, test authoring is the long pole

## 6. Ownership guide

Use this to decide where to fix.

### A. Spec / customization problem

Symptoms:

- `task_agent_authoring_customization_rejected`
- invalid public endpoints
- invalid public-check paths
- generic or broken learner surface before repo authoring starts

Fix in:

- `app/services/openai_task_agent_authoring.py`
- deterministic spec validation feedback

### B. Repo authoring problem

Symptoms:

- repo authoring times out
- authored files are missing or incoherent
- one deliverable diverges from the rest after a good spec

Fix in:

- `app/services/openai_repo_authoring.py`
- repo-authoring prompt contract
- hard-timeout boundary if requests are hanging too long

### C. Runtime/dependency-contract problem

Symptoms:

- lockfile/manifest mismatch
- toolchain mismatch
- install/build succeeds only after mutating dependency state unexpectedly

Fix in:

- `app/services/dependency_contract_materializer.py`
- authored `Dockerfile`
- authored `.coursegen/runtime/install.sh`

### D. Harness problem

Symptoms:

- authored Dockerfile exists but harness ignores it
- wrong runtime image is used
- replay smoke or runtime runner contradicts authored bundle

Fix in:

- `app/services/docker_sandbox_runner.py`
- `app/services/learner_studio_service.py`
- `app/services/failure_replay_smoke.py`

### E. Test-authoring problem

Symptoms:

- runtime is green
- hidden tests are weak, flaky, or too expensive to iterate

Fix in:

- `app/services/openai_test_script_authoring.py`
- baseline verifier and test-review loop

## 7. How to stop a wedged run

There is no nice cancel API yet. If a run is clearly wedged and you need to stop spend, do it explicitly.

First stop the local app process if it is actively driving background work.

Then mark both course and workflow blocked:

```bash
source .venv/bin/activate
python - <<'PY'
from datetime import UTC, datetime
from app.domain.course import CourseRunStage, CourseRunStatus
from app.domain.workflow import WorkflowStage, WorkflowStatus
from app.storage.sqlite_store import SQLiteWorkflowStore

course_run_id = "course_<COURSE_ID>"
workflow_run_id = "run_<WORKFLOW_ID>"
reason = "Stopped manually during debugging."

store = SQLiteWorkflowStore()
course = store.get_course_run(course_run_id)
run = store.get_run(workflow_run_id)
now = datetime.now(UTC)

course.stage = CourseRunStage.blocked
course.status = CourseRunStatus.blocked
course.active_operation = None
course.updated_at = now
course.last_error = reason
course.notes = list(dict.fromkeys([*course.notes, reason]))
store.save_course_run(course)
store.append_course_event(course.id, "course_generation_stopped", {
    "reason": reason,
    "shared_workflow_run_id": workflow_run_id,
})

run.stage = WorkflowStage.blocked
run.status = WorkflowStatus.blocked
run.pending_gate = None
run.updated_at = now
run.artifacts.notes = list(dict.fromkeys([*run.artifacts.notes, reason]))
store.save_run(run)
store.append_event(run.id, "workflow_stopped", {"reason": reason})
PY
```

Then restart the app cleanly.

## 8. Worked example: what preflight changed on Rust

Previous Rust run:

- run: `run_b77f09cca50b`
- invalid customization immediately fell back to deterministic scaffold
- stored origin: `generic_backend_service`

New Rust run with preflight:

- run: `run_9f7c446c03d5`
- first customization was rejected with:
  - `missing_health_endpoint`
  - `public_check_path_not_published`
- preflight fed those errors back and retried
- stored origin became:
  - `openai_customized:generic_backend_service`

Visible effect:

- old deliverables were generic:
  - `And Escalation ...`
- new deliverables became grounded:
  - `Create and fetch escalation records`
  - `Approve escalations and persist state`
  - `Expose escalation trace and failure details`

So preflight already paid for itself: it prevented cheap invalid customization from immediately collapsing into deterministic fallback.

## 9. What to do next when picking this up

Recommended order:

1. replay smoke on the last failure if the fix is local and targeted
2. if green, rerun the full course
3. if rerun fails, classify the owner using the guide above
4. stop the run once the next root cause is clear

Current next likely bottlenecks after the Rust preflight win:

- repo-authoring timeout boundary
- downstream authored repo/runtime consistency for individual deliverables
- test authoring cost and iteration speed
