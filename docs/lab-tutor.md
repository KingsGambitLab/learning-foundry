# Lab Tutor

## What it is

The Lab Tutor is an AI tutor embedded inside cloud-hosted VS Code (code-server) instances for learners doing graded coding assignments. Phase 1 ships the wiring skeleton: a VS Code extension (sidebar chat UI, status bar, and submit command), a FastAPI `/v1/tutor/*` backend with canned-response stubs, and the launch-time plumbing that ties them together. Real reactive coaching, Copilot hook integration, and viva orchestration land in later phases. The core contract is that the tutor is either fully present — extension loaded, env vars set, backend reachable — or completely absent; there is no partial state. The pedagogical spirit is "coach the prompting moment, don't replace the agent."

---

## Architecture

```
Learner browser
   └─ code-server container (Docker)
       ├─ code-server + VS Code workbench
       ├─ Lab Tutor extension  (pre-installed at /opt/lab-tutor/extensions,
       │   loaded via --extensions-dir when lab_tutor_enabled=True)
       │     ├─ Sidebar webview  (chat UI — labTutor.chat view)
       │     ├─ Status bar item  (state label + hint budget, links to lab.openTutor)
       │     └─ Submit command   (lab.submitAssignment — posts code snapshot to /submit)
       └─ Container env: LAB_TUTOR_BASE_URL, LAB_TUTOR_SESSION_ID
                         (injected only when lab_tutor_enabled=True)

   ↕ HTTP  (base URL from LAB_TUTOR_BASE_URL)

Backend (FastAPI, app/main.py)
   └─ /v1/tutor/*
       ├─ POST /chat    (TutorChatRequest → TutorChatResponse)  — currently canned
       └─ POST /submit  (TutorSubmitRequest → TutorSubmitResponse) — currently canned
```

### Extension layer

- At activation (`onStartupFinished`) the extension reads `process.env.LAB_TUTOR_BASE_URL` and `process.env.LAB_TUTOR_SESSION_ID`. If either is missing it falls back to the VS Code settings `labTutor.baseUrl` / `labTutor.sessionId`, then to hardcoded defaults (`http://localhost:8000` / `dev-session`). This means the extension will activate in any workspace — the env vars are what restrict it to tutor-enabled sessions.
- A `TutorClient` (`extensions/lab-tutor/src/services/tutor-client.ts`) wraps the two backend endpoints with typed `fetch` calls. Both requests carry `session_id` in the JSON body.
- The sidebar (`TutorSidebarProvider`) renders `extensions/lab-tutor/media/sidebar.html` as a webview with a simple message/reply UI. User messages are forwarded to `TutorClient.chat`; replies are posted back to the webview.
- The status bar (`TutorStatusBar`) shows four states — `watching`, `coaching`, `idle`, `reviewing` — appended with the current hint budget label (e.g., "Hints: 3/4"). It is backed by a `HintBudget` instance initialized with capacity 4.
- `lab.submitAssignment` captures the active editor's full text, posts it to `TutorClient.submit`, and on success shows a viva-question popup (`popup.ts`). The popup opens in a split view beside the editor with the viva questions rendered as numbered HTML list items. Failures surface as VS Code error notifications. (The viva recording UI is a Phase 4 item; the popup panel currently has no input.)

### Backend service layer

- Router mounted at `/v1/tutor` (`app/api/tutor.py`); two endpoints: `POST /chat` and `POST /submit`.
- `TutorService` (`app/services/tutor_service.py`) is the Phase 1 stub. For `/chat` it echoes the first 80 characters of the message back with a `(stub)` prefix and `hint_tier=None`. For `/submit` it returns `{"passed": True, "details": "stub"}` as `test_results` and two hard-coded viva questions.
- Domain types live in `app/domain/tutor.py`:
  - `TutorChatRequest` — `session_id`, `message`
  - `TutorChatResponse` — `reply`, `hint_tier: int | None` (Phase 2 budget authority hook)
  - `TutorSubmitRequest` — `session_id`, `code_snapshot`
  - `TutorSubmitResponse` — `test_results: dict`, `viva_questions: list[TutorVivaQuestion]`
- No authentication on these routes in Phase 1. Any client that can reach the process can call them.

### Launcher layer

- `LearnerStudioService.launch_editor` (`app/services/learner_studio_service.py`) receives `lab_tutor_enabled: bool` from the caller.
- When `True`: `_tutor_environment(session_id)` returns `{"LAB_TUTOR_BASE_URL": ..., "LAB_TUTOR_SESSION_ID": ...}` and those are added to the `docker run` `-e` args; `--extensions-dir /opt/lab-tutor/extensions` is appended to the code-server invocation.
- When `False`: neither the env vars nor `--extensions-dir` appear. The extension directory in the image is not referenced, so code-server loads no Lab Tutor extension.
- The image tag is computed from the Dockerfile and all source files under `extensions/lab-tutor/` (excluding `node_modules`, `dist`, `out`, `test-out`, and `.vsix` files). A SHA-1 over sorted file paths and contents produces a 12-character hex tag such as `course-gen-learner-studio:a3f9c1e82d40`. Changing any extension source file therefore triggers a fresh image build on next launch, with no manual rebuild step required.
- Toggle-reuse guard: if an existing session's `lab_tutor_enabled` differs from the requested value, the launcher tears down the stale container before starting a new one. This prevents a learner from seeing a mismatched tutor state after an admin changes the setting.

---

## The on/off toggle

### Where it lives

- `CourseRun.lab_tutor_enabled: bool` (default `False`), defined in `app/domain/course.py`.
- `LearnerWorkspaceSession.lab_tutor_enabled: bool` (default `False`), defined in `app/domain/learner.py`. This records the state at the moment the session was launched so it can be compared on the next launch.
- Both fields are persisted inside `payload_json` TEXT columns in SQLite. The course-run value lives in `course_runs.payload_json`; the session value in `learner_workspace_sessions.payload_json`.
- Scope: per CourseRun (i.e., per assignment). All learners enrolled in the same assignment see the same value simultaneously.

### Default

`False`. Existing assignments and newly-created assignments are OFF until explicitly enabled. Async generation rebuilds do not reset this value (see below).

### How to flip it

The setting flows through standard CourseRun update paths. Phase 1 has no dedicated admin endpoint; update via the existing CourseRun mutation API, or operate directly on the database for one-off ops work.

Raw SQL approach (SQLite syntax, safe to test the read first):

```sql
-- Read the current value for a course run
SELECT json_extract(payload_json, '$.lab_tutor_enabled')
FROM course_runs
WHERE course_run_id = '<your-course-run-id>';

-- Flip ON
UPDATE course_runs
SET payload_json = json_set(payload_json, '$.lab_tutor_enabled', json('true'))
WHERE course_run_id = '<your-course-run-id>';

-- Flip OFF
UPDATE course_runs
SET payload_json = json_set(payload_json, '$.lab_tutor_enabled', json('false'))
WHERE course_run_id = '<your-course-run-id>';
```

After a change, the next `launch_workspace` call for any learner in that assignment picks up the new value. No server restart is required. The value is read fresh from the store on every `launch_workspace` call.

### What OFF guarantees (the contract)

- The `docker run` command omits `LAB_TUTOR_BASE_URL` and `LAB_TUTOR_SESSION_ID` env vars entirely.
- The code-server invocation omits `--extensions-dir /opt/lab-tutor/extensions`.
- code-server loads no Lab Tutor extension — no sidebar, no status bar item, no `lab.submitAssignment` command, and no API traffic from the container.
- The pre-installed extension at `/opt/lab-tutor/extensions` inside the image is inert when not referenced via `--extensions-dir`. It does not self-activate.
- The `/v1/tutor/*` backend routes remain reachable by anyone with network access to the process, but nothing inside the container calls them.

### What OFF does NOT do immediately (and why)

An already-running editor container that was launched while `lab_tutor_enabled=True` keeps the tutor visible until the container is stopped. The toggle is read at launch time only, not enforced retroactively on live sessions.

However, session reuse refuses to reuse a mismatched container. When a learner next triggers `launch_workspace` (page reload, navigation away and back), the launcher detects `existing_session.lab_tutor_enabled != lab_tutor_enabled`, tears down the stale container, and starts a fresh one. The practical effect: the next page reload is the catch-up boundary.

If you need immediate retroactive disablement (for example, a security incident), force-stop running sessions through the existing session-stop path. This is rare and not the normal operational flow.

### Carry-through to the publish-certification gate

`PublishLearnerCertificationService` (`app/services/publish_learner_certification_service.py`) resolves `lab_tutor_enabled` from the CourseRun at publish time and passes it to `launch_editor`. This means the certification gate exercises the same runtime configuration learners will see. Regressions on tutor-enabled courses (for example, the extension failing to load) are caught at publish time, not at learner runtime.

### Async generation preservation

The async course generation path (`apply_generated_plan` in `app/services/course_workflow_service.py`) rebuilds `CourseRun` objects from scratch. The toggle is explicitly copied from the pre-rebuild run:

```python
course_run.lab_tutor_enabled = existing.lab_tutor_enabled
```

A "flip ON during a queued regeneration" therefore does not silently revert to `False`.

---

## Testing the toggle

For engineers verifying behavior locally. The SQLite store defaults to `data/store.db` at the repo root. You can confirm its location by checking `app/storage/sqlite_store.py` or the value of `STORE_PATH` in your `.claude/launch.json`.

```bash
# 1. Set the DB path
DB="$(pwd)/data/store.db"

# 2. Enable the tutor for a specific course run
sqlite3 "$DB" "UPDATE course_runs \
  SET payload_json = json_set(payload_json, '$.lab_tutor_enabled', json('true')) \
  WHERE course_run_id = '<your-course-run-id>'"

# 3. Confirm the write
sqlite3 "$DB" \
  "SELECT json_extract(payload_json, '$.lab_tutor_enabled') \
   FROM course_runs WHERE course_run_id = '<your-course-run-id>'"
# Expected output: 1

# 4. Launch the workspace through the normal LMS flow
#    (via the web UI, or via your existing test harness / API call)

# 5. Verify env vars and the extensions-dir flag reached the container
docker inspect <container-name> \
  | jq '.[0] | {Env: .Config.Env, Cmd: .Config.Cmd}'
# Expected: LAB_TUTOR_BASE_URL and LAB_TUTOR_SESSION_ID appear in Env;
#           --extensions-dir /opt/lab-tutor/extensions appears in Cmd.

# 6. Verify the extension is loaded inside the container
docker exec <container-name> \
  code-server --extensions-dir /opt/lab-tutor/extensions --list-extensions
# Expected: scaler.lab-tutor@0.1.0
```

For the OFF path, repeat step 5 — the same `docker inspect` should show no `LAB_TUTOR_*` env vars and no `--extensions-dir` flag.

To exercise the toggle-reuse guard, flip the toggle while a session is live and trigger `launch_workspace` again. Verify via `docker ps` that the old container name is gone and a fresh container appears. The new container should reflect the updated tutor setting.

To exercise the publish-certification path with the tutor enabled, ensure `lab_tutor_enabled=True` on the course run before running the cert check. The cert check's `launch_editor` call will include `--extensions-dir`, which means the extension must load cleanly for the check to pass.

---

## File map

| Concern | File |
|---|---|
| Domain model (`CourseRun.lab_tutor_enabled`) | `app/domain/course.py` |
| Domain model (`LearnerWorkspaceSession.lab_tutor_enabled`) | `app/domain/learner.py` |
| Launcher (env vars + extensions-dir + session reuse guard) | `app/services/learner_studio_service.py` |
| Caller: learner launch | `app/services/lms_service.py` — `launch_workspace` |
| Caller: publish cert | `app/services/publish_learner_certification_service.py` |
| Async-rebuild preservation | `app/services/course_workflow_service.py` — `apply_generated_plan` |
| Backend domain types | `app/domain/tutor.py` |
| Backend service (stub) | `app/services/tutor_service.py` |
| Backend router | `app/api/tutor.py` |
| Extension manifest | `extensions/lab-tutor/package.json` |
| Extension activation | `extensions/lab-tutor/src/extension.ts` |
| Extension sidebar UI | `extensions/lab-tutor/src/sidebar.ts`, `extensions/lab-tutor/media/sidebar.html` |
| Extension status bar | `extensions/lab-tutor/src/status-bar.ts` |
| Extension submit command | `extensions/lab-tutor/src/submit-command.ts` |
| Extension viva popup | `extensions/lab-tutor/src/popup.ts` |
| Extension HTTP client | `extensions/lab-tutor/src/services/tutor-client.ts` |
| Client-side hint budget | `extensions/lab-tutor/src/state/hint-budget.ts` |
| Image tag hashing | `app/services/learner_studio_service.py` — `_hash_learner_studio_inputs` |
| Docker image definition | `docker/learner-studio.Dockerfile` |

---

## Troubleshooting

### Learner reports no tutor sidebar despite the assignment being enabled

Work through this checklist in order:

1. **Confirm the DB value.** Run the read query from the SQL examples above. If it returns `0` or `NULL`, the toggle is off; flip it and have the learner relaunch.

2. **Confirm the container has the env vars.** `docker inspect <container-name> | jq '.[0].Config.Env'`. If `LAB_TUTOR_BASE_URL` and `LAB_TUTOR_SESSION_ID` are missing, the session was started when the toggle was off. The learner needs to trigger a new `launch_workspace` to get a fresh container.

3. **Confirm `--extensions-dir` is in the command.** `docker inspect <container-name> | jq '.[0].Config.Cmd'`. The array should contain `--extensions-dir` and `/opt/lab-tutor/extensions`. If missing, same root cause as step 2.

4. **Confirm the extension is installed.** Run `docker exec <container-name> code-server --extensions-dir /opt/lab-tutor/extensions --list-extensions`. If `scaler.lab-tutor` is absent, the image may have been built before the extension Dockerfile stage was added — rebuild the image by deleting the `course-gen-learner-studio:*` image tag and relaunching.

5. **Confirm the backend is reachable.** From inside the container: `docker exec <container-name> curl -s -o /dev/null -w "%{http_code}" $LAB_TUTOR_BASE_URL/v1/tutor/chat -X POST -H "Content-Type: application/json" -d '{"session_id":"x","message":"ping"}'`. A 200 means the backend is up. A connection-refused means `tutor_base_url` is pointed at the wrong address (most likely `localhost` inside the container when the backend is on the host — see the Known Limitations section).

### Session reuse left a tutor-enabled container after the toggle was flipped OFF

This is expected behavior for any session started before the toggle changed. The container continues running until the learner triggers `launch_workspace` again, at which point the launcher detects the mismatch (`existing_session.lab_tutor_enabled != lab_tutor_enabled`) and tears down the old container before starting a fresh one. If you need the old container gone immediately, stop it with `docker rm -f <container-name>`. The learner will see an error on their current page and need to relaunch.

### The extension activates but the sidebar shows no response to chat messages

This usually means the extension reached the backend URL but the backend returned an error. Check the backend process logs for request traces on `POST /v1/tutor/chat`. If the URL is `localhost:8000` inside the container and the backend is on the host, the request will fail with connection-refused — the status bar will still show and activate, but the chat will silently error. See the `localhost` limitation in the Known Limitations section above.

---

## Known limitations (Phase 2 carry-overs)

These were surfaced during Phase 1 reviews. They are not bugs but must be addressed before the tutor is used in production for real learners.

**`localhost` default base URL.**
`LearnerStudioService.__init__` defaults `tutor_base_url` to `http://localhost:8000` (falling back to the `LAB_TUTOR_BASE_URL` env var on the host process). Inside a Docker container, `localhost` refers to the container itself, not the host. For real learner workspaces hitting a tutor backend running on the host, override `tutor_base_url` when constructing `LearnerStudioService` — for example, `http://host.docker.internal:8000` on macOS, or the production tutor service hostname in a deployed environment. Without this override the extension silently fails to reach the backend on every request.

**HintBudget is client-side only.**
`HintBudget` in `extensions/lab-tutor/src/state/hint-budget.ts` lives entirely in the extension process. It initializes at capacity 4 on every container start and is decremented by the extension without server confirmation. A learner can reset it by restarting the editor, and the backend has no record of how many hints were actually consumed. Phase 2 should move authority server-side and have the extension read remaining budget from the `hint_tier` field on `TutorChatResponse` — that field is already plumbed in `app/domain/tutor.py` and passed through `TutorService.chat`; it just needs to be populated.

**Container env vars are read once.**
The extension reads `process.env.LAB_TUTOR_SESSION_ID` at activation. If a workspace container is reused without restart (the toggle-reuse guard mostly prevents this for the toggle case), the extension will use a stale session id for the lifetime of that container. The safe design invariant is: any change that affects which session id the extension should use must result in a container restart.

**No auth on `/v1/tutor/*`.**
Any HTTP client that can reach the backend process can call the tutor routes without credentials. Phase 1 is localhost-only so this is acceptable, but it must be addressed before the tutor backend is exposed outside a loopback interface. Phase 3 (Copilot hook integration) requires defining an authentication model; design that before exposing the routes to a real network. The `session_id` in request bodies is not validated as a real session — it is echoed back in stub responses but otherwise unused.

**Stub backend.**
`TutorService` returns canned responses for all requests. The chat stub echoes the first 80 characters of the message and returns `hint_tier=None`; the submit stub always returns `passed: true` with two fixed viva questions. Phase 2 wires real LLM-backed coaching in place of the stubs. Until then the tutor sidebar works end-to-end mechanically, but the replies are not pedagogically useful.

**Extension `types.ts` is not published.**
`extensions/lab-tutor/src/types.ts` defines `ChatReply`, `SubmitResult`, and `VivaQuestion` as TypeScript interfaces for the client. These must stay in sync with `app/domain/tutor.py`. There is currently no shared schema contract or codegen step — a field rename on the backend will silently break the extension until both sides are updated.

---

## Pointer to the original handover

The product-level handover doc (pedagogical thesis, non-negotiables, Phase 2–5 roadmap) is the authoritative source for the WHY and the HOW BEYOND Phase 1. This document covers the WHAT and the HOW NOW.

That handover lives in conversation history and should be moved into the repo separately if the team wants it version-tracked alongside the code. When it is added, a link here would complete the documentation chain: engineering spec (this file) → product spec (handover) → implementation (source).
