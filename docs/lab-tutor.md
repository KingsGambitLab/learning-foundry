# Lab Tutor

## What it is

The Lab Tutor is a page-embedded chat widget that gives learners a Socratic AI tutor on every LMS page, without requiring VS Code or any browser extension. The widget renders as a floating bubble in the bottom-right corner of the LMS SPA; clicking it expands a chat panel that calls `/v1/tutor/chat` (Claude Haiku 4.5) with the assignment title as context. A per-assignment kill switch on `CourseRun.lab_tutor_enabled` (default `False`) controls whether the widget is injected at all. Real reactive coaching, viva orchestration, and code-context awareness are Phase 2 and later.

---

## Architecture

```
Learner browser
   └─ LMS SPA (FastAPI Jinja shell + app/static/lms.js)
       └─ When the active enrollment's CourseRun has lab_tutor_enabled=true,
          lms.js injects <link> + <script> for the widget assets into <head>
              |
              v
          Floating bubble bottom-right  (app/static/lab-tutor.js)
              |
              | click
              v
          Chat panel with assignment-title-aware welcome card
              |
              | POST /v1/tutor/chat  {session_id, message, assignment_title}
              v
Backend (FastAPI on the same origin)
   └─ /v1/tutor/chat --> TutorService.chat --> Anthropic SDK --> Claude Haiku 4.5
```

### Widget layer (`app/static/lab-tutor.js`, `lab-tutor.css`)

Self-contained vanilla JS, no build step, no framework. Reads config from data
attributes on its own `<script>` tag when loaded standalone, or from the opts
object passed to `window.__labTutorMount(opts)` when driven by the SPA.
Exposes three functions on `window`:

- `__labTutorMount(opts)` — creates and shows the bubble + panel
- `__labTutorUnmount()` — hides them (DOM node kept; re-mount is cheap)
- `__labTutorUpdate(opts)` — updates `assignmentTitle` / `sessionId` without
  re-mounting (used when the learner switches enrollments)

Styles use a `lt-` class prefix and load from a separate stylesheet
(`lab-tutor.css`) so they cannot collide with LMS CSS.

### Mount logic layer (`app/static/lms.js`)

`syncLabTutor()` is called every time the active enrollment changes (on initial
page load, after catalog/enrollment refresh, and after enrollment switch).
It reads `lab_tutor_enabled` from the matching entry in the catalog response
(a `PublishedCourseSummary` field populated from the underlying `CourseRun`).

- First time the toggle is ON: injects `<link>` and `<script>` tags into
  `<head>`, then calls `window.__labTutorMount(...)` once the script loads.
- On subsequent enrollment switches where the toggle is ON: calls
  `window.__labTutorMount(...)` with updated opts (or `__labTutorUpdate(...)`
  if the widget was already mounted with a different title).
- When the toggle is OFF: calls `window.__labTutorUnmount()`.

The script injection guard (`labTutorScriptInjected`) ensures the `<script>`
tag is only added to `<head>` once per page lifetime regardless of how many
times `syncLabTutor` fires.

### Backend layer

Router mounted at `/v1/tutor` in `app/api/tutor.py`. Two routes:

- `POST /v1/tutor/chat` — takes a `TutorChatRequest`, returns a
  `TutorChatResponse` with a Claude Haiku 4.5 reply.
- `POST /v1/tutor/submit` — Phase 1 stub; always returns a canned response.

`TutorService` in `app/services/tutor_service.py` builds a Socratic-coach
system prompt that includes the assignment title when present (so the model
never asks "what are you working on"). Prompt caching is enabled on the system
prompt via `cache_control: {"type": "ephemeral"}` to keep per-call cost low
across many simultaneous learners.

---

## The on/off toggle

This is the load-bearing section. Read it before touching any toggle-related
code.

### Where it lives

- `CourseRun.lab_tutor_enabled: bool` (default `False`), defined in
  `app/domain/course.py`.
- Persisted inside `course_runs.payload_json` — a SQLite TEXT column that
  holds a JSON blob for the entire CourseRun.
- Surfaced to the browser via `PublishedCourseSummary.lab_tutor_enabled`
  (see `app/domain/learner.py`), which is populated from `CourseRun` by
  `app/services/lms_service.py` and returned in the catalog API response.
- Scope: per CourseRun (i.e., per assignment). Every learner enrolled in the
  same assignment sees the same value simultaneously.

### Default

`False`. Existing assignments and newly-created assignments are OFF until
explicitly enabled. Async generation rebuilds preserve the current value
(see "Async generation preservation" below).

### How to flip it

There is no admin UI in Phase 1. Use the SQLite database directly.

```sql
-- read current value
SELECT json_extract(payload_json, '$.lab_tutor_enabled')
FROM course_runs
WHERE course_run_id = '<id>';

-- enable
UPDATE course_runs
SET payload_json = json_set(payload_json, '$.lab_tutor_enabled', json('true'))
WHERE course_run_id = '<id>';

-- disable
UPDATE course_runs
SET payload_json = json_set(payload_json, '$.lab_tutor_enabled', json('false'))
WHERE course_run_id = '<id>';
```

After flipping, the change takes effect on the learner's next page render.
No server restart is required; the catalog is re-fetched on every SPA
initialisation.

### What OFF guarantees (the contract)

- The widget script and stylesheet are NOT injected into the page.
- No floating bubble appears. No background polling. No network calls to
  `/v1/tutor/*` from that learner's page.
- The backend routes remain reachable (other learners on enabled assignments
  can still call them) but nothing on this learner's page calls them.

### What OFF does NOT do

- It does not retroactively close an already-open chat panel in an existing
  browser tab. The mount/unmount logic runs when the active enrollment changes;
  if the same enrollment's toggle flips while the tab is still open, the learner
  must refresh to pick up the change.
- It does not delete chat history. There is no server-side chat history in
  Phase 1 — every message is stateless; `session_id` is generated per
  page-load and never stored on the backend.

### Async generation preservation

`apply_generated_plan` in `app/services/course_workflow_service.py` rebuilds
`CourseRun` objects from scratch during async course generation. The toggle is
explicitly copied from the pre-rebuild run:

```python
course_run.lab_tutor_enabled = existing.lab_tutor_enabled
```

Flipping ON during a queued regeneration therefore survives the rebuild without
reverting to `False`.

---

## Testing the toggle

Steps for an engineer to verify behavior locally. The database is at
`data/course_gen.db` relative to the repo root.

```bash
# 1. Find a course run with the tutor already enabled
sqlite3 data/course_gen.db \
  "SELECT course_run_id, json_extract(payload_json,'$.title') \
   FROM course_runs \
   WHERE json_extract(payload_json,'$.lab_tutor_enabled') = 1;"

# 2. Find an enrollment for that course run
sqlite3 data/course_gen.db \
  "SELECT enrollment_id FROM learner_enrollments \
   WHERE course_run_id = '<id>';"

# 3. Open the LMS in a browser and switch to that enrollment
#    (visit http://127.0.0.1:8012/ and click into the enrollment)

# 4. Confirm the widget appears bottom-right. Click the bubble;
#    the panel should expand. Type a message and verify a real
#    Claude Haiku 4.5 response arrives.

# 5. Disable the toggle
sqlite3 data/course_gen.db \
  "UPDATE course_runs \
   SET payload_json = json_set(payload_json,'$.lab_tutor_enabled',json('false')) \
   WHERE course_run_id = '<id>';"

# 6. Refresh the browser page. The bubble should be gone.
```

To test the ON path from scratch, run the enable query in step 5 (with
`json('true')`) against any published course run and then navigate to an
enrollment for that run.

---

## The widget (`app/static/lab-tutor.js`, `lab-tutor.css`)

- Self-contained vanilla JS. No build step, no framework, no globals. ES2020.
- Config comes from data attributes on the script tag or from the opts object
  passed to `__labTutorMount`:
  - `data-assignment-title` / `assignmentTitle` — shown in the welcome card
    and sent to the backend with every chat request
  - `data-session-id` / `sessionId` — a per-page-load identifier derived from
    the enrollment id (`"lms-" + enrollmentId`); the backend does not persist
    sessions in Phase 1
  - `data-base-url` / `baseUrl` — defaults to relative (same-origin); used as
    prefix for fetch calls and the CSS `<link>`
- The welcome card renders "Working on `<title>`" when an assignment title is
  set, or a generic prompt when it is not.
- Tutor replies are rendered with minimal Markdown: `**bold**` segments are
  converted to `<strong>` nodes. Other HTML is never injected — untrusted
  content is always set via `textContent`.
- Styles are isolated under the `lt-` class prefix. The stylesheet loads once
  per page; `syncLabTutor` guards against double-injection with a
  `querySelector` check on the `<link>` tag.

---

## The mount logic (`app/static/lms.js`)

`syncLabTutor()` is the single entry point for all widget lifecycle changes.
Callers:

- Immediately after `renderAll()` on SPA boot.
- After `loadEnrollment()` resolves (enrollment data arrives from the server).
- After `refreshCatalog()` + `refreshEnrollments()` settle (next enrollment
  determined).

Internal flow:

1. `labTutorEnabledForCurrentEnrollment()` looks up the active enrollment's
   `course_run_id` in `state.catalog.courses` and returns
   `course.lab_tutor_enabled === true`.
2. If `false`, call `window.__labTutorUnmount()` (no-op if never mounted) and
   return.
3. If `true` and `labTutorScriptInjected` is `false`: create a `<script>` tag
   pointing at `/static/lab-tutor.js`, set `data-session-id` and
   `data-assignment-title`, append to `<head>`, and call
   `window.__labTutorMount(...)` on `load`.
4. If `true` and the script is already injected: call
   `window.__labTutorMount(...)` directly (the widget is already in the DOM;
   mount is idempotent).

The session id passed to the widget is `"lms-" + enrollmentId`.
The assignment title is `experience.enrollment.course_title`.

---

## The backend (`/v1/tutor/*`)

Routes are defined in `app/api/tutor.py` and mounted at `/v1/tutor`.

### POST /v1/tutor/chat

Request schema (`TutorChatRequest` in `app/domain/tutor.py`):

```json
{
  "session_id": "lms-abc123",
  "message": "I'm stuck on the join condition",
  "assignment_title": "Build a REST API with FastAPI"
}
```

Response schema (`TutorChatResponse`):

```json
{
  "reply": "What does your current JOIN clause look like, and what result are you getting?",
  "hint_tier": null
}
```

`TutorService.chat` builds the system prompt from `_TUTOR_SYSTEM_BASE` plus an
assignment-title suffix (when `assignment_title` is set), caches the prompt
block via `cache_control: {"type": "ephemeral"}`, and sends a single-turn
message to Claude Haiku 4.5 with a 30-second timeout. The `hint_tier` field
is reserved for Phase 2 budget authority; it is always `null` in Phase 1.

If `ANTHROPIC_API_KEY` is missing or empty, the service returns a clear
"Tutor backend not configured" message rather than crashing.

### POST /v1/tutor/submit

Phase 1 stub. Always returns `{"passed": true, "details": "stub"}` in
`test_results` and two hard-coded viva questions. Real grading and viva
orchestration land later.

---

## Configuration

To run the backend with real Haiku replies, provide an API key via an env file:

```bash
# anthropic.env.keys contains: ANTHROPIC_API_KEY=sk-ant-...

COURSE_GEN_ANTHROPIC_ENV_FILE=/path/to/anthropic.env.keys \
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8012
```

The loader (`_load_env_file` in `tutor_service.py`) uses `setdefault` semantics:
it only sets a key if it is not already set in the environment AND its value
in the file is non-empty. A stale `ANTHROPIC_API_KEY=` exported by the parent
shell therefore does not shadow a real key in the file.

If the env file is absent or the key is empty after loading, the widget receives
a human-readable "Tutor backend not configured — set
COURSE_GEN_ANTHROPIC_ENV_FILE..." reply instead of an HTTP error.

---

## File map

| Concern | File |
|---|---|
| `CourseRun.lab_tutor_enabled` toggle | `app/domain/course.py` |
| Catalog exposure (`PublishedCourseSummary.lab_tutor_enabled`) | `app/domain/learner.py` |
| Catalog population | `app/services/lms_service.py` (passes `run.lab_tutor_enabled` to `PublishedCourseSummary.from_run`) |
| Async-rebuild preservation | `app/services/course_workflow_service.py` (`apply_generated_plan`) |
| Backend domain types | `app/domain/tutor.py` |
| Backend service (Haiku + Socratic prompt) | `app/services/tutor_service.py` |
| Backend router | `app/api/tutor.py` |
| Widget bundle | `app/static/lab-tutor.js` |
| Widget styles | `app/static/lab-tutor.css` |
| LMS SPA mount logic | `app/static/lms.js` (`syncLabTutor`, `labTutorEnabledForCurrentEnrollment`) |

---

## Known limitations (Phase 1)

**Widget only mounts on LMS SPA pages.** It does not appear inside the
code-server iframe when the learner launches the editor. Cross-iframe injection
into code-server is a separate problem deferred until LMS-page placement is
validated.

**Chat is stateless server-side.** `session_id` is sent with each message but
the backend does not persist conversation history. The model sees only the
current turn: system prompt + one user message. Multi-turn awareness requires a
server-side conversation store — Phase 2.

**`HintBudget` is gone.** It lived in the old VS Code extension. Re-introduce
server-side when hint enforcement is needed; the `hint_tier` field on
`TutorChatResponse` is already plumbed for it.

**No auth on `/v1/tutor/*` routes.** Any client that can reach the backend
process can call these endpoints. Phase 3 (production hardening) must add
authentication before the routes are exposed outside a loopback interface.

**`TutorService.submit` is a stub.** Real grading and viva orchestration land
later.

**No streaming.** The widget waits for the full Haiku response before rendering
it. Streaming would feel more responsive — Phase 2.

**Toggle flip not live.** If an admin flips `lab_tutor_enabled` while a learner
has the page open, the learner's current tab does not pick up the change until
they refresh. This is acceptable for Phase 1 operational cadences.

---

## Context wiring — invariants (DO NOT BREAK)

This has regressed more than once. The tutor is only useful if it
receives the assignment's **project brief + deliverables**, not just the
title. The whole chain hinges on ONE contract:

> **`session_id` is always `lms-<enrollmentId>`.**

Producers of `session_id` (all must keep the `lms-<enrollmentId>` form):

1. **LMS-page widget** — `app/static/lms.js` `syncLabTutor()` calls
   `__labTutorMount({ sessionId: "lms-" + enrollmentId, ... })`.
2. **In-editor widget** — nginx (`/etc/nginx/conf.d/course-gen-codex.conf`,
   editor `location`) `sub_filter`-injects
   `<script src="/static/lab-tutor-editor-boot.js" data-editor-port="$eport">`
   into the code-server HTML. `lab-tutor-editor-boot.js` calls
   `GET /v1/tutor/editor-context?port=<port>`, which maps the port →
   owning learner's enrollment (ownership-checked) and returns
   `session_id = "lms-<enrollmentId>"` + the real `assignment_title`.

Consumer: `TutorService._resolve_session_context(session_id)` (in
`app/services/tutor_service.py`). It MUST resolve the `lms-<enrollmentId>`
namespace — first by finding that enrollment's workspace session
(workspace files → brief/code), and if no workspace exists yet, by
falling back to the **publish snapshot's** `project_brief_markdown` /
`deliverables_markdown`. (The original code only matched
`LearnerWorkspaceSession.id == session_id`, which `lms-<eid>` never
equals → the tutor silently got no brief and asked the learner to paste
the spec. That is the canonical regression.)

Behavioral invariant (enforced in `_TUTOR_PERSONA`): the tutor must
**never** ask the learner to paste the assignment/spec/question, and
should include a Mermaid diagram whenever the content can be
visualized (picking the best diagram type for the content).

### Regression checklist (run after ANY tutor / nginx / lms.js change)

```bash
# 1. Resolver returns a real brief for the lms-<eid> namespace.
#    (login as a learner with an enrollment, then:)
curl -s -b jar -X POST http://18.236.242.248/v1/tutor/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"lms-<ENROLLMENT_ID>","message":"Explain the problem statement with a mindmap","assignment_title":"<title>"}'
#    PASS = reply contains a ```mermaid block AND does NOT say
#    "paste the assignment / spec / question".
# 2. Editor path: GET /v1/tutor/editor-context?port=<running editor port>
#    returns assignment_title + session_id "lms-<enrollmentId>".
# 3. Served code-server HTML still injects lab-tutor-editor-boot.js
#    (curl the /editor/<port>/ page, grep for the script tag).
```

If any step fails, the tutor is running context-blind — fix before
shipping.

## Historical note

This document previously described a VS Code extension that shipped inside
the learner-studio Docker image (pre-installed at `/opt/lab-tutor/extensions`,
loaded via `--extensions-dir`, driven by `LAB_TUTOR_BASE_URL` and
`LAB_TUTOR_SESSION_ID` container env vars). That approach was demolished at
commit `44137df3` after it became clear the VS Code extension API cannot
surface the tutor with the visual prominence the design required — there is no
floating-window primitive in the VS Code extension API, and sidebar webviews
are easily overlooked. Both `/codex:adversarial-review` passes during that
Phase 1 build are recorded in
`docs/superpowers/plans/2026-05-14-lab-tutor-phase-1.md` under the "Resolved
decisions" and "REVISED" banners.
