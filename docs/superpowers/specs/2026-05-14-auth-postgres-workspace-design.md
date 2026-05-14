# Auth, Postgres, and per-user workspace mapping

**Date:** 2026-05-14
**Status:** Design — awaiting user review before implementation planning

## Goal

Turn the single-user local prototype into a real multi-user backend without changing the creator-to-learner pipeline's behavior. Four coordinated changes:

1. Replace SQLite with Postgres behind the existing store seam.
2. Add user accounts (email + password, role at registration) and server-side sessions.
3. Bind enrollments to real authenticated users instead of the hard-coded `local-learner` identity.
4. Key learner workspaces and editor sessions by `<user_id, assignment_id>` instead of `<enrollment_id>`.

The migration must be done without disturbing the existing server running at `127.0.0.1:8010`. The new server runs in parallel on port `8030` so behavior can be diffed end-to-end before any cutover.

## Scope

In scope:

- `PostgresWorkflowStore` replacing `SQLiteWorkflowStore` behind a `WorkflowStore` Protocol.
- SQLAlchemy Core for the existing 12 tables (JSON-blob shape preserved 1:1); ORM models for the new `users` and `user_sessions` tables.
- Alembic migrations, initial revision plus one auth revision.
- `/auth/register`, `/auth/login`, `/auth/logout`, `/auth/me` JSON routes; minimal Jinja `/login` and `/register` pages.
- FastAPI dependencies for `current_user`, `current_user_optional`, `require_role(...)`.
- Role-guarded `/v1/*` routes (creator vs learner).
- Snapshot-based one-shot migrator: 8010's live SQLite → local snapshot DB → Postgres.
- Workspace path scheme change to `learner_workspaces/<user_id>/<assignment_id>/workspace/` and a workspace-rename pass in the migrator.
- Side-by-side verification script comparing 8010 (SQLite) and 8030 (Postgres) JSON outputs.
- Deployment notes for the demo EC2 host.

Out of scope (flagged so they are explicitly deferred):

- Email verification flow.
- Password reset flow.
- OAuth / social login.
- Per-deliverable workspaces (today's `shared_course` scope is preserved).
- Production hardening of the docker-socket spawn pattern (rootless docker, sysbox, dedicated runner host).
- Managed Postgres (RDS / Supabase) — deferred per user direction; Postgres runs inside the same compose file on the EC2 host for demo.

## Architecture overview

```
                ┌───────────────────────────────┐
   browser ───▶ │  FastAPI app (port 8030)      │
                │  ┌─────────────────────────┐  │
                │  │  /auth routes           │  │
                │  │  /v1/* (role-guarded)   │  │
                │  │  Jinja pages            │  │
                │  └────────────┬────────────┘  │
                │               │                │
                │  ┌────────────▼────────────┐  │
                │  │  Services               │  │
                │  │  (LMSService, etc.)     │  │
                │  └────────────┬────────────┘  │
                │               │                │
                │  ┌────────────▼────────────┐  │
                │  │  WorkflowStore Protocol │  │
                │  │   (48 methods)          │  │
                │  └────────────┬────────────┘  │
                │               │                │
                │  ┌────────────▼────────────┐  │
                │  │  PostgresWorkflowStore  │──┼──▶ Postgres (docker compose)
                │  └─────────────────────────┘  │
                └───────────────────────────────┘

           ┌──────────────────────────────────┐
8010   ───▶│  Old FastAPI app (SQLite)        │   untouched during migration
           └──────────────────────────────────┘
```

The four pieces are sequenced because each later piece depends on the earlier ones:

1. **Storage swap** — lands first because auth needs Postgres tables to live somewhere.
2. **Auth (users + sessions + routes)** — sits on top of Postgres.
3. **Enrollment rewiring** — sits on top of auth.
4. **Workspace key change** — smallest surface; lands last so we can verify the auth + storage swap is solid before touching disk paths.

## Section 1 — Storage layer

### Files added

- `app/storage/workflow_store.py` — `WorkflowStore` Protocol with the 48 public methods from today's `SQLiteWorkflowStore`. Method signatures and return types are unchanged. Services type-annotate against this Protocol; the concrete class swaps via dependency wiring in `app/main.py`.
- `app/storage/postgres_store.py` — new concrete store using SQLAlchemy Core + a psycopg engine. Same method names, identical JSON-blob behavior for legacy tables.
- `app/storage/database.py` — thin module owning the `Engine` and a `session()` context manager. Reads `DATABASE_URL`. `pool_pre_ping=True`, pool size 5 / max overflow 10.
- `alembic/` directory at repo root with `env.py`, `script.py.mako`, and `versions/0001_initial.py` (creates all 12 legacy tables) + `versions/0002_auth.py` (creates `users` and `user_sessions`).

### Files removed

- `app/storage/sqlite_store.py` is deleted after callers are switched. The migrator uses the stdlib `sqlite3` module directly to read the snapshot file — it does not depend on the deleted class. Production app code does not import `sqlite3` after the swap.

### Schema (legacy tables, ported 1:1)

| Table | Key columns kept normalized | Payload |
| --- | --- | --- |
| `workflow_runs` | `run_id PK, title, stage, status, created_at, updated_at` | `payload JSONB NOT NULL` |
| `workflow_events` | `id BIGSERIAL PK, run_id FK, sequence_no, event_type, created_at` | `payload JSONB NOT NULL` |
| `course_runs` | `course_run_id PK, title, package_type, stage, status, created_at, updated_at` | `payload JSONB NOT NULL` |
| `course_events` | `id BIGSERIAL PK, course_run_id FK, sequence_no, event_type, created_at` | `payload JSONB NOT NULL` |
| `learner_enrollments` | `enrollment_id PK, learner_id, course_run_id, status, created_at, updated_at` | `payload JSONB NOT NULL` |
| `learner_submissions` | `id PK, enrollment_id, deliverable_id, created_at, status` | `payload JSONB NOT NULL` |
| `learner_workspace_sessions` | `id PK, enrollment_id, deliverable_id, status, created_at, updated_at` | `payload JSONB NOT NULL` |
| `publish_snapshots` | `id PK, course_run_id, created_at` | `payload JSONB NOT NULL` |
| `creator_feedback`, `learner_feedback` | `id PK, course_run_id, created_at` | `payload JSONB NOT NULL` |
| `learner_eval_reports` | `id PK, enrollment_id, created_at` | `payload JSONB NOT NULL` |
| `creator_assets` | `id PK, course_run_id, created_at, updated_at` | `payload JSONB NOT NULL` |

Indexes carry over verbatim from `SQLiteWorkflowStore._ensure_schema`. `learner_id` on `learner_enrollments` gets a non-unique B-tree index for the new "list my enrollments" path.

### SQLite-isms swapped

- `INSERT OR REPLACE` → `INSERT … ON CONFLICT (<pk>) DO UPDATE SET …`
- `INTEGER PRIMARY KEY AUTOINCREMENT` → `BIGSERIAL PRIMARY KEY`
- `PRAGMA journal_mode=WAL` removed (Postgres MVCC).
- `threading.Lock` removed (Postgres handles concurrency).
- `payload_json TEXT` → `payload JSONB`. Encoders/decoders adjusted to pass Python dicts directly rather than `json.dumps` strings.

### Transaction semantics

Every public store method runs in its own `with engine.begin() as conn:` block — explicit transaction per call. This matches today's per-call SQLite semantics; no method silently reuses a connection from a caller.

### Tests

- `tests/conftest.py` gains a session-scoped `postgres_engine` fixture using `testcontainers-python`. A function-scoped `store` fixture truncates every table (`TRUNCATE … RESTART IDENTITY CASCADE`) between tests for isolation.
- Existing tests that construct `SQLiteWorkflowStore(tmp_path / "x.db")` are redirected to consume the new fixture.

## Section 2 — User identity, sessions, auth routes

### Tables (Alembic revision `0002_auth.py`)

```sql
CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         CITEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role          TEXT NOT NULL CHECK (role IN ('creator','learner')),
  display_name  TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE user_sessions (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at   TIMESTAMPTZ NOT NULL,
  last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ip           INET,
  user_agent   TEXT
);
CREATE INDEX user_sessions_expires_at_idx ON user_sessions(expires_at);
CREATE INDEX user_sessions_user_id_idx ON user_sessions(user_id);
```

### Modules

- `app/services/auth_passwords.py` — `hash_password(plain) -> str`, `verify_password(plain, hash) -> bool`. Backed by `passlib[bcrypt]`. Cost factor 12, overridable via `AUTH_BCRYPT_COST` env.
- `app/services/auth_session.py` — `create_session(user_id, ip, ua) -> session_id`, `load_session(session_id) -> User | None` (bumps `last_seen_at`, returns `None` if expired or missing), `revoke_session(session_id)`. Cookie name `coursegen_session`. 14-day expiry, sliding via `last_seen_at` bump.
- `app/api/deps.py` — `current_user`, `current_user_optional`, `require_role(*roles)` dependencies.

### Routes (`app/api/auth_routes.py`, mounted at `/auth`)

| Method | Path | Body | Response | Notes |
| --- | --- | --- | --- | --- |
| `POST` | `/auth/register` | `{email, password, role, display_name?}` | `201 {user_id, role}` | Creates user, creates session, sets cookie. Validates email format, password ≥ 8 chars, role ∈ {creator, learner}. |
| `POST` | `/auth/login` | `{email, password}` | `200 {user_id, role}` | Same response shape on bad email vs bad password (no user enumeration). |
| `POST` | `/auth/logout` | empty | `204` | Revokes session, clears cookie. |
| `GET` | `/auth/me` | — | `200 user` or `401` | Used by the JS frontend for auth-aware nav. |

### Pages

- `GET /login` → `app/templates/login.html`. Email + password form, link to register.
- `GET /register` → `app/templates/register.html`. Email + password + role radio + optional display name.

Both submit via JS to the JSON endpoints and redirect on success. A Jinja include `_auth_header.html` is added to existing pages — shows "Sign in" / "Register" when guest, "<display_name> · Log out" when authenticated.

### Cookie configuration

- Name: `coursegen_session`.
- `HttpOnly=True`, `SameSite=Lax`.
- `Secure` toggled by `SESSION_COOKIE_SECURE` env (off in dev, on in staging/prod).
- Value: opaque UUID (the session's primary key). No claims encoded — server-side state is authoritative.

### Route guards

- Every `/v1/*` route gets exactly one of: `Depends(current_user)`, `Depends(require_role("creator"))`, or `Depends(require_role("learner"))`.
- Creator routes: `/v1/courses/*`, `/v1/workflow/*`, `/v1/publish/*`, `/v1/creator/*`, `/v1/task-agent-authoring/*`.
- Learner routes: `/v1/lms/*` (catalog, enrollments, workspace, submissions, feedback).
- Page routes (`/`, `/courses`, `/create-course`, `/draft-timeline`) use `current_user_optional`; unauthenticated access to a creator-only page redirects to `/login?next=<original>`.

## Section 3 — Enrollment rewiring + data migration

### Domain & service changes

- `LearnerEnrollment.learner_id` now always carries a real `users.id` UUID. Pydantic validator enforces UUID shape on save.
- `CreateEnrollmentRequest` drops its `learner_id` field. The route reads `current_user.id` instead.
- `LMSService.list_enrollments(learner_id: str)` — route passes the authenticated user's id explicitly; no default value.
- `LMSService.enroll(request, *, learner_id)` — new keyword-only `learner_id` supplied by the route.
- Other LMSService methods already trust `enrollment.learner_id` and stay correct once enrollment rows carry real UUIDs.

### Snapshot + migrator: `scripts/migrate_sqlite_to_postgres.py`

Three logical phases, all idempotent so the script can be rerun safely.

**Phase 1 — Snapshot 8010's SQLite into a local file.**

- Source: `/Users/tushar/Desktop/codebases/course-gen-codex/data/course_gen.db` (default, CLI flag overrides).
- Target: `<worktree>/data/course_gen_snapshot.db`.
- Mechanism: open source read-only (`?mode=ro&immutable=0`), run `VACUUM INTO '<target>'`. This produces a consistent snapshot even while 8010 is actively writing.
- Re-running the snapshot phase overwrites the target.

**Phase 2 — Copy snapshot to Postgres.**

1. Verify `DATABASE_URL` schema is at Alembic head; abort with a clear message otherwise.
2. **Seed user:** ensure a synthetic `users` row exists — email `legacy-local-learner@coursegen.local`, role `learner`, password randomly generated and printed once to stdout. Idempotent via `INSERT … ON CONFLICT (email) DO NOTHING RETURNING id` + a subsequent `SELECT id`. Capture as `seed_learner_id`.
3. **Per-table copy** in FK-safe order: `course_runs` → `workflow_runs` → `publish_snapshots` → `learner_enrollments` → `learner_workspace_sessions` → `learner_submissions` → `creator_feedback`, `learner_feedback`, `learner_eval_reports` → `creator_assets` → `workflow_events`, `course_events`.
4. For each table: `SELECT *` from the snapshot, batch-insert into Postgres (batch size 500) with `ON CONFLICT (pk) DO NOTHING`. `payload_json` text is parsed once and inserted as JSON.
5. **Learner id rewrite:** during the `learner_enrollments` copy, replace `learner_id == "local-learner"` with `seed_learner_id` in both the normalized column and the JSON `payload`. The same rewrite is applied to the JSON `payload` of `learner_feedback` rows (which carry a `learner_id` field). No other table normalizes or nests `learner_id`, but the migrator does a final defensive grep across all migrated JSON payloads for the literal string `"local-learner"` and fails loudly if any survive.
6. **Row-count verification:** after each table, compare snapshot count to Postgres count. Non-match → exit non-zero.
7. **Payload sanity check:** for a sampled min(20, 5%) of rows per table, round-trip `json.loads(snapshot_row.payload_json)` and compare equal to the Postgres `payload` JSONB. Cheap detection of encoder drift.

**Phase 3 — Workspace directory rename.**

- For each row in the migrated `learner_enrollments`:
  - `old_root = <main-checkout>/learner_workspaces/<enrollment_id>/workspace`
  - `new_root = <worktree>/learner_workspaces/<learner_id>/<shared_workflow_run_id>/workspace`
- If a workspace directory needs to be present for verification: `cp -r old_root new_root` (file copy, not move — 8010 keeps its own copy on disk).
- If `new_root` already exists, skip.

**Editor sessions across the cutover.**

The `learner_workspace_sessions` payload carries `workspace_root` strings that reference the old layout. During Phase 2, when copying these rows, rewrite `workspace_root` in the payload to the new `<user_id>/<assignment_id>` path. On first boot of the new server, `LearnerStudioService.reconcile_stale_sessions` (already present in [app/main.py:161](app/main.py:161)) marks any sessions stopped whose backing container is gone — the learner relaunches and a fresh session row is written.

### Verification script: `scripts/verify_migration.py`

Compares 8010 (SQLite, untouched) to 8030 (Postgres, freshly migrated) end-to-end.

1. Hit `127.0.0.1:8010` and `127.0.0.1:8030` for each of these endpoints, capture JSON:
   - `GET /v1/lms/courses` (catalog)
   - `GET /v1/lms/enrollments?learner_id=local-learner` (against 8010) vs `GET /v1/lms/enrollments` (against 8030 with the seed-learner's session cookie). The two endpoints address the same logical learner under different keys; the diff normalizes `learner_id` before comparing so the rewrite from `"local-learner"` to `seed_learner_id` is not flagged as a mismatch.
   - `GET /v1/workflow/runs`
   - `GET /v1/course-runs/{id}/timeline` for a sampled course
   - `GET /v1/lms/enrollments/{id}/experience` for the first enrollment
2. Normalize both payloads (sort keys, project away volatile fields like `last_seen_at`, `updated_at` when those fields aren't load-bearing).
3. Diff. Mismatches written to `scripts/migration_verification_report.json`.
4. Spot-check: log in as the seed learner against 8030, launch the workspace, list workspace files — assert the file list matches what 8010 returns for the same enrollment.
5. Exit non-zero on any unaccounted-for mismatch.

## Section 4 — Workspace key change

### Path scheme

- New: `learner_workspaces/<user_id>/<assignment_id>/workspace/`
- `user_id` = the authenticated user's UUID string.
- `assignment_id` = `enrollment.shared_workflow_run_id` (the workflow run id that produced the assignment).
- `<user_id, assignment_id>` is 1:1 with `<user_id, course_run_id>` today (one assignment per course), so no uniqueness regression.

### Code change

[`LMSService._workspace_root`](app/services/lms_service.py:386):

```python
def _workspace_root(self, enrollment: LearnerEnrollment) -> Path:
    return (
        self.base_dir
        / enrollment.learner_id
        / enrollment.shared_workflow_run_id
        / "workspace"
    )
```

That's the only behavioral change. `_ensure_workspace_seeded`, the file read/write methods, `launch_workspace`, and the editor session row continue to flow through `_workspace_root` and stay correct.

### Backward compatibility note

The old `learner_workspaces/<enrollment_id>/workspace/` layout persists in the main checkout where 8010 runs. The new server's `learner_workspaces/` directory in the worktree uses only the new scheme. No code reads both schemes — the migration is one-way.

## Deployment notes (demo EC2)

Captured here so the deploy decision lives with the design rather than being relitigated later.

- Single large EC2 instance (recommended: `m6i.xlarge` or larger — 4 vCPU / 16 GB RAM minimum for demo headroom since each active learner spawns a code-server container).
- Docker Compose with services: `app`, `postgres`, `nginx`.
- `app` container bind-mounts `/var/run/docker.sock` so it can continue to spawn code-server and sandbox runtime containers. **Acceptable for staging/demo with team-only access; not safe for production with untrusted learners** — production hardening (rootless docker, sysbox, or a dedicated runner host) is a separate design.
- Persistent volumes: `postgres_data`, `learner_workspaces`.
- nginx (or Caddy) terminates TLS via Let's Encrypt and reverse-proxies to the FastAPI app on `:8030`. Dynamic code-server host ports are reverse-proxied through the same origin via subpaths or per-session subdomains (decision deferred — current local dev exposes raw host ports).
- Env-driven config: `DATABASE_URL`, `SESSION_SECRET`, `SESSION_COOKIE_SECURE=true`, `AUTH_BCRYPT_COST`, OpenAI env file path.
- Deploy mechanism for now: `git pull && docker compose up --build -d`. Promote to a GitHub Actions deploy when manual deploys become painful.
- Postgres lives **inside the same compose file on the same host** for demo simplicity. Splitting to RDS is a one-env-var change when DB ops become a chore.

## Test plan summary

| What | How |
| --- | --- |
| Storage swap parity | Side-by-side JSON diff of 8010 (SQLite) vs 8030 (Postgres) for the catalog, enrollments, workflow, timeline, and experience endpoints. Exit non-zero on any unaccounted mismatch. |
| Storage swap concurrency | `pytest tests/storage/` with the new Postgres fixture; existing tests run unchanged against the Protocol. |
| Auth happy path | `tests/auth/test_register_login_logout.py` — register a creator and a learner, log in, hit `/auth/me`, log out, confirm cookie invalidation. |
| Auth role guards | `tests/auth/test_role_guards.py` — a learner gets 403 on `/v1/courses`; a creator gets 403 on `/v1/lms/enrollments`. |
| Data migration idempotence | Run the migrator twice in a row; second run is a no-op (row counts unchanged). |
| Workspace rename | Pre-create a fake old-layout workspace; run Phase 3; assert new path exists and old is untouched (because 8010 owns it). |
| End-to-end on 8030 | Manually: register a learner against 8030, enroll in a course copied over by the migrator, launch the workspace, write a file, submit, see the grade report. |

## Open questions

None blocking. The following are deferred-by-choice (listed in **Out of scope**) rather than open:

- Email verification, password reset, OAuth.
- Per-deliverable workspaces.
- Prod hardening of the docker-socket spawn pattern.
- Splitting Postgres out to a managed service.

## Sequencing for the implementation plan

The implementation plan (next document) should sequence work as four layered milestones, each independently verifiable:

1. **M1 — Storage swap.** Land `WorkflowStore` Protocol, `PostgresWorkflowStore`, Alembic, docker-compose Postgres, test fixtures. App starts and serves the existing surface against Postgres. No data migrated yet.
2. **M2 — Auth.** Add `users` / `user_sessions`, hashing, session module, dependencies, routes, pages. Existing routes still accept the legacy `learner_id` query param so the app keeps working unauthenticated until M3.
3. **M3 — Wire enrollment to authenticated users + run the migrator.** Drop `learner_id` from request shapes, apply route guards, run snapshot → migrate → verify against 8010.
4. **M4 — Workspace path change.** Update `_workspace_root`, run the workspace-rename migrator phase, verify launch + file read/write + submit on 8030.
