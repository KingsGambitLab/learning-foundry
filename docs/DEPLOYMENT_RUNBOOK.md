# Deployment Runbook — Scaler Labs (EC2 staging)

Repeatable procedure for pushing the app to the staging box. Designed for
frequent (10–20×/day) deploys. Read "Invariants" once; then the
"Standard deploy" block is the only thing you run each time.

## Target

| | |
|---|---|
| Host | `18.236.242.248` (ec2-user) |
| SSH key | `~/Downloads/ai-agent-demo.pem` |
| App dir | `/opt/course-gen-codex` |
| venv | `/opt/course-gen-codex/.venv` |
| Service | `course-gen-codex.service` (systemd, uvicorn → `127.0.0.1:8040`) |
| Edge | nginx `:80` reverse proxy + `/editor/<port>/` code-server proxy |
| DB | Postgres container, loopback `127.0.0.1:5435` |
| Public URL | http://18.236.242.248 |

## Invariants (never break these)

1. **Never overwrite server-only files**: `.env`, `anthropic.env.keys`.
   They hold secrets/keys and are not in git.
2. **Never rsync over runtime data dirs**: `data/`, `learner_workspaces/`,
   `outcome_workspaces/`, `workspaces/`, `generated/`, `logs/`, `tmp/`.
   We only sync source (`app/`) + `pyproject.toml`.
3. **`app/` is source-only** → safe to rsync with `--delete` (prunes stale
   files like the old `app/routes.py` / `app/lms_page.py`).
4. The lab-tutor needs the `anthropic` SDK in the venv. `tutor_service.py`
   imports it at module top — if it's missing the app **will not boot**.
5. New Alembic migration ⇒ run `alembic upgrade head` (loads `.env`).
   No new migration ⇒ skip it (don't run blindly).
6. Backups of any DB-payload edits go to `/opt/course-gen-codex/tmp/`.

## Standard deploy

Run from the repo worktree root (where `app/` lives). Dry-run first.

```bash
KEY=~/Downloads/ai-agent-demo.pem
HOST=ec2-user@18.236.242.248
SSH="ssh -o StrictHostKeyChecking=no -i $KEY"

# 1. Dry-run: review what changes / gets deleted
rsync -az --dry-run --delete --itemize-changes \
  -e "ssh -i $KEY" --exclude '__pycache__' --exclude '*.pyc' \
  app/ $HOST:/opt/course-gen-codex/app/ | grep -E '^[<>ch*]|deleting'

# 2. Real sync (only after the dry-run looks right)
rsync -az --delete -e "ssh -i $KEY" \
  --exclude '__pycache__' --exclude '*.pyc' \
  app/ $HOST:/opt/course-gen-codex/app/
rsync -az -e "ssh -i $KEY" pyproject.toml $HOST:/opt/course-gen-codex/pyproject.toml

# 3. Deps: install anything new (idempotent; skip if no dep change)
$SSH $HOST 'cd /opt/course-gen-codex && .venv/bin/pip install -q -e ".[test]" 2>&1 | tail -1 || true'
#    (anthropic is NOT in pyproject; install once and it stays:)
$SSH $HOST 'cd /opt/course-gen-codex && .venv/bin/python -c "import anthropic" 2>/dev/null || .venv/bin/pip install -q "anthropic>=0.40,<1.0"'

# 4. Migrations: ONLY if alembic/versions/ gained a file this deploy
$SSH $HOST 'cd /opt/course-gen-codex && set -a && . ./.env && set +a && .venv/bin/alembic upgrade head'

# 5. Restart + verify boot
$SSH $HOST 'sudo systemctl restart course-gen-codex.service && sleep 4 && \
  systemctl is-active course-gen-codex.service && \
  sudo journalctl -u course-gen-codex.service -n 5 --no-pager | tail -3'
```

## Smoke test (after every deploy)

```bash
B=http://18.236.242.248
curl -s -o /dev/null -w "login %{http_code}\n"  $B/login
for f in static/lms.js static/lab-tutor.js static/lab-tutor.css static/vendor/mermaid.min.js; do
  printf "%-30s " "$f"; curl -s -o /dev/null -w "%{http_code}\n" "$B/$f"; done
for ep in chat submit triage; do
  printf "tutor/%s unauth " "$ep"; curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST "$B/v1/tutor/$ep" -H 'content-type: application/json' -d '{}'; done
# expect: login 200, all static 200, tutor endpoints 401
```

Full learner round-trip (registers a throwaway user — delete it after):

```bash
B=http://18.236.242.248; J=$(mktemp); E="smoke_$(date +%s)@example.com"
curl -s -c $J -X POST $B/auth/register -H 'content-type: application/json' \
  -d "{\"email\":\"$E\",\"password\":\"TestPass123!\",\"display_name\":\"Smoke\"}" >/dev/null
curl -s -b $J $B/v1/lms/catalog | python3 -c \
  "import sys,json;[print(c['course_run_id'],c.get('lab_tutor_enabled')) for c in json.load(sys.stdin)['courses']]"
curl -s -b $J -X POST $B/v1/lms/enrollments -H 'content-type: application/json' \
  -d '{"course_run_id":"course_f918e889a33c"}' -o /dev/null -w "enroll %{http_code}\n"
curl -s -b $J -X POST $B/v1/tutor/chat -H 'content-type: application/json' \
  -d '{"session_id":"smoke","message":"one-line BM25 intuition?","assignment_title":"Finance RAG"}' \
  -w "\ntutor/chat %{http_code}\n"
```

## Rollback

The deploy is just files + a service restart. To roll back, re-sync the
previous known-good `app/` (e.g. `git checkout <good-sha> -- app pyproject.toml`
in the worktree, then re-run the Standard deploy). DB-payload edits roll
back from the JSON backups in `/opt/course-gen-codex/tmp/` via
`PostgresWorkflowStore.save_*`.

## Appendix — one-off DB ops

These are **not** part of a code deploy; run only when intentionally
changing course data. Always source `.env` first:
`cd /opt/course-gen-codex && set -a && . ./.env && set +a && .venv/bin/python - <<'EOF' … EOF`

**Toggle lab tutor on a course** (no API; payload flag, reversible):

```python
from app.storage.postgres_store import PostgresWorkflowStore
st = PostgresWorkflowStore()
r = st.get_course_run("course_f918e889a33c")
r.lab_tutor_enabled = True
st.save_course_run(r)
```

**Edit a course's starter `requirements.txt`** — it lives inline in the
publish snapshot at `learner_package.deliverables[*].workspace_seed_files`
(`relative_path == "public/starter/requirements.txt"`). New workspaces seed
from the snapshot; already-materialised `learner_workspaces/*/*/workspace/...`
must be patched on disk too (they are not re-seeded — that would clobber
learner code). See `/opt/course-gen-codex/tmp/patch_course_requirements.py`
for the exact, idempotent procedure (backs up the snapshot payload first).

**CPU-only ML deps note**: course starters pin `torch==2.2.2+cpu` via
`--extra-index-url https://download.pytorch.org/whl/cpu` so a plain
`pip install -r requirements.txt` pulls the CPU PyTorch wheel and **zero**
`nvidia-*`/CUDA packages. `faiss-cpu` and `rank_bm25` are CPU-only by
design. Keep that index line if you edit the file.

---

## HTTPS migration (HTTP → TLS via nginx + certbot)

**Status: PLANNED — not executed (deploy freeze + needs a domain).**
Decision: domain + nginx + Let's Encrypt (`certbot`). A trusted cert
**cannot** be issued for the bare IP `18.236.242.248` — a domain is
mandatory.

### Prerequisites (gather before touching the host)

- `<DOMAIN>` — the hostname that will serve the app (e.g.
  `labs.example.com`). Fill in everywhere below.
- DNS: an **A record** `<DOMAIN> → 18.236.242.248`, propagated
  (`dig +short <DOMAIN>` returns the IP) BEFORE running certbot
  (HTTP-01 challenge hits `<DOMAIN>:80`).
- EC2 **security group**: inbound **443/tcp** open (and keep 80/tcp
  open — certbot HTTP-01 + the 80→443 redirect need it). AWS
  console/CLI op, separate from the app deploy.
- Host has `nginx`; install certbot: `sudo dnf install -y certbot
  python3-certbot-nginx` (AL2023) or distro equivalent.

### App-side changes (no code — env only; flip AT cutover)

Both live in the service env (systemd unit / `.env` consumed by
`course-gen-codex.service`). They must change **together with** TLS
going live, never before:

- `SESSION_COOKIE_SECURE=true` — auth cookies then carry `Secure`
  (`samesite=lax` already). If set true while still on http, the
  cookie is never sent → total lockout. Honored by
  `app/api/auth_routes.py` `_set_session_cookie`.
- `COURSE_GEN_EDITOR_PUBLIC_BASE=https://<DOMAIN>` — in-editor URLs
  (`learner_studio_service.py`) become https; avoids mixed-content /
  insecure code-server links.

No source changes: grep confirmed no hardcoded `http://18.236...` in
`app/` templates/JS; both couplings are env-driven.

### Procedure (run on unfreeze, in this order)

1. Confirm DNS: `dig +short <DOMAIN>` → `18.236.242.248`.
2. Open :443 in the EC2 SG; confirm 80 still open.
3. Issue cert + auto-rewrite nginx (keeps :80 serving during issuance):
   `sudo certbot --nginx -d <DOMAIN> --redirect -m <ops-email>
   --agree-tos -n`
   - `--redirect` adds the 80→443 redirect; certbot installs the
     renewal timer (`systemctl status certbot-renew.timer`).
4. Verify nginx still proxies BOTH the app and the editor over TLS:
   - `curl -sI https://<DOMAIN>/login` → 200
   - `curl -sI https://<DOMAIN>/static/lms.js` → 200
   - editor path: open a live `/editor/<port>/` over https.
   Ensure the `/editor/<port>/` `location` block survived the certbot
   rewrite (it edits the default server block; re-add the editor proxy
   to the 443 server if missing).
5. Flip the two env vars (`SESSION_COOKIE_SECURE=true`,
   `COURSE_GEN_EDITOR_PUBLIC_BASE=https://<DOMAIN>`) →
   `sudo systemctl restart course-gen-codex.service`.
6. Smoke (the standard block, but https): login, static assets,
   `/v1/tutor/*` 401, a full register→enroll→editor round-trip; verify
   the session cookie shows `Secure`; no mixed-content console errors;
   editor opens over https.

### Rollback

- App: revert the two env vars → restart (instant; back to working
  http behavior).
- Edge: certbot kept a backup of the nginx config
  (`/etc/nginx/…*.bak` / `certbot rollback`); `sudo nginx -t &&
  sudo systemctl reload nginx`. Cert files are inert if unreferenced.
- DNS/SG changes are independently reversible.

### Notes

- One cert covers app + editor (same host, nginx path-proxy) — no
  per-editor-port cert needed.
- Renewal is automatic (certbot timer); no app redeploy on renew.
- After cutover, update the smoke-test base in this runbook and any
  `COURSE_GEN_EDITOR_PUBLIC_BASE` references from
  `http://18.236.242.248` to `https://<DOMAIN>`.
