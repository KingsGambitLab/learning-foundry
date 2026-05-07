# Creator-First UI Handoff

This write-up is for the UI agent implementing the next creator flow.

The goal is to move the main author experience onto:

- `POST /v1/course-generation/creator-plan`
- `POST /v1/course-runs/from-creator-plan`

and keep the deep generation spec behind an **admin/debug surface**, not the main creator path.

## Product direction

The creator should experience the product like this:

1. Enter a problem statement.
2. Get suggested outcomes and edit them.
3. Answer a few practical setup questions:
   - scaffolded starter app vs implement from scratch
   - primary database
   - cache backend
   - later: tech stack
4. Review a proposed module plan in plain language.
5. Accept the plan and create a draft.
6. Open the created draft and play with it before publishing.

The creator should **not** be dropped straight into:

- contract schemas
- tool registries
- approval policy internals
- risk classes
- endpoint counts
- raw spec sections

Those belong in an **admin/debug view**.

## Important principle

This is a **presentation/orchestration change**, not a new parallel authoring system.

The creator plan is already backed by the same canonical backend spec and the same downstream generation/review/publish pipeline. The UI should use the new creator-plan surfaces as the primary entry point, rather than exposing the deep spec first.

## Backend surfaces to use

### 1. Suggest outcomes

`POST /v1/course-generation/suggest-outcomes`

Use this for the creator's first assist after they enter a problem statement.

Request:

```json
{
  "goal": "Build a flight booking system that is production ready. Mock external dependent services where required"
}
```

Response:

```json
{
  "source": "openai_live",
  "status": {
    "provider": "openai",
    "available": true,
    "source": "openai_live",
    "message": "..."
  },
  "learning_outcomes": [
    "Design booking workflows that stay correct under concurrency.",
    "Implement safe inventory reservation and release behavior.",
    "Use caching and persistence intentionally in the service design."
  ]
}
```

Notes:
- The backend now normalizes multiline and bullet-style outcomes into separate items.
- The UI should still present outcomes in an editable, creator-friendly list rather than a raw textarea blob.

### 2. Create a creator plan

`POST /v1/course-generation/creator-plan`

This is the **main planning endpoint** for the creator flow.

Request:

```json
{
  "goal": "Build a flight booking system that is production ready. Mock external dependent services where required",
  "learning_outcomes": [
    "Design booking workflows that stay correct under concurrency.",
    "Implement safe inventory reservation and release behavior.",
    "Use caching and persistence intentionally in the service design."
  ],
  "creator_choices": {
    "starter_type": "partial_implementation",
    "primary_database": "postgres",
    "cache_backend": "redis",
    "tech_stack": []
  }
}
```

Response shape:

- `source`
- `status`
- `learning_outcomes`
- `plan`

The important part of `plan` is:

- `title`
- `summary`
- `creator_choices`
- `modules[]`
- `creator_summary`
- `notes`

Each module includes:

- `module_slug`
- `title`
- `summary`
- `learning_outcomes`
- `creator_notes`

This is the object the creator should review and accept.

### 3. Create a draft from the creator plan

`POST /v1/course-runs/from-creator-plan`

Request:

```json
{
  "plan": {
    "...": "the full CreatorCoursePlan returned by /creator-plan"
  }
}
```

This creates the actual draft and feeds it into the existing course-generation/review pipeline.

### 4. Load the creator-facing draft view

`GET /v1/course-runs/{course_run_id}/creator-view`

This is the preferred draft/test-facing payload for the creator once the draft exists.

It includes:

- `course_run`
- `review`
- `published_versions`
- `creator_feedback`
- `latest_learner_evaluation`

Use this as the primary data source for:

- current draft status
- progress
- what is blocked
- what is ready
- learner eval summary

Do not force creators to navigate raw review/debug endpoints first.

### 5. Optional creator feedback

The UI may also use:

- `GET /v1/course-runs/{course_run_id}/feedback`
- `POST /v1/course-runs/{course_run_id}/feedback`

to let creators record friction or review feedback while using the draft.

## Recommended creator flow in the UI

### Step 1: Brief

Show:

- problem statement
- suggested outcomes
- editable outcomes list

Avoid:

- raw JSON
- hidden/internal status language

### Step 2: Practical setup choices

Show simple creator-facing questions:

- starter app or blank implementation
- database choice
- cache choice
- later: stack choice

These should map directly to `creator_choices`.

Suggested labels:

- `How much starter code should learners get?`
- `Which database should the course assume?`
- `Should learners have a cache available?`

### Step 3: Proposed module plan

After `creator-plan`, show a plain-language module ladder.

Each module card should focus on:

- title
- summary
- learning outcomes
- short creator notes

The creator's job here is:

- review the structure
- make sure the sequence feels right
- accept it if aligned

This is the main HIL point for creators before the draft is created.

### Step 4: Draft playground

After `from-creator-plan`, open the draft route and load `creator-view`.

This view should focus on:

- current status
- what is waiting on the creator vs the agent
- what is already approved
- learner evaluation summary
- next step toward publish

This is where creators should be able to inspect and play with the course before publishing.

## What should move behind admin/debug

Keep these out of the main creator flow unless explicitly requested:

- contract frame
- contract schemas
- run state schema
- trace schema
- tool registry
- approval policy internals
- detailed risk labels
- internal checks lists
- raw design spec sections

If needed, expose them through:

- an `Admin` tab
- a `Technical details` drawer
- a developer-only toggle

That surface can use the existing deeper review payloads, but it should not be the default creator experience.

## Strong recommendation for page structure

### Main creator path

Use the UI to create a clear funnel:

1. `Describe the system`
2. `Refine outcomes`
3. `Choose setup options`
4. `Review module plan`
5. `Create draft`
6. `Test before publish`

### Separate admin/debug path

Use a separate, lower-priority surface for:

- deep spec inspection
- workflow internals
- gates/debug
- technical diagnostics

## Files the UI agent will likely touch

Main creator surface:

- `/Users/tushar/Desktop/codebases/course-gen-codex/app/templates/dashboard.html`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/dashboard.js`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/dashboard.css`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/app-shell.css`

Thin wiring only if needed:

- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/dashboard_page.py`

## Hard boundary

The UI agent should **not** change the core generation/review/publish/grading pipeline to make this easier.

In particular, do not change:

- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/course_generation_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/course_workflow_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/workflow_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/langgraph_assignment_graph.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/publish_snapshot_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/lms_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/api/routes.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/domain/`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/storage/`

The backend contract for the creator-first flow already exists. The UI task is to **sit on top of it cleanly**.

## The one-sentence summary

Build the creator experience on top of:

- `suggest-outcomes`
- `creator-plan`
- `from-creator-plan`
- `creator-view`

and push deep spec inspection into an explicit admin/debug surface instead of making it the default author experience.
