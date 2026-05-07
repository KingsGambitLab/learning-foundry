# UI / UX Agent Handoff

## What this project is

This repo is building an end-to-end **course generation and delivery system**.

At a high level, the product flow is:

1. Author enters a goal + learning outcomes.
2. The system generates a course draft.
3. The course draft creates one or more assignment workflows.
4. Those workflows go through authoring/review, including Docker-backed verification.
5. The approved course is published into an immutable learner-facing snapshot.
6. Learners enroll through a lightweight LMS view.
7. Learners open a cloud VS Code workspace, run visible checks, submit work to the real hidden grader, and unlock the next module.

The system is already functional. This handoff is **only for improving the UI/UX**, not changing the underlying generation, review, publish, grading, or LMS business logic.

## Product surfaces

### 1. Author surface
- Route: `/create-course`
- Primary job:
  - create a course
  - monitor draft progress
  - review pending gates
  - understand what is blocked and what happens next

Main files:
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/templates/dashboard.html`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/dashboard.js`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/dashboard.css`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/app-shell.css`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/dashboard_page.py`

### 2. Learner / LMS surface
- Route: `/`
- Primary job:
  - browse published courses
  - enroll
  - understand current module
  - open workspace
  - run visible checks
  - submit for grading
  - see progression

Main files:
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/templates/lms_home.html`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/lms.js`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/lms.css`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/app-shell.css`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/lms_page.py`

### 3. API docs surface
- Route: `/docs`
- Not the main focus unless you need minor presentational cleanup.

Files:
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/templates/docs.html`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/docs.js`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/docs.css`

## How the system works conceptually

This context matters so the UI uses the right mental model.

### Authoring side
- Course generation is model-backed and can fall back deterministically.
- Assignment workflows are LangGraph-backed.
- Assignment authoring/review includes Docker sandbox verification.
- Some workflow stages are waiting-on-human gates.
- Progressive courses use one shared assignment workflow with checkpoint/module progression layered on top.

### Learner side
- Learners do **not** see private workflow/spec/grader internals.
- Learners see:
  - module writeup
  - starter workspace
  - visible checks
  - submit-for-grading
- Hidden grading is deeper than visible checks.
- Publish snapshots are immutable and are the learner-facing source of truth.

## What the UI should optimize for

### For authors
The author cares most about:

1. **What is the current state?**
2. **Is it waiting on me or the agent?**
3. **Why is it blocked?**
4. **How do I unblock it?**
5. **What has already been approved?**
6. **What is the next step after this one?**

The page should feel like a **focused review workspace**, not a system dashboard.

### For learners
The learner cares most about:

1. What am I supposed to build?
2. Which files should I edit?
3. How do I test locally?
4. What happens when I submit?
5. What unlocks next?

The learner page should feel like a **calm module workspace**, not an internal platform console.

## Known UX issues to improve

These are recurring issues we have already seen:

### Author UI issues
- Too much backend/state language leaks into the page.
- Review pages can feel empty or under-contextualized.
- Important actions are sometimes visible before the review context is understandable.
- The draft page can feel like a long dashboard instead of one primary task.
- Sticky areas have sometimes been too bulky or visually awkward.
- Progress/status information can overpower the thing the author is actually reviewing.
- The page should always preserve route state when a draft is selected.

### Learner UI issues
- Module/task descriptions must be concrete and readable.
- Learners need the workspace/test/submit flow to be obvious.
- The page should clearly separate:
  - visible checks for debugging
  - hidden grader on submit
- Module progression should feel understandable, not mysterious.

## Editable files

You may freely edit these:

### Preferred edit surface
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/templates/dashboard.html`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/templates/lms_home.html`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/dashboard.js`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/dashboard.css`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/lms.js`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/lms.css`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/static/app-shell.css`

### Thin presentation-layer files
Only if needed for view wiring or template state:
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/dashboard_page.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/lms_page.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/main.py`

### Tests you may update
- `/Users/tushar/Desktop/codebases/course-gen-codex/tests/test_api.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/tests/test_learner_studio_service.py`

## Hard boundary: do not touch the core generation pipeline

Do **not** change the behavior of the generation / review / publish / grading pipeline.

That means **do not edit** these files unless explicitly approved later:

- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/workflow_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/course_generation_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/course_workflow_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/langgraph_assignment_graph.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/openai_course_planner.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/openai_task_agent_authoring.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/task_agent_scaffolds.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/task_agent_workspace_authoring.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/task_agent_repair_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/docker_sandbox_runner.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/task_agent_grader.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/grader_planner.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/publish_snapshot_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/lms_service.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/spec_validation.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/learner_brief_builder.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/artifact_materializer.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/course_artifact_materializer.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/task_agent_blackbox_runner.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/services/assignment_workspace_manager.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/api/routes.py`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/domain/`
- `/Users/tushar/Desktop/codebases/course-gen-codex/app/storage/`

### Why this boundary exists
We want the UI/UX workstream to improve:
- layout
- hierarchy
- labels
- actions
- clarity
- routing
- progressive disclosure

without accidentally changing:
- workflow decisions
- gating logic
- model/planner behavior
- grading behavior
- publish snapshot structure
- learner progression semantics

If a UI improvement appears to require one of those backend changes, stop and hand that back instead of quietly changing the pipeline.

## Allowed kinds of changes

Good changes:
- change layout structure
- improve spacing, hierarchy, density, typography
- rename labels for clarity
- add accordions / expanders / tabs / summaries
- reorder information to match user priority
- improve sticky/header behavior
- improve empty states
- improve button placement and copy
- make route state survive reload
- hide system jargon from the UI
- make review context and actions easier to understand

Avoid:
- inventing fake statuses or placeholder data
- hiding critical actions completely
- changing business rules to make the UI easier
- changing API contracts as a shortcut

## Current UX direction

### Author surface direction
When a draft is active, the page should emphasize:

1. Current state
2. Why it is waiting
3. How to unblock it
4. The actual review object
5. Expandable progress details

The active review step should dominate the page.

### Learner surface direction
The learner page should make this flow obvious:

1. Read the module brief
2. Open / resume workspace
3. Run visible checks
4. Submit for grading
5. See result and unlock next module

## Verification checklist

Before handing back UI changes:

1. Run:
```bash
node --check /Users/tushar/Desktop/codebases/course-gen-codex/app/static/dashboard.js
node --check /Users/tushar/Desktop/codebases/course-gen-codex/app/static/lms.js
```

2. Run:
```bash
/Users/tushar/Desktop/codebases/course-gen-codex/.venv/bin/python -m unittest discover -s /Users/tushar/Desktop/codebases/course-gen-codex/tests -p 'test_*.py'
```

3. Test in browser on:
- `http://127.0.0.1:8010/create-course`
- `http://127.0.0.1:8010/`

4. Check these real scenarios:
- create draft
- reload with selected draft in URL
- open pending review step
- approve / request changes
- browse learner course
- open workspace
- understand visible checks vs submit for grading

## Summary for the UI agent

You are improving a real working system. Your job is to make it easier to understand and use, **without changing the engine underneath it**.

If you stay inside:
- templates
- static JS/CSS
- thin page-render wiring

you’re in the right lane.
