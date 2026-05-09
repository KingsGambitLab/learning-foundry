# Course Gen Codex

Explicit-spec assignment generation MVP for hands-on engineering projects.

This project turns the design we discussed into a runnable FastAPI backend with:

- a versioned design catalog for package types, domain packs, and overlays
- a richer learner-ready assignment spec for bounded, production-ready systems
- deterministic design inference from a high-level brief into explicit course structure, runtime dependencies, capabilities, and assessment strategy
- OpenAI-backed course planning with deterministic fallback
- OpenAI-backed task-agent authoring with deterministic fallback
- business validation for task-agent specs
- module gate computation for progressive assignments
- durable workflow runs backed by SQLite for local development
- editable HIL-gated assignment generation runs
- Dockerized sandbox verification for generated task-agent assignments
- persistent per-run workspaces with shared starter runtimes and runnable FastAPI starter wrappers
- LangGraph-driven authoring/reviewer loops with repair retries and review summaries
- sample course-pattern mappings for the current catalog
- a complete example spec for a support-triage agent
- a browser-based intake-first author page at the root URL for goal + outcomes -> course draft creation

## What this MVP covers

### Supported design space

- bounded service workflows with learner-editable starter code
- retrieval and grounded-answer systems over visible corpus fixtures
- stateful backend services with progressive or survey course packaging
- production overlays for observability, SLOs, freshness, and review-required flows

### Overlays

- `productionization_overlay`
- `scale_slo_overlay`
- `freshness_overlay`
- `adversarial_overlay` (marked later)

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

The root page starts empty and is focused on one flow: write a goal, add outcomes, and create a course draft. It does not preload catalog content anymore. Once a draft exists, it surfaces linked assignment workflow state, loop budgets, attempts used, and blockers.

Real workflow creation is not instant anymore: task-agent runs now prepare a persistent workspace and run Docker-backed compile + endpoint smoke verification before review opens, so a live `POST /v1/workflow-runs` can take tens of seconds.

## Docker note

The live app expects Docker for real assignment sandbox verification. The sandbox now verifies that each generated starter compiles, boots, serves `/health`, and answers `/run`, `/runs/{id}`, `/trace/{id}`, `/approve/{id}`, and `/eval` before author review opens. You can inspect availability with:

```bash
curl http://127.0.0.1:8000/v1/sandbox/status
```

The sandbox runner now reuses Docker images when the generated assignment workspace content is unchanged, so repeated reviewer passes and manual reruns are much faster than cold builds.

## Live course generation

The app can plan a course from a brief in two modes:

- **Live OpenAI planning** when the OpenAI SDK and API key are available
- **Deterministic fallback planning** when the live planner is unavailable or errors

Optional environment:

```bash
export COURSE_GEN_OPENAI_ENV_FILE=/absolute/path/to/openai.env.keys
```

The env file can contain:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4
```

You can optionally override the course planner model independently:

```bash
export COURSE_GEN_OPENAI_PLANNER_MODEL=gpt-5.4
```

## OpenAI task-agent authoring

Task-agent workflow creation customizes the deterministic scaffold with OpenAI before the LangGraph authoring and reviewer nodes run when OpenAI is configured.

Optional environment:

```bash
export COURSE_GEN_OPENAI_ENV_FILE=/absolute/path/to/openai.env.keys
```

The env file can contain:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4
```

Check status with:

```bash
curl http://127.0.0.1:8000/v1/task-agent-authoring/status
```

## Test

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

## Useful endpoints

- `GET /health`
- `GET /v1/sandbox/status`
- `GET /v1/task-agent-authoring/status`
- `GET /v1/registry`
- `POST /v1/designs/infer`
- `GET /v1/course-patterns`
- `GET /v1/course-patterns/{course_slug}`
- `GET /v1/course-generation/status`
- `POST /v1/course-runs/generate`
- `GET /v1/examples/task-agent/support-triage`
- `GET /v1/examples/task-agent/support-triage/submission`
- `POST /v1/specs/task-agent/validate`
- `POST /v1/specs/task-agent/gates/{module_id}`
- `POST /v1/specs/task-agent/grader-plans`
- `POST /v1/specs/task-agent/grader-plans/{module_id}`
- `POST /v1/specs/task-agent/grade/{module_id}`
- `POST /v1/specs/task-agent/grade-live/{module_id}`
- `POST /v1/workflow-runs`
- `GET /v1/workflow-runs`
- `GET /v1/workflow-runs/{run_id}`
- `GET /v1/workflow-runs/{run_id}/nodes`
- `GET /v1/workflow-runs/{run_id}/review`
- `GET /v1/workflow-runs/{run_id}/workspace`
- `GET /v1/workflow-runs/{run_id}/workspace/file?path=public/starter/module_1/app.py`
- `POST /v1/workflow-runs/{run_id}/nodes/execute`
- `GET /v1/workflow-runs/{run_id}/grader-plans`
- `GET /v1/workflow-runs/{run_id}/grader-plans/{module_id}`
- `POST /v1/workflow-runs/{run_id}/grade/{module_id}`
- `POST /v1/workflow-runs/{run_id}/grade-live/{module_id}`
- `POST /v1/course-runs`
- `GET /v1/course-runs`
- `GET /v1/course-runs/{course_run_id}`
- `GET /v1/course-runs/{course_run_id}/events`
- `GET /v1/course-runs/{course_run_id}/review`
- `POST /v1/course-runs/{course_run_id}/sync`
- `POST /v1/course-runs/{course_run_id}/publish`
- `POST /v1/course-runs/{course_run_id}/materialize`
- `GET /v1/course-runs/{course_run_id}/bundle`
- `GET /v1/course-runs/{course_run_id}/bundle/file?path=public/README.md`
- `PUT /v1/workflow-runs/{run_id}/task-agent-spec`
- `POST /v1/workflow-runs/{run_id}/decisions`
- `GET /v1/workflow-runs/{run_id}/events`
- `POST /v1/workflow-runs/{run_id}/materialize`
- `GET /v1/workflow-runs/{run_id}/bundle`
- `GET /v1/workflow-runs/{run_id}/bundle/file?path=public/README.md`

## Example design inference

```bash
curl -X POST http://127.0.0.1:8000/v1/designs/infer \
  -H "content-type: application/json" \
  -d '{
    "title": "Feature flag control plane",
    "problem_statement": "Build a service that evaluates feature flags, supports gradual rollouts, records audit trails, and ships with production-ready checks.",
    "learning_outcomes": [
      "rollout evaluation",
      "safe configuration updates",
      "observability",
      "auditability"
    ]
  }'
```

## Example brief-to-course flow

1. Check whether live generation is available:

```bash
curl http://127.0.0.1:8000/v1/course-generation/status
```

2. Create a course draft from a goal and outcomes:

```bash
curl -X POST http://127.0.0.1:8000/v1/course-runs/generate \
  -H "content-type: application/json" \
  -d '{
    "goal": "Build a production-ready feature flag control plane with rollout targeting, audit logs, and safe configuration updates.",
    "learning_outcomes": [
      "rollout evaluation",
      "safe updates",
      "observability"
    ]
  }'
```

## Example validation flow

1. Infer a starter project contract:

```bash
curl -X POST http://127.0.0.1:8000/v1/designs/infer \
  -H "content-type: application/json" \
  -d '{
    "title": "Feature flag control plane",
    "problem_statement": "Build a service that evaluates feature flags, supports gradual rollouts, and records audit trails."
  }'
```

2. Validate it:

```bash
curl -X POST http://127.0.0.1:8000/v1/specs/task-agent/validate \
  -H "content-type: application/json" \
  -d @support-triage.json
```

3. Compute the active gate for a module:

```bash
curl -X POST http://127.0.0.1:8000/v1/specs/task-agent/gates/module_4 \
  -H "content-type: application/json" \
  -d @support-triage.json
```

4. Expand the deterministic grader plan for one module or the full ladder:

```bash
curl -X POST http://127.0.0.1:8000/v1/specs/task-agent/grader-plans/module_5 \
  -H "content-type: application/json" \
  -d @support-triage.json

curl -X POST http://127.0.0.1:8000/v1/specs/task-agent/grader-plans \
  -H "content-type: application/json" \
  -d @support-triage.json
```

5. Fetch a passing evidence submission and grade it:

```bash
curl http://127.0.0.1:8000/v1/examples/task-agent/support-triage/submission

curl -X POST http://127.0.0.1:8000/v1/specs/task-agent/grade/module_8 \
  -H "content-type: application/json" \
  -d '{
    "spec": ...,
    "submission": ...
  }'
```

6. Run the black-box live grader against a learner-hosted app:

```bash
curl -X POST http://127.0.0.1:8000/v1/specs/task-agent/grade-live/module_8 \
  -H "content-type: application/json" \
  -d '{
    "spec": ...,
    "live": {
      "base_url": "http://127.0.0.1:8011"
    }
  }'
```

## Example workflow flow

1. Create a generation run:

```bash
curl -X POST http://127.0.0.1:8000/v1/workflow-runs \
  -H "content-type: application/json" \
  -d '{
    "intake": {
      "title": "Feature flag control plane",
      "problem_statement": "Build a service that evaluates feature flags, supports gradual rollouts, and records audit trails.",
      "learning_outcomes": ["rollout evaluation", "safe updates", "observability", "auditability"]
    }
  }'
```

2. Inspect the run:

```bash
curl http://127.0.0.1:8000/v1/workflow-runs/<run_id>
```

3. Inspect the LangGraph node executions and loop summary:

```bash
curl http://127.0.0.1:8000/v1/workflow-runs/<run_id>/nodes
curl http://127.0.0.1:8000/v1/workflow-runs/<run_id>/review
```

4. Save an edited task-agent draft:

```bash
curl -X PUT http://127.0.0.1:8000/v1/workflow-runs/<run_id>/task-agent-spec \
  -H "content-type: application/json" \
  -d @edited-spec.json
```

5. Approve a gate:

```bash
curl -X POST http://127.0.0.1:8000/v1/workflow-runs/<run_id>/decisions \
  -H "content-type: application/json" \
  -d '{"gate":"gate_1_spec_review","decision":"approve"}'
```

6. Review the event stream:

```bash
curl http://127.0.0.1:8000/v1/workflow-runs/<run_id>/events
```

7. Preview the grader plans derived from the workflow draft:

```bash
curl http://127.0.0.1:8000/v1/workflow-runs/<run_id>/grader-plans
curl http://127.0.0.1:8000/v1/workflow-runs/<run_id>/grader-plans/module_4
```

8. Grade a learner submission against the workflow draft:

```bash
curl -X POST http://127.0.0.1:8000/v1/workflow-runs/<run_id>/grade/module_8 \
  -H "content-type: application/json" \
  -d @submission.json
```

9. Or run the workflow against a live learner app URL:

```bash
curl -X POST http://127.0.0.1:8000/v1/workflow-runs/<run_id>/grade-live/module_8 \
  -H "content-type: application/json" \
  -d '{"base_url": "http://127.0.0.1:8011"}'
```

10. Materialize the generated assignment bundle:

```bash
curl -X POST http://127.0.0.1:8000/v1/workflow-runs/<run_id>/materialize \
  -H "content-type: application/json" \
  -d '{"overwrite": true}'
```

11. Inspect the bundle manifest or a generated file:

```bash
curl http://127.0.0.1:8000/v1/workflow-runs/<run_id>/bundle
curl 'http://127.0.0.1:8000/v1/workflow-runs/<run_id>/bundle/file?path=public/README.md'
```

## Example course workflow

1. Create a course draft from a known pattern:

```bash
curl -X POST http://127.0.0.1:8000/v1/course-runs \
  -H "content-type: application/json" \
  -d '{"pattern_slug": "tusharbisht-cs-demo-agent-to-production"}'
```

2. Or create a custom survey course with explicit module design specs:

```bash
curl -X POST http://127.0.0.1:8000/v1/course-runs \
  -H "content-type: application/json" \
  -d '{
    "title": "Backend Systems Survey",
    "summary": "A survey course across independent backend assignments.",
    "package_type": "survey_course",
    "modules": [
      {
        "module_slug": "tinyurl",
        "title": "TinyURL",
        "summary": "Build a URL shortener with collision resistance and concurrency safety.",
        "design_spec": {
          "course_structure": {
            "package_type": "survey_course",
            "workspace_scope": "per_module_workspace",
            "progression_mode": "independent_modules",
            "shared_codebase": false
          },
          "runtime_dependencies": {
            "execution_surface": "http_service",
            "editable_files": ["app.py"],
            "visible_fixture_files": [],
            "local_run_command": "python -m uvicorn app:app --host 127.0.0.1 --port 8000",
            "visible_check_command": "python checks/run_visible_checks.py",
            "preview_command": "python -m uvicorn app:app --host 127.0.0.1 --port 8000"
          },
          "capabilities": {
            "retrieval_mode": "none",
            "answer_synthesis_required": false,
            "citations_required": false,
            "abstention_required": false,
            "tool_use_required": false,
            "traceability_required": true,
            "durable_state_required": true,
            "approval_flow_required": false
          },
          "assessment_strategy": {
            "public_checks_required": true,
            "hidden_grader_required": true,
            "cumulative_module_gates": false,
            "learner_submission_enabled": true
          },
          "risk_class": "standard",
          "domain_pack": null,
          "overlays": []
        }
      },
      {
        "module_slug": "feature-flags",
        "title": "Feature flag control plane",
        "summary": "Build a feature flag service with rollouts, audit trails, and observability.",
        "domain_pack_hint": null,
        "overlays_hint": ["productionization_overlay"]
      }
    ]
  }'
```

3. Inspect, sync, and publish the course draft:

```bash
curl http://127.0.0.1:8000/v1/course-runs/<course_run_id>
curl http://127.0.0.1:8000/v1/course-runs/<course_run_id>/events
curl http://127.0.0.1:8000/v1/course-runs/<course_run_id>/review
curl -X POST http://127.0.0.1:8000/v1/course-runs/<course_run_id>/sync
curl -X POST http://127.0.0.1:8000/v1/course-runs/<course_run_id>/publish
```

Survey courses create one assignment workflow run per module. Progressive courses create one shared assignment workflow run and attach a module ladder on top of it.

4. Materialize the course authoring bundle:

```bash
curl -X POST http://127.0.0.1:8000/v1/course-runs/<course_run_id>/materialize \
  -H "content-type: application/json" \
  -d '{"overwrite": true}'

curl http://127.0.0.1:8000/v1/course-runs/<course_run_id>/bundle
curl 'http://127.0.0.1:8000/v1/course-runs/<course_run_id>/bundle/file?path=public/content/syllabus.md'
curl 'http://127.0.0.1:8000/v1/course-runs/<course_run_id>/bundle/file?path=public/content/review.md'
```

Course bundles contain:

- public `README.md`
- public `content/syllabus.md`
- public `content/review.md`
- public `content/module_sequence.md`
- public `content/modules/<module_slug>.md`
- private `course_snapshot.json`
- private `module_index.json`
- private `linked_workflow_runs.json`

Generated artifacts are written under:

```text
generated/<run_id>/
  public/
    content/
      module_1.md
      module_1_grading.md
      ...
    starter/
      module_1/
      ...
  private/
    grader_plans/
      index.json
      module_1.json
      ...
  manifest.json
```

## Reference learner app

A tiny feature-flag learner app can be created locally for black-box smoke tests:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8011
```

## Project shape

```text
app/
  api/routes.py
  domain/registry.py
  domain/task_agent.py
  services/course_patterns.py
  services/intake_router.py
  services/spec_validation.py
  main.py
tests/
  test_api.py
```
