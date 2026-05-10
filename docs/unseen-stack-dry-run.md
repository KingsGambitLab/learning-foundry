# Unseen Stack Dry Run

This is the target dry-run flow for a brand-new stack, like a creator choosing a language or framework the platform has never shipped before.

The rule is simple:

> creator owns the stack contract  
> authoring owns the repo and runtime protocol  
> harness executes and judges  
> retries revise the authored artifacts from real failure packets

## Preconditions

The creator chooses and confirms the full stack contract up front:

- implementation language
- language version
- framework
- framework version
- package manager / build tool
- optional primary database
- optional cache backend

That contract is persisted and must not be rewritten later.

## Dry-run node flow

### 1. Intake / creator flow

Responsible for:

- collecting the creator-owned stack contract
- persisting it to the course/workflow records
- passing the exact contract into design + authoring

Must not:

- infer missing database/cache after the fact
- silently swap package managers
- silently downgrade to another language/framework

### 2. Spec authoring

Responsible for:

- assignment intent
- public endpoints
- deliverables
- learner starter surface
- assessment shape

Inputs it needs:

- creator stack contract
- problem statement
- learning goals
- deliverable plan

Must not:

- invent the learner repo shape directly through language templates
- own runtime execution heuristics outside the authored bundle

### 3. Repo authoring

Responsible for authoring the actual starter bundle for one deliverable:

- dependency manifest
- source files
- `Dockerfile`
- `.coursegen/runtime/install.sh`
- `.coursegen/runtime/verify.sh`
- `.coursegen/runtime/run.sh`

Inputs it needs:

- creator stack contract
- materialized starter manifest
- README / learner brief
- public endpoints
- current authored files on retries
- failure packet on retries

Success criterion:

- the authored bundle contains a real repo and a real runtime protocol

Failure mode:

- if repo authoring leaves default placeholder runtime files or no learner repo files, reviewer/tests must block

### 4. `authoring_runtime`

Responsible for:

- materializing the authored workspace
- running the authored runtime protocol through the sandbox
- producing compile/runtime evidence

Inputs it needs:

- authored starter bundle
- authored `Dockerfile`
- authored runtime scripts

Must not:

- reconstruct install/run commands from language-specific platform templates

### 5. `authoring_tests`

Responsible for:

- writing visible and hidden test scripts against the actual authored starter workspace

Inputs it needs:

- the full current starter workspace
- authored runtime protocol files
- manifest
- learner README
- failure packet on retries

Success criterion:

- tests are real scripts
- visible tests teach
- hidden tests are stronger

### 6. Harness baseline verifier

Responsible for:

- executing the authored tests against baseline workspaces
- proving the tests discriminate correctly

Expected matrix:

- empty repo fails
- untouched starter fails hidden
- partial starter usually fails visible when core behavior is missing
- stronger implementation passes

If this matrix fails, the harness should return a concrete blocker, not let the workflow continue.

### 7. Reviewer nodes

`reviewer_runtime`

- reruns the authored starter through the runtime harness

`reviewer_code`

- checks learner-honest repo shape
- blocks fake starter surfaces

`reviewer_pedagogy`

- checks README / starter clarity
- checks learner-facing consistency

`reviewer_tests`

- blocks if repo bundle is still default
- blocks if runtime protocol is still default
- blocks if tests are still placeholders
- runs baseline verifier

### 8. Repair / retry

`authoring_repair` and `reviewer_repair` should not patch by language.

They should:

- pass the failure packet back to spec + repo + test authoring
- rerun materialization
- rerun authored runtime protocol
- rerun authored tests

Retry should stop early if:

- the same blocker signature survives
- the authored bundle did not materially change

## What the harness should own

The harness should only know the execution protocol:

- where runtime scripts live
- where visible and hidden tests live
- how to run them
- how to collect failure evidence

The harness should not own:

- language-specific starter templates
- language-specific repo skeletons
- language-specific runtime command synthesis as the source of truth

## Current code alignment

The current repo is now aligned on a few important pieces:

- repo authoring can author repo files and runtime protocol files
- test authoring now sees the full current starter workspace
- reviewer/tests block default repo bundles and default runtime protocol bundles
- starter protocol placeholders fail loudly until authoring replaces them

The main remaining architectural gap is that `assignment_design_inference.py` still synthesizes some runtime-plan service metadata like entrypoint hints and container images. Command synthesis is gone; authored runtime protocol files are now the execution source of truth.
