# Autonomous bug fix loop — 2026-05-15

**Scope:** Work through the 32-item bug list from the live RAG/CRAG smoke
PLUS the "harness never repaired" investigation. No API token spends —
mock LLM calls using real data captured in `logs/course_generation.log`
and `data/course_gen.db`.

**Workflow per item:** identify root cause from data → fix root cause →
test using mock data from logs/db → commit. Don't assume — look at the
real captured data.

## Harness-repair investigation (priority 0)

`grep -c grader_repair logs/course_generation.log` returns **0**. Same
for `starter_repair`. The repair nodes ran (the graph dispatcher routes
to them) but never emitted a single observable event.

### Findings

1. **`node_grader_repair` calls `oracle_author.author_oracle(spec)` with
   NO failure findings.** The docstring explicitly says
   `TODO(wave 5): pass validation findings into oracle_author for repair`.
   So every "repair" is a re-roll of the dice: the same spec goes in,
   the LLM tries again, statistically might produce something
   different — but there's no guidance from the previous failure.
   - File: `app/services/langgraph_outcome_graph.py:541-568`
   - Smoking-gun confirmation: 734 oracle_authoring_attempt_failed
     events (error_kind=validation), but no repair-attempted event.

2. **`OracleAuthoring.author_oracle` signature is `(self, spec)` — no
   `failure_context` parameter.** The grader-repair callsite has nowhere
   to pass findings even if it wanted to.
   - File: `app/services/oracle_authoring.py:459`

3. **`node_starter_repair` DOES build a `failure_context` dict** with
   findings + boot_result. Wires it via
   `repo_author.generate_bundle(spec=..., failure_context=...)`. The
   `OpenAIOutcomeRepoAuthor` adapter passes it to
   `build_outcome_repo_author_payload` so the LLM sees it. But the
   `DeterministicStarterShellFallback` does `del failure_context`,
   discarding it.
   - Files: `app/services/langgraph_outcome_graph.py:384-421`,
     `app/services/outcome_repo_author_adapter.py:94-100,191-220`

4. **Neither repair node calls `log_coursegen_event`.** No observability
   on (a) that repair was attempted, (b) what context was passed,
   (c) whether repair produced different output than the prior attempt.

### Fixes to land in this loop

- **F1.** Add a `failure_context: dict | None` parameter to
  `OracleAuthoring.author_oracle` so `node_grader_repair` can pass
  validation findings.
- **F2.** Thread that failure_context into the oracle authoring system
  prompt as a repair section (the LLM sees prior failures + the gap to
  close).
- **F3.** Emit `node_starter_repair_invoked` / `node_grader_repair_invoked`
  events with the findings count + a hash of the prior bundle's source
  so we can detect repair no-ops in the log.
- **F4.** Patch `node_grader_repair` to call `author_oracle(spec,
  failure_context=...)`.

### Mock test plan

Real-data fixtures:
- `tests/fixtures/oracle_authoring_validation_failures.json` — the
  73-blocking-reasons report we saw on `course_f918e889a33c`.
- `tests/fixtures/oracle_authoring_recorded_responses.json` — sample
  LLM-emit shapes captured from the logs.

Unit test: feed the failure report into `author_oracle(spec,
failure_context=...)`, verify the prompt sent to the (mocked) router
contains the failure findings, verify the returned bundle changes
shape vs the no-context call.

## The 32 bug list (from earlier triage)

Status legend: `[ ]` queued, `[~]` in progress, `[x]` fixed,
`[!]` deferred.

### Planner / spec authoring
- [x] 1. max_tokens 16K→32K (in code from earlier)
- [x] 2. timeout 240→480s (in code)
- [x] 3. dict[str,Any] → JSON-string schema fields for OpenAI strict (in code)
- [ ] 4. `_OutcomePlanPayload` lacks `benchmark` field — currently
       hardcoded sniff of brief text. Plumb properly: add discriminated
       union slot in payload + propagate to spec.
- [x] 5. Sniff now uses brief, not paraphrased payload.goal

### Starter authoring + verify
- [x] 6. workspace_boot parses Dockerfile EXPOSE
- [x] 7. materialize_starter patches verify.sh + httpx in requirements

### Oracle authoring + dataset
- [x] 8. datasets pip installed (deps file change pending)
- [x] 9. Quivr/CRAG HF config hardcoded
- [ ] 10. CRAG `crag_task_1_and_2` config has empty `search_results`
        — switch default to a subset config OR build a smarter loader
        that picks a populated subset.

### Scenario materialization
- [x] 11. materialize_oracle_bundle strips leading `scenarios/`

### Reference impl
- [x] 12. /health endpoint missing — fixed by reference-impl rewrite
- [x] 13. Reference-impl rewrite returns short typed spans
- [x] 14. detect_false_premise rewritten as BM25 + coverage gate

### Rubric library
- [ ] 15. LLM emits wrong rubric kwarg names — needs prompt fix +
        kwarg normalization layer in `_build_rubric`.
- [ ] 16. Two different path-resolution conventions (oracle_set_overlap
        wants no `setup_data.` prefix, llm_judge_* wants the prefix) —
        unify the convention.
- [x] 17. Trace body shape: `body:` at TraceStep level — fixed in
        curation; prompt also needs the constraint
- [ ] 18. Trace captures don't have `request` info — document
        constraint in scenario prompt OR add a `request` field to
        capture entry
- [ ] 19. BehavioralEquivalence.expected is literal — extend rubric
        to support path-vs-path comparison (or document the limitation
        in the scenario prompt)
- [x] 20. `_PERCENT_RE` trailing `\b` bug (in reference impl) — fixed
- [x] 21. `_COUNT_RE` requires immediate unit — fixed
- [ ] 22. Trivial-rubric warning fires too aggressively — scenarios
        with only structural rubrics + a `numeric_range` get flagged

### Grader-repair loop (architecture)
- [ ] 23. `node_grader_repair` re-authors → `materialize_oracle_bundle`
        WIPES `_setup/_reference/scenarios/`. Curated assets lost.
        Either: (a) make materialize-or-not toggleable when failure
        context indicates "human curated", or (b) repair LLM emits a
        DIFF rather than full re-write.

### Gate routes
- [x] 24. Gate enum names (was confusion from old POSTs; documented)

### State machine resume
- [x] 25. coursegen_resume.py script lets blocked courses re-run

### Setup_data loader
- [ ] 26. `.jsonl` left as raw text — loader should also parse jsonl
        into a list (and document the convention)

### Trace interpolation
- [ ] 27. `${setup_data.X}` placeholders don't resolve in trace
        bodies — extend interpolator to read setup_data (and
        course_meta) so curated scenarios can interpolate gold data
        into requests without inlining

### Publish
- [x] 28. node_publish doesn't emit publish_snapshot — fixed via
        `outcome_publish_snapshot.build_outcome_publish_snapshot`
- [x] 29. /lms/courses/{id} route was missing — added

### Tokenization
- [x] 30. Apostrophe handling — fixed in reference impl tokenizer

### Title / summary
- [x] 31. course_run.title stuck as truncation placeholder — fixed
        by refreshing from spec.title in _persist_outcome_state
- [x] 32. Planner title/goal don't surface learner skills — prompt
        updated; this course backfilled

## Working order (highest leverage first)

1. **Harness repair** (priority 0 above)
2. **Bug 15** (rubric kwarg drift) — every scenario every run hits this
3. **Bug 16** (path-prefix convention divergence) — same impact as 15
4. **Bug 27** (trace interpolation can't read setup_data) — forces all
   curated data to be inlined, blocking benchmark-grounded scenarios
5. **Bug 23** (grader_repair wipes curated assets) — depends on F1-F4
   from harness fix
6. **Bug 4** (planner payload missing benchmark) — needed to remove the
   hardcoded brief sniff
7. **Bug 22** (trivial-rubric over-firing)
8. **Bug 18** (request not captured) — coupled with 27
9. **Bug 19** (behavioral_equivalence single-value) — needed for
   adversarial scenarios
10. **Bug 26** (jsonl loader)
11. **Bug 10** (CRAG config selection)

## Mock data inventory (for tests)

To avoid token spends:
- `logs/course_generation.log` — every router error, every successful
  spec/oracle authoring (counts, error_kinds, timestamps).
- `data/course_gen.db` — every course_run's serialized state, including
  the LLM-emitted scenarios that broke and the ones that passed.
- `workspaces/outcome/course_f918e889a33c/private/` — the validated
  18-scenario bundle (curated) we know is well-formed.
- `tests/fixtures/` (TBD) — once each fix needs a mock, we capture the
  matching artifact here.

The mock pattern: `RecordedRouter` that takes a list of
`(text_format, response_payload)` tuples captured from previous runs
and serves them in order on each `parse_structured` call. Test
asserts against the input prompt structure (failure_context present,
correct kwarg names in spec, etc.) without ever hitting the real API.
