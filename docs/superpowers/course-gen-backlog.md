# Course-gen pipeline backlog

A running list of generalization fixes for the course-generation pipeline.
Every direct fix made to a specific course is mirrored here as a generic
fix to apply to the upstream pipeline. We pick up these items **after** the
RAG course end-to-end flow is verified working.

Sort order: roughly by impact. Tag with the originating commit / direct
fix so you can trace back what motivated each item.

---

## 1. Rubric path resolver: honor body-shorthand on capture targets

**Status:** DONE in pipeline (commit `94cace43`). Generalized as part of the
direct fix — the rubric resolver itself is the generalization. No further
work needed in the pipeline. Listed for the audit trail.

**Originating direct fix:** Versioned Prompt Eval course was scoring 1/20
because LLM-authored scenarios emitted `target: eval.summary.total_cases`
expecting the body-shorthand convention, but rubrics walked captures
literally. After the fix, score jumped 1/20 → 7/20 with no learner code
touched.

**Done:**
- `expand_capture_shorthand` + `resolve_capture_target` in
  `scenario_rubrics_base.py`
- All 13 rubric call sites routed through it
- 14 regression tests pinning the shorthand rules

---

## 2. Convention audit: LLM-emitted vs runtime expectations

**Status:** Survey task spawned (chip position 1, earlier session).

**Scope:** Find every place where the course-authoring LLM emits paths,
kwargs, or names that don't match what the runtime resolver expects. The
rubric path resolver was one such case; commit `5074b6f7` (bare `{X.Y}`
placeholders) was another. There's almost certainly more.

**Investigate:**
- `_RUBRIC_KWARG_ALIASES` and `_RUBRIC_KWARGS_TO_DROP` in
  `scenario_trace_runner.py` — recent course bundles may need more aliases
- `grader_runner_script_template.py` — does the on-disk runner use the same
  resolver convention?
- Authoring prompts vs schema expectations in `oracle_authoring.py`,
  `spec_authoring.py`

---

## 3. Starter authoring: HTTP status code conventions

**Status:** Survey task spawned (chip position 2, earlier session).

**Originating direct fix:** Versioned Prompt Eval starter returned `400`
for body-validation failures, scenarios expected `422`. Also returned
silent `200` for `promote=false` on a non-pinned version, scenarios
expected `4xx`.

**Generalization:** Teach the starter-authoring LLM REST conventions —
422 for body validation, 409 for state conflicts, 404 for not-found, 400
only for malformed JSON. Either prompt-injected rules or a post-generation
linter.

---

## 4. Shared behavior-string contract: scenario ↔ starter authoring

**Status:** Survey task spawned (chip position 3, earlier session).

**Originating direct fix:** Versioned Prompt Eval scenarios emit
`expected_behavior: "Output should contain X"` but the starter's
behavior evaluator only recognized `contains: X` (explicit colon form).
Every behavior assertion in 15+ scenarios fell through to a meaningless
substring check of the assertion text in the output.

**Generalization:** Pick a canonical behavior-string grammar and make
BOTH authoring stages aware of it. Options: structured spec field
`behaviors: [{kind, args}]`, markdown-bolded names, or strict
`Name — Description` prose.

---

## 5. Scenario state isolation across grading run

**Status:** Design task spawned (chip position 4, earlier session).

**Originating direct fix:** Versioned Prompt Eval scenario
`register_and_get_pinned_initially_null` failed because earlier scenarios
in the same grading container had already pinned a version. Globals
persist; scenarios are not order-independent.

**Generalization options:**
- (a) Boot fresh container per scenario (~20× cost)
- (b) Authoring guideline: scenarios must reset state in setup
- (c) Admin `/__reset__` endpoint the runner calls between scenarios
- (d) Unique entity ids per scenario (collision-free by construction)

---

## 6. Canonical skill-bullet format in course summaries

**Status:** Survey task spawned (chip position 1, current session).

**Originating direct fix:** BM25 course summary used numbered bullets with
`-` as the name/description separator AND had a trailing paragraph after
the bullets. The Versioned Prompt Eval course used dash bullets with no
separator. Two courses, two formats. The JS scorecard parser had to grow
heuristics to handle both. commit `d81a9f5a`.

**Generalization:** Pick one canonical format (e.g., `- <Name> — <Description>`
with Name ≤6 words) and constrain the authoring LLM to emit it. Or move
skills out of the prose summary into a structured `LearnerCoursePackage.skills`
list and render whatever prose the UI wants.

---

## 7. Starter authoring is producing over-complete starters

**Status:** Backlog. Direct fix shipped for the BM25 course as 3 demo
states (skeleton / partial / good).

**Originating direct fix:** The BM25 course's authored starter at
`workspaces/outcome/course_f918e889a33c/public/starter/app.py` is 220 lines
of working extraction logic. It scores **17/18** out of the box — the
learner has almost nothing to implement. By contrast, the calibration
target is:
- Skeleton (initial state): fails all/most graders (0-3/18)
- Partial implementation: 5-10/18
- Good solution: ≥15/18 (panel turns green)

**Generalization:** The starter-authoring prompt currently emits something
near a full reference implementation. Either:
- (a) Add a "leave key functions as TODO stubs" rule to the prompt, with
  examples of which kinds of functions to stub (the SCORING / EXTRACTION
  layer for RAG courses, NOT the request validation layer).
- (b) Define `StarterType.partial` more strictly in the spec — pick a
  target opening-score band (e.g., 15-25% passing) and have the authoring
  loop verify the starter falls in that band before publish.
- (c) Generate the starter as a "deleted" copy of the full reference impl:
  produce the full impl first, then strip 60-70% of the meat from the
  most-pedagogically-relevant functions, leaving signatures + TODOs.

(c) is the cleanest because the upstream tests still pin the contract; the
delete-and-stub step is mechanical.

---

## 9. Scenarios don't validate the skills the course advertises

**Status:** Backlog. Surfaced while comparing the V3 BM25 starter (80
lines of `set(question) & set(sentence)`) against the course's
advertised skill set (BM25 IDF, FAISS, Pinecone/pgvector, span
extraction).

**Originating direct fix:** None — this is a course-design gap.

**The pattern:** A learner can pass the BM25 RAG course's 18 scenarios
with boolean set intersection retrieval. None of the advertised
techniques are required:

- BM25 IDF/k1/b — not needed; corpora are 1-4 passages
- FAISS / dense embeddings — not needed; lexical overlap is enough
- Pinecone/pgvector — not needed; no vector store touched
- Span extraction by question intent — not needed; whole-sentence
  responses pass under `strictness: lenient`

The scenarios test the I/O contract (response shape, citation recall
≥50%, abstain on out-of-scope) but not the retrieval/extraction
quality. The course markets skills the grader doesn't enforce.

**What would close the gap (per skill):**

| Skill | Scenario shape that would enforce it |
|-------|--------------------------------------|
| BM25 | 20+ passages sharing the question's keywords, gold disambiguated only by IDF weighting; assert `min_precision=0.8` not `min_recall=0.5` |
| Embeddings/FAISS | Question paraphrased with zero lexical overlap to gold passage (synonyms, abbrevs); assert correct retrieval despite zero overlap |
| Span extraction | `strictness: strict` on LLM judge so sentence-with-cruft answers fail |
| Citation grounding | `min_precision` assertion, not just recall — over-citation should fail |

**Generalization:** The scenario-authoring stage needs to ground each
scenario in a SPECIFIC skill from `spec.learning_path` and verify that
naive baselines fail it. Concrete option:

(a) After scenario authoring, run the scenario set against a
    deliberate baseline (set-overlap retriever, ~20 lines). Any
    scenario the baseline passes either:
    - Gets re-authored harder (more distractors / paraphrased question / tighter rubric)
    - Gets reclassified as a `smoke_test` and excluded from skill-bar scoring

(b) Have the scenario-authoring prompt receive the spec's advertised
    skills and explicitly require each scenario to be "the kind of
    test that fails if the learner only does <weak baseline>".

Without one of these, the scorecard greenlights starters that don't
demonstrate the skills the course is sold on.

---

## 8. Scenario sets have binary difficulty cliffs

**Status:** Backlog. Surfaced while calibrating the BM25 RAG starter.

**Originating direct fix:** None — couldn't make this course's scenarios
hit the user's intended "partial 5-10/18" band. The course has three
score levels:

- 3/18 — skeleton returns wrong shape; only Pydantic-validated malformed
  inputs pass
- 15/18 — any reasonable retrieval (passage-level overlap is enough); the
  LLM-judge rubrics abstain in the no-router env, which inflates the
  middle of the curve
- 18/18 — retrieval + abstention

There's no natural 5-10 band because:
- `oracle_set_overlap` uses `min_recall=0.5` — tolerates over-citation
  (citing all passages always passes)
- `llm_judge_semantic_eq` abstains without an LLM router, so wrong
  answers don't fail
- `schema_match` is binary (shape OR not), no partial credit

The user expected a smooth learner-progression gradient. The scenarios
as authored don't produce one.

**Generalization options:**
- (a) Authoring guideline: scenarios should be authored at multiple
  precision tiers. Some rubrics tight (literal_match), some loose
  (recall threshold). Tighten 2-3 scenarios per category so partial
  implementations land mid-band.
- (b) Spec stage emits a target difficulty distribution (e.g., 30%
  scenarios passable with response-shape only, 40% need correct
  retrieval, 30% need full feature). Authoring verifies this before
  publish.
- (c) Accept that with abstained-LLM-judges, the scoring will always
  cliff, and document the calibration assumes a live judge.

---

## How to add to this list

When making a direct course fix:

1. Implement the per-course fix
2. Add a section here following the template:
   ```
   ## N. <Short title>
   **Status:** Backlog | Spawned | DONE in pipeline
   **Originating direct fix:** <course id> + <what broke>
   **Generalization:** <what to change in course-gen>
   ```
3. Where to fix lives in `app/services/{spec,oracle,starter,scenario}_authoring.py`
   for authoring-side concerns or `app/services/scenario_rubrics_*.py`
   for runtime/grader concerns
