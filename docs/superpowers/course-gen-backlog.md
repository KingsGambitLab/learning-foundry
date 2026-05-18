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

## 10. Benchmark data is loaded but dropped before grading [HIGH PRIORITY]

**Status:** Backlog. Surfaced 2026-05-15 when the user questioned why
a 80-line `set(question) & set(sentence)` retriever scores 17/18 on a
course tagged "BM25 / FAISS / Pinecone" over the Quivr/CRAG benchmark.

**Originating direct fix:** None — this is a pipeline bug, not a
per-course one.

**The flow that's broken:**

1. ✅ Spec selects `Quivr/CRAG` with `max_queries=20`, `use_split=validation`
2. ✅ `benchmark_loader.load_crag_benchmark` fetches the real dataset.
   Proof: `public/examples/sample_queries.json` carries through real
   CRAG `query_id` UUIDs ("4afd82c6-af57-41b2-9848-f0ec9479efd5"),
   real questions ("can you tell me what the lodger title was originally?"),
   and real domain tags (movie / finance / sports).
3. ✅ `oracle_authoring.py:570` calls `_crag_bundle_to_setup_files(hidden_crag)`
   which returns 3 `GeneratedSetupFile` entries:
   `queries.jsonl` / `gold_answers.json` / `search_results_index.json`.
4. ✅ `oracle_authoring.py:1023` merge: if `benchmark_setup_files`
   non-empty, those win over LLM-emitted setup files.
5. ❌ **On disk we see**: `queries.json` (not `.jsonl`),
   `gold_supports.json` (not `search_results_index.json`),
   `gold_answers.json` with **scenario IDs** (`happy_valid_q1`) as
   keys instead of CRAG `query_id` UUIDs. The hidden CRAG bundle
   never reached `_setup/`.
6. ❌ Scenarios were authored against an LLM-invented "Acme Corp" /
   "Globex" world with 1-4 hand-crafted passages each, not against
   real CRAG queries. All `passage_id`s are fictional (`acme_buyback`,
   `gb_fy23_margin`, `qt1`, `m1`).
7. ❌ Grading runs against the toy scenarios. Boolean set intersection
   clears 15-17/18. The advertised techniques are never required.

**Where the bug likely sits** (need to trace, didn't fully verify):

- `node_grader_repair` calls `author_oracle(spec)` again on each
  repair iteration. Possibly `benchmark_setup_files` lost on repair?
- Or `OracleAuthor.author_oracle` returns a result where the merge
  silently bypassed the benchmark bundle (empty queries from a stale
  filter, dataset-load failure that didn't raise, etc.).
- Or `materialize_oracle_bundle` wiping `_setup/` between attempts
  on a path where the second attempt had `benchmark_setup_files=[]`.

**Reproducer:**

  ```bash
  ls workspaces/outcome/course_f918e889a33c/private/grader/_setup/
  # gold_answers.json  gold_supports.json  queries.json
  # (should be: queries.jsonl, gold_answers.json, search_results_index.json)

  python -c "import json; q = json.load(open('.../queries.json')); print(list(q.keys())[:3])"
  # ['happy_valid_q1', 'happy_valid_q2', 'happy_valid_q3']
  # (should be CRAG UUIDs like '4afd82c6-af57-41b2-9848-f0ec9479efd5')
  ```

**Generalization:** add a post-author check in `node_oracle_authoring`
that asserts when `spec.benchmark` is set:
  - `_setup/` contains the loader's expected file names (`queries.jsonl`,
    `search_results_index.json`)
  - Top-level keys in setup files match the benchmark's id namespace
    (UUIDs for CRAG, BeIR ids for BeIR), NOT scenario_ids
  - Each scenario references `setup_data.search_results_index.<id>`
    for retrieval, doesn't inline its own `passage_id`s

Fail publish if these don't hold. The current pipeline silently
permits the LLM to ignore the loaded benchmark.

---

## 11. Calibration check: weak-baseline-runs as a publish gate

**Status:** Backlog. Surfaced during the BM25 RAG course direct-fix
work — once the grader bundle was anchored in real BeIR/fiqa data
(commit `8ec99202`), four implementations were submitted to verify
the difficulty gradient matches the advertised skills:

| Implementation                          | Score | Technique                           |
|-----------------------------------------|-------|-------------------------------------|
| V1 skeleton (placeholder return)        | 4/18  | none                                |
| V3 baseline                             | 5/18  | ``set(question) & set(passage)``    |
| V4 BM25                                 | 7/18  | ``rank_bm25`` Okapi (k1=1.5, b=.75) |
| V5 dense                                | 9/18  | sentence-transformers + FAISS       |

Each advertised technique buys ~2 scenarios over the previous tier.
The course's claim ("BM25 / FAISS / Pinecone") is now backed by a
test set where these techniques actually unlock specific scenarios.

What V5 still fails (the next pedagogical steps):
- 5 happy_path/boundary: retrieval finds the right passage but
  returns the WHOLE passage; judge says "answer extends far beyond
  the primary point". Span extraction is the missing skill.
- 3 out_of_scope: cosine-similarity-threshold abstention doesn't
  fire because hard distractors still have cosine > 0.30. Smart
  abstention (whether the question's specific concept is in any
  passage) is the next step.
- 1 adversarial: reorder + paraphrase breaks even dense retrieval.

**Generalization:** the publish gate should automatically run a
weak-baseline check before any course is marked publishable.
Concretely:

1. After scenarios are authored, materialize a deliberate baseline
   starter (the V3 pattern: 80 lines of bag-of-words intersection).
2. Run the grader against that baseline. Score it.
3. If the baseline scores >= ``starter_target_max`` (e.g. 0.3 of
   scenarios), the scenario set is too easy: either re-author
   harder scenarios or downgrade the spec's advertised skills.
4. If the baseline scores < ``starter_target_min`` (e.g. 0 of
   scenarios), the scenarios are unreachable — likely a
   contract/shape mismatch rather than a hard problem.

This catches both BM25-style courses that secretly need none of
their advertised techniques AND course generations where the
scenarios were authored against a broken spec.

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

## 12. Visible/hidden corpus passage leakage

**Status:** Backlog. Surfaced 2026-05-15 during a passage-overlap audit
of the BM25 RAG course (commit `c84ac7c5`).

**Originating direct fix:** None yet — the audit found leakage; this
item tracks the systematic fix.

**The bug:** The repair scripts pick visible queries and hidden queries
from disjoint `query_id` sets. But each query's 10-passage pool is
selected independently by token overlap with the query, sampled from
the same 57K-passage BeIR/fiqa corpus. Audit shows **2 of 50 visible
passages also appear in the hidden pool**:

  ```
  VISIBLE: 5 queries, 50 unique passages
  HIDDEN:  20 queries, 186 unique passages
  qid overlap:     0  (clean)
  passage overlap: 2  (leak — same passages in visible+hidden)
  ```

That's 4% leakage on visible passages. Marginal for this course, but
the leakage rate scales with topical similarity between visible and
hidden queries. A learner who memorizes those passages could over-fit.

**Generalization:** the visible-sample picker should:
- Track every passage_id allocated to the hidden pool first
- Exclude those passage_ids from the candidate set when picking
  distractors for visible queries
- Assert at the end that ``visible_pids ∩ hidden_pids == ∅`` and
  fail publish otherwise

Implement in ``benchmark_loader.split_crag_for_visibility`` /
``split_beir_for_visibility`` so future benchmark-backed courses get
clean splits by construction.

---

## 13. Visible samples lack scenario-category coverage

**Status:** Backlog. Same audit as #12.

**The bug:** The 18 hidden scenarios span 5 categories — happy_path,
boundary, malformed_input, out_of_scope, idempotency, adversarial.
The 5 visible samples are all "general queries" (~happy_path shape).
A learner can develop and verify retrieval/extraction locally, but
has no visible example for:
- abstention on false-premise / out-of-scope questions
- distractor injection / passage reordering robustness
- malformed-input validation responses

So learners can't iterate on those skills locally — they have to
guess what the hidden grader expects, submit, and iterate against
the scorecard.

**Generalization:** visible-sample picker should include at least one
example per scenario CATEGORY (out_of_scope, adversarial, etc.) with
its expected behavior labeled, so the learner has a worked example
for each skill the grader tests.

---

## 14. Hidden test set is too small relative to available data

**Status:** Backlog. User suggestion 2026-05-15 — "if we have 50
examples from the repo, maybe share 5 with learner and keep 45
hidden for private evaluation".

**The bug:** Current ratio is 5 visible : 20 hidden (1:4). User
suggests 1:9 (5 visible : 45 hidden). Today we use **0.30%** of
BeIR/fiqa's 6,648 queries — there's plenty of headroom.

**Concerns balancing this:**
- Each hidden scenario = ~1 haiku call for the LLM judge = ~$0.0017.
  18 → 45 scenarios = ~$0.08/submit, vs ~$0.03 today. Cost scales.
- Wall-clock: 18 scenarios takes ~30s today (per submit). 45 would be
  ~75s. Still fine.
- Statistical signal: 18 binary trials gives ±12% confidence interval
  around the true pass rate. 45 trials drops it to ±7%. Materially
  more accurate calibration.

**Generalization:** target a 1:9 visible/hidden ratio in the
benchmark-backed authoring path. Make the hidden set size a
spec-level parameter (``spec.benchmark.hidden_query_count``) with
a default of 45 and per-category quotas (e.g. 15 happy + 6 boundary
+ 6 malformed + 6 out_of_scope + 3 idempotency + 9 adversarial).

---

## 20. Course teaches re-ranking, not full-stack RAG (chunking + indexing + retrieval done upstream)

**Status:** Backlog. Surfaced 2026-05-15 when the user asked
"Where did chunking happen? And how was the search-result pool
selected?".

**Originating direct fix:** None — this is a scope/calibration gap
in the course design.

**The gap:** The BM25 RAG course markets five RAG skills:
BM25 retrieval, dense embeddings (FAISS/Pinecone), span extraction,
citation grounding, false-premise abstention. But the dataset
pipeline (FiQA-2018 + our repair script) does TWO of the five
upstream:

  1. **Chunking** — FiQA shipped pre-chunked passages
     (1 Stack Exchange answer = 1 retrievable unit). The learner
     never sees raw documents or decides chunk boundaries.
  2. **First-pass retrieval** — our
     ``scripts/repair_bm25_course_fiqa.py`` scores the full 57K-passage
     corpus by token overlap, then ships only the top 10 per query
     (gold + hard distractors) to the learner. The learner never
     queries a 1K+ corpus.

So when a learner implements "BM25" against 10 in-request passages,
they're not actually exercising production BM25 — they're rebuilding
the index per request over a tiny pool that's already been
ground-truthed by us. The IDF term in BM25 is computed over 10
documents instead of 57K; the document-length normalization is
operating on hand-picked similar-length passages; etc.

Same for FAISS: embedding 10 passages and doing a flat-IP search
is functionally identical to cosine-sort. The FAISS-specific value
(IVF / HNSW indexing, billion-scale ANN, persistence) never enters
the picture.

**What the course actually teaches:** RE-ranking, span extraction,
citation grounding, abstention. Reasonable scope for an
intermediate course; just doesn't match the marketed skills.

**Generalization options:**

(a) Rename/reframe the course. Title becomes "RAG re-ranking +
    extraction + grounding" instead of "BM25 retrieval + FAISS
    indexing". Drop the misleading skills tags.

(b) Expand scope: ship the FULL 57K-passage corpus to the learner
    and have them build the index themselves. First-pass retrieval
    becomes a learner concern. Per-scenario passage pool becomes
    "all 57K passages — you decide which 10 to consider".
    Significantly harder course; would need different scoring
    (recall@K with the learner's own retrieval, not just citation
    recall on a hand-picked pool).

(c) Add a SECOND deliverable: a chunking + indexing prerequisite
    where the learner builds a corpus index over a small documents
    dump, then deliverable 2 (current course) becomes the
    re-rank/extract layer.

(b) and (c) are the honest-to-the-marketing options. (a) is the
honest-to-the-current-scope option.

The pipeline pattern this reveals: when spec authoring lists a
skill, the materializer should mechanically check that the
scenarios test that skill in a way the dataset structure
ENABLES. If FiQA ships pre-chunked passages, "chunking" can't
be a skill in a FiQA-backed course. The spec validator should
reject `learning_path` entries that the chosen benchmark can't
support.

---

## 16. Visible samples ship with empty retrieval pools

**Status:** Backlog. Surfaced 2026-05-15 while answering the user's
question "can a learner run public checks themselves?".

**Originating direct fix:** `scripts/repair_bm25_course_visible_samples.py`
manually populated `public/examples/sample_queries.json` for the BM25
RAG course with real BeIR/fiqa passages.

**The bug:** The CRAG visibility splitter
(`split_crag_for_visibility`) correctly strips `search_results` from
visible queries (so the corpus isn't leaked to learners). But
nothing populates a development pool in its place. Every shipped
visible sample on the BM25 course had `"search_results": []`. A
learner can't develop retrieval against empty arrays — they'd have
to download a benchmark themselves to test locally.

This blocks the entire dev iteration loop the LMS advertises (the
visible checks runner that ships in `public/checks/run_visible_checks.py`
is functionally a no-op without samples to fire).

**Generalization:** For every benchmark-backed course, the authoring
stage must populate `public/examples/sample_queries.json` with a
small (≤5) but COMPLETE per-query retrieval pool — gold passages
plus a few topical distractors, with the labels marked for learner
iteration. The visibility splitter should produce this pool by
construction, not just leave search_results empty.

Cost is trivial: 5 × 10 passages = 50 corpus rows shipped to disk.

---

## 17. Authoring emits citation recall rubric but never the matching precision rubric

**Status:** Backlog. Surfaced 2026-05-15.

**Originating direct fix:** `scripts/repair_bm25_course_citation_precision.py`
added 11 `subset_match` rubrics to the BM25 course scenarios.

**The bug:** The scenario-authoring stage emits `oracle_set_overlap`
(citation recall: gold passages must appear in `body.citations`) but
never the corresponding `subset_match` rubric (citation precision:
every value in `body.citations` must be a `passage_id` present in
the request's `search_results`).

Without precision, a learner can pass by returning every passage_id
in the request (over-citing) — or even by fabricating IDs entirely
(no check that cited values trace back to the request). The course
markets "citation grounding" as a skill but the grader doesn't enforce
the grounding half.

**Generalization:** the scenario-authoring prompt should treat
recall + precision as a PAIR on any field that carries citations.
Concretely: when authoring a scenario with `kind: oracle_set_overlap`
on `body.citations`, the author must also emit
`kind: subset_match  target: <same>  acceptable_source: <step>.request.body.<corpus_field>`
with `min_overlap: 1.0`.

Could be enforced by a post-author validator that scans every
scenario YAML and flags missing precision-pair rubrics before
publish.

---

## 18. Learner-facing docs drift from the spec contract (README / deliverables.md)

**Status:** Backlog. Surfaced 2026-05-15 when the user asked
"Does the grader today check for `cited_chunks: list[str]`?"

**Originating direct fix:** `scripts/repair_bm25_course_readme.py`
rewrote the BM25 course's `README.md` to match the actual contract;
also patched `deliverables.md` and `publish_snapshot.workspace_seed_files`.

**The bug:** The original course shipped a README and `deliverables.md`
with a citation contract that didn't match the code. README promised:

  - `cited_chunks: list[str]` (URL-shaped) — code uses
    `citations: list[str]` (passage_id-shaped)
  - `page_url` / `page_snippet` / `page_result` fields on search_results
    — code uses `passage_id` / `text` / `title` / `source`
  - HTML parsing helpers at `app/utils/html_parsing.py` — file
    doesn't exist

`deliverables.md` had a stale 5-skill list ("Span extraction by...")
that didn't match the actual quality bars or the updated course
summary I patched earlier.

So the learner reading the docs was being lied to. They could try
to implement the documented contract and fail every scenario because
the rubrics check different field names.

**Generalization:** all learner-facing docs should be generated from
the authoritative spec at materialize time, not authored as
free-form prose that can drift:

- `README.md` — generated from `spec.endpoints` (request/response
  schemas), `spec.quality_bars` (the rubrics that will fire), and
  the calibration data (V1 baseline / V5 reference scores).
- `deliverables.md` — generated from `spec.learning_path` (skill
  bullets) + the same quality bars.
- The `cited_chunks` / `page_url` text in the BM25 README looks
  like it was copy-pasted from a different course's template (or
  from an early prompt iteration). Either way it's evidence that
  the authoring loop doesn't validate doc-text against the actual
  spec before publish.

Concrete next step: add a publish-time validator that asserts
every field name appearing in a fenced code block in README.md
also appears in `spec.endpoints[*].request_schema_json` /
`response_schema_json`. Fail publish on mismatch.

---

## 19. README is missing the step-by-step learner journey

**Status:** Backlog. Surfaced 2026-05-15.

**Originating direct fix:** Manually wrote a "How to solve this
assignment (step by step)" section into the BM25 course README,
plus a worked example walking through `sample_queries.json` with a
concrete sample + a 6-line Python dev loop the learner can copy.

**The bug:** A fresh learner who clicks "Open VS Code workspace"
gets dropped into a tree with `README.md` / `project_brief.md` /
`deliverables.md` / `public/` and no instructions on what to read
first, how to iterate locally, or when they're "done". The
implicit assumption is that the learner figures out the journey
themselves; in practice they get stuck at step ~5 (boot service,
realize sample queries are empty, can't develop).

**Generalization:** authoring should always emit a learner-journey
section in the README, with at minimum:

1. Read-order for the orientation files
2. Boot command for the local service
3. Visible-checks command + how to interpret results
4. When to submit + what the scorecard means
5. Green-band threshold (e.g. "≥15/18 turns the panel green")

Plus a worked example of the dev artifact (in this course,
`sample_queries.json` — what its fields mean + a snippet showing
how to fire one sample at the local service). This is template-able
across courses: the artifact path varies, the journey shape doesn't.

The current `outcome_artifact_materializer` writes `README.md`
from `spec.goal` + endpoint schemas. Extend it to ALSO write the
journey + walkthrough sections.

---

## 15. Visible samples expose `gold_passage_ids` labels

**Status:** Backlog. Same audit as #12.

**The current behavior:** Each visible sample includes
``gold_passage_ids`` — explicitly labels which passage in the pool
contains the answer. This is the same convention real research
benchmarks use for DEV splits: labeled gold to help iteration.

**The concern:** If the visible samples are used as a "training
set" and the learner over-fits to the labeled gold (e.g., hard-codes
"return passage 0" assuming gold is always first), the hidden set's
unlabeled gold won't match the assumption.

**Two possible directions:**
(a) Keep gold_passage_ids visible (learner needs a dev signal to
    iterate); label clearly in the README that "real submission
    gold is hidden — don't hard-code positions or IDs from these
    samples".
(b) Strip gold_passage_ids from visible; provide a `dev_check`
    endpoint or local utility that scores a learner-submitted answer
    against unlabeled gold. Closer to a real benchmark dev split.

(a) is faster to ship; (b) is more pedagogically honest. Either way
the **course brief must explicitly document the visible/hidden
convention** so a learner doesn't accidentally over-fit.

---

## 21. Core publish flow lacks the safe-republish behaviors hand-coded in install_support_bot.py

**Status:** Backlog (highest impact). Sub-item (a) also has a spawned
chip ("Repin enrollments on course re-publish").

**Originating direct fix:** Customer Support Bot. The course is
published via a bespoke one-off script
(`scripts/support_bot_course/install_support_bot.py`, mirrored at
`/opt/course-gen-codex/tmp/`). Every safety behavior needed for a
re-publish was hand-coded there because the core publish path does not
do them. A re-publish through the normal flow on ANY other lab silently
regresses learners.

**Generalization — fold ALL of these into the core course publish
pipeline so every lab gets them for free:**
- (a) **Re-pin active enrollments** to the freshly published snapshot.
  `get_deliverable_experience` serves `enrollment.publish_snapshot_id`,
  not the course's latest — so without repin, learners see the OLD
  brief/README while being graded on the NEW on-disk bundle (M33).
- (b) **Refresh already-seeded workspace docs** to the current scheme
  (single consolidated README) and delete retired files
  (`project_brief.md`, `deliverables.md`), **never touching
  learner-authored code** (M41). `seed_workspace_from_snapshot` skips
  existing files, so already-materialised workspaces need an explicit
  refresh step.
- (c) **Back up enrollment rows** (and the course run / snapshot)
  before any mutation (M33 — backups went to `tmp/`).
- (d) **Preserve `status=published` across re-publish.** Deep-copying a
  template course run inherits the template's status; if the template
  is hidden (`active`) the re-published lab silently drops out of the
  catalog (M41 regression — caught and patched in the script only).
- (e) **Populate a real structured `learner_brief`** instead of leaving
  the cloned template's (the Support Bot showed Wikipedia-QA copy until
  M33 set it explicitly).
- (f) **Embed the grader bundle in the publish snapshot** instead of
  rsync-to-disk + `payload_json.outcome_state.workspace_root` (the
  prior P0; bundle currently lives in the mutable
  `outcome_workspaces/<course>/` tree, not the immutable snapshot).

**Where to fix:** the course publish/snapshot service
(`app/services/publish_snapshot_service.py`,
`PostgresWorkflowStore.save_publish_snapshot`/`save_course_run`) plus
`seed_workspace_from_snapshot` for (b). Confirm with the user before
changing shared publish semantics (some snapshot pinning is
intentional for mid-assignment immutability).

---

## 22. Course publish is a per-course bespoke script, not a pipeline

**Status:** Backlog. Depends on / overlaps #21.

**Originating direct fix:** `install_support_bot.py` clones the
known-good `course_wikiqa_v1` snapshot and mutates it by hand (title,
summary, brief, deliverable, learner_brief, seed files, workspace_root,
visible_files). Onboarding any new lab today means writing another such
script.

**Generalization:** a parameterized publisher (course id + starter tree
+ grader bundle dir → snapshot + course run) or a proper
authoring→publish command, so new labs don't need bespoke glue and
inherit #21's safety behaviors automatically.

---

## 23. Learner-facing platform fixes already generalized (audit trail)

**Status:** DONE in pipeline — these session fixes landed in core
(`app/services/lms_service.py`, `app/static/lms.js`,
`app/static/lab-tutor.js`, `app/templates/`,
`app/services/learner_package_runtime.py`) and therefore already apply
to **every lab**, not just the Support Bot. Recorded for traceability;
no further generalization work.

- **M34** lab-tutor chat persisted in Postgres (was browser
  localStorage only) — core `tutor_service` + `tutor_chat_messages`.
- **M35** removed the brittle VS Code agent-panel intercept; clean
  reintroduction is a **spawned chip** ("Reintroduce agent-panel tutor
  triage") — the EditContext root cause is captured there.
- **M36** passing checks are expandable (positive worked example).
  *Limitation:* reports graded before M36 have no stored
  passing-example data and are not retrofitted — only new submissions.
- **M37** dead `#catalog-panel` link → `/courses`.
- **M38** learner-facing copy "course" → "lab".
- **M39** `/courses` is the post-login landing; bare `/` deprecated
  (but `/?enrollment=<id>` preserved for the workspace experience).
- **M40** failing-check feedback is consumable: Expected vs Your output
  are field-scoped (same target), no internal rubric-kind jargon,
  domain-neutral hints.
- **M41** one consolidated `README.md` per workspace (core
  `seed_workspace_from_snapshot`); the *refresh of already-seeded*
  workspaces is still bespoke — covered by #21(b)/#22.

**Known dead code to clean up (not lab-specific):** the orphaned
`renderCatalog()` block + bare-`/` hero path in `lms.js`/
`render_lms_home` (unreachable since the catalog moved to `/courses`).

---

## 24. Support Bot course-content methodology (audit trail)

**Status:** DONE as guidance — generalized in
`docs/COURSE_AUTHORING_PLAYBOOK.md`, not code. The Support Bot's
**M32** changes (dense-retrieval genuinely required, vocabulary-mismatch
scenarios, spec-authored gold instead of reference-derived, honest
"keyword tops out ~20/25" limitation) are course content. The reusable
*method* is the playbook (§4, §10b near-tie wall). Any new lab applies
it by following the playbook; nothing to change in the pipeline.

---

## 25. visible-check command metadata doesn't match the real command

**Status:** Backlog. Generic across all labs.

**Originating direct fix:** Customer Support Bot. The platform default
`visible_check_command` is `sh .coursegen/runtime/check_visible.sh`
(`postgres_store.py`, `artifact_materializer.py`), but the Support Bot
bundle has no `check_visible.sh` — the real visible check is
`python public/checks/run_visible_checks.py`, and the seeded
`.coursegen/runtime/` only ships `install.sh`/`run.sh`/`verify.sh`.
Anything that surfaces `visible_check_command` to a learner (or runs
it) points at a non-existent script.

**Generalization:** make the runtime-dependency metadata
(`local_run_command` / `visible_check_command` / `preview_command`)
derive from what the bundle actually ships (or require authoring to
emit a real `check_visible.sh`), and validate the referenced scripts
exist at publish time. Until then the README/brief points learners at
the correct `python public/checks/run_visible_checks.py`.

---

## 26. Seeded `.coursegen/review_areas/<id>/README.md` is a confusing duplicate the validator pins

**Status:** Backlog. Core/platform — needs coordinated validator change.

**Originating direct fix:** Customer Support Bot. M41 consolidated the
learner workspace to a single `README.md`; this session also stopped
seeding the unread `.coursegen/review_areas/index.json` +
`deliverables/index.json` clutter. But `.coursegen/review_areas/<id>/
README.md` (= `starter_readme`) is still seeded **because
`validate_seeded_learner_workspace` (publish-certification gate,
`bundle_validation.py` ~L760) hard-errors `seeded_workspace_missing_
review_area_readme` if it's absent.** So learners still see a second,
separate README inside a dot-folder.

**Generalization:** retarget `validate_seeded_learner_workspace` (and
the certification flow in `publish_learner_certification_service.py`)
to validate the single consolidated `README.md` instead of the
`.coursegen/review_areas/<id>/README.md` copy, then stop seeding the
duplicate in `seed_workspace_from_snapshot`. End state: the only
learner-facing doc is the one consolidated README; `.coursegen/` holds
nothing but the internal `workspace_seeded.txt` marker.

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
