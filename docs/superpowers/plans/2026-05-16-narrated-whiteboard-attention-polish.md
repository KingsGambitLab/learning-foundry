# Narrated Whiteboard — Attention Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shipped narrated-whiteboard card more attention-grabbing — spotlight the just-added node each step, polish the step-in motion, and add keyboard control + progress dots — with no voice/LLM/persistence/infra changes.

**Architecture:** Two new pure, Node-unit-tested helpers (`extractNodeLabels`, `diffNewNodeLabels`) added to the existing guarded helper block in `app/static/lab-tutor.js` and exported for tests. `renderNarratedInto` calls them after each step's Mermaid renders and adds a transient CSS class to the SVG node whose visible label is new (best-effort, never fatal). CSS gains a spotlight keyframe + spring step-in easing + progress-dot styles. Keyboard handlers are card-scoped and reuse the existing playback handler logic.

**Tech Stack:** Vanilla ES2020 (no bundler), existing vendored Mermaid, Node v24 built-in test runner (`node --test`, zero deps), CSS. No Python changes.

---

## File Structure

- `app/static/lab-tutor.js` (modify) — add `extractNodeLabels` + `diffNewNodeLabels` to the guarded helper block and to `module.exports`; add spotlight call in `renderStep`; extract shared playback handlers; add progress dots + card-scoped keyboard.
- `app/static/lab-tutor.css` (modify) — `.lt-narrated-spot` keyframe, spring step-in easing, `.lt-narrated-dots`/`.lt-narrated-dot` styles, reduced-motion coverage.
- `tests/js/test_narrated.js` (modify) — Node unit tests for the two new pure helpers.

No new files. No persistence/LLM/server changes. Manual browser check for DOM/CSS/keyboard (repo has no JS DOM harness — same tradeoff as the base feature).

---

## Task 1: Pure helpers `extractNodeLabels` + `diffNewNodeLabels`

**Files:**
- Modify: `app/static/lab-tutor.js` — insert two functions immediately BEFORE the export guard (the line `  if (typeof module !== "undefined" && module.exports) {`, currently ~line 103), and extend the `module.exports = {...}` object.
- Test: `tests/js/test_narrated.js` (append tests at end).

- [ ] **Step 1: Write the failing tests**

Append to `tests/js/test_narrated.js`:

```js

test("extractNodeLabels: bracketed flowchart labels", () => {
  const out = lt.extractNodeLabels("flowchart LR\n A[Query]-->B[Embed]");
  assert.deepEqual(out.slice().sort(), ["Embed", "Query"]);
});

test("extractNodeLabels: bare ids when no label declared", () => {
  const out = lt.extractNodeLabels("graph TD\n A-->B");
  assert.deepEqual(out.slice().sort(), ["A", "B"]);
});

test("extractNodeLabels: ignores edge-label text and direction/header", () => {
  const out = lt.extractNodeLabels("flowchart LR\n A[Q] -->|yes| B[E]");
  assert.deepEqual(out.slice().sort(), ["E", "Q"]);
  assert.equal(out.includes("yes"), false);
  assert.equal(out.includes("LR"), false);
  assert.equal(out.includes("flowchart"), false);
});

test("extractNodeLabels: dedupes by first declaration", () => {
  const out = lt.extractNodeLabels("flowchart LR\n A[Q]-->B[E]\n B[E]-->A[Q]");
  assert.deepEqual(out.slice().sort(), ["E", "Q"]);
});

test("extractNodeLabels: shape variants (rhombus/circle)", () => {
  const out = lt.extractNodeLabels("flowchart TD\n A{Decide}-->B((Done))");
  assert.deepEqual(out.slice().sort(), ["Decide", "Done"]);
});

test("extractNodeLabels: non-string / empty -> []", () => {
  assert.deepEqual(lt.extractNodeLabels(""), []);
  assert.deepEqual(lt.extractNodeLabels(null), []);
});

test("diffNewNodeLabels: no prev (step 0) -> []", () => {
  assert.deepEqual(lt.diffNewNodeLabels(undefined, "flowchart LR\n A[Q]"), []);
  assert.deepEqual(lt.diffNewNodeLabels("", "flowchart LR\n A[Q]"), []);
});

test("diffNewNodeLabels: returns only newly added label", () => {
  const out = lt.diffNewNodeLabels(
    "flowchart LR\n A[Q]-->B[E]",
    "flowchart LR\n A[Q]-->B[E]-->C[S]"
  );
  assert.deepEqual(out, ["S"]);
});

test("diffNewNodeLabels: no change -> []", () => {
  assert.deepEqual(
    lt.diffNewNodeLabels("flowchart LR\n A[Q]", "flowchart LR\n A[Q]"),
    []
  );
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `node --test tests/js/test_narrated.js`
Expected: FAIL — `lt.extractNodeLabels is not a function`.

- [ ] **Step 3: Implement the helpers**

In `app/static/lab-tutor.js`, immediately BEFORE this exact existing block:

```js
  // Node-only export hook for unit tests. In the browser `module` is
  // undefined so this is skipped and the widget bootstraps normally.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { parseNarrated, computeNoAudioMs, narratedReducer };
    return;
  }
```

insert:

```js
  // Distinct human-visible node labels from a Mermaid flowchart/graph
  // source, in declaration order. Pure; never throws. Used to spotlight
  // the node added by the current step. Worst case (exotic markup): a
  // label is missed and the spotlight simply no-ops — never breakage.
  function extractNodeLabels(src) {
    if (!src || typeof src !== "string") return [];
    const RESERVED = new Set([
      "flowchart", "graph", "subgraph", "end", "style", "classDef",
      "class", "linkStyle", "click", "direction", "stateDiagram",
      "stateDiagram-v2", "sequenceDiagram", "classDiagram", "erDiagram",
      "journey", "gantt", "pie", "mindmap",
      "LR", "RL", "TB", "BT", "TD", "DT",
    ]);
    // Most-specific bracket pairs first so e.g. ([stadium]) is not
    // mis-read by the (round) pattern.
    const SHAPES = [
      /([A-Za-z_]\w*)\s*\[\[([^\]]+)\]\]/g,  // [[subroutine]]
      /([A-Za-z_]\w*)\s*\[\(([^)]+)\)\]/g,   // [(cylinder)]
      /([A-Za-z_]\w*)\s*\(\(([^)]+)\)\)/g,   // ((circle))
      /([A-Za-z_]\w*)\s*\(\[([^\]]+)\]\)/g,  // ([stadium])
      /([A-Za-z_]\w*)\s*\{\{([^}]+)\}\}/g,   // {{hexagon}}
      /([A-Za-z_]\w*)\s*\[([^\]]+)\]/g,      // [rect]
      /([A-Za-z_]\w*)\s*\(([^)]+)\)/g,       // (round)
      /([A-Za-z_]\w*)\s*\{([^}]+)\}/g,       // {rhombus}
      /([A-Za-z_]\w*)\s*>([^\]]+)\]/g,       // >asymmetric]
    ];
    const labelById = Object.create(null);
    const order = [];
    // Drop edge labels |...| so they are not parsed as node content.
    let work = src.replace(/\|[^|]*\|/g, " ");
    // Blank a leading diagram header line.
    const lines = work.split(/\r?\n/);
    if (
      lines.length &&
      /^\s*(flowchart|graph|stateDiagram(-v2)?|sequenceDiagram|classDiagram|erDiagram|journey|gantt|pie|mindmap)\b/.test(
        lines[0]
      )
    ) {
      lines[0] = "";
    }
    work = lines.join("\n");
    let stripped = work;
    for (const re of SHAPES) {
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(work)) !== null) {
        const id = m[1];
        const label = m[2].trim();
        if (!(id in labelById) && label !== "") {
          labelById[id] = label;
          order.push(id);
        }
      }
      stripped = stripped.replace(re, " $1 ");
    }
    // Remove edge operators so arrowheads (x/o) are not read as ids.
    stripped = stripped.replace(/<?-{1,3}[>xo]?|-\.-?>?|={2,}>?/g, " ");
    const tokens = stripped.match(/[A-Za-z_]\w*/g) || [];
    for (const t of tokens) {
      if (RESERVED.has(t)) continue;
      if (!(t in labelById)) {
        labelById[t] = t;
        order.push(t);
      }
    }
    return order.map((id) => labelById[id]);
  }

  // Labels present in `currSrc` but not `prevSrc`. No prev (step 0) -> [].
  function diffNewNodeLabels(prevSrc, currSrc) {
    const curr = extractNodeLabels(currSrc);
    if (!prevSrc) return [];
    const prev = new Set(extractNodeLabels(prevSrc));
    return curr.filter((l) => !prev.has(l));
  }
```

Then change the export line from:

```js
    module.exports = { parseNarrated, computeNoAudioMs, narratedReducer };
```

to:

```js
    module.exports = { parseNarrated, computeNoAudioMs, narratedReducer, extractNodeLabels, diffNewNodeLabels };
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `node --test tests/js/test_narrated.js`
Expected: PASS — all tests (the original 10 + the 9 new) pass.

- [ ] **Step 5: Syntax check**

Run: `node --check app/static/lab-tutor.js`
Expected: exit 0, no output.

- [ ] **Step 6: Commit**

```bash
git add app/static/lab-tutor.js tests/js/test_narrated.js
git commit -m "feat(lab-tutor): node-label extraction + diff helpers for spotlight

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Spotlight DOM application + motion polish CSS

**Files:**
- Modify: `app/static/lab-tutor.js` — extend `renderStep` inside `renderNarratedInto` to spotlight new nodes.
- Modify: `app/static/lab-tutor.css` — spotlight keyframe, spring step-in, reduced-motion coverage.

No automated test (DOM); covered by Task 1 pure tests + Task 4 manual verification.

- [ ] **Step 1: Add the spotlight helper + call in `renderStep`**

In `app/static/lab-tutor.js`, find this EXACT current `renderStep` (inside `renderNarratedInto`):

```js
    async function renderStep(i) {
      const step = spec.steps[i];
      caption.textContent = step.say;
      stage.classList.remove("lt-narrated-stage--in");
      stage.innerHTML = "";
      // reflow so the entry transition re-triggers
      void stage.offsetWidth;
      await renderMermaidInto(stage, step.mermaid);
      stage.classList.add("lt-narrated-stage--in");
    }
```

Replace it with:

```js
    function spotlightNew(i) {
      try {
        if (i <= 0) return;
        const labels = diffNewNodeLabels(
          spec.steps[i - 1].mermaid,
          spec.steps[i].mermaid
        );
        if (!labels.length) return;
        let nodes = stage.querySelectorAll("g.node");
        if (!nodes.length) nodes = stage.querySelectorAll('[class*="node"]');
        const wanted = new Set(labels);
        const hit = new Set();
        nodes.forEach((el) => {
          const txt = (el.textContent || "").trim();
          if (wanted.has(txt) && !hit.has(txt)) {
            hit.add(txt);
            el.classList.add("lt-narrated-spot");
            setTimeout(() => {
              try { el.classList.remove("lt-narrated-spot"); } catch {}
            }, 1000);
          }
        });
      } catch {
        /* spotlight is a nicety — never a failure path */
      }
    }

    async function renderStep(i) {
      const step = spec.steps[i];
      caption.textContent = step.say;
      stage.classList.remove("lt-narrated-stage--in");
      stage.innerHTML = "";
      // reflow so the entry transition re-triggers
      void stage.offsetWidth;
      await renderMermaidInto(stage, step.mermaid);
      stage.classList.add("lt-narrated-stage--in");
      spotlightNew(i);
    }
```

- [ ] **Step 2: Motion polish + spotlight CSS**

In `app/static/lab-tutor.css`, find this EXACT current block:

```css
.lt-narrated-stage {
  opacity: 0;
  transform: translateY(6px) scale(0.98);
}
.lt-narrated-stage--in {
  opacity: 1;
  transform: none;
  transition: opacity 0.22s ease, transform 0.22s ease;
}
@media (prefers-reduced-motion: reduce) {
  .lt-narrated-stage,
  .lt-narrated-stage--in {
    opacity: 1;
    transform: none;
    transition: none;
  }
}
```

Replace it with:

```css
.lt-narrated-stage {
  opacity: 0;
  transform: translateY(10px) scale(0.96);
}
.lt-narrated-stage--in {
  opacity: 1;
  transform: none;
  transition: opacity 0.32s cubic-bezier(0.22, 1, 0.36, 1),
              transform 0.32s cubic-bezier(0.22, 1, 0.36, 1);
}
.lt-narrated-spot {
  animation: lt-narrated-pulse 0.9s ease-out 1;
}
@keyframes lt-narrated-pulse {
  0% {
    filter: drop-shadow(0 0 0 rgba(31, 111, 235, 0));
    transform: scale(1);
  }
  35% {
    filter: drop-shadow(0 0 6px rgba(31, 111, 235, 0.75));
    transform: scale(1.06);
  }
  100% {
    filter: drop-shadow(0 0 0 rgba(31, 111, 235, 0));
    transform: scale(1);
  }
}
@media (prefers-reduced-motion: reduce) {
  .lt-narrated-stage,
  .lt-narrated-stage--in {
    opacity: 1;
    transform: none;
    transition: none;
  }
  .lt-narrated-spot {
    animation: none;
  }
}
```

- [ ] **Step 3: Regression + syntax**

Run: `node --check app/static/lab-tutor.js && node --test tests/js/test_narrated.js`
Expected: `node --check` exit 0; all Node tests pass (the helpers/guard untouched).

- [ ] **Step 4: Commit**

```bash
git add app/static/lab-tutor.js app/static/lab-tutor.css
git commit -m "feat(lab-tutor): spotlight the newly added node + spring step-in

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Keyboard control + progress dots

**Files:**
- Modify: `app/static/lab-tutor.js` — extract shared playback handlers, add dots + card-scoped keyboard.
- Modify: `app/static/lab-tutor.css` — dot styles.

- [ ] **Step 1: Extract shared handlers and wire dots + keyboard**

In `app/static/lab-tutor.js`, find this EXACT current block (the four `addEventListener` handlers + the final initial-paint line, inside `renderNarratedInto`):

```js
    playBtn.addEventListener("click", () => {
      if (state.mode === "playing") {
        state = narratedReducer(state, { type: "PAUSE" });
        stopSpeech();
        syncControls();
      } else {
        const wasDone = state.mode === "done";
        state = narratedReducer(state, { type: wasDone ? "REPLAY" : "PLAY" });
        run();
      }
    });
    prevBtn.addEventListener("click", () => {
      stopSpeech();
      state = narratedReducer(state, { type: "PREV" });
      if (state.mode === "playing") { state = narratedReducer(state, { type: "PAUSE" }); }
      renderStep(state.step).then(syncControls);
    });
    nextBtn.addEventListener("click", () => {
      stopSpeech();
      state = narratedReducer(state, { type: "NEXT" });
      if (state.mode === "playing") { state = narratedReducer(state, { type: "PAUSE" }); }
      renderStep(state.step).then(syncControls);
    });
    muteBtn.addEventListener("click", () => {
      muted = !muted;
      muteBtn.textContent = muted ? "🔇" : "🔊";
      muteBtn.setAttribute("aria-label", muted ? "Unmute narration" : "Mute narration");
      if (muted) {
        stopSpeech();
        // Muting mid-playback must not rely on cancel() firing onerror to
        // advance — restart pacing via the no-audio timer explicitly.
        if (state.mode === "playing") {
          noAudioTimer = setTimeout(
            () => { if (state.mode === "playing") advance(); },
            computeNoAudioMs(spec.steps[state.step].say)
          );
        }
      }
    });

    // Initial paint: first step visible, idle (no autoplay).
    renderStep(0).then(syncControls);
  }
```

Replace it with:

```js
    function doPlayPause() {
      if (state.mode === "playing") {
        state = narratedReducer(state, { type: "PAUSE" });
        stopSpeech();
        syncControls();
      } else {
        const wasDone = state.mode === "done";
        state = narratedReducer(state, { type: wasDone ? "REPLAY" : "PLAY" });
        run();
      }
    }
    function doPrev() {
      stopSpeech();
      state = narratedReducer(state, { type: "PREV" });
      if (state.mode === "playing") { state = narratedReducer(state, { type: "PAUSE" }); }
      renderStep(state.step).then(syncControls);
    }
    function doNext() {
      stopSpeech();
      state = narratedReducer(state, { type: "NEXT" });
      if (state.mode === "playing") { state = narratedReducer(state, { type: "PAUSE" }); }
      renderStep(state.step).then(syncControls);
    }

    playBtn.addEventListener("click", doPlayPause);
    prevBtn.addEventListener("click", doPrev);
    nextBtn.addEventListener("click", doNext);
    muteBtn.addEventListener("click", () => {
      muted = !muted;
      muteBtn.textContent = muted ? "🔇" : "🔊";
      muteBtn.setAttribute("aria-label", muted ? "Unmute narration" : "Mute narration");
      if (muted) {
        stopSpeech();
        // Muting mid-playback must not rely on cancel() firing onerror to
        // advance — restart pacing via the no-audio timer explicitly.
        if (state.mode === "playing") {
          noAudioTimer = setTimeout(
            () => { if (state.mode === "playing") advance(); },
            computeNoAudioMs(spec.steps[state.step].say)
          );
        }
      }
    });

    // Card-scoped keyboard (NEVER document-level — must not leak into
    // code-server / Monaco / page shortcuts).
    card.tabIndex = 0;
    card.setAttribute("role", "group");
    card.setAttribute("aria-label", "Narrated whiteboard");
    card.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Spacebar") {
        e.preventDefault();
        doPlayPause();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        doNext();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        doPrev();
      }
    });

    // Initial paint: first step visible, idle (no autoplay).
    renderStep(0).then(syncControls);
  }
```

- [ ] **Step 2: Add the progress dots element**

In `app/static/lab-tutor.js`, find this EXACT current block (inside `renderNarratedInto`, just after the caption is appended and before the controls are built):

```js
    const caption = document.createElement("div");
    caption.className = "lt-narrated-caption";
    card.appendChild(caption);

    const controls = document.createElement("div");
    controls.className = "lt-narrated-controls";
```

Replace it with:

```js
    const caption = document.createElement("div");
    caption.className = "lt-narrated-caption";
    card.appendChild(caption);

    const dots = document.createElement("div");
    dots.className = "lt-narrated-dots";
    dots.setAttribute("aria-hidden", "true");
    const dotEls = [];
    for (let d = 0; d < spec.steps.length; d++) {
      const dot = document.createElement("span");
      dot.className = "lt-narrated-dot";
      dots.appendChild(dot);
      dotEls.push(dot);
    }
    card.appendChild(dots);

    const controls = document.createElement("div");
    controls.className = "lt-narrated-controls";
```

- [ ] **Step 3: Make `syncControls` update the active dot**

In `app/static/lab-tutor.js`, find this EXACT current `syncControls`:

```js
    function syncControls() {
      playBtn.textContent =
        state.mode === "playing" ? "⏸ Pause"
        : state.mode === "done" ? "↻ Replay"
        : "▶ Play narration";
      playBtn.setAttribute(
        "aria-label",
        state.mode === "playing" ? "Pause narration"
        : state.mode === "done" ? "Replay narration"
        : "Play narration"
      );
      prevBtn.disabled = state.step === 0;
      nextBtn.disabled = state.step >= total - 1 && state.mode !== "playing";
    }
```

Replace it with:

```js
    function syncControls() {
      playBtn.textContent =
        state.mode === "playing" ? "⏸ Pause"
        : state.mode === "done" ? "↻ Replay"
        : "▶ Play narration";
      playBtn.setAttribute(
        "aria-label",
        state.mode === "playing" ? "Pause narration"
        : state.mode === "done" ? "Replay narration"
        : "Play narration"
      );
      prevBtn.disabled = state.step === 0;
      nextBtn.disabled = state.step >= total - 1 && state.mode !== "playing";
      for (let d = 0; d < dotEls.length; d++) {
        dotEls[d].classList.toggle("lt-narrated-dot--active", d === state.step);
      }
    }
```

- [ ] **Step 4: Dot styles**

In `app/static/lab-tutor.css`, find this EXACT current block:

```css
.lt-narrated-caption {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.5;
  color: var(--lt-text);
  min-height: 1.5em;
}
```

Replace it with:

```css
.lt-narrated-caption {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.5;
  color: var(--lt-text);
  min-height: 1.5em;
}
.lt-narrated-dots {
  display: flex;
  gap: 6px;
  margin-top: 8px;
}
.lt-narrated-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--lt-border);
  transition: background 0.15s ease, transform 0.15s ease;
}
.lt-narrated-dot--active {
  background: var(--lt-accent, #1f6feb);
  transform: scale(1.4);
}
@media (prefers-reduced-motion: reduce) {
  .lt-narrated-dot {
    transition: none;
  }
}
```

- [ ] **Step 5: Regression + syntax**

Run: `node --check app/static/lab-tutor.js && node --test tests/js/test_narrated.js`
Expected: `node --check` exit 0; all Node tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/static/lab-tutor.js app/static/lab-tutor.css
git commit -m "feat(lab-tutor): card-scoped keyboard control + progress dots

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Full regression + manual verification

**Files:** none (verification only)

- [ ] **Step 1: Full automated regression**

Run:
```bash
node --check app/static/lab-tutor.js && node --test tests/js/test_narrated.js
/Users/tushar/Desktop/codebases/course-gen-codex/.venv/bin/python -m pytest tests/test_tutor_service.py tests/test_tutor_routes.py -q
```
Expected: `node --check` exit 0; all Node tests pass (19 total); pytest all pass (the persona/routes are untouched — pure regression guard).

- [ ] **Step 2: Manual browser verification (evidence required)**

Local only — no remote deploy. With the local server running (`http://127.0.0.1:8012`), launch a learner studio container, open the tutor, prompt a multi-step process ("walk me through the RAG pipeline step by step"). Record evidence (screenshot/observation) for EACH:
  - [ ] Each step pulses ONLY the newly added node (not the whole diagram).
  - [ ] A step that adds no new node (or a non-flowchart diagram) simply doesn't pulse — no error, card still plays.
  - [ ] Step-in motion reads as a settle (spring), not a hard blink; reduced-motion OS setting disables pulse + transition but dots/keys still work.
  - [ ] Progress dots track the current step during autoplay and prev/next.
  - [ ] With the card focused: `Space` toggles play/pause and does NOT scroll the page; `→`/`←` step; keys do NOT leak to the editor when the card is not focused.
  - [ ] Existing controls (play/pause/prev/next/mute) behave exactly as before; malformed `lt-narrated` still falls back to the raw card.

- [ ] **Step 3: Report**

Do not claim completion until every Step-2 box has recorded evidence (verification-before-completion). Report results. No `git push` / no remote deploy without explicit approval.

---

## Self-Review

**Spec coverage:**
- New-node spotlight (source diff + label-text match, graceful no-op) → Task 1 (`extractNodeLabels`/`diffNewNodeLabels` + tests) + Task 2 (`spotlightNew` DOM apply, try/catch, `g.node` then `[class*="node"]` fallback, 1000ms cleanup).
- Step 0 → no spotlight → `diffNewNodeLabels` returns `[]` for falsy prev (Task 1 test) and `spotlightNew` early-returns for `i<=0` (Task 2).
- Motion polish (spring cubic-bezier, more travel) → Task 2 CSS.
- Reduced-motion neutralizes stage transition AND spotlight → Task 2 `@media` block.
- Progress dots (decorative, `aria-hidden`, between caption and controls, active tracks `state.step`) → Task 3 Steps 2–4.
- Keyboard card-scoped (`tabindex`/`role`/`aria-label`, listener on card not document, Space preventDefault, arrows) reusing existing handler logic → Task 3 Step 1.
- Pure helpers Node-unit-tested; DOM/CSS/keyboard manual → Tasks 1 & 4.
- Non-goals respected (no voice/LLM/persistence/server/new-file; no tap-to-advance; no autoplay change) — no such tasks; Task 4 forbids push.

**Placeholder scan:** No TBD/TODO; every code step has complete code; commands have expected output.

**Type consistency:** `extractNodeLabels(src)→string[]`, `diffNewNodeLabels(prevSrc,currSrc)→string[]` consistent across Task 1 tests, the `module.exports` list, and Task 2's `spotlightNew`. `spotlightNew(i)` called only from `renderStep` after render. Shared handlers `doPlayPause/doPrev/doNext` defined once (Task 3 Step 1) and referenced by both click and keydown. `dotEls` created in Task 3 Step 2 and consumed in Task 3 Step 3's `syncControls`. CSS classes (`lt-narrated-spot`, `lt-narrated-dots`, `lt-narrated-dot`, `lt-narrated-dot--active`) match between JS and CSS tasks. `--lt-accent` referenced with a `#1f6feb` fallback in case the variable name differs.
