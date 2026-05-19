# Narrated Whiteboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Lab Tutor explain a multi-step concept as a step-by-step Mermaid reveal narrated by browser text-to-speech, with no extra LLM call and zero TTS cost.

**Architecture:** The tutor emits a fenced ```` ```lt-narrated ```` block (JSON with a `steps[]` array; each step has its own cumulative Mermaid source + a `say` line). The widget parses it and renders a playback card that, per step, re-renders the step's Mermaid and speaks `say` via `window.speechSynthesis`, advancing on the utterance `onend` event (event-based cursor, no timeline). Pure logic (parser, no-audio timer, playback reducer) is factored into testable functions guarded for Node so they can be unit-tested with Node's built-in test runner; DOM/playback wiring stays in `lab-tutor.js`.

**Tech Stack:** Vanilla ES2020 (no bundler), existing vendored `mermaid.min.js`, browser `SpeechSynthesis`, Python 3 / pytest + `unittest` for the tutor-prompt contract, Node v24 built-in test runner (`node --test`, zero new deps) for JS pure logic.

---

## File Structure

- `app/static/lab-tutor.js` (modify) — add three pure helpers + a Node-export guard near the top of the IIFE; add `renderNarratedInto` + `lt-narrated` fence handling in `appendTutor`.
- `app/static/lab-tutor.css` (modify) — add `.lt-narrated*` styles incl. per-step entry transition.
- `app/services/tutor_service.py` (modify) — extend `_TUTOR_PERSONA` with `lt-narrated` guidance + single-pass constraint.
- `tests/js/test_narrated.js` (create) — Node `node:test` unit tests for the three pure helpers.
- `tests/test_tutor_service.py` (modify) — add a test asserting the persona contract.

**V1 simplification (deliberate, flagged at handoff):** per-card playback position is NOT persisted. On reload the narrated card re-renders fresh at `idle` (ready to replay) — identical to how the existing Mermaid card already behaves. The spec's "persist last-played step" is deferred; adding it would require extending the localStorage history schema (`loadHistory` only retains `{role,text}`), which is disproportionate for V1.

---

## Task 1: Pure helpers + Node export guard

**Files:**
- Modify: `app/static/lab-tutor.js` (insert after line 11 `"use strict";`, before line 14 `const me = document.currentScript`)
- Test: `tests/js/test_narrated.js`

- [ ] **Step 1: Write the failing test**

Create `tests/js/test_narrated.js`:

```js
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

// Requiring lab-tutor.js under Node hits the export guard and returns the
// pure helpers without touching the DOM.
const lt = require(path.join(__dirname, "..", "..", "app", "static", "lab-tutor.js"));

test("parseNarrated: valid JSON returns normalized steps", () => {
  const body = JSON.stringify({
    steps: [
      { say: "Chunk the docs.", mermaid: "flowchart LR\n A-->B" },
      { say: "Embed them.", mermaid: "flowchart LR\n A-->B-->C" },
    ],
  });
  const out = lt.parseNarrated(body);
  assert.equal(out.steps.length, 2);
  assert.equal(out.steps[0].say, "Chunk the docs.");
  assert.equal(out.steps[1].mermaid, "flowchart LR\n A-->B-->C");
});

test("parseNarrated: repairs trailing commas", () => {
  const body = '{ "steps": [ { "say": "x", "mermaid": "graph TD\\n A", }, ], }';
  const out = lt.parseNarrated(body);
  assert.ok(out);
  assert.equal(out.steps.length, 1);
  assert.equal(out.steps[0].say, "x");
});

test("parseNarrated: strips stray prose around the object", () => {
  const body = 'Sure! Here it is:\n{ "steps": [ { "say": "a", "mermaid": "graph TD\\n A" } ] }\nHope that helps.';
  const out = lt.parseNarrated(body);
  assert.ok(out);
  assert.equal(out.steps[0].say, "a");
});

test("parseNarrated: garbage or empty -> null", () => {
  assert.equal(lt.parseNarrated("not json at all <<<"), null);
  assert.equal(lt.parseNarrated('{"steps": []}'), null);
  assert.equal(lt.parseNarrated('{"steps": [{"say": 1}]}'), null);
});

test("computeNoAudioMs: max(2000, words*240)", () => {
  assert.equal(lt.computeNoAudioMs("one two three"), 2000); // 3*240=720 -> floor 2000
  assert.equal(lt.computeNoAudioMs("a ".repeat(20).trim()), 20 * 240);
  assert.equal(lt.computeNoAudioMs(""), 2000);
});

test("narratedReducer: play/advance/done flow", () => {
  let s = { mode: "idle", step: 0, total: 3 };
  s = lt.narratedReducer(s, { type: "PLAY" });
  assert.deepEqual(s, { mode: "playing", step: 0, total: 3 });
  s = lt.narratedReducer(s, { type: "ADVANCE" });
  assert.deepEqual(s, { mode: "playing", step: 1, total: 3 });
  s = lt.narratedReducer(s, { type: "ADVANCE" });
  s = lt.narratedReducer(s, { type: "ADVANCE" }); // step 2 -> done
  assert.equal(s.mode, "done");
  assert.equal(s.step, 2);
});

test("narratedReducer: stale ADVANCE ignored when not playing", () => {
  let s = { mode: "paused", step: 1, total: 3 };
  const after = lt.narratedReducer(s, { type: "ADVANCE" });
  assert.deepEqual(after, s); // unchanged — stale-callback guard
});

test("narratedReducer: PAUSE/NEXT/PREV/REPLAY", () => {
  let s = { mode: "playing", step: 0, total: 3 };
  s = lt.narratedReducer(s, { type: "PAUSE" });
  assert.equal(s.mode, "paused");
  s = lt.narratedReducer(s, { type: "NEXT" });
  assert.deepEqual(s, { mode: "paused", step: 1, total: 3 });
  s = lt.narratedReducer(s, { type: "PREV" });
  assert.deepEqual(s, { mode: "paused", step: 0, total: 3 });
  s = lt.narratedReducer(s, { type: "PREV" }); // clamp at 0
  assert.equal(s.step, 0);
  s = lt.narratedReducer({ mode: "done", step: 2, total: 3 }, { type: "REPLAY" });
  assert.deepEqual(s, { mode: "playing", step: 0, total: 3 });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/js/test_narrated.js`
Expected: FAIL — `lt.parseNarrated is not a function` (the export guard / helpers don't exist yet).

- [ ] **Step 3: Write minimal implementation**

In `app/static/lab-tutor.js`, immediately after line 11 (`  "use strict";`) and before line 13's config comment, insert:

```js

  // ── Narrated-whiteboard pure helpers (also unit-tested under Node) ─────────
  // Parse an lt-narrated block body into { steps: [{say, mermaid}] } or null.
  function parseNarrated(body) {
    function tryParse(s) {
      try { return JSON.parse(s); } catch { return undefined; }
    }
    let obj = tryParse(body);
    if (obj === undefined) {
      let repaired = String(body);
      const open = repaired.indexOf("{");
      const close = repaired.lastIndexOf("}");
      if (open !== -1 && close > open) repaired = repaired.slice(open, close + 1);
      repaired = repaired.replace(/,(\s*[}\]])/g, "$1"); // trailing commas
      obj = tryParse(repaired);
    }
    if (!obj || !Array.isArray(obj.steps)) return null;
    const steps = obj.steps.filter(
      (st) =>
        st &&
        typeof st.say === "string" &&
        st.say.trim() !== "" &&
        typeof st.mermaid === "string" &&
        st.mermaid.trim() !== ""
    ).map((st) => ({ say: st.say, mermaid: st.mermaid }));
    return steps.length > 0 ? { steps } : null;
  }

  // No-audio / muted pacing: max(2s, words*240ms) (pattern from OpenMAIC).
  function computeNoAudioMs(say) {
    const words = String(say).trim().split(/\s+/).filter(Boolean).length;
    return Math.max(2000, words * 240);
  }

  // Pure playback state machine. State: {mode,step,total}.
  // mode: "idle" | "playing" | "paused" | "done".
  function narratedReducer(state, action) {
    const s = state;
    switch (action.type) {
      case "PLAY":
        if (s.mode === "done") return { mode: "playing", step: 0, total: s.total };
        return { mode: "playing", step: s.step, total: s.total };
      case "PAUSE":
        if (s.mode !== "playing") return s;
        return { mode: "paused", step: s.step, total: s.total };
      case "ADVANCE": {
        // Stale-callback guard: only the active "playing" mode advances.
        if (s.mode !== "playing") return s;
        const next = s.step + 1;
        if (next >= s.total) return { mode: "done", step: s.total - 1, total: s.total };
        return { mode: "playing", step: next, total: s.total };
      }
      case "NEXT": {
        const next = Math.min(s.step + 1, s.total - 1);
        const done = s.step + 1 >= s.total;
        return { mode: done ? "done" : s.mode, step: next, total: s.total };
      }
      case "PREV":
        return { mode: s.mode, step: Math.max(s.step - 1, 0), total: s.total };
      case "REPLAY":
        return { mode: "playing", step: 0, total: s.total };
      case "STOP":
        return { mode: "idle", step: 0, total: s.total };
      default:
        return s;
    }
  }

  // Node-only export hook for unit tests. In the browser `module` is
  // undefined so this is skipped and the widget bootstraps normally.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { parseNarrated, computeNoAudioMs, narratedReducer };
    return;
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/js/test_narrated.js`
Expected: PASS — all 8 tests pass.

- [ ] **Step 5: Verify the browser path is not broken**

Run: `node -e "global.window={};global.document={currentScript:null,querySelector:()=>null};try{require('./app/static/lab-tutor.js');console.log('browser-path require did not early-return (expected, guard only triggers under module.exports)')}catch(e){console.log('threw:',e.message)}"`
Expected: it threw (because real DOM is absent) OR printed the no-early-return line — either is fine. This step only confirms the guard is `module.exports`-gated, not that the widget runs headless. The authoritative browser check is Task 6.

- [ ] **Step 6: Commit**

```bash
git add app/static/lab-tutor.js tests/js/test_narrated.js
git commit -m "feat(lab-tutor): narrated-whiteboard pure helpers + Node unit tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Renderer + playback wiring in the widget

**Files:**
- Modify: `app/static/lab-tutor.js` — add `renderNarratedInto` near the Mermaid helpers (after `renderMermaidInto`, which ends ~line 137); add an `lt-narrated` scan in `appendTutor` (around lines 379-398).

No automated test (no DOM test harness in this repo by design). Logic correctness is covered by Task 1's pure-function tests; integration is verified manually in Task 6.

- [ ] **Step 1: Add the narrated renderer**

In `app/static/lab-tutor.js`, immediately AFTER the `renderMermaidInto` function (just before the `// ──` comment that follows it, ~line 137), insert:

```js

  // Render a narrated-whiteboard card: stepwise Mermaid reveal + TTS.
  let narratedIdCounter = 0;
  function renderNarratedInto(parent, spec) {
    const card = document.createElement("div");
    card.className = "lt-narrated";

    const stage = document.createElement("div");
    stage.className = "lt-narrated-stage";
    card.appendChild(stage);

    const caption = document.createElement("div");
    caption.className = "lt-narrated-caption";
    card.appendChild(caption);

    const controls = document.createElement("div");
    controls.className = "lt-narrated-controls";
    const mkBtn = (label, aria) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "lt-narrated-btn";
      b.textContent = label;
      b.setAttribute("aria-label", aria);
      return b;
    };
    const playBtn = mkBtn("▶ Play narration", "Play narration");
    const prevBtn = mkBtn("‹", "Previous step");
    const nextBtn = mkBtn("›", "Next step");
    const muteBtn = mkBtn("🔊", "Mute narration");
    controls.append(playBtn, prevBtn, nextBtn, muteBtn);
    card.appendChild(controls);
    parent.appendChild(card);

    const total = spec.steps.length;
    let state = { mode: "idle", step: 0, total };
    let muted = false;
    let noAudioTimer = null;
    const synth = window.speechSynthesis || null;
    const ttsOk = !!(synth && window.SpeechSynthesisUtterance);
    if (!ttsOk) {
      muteBtn.style.display = "none";
      muted = true;
    }

    function clearTimer() {
      if (noAudioTimer) { clearTimeout(noAudioTimer); noAudioTimer = null; }
    }
    function stopSpeech() {
      clearTimer();
      if (ttsOk) { try { synth.cancel(); } catch {} }
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
    }

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

    function speakCurrent() {
      const step = spec.steps[state.step];
      stopSpeech();
      const advance = () => {
        const tokenMode = state.mode;
        state = narratedReducer(state, { type: "ADVANCE" });
        if (state.mode === "playing" && tokenMode === "playing") {
          run();
        } else {
          syncControls();
        }
      };
      if (!muted && ttsOk) {
        const u = new SpeechSynthesisUtterance(step.say);
        u.onend = () => { if (state.mode === "playing") advance(); };
        u.onerror = () => { if (state.mode === "playing") advance(); };
        try { synth.speak(u); }
        catch { noAudioTimer = setTimeout(() => { if (state.mode === "playing") advance(); }, computeNoAudioMs(step.say)); }
      } else {
        noAudioTimer = setTimeout(
          () => { if (state.mode === "playing") advance(); },
          computeNoAudioMs(step.say)
        );
      }
    }

    async function run() {
      await renderStep(state.step);
      syncControls();
      if (state.mode === "playing") speakCurrent();
    }

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
      if (muted) stopSpeech();
    });

    // Initial paint: first step visible, idle (no autoplay).
    renderStep(0).then(syncControls);
  }
```

- [ ] **Step 2: Wire the `lt-narrated` fence into `appendTutor`**

In `appendTutor` (the mermaid-fence block ~lines 379-398), replace this exact block:

```js
      // Split on mermaid fences. Even indices = text, odd indices = mermaid code.
      const fence = /```mermaid\n([\s\S]*?)\n```/g;
      let lastIndex = 0;
      let m;
      let any = false;
      while ((m = fence.exec(text)) !== null) {
        any = true;
        const before = text.slice(lastIndex, m.index);
        if (before.trim()) appendParagraphsBold(wrap, before);
        const host = document.createElement("div");
        host.className = "lt-mermaid-host";
        wrap.appendChild(host);
        // Fire-and-forget; the placeholder appears synchronously.
        renderMermaidInto(host, m[1]);
        lastIndex = m.index + m[0].length;
      }
```

with:

```js
      // Split on lt-narrated AND mermaid fences. A combined scanner keeps
      // text/diagram/narrated ordering intact.
      const fence = /```(lt-narrated|mermaid)\n([\s\S]*?)\n```/g;
      let lastIndex = 0;
      let m;
      let any = false;
      while ((m = fence.exec(text)) !== null) {
        any = true;
        const before = text.slice(lastIndex, m.index);
        if (before.trim()) appendParagraphsBold(wrap, before);
        const host = document.createElement("div");
        wrap.appendChild(host);
        if (m[1] === "lt-narrated") {
          const spec = parseNarrated(m[2]);
          if (spec) {
            host.className = "lt-narrated-host";
            renderNarratedInto(host, spec);
          } else {
            // Fallback: treat the raw body as a failed-mermaid-style card.
            host.className = "lt-mermaid lt-mermaid--failed";
            const pre = document.createElement("pre");
            pre.textContent = m[2];
            host.appendChild(pre);
          }
        } else {
          host.className = "lt-mermaid-host";
          renderMermaidInto(host, m[2]); // fire-and-forget
        }
        lastIndex = m.index + m[0].length;
      }
```

- [ ] **Step 3: Re-run the pure-helper tests (regression guard)**

Run: `node --test tests/js/test_narrated.js`
Expected: PASS — 8/8. (Confirms the edits above didn't disturb the guarded helpers / early return.)

- [ ] **Step 4: Commit**

```bash
git add app/static/lab-tutor.js
git commit -m "feat(lab-tutor): narrated-whiteboard renderer + playback wiring

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Styles

**Files:**
- Modify: `app/static/lab-tutor.css` (append a new section at end of file, after the `.lt-mermaid` rules ending ~line 395)

- [ ] **Step 1: Append the narrated-whiteboard styles**

Append to `app/static/lab-tutor.css`:

```css

/* Narrated whiteboard */
.lt-narrated {
  margin: 8px 0;
  padding: 12px;
  background: #fafafa;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}
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
.lt-narrated-stage svg {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}
.lt-narrated-caption {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.5;
  color: var(--lt-text);
  min-height: 1.5em;
}
.lt-narrated-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}
.lt-narrated-btn {
  padding: 4px 10px;
  border: 1px solid var(--lt-border);
  background: var(--lt-surface);
  border-radius: 6px;
  font-size: 12px;
  font-family: var(--lt-font);
  color: var(--lt-text);
  cursor: pointer;
  transition: background 0.1s ease;
}
.lt-narrated-btn:hover:not(:disabled) {
  background: var(--lt-surface-2);
}
.lt-narrated-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.lt-narrated-controls .lt-narrated-btn:first-child {
  margin-right: auto;
}
```

- [ ] **Step 2: Commit**

```bash
git add app/static/lab-tutor.css
git commit -m "feat(lab-tutor): narrated-whiteboard styles + reduced-motion fallback

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Tutor prompt guidance (TDD)

**Files:**
- Modify: `app/services/tutor_service.py` — extend `_TUTOR_PERSONA` (string ends at line 44)
- Test: `tests/test_tutor_service.py` (append a new test method/class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tutor_service.py`:

```python
class TutorPersonaNarratedContract(unittest.TestCase):
    def test_persona_documents_lt_narrated_and_single_pass(self) -> None:
        from app.services.tutor_service import _TUTOR_PERSONA

        self.assertIn("```lt-narrated", _TUTOR_PERSONA)
        # JSON shape the widget parses
        self.assertIn('"steps"', _TUTOR_PERSONA)
        self.assertIn('"say"', _TUTOR_PERSONA)
        self.assertIn('"mermaid"', _TUTOR_PERSONA)
        # single-pass constraint must be stated
        self.assertIn("same reply", _TUTOR_PERSONA.lower())
        # plain mermaid path must still be documented
        self.assertIn("```mermaid", _TUTOR_PERSONA)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tutor_service.py::TutorPersonaNarratedContract -v`
Expected: FAIL — `AssertionError` (no `` ```lt-narrated `` in the persona yet).

- [ ] **Step 3: Add the guidance to the persona**

In `app/services/tutor_service.py`, replace the final bullet of `_TUTOR_PERSONA` (line 44):

```
- Don't draw a diagram for every reply. Skip it when the answer is a one-line concept, a syntax lookup, or a quick yes/no. The bar: would a real tutor reach for the whiteboard here? If not, stay in prose."""
```

with:

```
- Don't draw a diagram for every reply. Skip it when the answer is a one-line concept, a syntax lookup, or a quick yes/no. The bar: would a real tutor reach for the whiteboard here? If not, stay in prose.
- When the explanation is a multi-step PROCESS (a pipeline, a state machine, a data flow, an algorithm walkthrough) and building it up stage by stage would help more than one static picture, use a narrated whiteboard instead of a plain diagram. Emit a fenced block tagged lt-narrated whose body is JSON with a "steps" array; each step has a one-sentence "say" line and its own CUMULATIVE "mermaid" source (step N draws nodes 1..N). 3-6 steps. Produce the whole JSON in the SAME reply (one pass — never promise to send it next):
  ```lt-narrated
  {"steps":[
    {"say":"First the query is embedded.","mermaid":"flowchart LR\\n  Q[Query]-->E[Embed]"},
    {"say":"Then we search the vector store.","mermaid":"flowchart LR\\n  Q[Query]-->E[Embed]-->S[Search]"}
  ]}
  ```
  Use the plain ```mermaid path for a single static diagram; use lt-narrated only when the step-by-step build is the teaching point. Still pair it with one short follow-up question or hint."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tutor_service.py::TutorPersonaNarratedContract -v`
Expected: PASS.

- [ ] **Step 5: Run the full tutor-service suite (regression)**

Run: `python -m pytest tests/test_tutor_service.py -v`
Expected: PASS — all pre-existing tests still green (the line-98 "plain persona" test must not regress; the new bullet is appended, persona-only behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add app/services/tutor_service.py tests/test_tutor_service.py
git commit -m "feat(lab-tutor): persona guidance for lt-narrated (single-pass)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Full regression + manual browser verification

**Files:** none (verification only)

- [ ] **Step 1: Full JS + Python regression**

Run: `node --test tests/js/ && python -m pytest tests/test_tutor_service.py tests/test_tutor_routes.py -q`
Expected: all PASS.

- [ ] **Step 2: Manual browser verification (evidence required)**

Local only — no remote deploy. Start the app locally with `LAB_TUTOR_BASE_URL=http://127.0.0.1:8012`, launch a learner studio container, open the editor, open the tutor, and prompt something that triggers a multi-step process explanation (e.g. "walk me through the RAG pipeline step by step"). Confirm and record evidence (screenshot or written observation) for EACH:
  - [ ] A narrated card renders with a "▶ Play narration" button (no autoplay).
  - [ ] Play reveals step 1's diagram + caption, speaks it, then auto-advances to step 2 on speech end (event-based sync).
  - [ ] Pause stops speech and the cursor does not jump (stale-callback guard holds).
  - [ ] Prev/Next move steps and disable correctly at the ends.
  - [ ] Mute → playback continues visually on the no-audio timer (~2s+/step).
  - [ ] Reload the page: the card re-renders fresh at idle (documented V1 behavior — position not persisted), chat history otherwise intact.
  - [ ] A deliberately malformed `lt-narrated` block falls back to a raw card and does NOT break the message log.

- [ ] **Step 3: Update the running-feature todo and report**

Do NOT claim completion until every Step-2 box has recorded evidence (verification-before-completion). Report results to the user. No `git push` / no remote deploy without explicit approval.

---

## Self-Review

**Spec coverage:**
- Block format (`lt-narrated`, JSON, cumulative mermaid) → Task 1 (`parseNarrated`) + Task 4 (persona emits it).
- Widget renderer + Play button (no autoplay) + controls + caption fallback → Task 2 + Task 3.
- Event-based sync, no-audio timer `max(2s, words*240)`, stale-callback guard → Task 1 (reducer/timer) + Task 2 (wiring).
- Per-step CSS entry transition + reduced-motion → Task 3.
- Parser robustness ladder → Task 1.
- Tutor prompt single-pass guidance → Task 4.
- Verification (Python contract + JS units + manual) → Tasks 1, 4, 5.
- Non-goals (no games/video/server-TTS/OpenMAIC/course-gen/deploy) → respected; no such tasks; Task 5 Step 3 forbids push.
- Spec "persist last-played step" → consciously deferred; documented under File Structure as a V1 simplification and surfaced in Task 5 Step 2 / handoff.

**Placeholder scan:** No TBD/TODO; every code step contains complete code; commands have expected output.

**Type consistency:** `parseNarrated` returns `{steps:[{say,mermaid}]}` — consumed identically in Task 2. `narratedReducer(state,action)` state shape `{mode,step,total}` and action types `PLAY/PAUSE/ADVANCE/NEXT/PREV/REPLAY/STOP` consistent across Task 1 tests and Task 2 wiring. `computeNoAudioMs(say)` signature consistent. CSS class names (`lt-narrated`, `lt-narrated-stage`, `lt-narrated-stage--in`, `lt-narrated-caption`, `lt-narrated-controls`, `lt-narrated-btn`, `lt-narrated-host`) match between Task 2 and Task 3.
