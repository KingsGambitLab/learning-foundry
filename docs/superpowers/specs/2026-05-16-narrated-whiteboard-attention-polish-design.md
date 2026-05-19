# Narrated Whiteboard — Attention Polish Design

Date: 2026-05-16
Branch: `claude/agitated-hamilton-f1b773`
Status: Approved (design), pre-implementation
Builds on: `2026-05-16-narrated-whiteboard-design.md` (the shipped narrated-whiteboard feature)

## Goal

Increase engagement of the existing narrated-whiteboard card by making each
step visually arresting and learner-controllable — without touching voice
quality, adding LLM calls, persistence, or new infra. Scope is deliberately
"Bundle A: attention polish only": **new-node spotlight + motion polish +
keyboard/progress-dots**.

## Non-Goals (explicit)

- Caption word-sync, predict-the-next gate, recap takeaway, code-anchored
  narrative (these were Bundles B/C — out of scope here).
- Tap-to-advance; any change to the no-autoplay behavior.
- Any new LLM call, persistence-schema change, server change, or new file
  injection site.
- Voice/TTS changes of any kind.
- Remote deploy. Local build + verify only; no push without explicit approval.

## Context

The shipped feature renders an `lt-narrated` card via `renderNarratedInto`
in `app/static/lab-tutor.js`. Per step it sets a caption and re-renders the
step's **cumulative** Mermaid source (`renderMermaidInto`), then plays the
entry transition via the `.lt-narrated-stage--in` class. Playback is the
pure `narratedReducer` state machine; pure helpers (`parseNarrated`,
`computeNoAudioMs`, `narratedReducer`) are unit-tested under Node
(`tests/js/test_narrated.js`) via a module-export guard. There is no JS DOM
test harness in the repo by design.

## Architecture

Three independent, additive sub-features. Each degrades gracefully and is
off the critical path (if any fails, the card still plays exactly as today).

### 1. New-node spotlight

Per the original design, each step carries its OWN cumulative Mermaid
source. To emphasize only what changed, diff the SOURCE (not the rendered
SVG — SVG-id targeting is brittle across Mermaid versions and was
deliberately avoided in the base feature).

**Pure helpers (Node-unit-tested, added to the existing guarded block):**

- `extractNodeLabels(mermaidSrc)` → ordered array of distinct human-visible
  node label strings. Parses flowchart/graph node declarations:
  - `A[Label]`, `A(Label)`, `A((Label))`, `A{Label}`, `A>Label]`,
    `A([Label])`, `A[[Label]]` — capture the bracketed label.
  - Bare node ids that appear in edges without a label declaration
    (`A --> B`) contribute the id as its own label only if the id never
    receives a bracketed label anywhere in the source.
  - Ignores the diagram-type header line (`flowchart LR`, `graph TD`,
    `stateDiagram-v2`, etc.) and edge-label text (`A -->|text| B`).
  - Deterministic, no DOM, no regex catastrophic backtracking (bounded
    quantifiers).
- `diffNewNodeLabels(prevMermaidSrc, currMermaidSrc)` → labels present in
  `curr` but not in `prev` (set difference on `extractNodeLabels` output,
  order preserved by `curr`). For step 0 (`prev` undefined/empty) → `[]`
  (no spotlight on the first paint; nothing is "new").

**DOM application (in `renderNarratedInto`, best-effort, non-fatal):**

After `renderMermaidInto(stage, step.mermaid)` resolves for step i where
i > 0:
1. Compute `newLabels = diffNewNodeLabels(spec.steps[i-1].mermaid,
   spec.steps[i].mermaid)`.
2. For each new label, find the FIRST SVG node element in `stage` whose
   trimmed text content equals the label, scoped to Mermaid node containers
   (`g.node`, falling back to any `.nodeLabel`/`text` whose trimmed text
   matches). Add class `lt-narrated-spot`.
3. Remove `lt-narrated-spot` after 1000ms (slightly longer than the CSS
   animation) so re-render of the next step starts clean.
4. Any exception or zero matches → swallow and continue (spotlight is a
   nicety, never a failure path). Matching is by the label the learner
   actually sees, so it tracks Mermaid markup changes.

### 2. Motion polish

Replace the `.lt-narrated-stage--in` transition timing function with a
spring-ish `cubic-bezier(0.22, 1, 0.36, 1)` and increase initial travel
slightly (`translateY(10px) scale(0.96)` → settle). Pure CSS. The existing
`@media (prefers-reduced-motion: reduce)` block already neutralizes the
stage transform/opacity/transition and must also neutralize the new
spotlight animation.

### 3. Keyboard + progress dots

- **Progress dots:** add a `.lt-narrated-dots` container (built once from
  `spec.steps.length`) of `.lt-narrated-dot` spans. `syncControls` toggles
  `.lt-narrated-dot--active` on the dot at `state.step`. Decorative:
  `aria-hidden="true"` (the buttons already carry the accessible state).
  Placed between the caption and the controls row.
- **Keyboard:** the card root gets `tabindex="0"`, `role="group"`, and
  `aria-label="Narrated whiteboard"`. A `keydown` listener is attached to
  the CARD ELEMENT (never `document`) so it cannot interfere with
  code-server / Monaco / page shortcuts:
  - `Space` → same as clicking play/pause; `event.preventDefault()` to
    stop the page scrolling.
  - `ArrowRight` → same as Next button; `ArrowLeft` → same as Prev button.
  - Reuse the EXISTING button click handlers' logic (extract the
    play/pause, next, prev bodies into named functions the listeners and
    keys share — no behavior change, just shared call sites).
  - Any other key → ignored (no preventDefault).

## Data Flow

`spec.steps[i].mermaid` (already in memory) → `diffNewNodeLabels` → label
list → DOM query within the freshly rendered `stage` → transient CSS class.
No new data, no network, no storage. Dots derive purely from
`spec.steps.length` and `state.step`. Keyboard maps to existing playback
actions through `narratedReducer` (unchanged).

## Error Handling

- `extractNodeLabels`/`diffNewNodeLabels` are total functions: malformed or
  exotic Mermaid → return whatever labels parse (possibly `[]`); never
  throw. Unit tests cover empty input, no-new-node steps, label-shape
  variants, and a non-flowchart header.
- Spotlight DOM step is wrapped so any error (no SVG, no match, unexpected
  markup) is swallowed; the card continues normally.
- Keyboard handler only acts on the three keys; everything else passes
  through. Listener is card-scoped and removed implicitly when the message
  log is cleared (same lifecycle as today's listeners; no new global
  state).
- Reduced-motion users: no spotlight animation, no stage transition (CSS
  media query), but dots + keyboard still work.

## Testing / Verification

- **Node unit tests** (extend `tests/js/test_narrated.js`):
  `extractNodeLabels` (bracket shapes, bare-id edges, header/edge-label
  exclusion, dedupe/order) and `diffNewNodeLabels` (step 0 → [], added
  node, no-change step, label rename treated as add). Run:
  `node --test tests/js/test_narrated.js`.
- **node --check** on `lab-tutor.js`; full Node suite green.
- **Manual browser verification** (consistent with base feature; no DOM
  harness): trigger a narrated reply, confirm — only the newly added node
  pulses each step; motion feels like a settle not a blink; dots track the
  step; `Space`/`←`/`→` work when the card is focused and do NOT scroll the
  page or leak to the editor; reduced-motion disables animations but keeps
  dots/keys; a non-flowchart or odd diagram simply doesn't spotlight (no
  breakage). Evidence required before any "it works" claim.

## Rollout

Additive and local-only. If `diffNewNodeLabels` returns `[]` or matching
fails, behavior is identical to the shipped feature. No remote deploy, no
push without explicit approval.
