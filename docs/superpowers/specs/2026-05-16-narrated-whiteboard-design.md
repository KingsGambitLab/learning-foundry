# Narrated Whiteboard — Tutor Enhancement Design

Date: 2026-05-16
Branch: `claude/agitated-hamilton-f1b773`
Status: Approved (design), pre-implementation

## Goal

Make the Lab Tutor's conceptual explanations more engaging by delivering them
as a **step-by-step diagram reveal synced with voice narration** — the feel of
a short Khan-Academy explainer, without generating any video and without any
additional LLM call.

This is the lightweight, in-IDE reimplementation of OpenMAIC's narrated-
simulation idea. OpenMAIC itself is **not** integrated (AGPL-3.0, Node/LangGraph
stack, classroom-app UX that would force a context switch out of the editor).

## Non-Goals (explicit)

- Games / quizzes / MCQs.
- Video or talking-head avatar generation.
- Server-side TTS (OpenAI/ElevenLabs/VoxCPM). Browser TTS only.
- Any OpenMAIC integration.
- Any change to the course-gen authoring pipeline (assignment generation,
  judges, starters). Per standing rule, core course-gen edits are surfaced as
  a flagged todo, not bundled here.
- Any remote deployment. Build and verify locally only; no push to remote
  without explicit approval (remote is under a deploy freeze).

## Cost Model

- Browser `SpeechSynthesis` (Web Speech API): $0. Local OS voice engine — not
  an LLM, no API, no server, no network.
- Mermaid render + step reveal animation: $0. Client-side only.
- Narration script + diagram spec: LLM-generated, but **part of the existing
  single tutor reply** — no new API call. Marginal change in output tokens
  (a few hundred), roughly a wash with the prose it replaces.
- Hard constraint: **single-pass only**. The script must be emitted in the
  same tutor completion as the diagram. No second "rewrite into narration"
  LLM pass.

## Architecture

Reuses the existing structured-fenced-block pattern already proven by the
Mermaid feature (`app/static/lab-tutor.js` parses ```` ```mermaid ```` blocks
and renders them via a client-side renderer).

### Block format

The tutor emits a fenced block tagged `lt-narrated` whose body is JSON:

```json
{
  "steps": [
    { "say": "First we chunk the documents.",
      "mermaid": "flowchart LR\n  A[Docs] --> B[Chunk]" },
    { "say": "Each chunk is embedded into a vector.",
      "mermaid": "flowchart LR\n  A[Docs] --> B[Chunk] --> C[Embed]" }
  ]
}
```

- Each step carries its **own cumulative Mermaid source**: step N renders the
  diagram containing nodes/edges 1..N.
- Rationale for cumulative re-render over animating one base SVG: Mermaid
  auto-generates SVG element ids, so reliably targeting "reveal node 3" by id
  is brittle across diagram types and Mermaid versions. Re-rendering a small
  diagram per step is cheap (client-side, sub-frame for <10 nodes) and
  deterministic.
- Recommended shape: 3–6 steps, one short sentence per `say`, small diagrams.

### Widget renderer (`app/static/lab-tutor.js`)

A new renderer registered alongside the existing mermaid renderer.

- Parser: detect ```` ```lt-narrated ```` fences, `JSON.parse` the body.
  Malformed JSON or missing/empty `steps` → graceful fallback card showing the
  raw text (mirrors the existing `.lt-mermaid--failed` behavior).
- Render: a card with a **▶ Play narration** button. No autoplay:
  1. Browsers gate audio/speech behind a user gesture.
  2. Surprise speech at a learner mid-task is poor UX.
- Playback state machine per narrated card:
  - `idle → playing → (paused) → done`
  - On entering step i: render `steps[i].mermaid` into the card's diagram
    slot; show `steps[i].say` as a visible caption; speak `say` via
    `SpeechSynthesis`; on utterance `onend` advance to i+1; at end → `done`.
- Controls: Play/Pause, Prev, Next, Replay, Mute. Caption text is **always**
  visible regardless of audio (accessibility, sound-off, and
  `prefers-reduced-motion` users get a "show all steps" static fallback).
- Persistence: last-played step index + mute state stored in the existing
  enrollment-scoped localStorage chat record, so a narrated card survives
  reload like chat messages already do.

### Audio

- `window.speechSynthesis` only.
- Handle the `voiceschanged` load race (voices often unavailable on first
  synchronous `getVoices()` call): resolve voice list on `voiceschanged` or a
  short poll, pick a sane default (prefer a local en-US voice), fall back to
  the platform default.
- Mute toggle persisted per the persistence note above. Muted playback still
  advances steps on a timer derived from `say` length (so the visual still
  plays without audio).

### Tutor prompt (`app/services/tutor_service.py`)

Add guidance next to the existing Mermaid nudge in the persona/system prompt:

- When the explanation is a **multi-step process** (pipeline, state machine,
  data flow, algorithm walkthrough), prefer an `lt-narrated` block: 3–6 steps,
  cumulative diagrams, one short sentence per step.
- Keep the existing plain ```` ```mermaid ```` path for a single static
  diagram. Keep the "don't diagram every reply" bar.
- Emit the JSON in one reply (single-pass constraint).

## Error Handling

- Malformed `lt-narrated` JSON → fallback card with raw content + a short
  error line. Never throw into the message log.
- A step whose `mermaid` fails to render → show that step's `say` caption with
  a small "diagram unavailable" note; narration/stepping continues.
- `speechSynthesis` unavailable (older browser / blocked) → silent mode:
  captions + manual Next still work; Mute control hidden or disabled.
- Voices not yet loaded at Play time → defer first utterance until
  `voiceschanged` (bounded wait), then proceed.

## Testing / Verification

- **Python (TDD):** assert the tutor service / persona contract — the system
  prompt contains the `lt-narrated` guidance and the single-pass constraint;
  cover any server-side parsing/validation we add for the block.
- **JS (unit-testable seams):** factor the block parser and the playback
  step-state reducer as pure functions so they can be unit-tested without a
  DOM/browser; test malformed input, cumulative step progression, mute-timer
  advance, persistence (de)serialization.
- **Manual:** run a local code-server container with the widget injected,
  trigger a narrated explanation, verify reveal+voice sync, controls,
  reload persistence, and the no-audio fallback. Evidence before any
  "it works" claim (verification-before-completion).

## Rollout

- Local only. No remote deploy. No push without explicit approval.
- Feature is additive: if the tutor never emits `lt-narrated`, behavior is
  unchanged from today.
