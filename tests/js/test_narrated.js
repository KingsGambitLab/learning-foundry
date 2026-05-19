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

test("parseNarrated: repairs literal newlines/tabs inside string values", () => {
  // Raw newline + tab inside the mermaid value (invalid JSON as-is).
  const body = '{"steps":[{"say":"Build it.","mermaid":"flowchart LR\n\tA-->B\n\tB-->C"}]}';
  const out = lt.parseNarrated(body);
  assert.ok(out, "expected a parsed object, got null");
  assert.equal(out.steps.length, 1);
  assert.equal(out.steps[0].say, "Build it.");
  assert.equal(out.steps[0].mermaid, "flowchart LR\n\tA-->B\n\tB-->C");
});

test("narratedReducer: STOP resets to idle/step 0", () => {
  const s = lt.narratedReducer({ mode: "playing", step: 2, total: 4 }, { type: "STOP" });
  assert.deepEqual(s, { mode: "idle", step: 0, total: 4 });
});
