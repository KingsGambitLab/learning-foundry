import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { HintBudget } from "../src/state/hint-budget";

describe("HintBudget", () => {
  it("starts with the given capacity and zero consumed", () => {
    const b = new HintBudget(4);
    assert.equal(b.remaining, 4);
    assert.equal(b.consumed, 0);
  });

  it("decrements remaining on consume", () => {
    const b = new HintBudget(4);
    b.consume();
    assert.equal(b.remaining, 3);
    assert.equal(b.consumed, 1);
  });

  it("clamps at zero and reports exhausted", () => {
    const b = new HintBudget(1);
    b.consume();
    b.consume();
    assert.equal(b.remaining, 0);
    assert.equal(b.exhausted, true);
  });

  it("formats a human label", () => {
    const b = new HintBudget(4);
    b.consume();
    assert.equal(b.label, "Hints: 3/4");
  });

  it("formats the label when exhausted", () => {
    const b = new HintBudget(2);
    b.consume();
    b.consume();
    assert.equal(b.exhausted, true);
    assert.equal(b.label, "Hints: 0/2");
  });
});
