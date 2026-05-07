# UI Review Note — 2026-05-07

This note is based on a browser review of:

- `/create-course`
- the creator funnel: brief -> outcomes -> setup -> plan
- a live draft review page
- `/`
- `/courses`

## Overall read

The product direction is much better than before.

What feels meaningfully improved:
- the creator funnel is the right shape
- the setup step asks the right questions
- the module-plan review is understandable
- the learner flow explains visible checks vs hidden grading better

What still needs work:
- the transition from plan -> draft feels unreliable
- the draft review page still reads like an internal/admin surface
- the pages still carry more chrome and metadata than a normal creator needs

## Keep

### 1. Keep the creator funnel
This is the strongest part of the new flow:

1. Describe
2. Outcomes
3. Setup
4. Plan

This is much better than exposing raw spec review first.

### 2. Keep the setup choices
These are good creator-facing questions:
- starter code vs blank
- database
- cache

They feel practical and product-shaped.

### 3. Keep the module-plan review
The flight-booking example reads well:
- core booking contract and seat inventory
- pessimistic locking in postgres
- optimistic locking and retries in postgres
- redis for availability reads
- production hardening and failure drills

This is the right HIL surface for creators.

## Highest-priority fixes

### 1. Fix the `Start building` transition
This is the biggest issue.

Current behavior:
- clicking `Start building` shows a loading/status message
- the page stays on the plan screen
- it is not obvious whether the draft was actually created

Expected behavior:
- immediately route to the created draft URL
- make success obvious
- switch into the draft review/playground state automatically

Why this matters:
- this is the moment where creator trust is won or lost
- if this feels broken, the whole product feels unreliable

### 2. Split creator mode from admin/debug mode more aggressively
The draft review page still leads with internal concepts like:
- linked workflow
- review assignment spec
- contract / tools / checks / endpoints
- technical details

That is still too internal for the default creator experience.

Default creator draft view should emphasize:
- current state
- what is waiting on me vs the agent
- what is approved so far
- what happens next
- learner preview / draft playground
- publish readiness

Technical review/spec surfaces should still exist, but behind:
- an `Admin` tab
- a `Technical details` drawer
- or a clearly secondary debug surface

### 3. Compress the top-of-page chrome
The creator funnel is good, but the page still spends too much height on:
- top shell
- workflow chrome
- fallback banner/status
- tabs

The active task should dominate the page more quickly.

## Medium-priority fixes

### 4. Improve draft-list differentiation
The drafts list becomes hard to scan when multiple drafts have similar titles.

Add stronger identity signals, such as:
- latest/current badge
- version badge
- last meaningful action
- short goal snippet
- clearer timestamp hierarchy

### 5. Tighten the learner brief hierarchy
The learner page has the right information, but it is still text-heavy.

The top of the learner module should make these obvious immediately:
- file(s) to edit
- how to run visible checks
- how to submit
- what “done” means

The fuller brief can sit below that.

## Bugs / rough edges

### 6. Fix content formatting issues
Observed issue:
- learner brief sections like `What you will practice` are partially rendered as mashed text instead of clean bullets

This makes the content feel less trustworthy and harder to skim.

### 7. Fix the initial problem-statement interaction edge case
Observed issue:
- the field visually appeared filled from the placeholder/example
- but the app still treated the problem statement as empty until I actually typed

The state should be unambiguous:
- example text should look like example text
- real input should look like real input

## Suggested implementation order

1. Fix `Start building` -> draft routing and success state
2. Reduce admin/debug leakage in the default draft view
3. Compress top-of-page chrome so the active task appears sooner
4. Fix formatting/rendering issues in creator + learner content
5. Improve draft-list identity and scannability
6. Tighten learner-page top-of-page hierarchy

## One-line summary

The creator funnel is now the right product idea, but the post-plan transition and the draft review surface still feel too much like internal tooling.
