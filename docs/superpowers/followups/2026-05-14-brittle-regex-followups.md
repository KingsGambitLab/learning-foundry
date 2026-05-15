# Brittle regex / string-matching follow-ups

**Date:** 2026-05-14
**Branch:** `virtusa_assignment`
**Context:** Audit pass after the LLM-authored `core_entities` fix
(commit `adab73cc`). The same naive-substring pattern shows up in
several other harness paths; cataloging here so they can be picked off
in subsequent commits without inline detours during course runs.

## TIER A — Live false-positive risk

### A.1 Tool-use / approval / trace claim checks (substring)

**Location:** [`app/services/bundle_validation.py:397-413`](../../../app/services/bundle_validation.py)

```python
if not spec.capabilities.tool_use_required and ("tool-use" in lowered or "tool use" in lowered):
    _add_issue(... code="course_readme_unbacked_tooling_claim" ...)
if not spec.capabilities.approval_flow_required and "approval" in lowered:
    _add_issue(... code="course_readme_unbacked_approval_claim" ...)
if not spec.capabilities.traceability_required and "trace" in lowered:
    _add_issue(... code="course_readme_unbacked_trace_claim" ...)
```

**Brittle because:**
- *"tool use is NOT required for this course"* → flagged (negation ignored).
- *"PR approval workflow"* mention → flagged as approval-flow claim.
- *"stack trace"* in error-handling section → flagged as trace claim.

**Fix shape:** one Haiku judge per check — given README + capability flag, decide whether the README actually claims to implement the capability. Same plumbing as the domain-grounding judge in `public_surface_quality_llm.py` (commit `2992f68f`), reusing `BundleValidationIssue.hint` for actionable suggestions.

**Effort:** ~1 hr (extract a generic `evaluate_capability_claim(content, capability, required) -> CapabilityClaimVerdict` helper, fan out to the 3 callsites with a deterministic-fallback hook).

### A.2 `extract_project_entities` still runs on the live path

**Location:** [`app/services/public_surface_quality.py:124`](../../../app/services/public_surface_quality.py), called from `assignment_design_inference.build_project_contract` and indirectly elsewhere.

**Brittle because:** the LLM-authored entities now flow through (`adab73cc`), but the regex still runs in parallel and leaks into `inferred_entities` for `system_kind` template substitution and per-deliverable archetype detection. That re-introduces `"a small"`-class output even when the LLM gave us better entities.

**Fix shape:** when the design-inference path has access to `ProjectContractSpec` already populated with LLM-authored entities, **skip** `extract_project_entities` entirely. Keep the regex strictly as a no-LLM fallback when `core_entities` is empty.

**Effort:** ~30 min (guard the call site; rebuild the `primary_entity` derivation from the LLM-authored list when present).

### A.3 Design archetype keyword matching

**Location:** [`app/services/assignment_design_inference.py:311`](../../../app/services/assignment_design_inference.py) — `text = " ".join([title, problem_statement]).lower()` then substring matches to pick `ProjectFamily.grounded_retrieval_service` / `search_service` / `workflow_service` / `control_plane` / `service`.

**Brittle because:** novel domains (MLOps, agentic, data pipelines, prompt-eng) fall through to the generic `service` family. Wrong family → wrong `runtime_plan` → wrong starter scaffold. The promptfoo brief hit this — fell to generic.

**Fix shape:** have the course-planner LLM emit `project_family` directly, same pattern as `system_kind/core_entities` in `adab73cc`. Regex stays as no-LLM fallback.

**Effort:** ~2 hr. The field has more downstream consumers than `system_kind/core_entities`; need to verify everywhere that reads `ProjectContractSpec.family` is OK with values authored from a richer set.

## TIER B — Working but brittle

### B.1 Sandbox failure summarization drops the full stack

**Location:** [`app/services/docker_sandbox_runner._summarize_stage_failure`](../../../app/services/docker_sandbox_runner.py) (around line 2003+).

**Observed:** on the promptfoo run, a 30-line npm-gyp error got summarized to one line (`gyp failed with exit code 1`). The repair LLM lost the actionable detail (which package, which native dep, what was missing). Sonnet 4.6 recovered anyway, but a less-canonical failure would have cycled through every attempt.

**Fix shape:** stuff the last N lines (say 50) of stderr into `BundleValidationIssue.detail` (or a new `evidence` field). Don't truncate to the regex-extracted headline.

**Effort:** ~30 min.

### B.2 `_owner_hint` platform-marker detection

**Location:** [`app/services/failure_context_builder._owner_hint:279`](../../../app/services/failure_context_builder.py).

Substring match list (`"cannot connect to the docker daemon"`, `"port is already allocated"`, etc.) to classify "platform fault" vs "learner-side bug". Works today because Docker error strings are stable, but couples repair-routing to Docker's exact wording.

**Fix shape:** have `DockerSandboxRunner` emit a structured `SandboxFailureKind` enum at the source (where it knows whether it tried `docker network create` vs `docker run`), rather than parsing the resulting log string after the fact.

**Effort:** ~2 hr. Touches the sandbox runner's error-construction sites.

### B.3 `_OVERSTATED_WORKFLOW_MARKERS` substring list

**Location:** [`app/services/bundle_validation.py:78`](../../../app/services/bundle_validation.py).

Specific phrases (`"agentic system"`, `"tool-use policies"`). Low false-positive risk because the phrases are unusual. **Acceptable as-is** unless a future course brief happens to mention them in a legitimate sense.

## TIER C — Deterministic, no action

| Site | Why it's fine |
|---|---|
| `langgraph_assignment_graph.py:1006` title-to-slug `re.sub(r"[^a-z0-9]+", "_", ...)` | Alphanumeric squashing, idempotent. |
| `task_agent_retry_service.py:483` same pattern | Idem. |
| `creator_asset_service.py:97,117` filename sanitization | Idem. |
| `failure_context_builder.py:457` whitespace + address anonymization | Defensive normalization, no semantic decisions. |
| `stack_catalog_service.py:528-544` version parsing `r"\b(?:v)?(\d+(?:\.\d+)*)\b"` | Works for semver; non-semver pre-release / build-metadata is an edge case worth a separate eval, not a brittleness fix. |

## Suggested order to land

1. **A.2** (skip `extract_project_entities` on live LLM path) — smallest diff, immediately compounds `adab73cc`.
2. **B.1** (full stderr in failure detail) — independent, high-leverage for repair-loop quality.
3. **A.1** (LLM judge for the 3 capability-claim checks) — same plumbing pattern as the domain-grounding judge.
4. **A.3** (LLM-authored `project_family`) — larger, can wait until 1-3 are in.

## Out of scope for this list

- Prompt-engineering changes to teach the LLM specific phrasings — that's a different lever and we've already added two prompts (commits `7e3c6727`, `adab73cc`). Further prompt tweaks belong with the corresponding code change.
- LLM-side determinism / temperature tuning — separate concern.
- Replacing `BundleValidationIssue` with a richer "evidence + suggestion + hint" model — refactor; defer until the per-finding pattern proves it needs a structured slot for evidence (B.1 may force this).
