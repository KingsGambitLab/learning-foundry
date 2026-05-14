# Scenario Rubrics — RAG MVP

**Date:** 2026-05-14
**Status:** Design + implementation in progress
**Scope:** Phase-1 implementation of the rubric library that powers
scenario-driven grading. RAG-only for now.

## Background

The single-outcome course pipeline (see `docs/superpowers/specs/...`)
grades learner submissions by running curated scenarios against the
learner's service and applying rubrics to the captured HTTP responses.
Each rubric is a small class with one method: given a `RubricContext`
(captures from the trace run + setup data + course meta), return a
`Verdict` (pass / fail / abstain + rationale + diagnostic).

Rubric classes split cleanly into three flavors:

- **Structural** — inspect shape and primitive values; no network, no LLM.
- **Set / semantic-equivalence** — compare collections or labeled
  behavior; no LLM.
- **LLM-judged** — call Haiku to evaluate subjective dimensions like
  answer faithfulness; needs `LLMRouter`.

## Class roster

The full design surface is **seventeen rubric classes**, organized by
the question they answer. This MVP implements the **eight needed for
RAG**; the other nine are documented here so the framework is shaped to
admit them without rewrites, but their bodies are deferred.

| # | Class | Question it answers | Flavor | Status |
|---|---|---|---|---|
| 1 | `SchemaMatch` | "Does the response have these fields with these types?" | Structural | **IMPLEMENT** |
| 2 | `LiteralMatch` | "Does this value equal an exact expected value?" | Structural | **IMPLEMENT** |
| 3 | `RegexMatch` | "Does this string match a pattern?" | Structural | **IMPLEMENT** |
| 4 | `NumericRange` | "Is this number in [min, max]?" | Structural | **IMPLEMENT** |
| 5 | `NumericTolerance` | "Is this number within ε of an expected value?" | Structural | DEFER |
| 6 | `SubsetMatch` | "Is set A ⊆ set B?" | Set | **IMPLEMENT** |
| 7 | `SetMatch` | "Does set A == set B (order doesn't matter)?" | Set | DEFER |
| 8 | `OrderPreservation` | "Is the list sorted by a key (with tie tolerance)?" | Set | DEFER |
| 9 | `BehavioralEquivalence` | "Does this categorical behavior match expected?" | Set | **IMPLEMENT** |
| 10 | `OracleSetOverlap` | "Does retrieval top-k overlap the gold set by ≥ X?" | Set / oracle | **IMPLEMENT** |
| 11 | `LLMJudgeCoverage` | "Does the answer cover these required facts?" | LLM | **IMPLEMENT** |
| 12 | `LLMJudgeSemanticEq` | "Are two passages semantically equivalent?" | LLM | DEFER |
| 13 | `ConcurrencyInvariant` | "Does a global invariant hold across N parallel runs?" | Scenario | DEFER |
| 14 | `TraceStructure` | "Does the agent's tool-call trace match a required kind sequence?" | LLM-or-structural | DEFER |
| 15 | `EntityAllocationInvariant` | "Is every allocated entity uniquely-referenced and within bounds?" | Set | DEFER |
| 16 | `TranscriptEquality` | "Under a fixed RNG seed, does the trace byte-equal the reference?" | Structural | DEFER |
| 17 | `LoadTestHarness` | "Does p95 / throughput meet the bar under N concurrent clients?" | Scenario | DEFER |

## Foundation (shared across all rubrics)

Located in `app/services/scenario_rubrics_base.py`.

### `Verdict`

```python
class Verdict(BaseModel):
    status: Literal["pass", "fail", "abstain"]
    rationale: str
    diagnostic: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0
```

### `RubricContext`

```python
class RubricContext(BaseModel):
    captures: dict[str, Any]                  # everything captured during the scenario trace
    setup_data: dict[str, Any] = {}           # loaded from private/grader/_setup/
    course_meta: dict[str, Any] = {}          # spec subset (entities, capabilities, ...)
```

### `Rubric` (abstract base)

```python
class Rubric(ABC):
    name: ClassVar[str]                       # YAML kind value
    @abstractmethod
    def judge(self, ctx: RubricContext) -> Verdict: ...
```

### Path resolver

```python
def resolve_path(captures: Mapping[str, Any], dotted_path: str) -> Any:
    """Walk ``dotted_path`` through nested dicts and lists.
    Examples: 'answer_response', 'answer_response.cited_chunk_ids',
    'retrieval.ranked_chunks[0].doc_id'."""
```

### Registry

```python
RUBRIC_REGISTRY: dict[str, type[Rubric]] = {}

def register_rubric(cls: type[Rubric]) -> type[Rubric]:
    RUBRIC_REGISTRY[cls.name] = cls
    return cls
```

The eight implemented rubrics decorate themselves on import.

## Implementation split across parallel code owners

To complete faster, the eight rubrics are split into four
parallel-implementable units. Each unit owns its own file + tests,
shares only the foundation interface, and uses TDD.

| Agent | Owns | File | Test file |
|---|---|---|---|
| **A** (structural) | `SchemaMatch`, `LiteralMatch`, `RegexMatch`, `NumericRange` | `app/services/scenario_rubrics_structural.py` | `tests/test_scenario_rubrics_structural.py` |
| **B** (set) | `SubsetMatch`, `BehavioralEquivalence` | `app/services/scenario_rubrics_set.py` | `tests/test_scenario_rubrics_set.py` |
| **C** (LLM) | `LLMJudgeCoverage` | `app/services/scenario_rubrics_llm.py` | `tests/test_scenario_rubrics_llm.py` |
| **D** (oracle) | `OracleSetOverlap` | `app/services/scenario_rubrics_oracle.py` | `tests/test_scenario_rubrics_oracle.py` |

Agents A, B, D have no LLM dependency and run pure-deterministically.
Agent C uses `LLMRouter` exactly like `public_surface_quality_llm.py`
already does, with the same `router=None` → return `None` (abstain)
fallback for offline / no-key environments.

## Scope explicitly out of MVP

- Scenario YAML loader and trace runner (separate PR)
- Reference-impl oracle pass and `_oracle/outputs.json` generation
  (separate PR)
- Integration into `langgraph_assignment_graph.py` (separate PR)
- The nine deferred rubric classes
- Composite rubrics (`one_of`, `all_of`) — applied at the runner level
  later, not part of the rubric library

## Acceptance for this PR

- Foundation file + tests pass
- Each of the eight rubrics has at least one passing test for each
  observable behavior (pass case, fail case, edge case)
- Rubrics register themselves into `RUBRIC_REGISTRY` on import
- No regressions in the existing test suite
