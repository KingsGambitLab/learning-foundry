# Anthropic LLM Router — Design

**Date:** 2026-05-13
**Branch:** `virtusa_assignment`
**Status:** approved (user LGTM 2026-05-13)

## Why

The platform currently calls OpenAI's `responses.parse` from six service
files. The user's OpenAI quota is exhausted, blocking the RAG course run
that motivates this branch. We need to route all structured-output LLM
calls to Anthropic (Claude Sonnet 4.6 / Haiku 4.5) without:

- losing the hard-kill subprocess timeout guardrail (README, "Structured-
  output hard-kill on LLM calls"),
- losing the retry / backoff already wrapped around each callsite,
- forcing every callsite to learn a second SDK.

OpenAI must remain a one-env-var fallback so we can flip back if a model
regresses.

## Non-goals

- No third-party router (LiteLLM, OpenRouter, Portkey). Reasons in the
  brainstorming transcript: the abstraction we need is ~80 lines, third-
  party shims introduce their own structured-output bugs, and we only
  have two providers.
- No rename of `OpenAI*AuthoringService` classes. They become orchestrators
  that happen to call the router. Cosmetic rename is a follow-up.
- No model auto-escalation (Haiku → Sonnet on validation failure). The
  per-callsite tier is fixed; existing retry loops already handle bad
  output.
- No streaming. All current callsites are batch structured-output.
- No prompt-cache tuning beyond what's obvious for the system/tool blocks.
  Aggressive caching is a follow-up once we have a baseline.

## Architecture

```
+----------------------------+
|  OpenAITaskAgentAuthoring  |    ... 5 other authoring services
|  (and 4 siblings)          |
+--------------+-------------+
               | router.parse_structured(tier=..., text_format=PydanticModel)
               v
+--------------+-------------+
|         LLMRouter          |     app/services/llm_router.py
|  - provider selection      |     env: COURSE_GEN_LLM_PROVIDER
|  - hard-kill subprocess    |     (reuses openai_runtime_support helper
|    timeout wrapper         |     for the subprocess plumbing)
+----+------------------+----+
     |                  |
     v                  v
+----+----+        +----+-------+
|Anthropic|        |   OpenAI   |
|Provider |        |  Provider  |
+----+----+        +-----+------+
     |                   |
     v                   v
 anthropic         openai SDK (existing)
 SDK (new dep)
```

### Module layout

| File | Status | Responsibility |
|---|---|---|
| `app/services/llm_router.py` | new | `LLMRouter`, `LLMTier`, provider selection, the public `parse_structured` entry point. Owns the hard-kill subprocess wrapper. |
| `app/services/anthropic_runtime_support.py` | new | Env-file loading (mirrors `openai_runtime_support.resolve_openai_env_file`), Anthropic SDK client factory, JSON-schema-from-Pydantic, forced `tool_use` call, response-to-Pydantic decoding, `AIUsageSummary` extractor. |
| `app/services/openai_runtime_support.py` | modified | No behavior change; extract the subprocess helper into a provider-agnostic shape so the router can reuse it. Public surface kept stable. |
| `app/services/openai_course_planner.py` | modified | Replace `client.responses.parse(...)` (and the subprocess fallback) with `router.parse_structured(tier="sonnet", ...)`. |
| `app/services/openai_task_agent_authoring.py` | modified | Same swap, tier=sonnet. |
| `app/services/openai_repo_authoring.py` | modified | Same swap, tier=sonnet. |
| `app/services/openai_test_script_authoring.py` | modified | Same swap, tier=sonnet. |
| `app/services/openai_learner_feedback.py` | modified | Same swap, tier=**haiku**. |
| `app/api/routes.py` | modified | `/v1/task-agent-authoring/status` reports `provider`, both model ids, key presence. |
| `pyproject.toml` | modified | Add `anthropic>=0.95,<1.0` (current series, latest is 0.101 as of 2026-05-13). |
| `tests/test_anthropic_runtime_support.py` | new | Unit-tests the Pydantic→JSON-schema→tool_use mapping, response decoding, validation errors. Mocks the SDK. |
| `tests/test_llm_router.py` | new | Provider selection by env var, tier→model id mapping, hard-kill timeout behavior, validation passthrough. |

## Public API

```python
class LLMTier(StrEnum):
    sonnet = "sonnet"   # smart tier
    haiku = "haiku"     # fast tier


@dataclass
class ParsedResult:
    """Provider-agnostic result of a structured-output call."""
    parsed: BaseModel             # validated Pydantic instance
    usage: AIUsageSummary | None  # tokens + cost when the provider reports it


class LLMRouter:
    def __init__(
        self,
        *,
        provider: Literal["anthropic", "openai"] | None = None,
        anthropic_env_file: str | None = None,
        openai_env_file: str | None = None,
    ) -> None: ...

    def parse_structured(
        self,
        *,
        tier: LLMTier | Literal["sonnet", "haiku"],
        system: str,
        user: str,
        text_format: type[BaseModel],
        request_timeout_s: float = 240.0,
        max_request_retries: int = 2,
        workflow_run_id: str | None = None,
        usage_label: str | None = None,
    ) -> ParsedResult: ...

    def status(self) -> dict[str, Any]:
        """Shape compatible with /v1/task-agent-authoring/status."""
```

## Provider selection

- Constructor `provider=` wins if set explicitly (used in tests).
- Otherwise read `COURSE_GEN_LLM_PROVIDER` (case-insensitive).
- Default = `"anthropic"`.
- Invalid value raises at construction.

## Anthropic provider details

### Env loading

Mirrors `openai_runtime_support`. Reads `COURSE_GEN_ANTHROPIC_ENV_FILE`
(default `/Users/tushar/Desktop/anthropic.env.keys` only as a doc hint;
no hard-coded fallback). Recognized keys inside the file:

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_SONNET=claude-sonnet-4-6        # optional, default below
ANTHROPIC_MODEL_HAIKU=claude-haiku-4-5-20251001 # optional, default below
ANTHROPIC_BASE_URL=...                          # optional
```

Defaults baked into code (aliases, no date suffix per `anthropic-skills:claude-api`):

| Tier | Default model id |
|---|---|
| `sonnet` | `claude-sonnet-4-6` |
| `haiku` | `claude-haiku-4-5` |

These can be rotated by editing the env file, no code change.

### Structured output via `client.messages.parse()`

The Anthropic SDK ships a Pydantic-validated structured-output entry
point: `client.messages.parse(model=..., output_format=PydanticModel,
...)`. Supported on Sonnet 4.6 / Haiku 4.5. The SDK auto-strips JSON-
schema features Anthropic doesn't honor (numerical bounds, string
length, recursive `$ref`) and validates the response client-side, so we
do not need to forge a forced-`tool_use` request or flatten `$defs`.

Call shape:

```python
response = client.messages.parse(
    model=model_id,
    max_tokens=max_tokens,
    system=system_blocks,
    messages=[{"role": "user", "content": user_prompt}],
    output_format=text_format,
    thinking={"type": "disabled"},   # structured extraction, no chain-of-thought
)
parsed: BaseModel = response.parsed_output  # already a validated Pydantic instance
```

If `parsed_output` is `None` (refusal / safety stop / max_tokens early
cut) → raise `LLMStructuredOutputError`, which the callsite's existing
retry loop handles.

### Usage tracking

`response.usage` shape: `{input_tokens, output_tokens,
cache_creation_input_tokens, cache_read_input_tokens}`. Pricing table per
tier (from public Anthropic pricing) maintained in
`anthropic_runtime_support._pricing_for_model`. Returns `AIUsageSummary`
shape-compatible with the existing OpenAI version so usage logging keeps
working unchanged.

### Prompt caching

Initial implementation places `cache_control={"type": "ephemeral"}` on:
- the system prompt block,
- the tool definition (one tool per call).

That covers the bulk of repeated tokens in authoring/repair loops. We
intentionally skip user-message caching in v1 — they vary per attempt.
Revisit after a course run produces a real workload profile.

## OpenAI provider details

Existing `openai_runtime_support.parse_structured_openai_response_with_hard_timeout`
is wrapped in a thin `OpenAIProvider` that adapts:
- `tier="sonnet"` → `OPENAI_MODEL` env value (today `gpt-5.4`),
- `tier="haiku"` → `OPENAI_MODEL_FAST` env value, fall back to `OPENAI_MODEL`.

No model auto-mapping invented. If the OpenAI env file doesn't define
`OPENAI_MODEL_FAST`, both tiers use the same OpenAI model — acceptable
fallback because today's setup already used one model for everything.

## Hard-kill timeout

`LLMRouter.parse_structured` runs each provider call inside the existing
`multiprocessing.Process` wrapper from `openai_runtime_support`. The
wrapper is provider-agnostic — it takes a callable + args and returns
its result or raises after `request_timeout_s`. The router supplies the
provider-specific callable.

Why we keep this: it's the guardrail described in
README§"Structured-output hard-kill on LLM calls" — a hung SDK call must
not pin a workflow thread. Both Anthropic and OpenAI SDKs sit on httpx
and can wedge identically.

## Per-callsite tier assignment

| Service | Tier | Justification |
|---|---|---|
| `OpenAICoursePlanner` | sonnet | Single shot, defines the whole course direction. |
| `OpenAITaskAgentAuthoringService` | sonnet | Deep structured-output, drives every downstream node. |
| `OpenAIStarterRepoAuthoringService` | sonnet | Most failure-sensitive call; prompts enforce toolchain rules, lockfile policy, path conventions. Haiku will miss adversarial cases. |
| `OpenAITestScriptAuthoringService` | sonnet | Test scripts feed the baseline matrix verifier. Weak scripts burn reviewer attempts. |
| `OpenAILearnerFeedbackService` | haiku | Read-and-summarize, no schema authority over downstream. Volume-sensitive at LMS scale. |

## Status endpoint shape

`GET /v1/task-agent-authoring/status` keeps the same top-level keys and
adds two:

```json
{
  "provider": "anthropic",
  "available": true,
  "source": "anthropic_live",
  "message": "...",
  "sdk_installed": true,
  "api_key_present": true,
  "model_id_sonnet": "claude-sonnet-4-6",
  "model_id_haiku": "claude-haiku-4-5-20251001",
  "env_file": "/Users/tushar/Desktop/anthropic.env.keys",
  "fallback_provider_available": true
}
```

The pre-existing `model_id` key is preserved (set to the sonnet id) so
older dashboard code keeps rendering.

## Testing strategy (TDD)

For every new module, tests precede code.

1. **`tests/test_anthropic_runtime_support.py`**
   - Env file loading happy / missing-file / missing-key paths.
   - Pydantic→tool_use payload shape (no `$ref` survive in input_schema,
     tool name = model class name, tool_choice forced).
   - Decoding of a stub SDK response with a `tool_use` block → returns
     a validated Pydantic instance.
   - Decoding when no tool_use block present → raises
     `LLMStructuredOutputError`.
   - Usage extraction maps Anthropic's
     `(input_tokens, output_tokens, cache_*)` into `AIUsageSummary`.

2. **`tests/test_llm_router.py`**
   - Provider selection by env var (anthropic / openai / invalid raises).
   - `tier="sonnet"` → sonnet model id; `tier="haiku"` → haiku model id.
   - Hard-kill timeout triggers and raises a deterministic error type.
   - End-to-end with mocked Anthropic SDK: prompt → parsed Pydantic.

3. Existing callsite tests
   (`tests/test_authoring_payloads.py`,
   `tests/test_authoring_resilience.py`,
   `tests/test_course_generation_async.py`,
   `tests/test_generated_test_loop.py`)
   must keep passing without modification beyond fixture/mocks adjusted
   to the router seam. If any of them imported `openai_runtime_support`
   directly, switch to mocking the router instead.

4. No real API key required for any test. The Anthropic SDK is mocked.

## Rollout plan

1. Land the router + provider + tests on `virtusa_assignment`.
2. Flip the running server (port 8020) to the patched code.
3. Re-submit the RAG course brief (`/tmp/rag_course.json`) and confirm
   the workflow reaches gate 1 using Anthropic.
4. Verify the cache-key + normalization fixes from the prior commit
   (`c72aa4e4`) cooperate: each LangGraph node reuses the runtime image
   after the first build.
5. If Anthropic falters on a specific schema, flip
   `COURSE_GEN_LLM_PROVIDER=openai` (once the user's OpenAI quota refills)
   and re-run without code changes.

## Risks

| Risk | Mitigation |
|---|---|
| Anthropic refuses certain JSON-schema features the Pydantic models use (e.g. recursive `$ref`, `oneOf`). | The runtime support module flattens / inlines refs and rejects features Anthropic doesn't support, raising a clear error at startup-mock time (covered by tests). If a real prompt hits the wall, the OpenAI fallback path remains one env var away. |
| Forced tool_use causes Anthropic to occasionally over-explain in text blocks before the tool call. | We only read the tool_use block. Text blocks are ignored. |
| Token usage / cost differs materially between providers. | `AIUsageSummary` keeps tracking it. The status endpoint exposes which provider produced each call. Out-of-scope for this PR to alarm on cost regressions. |
| Anthropic SDK version drift. | Pin `anthropic>=0.40,<1.0` (current 0.x series). When 1.x lands, follow the migration via `anthropic-skills:claude-api`. |
