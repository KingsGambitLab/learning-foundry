"""Capability-gated learner-facing README templater (Wave 6.7a).

The single-outcome pipeline used to dump whatever the LLM authored as
the README. That was opaque: the learner couldn't tell which runtime
primitives were available (e.g. the sandbox LLM proxy), the endpoint
shape didn't always survive the LLM's free-form prose, and domain
content was inconsistently surfaced.

This module renders ``public/README.md`` deterministically from the
:class:`CourseOutcomeSpec`:

- Title + goal come from the spec.
- The endpoint table is generated from ``spec.endpoints`` so the
  contract surface is always navigable.
- Quality bars are listed verbatim (id + threshold + description) so
  the learner knows exactly which bars gate "done".
- Capability-gated sections only appear when the spec opts in via
  :class:`CapabilityFlags` — a course that doesn't need the sandbox
  LLM proxy never sees proxy docs.
- The materializer appends sibling-scaffold-generated blocks (e.g. the
  RAG scaffold's HTML-parsing primer) into a dedicated section so
  family-specific content lives outside the framework.

Section order:

  1. Title + goal
  2. Endpoint contract
  3. Quality bars
  4. Capability-gated sections:
       - LLM proxy access (``runtime_llm_required``)
       - Persistence (``durable_state_required``)
       - Database sidecar (``sidecar_database != "none"``)
       - Structured logging (``structured_logging_required``)
  5. Local development quickstart
  6. Visible self-test instructions (always documented; the materializer
     decides whether the script ships under ``public/checks/``)
  7. Scaffold blocks (caller-supplied)
  8. Learning path (when ``spec.learning_path`` is non-empty)

The function is a pure string builder: no I/O, no globals. Tests cover
each section in isolation.
"""
from __future__ import annotations

import os
from typing import Iterable

from app.services.course_outcome_models import (
    CapabilityFlags,
    CourseOutcomeSpec,
    EndpointContract,
    LearningHint,
    QualityBar,
)


__all__ = [
    "render_outcome_readme",
    "DEFAULT_SANDBOX_LLM_PROXY_URL",
    "DEFAULT_MAX_TOKENS_PER_CALL",
    "DEFAULT_MAX_TOKENS_PER_SUBMISSION",
    "ENV_LLM_PROXY_URL",
]


# ----- defaults -----
#
# These mirror the hard-coded defaults in
# ``app/services/sandbox_llm_proxy.py``. Keeping them as named module
# constants means the README documents the real values, and a course
# author who needs to override the proxy hostname in the README (e.g.
# for a non-standard sidecar layout) can set ``COURSEGEN_LLM_PROXY_URL``
# in the environment before the materializer runs.
ENV_LLM_PROXY_URL = "COURSEGEN_LLM_PROXY_URL"
DEFAULT_SANDBOX_LLM_PROXY_URL = "http://coursegen-llm:8080/v1/messages"
DEFAULT_MAX_TOKENS_PER_CALL = 2000
DEFAULT_MAX_TOKENS_PER_SUBMISSION = 100_000


# ----- public API -----


def render_outcome_readme(
    spec: CourseOutcomeSpec,
    *,
    scaffold_blocks: Iterable[str] | None = None,
) -> str:
    """Render the learner-facing README for a single-outcome course.

    Args:
        spec: The course spec. The templater handles partial / malformed
            inputs gracefully: an empty ``learning_path`` simply skips
            section 8 (rather than emitting an empty bullet list), and
            capabilities default to all-off so a planner that omitted
            the field still produces a coherent README.
        scaffold_blocks: Pre-rendered markdown blocks to append between
            the visible-self-test instructions and the learning path.
            Each block is treated as a standalone subsection (it owns
            its own ``##`` header). Order is preserved.

    Returns:
        The full README markdown as a single string, ready to write to
        ``public/README.md``.
    """
    blocks = list(scaffold_blocks or [])
    parts: list[str] = []

    parts.append(_render_title_and_goal(spec))
    parts.append(_render_endpoint_section(spec.endpoints))
    parts.append(_render_quality_bars_section(spec.quality_bars))

    caps = spec.capabilities or CapabilityFlags()
    if caps.runtime_llm_required:
        parts.append(_render_llm_proxy_section())
    if caps.durable_state_required:
        parts.append(_render_persistence_section())
    if caps.sidecar_database and caps.sidecar_database != "none":
        parts.append(_render_database_section(caps.sidecar_database))
    if caps.structured_logging_required:
        parts.append(_render_structured_logging_section())

    parts.append(_render_local_dev_section())
    parts.append(_render_visible_self_test_section())

    for block in blocks:
        if block and block.strip():
            parts.append(block.strip() + "\n")

    if spec.learning_path:
        parts.append(_render_learning_path_section(spec.learning_path))

    return "\n".join(parts).rstrip() + "\n"


# ----- section renderers -----


def _render_title_and_goal(spec: CourseOutcomeSpec) -> str:
    title = (spec.title or "Course outcome").strip()
    goal = (spec.goal or "").strip()
    if goal:
        return f"# {title}\n\n{goal}\n"
    return f"# {title}\n"


def _render_endpoint_section(endpoints: list[EndpointContract]) -> str:
    if not endpoints:
        return "## Endpoint contract\n\n_No endpoints declared._\n"
    lines = ["## Endpoint contract", ""]
    for ep in endpoints:
        method = ep.method.value if hasattr(ep.method, "value") else str(ep.method)
        lines.append(f"### `{method} {ep.path}`")
        lines.append("")
        if ep.description:
            lines.append(ep.description.strip())
            lines.append("")
        if ep.request_schema is not None:
            lines.append("Request shape:")
            lines.append("")
            lines.append("```json")
            lines.append(_format_schema(ep.request_schema))
            lines.append("```")
            lines.append("")
        lines.append("Response shape:")
        lines.append("")
        lines.append("```json")
        lines.append(_format_schema(ep.response_schema))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_quality_bars_section(bars: list[QualityBar]) -> str:
    if not bars:
        return "## Quality bars\n\n_No quality bars declared._\n"
    lines = ["## Quality bars", ""]
    lines.append(
        "Your service is graded against the following bars. Each bar's "
        "threshold is evaluated by the rubric named in `judged_by`."
    )
    lines.append("")
    for bar in bars:
        judge = (
            bar.judged_by.value
            if hasattr(bar.judged_by, "value")
            else str(bar.judged_by)
        )
        aggregation = (
            bar.aggregation.value
            if hasattr(bar.aggregation, "value")
            else str(bar.aggregation)
        )
        lines.append(f"- **`{bar.id}`** — threshold `{bar.threshold}`")
        lines.append(f"  - Metric: {bar.metric_description}")
        lines.append(
            f"  - Judged by: `{judge}` · sample size: `{bar.sample_size}` · "
            f"aggregation: `{aggregation}`"
        )
    lines.append("")
    return "\n".join(lines)


def _render_llm_proxy_section() -> str:
    proxy_url = os.environ.get(ENV_LLM_PROXY_URL, DEFAULT_SANDBOX_LLM_PROXY_URL)
    lines = [
        "## LLM access inside the sandbox",
        "",
        "Your service can call a managed LLM endpoint that the sandbox "
        "harness operates on the internal Docker network. No API key is "
        "required — the sandbox network boundary is the auth.",
        "",
        f"- Endpoint URL: `{proxy_url}`",
        "- Method: `POST`",
        "- Default allowed tier: `haiku` (Sonnet is gated; check your "
        "submission limits)",
        "",
        "Request shape:",
        "",
        "```json",
        "{",
        '  "tier": "haiku",',
        '  "system": "<system prompt>",',
        '  "messages": [{"role": "user", "content": "..."}],',
        '  "max_tokens": 1024',
        "}",
        "```",
        "",
        "Response shape:",
        "",
        "```json",
        "{",
        '  "content": "<assistant text>",',
        '  "usage": {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...},',
        '  "model_id": "<resolved model id>",',
        '  "cost_usd": 0.0',
        "}",
        "```",
        "",
        "Limits (set by the harness on the proxy container):",
        "",
        f"- Per-call cap on `max_tokens`: `{DEFAULT_MAX_TOKENS_PER_CALL}`",
        f"- Per-submission cumulative token budget: `{DEFAULT_MAX_TOKENS_PER_SUBMISSION}`",
        "- A handful of calls per submission is fine; tight retry loops "
        "will exhaust the budget.",
        "",
    ]
    return "\n".join(lines)


def _render_persistence_section() -> str:
    return (
        "## Persistence\n\n"
        "Your service must persist state across restarts. The sandbox "
        "mounts a writable volume at `/data` for this — every file you "
        "write under `/data` will be available to a follow-up health "
        "check or restart. State written elsewhere will be lost when "
        "the container is recycled.\n"
    )


def _render_database_section(sidecar: str) -> str:
    if sidecar == "postgres":
        return (
            "## Database sidecar\n\n"
            "The sandbox network exposes a Postgres sidecar at "
            "`postgresql://coursegen:coursegen@coursegen-db:5432/coursegen`. "
            "Schema creation is your responsibility — the sidecar boots "
            "with an empty database. Use the connection string above "
            "from your service container.\n"
        )
    if sidecar == "redis":
        return (
            "## Cache sidecar\n\n"
            "The sandbox network exposes a Redis sidecar at "
            "`redis://coursegen-cache:6379/0`. The cache starts empty "
            "on every grading run.\n"
        )
    return ""


def _render_structured_logging_section() -> str:
    return (
        "## Structured logging\n\n"
        "Emit structured JSON log records to stdout. The grader captures "
        "stdout per request and inspects fields like `latency_ms`, "
        "`request_id`, and `outcome` against the bars in the rubric. A "
        "minimum record looks like:\n\n"
        "```json\n"
        "{\"event\": \"answer\", \"latency_ms\": 142, \"request_id\": \"...\"}\n"
        "```\n"
    )


def _render_local_dev_section() -> str:
    return (
        "## Local development quickstart\n\n"
        "1. Build the starter image:\n"
        "   ```\n"
        "   docker build -t my-service public/starter\n"
        "   ```\n"
        "2. Run the service:\n"
        "   ```\n"
        "   docker run --rm -p 8080:8080 my-service\n"
        "   ```\n"
        "3. Hit the endpoints documented above and confirm the response "
        "shapes match the contract.\n"
    )


def _render_visible_self_test_section() -> str:
    return (
        "## Visible self-test\n\n"
        "When the course ships a visible-check runner, you'll find it at "
        "`public/checks/run_visible_checks.py`. It calls each endpoint "
        "against the visible sample queries under `public/examples/` and "
        "reports pass / fail per case. Run it before submission to catch "
        "contract mismatches early:\n\n"
        "```\n"
        "python public/checks/run_visible_checks.py\n"
        "```\n"
    )


def _render_learning_path_section(hints: list[LearningHint]) -> str:
    lines = ["## Learning path", ""]
    lines.append(
        "When a specific quality bar fails, the following hints are "
        "surfaced. Each hint targets the failing bar by id."
    )
    lines.append("")
    for hint in hints:
        lines.append(f"- **`{hint.on_metric_fail}`** — {hint.hint}")
    lines.append("")
    return "\n".join(lines)


# ----- formatting helpers -----


def _format_schema(schema: dict[str, object] | None) -> str:
    """Render a schema map as compact JSON-ish so the README block is
    readable. We don't ship full JSON Schema here — the spec keeps shape
    maps as ``dict[str, Any]`` (e.g. ``{"answer": "str"}``). Use a
    deterministic key order so successive materializations of the same
    spec produce byte-identical README content.
    """
    import json

    if not schema:
        return "{}"
    try:
        return json.dumps(schema, indent=2, sort_keys=True)
    except TypeError:
        # Fall back to ``repr`` for non-JSON-serializable values; the
        # README still surfaces the shape, just less prettily.
        return repr(schema)
