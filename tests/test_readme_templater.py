"""Tests for the capability-gated learner-facing README templater.

The templater renders ``public/README.md`` for a single-outcome course
from a :class:`CourseOutcomeSpec` plus an optional list of scaffold-
generated markdown blocks. Sections are composed in a deterministic
order so a reviewer (or a learner) sees the same shape across every
course; capability-gated sections only appear when the spec opts in.

These tests do NOT exercise the materializer integration — they check
``render_outcome_readme`` in isolation. Materializer-side tests live in
``tests/test_outcome_artifact_materializer.py``.
"""
from __future__ import annotations

from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CapabilityFlags,
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    LearningHint,
    QualityBar,
    StarterType,
)
from app.services.readme_templater import render_outcome_readme


# ---------------- helpers ----------------


def _spec(
    *,
    capabilities: CapabilityFlags | None = None,
    learning_path: list[LearningHint] | None = None,
    quality_bars: list[QualityBar] | None = None,
    endpoints: list[EndpointContract] | None = None,
) -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title="Grounded retrieval QA service",
        goal=(
            "Build a small retrieval-augmented QA service that returns "
            "grounded answers with citations."
        ),
        starter_type=StarterType.partial,
        endpoints=endpoints
        or [
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str", "citations": "list[str]"},
                description="Return a grounded answer with citations.",
            ),
        ],
        quality_bars=quality_bars
        or [
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite supporting passages.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
        ],
        learning_path=learning_path or [],
        package_type=PackageType.progressive_codebase_course,
        capabilities=capabilities or CapabilityFlags(),
    )


# ---------------- rendering ----------------


def test_render_outcome_readme_includes_title_goal_and_endpoint_table() -> None:
    """Section 1 (title + goal) and section 2 (endpoint contract) come
    straight from the spec. A minimal spec without any capabilities or
    learning path still produces a valid, navigable README."""
    spec = _spec()
    md = render_outcome_readme(spec)
    assert "# Grounded retrieval QA service" in md
    # Goal paragraph appears verbatim.
    assert "grounded answers with citations" in md
    # Endpoint contract section names each endpoint by method and path.
    assert "POST" in md
    assert "/answer" in md
    # Endpoint description is surfaced so a learner knows the contract
    # shape without reading the spec JSON.
    assert "Return a grounded answer" in md


def test_render_outcome_readme_includes_quality_bars_section() -> None:
    """Quality bars surface so the learner knows the bar IDs and the
    thresholds they need to clear. Threshold expressions are rendered
    verbatim so the section is self-describing."""
    spec = _spec(
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite supporting passages.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
            QualityBar(
                id="recall_at_5",
                metric_description="Recall@5 over the labeled retrieval oracle.",
                threshold=">= 0.7",
                judged_by=JudgeKind.oracle_set_overlap,
                sample_size=20,
            ),
        ],
    )
    md = render_outcome_readme(spec)
    assert "faithfulness" in md
    assert "recall_at_5" in md
    assert ">= 0.8" in md
    assert ">= 0.7" in md
    # Metric descriptions land in the section too so a learner sees what
    # is being measured.
    assert "supporting passages" in md
    assert "labeled retrieval oracle" in md


def test_render_outcome_readme_omits_llm_proxy_section_when_capability_off() -> None:
    """No mention of the sandbox LLM proxy when the spec has not opted
    in. Otherwise every course README would dump irrelevant proxy docs
    on a learner whose course has no LLM use."""
    spec = _spec(capabilities=CapabilityFlags(runtime_llm_required=False))
    md = render_outcome_readme(spec)
    assert "coursegen-llm" not in md
    assert "LLM access inside the sandbox" not in md
    assert "/v1/messages" not in md


def test_render_outcome_readme_includes_llm_proxy_section_when_capability_on() -> None:
    """When the spec declares ``runtime_llm_required=True`` (RAG answer
    synthesis, classifier justification, ...), the README surfaces the
    proxy URL, the request shape, and the per-call / per-submission
    caps so a learner can wire their service to the proxy without
    hunting the docs."""
    spec = _spec(capabilities=CapabilityFlags(runtime_llm_required=True))
    md = render_outcome_readme(spec)
    # Section header.
    assert "LLM access inside the sandbox" in md
    # Endpoint URL of the in-network proxy.
    assert "coursegen-llm" in md
    assert "/v1/messages" in md
    # Request shape — tier and messages field must be documented.
    assert "tier" in md
    assert "messages" in md
    # Caps the learner needs to design around.
    assert "max_tokens" in md
    # Per-submission cumulative cap is documented (default 100000).
    assert "100000" in md or "100,000" in md


def test_render_outcome_readme_includes_scaffold_blocks_when_passed() -> None:
    """Section 7 takes pre-rendered markdown blocks the materializer
    gathers from sibling family scaffolds (e.g. the RAG scaffold's
    HTML-parsing primer in Wave 6.7b). The templater appends them
    verbatim so the scaffold owner controls section content."""
    spec = _spec()
    block_a = "## HTML parsing primer\n\nUse `selectolax` for fast DOM walks."
    block_b = "## Embedding cheatsheet\n\nFAISS is wired in `db/index.py`."
    md = render_outcome_readme(spec, scaffold_blocks=[block_a, block_b])
    assert "HTML parsing primer" in md
    assert "selectolax" in md
    assert "Embedding cheatsheet" in md
    assert "FAISS is wired" in md


def test_render_outcome_readme_includes_learning_path_when_non_empty() -> None:
    """The learning path (advisory hints keyed by quality_bar.id) renders
    only when the spec has at least one hint. Sequence-of-empty bullet
    sections is a smell; omit the section entirely when empty."""
    spec_no_path = _spec(learning_path=[])
    md_no_path = render_outcome_readme(spec_no_path)
    assert "Learning path" not in md_no_path

    spec_with_path = _spec(
        learning_path=[
            LearningHint(
                on_metric_fail="faithfulness",
                hint="Cite the chunk id you copied each sentence from.",
            ),
        ],
    )
    md_with_path = render_outcome_readme(spec_with_path)
    assert "Learning path" in md_with_path
    assert "faithfulness" in md_with_path
    assert "Cite the chunk id" in md_with_path


def test_render_outcome_readme_includes_local_dev_quickstart() -> None:
    """Every course README ships a local-dev quickstart so a learner can
    boot the starter without hunting the spec. The quickstart points
    at the starter Dockerfile path and mentions the visible-self-test
    runner under ``public/checks/``."""
    spec = _spec()
    md = render_outcome_readme(spec)
    assert "Local development" in md or "Quickstart" in md
    # Dockerfile path under public/starter
    assert "public/starter" in md
