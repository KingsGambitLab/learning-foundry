"""LLM judge for domain-grounding in course / starter READMEs.

The substring-based ``content_lacks_domain_grounding`` rule in
``public_surface_quality`` rejects content that uses synonyms or
morphological variants of the canonical entity strings (e.g. it accepts
``"retrieval corpus"`` but rejects ``"retrieval-corpuses"``). On real
courses this fires false positives that the repair loop cannot resolve
because the finding carries no actionable hint.

This module adds a small LLM judge (Haiku tier) that reads the actual
content and decides whether it adequately describes the project using
the spec's domain entities, returning suggested revisions when it
doesn't. Callers should treat ``None`` as "judge unavailable, fall back
to the substring rule" so offline / no-API-key environments keep
working.
"""
from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field


_JUDGE_SYSTEM_PROMPT = (
    "You judge whether a learner-facing course README is grounded in the "
    "concrete domain entities of the project it describes — not in generic "
    "backend / service / API wording. You accept any reasonable synonym or "
    "morphological variant of each entity (singular / plural / hyphenated / "
    "compounded with adjectives). A README is grounded if a reader who knows "
    "the domain would recognize, from the text alone, what the project is "
    "actually about.\n\n"
    "When you find the README is NOT grounded, produce 1-4 short, "
    "copy-pasteable revision suggestions that name the exact phrases the "
    "author should add or rewrite. Suggestions must be specific to this "
    "README and these entities — not generic advice."
)


class DomainGroundingVerdict(BaseModel):
    """Judge verdict on whether a piece of content describes a project
    using its declared domain entities (or recognizable variants)
    rather than only generic backend wording."""

    is_grounded: bool = Field(
        ...,
        description=(
            "True when the content makes the domain recognizable to a "
            "reader who knows the entities. Accept synonyms / variants."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "One sentence explaining the verdict. Reference specific "
            "phrases from the content."
        ),
    )
    suggested_revisions: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, copy-pasteable phrases to add or rewrite. Empty "
            "when is_grounded=True. 1-4 items when is_grounded=False."
        ),
    )


def evaluate_domain_grounding(
    *,
    content: str,
    entities: Iterable[str],
    system_kind: str | None,
    router=None,
) -> DomainGroundingVerdict | None:
    """Ask an LLM judge whether ``content`` describes the project using
    the declared ``entities`` (or recognizable variants).

    Returns the verdict on success.

    Returns ``None`` when the judge is unavailable so the caller can
    fall back to the deterministic substring rule:
    - ``router`` is None (no provider configured, test mode), or
    - the LLM call raises (network blip, quota, schema rejected by
      provider — fail open rather than block on judge availability),
    - there are no entities to judge against.
    """
    entity_list = [e.strip() for e in entities if e and e.strip()]
    if not entity_list:
        return None
    if router is None:
        return None

    user_prompt = (
        f"System kind: {system_kind or 'unspecified'}\n"
        f"Declared core entities: {entity_list!r}\n\n"
        "README content to judge:\n"
        "<<<README>>>\n"
        f"{content}\n"
        "<<<END>>>\n\n"
        "Return your verdict as a DomainGroundingVerdict."
    )
    try:
        # Avoid importing LLMTier at module load — keeps this file
        # importable from places that don't have the router wired.
        from app.services.llm_router import LLMTier

        result = router.parse_structured(
            tier=LLMTier.haiku,
            system=_JUDGE_SYSTEM_PROMPT,
            user=user_prompt,
            text_format=DomainGroundingVerdict,
            request_timeout_s=60.0,
            max_tokens=1024,
        )
    except Exception:
        return None
    parsed = getattr(result, "parsed", None) or getattr(result, "output_parsed", None)
    if not isinstance(parsed, DomainGroundingVerdict):
        return None
    return parsed
