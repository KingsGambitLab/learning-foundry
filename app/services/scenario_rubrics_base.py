"""Foundation types for the scenario-rubric library.

Every rubric (structural, set, LLM-judge, oracle) implements the
``Rubric`` ABC defined here. Rubrics are pure functions of a
``RubricContext`` and return a ``Verdict``. Concrete rubrics live in
sibling modules (``scenario_rubrics_structural``,
``scenario_rubrics_set``, ``scenario_rubrics_llm``,
``scenario_rubrics_oracle``) and self-register into ``RUBRIC_REGISTRY``
via the ``@register_rubric`` decorator at import time.

See ``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``
for the full design and roster of rubrics.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Mapping

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """Outcome of a rubric's evaluation.

    ``status``: tri-valued so rubrics can decline when their inputs are
    not available (e.g., an LLM judge with no router) instead of
    forcing a binary decision.

    ``rationale``: one short sentence the learner / repair LLM reads.

    ``diagnostic``: structured details for the feedback synthesizer
    (which fields were missing, what the threshold was, etc.).

    ``cost_usd``: for LLM-judged rubrics; surfaces in the per-grade
    cost report. Free rubrics leave it at 0.0.
    """

    status: Literal["pass", "fail", "abstain"]
    rationale: str
    diagnostic: dict[str, Any] = Field(default_factory=dict)
    cost_usd: float = 0.0


class RubricContext(BaseModel):
    """All inputs available to a rubric at judge time.

    ``captures``: everything the scenario trace runner captured from
    the learner's service (request bodies, response bodies, headers,
    status codes, timings).

    ``setup_data``: course-level assets the grader loads from
    ``private/grader/_setup/`` — gold-label files, hidden corpora,
    seed lists, anything the rubrics may need to compare against.

    ``course_meta``: a subset of the ``CourseOutcomeSpec`` (declared
    entities, capability flags, endpoint contracts) that some rubrics
    consult.

    All three are dicts so each rubric reads only what it needs and
    the framework doesn't have to know rubric-specific shapes.
    """

    captures: dict[str, Any]
    setup_data: dict[str, Any] = Field(default_factory=dict)
    course_meta: dict[str, Any] = Field(default_factory=dict)


class Rubric(ABC):
    """Abstract base for every rubric class.

    Subclasses set ``name`` (the YAML ``kind`` value) as a class
    attribute and implement ``judge``. They typically take their
    per-scenario config (target path, threshold, expected value, ...)
    via ``__init__``, and consult ``ctx.captures`` / ``ctx.setup_data``
    inside ``judge``.

    Subclasses register themselves into ``RUBRIC_REGISTRY`` by
    decorating with ``@register_rubric``.
    """

    name: ClassVar[str]

    @abstractmethod
    def judge(self, ctx: RubricContext) -> Verdict:
        """Evaluate the rubric and return a verdict.

        Must not raise on normal pass/fail conditions — encode the
        outcome in the returned ``Verdict``. Reserve exceptions for
        bugs (misconfigured rubric, malformed inputs).
        """


RUBRIC_REGISTRY: dict[str, type[Rubric]] = {}


def register_rubric(cls: type[Rubric]) -> type[Rubric]:
    """Register a rubric class under its ``name`` for YAML lookup.

    Raises ``ValueError`` on duplicate names so two rubrics can't
    claim the same ``kind`` token.
    """
    rubric_name = cls.name
    if rubric_name in RUBRIC_REGISTRY:
        raise ValueError(
            f"Rubric name '{rubric_name}' already registered by "
            f"{RUBRIC_REGISTRY[rubric_name].__qualname__}; "
            f"second registration from {cls.__qualname__} rejected."
        )
    RUBRIC_REGISTRY[rubric_name] = cls
    return cls


# ---------------- Path resolution ----------------

_INDEX_PATTERN = re.compile(r"^(?P<name>[^\[\]]*)\[(?P<index>\d+)\](?P<rest>.*)$")


def resolve_path(captures: Mapping[str, Any], dotted_path: str) -> Any:
    """Walk ``dotted_path`` through nested dicts and lists in ``captures``.

    Supported syntax:
    - Top-level key: ``"resp"``
    - Nested key: ``"resp.body.answer"``
    - List index anywhere: ``"chunks[0].doc_id"``, ``"results[2]"``
    - Empty string returns the root mapping unchanged.

    Raises:
      ``KeyError`` when a dict key is missing.
      ``IndexError`` when a list index is out of range.
    """
    if dotted_path == "":
        return captures

    current: Any = captures
    parts = dotted_path.split(".")
    for part in parts:
        # Drain bracket indexing (may appear after a name OR alone, e.g. "[2]").
        while True:
            match = _INDEX_PATTERN.match(part)
            if match is None:
                break
            name = match.group("name")
            index = int(match.group("index"))
            part = match.group("rest").lstrip(".")
            if name:
                # "chunks[0]" — first descend into "chunks", then index.
                current = current[name]
            current = current[index]
            if not part:
                break
        if not part:
            continue
        # Remaining ``part`` is a plain dict key.
        current = current[part]
    return current
