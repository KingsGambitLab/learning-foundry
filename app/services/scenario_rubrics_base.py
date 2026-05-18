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

# Capture entries have a fixed shape — ``{status, headers, body, raw, request}``.
# A target path's second segment is one of these only when the scenario
# author explicitly wants to inspect that part of the capture; otherwise
# the segment is a body field and the path needs a ``body.`` injection.
_CAPTURE_PARTS = frozenset({"status", "headers", "body", "raw", "request"})

# Top-level path prefixes that route OUT of captures (into setup_data /
# course_meta merged-context resolution). These never get the body
# shorthand applied because they aren't capture targets at all.
_NON_CAPTURE_PREFIXES = frozenset({"setup_data", "course_meta", "captures"})


def expand_capture_shorthand(target: str) -> str:
    """Inject ``body`` after the capture id when the LLM-authored target
    uses the body-shorthand convention.

    The trace-runner's placeholder ``${X.Y}`` uses the same shorthand:
    a path like ``eval.summary`` against captures resolves through
    ``eval.body.summary``. Rubrics call :func:`resolve_path` directly
    which is shorthand-blind, so without this expansion every rubric
    using the shorthand fails with "X not found in captures".

    Rules:
    - Empty string: pass through unchanged.
    - Path starts with ``setup_data.`` / ``course_meta.`` / ``captures.``:
      pass through (not a capture target).
    - Single segment (``eval``): inject ``body`` so a top-level
      schema_match against the capture id checks fields on the body
      dict, not the ``{status, headers, body, ...}`` entry.
    - Multi-segment with second segment in ``{status, headers, body,
      raw, request}``: pass through (author was explicit).
    - Multi-segment with body-field second segment: inject ``body``
      after the capture id.
    """
    if not target:
        return target
    segments = target.split(".")
    leading = segments[0]
    if leading in _NON_CAPTURE_PREFIXES:
        return target
    if len(segments) == 1:
        return f"{leading}.body"
    second = segments[1]
    # Strip a trailing list-index suffix (``case_results[0]``) before
    # the membership check — ``case_results[0]`` is still a body field.
    second_name = second.split("[", 1)[0]
    if second_name in _CAPTURE_PARTS:
        return target
    return ".".join([leading, "body"] + segments[1:])


def resolve_capture_target(captures: Mapping[str, Any], target: str) -> Any:
    """Resolve a rubric target against captures, honoring the
    body-shorthand convention.

    Strategy: try the path AS-IS first (preserves canonical
    ``resp.body.X`` paths and legacy test fixtures with custom capture
    shapes), then fall back to the body-shorthand-expanded form if the
    as-is walk fails. The fallback is the LLM-author convention —
    targets like ``eval.summary`` get expanded to
    ``eval.body.summary`` when the literal walk can't find them.

    On total failure, raises the original ``KeyError`` / ``IndexError``
    from the as-is attempt so the rubric's diagnostic still names the
    path the author wrote, not the expanded form.
    """
    try:
        value = resolve_path(captures, target)
    except (KeyError, IndexError, TypeError) as exc:
        expanded = expand_capture_shorthand(target)
        if expanded == target:
            raise
        try:
            return resolve_path(captures, expanded)
        except (KeyError, IndexError, TypeError):
            # Re-raise the ORIGINAL exception so the diagnostic shows
            # the author's path, not the internal expansion.
            raise exc

    # Single-segment target that lands on a canonical capture-entry-shaped
    # dict (has both ``body`` and at least one of ``status`` / ``headers``):
    # walk into ``body`` so rubrics inspect the response payload, not the
    # entry envelope. Legacy test fixtures using flat capture dicts (no
    # ``body`` key) are unaffected.
    if (
        "." not in target
        and isinstance(value, dict)
        and "body" in value
        and ("status" in value or "headers" in value)
    ):
        return value["body"]
    return value


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
