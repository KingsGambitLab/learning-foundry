"""Structural rubrics: shape and primitive-value checks on captures.

This module hosts the four rubrics that inspect ``ctx.captures`` shape
without consulting ``setup_data``, calling out to an LLM, or running
oracle code. They share a single error path: when ``resolve_path``
cannot reach the configured ``target``, the rubric returns a FAIL
verdict carrying the missing path in its diagnostic, never raising.

See ``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``
for how these compose with set, LLM-judge, and oracle rubrics.
"""
from __future__ import annotations

import re
from typing import Any

from app.services.scenario_rubrics_base import (
    Rubric,
    RubricContext,
    Verdict,
    register_rubric,
    resolve_capture_target,
    resolve_path,
)


def _missing_path_verdict(path: str, exc: Exception) -> Verdict:
    """Shared FAIL verdict for an unresolvable target path."""
    return Verdict(
        status="fail",
        rationale=f"{path} not found in captures",
        diagnostic={"missing_path": path, "error": str(exc)},
    )


@register_rubric
class SchemaMatch(Rubric):
    """Verify a dict-shaped capture has every required key, optionally
    with the right Python type per field.

    Designed for response-body shape checks: e.g., ``resp.body`` must
    carry an ``answer`` (str) and a ``citations`` (list).
    """

    name = "schema_match"

    def __init__(
        self,
        target: str,
        must_have_fields: list[str],
        field_types: dict[str, type] | None = None,
    ) -> None:
        self.target = target
        self.must_have_fields = list(must_have_fields)
        self.field_types = dict(field_types) if field_types else {}

    def judge(self, ctx: RubricContext) -> Verdict:
        try:
            value = resolve_capture_target(ctx.captures, self.target)
        except (KeyError, IndexError) as exc:
            return _missing_path_verdict(self.target, exc)

        if not isinstance(value, dict):
            return Verdict(
                status="fail",
                rationale="target is not a dict-shaped object",
                diagnostic={"target": self.target, "got_type": type(value).__name__},
            )

        missing_fields = [f for f in self.must_have_fields if f not in value]
        type_mismatches: dict[str, str] = {}
        for field_name, expected_type in self.field_types.items():
            if field_name not in value:
                # already accounted for in missing_fields if required
                continue
            actual = value[field_name]
            if not isinstance(actual, expected_type):
                type_mismatches[field_name] = (
                    f"got_{type(actual).__name__}, "
                    f"expected_{expected_type.__name__}"
                )

        if missing_fields or type_mismatches:
            return Verdict(
                status="fail",
                rationale="target dict failed schema check",
                diagnostic={
                    "missing_fields": missing_fields,
                    "type_mismatches": type_mismatches,
                },
            )
        return Verdict(status="pass", rationale="schema satisfied")


@register_rubric
class LiteralMatch(Rubric):
    """Verify a resolved value equals a literal expected value.

    Use for fixed primitives like status codes, mode flags, or enum
    strings. Comparison is plain ``==`` so the same rubric works for
    ints, strings, bools, and even small dicts / lists when the
    scenario author wants an exact-shape check.
    """

    name = "literal_match"

    def __init__(self, target: str, expected: Any) -> None:
        self.target = target
        self.expected = expected

    def judge(self, ctx: RubricContext) -> Verdict:
        try:
            value = resolve_capture_target(ctx.captures, self.target)
        except (KeyError, IndexError) as exc:
            return _missing_path_verdict(self.target, exc)

        if value == self.expected:
            return Verdict(
                status="pass",
                rationale=f"{self.target} matches expected literal",
            )
        return Verdict(
            status="fail",
            rationale=f"{self.target} does not equal expected literal",
            diagnostic={"got": value, "expected": self.expected},
        )


@register_rubric
class RegexMatch(Rubric):
    """Verify a resolved string value fully matches a regex pattern.

    Uses ``re.fullmatch`` so the pattern is implicitly anchored — a
    partial substring match doesn't count. Useful for ID formats,
    canonical mode names, or any predictable string shape.
    """

    name = "regex_match"

    def __init__(self, target: str, pattern: str) -> None:
        self.target = target
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def judge(self, ctx: RubricContext) -> Verdict:
        try:
            value = resolve_capture_target(ctx.captures, self.target)
        except (KeyError, IndexError) as exc:
            return _missing_path_verdict(self.target, exc)

        if not isinstance(value, str):
            return Verdict(
                status="fail",
                rationale="target value is not a string",
                diagnostic={"got": value, "got_type": type(value).__name__},
            )

        if self._compiled.fullmatch(value) is None:
            return Verdict(
                status="fail",
                rationale=f"{self.target} does not match pattern",
                diagnostic={"got": value, "pattern": self.pattern},
            )
        return Verdict(
            status="pass",
            rationale=f"{self.target} matches pattern",
        )


@register_rubric
class NumericRange(Rubric):
    """Verify a numeric capture lies within ``[min_value, max_value]``.

    Either bound is optional but at least one must be set (a rubric
    with no bounds would be a no-op, almost certainly a config bug).
    ``bool`` values are rejected up front: although Python treats
    ``True``/``False`` as ints, the scenario author who wrote a
    NumericRange rubric meant numbers, not flags.
    """

    name = "numeric_range"

    def __init__(
        self,
        target: str,
        min_value: float | int | None,
        max_value: float | int | None,
    ) -> None:
        if min_value is None and max_value is None:
            raise ValueError(
                "NumericRange requires at least one of min_value or max_value"
            )
        self.target = target
        self.min_value = min_value
        self.max_value = max_value

    def judge(self, ctx: RubricContext) -> Verdict:
        try:
            value = resolve_capture_target(ctx.captures, self.target)
        except (KeyError, IndexError) as exc:
            return _missing_path_verdict(self.target, exc)

        # bool is a subclass of int — reject it explicitly so a True
        # doesn't masquerade as a score of 1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return Verdict(
                status="fail",
                rationale="target value is not numeric",
                diagnostic={"got": value, "got_type": type(value).__name__},
            )

        below_min = self.min_value is not None and value < self.min_value
        above_max = self.max_value is not None and value > self.max_value
        if below_min or above_max:
            return Verdict(
                status="fail",
                rationale=f"{self.target} is outside [{self.min_value}, {self.max_value}]",
                diagnostic={
                    "got": value,
                    "min_value": self.min_value,
                    "max_value": self.max_value,
                },
            )
        return Verdict(
            status="pass",
            rationale=f"{self.target} is within range",
        )
