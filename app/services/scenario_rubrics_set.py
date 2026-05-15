"""Set-style scenario rubrics: ``SubsetMatch`` and ``BehavioralEquivalence``.

These two rubrics compare values pulled out of the scenario trace
(``captures``) against an "acceptable" set or an expected scalar. They
are pure, deterministic, and free — no LLM calls — so they run on every
grade.

See ``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``
for the broader rubric roster and the YAML kinds they bind to
(``subset_match`` and ``behavioral_equivalence``).
"""
from __future__ import annotations

from typing import Any

from app.services.scenario_rubrics_base import (
    Rubric,
    RubricContext,
    Verdict,
    register_rubric,
    resolve_capture_target,
    resolve_path,
)

_SETUP_PREFIX = "setup_data."


def _resolve_in_context(ctx: RubricContext, dotted_path: str) -> Any:
    """Resolve ``dotted_path`` against either ``setup_data`` or ``captures``.

    Paths starting with ``setup_data.`` look up inside
    ``ctx.setup_data``; all other paths resolve against ``ctx.captures``.
    Raises ``KeyError`` / ``IndexError`` like ``resolve_path``.
    """
    if dotted_path.startswith(_SETUP_PREFIX):
        return resolve_path(ctx.setup_data, dotted_path[len(_SETUP_PREFIX) :])
    return resolve_capture_target(ctx.captures, dotted_path)


@register_rubric
class SubsetMatch(Rubric):
    """Verify a captured set of values is (mostly) a subset of an acceptable set.

    Used for things like "every ``cited_chunk_id`` the learner returned
    must come from a chunk that actually exists in the hidden corpus."

    Config:
      - ``target``: dotted path (in ``captures``) to a list of values.
      - ``acceptable_source``: dotted path to the acceptable collection.
        Prefix with ``setup_data.`` to read from ``ctx.setup_data``;
        otherwise resolves against ``ctx.captures``.
      - ``acceptable_key``: if the acceptable source is a list of dicts,
        pull this key from each. When ``None``, treat each element as a
        value directly.
      - ``min_overlap``: fraction of target values that must be in the
        acceptable set. Default ``1.0`` enforces strict subset.
    """

    name = "subset_match"

    def __init__(
        self,
        *,
        target: str,
        acceptable_source: str,
        acceptable_key: str | None = None,
        min_overlap: float = 1.0,
    ) -> None:
        self.target = target
        self.acceptable_source = acceptable_source
        self.acceptable_key = acceptable_key
        self.min_overlap = min_overlap

    def judge(self, ctx: RubricContext) -> Verdict:
        try:
            target_values = resolve_capture_target(ctx.captures, self.target)
        except (KeyError, IndexError):
            return Verdict(
                status="fail",
                rationale=f"target path '{self.target}' not found in captures",
                diagnostic={"missing_path": self.target},
            )

        try:
            acceptable_raw = _resolve_in_context(ctx, self.acceptable_source)
        except (KeyError, IndexError):
            return Verdict(
                status="fail",
                rationale=(
                    f"acceptable_source path '{self.acceptable_source}' not found"
                ),
                diagnostic={"missing_path": self.acceptable_source},
            )

        target_list = list(target_values)
        if len(target_list) == 0:
            return Verdict(
                status="fail",
                rationale="target is empty; cannot check subset",
                diagnostic={
                    "target_size": 0,
                    "overlap_fraction": 0.0,
                    "min_overlap": self.min_overlap,
                    "invalid_values": [],
                },
            )

        if self.acceptable_key is not None:
            acceptable_values = {
                item[self.acceptable_key]
                for item in acceptable_raw
                if isinstance(item, dict) and self.acceptable_key in item
            }
        else:
            acceptable_values = set(acceptable_raw)

        if len(acceptable_values) == 0:
            return Verdict(
                status="fail",
                rationale="acceptable set is empty; cannot check subset",
                diagnostic={
                    "target_size": len(target_list),
                    "overlap_fraction": 0.0,
                    "min_overlap": self.min_overlap,
                    "invalid_values": list(target_list),
                },
            )

        invalid = [v for v in target_list if v not in acceptable_values]
        overlap_fraction = (len(target_list) - len(invalid)) / len(target_list)
        diagnostic = {
            "invalid_values": invalid,
            "target_size": len(target_list),
            "overlap_fraction": overlap_fraction,
            "min_overlap": self.min_overlap,
        }
        if overlap_fraction >= self.min_overlap:
            return Verdict(
                status="pass",
                rationale=(
                    f"{len(target_list) - len(invalid)} of {len(target_list)} "
                    f"target values fall inside the acceptable set"
                ),
                diagnostic=diagnostic,
            )
        return Verdict(
            status="fail",
            rationale=(
                f"only {overlap_fraction:.0%} of target values are in the "
                f"acceptable set (need {self.min_overlap:.0%})"
            ),
            diagnostic=diagnostic,
        )


# Sentinel for "expected was not supplied". Plain ``None`` is a valid
# literal value (e.g., ``abstained == None``) so we can't use it.
_MISSING = object()


@register_rubric
class BehavioralEquivalence(Rubric):
    """Verify a captured scalar equals an expected value or another path.

    Used for categorical / boolean behavior assertions like "this
    question is out-of-corpus, so ``abstained`` must be ``true``" or
    "this endpoint should return status ``404``", AND for path-vs-path
    equivalence like "the response from ``first_call`` matches the
    response from ``second_call``" (idempotency / adversarial-reorder
    scenarios).

    Config — provide exactly one of ``expected`` / ``expected_path``:
      - ``target``: dotted path (in ``captures``) to a value.
      - ``expected``: a literal value the target must equal. Used for
        boolean / categorical assertions.
      - ``expected_path``: dotted path to another value the target must
        match. Resolved like ``target`` (uses ``ctx.captures``). When
        both are provided, ``expected_path`` wins.
      - ``case_sensitive``: when ``False`` and both sides are strings,
        compare case-insensitively.

    Bug 19 (2026-05-15): the original rubric only supported literal
    ``expected`` values. Cross-step equivalence — the canonical
    idempotency check ``first_call.body.answer == second_call.body.answer``
    — was impossible. ``expected_path`` closes that gap without
    breaking the literal-value form.
    """

    name = "behavioral_equivalence"

    def __init__(
        self,
        *,
        target: str,
        expected: Any = _MISSING,
        expected_path: str | None = None,
        case_sensitive: bool = True,
    ) -> None:
        if expected is _MISSING and expected_path is None:
            raise TypeError(
                "BehavioralEquivalence requires either ``expected`` "
                "(literal value) or ``expected_path`` (path into captures)."
            )
        self.target = target
        self.expected = expected
        self.expected_path = expected_path
        self.case_sensitive = case_sensitive

    def judge(self, ctx: RubricContext) -> Verdict:
        try:
            got = resolve_capture_target(ctx.captures, self.target)
        except (KeyError, IndexError):
            return Verdict(
                status="fail",
                rationale=f"target path '{self.target}' not found in captures",
                diagnostic={"missing_path": self.target},
            )

        # Resolve ``expected`` from ``expected_path`` when provided.
        # Path-vs-path takes priority over literal — the explicit
        # ``expected_path`` kwarg signals "compare to a captured value".
        if self.expected_path is not None:
            try:
                expected_value = resolve_capture_target(ctx.captures, self.expected_path)
            except (KeyError, IndexError):
                return Verdict(
                    status="fail",
                    rationale=(
                        f"expected_path '{self.expected_path}' not found "
                        f"in captures"
                    ),
                    diagnostic={"missing_path": self.expected_path},
                )
        else:
            expected_value = self.expected

        if (
            not self.case_sensitive
            and isinstance(got, str)
            and isinstance(expected_value, str)
        ):
            equal = got.casefold() == expected_value.casefold()
        else:
            equal = got == expected_value

        if equal:
            return Verdict(
                status="pass",
                rationale=f"target equals expected ({expected_value!r})",
                diagnostic={"got": got, "expected": expected_value},
            )
        return Verdict(
            status="fail",
            rationale=(
                f"expected {expected_value!r} at '{self.target}', got {got!r}"
            ),
            diagnostic={"got": got, "expected": expected_value},
        )
