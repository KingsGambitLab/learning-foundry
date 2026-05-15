"""Feedback synthesizer for the scenario-rubric grader pipeline.

The trace runner produces a list of ``ScenarioOutcome``s — each one a
collection of ``(rubric_kind, Verdict)`` pairs together with the
``QualityBar.id``s the scenario contributes to. This module rolls those
per-scenario verdicts up into per-``QualityBar`` reports and into a
final ``GraderFeedbackReport`` consumed by the learner UI and the
reviewer-repair LLM.

The bridge ``report_to_reviewer_findings`` converts each failed bar
into a ``ReviewerFinding`` so the structured feedback flows through the
existing reviewer-repair channel without bespoke plumbing — preserving
the matched ``LearningHint`` on the ``hint`` field, which is the key
plumbing the spec calls out as user-visible.

See ``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``
"Stage 5 — Feedback synthesis" for the design.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    LearningHint,
    QualityBar,
    QualityBarAggregation,
)
from app.services.scenario_rubrics_base import Verdict

__all__ = [
    "ScenarioOutcome",
    "QualityBarReport",
    "GraderFeedbackReport",
    "ThresholdParseError",
    "UnsupportedThresholdError",
    "synthesize_grader_feedback",
    "report_to_reviewer_findings",
]


# ---------------- Input / output models ----------------


class ScenarioOutcome(BaseModel):
    """Per-scenario rollup of rubric verdicts produced by the trace runner.

    ``verdicts`` is a list of ``(rubric_kind, Verdict)`` tuples because a
    single scenario can be judged by multiple rubrics (structural +
    behavioral + LLM). ``quality_bar_ids`` records which bars this
    scenario contributes evidence toward; the synthesizer fans the
    scenario's pass/fail call out to each listed bar.
    """

    scenario_id: str
    category: str
    description: str
    verdicts: list[tuple[str, Verdict]]
    quality_bar_ids: list[str]


class QualityBarReport(BaseModel):
    """Per-bar rollup. One per ``QualityBar`` in the spec.

    ``threshold_type`` records how the threshold was interpreted so
    downstream consumers (feedback synthesizer, repair LLM, learner UI)
    know which kind of bar they're reading without re-parsing the raw
    threshold string. Emitted values are ``"ratio"``, ``"count"``, and
    ``"categorical"``; ``"measured_value"`` is reserved for a future
    wave that ships measurement rubrics (latency / throughput / memory).

    Uncovered bars (no scenarios target them) record ``threshold_type``
    as ``None`` since the threshold was never parsed, and ``status`` is
    ``"fail"`` with ``rationale`` explaining the configuration error
    (Codex review #4 finding #2). ``"abstain"`` is reserved for future
    use; today no code path emits it.

    ``rationale`` is set when the bar's status is driven by something
    other than the per-scenario rollup — currently only the uncovered-
    bar case writes a non-``None`` value. Distinct from
    ``failure_diagnostics`` (which collects per-rationale strings from
    failing verdicts) so consumers can pull the single load-bearing
    explanation without scanning a list.
    """

    bar_id: str
    metric_description: str
    threshold: str
    threshold_type: (
        Literal["ratio", "count", "categorical", "measured_value"] | None
    ) = None
    observed_value: str
    status: Literal["pass", "fail", "abstain"]
    rationale: str | None = None
    failing_scenario_ids: list[str] = Field(default_factory=list)
    failure_diagnostics: list[str] = Field(default_factory=list)
    matched_hint: str | None = None


class GraderFeedbackReport(BaseModel):
    """Top-level structured feedback for a single grade run.

    ``coverage_failures`` lists the ``QualityBar.id``s declared in the
    spec but referenced by zero scenarios — a course-authoring
    configuration error that must NOT be silently graded as a learner
    pass (Codex review #4 finding #2). When ``coverage_failures`` is
    non-empty, ``overall_status`` is forced to ``"fail"`` independently
    of the per-bar rollup so a misconfigured grader never reaches the
    learner as a passing grade.

    ``actionable_feedback`` holds two flavors of remediation:

    - Learner-targeted lines like
      ``[faithfulness] threshold >= 0.8, observed 0.62. <hint>``
    - Course-author-targeted lines, marker-prefixed with
      ``[course author]`` so the UI / repair LLM can route them to the
      author-facing channel without parsing the bar id.

    Decision: marker prefix in a single list (over a parallel
    ``course_author_feedback`` field) because every existing consumer
    already iterates ``actionable_feedback`` — keeping one list avoids
    a downstream-schema break and the marker is load-bearing and
    cheap to filter.
    """

    course_title: str
    total_scenarios: int
    passed_scenarios: int
    quality_bar_reports: list[QualityBarReport]
    overall_status: Literal["pass", "fail"]
    summary: str
    actionable_feedback: list[str] = Field(default_factory=list)
    coverage_failures: list[str] = Field(default_factory=list)


class ThresholdParseError(ValueError):
    """Raised when a ``QualityBar.threshold`` can't be parsed.

    A misconfigured spec should fail loudly at synthesis time rather
    than silently grade-time, so the course author can fix the YAML.
    """


class UnsupportedThresholdError(ValueError):
    """Raised when a threshold uses a measurement unit we don't yet support.

    Patterns like ``"<= 2000ms"`` or ``">= 1000 req/s"`` require rubrics
    that emit per-scenario ``measured_value`` diagnostics (latency,
    throughput, memory). Those rubrics ship in a later wave; until then
    we fail loud at synthesis time so course authors aren't surprised
    by silently mis-graded bars when learners hit "Run".
    """


# ---------------- Threshold parsing ----------------

# Top-level shape: an operator followed by a non-empty RHS.
# Operators are listed longest-first so the regex tokenizer picks ``>=``
# over ``>`` etc. when scanning the threshold string. ``!=`` is included
# so categorical bars can express the "absence of behavior" form
# ``!= true`` (semantic alias of ``== false``).
_THRESHOLD_PATTERN = re.compile(
    r"^\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<rhs>.+?)\s*$",
)

# A bare numeric ratio — optional sign + digits + optional fractional
# part + optional ``%`` (for the human-friendly percent form). No
# other unit characters allowed; if any are present the RHS instead
# matches ``_UNIT_BEARING_PATTERN`` below.
_RATIO_RHS_PATTERN = re.compile(
    r"^(?P<num>-?\d+(?:\.\d+)?)\s*(?P<percent>%)?$",
)

# A number followed by a unit token (``ms``, ``s``, ``MB``, ``req/s``).
# We deliberately match anything non-empty trailing the number that
# isn't ``%`` — those are the patterns we reject in v1.
_UNIT_BEARING_PATTERN = re.compile(
    r"^(?P<num>-?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z][A-Za-z0-9/]*)$",
)

# A bare non-negative integer — for count aggregations the RHS must be
# an integer count (no fractional part, no unit).
_INT_RHS_PATTERN = re.compile(r"^(?P<num>\d+)$")


def _parse_threshold(
    threshold_str: str,
    aggregation: QualityBarAggregation = QualityBarAggregation.ratio,
) -> tuple[
    str,
    float | int | bool | str,
    Literal["ratio", "count", "categorical"],
]:
    """Parse a threshold expression into ``(operator, value, kind)``.

    Supported operators: ``>=``, ``<=``, ``==``, ``!=``, ``>``, ``<``.
    ``!=`` is reserved for categorical bars (see below).

    The ``aggregation`` argument selects how the right-hand side is
    interpreted:

    - ``ratio`` (default) — RHS must be a numeric pass-rate in ``[0, 1]``
      (the percent form ``"70%"`` is normalized to ``0.70``). Returns
      a ``float`` value tagged ``"ratio"``. ``0.0`` is a valid ratio
      threshold; bar authors who actually mean "zero failures" should
      opt in to ``count_failing`` aggregation rather than relying on
      ``ratio`` semantics.
    - ``count_failing`` / ``count_passing`` — RHS must be a non-negative
      integer. Returns an ``int`` value tagged ``"count"``. A mismatch
      (e.g. ``>= 0.7`` under a count aggregation) raises
      ``ThresholdParseError`` so the misconfigured spec fails loudly at
      synthesis time rather than silently mis-graded.
    - ``categorical`` (or any bar whose RHS does not parse as a ratio) —
      RHS must be exactly ``"true"`` or ``"false"`` (case-insensitive) in
      v1; operator must be ``==`` or ``!=``. Returns a ``bool`` value
      tagged ``"categorical"``. Non-boolean tokens like ``"json"`` and
      comparison operators like ``>=`` raise ``ThresholdParseError`` —
      v1 has no measurement rubric that emits a non-boolean observed
      value, so accepting them would silently misgrade the bar. This is
      the Codex review #5 fix.

    Unit-bearing RHSes (``"2000ms"``, ``"1000 req/s"``, ``"100MB"``)
    raise ``UnsupportedThresholdError`` because the measurement rubrics
    that would emit a numeric observed value are not yet shipped; a
    future wave will add ``"measured_value"`` as a fourth kind.
    """
    match = _THRESHOLD_PATTERN.match(threshold_str)
    if match is None:
        raise ThresholdParseError(
            f"Unparseable threshold expression: {threshold_str!r}. "
            f"Expected one of '>=', '<=', '==', '!=', '>', '<' followed by "
            f"a value."
        )
    op = match.group("op")
    rhs = match.group("rhs").strip()

    # Count aggregations: RHS must be a bare non-negative integer.
    # Anything else (fractional, percent, unit-bearing, free-form token)
    # is a configuration mismatch we surface loudly.
    if aggregation in (
        QualityBarAggregation.count_failing,
        QualityBarAggregation.count_passing,
    ):
        int_match = _INT_RHS_PATTERN.match(rhs)
        if int_match is None:
            raise ThresholdParseError(
                f"Threshold {threshold_str!r} with aggregation "
                f"{aggregation.value!r} must be a non-negative integer count "
                f"(e.g. '== 0', '<= 2', '>= 8'). Got RHS {rhs!r}."
            )
        return op, int(int_match.group("num")), "count"

    # Ratio first: matches digits with an optional ``%`` and nothing else.
    ratio_match = _RATIO_RHS_PATTERN.match(rhs)
    if ratio_match is not None:
        value = float(ratio_match.group("num"))
        if ratio_match.group("percent"):
            value /= 100.0
        if not (0.0 <= value <= 1.0):
            raise ThresholdParseError(
                f"Threshold {threshold_str!r} is out of range: "
                f"pass-rate ratios must be in [0, 1] after normalization "
                f"(got {value}). For raw numeric thresholds like '<= 2000ms', "
                f"use a unit and note that measurement rubrics are not yet "
                f"supported."
            )
        return op, value, "ratio"

    # Unit-bearing: numeric value with a non-percent suffix. v1 has no
    # measurement rubrics, so we fail loud at synthesis time.
    unit_match = _UNIT_BEARING_PATTERN.match(rhs)
    if unit_match is not None:
        unit = unit_match.group("unit")
        raise UnsupportedThresholdError(
            f"Threshold {threshold_str!r} uses unit {unit!r}; "
            f"measurement-rubric support is not yet implemented. "
            f"Use a ratio threshold (e.g., '>= 0.7' or '>= 70%') for v1."
        )

    # Categorical: v1 only supports boolean RHS (``true``/``false``) and
    # the equality operators ``==`` / ``!=``. Non-boolean tokens like
    # ``"json"`` would need a measurement rubric that emits a non-boolean
    # observed value — none ship yet, and accepting them silently
    # caused the misgrading bug behind Codex review #5.
    rhs_lower = rhs.lower()
    if rhs_lower not in ("true", "false"):
        raise ThresholdParseError(
            f"Threshold {threshold_str!r} is categorical but its RHS "
            f"{rhs!r} is not a supported boolean token. Categorical bars "
            f"in v1 must use 'true' or 'false' (case-insensitive); "
            f"non-boolean observed values require a measurement rubric "
            f"that does not ship yet."
        )
    if op not in ("==", "!="):
        raise ThresholdParseError(
            f"Threshold {threshold_str!r} uses comparison operator "
            f"{op!r} against a boolean RHS, which is not meaningful. "
            f"Categorical bars must use '==' or '!='."
        )
    return op, rhs_lower == "true", "categorical"


def _compare_numeric(
    op: str, observed: float | int, threshold: float | int
) -> bool:
    if op == ">=":
        return observed >= threshold
    if op == "<=":
        return observed <= threshold
    if op == "==":
        return observed == threshold
    if op == ">":
        return observed > threshold
    if op == "<":
        return observed < threshold
    # _parse_threshold has already validated op, so reaching here is a bug.
    raise ThresholdParseError(f"Unknown operator after parse: {op!r}")


# ---------------- Helpers ----------------


def _scenario_passes(outcome: ScenarioOutcome) -> bool:
    """A scenario contributes a 'pass' to its bars only if EVERY rubric
    verdict it carries is ``pass``.

    Any ``fail`` flips the scenario to fail. Abstains are not failures
    on their own — a scenario that only has abstains is not counted as
    a pass (it cannot prove success) and not counted as a hard fail
    (it didn't disprove anything either). See ``_scenario_fails``.
    """
    statuses = [v.status for _, v in outcome.verdicts]
    if not statuses:
        return False
    return all(s == "pass" for s in statuses)


def _scenario_fails(outcome: ScenarioOutcome) -> bool:
    """A scenario fails iff at least one of its verdicts is ``fail``."""
    return any(v.status == "fail" for _, v in outcome.verdicts)


def _collect_failing_rationales(outcomes: list[ScenarioOutcome]) -> list[str]:
    """Dedupe rationales from ``fail`` verdicts (preserve first-seen order)."""
    seen: set[str] = set()
    rationales: list[str] = []
    for outcome in outcomes:
        for _, verdict in outcome.verdicts:
            if verdict.status == "fail" and verdict.rationale not in seen:
                seen.add(verdict.rationale)
                rationales.append(verdict.rationale)
    return rationales


def _find_hint(
    learning_path: list[LearningHint], bar_id: str
) -> str | None:
    for hint in learning_path:
        if hint.on_metric_fail == bar_id:
            return hint.hint
    return None


def _build_bar_report(
    bar: QualityBar,
    relevant: list[ScenarioOutcome],
    matched_hint: str | None,
) -> QualityBarReport:
    """Roll the scenarios that touch this bar into a ``QualityBarReport``.

    The bar's ``aggregation`` field drives the comparison and the
    ``observed_value`` shape:

    - ratio: ``"0.62 (47 of 50 scenarios passed)"`` — pass-rate float
      drives the comparison; N-of-M tail gives humans context.
    - count_failing: ``"2 failing scenarios"`` — the integer failure
      count is compared to the threshold. Fixes the inverted-grading
      bug for bars like ``stub_resistance == 0``.
    - count_passing: ``"8 passing scenarios"`` — the integer pass count
      drives the comparison.
    - categorical: ``"true"`` / ``"false"`` — bar is "true" only when
      every targeted scenario passed.
    - abstain (no targeted scenarios): ``"0 of 0 scenarios passed"``;
      ``threshold_type`` is left ``None`` because we never parsed it.
    """
    total = len(relevant)
    failing = [o for o in relevant if _scenario_fails(o)]
    passing = [o for o in relevant if _scenario_passes(o)]

    if total == 0:
        # No scenarios target this bar — this is a course-authoring
        # configuration error, NOT a learner-graded outcome. We
        # deliberately return ``status='fail'`` (not ``'abstain'``) so
        # the per-bar report is honest in isolation: even if the
        # synthesizer-level coverage check is bypassed, downstream
        # consumers see a clear fail with a config-error diagnostic.
        # See Codex review #4 finding #2.
        return QualityBarReport(
            bar_id=bar.id,
            metric_description=bar.metric_description,
            threshold=bar.threshold,
            threshold_type=None,
            observed_value=f"{len(passing)} of {total} scenarios passed",
            status="fail",
            rationale=(
                "No scenarios contribute to this bar — grader is incomplete."
            ),
            failing_scenario_ids=[],
            failure_diagnostics=[
                (
                    f"Bar '{bar.id}' is declared but no scenario references "
                    f"it. This is a grader configuration error, not a "
                    f"learner failure."
                )
            ],
            matched_hint=matched_hint,
        )

    op, threshold_value, kind = _parse_threshold(bar.threshold, bar.aggregation)
    status: Literal["pass", "fail", "abstain"]
    if bar.aggregation is QualityBarAggregation.count_failing:
        # Count-of-failures bar: observed is the integer failure count.
        assert isinstance(threshold_value, int)
        observed_count = len(failing)
        status = (
            "pass"
            if _compare_numeric(op, observed_count, threshold_value)
            else "fail"
        )
        observed_value = f"{observed_count} failing scenarios"
    elif bar.aggregation is QualityBarAggregation.count_passing:
        # Count-of-passes bar: observed is the integer pass count.
        assert isinstance(threshold_value, int)
        observed_count = len(passing)
        status = (
            "pass"
            if _compare_numeric(op, observed_count, threshold_value)
            else "fail"
        )
        observed_value = f"{observed_count} passing scenarios"
    elif kind == "ratio":
        # Ratio bar: compare pass-rate against the normalized threshold.
        # ``bar.aggregation`` is either ratio (default) or categorical;
        # ``kind`` is what the parser actually inferred from the RHS so
        # a bar that opted in to categorical aggregation but happened
        # to write a numeric threshold still routes through here.
        assert isinstance(threshold_value, float)
        ratio = len(passing) / total
        status = (
            "pass" if _compare_numeric(op, ratio, threshold_value) else "fail"
        )
        observed_value = (
            f"{ratio:.2f} ({len(passing)} of {total} scenarios passed)"
        )
    else:
        # Categorical bar: observed truthiness is "every relevant
        # scenario passed". Compare against the configured boolean
        # threshold using the configured operator (Codex review #5
        # fix — previously ``op`` and ``threshold_value`` were parsed
        # but discarded, so ``== false`` could never pass).
        assert isinstance(threshold_value, bool)
        observed_bool = len(passing) == total
        if op == "==":
            passed = observed_bool == threshold_value
        elif op == "!=":
            passed = observed_bool != threshold_value
        else:
            # _parse_threshold has already rejected non-equality operators
            # for categorical kind, so reaching here is a bug.
            raise ThresholdParseError(
                f"Categorical bar '{bar.id}' reached the reporter with "
                f"unsupported operator {op!r}; the parser should have "
                f"rejected this at synthesis time."
            )
        status = "pass" if passed else "fail"
        observed_value = "true" if observed_bool else "false"

    rationales = _collect_failing_rationales(failing)[:5]
    # Categorical-specific diagnostic: when the bar fails, surface the
    # observed-vs-expected truthiness explicitly so downstream consumers
    # (repair LLM, learner UI) know which side of the equality drove
    # the failure without re-deriving it from ``observed_value``.
    if kind == "categorical" and status == "fail":
        expected_label = "true" if threshold_value else "false"
        rationales = [
            (
                f"Categorical bar '{bar.id}': expected {expected_label!r}, "
                f"observed {observed_value!r} ({len(passing)} of {total} "
                f"scenarios passed)."
            ),
            *rationales,
        ][:5]

    return QualityBarReport(
        bar_id=bar.id,
        metric_description=bar.metric_description,
        threshold=bar.threshold,
        threshold_type=kind,
        observed_value=observed_value,
        status=status,
        failing_scenario_ids=[o.scenario_id for o in failing],
        failure_diagnostics=rationales,
        matched_hint=matched_hint,
    )


def _build_summary(
    *,
    total_scenarios: int,
    passed_scenarios: int,
    bar_reports: list[QualityBarReport],
) -> str:
    failed_bars = [b for b in bar_reports if b.status == "fail"]
    n_bars = len(bar_reports)
    if not failed_bars:
        return (
            f"Passed {passed_scenarios} of {total_scenarios} scenarios. "
            f"All {n_bars} quality bars passed."
        )
    failed_ids = ", ".join(b.bar_id for b in failed_bars)
    return (
        f"Passed {passed_scenarios} of {total_scenarios} scenarios. "
        f"{len(failed_bars)} of {n_bars} quality bars failed: {failed_ids}."
    )


def _build_actionable_feedback(
    bar_reports: list[QualityBarReport],
    uncovered_bar_ids: set[str],
) -> list[str]:
    """Build the actionable-feedback list.

    Two flavors are mixed into the same list, with the course-author
    flavor carrying a ``[course author]`` marker prefix so downstream
    consumers can route the two flows independently without parsing
    the bar id. See ``GraderFeedbackReport`` docstring for the
    decision rationale.
    """
    lines: list[str] = []
    for bar in bar_reports:
        if bar.status != "fail":
            continue
        if bar.bar_id in uncovered_bar_ids:
            # Course-author-targeted: the learner cannot influence this.
            lines.append(
                f"[course author] Quality bar `{bar.bar_id}` has no "
                f"contributing scenarios. Either author scenarios that "
                f"reference this bar in `quality_bar_ids`, or remove the "
                f"bar from the spec."
            )
            continue
        suffix = (
            bar.matched_hint
            if bar.matched_hint
            else "No learning_path hint configured."
        )
        lines.append(
            f"[{bar.bar_id}] threshold {bar.threshold}, "
            f"observed {bar.observed_value}. {suffix}"
        )
    return lines


# ---------------- Public API ----------------


def synthesize_grader_feedback(
    *,
    spec: CourseOutcomeSpec,
    scenario_outcomes: list[ScenarioOutcome],
) -> GraderFeedbackReport:
    """Roll per-scenario verdicts up to per-quality-bar reports and into
    the final structured feedback the learner / repair LLM consumes.

    Coverage validation (Codex review #4 finding #2): any
    ``QualityBar.id`` declared in the spec but referenced by zero
    scenarios is recorded in ``coverage_failures`` and flips
    ``overall_status`` to ``fail`` independently of the per-bar
    rollup. Coverage is determined purely by ``quality_bar_ids``
    membership on the scenario outcomes — a scenario that references
    the bar counts toward coverage even when its rubrics all abstain.
    The rationale: "did the course author author SOMETHING that points
    at this bar?" is the configuration question we are checking. The
    separate question "did those rubrics actually evaluate the bar?"
    is the per-bar status (which still surfaces a fail when no verdict
    is conclusive).
    """
    # --- Coverage check (course-authoring configuration error) ---
    targeted_bar_ids: set[str] = {
        bar_id
        for outcome in scenario_outcomes
        for bar_id in outcome.quality_bar_ids
    }
    coverage_failures: list[str] = [
        (
            f"Quality bar '{bar.id}' is declared in the spec but no "
            f"scenario contributes to it"
        )
        for bar in spec.quality_bars
        if bar.id not in targeted_bar_ids
    ]
    uncovered_bar_ids: set[str] = {
        bar.id for bar in spec.quality_bars if bar.id not in targeted_bar_ids
    }

    bar_reports: list[QualityBarReport] = []
    for bar in spec.quality_bars:
        relevant = [
            o for o in scenario_outcomes if bar.id in o.quality_bar_ids
        ]
        matched_hint = _find_hint(spec.learning_path, bar.id)
        bar_reports.append(_build_bar_report(bar, relevant, matched_hint))

    total_scenarios = len(scenario_outcomes)
    passed_scenarios = sum(1 for o in scenario_outcomes if _scenario_passes(o))

    # Coverage failures force overall fail regardless of per-bar status.
    has_failing_bar = any(b.status == "fail" for b in bar_reports)
    overall_status: Literal["pass", "fail"] = (
        "fail" if (coverage_failures or has_failing_bar) else "pass"
    )

    summary = _build_summary(
        total_scenarios=total_scenarios,
        passed_scenarios=passed_scenarios,
        bar_reports=bar_reports,
    )
    actionable = _build_actionable_feedback(bar_reports, uncovered_bar_ids)

    return GraderFeedbackReport(
        course_title=spec.title,
        total_scenarios=total_scenarios,
        passed_scenarios=passed_scenarios,
        quality_bar_reports=bar_reports,
        overall_status=overall_status,
        summary=summary,
        actionable_feedback=actionable,
        coverage_failures=coverage_failures,
    )


def report_to_reviewer_findings(
    report: GraderFeedbackReport,
) -> list[ReviewerFinding]:
    """Convert each failed ``QualityBarReport`` into a ``ReviewerFinding``
    so the structured feedback flows through the existing reviewer-repair
    channel — one finding per failed bar, with ``hint`` populated from the
    matched ``LearningHint`` (the load-bearing plumbing for the repair LLM
    and the learner UI)."""
    findings: list[ReviewerFinding] = []
    for bar in report.quality_bar_reports:
        if bar.status != "fail":
            continue
        fallback_hint = (
            f"[{bar.bar_id}] threshold {bar.threshold}, "
            f"observed {bar.observed_value}. No learning_path hint configured."
        )
        detail_parts = [
            f"Quality bar '{bar.bar_id}' failed: "
            f"threshold {bar.threshold}, observed {bar.observed_value}."
        ]
        if bar.failure_diagnostics:
            detail_parts.append(
                "Diagnostics: " + " | ".join(bar.failure_diagnostics)
            )
        findings.append(
            ReviewerFinding(
                category="quality_bar",
                severity=ReviewerFindingSeverity.error,
                title=f"Quality bar '{bar.bar_id}' failed",
                detail=" ".join(detail_parts),
                code=f"quality_bar_failed_{bar.bar_id}",
                hint=bar.matched_hint if bar.matched_hint else fallback_hint,
            )
        )
    return findings
