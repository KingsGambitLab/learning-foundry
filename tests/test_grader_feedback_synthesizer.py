"""Tests for the grader feedback synthesizer.

Covers ``_parse_threshold`` helper, ``synthesize_grader_feedback``
rollup logic, and the ``report_to_reviewer_findings`` bridge into the
canonical reviewer-repair channel.
"""
from __future__ import annotations

import pytest

from app.domain.registry import PackageType
from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    LearningHint,
    QualityBar,
    QualityBarAggregation,
    StarterType,
)
from app.services.grader_feedback_synthesizer import (
    GraderFeedbackReport,
    QualityBarReport,
    ScenarioOutcome,
    ThresholdParseError,
    UnsupportedThresholdError,
    _build_bar_report,
    _parse_threshold,
    report_to_reviewer_findings,
    synthesize_grader_feedback,
)
from app.services.scenario_rubrics_base import Verdict


# ---------------- helpers ----------------

def _make_spec(
    *,
    quality_bars: list[QualityBar],
    learning_path: list[LearningHint] | None = None,
    title: str = "Test Course Title",
) -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title=title,
        goal="Build a retrieval service that meets quality bars.",
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/ask",
                request_schema={"query": "str"},
                response_schema={"answer": "str"},
                description="Answer a query.",
            )
        ],
        quality_bars=quality_bars,
        learning_path=learning_path or [],
        package_type=PackageType.progressive_codebase_course,
    )


def _v(status: str, rationale: str = "ok") -> Verdict:
    return Verdict(status=status, rationale=rationale)


def _outcome(
    scenario_id: str,
    bar_ids: list[str],
    verdicts: list[tuple[str, Verdict]],
    *,
    category: str = "happy_path",
    description: str = "scenario",
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario_id,
        category=category,
        description=description,
        verdicts=verdicts,
        quality_bar_ids=bar_ids,
    )


# ---------------- _parse_threshold ----------------

def test_parse_threshold_gte() -> None:
    assert _parse_threshold(">= 0.8") == (">=", 0.8, "ratio")


def test_parse_threshold_lte() -> None:
    # Bare numeric values without a unit are parsed as ratios; 2000 is
    # out of the [0, 1] ratio range and must fail loudly.
    with pytest.raises(ThresholdParseError):
        _parse_threshold("<= 2000")


def test_parse_threshold_eq() -> None:
    assert _parse_threshold("== 1.0") == ("==", 1.0, "ratio")


def test_parse_threshold_gt() -> None:
    assert _parse_threshold("> 0.5") == (">", 0.5, "ratio")


def test_parse_threshold_lt() -> None:
    # Same as test_parse_threshold_lte: bare 100 is not a ratio.
    with pytest.raises(ThresholdParseError):
        _parse_threshold("< 100")


def test_parse_threshold_unit_bearing_rejected() -> None:
    # Unit-bearing thresholds (latency / throughput / size) require
    # measurement rubrics that are not shipped in v1; the parser must
    # reject them at synthesis time rather than silently mis-grading.
    with pytest.raises(UnsupportedThresholdError) as excinfo:
        _parse_threshold("<= 2000ms")
    msg = str(excinfo.value)
    assert "2000ms" in msg or "ms" in msg
    assert "measurement" in msg.lower() or "not yet" in msg.lower()


def test_parse_threshold_percent_normalized_to_ratio() -> None:
    # ``95%`` is the human-friendly form of ``0.95`` — the parser
    # must normalize and tag the result as a ratio.
    assert _parse_threshold(">= 95%") == (">=", 0.95, "ratio")


def test_parse_threshold_percent_seventy_normalized() -> None:
    # ``70%`` normalizes to ``0.70`` and is tagged ratio.
    assert _parse_threshold(">= 70%") == (">=", 0.70, "ratio")


def test_parse_threshold_ratio_out_of_range_raises() -> None:
    # After normalization a ratio must be in [0, 1]. ``>= 1.5`` is
    # nonsense; fail loudly.
    with pytest.raises(ThresholdParseError):
        _parse_threshold(">= 1.5")


def test_parse_threshold_negative_ratio_raises() -> None:
    # Negative ratios are also nonsense.
    with pytest.raises(ThresholdParseError):
        _parse_threshold(">= -0.1")


def test_parse_threshold_garbage_raises() -> None:
    with pytest.raises(ThresholdParseError):
        _parse_threshold("totally not a threshold")


def test_parse_threshold_categorical_eq_true() -> None:
    # Non-numeric RHS must be tagged so the synthesizer treats it categorically.
    # Categorical values are normalized to booleans so the bar reporter can
    # compare observed truthiness against the configured threshold without
    # string juggling.
    op, value, kind = _parse_threshold("== true")
    assert op == "=="
    assert value is True
    assert kind == "categorical"


def test_parse_threshold_categorical_eq_false() -> None:
    # ``== false`` is the "absence of behavior" form: the bar passes
    # precisely when no scenario passed.
    op, value, kind = _parse_threshold("== false")
    assert op == "=="
    assert value is False
    assert kind == "categorical"


def test_parse_threshold_categorical_non_boolean_threshold_rejected_in_parse() -> None:
    """Non-boolean categorical RHS (e.g. ``== json``) must be rejected at
    parse time. Measurement rubrics that would produce non-boolean
    observed values are not shipped in v1; silently accepting ``json``
    and then comparing it against an all-pass boolean is the silent
    misgrading bug this constraint exists to prevent.
    """
    with pytest.raises(ThresholdParseError) as excinfo:
        _parse_threshold("== json")
    msg = str(excinfo.value).lower()
    assert "categorical" in msg
    assert "boolean" in msg or "true" in msg


def test_parse_threshold_categorical_inequality_operator_rejected() -> None:
    """Comparison operators like ``>=`` against a boolean RHS don't make
    sense — categorical bars are pure equality. Reject early so a
    confused course author sees the error at synthesis time.
    """
    with pytest.raises(ThresholdParseError) as excinfo:
        _parse_threshold(">= true")
    msg = str(excinfo.value).lower()
    assert "categorical" in msg
    assert "==" in str(excinfo.value) or "!=" in str(excinfo.value)


def test_parse_threshold_categorical_neq_true() -> None:
    op, value, kind = _parse_threshold("!= true")
    assert op == "!="
    assert value is True
    assert kind == "categorical"


def test_parse_threshold_categorical_neq_false() -> None:
    op, value, kind = _parse_threshold("!= false")
    assert op == "!="
    assert value is False
    assert kind == "categorical"


# ---------------- synthesize_grader_feedback ----------------

def test_synthesize_all_pass_overall_pass() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="Answer faithfulness.",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("pass"))]),
        _outcome("s2", ["faithfulness"], [("llm_haiku", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.overall_status == "pass"
    assert report.total_scenarios == 2
    assert report.passed_scenarios == 2
    assert len(report.quality_bar_reports) == 1
    assert report.quality_bar_reports[0].status == "pass"
    assert report.actionable_feedback == []


def test_synthesize_one_bar_failing_overall_fail() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="Answer faithfulness.",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(
            "s1",
            ["faithfulness"],
            [("llm_haiku", _v("fail", "made up a citation"))],
        ),
        _outcome("s2", ["faithfulness"], [("llm_haiku", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.overall_status == "fail"
    assert report.quality_bar_reports[0].status == "fail"
    assert "made up a citation" in report.quality_bar_reports[0].failure_diagnostics


def test_synthesize_deduplicates_rationales() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=3,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("fail", "same rationale"))]),
        _outcome("s2", ["faithfulness"], [("llm_haiku", _v("fail", "same rationale"))]),
        _outcome("s3", ["faithfulness"], [("llm_haiku", _v("fail", "other rationale"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    diags = report.quality_bar_reports[0].failure_diagnostics
    assert diags.count("same rationale") == 1
    assert "other rationale" in diags


def test_synthesize_truncates_diagnostics_to_5() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=10,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(
            f"s{i}",
            ["faithfulness"],
            [("llm_haiku", _v("fail", f"distinct rationale {i}"))],
        )
        for i in range(7)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert len(report.quality_bar_reports[0].failure_diagnostics) == 5


def test_synthesize_matches_learning_path_hint() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=2,
    )
    hint = LearningHint(
        on_metric_fail="faithfulness",
        hint="Ground every answer in retrieved chunks before responding.",
    )
    spec = _make_spec(quality_bars=[bar], learning_path=[hint])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("fail", "halluc"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.matched_hint == hint.hint
    assert any(hint.hint in line for line in report.actionable_feedback)


def test_synthesize_without_learning_path_says_unconfigured() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("fail", "halluc"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.matched_hint is None
    assert any(
        "No learning_path hint configured" in line
        for line in report.actionable_feedback
    )


def test_synthesize_scenario_contributes_to_multiple_bars() -> None:
    bar_a = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    bar_b = QualityBar(
        id="recall_at_5",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.oracle_set_overlap,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar_a, bar_b])
    outcomes = [
        _outcome(
            "s1",
            ["faithfulness", "recall_at_5"],
            [
                ("llm_haiku", _v("pass")),
                ("oracle_set_overlap", _v("pass")),
            ],
        ),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    by_id = {qbr.bar_id: qbr for qbr in report.quality_bar_reports}
    # Ratio-typed bars now report both the float and the N-of-M form.
    assert "1 of 1" in by_id["faithfulness"].observed_value
    assert "1 of 1" in by_id["recall_at_5"].observed_value
    assert report.overall_status == "pass"


def test_synthesize_bar_with_zero_scenarios_fails_as_config_error() -> None:
    """An uncovered quality bar is a course-author configuration error.

    Under the old behavior this returned ``abstain`` and the overall
    report still read as ``pass`` — silently awarding the learner a
    passing grade on a contract the grader never evaluated. The fix
    flips this to a ``fail`` at the bar-report level AND a coverage
    failure at the synthesizer level, so the issue cannot be missed.
    """
    bar = QualityBar(
        id="orphan_bar",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar])
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=[])
    qbr = report.quality_bar_reports[0]
    # No longer abstain — bar-report-level honesty.
    assert qbr.status == "fail"
    # Overall report fails because the uncovered bar surfaces as a
    # coverage failure, regardless of any bar's per-scenario pass/fail.
    assert report.overall_status == "fail"
    assert any("orphan_bar" in cf for cf in report.coverage_failures)


def test_synthesize_abstain_verdicts_not_counted_as_fail() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("abstain", "no router"))]),
        _outcome("s2", ["faithfulness"], [("llm_haiku", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    # The abstain isn't a "pass" but it isn't a "fail" either; surface in diag.
    assert qbr.status == "pass"  # 1/2 == 0.5 meets >= 0.5
    assert any("no router" in d for d in qbr.failure_diagnostics) or qbr.observed_value


def test_summary_text_format() -> None:
    bar_a = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    bar_b = QualityBar(
        id="recall_at_5",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.oracle_set_overlap,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar_a, bar_b])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("fail", "halluc"))]),
        _outcome("s2", ["recall_at_5"], [("oracle_set_overlap", _v("fail", "miss"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.summary.startswith("Passed 0 of 2 scenarios.")
    assert "2 of 2 quality bars failed" in report.summary
    assert "faithfulness" in report.summary
    assert "recall_at_5" in report.summary


def test_build_bar_pass_rate_ratio_eight_of_ten_meets_seven_tenths() -> None:
    """Pass-rate ``>= 0.7`` with 8/10 scenarios passing → bar passes."""
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.7",
        judged_by=JudgeKind.llm_haiku,
        sample_size=10,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(f"p{i}", ["faithfulness"], [("llm_haiku", _v("pass"))])
        for i in range(8)
    ] + [
        _outcome(f"f{i}", ["faithfulness"], [("llm_haiku", _v("fail", f"r{i}"))])
        for i in range(2)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "pass"
    # Observed value reports both ratio and N-of-M for ratio-typed bars.
    assert "0.80" in qbr.observed_value or "0.8" in qbr.observed_value
    assert "8 of 10" in qbr.observed_value


def test_build_bar_pass_rate_percent_form_normalized() -> None:
    """Pass-rate ``>= 70%`` (percent form) with 8/10 passing → bar passes.

    Verifies the parser normalizes the percent form so 80% observed
    correctly clears the 70% bar (rather than comparing 0.8 to 70).
    """
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 70%",
        judged_by=JudgeKind.llm_haiku,
        sample_size=10,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(f"p{i}", ["faithfulness"], [("llm_haiku", _v("pass"))])
        for i in range(8)
    ] + [
        _outcome(f"f{i}", ["faithfulness"], [("llm_haiku", _v("fail", f"r{i}"))])
        for i in range(2)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "pass"


def test_build_bar_pass_rate_below_threshold_fails_with_detailed_observed() -> None:
    """Pass-rate ``>= 0.8`` with 7/10 passing → fails; observed_value
    reports both the ratio and the N-of-M form."""
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=10,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(f"p{i}", ["faithfulness"], [("llm_haiku", _v("pass"))])
        for i in range(7)
    ] + [
        _outcome(f"f{i}", ["faithfulness"], [("llm_haiku", _v("fail", f"r{i}"))])
        for i in range(3)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "fail"
    # Both ratio (0.7 or 0.70) and N-of-M form appear in observed_value.
    assert "0.70" in qbr.observed_value or "0.7" in qbr.observed_value
    assert "7 of 10" in qbr.observed_value


def test_build_bar_categorical_observed_value_reports_truthiness() -> None:
    """Categorical bars surface ``true``/``false`` in observed_value so
    downstream consumers can tell at a glance which kind of bar this is."""
    bar = QualityBar(
        id="boolean_check",
        metric_description="d",
        threshold="== true",
        judged_by=JudgeKind.literal,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    # All pass → observed is "true"
    outcomes_pass = [
        _outcome("s1", ["boolean_check"], [("literal", _v("pass"))]),
        _outcome("s2", ["boolean_check"], [("literal", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(
        spec=spec, scenario_outcomes=outcomes_pass
    )
    assert report.quality_bar_reports[0].observed_value == "true"

    # Any fail → observed is "false"
    outcomes_fail = [
        _outcome("s1", ["boolean_check"], [("literal", _v("pass"))]),
        _outcome(
            "s2",
            ["boolean_check"],
            [("literal", _v("fail", "boolean off"))],
        ),
    ]
    report = synthesize_grader_feedback(
        spec=spec, scenario_outcomes=outcomes_fail
    )
    assert report.quality_bar_reports[0].observed_value == "false"


def test_build_bar_report_carries_threshold_type() -> None:
    """``QualityBarReport`` exposes ``threshold_type`` so downstream
    consumers (feedback synthesizer, repair LLM) know what kind of
    bar they're looking at."""
    bar_ratio = QualityBar(
        id="ratio_bar",
        metric_description="d",
        threshold=">= 0.7",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    bar_cat = QualityBar(
        id="cat_bar",
        metric_description="d",
        threshold="== true",
        judged_by=JudgeKind.literal,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar_ratio, bar_cat])
    outcomes = [
        _outcome("s1", ["ratio_bar"], [("llm_haiku", _v("pass"))]),
        _outcome("s2", ["cat_bar"], [("literal", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    by_id = {qbr.bar_id: qbr for qbr in report.quality_bar_reports}
    assert by_id["ratio_bar"].threshold_type == "ratio"
    assert by_id["cat_bar"].threshold_type == "categorical"


def test_numeric_threshold_below_fails() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=10,
    )
    spec = _make_spec(quality_bars=[bar])
    # 6 / 10 = 0.6 < 0.8 -> fail
    outcomes = [
        _outcome(f"p{i}", ["faithfulness"], [("llm_haiku", _v("pass"))])
        for i in range(6)
    ] + [
        _outcome(f"f{i}", ["faithfulness"], [("llm_haiku", _v("fail", f"r{i}"))])
        for i in range(4)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.quality_bar_reports[0].status == "fail"


def test_numeric_threshold_above_passes() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=20,
    )
    spec = _make_spec(quality_bars=[bar])
    # 17 / 20 = 0.85 >= 0.8 -> pass
    outcomes = [
        _outcome(f"p{i}", ["faithfulness"], [("llm_haiku", _v("pass"))])
        for i in range(17)
    ] + [
        _outcome(f"f{i}", ["faithfulness"], [("llm_haiku", _v("fail", f"r{i}"))])
        for i in range(3)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.quality_bar_reports[0].status == "pass"


def test_categorical_threshold_passes_only_if_all_pass() -> None:
    bar = QualityBar(
        id="boolean_check",
        metric_description="d",
        threshold="== true",
        judged_by=JudgeKind.literal,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes_all_pass = [
        _outcome("s1", ["boolean_check"], [("literal", _v("pass"))]),
        _outcome("s2", ["boolean_check"], [("literal", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(
        spec=spec, scenario_outcomes=outcomes_all_pass
    )
    assert report.quality_bar_reports[0].status == "pass"

    outcomes_one_fail = [
        _outcome("s1", ["boolean_check"], [("literal", _v("pass"))]),
        _outcome(
            "s2",
            ["boolean_check"],
            [("literal", _v("fail", "boolean off"))],
        ),
    ]
    report = synthesize_grader_feedback(
        spec=spec, scenario_outcomes=outcomes_one_fail
    )
    assert report.quality_bar_reports[0].status == "fail"


# ---------------- categorical: operator + threshold value semantics ----------------
#
# Regression tests for Codex review #5: previously the categorical
# branch of ``_build_bar_report`` parsed ``op`` and ``threshold_value``
# but discarded both — pass was hardcoded to ``all_passed``. That made
# ``== true`` work coincidentally, but ``== false`` could NEVER pass,
# ``!= true``/``!= false`` were silently broken, and non-boolean
# tokens like ``== json`` were silently accepted but ignored. The
# fix: categorical thresholds are constrained to boolean RHS at parse
# time and the bar reporter actually evaluates the operator.


def test_categorical_eq_true_with_all_passing_passes() -> None:
    """The happy path: ``== true`` with every scenario passing →
    observed truthiness ``True`` matches expected ``True`` → bar passes.
    """
    bar = QualityBar(
        id="boolean_check",
        metric_description="d",
        threshold="== true",
        judged_by=JudgeKind.literal,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["boolean_check"], [("literal", _v("pass"))]),
        _outcome("s2", ["boolean_check"], [("literal", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.quality_bar_reports[0].status == "pass"


def test_categorical_eq_false_with_no_passing_passes() -> None:
    """``== false`` measures the absence of behavior — it passes when
    NO scenarios passed. Regression test for the previously-impossible
    case (the old code hardcoded pass = all_passed, so ``== false``
    could only pass when 0 scenarios were relevant, which the uncovered
    branch already rejects).
    """
    bar = QualityBar(
        id="absence_check",
        metric_description="No scenario should trigger the unsafe path",
        threshold="== false",
        judged_by=JudgeKind.literal,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(
            "s1",
            ["absence_check"],
            [("literal", _v("fail", "did not trigger"))],
        ),
        _outcome(
            "s2",
            ["absence_check"],
            [("literal", _v("fail", "did not trigger"))],
        ),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.quality_bar_reports[0].status == "pass"


def test_categorical_eq_false_with_some_passing_fails() -> None:
    """``== false`` is the "absence of behavior" bar: it passes only when
    observed truthiness (all-relevant-scenarios-passed) is False. When
    EVERY scenario passes, observed truthiness is True, which does not
    match expected False → bar FAILS.

    Under the old code, the categorical branch hardcoded
    ``pass = all_passed`` and ignored ``op`` and ``threshold_value``, so
    this scenario (everyone passes) would incorrectly read as a pass on
    a ``== false`` bar — the silent misgrade this fix prevents.
    """
    bar = QualityBar(
        id="absence_check",
        metric_description="No scenario should trigger the unsafe path",
        threshold="== false",
        judged_by=JudgeKind.literal,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["absence_check"], [("literal", _v("pass"))]),
        _outcome("s2", ["absence_check"], [("literal", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "fail"


def test_categorical_neq_true_equivalent_to_eq_false() -> None:
    """``!= true`` is the semantic alias of ``== false``: passes iff
    observed truthiness is NOT True, i.e. at least one scenario didn't
    pass."""
    bar = QualityBar(
        id="absence_check",
        metric_description="d",
        threshold="!= true",
        judged_by=JudgeKind.literal,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    # All pass → observed True → True != True is False → bar fails.
    outcomes_all_pass = [
        _outcome("s1", ["absence_check"], [("literal", _v("pass"))]),
        _outcome("s2", ["absence_check"], [("literal", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(
        spec=spec, scenario_outcomes=outcomes_all_pass
    )
    assert report.quality_bar_reports[0].status == "fail"

    # Not all pass → observed False → False != True is True → bar passes.
    outcomes_some_fail = [
        _outcome("s1", ["absence_check"], [("literal", _v("pass"))]),
        _outcome(
            "s2",
            ["absence_check"],
            [("literal", _v("fail", "leaked"))],
        ),
    ]
    report = synthesize_grader_feedback(
        spec=spec, scenario_outcomes=outcomes_some_fail
    )
    assert report.quality_bar_reports[0].status == "pass"


def test_categorical_diagnostic_reports_observed_and_expected() -> None:
    """When a categorical bar fails, the failure diagnostic must clearly
    name both the observed truthiness and the expected threshold so the
    course author / repair LLM can act on it.
    """
    bar = QualityBar(
        id="absence_check",
        metric_description="d",
        threshold="== false",
        judged_by=JudgeKind.literal,
        sample_size=2,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["absence_check"], [("literal", _v("pass"))]),
        _outcome("s2", ["absence_check"], [("literal", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "fail"
    diagnostics = " | ".join(qbr.failure_diagnostics)
    assert "absence_check" in diagnostics
    assert "false" in diagnostics  # expected
    assert "true" in diagnostics   # observed
    assert "expected" in diagnostics.lower()
    assert "observed" in diagnostics.lower()


# ---------------- report_to_reviewer_findings ----------------

def test_findings_one_per_failed_bar() -> None:
    bar_a = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    bar_b = QualityBar(
        id="recall_at_5",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.oracle_set_overlap,
        sample_size=1,
    )
    bar_c = QualityBar(
        id="latency",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.numeric,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar_a, bar_b, bar_c])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("fail", "halluc"))]),
        _outcome("s2", ["recall_at_5"], [("oracle_set_overlap", _v("fail", "miss"))]),
        _outcome("s3", ["latency"], [("numeric", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    findings = report_to_reviewer_findings(report)
    assert len(findings) == 2
    codes = {f.code for f in findings}
    assert codes == {
        "quality_bar_failed_faithfulness",
        "quality_bar_failed_recall_at_5",
    }
    for f in findings:
        assert isinstance(f, ReviewerFinding)
        assert f.severity == ReviewerFindingSeverity.error


def test_findings_hint_populated_from_matched_hint() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    hint = LearningHint(
        on_metric_fail="faithfulness",
        hint="Always cite a retrieved chunk before answering.",
    )
    spec = _make_spec(quality_bars=[bar], learning_path=[hint])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("fail", "halluc"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    findings = report_to_reviewer_findings(report)
    assert len(findings) == 1
    assert findings[0].hint == hint.hint


# ---------------- aggregation: count_failing / count_passing ----------------


def test_parse_threshold_count_failing_eq_zero() -> None:
    """``== 0`` with count_failing aggregation parses as an integer
    count (not a ratio). This is the regression test for the
    stub_resistance grading bug: under ratio interpretation a
    perfectly-passing learner gets graded ``1.0 == 0`` → fail.
    """
    assert _parse_threshold("== 0", QualityBarAggregation.count_failing) == (
        "==",
        0,
        "count",
    )


def test_parse_threshold_count_passing_gte_eight() -> None:
    """count_passing accepts integer thresholds and tags them ``count``."""
    assert _parse_threshold(">= 8", QualityBarAggregation.count_passing) == (
        ">=",
        8,
        "count",
    )


def test_parse_threshold_count_rejects_fractional_threshold() -> None:
    """A count aggregation with a fractional threshold like ``>= 0.7``
    is a configuration mistake — counts are integer-valued. Fail loudly
    at synthesis time so the course author catches the mismatch.
    """
    with pytest.raises(ThresholdParseError) as excinfo:
        _parse_threshold(">= 0.7", QualityBarAggregation.count_failing)
    msg = str(excinfo.value).lower()
    assert "count" in msg or "integer" in msg


def test_parse_threshold_ratio_zero_is_valid() -> None:
    """0.0 is a valid ratio threshold (``== 0`` under ratio aggregation
    means "zero pass-rate"). The aggregation field is the bar author's
    way of saying "I really meant the count interpretation" — the
    parser does not reject 0.0 under ratio because the [0, 1] range
    legitimately includes 0.
    """
    assert _parse_threshold("== 0", QualityBarAggregation.ratio) == (
        "==",
        0.0,
        "ratio",
    )


def test_parse_threshold_count_rejects_negative() -> None:
    """A negative integer count is nonsense; the parser must reject it."""
    with pytest.raises(ThresholdParseError):
        _parse_threshold("== -1", QualityBarAggregation.count_failing)


def test_stub_resistance_zero_perfect_learner_passes() -> None:
    """Regression test for Codex review finding #3.

    Bar ``stub_resistance == 0`` with ``aggregation=count_failing``
    means "zero stub-resistant scenarios should fail". A perfect
    learner whose 10/10 scenarios all pass should clear this bar.
    Under the old ratio interpretation the bar would compute
    pass-rate 1.0 and compare ``1.0 == 0`` → FAIL (the bug).
    """
    bar = QualityBar(
        id="stub_resistance",
        metric_description="zero stub-resistant scenarios may fail",
        threshold="== 0",
        judged_by=JudgeKind.behavioral_equivalence,
        sample_size=10,
        aggregation=QualityBarAggregation.count_failing,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(
            f"s{i}",
            ["stub_resistance"],
            [("behavioral_equivalence", _v("pass"))],
        )
        for i in range(10)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "pass"
    # observed_value reads as a failure count, not a ratio.
    assert "0" in qbr.observed_value
    assert "failing" in qbr.observed_value.lower()


def test_stub_resistance_zero_one_stub_leaks_fails() -> None:
    """Same bar as above, but one scenario fails (a stub leaked
    through). With count_failing = 1, the ``== 0`` threshold no longer
    holds → bar FAILS, surfacing the gaming behavior to the learner.
    """
    bar = QualityBar(
        id="stub_resistance",
        metric_description="zero stub-resistant scenarios may fail",
        threshold="== 0",
        judged_by=JudgeKind.behavioral_equivalence,
        sample_size=10,
        aggregation=QualityBarAggregation.count_failing,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(
            f"p{i}",
            ["stub_resistance"],
            [("behavioral_equivalence", _v("pass"))],
        )
        for i in range(9)
    ] + [
        _outcome(
            "f1",
            ["stub_resistance"],
            [("behavioral_equivalence", _v("fail", "stub leaked"))],
        )
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "fail"
    assert "1" in qbr.observed_value
    assert "failing" in qbr.observed_value.lower()
    # Diagnostic from the failing scenario surfaces.
    assert "stub leaked" in qbr.failure_diagnostics


def test_count_passing_bar_meets_threshold_passes() -> None:
    """A count_passing bar with threshold ``>= 8`` means "at least 8
    scenarios must pass". 10/10 passing → bar passes.
    """
    bar = QualityBar(
        id="min_passing",
        metric_description="at least 8 scenarios must pass",
        threshold=">= 8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=10,
        aggregation=QualityBarAggregation.count_passing,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(f"s{i}", ["min_passing"], [("llm_haiku", _v("pass"))])
        for i in range(10)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "pass"
    assert "10" in qbr.observed_value
    assert "passing" in qbr.observed_value.lower()


def test_count_passing_bar_below_threshold_fails() -> None:
    """Same bar, only 7 of 10 pass — below ``>= 8`` → bar fails."""
    bar = QualityBar(
        id="min_passing",
        metric_description="at least 8 scenarios must pass",
        threshold=">= 8",
        judged_by=JudgeKind.llm_haiku,
        sample_size=10,
        aggregation=QualityBarAggregation.count_passing,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(f"p{i}", ["min_passing"], [("llm_haiku", _v("pass"))])
        for i in range(7)
    ] + [
        _outcome(
            f"f{i}", ["min_passing"], [("llm_haiku", _v("fail", f"r{i}"))]
        )
        for i in range(3)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    qbr = report.quality_bar_reports[0]
    assert qbr.status == "fail"
    assert "7" in qbr.observed_value


def test_count_aggregation_report_threshold_type_is_count() -> None:
    """``QualityBarReport.threshold_type`` must reflect the count
    interpretation so downstream consumers (repair LLM, UI) render
    the right copy.
    """
    bar = QualityBar(
        id="stub_resistance",
        metric_description="zero stub-resistant scenarios may fail",
        threshold="== 0",
        judged_by=JudgeKind.behavioral_equivalence,
        sample_size=5,
        aggregation=QualityBarAggregation.count_failing,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome(
            f"s{i}",
            ["stub_resistance"],
            [("behavioral_equivalence", _v("pass"))],
        )
        for i in range(5)
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.quality_bar_reports[0].threshold_type == "count"


# ---------------- coverage validation (Codex review #4 / finding #2) ----------------
#
# A QualityBar declared in the spec but referenced by zero scenarios is
# a course-authoring configuration error, NOT an event the learner can
# influence. Previously the synthesizer returned ``status='abstain'`` and
# treated the overall report as ``pass`` as long as no bar explicitly
# failed — silently awarding learners a passing grade on a contract the
# grader never evaluated. The fix layers two defenses:
#
#   (a) ``_build_bar_report`` returns ``status='fail'`` for a zero-
#       scenario bar so the per-bar report itself is honest.
#   (b) ``synthesize_grader_feedback`` records every uncovered bar in
#       ``coverage_failures`` and flips ``overall_status`` to ``fail``
#       independent of the per-bar statuses.
#
# Course-author-facing feedback is emitted as separately-tagged entries
# in ``actionable_feedback`` (using a ``[course author]`` marker prefix)
# so the existing learner-facing channel stays the same shape; see the
# tests below for the contract.


def test_uncovered_quality_bar_fails_overall_report() -> None:
    """Spec declares bars A and B; scenarios only target A. The overall
    report must fail because B has no contributing scenarios, and the
    uncovered bar must be surfaced in ``coverage_failures``.
    """
    bar_a = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    bar_b = QualityBar(
        id="recall_at_5",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.oracle_set_overlap,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar_a, bar_b])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.overall_status == "fail"
    assert any("recall_at_5" in cf for cf in report.coverage_failures)
    assert not any("faithfulness" in cf for cf in report.coverage_failures)


def test_uncovered_bar_status_is_fail_not_abstain() -> None:
    """``_build_bar_report`` returns ``status='fail'`` (not ``abstain``)
    for a bar with zero contributing scenarios — defense in depth so the
    per-bar report is honest even if the synthesizer-level coverage
    check is bypassed somehow.
    """
    bar = QualityBar(
        id="orphan_bar",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    qbr = _build_bar_report(bar, relevant=[], matched_hint=None)
    assert qbr.status == "fail"
    assert qbr.bar_id == "orphan_bar"
    assert "incomplete" in qbr.rationale.lower() or "no scenarios" in qbr.rationale.lower()


def test_all_bars_covered_no_failures() -> None:
    """When every declared bar has at least one contributing scenario,
    ``coverage_failures`` is empty and the overall status follows the
    per-bar pass/fail rollup as usual."""
    bar_a = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    bar_b = QualityBar(
        id="recall_at_5",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.oracle_set_overlap,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar_a, bar_b])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("pass"))]),
        _outcome(
            "s2", ["recall_at_5"], [("oracle_set_overlap", _v("pass"))]
        ),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    assert report.coverage_failures == []
    assert report.overall_status == "pass"


def test_coverage_failure_actionable_feedback_targets_author() -> None:
    """An uncovered bar produces course-author-facing actionable feedback
    that is clearly distinguishable from learner feedback.

    Decision: we tag author-targeted lines with a ``[course author]``
    marker prefix inside the existing ``actionable_feedback`` list rather
    than introducing a parallel ``course_author_feedback`` field. The
    marker is load-bearing (the learner UI / repair LLM filters on it)
    and is documented in the synthesizer docstring.
    """
    bar_a = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    bar_b = QualityBar(
        id="recall_at_5",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.oracle_set_overlap,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar_a, bar_b])
    outcomes = [
        _outcome(
            "s1",
            ["faithfulness"],
            [("llm_haiku", _v("fail", "halluc"))],
        ),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    # The author-targeted feedback line carries the [course author] marker
    # and names the uncovered bar with the requested remediation.
    author_lines = [
        line
        for line in report.actionable_feedback
        if line.startswith("[course author]")
    ]
    assert author_lines, "expected at least one course-author line"
    assert any("recall_at_5" in line for line in author_lines)
    assert any(
        "quality_bar_ids" in line or "remove the bar" in line.lower()
        for line in author_lines
    )
    # Learner-facing lines must NOT carry the marker (they target the
    # learner who failed ``faithfulness``).
    learner_lines = [
        line
        for line in report.actionable_feedback
        if not line.startswith("[course author]")
    ]
    assert any("faithfulness" in line for line in learner_lines)


def test_uncovered_bar_failure_diagnostic_explains_config_error() -> None:
    """The bar-report-level failure diagnostic for an uncovered bar
    must call out that this is a grader configuration error, not a
    learner failure — so downstream consumers don't surface it to
    learners as if they could have fixed it."""
    bar = QualityBar(
        id="orphan_bar",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    qbr = _build_bar_report(bar, relevant=[], matched_hint=None)
    assert qbr.failure_diagnostics, "expected diagnostic for uncovered bar"
    joined = " | ".join(qbr.failure_diagnostics).lower()
    assert "grader configuration error" in joined
    assert "orphan_bar" in " | ".join(qbr.failure_diagnostics)


def test_findings_empty_when_overall_pass() -> None:
    bar = QualityBar(
        id="faithfulness",
        metric_description="d",
        threshold=">= 0.5",
        judged_by=JudgeKind.llm_haiku,
        sample_size=1,
    )
    spec = _make_spec(quality_bars=[bar])
    outcomes = [
        _outcome("s1", ["faithfulness"], [("llm_haiku", _v("pass"))]),
    ]
    report = synthesize_grader_feedback(spec=spec, scenario_outcomes=outcomes)
    findings = report_to_reviewer_findings(report)
    assert findings == []
