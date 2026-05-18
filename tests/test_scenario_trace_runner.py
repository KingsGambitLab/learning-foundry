"""Tests for the scenario trace runner.

The trace runner is responsible for:
1. Resolving ``${id.dotted.path}`` placeholders against captured step responses.
2. Walking ``setup`` then ``trace`` steps, issuing HTTP requests via an injected
   client, capturing responses, and applying per-step ``expect`` assertions.
3. Running each scenario's rubrics against the resulting ``RubricContext`` and
   aggregating verdicts into a ``ScenarioVerdictReport``.

These tests use a ``FakeHttpClient`` with canned responses keyed by
``(method, url)`` — no real network — so the runner's behaviour is fully
deterministic and inspectable.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.services.scenario_loader import (
    HttpExpectation,
    RubricSpec,
    Scenario,
    TraceStep,
)
from app.services.scenario_rubrics_base import (
    Rubric,
    RubricContext,
    Verdict,
    register_rubric,
)
from app.services.scenario_trace_runner import (
    InterpolationError,
    ScenarioHttpClient,
    ScenarioRunResult,
    ScenarioVerdictReport,
    TraceStepResult,
    UrllibHttpClient,
    interpolate,
    run_scenario,
)


# ---------------- FakeHttpClient ----------------


class FakeHttpClient:
    """Canned-response HTTP client keyed by ``(method, url)``.

    Each value is a 3-tuple ``(status, headers, body)`` matching the
    ``ScenarioHttpClient.request`` return contract.
    """

    def __init__(self, responses: dict[tuple[str, str], tuple[int, dict[str, str], Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any | None,
        follow_redirects: bool,
        timeout: float,
    ) -> tuple[int, dict[str, str], Any]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body,
                "follow_redirects": follow_redirects,
                "timeout": timeout,
            }
        )
        key = (method.upper(), url)
        if key not in self.responses:
            raise AssertionError(f"FakeHttpClient: no canned response for {key}")
        return self.responses[key]


# ---------------- interpolate ----------------


def test_interpolate_single_placeholder_body_shorthand() -> None:
    captures = {"created": {"status": 201, "headers": {}, "body": {"short_code": "abc123"}}}
    assert interpolate("/links/${created.short_code}", captures) == "/links/abc123"


def test_interpolate_explicit_body_prefix() -> None:
    captures = {"created": {"status": 201, "headers": {}, "body": {"short_code": "abc123"}}}
    assert interpolate("/links/${created.body.short_code}", captures) == "/links/abc123"


def test_interpolate_status_no_prefix() -> None:
    captures = {"step1": {"status": 200, "headers": {}, "body": {}}}
    assert interpolate("status=${step1.status}", captures) == "status=200"


def test_interpolate_headers_no_prefix() -> None:
    captures = {
        "step1": {"status": 200, "headers": {"location": "/x"}, "body": None}
    }
    assert interpolate("loc=${step1.headers.location}", captures) == "loc=/x"


def test_interpolate_multiple_placeholders() -> None:
    captures = {
        "a": {"status": 200, "headers": {}, "body": {"x": "alpha"}},
        "b": {"status": 200, "headers": {}, "body": {"y": "beta"}},
    }
    assert (
        interpolate("/p/${a.x}/q/${b.y}", captures) == "/p/alpha/q/beta"
    )


def test_interpolate_raises_on_missing_capture() -> None:
    captures = {"created": {"status": 201, "headers": {}, "body": {}}}
    with pytest.raises(InterpolationError):
        interpolate("/links/${created.short_code}", captures)


def test_interpolate_raises_on_unknown_capture_id() -> None:
    with pytest.raises(InterpolationError):
        interpolate("/x/${nope.field}", {})


# ---------------- Pydantic models ----------------


def test_trace_step_result_construction() -> None:
    result = TraceStepResult(
        step_id="s1",
        status=200,
        headers={"content-type": "application/json"},
        body={"ok": True},
        latency_ms=12.5,
        expect_passed=True,
        expect_diagnostic=None,
    )
    assert result.step_id == "s1"
    assert result.status == 200
    assert result.headers == {"content-type": "application/json"}
    assert result.body == {"ok": True}
    assert result.latency_ms == 12.5
    assert result.expect_passed is True
    assert result.expect_diagnostic is None


def test_scenario_run_result_construction() -> None:
    step = TraceStepResult(
        step_id="s1",
        status=200,
        headers={},
        body=None,
        latency_ms=0.0,
        expect_passed=True,
        expect_diagnostic=None,
    )
    run = ScenarioRunResult(
        scenario_id="sc1",
        setup_results=[],
        trace_results=[step],
        captures={"s1": {"status": 200, "headers": {}, "body": None}},
        aborted=False,
        abort_reason=None,
    )
    assert run.scenario_id == "sc1"
    assert run.trace_results[0].step_id == "s1"
    assert run.aborted is False


def test_scenario_verdict_report_overall_status_all_pass() -> None:
    run = ScenarioRunResult(
        scenario_id="sc1",
        setup_results=[],
        trace_results=[],
        captures={},
        aborted=False,
        abort_reason=None,
    )
    report = ScenarioVerdictReport(
        scenario_id="sc1",
        category="happy_path",
        run_result=run,
        verdicts=[
            ("schema_match", Verdict(status="pass", rationale="ok")),
            ("literal_match", Verdict(status="pass", rationale="ok")),
        ],
    )
    assert report.overall_status == "pass"


def test_scenario_verdict_report_overall_status_any_fail() -> None:
    run = ScenarioRunResult(
        scenario_id="sc1",
        setup_results=[],
        trace_results=[],
        captures={},
        aborted=False,
        abort_reason=None,
    )
    report = ScenarioVerdictReport(
        scenario_id="sc1",
        category="happy_path",
        run_result=run,
        verdicts=[
            ("schema_match", Verdict(status="pass", rationale="ok")),
            ("literal_match", Verdict(status="fail", rationale="nope")),
        ],
    )
    assert report.overall_status == "fail"


def test_scenario_verdict_report_overall_status_abstain() -> None:
    run = ScenarioRunResult(
        scenario_id="sc1",
        setup_results=[],
        trace_results=[],
        captures={},
        aborted=False,
        abort_reason=None,
    )
    report = ScenarioVerdictReport(
        scenario_id="sc1",
        category="happy_path",
        run_result=run,
        verdicts=[
            ("schema_match", Verdict(status="pass", rationale="ok")),
            ("llm_judge_coverage", Verdict(status="abstain", rationale="no router")),
        ],
    )
    assert report.overall_status == "abstain"


# ---------------- run_scenario ----------------


def _scenario_single_step() -> Scenario:
    return Scenario(
        id="sc-single",
        description="single step",
        category="happy_path",
        trace=[TraceStep(id="ping", method="GET", path="/ping")],
        rubrics=[
            RubricSpec(
                kind="schema_match",
                config={"target": "ping.body", "must_have_fields": ["ok"]},
            )
        ],
    )


def test_run_scenario_single_step_captures_response() -> None:
    scenario = _scenario_single_step()
    client = FakeHttpClient(
        {("GET", "http://api/ping"): (200, {"content-type": "application/json"}, {"ok": True})}
    )
    report = run_scenario(
        scenario=scenario,
        base_url="http://api",
        http_client=client,
    )
    assert report.scenario_id == "sc-single"
    assert report.run_result.captures["ping"]["status"] == 200
    assert report.run_result.captures["ping"]["body"] == {"ok": True}
    assert report.overall_status == "pass"


def test_run_scenario_multi_step_with_variable_substitution() -> None:
    scenario = Scenario(
        id="sc-multi",
        description="create then read",
        category="happy_path",
        trace=[
            TraceStep(
                id="create",
                method="POST",
                path="/links",
                body={"url": "https://example.com"},
                expect=HttpExpectation(status=201),
            ),
            TraceStep(
                id="read",
                method="GET",
                path="/links/${create.short_code}",
                expect=HttpExpectation(status=200),
            ),
        ],
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "read.status", "expected": 200},
            )
        ],
    )
    client = FakeHttpClient(
        {
            ("POST", "http://api/links"): (
                201,
                {"content-type": "application/json"},
                {"short_code": "abc"},
            ),
            ("GET", "http://api/links/abc"): (
                200,
                {"content-type": "application/json"},
                {"url": "https://example.com"},
            ),
        }
    )
    report = run_scenario(
        scenario=scenario, base_url="http://api", http_client=client
    )
    assert report.overall_status == "pass"
    assert len(report.run_result.trace_results) == 2
    assert report.run_result.captures["create"]["body"]["short_code"] == "abc"
    assert report.run_result.captures["read"]["status"] == 200
    # second call's URL was interpolated
    assert client.calls[1]["url"] == "http://api/links/abc"


def test_run_scenario_expect_passing() -> None:
    scenario = Scenario(
        id="sc-expect-pass",
        description="x",
        category="happy_path",
        trace=[
            TraceStep(
                id="ping",
                method="GET",
                path="/ping",
                expect=HttpExpectation(status=[200, 201]),
            )
        ],
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "ping.status", "expected": 200},
            )
        ],
    )
    client = FakeHttpClient({("GET", "http://api/ping"): (200, {}, None)})
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert report.run_result.trace_results[0].expect_passed is True
    assert report.run_result.aborted is False


def test_run_scenario_expect_failing_aborts() -> None:
    scenario = Scenario(
        id="sc-expect-fail",
        description="x",
        category="happy_path",
        trace=[
            TraceStep(
                id="s1",
                method="GET",
                path="/s1",
                expect=HttpExpectation(status=200),
            ),
            TraceStep(id="s2", method="GET", path="/s2"),
        ],
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "s1.status", "expected": 200},
            )
        ],
    )
    client = FakeHttpClient(
        {
            ("GET", "http://api/s1"): (500, {}, None),
            ("GET", "http://api/s2"): (200, {}, None),
        }
    )
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert report.run_result.aborted is True
    assert report.run_result.abort_reason is not None
    assert "s1" in report.run_result.abort_reason
    # second step was skipped → only one trace_result
    assert len(report.run_result.trace_results) == 1
    assert report.run_result.trace_results[0].expect_passed is False
    # s2 was never called
    assert len(client.calls) == 1


def test_run_scenario_with_setup_steps() -> None:
    scenario = Scenario(
        id="sc-setup",
        description="x",
        category="happy_path",
        setup=[TraceStep(id="seed", method="POST", path="/seed", body={"foo": "bar"})],
        trace=[TraceStep(id="ping", method="GET", path="/ping")],
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "ping.status", "expected": 200},
            )
        ],
    )
    client = FakeHttpClient(
        {
            ("POST", "http://api/seed"): (201, {}, {"seeded": True}),
            ("GET", "http://api/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
        }
    )
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert len(report.run_result.setup_results) == 1
    assert report.run_result.setup_results[0].step_id == "seed"
    assert report.run_result.captures["seed"]["status"] == 201


def test_run_scenario_rubric_pass() -> None:
    scenario = _scenario_single_step()
    client = FakeHttpClient(
        {("GET", "http://api/ping"): (200, {"content-type": "application/json"}, {"ok": True})}
    )
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert report.overall_status == "pass"
    assert report.verdicts[0][0] == "schema_match"
    assert report.verdicts[0][1].status == "pass"


def test_run_scenario_rubric_fail_makes_overall_fail() -> None:
    scenario = Scenario(
        id="sc-rubric-fail",
        description="x",
        category="happy_path",
        trace=[TraceStep(id="ping", method="GET", path="/ping")],
        rubrics=[
            RubricSpec(
                kind="schema_match",
                config={"target": "ping.body", "must_have_fields": ["missing_field"]},
            )
        ],
    )
    client = FakeHttpClient(
        {("GET", "http://api/ping"): (200, {"content-type": "application/json"}, {"ok": True})}
    )
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert report.overall_status == "fail"
    assert report.verdicts[0][1].status == "fail"


def test_run_scenario_rubric_abstain_makes_overall_abstain() -> None:
    scenario = Scenario(
        id="sc-rubric-abstain",
        description="x",
        category="happy_path",
        trace=[TraceStep(id="ping", method="GET", path="/ping")],
        rubrics=[
            RubricSpec(
                kind="llm_judge_coverage",
                config={
                    "target": "ping.body.answer",
                    "must_contain_facts": ["fact1"],
                },
            )
        ],
    )
    client = FakeHttpClient(
        {
            ("GET", "http://api/ping"): (
                200,
                {"content-type": "application/json"},
                {"answer": "anything"},
            )
        }
    )
    # No router → llm_judge_coverage abstains.
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert report.verdicts[0][1].status == "abstain"
    assert report.overall_status == "abstain"


def test_run_scenario_passes_router_to_llm_rubric() -> None:
    """When a rubric class accepts ``router`` in its __init__, the runner
    must inject the runner's router into its kwargs."""
    received: dict[str, Any] = {}

    class FakeRouter:
        def parse_structured(self, **kwargs: Any) -> Any:
            received["called"] = True
            # Return something that fails the parsed-shape check so the rubric abstains,
            # but at minimum we've shown the router was passed.
            return type("R", (), {"parsed": None, "usage_summary": None})()

    router = FakeRouter()
    scenario = Scenario(
        id="sc-router",
        description="x",
        category="happy_path",
        trace=[TraceStep(id="ping", method="GET", path="/ping")],
        rubrics=[
            RubricSpec(
                kind="llm_judge_coverage",
                config={"target": "ping.body.answer", "must_contain_facts": ["a"]},
            )
        ],
    )
    client = FakeHttpClient(
        {
            ("GET", "http://api/ping"): (
                200,
                {"content-type": "application/json"},
                {"answer": "x"},
            )
        }
    )
    report = run_scenario(
        scenario=scenario,
        base_url="http://api",
        router=router,
        http_client=client,
    )
    # Router was injected → parse_structured was attempted on it.
    assert received.get("called") is True
    # We don't care about the verdict value for this test — only the wiring.
    assert report.verdicts[0][0] == "llm_judge_coverage"


def test_run_scenario_does_not_pass_router_to_non_llm_rubric() -> None:
    """Rubrics whose __init__ does not accept ``router`` must NOT receive it.

    We use a stub rubric that explodes on any unexpected kwarg.
    """
    from app.services.scenario_rubrics_base import RUBRIC_REGISTRY

    init_kwargs: dict[str, Any] = {}

    class StrictNoRouterRubric(Rubric):
        name = "strict_no_router_test_rubric"

        def __init__(self, target: str) -> None:  # noqa: D401
            init_kwargs["target"] = target

        def judge(self, ctx: RubricContext) -> Verdict:  # noqa: D401
            return Verdict(status="pass", rationale="ok")

    # Register manually, bypassing the dup-check guard.
    RUBRIC_REGISTRY["strict_no_router_test_rubric"] = StrictNoRouterRubric
    try:
        scenario = Scenario(
            id="sc-no-router",
            description="x",
            category="happy_path",
            trace=[TraceStep(id="ping", method="GET", path="/ping")],
            rubrics=[
                RubricSpec(
                    kind="strict_no_router_test_rubric",
                    config={"target": "ping.body"},
                )
            ],
        )
        client = FakeHttpClient({("GET", "http://api/ping"): (200, {}, None)})

        class FakeRouter:
            pass

        report = run_scenario(
            scenario=scenario,
            base_url="http://api",
            router=FakeRouter(),
            http_client=client,
        )
        assert init_kwargs == {"target": "ping.body"}
        assert report.verdicts[0][1].status == "pass"
    finally:
        del RUBRIC_REGISTRY["strict_no_router_test_rubric"]


def test_run_scenario_handles_json_text_and_empty_bodies() -> None:
    scenario = Scenario(
        id="sc-bodies",
        description="x",
        category="happy_path",
        trace=[
            TraceStep(id="j", method="GET", path="/json"),
            TraceStep(id="t", method="GET", path="/text"),
            TraceStep(id="e", method="GET", path="/empty"),
        ],
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "j.body.ok", "expected": True},
            )
        ],
    )
    client = FakeHttpClient(
        {
            ("GET", "http://api/json"): (
                200,
                {"content-type": "application/json"},
                {"ok": True},
            ),
            ("GET", "http://api/text"): (
                200,
                {"content-type": "text/plain"},
                "hello",
            ),
            ("GET", "http://api/empty"): (204, {}, None),
        }
    )
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert report.run_result.captures["j"]["body"] == {"ok": True}
    assert report.run_result.captures["t"]["body"] == "hello"
    assert report.run_result.captures["e"]["body"] is None


def test_run_scenario_follow_redirects_false_captures_location() -> None:
    scenario = Scenario(
        id="sc-redirect",
        description="x",
        category="happy_path",
        trace=[
            TraceStep(
                id="r",
                method="GET",
                path="/r",
                follow_redirects=False,
                expect=HttpExpectation(status=302),
            )
        ],
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "r.status", "expected": 302},
            )
        ],
    )
    client = FakeHttpClient(
        {("GET", "http://api/r"): (302, {"location": "/elsewhere"}, None)}
    )
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    assert report.run_result.captures["r"]["status"] == 302
    assert report.run_result.captures["r"]["headers"].get("location") == "/elsewhere"
    # The HTTP client got the follow_redirects=False flag.
    assert client.calls[0]["follow_redirects"] is False


def test_run_scenario_interpolates_in_body_leaves() -> None:
    scenario = Scenario(
        id="sc-body-interp",
        description="x",
        category="happy_path",
        trace=[
            TraceStep(id="a", method="POST", path="/a", body={"v": "x"}),
            TraceStep(
                id="b",
                method="POST",
                path="/b",
                body={"ref": "${a.v}"},
            ),
        ],
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "b.status", "expected": 200},
            )
        ],
    )
    client = FakeHttpClient(
        {
            ("POST", "http://api/a"): (
                200,
                {"content-type": "application/json"},
                {"v": "ALPHA"},
            ),
            ("POST", "http://api/b"): (200, {}, None),
        }
    )
    report = run_scenario(scenario=scenario, base_url="http://api", http_client=client)
    # The second call's body had its placeholder resolved.
    assert client.calls[1]["body"] == {"ref": "ALPHA"}
    assert report.overall_status == "pass"


def test_urllib_http_client_is_a_scenario_http_client() -> None:
    """Sanity check that UrllibHttpClient satisfies the Protocol surface."""
    c: ScenarioHttpClient = UrllibHttpClient()  # noqa: F841 — assignment is the assertion.


# ---------------- Rubric-exception defensive wrapper ----------------
#
# Codex review #6 finding #4 part B: a rubric that raises out of
# ``judge()`` (rubric library bug, unexpected learner payload that
# slipped past per-rubric validation, etc.) must NOT cascade into a
# total scenario crash. The runner catches the exception, converts it
# to a FAIL verdict with the exception type/message in the diagnostic,
# logs a WARNING, and continues running remaining rubrics.


def test_run_scenario_catches_rubric_exception() -> None:
    """A rubric that raises during ``judge()`` must produce a FAIL
    verdict with diagnostic instead of crashing the runner. Other
    rubrics in the scenario must still run."""
    from app.services.scenario_rubrics_base import RUBRIC_REGISTRY

    class FaultyRubric(Rubric):
        name = "faulty_test_rubric"

        def __init__(self) -> None:  # noqa: D401
            pass

        def judge(self, ctx: RubricContext) -> Verdict:  # noqa: D401
            raise RuntimeError("synthetic rubric failure")

    RUBRIC_REGISTRY["faulty_test_rubric"] = FaultyRubric
    try:
        scenario = Scenario(
            id="sc-faulty",
            description="x",
            category="happy_path",
            trace=[TraceStep(id="ping", method="GET", path="/ping")],
            rubrics=[
                RubricSpec(kind="faulty_test_rubric", config={}),
                RubricSpec(
                    kind="literal_match",
                    config={"target": "ping.status", "expected": 200},
                ),
            ],
        )
        client = FakeHttpClient({("GET", "http://api/ping"): (200, {}, None)})
        # The runner must NOT raise.
        report = run_scenario(
            scenario=scenario, base_url="http://api", http_client=client
        )
        # First rubric: converted to fail verdict.
        assert report.verdicts[0][0] == "faulty_test_rubric"
        assert report.verdicts[0][1].status == "fail"
        assert "RuntimeError" in report.verdicts[0][1].rationale
        assert "synthetic rubric failure" in report.verdicts[0][1].rationale
        assert report.verdicts[0][1].diagnostic.get("exception_type") == "RuntimeError"
        assert (
            report.verdicts[0][1].diagnostic.get("exception_message")
            == "synthetic rubric failure"
        )
        # Second rubric still ran and produced its own verdict.
        assert report.verdicts[1][0] == "literal_match"
        assert report.verdicts[1][1].status == "pass"
    finally:
        del RUBRIC_REGISTRY["faulty_test_rubric"]


def test_run_scenario_logs_rubric_exception_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a rubric raises, the runner emits a WARNING-level log so
    the issue surfaces in operational logs."""
    import logging

    from app.services.scenario_rubrics_base import RUBRIC_REGISTRY

    class FaultyRubric(Rubric):
        name = "faulty_log_test_rubric"

        def __init__(self) -> None:  # noqa: D401
            pass

        def judge(self, ctx: RubricContext) -> Verdict:  # noqa: D401
            raise ValueError("kaboom")

    RUBRIC_REGISTRY["faulty_log_test_rubric"] = FaultyRubric
    try:
        scenario = Scenario(
            id="sc-faulty-log",
            description="x",
            category="happy_path",
            trace=[TraceStep(id="ping", method="GET", path="/ping")],
            rubrics=[RubricSpec(kind="faulty_log_test_rubric", config={})],
        )
        client = FakeHttpClient({("GET", "http://api/ping"): (200, {}, None)})
        with caplog.at_level(logging.WARNING):
            run_scenario(
                scenario=scenario, base_url="http://api", http_client=client
            )
        # Some WARNING-level record mentions the rubric kind and the exception.
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "expected at least one WARNING log"
        joined = " ".join(r.getMessage() for r in warning_records)
        assert "faulty_log_test_rubric" in joined
        assert "ValueError" in joined or "kaboom" in joined
    finally:
        del RUBRIC_REGISTRY["faulty_log_test_rubric"]


def test_run_scenario_continues_after_one_rubric_crash() -> None:
    """3 rubrics, the middle one raises — the first and third must
    still produce verdicts. The middle one is recorded as fail."""
    from app.services.scenario_rubrics_base import RUBRIC_REGISTRY

    class MiddleFaultyRubric(Rubric):
        name = "middle_faulty_test_rubric"

        def __init__(self) -> None:  # noqa: D401
            pass

        def judge(self, ctx: RubricContext) -> Verdict:  # noqa: D401
            raise KeyError("missing thing")

    RUBRIC_REGISTRY["middle_faulty_test_rubric"] = MiddleFaultyRubric
    try:
        scenario = Scenario(
            id="sc-three",
            description="x",
            category="happy_path",
            trace=[TraceStep(id="ping", method="GET", path="/ping")],
            rubrics=[
                RubricSpec(
                    kind="literal_match",
                    config={"target": "ping.status", "expected": 200},
                ),
                RubricSpec(kind="middle_faulty_test_rubric", config={}),
                RubricSpec(
                    kind="literal_match",
                    config={"target": "ping.status", "expected": 200},
                ),
            ],
        )
        client = FakeHttpClient({("GET", "http://api/ping"): (200, {}, None)})
        report = run_scenario(
            scenario=scenario, base_url="http://api", http_client=client
        )
        assert len(report.verdicts) == 3
        kinds = [k for k, _ in report.verdicts]
        assert kinds == [
            "literal_match",
            "middle_faulty_test_rubric",
            "literal_match",
        ]
        assert report.verdicts[0][1].status == "pass"
        assert report.verdicts[1][1].status == "fail"
        assert report.verdicts[1][1].diagnostic.get("exception_type") == "KeyError"
        assert report.verdicts[2][1].status == "pass"
        # One fail in the middle → overall fail.
        assert report.overall_status == "fail"
    finally:
        del RUBRIC_REGISTRY["middle_faulty_test_rubric"]


def test_run_scenario_setup_data_and_course_meta_reach_rubric() -> None:
    """The runner must build a RubricContext that includes setup_data
    and course_meta as the rubric sees them."""
    from app.services.scenario_rubrics_base import RUBRIC_REGISTRY

    seen: dict[str, Any] = {}

    class CapturingRubric(Rubric):
        name = "capturing_ctx_test_rubric"

        def __init__(self) -> None:  # noqa: D401
            pass

        def judge(self, ctx: RubricContext) -> Verdict:  # noqa: D401
            seen["captures"] = dict(ctx.captures)
            seen["setup_data"] = dict(ctx.setup_data)
            seen["course_meta"] = dict(ctx.course_meta)
            return Verdict(status="pass", rationale="ok")

    RUBRIC_REGISTRY["capturing_ctx_test_rubric"] = CapturingRubric
    try:
        scenario = Scenario(
            id="sc-ctx",
            description="x",
            category="happy_path",
            trace=[TraceStep(id="ping", method="GET", path="/ping")],
            rubrics=[RubricSpec(kind="capturing_ctx_test_rubric", config={})],
        )
        client = FakeHttpClient({("GET", "http://api/ping"): (200, {}, None)})
        run_scenario(
            scenario=scenario,
            base_url="http://api",
            http_client=client,
            setup_data={"oracle": {"x": 1}},
            course_meta={"entities": ["Link"]},
        )
        assert seen["setup_data"] == {"oracle": {"x": 1}}
        assert seen["course_meta"] == {"entities": ["Link"]}
        assert "ping" in seen["captures"]
    finally:
        del RUBRIC_REGISTRY["capturing_ctx_test_rubric"]
