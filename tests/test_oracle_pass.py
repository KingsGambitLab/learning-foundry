"""Tests for the oracle_pass node.

The oracle_pass boots the reference impl in a sandbox (Docker, faked
here), executes every scenario against it via the trace runner, and
persists the captured outputs as ground truth.

These tests use a ``FakeSandboxRunner`` (no Docker), a
``FakeHttpClient`` keyed by ``(method, url)`` (no network), and rubrics
from the structural / set registry that don't need an LLM router. The
goal: pin every behaviour the orchestration logic owns — hashing,
counts, per-scenario isolation, teardown — without touching the real
infra.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services.scenario_loader import (
    HttpExpectation,
    RubricSpec,
    Scenario,
    TraceStep,
)
from app.services.oracle_pass import (
    OraclePass,
    OraclePassResult,
    OracleScenarioOutput,
    persist_oracle_outputs,
)


# ---------------- Fakes ----------------


class FakeSandboxHandle:
    """Plain object returned from ``FakeSandboxRunner.boot``."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class FakeSandboxRunner:
    """Test-only sandbox: returns a canned ``base_url`` and records calls.

    Mirrors the protocol the real Docker sandbox runner is expected to
    expose for oracle_pass: ``boot(reference_impl_dir) -> handle`` and
    ``teardown(handle)`` (called inside a ``finally``).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:12345") -> None:
        self.base_url = base_url
        self.boot_calls: list[Path] = []
        self.teardown_calls: list[FakeSandboxHandle] = []
        self.raise_on_boot: Exception | None = None

    def boot(self, reference_impl_dir: Path) -> FakeSandboxHandle:
        if self.raise_on_boot is not None:
            raise self.raise_on_boot
        self.boot_calls.append(reference_impl_dir)
        return FakeSandboxHandle(self.base_url)

    def teardown(self, handle: FakeSandboxHandle) -> None:
        self.teardown_calls.append(handle)


class FakeHttpClient:
    """Canned-response HTTP client keyed by ``(method, url)``."""

    def __init__(
        self,
        responses: dict[tuple[str, str], tuple[int, dict[str, str], Any]],
    ) -> None:
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
            {"method": method, "url": url, "headers": dict(headers), "body": body}
        )
        key = (method.upper(), url)
        if key not in self.responses:
            raise AssertionError(f"FakeHttpClient: no canned response for {key}")
        return self.responses[key]


# ---------------- Fixtures / helpers ----------------


def _make_scenario(
    scenario_id: str = "s1",
    rubrics: list[RubricSpec] | None = None,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        description=f"scenario {scenario_id}",
        category="happy_path",
        trace=[
            TraceStep(
                id="get_index",
                method="GET",
                path="/ping",
                expect=HttpExpectation(status=200),
            )
        ],
        rubrics=rubrics
        or [
            RubricSpec(
                kind="literal_match",
                config={"target": "get_index.body.ok", "expected": True},
            )
        ],
    )


def _make_reference_impl(tmp_path: Path, contents: str = "FROM python:3.12\n") -> Path:
    ref_dir = tmp_path / "refimpl"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "Dockerfile").write_text(contents)
    (ref_dir / "app.py").write_text("# ref impl\n")
    return ref_dir


# ---------------- Pydantic types ----------------


def test_oracle_scenario_output_construction() -> None:
    out = OracleScenarioOutput(
        scenario_id="s1",
        category="happy_path",
        captures={"get_index": {"status": 200, "headers": {}, "body": {"ok": True}}},
        verdicts=[("literal_match", {"status": "pass", "rationale": "ok"})],
        aborted=False,
        abort_reason=None,
    )
    assert out.scenario_id == "s1"
    assert out.captures["get_index"]["status"] == 200
    assert out.verdicts[0][0] == "literal_match"
    assert out.aborted is False


def test_oracle_pass_result_construction() -> None:
    result = OraclePassResult(
        reference_impl_hash="abc",
        scenario_set_hash="def",
        generated_at="2026-05-14T00:00:00+00:00",
        scenario_outputs=[],
        total_scenarios=0,
        passed_scenarios=0,
        failed_scenarios=0,
        abstained_scenarios=0,
    )
    assert result.reference_impl_hash == "abc"
    assert result.total_scenarios == 0


# ---------------- Hashing ----------------


def test_hash_determinism_same_inputs_produce_same_hashes(tmp_path: Path) -> None:
    ref_dir = _make_reference_impl(tmp_path)
    scenarios = [_make_scenario("s1"), _make_scenario("s2")]

    runner = OraclePass(sandbox_runner=FakeSandboxRunner(), http_client=FakeHttpClient({
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }))

    result1 = runner.run(scenarios=scenarios, reference_impl_dir=ref_dir)
    result2 = runner.run(scenarios=scenarios, reference_impl_dir=ref_dir)

    assert result1.reference_impl_hash == result2.reference_impl_hash
    assert result1.scenario_set_hash == result2.scenario_set_hash


def test_hash_sensitivity_different_reference_impl_content(tmp_path: Path) -> None:
    ref_a = _make_reference_impl(tmp_path / "a", contents="FROM python:3.12\n")
    ref_b_dir = tmp_path / "b"
    ref_b_dir.mkdir()
    (ref_b_dir / "Dockerfile").write_text("FROM python:3.13\n")  # different
    (ref_b_dir / "app.py").write_text("# ref impl\n")

    scenarios = [_make_scenario("s1")]
    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }

    runner = OraclePass(
        sandbox_runner=FakeSandboxRunner(),
        http_client=FakeHttpClient(responses),
    )
    res_a = runner.run(scenarios=scenarios, reference_impl_dir=ref_a)
    res_b = runner.run(scenarios=scenarios, reference_impl_dir=ref_b_dir)

    assert res_a.reference_impl_hash != res_b.reference_impl_hash


def test_hash_sensitivity_different_scenario_set(tmp_path: Path) -> None:
    ref_dir = _make_reference_impl(tmp_path)
    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }
    runner = OraclePass(
        sandbox_runner=FakeSandboxRunner(),
        http_client=FakeHttpClient(responses),
    )
    res_one = runner.run(scenarios=[_make_scenario("s1")], reference_impl_dir=ref_dir)
    res_two = runner.run(
        scenarios=[_make_scenario("s1"), _make_scenario("s2")],
        reference_impl_dir=ref_dir,
    )
    assert res_one.scenario_set_hash != res_two.scenario_set_hash


# ---------------- Run happy path ----------------


def test_run_happy_path_all_scenarios_pass(tmp_path: Path) -> None:
    ref_dir = _make_reference_impl(tmp_path)
    scenarios = [_make_scenario("s1"), _make_scenario("s2")]
    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }
    sandbox = FakeSandboxRunner()
    runner = OraclePass(sandbox_runner=sandbox, http_client=FakeHttpClient(responses))

    result = runner.run(scenarios=scenarios, reference_impl_dir=ref_dir)

    assert result.total_scenarios == 2
    assert result.passed_scenarios == 2
    assert result.failed_scenarios == 0
    assert result.abstained_scenarios == 0
    assert len(result.scenario_outputs) == 2
    assert {o.scenario_id for o in result.scenario_outputs} == {"s1", "s2"}
    # generated_at is an ISO-8601 string with timezone
    assert "T" in result.generated_at and result.generated_at.endswith("+00:00")
    # sandbox booted once and torn down once
    assert len(sandbox.boot_calls) == 1
    assert len(sandbox.teardown_calls) == 1


def test_run_with_one_failing_scenario_reflected_in_counts(tmp_path: Path) -> None:
    ref_dir = _make_reference_impl(tmp_path)
    passing = _make_scenario("pass1")
    failing = _make_scenario(
        "fail1",
        rubrics=[
            RubricSpec(
                kind="literal_match",
                config={"target": "get_index.body.ok", "expected": "definitely-not-this"},
            )
        ],
    )
    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }
    runner = OraclePass(
        sandbox_runner=FakeSandboxRunner(),
        http_client=FakeHttpClient(responses),
    )
    result = runner.run(scenarios=[passing, failing], reference_impl_dir=ref_dir)
    assert result.total_scenarios == 2
    assert result.passed_scenarios == 1
    assert result.failed_scenarios == 1
    assert result.abstained_scenarios == 0


def test_run_with_one_abstaining_scenario_reflected_in_counts(tmp_path: Path) -> None:
    """An llm_judge rubric without a router abstains (its standard
    behaviour). One abstain among otherwise-passing rubrics makes the
    scenario abstain overall."""
    ref_dir = _make_reference_impl(tmp_path)
    abstaining = _make_scenario(
        "abstain1",
        rubrics=[
            RubricSpec(
                kind="llm_judge_coverage",
                config={
                    "target": "get_index.body",
                    "must_contain_facts": ["friendliness"],
                },
            )
        ],
    )
    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }
    runner = OraclePass(
        sandbox_runner=FakeSandboxRunner(),
        http_client=FakeHttpClient(responses),
    )
    result = runner.run(
        scenarios=[_make_scenario("ok1"), abstaining],
        reference_impl_dir=ref_dir,
        router=None,
    )
    assert result.passed_scenarios == 1
    assert result.abstained_scenarios == 1
    assert result.failed_scenarios == 0


# ---------------- Resilience ----------------


def test_run_catches_per_scenario_exceptions(tmp_path: Path) -> None:
    """A scenario whose HTTP call raises shouldn't abort the whole pass;
    it should be recorded as ``aborted=True`` and others should still run."""
    ref_dir = _make_reference_impl(tmp_path)
    boom_scenario = Scenario(
        id="boom",
        description="will raise",
        category="happy_path",
        trace=[TraceStep(id="t", method="GET", path="/does-not-exist", expect=HttpExpectation(status=200))],
        rubrics=[RubricSpec(kind="literal_match", config={"target": "t.body.ok", "expected": True})],
    )
    ok_scenario = _make_scenario("ok1")

    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
        # NOTE: no canned response for /does-not-exist — FakeHttpClient raises AssertionError
    }
    runner = OraclePass(
        sandbox_runner=FakeSandboxRunner(),
        http_client=FakeHttpClient(responses),
    )
    result = runner.run(scenarios=[boom_scenario, ok_scenario], reference_impl_dir=ref_dir)

    assert result.total_scenarios == 2
    # ok_scenario passes
    ok_out = next(o for o in result.scenario_outputs if o.scenario_id == "ok1")
    assert ok_out.aborted is False
    # boom_scenario is aborted with a reason
    boom_out = next(o for o in result.scenario_outputs if o.scenario_id == "boom")
    assert boom_out.aborted is True
    assert boom_out.abort_reason is not None
    assert "no canned response" in boom_out.abort_reason.lower() or "boom" in boom_out.abort_reason.lower() or boom_out.abort_reason


def test_run_calls_teardown_in_finally_even_on_crash(tmp_path: Path) -> None:
    """If a non-recoverable error happens after boot (we simulate by
    making the http client raise on first call AND the runner's
    per-scenario try/except still catches), teardown is still called.
    Stronger guarantee: even if we monkey-patch run_scenario to raise
    something not caught per-scenario, teardown still runs."""
    ref_dir = _make_reference_impl(tmp_path)

    class ExplodingSandbox(FakeSandboxRunner):
        pass

    sandbox = ExplodingSandbox()

    # Patch the run_scenario call inside oracle_pass to raise a fatal
    # error that bypasses per-scenario handling — to verify finally.
    import app.services.oracle_pass as op_mod

    original = op_mod.run_scenario

    def boom(**_kwargs):
        raise RuntimeError("simulated fatal crash inside oracle_pass orchestration")

    op_mod.run_scenario = boom  # type: ignore[assignment]
    runner = OraclePass(sandbox_runner=sandbox, http_client=FakeHttpClient({}))
    try:
        # If the implementation correctly wraps per-scenario errors,
        # this won't raise. If not (or if a different error escapes),
        # teardown still needs to be called.
        try:
            runner.run(scenarios=[_make_scenario("s1")], reference_impl_dir=ref_dir)
        except Exception:
            pass
    finally:
        op_mod.run_scenario = original  # type: ignore[assignment]

    assert len(sandbox.teardown_calls) == 1


def test_run_loads_setup_data_from_directory(tmp_path: Path) -> None:
    """``setup_data_dir`` is walked: JSON files parsed, text files raw,
    keys are file stems."""
    ref_dir = _make_reference_impl(tmp_path)
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    (setup_dir / "gold_answers.json").write_text(json.dumps({"q1": "alpha"}))
    (setup_dir / "rubric_notes.txt").write_text("manual notes here")

    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }

    seen_setup_data: list[dict[str, Any]] = []

    # Use a scenario whose rubric reads setup_data — easiest: capture
    # via a custom rubric. We can sniff via monkeypatching run_scenario.
    import app.services.oracle_pass as op_mod
    original = op_mod.run_scenario

    def spy(**kwargs):
        seen_setup_data.append(kwargs.get("setup_data") or {})
        return original(**kwargs)

    op_mod.run_scenario = spy  # type: ignore[assignment]
    try:
        runner = OraclePass(
            sandbox_runner=FakeSandboxRunner(),
            http_client=FakeHttpClient(responses),
        )
        runner.run(
            scenarios=[_make_scenario("s1")],
            reference_impl_dir=ref_dir,
            setup_data_dir=setup_dir,
        )
    finally:
        op_mod.run_scenario = original  # type: ignore[assignment]

    assert len(seen_setup_data) == 1
    data = seen_setup_data[0]
    assert data["gold_answers"] == {"q1": "alpha"}
    assert data["rubric_notes"] == "manual notes here"


def test_run_passes_course_meta_and_setup_data_to_each_scenario(tmp_path: Path) -> None:
    ref_dir = _make_reference_impl(tmp_path)
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    (setup_dir / "k.json").write_text(json.dumps({"v": 1}))
    course_meta = {"course_id": "demo-101"}

    responses = {
        ("GET", "http://127.0.0.1:12345/ping"): (200, {"content-type": "application/json"}, {"ok": True}),
    }
    seen: list[tuple[Any, Any]] = []
    import app.services.oracle_pass as op_mod
    original = op_mod.run_scenario

    def spy(**kwargs):
        seen.append((kwargs.get("course_meta"), kwargs.get("setup_data")))
        return original(**kwargs)

    op_mod.run_scenario = spy  # type: ignore[assignment]
    try:
        runner = OraclePass(
            sandbox_runner=FakeSandboxRunner(),
            http_client=FakeHttpClient(responses),
        )
        runner.run(
            scenarios=[_make_scenario("s1"), _make_scenario("s2")],
            reference_impl_dir=ref_dir,
            setup_data_dir=setup_dir,
            course_meta=course_meta,
        )
    finally:
        op_mod.run_scenario = original  # type: ignore[assignment]

    assert len(seen) == 2
    for meta, sd in seen:
        assert meta == course_meta
        assert sd == {"k": {"v": 1}}


# ---------------- Persistence ----------------


def test_persist_round_trips_to_equivalent_result(tmp_path: Path) -> None:
    result = OraclePassResult(
        reference_impl_hash="ref-hash",
        scenario_set_hash="scn-hash",
        generated_at="2026-05-14T00:00:00+00:00",
        scenario_outputs=[
            OracleScenarioOutput(
                scenario_id="s1",
                category="happy_path",
                captures={"step": {"status": 200, "headers": {}, "body": {"ok": True}}},
                verdicts=[("literal_match", {"status": "pass", "rationale": "ok", "diagnostic": {}, "cost_usd": 0.0})],
                aborted=False,
                abort_reason=None,
            )
        ],
        total_scenarios=1,
        passed_scenarios=1,
        failed_scenarios=0,
        abstained_scenarios=0,
    )
    out_path = tmp_path / "oracle" / "outputs.json"
    persist_oracle_outputs(result, out_path)

    assert out_path.exists()
    raw = json.loads(out_path.read_text())
    round_tripped = OraclePassResult.model_validate(raw)
    assert round_tripped.model_dump() == result.model_dump()


def test_persist_creates_parent_directories(tmp_path: Path) -> None:
    result = OraclePassResult(
        reference_impl_hash="h",
        scenario_set_hash="h",
        generated_at="2026-05-14T00:00:00+00:00",
        scenario_outputs=[],
        total_scenarios=0,
        passed_scenarios=0,
        failed_scenarios=0,
        abstained_scenarios=0,
    )
    deep = tmp_path / "a" / "b" / "c" / "outputs.json"
    persist_oracle_outputs(result, deep)
    assert deep.exists()
    assert deep.parent.is_dir()
