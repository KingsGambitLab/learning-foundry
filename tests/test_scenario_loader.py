"""Tests for the scenario YAML schema and loader.

The loader is parse-only: it validates structure, looks up every
rubric ``kind`` against ``RUBRIC_REGISTRY``, and rejects duplicate
scenario ids — but it does NOT resolve ``${...}`` interpolation
(that's the trace runner's job) and does NOT execute any HTTP calls.

These tests exercise the public surface:
``Scenario``, ``TraceStep``, ``HttpExpectation``, ``RubricSpec``,
``ScenarioLoadError``, and ``load_scenarios_from_dir``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.services.scenario_loader import (
    HttpExpectation,
    RubricSpec,
    Scenario,
    ScenarioLoadError,
    TraceStep,
    load_scenarios_from_dir,
)


# ---------------- HttpExpectation ----------------


def test_http_expectation_with_single_status() -> None:
    e = HttpExpectation(status=200)
    assert e.status == 200


def test_http_expectation_with_status_list() -> None:
    e = HttpExpectation(status=[200, 201])
    assert e.status == [200, 201]


def test_http_expectation_defaults_to_none() -> None:
    e = HttpExpectation()
    assert e.status is None


# ---------------- TraceStep ----------------


def test_trace_step_minimal() -> None:
    step = TraceStep(id="get_root", method="GET", path="/")
    assert step.id == "get_root"
    assert step.method == "GET"
    assert step.path == "/"
    assert step.body is None
    assert step.headers == {}
    assert step.follow_redirects is True
    assert step.expect is None
    assert step.capture is None


def test_trace_step_with_body_and_expect() -> None:
    step = TraceStep(
        id="create_link",
        method="POST",
        path="/links",
        body={"url": "https://example.com"},
        expect=HttpExpectation(status=201),
        capture="created",
    )
    assert step.body == {"url": "https://example.com"}
    assert step.expect is not None
    assert step.expect.status == 201
    assert step.capture == "created"


def test_trace_step_preserves_variable_substitution_strings() -> None:
    """Loader must NOT resolve ``${id.field}`` placeholders — they
    pass through verbatim for the runner to resolve later."""
    step = TraceStep(
        id="follow",
        method="GET",
        path="/links/${created.body.short_code}",
        body={"caller": "${created.body.owner_id}"},
    )
    assert step.path == "/links/${created.body.short_code}"
    assert step.body == {"caller": "${created.body.owner_id}"}


def test_trace_step_rejects_invalid_method() -> None:
    with pytest.raises(ValidationError):
        TraceStep(id="x", method="FLY", path="/")


def test_trace_step_follow_redirects_false() -> None:
    step = TraceStep(
        id="follow_short",
        method="GET",
        path="/r/abc",
        follow_redirects=False,
    )
    assert step.follow_redirects is False


# ---------------- RubricSpec ----------------


def test_rubric_spec_packs_extra_fields_into_config() -> None:
    """A rubric line in YAML has free-form keys per rubric kind; the
    loader packs every non-``kind`` field into ``config``."""
    spec = RubricSpec(
        kind="schema_match",
        target="resp.body",
        must_have_fields=["answer", "citations"],
    )
    assert spec.kind == "schema_match"
    assert spec.config == {
        "target": "resp.body",
        "must_have_fields": ["answer", "citations"],
    }


def test_rubric_spec_with_explicit_config_dict() -> None:
    """Authors may also write ``config: {...}`` explicitly."""
    spec = RubricSpec(
        kind="literal_match",
        config={"target": "resp.status", "expected": 200},
    )
    assert spec.kind == "literal_match"
    assert spec.config == {"target": "resp.status", "expected": 200}


def test_rubric_spec_empty_config_when_only_kind() -> None:
    spec = RubricSpec(kind="schema_match")
    assert spec.config == {}


def test_rubric_spec_unknown_kind_raises() -> None:
    with pytest.raises(ScenarioLoadError) as exc_info:
        RubricSpec(kind="not_a_real_rubric")
    msg = str(exc_info.value)
    assert "not_a_real_rubric" in msg
    # the message should hint at the registered kinds
    assert "schema_match" in msg or "registered" in msg.lower()


# ---------------- Scenario ----------------


def test_scenario_happy_path_from_dict() -> None:
    sc = Scenario(
        id="q_rag_definition",
        description="learner answers what RAG is",
        category="happy_path",
        trace=[
            {
                "id": "ask",
                "method": "POST",
                "path": "/answer",
                "body": {"question": "What is RAG?"},
                "expect": {"status": 200},
                "capture": "resp",
            }
        ],
        rubrics=[
            {
                "kind": "schema_match",
                "target": "resp.body",
                "must_have_fields": ["answer", "citations"],
            }
        ],
    )
    assert sc.id == "q_rag_definition"
    assert sc.category == "happy_path"
    assert sc.mode == "independent"  # default
    assert sc.setup == []
    assert len(sc.trace) == 1
    assert sc.trace[0].id == "ask"
    assert sc.rubrics[0].kind == "schema_match"
    assert sc.rubrics[0].config["target"] == "resp.body"


def test_scenario_mode_defaults_to_independent() -> None:
    sc = Scenario(
        id="x",
        description="d",
        category="happy_path",
        trace=[{"id": "a", "method": "GET", "path": "/"}],
        rubrics=[{"kind": "schema_match", "target": "a"}],
    )
    assert sc.mode == "independent"


def test_scenario_mode_live_accepted() -> None:
    sc = Scenario(
        id="x",
        description="d",
        category="happy_path",
        trace=[{"id": "a", "method": "GET", "path": "/"}],
        rubrics=[{"kind": "schema_match", "target": "a"}],
        mode="live",
    )
    assert sc.mode == "live"


def test_scenario_empty_trace_rejected() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="x",
            description="d",
            category="happy_path",
            trace=[],
            rubrics=[{"kind": "schema_match"}],
        )


def test_scenario_no_rubrics_rejected() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="x",
            description="d",
            category="happy_path",
            trace=[{"id": "a", "method": "GET", "path": "/"}],
            rubrics=[],
        )


def test_scenario_invalid_category_rejected() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="x",
            description="d",
            category="not_a_category",
            trace=[{"id": "a", "method": "GET", "path": "/"}],
            rubrics=[{"kind": "schema_match"}],
        )


def test_scenario_invalid_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            id="x",
            description="d",
            category="happy_path",
            trace=[{"id": "a", "method": "GET", "path": "/"}],
            rubrics=[{"kind": "schema_match"}],
            mode="ghost",
        )


def test_scenario_supports_setup_steps() -> None:
    sc = Scenario(
        id="x",
        description="d",
        category="happy_path",
        setup=[
            {"id": "ingest", "method": "POST", "path": "/ingest", "body": {"text": "doc"}}
        ],
        trace=[{"id": "a", "method": "GET", "path": "/"}],
        rubrics=[{"kind": "schema_match"}],
    )
    assert len(sc.setup) == 1
    assert sc.setup[0].id == "ingest"


# ---------------- load_scenarios_from_dir ----------------


_VALID_YAML_A = """
id: q_rag_definition
description: learner answers what RAG is
category: happy_path
trace:
  - id: ask
    method: POST
    path: /answer
    body:
      question: What is RAG?
    expect:
      status: 200
    capture: resp
rubrics:
  - kind: schema_match
    target: resp.body
    must_have_fields: [answer, citations]
"""

_VALID_YAML_B = """
id: q_link_redirect
description: short-link round trip
category: state_persistence
setup:
  - id: create
    method: POST
    path: /links
    body:
      url: https://example.com
    capture: created
trace:
  - id: visit
    method: GET
    path: /r/${created.body.short_code}
    follow_redirects: false
    expect:
      status: [301, 302]
    capture: hit
rubrics:
  - kind: literal_match
    target: hit.status
    expected: 302
"""


def test_load_scenarios_from_dir_happy(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(_VALID_YAML_A)
    (tmp_path / "b.yaml").write_text(_VALID_YAML_B)

    scenarios = load_scenarios_from_dir(tmp_path)
    ids = {s.id for s in scenarios}
    assert ids == {"q_rag_definition", "q_link_redirect"}


def test_load_scenarios_from_dir_preserves_var_syntax(tmp_path: Path) -> None:
    """The runner — not the loader — resolves ``${...}``; the loader
    must leave raw placeholder strings untouched."""
    (tmp_path / "b.yaml").write_text(_VALID_YAML_B)
    [sc] = load_scenarios_from_dir(tmp_path)
    assert sc.trace[0].path == "/r/${created.body.short_code}"


def test_load_scenarios_from_dir_malformed_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: x\ndescription: [unterminated\n")
    with pytest.raises(ScenarioLoadError) as exc_info:
        load_scenarios_from_dir(tmp_path)
    assert "bad.yaml" in str(exc_info.value)


def test_load_scenarios_from_dir_duplicate_ids_raises(tmp_path: Path) -> None:
    (tmp_path / "one.yaml").write_text(_VALID_YAML_A)
    # Same id, different file.
    (tmp_path / "two.yaml").write_text(_VALID_YAML_A)
    with pytest.raises(ScenarioLoadError) as exc_info:
        load_scenarios_from_dir(tmp_path)
    msg = str(exc_info.value)
    assert "q_rag_definition" in msg
    assert "duplicate" in msg.lower() or "already" in msg.lower()


def test_load_scenarios_from_dir_unknown_rubric_kind_raises(tmp_path: Path) -> None:
    (tmp_path / "x.yaml").write_text(
        """
id: x
description: d
category: happy_path
trace:
  - id: a
    method: GET
    path: /
rubrics:
  - kind: foo_bar
"""
    )
    with pytest.raises(ScenarioLoadError) as exc_info:
        load_scenarios_from_dir(tmp_path)
    msg = str(exc_info.value)
    assert "foo_bar" in msg
    assert "x.yaml" in msg


def test_load_scenarios_from_dir_validation_error_names_file(tmp_path: Path) -> None:
    """A pydantic ValidationError (e.g., bad category) gets wrapped in
    ScenarioLoadError so callers don't have to know about pydantic."""
    (tmp_path / "nope.yaml").write_text(
        """
id: x
description: d
category: not_a_category
trace:
  - id: a
    method: GET
    path: /
rubrics:
  - kind: schema_match
"""
    )
    with pytest.raises(ScenarioLoadError) as exc_info:
        load_scenarios_from_dir(tmp_path)
    assert "nope.yaml" in str(exc_info.value)


def test_load_scenarios_from_dir_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_scenarios_from_dir(tmp_path) == []


def test_load_scenarios_from_dir_ignores_non_yaml_files(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(_VALID_YAML_A)
    (tmp_path / "README.md").write_text("not a scenario")
    (tmp_path / "notes.txt").write_text("ignore me")
    scenarios = load_scenarios_from_dir(tmp_path)
    assert [s.id for s in scenarios] == ["q_rag_definition"]


def test_load_scenarios_from_dir_accepts_string_path(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(_VALID_YAML_A)
    scenarios = load_scenarios_from_dir(str(tmp_path))
    assert len(scenarios) == 1


def test_load_scenarios_from_dir_missing_dir_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ScenarioLoadError):
        load_scenarios_from_dir(missing)
