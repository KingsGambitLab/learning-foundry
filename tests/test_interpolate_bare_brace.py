"""Regression test: bare-brace ``{X.Y}`` placeholders interpolate too.

Bug surfaced by live run 2 (course_67915786afec, 2026-05-15 promptfoo
brief): 13/20 oracle_pass scenarios failed because the LLM-authored
trace steps emit FastAPI / OpenAPI-style path-param placeholders —
``/evaluations/{create_base_suite.body.suite_id}`` — instead of the
canonical ``${X.Y}`` form. The trace runner's ``_PLACEHOLDER_PATTERN``
only matched ``${...}``, so the literal ``{...}`` survived into the
HTTP request URL and the service returned 404, cascading into oracle
failures on every scenario that chained two steps together.

The dollar prefix is canonical; the bare form is a tolerated alias
admitted only when the contents look like a dotted expression
(contains a ``.``), so literal braces in JSON / regex / curly-set URL
syntax don't mis-fire.
"""
from __future__ import annotations

import pytest

from app.services.scenario_trace_runner import InterpolationError, interpolate


def test_bare_brace_resolves_dotted_capture() -> None:
    """The exact failure URL from Run 2 — bare braces with a dotted
    capture expression — should now resolve."""
    captures = {
        "create_base_suite": {
            "status": 201,
            "headers": {},
            "body": {"suite_id": "suite_abc"},
        }
    }
    assert (
        interpolate(
            "/evaluations/{create_base_suite.body.suite_id}", captures
        )
        == "/evaluations/suite_abc"
    )


def test_bare_brace_two_segment_shorthand() -> None:
    """Two-segment bare placeholder gets the same ``body`` auto-prefix as
    the dollar form."""
    captures = {
        "created": {"status": 201, "headers": {}, "body": {"short_code": "abc123"}}
    }
    assert interpolate("/links/{created.short_code}", captures) == "/links/abc123"


def test_dollar_form_still_works() -> None:
    """The canonical form is unchanged — regression guard."""
    captures = {
        "created": {"status": 201, "headers": {}, "body": {"short_code": "abc"}}
    }
    assert interpolate("/links/${created.short_code}", captures) == "/links/abc"


def test_mixed_dollar_and_bare_in_one_template() -> None:
    """A template can mix both forms; each placeholder resolves
    independently."""
    captures = {
        "a": {"status": 200, "headers": {}, "body": {"x": "alpha"}},
        "b": {"status": 200, "headers": {}, "body": {"y": "beta"}},
    }
    assert (
        interpolate("/p/${a.x}/q/{b.y}", captures) == "/p/alpha/q/beta"
    )


def test_bare_single_segment_left_alone() -> None:
    """``{var}`` without a dot is NOT a placeholder — it's an opaque
    literal (could be a path-param name in a route template the test
    author wants to keep, a curly-set URL syntax, etc). The runner
    must not crash on it and must not try to resolve it."""
    captures: dict[str, dict] = {}
    # No dot → not a placeholder; passes through unchanged.
    assert interpolate("/users/{user_id}", captures) == "/users/{user_id}"


def test_bare_brace_unknown_capture_raises() -> None:
    """Once the runner DOES recognize a bare ``{X.Y}`` as a placeholder,
    an unknown leading id surfaces the same InterpolationError as the
    dollar form — fail loudly, don't silently leak literal braces."""
    with pytest.raises(InterpolationError):
        interpolate("/x/{nope.field}", {})


def test_bare_brace_setup_data() -> None:
    """Bare placeholders honor the same ``setup_data.`` / ``course_meta.``
    routing as the dollar form."""
    setup_data = {"queries": {"q1": {"text": "hello world"}}}
    assert (
        interpolate(
            "q={setup_data.queries.q1.text}", {}, setup_data=setup_data
        )
        == "q=hello world"
    )
