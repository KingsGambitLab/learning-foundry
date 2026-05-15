"""Regression test for capture entries carrying the raw response text.

Bug surfaced by live run 2 (course_67915786afec, 2026-05-15 promptfoo
brief): 11+ of the 20 oracle_pass failures were ``<step>.raw not
found in captures``. The scenario-author LLM emits ``regex_match``
rubrics that target ``<step_id>.raw`` to assert against the raw JSON
TEXT of the response — but capture entries only carried parsed
``body`` (dict/list) and there was no ``raw`` key.

Fix: ``_execute_step`` now stores a ``raw`` string field on every
capture entry. It's the JSON-encoded re-serialization of ``body``
when body is a dict/list, the body verbatim when it's already a
string, or empty for ``None`` bodies. Sufficient for rubrics that
need text-level pattern matching against API responses.
"""
from __future__ import annotations

import unittest

from app.services.scenario_loader import (
    HttpExpectation,
    RubricSpec,
    Scenario,
    TraceStep,
)
from app.services.scenario_trace_runner import run_scenario


class _ScriptedHttp:
    def __init__(self, *responses) -> None:
        self._responses = list(responses)

    def request(self, *, method, url, headers, body, follow_redirects, timeout):
        return self._responses.pop(0)


class CaptureRawTests(unittest.TestCase):
    def test_capture_carries_raw_text_for_json_body(self) -> None:
        scenario = Scenario(
            id="raw_t",
            description="raw key in capture",
            category="happy_path",
            quality_bar_ids=["x"],
            trace=[
                TraceStep(
                    id="call",
                    method="POST",
                    path="/evaluations",
                    body={"suite_id": "s1"},
                    expect=HttpExpectation(status_code=200),
                )
            ],
            rubrics=[
                RubricSpec(kind="schema_match", config={"target": "call.body"})
            ],
        )
        http = _ScriptedHttp(
            (
                200,
                {"content-type": "application/json"},
                {
                    "runId": "run_abc",
                    "results": [
                        {"testIdx": 0, "providerOutput": "Y", "gradingResult": {"pass": True}}
                    ],
                },
            )
        )
        report = run_scenario(
            scenario=scenario,
            base_url="http://localhost:8000",
            http_client=http,
        )
        cap = report.run_result.captures["call"]
        self.assertIn("raw", cap)
        self.assertIsInstance(cap["raw"], str)
        # Raw text re-serializes the body — promptfoo-shaped JSON fields
        # surface as text that regex_match rubrics can assert on.
        self.assertIn("testIdx", cap["raw"])
        self.assertIn("gradingResult", cap["raw"])
        self.assertIn("providerOutput", cap["raw"])
        # Parsed body unchanged.
        self.assertEqual(cap["body"]["runId"], "run_abc")

    def test_capture_raw_is_empty_string_for_none_body(self) -> None:
        scenario = Scenario(
            id="raw_n",
            description="None body",
            category="happy_path",
            quality_bar_ids=["x"],
            trace=[
                TraceStep(
                    id="call",
                    method="GET",
                    path="/no-body",
                    expect=HttpExpectation(status_code=204),
                )
            ],
            rubrics=[
                RubricSpec(kind="schema_match", config={"target": "call.body"})
            ],
        )
        http = _ScriptedHttp((204, {}, None))
        report = run_scenario(
            scenario=scenario,
            base_url="http://localhost:8000",
            http_client=http,
        )
        self.assertEqual(report.run_result.captures["call"]["raw"], "")


if __name__ == "__main__":
    unittest.main()
