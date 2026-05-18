"""Regression test for capture entries carrying request info.

Bug 18 from docs/superpowers/bugs/2026-05-15-autonomous-fix-loop.md.

Before this fix, ``captures[step_id]`` carried only ``status / headers
/ body``. Rubrics that wanted to check "the citations subset of the
request's ``search_results``" had to inline the passages twice — once
in the trace body and once in the rubric config — because the
``acceptable_source: trace.<step>.request.json.search_results`` path
the LLM kept emitting wouldn't resolve.

After the fix, capture entries include a ``request`` block carrying
the (interpolated) method, path, headers, body, and URL. Rubrics can
now address ``<step>.request.body.search_results``.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.services.scenario_loader import (
    HttpExpectation,
    RubricSpec,
    Scenario,
    TraceStep,
)
from app.services.scenario_trace_runner import (
    UrllibHttpClient,
    run_scenario,
)


class _ScriptedHttp:
    """HTTP fake that returns canned responses without hitting the network."""

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def request(self, *, method, url, headers, body, follow_redirects, timeout):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": dict(headers),
            "body": body,
        })
        return self._responses.pop(0)


class CaptureRequestTests(unittest.TestCase):
    def test_capture_entry_includes_request(self) -> None:
        scenario = Scenario(
            id="t",
            description="capture-request test",
            category="happy_path",
            quality_bar_ids=["x"],
            trace=[
                TraceStep(
                    id="call",
                    method="POST",
                    path="/finance/answer",
                    body={
                        "question": "What was Q3 revenue?",
                        "search_results": [
                            {"passage_id": "p1", "text": "Q3 revenue was $12B."},
                            {"passage_id": "p2", "text": "Operating margin was 18%."},
                        ],
                    },
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
                {"answer": "$12B", "citations": ["p1"], "abstained": False},
            )
        )
        report = run_scenario(
            scenario=scenario,
            base_url="http://localhost:8000",
            http_client=http,
        )
        captures = report.run_result.captures
        self.assertIn("request", captures["call"])
        req = captures["call"]["request"]
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["path"], "/finance/answer")
        # Body is the SAME dict the runner sent on the wire — including
        # the search_results so rubrics can check citation subset.
        self.assertEqual(req["body"]["question"], "What was Q3 revenue?")
        self.assertEqual(len(req["body"]["search_results"]), 2)
        self.assertEqual(req["body"]["search_results"][0]["passage_id"], "p1")


if __name__ == "__main__":
    unittest.main()
