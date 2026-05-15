"""Regression test for setup_data JSONL parsing.

Bug 26 from docs/superpowers/bugs/2026-05-15-autonomous-fix-loop.md.

Before this fix, ``.jsonl`` files in ``_setup/`` were loaded as raw
text. The CRAG benchmark loader writes ``queries.jsonl``, and curated
scenarios referencing ``${setup_data.queries.0.query}`` would crash
because ``setup_data.queries`` was a string, not a list.

The fixture uses the exact JSONL line format CRAG produces (captured
from workspaces/outcome/course_f918e889a33c/private/grader/_setup/
during the live smoke).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.services.oracle_pass import _load_setup_data


# Real CRAG queries.jsonl line shape, captured from the live smoke.
LIVE_CRAG_JSONL_LINES = [
    {
        "alt_ans": [],
        "answer": "4 3-points attempts per game",
        "answer_type": "valid",
        "domain": "sports",
        "query": "how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club?",
        "query_id": "7bb29eb4-12f9-45f9-bf8a-66832b3c8962",
        "question_type": "post-processing",
        "search_results": [],
    },
    {
        "alt_ans": [],
        "answer": "finding nemo",
        "answer_type": "valid",
        "domain": "movie",
        "query": "in 2004, which animated film was recognized with the best animated feature film oscar?",
        "query_id": "8163a6f0-3238-4a69-ba60-a4a06090bc6f",
        "question_type": "simple",
        "search_results": [],
    },
]


class JsonlLoaderTests(unittest.TestCase):
    def test_jsonl_loaded_as_list_of_dicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "queries.jsonl").write_text(
                "\n".join(json.dumps(line) for line in LIVE_CRAG_JSONL_LINES)
                + "\n"
            )
            result = _load_setup_data(path)
        self.assertIn("queries", result)
        self.assertIsInstance(result["queries"], list)
        self.assertEqual(len(result["queries"]), 2)
        self.assertEqual(result["queries"][0]["answer"], "4 3-points attempts per game")

    def test_jsonl_with_blank_lines_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "queries.jsonl").write_text(
                json.dumps(LIVE_CRAG_JSONL_LINES[0]) + "\n\n\n"
                + json.dumps(LIVE_CRAG_JSONL_LINES[1]) + "\n"
            )
            result = _load_setup_data(path)
        self.assertEqual(len(result["queries"]), 2)

    def test_jsonl_with_malformed_line_falls_back_to_raw_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "queries.jsonl").write_text(
                json.dumps(LIVE_CRAG_JSONL_LINES[0]) + "\n"
                "not valid JSON\n"
            )
            result = _load_setup_data(path)
        # Falls back to raw text so the operator can see the file.
        self.assertIsInstance(result["queries"], str)
        self.assertIn("not valid JSON", result["queries"])

    def test_json_files_still_parsed_into_dicts(self) -> None:
        """Regression: .json behavior unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "gold.json").write_text(
                json.dumps({"q1": {"answer": "A"}, "q2": {"answer": "B"}})
            )
            result = _load_setup_data(path)
        self.assertIsInstance(result["gold"], dict)
        self.assertEqual(result["gold"]["q1"]["answer"], "A")


if __name__ == "__main__":
    unittest.main()
