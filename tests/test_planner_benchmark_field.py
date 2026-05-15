"""Regression tests for the planner payload's ``benchmark_json`` field.

Bug 4 from docs/superpowers/bugs/2026-05-15-autonomous-fix-loop.md.

Before this fix, ``_OutcomePlanPayload`` had no slot for a benchmark
binding — the LLM could not emit "this course grades against Quivr/CRAG"
even when the brief named it. The Wave 5 hardcode sniffed the brief
text for ``quivr/crag`` and injected a default ``CRAGBenchmarkSource``;
the actual planner-emitted payload had no influence.

After this fix:
- ``_OutcomePlanPayload.benchmark_json`` carries the JSON-stringified
  ``BenchmarkSource`` (discriminated union of HF + CRAG shapes).
- ``_normalize_payload`` parses it and validates against the model.
- The brief-text sniff is a FALLBACK that fires only when the LLM
  omits the field.
- ``oracle_source`` flips to ``hybrid`` whenever a benchmark binds.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from app.domain.course import GenerateCourseFromBriefRequest
from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CRAGBenchmarkSource,
    HFBenchmarkSource,
    OracleSource,
)
from app.services.course_outcome_planner import (
    OutcomeCoursePlanner,
    _OutcomePlanPayload,
)


def _valid_payload_dict(**overrides) -> dict:
    base = {
        "title": "Title for benchmark binding test",
        "goal": "G with enough length for the model validator to accept",
        "starter_type": "partial",
        "endpoints": [
            {
                "method": "POST",
                "path": "/x",
                "request_schema_json": "{}",
                "response_schema_json": '{"type": "object"}',
                "description": "x",
            }
        ],
        "quality_bars": [
            {
                "id": "b1",
                "metric_description": "Some metric on the system",
                "threshold": ">= 0.5",
                "judged_by": "literal",
                "sample_size": 5,
            }
        ],
        "learning_path": [],
    }
    base.update(overrides)
    return base


def _request() -> GenerateCourseFromBriefRequest:
    return GenerateCourseFromBriefRequest(
        goal=(
            "Build a retrieval service for some domain — sufficient length "
            "to satisfy the brief's min_length validator."
        ),
        title=None,
        package_type_hint=PackageType.progressive_codebase_course,
    )


class BenchmarkJsonParsingTests(unittest.TestCase):
    """The LLM-emitted ``benchmark_json`` is parsed into the discriminated
    BenchmarkSource union and threaded onto the spec.
    """

    def test_crag_benchmark_json_parses(self) -> None:
        planner = OutcomeCoursePlanner(router=MagicMock())
        payload_dict = _valid_payload_dict(
            benchmark_json=json.dumps(
                {
                    "kind": "crag",
                    "dataset": "Quivr/CRAG",
                    "max_queries": 25,
                }
            )
        )
        payload = _OutcomePlanPayload.model_validate(payload_dict)
        spec = planner._normalize_payload(_request(), payload)
        self.assertIsInstance(spec.benchmark, CRAGBenchmarkSource)
        self.assertEqual(spec.benchmark.dataset, "Quivr/CRAG")
        self.assertEqual(spec.benchmark.max_queries, 25)
        # When a benchmark binds, oracle_source flips to hybrid.
        self.assertEqual(spec.oracle_source, OracleSource.hybrid)

    def test_huggingface_benchmark_json_parses(self) -> None:
        planner = OutcomeCoursePlanner(router=MagicMock())
        payload_dict = _valid_payload_dict(
            benchmark_json=json.dumps(
                {
                    "kind": "huggingface",
                    "corpus_dataset": "BeIR/scifact",
                    "qrels_dataset": "BeIR/scifact-qrels",
                    "max_queries": 50,
                }
            )
        )
        payload = _OutcomePlanPayload.model_validate(payload_dict)
        spec = planner._normalize_payload(_request(), payload)
        self.assertIsInstance(spec.benchmark, HFBenchmarkSource)
        self.assertEqual(spec.benchmark.corpus_dataset, "BeIR/scifact")
        self.assertEqual(spec.oracle_source, OracleSource.hybrid)

    def test_missing_benchmark_json_returns_none_when_no_brief_sniff_match(self) -> None:
        planner = OutcomeCoursePlanner(router=MagicMock())
        payload = _OutcomePlanPayload.model_validate(_valid_payload_dict())
        spec = planner._normalize_payload(_request(), payload)
        self.assertIsNone(spec.benchmark)

    def test_invalid_benchmark_kind_falls_back_to_none(self) -> None:
        """An unrecognized ``kind`` is logged and ignored — we don't crash."""
        planner = OutcomeCoursePlanner(router=MagicMock())
        payload_dict = _valid_payload_dict(
            benchmark_json=json.dumps({"kind": "unknown_shape", "x": 1})
        )
        payload = _OutcomePlanPayload.model_validate(payload_dict)
        spec = planner._normalize_payload(_request(), payload)
        self.assertIsNone(spec.benchmark)

    def test_brief_sniff_still_works_when_payload_omits_benchmark(self) -> None:
        """Legacy briefs that name Quivr/CRAG without the new payload
        field still bind correctly (fallback path)."""
        planner = OutcomeCoursePlanner(router=MagicMock())
        request = GenerateCourseFromBriefRequest(
            goal=(
                "Build a service that answers questions over the Quivr/CRAG "
                "finance benchmark."
            ),
            title=None,
            package_type_hint=PackageType.progressive_codebase_course,
        )
        payload = _OutcomePlanPayload.model_validate(_valid_payload_dict())
        spec = planner._normalize_payload(request, payload)
        self.assertIsInstance(spec.benchmark, CRAGBenchmarkSource)

    def test_explicit_benchmark_json_overrides_brief_sniff(self) -> None:
        """When the LLM emits ``benchmark_json``, the sniff doesn't fire
        even if the brief mentions Quivr/CRAG — the planner-emitted
        value wins."""
        planner = OutcomeCoursePlanner(router=MagicMock())
        request = GenerateCourseFromBriefRequest(
            goal=(
                "Build a service that answers questions over the Quivr/CRAG "
                "finance benchmark — but use 100 queries not 20."
            ),
            title=None,
            package_type_hint=PackageType.progressive_codebase_course,
        )
        payload_dict = _valid_payload_dict(
            benchmark_json=json.dumps(
                {
                    "kind": "crag",
                    "dataset": "Quivr/CRAG",
                    "max_queries": 100,  # honored, NOT overridden to 20
                }
            )
        )
        payload = _OutcomePlanPayload.model_validate(payload_dict)
        spec = planner._normalize_payload(request, payload)
        self.assertEqual(spec.benchmark.max_queries, 100)


if __name__ == "__main__":
    unittest.main()
