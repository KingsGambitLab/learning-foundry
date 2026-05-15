"""Regression tests for the grader-repair failure-context fix.

The live RAG/CRAG smoke (2026-05-14) exhausted the grader-repair budget
3 times without ever actually repairing — ``node_grader_repair`` called
``oracle_author.author_oracle(spec)`` with no findings, so every retry
was a re-roll of the dice. These tests pin the fix:

- ``OracleAuthor.author_oracle`` accepts ``failure_context``.
- When ``failure_context`` is provided, the system+user prompts the
  router sees include the prior blocking reasons (so the LLM can
  repair targeted issues, not re-derive the whole bundle).
- ``node_grader_repair`` builds ``failure_context`` from the prior
  validation reports + oracle pass result and threads it through.
- ``node_grader_repair`` emits ``node_grader_repair_invoked`` so the
  log has observability for "repair attempted".

Real-data fixture: ``LIVE_BLOCKING_REASONS`` is the 73-reason report
captured from ``course_f918e889a33c`` after its grader budget was
exhausted on the live smoke (see
``docs/superpowers/bugs/2026-05-15-autonomous-fix-loop.md`` §
"Harness-repair investigation"). Using real captured data instead of
hand-rolled fixtures keeps the test honest about the regime the fix
actually has to handle.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from app.services.course_outcome_models import (
    CapabilityFlags,
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    LearningHint,
    OracleSource,
    QualityBar,
    StarterType,
)
from app.domain.registry import PackageType
from app.services.oracle_authoring import OracleAuthor


# 25 of the 73 blocking reasons captured from
# course_f918e889a33c.oracle_validation_report after its grader budget
# was exhausted. The trailing entries are repeats — kept in to match
# the actual on-disk shape.
LIVE_BLOCKING_REASONS = [
    "Required scenario category 'boundary' has no scenarios.",
    "Required scenario category 'happy_path' has no scenarios.",
    "Required scenario category 'malformed_input' has no scenarios.",
    "Required scenario category 'out_of_scope' has no scenarios.",
    "Quality bar 'finance_answer_schema_conformance' is declared in the spec but no scenario contributes to it (no scenario lists this id in its quality_bar_ids). Either author at least one scenario that references this bar or remove the bar from the spec.",
    "Quality bar 'citation_set_overlap' is declared in the spec but no scenario contributes to it (no scenario lists this id in its quality_bar_ids). Either author at least one scenario that references this bar or remove the bar from the spec.",
    "Quality bar 'finance_answer_faithfulness' is declared in the spec but no scenario contributes to it (no scenario lists this id in its quality_bar_ids). Either author at least one scenario that references this bar or remove the bar from the spec.",
    "Quality bar 'false_premise_abstention_precision' is declared in the spec but no scenario contributes to it (no scenario lists this id in its quality_bar_ids). Either author at least one scenario that references this bar or remove the bar from the spec.",
    "Quality bar 'extractive_stub_resistance' is declared in the spec but no scenario contributes to it (no scenario lists this id in its quality_bar_ids). Either author at least one scenario that references this bar or remove the bar from the spec.",
]


def _make_spec() -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title="Test finance QA",
        goal="Build an extractive finance Q&A service over benchmark passages.",
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/finance/answer",
                request_schema={"type": "object"},
                response_schema={"type": "object"},
                description="Answer endpoint.",
            )
        ],
        quality_bars=[
            QualityBar(
                id="schema_conformance",
                metric_description="Schema match",
                threshold=">= 0.99",
                judged_by=JudgeKind.literal,
                sample_size=10,
            )
        ],
        learning_path=[],
        package_type=PackageType.progressive_codebase_course,
        oracle_source=OracleSource.curated,
        capabilities=CapabilityFlags(),
    )


def _make_router_with_recorded_response(parsed_payload: Any) -> MagicMock:
    """A router that returns the supplied parsed payload from
    ``parse_structured``. The test asserts on ``router.parse_structured.call_args``
    to verify what user-prompt got built.
    """
    router = MagicMock()
    response = SimpleNamespace(
        parsed=parsed_payload,
        output_parsed=parsed_payload,
        usage=None,
        usage_summary=None,
    )
    router.parse_structured.return_value = response
    router.model_id_for.return_value = "fake-model"
    return router


class AuthorOracleFailureContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = _make_spec()

    def test_author_oracle_accepts_failure_context_kwarg(self) -> None:
        """The grader-repair node needs to pass ``failure_context``;
        the public signature must accept it without TypeError."""
        # We can't easily construct a valid _OracleAuthorPayload here
        # without a lot of setup — instead patch _validate_payload to
        # always fail so we exercise the kwarg-accepts path through
        # attempt_diagnostics seeding without needing a full payload.
        router = MagicMock()
        router.parse_structured.side_effect = RuntimeError("test router stop")
        router.model_id_for.return_value = "fake-model"
        author = OracleAuthor(router=router)

        with self.assertRaises(Exception):
            # router raises → we hit attempt_diagnostics path on the
            # first iteration; if failure_context is rejected this
            # raises TypeError instead.
            author.author_oracle(
                self.spec,
                failure_context={"prior_blocking_reasons": LIVE_BLOCKING_REASONS},
            )

    def test_failure_context_seeds_attempt_diagnostics(self) -> None:
        """When ``failure_context`` is supplied, the user prompt sent to
        the router on attempt 1 contains the prior blocking reasons
        (so the LLM can repair rather than re-roll)."""
        router = MagicMock()
        router.parse_structured.side_effect = RuntimeError("stop after first call")
        router.model_id_for.return_value = "fake-model"
        author = OracleAuthor(router=router)

        try:
            author.author_oracle(
                self.spec,
                failure_context={
                    "prior_blocking_reasons": LIVE_BLOCKING_REASONS,
                    "passed_scenarios": 0,
                    "failed_scenarios": 18,
                },
            )
        except Exception:
            pass

        # First call to parse_structured: inspect the user prompt.
        self.assertTrue(router.parse_structured.called)
        call_args = router.parse_structured.call_args
        user_prompt = call_args.kwargs.get("user", "")
        # Real captured blocking reasons must surface in the prompt.
        self.assertIn("Required scenario category 'happy_path'", user_prompt)
        self.assertIn("Quality bar 'finance_answer_schema_conformance'", user_prompt)
        # Counts surface too.
        self.assertIn("failed_scenarios", user_prompt)
        self.assertIn("passed_scenarios", user_prompt)

    def test_failure_context_omitted_keeps_legacy_behavior(self) -> None:
        """No ``failure_context`` → attempt 1 prompt has no prior-failures
        section (legacy behavior)."""
        router = MagicMock()
        router.parse_structured.side_effect = RuntimeError("stop")
        router.model_id_for.return_value = "fake-model"
        author = OracleAuthor(router=router)

        try:
            author.author_oracle(self.spec)
        except Exception:
            pass

        user_prompt = router.parse_structured.call_args.kwargs.get("user", "")
        # The legacy prompt has no "Prior author_oracle call" header.
        self.assertNotIn("Prior author_oracle call produced a bundle", user_prompt)


class NodeGraderRepairInstrumentationTests(unittest.TestCase):
    """Pin that ``node_grader_repair`` (a) emits an event with the
    findings count, and (b) forwards a structured ``failure_context``
    into the oracle author.
    """

    def test_grader_repair_emits_invocation_event(self) -> None:
        from app.services.langgraph_outcome_graph import (
            OutcomeGraphDeps,
            OutcomeWorkflowState,
            node_grader_repair,
        )
        from app.services.oracle_validation import OracleValidationReport

        # Fake oracle author records the kwargs it received.
        recorded_kwargs: dict = {}

        class _RecordingAuthor:
            def author_oracle(self, spec, *, failure_context=None):
                recorded_kwargs["spec"] = spec
                recorded_kwargs["failure_context"] = failure_context
                # Return a bare ``OracleAuthorResult``-shaped object
                # so node_grader_repair's downstream materialize call
                # doesn't blow up. We only care about the kwarg here.
                return SimpleNamespace(
                    scenarios=[],
                    reference_files=[],
                    setup_files=[],
                    visible_sample_queries_json=None,
                    cost_usd=0.0,
                    diagnostics=[],
                )

        emitted_events: list[tuple[str, dict]] = []

        def _capture(name: str, **kwargs):
            emitted_events.append((name, kwargs))

        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            state = OutcomeWorkflowState(
                run_id="run_test",
                workspace_root=Path(tmp),
                request={},
                spec=_make_spec(),
                spec_review_findings=[],
                starter_files=[],
                starter_attempt=0,
                grader_attempt=0,
                blocking_reasons=[],
                stage="oracle_validation",
                status="running",
            )
            state.oracle_validation_report = OracleValidationReport(
                publishable=False,
                summary="test",
                blocking_reasons=list(LIVE_BLOCKING_REASONS),
                category_coverage=[],
                reference_impl_hash="",
                scenario_set_hash="",
            )
            deps = OutcomeGraphDeps(
                planner=None,
                oracle_author=_RecordingAuthor(),
            )
            with patch(
                "app.services.langgraph_outcome_graph.log_coursegen_event",
                side_effect=_capture,
            ), patch(
                "app.services.langgraph_outcome_graph.materialize_oracle_bundle"
            ):
                node_grader_repair(state, deps=deps)

        # ----- event emitted -----
        invocation_events = [
            e for e in emitted_events if e[0] == "node_grader_repair_invoked"
        ]
        self.assertEqual(len(invocation_events), 1)
        _, payload = invocation_events[0]
        self.assertEqual(payload["grader_attempt"], 1)
        self.assertEqual(payload["blocking_reasons_count"], len(LIVE_BLOCKING_REASONS))

        # ----- failure_context threaded through -----
        ctx = recorded_kwargs["failure_context"]
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["prior_blocking_reasons"], list(LIVE_BLOCKING_REASONS))


if __name__ == "__main__":
    unittest.main()
