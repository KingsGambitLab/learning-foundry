from __future__ import annotations

from app.domain.grading import (
    ApprovalRecord,
    EscalationRecord,
    EvalRunEvidence,
    FailureInjectionRecord,
    FallbackActionRecord,
    TaskAgentSubmission,
    ToolCallRecord,
    ToolCallStatus,
)


def get_generic_project_submission() -> TaskAgentSubmission:
    """Return a stable passing submission used by legacy API tests."""

    return TaskAgentSubmission(
        submission_id="generic_project_reference_submission_v1",
        metadata={"scenario": "reference_passing_submission"},
        runs=[
            EvalRunEvidence(
                run_id="run-billing-001",
                case_id="billing_refund_simple",
                output={
                    "disposition": "resolve",
                    "priority": "medium",
                    "reply_draft": "I found the duplicate charge and have started the refund process.",
                    "confidence": 0.94,
                    "needs_human": False,
                },
                trace_events=[
                    "run_started",
                    "model_called",
                    "tool_selected",
                    "tool_called",
                    "tool_result",
                    "approval_requested",
                    "fallback_used",
                    "run_completed",
                ],
                step_count=5,
                latency_ms=2400,
                cost_usd=0.022,
                tool_calls=[
                    ToolCallRecord(
                        order=1,
                        tool_id="lookup_account",
                        args={"ticket_id": "T-100"},
                        status=ToolCallStatus.ok,
                    ),
                    ToolCallRecord(
                        order=2,
                        tool_id="search_kb",
                        args={"query": "double charged invoice refund"},
                        status=ToolCallStatus.timeout,
                    ),
                    ToolCallRecord(
                        order=3,
                        tool_id="lookup_account",
                        args={"ticket_id": "T-100"},
                        status=ToolCallStatus.ok,
                    ),
                    ToolCallRecord(
                        order=4,
                        tool_id="send_reply",
                        args={
                            "ticket_id": "T-100",
                            "reply_draft": "I found the duplicate charge and have started the refund process.",
                        },
                        status=ToolCallStatus.ok,
                        idempotency_key="T-100",
                        approval_id="appr-billing-1",
                    ),
                ],
                approvals=[
                    ApprovalRecord(
                        approval_id="appr-billing-1",
                        order=3,
                        tool_id="send_reply",
                        approved=True,
                    )
                ],
                failure_injections=[
                    FailureInjectionRecord(
                        target="tool",
                        target_id="search_kb",
                        failure_mode="timeout",
                    )
                ],
                fallback_actions=[
                    FallbackActionRecord(
                        trigger="tool_timeout",
                        action="switch_tool",
                        target_id="lookup_account",
                    )
                ],
                resumed_after_pause=True,
                success=True,
                quality_score=0.90,
            ),
            EvalRunEvidence(
                run_id="run-billing-dry-001",
                case_id="billing_refund_simple",
                dry_run=True,
                output={
                    "disposition": "resolve",
                    "priority": "medium",
                    "reply_draft": "Previewed refund response without sending it.",
                    "confidence": 0.90,
                    "needs_human": False,
                },
                trace_events=[
                    "run_started",
                    "model_called",
                    "tool_selected",
                    "tool_called",
                    "tool_result",
                    "run_completed",
                ],
                step_count=3,
                latency_ms=1200,
                cost_usd=0.010,
                tool_calls=[
                    ToolCallRecord(
                        order=1,
                        tool_id="lookup_account",
                        args={"ticket_id": "T-100"},
                        status=ToolCallStatus.ok,
                    ),
                    ToolCallRecord(
                        order=2,
                        tool_id="search_kb",
                        args={"query": "double charged invoice refund"},
                        status=ToolCallStatus.ok,
                    ),
                    ToolCallRecord(
                        order=3,
                        tool_id="send_reply",
                        args={
                            "ticket_id": "T-100",
                            "reply_draft": "Previewed refund response without sending it.",
                        },
                        status=ToolCallStatus.preview,
                        idempotency_key="T-100",
                    ),
                ],
                success=True,
                quality_score=0.88,
            ),
            EvalRunEvidence(
                run_id="run-outage-001",
                case_id="enterprise_outage",
                output={
                    "disposition": "escalate",
                    "priority": "urgent",
                    "reply_draft": "We have escalated this outage to the on-call support team.",
                    "confidence": 0.91,
                    "needs_human": True,
                },
                trace_events=[
                    "run_started",
                    "model_called",
                    "tool_selected",
                    "tool_called",
                    "tool_result",
                    "escalated",
                    "run_completed",
                ],
                step_count=3,
                latency_ms=1800,
                cost_usd=0.017,
                tool_calls=[
                    ToolCallRecord(
                        order=1,
                        tool_id="fetch_ticket_context",
                        args={"ticket_id": "T-101"},
                        status=ToolCallStatus.ok,
                    ),
                    ToolCallRecord(
                        order=2,
                        tool_id="create_escalation",
                        args={"ticket_id": "T-101", "reason": "service_outage"},
                        status=ToolCallStatus.ok,
                        idempotency_key="T-101",
                    ),
                    ToolCallRecord(
                        order=3,
                        tool_id="create_escalation",
                        args={"ticket_id": "T-101", "reason": "service_outage"},
                        status=ToolCallStatus.deduplicated,
                        idempotency_key="T-101",
                        deduplicated=True,
                    ),
                ],
                escalations=[EscalationRecord(order=2, reason="ambiguous_request")],
                success=True,
                quality_score=0.91,
            ),
            EvalRunEvidence(
                run_id="run-policy-001",
                case_id="ambiguous_policy_question",
                output={
                    "disposition": "needs_info",
                    "priority": "high",
                    "reply_draft": "I need a human to confirm the regional contract terms before replying.",
                    "confidence": 0.86,
                    "needs_human": True,
                },
                trace_events=[
                    "run_started",
                    "model_called",
                    "tool_selected",
                    "tool_called",
                    "tool_result",
                    "escalated",
                    "run_completed",
                ],
                step_count=2,
                latency_ms=1600,
                cost_usd=0.013,
                tool_calls=[
                    ToolCallRecord(
                        order=1,
                        tool_id="lookup_account",
                        args={"ticket_id": "T-102"},
                        status=ToolCallStatus.ok,
                    )
                ],
                escalations=[EscalationRecord(order=2, reason="ambiguous_request")],
                success=True,
                quality_score=0.87,
            ),
        ],
    )
