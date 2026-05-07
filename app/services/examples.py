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
from app.domain.registry import PackageType, RiskClass, StarterType
from app.domain.task_agent import (
    AgentMode,
    ApprovalPolicy,
    AssessmentStrategySpec,
    BehaviorSpec,
    BudgetPolicy,
    CapabilitySpec,
    CourseStructureSpec,
    CostPerSuccessTestParams,
    DryRunSemanticsTestParams,
    DurableResumeTestParams,
    ExecutionSurface,
    EscalationExpectation,
    EscalationPolicyTestParams,
    EscalationRule,
    EvalDataset,
    FallbackPolicy,
    FallbackPolicyTestParams,
    FallbackStep,
    FaultInjection,
    ModuleSpec,
    OutputSchemaTestParams,
    P95RunLatencyTestParams,
    ProductionContract,
    ProgressionMode,
    QualitySpec,
    RetrievalMode,
    RuntimeDependencySpec,
    SLOProfile,
    StepBudgetEnforcementTestParams,
    TaskAgentServiceSpec,
    TaskEvalCase,
    TaskOutputQualityJudgeTestParams,
    TaskSuccessRateTestParams,
    ToolChoiceExpectation,
    ToolInvocationCorrectnessTestParams,
    ToolInvocationExpectation,
    ToolRegistry,
    ToolSafety,
    ToolSelectionTestParams,
    ToolSpec,
    TraceContract,
    TraceEventType,
    TraceSchemaTestParams,
    ApprovalGateTestParams,
    ApprovalExpectation,
    IdempotentActionTestParams,
    ConfidenceCalibrationJudgeTestParams,
    EscalationPrecisionTestParams,
    WorkspaceScope,
)
from app.services.learner_brief_builder import ensure_task_agent_module_briefs


def get_support_triage_example() -> TaskAgentServiceSpec:
    spec = TaskAgentServiceSpec(
        title="Support Triage Agent",
        summary="Bounded support agent that triages tickets, drafts replies, escalates edge cases, and exposes production controls.",
        package_type=PackageType.progressive_codebase_course,
        risk_class=RiskClass.standard,
        domain_pack="support_triage",
        course_structure=CourseStructureSpec(
            package_type=PackageType.progressive_codebase_course,
            workspace_scope=WorkspaceScope.shared_course_workspace,
            progression_mode=ProgressionMode.independent_modules,
            shared_codebase=True,
        ),
        runtime_dependencies=RuntimeDependencySpec(
            execution_surface=ExecutionSurface.http_service,
            editable_files=["app.py"],
            visible_fixture_files=[],
            local_run_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
            visible_check_command="python checks/run_visible_checks.py",
            preview_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
        ),
        capabilities=CapabilitySpec(
            retrieval_mode=RetrievalMode.none,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=True,
            traceability_required=True,
            durable_state_required=True,
            approval_flow_required=True,
        ),
        assessment_strategy=AssessmentStrategySpec(
            public_checks_required=True,
            hidden_grader_required=True,
            cumulative_module_gates=False,
            learner_submission_enabled=True,
        ),
        supported_modes=[
            AgentMode.tool_using_single_run,
            AgentMode.multi_step_workflow,
            AgentMode.async_human_in_loop,
        ],
        modules=[
            ModuleSpec(
                id="module_1",
                title="Structured output and basic run contract",
                objective="Return valid triage decisions and reply drafts through a stable /run endpoint.",
                starter_type=StarterType.working_buggy,
            ),
            ModuleSpec(
                id="module_2",
                title="Tool selection and invocation correctness",
                objective="Pick the right tools and pass the right arguments before drafting a response.",
                starter_type=StarterType.partial_implementation,
            ),
            ModuleSpec(
                id="module_3",
                title="Multi-step execution and durable state",
                objective="Persist enough state to resume safely when the run pauses for approval.",
                starter_type=StarterType.partial_implementation,
            ),
            ModuleSpec(
                id="module_4",
                title="Escalation and approval gates",
                objective="Know when to escalate and when sending a reply needs approval.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            ModuleSpec(
                id="module_5",
                title="Fallbacks, dry-run, and idempotent actions",
                objective="Recover from tool failures and keep writes safe and repeatable.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            ModuleSpec(
                id="module_6",
                title="Observability and trace completeness",
                objective="Emit enough trace data to inspect every meaningful run decision.",
                starter_type=StarterType.partial_implementation,
                overlay_ids=["productionization_overlay"],
            ),
            ModuleSpec(
                id="module_7",
                title="Eval-driven quality control",
                objective="Use a frozen eval set to improve task success, escalation precision, and output quality.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay"],
            ),
            ModuleSpec(
                id="module_8",
                title="Production final at SLO",
                objective="Meet the full success, latency, cost, and calibration targets together.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay", "scale_slo_overlay"],
            ),
        ],
        task_schema={
            "type": "object",
            "required": ["ticket_id", "customer_message", "account_tier"],
            "properties": {
                "ticket_id": {"type": "string"},
                "customer_message": {"type": "string"},
                "account_tier": {"type": "string", "enum": ["free", "pro", "enterprise"]},
                "dry_run": {"type": "boolean"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["disposition", "priority", "reply_draft", "confidence", "needs_human"],
            "properties": {
                "disposition": {"type": "string", "enum": ["resolve", "escalate", "needs_info"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "reply_draft": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "needs_human": {"type": "boolean"},
            },
        },
        trace_schema={
            "type": "object",
            "required": ["run_id", "events"],
            "properties": {
                "run_id": {"type": "string"},
                "events": {"type": "array"},
            },
        },
        run_state_schema={
            "type": "object",
            "required": ["ticket_id", "step_count", "tool_history"],
            "properties": {
                "ticket_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "tool_history": {"type": "array"},
                "pending_approval": {"type": "boolean"},
            },
        },
        tool_registry=ToolRegistry(
            tools=[
                ToolSpec(
                    id="fetch_ticket_context",
                    description="Fetch ticket metadata and previous interactions.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["ticket_id"]},
                    output_schema={"type": "object"},
                    grader_fixture_id="fixture.fetch_ticket_context",
                ),
                ToolSpec(
                    id="lookup_account",
                    description="Lookup account metadata and entitlements.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["ticket_id"]},
                    output_schema={"type": "object"},
                    grader_fixture_id="fixture.lookup_account",
                ),
                ToolSpec(
                    id="search_kb",
                    description="Search the support knowledge base.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                    grader_fixture_id="fixture.search_kb",
                ),
                ToolSpec(
                    id="create_escalation",
                    description="Open a handoff task for a human support engineer.",
                    safety=ToolSafety.write,
                    input_schema={"type": "object", "required": ["ticket_id", "reason"]},
                    output_schema={"type": "object"},
                    grader_fixture_id="fixture.create_escalation",
                    idempotency_key_arg="ticket_id",
                ),
                ToolSpec(
                    id="send_reply",
                    description="Send the drafted response to the customer.",
                    safety=ToolSafety.irreversible,
                    input_schema={"type": "object", "required": ["ticket_id", "reply_draft"]},
                    output_schema={"type": "object"},
                    grader_fixture_id="fixture.send_reply",
                    approval_required=True,
                    idempotency_key_arg="ticket_id",
                ),
            ],
            max_steps_per_run=6,
            max_parallel_tool_calls=1,
            allow_tool_retries=True,
        ),
        eval_dataset=EvalDataset(
            id="support_triage_eval_v1",
            cases=[
                TaskEvalCase(
                    id="billing_refund_simple",
                    input={
                        "ticket_id": "T-100",
                        "customer_message": "I was double charged on my last invoice.",
                        "account_tier": "pro",
                    },
                    expected_output={"disposition": "resolve"},
                    should_escalate=False,
                    requires_approval=True,
                    must_use_any_of_tools=["lookup_account", "search_kb"],
                ),
                TaskEvalCase(
                    id="enterprise_outage",
                    input={
                        "ticket_id": "T-101",
                        "customer_message": "Our enterprise workspace is down for every user in EMEA.",
                        "account_tier": "enterprise",
                    },
                    expected_output={"disposition": "escalate"},
                    should_escalate=True,
                    requires_approval=False,
                    must_use_any_of_tools=["fetch_ticket_context", "create_escalation"],
                ),
                TaskEvalCase(
                    id="ambiguous_policy_question",
                    input={
                        "ticket_id": "T-102",
                        "customer_message": "Can you confirm whether our contract allows data export to a partner-owned bucket in a different region?",
                        "account_tier": "enterprise",
                    },
                    expected_output={"disposition": "needs_info"},
                    should_escalate=True,
                    requires_approval=False,
                    must_use_any_of_tools=["lookup_account"],
                ),
            ],
        ),
        production_contract=ProductionContract(
            supports_async_runs=True,
            supports_resume=True,
            supports_dry_run=True,
            state_backend="postgres",
            trace_retention_days=30,
            budget_policy=BudgetPolicy(
                max_steps=6,
                max_tool_calls=8,
                max_runtime_ms=20_000,
                max_cost_usd=0.08,
            ),
            approval_policy=ApprovalPolicy(
                require_for_irreversible=True,
                require_for_tools=["send_reply"],
            ),
            escalation_policy=[
                EscalationRule(reason="low_confidence", action="escalate"),
                EscalationRule(reason="ambiguous_request", action="escalate"),
                EscalationRule(reason="tool_failure", action="escalate"),
                EscalationRule(reason="policy_block", action="request_approval"),
            ],
            fallback_policy=FallbackPolicy(
                steps=[
                    FallbackStep(
                        trigger="tool_timeout",
                        action="switch_tool",
                        target_id="lookup_account",
                        max_retries=1,
                    ),
                    FallbackStep(
                        trigger="low_confidence",
                        action="escalate",
                    ),
                ]
            ),
            trace_contract=TraceContract(
                required_events=[
                    TraceEventType.run_started,
                    TraceEventType.model_called,
                    TraceEventType.tool_selected,
                    TraceEventType.tool_called,
                    TraceEventType.tool_result,
                    TraceEventType.run_completed,
                ],
                require_prompt_version=True,
                require_model_name=True,
                require_token_usage=True,
                require_cost_usd=True,
            ),
            slos=SLOProfile(
                p95_run_latency_ms=3_500,
                min_task_success_rate=0.9,
                max_cost_per_success_usd=0.04,
                min_escalation_precision=0.85,
            ),
        ),
        behaviors=[
            BehaviorSpec(
                id="structured_output",
                description="The agent returns a valid structured triage response for each eval case.",
                first_required_in="module_1",
                test=OutputSchemaTestParams(
                    type="output_schema_test",
                    case_ids=["billing_refund_simple", "enterprise_outage", "ambiguous_policy_question"],
                ),
            ),
            BehaviorSpec(
                id="tool_choice_matches_ticket_type",
                description="The agent chooses tools that match the ticket type and avoids irrelevant tools.",
                first_required_in="module_2",
                test=ToolSelectionTestParams(
                    type="tool_selection_test",
                    expectations=[
                        ToolChoiceExpectation(
                            case_id="billing_refund_simple",
                            must_call_any_of=["lookup_account", "search_kb"],
                            must_not_call=["create_escalation"],
                        ),
                        ToolChoiceExpectation(
                            case_id="enterprise_outage",
                            must_call_any_of=["fetch_ticket_context", "create_escalation"],
                        ),
                    ],
                ),
            ),
            BehaviorSpec(
                id="tool_arguments_are_correct",
                description="The agent passes the right argument shape when invoking tools.",
                first_required_in="module_2",
                test=ToolInvocationCorrectnessTestParams(
                    type="tool_invocation_correctness_test",
                    expectations=[
                        ToolInvocationExpectation(
                            case_id="billing_refund_simple",
                            tool_id="lookup_account",
                            required_args_subset={"ticket_id": "T-100"},
                        )
                    ],
                ),
            ),
            BehaviorSpec(
                id="resume_after_approval_pause",
                description="The agent can resume a paused run after waiting for approval.",
                first_required_in="module_3",
                test=DurableResumeTestParams(
                    type="durable_resume_test",
                    case_id="billing_refund_simple",
                    interrupt_after_event=TraceEventType.approval_requested,
                ),
            ),
            BehaviorSpec(
                id="escalate_low_confidence_cases",
                description="The agent escalates low-confidence or ambiguous cases instead of bluffing.",
                first_required_in="module_4",
                test=EscalationPolicyTestParams(
                    type="escalation_policy_test",
                    expectations=[
                        EscalationExpectation(
                            case_id="ambiguous_policy_question",
                            must_escalate=True,
                            allowed_reasons=["low_confidence", "ambiguous_request"],
                        )
                    ],
                ),
            ),
            BehaviorSpec(
                id="approval_before_irreversible_reply",
                description="Sending a customer reply requires approval before the irreversible action fires.",
                first_required_in="module_4",
                test=ApprovalGateTestParams(
                    type="approval_gate_test",
                    expectations=[
                        ApprovalExpectation(
                            case_id="billing_refund_simple",
                            tool_id="send_reply",
                            requires_approval=True,
                        )
                    ],
                ),
            ),
            BehaviorSpec(
                id="fallback_on_tool_failure",
                description="If the KB tool fails, the agent should recover through the declared fallback path.",
                first_required_in="module_5",
                test=FallbackPolicyTestParams(
                    type="fallback_policy_test",
                    injections=[
                        FaultInjection(
                            case_id="billing_refund_simple",
                            target="tool",
                            target_id="search_kb",
                            failure_mode="timeout",
                        )
                    ],
                    min_success_after_fallback=0.66,
                ),
            ),
            BehaviorSpec(
                id="dry_run_blocks_mutations",
                description="Dry-run mode exercises the flow without mutating customer-visible systems.",
                first_required_in="module_5",
                test=DryRunSemanticsTestParams(
                    type="dry_run_semantics_test",
                    case_ids=["billing_refund_simple"],
                    mutating_tool_ids=["create_escalation", "send_reply"],
                ),
            ),
            BehaviorSpec(
                id="idempotent_escalation_creation",
                description="Escalation creation is idempotent across retries for the same ticket.",
                first_required_in="module_5",
                test=IdempotentActionTestParams(
                    type="idempotent_action_test",
                    case_ids=["enterprise_outage"],
                    idempotency_key_field="ticket_id",
                ),
            ),
            BehaviorSpec(
                id="trace_contract_is_complete",
                description="Every meaningful run exposes a complete trace contract for debugging and replay.",
                first_required_in="module_6",
                test=TraceSchemaTestParams(
                    type="trace_schema_test",
                    case_ids=["billing_refund_simple", "enterprise_outage"],
                    required_events=[
                        TraceEventType.run_started,
                        TraceEventType.model_called,
                        TraceEventType.tool_selected,
                        TraceEventType.tool_called,
                        TraceEventType.tool_result,
                        TraceEventType.run_completed,
                    ],
                ),
            ),
            BehaviorSpec(
                id="step_budget_is_enforced",
                description="The agent cannot exceed the configured step budget.",
                first_required_in="module_7",
                test=StepBudgetEnforcementTestParams(
                    type="step_budget_enforcement_test",
                    case_ids=["billing_refund_simple", "enterprise_outage", "ambiguous_policy_question"],
                    max_steps=6,
                ),
            ),
        ],
        qualities=[
            QualitySpec(
                id="success_rate_eval_gate",
                description="The agent meets the minimum success rate on the frozen eval set.",
                first_required_in="module_7",
                test=TaskSuccessRateTestParams(
                    type="task_success_rate_test",
                    dataset_id="support_triage_eval_v1",
                    min_success_rate=0.8,
                ),
            ),
            QualitySpec(
                id="escalation_precision_eval_gate",
                description="Escalations should be meaningfully precise rather than overused.",
                first_required_in="module_7",
                test=EscalationPrecisionTestParams(
                    type="escalation_precision_test",
                    dataset_id="support_triage_eval_v1",
                    min_precision=0.75,
                ),
            ),
            QualitySpec(
                id="task_output_quality_final",
                description="The final outputs should meet the quality rubric on average.",
                first_required_in="module_8",
                test=TaskOutputQualityJudgeTestParams(
                    type="task_output_quality_judge_test",
                    dataset_id="support_triage_eval_v1",
                    judge_id="support_triage_quality_v1",
                    rubric_id="support_triage_rubric_v1",
                    min_avg_score=0.85,
                ),
            ),
            QualitySpec(
                id="latency_final",
                description="The final service meets the p95 latency target.",
                first_required_in="module_8",
                test=P95RunLatencyTestParams(
                    type="p95_run_latency_test",
                    dataset_id="support_triage_eval_v1",
                    concurrency=8,
                    p95_ms=3500,
                ),
            ),
            QualitySpec(
                id="cost_final",
                description="The final service stays within the cost budget per success.",
                first_required_in="module_8",
                test=CostPerSuccessTestParams(
                    type="cost_per_success_test",
                    dataset_id="support_triage_eval_v1",
                    max_cost_usd=0.04,
                ),
            ),
            QualitySpec(
                id="confidence_calibration_final",
                description="Reported confidence should be reasonably calibrated.",
                first_required_in="module_8",
                test=ConfidenceCalibrationJudgeTestParams(
                    type="confidence_calibration_judge_test",
                    dataset_id="support_triage_eval_v1",
                    max_expected_calibration_error=0.15,
                ),
            ),
        ],
    )
    return ensure_task_agent_module_briefs(spec, overwrite=True)


def get_support_triage_passing_submission() -> TaskAgentSubmission:
    return TaskAgentSubmission(
        submission_id="support_triage_passing_submission_v1",
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
