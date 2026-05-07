from __future__ import annotations

import re

from app.domain.registry import PackageType, RiskClass, StarterType
from app.domain.task_agent import (
    AgentMode,
    AssignmentDesignSpec,
    ApprovalPolicy,
    BehaviorSpec,
    BudgetPolicy,
    CapabilitySpec,
    ConfidenceCalibrationJudgeTestParams,
    CourseStructureSpec,
    CostPerSuccessTestParams,
    DryRunSemanticsTestParams,
    ExecutionSurface,
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
    ProgressionMode,
    ProductionContract,
    QualitySpec,
    RetrievalMode,
    RuntimeDependencySpec,
    SLOProfile,
    AssessmentStrategySpec,
    TaskAgentServiceSpec,
    TaskEvalCase,
    TaskOutputQualityJudgeTestParams,
    TaskSuccessRateTestParams,
    ToolChoiceExpectation,
    ToolRegistry,
    ToolSafety,
    ToolSelectionTestParams,
    ToolSpec,
    TraceContract,
    TraceEventType,
    TraceSchemaTestParams,
    WorkspaceScope,
)
from app.services.examples import get_support_triage_example
from app.services.learner_brief_builder import ensure_task_agent_module_briefs


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "agent_assignment"


def _apply_overlays(spec: TaskAgentServiceSpec, overlays: list[str]) -> TaskAgentServiceSpec:
    if not overlays:
        return spec
    for module in spec.modules:
        overlay_ids = list(module.overlay_ids)
        if module.id == spec.modules[-1].id:
            for overlay in overlays:
                if overlay not in overlay_ids:
                    overlay_ids.append(overlay)
            module.overlay_ids = overlay_ids
    return spec


def _compress_to_survey(spec: TaskAgentServiceSpec) -> TaskAgentServiceSpec:
    mapping = {
        spec.modules[0].id: "module_1",
        spec.modules[1].id: "module_2",
    }
    for module in spec.modules[2:]:
        mapping[module.id] = "module_3"

    spec.modules = [
        ModuleSpec(
            id="module_1",
            title="Core contract",
            objective="Return structured outputs through a stable run contract.",
            starter_type=StarterType.working_buggy,
        ),
        ModuleSpec(
            id="module_2",
            title="Tool use and control flow",
            objective="Use tools correctly and enforce the core safety policies.",
            starter_type=StarterType.partial_implementation,
            overlay_ids=["productionization_overlay"],
        ),
        ModuleSpec(
            id="module_3",
            title="Final quality bar",
            objective="Meet the final task-success, latency, and quality targets.",
            starter_type=StarterType.working_suboptimal,
            overlay_ids=["productionization_overlay", "scale_slo_overlay"],
        ),
    ]

    for behavior in spec.behaviors:
        behavior.first_required_in = mapping[behavior.first_required_in]
    for quality in spec.qualities:
        quality.first_required_in = "module_3"
    return spec


def _build_generic_task_agent_scaffold(
    *,
    title: str,
    summary: str,
    risk_class: RiskClass,
    domain_pack: str | None,
) -> tuple[TaskAgentServiceSpec, str]:
    if domain_pack == "support_triage":
        return get_support_triage_example().model_copy(deep=True), "support_triage"

    slug = _slugify(title)
    spec = TaskAgentServiceSpec(
        title=title,
        summary=summary,
        package_type=PackageType.progressive_codebase_course,
        risk_class=risk_class,
        domain_pack=domain_pack,
        course_structure=CourseStructureSpec(
            package_type=PackageType.progressive_codebase_course,
            workspace_scope=WorkspaceScope.shared_course_workspace,
            progression_mode=ProgressionMode.cumulative_module_gates,
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
            cumulative_module_gates=True,
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
                title="Structured output and run contract",
                objective="Return the expected structured output for a bounded task.",
                starter_type=StarterType.working_buggy,
            ),
            ModuleSpec(
                id="module_2",
                title="Tool selection and safe action routing",
                objective="Choose the right tools and keep writes behind a clear policy boundary.",
                starter_type=StarterType.partial_implementation,
            ),
            ModuleSpec(
                id="module_3",
                title="Fallbacks, approvals, and traceability",
                objective="Recover from failures and expose a replayable run trace.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            ModuleSpec(
                id="module_4",
                title="Production final",
                objective="Meet the quality, latency, and calibration targets together.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay", "scale_slo_overlay"],
            ),
        ],
        task_schema={
            "type": "object",
            "required": ["request_id", "task_input"],
            "properties": {
                "request_id": {"type": "string"},
                "task_input": {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["result", "confidence", "needs_human"],
            "properties": {
                "result": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "needs_human": {"type": "boolean"},
            },
        },
        trace_schema={
            "type": "object",
            "required": ["run_id", "events"],
            "properties": {"run_id": {"type": "string"}, "events": {"type": "array"}},
        },
        run_state_schema={
            "type": "object",
            "required": ["request_id", "step_count"],
            "properties": {
                "request_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "pending_approval": {"type": "boolean"},
            },
        },
        tool_registry=ToolRegistry(
            tools=[
                ToolSpec(
                    id="lookup_context",
                    description="Fetch structured context for the current task.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.lookup_context",
                ),
                ToolSpec(
                    id="search_knowledge",
                    description="Search a bounded knowledge source.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.search_knowledge",
                ),
                ToolSpec(
                    id="perform_action",
                    description="Take a reversible or internal write action.",
                    safety=ToolSafety.write,
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.perform_action",
                    idempotency_key_arg="request_id",
                ),
                ToolSpec(
                    id="send_final_output",
                    description="Trigger the irreversible external side effect.",
                    safety=ToolSafety.irreversible,
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.send_final_output",
                    approval_required=True,
                    idempotency_key_arg="request_id",
                ),
            ]
        ),
        eval_dataset=EvalDataset(
            id=f"{slug}_eval_v1",
            cases=[
                TaskEvalCase(
                    id="happy_path",
                    input={"request_id": "REQ-1", "task_input": "Handle the routine case cleanly."},
                    expected_output={"needs_human": False},
                ),
                TaskEvalCase(
                    id="escalation_case",
                    input={"request_id": "REQ-2", "task_input": "Handle the ambiguous or risky case."},
                    expected_output={"needs_human": True},
                    should_escalate=True,
                    requires_approval=True,
                ),
            ],
        ),
        production_contract=ProductionContract(
            supports_async_runs=True,
            supports_resume=True,
            supports_dry_run=True,
            state_backend="postgres",
            budget_policy=BudgetPolicy(
                max_steps=5,
                max_tool_calls=6,
                max_runtime_ms=15_000,
                max_cost_usd=0.05,
            ),
            approval_policy=ApprovalPolicy(
                require_for_irreversible=True,
                require_for_tools=["send_final_output"],
            ),
            escalation_policy=[
                EscalationRule(reason="low_confidence", action="escalate"),
                EscalationRule(reason="ambiguous_request", action="escalate"),
                EscalationRule(reason="policy_block", action="request_approval"),
            ],
            fallback_policy=FallbackPolicy(
                steps=[
                    FallbackStep(trigger="tool_timeout", action="switch_tool", target_id="lookup_context"),
                    FallbackStep(trigger="low_confidence", action="escalate"),
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
                ]
            ),
            slos=SLOProfile(
                p95_run_latency_ms=2500,
                min_task_success_rate=0.85,
                max_cost_per_success_usd=0.03,
                min_escalation_precision=0.8,
            ),
        ),
        behaviors=[
            BehaviorSpec(
                id="structured_output",
                description="The agent returns the expected structured result.",
                first_required_in="module_1",
                test=OutputSchemaTestParams(type="output_schema_test", case_ids=["happy_path", "escalation_case"]),
            ),
            BehaviorSpec(
                id="tool_selection",
                description="The agent chooses the bounded toolset appropriately.",
                first_required_in="module_2",
                test=ToolSelectionTestParams(
                    type="tool_selection_test",
                    expectations=[
                        ToolChoiceExpectation(case_id="happy_path", must_call_any_of=["lookup_context", "search_knowledge"]),
                        ToolChoiceExpectation(case_id="escalation_case", must_call_any_of=["lookup_context", "perform_action"]),
                    ],
                ),
            ),
            BehaviorSpec(
                id="escalation_policy",
                description="The agent escalates ambiguous cases rather than bluffing.",
                first_required_in="module_3",
                test=EscalationPolicyTestParams(
                    type="escalation_policy_test",
                    expectations=[
                        {"case_id": "escalation_case", "must_escalate": True, "allowed_reasons": ["low_confidence", "ambiguous_request"]}
                    ],
                ),
            ),
            BehaviorSpec(
                id="fallbacks_and_dry_run",
                description="The agent recovers from tool failures and avoids mutations in dry-run mode.",
                first_required_in="module_3",
                test=FallbackPolicyTestParams(
                    type="fallback_policy_test",
                    injections=[
                        FaultInjection(case_id="happy_path", target="tool", target_id="search_knowledge", failure_mode="timeout")
                    ],
                    min_success_after_fallback=0.5,
                ),
            ),
            BehaviorSpec(
                id="dry_run_semantics",
                description="Dry-run mode blocks mutating side effects.",
                first_required_in="module_3",
                test=DryRunSemanticsTestParams(
                    type="dry_run_semantics_test",
                    case_ids=["happy_path"],
                    mutating_tool_ids=["perform_action", "send_final_output"],
                ),
            ),
        ],
        qualities=[
            QualitySpec(
                id="task_success_final",
                description="The agent meets the target success rate.",
                first_required_in="module_4",
                test=TaskSuccessRateTestParams(
                    type="task_success_rate_test",
                    dataset_id=f"{slug}_eval_v1",
                    min_success_rate=0.85,
                ),
            ),
            QualitySpec(
                id="quality_final",
                description="The agent output quality meets the rubric.",
                first_required_in="module_4",
                test=TaskOutputQualityJudgeTestParams(
                    type="task_output_quality_judge_test",
                    dataset_id=f"{slug}_eval_v1",
                    judge_id=f"{slug}_quality_judge_v1",
                    rubric_id=f"{slug}_rubric_v1",
                    min_avg_score=0.8,
                ),
            ),
            QualitySpec(
                id="latency_final",
                description="The agent meets the p95 latency target.",
                first_required_in="module_4",
                test=P95RunLatencyTestParams(
                    type="p95_run_latency_test",
                    dataset_id=f"{slug}_eval_v1",
                    concurrency=4,
                    p95_ms=2500,
                ),
            ),
            QualitySpec(
                id="cost_final",
                description="The agent stays inside the cost budget.",
                first_required_in="module_4",
                test=CostPerSuccessTestParams(
                    type="cost_per_success_test",
                    dataset_id=f"{slug}_eval_v1",
                    max_cost_usd=0.03,
                ),
            ),
            QualitySpec(
                id="confidence_final",
                description="The agent confidence is reasonably calibrated.",
                first_required_in="module_4",
                test=ConfidenceCalibrationJudgeTestParams(
                    type="confidence_calibration_judge_test",
                    dataset_id=f"{slug}_eval_v1",
                    max_expected_calibration_error=0.2,
                ),
            ),
        ],
    )
    return spec, "generic_task_agent"


def _build_grounded_rag_scaffold(
    *,
    title: str,
    summary: str,
    risk_class: RiskClass,
) -> tuple[TaskAgentServiceSpec, str]:
    slug = _slugify(title)
    spec = TaskAgentServiceSpec(
        title=title,
        summary=summary,
        package_type=PackageType.progressive_codebase_course,
        risk_class=risk_class,
        course_structure=CourseStructureSpec(
            package_type=PackageType.progressive_codebase_course,
            workspace_scope=WorkspaceScope.shared_course_workspace,
            progression_mode=ProgressionMode.cumulative_module_gates,
            shared_codebase=True,
        ),
        runtime_dependencies=RuntimeDependencySpec(
            execution_surface=ExecutionSurface.http_service,
            editable_files=["app.py"],
            visible_fixture_files=["data/corpus.json"],
            local_run_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
            visible_check_command="python checks/run_visible_checks.py",
            preview_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
        ),
        capabilities=CapabilitySpec(
            retrieval_mode=RetrievalMode.grounded_answers,
            answer_synthesis_required=True,
            citations_required=True,
            abstention_required=True,
            tool_use_required=True,
            traceability_required=True,
            durable_state_required=False,
            approval_flow_required=False,
        ),
        assessment_strategy=AssessmentStrategySpec(
            public_checks_required=True,
            hidden_grader_required=True,
            cumulative_module_gates=True,
            learner_submission_enabled=True,
        ),
        supported_modes=[
            AgentMode.routed_single_step,
            AgentMode.retrieval_plus_action,
        ],
        modules=[
            ModuleSpec(
                id="module_1",
                title="Grounded answer contract and citation schema",
                objective="Return grounded answers with supporting citations through a stable /run endpoint.",
                starter_type=StarterType.working_buggy,
            ),
            ModuleSpec(
                id="module_2",
                title="Retrieval selection and evidence ranking",
                objective="Use the retrieval tools to surface the strongest supporting evidence before answering.",
                starter_type=StarterType.partial_implementation,
            ),
            ModuleSpec(
                id="module_3",
                title="Abstention and traceable retrieval flow",
                objective="Abstain when evidence is weak and expose a readable retrieval trace.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            ModuleSpec(
                id="module_4",
                title="Production final at groundedness bar",
                objective="Meet the groundedness, latency, and operating-cost targets together.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay", "scale_slo_overlay"],
            ),
        ],
        task_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
            },
        },
        output_schema={
            "type": "object",
            "required": ["answer", "citations", "confidence", "abstained"],
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "abstained": {"type": "boolean"},
            },
        },
        trace_schema={
            "type": "object",
            "required": ["run_id", "events"],
            "properties": {"run_id": {"type": "string"}, "events": {"type": "array"}},
        },
        run_state_schema={
            "type": "object",
            "required": ["query", "retrieval_count"],
            "properties": {
                "query": {"type": "string"},
                "retrieval_count": {"type": "integer"},
                "selected_doc_ids": {"type": "array"},
            },
        },
        tool_registry=ToolRegistry(
            tools=[
                ToolSpec(
                    id="search_corpus",
                    description="Search the visible corpus for passages relevant to the question.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.search_corpus",
                ),
                ToolSpec(
                    id="fetch_document",
                    description="Fetch a document body by document id so the answer can cite grounded support.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["doc_id"]},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.fetch_document",
                ),
                ToolSpec(
                    id="rerank_passages",
                    description="Reorder retrieved passages so the strongest support appears first.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.rerank_passages",
                ),
            ],
            max_steps_per_run=4,
            max_parallel_tool_calls=1,
            allow_tool_retries=True,
        ),
        eval_dataset=EvalDataset(
            id=f"{slug}_eval_v1",
            cases=[
                TaskEvalCase(
                    id="ada_birth",
                    input={"query": "Where was Ada Lovelace born?"},
                    expected_output={
                        "answer": "Ada Lovelace was born in London, England.",
                        "citations": ["doc:ada_lovelace"],
                        "abstained": False,
                    },
                    must_use_any_of_tools=["search_corpus", "fetch_document"],
                ),
                TaskEvalCase(
                    id="turing_role",
                    input={"query": "What was Alan Turing known for?"},
                    expected_output={
                        "answer": "Alan Turing was an English mathematician and computer scientist.",
                        "citations": ["doc:alan_turing"],
                        "abstained": False,
                    },
                    must_use_any_of_tools=["search_corpus", "rerank_passages"],
                ),
                TaskEvalCase(
                    id="unsupported_lunar_policy",
                    input={"query": "What is the maintenance window for the lunar base policy manual?"},
                    expected_output={
                        "answer": "I do not have enough grounded support in the corpus to answer that question.",
                        "citations": [],
                        "abstained": True,
                    },
                    must_use_any_of_tools=["search_corpus"],
                ),
            ],
        ),
        production_contract=ProductionContract(
            supports_async_runs=False,
            supports_resume=False,
            supports_dry_run=False,
            state_backend="sqlite",
            budget_policy=BudgetPolicy(
                max_steps=4,
                max_tool_calls=4,
                max_runtime_ms=10_000,
                max_cost_usd=0.02,
            ),
            approval_policy=ApprovalPolicy(
                require_for_irreversible=False,
                require_for_tools=[],
            ),
            escalation_policy=[],
            fallback_policy=FallbackPolicy(
                steps=[
                    FallbackStep(trigger="tool_timeout", action="switch_tool", target_id="search_corpus"),
                    FallbackStep(trigger="low_confidence", action="return_partial"),
                ]
            ),
            trace_contract=TraceContract(
                required_events=[
                    TraceEventType.run_started,
                    TraceEventType.tool_selected,
                    TraceEventType.tool_called,
                    TraceEventType.tool_result,
                    TraceEventType.run_completed,
                ]
            ),
            slos=SLOProfile(
                p95_run_latency_ms=1800,
                max_cost_per_success_usd=0.02,
            ),
        ),
        behaviors=[
            BehaviorSpec(
                id="grounded_answer_contract",
                description="The service returns grounded answers with the required citation schema.",
                first_required_in="module_1",
                test=OutputSchemaTestParams(
                    type="output_schema_test",
                    case_ids=["ada_birth", "turing_role", "unsupported_lunar_policy"],
                ),
            ),
            BehaviorSpec(
                id="retrieval_tool_selection",
                description="The service uses retrieval tools before answering supported questions.",
                first_required_in="module_2",
                test=ToolSelectionTestParams(
                    type="tool_selection_test",
                    expectations=[
                        ToolChoiceExpectation(case_id="ada_birth", must_call_any_of=["search_corpus", "fetch_document"]),
                        ToolChoiceExpectation(case_id="turing_role", must_call_any_of=["search_corpus", "rerank_passages"]),
                        ToolChoiceExpectation(case_id="unsupported_lunar_policy", must_call_any_of=["search_corpus"]),
                    ],
                ),
            ),
            BehaviorSpec(
                id="retrieval_trace_completeness",
                description="The service emits a readable retrieval trace for the grounded answer flow.",
                first_required_in="module_3",
                test=TraceSchemaTestParams(
                    type="trace_schema_test",
                    case_ids=["ada_birth", "unsupported_lunar_policy"],
                    required_events=[
                        TraceEventType.run_started,
                        TraceEventType.tool_called,
                        TraceEventType.run_completed,
                    ],
                ),
            ),
        ],
        qualities=[
            QualitySpec(
                id="latency_final",
                description="The service meets the p95 latency target.",
                first_required_in="module_4",
                test=P95RunLatencyTestParams(
                    type="p95_run_latency_test",
                    dataset_id=f"{slug}_eval_v1",
                    concurrency=3,
                    p95_ms=1800,
                ),
            ),
            QualitySpec(
                id="cost_final",
                description="The service stays inside the cost budget.",
                first_required_in="module_4",
                test=CostPerSuccessTestParams(
                    type="cost_per_success_test",
                    dataset_id=f"{slug}_eval_v1",
                    max_cost_usd=0.02,
                ),
            ),
        ],
    )
    return spec, "generic_grounded_rag"


def _build_retrieval_scaffold(
    *,
    title: str,
    summary: str,
    risk_class: RiskClass,
) -> tuple[TaskAgentServiceSpec, str]:
    slug = _slugify(title)
    spec = TaskAgentServiceSpec(
        title=title,
        summary=summary,
        package_type=PackageType.progressive_codebase_course,
        risk_class=risk_class,
        course_structure=CourseStructureSpec(
            package_type=PackageType.progressive_codebase_course,
            workspace_scope=WorkspaceScope.shared_course_workspace,
            progression_mode=ProgressionMode.cumulative_module_gates,
            shared_codebase=True,
        ),
        runtime_dependencies=RuntimeDependencySpec(
            execution_surface=ExecutionSurface.http_service,
            editable_files=["app.py"],
            visible_fixture_files=["data/corpus.json"],
            local_run_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
            visible_check_command="python checks/run_visible_checks.py",
            preview_command="python -m uvicorn app:app --host 127.0.0.1 --port 8000",
        ),
        capabilities=CapabilitySpec(
            retrieval_mode=RetrievalMode.ranked_results,
            answer_synthesis_required=False,
            citations_required=False,
            abstention_required=False,
            tool_use_required=False,
            traceability_required=True,
            durable_state_required=False,
            approval_flow_required=False,
        ),
        assessment_strategy=AssessmentStrategySpec(
            public_checks_required=True,
            hidden_grader_required=True,
            cumulative_module_gates=True,
            learner_submission_enabled=True,
        ),
        supported_modes=[AgentMode.routed_single_step],
        modules=[
            ModuleSpec(
                id="module_1",
                title="Retrieval contract and result schema",
                objective="Return ranked search results through a stable /run endpoint.",
                starter_type=StarterType.working_buggy,
            ),
            ModuleSpec(
                id="module_2",
                title="Ranking quality and metadata filters",
                objective="Improve ordering quality and respect simple filter constraints.",
                starter_type=StarterType.partial_implementation,
            ),
            ModuleSpec(
                id="module_3",
                title="Traceable retrieval flow",
                objective="Expose retrieval traces that explain why documents were returned.",
                starter_type=StarterType.working_buggy,
                overlay_ids=["productionization_overlay"],
            ),
            ModuleSpec(
                id="module_4",
                title="Production retrieval final",
                objective="Meet quality, latency, and operating-cost targets together.",
                starter_type=StarterType.working_suboptimal,
                overlay_ids=["productionization_overlay", "scale_slo_overlay"],
            ),
        ],
        task_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
                "filters": {"type": "object"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                }
            },
        },
        trace_schema={
            "type": "object",
            "required": ["run_id", "events"],
            "properties": {"run_id": {"type": "string"}, "events": {"type": "array"}},
        },
        run_state_schema={
            "type": "object",
            "required": ["query", "retrieval_steps"],
            "properties": {
                "query": {"type": "string"},
                "retrieval_steps": {"type": "integer"},
            },
        },
        tool_registry=ToolRegistry(
            tools=[
                ToolSpec(
                    id="search_corpus",
                    description="Search the indexed corpus for candidate matches.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["query"]},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.search_corpus",
                ),
                ToolSpec(
                    id="rerank_passages",
                    description="Improve the ranking order of candidate results.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["query", "candidate_ids"]},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.rerank_passages",
                ),
                ToolSpec(
                    id="fetch_document",
                    description="Load the document body for a ranked result.",
                    safety=ToolSafety.read,
                    input_schema={"type": "object", "required": ["doc_id"]},
                    output_schema={"type": "object"},
                    grader_fixture_id=f"{slug}.fetch_document",
                ),
            ]
        ),
        eval_dataset=EvalDataset(
            id=f"{slug}_eval_v1",
            cases=[
                TaskEvalCase(
                    id="ada_birth",
                    input={"query": "When was Ada Lovelace born?", "top_k": 3},
                    expected_output={},
                    must_use_any_of_tools=["search_corpus"],
                ),
                TaskEvalCase(
                    id="turing_role",
                    input={"query": "What role did Alan Turing play at Bletchley Park?", "top_k": 3},
                    expected_output={},
                    must_use_any_of_tools=["search_corpus", "rerank_passages"],
                ),
            ],
        ),
        production_contract=ProductionContract(
            supports_async_runs=False,
            supports_resume=False,
            supports_dry_run=False,
            state_backend="memory",
            budget_policy=BudgetPolicy(
                max_steps=4,
                max_tool_calls=5,
                max_runtime_ms=10_000,
                max_cost_usd=0.02,
            ),
            approval_policy=ApprovalPolicy(require_for_irreversible=True),
            escalation_policy=[],
            fallback_policy=FallbackPolicy(
                steps=[FallbackStep(trigger="tool_timeout", action="return_partial")]
            ),
            trace_contract=TraceContract(
                required_events=[
                    TraceEventType.run_started,
                    TraceEventType.tool_called,
                    TraceEventType.run_completed,
                ]
            ),
            slos=SLOProfile(
                p95_run_latency_ms=1800,
                min_task_success_rate=0.85,
                max_cost_per_success_usd=0.02,
            ),
        ),
        behaviors=[
            BehaviorSpec(
                id="ranked_results_schema",
                description="The service returns a stable ranked results payload.",
                first_required_in="module_1",
                test=OutputSchemaTestParams(type="output_schema_test", case_ids=["ada_birth", "turing_role"]),
            ),
            BehaviorSpec(
                id="retrieval_tool_selection",
                description="The service uses retrieval tools before returning results.",
                first_required_in="module_2",
                test=ToolSelectionTestParams(
                    type="tool_selection_test",
                    expectations=[
                        ToolChoiceExpectation(case_id="ada_birth", must_call_any_of=["search_corpus", "fetch_document"]),
                        ToolChoiceExpectation(case_id="turing_role", must_call_any_of=["search_corpus", "rerank_passages"]),
                    ],
                ),
            ),
            BehaviorSpec(
                id="retrieval_trace_completeness",
                description="The service emits a readable retrieval trace.",
                first_required_in="module_3",
                test=TraceSchemaTestParams(
                    type="trace_schema_test",
                    case_ids=["ada_birth", "turing_role"],
                    required_events=[
                        TraceEventType.run_started,
                        TraceEventType.tool_called,
                        TraceEventType.run_completed,
                    ],
                ),
            ),
        ],
        qualities=[
            QualitySpec(
                id="latency_final",
                description="The service meets the p95 latency target.",
                first_required_in="module_4",
                test=P95RunLatencyTestParams(
                    type="p95_run_latency_test",
                    dataset_id=f"{slug}_eval_v1",
                    concurrency=3,
                    p95_ms=1800,
                ),
            ),
            QualitySpec(
                id="cost_final",
                description="The service stays inside the cost budget.",
                first_required_in="module_4",
                test=CostPerSuccessTestParams(
                    type="cost_per_success_test",
                    dataset_id=f"{slug}_eval_v1",
                    max_cost_usd=0.02,
                ),
            ),
        ],
    )
    return spec, "generic_retrieval_service"


def build_task_agent_scaffold(
    *,
    title: str,
    summary: str,
    design_spec: AssignmentDesignSpec,
) -> tuple[TaskAgentServiceSpec, str]:
    design = design_spec

    if design.capabilities.is_grounded_answer_system:
        spec, origin_template = _build_grounded_rag_scaffold(
            title=title,
            summary=summary,
            risk_class=design.risk_class,
        )
    elif design.capabilities.retrieval_mode == RetrievalMode.ranked_results:
        spec, origin_template = _build_retrieval_scaffold(
            title=title,
            summary=summary,
            risk_class=design.risk_class,
        )
    else:
        spec, origin_template = _build_generic_task_agent_scaffold(
            title=title,
            summary=summary,
            risk_class=design.risk_class,
            domain_pack=design.domain_pack,
        )

    spec.title = title
    spec.summary = summary
    spec.package_type = design.course_structure.package_type
    spec.course_structure = design.course_structure
    spec.runtime_dependencies = design.runtime_dependencies
    spec.capabilities = design.capabilities
    spec.assessment_strategy = design.assessment_strategy
    spec.risk_class = design.risk_class
    spec.domain_pack = design.domain_pack
    spec.eval_dataset.id = f"{_slugify(title)}_eval_v1"

    for quality in spec.qualities:
        if hasattr(quality.test, "dataset_id"):
            quality.test.dataset_id = spec.eval_dataset.id

    spec = _apply_overlays(spec, design.overlays)
    if design.course_structure.package_type == PackageType.survey_course:
        spec = _compress_to_survey(spec)

    spec = spec.model_copy(
        update={
            "summary": summary,
            "title": title,
            "domain_pack": design.domain_pack,
            "overlays": list(design.overlays),
            "risk_class": design.risk_class,
            "course_structure": design.course_structure,
            "runtime_dependencies": design.runtime_dependencies,
            "capabilities": design.capabilities,
            "assessment_strategy": design.assessment_strategy,
            "package_type": design.course_structure.package_type,
        }
    )
    return ensure_task_agent_module_briefs(spec, overwrite=True), origin_template
