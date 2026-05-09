from __future__ import annotations

import json
import os
import tempfile
import time
import unittest

from app.domain.ai import AIUsageSummary
from app.domain.task_agent import AssignmentDesignSpec, EndpointSpec
from app.domain.workflow import (
    DecisionOutcome,
    GateDecisionRequest,
    HILGate,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowNodeStatus,
    WorkflowStage,
    WorkflowStatus,
)
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.task_agent_contract_surface import primary_submit_endpoint
from app.services.openai_task_agent_authoring import (
    OpenAITaskAgentAuthoringService,
    StarterScenarioCustomization,
    StarterSurfaceCustomization,
    TaskAgentCustomization,
    TaskAgentAuthoringResult,
    TaskAgentAuthoringSource,
    TaskAgentAuthoringStatus,
    DeliverableCustomization,
    EvalCaseCustomization,
)
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.services.task_agent_starter_templates import render_task_agent_runtime_entrypoint
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs
from app.services.spec_validation import validate_task_agent_spec
from app.services.workflow_service import WorkflowConflictError, WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _grounded_design_spec() -> AssignmentDesignSpec:
    inferred = infer_assignment_design(
        title="Build a Grounded Internal Docs Assistant",
        problem_statement=(
            "Build a backend service that answers internal docs questions from a visible corpus, "
            "returns citations, and abstains when the evidence is weak."
        ),
    )
    assert inferred.design_spec is not None
    return inferred.design_spec


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.output_text = json.dumps(payload)
        self.usage = None


class _TimeoutThenSuccessClient:
    def __init__(self) -> None:
        self.calls = 0
        self.responses = self

    def create(self, **_: object) -> _FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("Request timed out.")
        return _FakeResponse(
            {
                "summary": "Customized after a retry.",
                "canonical_endpoints": [
                    {"method": "POST", "path": "/questions/query", "required": True},
                    {"method": "GET", "path": "/health", "required": True},
                ],
                "deliverables": [],
                "tools": [],
                "eval_cases": [],
                "notes": ["retried successfully"],
            }
        )


class _BlockingClient:
    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s
        self.responses = self

    def create(self, **_: object) -> _FakeResponse:
        time.sleep(self.sleep_s)
        return _FakeResponse(
            {
                "summary": "This response arrived too late to be useful.",
                "canonical_endpoints": [
                    {"method": "POST", "path": "/late", "required": True},
                    {"method": "GET", "path": "/health", "required": True},
                ],
                "deliverables": [],
                "tools": [],
                "eval_cases": [],
                "notes": ["late response"],
            }
        )


class _FailedLiveAuthoringService:
    def status(self) -> TaskAgentAuthoringStatus:
        return TaskAgentAuthoringStatus(
            available=False,
            source=TaskAgentAuthoringSource.deterministic_fallback,
            message=(
                "OpenAI task-agent authoring failed and fell back to the deterministic starter template: "
                "Request timed out."
            ),
            sdk_installed=True,
            api_key_present=True,
            model_id="gpt-5.4",
            env_file="/tmp/fake-openai.env",
        )

    def generate_scaffold(self, *, title, summary, design_spec) -> TaskAgentAuthoringResult:
        spec, origin_template = build_task_agent_scaffold(
            title=title,
            summary=summary,
            design_spec=design_spec,
        )
        return TaskAgentAuthoringResult(
            spec=spec,
            origin_template=origin_template,
            source=TaskAgentAuthoringSource.deterministic_fallback,
            notes=[self.status().message],
            status=self.status(),
            usage=AIUsageSummary(),
        )


class _InvalidRevisionAuthoringService:
    def __init__(self, spec) -> None:  # noqa: ANN001
        self.spec = spec

    def revise_spec(self, **kwargs):  # noqa: ANN001
        invalid = self.spec.model_copy(deep=True)
        invalid.runtime_dependencies.editable_files = []
        invalid.deliverables[0].learner_brief = None
        status = TaskAgentAuthoringStatus(
            available=True,
            source=TaskAgentAuthoringSource.openai_live,
            message="invalid reviser",
            sdk_installed=True,
            api_key_present=True,
            model_id="fake-model",
            env_file=None,
        )
        return TaskAgentAuthoringResult(
            spec=invalid,
            origin_template="openai_revision:task_agent_spec",
            source=TaskAgentAuthoringSource.openai_live,
            notes=["returned an invalid revision"],
            status=status,
            usage=AIUsageSummary(),
        )


class _AlwaysReadyAuthoringService(OpenAITaskAgentAuthoringService):
    def status(self) -> TaskAgentAuthoringStatus:
        return TaskAgentAuthoringStatus(
            available=True,
            source=TaskAgentAuthoringSource.openai_live,
            message="OpenAI task-agent authoring is ready to customize task-agent specs.",
            sdk_installed=True,
            api_key_present=True,
            model_id="gpt-5.4",
            env_file="/tmp/fake-openai.env",
            customization_validation_rejection_count=self._customization_validation_rejection_count,
            last_customization_validation_error=self._last_customization_validation_error,
        )


class AuthoringResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        if self.previous_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self.previous_api_key

    def test_generate_scaffold_retries_transient_timeout(self) -> None:
        fake_client = _TimeoutThenSuccessClient()
        service = OpenAITaskAgentAuthoringService(
            client_factory=lambda api_key, base_url: fake_client,
            request_timeout_s=0.1,
            max_request_retries=1,
        )

        result = service.generate_scaffold(
            title="Build a Grounded Internal Docs Assistant",
            summary="Answer docs questions with citations and abstention.",
            design_spec=_grounded_design_spec(),
        )

        self.assertEqual(fake_client.calls, 2)
        self.assertEqual(result.source, TaskAgentAuthoringSource.openai_live)
        self.assertEqual(result.spec.summary, "Customized after a retry.")

    def test_generate_scaffold_hard_times_out_blocking_live_request(self) -> None:
        blocking_client = _BlockingClient(sleep_s=0.2)
        service = OpenAITaskAgentAuthoringService(
            client_factory=lambda api_key, base_url: blocking_client,
            request_timeout_s=0.05,
            max_request_retries=0,
        )

        started_at = time.time()
        result = service.generate_scaffold(
            title="Build a Grounded Internal Docs Assistant",
            summary="Answer docs questions with citations and abstention.",
            design_spec=_grounded_design_spec(),
        )
        elapsed_s = time.time() - started_at

        self.assertLess(elapsed_s, 0.2)
        self.assertEqual(result.source, TaskAgentAuthoringSource.deterministic_fallback)
        self.assertIn("exceeded 0.1s", result.status.message)

    def test_workflow_blocks_early_when_live_authoring_falls_back(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        try:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir.name}/test.db")
            workflow_service = WorkflowService(
                store,
                materializer=ArtifactMaterializer(base_dir=f"{temp_dir.name}/generated"),
                task_agent_authoring_service=_FailedLiveAuthoringService(),
            )

            run = workflow_service.create_run_from_explicit_plan(
                intake=GenerationIntake(
                    title="Build a Grounded Internal Docs Assistant",
                    problem_statement=(
                        "Build a backend service that answers internal docs questions from a visible corpus, "
                        "returns citations, and abstains when the evidence is weak."
                    ),
                ),
                design_spec=_grounded_design_spec(),
            )

            self.assertEqual(run.stage, WorkflowStage.blocked)
            self.assertEqual(run.status, WorkflowStatus.blocked)
            self.assertIsNone(run.artifacts.task_agent_spec)
            events = workflow_service.list_events(run.id)
            self.assertTrue(any(event.event_type == "workflow_authoring_failed" for event in events))
        finally:
            temp_dir.cleanup()

    def test_authored_scenarios_drive_public_checks_when_eval_payloads_are_loose(self) -> None:
        base_spec, _ = build_task_agent_scaffold(
            title="Build a Concurrent Inventory Reservation Service",
            summary=(
                "Create a production-ready transactional backend that keeps inventory reservations "
                "correct under retries, concurrency, and warehouse stock movement."
            ),
            design_spec=infer_assignment_design(
                title="Build a Concurrent Inventory Reservation Service",
                problem_statement=(
                    "Create a production-ready transactional backend that keeps inventory reservations "
                    "correct under retries, concurrency, and warehouse stock movement."
                ),
                implementation_language="python",
                application_framework="fastapi",
                primary_database="postgres",
                cache_backend="redis",
            ).design_spec,
        )
        primary_case_id = base_spec.eval_dataset.cases[0].id
        edge_case_id = base_spec.eval_dataset.cases[1].id
        base_spec = ensure_task_agent_deliverable_briefs(base_spec, overwrite=True)
        service = OpenAITaskAgentAuthoringService(enabled=False)
        customization = TaskAgentCustomization(
            deliverables=[
                DeliverableCustomization(
                    id="deliverable_1",
                    learner_starter_surface=StarterSurfaceCustomization(
                        starter_summary="Build the real inventory reservation API in app.py.",
                        implementation_checklist=[
                            "Persist reservation workflow state durably.",
                        ],
                        domain_scenarios=[
                            StarterScenarioCustomization(
                                title="Reserve available stock from one warehouse",
                                request_summary=(
                                    "A client reserves 3 units of SKU-RED-CHAIR from warehouse WH-EAST for order ORD-1001."
                                ),
                                expected_behavior=(
                                    "Create exactly one durable reservation and decrease allocatable stock without going negative."
                                ),
                            ),
                            StarterScenarioCustomization(
                                title="Retry the same reservation request after a timeout",
                                request_summary=(
                                    "The same request id and reservation payload arrive again after the caller timed out."
                                ),
                                expected_behavior=(
                                    "Return the original reservation outcome without decrementing stock a second time."
                                ),
                            ),
                        ],
                    ),
                )
            ],
            eval_cases=[
                EvalCaseCustomization(
                    id=primary_case_id,
                    title="Reserve available stock from one warehouse",
                    input={
                        "task-specific": {
                            "request_id": "REQ-INV-1001",
                            "task_input": "Reserve 3 units of SKU-RED-CHAIR from warehouse WH-EAST for order ORD-1001.",
                        }
                    },
                    expected_output={
                        "result": "<result>",
                        "confidence": "<confidence>",
                        "needs_human": False,
                    },
                    tags=["deliverable_1"],
                ),
                EvalCaseCustomization(
                    id=edge_case_id,
                    title="Escalate inconsistent inventory state for human review",
                    input={
                        "task-specific": {
                            "request_id": "REQ-INV-2002",
                            "task_input": "Move stock after detecting inconsistent reserved totals.",
                        }
                    },
                    expected_output={
                        "result": "<result>",
                        "confidence": "<confidence>",
                        "needs_human": True,
                    },
                    should_escalate=True,
                    requires_approval=True,
                    tags=["deliverable_1"],
                ),
            ],
        )

        updated = service._apply_customization(base_spec, customization)
        updated = ensure_task_agent_deliverable_briefs(updated, overwrite=True)
        validation = validate_task_agent_spec(updated)

        self.assertFalse(any(error.code == "placeholder_public_check" for error in validation.errors))
        self.assertEqual(
            updated.deliverables[0].public_checks[0].title,
            "Reserve available stock from one warehouse",
        )
        self.assertIn(
            "decrease allocatable stock without going negative",
            " ".join(updated.deliverables[0].public_checks[0].expected_assertions).lower(),
        )

    def test_validation_rejects_placeholder_schema_field_names(self) -> None:
        spec, _ = build_task_agent_scaffold(
            title="Build a Grounded Internal Docs Assistant",
            summary="Answer docs questions with citations and abstention.",
            design_spec=_grounded_design_spec(),
        )
        spec.task_schema = {
            "type": "object",
            "required": ["domain_field"],
            "properties": {"domain_field": {"type": "string"}},
        }
        spec.output_schema = {
            "type": "object",
            "required": ["domain_result"],
            "properties": {"domain_result": {"type": "string"}},
        }

        validation = validate_task_agent_spec(spec)
        error_codes = {error.code for error in validation.errors}

        self.assertIn("placeholder_task_schema_field", error_codes)
        self.assertIn("placeholder_output_schema_field", error_codes)

    def test_specialized_projects_render_non_placeholder_public_endpoints(self) -> None:
        inferred = infer_assignment_design(
            title="Build a Concurrent Inventory Reservation Service",
            problem_statement=(
                "Create a production-ready transactional backend that keeps inventory reservations "
                "correct under retries, concurrency, and warehouse stock movement."
            ),
            implementation_language="python",
            application_framework="fastapi",
            primary_database="postgres",
            cache_backend="redis",
        )
        spec, _ = build_task_agent_scaffold(
            title="Build a Concurrent Inventory Reservation Service",
            summary="Keep reservations correct under concurrent requests and retries.",
            design_spec=inferred.design_spec,
        )

        validation = validate_task_agent_spec(spec)
        self.assertNotIn("placeholder_service_endpoints", {error.code for error in validation.errors})
        self.assertTrue(
            any(endpoint.path.startswith("/concurrent-inventory") for endpoint in spec.production_contract.canonical_endpoints)
        )

    def test_primary_submit_endpoint_prefers_non_parameterized_post(self) -> None:
        endpoint = primary_submit_endpoint(
            [
                EndpointSpec(method="POST", path="/reservations/{reservation_id}/confirm", required=True),
                EndpointSpec(method="POST", path="/reservations", required=True),
                EndpointSpec(method="GET", path="/health", required=True),
            ]
        )

        self.assertIsNotNone(endpoint)
        self.assertEqual(endpoint.path, "/reservations")

    def test_python_runtime_entrypoint_exposes_public_routes_and_internal_harness(self) -> None:
        inferred = infer_assignment_design(
            title="Build a Concurrent Inventory Reservation Service",
            problem_statement=(
                "Create a production-ready transactional backend that keeps inventory reservations "
                "correct under retries, concurrency, and warehouse stock movement."
            ),
            implementation_language="python",
            application_framework="fastapi",
            primary_database="postgres",
            cache_backend="redis",
        )
        spec, _ = build_task_agent_scaffold(
            title="Build a Concurrent Inventory Reservation Service",
            summary="Keep reservations correct under concurrent requests and retries.",
            design_spec=inferred.design_spec,
        )
        spec.production_contract.canonical_endpoints = [
            EndpointSpec(method="POST", path="/reservations", required=True),
            EndpointSpec(method="GET", path="/reservations/{reservation_id}", required=True),
            EndpointSpec(method="GET", path="/health", required=True),
        ]

        rendered = render_task_agent_runtime_entrypoint(spec, for_root_workspace=False)

        self.assertIn("/_coursegen/run", rendered)
        self.assertIn("x-coursegen-run-id", rendered)
        self.assertIn("app.add_api_route(", rendered)
        self.assertIn("_public_endpoints(manifest)", rendered)
        self.assertNotIn('@app.post("/run")', rendered)

    def test_scaffold_drops_approval_lane_when_capability_disabled(self) -> None:
        inferred = infer_assignment_design(
            title="Build a Concurrent Inventory Reservation Service",
            problem_statement=(
                "Create a production-ready transactional backend that keeps inventory reservations "
                "correct under retries, concurrency, and warehouse stock movement."
            ),
            implementation_language="python",
            application_framework="fastapi",
            primary_database="postgres",
            cache_backend="redis",
        )
        spec, _ = build_task_agent_scaffold(
            title="Build a Concurrent Inventory Reservation Service",
            summary="Keep reservations correct under concurrent requests and retries.",
            design_spec=inferred.design_spec,
        )

        self.assertFalse(spec.capabilities.approval_flow_required)
        self.assertNotIn("async_human_in_loop", [mode.value for mode in spec.supported_modes])
        self.assertFalse(spec.production_contract.supports_async_runs)
        self.assertFalse(any(tool.approval_required for tool in spec.tool_registry.tools))
        self.assertFalse(any(case.requires_approval for case in spec.eval_dataset.cases))

    def test_human_feedback_revision_rejects_invalid_reauthored_spec(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        try:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir.name}/test.db")
            workflow_service = WorkflowService(
                store,
                materializer=ArtifactMaterializer(base_dir=f"{temp_dir.name}/generated"),
            )
            run = workflow_service.create_run_from_explicit_plan(
                intake=GenerationIntake(
                    title="Build a Grounded Internal Docs Assistant",
                    problem_statement=(
                        "Build a backend service that answers internal docs questions from a visible corpus, "
                        "returns citations, and abstains when the evidence is weak."
                    ),
                ),
                design_spec=_grounded_design_spec(),
                execute_nodes=False,
            )
            original_spec = run.artifacts.task_agent_spec.model_dump(mode="json")
            original_validation = dict(run.artifacts.validation_summary or {})
            workflow_service.task_agent_authoring_service = _InvalidRevisionAuthoringService(
                run.artifacts.task_agent_spec
            )

            with self.assertRaises(WorkflowConflictError):
                workflow_service._apply_human_feedback_revision(run, "Tighten the learner-facing contract.")

            self.assertEqual(run.artifacts.task_agent_spec.model_dump(mode="json"), original_spec)
            self.assertEqual(run.artifacts.validation_summary, original_validation)
        finally:
            temp_dir.cleanup()

    def test_status_surfaces_customization_validation_rejections(self) -> None:
        class _MinimalClient:
            def __init__(self) -> None:
                self.responses = self

            def create(self, **_: object) -> _FakeResponse:
                return _FakeResponse({"notes": ["minimal customization"]})

        service = _AlwaysReadyAuthoringService(client_factory=lambda api_key, base_url: _MinimalClient())
        original_apply = service._apply_customization

        def _invalid_apply(base_spec, customization):  # noqa: ANN001
            invalid = original_apply(base_spec, customization)
            invalid.runtime_dependencies.editable_files = []
            invalid.deliverables[0].learner_brief = None
            return invalid

        service._apply_customization = _invalid_apply  # type: ignore[method-assign]

        result = service.generate_scaffold(
            title="Build a Grounded Internal Docs Assistant",
            summary="Answer docs questions with citations and abstention.",
            design_spec=_grounded_design_spec(),
        )
        status = service.status()

        self.assertEqual(result.source, TaskAgentAuthoringSource.deterministic_fallback)
        self.assertEqual(status.customization_validation_rejection_count, 1)
        self.assertIn(
            "OpenAI customization produced an invalid spec",
            status.last_customization_validation_error or "",
        )

    def test_gate_one_approval_fails_closed_when_validation_summary_is_missing(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        try:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir.name}/test.db")
            workflow_service = WorkflowService(
                store,
                materializer=ArtifactMaterializer(base_dir=f"{temp_dir.name}/generated"),
            )
            run = workflow_service.create_run_from_explicit_plan(
                intake=GenerationIntake(
                    title="Build a Grounded Internal Docs Assistant",
                    problem_statement=(
                        "Build a backend service that answers internal docs questions from a visible corpus, "
                        "returns citations, and abstains when the evidence is weak."
                    ),
                ),
                design_spec=_grounded_design_spec(),
                execute_nodes=False,
            )
            run.artifacts.validation_summary = None
            run.artifacts.node_executions = [
                WorkflowNodeExecution(
                    node_id="authoring_runtime_1",
                    kind=WorkflowNodeKind.authoring_runtime,
                    iteration=1,
                    attempt=1,
                    status=WorkflowNodeStatus.passed,
                    summary="Generated assignment compiled and booted inside the Docker sandbox.",
                    created_at=run.created_at,
                    sandbox_result=None,
                    findings=[],
                )
            ]
            store.save_run(run)

            with self.assertRaises(WorkflowConflictError):
                workflow_service.apply_gate_decision(
                    run.id,
                    GateDecisionRequest(
                        gate=HILGate.gate_1_spec_review,
                        decision=DecisionOutcome.approve,
                    ),
                )
        finally:
            temp_dir.cleanup()

    def test_gate_one_approval_requires_fresh_authoring_after_artifact_invalidation(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        try:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir.name}/test.db")
            workflow_service = WorkflowService(
                store,
                materializer=ArtifactMaterializer(base_dir=f"{temp_dir.name}/generated"),
            )
            run = workflow_service.create_run_from_explicit_plan(
                intake=GenerationIntake(
                    title="Build a Grounded Internal Docs Assistant",
                    problem_statement=(
                        "Build a backend service that answers internal docs questions from a visible corpus, "
                        "returns citations, and abstains when the evidence is weak."
                    ),
                ),
                design_spec=_grounded_design_spec(),
                execute_nodes=False,
            )
            run.artifacts.validation_summary = {"valid": True, "errors": [], "warnings": []}
            run.artifacts.workspace_snapshot = None
            run.artifacts.node_executions = [
                WorkflowNodeExecution(
                    node_id="authoring_runtime_1",
                    kind=WorkflowNodeKind.authoring_runtime,
                    iteration=1,
                    attempt=1,
                    status=WorkflowNodeStatus.passed,
                    summary="Generated assignment compiled and booted inside the Docker sandbox.",
                    created_at=run.created_at,
                    sandbox_result=None,
                    findings=[],
                )
            ]
            store.save_run(run)

            with self.assertRaises(WorkflowConflictError):
                workflow_service.apply_gate_decision(
                    run.id,
                    GateDecisionRequest(
                        gate=HILGate.gate_1_spec_review,
                        decision=DecisionOutcome.approve,
                    ),
                )
        finally:
            temp_dir.cleanup()
