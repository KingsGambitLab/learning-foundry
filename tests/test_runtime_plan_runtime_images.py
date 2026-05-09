from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.grading import ApprovalRecord
from app.domain.registry import PackageType
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import (
    dependency_container_image,
    infer_assignment_design,
    runtime_target_commands_for_stack,
)
from app.services.learner_studio_service import LearnerStudioService
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs
from app.services.openai_task_agent_authoring import (
    DeliverableCustomization,
    EvalCaseCustomization,
    OpenAITaskAgentAuthoringService,
    SchemaCustomization,
    StarterScenarioCustomization,
    StarterSurfaceCustomization,
    TaskAgentCustomization,
)
from app.services.spec_validation import validate_task_agent_spec
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.services.task_agent_starter_templates import (
    build_task_agent_starter_files,
    task_agent_entrypoint_path,
    render_task_agent_runtime_deliverable,
)


def _build_spec(
    *,
    title: str,
    summary: str,
    problem_statement: str,
):
    inferred = infer_assignment_design(
        title=title,
        problem_statement=problem_statement,
        learning_outcomes=[
            "Design a production-grade backend surface.",
            "Ship a runtime that can be graded end to end.",
        ],
        package_type_hint=PackageType.progressive_codebase_course,
    )
    assert inferred.design_spec is not None
    spec, _origin = build_task_agent_scaffold(
        title=title,
        summary=summary,
        design_spec=inferred.design_spec,
    )
    return spec


def test_typescript_starter_dockerfile_follows_runtime_plan() -> None:
    spec = _build_spec(
        title="Feature Flag Control Plane",
        summary="Build a feature flag control plane service.",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, "
            "NestJS 11, Node 22, MongoDB 7, pnpm, audit logs, and safe config updates."
        ),
    )

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    dockerfile = starter_files["Dockerfile"]

    assert dockerfile.startswith("FROM ")
    assert (
        "FROM node:22-bookworm-slim" in dockerfile
        or "FROM sha256:" in dockerfile
    )
    assert "ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0" in dockerfile
    assert (
        "RUN corepack enable" in dockerfile
        or "apt-get install -y --no-install-recommends nodejs npm" in dockerfile
    )
    assert "RUN pnpm install --yes --dangerously-allow-all-builds" in dockerfile
    assert "COPY . /workspace" in dockerfile
    assert "EXPOSE 8000" in dockerfile


def test_runtime_target_commands_for_pnpm_allow_noninteractive_builds() -> None:
    install_command, run_command, check_command = runtime_target_commands_for_stack(
        implementation_language="typescript",
        application_framework="nestjs",
        package_manager="pnpm",
    )

    assert install_command == "pnpm install --yes --dangerously-allow-all-builds"
    assert run_command == "pnpm start:dev"
    assert check_command == "python checks/run_visible_checks.py"


def test_runtime_plan_prefers_lightweight_dependency_images() -> None:
    assert dependency_container_image(technology="postgres", version_hint=None) == "postgres:16-alpine"
    assert dependency_container_image(technology="redis", version_hint=None) == "redis:7-alpine"


def test_assignment_runtime_dockerfile_reuses_app_base_and_adds_verifier_python() -> None:
    spec = _build_spec(
        title="Feature Flag Control Plane",
        summary="Build a feature flag control plane service.",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, "
            "NestJS 11, Node 22, MongoDB 7, pnpm, audit logs, and safe config updates."
        ),
    )

    materializer = ArtifactMaterializer()
    dockerfile = materializer._assignment_runtime_dockerfile(spec)

    assert dockerfile.startswith("FROM ")
    assert "apt-get install -y --no-install-recommends python3" in dockerfile
    assert (
        "corepack enable" in dockerfile
        or "apt-get install -y --no-install-recommends nodejs npm" in dockerfile
    )
    assert 'CMD ["python3", "runtime/verify_assignment.py"]' in dockerfile


def test_starter_runtime_emits_stable_approval_ids() -> None:
    runtime_source = render_task_agent_runtime_deliverable()

    assert '"approval_id": f"{run_id}::approval::0"' in runtime_source


def test_blackbox_runner_normalizes_missing_approval_ids() -> None:
    runner = TaskAgentBlackBoxRunner()

    approvals = runner._parse_records(
        [{"tool_id": "send_final_output", "status": "approved"}],
        ApprovalRecord,
        include_order=True,
    )

    assert len(approvals) == 1
    assert approvals[0].approval_id == "approval::send_final_output::0"
    assert approvals[0].approved is True


def test_learner_runtime_launch_script_exports_corepack_prompt_override() -> None:
    spec = _build_spec(
        title="Feature Flag Control Plane",
        summary="Build a feature flag control plane service.",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, "
            "NestJS 11, Node 22, MongoDB 7, pnpm, audit logs, and safe config updates."
        ),
    )
    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    with TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        for relative_path, content in starter_files.items():
            output_path = workspace_path / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        manifest = json.loads((workspace_path / "starter_manifest.json").read_text(encoding="utf-8"))
        assert manifest["runtime_plan"]["package_manager"] == "pnpm"

        service = LearnerStudioService()
        launch_script = service._runtime_launch_script(
            workspace_path=workspace_path,
            spec=spec,
            include_setup=True,
        )

    assert "export COREPACK_ENABLE_DOWNLOAD_PROMPT=0" in launch_script
    assert "corepack enable" in launch_script
    assert "pnpm install --yes --dangerously-allow-all-builds" in launch_script


def test_starter_surface_is_authored_and_python_entrypoint_is_not_a_wrapper() -> None:
    spec = _build_spec(
        title="Grounded Internal Docs Assistant",
        summary="Build a grounded assistant over a visible internal docs corpus.",
        problem_statement=(
            "Build a grounded internal docs assistant that answers from a visible corpus with citations "
            "and abstains when support is weak."
        ),
    )

    validation = validate_task_agent_spec(spec)
    assert validation.valid
    starter_surface = spec.deliverables[0].learner_starter_surface
    assert starter_surface is not None
    assert starter_surface.primary_editable_paths
    assert starter_surface.required_endpoints
    assert starter_surface.domain_scenarios

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    entrypoint_path = task_agent_entrypoint_path(spec)
    source = starter_files[entrypoint_path]
    manifest = json.loads(starter_files["starter_manifest.json"])

    assert "from runtime.task_agent_runtime import" not in source
    assert "def create_app_from_manifest(" in source
    assert manifest["learner_starter_surface"]["primary_editable_paths"]
    assert manifest["learner_starter_surface"]["required_endpoints"]


def test_generic_placeholder_workflow_specs_now_fail_validation() -> None:
    spec = _build_spec(
        title="Workflow Agent",
        summary="Build a generic workflow agent.",
        problem_statement=(
            "Build an agent that uses tools, approvals, and traceability to complete bounded workflows."
        ),
    )

    validation = validate_task_agent_spec(spec)

    assert not validation.valid
    error_codes = {issue.code for issue in validation.errors}
    assert "placeholder_domain_scenario" in error_codes or "placeholder_public_check" in error_codes


def test_authoring_customization_can_make_generic_workflow_spec_domain_specific() -> None:
    spec = _build_spec(
        title="Production Customer Support Bot",
        summary="Build a production-ready customer support bot.",
        problem_statement=(
            "Build a customer support bot that handles refunds, outage updates, suspicious logins, "
            "approval gates, and production-ready tracing."
        ),
    )
    initial_validation = validate_task_agent_spec(spec)
    assert not initial_validation.valid

    service = OpenAITaskAgentAuthoringService(enabled=False)
    customized = service._apply_customization(
        spec,
        TaskAgentCustomization(
            task_schema=SchemaCustomization(
                required=["ticket_id", "customer_message", "issue_type"],
                properties={
                    "ticket_id": {"type": "string"},
                    "customer_message": {"type": "string"},
                    "issue_type": {"type": "string"},
                    "account_tier": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
            ),
            output_schema=SchemaCustomization(
                required=["decision", "priority", "response_summary", "confidence", "needs_human"],
                properties={
                    "decision": {"type": "string"},
                    "priority": {"type": "string"},
                    "response_summary": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "needs_human": {"type": "boolean"},
                },
            ),
            eval_cases=[
                EvalCaseCustomization(
                    id="happy_path",
                    title="Billing refund request",
                    input={
                        "ticket_id": "T-100",
                        "customer_message": "I was charged twice and need a refund.",
                        "issue_type": "billing_refund",
                        "account_tier": "pro",
                    },
                    expected_output={
                        "decision": "draft_refund_reply",
                        "priority": "medium",
                        "needs_human": True,
                    },
                    should_escalate=True,
                    requires_approval=True,
                ),
                EvalCaseCustomization(
                    id="escalation_case",
                    title="Suspicious login request",
                    input={
                        "ticket_id": "T-102",
                        "customer_message": "My account was accessed from a country I have never visited.",
                        "issue_type": "suspicious_login",
                        "account_tier": "business",
                    },
                    expected_output={
                        "decision": "security_escalation",
                        "priority": "urgent",
                        "needs_human": True,
                    },
                    should_escalate=True,
                    requires_approval=False,
                ),
            ],
            deliverables=[
                DeliverableCustomization(
                    id=spec.deliverables[0].id,
                    learner_starter_surface=StarterSurfaceCustomization(
                        starter_summary=(
                            "Build the real support workflow in learner-owned code so refund, outage, and "
                            "security tickets move through triage, approval, and reply drafting"
                        ),
                        implementation_checklist=[
                            "Persist a traceable support run id through the request lifecycle.",
                            "Keep risky ticket decisions reviewable before a final reply is sent.",
                        ],
                        domain_scenarios=[
                            StarterScenarioCustomization(
                                id="billing_refund",
                                title="Billing refund request",
                                request_summary=(
                                    "A pro customer says they were charged twice and wants a refund on the same ticket."
                                ),
                                expected_behavior=(
                                    "Gather the needed context, draft the refund path, and require review before any irreversible action."
                                ),
                            ),
                            StarterScenarioCustomization(
                                id="suspicious_login",
                                title="Suspicious login request",
                                request_summary=(
                                    "A business customer reports account access from a country they do not recognize."
                                ),
                                expected_behavior=(
                                    "Treat it as a security-sensitive path, escalate quickly, and keep the trace explicit."
                                ),
                            ),
                        ],
                    ),
                )
            ],
        ),
    )
    customized = ensure_task_agent_deliverable_briefs(customized, overwrite=True)

    validation = validate_task_agent_spec(customized)
    assert validation.valid
    starter_surface = customized.deliverables[0].learner_starter_surface
    assert starter_surface is not None
    assert starter_surface.starter_summary.startswith("Build the real support workflow")
    assert starter_surface.domain_scenarios[0].title == "Billing refund request"
    assert "charged twice" in starter_surface.domain_scenarios[0].request_summary
    assert customized.deliverables[0].public_checks[0].title == "Billing refund request"
    assert "Build the real support workflow" in customized.deliverables[0].learner_brief.task_to_build
    assert any(
        "Persist a traceable support run id" in item
        for item in customized.deliverables[0].learner_brief.definition_of_done
    )
