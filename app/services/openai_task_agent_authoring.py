from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import (
    AssignmentDesignSpec,
    ApprovalExpectation,
    CostPerSuccessTestParams,
    DryRunSemanticsTestParams,
    EscalationExpectation,
    EscalationPolicyTestParams,
    OutputSchemaTestParams,
    P95RunLatencyTestParams,
    QualitySpec,
    TaskAgentServiceSpec,
    TaskEvalCase,
    TaskOutputQualityJudgeTestParams,
    TaskSuccessRateTestParams,
    ToolSelectionTestParams,
    TraceSchemaTestParams,
    ConfidenceCalibrationJudgeTestParams,
    EscalationPrecisionTestParams,
)
from app.services.spec_validation import validate_task_agent_spec
from app.services.learner_brief_builder import ensure_task_agent_module_briefs
from app.services.task_agent_scaffolds import build_task_agent_scaffold


class TaskAgentAuthoringSource(str, Enum):
    openai_live = "openai_live"
    deterministic_fallback = "deterministic_fallback"


class TaskAgentAuthoringStatus(BaseModel):
    provider: str = "openai"
    available: bool
    source: TaskAgentAuthoringSource
    message: str
    sdk_installed: bool = False
    api_key_present: bool = False
    model_id: str | None = None
    env_file: str | None = None


class ModuleCustomization(BaseModel):
    id: str
    title: str | None = None
    objective: str | None = None
    overlay_ids: list[str] = Field(default_factory=list)


class ToolCustomization(BaseModel):
    id: str
    description: str | None = None


class EvalCaseCustomization(BaseModel):
    id: str
    input: dict[str, Any] | None = None
    expected_output: dict[str, Any] | None = None
    should_escalate: bool | None = None
    requires_approval: bool | None = None
    must_use_any_of_tools: list[str] = Field(default_factory=list)
    must_not_use_tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class QualityTargetCustomization(BaseModel):
    min_task_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    p95_run_latency_ms: int | None = Field(default=None, ge=1)
    max_cost_per_success_usd: float | None = Field(default=None, ge=0.0)
    min_escalation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    min_output_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_expected_calibration_error: float | None = Field(default=None, ge=0.0, le=1.0)


class TaskAgentCustomization(BaseModel):
    summary: str | None = None
    modules: list[ModuleCustomization] = Field(default_factory=list)
    tools: list[ToolCustomization] = Field(default_factory=list)
    eval_cases: list[EvalCaseCustomization] = Field(default_factory=list)
    quality_targets: QualityTargetCustomization | None = None
    notes: list[str] = Field(default_factory=list)


class TaskAgentAuthoringResult(BaseModel):
    spec: TaskAgentServiceSpec
    origin_template: str
    source: TaskAgentAuthoringSource
    notes: list[str] = Field(default_factory=list)
    status: TaskAgentAuthoringStatus


def _matches_schema_rule(schema: dict[str, Any], value: Any) -> bool:
    expected_type = schema.get("type")
    if not expected_type:
        return True
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


class OpenAITaskAgentAuthoringService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        env_file: str | None = None,
        model: str | None = None,
        client_factory=None,
    ) -> None:
        self.enabled = enabled
        self.env_file = env_file or os.environ.get("COURSE_GEN_OPENAI_ENV_FILE") or os.environ.get("OPENAI_ENV_FILE")
        self.model = model
        self.client_factory = client_factory

    def status(self) -> TaskAgentAuthoringStatus:
        config = self._config()
        sdk_installed = self._openai_sdk_available()
        api_key_present = bool(config.get("OPENAI_API_KEY"))
        model_id = config.get("OPENAI_MODEL") or self.model or "gpt-5.4"

        if not self.enabled:
            return TaskAgentAuthoringStatus(
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message="OpenAI task-agent authoring is disabled for this app instance.",
                sdk_installed=sdk_installed,
                api_key_present=api_key_present,
                model_id=model_id,
                env_file=self.env_file,
            )
        if not sdk_installed:
            return TaskAgentAuthoringStatus(
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message="The OpenAI Python SDK is not installed, so task-agent authoring will use the deterministic scaffold.",
                sdk_installed=False,
                api_key_present=api_key_present,
                model_id=model_id,
                env_file=self.env_file,
            )
        if not api_key_present:
            return TaskAgentAuthoringStatus(
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message="OPENAI_API_KEY is not configured, so task-agent authoring will use the deterministic scaffold.",
                sdk_installed=True,
                api_key_present=False,
                model_id=model_id,
                env_file=self.env_file,
            )
        return TaskAgentAuthoringStatus(
            available=True,
            source=TaskAgentAuthoringSource.openai_live,
            message="OpenAI task-agent authoring is ready to customize task-agent specs.",
            sdk_installed=True,
            api_key_present=True,
            model_id=model_id,
            env_file=self.env_file,
        )

    def generate_scaffold(
        self,
        *,
        title: str,
        summary: str,
        design_spec: AssignmentDesignSpec,
    ) -> TaskAgentAuthoringResult:
        base_spec, origin_template = build_task_agent_scaffold(
            title=title,
            summary=summary,
            design_spec=design_spec,
        )
        base_spec = ensure_task_agent_module_briefs(base_spec, overwrite=True)
        status = self.status()
        if not status.available:
            return TaskAgentAuthoringResult(
                spec=base_spec,
                origin_template=origin_template,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                notes=[status.message],
                status=status,
            )

        try:
            customization = self._generate_customization(
                base_spec=base_spec,
                origin_template=origin_template,
                title=title,
                summary=summary,
                package_type=design_spec.course_structure.package_type,
                domain_pack=design_spec.domain_pack,
                risk_class=design_spec.risk_class,
                overlays=design_spec.overlays,
                model_id=status.model_id or "gpt-5.4",
            )
            customized_spec = self._apply_customization(base_spec, customization)
            customized_spec = ensure_task_agent_module_briefs(customized_spec, overwrite=True)
            validation = validate_task_agent_spec(customized_spec)
            if not validation.valid:
                raise ValueError(
                    "OpenAI customization produced an invalid spec: "
                    + "; ".join(error.code for error in validation.errors[:3])
                )
            return TaskAgentAuthoringResult(
                spec=customized_spec,
                origin_template=f"openai_customized:{origin_template}",
                source=TaskAgentAuthoringSource.openai_live,
                notes=[
                    f"Customized with OpenAI model `{status.model_id}`.",
                    *customization.notes[:3],
                ],
                status=status,
            )
        except Exception as exc:  # pragma: no cover - exact SDK failures vary by environment
            fallback_status = TaskAgentAuthoringStatus(
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message=f"OpenAI task-agent authoring failed and fell back to deterministic scaffolding: {exc}",
                sdk_installed=status.sdk_installed,
                api_key_present=status.api_key_present,
                model_id=status.model_id,
                env_file=status.env_file,
            )
            return TaskAgentAuthoringResult(
                spec=base_spec,
                origin_template=origin_template,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                notes=[fallback_status.message],
                status=fallback_status,
            )

    def revise_spec(
        self,
        *,
        spec: TaskAgentServiceSpec,
        title: str,
        summary: str,
        package_type: PackageType,
        domain_pack: str | None,
        risk_class: RiskClass,
        overlays: list[str],
        feedback: str,
        origin_template: str | None = None,
    ) -> TaskAgentAuthoringResult:
        status = self.status()
        origin = origin_template or "task_agent_spec"
        if not status.available:
            return TaskAgentAuthoringResult(
                spec=spec,
                origin_template=origin,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                notes=[status.message],
                status=status,
            )

        try:
            customization = self._generate_customization(
                base_spec=spec,
                origin_template=origin,
                title=title,
                summary=summary,
                package_type=package_type,
                domain_pack=domain_pack,
                risk_class=risk_class,
                overlays=overlays,
                model_id=status.model_id or "gpt-5.4",
                feedback=feedback,
            )
            revised_spec = self._apply_customization(spec, customization)
            revised_spec = ensure_task_agent_module_briefs(revised_spec, overwrite=True)
            validation = validate_task_agent_spec(revised_spec)
            if not validation.valid:
                raise ValueError(
                    "OpenAI revision produced an invalid spec: "
                    + "; ".join(error.code for error in validation.errors[:3])
                )
            return TaskAgentAuthoringResult(
                spec=revised_spec,
                origin_template=f"openai_revision:{origin}",
                source=TaskAgentAuthoringSource.openai_live,
                notes=[
                    f"Revised with OpenAI model `{status.model_id}` using human review feedback.",
                    *customization.notes[:3],
                ],
                status=status,
            )
        except Exception as exc:  # pragma: no cover - exact SDK failures vary by environment
            fallback_status = TaskAgentAuthoringStatus(
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message=f"OpenAI revision failed and left the current spec unchanged: {exc}",
                sdk_installed=status.sdk_installed,
                api_key_present=status.api_key_present,
                model_id=status.model_id,
                env_file=status.env_file,
            )
            return TaskAgentAuthoringResult(
                spec=spec,
                origin_template=origin,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                notes=[fallback_status.message],
                status=fallback_status,
            )

    def _generate_customization(
        self,
        *,
        base_spec: TaskAgentServiceSpec,
        origin_template: str,
        title: str,
        summary: str,
        package_type: PackageType,
        domain_pack: str | None,
        risk_class: RiskClass,
        overlays: list[str],
        model_id: str,
        feedback: str | None = None,
    ) -> TaskAgentCustomization:
        config = self._config()
        client = self._client(
            api_key=config.get("OPENAI_API_KEY", ""),
            base_url=config.get("OPENAI_BASE_URL"),
        )
        prompt = self._prompt_payload(
            base_spec=base_spec,
            origin_template=origin_template,
            title=title,
            summary=summary,
            package_type=package_type,
            domain_pack=domain_pack,
            risk_class=risk_class,
            overlays=overlays,
            feedback=feedback,
        )
        response = client.responses.create(
            model=model_id,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You customize learner-ready backend assignments for software engineering courses. "
                        "Return JSON only. Do not rename module ids, tool ids, or eval case ids. "
                        "Keep the assignment production-minded, teachable, and safety-aware. "
                        "Preserve grounded contracts, citations, abstention behavior, and approval gates whenever they apply. "
                        "If human review feedback is provided, revise the current assignment to address it directly."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, indent=2)},
            ],
            temperature=0.2,
        )
        payload = self._extract_json(getattr(response, "output_text", ""))
        return TaskAgentCustomization.model_validate(payload)

    def _apply_customization(
        self,
        spec: TaskAgentServiceSpec,
        customization: TaskAgentCustomization,
    ) -> TaskAgentServiceSpec:
        updated = spec.model_copy(deep=True)
        tool_ids = updated.tool_ids
        allowed_input_keys = set(((updated.task_schema.get("properties") or {}).keys()))
        allowed_output_keys = set(((updated.output_schema.get("properties") or {}).keys()))
        output_properties = updated.output_schema.get("properties") or {}

        if customization.summary:
            updated.summary = customization.summary.strip()

        modules_by_id = {module.id: module for module in updated.modules}
        for patch in customization.modules:
            module = modules_by_id.get(patch.id)
            if module is None:
                continue
            if patch.title:
                module.title = patch.title.strip()
            if patch.objective:
                module.objective = patch.objective.strip()
            if patch.overlay_ids:
                module.overlay_ids = list(dict.fromkeys(patch.overlay_ids))

        tools_by_id = {tool.id: tool for tool in updated.tool_registry.tools}
        for patch in customization.tools:
            tool = tools_by_id.get(patch.id)
            if tool is None:
                continue
            if patch.description:
                tool.description = patch.description.strip()

        cases_by_id = {case.id: case for case in updated.eval_dataset.cases}
        for patch in customization.eval_cases:
            case = cases_by_id.get(patch.id)
            if case is None:
                continue
            if patch.input:
                filtered_input = {
                    key: value for key, value in patch.input.items() if not allowed_input_keys or key in allowed_input_keys
                }
                if filtered_input:
                    case.input = filtered_input
            if patch.expected_output is not None:
                filtered_output = {
                    key: value
                    for key, value in patch.expected_output.items()
                    if (
                        (not allowed_output_keys or key in allowed_output_keys)
                        and _matches_schema_rule(output_properties.get(key, {}), value)
                    )
                }
                if filtered_output:
                    merged_output = dict(case.expected_output or {})
                    merged_output.update(filtered_output)
                    case.expected_output = merged_output
            if patch.should_escalate is not None:
                case.should_escalate = patch.should_escalate
            if patch.requires_approval is not None:
                case.requires_approval = patch.requires_approval
            case.must_use_any_of_tools = [tool_id for tool_id in patch.must_use_any_of_tools if tool_id in tool_ids]
            case.must_not_use_tools = [tool_id for tool_id in patch.must_not_use_tools if tool_id in tool_ids]
            case.tags = patch.tags

        self._sync_behavior_tests(updated)
        self._sync_quality_targets(updated, customization.quality_targets)
        return updated

    def _sync_behavior_tests(self, spec: TaskAgentServiceSpec) -> None:
        eval_cases = list(spec.eval_dataset.cases)
        if not eval_cases:
            return

        output_case_ids = [case.id for case in eval_cases[: min(2, len(eval_cases))]]
        if spec.capabilities.abstention_required or spec.capabilities.is_grounded_answer_system:
            abstention_case = next(
                (
                    case.id
                    for case in eval_cases
                    if isinstance(case.expected_output, dict) and case.expected_output.get("abstained") is True
                ),
                None,
            )
            if abstention_case is not None and abstention_case not in output_case_ids:
                output_case_ids.append(abstention_case)
        escalation_cases = [case for case in eval_cases if case.should_escalate]
        approval_cases = [case for case in eval_cases if case.requires_approval]
        mutating_tools = [
            tool.id
            for tool in spec.tool_registry.tools
            if tool.safety.value != "read"
        ]

        for behavior in spec.behaviors:
            test = behavior.test
            if isinstance(test, OutputSchemaTestParams):
                test.case_ids = output_case_ids
            elif isinstance(test, TraceSchemaTestParams):
                test.case_ids = output_case_ids
            elif isinstance(test, ToolSelectionTestParams):
                for expectation in test.expectations:
                    matching = next((case for case in eval_cases if case.id == expectation.case_id), None)
                    if matching is None:
                        continue
                    if matching.must_use_any_of_tools:
                        expectation.must_call_any_of = matching.must_use_any_of_tools
                    expectation.must_not_call = matching.must_not_use_tools
            elif isinstance(test, EscalationPolicyTestParams) and escalation_cases:
                test.expectations = [
                    EscalationExpectation(
                        case_id=case.id,
                        must_escalate=bool(case.should_escalate),
                        allowed_reasons=["low_confidence", "ambiguous_request"],
                    )
                    for case in escalation_cases
                ]
            elif isinstance(test, DryRunSemanticsTestParams):
                test.case_ids = [output_case_ids[0]]
                test.mutating_tool_ids = mutating_tools or test.mutating_tool_ids

        for quality in spec.qualities:
            if isinstance(quality.test, P95RunLatencyTestParams):
                quality.test.concurrency = min(max(len(eval_cases), 1), 8)

        if approval_cases:
            for behavior in spec.behaviors:
                if hasattr(behavior.test, "expectations") and behavior.test.type == "approval_gate_test":
                    behavior.test.expectations = [
                        ApprovalExpectation(
                            case_id=case.id,
                            tool_id=next(
                                (
                                    tool.id
                                    for tool in spec.tool_registry.tools
                                    if tool.approval_required
                                ),
                                spec.tool_registry.tools[-1].id,
                            ),
                            requires_approval=True,
                        )
                        for case in approval_cases
                    ]

    def _sync_quality_targets(
        self,
        spec: TaskAgentServiceSpec,
        quality_targets: QualityTargetCustomization | None,
    ) -> None:
        if quality_targets is None:
            return

        slos = spec.production_contract.slos
        if quality_targets.min_task_success_rate is not None:
            slos.min_task_success_rate = quality_targets.min_task_success_rate
        if quality_targets.p95_run_latency_ms is not None:
            slos.p95_run_latency_ms = quality_targets.p95_run_latency_ms
        if quality_targets.max_cost_per_success_usd is not None:
            slos.max_cost_per_success_usd = quality_targets.max_cost_per_success_usd
        if quality_targets.min_escalation_precision is not None:
            slos.min_escalation_precision = quality_targets.min_escalation_precision

        for quality in spec.qualities:
            test = quality.test
            if isinstance(test, TaskSuccessRateTestParams) and quality_targets.min_task_success_rate is not None:
                test.min_success_rate = quality_targets.min_task_success_rate
            elif isinstance(test, P95RunLatencyTestParams) and quality_targets.p95_run_latency_ms is not None:
                test.p95_ms = quality_targets.p95_run_latency_ms
            elif isinstance(test, CostPerSuccessTestParams) and quality_targets.max_cost_per_success_usd is not None:
                test.max_cost_usd = quality_targets.max_cost_per_success_usd
            elif isinstance(test, EscalationPrecisionTestParams) and quality_targets.min_escalation_precision is not None:
                test.min_precision = quality_targets.min_escalation_precision
            elif isinstance(test, TaskOutputQualityJudgeTestParams) and quality_targets.min_output_quality_score is not None:
                test.min_avg_score = quality_targets.min_output_quality_score
            elif (
                isinstance(test, ConfidenceCalibrationJudgeTestParams)
                and quality_targets.max_expected_calibration_error is not None
            ):
                test.max_expected_calibration_error = quality_targets.max_expected_calibration_error

    def _prompt_payload(
        self,
        *,
        base_spec: TaskAgentServiceSpec,
        origin_template: str,
        title: str,
        summary: str,
        package_type: PackageType,
        domain_pack: str | None,
        risk_class: RiskClass,
        overlays: list[str],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        output_fields = list((base_spec.output_schema.get("properties") or {}).keys())[:3]
        example_expected_output = {field: f"<{field}>" for field in output_fields}
        payload = {
            "goal": {
                "title": title,
                "summary": summary,
                "package_type": package_type.value,
                "domain_pack": domain_pack,
                "risk_class": risk_class.value,
                "overlays": overlays,
                "course_structure": base_spec.course_structure.model_dump(mode="json"),
                "runtime_dependencies": base_spec.runtime_dependencies.model_dump(mode="json"),
                "capabilities": base_spec.capabilities.model_dump(mode="json"),
                "assessment_strategy": base_spec.assessment_strategy.model_dump(mode="json"),
            },
            "base_template": origin_template,
            "requirements": {
                "keep_module_ids": [module.id for module in base_spec.modules],
                "keep_tool_ids": [tool.id for tool in base_spec.tool_registry.tools],
                "keep_eval_case_ids": [case.id for case in base_spec.eval_dataset.cases],
                "return_shape": {
                    "summary": "string or null",
                    "modules": [
                        {"id": "module id", "title": "new title", "objective": "new objective", "overlay_ids": ["optional overlays"]}
                    ],
                    "tools": [{"id": "tool id", "description": "updated description"}],
                    "eval_cases": [
                        {
                            "id": "existing case id",
                            "input": {"task-specific": "payload"},
                            "expected_output": example_expected_output,
                            "should_escalate": False,
                            "requires_approval": False,
                            "must_use_any_of_tools": ["existing_tool_id"],
                            "must_not_use_tools": ["existing_tool_id"],
                            "tags": ["optional"]
                        }
                    ],
                    "quality_targets": {
                        "min_task_success_rate": 0.9,
                        "p95_run_latency_ms": 1800,
                        "max_cost_per_success_usd": 0.03,
                        "min_escalation_precision": 0.8,
                        "min_output_quality_score": 0.8,
                        "max_expected_calibration_error": 0.2
                    },
                    "notes": ["short notes"]
                },
            },
            "base_spec": {
                "summary": base_spec.summary,
                "modules": [
                    {
                        "id": module.id,
                        "title": module.title,
                        "objective": module.objective,
                        "starter_type": module.starter_type.value,
                        "overlay_ids": module.overlay_ids,
                    }
                    for module in base_spec.modules
                ],
                "tools": [
                    {
                        "id": tool.id,
                        "description": tool.description,
                        "safety": tool.safety.value,
                        "approval_required": tool.approval_required,
                    }
                    for tool in base_spec.tool_registry.tools
                ],
                "eval_cases": [
                    {
                        "id": case.id,
                        "input": case.input,
                        "expected_output": case.expected_output,
                        "should_escalate": case.should_escalate,
                        "requires_approval": case.requires_approval,
                        "must_use_any_of_tools": case.must_use_any_of_tools,
                        "must_not_use_tools": case.must_not_use_tools,
                    }
                    for case in base_spec.eval_dataset.cases
                ],
                "quality_targets": {
                    "min_task_success_rate": base_spec.production_contract.slos.min_task_success_rate,
                    "p95_run_latency_ms": base_spec.production_contract.slos.p95_run_latency_ms,
                    "max_cost_per_success_usd": base_spec.production_contract.slos.max_cost_per_success_usd,
                    "min_escalation_precision": base_spec.production_contract.slos.min_escalation_precision,
                },
            },
        }
        if feedback:
            payload["human_review_feedback"] = feedback
        return payload

    def _client(self, *, api_key: str, base_url: str | None):
        if self.client_factory is not None:
            return self.client_factory(api_key, base_url)
        from openai import OpenAI

        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)

    def _config(self) -> dict[str, str]:
        config: dict[str, str] = {}
        if self.env_file:
            config.update(self._load_env_file(self.env_file))
        for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
            value = os.environ.get(key)
            if value:
                config[key] = value
        if "OPENAI_MODEL" not in config and self.model:
            config["OPENAI_MODEL"] = self.model
        elif "OPENAI_MODEL" not in config:
            config["OPENAI_MODEL"] = "gpt-5.4"
        return config

    def _load_env_file(self, path: str) -> dict[str, str]:
        env: dict[str, str] = {}
        env_path = Path(path).expanduser()
        if not env_path.exists():
            return env
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = self._strip_quotes(value.strip())
        return env

    def _strip_quotes(self, value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    def _extract_json(self, text: str) -> dict[str, Any]:
        if not text:
            raise ValueError("The OpenAI response did not contain text output.")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("The OpenAI response did not contain a JSON object.")
        return json.loads(text[start : end + 1])

    def _openai_sdk_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True
