from __future__ import annotations

import json
import os
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.ai import AIUsageSummary, merge_ai_usage
from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import (
    AssignmentDesignSpec,
    DeliverableSpec,
    EndpointSpec,
    LearnerStarterSurfaceSpec,
    PublicCheckSpec,
    StarterScenarioSpec,
    TaskAgentServiceSpec,
)
from app.domain.workflow import FailureContext
from app.services.coursegen_logging import log_coursegen_event
from app.services.spec_validation import validate_task_agent_spec
from app.services.learner_brief_builder import ensure_task_agent_deliverable_briefs
from app.services.public_surface_quality import meaningful_domain_entities, normalized_tokens
from app.services.task_agent_contract_surface import learner_editable_paths_for_deliverable
from app.services.openai_runtime_support import (
    extract_openai_usage,
    load_openai_env_file,
    parse_structured_openai_response_with_hard_timeout,
    resolve_openai_env_file,
    strip_quotes,
)
from app.services.task_agent_scaffolds import build_task_agent_scaffold


class TaskAgentAuthoringSource(str, Enum):
    openai_live = "openai_live"
    anthropic_live = "anthropic_live"
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
    customization_validation_rejection_count: int = 0
    last_customization_validation_error: str | None = None


class EndpointCustomization(BaseModel):
    method: str
    path: str
    required: bool = True


class StarterScenarioCustomization(BaseModel):
    id: str | None = None
    title: str | None = None
    request_summary: str | None = None
    expected_behavior: str | None = None


class StarterSurfaceCustomization(BaseModel):
    starter_summary: str | None = None
    primary_editable_paths: list[str] = Field(default_factory=list)
    implementation_checklist: list[str] = Field(default_factory=list)
    domain_scenarios: list[StarterScenarioCustomization] = Field(default_factory=list)


class PublicCheckCustomization(BaseModel):
    id: str
    title: str | None = None
    learner_goal: str | None = None
    request_method: str | None = None
    request_path: str | None = None
    request_body_json: str | None = None
    expected_status: int | None = None
    expected_response_contains: list[str] = Field(default_factory=list)


class DeliverableCustomization(BaseModel):
    id: str
    title: str | None = None
    objective: str | None = None
    overlay_ids: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)
    learner_starter_surface: StarterSurfaceCustomization | None = None
    public_checks: list[PublicCheckCustomization] = Field(default_factory=list)


class TaskAgentCustomization(BaseModel):
    summary: str | None = None
    public_endpoints: list[EndpointCustomization] = Field(default_factory=list)
    deliverables: list[DeliverableCustomization] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TaskAgentAuthoringResult(BaseModel):
    spec: TaskAgentServiceSpec
    origin_template: str
    source: TaskAgentAuthoringSource
    notes: list[str] = Field(default_factory=list)
    status: TaskAgentAuthoringStatus
    usage: AIUsageSummary | None = None


def _normalize_text_list(items: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in normalized:
            continue
        normalized.append(cleaned)
    return normalized


def _scenario_identifier(value: str | None, *, fallback_index: int) -> str:
    if isinstance(value, str):
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
        normalized = "_".join(part for part in normalized.split("_") if part)
        if normalized:
            return normalized
    return f"starter_scenario_{fallback_index}"


def _parse_request_body_json(raw_body: str | None) -> dict[str, Any]:
    if not isinstance(raw_body, str):
        return {}
    candidate = raw_body.strip()
    if not candidate:
        return {}
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_authored_domain_scenarios(
    scenarios: list[StarterScenarioCustomization],
) -> list[StarterScenarioSpec]:
    authored: list[StarterScenarioSpec] = []
    for index, scenario in enumerate(scenarios, start=1):
        title = scenario.title.strip() if isinstance(scenario.title, str) else ""
        request_summary = scenario.request_summary.strip() if isinstance(scenario.request_summary, str) else ""
        expected_behavior = scenario.expected_behavior.strip() if isinstance(scenario.expected_behavior, str) else ""
        if not title or not request_summary or not expected_behavior:
            continue
        authored.append(
            StarterScenarioSpec(
                id=_scenario_identifier(scenario.id or title, fallback_index=index),
                title=title,
                request_summary=request_summary,
                expected_behavior=expected_behavior,
            )
        )
    return authored


class OpenAITaskAgentAuthoringService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        env_file: str | None = None,
        model: str | None = None,
        client_factory=None,
        # Bumped from 240s — Anthropic Sonnet 4.6 on the deep
        # TaskAgentServiceSpec schema with max_tokens=16000 routinely
        # hits ~5 min wall-clock. Keep ample headroom; subprocess hard-
        # kill remains the authoritative deadline.
        request_timeout_s: float = 600.0,
        max_request_retries: int = 2,
        max_customization_validation_retries: int = 2,
    ) -> None:
        self.enabled = enabled
        self.env_file = resolve_openai_env_file(env_file)
        self.model = model
        self.client_factory = client_factory
        self.request_timeout_s = request_timeout_s
        self.max_request_retries = max(0, max_request_retries)
        self.max_customization_validation_retries = max(0, max_customization_validation_retries)
        self._customization_validation_rejection_count = 0
        self._last_customization_validation_error: str | None = None

    def status(self) -> TaskAgentAuthoringStatus:
        # Authoring is now routed through the LLMRouter. Use its
        # provider-aware status as the source of truth so /v1/task-agent-
        # authoring/status reflects whichever provider is currently
        # active (Anthropic by default, OpenAI when COURSE_GEN_LLM_PROVIDER
        # is flipped).
        from app.services.llm_router import get_default_router

        router = get_default_router()
        router_status = router.status()
        if not self.enabled:
            return TaskAgentAuthoringStatus(
                provider=router_status["provider"],
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message="Authoring is disabled for this app instance.",
                sdk_installed=router_status["sdk_installed"],
                api_key_present=router_status["api_key_present"],
                model_id=router_status["model_id"],
                env_file=router_status["env_file"],
                customization_validation_rejection_count=self._customization_validation_rejection_count,
                last_customization_validation_error=self._last_customization_validation_error,
            )
        available = router_status["available"]
        if not available:
            return TaskAgentAuthoringStatus(
                provider=router_status["provider"],
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message=router_status["message"],
                sdk_installed=router_status["sdk_installed"],
                api_key_present=router_status["api_key_present"],
                model_id=router_status["model_id"],
                env_file=router_status["env_file"],
                customization_validation_rejection_count=self._customization_validation_rejection_count,
                last_customization_validation_error=self._last_customization_validation_error,
            )
        provider_value = router_status["provider"]
        live_source = (
            TaskAgentAuthoringSource.anthropic_live
            if provider_value == "anthropic"
            else TaskAgentAuthoringSource.openai_live
        )
        return TaskAgentAuthoringStatus(
            provider=provider_value,
            available=True,
            source=live_source,
            message=router_status["message"],
            sdk_installed=True,
            api_key_present=True,
            model_id=router_status["model_id"],
            env_file=router_status["env_file"],
            customization_validation_rejection_count=self._customization_validation_rejection_count,
            last_customization_validation_error=self._last_customization_validation_error,
        )

    def generate_scaffold(
        self,
        *,
        title: str,
        summary: str,
        design_spec: AssignmentDesignSpec,
        planner_deliverables: list[DeliverableSpec],
    ) -> TaskAgentAuthoringResult:
        log_coursegen_event(
            "task_agent_authoring_generate_started",
            title=title,
            package_type=design_spec.course_structure.package_type.value,
            implementation_language=design_spec.runtime_dependencies.implementation_language,
            application_framework=design_spec.runtime_dependencies.application_framework,
        )
        base_spec, origin_template = build_task_agent_scaffold(
            title=title,
            summary=summary,
            design_spec=design_spec,
            planner_deliverables=planner_deliverables,
        )
        base_spec = ensure_task_agent_deliverable_briefs(base_spec, overwrite=True)
        status = self.status()
        if not status.available:
            return TaskAgentAuthoringResult(
                spec=base_spec,
                origin_template=origin_template,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                notes=[status.message],
                status=status,
                usage=None,
            )

        try:
            customized_spec, customization, usage = self._generate_valid_customized_spec(
                stage="generate",
                base_spec=base_spec,
                title=title,
                summary=summary,
                package_type=design_spec.course_structure.package_type,
                domain_pack=design_spec.domain_pack,
                risk_class=design_spec.risk_class,
                overlays=design_spec.overlays,
                model_id=status.model_id or "gpt-5.4",
            )
            final_status = self.status()
            return TaskAgentAuthoringResult(
                spec=customized_spec,
                origin_template=f"openai_customized:{origin_template}",
                source=TaskAgentAuthoringSource.openai_live,
                notes=[f"Customized with OpenAI model `{status.model_id}`.", *customization.notes[:3]],
                status=final_status,
                usage=usage,
            )
        except Exception as exc:  # pragma: no cover
            fallback_status = TaskAgentAuthoringStatus(
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message=f"OpenAI authoring failed and fell back to the deterministic scaffold: {exc}",
                sdk_installed=status.sdk_installed,
                api_key_present=status.api_key_present,
                model_id=status.model_id,
                env_file=status.env_file,
                customization_validation_rejection_count=self._customization_validation_rejection_count,
                last_customization_validation_error=self._last_customization_validation_error,
            )
            return TaskAgentAuthoringResult(
                spec=base_spec,
                origin_template=origin_template,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                notes=[fallback_status.message],
                status=fallback_status,
                usage=None,
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
        failure_context: FailureContext | None = None,
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
                usage=None,
            )

        try:
            revised_spec, customization, usage = self._generate_valid_customized_spec(
                stage="revise",
                base_spec=spec,
                title=title,
                summary=summary,
                package_type=package_type,
                domain_pack=domain_pack,
                risk_class=risk_class,
                overlays=overlays,
                model_id=status.model_id or "gpt-5.4",
                feedback=feedback,
                failure_context=failure_context,
            )
            final_status = self.status()
            return TaskAgentAuthoringResult(
                spec=revised_spec,
                origin_template=f"openai_revision:{origin}",
                source=TaskAgentAuthoringSource.openai_live,
                notes=[f"Revised with OpenAI model `{status.model_id}`.", *customization.notes[:3]],
                status=final_status,
                usage=usage,
            )
        except Exception as exc:  # pragma: no cover
            fallback_status = TaskAgentAuthoringStatus(
                available=False,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                message=f"OpenAI authoring revision failed and left the prior draft unchanged: {exc}",
                sdk_installed=status.sdk_installed,
                api_key_present=status.api_key_present,
                model_id=status.model_id,
                env_file=status.env_file,
                customization_validation_rejection_count=self._customization_validation_rejection_count,
                last_customization_validation_error=self._last_customization_validation_error,
            )
            return TaskAgentAuthoringResult(
                spec=spec,
                origin_template=origin,
                source=TaskAgentAuthoringSource.deterministic_fallback,
                notes=[fallback_status.message],
                status=fallback_status,
                usage=None,
            )

    def _generate_valid_customized_spec(
        self,
        *,
        stage: str,
        base_spec: TaskAgentServiceSpec,
        title: str,
        summary: str,
        package_type: PackageType,
        domain_pack: str | None,
        risk_class: RiskClass,
        overlays: list[str],
        model_id: str,
        feedback: str | None = None,
        failure_context: FailureContext | None = None,
    ) -> tuple[TaskAgentServiceSpec, TaskAgentCustomization, AIUsageSummary | None]:
        accumulated_usage = AIUsageSummary()
        latest_feedback = feedback
        last_error_message: str | None = None

        for validation_attempt in range(1, self.max_customization_validation_retries + 2):
            customization, usage = self._generate_customization(
                base_spec=base_spec,
                title=title,
                summary=summary,
                package_type=package_type,
                domain_pack=domain_pack,
                risk_class=risk_class,
                overlays=overlays,
                model_id=model_id,
                feedback=latest_feedback,
                failure_context=failure_context,
            )
            accumulated_usage = merge_ai_usage(accumulated_usage, usage)
            customized_spec = self._apply_customization(base_spec, customization)
            customized_spec = ensure_task_agent_deliverable_briefs(customized_spec, overwrite=True)
            validation = validate_task_agent_spec(customized_spec)
            if validation.valid:
                return (
                    customized_spec,
                    customization,
                    accumulated_usage if accumulated_usage.request_count else None,
                )

            last_error_message = self._validation_error_message(stage=stage, validation=validation)
            self._record_customization_validation_rejection(
                stage=stage,
                error_message=last_error_message,
            )
            if validation_attempt > self.max_customization_validation_retries:
                break
            latest_feedback = self._preflight_feedback(
                existing_feedback=feedback,
                validation=validation,
                stage=stage,
                validation_attempt=validation_attempt,
            )
            log_coursegen_event(
                "task_agent_authoring_customization_preflight_retry",
                stage=stage,
                validation_attempt=validation_attempt + 1,
                error_codes=[issue.code for issue in validation.errors[:5]],
            )

        raise ValueError(last_error_message or "OpenAI customization preflight failed.")

    def _apply_customization(
        self,
        base_spec: TaskAgentServiceSpec,
        customization: TaskAgentCustomization,
    ) -> TaskAgentServiceSpec:
        spec = base_spec.model_copy(deep=True)
        if customization.summary:
            spec.summary = customization.summary.strip()
        if customization.public_endpoints:
            endpoints = []
            for endpoint in customization.public_endpoints:
                method = endpoint.method.strip().upper()
                path = endpoint.path.strip()
                if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"} or not path.startswith("/"):
                    continue
                endpoints.append(
                    EndpointSpec(method=method, path=path, required=endpoint.required)
                )
            if endpoints:
                spec.public_endpoints = endpoints

        deliverables_by_id = {deliverable.id: deliverable for deliverable in spec.deliverables}
        for patch in customization.deliverables:
            deliverable = deliverables_by_id.get(patch.id)
            if deliverable is None:
                continue
            if patch.title:
                deliverable.title = patch.title.strip()
            if patch.objective:
                deliverable.objective = patch.objective.strip()
            if patch.overlay_ids:
                deliverable.overlay_ids = _normalize_text_list(patch.overlay_ids)
            if patch.learning_outcomes:
                deliverable.learning_outcomes = _normalize_text_list(patch.learning_outcomes)
            if patch.learner_starter_surface is not None:
                authored_surface = deliverable.learner_starter_surface or LearnerStarterSurfaceSpec(
                    starter_summary="",
                    primary_editable_paths=[],
                    support_paths=[],
                    required_endpoints=[endpoint.model_copy(deep=True) for endpoint in spec.public_endpoints],
                    implementation_checklist=[],
                    domain_scenarios=[],
                )
                if patch.learner_starter_surface.starter_summary:
                    authored_surface.starter_summary = patch.learner_starter_surface.starter_summary.strip()
                if patch.learner_starter_surface.primary_editable_paths:
                    authored_surface.primary_editable_paths = _normalize_text_list(
                        patch.learner_starter_surface.primary_editable_paths
                    )
                if patch.learner_starter_surface.implementation_checklist:
                    authored_surface.implementation_checklist = _normalize_text_list(
                        patch.learner_starter_surface.implementation_checklist
                    )
                authored_domain_scenarios = _build_authored_domain_scenarios(
                    patch.learner_starter_surface.domain_scenarios
                )
                if authored_domain_scenarios:
                    authored_surface.domain_scenarios = authored_domain_scenarios
                deliverable.learner_starter_surface = authored_surface
            if patch.public_checks:
                editable_paths = learner_editable_paths_for_deliverable(spec, deliverable)
                public_checks: list[PublicCheckSpec] = []
                for index, check in enumerate(patch.public_checks, start=1):
                    request_method = (check.request_method or "POST").strip().upper()
                    request_path = (check.request_path or "").strip()
                    if request_method not in {"GET", "POST", "PUT", "PATCH", "DELETE"} or not request_path.startswith("/"):
                        continue
                    public_checks.append(
                        PublicCheckSpec(
                            id=check.id.strip() or f"{deliverable.id}_check_{index}",
                            title=(check.title or f"Visible check {index}").strip(),
                            learner_goal=(check.learner_goal or deliverable.objective).strip(),
                            request_method=request_method,
                            request_path=request_path,
                            request_body=_parse_request_body_json(check.request_body_json),
                            expected_status=check.expected_status or 200,
                            expected_response_contains=_normalize_text_list(check.expected_response_contains),
                            files_to_use=list(editable_paths),
                        )
                    )
                if public_checks:
                    deliverable.public_checks = public_checks
        return spec

    def _generate_customization(
        self,
        *,
        base_spec: TaskAgentServiceSpec,
        title: str,
        summary: str,
        package_type: PackageType,
        domain_pack: str | None,
        risk_class: RiskClass,
        overlays: list[str],
        model_id: str,
        feedback: str | None = None,
        failure_context: FailureContext | None = None,
    ) -> tuple[TaskAgentCustomization, AIUsageSummary | None]:
        config = self._config()
        client = (
            self._client(
                api_key=config.get("OPENAI_API_KEY", ""),
                base_url=config.get("OPENAI_BASE_URL"),
            )
            if self.client_factory is not None
            else None
        )
        prompt_payload = {
            "title": title,
            "summary": summary,
            "package_type": package_type.value,
            "domain_pack": domain_pack,
            "risk_class": risk_class.value,
            "overlays": overlays,
            "project_contract": base_spec.project_contract.model_dump(mode="json"),
            "runtime_plan": base_spec.project_contract.runtime_plan.model_dump(mode="json"),
            "runtime_dependencies": base_spec.runtime_dependencies.model_dump(mode="json"),
            "public_endpoints": [endpoint.model_dump(mode="json") for endpoint in base_spec.public_endpoints],
            "deliverables": [deliverable.model_dump(mode="json") for deliverable in base_spec.deliverables],
            "concrete_entity_hints": meaningful_domain_entities(base_spec.project_contract.core_entities),
            "title_slug_tokens": normalized_tokens(title),
            "feedback": feedback,
            "failure_context": failure_context.model_dump(mode="json") if failure_context is not None else None,
        }
        response = self._create_response_with_retries(
            client,
            model=model_id,
            api_key=config.get("OPENAI_API_KEY", ""),
            base_url=config.get("OPENAI_BASE_URL"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are authoring a learner-facing software project bundle. "
                        "Return JSON only. Do not invent tools, traces, approvals, confidence scores, or synthetic workflow semantics "
                        "unless the prompt explicitly requires them. "
                        "Focus on the real public endpoints, deliverables, starter guidance, concrete scenarios, and visible checks. "
                        "Use concrete resource nouns from `concrete_entity_hints` whenever possible. "
                        "The `project_contract.core_entities` and `project_contract.system_kind` are the project's domain spine. "
                        "They MUST be concrete, multi-word, course-specific noun phrases drawn from the brief and the runtime plan — "
                        "for example 'prompt template', 'promptfoo eval suite', 'retrieval corpus', 'grounded response', "
                        "'inventory reservation', 'rate limit decision'. "
                        "They MUST NOT be placeholder articles like 'a small', 'the X', 'an item'; "
                        "single common nouns like 'thing', 'item', 'object', 'request', 'response', 'service request'; "
                        "the course title turned into a noun; or generic words that could apply to any backend. "
                        "If `concrete_entity_hints` is empty or itself looks generic, derive entities from the brief's "
                        "tech_stack and deliverable workload, not from a template. `system_kind` MUST be a 3-6 word phrase "
                        "that a learner could read and immediately recognize what THIS project is for, NOT a templated "
                        "'<entity> service' or '<entity> system' pattern. "
                        "Do not expose public paths that are just the course title turned into a URL slug, and do not use words like "
                        "`service`, `system`, `api`, `backend`, `bot`, or `agent` as the primary public resource path. "
                        "Starter scenarios and visible checks should use concrete domain language, not labels like `Primary request` or `Edge or failure path`. "
                        "Deliverable titles such as `Service contract`, `Operational hardening`, or other generic scaffolding labels are too weak; "
                        "when a public check needs a body, put a compact JSON object string in `request_body_json` and leave it null for bodyless requests. "
                        "keep the titles grounded in the project's actual resources and workflows. "
                        "For every deliverable, set `learner_starter_surface.primary_editable_paths` to the relative paths "
                        "the learner edits in the chosen stack and the layout you will author. Pick paths that match the creator-selected "
                        "implementation language, application framework, and package manager rather than copying a previous stack's defaults. "
                        "Do not leave `primary_editable_paths` empty: a starter without a primary file is invalid."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, indent=2)},
            ],
            temperature=0.2,
            text_format=TaskAgentCustomization,
        )
        customization = response.output_parsed
        if customization is None:
            raise ValueError("OpenAI authoring returned no parsed customization.")
        from app.services.llm_router import usage_summary_from_response

        return customization, usage_summary_from_response(response, model_id=model_id)

    def _create_response_with_retries(
        self,
        client,
        *,
        model: str,
        api_key: str,
        base_url: str | None,
        input: list[dict[str, Any]],
        temperature: float,
        text_format: type[BaseModel],
    ):
        last_error: Exception | None = None
        for attempt in range(1, self.max_request_retries + 2):
            log_coursegen_event(
                "task_agent_authoring_request_attempt_started",
                model_id=model,
                attempt=attempt,
            )
            try:
                if self.client_factory is not None:
                    return client.responses.parse(
                        model=model,
                        input=input,
                        temperature=temperature,
                        text_format=text_format,
                        timeout=self.request_timeout_s,
                    )
                # Route the structured-output call through LLMRouter. The
                # router resolves the active provider (Anthropic default,
                # OpenAI fallback) and the tier→model mapping.
                from app.services.llm_router import (
                    LLMTier,
                    get_default_router,
                    messages_to_system_user,
                )

                router = get_default_router()
                system, user = messages_to_system_user(input)
                return router.parse_structured(
                    tier=LLMTier.sonnet,
                    system=system,
                    user=user,
                    text_format=text_format,
                    request_timeout_s=self.request_timeout_s,
                    max_tokens=16_000,
                    extra_request_kwargs={"temperature": temperature},
                )
            except Exception as exc:  # pragma: no cover
                last_error = exc
                log_coursegen_event(
                    "task_agent_authoring_request_attempt_failed",
                    model_id=model,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt > self.max_request_retries:
                    break
                time.sleep(min(2**attempt, 4))
        assert last_error is not None
        raise last_error

    def _record_customization_validation_rejection(self, *, stage: str, error_message: str) -> None:
        self._customization_validation_rejection_count += 1
        self._last_customization_validation_error = error_message
        log_coursegen_event(
            "task_agent_authoring_customization_rejected",
            stage=stage,
            rejection_count=self._customization_validation_rejection_count,
            error=error_message,
        )

    def _validation_error_message(self, *, stage: str, validation) -> str:  # noqa: ANN001
        prefix = "OpenAI customization produced an invalid spec" if stage == "generate" else "OpenAI revision produced an invalid spec"
        return prefix + ": " + "; ".join(error.code for error in validation.errors[:3])

    def _preflight_feedback(
        self,
        *,
        existing_feedback: str | None,
        validation,
        stage: str,
        validation_attempt: int,
    ) -> str:
        sections: list[str] = []
        if existing_feedback and existing_feedback.strip():
            sections.append(existing_feedback.strip())
        sections.append(
            (
                "The last OpenAI customization was rejected by deterministic validation. "
                if stage == "generate"
                else "The last OpenAI revision was rejected by deterministic validation. "
            )
            + f"Fix these exact issues on preflight attempt {validation_attempt + 1}:"
        )
        for issue in validation.errors[:8]:
            sections.append(f"- {issue.code} at {issue.location}: {issue.message}")
        sections.append(
            "Return another full customization JSON response that keeps the creator-selected stack intact and fixes the listed validation issues."
        )
        return "\n".join(sections)

    def _config(self) -> dict[str, str]:
        config = load_openai_env_file(self.env_file)
        env_values = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
            "OPENAI_MODEL": os.environ.get("OPENAI_MODEL"),
        }
        for key, value in env_values.items():
            if value:
                config[key] = strip_quotes(value)
        return config

    def _openai_sdk_available(self) -> bool:
        try:
            import openai  # noqa: F401

            return True
        except Exception:
            return False

    def _client(self, *, api_key: str, base_url: str | None):
        if self.client_factory is not None:
            return self.client_factory(api_key=api_key, base_url=base_url)
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
