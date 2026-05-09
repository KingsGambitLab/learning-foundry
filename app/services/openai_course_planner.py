from __future__ import annotations

import json
import os
from typing import Any

from app.domain.ai import AIUsageSummary
from app.domain.course import (
    CourseGenerationSource,
    CourseGenerationStatus,
    CreateCourseDeliverableRequest,
    GenerateCourseFromBriefRequest,
    GeneratedCoursePlan,
    SuggestLearningOutcomesRequest,
)
from app.services.assignment_design_inference import infer_assignment_design
from app.services.openai_runtime_support import (
    extract_openai_usage,
    load_openai_env_file,
    resolve_openai_env_file,
    strip_quotes,
)


class OpenAICoursePlannerUnavailable(RuntimeError):
    """Raised when live OpenAI course planning is not available."""


class OpenAICourseGenerationError(RuntimeError):
    """Raised when OpenAI course generation fails after fallback attempts."""


class OpenAICoursePlanner:
    def __init__(
        self,
        *,
        enabled: bool = True,
        env_file: str | None = None,
        model: str | None = None,
        client_factory=None,
    ) -> None:
        self.enabled = enabled
        self.env_file = resolve_openai_env_file(env_file)
        self.model = model
        self.client_factory = client_factory

    def status(self) -> CourseGenerationStatus:
        config = self._config()
        sdk_installed = self._openai_sdk_available()
        api_key_present = bool(config.get("OPENAI_API_KEY"))
        model_id = config.get("OPENAI_MODEL") or self.model or "gpt-5.4"

        if not self.enabled:
            return CourseGenerationStatus(
                provider="openai",
                available=False,
                source=CourseGenerationSource.deterministic_fallback,
                message="Live OpenAI course generation is disabled for this app instance.",
                sdk_installed=sdk_installed,
                api_key_present=api_key_present,
                model_id=model_id,
                env_file=self.env_file,
            )
        if not sdk_installed:
            return CourseGenerationStatus(
                provider="openai",
                available=False,
                source=CourseGenerationSource.deterministic_fallback,
                message="The OpenAI Python SDK is not installed, so course generation will use deterministic fallback planning.",
                sdk_installed=False,
                api_key_present=api_key_present,
                model_id=model_id,
                env_file=self.env_file,
            )
        if not api_key_present:
            return CourseGenerationStatus(
                provider="openai",
                available=False,
                source=CourseGenerationSource.deterministic_fallback,
                message="OPENAI_API_KEY is not configured, so course generation will use deterministic fallback planning.",
                sdk_installed=True,
                api_key_present=False,
                model_id=model_id,
                env_file=self.env_file,
            )
        return CourseGenerationStatus(
            provider="openai",
            available=True,
            source=CourseGenerationSource.openai_live,
            message="OpenAI course planning is ready to generate course plans from the brief.",
            sdk_installed=True,
            api_key_present=True,
            model_id=model_id,
            env_file=self.env_file,
        )

    def plan_course(
        self,
        request: GenerateCourseFromBriefRequest,
    ) -> tuple[GeneratedCoursePlan, CourseGenerationStatus, AIUsageSummary | None]:
        status = self.status()
        if not status.available:
            raise OpenAICoursePlannerUnavailable(status.message)

        config = self._config()
        client = self._client(
            api_key=config.get("OPENAI_API_KEY", ""),
            base_url=config.get("OPENAI_BASE_URL"),
        )
        prompt = self._prompt_payload(request)

        try:
            response = client.responses.create(
                model=status.model_id or "gpt-5.4",
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You design practical engineering courses for a hands-on backend learning platform. "
                            "Return JSON only. Propose a clear course title, summary, package type, and deliverable plan. "
                            "Use the inferred project contract, runtime binding, and runtime plan as the source of truth. "
                            "Each deliverable must represent a real engineering concern or subsystem, not a maturity stage. "
                            "Avoid generic sequences like 'run contract', 'tooling', or 'approvals' unless the project contract explicitly requires them."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
                temperature=0.2,
            )
            raw_text = getattr(response, "output_text", "")
            raw_plan = self._extract_json(raw_text)
            plan = self._normalize_raw_plan(request, raw_plan)
            usage = extract_openai_usage(response, status.model_id)
        except Exception as exc:  # pragma: no cover - network and SDK failures vary
            raise OpenAICourseGenerationError(str(exc)) from exc

        return (
            plan,
            CourseGenerationStatus(
                provider="openai",
                available=True,
                source=CourseGenerationSource.openai_live,
                message=f"Generated course plan with OpenAI model `{status.model_id}`.",
                sdk_installed=True,
                api_key_present=True,
                model_id=status.model_id,
                env_file=status.env_file,
            ),
            usage,
        )

    def suggest_learning_outcomes(
        self,
        request: SuggestLearningOutcomesRequest,
    ) -> tuple[list[str], CourseGenerationStatus, AIUsageSummary | None]:
        status = self.status()
        if not status.available:
            raise OpenAICoursePlannerUnavailable(status.message)

        config = self._config()
        client = self._client(
            api_key=config.get("OPENAI_API_KEY", ""),
            base_url=config.get("OPENAI_BASE_URL"),
        )
        prompt = self._outcome_prompt_payload(request)

        try:
            response = client.responses.create(
                model=status.model_id or "gpt-5.4",
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You help course authors draft concrete learning outcomes for engineering projects. "
                            "Return JSON only. Write 4 to 6 concise, teachable outcomes that are specific enough to guide a hands-on build. "
                            "Use action-oriented language. Avoid vague outcomes like 'understand the topic'."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
                temperature=0.2,
            )
            payload = self._extract_json(getattr(response, "output_text", ""))
            outcomes = [
                str(item).strip()
                for item in payload.get("learning_outcomes", [])
                if str(item).strip()
            ][:6]
            if not outcomes:
                raise ValueError("The OpenAI response did not contain any learning outcomes.")
            usage = extract_openai_usage(response, status.model_id)
        except Exception as exc:  # pragma: no cover - network and SDK failures vary
            raise OpenAICourseGenerationError(str(exc)) from exc

        return (
            outcomes,
            CourseGenerationStatus(
                provider="openai",
                available=True,
                source=CourseGenerationSource.openai_live,
                message=f"Suggested learning outcomes with OpenAI model `{status.model_id}`.",
                sdk_installed=True,
                api_key_present=True,
                model_id=status.model_id,
                env_file=status.env_file,
            ),
            usage,
        )

    def _normalize_raw_plan(
        self,
        request: GenerateCourseFromBriefRequest,
        raw_plan: dict[str, Any],
    ) -> GeneratedCoursePlan:
        title = str(raw_plan.get("title") or request.title or "Generated Course Draft").strip()
        summary = str(raw_plan.get("summary") or request.goal).strip()
        package_type_raw = raw_plan.get("package_type") or request.package_type_hint or "progressive_codebase_course"
        package_type = package_type_raw if hasattr(package_type_raw, "value") else str(package_type_raw)
        shared_design_spec = infer_assignment_design(
            title=title,
            problem_statement=request.goal,
            package_type_hint=request.package_type_hint,
            starter_type=request.creator_setup.starter_type,
            implementation_language=request.creator_setup.implementation_language,
            application_framework=request.creator_setup.application_framework,
            primary_database=request.creator_setup.primary_database,
            cache_backend=request.creator_setup.cache_backend,
            tech_stack=list(request.creator_setup.tech_stack),
            data_sources=list(request.creator_setup.data_sources),
        ).design_spec
        if shared_design_spec is None:
            raise ValueError("This brief is outside the current learner-ready generation scope.")

        deliverable_items = raw_plan.get("deliverables")
        if not isinstance(deliverable_items, list):
            deliverable_items = raw_plan.get("deliverables", [])

        deliverables: list[CreateCourseDeliverableRequest] = []
        for raw_deliverable in deliverable_items:
            if not isinstance(raw_deliverable, dict):
                continue
            deliverable_title = str(raw_deliverable.get("title") or "").strip()
            if not deliverable_title:
                continue
            deliverable_summary = str(raw_deliverable.get("summary") or deliverable_title).strip()
            deliverable_outcomes = [
                str(item).strip()
                for item in raw_deliverable.get("learning_outcomes", [])
                if str(item).strip()
            ][:3]
            deliverable_design_spec = infer_assignment_design(
                title=deliverable_title,
                problem_statement=deliverable_summary,
                package_type_hint=request.package_type_hint,
                starter_type=request.creator_setup.starter_type,
                implementation_language=request.creator_setup.implementation_language,
                application_framework=request.creator_setup.application_framework,
                primary_database=request.creator_setup.primary_database,
                cache_backend=request.creator_setup.cache_backend,
                tech_stack=list(request.creator_setup.tech_stack),
                data_sources=list(request.creator_setup.data_sources),
            ).design_spec or shared_design_spec
            deliverables.append(
                CreateCourseDeliverableRequest(
                    deliverable_slug=raw_deliverable.get("deliverable_slug") or raw_deliverable.get("deliverable_slug"),
                    title=deliverable_title,
                    summary=deliverable_summary,
                    learning_outcomes=deliverable_outcomes,
                    design_spec=deliverable_design_spec,
                    domain_pack_hint=deliverable_design_spec.domain_pack,
                    overlays_hint=list(deliverable_design_spec.overlays),
                )
            )

        if not deliverables:
            raise ValueError("The OpenAI response did not include any valid course deliverables.")

        return GeneratedCoursePlan(
            title=title,
            summary=summary,
            package_type=package_type,
            shared_design_spec=shared_design_spec,
            deliverables=deliverables,
            notes=[str(item).strip() for item in raw_plan.get("notes", []) if str(item).strip()],
        )

    def _prompt_payload(self, request: GenerateCourseFromBriefRequest) -> dict[str, Any]:
        hint = request.package_type_hint.value if request.package_type_hint else "infer from the brief"
        design_spec = infer_assignment_design(
            title=request.title or request.goal,
            problem_statement=request.goal,
            package_type_hint=request.package_type_hint,
            starter_type=request.creator_setup.starter_type,
            implementation_language=request.creator_setup.implementation_language,
            application_framework=request.creator_setup.application_framework,
            primary_database=request.creator_setup.primary_database,
            cache_backend=request.creator_setup.cache_backend,
            tech_stack=list(request.creator_setup.tech_stack),
            data_sources=list(request.creator_setup.data_sources),
        ).design_spec
        creator_setup = {
            "starter_type": request.creator_setup.starter_type.value if request.creator_setup.starter_type else None,
            "implementation_language": request.creator_setup.implementation_language,
            "application_framework": request.creator_setup.application_framework,
            "primary_database": request.creator_setup.primary_database,
            "cache_backend": request.creator_setup.cache_backend,
            "tech_stack": list(request.creator_setup.tech_stack),
            "data_sources": [source.model_dump(mode="json") for source in request.creator_setup.data_sources],
        }
        return {
            "goal": request.goal,
            "title_hint": request.title,
            "package_type_hint": hint,
            "creator_setup": creator_setup,
            "design_signal": (
                {
                    "course_structure": design_spec.course_structure.model_dump(mode="json"),
                    "runtime_dependencies": design_spec.runtime_dependencies.model_dump(mode="json"),
                    "capabilities": design_spec.capabilities.model_dump(mode="json"),
                    "project_contract": design_spec.project_contract.model_dump(mode="json"),
                }
                if design_spec is not None
                else None
            ),
            "constraints": [
                "Produce 4 to 8 deliverables for progressive courses, or 3 to 6 deliverables for survey courses.",
                "Use the creator setup as a real input to the deliverable plan and runtime assumptions.",
                "Return a deliverables array for one shared project, not a tutorial sequence.",
                "Every deliverable needs a concrete title, summary, and one to three learning outcomes derived from the work learners will do.",
                "Each deliverable should own a distinct engineering concern, subsystem, or operational capability.",
                "Do not generate generic agentic deliverables unless the project contract explicitly mentions tool routing, approvals, or operator handoffs.",
                "Prefer progressive codebase courses when one evolving system is the teaching shape.",
                "Prefer survey courses when deliverables are independent systems.",
                "Keep deliverables tightly tied to the work the learner will actually build.",
            ],
            "required_output_shape": {
                "title": "string",
                "summary": "string",
                "package_type": "survey_course | progressive_codebase_course",
                "deliverables": [
                    {
                        "deliverable_slug": "optional slug",
                        "title": "string",
                        "summary": "string",
                        "learning_outcomes": ["1 to 3 concise outcomes"],
                    }
                ],
                "notes": ["short notes"],
            },
        }

    def _outcome_prompt_payload(self, request: SuggestLearningOutcomesRequest) -> dict[str, Any]:
        return {
            "goal": request.goal,
            "title_hint": request.title,
            "constraints": [
                "Return 4 to 6 outcomes.",
                "Each outcome should describe a concrete capability the learner will build or verify.",
                "Favor production-minded outcomes when the goal implies a real deployed system.",
                "Keep each outcome to one short sentence or phrase.",
            ],
            "required_output_shape": {
                "learning_outcomes": [
                    "A concrete, editable outcome written for a hands-on engineering course."
                ],
            },
        }

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
            config.update(load_openai_env_file(self.env_file))
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_BASE_URL",
            "COURSE_GEN_OPENAI_PLANNER_MODEL",
        ):
            value = os.environ.get(key)
            if value:
                config[key] = value
        if "OPENAI_MODEL" not in config:
            config["OPENAI_MODEL"] = config.get("COURSE_GEN_OPENAI_PLANNER_MODEL") or self.model or "gpt-5.4"
        return config

    def _load_env_file(self, path: str) -> dict[str, str]:
        return load_openai_env_file(path)

    def _strip_quotes(self, value: str) -> str:
        return strip_quotes(value)

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("OpenAI returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            return json.loads(text[start : end + 1])

    def _openai_sdk_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True
