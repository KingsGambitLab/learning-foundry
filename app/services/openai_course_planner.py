from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.domain.course import (
    CourseGenerationSource,
    CourseGenerationStatus,
    CreateCourseModuleRequest,
    GenerateCourseFromBriefRequest,
    GeneratedCoursePlan,
    SuggestLearningOutcomesRequest,
)
from app.services.assignment_design_inference import infer_assignment_design


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
        self.env_file = env_file or os.environ.get("COURSE_GEN_OPENAI_ENV_FILE") or os.environ.get("OPENAI_ENV_FILE")
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

    def plan_course(self, request: GenerateCourseFromBriefRequest) -> tuple[GeneratedCoursePlan, CourseGenerationStatus]:
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
                            "Return JSON only. Propose a clear course title, summary, package type, and module ladder. "
                            "Keep module ladders concrete, realistic, and teachable."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
                temperature=0.2,
            )
            raw_text = getattr(response, "output_text", "")
            raw_plan = self._extract_json(raw_text)
            plan = self._normalize_raw_plan(request, raw_plan)
        except Exception as exc:  # pragma: no cover - network and SDK failures vary
            raise OpenAICourseGenerationError(str(exc)) from exc

        return plan, CourseGenerationStatus(
            provider="openai",
            available=True,
            source=CourseGenerationSource.openai_live,
            message=f"Generated course plan with OpenAI model `{status.model_id}`.",
            sdk_installed=True,
            api_key_present=True,
            model_id=status.model_id,
            env_file=status.env_file,
        )

    def suggest_learning_outcomes(
        self,
        request: SuggestLearningOutcomesRequest,
    ) -> tuple[list[str], CourseGenerationStatus]:
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
        except Exception as exc:  # pragma: no cover - network and SDK failures vary
            raise OpenAICourseGenerationError(str(exc)) from exc

        return outcomes, CourseGenerationStatus(
            provider="openai",
            available=True,
            source=CourseGenerationSource.openai_live,
            message=f"Suggested learning outcomes with OpenAI model `{status.model_id}`.",
            sdk_installed=True,
            api_key_present=True,
            model_id=status.model_id,
            env_file=status.env_file,
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
            learning_outcomes=request.learning_outcomes,
            package_type_hint=request.package_type_hint,
        ).design_spec
        if shared_design_spec is None:
            raise ValueError("This brief is outside the current learner-ready generation scope.")

        modules: list[CreateCourseModuleRequest] = []
        for raw_module in raw_plan.get("modules", []):
            if not isinstance(raw_module, dict):
                continue
            module_title = str(raw_module.get("title") or "").strip()
            if not module_title:
                continue
            module_summary = str(raw_module.get("summary") or module_title).strip()
            module_outcomes = [
                str(item).strip()
                for item in raw_module.get("learning_outcomes", [])
                if str(item).strip()
            ][:3]
            module_design_spec = infer_assignment_design(
                title=module_title,
                problem_statement=module_summary,
                learning_outcomes=module_outcomes or request.learning_outcomes,
                package_type_hint=request.package_type_hint,
            ).design_spec or shared_design_spec
            modules.append(
                CreateCourseModuleRequest(
                    module_slug=raw_module.get("module_slug"),
                    title=module_title,
                    summary=module_summary,
                    learning_outcomes=module_outcomes,
                    design_spec=module_design_spec,
                    domain_pack_hint=module_design_spec.domain_pack,
                    overlays_hint=list(module_design_spec.overlays),
                )
            )

        if not modules:
            raise ValueError("The OpenAI response did not include any valid course modules.")

        return GeneratedCoursePlan(
            title=title,
            summary=summary,
            package_type=package_type,
            shared_design_spec=shared_design_spec,
            modules=modules,
            notes=[str(item).strip() for item in raw_plan.get("notes", []) if str(item).strip()],
        )

    def _prompt_payload(self, request: GenerateCourseFromBriefRequest) -> dict[str, Any]:
        hint = request.package_type_hint.value if request.package_type_hint else "infer from the brief"
        return {
            "goal": request.goal,
            "learning_outcomes": request.learning_outcomes,
            "title_hint": request.title,
            "package_type_hint": hint,
            "constraints": [
                "Produce 4 to 8 modules for progressive courses, or 3 to 6 modules for survey courses.",
                "Every module needs a concrete title, summary, and one to three outcomes.",
                "Prefer progressive codebase courses when one evolving system is the teaching shape.",
                "Prefer survey courses when modules are independent systems.",
                "Keep modules tightly tied to the work the learner will actually build.",
            ],
            "required_output_shape": {
                "title": "string",
                "summary": "string",
                "package_type": "survey_course | progressive_codebase_course",
                "modules": [
                    {
                        "module_slug": "optional slug",
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
            config.update(self._load_env_file(self.env_file))
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
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

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
