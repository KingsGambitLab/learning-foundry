from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.domain.grading import (
    AssignmentGradeReport,
    GradeStatus,
    LearnerReviewGuidance,
    ReviewAreaGradeReport,
    TestGradeResult,
)
from app.domain.publish import LearnerCoursePackage, LearnerDeliverablePackage
from app.domain.task_agent import TaskAgentServiceSpec
from app.services.openai_runtime_support import load_openai_env_file, resolve_openai_env_file, strip_quotes


class OpenAILearnerFeedbackService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        env_file: str | None = None,
        model: str | None = None,
        client_factory=None,
        max_editable_file_chars: int = 12_000,
    ) -> None:
        self.enabled = enabled
        self.env_file = resolve_openai_env_file(env_file)
        self.model = model
        self.client_factory = client_factory
        self.max_editable_file_chars = max_editable_file_chars

    def available(self) -> bool:
        config = self._config()
        return self.enabled and self._openai_sdk_available() and bool(config.get("OPENAI_API_KEY"))

    def annotate_assignment_report(
        self,
        *,
        project_brief_markdown: str,
        learner_package: LearnerCoursePackage,
        assignment_report: AssignmentGradeReport,
        workspace_root: str | Path,
        spec: TaskAgentServiceSpec,
    ) -> AssignmentGradeReport:
        if assignment_report.status == GradeStatus.passed:
            return assignment_report
        failed_review_areas = [area for area in assignment_report.review_areas if area.grade_report.status == GradeStatus.failed]
        if not failed_review_areas:
            return assignment_report
        if not self.available():
            return assignment_report

        config = self._config()
        client = self._client(
            api_key=config.get("OPENAI_API_KEY", ""),
            base_url=config.get("OPENAI_BASE_URL"),
        )
        editable_files = self._editable_file_context(workspace_root, spec.runtime_dependencies.editable_files)
        updated_review_areas = []
        for review_area in assignment_report.review_areas:
            if review_area.grade_report.status != GradeStatus.failed:
                updated_review_areas.append(review_area)
                continue
            deliverable = self._deliverable_package(learner_package, review_area.deliverable_id)
            if deliverable is None:
                updated_review_areas.append(review_area)
                continue
            try:
                prompt_payload = self._prompt_payload(
                    project_brief_markdown=project_brief_markdown,
                    failed_review_area=review_area.model_dump(mode="json"),
                    passed_review_areas=[
                        {
                            "title": passed_area.title,
                            "objective": passed_area.objective,
                            "status": passed_area.grade_report.status,
                            "summaries": [result.summary for result in passed_area.grade_report.results],
                        }
                        for passed_area in assignment_report.review_areas
                        if passed_area.deliverable_id != review_area.deliverable_id
                        and passed_area.grade_report.status == GradeStatus.passed
                    ],
                    editable_files=editable_files,
                    deliverable=deliverable,
                )
                response = client.responses.create(
                    model=config.get("OPENAI_MODEL") or self.model or "gpt-5.4",
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are a thoughtful tech lead reviewing a learner submission for a backend project. "
                                "Use only the provided project brief, learner-visible expectations, code, and grader results. "
                                "Explain the fundamental gap without spoon-feeding implementation. "
                                "Be concrete about what already looks strong, what is weak, and where the learner should investigate next. "
                                "Talk like a calm, technically sharp tech lead. Do not write code. "
                                "Do not mention hidden tests, graders, or internal tooling. "
                                "Return JSON only with non-empty keys: strengths, fundamental_gap, why_it_matters, likely_root_cause, investigation_steps, learner_feedback."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(prompt_payload, indent=2),
                        },
                    ],
                    temperature=0.2,
                )
                feedback = LearnerReviewGuidance.model_validate(
                    self._normalize_feedback_payload(
                        self._extract_json(getattr(response, "output_text", "")),
                        failed_review_area=review_area,
                        passed_review_areas=[
                            passed_area
                            for passed_area in assignment_report.review_areas
                            if passed_area.deliverable_id != review_area.deliverable_id
                            and passed_area.grade_report.status == GradeStatus.passed
                        ],
                        deliverable=deliverable,
                        editable_files=editable_files,
                    )
                )
                updated_review_areas.append(review_area.model_copy(update={"feedback": feedback}))
            except Exception:
                updated_review_areas.append(review_area)

        return assignment_report.model_copy(update={"review_areas": updated_review_areas})

    def _prompt_payload(
        self,
        *,
        project_brief_markdown: str,
        failed_review_area: dict[str, Any],
        passed_review_areas: list[dict[str, Any]],
        editable_files: dict[str, str],
        deliverable: LearnerDeliverablePackage,
    ) -> dict[str, Any]:
        learner_visible_expectations = {
            "deliverable_title": deliverable.title,
            "deliverable_objective": deliverable.objective,
            "editable_files": [path for path in deliverable.visible_files if path in editable_files] or ["app.py"],
            "public_checks": [
                {
                    "title": check.title,
                    "learner_goal": check.learner_goal,
                    "expected_assertions": check.expected_assertions,
                    "files_to_use": check.files_to_use,
                }
                for check in deliverable.public_checks
            ],
        }
        return {
            "project_brief": project_brief_markdown,
            "failed_deliverable": failed_review_area,
            "passed_deliverables": passed_review_areas,
            "editable_files": editable_files,
            "learner_visible_expectations": learner_visible_expectations,
        }

    def _editable_file_context(self, workspace_root: str | Path, editable_files: list[str]) -> dict[str, str]:
        root = Path(workspace_root)
        context: dict[str, str] = {}
        for relative_path in editable_files[:4]:
            target = root / relative_path
            if not target.exists() or not target.is_file():
                continue
            content = target.read_text(encoding="utf-8")
            if len(content) > self.max_editable_file_chars:
                content = content[: self.max_editable_file_chars] + "\n...<truncated>..."
            context[relative_path] = content
        return context

    def _deliverable_package(
        self,
        learner_package: LearnerCoursePackage,
        deliverable_id: str,
    ) -> LearnerDeliverablePackage | None:
        return next(
            (deliverable for deliverable in learner_package.deliverables if deliverable.deliverable_id == deliverable_id),
            None,
        )

    def _normalize_feedback_payload(
        self,
        payload: dict[str, Any],
        *,
        failed_review_area: ReviewAreaGradeReport,
        passed_review_areas: list[ReviewAreaGradeReport],
        deliverable: LearnerDeliverablePackage,
        editable_files: dict[str, str],
    ) -> dict[str, Any]:
        def ensure_list(value: Any, *, limit: int) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                cleaned = value.strip()
                return [cleaned] if cleaned else []
            if not isinstance(value, list):
                return []
            items = [str(item).strip() for item in value if str(item).strip()]
            return self._dedupe(items)[:limit]

        fundamental_gap = str(payload.get("fundamental_gap") or "").strip()
        if not fundamental_gap:
            fundamental_gap = self._fallback_fundamental_gap(failed_review_area, deliverable)

        strengths = ensure_list(payload.get("strengths"), limit=4)
        if not strengths:
            strengths = self._fallback_strengths(
                failed_review_area=failed_review_area,
                passed_review_areas=passed_review_areas,
            )

        why_it_matters = ensure_list(payload.get("why_it_matters"), limit=4)
        if not why_it_matters:
            why_it_matters = self._fallback_why_it_matters(deliverable)

        likely_root_cause = ensure_list(payload.get("likely_root_cause"), limit=5)
        if not likely_root_cause:
            likely_root_cause = self._fallback_likely_root_cause(failed_review_area, deliverable)

        investigation_steps = ensure_list(payload.get("investigation_steps"), limit=6)
        if not investigation_steps:
            investigation_steps = self._fallback_investigation_steps(
                failed_review_area=failed_review_area,
                deliverable=deliverable,
                editable_files=editable_files,
            )

        learner_feedback = str(payload.get("learner_feedback") or "").strip()
        if not learner_feedback:
            learner_feedback = self._fallback_learner_feedback(
                deliverable=deliverable,
                strengths=strengths,
                fundamental_gap=fundamental_gap,
                investigation_steps=investigation_steps,
            )

        return {
            "strengths": strengths,
            "fundamental_gap": fundamental_gap,
            "why_it_matters": why_it_matters,
            "likely_root_cause": likely_root_cause,
            "investigation_steps": investigation_steps,
            "learner_feedback": learner_feedback,
        }

    def _fallback_strengths(
        self,
        *,
        failed_review_area: ReviewAreaGradeReport,
        passed_review_areas: list[ReviewAreaGradeReport],
    ) -> list[str]:
        strengths: list[str] = []
        if failed_review_area.grade_report.passed_tests:
            strengths.append(
                f"Some of the checks for {failed_review_area.title} are already passing, so part of this deliverable is in place."
            )
        for area in passed_review_areas[:2]:
            strengths.append(f"{area.title} is already passing its current review checks.")
        if not strengths:
            strengths.append("The project is reaching the review run, so the remaining work is in behavior rather than setup.")
        return self._dedupe(strengths)[:4]

    def _fallback_fundamental_gap(
        self,
        failed_review_area: ReviewAreaGradeReport,
        deliverable: LearnerDeliverablePackage,
    ) -> str:
        failed_results = [result for result in failed_review_area.grade_report.results if result.status == GradeStatus.failed]
        if failed_results:
            first_summary = failed_results[0].summary.strip().rstrip(".")
            return f"{deliverable.title} is still missing behavior that satisfies the review contract: {first_summary}."
        return f"{deliverable.title} is not yet meeting the review contract for this deliverable."

    def _fallback_why_it_matters(self, deliverable: LearnerDeliverablePackage) -> list[str]:
        reasons: list[str] = []
        if deliverable.objective:
            reasons.append(deliverable.objective)
        for check in deliverable.public_checks[:2]:
            if check.learner_goal:
                reasons.append(check.learner_goal)
        if not reasons:
            reasons.append(f"This deliverable is part of the core project behavior the reviewer expects to see working.")
        return self._dedupe(reasons)[:4]

    def _fallback_likely_root_cause(
        self,
        failed_review_area: ReviewAreaGradeReport,
        deliverable: LearnerDeliverablePackage,
    ) -> list[str]:
        diagnostics = " ".join(self._diagnostic_clues(failed_review_area.grade_report.results)).lower()
        likely: list[str] = []
        if "citation" in diagnostics:
            likely.append("The response assembly path is probably not attaching or clearing citations consistently.")
        if "abstain" in diagnostics or "unsupported" in diagnostics:
            likely.append("The unsupported-case branch likely is not returning the expected abstention behavior.")
        if "contract" in diagnostics or "subset" in diagnostics or "output" in diagnostics:
            likely.append("The final response shape still diverges from the contract expected by the review checks.")
        if "timeout" in diagnostics or "connection refused" in diagnostics:
            likely.append("One code path is still failing before it returns the behavior this deliverable is supposed to provide.")
        if not likely:
            likely.append(
                f"A branch tied to {deliverable.title} is still returning behavior that does not line up with the expected contract."
            )
        return self._dedupe(likely)[:5]

    def _fallback_investigation_steps(
        self,
        *,
        failed_review_area: ReviewAreaGradeReport,
        deliverable: LearnerDeliverablePackage,
        editable_files: dict[str, str],
    ) -> list[str]:
        steps: list[str] = []
        if deliverable.public_checks:
            first_check = deliverable.public_checks[0]
            if first_check.title:
                steps.append(f"Run the visible check '{first_check.title}' locally and inspect the raw response it produces.")
            if first_check.expected_assertions:
                steps.append(
                    f"Compare the response against these expectations: {self._join_phrases(first_check.expected_assertions[:2])}."
                )
        focus_files = [path for path in deliverable.visible_files if path in editable_files] or list(editable_files.keys())[:2]
        if focus_files:
            steps.append(f"Trace the code path in {self._join_phrases(focus_files[:2])} that builds the final response for this deliverable.")
        failed_summaries = [
            result.summary.strip().rstrip(".")
            for result in failed_review_area.grade_report.results
            if result.status == GradeStatus.failed and result.summary.strip()
        ]
        if failed_summaries:
            steps.append(f"Use the failing review summary as a checklist while you retest: {failed_summaries[0]}.")
        if not steps:
            steps.append("Run the visible checks locally, inspect the response shape, and then trace the code that assembles the final output.")
        return self._dedupe(steps)[:6]

    def _fallback_learner_feedback(
        self,
        *,
        deliverable: LearnerDeliverablePackage,
        strengths: list[str],
        fundamental_gap: str,
        investigation_steps: list[str],
    ) -> str:
        strengths_line = strengths[0] if strengths else "The project is already reaching the review run."
        next_step = investigation_steps[0] if investigation_steps else "Run the visible checks locally and inspect the response shape."
        return (
            f"{strengths_line} The main gap is in {deliverable.title.lower()}: {fundamental_gap} "
            f"Start by {next_step[0].lower() + next_step[1:] if next_step else 'reviewing the visible checks and the final response path.'}"
        )

    def _diagnostic_clues(self, results: list[TestGradeResult]) -> list[str]:
        clues: list[str] = []
        for result in results:
            if result.status != GradeStatus.failed:
                continue
            if result.summary.strip():
                clues.append(result.summary.strip())
            clues.extend(diag.strip() for diag in result.diagnostics if diag.strip())
        return clues

    def _join_phrases(self, items: list[str]) -> str:
        quoted = [item.strip() for item in items if item.strip()]
        if not quoted:
            return ""
        if len(quoted) == 1:
            return quoted[0]
        if len(quoted) == 2:
            return f"{quoted[0]} and {quoted[1]}"
        return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            normalized = " ".join(item.split()).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalized)
        return ordered

    def _config(self) -> dict[str, str]:
        config: dict[str, str] = {}
        if self.env_file:
            config.update(self._load_env_file(self.env_file))
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "COURSE_GEN_OPENAI_FEEDBACK_MODEL", "OPENAI_MODEL"):
            value = os.environ.get(key)
            if value:
                config[key] = value
        if "OPENAI_MODEL" not in config:
            config["OPENAI_MODEL"] = config.get("COURSE_GEN_OPENAI_FEEDBACK_MODEL") or self.model or "gpt-5.4"
        return config

    def _client(self, *, api_key: str, base_url: str | None):
        if self.client_factory is not None:
            return self.client_factory(api_key=api_key, base_url=base_url)
        from openai import OpenAI

        if base_url:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=20.0,
                max_retries=0,
            )
        return OpenAI(
            api_key=api_key,
            timeout=20.0,
            max_retries=0,
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        if not text:
            raise ValueError("The OpenAI response did not contain text output.")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("The OpenAI response did not contain a JSON object.")
        return json.loads(text[start : end + 1])

    def _load_env_file(self, path: str) -> dict[str, str]:
        return load_openai_env_file(path)

    def _strip_quotes(self, value: str) -> str:
        return strip_quotes(value)

    def _openai_sdk_available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True
