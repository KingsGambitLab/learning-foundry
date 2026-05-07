from __future__ import annotations

import re
import threading
from collections.abc import Callable

from app.domain.course import (
    CourseGenerationSource,
    CourseGenerationStatus,
    CreateCourseModuleRequest,
    CreateCourseRunRequest,
    GenerateCourseFromBriefRequest,
    GenerateCourseFromBriefResponse,
    GeneratedCoursePlan,
    QueueCourseGenerationResponse,
    SuggestLearningOutcomesRequest,
    SuggestLearningOutcomesResponse,
)
from app.domain.registry import PackageType, RiskClass
from app.domain.task_agent import (
    AssignmentDesignSpec,
    ProgressionMode,
    RetrievalMode,
    WorkspaceScope,
)
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.course_workflow_service import CourseWorkflowService
from app.services.openai_course_planner import (
    OpenAICourseGenerationError,
    OpenAICoursePlanner,
    OpenAICoursePlannerUnavailable,
)


class CourseGenerationService:
    def __init__(
        self,
        course_workflow_service: CourseWorkflowService,
        *,
        live_planner: OpenAICoursePlanner | None = None,
        job_runner: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.course_workflow_service = course_workflow_service
        self.live_planner = live_planner or OpenAICoursePlanner()
        self.job_runner = job_runner or self._run_job_in_background

    def status(self) -> CourseGenerationStatus:
        return self.live_planner.status()

    def queue_course_run_generation(
        self,
        request: GenerateCourseFromBriefRequest,
    ) -> QueueCourseGenerationResponse:
        planner_status = self.live_planner.status()
        course_run = self.course_workflow_service.create_generation_placeholder(
            title=request.title or self._title_from_goal(request.goal),
            goal=request.goal,
            learning_outcomes=request.learning_outcomes,
            package_type_hint=request.package_type_hint,
            generation_status=planner_status,
        )
        self.course_workflow_service.store.append_course_event(
            course_run.id,
            "course_generation_started",
            {
                "provider": planner_status.provider,
                "source": planner_status.source.value,
                "message": planner_status.message,
                "model_id": planner_status.model_id,
            },
        )
        self.job_runner(lambda: self._finish_queued_course_generation(course_run.id, request))
        latest = self.course_workflow_service.get_run(course_run.id) or course_run
        return QueueCourseGenerationResponse(
            queued=True,
            status=planner_status,
            course_run=latest,
        )

    def suggest_learning_outcomes(
        self,
        request: SuggestLearningOutcomesRequest,
    ) -> SuggestLearningOutcomesResponse:
        source = CourseGenerationSource.deterministic_fallback
        status = self.live_planner.status()

        if status.available:
            try:
                outcomes, status = self.live_planner.suggest_learning_outcomes(request)
                source = CourseGenerationSource.openai_live
                return SuggestLearningOutcomesResponse(
                    source=source,
                    status=status,
                    learning_outcomes=outcomes,
                )
            except (OpenAICourseGenerationError, OpenAICoursePlannerUnavailable) as exc:
                status = CourseGenerationStatus(
                    provider="openai",
                    available=False,
                    source=CourseGenerationSource.deterministic_fallback,
                    message=f"Live outcome suggestions failed and fell back to deterministic suggestions: {exc}",
                    sdk_installed=status.sdk_installed,
                    api_key_present=status.api_key_present,
                    model_id=status.model_id,
                    env_file=status.env_file,
                )

        return SuggestLearningOutcomesResponse(
            source=source,
            status=status,
            learning_outcomes=self._fallback_learning_outcomes(request.goal),
        )

    def generate_course_run(self, request: GenerateCourseFromBriefRequest) -> GenerateCourseFromBriefResponse:
        normalized_plan, source, status = self._generate_normalized_plan(request)
        course_run = self.course_workflow_service.create_run(
            CreateCourseRunRequest(
                title=normalized_plan.title,
                summary=normalized_plan.summary,
                package_type=normalized_plan.package_type,
                shared_design_spec=normalized_plan.shared_design_spec,
                modules=normalized_plan.modules,
            )
        )
        aligned_plan = self.course_workflow_service.generated_plan_from_run(
            course_run,
            notes=normalized_plan.notes,
        )
        course_run.goal = request.goal
        course_run.requested_learning_outcomes = list(request.learning_outcomes)
        course_run.generated_plan = aligned_plan
        course_run.generation_source = source
        course_run.generation_status = status
        course_run.notes.append(
            (
                f"Course brief generated via `{source.value}`."
                if source == CourseGenerationSource.openai_live
                else "Course brief generated via deterministic fallback planning."
            )
        )
        if status.model_id:
            course_run.notes.append(f"Planner model: `{status.model_id}`.")
        self.course_workflow_service.store.save_course_run(course_run)
        self.course_workflow_service.store.append_course_event(
            course_run.id,
            "course_brief_generated",
            {
                "source": source.value,
                "provider": status.provider,
                "model_id": status.model_id,
                "message": status.message,
            },
        )
        review = self.course_workflow_service.review_run(course_run.id)
        return GenerateCourseFromBriefResponse(
            source=source,
            status=status,
            plan=aligned_plan,
            course_run=course_run,
            review=review,
        )

    def _finish_queued_course_generation(
        self,
        course_run_id: str,
        request: GenerateCourseFromBriefRequest,
    ) -> None:
        try:
            normalized_plan, source, status = self._generate_normalized_plan(request)
            self.course_workflow_service.apply_generated_plan(
                course_run_id,
                plan=normalized_plan,
                source=source,
                generation_status=status,
            )
        except Exception as exc:
            status = self.live_planner.status()
            self.course_workflow_service.mark_generation_failed(
                course_run_id,
                error=str(exc),
                generation_status=status,
            )

    def _generate_normalized_plan(
        self,
        request: GenerateCourseFromBriefRequest,
    ) -> tuple[GeneratedCoursePlan, CourseGenerationSource, CourseGenerationStatus]:
        source = CourseGenerationSource.deterministic_fallback
        status = self.live_planner.status()
        plan: GeneratedCoursePlan

        if status.available:
            try:
                plan, status = self.live_planner.plan_course(request)
                source = CourseGenerationSource.openai_live
            except (OpenAICourseGenerationError, OpenAICoursePlannerUnavailable) as exc:
                status = CourseGenerationStatus(
                    provider="openai",
                    available=False,
                    source=CourseGenerationSource.deterministic_fallback,
                    message=f"Live generation failed and fell back to deterministic planning: {exc}",
                    sdk_installed=status.sdk_installed,
                    api_key_present=status.api_key_present,
                    model_id=status.model_id,
                    env_file=status.env_file,
                )
                plan = self._fallback_plan(request)
        else:
            plan = self._fallback_plan(request)

        return self._normalize_plan(plan, request), source, status

    def _run_job_in_background(self, job: Callable[[], None]) -> None:
        thread = threading.Thread(target=job, daemon=True)
        thread.start()

    def _normalize_plan(
        self,
        plan: GeneratedCoursePlan,
        request: GenerateCourseFromBriefRequest,
    ) -> GeneratedCoursePlan:
        normalized = plan.model_copy(deep=True)
        intake = GenerationIntake(
            title=request.title or plan.title or self._title_from_goal(request.goal),
            problem_statement=request.goal,
            learning_outcomes=request.learning_outcomes,
            package_type_hint=request.package_type_hint or plan.package_type,
        )
        inferred = infer_assignment_design(
            title=intake.title,
            problem_statement=intake.problem_statement,
            learning_outcomes=intake.learning_outcomes,
            package_type_hint=intake.package_type_hint,
        )

        if request.package_type_hint is not None:
            normalized.package_type = request.package_type_hint
        if not normalized.title:
            normalized.title = intake.title
        if not normalized.summary:
            normalized.summary = request.goal

        shared_design_spec = normalized.shared_design_spec or inferred.design_spec
        if shared_design_spec is None:
            raise ValueError("This brief is outside the current learner-ready generation scope.")
        shared_design_spec = self._with_package_type(shared_design_spec, normalized.package_type)
        normalized.shared_design_spec = shared_design_spec

        modules: list[CreateCourseModuleRequest] = []
        for module in normalized.modules:
            design_spec = module.design_spec or shared_design_spec
            design_spec = self._with_package_type(design_spec, normalized.package_type)
            learning_outcomes = module.learning_outcomes or request.learning_outcomes[:3]
            modules.append(
                CreateCourseModuleRequest(
                    module_slug=module.module_slug,
                    title=module.title.strip(),
                    summary=(module.summary or module.title).strip(),
                    learning_outcomes=learning_outcomes[:3],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                )
            )

        if not modules:
            modules = self._fallback_modules(request, shared_design_spec, normalized.package_type)

        if normalized.package_type == PackageType.progressive_codebase_course:
            modules = [
                module.model_copy(update={"design_spec": shared_design_spec})
                for module in modules
            ]

        normalized.modules = modules
        normalized.notes = list(dict.fromkeys(normalized.notes))
        return normalized

    def _fallback_plan(self, request: GenerateCourseFromBriefRequest) -> GeneratedCoursePlan:
        title = request.title or self._title_from_goal(request.goal)
        inferred = infer_assignment_design(
            title=title,
            problem_statement=request.goal,
            learning_outcomes=request.learning_outcomes,
            package_type_hint=request.package_type_hint,
        )
        if inferred.design_spec is None:
            raise ValueError("This brief is outside the current learner-ready generation scope.")

        package_type = self._preferred_package_type(request, inferred.package_type)
        design_spec = self._with_package_type(inferred.design_spec, package_type)

        return GeneratedCoursePlan(
            title=title,
            summary=request.goal.strip(),
            package_type=package_type,
            shared_design_spec=design_spec,
            modules=self._fallback_modules(request, design_spec, package_type),
            notes=[
                "Built from deterministic fallback planning.",
                "The course structure was inferred from the explicit assignment design because live OpenAI planning was unavailable.",
            ],
        )

    def _fallback_modules(
        self,
        request: GenerateCourseFromBriefRequest,
        design_spec: AssignmentDesignSpec,
        package_type: PackageType,
    ) -> list[CreateCourseModuleRequest]:
        outcomes = request.learning_outcomes
        if design_spec.capabilities.retrieval_mode == RetrievalMode.grounded_answers:
            return [
                CreateCourseModuleRequest(
                    title="Corpus ingestion and chunking",
                    summary="Stand up the retrieval substrate and make the corpus queryable.",
                    learning_outcomes=outcomes[:2] or ["chunking strategy", "retrieval setup"],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Grounded retrieval and citations",
                    summary="Return answers that stay anchored to the corpus and cite supporting evidence.",
                    learning_outcomes=outcomes[:3] or ["citation correctness", "faithfulness", "abstention"],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Quality tuning and evals",
                    summary="Improve answer quality with decomposition, reranking, and eval-driven iteration.",
                    learning_outcomes=outcomes[:3] or ["query decomposition", "reranking", "evaluation"],
                    design_spec=design_spec,
                    overlays_hint=["productionization_overlay"],
                ),
                CreateCourseModuleRequest(
                    title="Scale, freshness, and final SLO",
                    summary="Push the system to production bars for latency, freshness, and operating cost.",
                    learning_outcomes=outcomes[:3] or ["latency tuning", "freshness", "cost control"],
                    design_spec=design_spec,
                    overlays_hint=["scale_slo_overlay", "freshness_overlay"],
                ),
            ]

        if design_spec.capabilities.retrieval_mode == RetrievalMode.ranked_results:
            return [
                CreateCourseModuleRequest(
                    title="Index design and retrieval contract",
                    summary="Build the corpus, query interface, and ranking baseline.",
                    learning_outcomes=outcomes[:2] or ["index design", "retrieval contract"],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Ranking quality and filtering",
                    summary="Improve retrieval precision, ordering, and metadata-aware filters.",
                    learning_outcomes=outcomes[:3] or ["ranking quality", "filtering", "query analysis"],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Production retrieval final",
                    summary="Meet quality and latency expectations for a production retrieval service.",
                    learning_outcomes=outcomes[:3] or ["latency tuning", "quality checks", "operational readiness"],
                    design_spec=design_spec,
                    overlays_hint=["scale_slo_overlay"],
                ),
            ]

        if design_spec.capabilities.durable_state_required and not design_spec.capabilities.tool_use_required:
            return [
                CreateCourseModuleRequest(
                    title="Contract and data model",
                    summary="Define the service surface, persistence model, and baseline invariants.",
                    learning_outcomes=outcomes[:2] or ["contract design", "data modeling"],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Correctness under concurrency",
                    summary="Make the state transitions safe under duplicate requests and parallel access.",
                    learning_outcomes=outcomes[:3] or ["idempotency", "concurrency safety", "error handling"],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Throughput and production final",
                    summary="Harden the service for real traffic, latency, and failure handling.",
                    learning_outcomes=outcomes[:3] or ["throughput", "latency", "operational readiness"],
                    design_spec=design_spec,
                    overlays_hint=["scale_slo_overlay"],
                ),
            ]

        modules = [
            CreateCourseModuleRequest(
                title="Run contract and structured output",
                summary="Get the service onto a stable run contract with a reliable output schema.",
                learning_outcomes=outcomes[:2] or ["structured output", "run contract"],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=list(design_spec.overlays),
            ),
            CreateCourseModuleRequest(
                title="Tooling and control flow",
                summary="Teach the system how to choose tools and execute bounded workflows.",
                learning_outcomes=outcomes[:3] or ["tool selection", "multi-step execution", "state handling"],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=list(design_spec.overlays),
            ),
            CreateCourseModuleRequest(
                title="Approvals, fallbacks, and observability",
                summary="Add safety controls, error recovery, and traces that make the system operable.",
                learning_outcomes=outcomes[:3] or ["approval gates", "fallback handling", "observability"],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=["productionization_overlay"],
            ),
        ]
        if package_type == PackageType.progressive_codebase_course:
            modules.extend(
                [
                    CreateCourseModuleRequest(
                        title="Eval-driven quality improvements",
                        summary="Use evals to improve reliability, escalation quality, and output usefulness.",
                        learning_outcomes=outcomes[:3] or ["evaluation", "quality tuning", "regression control"],
                        design_spec=design_spec,
                        domain_pack_hint=design_spec.domain_pack,
                        overlays_hint=["productionization_overlay"],
                    ),
                    CreateCourseModuleRequest(
                        title="Production final at SLO",
                        summary="Push the system to its final reliability, latency, and cost bar.",
                        learning_outcomes=outcomes[:3] or ["latency", "cost control", "production readiness"],
                        design_spec=design_spec,
                        domain_pack_hint=design_spec.domain_pack,
                        overlays_hint=["productionization_overlay", "scale_slo_overlay"],
                    ),
                ]
            )
        return modules

    def _fallback_learning_outcomes(self, goal: str) -> list[str]:
        design_spec = infer_assignment_design(
            title=self._title_from_goal(goal),
            problem_statement=goal,
            learning_outcomes=[],
        ).design_spec

        if design_spec is None:
            return [
                "Define the system contract and the core behaviors the learner must make reliable.",
                "Implement the primary workflow end to end with concrete success and failure handling.",
                "Add observability or evaluation checks that make quality visible during development.",
                "Raise the system to a production-minded bar for correctness, safety, or reliability.",
            ]

        if design_spec.capabilities.retrieval_mode == RetrievalMode.grounded_answers:
            return [
                "Build a retrieval pipeline that returns grounded answers with citations.",
                "Measure retrieval quality and reduce unsupported or hallucinated responses.",
                "Tune latency and cost so the system can run at a practical production bar.",
                "Add evaluation loops that surface failure cases and track quality over time.",
            ]

        if design_spec.capabilities.retrieval_mode == RetrievalMode.ranked_results:
            return [
                "Design an index and query flow that retrieves relevant results reliably.",
                "Measure ranking quality with concrete retrieval metrics and fixtures.",
                "Handle filters, freshness, and edge cases in the retrieval contract.",
                "Tune the system for practical latency under realistic query load.",
            ]

        if design_spec.capabilities.durable_state_required and not design_spec.capabilities.tool_use_required:
            return [
                "Model the core state transitions and the invariants the service must preserve.",
                "Make duplicate requests and concurrent access safe to handle in production.",
                "Capture traces or audit records that make stateful failures easier to debug.",
                "Raise the system to a production-minded bar for latency, reliability, and correctness.",
            ]

        return [
            "Define a bounded service contract with clear inputs, outputs, and failure handling.",
            "Implement the primary workflow end to end with observable, testable behavior.",
            "Add evaluation checks that make quality visible during development.",
            "Raise the system to a production-minded bar for correctness, safety, or reliability.",
        ]

    def _title_from_goal(self, goal: str) -> str:
        text = goal.strip()
        text = re.sub(r"^(build|create|design|make)\s+", "", text, flags=re.IGNORECASE)
        words = text.split()
        if not words:
            return "Generated Course Draft"
        return " ".join(word.capitalize() for word in words[:6])

    def _preferred_package_type(
        self,
        request: GenerateCourseFromBriefRequest,
        recommended: PackageType | None,
    ) -> PackageType:
        if request.package_type_hint is not None:
            return request.package_type_hint

        brief = " ".join([request.goal, *request.learning_outcomes]).lower()
        survey_markers = [
            "survey",
            "compare",
            "multiple systems",
            "across different systems",
            "variety of systems",
            "several systems",
        ]
        if any(marker in brief for marker in survey_markers):
            return PackageType.survey_course
        return recommended or PackageType.progressive_codebase_course

    def _with_package_type(
        self,
        design_spec: AssignmentDesignSpec,
        package_type: PackageType,
    ) -> AssignmentDesignSpec:
        shared_codebase = package_type == PackageType.progressive_codebase_course
        return design_spec.model_copy(
            update={
                "course_structure": design_spec.course_structure.model_copy(
                    update={
                        "package_type": package_type,
                        "workspace_scope": (
                            WorkspaceScope.shared_course_workspace
                            if shared_codebase
                            else WorkspaceScope.per_module_workspace
                        ),
                        "progression_mode": (
                            ProgressionMode.cumulative_module_gates
                            if shared_codebase
                            else ProgressionMode.independent_modules
                        ),
                        "shared_codebase": shared_codebase,
                    }
                ),
                "assessment_strategy": design_spec.assessment_strategy.model_copy(
                    update={"cumulative_module_gates": shared_codebase}
                ),
            }
        )
