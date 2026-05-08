from __future__ import annotations

import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from app.domain.ai import AIUsageSummary
from app.domain.course import (
    CreateCourseFromCreatorPlanRequest,
    CourseGenerationSource,
    CourseGenerationStatus,
    CourseRun,
    CreatorCourseSetupChoices,
    CreatorCourseSetupInput,
    CreatorCourseModulePlan,
    CreatorCoursePlan,
    CreateCourseModuleRequest,
    CreateCourseRunRequest,
    GenerateCreatorCoursePlanRequest,
    GenerateCreatorCoursePlanResponse,
    GenerateCourseFromBriefRequest,
    GenerateCourseFromBriefResponse,
    GeneratedCoursePlan,
    QueueCourseGenerationResponse,
    SuggestLearningOutcomesRequest,
    SuggestLearningOutcomesResponse,
)
from app.domain.registry import PackageType, RiskClass, StarterType
from app.domain.task_agent import (
    AssignmentDesignSpec,
    DataSourceKind,
    DataSourcePurpose,
    DataSourceSpec,
    ProgressionMode,
    RetrievalMode,
    WorkspaceScope,
)
from app.domain.workflow import DecisionOutcome, GateDecisionRequest
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
        resolved_setup = self._resolve_creator_setup(request.goal, request.creator_setup)
        course_run = self.course_workflow_service.create_generation_placeholder(
            title=request.title or self._title_from_goal(request.goal),
            goal=request.goal,
            learning_outcomes=[],
            package_type_hint=request.package_type_hint,
            creator_choices=resolved_setup,
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
                outcomes, status, _usage = self.live_planner.suggest_learning_outcomes(request)
                source = CourseGenerationSource.openai_live
                return SuggestLearningOutcomesResponse(
                    source=source,
                    status=status,
                    learning_outcomes=self._normalize_learning_outcomes(outcomes),
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
            learning_outcomes=self._normalize_learning_outcomes(self._fallback_learning_outcomes(request.goal)),
        )

    def generate_creator_plan(
        self,
        request: GenerateCreatorCoursePlanRequest,
    ) -> GenerateCreatorCoursePlanResponse:
        resolved_setup = self._resolve_creator_setup(request.goal, request.creator_choices)
        plan_request = GenerateCourseFromBriefRequest(
            goal=request.goal,
            title=request.title,
            package_type_hint=request.package_type_hint,
            creator_setup=CreatorCourseSetupInput(**resolved_setup.model_dump(mode="json")),
        )
        normalized_plan, source, status, _usage = self._generate_normalized_plan(plan_request)
        adjusted_shared_design_spec = self._apply_creator_choices_to_design_spec(
            normalized_plan.shared_design_spec,
            resolved_setup,
        )
        creator_modules = self._creator_plan_modules(
            request=plan_request,
            design_spec=adjusted_shared_design_spec,
            default_modules=normalized_plan.modules,
            creator_summary=normalized_plan.summary,
            creator_choices=resolved_setup,
        )
        normalized_outcomes = self._derive_plan_learning_outcomes(
            creator_modules,
            adjusted_shared_design_spec,
        )
        creator_plan = CreatorCoursePlan(
            goal=request.goal,
            learning_outcomes=normalized_outcomes,
            title=normalized_plan.title,
            summary=normalized_plan.summary,
            package_type=normalized_plan.package_type,
            creator_choices=resolved_setup,
            shared_design_spec=adjusted_shared_design_spec,
            modules=creator_modules,
            creator_summary=self._creator_summary(adjusted_shared_design_spec, resolved_setup),
            notes=list(
                dict.fromkeys(
                    [
                        *normalized_plan.notes,
                        "Review the module ladder before creating the draft.",
                        "The approved creator plan feeds the same course-generation and review pipeline.",
                    ]
                )
            ),
        )
        return GenerateCreatorCoursePlanResponse(
            source=source,
            status=status,
            learning_outcomes=normalized_outcomes,
            plan=creator_plan,
        )

    def create_course_run_from_creator_plan(
        self,
        request: CreateCourseFromCreatorPlanRequest,
    ) -> CourseRun:
        plan = request.plan
        generated_plan = self._generated_plan_from_creator_plan(plan)
        course_run = self.course_workflow_service.create_run(
            CreateCourseRunRequest(
                title=generated_plan.title,
                summary=generated_plan.summary,
                package_type=generated_plan.package_type,
                shared_design_spec=generated_plan.shared_design_spec,
                modules=generated_plan.modules,
            )
        )
        if len(course_run.modules) == len(plan.modules):
            for stored_module, planned_module in zip(course_run.modules, plan.modules, strict=False):
                stored_module.title = planned_module.title
                stored_module.summary = planned_module.summary
                stored_module.learning_outcomes = list(planned_module.learning_outcomes)
                stored_module.notes = list(
                    dict.fromkeys(
                        [
                            *stored_module.notes,
                            *planned_module.creator_notes,
                        ]
                    )
                )
        course_run.notes = list(
            dict.fromkeys(
                [
                    *course_run.notes,
                    "Draft created from an approved creator plan.",
                    f"Starter preference: `{plan.creator_choices.starter_type.value}`.",
                    *( [f"Primary database: `{plan.creator_choices.primary_database}`."] if plan.creator_choices.primary_database else [] ),
                    *( [f"Cache backend: `{plan.creator_choices.cache_backend}`."] if plan.creator_choices.cache_backend else [] ),
                    *(
                        [
                            "Attached data sources: "
                            + ", ".join(f"`{source.title}`" for source in plan.creator_choices.data_sources[:3])
                            + "."
                        ]
                        if plan.creator_choices.data_sources
                        else []
                    ),
                ]
            )
        )
        course_run.goal = plan.goal
        course_run.requested_learning_outcomes = list(plan.learning_outcomes)
        course_run.generated_plan = generated_plan
        self.course_workflow_service.store.save_course_run(course_run)
        self.course_workflow_service.store.append_course_event(
            course_run.id,
            "creator_plan_accepted",
            {
                "module_count": len(plan.modules),
                "goal": plan.goal,
                "learning_outcome_count": len(plan.learning_outcomes),
                "starter_type": plan.creator_choices.starter_type.value,
                "primary_database": plan.creator_choices.primary_database,
                "cache_backend": plan.creator_choices.cache_backend,
                "data_source_count": len(plan.creator_choices.data_sources),
            },
        )
        return course_run

    def queue_course_run_from_creator_plan(
        self,
        request: CreateCourseFromCreatorPlanRequest,
    ) -> QueueCourseGenerationResponse:
        plan = request.plan
        planner_status = self.live_planner.status()
        course_run = self.course_workflow_service.create_generation_placeholder(
            title=plan.title,
            goal=plan.goal or plan.summary,
            learning_outcomes=plan.learning_outcomes,
            package_type_hint=plan.package_type,
            creator_choices=plan.creator_choices,
            generation_status=planner_status,
        )
        course_run.summary = plan.summary
        course_run.goal = plan.goal
        course_run.requested_learning_outcomes = list(plan.learning_outcomes)
        course_run.generated_plan = self._generated_plan_from_creator_plan(plan)
        course_run.notes = list(
            dict.fromkeys(
                [
                    *course_run.notes,
                    "Creator-approved module plan queued.",
                    f"Starter preference: `{plan.creator_choices.starter_type.value}`.",
                    *([f"Primary database: `{plan.creator_choices.primary_database}`."] if plan.creator_choices.primary_database else []),
                    *([f"Cache backend: `{plan.creator_choices.cache_backend}`."] if plan.creator_choices.cache_backend else []),
                    *(
                        [
                            "Attached data sources: "
                            + ", ".join(f"`{source.title}`" for source in plan.creator_choices.data_sources[:3])
                            + "."
                        ]
                        if plan.creator_choices.data_sources
                        else []
                    ),
                ]
            )
        )
        self.course_workflow_service.store.save_course_run(course_run)
        self.course_workflow_service.store.append_course_event(
            course_run.id,
            "creator_plan_accepted",
            {
                "module_count": len(plan.modules),
                "goal": plan.goal,
                "learning_outcome_count": len(plan.learning_outcomes),
                "starter_type": plan.creator_choices.starter_type.value,
                "primary_database": plan.creator_choices.primary_database,
                "cache_backend": plan.creator_choices.cache_backend,
                "data_source_count": len(plan.creator_choices.data_sources),
                "message": "Approved module plan accepted. Building the draft in the background.",
            },
        )
        self.course_workflow_service.store.append_course_event(
            course_run.id,
            "course_generation_started",
            {
                "provider": planner_status.provider,
                "source": planner_status.source.value,
                "message": "Building the draft from your approved creator plan.",
                "model_id": planner_status.model_id,
            },
        )
        self.job_runner(lambda: self._finish_queued_creator_plan(course_run.id, request))
        latest = self.course_workflow_service.get_run(course_run.id) or course_run
        return QueueCourseGenerationResponse(
            queued=True,
            status=planner_status,
            course_run=latest,
        )

    def generate_course_run(self, request: GenerateCourseFromBriefRequest) -> GenerateCourseFromBriefResponse:
        normalized_plan, source, status, usage = self._generate_normalized_plan(request)
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
        course_run.requested_learning_outcomes = self._derive_plan_learning_outcomes(
            aligned_plan.modules,
            aligned_plan.shared_design_spec,
        )
        course_run.generated_plan = aligned_plan
        course_run.generation_source = source
        course_run.generation_status = status
        course_run.own_ai_usage = usage or AIUsageSummary()
        course_run.ai_usage = course_run.own_ai_usage
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
                "ai_usage": (usage.model_dump(mode="json") if usage is not None else None),
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
            normalized_plan, source, status, usage = self._generate_normalized_plan(request)
            built = self.course_workflow_service.apply_generated_plan(
                course_run_id,
                plan=normalized_plan,
                source=source,
                generation_status=status,
                usage=usage,
                execute_shared_workflow_nodes=False,
                clear_active_operation=False,
            )
            self._finalize_background_generation(
                built.id,
                generation_status=status,
                completion_message="Draft finished building from the generated course brief.",
            )
        except Exception as exc:
            status = self.live_planner.status()
            self.course_workflow_service.mark_generation_failed(
                course_run_id,
                error=str(exc),
                generation_status=status,
            )

    def _finish_queued_creator_plan(
        self,
        course_run_id: str,
        request: CreateCourseFromCreatorPlanRequest,
    ) -> None:
        try:
            plan = request.plan
            built = self.course_workflow_service.apply_generated_plan(
                course_run_id,
                plan=self._generated_plan_from_creator_plan(plan),
                source=CourseGenerationSource.deterministic_fallback,
                generation_status=self.live_planner.status(),
                execute_shared_workflow_nodes=False,
                clear_active_operation=False,
            )
            if len(built.modules) == len(plan.modules):
                for stored_module, planned_module in zip(built.modules, plan.modules, strict=False):
                    stored_module.title = planned_module.title
                    stored_module.summary = planned_module.summary
                    stored_module.learning_outcomes = list(planned_module.learning_outcomes)
                    stored_module.notes = list(
                        dict.fromkeys(
                            [
                                *stored_module.notes,
                                *planned_module.creator_notes,
                            ]
                        )
                    )
            built.summary = plan.summary
            built.goal = plan.goal
            built.requested_learning_outcomes = list(plan.learning_outcomes)
            built.generated_plan = self._generated_plan_from_creator_plan(plan)
            built.notes = list(
                dict.fromkeys(
                    [
                        *built.notes,
                        "Draft created from an approved creator plan.",
                        f"Starter preference: `{plan.creator_choices.starter_type.value}`.",
                        *([f"Primary database: `{plan.creator_choices.primary_database}`."] if plan.creator_choices.primary_database else []),
                        *([f"Cache backend: `{plan.creator_choices.cache_backend}`."] if plan.creator_choices.cache_backend else []),
                    ]
                )
            )
            self.course_workflow_service.store.save_course_run(built)
            self.course_workflow_service.store.append_course_event(
                built.id,
                "creator_plan_applied",
                {
                    "module_count": len(plan.modules),
                    "message": "Draft shell created from the approved creator plan. Review checks are still running.",
                },
            )
            self._finalize_background_generation(
                built.id,
                generation_status=self.live_planner.status(),
                completion_message="Draft finished building from the approved creator plan.",
            )
        except Exception as exc:
            self.course_workflow_service.mark_generation_failed(
                course_run_id,
                error=str(exc),
                generation_status=self.live_planner.status(),
            )

    def _finalize_background_generation(
        self,
        course_run_id: str,
        *,
        generation_status: CourseGenerationStatus,
        completion_message: str,
    ) -> None:
        course_run = self.course_workflow_service.get_run(course_run_id)
        if course_run is None:
            raise KeyError(course_run_id)
        if course_run.shared_workflow_run_id is not None:
            workflow_run = self.course_workflow_service.workflow_service.execute_langgraph_nodes(
                course_run.shared_workflow_run_id
            )
            while workflow_run.pending_gate is not None:
                workflow_run = self.course_workflow_service.workflow_service.apply_gate_decision(
                    workflow_run.id,
                    GateDecisionRequest(
                        gate=workflow_run.pending_gate,
                        decision=DecisionOutcome.approve,
                        comment="Auto-approved by the default creator-flow setting.",
                    ),
                )
            course_run = self.course_workflow_service.sync_run(course_run.id)
        course_run.active_operation = None
        course_run.updated_at = datetime.now(UTC)
        course_run.generation_status = generation_status
        course_run.last_error = None
        self.course_workflow_service.store.save_course_run(course_run)
        self.course_workflow_service.store.append_course_event(
            course_run.id,
            "course_generation_completed",
            {
                "message": completion_message,
                "deliverable_count": len(course_run.modules),
                "shared_workflow_run_id": course_run.shared_workflow_run_id,
            },
        )

    def _generate_normalized_plan(
        self,
        request: GenerateCourseFromBriefRequest,
    ) -> tuple[GeneratedCoursePlan, CourseGenerationSource, CourseGenerationStatus]:
        source = CourseGenerationSource.deterministic_fallback
        status = self.live_planner.status()
        plan: GeneratedCoursePlan
        usage: AIUsageSummary | None = None

        if status.available:
            try:
                plan, status, usage = self.live_planner.plan_course(request)
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

        return self._normalize_plan(plan, request), source, status, usage

    def _run_job_in_background(self, job: Callable[[], None]) -> None:
        thread = threading.Thread(target=job, daemon=True)
        thread.start()

    def _generated_plan_from_creator_plan(self, plan: CreatorCoursePlan) -> GeneratedCoursePlan:
        shared_design_spec = self._apply_creator_choices_to_design_spec(
            plan.shared_design_spec,
            plan.creator_choices,
        )
        modules = [
            CreateCourseModuleRequest(
                module_slug=module.module_slug,
                title=module.title,
                summary=module.summary,
                learning_outcomes=module.learning_outcomes,
                design_spec=self._apply_creator_choices_to_design_spec(module.design_spec or shared_design_spec, plan.creator_choices),
            )
            for module in plan.modules
        ]
        return GeneratedCoursePlan(
            title=plan.title,
            summary=plan.summary,
            package_type=plan.package_type,
            shared_design_spec=shared_design_spec,
            modules=modules,
            notes=list(
                dict.fromkeys(
                    [
                        *plan.notes,
                        "Creator-approved module plan.",
                    ]
                )
            ),
        )

    def _normalize_plan(
        self,
        plan: GeneratedCoursePlan,
        request: GenerateCourseFromBriefRequest,
    ) -> GeneratedCoursePlan:
        normalized = plan.model_copy(deep=True)
        creator_choices = self._resolve_creator_setup(request.goal, request.creator_setup)
        intake = GenerationIntake(
            title=request.title or plan.title or self._title_from_goal(request.goal),
            problem_statement=request.goal,
            package_type_hint=request.package_type_hint or plan.package_type,
            starter_type=creator_choices.starter_type,
            primary_database=creator_choices.primary_database,
            cache_backend=creator_choices.cache_backend,
            tech_stack=list(creator_choices.tech_stack),
            data_sources=list(creator_choices.data_sources),
        )
        inferred = infer_assignment_design(
            title=intake.title,
            problem_statement=intake.problem_statement,
            package_type_hint=intake.package_type_hint,
            starter_type=intake.starter_type,
            primary_database=intake.primary_database,
            cache_backend=intake.cache_backend,
            tech_stack=intake.tech_stack,
            data_sources=intake.data_sources,
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
        shared_design_spec = self._apply_creator_choices_to_design_spec(shared_design_spec, creator_choices)
        normalized.shared_design_spec = shared_design_spec

        modules: list[CreateCourseModuleRequest] = []
        for module in normalized.modules:
            design_spec = module.design_spec or shared_design_spec
            design_spec = self._with_package_type(design_spec, normalized.package_type)
            design_spec = self._apply_creator_choices_to_design_spec(design_spec, creator_choices)
            learning_outcomes = self._normalize_learning_outcomes(
                module.learning_outcomes or self._derive_module_learning_outcomes(module.title, module.summary or module.title, design_spec)
            )
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
        creator_choices = self._resolve_creator_setup(request.goal, request.creator_setup)
        inferred = infer_assignment_design(
            title=title,
            problem_statement=request.goal,
            package_type_hint=request.package_type_hint,
            starter_type=creator_choices.starter_type,
            primary_database=creator_choices.primary_database,
            cache_backend=creator_choices.cache_backend,
            tech_stack=list(creator_choices.tech_stack),
            data_sources=list(creator_choices.data_sources),
        )
        if inferred.design_spec is None:
            raise ValueError("This brief is outside the current learner-ready generation scope.")

        package_type = self._preferred_package_type(request, inferred.package_type)
        design_spec = self._with_package_type(inferred.design_spec, package_type)
        design_spec = self._apply_creator_choices_to_design_spec(design_spec, creator_choices)

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
        if design_spec.capabilities.retrieval_mode == RetrievalMode.grounded_answers:
            return [
                CreateCourseModuleRequest(
                    title="Corpus ingestion and chunking",
                    summary="Stand up the retrieval substrate and make the corpus queryable.",
                    learning_outcomes=[
                        "Stand up the retrieval substrate so learner-visible documents can be queried reliably.",
                        "Shape the corpus and retrieval contract before answer synthesis begins.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Grounded retrieval and citations",
                    summary="Return answers that stay anchored to the corpus and cite supporting evidence.",
                    learning_outcomes=[
                        "Return answers that stay grounded in retrieved evidence instead of guessing.",
                        "Use citations and abstention to make evidence coverage visible to the learner.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Quality tuning and evals",
                    summary="Improve answer quality with decomposition, reranking, and eval-driven iteration.",
                    learning_outcomes=[
                        "Use visible evaluation cases to improve grounded answer quality deliberately.",
                        "Refine retrieval and answer composition without regressing supported scenarios.",
                    ],
                    design_spec=design_spec,
                    overlays_hint=["productionization_overlay"],
                ),
                CreateCourseModuleRequest(
                    title="Scale, freshness, and final SLO",
                    summary="Push the system to production bars for latency, freshness, and operating cost.",
                    learning_outcomes=[
                        "Tune the service for production-minded latency and operating cost.",
                        "Keep retrieved evidence fresh enough for realistic deployment expectations.",
                    ],
                    design_spec=design_spec,
                    overlays_hint=["scale_slo_overlay", "freshness_overlay"],
                ),
            ]

        if design_spec.capabilities.retrieval_mode == RetrievalMode.ranked_results:
            return [
                CreateCourseModuleRequest(
                    title="Index design and retrieval contract",
                    summary="Build the corpus, query interface, and ranking baseline.",
                    learning_outcomes=[
                        "Design the retrieval contract and corpus shape the service will expose.",
                        "Build a ranking baseline over learner-visible documents and filters.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Ranking quality and filtering",
                    summary="Improve retrieval precision, ordering, and metadata-aware filters.",
                    learning_outcomes=[
                        "Improve retrieval precision with ranking and metadata-aware filters.",
                        "Handle query analysis decisions without destabilizing the retrieval contract.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Production retrieval final",
                    summary="Meet quality and latency expectations for a production retrieval service.",
                    learning_outcomes=[
                        "Raise retrieval quality and latency to a production-minded bar.",
                        "Use visible checks and final grading to prove the service is operationally credible.",
                    ],
                    design_spec=design_spec,
                    overlays_hint=["scale_slo_overlay"],
                ),
            ]

        if design_spec.capabilities.durable_state_required and not design_spec.capabilities.tool_use_required:
            return [
                CreateCourseModuleRequest(
                    title="Contract and data model",
                    summary="Define the service surface, persistence model, and baseline invariants.",
                    learning_outcomes=[
                        "Define the service contract and persistence model around the core invariants.",
                        "Model data so the workflow can stay correct as mutable state changes.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Correctness under concurrency",
                    summary="Make the state transitions safe under duplicate requests and parallel access.",
                    learning_outcomes=[
                        "Keep concurrent or repeated requests from breaking critical invariants.",
                        "Use retries, idempotency, or locking to make mutable state safe under load.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                ),
                CreateCourseModuleRequest(
                    title="Throughput and production final",
                    summary="Harden the service for real traffic, latency, and failure handling.",
                    learning_outcomes=[
                        "Raise the service to a production-minded bar for throughput, latency, and reliability.",
                        "Make stateful failures visible enough for an operator to debug them confidently.",
                    ],
                    design_spec=design_spec,
                    overlays_hint=["scale_slo_overlay"],
                ),
            ]

        modules = [
            CreateCourseModuleRequest(
                title="Run contract and structured output",
                summary="Get the service onto a stable run contract with a reliable output schema.",
                learning_outcomes=[
                    "Implement a stable run contract with the expected structured response shape.",
                    "Return reliable outputs for the visible scenarios before adding deeper controls.",
                ],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=list(design_spec.overlays),
            ),
            CreateCourseModuleRequest(
                title="Tooling and control flow",
                summary="Teach the system how to choose tools and execute bounded workflows.",
                learning_outcomes=[
                    "Choose the right tools and bounded control flow for the supported workflow paths.",
                    "Keep tool usage explicit enough that the learner can debug why each step happened.",
                ],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=list(design_spec.overlays),
            ),
            CreateCourseModuleRequest(
                title="Approvals, fallbacks, and observability",
                    summary="Add safety controls, error recovery, and traces that make the system operable.",
                    learning_outcomes=[
                        "Add approvals and fallback behavior without breaking the public contract.",
                        "Emit observability signals that make failures understandable to an operator.",
                    ],
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
                        learning_outcomes=[
                            "Use evaluation feedback to improve behavior without guessing what changed.",
                            "Tighten the system while protecting earlier supported scenarios from regression.",
                        ],
                        design_spec=design_spec,
                        domain_pack_hint=design_spec.domain_pack,
                        overlays_hint=["productionization_overlay"],
                    ),
                    CreateCourseModuleRequest(
                        title="Production final at SLO",
                        summary="Push the system to its final reliability, latency, and cost bar.",
                        learning_outcomes=[
                            "Integrate the full system until it meets a production-minded reliability and latency bar.",
                            "Balance quality, latency, and cost without regressing the learner-visible contract.",
                        ],
                        design_spec=design_spec,
                        domain_pack_hint=design_spec.domain_pack,
                        overlays_hint=["productionization_overlay", "scale_slo_overlay"],
                    ),
                ]
            )
        return modules

    def _creator_plan_modules(
        self,
        *,
        request: GenerateCourseFromBriefRequest,
        design_spec: AssignmentDesignSpec,
        default_modules: list[CreateCourseModuleRequest],
        creator_summary: str,
        creator_choices,
    ) -> list[CreatorCourseModulePlan]:
        text = " ".join([request.goal, creator_summary]).lower()
        modules: list[CreateCourseModuleRequest]
        if (
            design_spec.capabilities.durable_state_required
            and any(keyword in text for keyword in ["flight", "booking", "reservation", "inventory"])
        ):
            cache_label = creator_choices.cache_backend.title() if creator_choices.cache_backend else "Caching"
            cache_summary = (
                f"Introduce {creator_choices.cache_backend} caching for availability lookups and protect freshness."
                if creator_choices.cache_backend
                else "Improve read-path performance without breaking booking correctness."
            )
            db_label = creator_choices.primary_database or "the primary database"
            modules = [
                CreateCourseModuleRequest(
                    module_slug="exercise/01-core-booking-flow",
                    title="Core booking contract and seat inventory",
                    summary="Model the booking flow, inventory records, and baseline invariants before adding concurrency controls.",
                    learning_outcomes=[
                        "Define the booking workflow and the invariants that must never break.",
                        "Model seats, reservations, and booking state transitions clearly.",
                    ],
                    design_spec=design_spec,
                ),
                CreateCourseModuleRequest(
                    module_slug="exercise/02-pessimistic-locking",
                    title=f"Pessimistic locking in {db_label}",
                    summary="Prevent overselling by introducing pessimistic locking around the critical reservation path.",
                    learning_outcomes=[
                        "Use pessimistic locking to protect the hot booking path.",
                        "Explain the tradeoff between safety and throughput under contention.",
                    ],
                    design_spec=design_spec,
                ),
                CreateCourseModuleRequest(
                    module_slug="exercise/03-optimistic-locking",
                    title=f"Optimistic locking and retries in {db_label}",
                    summary="Shift the booking path to optimistic control with version checks, retries, and clear failure responses.",
                    learning_outcomes=[
                        "Implement optimistic concurrency control with version-aware writes.",
                        "Design retry and conflict handling that feels safe in production.",
                    ],
                    design_spec=design_spec,
                ),
                CreateCourseModuleRequest(
                    module_slug="exercise/04-caching",
                    title=f"{cache_label} for availability reads",
                    summary=cache_summary,
                    learning_outcomes=[
                        "Use caching to speed up read-heavy traffic without corrupting booking correctness.",
                        "Explain the freshness and invalidation tradeoffs in the booking workflow.",
                    ],
                    design_spec=design_spec,
                ),
                CreateCourseModuleRequest(
                    module_slug="final/production-readiness",
                    title="Production hardening and failure drills",
                    summary="Pull the booking service together with observability, retries, and realistic operational drills.",
                    learning_outcomes=[
                        "Make the service observable and debuggable under failure.",
                        "Prepare the system for production traffic and operator confidence.",
                    ],
                    design_spec=design_spec.model_copy(
                        update={"overlays": list(dict.fromkeys([*design_spec.overlays, "productionization_overlay"]))}
                    ),
                ),
            ]
        else:
            fallback_modules = self._fallback_modules(
                request,
                design_spec,
                design_spec.course_structure.package_type,
            )
            if fallback_modules and len(default_modules) > len(fallback_modules):
                modules = fallback_modules
            else:
                modules = default_modules

        return [
            CreatorCourseModulePlan(
                module_slug=module.module_slug or f"module-{index}",
                title=module.title,
                summary=module.summary or module.title,
                learning_outcomes=self._normalize_learning_outcomes(
                    module.learning_outcomes
                    or self._derive_module_learning_outcomes(
                        module.title,
                        module.summary or module.title,
                        module.design_spec or design_spec,
                    )
                ),
                creator_notes=self._creator_notes_for_module(module, creator_choices),
                design_spec=self._apply_creator_choices_to_design_spec(module.design_spec or design_spec, creator_choices),
            )
            for index, module in enumerate(modules, start=1)
        ]

    def _fallback_learning_outcomes(self, goal: str) -> list[str]:
        design_spec = infer_assignment_design(
            title=self._title_from_goal(goal),
            problem_statement=goal,
            learning_outcomes=[],
        ).design_spec

        if design_spec is None:
            return [
                "Turn the problem statement into a production-ready service contract with clear boundaries.",
                "Implement the core workflow end to end and make failure handling visible.",
                "Add debugging and quality signals so a teammate can trust the system under load.",
                "Push the system to a production bar for correctness, safety, or operational confidence.",
            ]

        if design_spec.capabilities.retrieval_mode == RetrievalMode.grounded_answers:
            return [
                "Build a retrieval flow that answers from evidence instead of guessing.",
                "Use citations and abstention to make groundedness visible to the learner.",
                "Tune latency and cost until the system feels believable in production.",
                "Add evaluation loops that catch regressions before they hit users.",
            ]

        if design_spec.capabilities.retrieval_mode == RetrievalMode.ranked_results:
            return [
                "Design a retrieval contract that returns relevant results consistently.",
                "Measure ranking quality with concrete fixtures instead of intuition.",
                "Handle filters, freshness, and read-path edge cases without surprises.",
                "Tune the service for practical latency under realistic query load.",
            ]

        if design_spec.capabilities.durable_state_required and not design_spec.capabilities.tool_use_required:
            lowered_goal = goal.lower()
            if any(keyword in lowered_goal for keyword in ["flight", "booking", "reservation", "inventory"]):
                return [
                    "Model bookings and seat inventory so the service preserves the right invariants.",
                    "Use locking and retries to keep concurrent reservations safe under load.",
                    "Add caching and observability without making availability stale or misleading.",
                    "Raise the service to a production bar for correctness, latency, and operator trust.",
                ]
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

    def _normalize_learning_outcomes(self, outcomes: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in outcomes:
            parts = re.split(r"(?:\r?\n|;)", raw)
            for part in parts:
                cleaned = part.strip()
                cleaned = re.sub(r"^[-*•\d\.\)\s]+", "", cleaned).strip()
                if not cleaned:
                    continue
                cleaned = re.sub(r"\s+", " ", cleaned)
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(cleaned)
        return normalized

    def _derive_plan_learning_outcomes(
        self,
        modules: list[CreateCourseModuleRequest] | list[CreatorCourseModulePlan],
        design_spec: AssignmentDesignSpec | None,
    ) -> list[str]:
        outcomes: list[str] = []
        for module in modules:
            module_outcomes = list(module.learning_outcomes) if getattr(module, "learning_outcomes", None) else []
            if not module_outcomes:
                module_design_spec = getattr(module, "design_spec", None) or design_spec
                module_outcomes = self._derive_module_learning_outcomes(
                    getattr(module, "title", "Module"),
                    getattr(module, "summary", "Build the module"),
                    module_design_spec,
                )
            outcomes.extend(module_outcomes[:2])
        return self._normalize_learning_outcomes(outcomes)[:6]

    def _derive_module_learning_outcomes(
        self,
        title: str,
        summary: str,
        design_spec: AssignmentDesignSpec | None,
    ) -> list[str]:
        title_lower = title.lower()
        summary_text = summary.strip().rstrip(".")

        if any(keyword in title_lower for keyword in ["lock", "concurrency", "retry", "idempot"]):
            return [
                "Keep the critical workflow correct under concurrent or repeated requests.",
                "Use locking, retries, or idempotency controls to preserve core invariants.",
            ]
        if "cache" in title_lower:
            return [
                "Use the configured cache to improve read performance without breaking correctness.",
                "Explain the freshness and invalidation tradeoffs introduced by the cache layer.",
            ]
        if any(keyword in title_lower for keyword in ["retrieval", "citation", "grounded", "corpus", "search"]):
            return [
                "Use the learner-visible data source to return relevant evidence for each request.",
                "Keep retrieval or grounded answers faithful to the available evidence.",
            ]
        if any(keyword in title_lower for keyword in ["tool", "control flow", "workflow"]):
            return [
                "Choose the right bounded workflow or tool path for each supported request.",
                "Keep the service contract stable while the internal workflow becomes more capable.",
            ]
        if any(keyword in title_lower for keyword in ["observability", "trace", "production", "slo"]):
            return [
                "Make the service observable enough to explain what happened during a run.",
                "Raise the module to a production-minded bar for reliability, latency, or operator trust.",
            ]

        outcomes: list[str] = []
        if summary_text:
            outcomes.append(summary_text + ".")
        if design_spec is not None and design_spec.capabilities.durable_state_required:
            outcomes.append("Protect the system invariants while the service mutates durable state.")
        elif design_spec is not None and design_spec.capabilities.retrieval_mode != RetrievalMode.none:
            outcomes.append("Turn the available data source into a dependable learner-visible service behavior.")
        else:
            outcomes.append("Implement the learner-visible behavior end to end and verify it with the provided checks.")
        return self._normalize_learning_outcomes(outcomes)[:3]

    def _resolve_creator_setup(
        self,
        goal: str,
        creator_setup: CreatorCourseSetupInput | CreatorCourseSetupChoices | None,
    ) -> CreatorCourseSetupChoices:
        setup = creator_setup or CreatorCourseSetupInput()
        lowered_goal = goal.lower()
        starter_type = setup.starter_type or self._starter_type_for_goal(lowered_goal)
        inferred_sources = self._infer_default_data_sources(lowered_goal)
        raw_sources = list(setup.data_sources or inferred_sources)
        seen_source_keys: set[str] = set()
        data_sources = []
        for source in raw_sources:
            key = source.asset_id or source.workspace_path or source.id
            if key in seen_source_keys:
                continue
            seen_source_keys.add(key)
            data_sources.append(source)

        primary_database = setup.primary_database
        if primary_database is None and any(
            keyword in lowered_goal
            for keyword in ["booking", "reservation", "inventory", "payment", "wallet", "transaction", "order"]
        ):
            primary_database = "postgres"

        cache_backend = setup.cache_backend
        if cache_backend is None and any(
            keyword in lowered_goal
            for keyword in ["cache", "caching", "booking", "reservation", "availability", "read-heavy", "latency"]
        ):
            cache_backend = "redis"

        return CreatorCourseSetupChoices(
            starter_type=starter_type,
            primary_database=primary_database,
            cache_backend=cache_backend,
            tech_stack=list(setup.tech_stack),
            data_sources=data_sources,
        )

    def _starter_type_for_goal(self, lowered_goal: str) -> StarterType:
        if any(keyword in lowered_goal for keyword in ["buggy", "fix", "debug", "legacy", "broken"]):
            return StarterType.working_buggy
        if any(keyword in lowered_goal for keyword in ["refactor", "improve", "optimize", "suboptimal"]):
            return StarterType.working_suboptimal
        if any(keyword in lowered_goal for keyword in ["from scratch", "blank", "implement everything"]):
            return StarterType.bare_stub
        return StarterType.partial_implementation

    def _infer_default_data_sources(self, lowered_goal: str) -> list[DataSourceSpec]:
        if any(
            keyword in lowered_goal
            for keyword in ["rag", "retrieval", "knowledge base", "documents", "corpus", "wiki", "search"]
        ):
            return [
                DataSourceSpec(
                    id="primary_corpus",
                    kind=DataSourceKind.uploaded_file,
                    title="Primary learner-visible corpus",
                    purpose=DataSourcePurpose.retrieval,
                    learner_visible=True,
                    format="json",
                    workspace_path="data/corpus.json",
                    description="A learner-visible corpus or uploaded file used for retrieval and grounded answers.",
                )
            ]
        return []

    def _apply_creator_choices_to_design_spec(
        self,
        design_spec: AssignmentDesignSpec | None,
        creator_choices,
    ) -> AssignmentDesignSpec | None:
        if design_spec is None:
            return None
        return design_spec.model_copy(
            update={
                "runtime_dependencies": design_spec.runtime_dependencies.model_copy(
                    update={
                        "starter_type": creator_choices.starter_type,
                        "visible_fixture_files": [
                            source.workspace_path
                            for source in creator_choices.data_sources
                            if source.learner_visible and source.workspace_path
                        ]
                        or list(design_spec.runtime_dependencies.visible_fixture_files),
                        "primary_database": creator_choices.primary_database,
                        "cache_backend": creator_choices.cache_backend,
                        "tech_stack": list(creator_choices.tech_stack),
                        "data_sources": list(creator_choices.data_sources),
                    }
                )
            }
        )

    def _creator_summary(self, design_spec: AssignmentDesignSpec, creator_choices) -> str:
        parts = [
            "We will create the course as a shared production-ready codebase."
            if design_spec.course_structure.shared_codebase
            else "We will create the course as separate module projects.",
            (
                "Learners start from a scaffolded starter app."
                if creator_choices.starter_type != StarterType.bare_stub
                else "Learners start closer to a blank scaffold and implement most of the system themselves."
            ),
        ]
        if creator_choices.primary_database:
            parts.append(f"The current plan assumes `{creator_choices.primary_database}` as the primary database.")
        if creator_choices.cache_backend:
            parts.append(f"The plan also gives learners access to `{creator_choices.cache_backend}` for caching work.")
        if creator_choices.data_sources:
            labels = ", ".join(f"`{source.title}`" for source in creator_choices.data_sources[:3])
            parts.append(f"Learners will also work with data sources such as {labels}.")
        capability_labels = ", ".join(design_spec.capabilities.summary_labels())
        parts.append(f"Under the hood, the generation pipeline will target {capability_labels}.")
        return " ".join(parts)

    def _creator_notes_for_module(self, module: CreateCourseModuleRequest, creator_choices) -> list[str]:
        notes: list[str] = []
        summary_lower = (module.summary or "").lower()
        if creator_choices.primary_database and any(keyword in summary_lower for keyword in ["lock", "transaction", "concurrency"]):
            notes.append(f"Expected to use `{creator_choices.primary_database}` in this module.")
        if creator_choices.cache_backend and "cache" in summary_lower:
            notes.append(f"Expected to use `{creator_choices.cache_backend}` in this module.")
        if creator_choices.data_sources and any(keyword in summary_lower for keyword in ["retrieval", "grounded", "corpus", "search", "citation"]):
            notes.append(
                "This module should use learner-visible data sources such as "
                + ", ".join(f"`{source.title}`" for source in creator_choices.data_sources[:2])
                + "."
            )
        if creator_choices.starter_type.name.startswith("working"):
            notes.append("Learners should inherit a starter that already runs, then improve it.")
        elif creator_choices.starter_type == creator_choices.starter_type.partial_implementation:
            notes.append("Learners should inherit a partial starter so they can focus on the core change.")
        else:
            notes.append("Learners should implement most of this module themselves from a bare scaffold.")
        return notes

    def _preferred_package_type(
        self,
        request: GenerateCourseFromBriefRequest,
        recommended: PackageType | None,
    ) -> PackageType:
        if request.package_type_hint is not None:
            return request.package_type_hint

        brief = request.goal.lower()
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
                        "progression_mode": ProgressionMode.independent_modules,
                        "shared_codebase": shared_codebase,
                    }
                ),
                "assessment_strategy": design_spec.assessment_strategy.model_copy(
                    update={"cumulative_module_gates": False}
                ),
            }
        )
