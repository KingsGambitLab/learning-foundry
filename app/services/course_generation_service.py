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
    CreatorCourseDeliverablePlan,
    CreatorCoursePlan,
    CreateCourseDeliverableRequest,
    CreateCourseRunRequest,
    GenerateCreatorCoursePlanRequest,
    GenerateCreatorCoursePlanResponse,
    GenerateCourseFromBriefRequest,
    GenerateCourseFromBriefResponse,
    GeneratedCoursePlan,
    QueueCourseGenerationResponse,
    SuggestLearningOutcomesRequest,
    SuggestLearningOutcomesResponse,
    RecommendCreatorStackContractRequest,
    RecommendCreatorStackContractResponse,
)
from app.domain.registry import PackageType, RiskClass, StarterType
from app.domain.task_agent import (
    AssignmentDesignSpec,
    DataSourceKind,
    DataSourcePurpose,
    DataSourceSpec,
    ProjectFamily,
    ProgressionMode,
    RetrievalMode,
    WorkspaceScope,
)
from app.services.assignment_design_inference import (
    GenerationIntake,
    build_project_runtime_binding,
    build_project_runtime_plan,
    infer_assignment_design,
)
from app.services.coursegen_logging import coursegen_log_path, log_coursegen_event
from app.services.course_workflow_service import CourseWorkflowService
from app.services.openai_course_planner import (
    OpenAICourseGenerationError,
    OpenAICoursePlanner,
    OpenAICoursePlannerUnavailable,
)
from app.services.stack_catalog_service import StackCatalogService


class CourseGenerationService:
    def __init__(
        self,
        course_workflow_service: CourseWorkflowService,
        *,
        live_planner: OpenAICoursePlanner | None = None,
        stack_catalog_service: StackCatalogService | None = None,
        job_runner: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.course_workflow_service = course_workflow_service
        self.live_planner = live_planner or OpenAICoursePlanner()
        self.stack_catalog_service = stack_catalog_service or StackCatalogService()
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
        creator_deliverables = self._creator_plan_deliverables(
            request=plan_request,
            design_spec=adjusted_shared_design_spec,
            default_deliverables=normalized_plan.deliverables,
            creator_choices=resolved_setup,
        )
        normalized_outcomes = self._derive_plan_learning_outcomes(
            creator_deliverables,
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
            deliverables=creator_deliverables,
            creator_summary=self._creator_summary(adjusted_shared_design_spec, resolved_setup),
            notes=list(
                dict.fromkeys(
                    [
                        *normalized_plan.notes,
                        "Review the deliverable plan before creating the draft.",
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

    def recommend_creator_stack_contract(
        self,
        request: RecommendCreatorStackContractRequest,
    ) -> RecommendCreatorStackContractResponse:
        resolved_setup = self._resolve_creator_setup(request.goal, request.creator_setup)
        return self.stack_catalog_service.describe_choices(resolved_setup)

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
                creator_choices=plan.creator_choices,
                shared_design_spec=generated_plan.shared_design_spec,
                deliverables=generated_plan.deliverables,
            )
        )
        if len(course_run.deliverables) == len(plan.deliverables):
            for stored_deliverable, planned_deliverable in zip(course_run.deliverables, plan.deliverables, strict=False):
                stored_deliverable.title = planned_deliverable.title
                stored_deliverable.summary = planned_deliverable.summary
                stored_deliverable.learning_outcomes = list(planned_deliverable.learning_outcomes)
                stored_deliverable.notes = list(
                    dict.fromkeys(
                        [
                            *stored_deliverable.notes,
                            *planned_deliverable.creator_notes,
                        ]
                    )
                )
        course_run.notes = list(
            dict.fromkeys(
                [
                    *course_run.notes,
                    "Draft created from an approved creator plan.",
                    f"Starter preference: `{plan.creator_choices.starter_type.value}`.",
                    *([f"Implementation language: `{plan.creator_choices.implementation_language}`."] if plan.creator_choices.implementation_language else []),
                    *([f"Application framework: `{plan.creator_choices.application_framework}`."] if plan.creator_choices.application_framework else []),
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
                "deliverable_count": len(plan.deliverables),
                "deliverable_count": len(plan.deliverables),
                "goal": plan.goal,
                "learning_outcome_count": len(plan.learning_outcomes),
                "starter_type": plan.creator_choices.starter_type.value,
                "implementation_language": plan.creator_choices.implementation_language,
                "application_framework": plan.creator_choices.application_framework,
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
                    "Creator-approved deliverable plan queued.",
                    f"Starter preference: `{plan.creator_choices.starter_type.value}`.",
                    *([f"Implementation language: `{plan.creator_choices.implementation_language}`."] if plan.creator_choices.implementation_language else []),
                    *([f"Application framework: `{plan.creator_choices.application_framework}`."] if plan.creator_choices.application_framework else []),
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
                "deliverable_count": len(plan.deliverables),
                "deliverable_count": len(plan.deliverables),
                "goal": plan.goal,
                "learning_outcome_count": len(plan.learning_outcomes),
                "starter_type": plan.creator_choices.starter_type.value,
                "implementation_language": plan.creator_choices.implementation_language,
                "application_framework": plan.creator_choices.application_framework,
                "primary_database": plan.creator_choices.primary_database,
                "cache_backend": plan.creator_choices.cache_backend,
                "data_source_count": len(plan.creator_choices.data_sources),
                "message": "Approved deliverable plan accepted. Building the draft in the background.",
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
                deliverables=normalized_plan.deliverables,
            )
        )
        aligned_plan = self.course_workflow_service.generated_plan_from_run(
            course_run,
            notes=normalized_plan.notes,
        )
        course_run.goal = request.goal
        course_run.requested_learning_outcomes = self._derive_plan_learning_outcomes(
            aligned_plan.deliverables,
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
            log_coursegen_event(
                "course_generation_job_started",
                course_run_id=course_run_id,
                mode="brief",
                goal=request.goal,
                log_path=str(coursegen_log_path()),
            )
            normalized_plan, source, status, usage = self._generate_normalized_plan(request)
            log_coursegen_event(
                "course_generation_plan_ready",
                course_run_id=course_run_id,
                source=source.value,
                deliverable_count=len(normalized_plan.deliverables),
            )
            built = self.course_workflow_service.apply_generated_plan(
                course_run_id,
                plan=normalized_plan,
                source=source,
                generation_status=status,
                usage=usage,
                execute_shared_workflow_nodes=False,
                clear_active_operation=False,
            )
            log_coursegen_event(
                "course_generation_plan_applied",
                course_run_id=built.id,
                shared_workflow_run_id=built.shared_workflow_run_id,
                stage=built.stage.value,
                status=built.status.value,
            )
            self._finalize_background_generation(
                built.id,
                generation_status=status,
                completion_message="Draft finished building from the generated course brief.",
            )
        except Exception as exc:
            status = self.live_planner.status()
            log_coursegen_event(
                "course_generation_job_failed",
                course_run_id=course_run_id,
                mode="brief",
                error=str(exc),
            )
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
            log_coursegen_event(
                "course_generation_job_started",
                course_run_id=course_run_id,
                mode="creator_plan",
                goal=plan.goal,
                deliverable_count=len(plan.deliverables),
                log_path=str(coursegen_log_path()),
            )
            log_coursegen_event(
                "course_generation_creator_plan_compilation_started",
                course_run_id=course_run_id,
                goal=plan.goal,
                deliverable_count=len(plan.deliverables),
            )
            generated_plan = self._generated_plan_from_creator_plan(plan)
            log_coursegen_event(
                "course_generation_creator_plan_compilation_completed",
                course_run_id=course_run_id,
                deliverable_count=len(generated_plan.deliverables),
                package_type=generated_plan.package_type.value,
            )
            log_coursegen_event(
                "course_generation_apply_generated_plan_started",
                course_run_id=course_run_id,
                deliverable_count=len(generated_plan.deliverables),
            )
            built = self.course_workflow_service.apply_generated_plan(
                course_run_id,
                plan=generated_plan,
                source=CourseGenerationSource.deterministic_fallback,
                generation_status=self.live_planner.status(),
                execute_shared_workflow_nodes=False,
                clear_active_operation=False,
            )
            log_coursegen_event(
                "course_generation_plan_applied",
                course_run_id=built.id,
                shared_workflow_run_id=built.shared_workflow_run_id,
                stage=built.stage.value,
                status=built.status.value,
            )
            if len(built.deliverables) == len(plan.deliverables):
                for stored_deliverable, planned_deliverable in zip(built.deliverables, plan.deliverables, strict=False):
                    stored_deliverable.notes = list(
                        dict.fromkeys(
                            [
                                *stored_deliverable.notes,
                                *planned_deliverable.creator_notes,
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
                    "deliverable_count": len(plan.deliverables),
                    "deliverable_count": len(plan.deliverables),
                    "message": "Draft shell created from the approved creator plan. Review checks are still running.",
                },
            )
            self._finalize_background_generation(
                built.id,
                generation_status=self.live_planner.status(),
                completion_message="Draft finished building from the approved creator plan.",
            )
        except Exception as exc:
            log_coursegen_event(
                "course_generation_job_failed",
                course_run_id=course_run_id,
                mode="creator_plan",
                error=str(exc),
            )
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
            log_coursegen_event(
                "course_generation_workflow_execution_started",
                course_run_id=course_run.id,
                shared_workflow_run_id=course_run.shared_workflow_run_id,
            )
            if self.course_workflow_service.workflow_service.node_runtime is not None:
                self.course_workflow_service.workflow_service.execute_langgraph_nodes(
                    course_run.shared_workflow_run_id
                )
            course_run = self.course_workflow_service.sync_run(course_run.id)
            log_coursegen_event(
                "course_generation_workflow_execution_completed",
                course_run_id=course_run.id,
                shared_workflow_run_id=course_run.shared_workflow_run_id,
                stage=course_run.stage.value,
                status=course_run.status.value,
            )
        course_run.active_operation = None
        course_run.updated_at = datetime.now(UTC)
        course_run.generation_status = generation_status
        course_run.last_error = None
        self.course_workflow_service.store.save_course_run(course_run)
        course_run = self.course_workflow_service.sync_run(course_run.id)
        self.course_workflow_service.store.append_course_event(
            course_run.id,
            "course_generation_completed",
            {
                "message": completion_message,
                "deliverable_count": len(course_run.deliverables),
                "shared_workflow_run_id": course_run.shared_workflow_run_id,
            },
        )
        log_coursegen_event(
            "course_generation_job_completed",
            course_run_id=course_run.id,
            stage=course_run.stage.value,
            status=course_run.status.value,
            shared_workflow_run_id=course_run.shared_workflow_run_id,
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
        deliverables = [
            CreateCourseDeliverableRequest(
                deliverable_slug=deliverable.deliverable_slug,
                title=deliverable.title,
                summary=deliverable.summary,
                learning_outcomes=deliverable.learning_outcomes,
                design_spec=self._apply_creator_choices_to_design_spec(deliverable.design_spec or shared_design_spec, plan.creator_choices),
            )
            for deliverable in plan.deliverables
        ]
        return GeneratedCoursePlan(
            title=plan.title,
            summary=plan.summary,
            package_type=plan.package_type,
            shared_design_spec=shared_design_spec,
            deliverables=deliverables,
            notes=list(
                dict.fromkeys(
                    [
                        *plan.notes,
                        "Creator-approved deliverable plan.",
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
            implementation_language=creator_choices.implementation_language,
            language_version=creator_choices.language_version,
            application_framework=creator_choices.application_framework,
            framework_version=creator_choices.framework_version,
            package_manager=creator_choices.package_manager,
            primary_database=creator_choices.primary_database,
            primary_database_version=creator_choices.primary_database_version,
            cache_backend=creator_choices.cache_backend,
            cache_backend_version=creator_choices.cache_backend_version,
            tech_stack=list(creator_choices.tech_stack),
            data_sources=list(creator_choices.data_sources),
        )
        inferred = infer_assignment_design(
            title=intake.title,
            problem_statement=intake.problem_statement,
            package_type_hint=intake.package_type_hint,
            starter_type=intake.starter_type,
            implementation_language=intake.implementation_language,
            language_version=intake.language_version,
            application_framework=intake.application_framework,
            framework_version=intake.framework_version,
            package_manager=intake.package_manager,
            primary_database=intake.primary_database,
            primary_database_version=intake.primary_database_version,
            cache_backend=intake.cache_backend,
            cache_backend_version=intake.cache_backend_version,
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

        deliverables: list[CreateCourseDeliverableRequest] = []
        for deliverable in normalized.deliverables:
            design_spec = deliverable.design_spec or shared_design_spec
            design_spec = self._with_package_type(design_spec, normalized.package_type)
            design_spec = self._apply_creator_choices_to_design_spec(design_spec, creator_choices)
            learning_outcomes = self._normalize_learning_outcomes(
                deliverable.learning_outcomes or self._derive_deliverable_learning_outcomes(deliverable.title, deliverable.summary or deliverable.title, design_spec)
            )
            deliverables.append(
                CreateCourseDeliverableRequest(
                    deliverable_slug=deliverable.deliverable_slug,
                    title=deliverable.title.strip(),
                    summary=(deliverable.summary or deliverable.title).strip(),
                    learning_outcomes=learning_outcomes[:3],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=list(design_spec.overlays),
                )
            )

        if not deliverables:
            deliverables = self._fallback_deliverables(request, shared_design_spec, normalized.package_type)

        if normalized.package_type == PackageType.progressive_codebase_course:
            deliverables = [
                deliverable.model_copy(update={"design_spec": shared_design_spec})
                for deliverable in deliverables
            ]

        normalized.deliverables = deliverables
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
            implementation_language=creator_choices.implementation_language,
            language_version=creator_choices.language_version,
            application_framework=creator_choices.application_framework,
            framework_version=creator_choices.framework_version,
            package_manager=creator_choices.package_manager,
            primary_database=creator_choices.primary_database,
            primary_database_version=creator_choices.primary_database_version,
            cache_backend=creator_choices.cache_backend,
            cache_backend_version=creator_choices.cache_backend_version,
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
            deliverables=self._fallback_deliverables(request, design_spec, package_type),
            notes=[
                "Built from deterministic fallback planning.",
                "The course structure was inferred from the explicit assignment design because live OpenAI planning was unavailable.",
            ],
        )

    def _fallback_deliverables(
        self,
        request: GenerateCourseFromBriefRequest,
        design_spec: AssignmentDesignSpec,
        package_type: PackageType,
    ) -> list[CreateCourseDeliverableRequest]:
        family = design_spec.project_contract.family
        if design_spec.capabilities.retrieval_mode == RetrievalMode.grounded_answers:
            return [
                CreateCourseDeliverableRequest(
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
                CreateCourseDeliverableRequest(
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
                CreateCourseDeliverableRequest(
                    title="Quality tuning and evals",
                    summary="Improve answer quality with decomposition, reranking, and eval-driven iteration.",
                    learning_outcomes=[
                        "Use visible evaluation cases to improve grounded answer quality deliberately.",
                        "Refine retrieval and answer composition without regressing supported scenarios.",
                    ],
                    design_spec=design_spec,
                    overlays_hint=["productionization_overlay"],
                ),
                CreateCourseDeliverableRequest(
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
                CreateCourseDeliverableRequest(
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
                CreateCourseDeliverableRequest(
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
                CreateCourseDeliverableRequest(
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

        return self._compile_contract_deliverables(design_spec, package_type)

    def _compile_contract_deliverables(
        self,
        design_spec: AssignmentDesignSpec,
        package_type: PackageType,
    ) -> list[CreateCourseDeliverableRequest]:
        contract = design_spec.project_contract
        runtime_binding = contract.runtime_binding
        read_focus = contract.primary_read_paths[0] if contract.primary_read_paths else "serve the main request path reliably"
        write_focus = (
            contract.primary_write_paths[0]
            if contract.primary_write_paths
            else "apply changes without breaking the public contract"
        )
        invariant_focus = contract.invariants[0] if contract.invariants else "preserve the public contract for supported requests"
        operational_focus = (
            ", ".join(contract.operational_concerns[:2])
            if contract.operational_concerns
            else "observability and reliability"
        )
        runtime_focus = (
            runtime_binding.integration_points[0]
            if runtime_binding.integration_points
            else "Integrate the runtime pieces without breaking the project contract."
        )
        seed_focus = (
            runtime_binding.seed_artifacts[0]
            if runtime_binding.seed_artifacts
            else "Keep the starter and seeded data honest enough for the visible checks."
        )
        service_labels = [
            binding.technology or binding.service_id
            for binding in runtime_binding.backing_services
        ]
        services_summary = ", ".join(service_labels[:2]) if service_labels else "the runtime dependencies"
        base_overlays = list(design_spec.overlays)

        if contract.family == ProjectFamily.control_plane_service:
            deliverables = [
                CreateCourseDeliverableRequest(
                    title="Evaluation contract and state model",
                    summary=f"Define the control-plane surface, core entities, and invariants so the service can {read_focus}.",
                    learning_outcomes=[
                        f"Model the service around the invariant that {invariant_focus.rstrip('.')}.",
                        "Make the evaluation contract deterministic enough to debug confidently.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Live read path and decision logic",
                    summary=f"Implement the main read path so the service can {read_focus}.",
                    learning_outcomes=[
                        "Build the read path around clear decision rules and request context.",
                        "Keep read behavior understandable enough to explain why each decision happened.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Safe mutation path and runtime integration",
                    summary=f"Connect {services_summary} and the mutation path so the service can {write_focus}.",
                    learning_outcomes=[
                        f"Wire the runtime around {services_summary} without losing configuration coherence.",
                        seed_focus,
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Auditability and production confidence",
                    summary=f"Raise the service to a production bar for {operational_focus}.",
                    learning_outcomes=[
                        "Record enough traces and audit evidence for an operator to trust the system.",
                        "Harden the project against realistic rollout and incident scenarios.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=["productionization_overlay", "scale_slo_overlay"],
                ),
            ]
            return deliverables

        if contract.family == ProjectFamily.transactional_stateful_service:
            return [
                CreateCourseDeliverableRequest(
                    title="Service contract and durable model",
                    summary=f"Define the service surface, core entities, and persistence plan so the system can {read_focus}.",
                    learning_outcomes=[
                        f"Model the durable state around the invariant that {invariant_focus.rstrip('.')}.",
                        "Define a service contract that stays stable as state changes.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Read and write path correctness",
                    summary=f"Implement the main read and write paths so the service can {write_focus}.",
                    learning_outcomes=[
                        "Keep repeated or concurrent requests from breaking critical state transitions.",
                        "Connect the core read and write paths without hiding correctness tradeoffs.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Runtime integration and failure recovery",
                    summary=f"Connect {services_summary} and recovery behavior so the service can stay correct under load.",
                    learning_outcomes=[
                        runtime_focus,
                        seed_focus,
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Operational hardening",
                    summary=f"Raise the project to a production-minded bar for {operational_focus}.",
                    learning_outcomes=[
                        "Make failures visible enough for an operator to debug them quickly.",
                        "Push the project toward believable production reliability and latency.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=["productionization_overlay", "scale_slo_overlay"],
                ),
            ]

        if contract.family == ProjectFamily.workflow_agent_service:
            deliverables = [
                CreateCourseDeliverableRequest(
                    title="Request contract and bounded workflow",
                    summary=f"Define the workflow surface so the service can {read_focus}.",
                    learning_outcomes=[
                        "Define the request and response contract around bounded workflow steps.",
                        f"Keep the workflow honest about the invariant that {invariant_focus.rstrip('.')}.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Routing, tools, and control flow",
                    summary=f"Implement the workflow path so the service can {write_focus}.",
                    learning_outcomes=[
                        "Choose tools and control flow deliberately enough to explain each step.",
                        "Keep bounded routing behavior stable across the visible scenarios.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=base_overlays,
                ),
                CreateCourseDeliverableRequest(
                    title="Fallbacks, approvals, and runtime wiring",
                    summary=f"Connect {services_summary} and the failure path so the project can stay operable.",
                    learning_outcomes=[
                        runtime_focus,
                        "Add fallback or approval behavior without breaking the public contract.",
                    ],
                    design_spec=design_spec,
                    domain_pack_hint=design_spec.domain_pack,
                    overlays_hint=["productionization_overlay"],
                ),
            ]
            if package_type == PackageType.progressive_codebase_course:
                deliverables.append(
                    CreateCourseDeliverableRequest(
                        title="Evaluation and production polish",
                        summary=f"Use evaluation feedback to raise the workflow to a production bar for {operational_focus}.",
                        learning_outcomes=[
                            "Use evaluation feedback to improve workflow quality without guessing what regressed.",
                            "Balance reliability, latency, and operator visibility in the final system.",
                        ],
                        design_spec=design_spec,
                        domain_pack_hint=design_spec.domain_pack,
                        overlays_hint=["productionization_overlay", "scale_slo_overlay"],
                    )
                )
            return deliverables

        deliverables = [
            CreateCourseDeliverableRequest(
                title="Project contract and core behavior",
                summary=f"Define the service surface, core entities, and visible behavior so the project can {read_focus}.",
                learning_outcomes=[
                    "Turn the project brief into a bounded contract with clear observable behavior.",
                    f"Keep the implementation aligned to the invariant that {invariant_focus.rstrip('.')}.",
                ],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=base_overlays,
            ),
            CreateCourseDeliverableRequest(
                title="Runtime integrations and data flow",
                summary=f"Connect {services_summary} and the main workflow so the project can {write_focus}.",
                learning_outcomes=[
                    runtime_focus,
                    seed_focus,
                ],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=base_overlays,
            ),
            CreateCourseDeliverableRequest(
                title="Quality signals and production hardening",
                summary=f"Add checks, observability, and operational polish around {operational_focus}.",
                learning_outcomes=[
                    "Make failures and regressions visible through explicit checks and diagnostics.",
                    "Raise the project to a production-minded bar for reliability and operator trust.",
                ],
                design_spec=design_spec,
                domain_pack_hint=design_spec.domain_pack,
                overlays_hint=["productionization_overlay", "scale_slo_overlay"],
            ),
        ]
        return deliverables

    def _creator_plan_deliverables(
        self,
        *,
        request: GenerateCourseFromBriefRequest,
        design_spec: AssignmentDesignSpec,
        default_deliverables: list[CreateCourseDeliverableRequest],
        creator_choices,
    ) -> list[CreatorCourseDeliverablePlan]:
        deliverables: list[CreateCourseDeliverableRequest]
        fallback_deliverables = self._fallback_deliverables(
            request,
            design_spec,
            design_spec.course_structure.package_type,
        )
        if fallback_deliverables and (
            len(default_deliverables) > len(fallback_deliverables)
            or self._deliverable_plan_needs_override(default_deliverables, design_spec)
        ):
            deliverables = fallback_deliverables
        else:
            deliverables = default_deliverables

        return [
            CreatorCourseDeliverablePlan(
                deliverable_slug=deliverable.deliverable_slug or f"deliverable-{index}",
                title=deliverable.title,
                summary=deliverable.summary or deliverable.title,
                learning_outcomes=self._normalize_learning_outcomes(
                    deliverable.learning_outcomes
                    or self._derive_deliverable_learning_outcomes(
                        deliverable.title,
                        deliverable.summary or deliverable.title,
                        deliverable.design_spec or design_spec,
                    )
                ),
                creator_notes=self._creator_notes_for_deliverable(deliverable, creator_choices),
                design_spec=self._apply_creator_choices_to_design_spec(deliverable.design_spec or design_spec, creator_choices),
            )
            for index, deliverable in enumerate(deliverables, start=1)
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

        if design_spec.project_contract.family == ProjectFamily.control_plane_service:
            return self._contract_learning_outcomes(design_spec)

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
            return self._contract_learning_outcomes(design_spec)

        return self._contract_learning_outcomes(design_spec)

    def _contract_learning_outcomes(self, design_spec: AssignmentDesignSpec) -> list[str]:
        contract = design_spec.project_contract
        runtime_binding = contract.runtime_binding
        invariant = contract.invariants[0] if contract.invariants else "preserve the service contract for supported requests"
        read_path = contract.primary_read_paths[0] if contract.primary_read_paths else "serve the main request path reliably"
        write_path = contract.primary_write_paths[0] if contract.primary_write_paths else "evolve the implementation without breaking the public contract"
        operational_focus = (
            ", ".join(contract.operational_concerns[:2])
            if contract.operational_concerns
            else "observability and production reliability"
        )
        runtime_focus = (
            runtime_binding.integration_points[0]
            if runtime_binding.integration_points
            else "Integrate the runtime pieces without losing correctness."
        )
        return [
            f"Define the project contract so the system can {read_path}.",
            f"Implement the core workflow while preserving the invariant that {invariant.rstrip('.')}.",
            runtime_focus,
            f"Raise the project to a production-minded bar for {operational_focus}.",
        ]

    def _deliverable_plan_needs_override(
        self,
        deliverables: list[CreateCourseDeliverableRequest],
        design_spec: AssignmentDesignSpec,
    ) -> bool:
        if not deliverables:
            return False
        family = design_spec.project_contract.family
        lowered = " ".join(
            f"{deliverable.title} {deliverable.summary or ''}"
            for deliverable in deliverables
        ).lower()
        generic_markers = [
            "run contract",
            "structured output",
            "tooling and control flow",
            "approvals, fallbacks",
            "eval-driven",
            "production final at slo",
        ]
        if any(marker in lowered for marker in generic_markers):
            return True
        if family in {ProjectFamily.control_plane_service, ProjectFamily.transactional_stateful_service}:
            return all(marker not in lowered for marker in ["contract", "state", "read", "write", "runtime", "audit", "recovery"])
        return False

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
        deliverables: list[CreateCourseDeliverableRequest] | list[CreatorCourseDeliverablePlan],
        design_spec: AssignmentDesignSpec | None,
    ) -> list[str]:
        outcomes: list[str] = []
        for deliverable in deliverables:
            deliverable_outcomes = list(deliverable.learning_outcomes) if getattr(deliverable, "learning_outcomes", None) else []
            if not deliverable_outcomes:
                deliverable_design_spec = getattr(deliverable, "design_spec", None) or design_spec
                deliverable_outcomes = self._derive_deliverable_learning_outcomes(
                    getattr(deliverable, "title", "Deliverable"),
                    getattr(deliverable, "summary", "Build the deliverable"),
                    deliverable_design_spec,
                )
            outcomes.extend(deliverable_outcomes[:2])
        return self._normalize_learning_outcomes(outcomes)[:6]

    def _derive_deliverable_learning_outcomes(
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
                "Raise the deliverable to a production-minded bar for reliability, latency, or operator trust.",
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

        implementation_language = (setup.implementation_language or "").strip().lower() or None
        language_version = (setup.language_version or "").strip() or None
        application_framework = (setup.application_framework or "").strip().lower() or None
        framework_version = (setup.framework_version or "").strip() or None
        package_manager = (setup.package_manager or "").strip().lower() or None
        primary_database_version = (setup.primary_database_version or "").strip() or None
        cache_backend_version = (setup.cache_backend_version or "").strip() or None

        primary_database = (setup.primary_database or "").strip().lower() or None
        cache_backend = (setup.cache_backend or "").strip().lower() or None
        normalized = self.stack_catalog_service.describe_choices(
            CreatorCourseSetupChoices(
                starter_type=starter_type,
                implementation_language=implementation_language,
                language_version=language_version,
                application_framework=application_framework,
                framework_version=framework_version,
                package_manager=package_manager,
                primary_database=primary_database,
                primary_database_version=primary_database_version,
                cache_backend=cache_backend,
                cache_backend_version=cache_backend_version,
                tech_stack=list(setup.tech_stack),
                data_sources=data_sources,
            )
        ).creator_choices

        return CreatorCourseSetupChoices(
            starter_type=starter_type,
            implementation_language=normalized.implementation_language,
            language_version=normalized.language_version,
            application_framework=normalized.application_framework,
            framework_version=normalized.framework_version,
            package_manager=normalized.package_manager,
            primary_database=normalized.primary_database,
            primary_database_version=normalized.primary_database_version,
            cache_backend=normalized.cache_backend,
            cache_backend_version=normalized.cache_backend_version,
            tech_stack=list(setup.tech_stack),
            data_sources=data_sources,
        )

    def _starter_type_for_goal(self, lowered_goal: str) -> StarterType:
        if any(keyword in lowered_goal for keyword in ["from scratch", "blank", "implement everything"]):
            return StarterType.empty
        return StarterType.partial

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
        runtime_binding = build_project_runtime_binding(
            family=design_spec.project_contract.family,
            implementation_language=creator_choices.implementation_language,
            application_framework=creator_choices.application_framework,
            primary_database=creator_choices.primary_database,
            cache_backend=creator_choices.cache_backend,
            tech_stack=list(creator_choices.tech_stack),
            data_sources=list(creator_choices.data_sources),
        )
        runtime_plan = build_project_runtime_plan(
            family=design_spec.project_contract.family,
            implementation_language=creator_choices.implementation_language,
            language_version=creator_choices.language_version,
            application_framework=creator_choices.application_framework,
            framework_version=creator_choices.framework_version,
            package_manager=creator_choices.package_manager,
            primary_database=creator_choices.primary_database,
            primary_database_version=creator_choices.primary_database_version,
            cache_backend=creator_choices.cache_backend,
            cache_backend_version=creator_choices.cache_backend_version,
            tech_stack=list(creator_choices.tech_stack),
            data_sources=list(creator_choices.data_sources),
            allow_inference=False,
        )
        return design_spec.model_copy(
            update={
                "runtime_dependencies": design_spec.runtime_dependencies.model_copy(
                    update={
                        "starter_type": creator_choices.starter_type,
                        "implementation_language": creator_choices.implementation_language,
                        "language_version": creator_choices.language_version,
                        "application_framework": creator_choices.application_framework,
                        "framework_version": creator_choices.framework_version,
                        "package_manager": creator_choices.package_manager,
                        "visible_fixture_files": [
                            source.workspace_path
                            for source in creator_choices.data_sources
                            if source.learner_visible and source.workspace_path
                        ]
                        or list(design_spec.runtime_dependencies.visible_fixture_files),
                        "primary_database": creator_choices.primary_database,
                        "primary_database_version": creator_choices.primary_database_version,
                        "cache_backend": creator_choices.cache_backend,
                        "cache_backend_version": creator_choices.cache_backend_version,
                        "tech_stack": list(creator_choices.tech_stack),
                        "data_sources": list(creator_choices.data_sources),
                    }
                ),
                "project_contract": design_spec.project_contract.model_copy(
                    update={
                        "runtime_binding": runtime_binding,
                        "runtime_plan": runtime_plan,
                    }
                ),
            }
        )

    def _creator_summary(self, design_spec: AssignmentDesignSpec, creator_choices) -> str:
        parts = [
            "We will create the course as a shared production-ready codebase."
            if design_spec.course_structure.shared_codebase
            else "We will create the course as separate deliverable projects.",
            (
                "Learners start from a starter app with key pieces already wired."
                if creator_choices.starter_type != StarterType.empty
                else "Learners start closer to a blank starter and implement most of the system themselves."
            ),
        ]
        if creator_choices.implementation_language:
            stack_note = f"The current plan targets `{creator_choices.implementation_language}`"
            if creator_choices.language_version:
                stack_note += f" version `{creator_choices.language_version}`"
            if creator_choices.application_framework:
                stack_note += f" with `{creator_choices.application_framework}`"
                if creator_choices.framework_version:
                    stack_note += f" version `{creator_choices.framework_version}`"
            if creator_choices.package_manager:
                stack_note += f" via `{creator_choices.package_manager}`"
            parts.append(stack_note + ".")
        if creator_choices.primary_database:
            database_note = f"The current plan assumes `{creator_choices.primary_database}`"
            if creator_choices.primary_database_version:
                database_note += f" `{creator_choices.primary_database_version}`"
            parts.append(database_note + " as the primary database.")
        if creator_choices.cache_backend:
            cache_note = f"The plan also gives learners access to `{creator_choices.cache_backend}`"
            if creator_choices.cache_backend_version:
                cache_note += f" `{creator_choices.cache_backend_version}`"
            parts.append(cache_note + " for caching work.")
        if creator_choices.data_sources:
            labels = ", ".join(f"`{source.title}`" for source in creator_choices.data_sources[:3])
            parts.append(f"Learners will also work with data sources such as {labels}.")
        if creator_choices.tech_stack:
            parts.append(
                "The runtime should honor explicit requirements such as "
                + ", ".join(f"`{item}`" for item in creator_choices.tech_stack[:4])
                + "."
            )
        capability_labels = ", ".join(design_spec.capabilities.summary_labels())
        parts.append(f"Under the hood, the generation pipeline will target {capability_labels}.")
        return " ".join(parts)

    def _creator_notes_for_deliverable(self, deliverable: CreateCourseDeliverableRequest, creator_choices) -> list[str]:
        notes: list[str] = []
        summary_lower = (deliverable.summary or "").lower()
        if creator_choices.primary_database and any(keyword in summary_lower for keyword in ["lock", "transaction", "concurrency"]):
            notes.append(f"Expected to use `{creator_choices.primary_database}` in this deliverable.")
        if creator_choices.cache_backend and "cache" in summary_lower:
            notes.append(f"Expected to use `{creator_choices.cache_backend}` in this deliverable.")
        if creator_choices.data_sources and any(keyword in summary_lower for keyword in ["retrieval", "grounded", "corpus", "search", "citation"]):
            notes.append(
                "This deliverable should use learner-visible data sources such as "
                + ", ".join(f"`{source.title}`" for source in creator_choices.data_sources[:2])
                + "."
            )
        if creator_choices.implementation_language:
            stack_note = f"Implement this deliverable in `{creator_choices.implementation_language}`"
            if creator_choices.language_version:
                stack_note += f" version `{creator_choices.language_version}`"
            if creator_choices.application_framework:
                stack_note += f" using `{creator_choices.application_framework}`"
                if creator_choices.framework_version:
                    stack_note += f" version `{creator_choices.framework_version}`"
            if creator_choices.package_manager:
                stack_note += f" with `{creator_choices.package_manager}`"
            notes.append(stack_note + ".")
        if creator_choices.tech_stack:
            notes.append(
                "Keep this deliverable aligned with runtime requirements like "
                + ", ".join(f"`{item}`" for item in creator_choices.tech_stack[:3])
                + "."
            )
        if creator_choices.starter_type == StarterType.partial:
            notes.append("Learners should inherit a partial starter so they can focus on the core change.")
        else:
            notes.append("Learners should implement most of this deliverable themselves from a bare starter.")
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
                            else WorkspaceScope.per_deliverable_workspace
                        ),
                        "progression_mode": ProgressionMode.independent_deliverables,
                        "shared_codebase": shared_codebase,
                    }
                ),
                "assessment_strategy": design_spec.assessment_strategy.model_copy(
                    update={"cumulative_deliverable_gates": False}
                ),
            }
        )
