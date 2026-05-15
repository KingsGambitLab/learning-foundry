from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import current_user, require_role, verify_enrollment_owner
from app.domain.auth import Role, User

from app.domain.assets import (
    CreateCreatorAssetRequest,
    CreatorAssetList,
    CreatorAssetRecord,
    DeleteCreatorAssetResult,
)
from app.domain.course import (
    CourseEvent,
    CourseGenerationStatus,
    CreateCourseFromCreatorPlanRequest,
    DraftTimelineResponse,
    LocalDraftResetResult,
    CourseReviewReport,
    CourseRun,
    CourseRunList,
    CreateCourseRunRequest,
    GenerateCreatorCoursePlanRequest,
    GenerateCreatorCoursePlanResponse,
    GenerateCourseFromBriefRequest,
    GenerateCourseFromBriefResponse,
    QueueCourseGenerationResponse,
    QueueCourseOperationResponse,
    QueueCourseRevisionResponse,
    RecommendCreatorStackContractRequest,
    RecommendCreatorStackContractResponse,
    SuggestLearningOutcomesRequest,
    SuggestLearningOutcomesResponse,
)
from app.domain.publish import PublishedVersionList
from app.domain.grader import DeliverableGraderPlan, TaskAgentGraderPlanCollection
from app.domain.learner import (
    CreateEnrollmentRequest,
    LaunchWorkspaceRequest,
    LearnerEnrollment,
    LearnerEnrollmentList,
    LearnerDeliverableExperience,
    LearnerWorkspaceFileContent,
    LearnerWorkspaceFileList,
    LearnerWorkspaceFileWriteResult,
    LearnerWorkspaceSession,
    PublishedCourseCatalog,
    SubmitDeliverableRequest,
    WriteLearnerWorkspaceFileRequest,
)
from app.domain.testing import (
    CreateCreatorFeedbackRequest,
    CreateLearnerEvaluationReportRequest,
    CreateLearnerFeedbackRequest,
    CreatorFeedbackList,
    CreatorFeedbackRecord,
    CreatorTestingView,
    LearnerCourseEvaluationReport,
    LearnerFeedbackList,
    LearnerFeedbackRecord,
    LearnerTestingView,
)
from app.domain.registry import DESIGN_CATALOG
from app.domain.sandbox import SandboxAvailability
from app.domain.task_agent import DeliverableGate, TaskAgentServiceSpec
from app.domain.workflow import (
    BundleFileContent,
    CreateWorkflowRunRequest,
    GateDecisionRequest,
    MaterializedBundle,
    MaterializeBundleRequest,
    WorkflowEvent,
    WorkflowNodeExecution,
    WorkflowReviewSummary,
    WorkflowRun,
    WorkflowRunList,
)
from app.services.course_patterns import CATALOG_PATTERNS, course_pattern_by_slug
from app.services.assignment_design_inference import AssignmentDesignInference, GenerationIntake, infer_assignment_design
from app.services.course_generation_service import CourseGenerationService
from app.services.creator_asset_service import CreatorAssetService
from app.services.course_workflow_service import CourseWorkflowConflictError, CourseWorkflowService
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.grader_planner import build_all_task_agent_grader_plans, build_task_agent_grader_plan
from app.services.lms_service import LMSConflictError, LMSService
from app.services.spec_validation import ValidationResult, compute_task_agent_gate, validate_task_agent_spec
from app.services.workflow_service import (
    WorkflowConflictError,
    WorkflowGateRefused,
    WorkflowService,
)

router = APIRouter()


def _workflow_service(request: Request) -> WorkflowService:
    return request.app.state.workflow_service


def _course_workflow_service(request: Request) -> CourseWorkflowService:
    return request.app.state.course_workflow_service


def _course_generation_service(request: Request) -> CourseGenerationService:
    return request.app.state.course_generation_service


def _creator_asset_service(request: Request) -> CreatorAssetService:
    return request.app.state.creator_asset_service


def _docker_sandbox_runner(request: Request) -> DockerSandboxRunner:
    return request.app.state.docker_sandbox_runner


def _lms_service(request: Request) -> LMSService:
    return request.app.state.lms_service


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/sandbox/status", response_model=SandboxAvailability, tags=["system"], dependencies=[Depends(require_role(Role.creator))])
def sandbox_status(request: Request) -> SandboxAvailability:
    return _docker_sandbox_runner(request).status()


@router.get("/v1/registry", tags=["registry"], dependencies=[Depends(require_role(Role.creator))])
def get_registry():
    return {
        "package_types": [package_type.value for package_type in DESIGN_CATALOG.package_types],
        "domain_packs": [domain_pack.model_dump(mode="json") for domain_pack in DESIGN_CATALOG.domain_packs],
        "overlays": [overlay.model_dump(mode="json") for overlay in DESIGN_CATALOG.overlays],
    }


@router.get("/v1/domain-packs", tags=["registry"], dependencies=[Depends(require_role(Role.creator))])
def list_domain_packs():
    return DESIGN_CATALOG.domain_packs


@router.get("/v1/overlays", tags=["registry"], dependencies=[Depends(require_role(Role.creator))])
def list_overlays():
    return DESIGN_CATALOG.overlays


@router.get("/v1/course-patterns", tags=["registry"], dependencies=[Depends(require_role(Role.creator))])
def get_course_patterns():
    return CATALOG_PATTERNS


@router.get("/v1/course-patterns/{course_slug}", tags=["registry"], dependencies=[Depends(require_role(Role.creator))])
def get_course_pattern(course_slug: str):
    pattern = course_pattern_by_slug(course_slug)
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"Unknown course pattern '{course_slug}'.")
    return pattern


@router.post("/v1/designs/infer", response_model=AssignmentDesignInference, tags=["intake"], dependencies=[Depends(require_role(Role.creator))])
def infer_assignment_design_for_intake(intake: GenerationIntake) -> AssignmentDesignInference:
    return infer_assignment_design(
        title=intake.title,
        problem_statement=intake.problem_statement,
        learning_outcomes=intake.learning_outcomes,
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


@router.post("/v1/specs/task-agent/validate", response_model=ValidationResult, tags=["validation"], dependencies=[Depends(require_role(Role.creator))])
def validate_task_agent(spec: TaskAgentServiceSpec) -> ValidationResult:
    return validate_task_agent_spec(spec)


@router.post("/v1/specs/task-agent/gates/{deliverable_id}", response_model=DeliverableGate, tags=["validation"], dependencies=[Depends(require_role(Role.creator))])
def compute_gate(deliverable_id: str, spec: TaskAgentServiceSpec) -> DeliverableGate:
    try:
        return compute_task_agent_gate(spec, deliverable_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/v1/specs/task-agent/grader-plans", response_model=TaskAgentGraderPlanCollection, tags=["validation"], dependencies=[Depends(require_role(Role.creator))])
def build_task_agent_grader_plans(spec: TaskAgentServiceSpec) -> TaskAgentGraderPlanCollection:
    return build_all_task_agent_grader_plans(spec)


@router.post("/v1/specs/task-agent/grader-plans/{deliverable_id}", response_model=DeliverableGraderPlan, tags=["validation"], dependencies=[Depends(require_role(Role.creator))])
def build_task_agent_grader_plan_for_deliverable(deliverable_id: str, spec: TaskAgentServiceSpec) -> DeliverableGraderPlan:
    try:
        return build_task_agent_grader_plan(spec, deliverable_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/v1/workflow-runs", response_model=WorkflowRunList, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def list_workflow_runs(request: Request) -> WorkflowRunList:
    return _workflow_service(request).list_runs()


@router.post("/v1/workflow-runs", response_model=WorkflowRun, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def create_workflow_run(payload: CreateWorkflowRunRequest, request: Request) -> WorkflowRun:
    return _workflow_service(request).create_run(payload.intake)


@router.get("/v1/workflow-runs/{run_id}", response_model=WorkflowRun, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def get_workflow_run(run_id: str, request: Request) -> WorkflowRun:
    run = _workflow_service(request).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.")
    return run


@router.get("/v1/workflow-runs/{run_id}/events", response_model=list[WorkflowEvent], tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def get_workflow_events(run_id: str, request: Request) -> list[WorkflowEvent]:
    service = _workflow_service(request)
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.")
    return service.list_events(run_id)


@router.get("/v1/workflow-runs/{run_id}/nodes", response_model=list[WorkflowNodeExecution], tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def get_workflow_nodes(run_id: str, request: Request) -> list[WorkflowNodeExecution]:
    service = _workflow_service(request)
    try:
        return service.list_node_executions(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc


@router.get("/v1/workflow-runs/{run_id}/review", response_model=WorkflowReviewSummary, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def get_workflow_review(run_id: str, request: Request) -> WorkflowReviewSummary:
    service = _workflow_service(request)
    try:
        return service.get_review_summary(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc


@router.get("/v1/workflow-runs/{run_id}/workspace", response_model=MaterializedBundle, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def get_workflow_workspace(run_id: str, request: Request) -> MaterializedBundle:
    service = _workflow_service(request)
    try:
        return service.get_workspace(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/workflow-runs/{run_id}/workspace/file", response_model=BundleFileContent, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def get_workflow_workspace_file(
    run_id: str,
    request: Request,
    path: str = Query(..., description="Relative path inside the prepared workspace, e.g. public/starter/deliverable_1/app.py"),
) -> BundleFileContent:
    service = _workflow_service(request)
    try:
        return service.read_workspace_file(run_id, path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workspace file '{path}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/workflow-runs/{run_id}/nodes/execute", response_model=WorkflowRun, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def execute_workflow_nodes(
    run_id: str,
    request: Request,
    start_node: str | None = None,
) -> WorkflowRun:
    """Legacy LangGraph node-loop trigger — retired in Wave 5b.

    The per-deliverable authoring/reviewer loop was removed when the
    outcome-mode graph became the only live generation path. The route
    is preserved for API stability but always returns ``409`` so any
    surviving caller fails loudly rather than silently no-oping.
    """
    del start_node
    service = _workflow_service(request)
    if service.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.")
    raise HTTPException(
        status_code=409,
        detail=(
            "The per-deliverable LangGraph node loop is retired; outcome-mode "
            "generation drives the live pipeline."
        ),
    )


@router.post("/v1/workflow-runs/{run_id}/decisions", response_model=WorkflowRun, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def decide_workflow_gate(run_id: str, decision: GateDecisionRequest, request: Request) -> WorkflowRun:
    service = _workflow_service(request)
    try:
        return service.apply_gate_decision(run_id, decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowGateRefused as exc:
        # Codex review #7: refusing to advance a legacy run without
        # execution evidence is a conflict (state, not validation), so
        # 409 matches how WorkflowConflictError is already surfaced.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/v1/course-runs/{course_run_id}/decisions",
    response_model=GenerateCourseFromBriefResponse,
    tags=["course"],
    dependencies=[Depends(require_role(Role.creator))],
)
def decide_course_run_gate(
    course_run_id: str,
    decision: GateDecisionRequest,
    request: Request,
) -> GenerateCourseFromBriefResponse:
    """Apply a gate decision to an outcome-mode course run.

    Codex review #6 finding #1: outcome runs are course-run-scoped and
    were unreachable via the legacy ``/v1/workflow-runs/.../decisions``
    route. This endpoint dispatches the decision to
    ``CourseGenerationService.resume_outcome_workflow_after_gate``.

    Returns
    -------
    The adapted ``GenerateCourseFromBriefResponse`` describing the
    post-decision state (next pending gate, blocking reasons, etc.).

    Errors
    ------
    * 404 — the course_run_id is unknown OR the row does not carry an
      ``outcome_state`` (i.e. it's a legacy workflow run and the
      caller should hit ``/v1/workflow-runs/{run_id}/decisions``).
    * 409 — the run is not currently paused at a gate awaiting human.
    * 400 — the gate/decision payload is structurally invalid.
    """
    course_workflow = _course_workflow_service(request)
    course_run = course_workflow.store.get_course_run(course_run_id)
    if course_run is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown course run '{course_run_id}'."
        )
    # Legacy workflow runs persist their state through
    # ``WorkflowService.apply_gate_decision``; the course-run route is
    # outcome-specific. A 404 here points the caller at the right
    # endpoint without leaking the row's existence.
    if not (course_run.payload_json or {}).get("outcome_state"):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Course run '{course_run_id}' is not an outcome-mode run; "
                "use POST /v1/workflow-runs/<id>/decisions for legacy runs."
            ),
        )

    # Conflict check: the run must be paused at a gate awaiting human.
    outcome_state = course_run.payload_json["outcome_state"]
    if outcome_state.get("status") != "awaiting_human":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Course run '{course_run_id}' is not awaiting a gate decision "
                f"(status={outcome_state.get('status')!r}, "
                f"stage={outcome_state.get('stage')!r})."
            ),
        )

    service = _course_generation_service(request)
    try:
        return service.resume_outcome_workflow_after_gate(
            course_run_id,
            gate=decision.gate,
            decision=decision.decision,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown course run '{course_run_id}'."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/workflow-runs/{run_id}/materialize", response_model=WorkflowRun, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def materialize_workflow_run(run_id: str, payload: MaterializeBundleRequest, request: Request) -> WorkflowRun:
    service = _workflow_service(request)
    try:
        return service.materialize_run(run_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/course-runs", response_model=CourseRunList, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def list_course_runs(request: Request) -> CourseRunList:
    return _course_workflow_service(request).list_runs()


@router.get("/v1/course-generation/status", response_model=CourseGenerationStatus, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def get_course_generation_status(request: Request) -> CourseGenerationStatus:
    return _course_generation_service(request).status()


@router.get("/v1/creator-assets", response_model=CreatorAssetList, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def list_creator_assets(request: Request) -> CreatorAssetList:
    return _creator_asset_service(request).list_assets()


@router.post("/v1/creator-assets", response_model=CreatorAssetRecord, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def create_creator_asset(
    payload: CreateCreatorAssetRequest,
    request: Request,
) -> CreatorAssetRecord:
    try:
        return _creator_asset_service(request).create_asset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/v1/creator-assets/{asset_id}", response_model=DeleteCreatorAssetResult, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def delete_creator_asset(asset_id: str, request: Request) -> DeleteCreatorAssetResult:
    deleted = _creator_asset_service(request).delete_asset(asset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown creator asset '{asset_id}'.")
    return DeleteCreatorAssetResult(asset_id=asset_id)


@router.post("/v1/course-generation/suggest-outcomes", response_model=SuggestLearningOutcomesResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def suggest_learning_outcomes(
    payload: SuggestLearningOutcomesRequest,
    request: Request,
) -> SuggestLearningOutcomesResponse:
    try:
        return _course_generation_service(request).suggest_learning_outcomes(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-generation/creator-plan", response_model=GenerateCreatorCoursePlanResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def generate_creator_course_plan(
    payload: GenerateCreatorCoursePlanRequest,
    request: Request,
) -> GenerateCreatorCoursePlanResponse:
    try:
        return _course_generation_service(request).generate_creator_plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-generation/creator-stack-contract", response_model=RecommendCreatorStackContractResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def recommend_creator_stack_contract(
    payload: RecommendCreatorStackContractRequest,
    request: Request,
) -> RecommendCreatorStackContractResponse:
    try:
        return _course_generation_service(request).recommend_creator_stack_contract(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/generate", response_model=GenerateCourseFromBriefResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def generate_course_run_from_brief(
    payload: GenerateCourseFromBriefRequest,
    request: Request,
) -> GenerateCourseFromBriefResponse:
    try:
        return _course_generation_service(request).generate_course_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/from-creator-plan", response_model=CourseRun, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def create_course_run_from_creator_plan(
    payload: CreateCourseFromCreatorPlanRequest,
    request: Request,
) -> CourseRun:
    try:
        return _course_generation_service(request).create_course_run_from_creator_plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/from-creator-plan-async", response_model=QueueCourseGenerationResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def queue_course_run_from_creator_plan(
    payload: CreateCourseFromCreatorPlanRequest,
    request: Request,
) -> QueueCourseGenerationResponse:
    try:
        return _course_generation_service(request).queue_course_run_from_creator_plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/generate-async", response_model=QueueCourseGenerationResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def queue_course_run_from_brief(
    payload: GenerateCourseFromBriefRequest,
    request: Request,
) -> QueueCourseGenerationResponse:
    try:
        return _course_generation_service(request).queue_course_run_generation(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/reset-local", response_model=LocalDraftResetResult, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def reset_local_course_state(request: Request) -> LocalDraftResetResult:
    return _course_workflow_service(request).reset_local_state()


@router.post("/v1/course-runs", response_model=CourseRun, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def create_course_run(payload: CreateCourseRunRequest, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.create_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/v1/course-runs/{course_run_id}", response_model=CourseRun, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def get_course_run(course_run_id: str, request: Request) -> CourseRun:
    run = _course_workflow_service(request).get_run(course_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.")
    return run


@router.get("/v1/course-runs/{course_run_id}/events", response_model=list[CourseEvent], tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def get_course_events(course_run_id: str, request: Request) -> list[CourseEvent]:
    service = _course_workflow_service(request)
    run = service.get_run(course_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.")
    return service.list_events(course_run_id)


@router.get("/v1/course-runs/{course_run_id}/timeline", response_model=DraftTimelineResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def get_course_timeline(course_run_id: str, request: Request) -> DraftTimelineResponse:
    service = _course_workflow_service(request)
    try:
        return service.timeline(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.get("/v1/course-runs/{course_run_id}/review", response_model=CourseReviewReport, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def review_course_run(course_run_id: str, request: Request) -> CourseReviewReport:
    service = _course_workflow_service(request)
    try:
        return service.review_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.get("/v1/course-runs/{course_run_id}/creator-view", response_model=CreatorTestingView, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def get_creator_testing_view(course_run_id: str, request: Request) -> CreatorTestingView:
    service = _course_workflow_service(request)
    try:
        return service.creator_view(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.get("/v1/course-runs/{course_run_id}/feedback", response_model=CreatorFeedbackList, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def list_creator_feedback(course_run_id: str, request: Request) -> CreatorFeedbackList:
    service = _course_workflow_service(request)
    try:
        return service.list_creator_feedback(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.post("/v1/course-runs/{course_run_id}/feedback", response_model=CreatorFeedbackRecord, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def create_creator_feedback(
    course_run_id: str,
    payload: CreateCreatorFeedbackRequest,
    request: Request,
) -> CreatorFeedbackRecord:
    service = _course_workflow_service(request)
    try:
        return service.record_creator_feedback(course_run_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/v1/course-runs/{course_run_id}/learner-eval", response_model=LearnerCourseEvaluationReport, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def get_latest_learner_evaluation(
    course_run_id: str,
    request: Request,
) -> LearnerCourseEvaluationReport:
    service = _course_workflow_service(request)
    try:
        report = service.get_latest_learner_evaluation(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    if report is None:
        raise HTTPException(status_code=404, detail=f"No learner evaluation report is recorded for '{course_run_id}'.")
    return report


@router.post("/v1/course-runs/{course_run_id}/learner-eval", response_model=LearnerCourseEvaluationReport, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def create_learner_evaluation(
    course_run_id: str,
    payload: CreateLearnerEvaluationReportRequest,
    request: Request,
) -> LearnerCourseEvaluationReport:
    service = _course_workflow_service(request)
    try:
        return service.record_learner_evaluation(course_run_id, payload)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else course_run_id
        if detail == course_run_id:
            raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{detail}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/course-runs/{course_run_id}/published-versions", response_model=PublishedVersionList, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def list_published_versions(course_run_id: str, request: Request) -> PublishedVersionList:
    service = _course_workflow_service(request)
    try:
        return service.list_published_versions(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.post("/v1/course-runs/{course_run_id}/sync", response_model=CourseRun, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def sync_course_run(course_run_id: str, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.sync_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.post("/v1/course-runs/{course_run_id}/publish", response_model=CourseRun, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def publish_course_run(course_run_id: str, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.publish_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/publish-async", response_model=QueueCourseOperationResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def queue_publish_course_run(course_run_id: str, request: Request) -> QueueCourseOperationResponse:
    service = _course_workflow_service(request)
    try:
        return service.queue_publish_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/create-revision", response_model=CourseRun, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def create_course_revision(course_run_id: str, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.create_revision(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/create-revision-async", response_model=QueueCourseRevisionResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def queue_course_revision(course_run_id: str, request: Request) -> QueueCourseRevisionResponse:
    service = _course_workflow_service(request)
    try:
        return service.queue_revision(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/materialize", response_model=CourseRun, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def materialize_course_run(
    course_run_id: str,
    payload: MaterializeBundleRequest,
    request: Request,
) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.materialize_run(course_run_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/materialize-async", response_model=QueueCourseOperationResponse, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def queue_materialize_course_run(
    course_run_id: str,
    payload: MaterializeBundleRequest,
    request: Request,
) -> QueueCourseOperationResponse:
    service = _course_workflow_service(request)
    try:
        return service.queue_materialize_run(course_run_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/course-runs/{course_run_id}/bundle", response_model=MaterializedBundle, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def get_course_bundle(course_run_id: str, request: Request) -> MaterializedBundle:
    run = _course_workflow_service(request).get_run(course_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.")
    if run.materialized_bundle is None:
        raise HTTPException(status_code=404, detail="This course run does not have a materialized bundle yet.")
    return run.materialized_bundle


@router.get("/v1/course-runs/{course_run_id}/bundle/file", response_model=BundleFileContent, tags=["course"], dependencies=[Depends(require_role(Role.creator))])
def read_course_bundle_file(
    course_run_id: str,
    request: Request,
    path: str = Query(..., description="Relative path inside the materialized course bundle, e.g. public/README.md"),
) -> BundleFileContent:
    service = _course_workflow_service(request)
    try:
        return service.read_bundle_file(course_run_id, path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown bundle file '{exc.args[0]}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/v1/workflow-runs/{run_id}/bundle", response_model=MaterializedBundle, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def get_workflow_bundle(run_id: str, request: Request) -> MaterializedBundle:
    service = _workflow_service(request)
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.")
    if run.artifacts.materialized_bundle is None:
        raise HTTPException(status_code=404, detail="This run does not have a materialized bundle yet.")
    return run.artifacts.materialized_bundle


@router.get("/v1/workflow-runs/{run_id}/bundle/file", response_model=BundleFileContent, tags=["workflow"], dependencies=[Depends(require_role(Role.creator))])
def read_workflow_bundle_file(
    run_id: str,
    request: Request,
    path: str = Query(..., description="Relative path inside the materialized bundle, e.g. public/README.md"),
) -> BundleFileContent:
    service = _workflow_service(request)
    try:
        return service.read_bundle_file(run_id, path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown bundle file '{exc.args[0]}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# The published-course catalog is not learner-exclusive — creators browse
# it too (e.g. to preview what learners see). Gate on authentication only,
# not role. Previously this required Role.learner, which 403'd logged-in
# creators and produced the UI's "couldn't refresh the published course
# catalog" error.
@router.get("/v1/lms/catalog", response_model=PublishedCourseCatalog, tags=["lms"], dependencies=[Depends(current_user)])
def list_lms_catalog(request: Request) -> PublishedCourseCatalog:
    return _lms_service(request).list_catalog()


@router.get("/v1/lms/enrollments", response_model=LearnerEnrollmentList, tags=["lms"], dependencies=[Depends(require_role(Role.learner))])
def list_lms_enrollments(
    request: Request,
    user: User = Depends(current_user),
) -> LearnerEnrollmentList:
    return _lms_service(request).list_enrollments(learner_id=str(user.id))


@router.post("/v1/lms/enrollments", response_model=LearnerEnrollment, tags=["lms"], dependencies=[Depends(require_role(Role.learner))])
def create_lms_enrollment(
    payload: CreateEnrollmentRequest,
    request: Request,
    user: User = Depends(current_user),
) -> LearnerEnrollment:
    service = _lms_service(request)
    try:
        return service.enroll(payload, learner_id=str(user.id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{payload.course_run_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/lms/enrollments/{enrollment_id}", response_model=LearnerEnrollment, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def get_lms_enrollment(enrollment_id: str, request: Request) -> LearnerEnrollment:
    service = _lms_service(request)
    try:
        return service.get_enrollment(enrollment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc


@router.get("/v1/lms/enrollments/{enrollment_id}/experience", response_model=LearnerDeliverableExperience, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def get_lms_deliverable_experience(
    enrollment_id: str,
    request: Request,
    deliverable_id: str | None = Query(None, description="Optional project deliverable id, e.g. exercise/01-contract"),
) -> LearnerDeliverableExperience:
    service = _lms_service(request)
    try:
        return service.get_deliverable_experience(enrollment_id, deliverable_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/lms/enrollments/{enrollment_id}/learner-view", response_model=LearnerTestingView, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def get_lms_learner_view(
    enrollment_id: str,
    request: Request,
    deliverable_id: str | None = Query(None, description="Optional learner-facing deliverable id."),
) -> LearnerTestingView:
    service = _lms_service(request)
    try:
        return service.get_learner_view(enrollment_id, deliverable_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/lms/enrollments/{enrollment_id}/feedback", response_model=LearnerFeedbackList, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def list_lms_feedback(enrollment_id: str, request: Request) -> LearnerFeedbackList:
    service = _lms_service(request)
    try:
        return service.list_feedback(enrollment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc


@router.post("/v1/lms/enrollments/{enrollment_id}/feedback", response_model=LearnerFeedbackRecord, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def create_lms_feedback(
    enrollment_id: str,
    payload: CreateLearnerFeedbackRequest,
    request: Request,
) -> LearnerFeedbackRecord:
    service = _lms_service(request)
    try:
        return service.record_feedback(enrollment_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/lms/enrollments/{enrollment_id}/workspace", response_model=LearnerEnrollment, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def launch_lms_workspace(
    enrollment_id: str,
    payload: LaunchWorkspaceRequest,
    request: Request,
) -> LearnerEnrollment:
    service = _lms_service(request)
    try:
        return service.launch_workspace(enrollment_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/v1/lms/enrollments/{enrollment_id}/workspace/files", response_model=LearnerWorkspaceFileList, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def list_lms_workspace_files(
    enrollment_id: str,
    request: Request,
    deliverable_id: str | None = Query(None, description="Optional learner-facing deliverable id."),
) -> LearnerWorkspaceFileList:
    service = _lms_service(request)
    try:
        return service.list_workspace_files(enrollment_id, deliverable_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/lms/enrollments/{enrollment_id}/workspace/file", response_model=LearnerWorkspaceFileContent, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def read_lms_workspace_file(
    enrollment_id: str,
    request: Request,
    path: str = Query(..., description="Relative path inside the learner workspace, e.g. app.py"),
    deliverable_id: str | None = Query(None, description="Optional learner-facing deliverable id."),
) -> LearnerWorkspaceFileContent:
    service = _lms_service(request)
    try:
        return service.read_workspace_file(enrollment_id, path, deliverable_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workspace file '{path}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/v1/lms/enrollments/{enrollment_id}/workspace/file", response_model=LearnerWorkspaceFileWriteResult, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def write_lms_workspace_file(
    enrollment_id: str,
    payload: WriteLearnerWorkspaceFileRequest,
    request: Request,
) -> LearnerWorkspaceFileWriteResult:
    service = _lms_service(request)
    try:
        return service.write_workspace_file(enrollment_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/lms/enrollments/{enrollment_id}/submit", response_model=LearnerDeliverableExperience, tags=["lms"], dependencies=[Depends(require_role(Role.learner)), Depends(verify_enrollment_owner)])
def submit_lms_deliverable(
    enrollment_id: str,
    payload: SubmitDeliverableRequest,
    request: Request,
) -> LearnerDeliverableExperience:
    service = _lms_service(request)
    try:
        return service.submit_project(enrollment_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
