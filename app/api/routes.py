from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

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
from app.domain.grading import (
    GradeTaskAgentRequest,
    LiveGradeTaskAgentRequest,
    LiveGradeTaskAgentSpecRequest,
    LiveTaskAgentGradeReport,
    DeliverableGradeReport,
    TaskAgentSubmission,
)
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
from app.services.openai_task_agent_authoring import TaskAgentAuthoringStatus
from app.services.spec_validation import ValidationResult, compute_task_agent_gate, validate_task_agent_spec
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner, TaskAgentRunnerError
from app.services.task_agent_grader import grade_task_agent_submission
from app.services.workflow_service import WorkflowConflictError, WorkflowService

router = APIRouter()


def _workflow_service(request: Request) -> WorkflowService:
    return request.app.state.workflow_service


def _course_workflow_service(request: Request) -> CourseWorkflowService:
    return request.app.state.course_workflow_service


def _course_generation_service(request: Request) -> CourseGenerationService:
    return request.app.state.course_generation_service


def _creator_asset_service(request: Request) -> CreatorAssetService:
    return request.app.state.creator_asset_service


def _task_agent_blackbox_runner(request: Request) -> TaskAgentBlackBoxRunner:
    return request.app.state.task_agent_blackbox_runner


def _docker_sandbox_runner(request: Request) -> DockerSandboxRunner:
    return request.app.state.docker_sandbox_runner


def _lms_service(request: Request) -> LMSService:
    return request.app.state.lms_service


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/sandbox/status", response_model=SandboxAvailability, tags=["system"])
def sandbox_status(request: Request) -> SandboxAvailability:
    return _docker_sandbox_runner(request).status()


@router.get("/v1/task-agent-authoring/status", response_model=TaskAgentAuthoringStatus, tags=["system"])
def task_agent_authoring_status(request: Request) -> TaskAgentAuthoringStatus:
    return _workflow_service(request).task_agent_authoring_status()


@router.get("/v1/registry", tags=["registry"])
def get_registry():
    return {
        "package_types": [package_type.value for package_type in DESIGN_CATALOG.package_types],
        "domain_packs": [domain_pack.model_dump(mode="json") for domain_pack in DESIGN_CATALOG.domain_packs],
        "overlays": [overlay.model_dump(mode="json") for overlay in DESIGN_CATALOG.overlays],
    }


@router.get("/v1/domain-packs", tags=["registry"])
def list_domain_packs():
    return DESIGN_CATALOG.domain_packs


@router.get("/v1/overlays", tags=["registry"])
def list_overlays():
    return DESIGN_CATALOG.overlays


@router.get("/v1/course-patterns", tags=["registry"])
def get_course_patterns():
    return CATALOG_PATTERNS


@router.get("/v1/course-patterns/{course_slug}", tags=["registry"])
def get_course_pattern(course_slug: str):
    pattern = course_pattern_by_slug(course_slug)
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"Unknown course pattern '{course_slug}'.")
    return pattern


@router.post("/v1/designs/infer", response_model=AssignmentDesignInference, tags=["intake"])
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


@router.post("/v1/specs/task-agent/validate", response_model=ValidationResult, tags=["validation"])
def validate_task_agent(spec: TaskAgentServiceSpec) -> ValidationResult:
    return validate_task_agent_spec(spec)


@router.post("/v1/specs/task-agent/gates/{deliverable_id}", response_model=DeliverableGate, tags=["validation"])
def compute_gate(deliverable_id: str, spec: TaskAgentServiceSpec) -> DeliverableGate:
    try:
        return compute_task_agent_gate(spec, deliverable_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/v1/specs/task-agent/grader-plans", response_model=TaskAgentGraderPlanCollection, tags=["validation"])
def build_task_agent_grader_plans(spec: TaskAgentServiceSpec) -> TaskAgentGraderPlanCollection:
    return build_all_task_agent_grader_plans(spec)


@router.post("/v1/specs/task-agent/grader-plans/{deliverable_id}", response_model=DeliverableGraderPlan, tags=["validation"])
def build_task_agent_grader_plan_for_deliverable(deliverable_id: str, spec: TaskAgentServiceSpec) -> DeliverableGraderPlan:
    try:
        return build_task_agent_grader_plan(spec, deliverable_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/v1/specs/task-agent/grade/{deliverable_id}", response_model=DeliverableGradeReport, tags=["grading"])
def grade_task_agent_spec(deliverable_id: str, payload: GradeTaskAgentRequest) -> DeliverableGradeReport:
    try:
        return grade_task_agent_submission(payload.spec, deliverable_id, payload.submission)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/v1/specs/task-agent/grade-live/{deliverable_id}", response_model=LiveTaskAgentGradeReport, tags=["grading"])
def grade_task_agent_spec_live(
    deliverable_id: str,
    payload: LiveGradeTaskAgentSpecRequest,
    request: Request,
) -> LiveTaskAgentGradeReport:
    runner = _task_agent_blackbox_runner(request)
    try:
        return runner.grade_live(payload.spec, deliverable_id, payload.live)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TaskAgentRunnerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/v1/workflow-runs", response_model=WorkflowRunList, tags=["workflow"])
def list_workflow_runs(request: Request) -> WorkflowRunList:
    return _workflow_service(request).list_runs()


@router.post("/v1/workflow-runs", response_model=WorkflowRun, tags=["workflow"])
def create_workflow_run(payload: CreateWorkflowRunRequest, request: Request) -> WorkflowRun:
    return _workflow_service(request).create_run(payload.intake)


@router.get("/v1/workflow-runs/{run_id}", response_model=WorkflowRun, tags=["workflow"])
def get_workflow_run(run_id: str, request: Request) -> WorkflowRun:
    run = _workflow_service(request).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.")
    return run


@router.get("/v1/workflow-runs/{run_id}/events", response_model=list[WorkflowEvent], tags=["workflow"])
def get_workflow_events(run_id: str, request: Request) -> list[WorkflowEvent]:
    service = _workflow_service(request)
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.")
    return service.list_events(run_id)


@router.get("/v1/workflow-runs/{run_id}/nodes", response_model=list[WorkflowNodeExecution], tags=["workflow"])
def get_workflow_nodes(run_id: str, request: Request) -> list[WorkflowNodeExecution]:
    service = _workflow_service(request)
    try:
        return service.list_node_executions(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc


@router.get("/v1/workflow-runs/{run_id}/review", response_model=WorkflowReviewSummary, tags=["workflow"])
def get_workflow_review(run_id: str, request: Request) -> WorkflowReviewSummary:
    service = _workflow_service(request)
    try:
        return service.get_review_summary(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc


@router.get("/v1/workflow-runs/{run_id}/workspace", response_model=MaterializedBundle, tags=["workflow"])
def get_workflow_workspace(run_id: str, request: Request) -> MaterializedBundle:
    service = _workflow_service(request)
    try:
        return service.get_workspace(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/workflow-runs/{run_id}/workspace/file", response_model=BundleFileContent, tags=["workflow"])
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


@router.post("/v1/workflow-runs/{run_id}/nodes/execute", response_model=WorkflowRun, tags=["workflow"])
def execute_workflow_nodes(
    run_id: str,
    request: Request,
    start_node: str | None = None,
) -> WorkflowRun:
    """Re-execute the LangGraph node loop.

    Optional query param ``start_node`` (e.g. ``reviewer_code``) lets
    callers resume from a specific node instead of restarting from
    ``authoring_runtime``. Useful for unblocking a run that failed deep
    in the reviewer lane after the underlying check was fixed.
    """
    service = _workflow_service(request)
    try:
        return service.execute_langgraph_nodes(run_id, start_node=start_node)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        # LangGraph executor rejects unknown start_node values.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/v1/workflow-runs/{run_id}/grader-plans", response_model=TaskAgentGraderPlanCollection, tags=["workflow"])
def list_workflow_grader_plans(run_id: str, request: Request) -> TaskAgentGraderPlanCollection:
    service = _workflow_service(request)
    try:
        return service.list_task_agent_grader_plans(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/v1/workflow-runs/{run_id}/grader-plans/{deliverable_id}", response_model=DeliverableGraderPlan, tags=["workflow"])
def get_workflow_grader_plan(run_id: str, deliverable_id: str, request: Request) -> DeliverableGraderPlan:
    service = _workflow_service(request)
    try:
        return service.get_task_agent_grader_plan(run_id, deliverable_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/v1/workflow-runs/{run_id}/grade/{deliverable_id}", response_model=DeliverableGradeReport, tags=["workflow"])
def grade_workflow_submission(run_id: str, deliverable_id: str, submission: TaskAgentSubmission, request: Request) -> DeliverableGradeReport:
    service = _workflow_service(request)
    try:
        return service.grade_task_agent_run(run_id, deliverable_id, submission)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/v1/workflow-runs/{run_id}/grade-live/{deliverable_id}", response_model=LiveTaskAgentGradeReport, tags=["workflow"])
def grade_workflow_submission_live(
    run_id: str,
    deliverable_id: str,
    payload: LiveGradeTaskAgentRequest,
    request: Request,
) -> LiveTaskAgentGradeReport:
    service = _workflow_service(request)
    try:
        return service.grade_task_agent_run_live(run_id, deliverable_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TaskAgentRunnerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.put("/v1/workflow-runs/{run_id}/task-agent-spec", response_model=WorkflowRun, tags=["workflow"])
def update_task_agent_workflow_spec(run_id: str, spec: TaskAgentServiceSpec, request: Request) -> WorkflowRun:
    service = _workflow_service(request)
    try:
        return service.update_task_agent_spec(run_id, spec)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/workflow-runs/{run_id}/decisions", response_model=WorkflowRun, tags=["workflow"])
def decide_workflow_gate(run_id: str, decision: GateDecisionRequest, request: Request) -> WorkflowRun:
    service = _workflow_service(request)
    try:
        return service.apply_gate_decision(run_id, decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.") from exc
    except WorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/workflow-runs/{run_id}/materialize", response_model=WorkflowRun, tags=["workflow"])
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


@router.get("/v1/course-runs", response_model=CourseRunList, tags=["course"])
def list_course_runs(request: Request) -> CourseRunList:
    return _course_workflow_service(request).list_runs()


@router.get("/v1/course-generation/status", response_model=CourseGenerationStatus, tags=["course"])
def get_course_generation_status(request: Request) -> CourseGenerationStatus:
    return _course_generation_service(request).status()


@router.get("/v1/creator-assets", response_model=CreatorAssetList, tags=["course"])
def list_creator_assets(request: Request) -> CreatorAssetList:
    return _creator_asset_service(request).list_assets()


@router.post("/v1/creator-assets", response_model=CreatorAssetRecord, tags=["course"])
def create_creator_asset(
    payload: CreateCreatorAssetRequest,
    request: Request,
) -> CreatorAssetRecord:
    try:
        return _creator_asset_service(request).create_asset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/v1/creator-assets/{asset_id}", response_model=DeleteCreatorAssetResult, tags=["course"])
def delete_creator_asset(asset_id: str, request: Request) -> DeleteCreatorAssetResult:
    deleted = _creator_asset_service(request).delete_asset(asset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown creator asset '{asset_id}'.")
    return DeleteCreatorAssetResult(asset_id=asset_id)


@router.post("/v1/course-generation/suggest-outcomes", response_model=SuggestLearningOutcomesResponse, tags=["course"])
def suggest_learning_outcomes(
    payload: SuggestLearningOutcomesRequest,
    request: Request,
) -> SuggestLearningOutcomesResponse:
    try:
        return _course_generation_service(request).suggest_learning_outcomes(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-generation/creator-plan", response_model=GenerateCreatorCoursePlanResponse, tags=["course"])
def generate_creator_course_plan(
    payload: GenerateCreatorCoursePlanRequest,
    request: Request,
) -> GenerateCreatorCoursePlanResponse:
    try:
        return _course_generation_service(request).generate_creator_plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-generation/creator-stack-contract", response_model=RecommendCreatorStackContractResponse, tags=["course"])
def recommend_creator_stack_contract(
    payload: RecommendCreatorStackContractRequest,
    request: Request,
) -> RecommendCreatorStackContractResponse:
    try:
        return _course_generation_service(request).recommend_creator_stack_contract(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/generate", response_model=GenerateCourseFromBriefResponse, tags=["course"])
def generate_course_run_from_brief(
    payload: GenerateCourseFromBriefRequest,
    request: Request,
) -> GenerateCourseFromBriefResponse:
    try:
        return _course_generation_service(request).generate_course_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/from-creator-plan", response_model=CourseRun, tags=["course"])
def create_course_run_from_creator_plan(
    payload: CreateCourseFromCreatorPlanRequest,
    request: Request,
) -> CourseRun:
    try:
        return _course_generation_service(request).create_course_run_from_creator_plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/from-creator-plan-async", response_model=QueueCourseGenerationResponse, tags=["course"])
def queue_course_run_from_creator_plan(
    payload: CreateCourseFromCreatorPlanRequest,
    request: Request,
) -> QueueCourseGenerationResponse:
    try:
        return _course_generation_service(request).queue_course_run_from_creator_plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/generate-async", response_model=QueueCourseGenerationResponse, tags=["course"])
def queue_course_run_from_brief(
    payload: GenerateCourseFromBriefRequest,
    request: Request,
) -> QueueCourseGenerationResponse:
    try:
        return _course_generation_service(request).queue_course_run_generation(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/course-runs/reset-local", response_model=LocalDraftResetResult, tags=["course"])
def reset_local_course_state(request: Request) -> LocalDraftResetResult:
    return _course_workflow_service(request).reset_local_state()


@router.post("/v1/course-runs", response_model=CourseRun, tags=["course"])
def create_course_run(payload: CreateCourseRunRequest, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.create_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/v1/course-runs/{course_run_id}", response_model=CourseRun, tags=["course"])
def get_course_run(course_run_id: str, request: Request) -> CourseRun:
    run = _course_workflow_service(request).get_run(course_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.")
    return run


@router.get("/v1/course-runs/{course_run_id}/events", response_model=list[CourseEvent], tags=["course"])
def get_course_events(course_run_id: str, request: Request) -> list[CourseEvent]:
    service = _course_workflow_service(request)
    run = service.get_run(course_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.")
    return service.list_events(course_run_id)


@router.get("/v1/course-runs/{course_run_id}/timeline", response_model=DraftTimelineResponse, tags=["course"])
def get_course_timeline(course_run_id: str, request: Request) -> DraftTimelineResponse:
    service = _course_workflow_service(request)
    try:
        return service.timeline(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.get("/v1/course-runs/{course_run_id}/review", response_model=CourseReviewReport, tags=["course"])
def review_course_run(course_run_id: str, request: Request) -> CourseReviewReport:
    service = _course_workflow_service(request)
    try:
        return service.review_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.get("/v1/course-runs/{course_run_id}/creator-view", response_model=CreatorTestingView, tags=["course"])
def get_creator_testing_view(course_run_id: str, request: Request) -> CreatorTestingView:
    service = _course_workflow_service(request)
    try:
        return service.creator_view(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.get("/v1/course-runs/{course_run_id}/feedback", response_model=CreatorFeedbackList, tags=["course"])
def list_creator_feedback(course_run_id: str, request: Request) -> CreatorFeedbackList:
    service = _course_workflow_service(request)
    try:
        return service.list_creator_feedback(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.post("/v1/course-runs/{course_run_id}/feedback", response_model=CreatorFeedbackRecord, tags=["course"])
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


@router.get("/v1/course-runs/{course_run_id}/learner-eval", response_model=LearnerCourseEvaluationReport, tags=["course"])
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


@router.post("/v1/course-runs/{course_run_id}/learner-eval", response_model=LearnerCourseEvaluationReport, tags=["course"])
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


@router.get("/v1/course-runs/{course_run_id}/published-versions", response_model=PublishedVersionList, tags=["course"])
def list_published_versions(course_run_id: str, request: Request) -> PublishedVersionList:
    service = _course_workflow_service(request)
    try:
        return service.list_published_versions(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.post("/v1/course-runs/{course_run_id}/sync", response_model=CourseRun, tags=["course"])
def sync_course_run(course_run_id: str, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.sync_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc


@router.post("/v1/course-runs/{course_run_id}/publish", response_model=CourseRun, tags=["course"])
def publish_course_run(course_run_id: str, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.publish_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/publish-async", response_model=QueueCourseOperationResponse, tags=["course"])
def queue_publish_course_run(course_run_id: str, request: Request) -> QueueCourseOperationResponse:
    service = _course_workflow_service(request)
    try:
        return service.queue_publish_run(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/create-revision", response_model=CourseRun, tags=["course"])
def create_course_revision(course_run_id: str, request: Request) -> CourseRun:
    service = _course_workflow_service(request)
    try:
        return service.create_revision(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/create-revision-async", response_model=QueueCourseRevisionResponse, tags=["course"])
def queue_course_revision(course_run_id: str, request: Request) -> QueueCourseRevisionResponse:
    service = _course_workflow_service(request)
    try:
        return service.queue_revision(course_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.") from exc
    except CourseWorkflowConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/course-runs/{course_run_id}/materialize", response_model=CourseRun, tags=["course"])
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


@router.post("/v1/course-runs/{course_run_id}/materialize-async", response_model=QueueCourseOperationResponse, tags=["course"])
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


@router.get("/v1/course-runs/{course_run_id}/bundle", response_model=MaterializedBundle, tags=["course"])
def get_course_bundle(course_run_id: str, request: Request) -> MaterializedBundle:
    run = _course_workflow_service(request).get_run(course_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{course_run_id}'.")
    if run.materialized_bundle is None:
        raise HTTPException(status_code=404, detail="This course run does not have a materialized bundle yet.")
    return run.materialized_bundle


@router.get("/v1/course-runs/{course_run_id}/bundle/file", response_model=BundleFileContent, tags=["course"])
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


@router.get("/v1/workflow-runs/{run_id}/bundle", response_model=MaterializedBundle, tags=["workflow"])
def get_workflow_bundle(run_id: str, request: Request) -> MaterializedBundle:
    service = _workflow_service(request)
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown workflow run '{run_id}'.")
    if run.artifacts.materialized_bundle is None:
        raise HTTPException(status_code=404, detail="This run does not have a materialized bundle yet.")
    return run.artifacts.materialized_bundle


@router.get("/v1/workflow-runs/{run_id}/bundle/file", response_model=BundleFileContent, tags=["workflow"])
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


@router.get("/v1/lms/catalog", response_model=PublishedCourseCatalog, tags=["lms"])
def list_lms_catalog(request: Request) -> PublishedCourseCatalog:
    return _lms_service(request).list_catalog()


@router.get("/v1/lms/enrollments", response_model=LearnerEnrollmentList, tags=["lms"])
def list_lms_enrollments(
    request: Request,
    learner_id: str = Query("local-learner", description="Local learner identity for the LMS prototype."),
) -> LearnerEnrollmentList:
    return _lms_service(request).list_enrollments(learner_id=learner_id)


@router.post("/v1/lms/enrollments", response_model=LearnerEnrollment, tags=["lms"])
def create_lms_enrollment(payload: CreateEnrollmentRequest, request: Request) -> LearnerEnrollment:
    service = _lms_service(request)
    try:
        return service.enroll(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown course run '{payload.course_run_id}'.") from exc
    except LMSConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/v1/lms/enrollments/{enrollment_id}", response_model=LearnerEnrollment, tags=["lms"])
def get_lms_enrollment(enrollment_id: str, request: Request) -> LearnerEnrollment:
    service = _lms_service(request)
    try:
        return service.get_enrollment(enrollment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc


@router.get("/v1/lms/enrollments/{enrollment_id}/experience", response_model=LearnerDeliverableExperience, tags=["lms"])
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


@router.get("/v1/lms/enrollments/{enrollment_id}/learner-view", response_model=LearnerTestingView, tags=["lms"])
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


@router.get("/v1/lms/enrollments/{enrollment_id}/feedback", response_model=LearnerFeedbackList, tags=["lms"])
def list_lms_feedback(enrollment_id: str, request: Request) -> LearnerFeedbackList:
    service = _lms_service(request)
    try:
        return service.list_feedback(enrollment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown enrollment '{enrollment_id}'.") from exc


@router.post("/v1/lms/enrollments/{enrollment_id}/feedback", response_model=LearnerFeedbackRecord, tags=["lms"])
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


@router.post("/v1/lms/enrollments/{enrollment_id}/workspace", response_model=LearnerEnrollment, tags=["lms"])
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


@router.get("/v1/lms/enrollments/{enrollment_id}/workspace/files", response_model=LearnerWorkspaceFileList, tags=["lms"])
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


@router.get("/v1/lms/enrollments/{enrollment_id}/workspace/file", response_model=LearnerWorkspaceFileContent, tags=["lms"])
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


@router.put("/v1/lms/enrollments/{enrollment_id}/workspace/file", response_model=LearnerWorkspaceFileWriteResult, tags=["lms"])
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


@router.post("/v1/lms/enrollments/{enrollment_id}/submit", response_model=LearnerDeliverableExperience, tags=["lms"])
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
