from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.course_artifact_materializer import CourseArtifactMaterializer
from app.services.course_generation_service import CourseGenerationService
from app.services.course_workflow_service import CourseWorkflowService
from app.services.creator_asset_service import CreatorAssetService
from app.services.dashboard_page import build_dashboard_state, render_author_dashboard
from app.services.draft_timeline_page import build_draft_timeline_state, render_draft_timeline_page
from app.services.docs_page import build_docs_state, render_docs_page
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
from app.services.lms_page import build_lms_state, render_lms_courses_page, render_lms_home
from app.services.lms_service import LMSService
from app.services.publish_learner_certification_service import PublishLearnerCertificationService
from app.services.learner_studio_service import LearnerStudioService
from app.services.openai_learner_feedback import OpenAILearnerFeedbackService
from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService
from app.services.openai_task_agent_authoring import OpenAITaskAgentAuthoringService
from app.services.openai_test_script_authoring import OpenAITestScriptAuthoringService
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


def _ensure_lms_service(app: FastAPI) -> LMSService:
    store = getattr(getattr(app.state, "workflow_service", None), "store", None) or SQLiteWorkflowStore()
    if (not hasattr(app.state, "creator_asset_service")) or app.state.creator_asset_service.store is not store:
        app.state.creator_asset_service = CreatorAssetService(store)
    if not hasattr(app.state, "learner_studio_service"):
        app.state.learner_studio_service = LearnerStudioService(
            runner=getattr(app.state, "task_agent_blackbox_runner", TaskAgentBlackBoxRunner()),
        )
    if not hasattr(app.state, "learner_feedback_service"):
        app.state.learner_feedback_service = OpenAILearnerFeedbackService(
            env_file=os.environ.get("COURSE_GEN_OPENAI_ENV_FILE"),
        )
    if not hasattr(app.state, "workflow_service"):
        app.state.workflow_service = WorkflowService(
            store,
            ArtifactMaterializer(creator_asset_service=app.state.creator_asset_service),
            getattr(app.state, "task_agent_blackbox_runner", TaskAgentBlackBoxRunner()),
            getattr(app.state, "assignment_node_runtime", None),
            getattr(app.state, "task_agent_authoring_service", OpenAITaskAgentAuthoringService(enabled=False)),
            getattr(app.state, "assignment_workspace_manager", AssignmentWorkspaceManager()),
        )
    if (not hasattr(app.state, "lms_service")) or app.state.lms_service.store is not app.state.workflow_service.store:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=app.state.learner_studio_service,
            learner_feedback_service=app.state.learner_feedback_service,
        )
    return app.state.lms_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = getattr(getattr(app.state, "workflow_service", None), "store", None) or SQLiteWorkflowStore()
    if (not hasattr(app.state, "creator_asset_service")) or app.state.creator_asset_service.store is not store:
        app.state.creator_asset_service = CreatorAssetService(store)
    if not hasattr(app.state, "assignment_workspace_manager"):
        app.state.assignment_workspace_manager = AssignmentWorkspaceManager()
    if not hasattr(app.state, "docker_sandbox_runner"):
        app.state.docker_sandbox_runner = DockerSandboxRunner(
            workspace_manager=app.state.assignment_workspace_manager
        )
    if not hasattr(app.state, "task_agent_workspace_authoring_service"):
        app.state.task_agent_workspace_authoring_service = TaskAgentWorkspaceAuthoringService(
            workspace_manager=app.state.assignment_workspace_manager,
            repo_authoring_service=OpenAIStarterRepoAuthoringService(
                enabled=True,
                env_file=os.environ.get("COURSE_GEN_OPENAI_ENV_FILE"),
            ),
        )
    if not hasattr(app.state, "task_agent_authoring_service"):
        app.state.task_agent_authoring_service = OpenAITaskAgentAuthoringService(
            env_file=os.environ.get("COURSE_GEN_OPENAI_ENV_FILE"),
        )
    if not hasattr(app.state, "test_script_authoring_service"):
        app.state.test_script_authoring_service = OpenAITestScriptAuthoringService(
            env_file=os.environ.get("COURSE_GEN_OPENAI_ENV_FILE"),
        )
    if not hasattr(app.state, "assignment_node_runtime"):
        app.state.assignment_node_runtime = LangGraphAssignmentGraph(
            app.state.docker_sandbox_runner,
            authoring_service=app.state.task_agent_authoring_service,
            test_authoring_service=app.state.test_script_authoring_service,
            workspace_authoring_service=app.state.task_agent_workspace_authoring_service,
        )
    if not hasattr(app.state, "task_agent_blackbox_runner"):
        app.state.task_agent_blackbox_runner = TaskAgentBlackBoxRunner()
    if not hasattr(app.state, "learner_studio_service"):
        app.state.learner_studio_service = LearnerStudioService(
            runner=app.state.task_agent_blackbox_runner,
        )
    if not hasattr(app.state, "learner_feedback_service"):
        app.state.learner_feedback_service = OpenAILearnerFeedbackService(
            env_file=os.environ.get("COURSE_GEN_OPENAI_ENV_FILE"),
        )
    if not hasattr(app.state, "workflow_service"):
        app.state.workflow_service = WorkflowService(
            store,
            ArtifactMaterializer(creator_asset_service=app.state.creator_asset_service),
            app.state.task_agent_blackbox_runner,
            app.state.assignment_node_runtime,
            app.state.task_agent_authoring_service,
            app.state.assignment_workspace_manager,
        )
    if not hasattr(app.state, "course_workflow_service"):
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(),
            publish_certification_service=PublishLearnerCertificationService(
                learner_studio_service=app.state.learner_studio_service,
                enabled=True,
            ),
            creator_asset_service=app.state.creator_asset_service,
        )
    if not hasattr(app.state, "course_generation_service"):
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
        )
    if not hasattr(app.state, "lms_service"):
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=app.state.learner_studio_service,
            learner_feedback_service=app.state.learner_feedback_service,
        )
    yield

app = FastAPI(
    title="Course Gen Codex",
    version="0.1.0",
    summary="Archetype-driven assignment generation MVP for engineering projects.",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)
app.include_router(router)


@app.get("/create-course", tags=["system"], include_in_schema=False)
def create_course(request: Request) -> HTMLResponse:
    dashboard_state = build_dashboard_state(
        generation_status=request.app.state.course_generation_service.status(),
    )
    return HTMLResponse(render_author_dashboard(dashboard_state))


@app.get("/draft-timeline", tags=["system"], include_in_schema=False)
def draft_timeline(request: Request, draft: str | None = None) -> HTMLResponse:
    return HTMLResponse(
        render_draft_timeline_page(
            build_draft_timeline_state(draft_id=draft)
        )
    )


@app.get("/", tags=["system"], include_in_schema=False)
def root(request: Request) -> HTMLResponse:
    lms_state = build_lms_state(
        catalog=_ensure_lms_service(request.app).list_catalog(),
        enrollments=_ensure_lms_service(request.app).list_enrollments(),
    )
    return HTMLResponse(render_lms_home(lms_state))


@app.get("/courses", tags=["system"], include_in_schema=False)
def courses(request: Request) -> HTMLResponse:
    lms_state = build_lms_state(
        catalog=_ensure_lms_service(request.app).list_catalog(),
        enrollments=_ensure_lms_service(request.app).list_enrollments(),
    )
    return HTMLResponse(render_lms_courses_page(lms_state))


@app.get("/docs", tags=["system"], include_in_schema=False)
def docs(request: Request) -> HTMLResponse:
    docs_state = build_docs_state(openapi_schema=request.app.openapi())
    return HTMLResponse(render_docs_page(docs_state))
