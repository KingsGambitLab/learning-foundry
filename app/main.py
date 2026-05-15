from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.auth_routes import router as auth_router
from app.api.deps import current_user_optional
from app.api.routes import router
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.course_artifact_materializer import CourseArtifactMaterializer
from app.services.course_generation_service import CourseGenerationService
from app.services.course_workflow_service import CourseWorkflowService
from app.services.coursegen_logging import log_coursegen_event
from app.services.creator_asset_service import CreatorAssetService
from app.services.dashboard_page import build_dashboard_state, render_author_dashboard
from app.services.draft_timeline_page import build_draft_timeline_state, render_draft_timeline_page
from app.services.docs_page import build_docs_state, render_docs_page
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.lms_page import build_lms_state, render_lms_courses_page, render_lms_home
from app.services.lms_service import LMSService
from app.services.publish_learner_certification_service import PublishLearnerCertificationService
from app.services.learner_studio_service import LearnerStudioService
from app.services.openai_learner_feedback import OpenAILearnerFeedbackService
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.auth_session import SessionService
from app.services.workflow_service import WorkflowService
from app.storage.postgres_store import PostgresWorkflowStore


def _ensure_lms_service(app: FastAPI) -> LMSService:
    store = getattr(getattr(app.state, "workflow_service", None), "store", None) or PostgresWorkflowStore()
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
    store = getattr(getattr(app.state, "workflow_service", None), "store", None) or PostgresWorkflowStore()
    if (not hasattr(app.state, "creator_asset_service")) or app.state.creator_asset_service.store is not store:
        app.state.creator_asset_service = CreatorAssetService(store)
    if not hasattr(app.state, "assignment_workspace_manager"):
        app.state.assignment_workspace_manager = AssignmentWorkspaceManager()
    if not hasattr(app.state, "docker_sandbox_runner"):
        app.state.docker_sandbox_runner = DockerSandboxRunner(
            workspace_manager=app.state.assignment_workspace_manager
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
            app.state.assignment_workspace_manager,
        )
    if not hasattr(app.state, "session_service"):
        app.state.session_service = SessionService(store=app.state.workflow_service.store)
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

    # Reconcile any course_runs whose `active_operation` survived a prior
    # process restart. Background tasks don't survive uvicorn shutdown, so
    # any non-null `active_operation` at this point is by definition stale.
    # Without this, publish/revise/materialize endpoints return "already
    # busy with `<operation>`" until the row is manually patched.
    try:
        reconciled = app.state.course_workflow_service.reconcile_stale_active_operations()
        if reconciled:
            log_coursegen_event(
                "course_active_operations_reconciled_on_startup",
                count=len(reconciled),
                course_run_ids=reconciled,
            )
    except Exception as exc:  # noqa: BLE001
        log_coursegen_event(
            "course_active_operations_reconciliation_failed",
            error=str(exc),
        )

    # Reconcile learner_workspace_sessions whose backing editor container
    # was killed by the prior process shutdown. Without this, the web UI
    # keeps showing the editor URL as active and the learner gets a 404.
    try:
        reconciled_sessions = app.state.learner_studio_service.reconcile_stale_sessions(store)
        if reconciled_sessions:
            log_coursegen_event(
                "learner_workspace_sessions_reconciled_on_startup",
                count=len(reconciled_sessions),
                session_ids=reconciled_sessions,
            )
    except Exception as exc:  # noqa: BLE001
        log_coursegen_event(
            "learner_workspace_sessions_reconciliation_failed",
            error=str(exc),
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
app.include_router(auth_router)

templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


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
    user = current_user_optional(request)
    svc = _ensure_lms_service(request.app)
    lms_state = build_lms_state(
        catalog=svc.list_catalog(),
        enrollments=svc.list_enrollments(learner_id=str(user.id)) if user is not None else svc.list_enrollments(learner_id=""),
    )
    return HTMLResponse(render_lms_home(lms_state))


@app.get("/courses", tags=["system"], include_in_schema=False)
def courses(request: Request) -> HTMLResponse:
    user = current_user_optional(request)
    svc = _ensure_lms_service(request.app)
    lms_state = build_lms_state(
        catalog=svc.list_catalog(),
        enrollments=svc.list_enrollments(learner_id=str(user.id)) if user is not None else svc.list_enrollments(learner_id=""),
    )
    return HTMLResponse(render_lms_courses_page(lms_state))


@app.get("/login", response_class=HTMLResponse, tags=["system"], include_in_schema=False)
def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {})


@app.get("/register", response_class=HTMLResponse, tags=["system"], include_in_schema=False)
def register_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "register.html", {})


@app.get("/lms/courses/{course_run_id}", tags=["system"], include_in_schema=False)
def lms_course_detail(course_run_id: str, request: Request) -> HTMLResponse:
    """Outcome-mode learner-preview page.

    The legacy LMS flow assumes a ``publish_snapshot`` with a
    ``learner_package`` (deliverables, starter readme, ...). Outcome-mode
    courses don't (yet) emit a snapshot — ``node_publish`` only writes
    the README + course_spec + grader runner. This route renders a thin
    page that surfaces what the published outcome bundle DOES carry:
    title, goal, capability flags, endpoint contracts, quality bars,
    and the on-disk workspace path so the operator can inspect the
    artifacts.

    Returns 404 only when the course_run is unknown.
    """
    store = request.app.state.workflow_service.store
    course_run = store.get_course_run(course_run_id)
    if course_run is None:
        return HTMLResponse(
            f"<h1>404 — Course not found</h1><p>No course_run with id "
            f"<code>{course_run_id}</code>.</p>",
            status_code=404,
        )
    outcome_state = (course_run.payload_json or {}).get("outcome_state") or {}
    spec = outcome_state.get("spec") or {}
    workspace_root = outcome_state.get("workspace_root") or "?"
    endpoints = spec.get("endpoints") or []
    quality_bars = spec.get("quality_bars") or []
    learning_path = spec.get("learning_path") or []
    bench = spec.get("benchmark") or {}
    caps = spec.get("capabilities") or {}

    def _esc(value: object) -> str:
        import html as _html
        return _html.escape(str(value))

    endpoints_html = "".join(
        f"<li><code>{_esc(ep.get('method'))} {_esc(ep.get('path'))}</code> — "
        f"{_esc(ep.get('description'))}</li>"
        for ep in endpoints
    )
    bars_html = "".join(
        f"<li><code>{_esc(b.get('id'))}</code> {_esc(b.get('threshold'))} "
        f"via <em>{_esc(b.get('judged_by'))}</em> "
        f"(aggregation: {_esc(b.get('aggregation') or 'ratio')}, "
        f"n={_esc(b.get('sample_size'))})<br>"
        f"<small>{_esc(b.get('metric_description'))}</small></li>"
        for b in quality_bars
    )
    hints_html = "".join(
        f"<li><code>{_esc(h.get('on_metric_fail'))}</code>: "
        f"{_esc(h.get('hint'))}</li>"
        for h in learning_path
    )
    benchmark_html = (
        f"<dt>Benchmark</dt><dd><code>{_esc(bench.get('kind'))}</code> "
        f"&middot; <code>{_esc(bench.get('dataset'))}</code> "
        f"(split=<code>{_esc(bench.get('use_split'))}</code>, "
        f"max_queries=<code>{_esc(bench.get('max_queries'))}</code>)</dd>"
        if bench
        else ""
    )
    body = (
        "<!doctype html><html><head>"
        f"<title>{_esc(spec.get('title') or course_run.title or course_run_id)}</title>"
        "<style>"
        "body{font:14px/1.5 system-ui;max-width:880px;margin:40px auto;color:#0f1419}"
        "h1{margin-bottom:.2em}h2{margin-top:1.6em;color:#0a4b7c}"
        "dt{font-weight:600;margin-top:.6em}dd{margin-left:1em}"
        "code{background:#f2f4f7;padding:.1em .35em;border-radius:3px}"
        "li{margin:.4em 0}small{color:#566}"
        "a{color:#0a4b7c}.muted{color:#999}"
        "</style></head><body>"
        f"<p><a href='/courses'>&larr; All courses</a></p>"
        f"<h1>{_esc(spec.get('title') or course_run.title or course_run_id)}</h1>"
        f"<p>{_esc(spec.get('goal') or course_run.summary or '')}</p>"
        f"<dl>"
        f"<dt>Course run</dt><dd><code>{_esc(course_run_id)}</code> "
        f"(stage: <code>{_esc(course_run.stage)}</code>, "
        f"status: <code>{_esc(course_run.status)}</code>)</dd>"
        f"<dt>Starter type</dt><dd><code>{_esc(spec.get('starter_type'))}</code></dd>"
        f"<dt>Oracle source</dt><dd><code>{_esc(spec.get('oracle_source'))}</code></dd>"
        f"{benchmark_html}"
        f"<dt>Capabilities</dt><dd>{_esc(caps)}</dd>"
        f"<dt>Workspace</dt><dd><code>{_esc(workspace_root)}</code></dd>"
        f"</dl>"
        f"<h2>Endpoint contracts</h2><ul>{endpoints_html}</ul>"
        f"<h2>Quality bars ({len(quality_bars)})</h2><ul>{bars_html}</ul>"
        f"<h2>Learning path</h2><ul>{hints_html or '<li class=muted>none</li>'}</ul>"
        f"<h2>Inspect raw state</h2>"
        f"<p><a href='/v1/course-runs/{_esc(course_run_id)}'>"
        f"GET /v1/course-runs/{_esc(course_run_id)}</a></p>"
        "</body></html>"
    )
    return HTMLResponse(body)


@app.get("/docs", tags=["system"], include_in_schema=False)
def docs(request: Request) -> HTMLResponse:
    docs_state = build_docs_state(openapi_schema=request.app.openapi())
    return HTMLResponse(render_docs_page(docs_state))
