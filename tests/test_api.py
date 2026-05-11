from __future__ import annotations

import copy
import json
import os
import py_compile
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.domain.ai import AIUsageSummary
from app.domain.course import (
    CreatorCourseSetupChoices,
    CreatorStackCatalog,
    CreatorStackCatalogOption,
    CourseAsyncOperation,
    CourseGenerationSource,
    CourseGenerationStatus,
    CreateCourseDeliverableRequest,
    GenerateCourseFromBriefRequest,
    GeneratedCoursePlan,
    RecommendCreatorStackContractResponse,
)
from app.domain.grading import (
    AssignmentGradeReport,
    GradeStatus,
    LearnerReviewGuidance,
    LiveAssignmentGradeReport,
    LiveTaskAgentGradeReport,
    DeliverableGradeReport,
    ReviewAreaGradeReport,
    TestGradeResult,
)
from app.domain.registry import PackageType, RiskClass
from app.domain.learner import LearnerWorkspaceScope, LearnerWorkspaceSession, LearnerWorkspaceSessionStatus
from app.domain.publish import LearnerCoursePackage, LearnerDeliverablePackage
from app.domain.sandbox import (
    DeliverableSandboxReport,
    SandboxAvailability,
    SandboxExecutionResult,
    SandboxExecutionStatus,
)
from app.main import app
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.assignment_design_inference import infer_assignment_design
from app.services.course_artifact_materializer import CourseArtifactMaterializer
from app.services.course_generation_service import CourseGenerationService
from app.services.course_workflow_service import CourseWorkflowService
from app.services.creator_asset_service import CreatorAssetService
from app.services.docker_sandbox_runner import DockerSandboxRunner
from app.services.examples import get_generic_project_submission
from app.services.intake_router import GenerationIntake
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
from app.services.generated_test_harness import BaselineValidationResult
from app.services.lms_service import LMSService
from app.services.openai_course_planner import OpenAICoursePlanner
from app.services.openai_learner_feedback import OpenAILearnerFeedbackService
from app.services.openai_repo_authoring import RepoAuthoringResult, RepoAuthoringSource
from app.services.openai_test_script_authoring import (
    TestScriptAuthoringResult,
    TestScriptAuthoringSource,
)
from app.services.publish_learner_certification_service import PublishLearnerCertificationService
from app.services import openai_runtime_support
from app.services.openai_task_agent_authoring import (
    DeliverableCustomization,
    PublicCheckCustomization,
    OpenAITaskAgentAuthoringService,
    TaskAgentCustomization,
    TaskAgentAuthoringResult,
    TaskAgentAuthoringSource,
    TaskAgentAuthoringStatus,
)
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.task_agent_contract_surface import learner_editable_paths_for_deliverable
from app.services.task_agent_grader import grade_assignment_submission, grade_task_agent_submission
from app.services.learner_studio_service import LearnerStudioError
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.services.task_agent_starter_templates import (
    build_task_agent_starter_files,
    HIDDEN_MANIFEST_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
)
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore
from app.domain.workflow import (
    FailureContext,
    FailureContextSandboxSummary,
    MaterializeBundleRequest,
    ReviewerFinding,
    ReviewerFindingSeverity,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowNodeStatus,
)


def _design_spec(
    *,
    title: str,
    problem_statement: str,
    learning_outcomes: list[str],
    package_type: PackageType = PackageType.progressive_codebase_course,
):
    inferred = infer_assignment_design(
        title=title,
        problem_statement=problem_statement,
        learning_outcomes=learning_outcomes,
        package_type_hint=package_type,
    )
    assert inferred.design_spec is not None
    return inferred.design_spec


class FakeLivePlanner:
    def status(self) -> CourseGenerationStatus:
        return CourseGenerationStatus(
            provider="openai",
            available=True,
            source=CourseGenerationSource.openai_live,
            message="Ready to generate with fake OpenAI.",
            sdk_installed=True,
            api_key_present=True,
            model_id="gpt-5.4",
            env_file="/tmp/fake-openai.env",
        )

    def plan_course(self, request) -> tuple[GeneratedCoursePlan, CourseGenerationStatus, AIUsageSummary]:
        shared_design_spec = _design_spec(
            title=request.title or "Fake Live Planner Course",
            problem_statement=request.goal,
            learning_outcomes=request.learning_outcomes,
        )
        plan = GeneratedCoursePlan(
            title=request.title or "Fake Live Planner Course",
            summary=request.goal,
            package_type=PackageType.progressive_codebase_course,
            shared_design_spec=shared_design_spec,
            deliverables=[
                CreateCourseDeliverableRequest(
                    title="Live planning foundation",
                    summary="Generated from the fake live planner.",
                    learning_outcomes=request.learning_outcomes[:2],
                    design_spec=shared_design_spec,
                    domain_pack_hint="generic_project_example",
                ),
                CreateCourseDeliverableRequest(
                    title="Live planning production deliverable",
                    summary="Adds production controls and evaluation.",
                    learning_outcomes=request.learning_outcomes[:3],
                    design_spec=shared_design_spec.model_copy(update={"overlays": ["productionization_overlay"]}),
                    domain_pack_hint="generic_project_example",
                    overlays_hint=["productionization_overlay"],
                ),
            ],
            notes=["Built by the fake live planner test double."],
        )
        return (
            plan,
            self.status(),
            AIUsageSummary(
                request_count=1,
                input_tokens=1200,
                output_tokens=450,
                total_tokens=1650,
                estimated_cost_usd=0.006,
                models=["gpt-5.4"],
            ),
        )

    def suggest_learning_outcomes(self, request):
        return (
            [
                "Define the core system contract and learner-visible success criteria.",
                "Implement the key workflow with production-minded safeguards.",
                "Add observability or evaluation checks that make quality visible.",
                "Refine the system until it meets a realistic engineering bar.",
            ],
            self.status(),
            AIUsageSummary(
                request_count=1,
                input_tokens=900,
                output_tokens=180,
                total_tokens=1080,
                estimated_cost_usd=0.0029,
                models=["gpt-5.4"],
            ),
        )


class FakeMultilineOutcomePlanner(FakeLivePlanner):
    def suggest_learning_outcomes(self, request):
        return (
            [
                "- Model the booking workflow clearly.\n- Handle concurrent reservations safely.",
                "Use caching carefully for read-heavy traffic.",
            ],
            self.status(),
            AIUsageSummary(
                request_count=1,
                input_tokens=700,
                output_tokens=120,
                total_tokens=820,
                estimated_cost_usd=0.0021,
                models=["gpt-5.4"],
            ),
        )


class FakeStackCatalogService:
    def describe_choices(self, choices: CreatorCourseSetupChoices) -> RecommendCreatorStackContractResponse:
        creator_choices = choices.model_copy(
            update={
                "implementation_language": choices.implementation_language or "go",
                "language_version": choices.language_version or "1.26",
                "application_framework": choices.application_framework or "gin",
                "framework_version": choices.framework_version or "1.12.0",
                "package_manager": choices.package_manager or "go",
                "primary_database": choices.primary_database,
                "primary_database_version": choices.primary_database_version or ("18" if choices.primary_database else None),
                "cache_backend": choices.cache_backend,
                "cache_backend_version": choices.cache_backend_version or ("8" if choices.cache_backend else None),
            }
        )
        return RecommendCreatorStackContractResponse(
            creator_choices=creator_choices,
            catalog=CreatorStackCatalog(
                languages=[
                    CreatorStackCatalogOption(value="go", label="Go", source_url="https://go.dev/dl/?mode=json", recommended=True),
                ],
                frameworks_by_language={
                    "go": [
                        CreatorStackCatalogOption(
                            value="gin",
                            label="Gin",
                            source_url="https://pkg.go.dev/github.com/gin-gonic/gin",
                            recommended=True,
                        )
                    ]
                },
                package_managers_by_language={
                    "go": [
                        CreatorStackCatalogOption(value="go", label="go", recommended=True),
                    ]
                },
                databases=[
                    CreatorStackCatalogOption(
                        value="postgres",
                        label="PostgreSQL",
                        source_url="https://hub.docker.com/_/postgres",
                        recommended=True,
                    )
                ],
                caches=[
                    CreatorStackCatalogOption(
                        value="redis",
                        label="Redis",
                        source_url="https://hub.docker.com/_/redis",
                        recommended=True,
                    )
                ],
            ),
            language_versions=[
                CreatorStackCatalogOption(value="1.26", label="1.26", source_url="https://go.dev/dl/?mode=json", recommended=True),
            ],
            framework_versions=[
                CreatorStackCatalogOption(
                    value="1.12.0",
                    label="1.12.0",
                    source_url="https://pkg.go.dev/github.com/gin-gonic/gin",
                    recommended=True,
                )
            ],
            database_versions=[
                CreatorStackCatalogOption(
                    value="18",
                    label="18",
                    source_url="https://hub.docker.com/_/postgres",
                    recommended=True,
                )
            ],
            cache_versions=[
                CreatorStackCatalogOption(
                    value="8",
                    label="8",
                    source_url="https://hub.docker.com/_/redis",
                    recommended=True,
                )
            ],
            notes=["Creator approves the final stack contract before generation."],
        )


class FakeSandboxRunner:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.calls: list[str] = []

    def status(self) -> SandboxAvailability:
        return SandboxAvailability(
            available=True,
            message="Fake Docker sandbox is ready.",
            docker_version="test",
        )

    def execute(self, run) -> SandboxExecutionResult:
        self.calls.append(run.id)
        reports = []
        if run.artifacts.task_agent_spec is not None:
            for deliverable in run.artifacts.task_agent_spec.deliverables:
                reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=self.success,
                        runtime_succeeded=self.success,
                        health_status_code=200 if self.success else None,
                        stdout="sandbox ok" if self.success else "",
                        stderr="" if self.success else "sandbox failed",
                        error=None if self.success else "sandbox failed",
                    )
                )
        return SandboxExecutionResult(
            status=SandboxExecutionStatus.passed if self.success else SandboxExecutionStatus.failed,
            available=True,
            build_succeeded=self.success,
            run_succeeded=self.success,
            generated_at=datetime.now(UTC),
            duration_ms=5,
            workspace_root="/tmp/fake-sandbox",
            image_tag="fake-image",
            build_command=["docker", "build"],
            run_command=["docker", "run"],
            build_stdout="build ok" if self.success else "",
            build_stderr="" if self.success else "build failed",
            run_stdout='{"success": true}' if self.success else "",
            run_stderr="" if self.success else "run failed",
            deliverable_reports=reports,
            error=None if self.success else "sandbox failed",
        )


class FakeTaskAgentAuthoringService:
    def __init__(self) -> None:
        self.last_failure_context = None

    def status(self) -> TaskAgentAuthoringStatus:
        return TaskAgentAuthoringStatus(
            available=True,
            source=TaskAgentAuthoringSource.openai_live,
            message="Fake OpenAI authoring is ready.",
            sdk_installed=True,
            api_key_present=True,
            model_id="gpt-5.4",
            env_file="/tmp/fake-openai.env",
        )

    def generate_scaffold(self, *, title, summary, design_spec) -> TaskAgentAuthoringResult:
        spec, origin_template = build_task_agent_scaffold(
            title=title,
            summary=summary,
            design_spec=design_spec,
        )
        spec.deliverables[0].title = "OpenAI-authored foundation"
        spec.summary = f"{summary} Generated with fake OpenAI."
        return TaskAgentAuthoringResult(
            spec=spec,
            origin_template=f"openai_customized:{origin_template}",
            source=TaskAgentAuthoringSource.openai_live,
            notes=["Customized with fake OpenAI."],
            status=self.status(),
        )

    def revise_spec(
        self,
        *,
        spec,
        title,
        summary,
        package_type,
        domain_pack,
        risk_class,
        overlays,
        feedback,
        failure_context=None,
        origin_template=None,
    ) -> TaskAgentAuthoringResult:
        self.last_failure_context = failure_context
        revised = spec.model_copy(deep=True)
        revised.deliverables[0].title = f"Revised after feedback: {feedback[:32]}"
        revised.summary = f"{summary} Revised from human review feedback."
        return TaskAgentAuthoringResult(
            spec=revised,
            origin_template=f"openai_revision:{origin_template or 'task_agent_spec'}",
            source=TaskAgentAuthoringSource.openai_live,
            notes=[f"Revised from fake OpenAI using feedback: {feedback}"],
            status=self.status(),
        )


class _AlwaysValidBaselineVerifier:
    def verify_deliverable(self, **kwargs):  # noqa: ANN003
        return BaselineValidationResult(valid=True)


class FakeRepoAuthoringService:
    def author_workspace_repo(self, run, **kwargs):  # noqa: ANN003
        spec = run.artifacts.task_agent_spec
        workspace = run.artifacts.workspace_snapshot
        if spec is None or workspace is None:
            return run, RepoAuthoringResult(
                source=RepoAuthoringSource.unavailable,
                updated_files=[],
                usage=None,
                notes=[],
                message="fake repo authoring unavailable",
                available=False,
            )
        updated_files: list[str] = []
        public_root = Path(workspace.public_dir)
        for deliverable in spec.deliverables:
            starter_root = public_root / "starter" / deliverable.id
            manifest_path = starter_root / HIDDEN_MANIFEST_PATH
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            starter_surface = deliverable.learner_starter_surface
            editable_paths = (
                list(starter_surface.primary_editable_paths)
                if starter_surface is not None and starter_surface.primary_editable_paths
                else list(spec.runtime_dependencies.editable_files)
            )
            if not editable_paths:
                editable_paths = ["app.py"]
            for relative_path in editable_paths:
                target = starter_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                if relative_path.endswith(".py"):
                    target.write_text(
                        "\n".join(
                            [
                                "from fastapi import FastAPI",
                                "",
                                "app = FastAPI()",
                                "",
                                "@app.get('/health')",
                                "def health():",
                                "    return {'ok': True}",
                                "",
                            ]
                        ),
                        encoding="utf-8",
                    )
                else:
                    target.write_text("// fake authored repo file\n", encoding="utf-8")
                updated_files.append(str(target.relative_to(workspace.root_dir)))
            dockerfile = starter_root / "Dockerfile"
            dockerfile.write_text(
                "\n".join(
                    [
                        "FROM python:3.12-slim",
                        "WORKDIR /workspace",
                        "COPY . /workspace",
                        "RUN sh .coursegen/runtime/install.sh",
                        'CMD ["sh", ".coursegen/runtime/run.sh"]',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            install_script = starter_root / RUNTIME_INSTALL_SCRIPT_PATH
            install_script.parent.mkdir(parents=True, exist_ok=True)
            install_script.write_text("#!/usr/bin/env sh\nset -eu\nexit 0\n", encoding="utf-8")
            verify_script = starter_root / RUNTIME_VERIFY_SCRIPT_PATH
            verify_script.write_text("#!/usr/bin/env sh\nset -eu\nexit 0\n", encoding="utf-8")
            run_script = starter_root / RUNTIME_RUN_SCRIPT_PATH
            run_script.write_text(
                "#!/usr/bin/env sh\nset -eu\npython -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}\n",
                encoding="utf-8",
            )
            manifest["starter_repo_bundle"] = {
                "source": "openai_live",
                "generated_for_deliverable": deliverable.id,
                "authored_paths": editable_paths,
            }
            manifest["runtime_protocol_bundle"] = {
                "source": "openai_live",
                "generated_for_deliverable": deliverable.id,
                "authored_paths": [
                    "Dockerfile",
                    RUNTIME_INSTALL_SCRIPT_PATH,
                    RUNTIME_VERIFY_SCRIPT_PATH,
                    RUNTIME_RUN_SCRIPT_PATH,
                ],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            updated_files.extend(
                [
                    str(dockerfile.relative_to(workspace.root_dir)),
                    str(install_script.relative_to(workspace.root_dir)),
                    str(verify_script.relative_to(workspace.root_dir)),
                    str(run_script.relative_to(workspace.root_dir)),
                    str(manifest_path.relative_to(workspace.root_dir)),
                ]
            )
        return run, RepoAuthoringResult(
            source=RepoAuthoringSource.openai_live,
            updated_files=updated_files,
            usage=None,
            notes=["Fake repo authoring populated learner-owned repo files and runtime protocol."],
            message="fake repo authoring completed",
            available=True,
        )


class FakeTestScriptAuthoringService:
    def author_workspace_tests(self, run, **kwargs):  # noqa: ANN003
        spec = run.artifacts.task_agent_spec
        workspace = run.artifacts.workspace_snapshot
        if spec is None or workspace is None:
            return run, TestScriptAuthoringResult(
                source=TestScriptAuthoringSource.unavailable,
                updated_files=[],
                usage=None,
                notes=[],
                message="fake test authoring unavailable",
                available=False,
            )
        updated_files: list[str] = []
        public_root = Path(workspace.public_dir)
        passing_script = "\n".join(
            [
                "import json, os",
                "payload = {'summary': 'ok', 'tests': [{'id': 'fake', 'title': 'fake', 'status': 'passed', 'summary': 'ok', 'diagnostics': []}]}",
                "report_path = os.environ.get('REPORT_PATH')",
                "if report_path:",
                "    with open(report_path, 'w', encoding='utf-8') as fh:",
                "        json.dump(payload, fh)",
                "else:",
                "    print(json.dumps(payload))",
                "raise SystemExit(0)",
                "",
            ]
        )
        for deliverable in spec.deliverables:
            starter_root = public_root / "starter" / deliverable.id
            visible_path = starter_root / "checks" / "run_visible_checks.py"
            hidden_path = starter_root / ".coursegen" / "grader" / "run_hidden_checks.py"
            visible_path.parent.mkdir(parents=True, exist_ok=True)
            hidden_path.parent.mkdir(parents=True, exist_ok=True)
            visible_path.write_text(passing_script, encoding="utf-8")
            hidden_path.write_text(passing_script, encoding="utf-8")
            manifest_path = starter_root / HIDDEN_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["generated_test_scripts"] = {
                "source": "openai_live",
                "generated_for_deliverable": deliverable.id,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            updated_files.extend(
                [
                    str(visible_path.relative_to(workspace.root_dir)),
                    str(hidden_path.relative_to(workspace.root_dir)),
                    str(manifest_path.relative_to(workspace.root_dir)),
                ]
            )
        return run, TestScriptAuthoringResult(
            source=TestScriptAuthoringSource.openai_live,
            updated_files=updated_files,
            usage=None,
            notes=["Fake test authoring populated visible and hidden scripts."],
            message="fake test authoring completed",
            available=True,
        )


class WorkspaceCompileSandboxRunner(FakeSandboxRunner):
    def execute(self, run) -> SandboxExecutionResult:
        self.calls.append(run.id)
        workspace = run.artifacts.workspace_snapshot
        reports = []
        success = True
        workspace_root = workspace.public_dir if workspace is not None else "/tmp/missing-workspace"
        if run.artifacts.task_agent_spec is not None and workspace is not None:
            public_dir = Path(workspace.public_dir)
            for deliverable in run.artifacts.task_agent_spec.deliverables:
                editable_paths = learner_editable_paths_for_deliverable(run.artifacts.task_agent_spec, deliverable)
                compile_targets = [
                    public_dir / "starter" / deliverable.id / relative_path
                    for relative_path in editable_paths
                    if relative_path.endswith(".py")
                ]
                content_targets = [
                    public_dir / "starter" / deliverable.id / relative_path
                    for relative_path in editable_paths
                    if not relative_path.endswith(".py")
                ]
                missing_targets = [
                    str((public_dir / "starter" / deliverable.id / relative_path).relative_to(public_dir))
                    for relative_path in editable_paths
                    if not (public_dir / "starter" / deliverable.id / relative_path).exists()
                ]
                try:
                    if missing_targets:
                        raise FileNotFoundError(f"Missing editable files: {', '.join(missing_targets)}")
                    for target in compile_targets:
                        py_compile.compile(str(target), doraise=True)
                    for target in content_targets:
                        if "BROKEN_STARTER_SENTINEL" in target.read_text(encoding="utf-8"):
                            raise ValueError(f"Broken starter sentinel found in {target.relative_to(public_dir)}")
                    compile_succeeded = True
                    error = None
                except Exception as exc:
                    compile_succeeded = False
                    error = str(exc)
                    success = False
                reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=compile_succeeded,
                        runtime_succeeded=compile_succeeded,
                        health_status_code=200 if compile_succeeded else None,
                        stdout="workspace ok" if compile_succeeded else "",
                        stderr="" if compile_succeeded else error or "compile failed",
                        error=error,
                    )
                )
        return SandboxExecutionResult(
            status=SandboxExecutionStatus.passed if success else SandboxExecutionStatus.failed,
            available=True,
            build_succeeded=success,
            run_succeeded=success,
            generated_at=datetime.now(UTC),
            duration_ms=5,
            workspace_root=workspace_root,
            image_tag="fake-image",
            build_command=["docker", "build"],
            run_command=["docker", "run"],
            build_stdout="build ok" if success else "",
            build_stderr="" if success else "build failed",
            run_stdout='{"success": true}' if success else "",
            run_stderr="" if success else "run failed",
            deliverable_reports=reports,
            error=None if success else "workspace compile failed",
        )


class BrokenFirstWorkspaceAuthoringService(TaskAgentWorkspaceAuthoringService):
    def __init__(self, workspace_manager: AssignmentWorkspaceManager, repo_authoring_service=None) -> None:  # noqa: ANN001
        super().__init__(workspace_manager=workspace_manager, repo_authoring_service=repo_authoring_service)
        self.author_calls = 0

    def author_workspace(self, run):
        run, result = super().author_workspace(run)
        self.author_calls += 1
        if self.author_calls == 1 and run.artifacts.workspace_snapshot is not None:
            deliverable = run.artifacts.task_agent_spec.deliverables[0]
            editable_paths = learner_editable_paths_for_deliverable(run.artifacts.task_agent_spec, deliverable)
            if not editable_paths:
                return run, result
            broken_path = (
                Path(run.artifacts.workspace_snapshot.public_dir)
                / "starter"
                / deliverable.id
                / editable_paths[0]
            )
            broken_path.parent.mkdir(parents=True, exist_ok=True)
            broken_contents = "def broken(:\n" if broken_path.suffix == ".py" else "BROKEN_STARTER_SENTINEL\n"
            broken_path.write_text(broken_contents, encoding="utf-8")
            result.updated_files.append(str(broken_path.relative_to(run.artifacts.workspace_snapshot.root_dir)))
            result.message = "Injected a broken starter on the first authoring pass to exercise the repair loop."
        return run, result


class FakeLearnerStudioService:
    def launch_editor(
        self,
        *,
        enrollment_id: str,
        deliverable_id: str,
        workspace_root: str,
        scope: LearnerWorkspaceScope,
        existing_session: LearnerWorkspaceSession | None = None,
    ) -> LearnerWorkspaceSession:
        now = datetime.now(UTC)
        return LearnerWorkspaceSession(
            id=existing_session.id if existing_session is not None else "studio_test_session",
            enrollment_id=enrollment_id,
            deliverable_id=deliverable_id,
            scope=scope,
            created_at=existing_session.created_at if existing_session is not None else now,
            updated_at=now,
            status=LearnerWorkspaceSessionStatus.running,
            workspace_root=str(workspace_root),
            container_name="fake-learner-studio",
            host_port=18080,
            editor_url="http://127.0.0.1:18080/",
            image_name="fake-learner-studio:latest",
            notes=["Fake learner studio."],
        )

    def grade_workspace(self, *, workspace_root: str, spec, deliverable_id: str):
        submission = get_generic_project_submission()
        grade_report = grade_task_agent_submission(spec, deliverable_id, submission)
        return LiveTaskAgentGradeReport(
            base_url="http://127.0.0.1:18080",
            submission=submission,
            grade_report=grade_report,
        )

    def grade_assignment(self, *, workspace_root: str, spec):
        submission = get_generic_project_submission()
        assignment_report = grade_assignment_submission(spec, submission)
        return LiveAssignmentGradeReport(
            base_url="http://127.0.0.1:18080",
            submission=submission,
            assignment_report=assignment_report,
        )

    def stop_editor(self, session: LearnerWorkspaceSession | None) -> None:
        return None


class FakeFailingLearnerStudioService(FakeLearnerStudioService):
    def grade_assignment(self, *, workspace_root: str, spec):
        submission = get_generic_project_submission().model_copy(deep=True)
        submission.runs = submission.runs[:1]
        assignment_report = grade_assignment_submission(spec, submission)
        return LiveAssignmentGradeReport(
            base_url="http://127.0.0.1:18080",
            submission=submission,
            assignment_report=assignment_report,
        )


class FakeBrokenAppLearnerStudioService(FakeLearnerStudioService):
    def grade_assignment(self, *, workspace_root: str, spec):
        raise LearnerStudioError(
            "Timed out waiting for 'http://127.0.0.1:18080/health' to respond. Last error: [Errno 61] Connection refused\n\n"
            "Container logs:\nTraceback (most recent call last):\nImportError: broken generated app"
        )


class FakePlatformBrokenLearnerStudioService(FakeLearnerStudioService):
    def launch_editor(
        self,
        *,
        enrollment_id: str,
        deliverable_id: str,
        workspace_root: str,
        scope: LearnerWorkspaceScope,
        existing_session: LearnerWorkspaceSession | None = None,
    ) -> LearnerWorkspaceSession:
        raise LearnerStudioError("Could not start learner editor container.")


class FakeLearnerFeedbackService:
    def __init__(self) -> None:
        self.calls = 0

    def annotate_assignment_report(self, *, assignment_report, **_kwargs):
        self.calls += 1
        updated = []
        for area in assignment_report.review_areas:
            if area.grade_report.status == GradeStatus.failed:
                updated.append(
                    area.model_copy(
                        update={
                            "feedback": LearnerReviewGuidance(
                                strengths=["The supported cases are already behaving correctly."],
                                fundamental_gap="The unsupported query path is still returning the wrong contract.",
                                why_it_matters=["A grounded assistant needs to abstain cleanly when support is weak."],
                                likely_root_cause=["The abstention branch is not normalizing the final response."],
                                investigation_steps=[
                                    "Run the unsupported case locally and inspect the raw JSON response.",
                                    "Check whether citations are cleared when the assistant abstains.",
                                ],
                                learner_feedback="Focus on the unsupported-query path next. Positive cases are already in decent shape.",
                            ),
                        }
                    )
                )
            else:
                updated.append(area)
        return assignment_report.model_copy(update={"review_areas": updated})


class OpenAILearnerFeedbackServiceTests(unittest.TestCase):
    def test_sparse_llm_feedback_is_filled_from_deliverable_context(self) -> None:
        spec, _origin = build_task_agent_scaffold(
            title="Grounded docs assistant",
            summary="Answer from a visible corpus and abstain when support is weak.",
            design_spec=_design_spec(
                title="Grounded docs assistant",
                problem_statement="Build a grounded internal docs assistant with citations and abstention.",
                learning_outcomes=["retrieval", "grounded answers", "abstention"],
            ),
        )

        failed_deliverable = spec.deliverables[0]
        passed_deliverable = spec.deliverables[1]
        learner_package = LearnerCoursePackage(
            course_run_id="course_demo",
            title=spec.title,
            summary=spec.summary,
            package_type=spec.package_type,
            published_at=datetime.now(UTC),
            workspace_scope=LearnerWorkspaceScope.shared_course,
            project_brief_markdown="Build a grounded internal docs assistant that cites the visible corpus and abstains when support is weak.",
            deliverables=[
                LearnerDeliverablePackage(
                    deliverable_id=failed_deliverable.id,
                    title=failed_deliverable.title,
                    objective=failed_deliverable.objective,
                    deliverable_index=1,
                    learner_brief=failed_deliverable.learner_brief,
                    public_checks=failed_deliverable.public_checks,
                    content_markdown="Failing deliverable content",
                    starter_readme="Failing deliverable README",
                    learning_outcomes=failed_deliverable.learning_outcomes,
                    active_test_ids=[],
                    completion_rule="Pass the review checks for this deliverable.",
                    visible_files=spec.runtime_dependencies.editable_files,
                ),
                LearnerDeliverablePackage(
                    deliverable_id=passed_deliverable.id,
                    title=passed_deliverable.title,
                    objective=passed_deliverable.objective,
                    deliverable_index=2,
                    learner_brief=passed_deliverable.learner_brief,
                    public_checks=passed_deliverable.public_checks,
                    content_markdown="Passing deliverable content",
                    starter_readme="Passing deliverable README",
                    learning_outcomes=passed_deliverable.learning_outcomes,
                    active_test_ids=[],
                    completion_rule="Pass the review checks for this deliverable.",
                    visible_files=spec.runtime_dependencies.editable_files,
                ),
            ],
        )

        assignment_report = AssignmentGradeReport(
            total_tests=3,
            passed_tests=1,
            failed_tests=2,
            pass_rate=1 / 3,
            status=GradeStatus.failed,
            review_areas=[
                ReviewAreaGradeReport(
                    deliverable_id=failed_deliverable.id,
                    title=failed_deliverable.title,
                    objective=failed_deliverable.objective,
                    deliverable_index=1,
                    grade_report=DeliverableGradeReport(
                        deliverable_id=failed_deliverable.id,
                        total_tests=2,
                        passed_tests=0,
                        failed_tests=2,
                        pass_rate=0.0,
                        status=GradeStatus.failed,
                        results=[
                            TestGradeResult(
                                test_id="unsupported_query",
                                test_type="behavior",
                                kind="expected_output",
                                status=GradeStatus.failed,
                                score=0.0,
                                summary="Case 'unsupported_lunar_policy' output does not match expected subset",
                                diagnostics=[
                                    "Expected abstained=True and an empty citations list for unsupported questions.",
                                ],
                            ),
                            TestGradeResult(
                                test_id="answer_contract",
                                test_type="behavior",
                                kind="expected_output",
                                status=GradeStatus.failed,
                                score=0.0,
                                summary="Supported answer contract is incomplete",
                                diagnostics=[
                                    "The response shape still differs from the expected contract.",
                                ],
                            ),
                        ],
                    ),
                ),
                ReviewAreaGradeReport(
                    deliverable_id=passed_deliverable.id,
                    title=passed_deliverable.title,
                    objective=passed_deliverable.objective,
                    deliverable_index=2,
                    grade_report=DeliverableGradeReport(
                        deliverable_id=passed_deliverable.id,
                        total_tests=1,
                        passed_tests=1,
                        failed_tests=0,
                        pass_rate=1.0,
                        status=GradeStatus.passed,
                        results=[
                            TestGradeResult(
                                test_id="grounded_answer",
                                test_type="behavior",
                                kind="expected_output",
                                status=GradeStatus.passed,
                                score=1.0,
                                summary="Grounded answer cases passed",
                                diagnostics=[],
                            ),
                        ],
                    ),
                ),
            ],
        )

        class SparseResponsesClient:
            def __init__(self) -> None:
                self.responses = self

            def create(self, **_kwargs):
                return type(
                    "Response",
                    (),
                    {"output_text": json.dumps({"fundamental_gap": "Unsupported questions still return a confident answer."})},
                )()

        service = OpenAILearnerFeedbackService(
            client_factory=lambda **_kwargs: SparseResponsesClient(),
        )

        with tempfile.TemporaryDirectory() as workspace_dir, patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch.object(
            service,
            "_openai_sdk_available",
            return_value=True,
        ):
            (Path(workspace_dir) / "app.py").write_text("print('hello')\n", encoding="utf-8")
            updated = service.annotate_assignment_report(
                project_brief_markdown=learner_package.project_brief_markdown,
                learner_package=learner_package,
                assignment_report=assignment_report,
                workspace_root=workspace_dir,
                spec=spec,
            )

        failed_feedback = updated.review_areas[0].feedback
        self.assertIsNotNone(failed_feedback)
        assert failed_feedback is not None
        self.assertTrue(failed_feedback.strengths)
        self.assertTrue(failed_feedback.why_it_matters)
        self.assertTrue(failed_feedback.likely_root_cause)
        self.assertTrue(failed_feedback.investigation_steps)
        self.assertTrue(failed_feedback.learner_feedback)
        self.assertIn("Unsupported questions still return a confident answer.", failed_feedback.fundamental_gap)
        self.assertTrue(any("passing" in item.lower() for item in failed_feedback.strengths))
        self.assertTrue(any("visible check" in item.lower() or "response" in item.lower() for item in failed_feedback.investigation_steps))


class CourseGenCodexApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        store = SQLiteWorkflowStore(db_path=f"{self.temp_dir.name}/test.db")
        self.fake_sandbox_runner = FakeSandboxRunner()
        self.workspace_manager = AssignmentWorkspaceManager(base_dir=f"{self.temp_dir.name}/workspaces")
        self.fake_repo_authoring_service = FakeRepoAuthoringService()
        self.fake_test_authoring_service = FakeTestScriptAuthoringService()
        self.workspace_authoring_service = TaskAgentWorkspaceAuthoringService(
            self.workspace_manager,
            repo_authoring_service=self.fake_repo_authoring_service,
        )
        self.disabled_authoring_service = OpenAITaskAgentAuthoringService(enabled=False)
        self.creator_asset_service = CreatorAssetService(
            store,
            base_dir=f"{self.temp_dir.name}/creator-assets",
        )
        app.state.docker_sandbox_runner = self.fake_sandbox_runner
        app.state.task_agent_workspace_authoring_service = self.workspace_authoring_service
        app.state.test_script_authoring_service = self.fake_test_authoring_service
        app.state.assignment_node_runtime = LangGraphAssignmentGraph(
            self.fake_sandbox_runner,
            workspace_authoring_service=self.workspace_authoring_service,
            test_authoring_service=self.fake_test_authoring_service,
            baseline_verifier=_AlwaysValidBaselineVerifier(),
        )
        app.state.task_agent_blackbox_runner = TaskAgentBlackBoxRunner()
        app.state.learner_feedback_service = OpenAILearnerFeedbackService(enabled=False)
        app.state.task_agent_authoring_service = self.disabled_authoring_service
        app.state.assignment_workspace_manager = self.workspace_manager
        app.state.creator_asset_service = self.creator_asset_service
        app.state.workflow_service = WorkflowService(
            store,
            ArtifactMaterializer(
                base_dir=f"{self.temp_dir.name}/generated",
                creator_asset_service=self.creator_asset_service,
            ),
            app.state.task_agent_blackbox_runner,
            app.state.assignment_node_runtime,
            app.state.task_agent_authoring_service,
            app.state.assignment_workspace_manager,
        )
        app.state.course_workflow_service = CourseWorkflowService(
            store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            creator_asset_service=self.creator_asset_service,
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def test_typescript_runtime_plan_drives_starter_files(self) -> None:
        inferred = infer_assignment_design(
            title="Build a feature flag control plane",
            problem_statement=(
                "Build a feature flag service using TypeScript, NestJS 11, MongoDB 7, and pnpm. "
                "Support gradual rollouts, targeting rules, and audit logs."
            ),
            learning_outcomes=["Ship a stateful control-plane backend."],
            implementation_language="typescript",
            application_framework="nestjs",
            primary_database="mongodb",
            tech_stack=["Node 22", "NestJS 11", "MongoDB 7", "pnpm"],
        )
        assert inferred.design_spec is not None

        spec, _origin_template = build_task_agent_scaffold(
            title="Build a feature flag control plane",
            summary="Feature flag runtime",
            design_spec=inferred.design_spec,
        )
        starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
        starter_manifest = json.loads(starter_files["starter_manifest.json"])
        package_json = json.loads(starter_files["package.json"])

        self.assertEqual(spec.runtime_dependencies.editable_files, ["src/main.ts"])
        self.assertIn("src/main.ts", starter_files)
        self.assertIn("package.json", starter_files)
        self.assertIn("tsconfig.json", starter_files)
        self.assertEqual(starter_manifest["entrypoint_path"], "src/main.ts")
        self.assertEqual(starter_manifest["preview_command"], "pnpm start:dev")
        self.assertEqual(starter_manifest["runtime_dependencies"]["editable_files"], ["src/main.ts"])
        self.assertEqual(package_json["scripts"]["start:dev"], "tsx src/main.ts")
        self.assertIn("@nestjs/core", package_json["dependencies"])

    def _install_mock_blackbox_runner(self) -> None:
        reference_submission = get_generic_project_submission().model_dump(mode="json")
        reference_runs = {run["run_id"]: run for run in reference_submission["runs"]}
        runtime_runs: dict[str, dict] = {}

        def response(payload: dict, status_code: int = 200) -> httpx.Response:
            return httpx.Response(status_code=status_code, json=payload)

        def response_shape(run: dict) -> dict:
            return {
                "output": run.get("output", {}),
                "trace_events": run.get("trace_events", []),
                "step_count": run.get("step_count", 0),
                "latency_ms": run.get("latency_ms", 0),
                "cost_usd": run.get("cost_usd", 0.0),
                "tool_calls": run.get("tool_calls", []),
                "approvals": run.get("approvals", []),
                "escalations": run.get("escalations", []),
                "failure_injections": run.get("failure_injections", []),
                "fallback_actions": run.get("fallback_actions", []),
                "resumed_after_pause": run.get("resumed_after_pause", False),
                "success": run.get("success", True),
                "quality_score": run.get("quality_score"),
                "notes": run.get("notes", []),
            }

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            payload = json.loads(request.content.decode() or "{}") if request.content else {}

            if request.method == "POST" and path == "/run":
                ticket_id = payload.get("ticket_id")
                dry_run = bool(payload.get("dry_run", False))

                if ticket_id == "T-100" and dry_run:
                    run = copy.deepcopy(reference_runs["run-billing-dry-001"])
                    run["run_id"] = "mock-billing-dry"
                    run["status"] = "completed"
                    runtime_runs[run["run_id"]] = run
                    return response({"run_id": run["run_id"], "status": "completed", **response_shape(run)})

                if ticket_id == "T-100":
                    final_run = copy.deepcopy(reference_runs["run-billing-001"])
                    final_run["run_id"] = "mock-billing"
                    runtime_runs["mock-billing"] = {"pending": True, "status": "awaiting_approval", "final": final_run}
                    return response({"run_id": "mock-billing", "status": "awaiting_approval"})

                if ticket_id == "T-101":
                    run = copy.deepcopy(reference_runs["run-outage-001"])
                    run["run_id"] = "mock-outage"
                elif ticket_id == "T-102":
                    run = copy.deepcopy(reference_runs["run-policy-001"])
                    run["run_id"] = "mock-policy"
                else:
                    return response({"detail": "unknown ticket"}, status_code=404)

                run["status"] = "completed"
                runtime_runs[run["run_id"]] = run
                return response({"run_id": run["run_id"], "status": "completed", **response_shape(run)})

            if request.method == "GET" and path.startswith("/runs/"):
                run_id = path.split("/")[-1]
                if run_id not in runtime_runs:
                    return response({"detail": "missing run"}, status_code=404)
                run = runtime_runs[run_id]
                if run.get("pending"):
                    return response({"run_id": run_id, "status": "awaiting_approval"})
                return response({"run_id": run_id, "status": run.get("status", "completed"), **response_shape(run)})

            if request.method == "GET" and path.startswith("/trace/"):
                run_id = path.split("/")[-1]
                if run_id not in runtime_runs:
                    return response({"detail": "missing run"}, status_code=404)
                run = runtime_runs[run_id]
                if run.get("pending"):
                    return response({"run_id": run_id, "events": ["run_started", "model_called", "tool_selected", "tool_called", "tool_result", "approval_requested"]})
                return response({"run_id": run_id, "events": run.get("trace_events", [])})

            if request.method == "POST" and path.startswith("/approve/"):
                run_id = path.split("/")[-1]
                if run_id not in runtime_runs:
                    return response({"detail": "missing run"}, status_code=404)
                run = runtime_runs[run_id]
                if run.get("pending"):
                    final_run = run["final"]
                    final_run["status"] = "completed"
                    runtime_runs[run_id] = final_run
                    return response({"run_id": run_id, "status": "completed", **response_shape(final_run)})
                return response({"run_id": run_id, "status": run.get("status", "completed"), **response_shape(run)})

            return response({"detail": "unknown route"}, status_code=404)

        runner = TaskAgentBlackBoxRunner(
            client_factory=lambda base_url, timeout_s: httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url=base_url,
                timeout=timeout_s,
            )
        )
        app.state.task_agent_blackbox_runner = runner
        app.state.workflow_service.runner = runner

    def test_registry_lists_design_catalog(self) -> None:
        response = self.client.get("/v1/registry")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("progressive_codebase_course", body["package_types"])
        domain_packs = {item["id"] for item in body["domain_packs"]}
        overlays = {item["id"] for item in body["overlays"]}
        self.assertIn("generic_project_example", domain_packs)
        self.assertIn("productionization_overlay", overlays)

    def test_root_renders_lms_home(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        body = response.text
        self.assertIn("Course LMS", body)
        self.assertIn("Learner LMS", body)
        self.assertIn("Course builder", body)
        self.assertIn("Open a course to see its deliverables.", body)
        self.assertIn('/static/lms.css', body)
        self.assertIn('/static/lms.js', body)
        self.assertIn('id="lms-state"', body)
        self.assertIn("/create-course", body)

    def test_create_course_renders_authoring_workspace(self) -> None:
        response = self.client.get("/create-course")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        body = response.text
        self.assertIn("Create and review learner-ready course drafts", body)
        self.assertIn("Goal and learning outcomes", body)
        self.assertIn("Workflow progress", body)
        self.assertIn("Suggest outcomes", body)
        self.assertIn("Start building", body)
        self.assertIn("Recent drafts", body)
        self.assertIn("Current state", body)
        self.assertIn("Recent activity", body)
        self.assertIn("Where we are", body)
        self.assertIn("Draft overview", body)
        self.assertIn("Review this step", body)
        self.assertIn("Published versions", body)
        self.assertIn("Start new version", body)
        self.assertIn("Clear local data", body)
        self.assertIn('/static/dashboard.css', body)
        self.assertIn('/static/dashboard.js', body)
        self.assertIn('id="dashboard-state"', body)
        self.assertNotIn("Catalog Patterns", body)

    def test_courses_renders_my_and_all_courses_page(self) -> None:
        response = self.client.get("/courses")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        body = response.text
        self.assertIn("My courses", body)
        self.assertIn("All courses", body)
        self.assertIn('/static/lms.css', body)
        self.assertIn('/static/lms-courses.js', body)
        self.assertIn('id="lms-state"', body)
        self.assertIn("/create-course", body)

    def test_dashboard_static_assets_are_served(self) -> None:
        script = self.client.get("/static/dashboard.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("javascript", script.headers["content-type"])
        self.assertIn("Approve", script.text)
        self.assertIn("Request changes", script.text)
        self.assertIn("Reviewer note", script.text)
        self.assertIn('searchParams.get("draft")', script.text)
        self.assertIn("Assignment spec snapshot", script.text)

        stylesheet = self.client.get("/static/dashboard.css")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn("text/css", stylesheet.headers["content-type"])
        self.assertIn(".tab-strip", stylesheet.text)

    def test_lms_static_assets_are_served(self) -> None:
        script = self.client.get("/static/lms.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("javascript", script.headers["content-type"])
        self.assertIn("Workspace ready", script.text)
        self.assertIn("Open a course to see its deliverables.", script.text)

        stylesheet = self.client.get("/static/lms.css")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn("text/css", stylesheet.headers["content-type"])
        self.assertIn(".learner-focus", stylesheet.text)
        self.assertIn(".catalog-grid", stylesheet.text)

    def test_sandbox_status_endpoint_reports_backend_support(self) -> None:
        response = self.client.get("/v1/sandbox/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["available"])
        self.assertEqual(body["engine"], "docker")

    def test_task_agent_authoring_status_endpoint_reports_fallback_when_unconfigured(self) -> None:
        response = self.client.get("/v1/task-agent-authoring/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["available"])
        self.assertEqual(body["source"], "deterministic_fallback")

    def test_course_generation_status_reports_fallback_when_live_planner_is_disabled(self) -> None:
        response = self.client.get("/v1/course-generation/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["available"])
        self.assertEqual(body["source"], "deterministic_fallback")
        self.assertIn("disabled", body["message"].lower())

    def test_openai_planner_status_uses_default_env_file_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "openai.env.keys"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            with patch.object(openai_runtime_support, "DEFAULT_OPENAI_ENV_FILES", (env_path,)):
                planner = OpenAICoursePlanner(enabled=True)
                status = planner.status()
        self.assertTrue(status.available)
        self.assertTrue(status.api_key_present)
        self.assertEqual(status.env_file, str(env_path))

    def test_generate_course_from_brief_uses_fallback_planner(self) -> None:
        response = self.client.post(
            "/v1/course-runs/generate",
            json={
                "goal": "Build a production-ready feature flag service that evaluates rollout requests, uses tools safely, and can be reviewed as a live course.",
                "learning_outcomes": [
                    "tool selection",
                    "approval gates",
                    "observability",
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "deterministic_fallback")
        self.assertEqual(body["course_run"]["package_type"], "progressive_codebase_course")
        self.assertGreaterEqual(len(body["plan"]["deliverables"]), 3)
        self.assertEqual(body["review"]["counts"]["total_deliverables"], len(body["course_run"]["deliverables"]))
        self.assertIn("ai_usage", body["course_run"])

    def test_generate_course_from_brief_preserves_survey_package_from_router(self) -> None:
        response = self.client.post(
            "/v1/course-runs/generate",
            json={
                "goal": "Create a backend systems course covering retrieval, stateful services, and agents.",
                "learning_outcomes": [
                    "Ship one hands-on assignment per system type",
                    "Practice the core engineering tradeoffs for each system",
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "deterministic_fallback")
        self.assertEqual(body["plan"]["package_type"], "survey_course")
        self.assertEqual(body["course_run"]["package_type"], "survey_course")

    def test_queue_course_generation_persists_draft_before_background_work(self) -> None:
        queued_jobs: list[object] = []
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
            job_runner=lambda job: queued_jobs.append(job),
        )

        response = self.client.post(
            "/v1/course-runs/generate-async",
            json={
                "goal": "Build a production-ready feature flag service that evaluates rollout requests and uses tools safely.",
                "learning_outcomes": ["tool selection", "observability"],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["queued"])
        self.assertEqual(body["course_run"]["stage"], "drafting")
        self.assertEqual(body["course_run"]["status"], "active")
        self.assertEqual(body["course_run"]["deliverables"], [])
        self.assertEqual(len(queued_jobs), 1)

        course_run_id = body["course_run"]["id"]
        events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(events.status_code, 200)
        event_types = [event["event_type"] for event in events.json()]
        self.assertIn("course_generation_queued", event_types)
        self.assertIn("course_generation_started", event_types)

        queued_jobs[0]()

        completed = self.client.get(f"/v1/course-runs/{course_run_id}")
        self.assertEqual(completed.status_code, 200)
        completed_body = completed.json()
        self.assertNotEqual(completed_body["stage"], "drafting")
        self.assertGreaterEqual(len(completed_body["deliverables"]), 1)
        self.assertIsNotNone(completed_body["generated_plan"])

    def test_generate_course_from_brief_can_use_live_planner(self) -> None:
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=FakeLivePlanner(),
        )
        response = self.client.post(
            "/v1/course-runs/generate",
            json={
                "goal": "Build a control plane service course that feels production ready.",
                "learning_outcomes": [
                    "tool selection",
                    "approval gates",
                    "observability",
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "openai_live")
        self.assertEqual(body["status"]["model_id"], "gpt-5.4")
        self.assertIsNotNone(body["course_run"]["shared_design_spec"])
        self.assertTrue(body["course_run"]["shared_design_spec"]["capabilities"]["tool_use_required"])
        self.assertEqual(body["plan"]["deliverables"][0]["title"], body["course_run"]["deliverables"][0]["title"])
        self.assertNotIn("checkpoint_deliverable_ids", body["plan"]["deliverables"][0])
        self.assertIn("Live planning", body["plan"]["deliverables"][0]["title"])

    def test_progressive_course_deliverables_do_not_expose_checkpoint_mappings(self) -> None:
        response = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        first_deliverable = body["deliverables"][0]

        self.assertNotIn("checkpoint_deliverable_ids", first_deliverable)
        self.assertTrue(first_deliverable["title"])
        self.assertTrue(first_deliverable["summary"])

    def test_suggest_learning_outcomes_can_use_live_planner(self) -> None:
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=FakeLivePlanner(),
        )
        response = self.client.post(
            "/v1/course-generation/suggest-outcomes",
            json={
                "goal": "Build a production-ready feature flag service that evaluates rollout requests, uses tools safely, and ships with evals.",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "openai_live")
        self.assertEqual(body["status"]["model_id"], "gpt-5.4")
        self.assertGreaterEqual(len(body["learning_outcomes"]), 4)

    def test_suggest_learning_outcomes_falls_back_when_live_planner_disabled(self) -> None:
        response = self.client.post(
            "/v1/course-generation/suggest-outcomes",
            json={
                "goal": "Build a production-ready feature flag service that evaluates rollout requests, uses tools safely, and ships with evals.",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "deterministic_fallback")
        self.assertGreaterEqual(len(body["learning_outcomes"]), 4)

    def test_suggest_learning_outcomes_normalizes_multiline_items(self) -> None:
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=FakeMultilineOutcomePlanner(),
        )
        response = self.client.post(
            "/v1/course-generation/suggest-outcomes",
            json={"goal": "Build a production-ready flight booking system."},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["learning_outcomes"],
            [
                "Model the booking workflow clearly.",
                "Handle concurrent reservations safely.",
                "Use caching carefully for read-heavy traffic.",
            ],
        )

    def test_creator_plan_endpoint_shapes_flight_booking_course(self) -> None:
        response = self.client.post(
            "/v1/course-generation/creator-plan",
            json={
                "goal": "Build a flight booking system that is production ready. Mock external dependent services where required.",
                "learning_outcomes": [
                    "Keep seat inventory correct under load.",
                    "Explain the tradeoffs between different locking strategies.",
                ],
                "creator_choices": {
                    "starter_type": "partial",
                    "primary_database": "postgres",
                    "cache_backend": "redis",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["plan"]["creator_choices"]["primary_database"], "postgres")
        self.assertEqual(body["plan"]["creator_choices"]["cache_backend"], "redis")
        self.assertEqual(
            body["plan"]["goal"],
            "Build a flight booking system that is production ready. Mock external dependent services where required.",
        )
        self.assertIn(
            "Model the durable state around the invariant that Concurrent or repeated writes do not corrupt critical state.",
            body["plan"]["learning_outcomes"],
        )
        self.assertNotIn("Use pessimistic locking to protect the hot booking path.", body["plan"]["learning_outcomes"])
        deliverable_titles = [deliverable["title"] for deliverable in body["plan"]["deliverables"]]
        self.assertEqual(
            deliverable_titles,
            [
                "Service contract and durable model",
                "Read and write path correctness",
                "Runtime integration and failure recovery",
                "Operational hardening",
            ],
        )
        self.assertNotIn("Pessimistic locking in postgres", deliverable_titles)
        self.assertNotIn("Optimistic locking and retries in postgres", deliverable_titles)
        self.assertIn("shared production-ready codebase", body["plan"]["creator_summary"].lower())

    def test_creator_stack_contract_endpoint_returns_structured_recommendations(self) -> None:
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            stack_catalog_service=FakeStackCatalogService(),
        )
        response = self.client.post(
            "/v1/course-generation/creator-stack-contract",
            json={
                "goal": "Build a production-grade Go reservation service with Gin, Postgres, and Redis.",
                "creator_setup": {
                    "implementation_language": "go",
                    "application_framework": "gin",
                    "primary_database": "postgres",
                    "cache_backend": "redis",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["creator_choices"]["implementation_language"], "go")
        self.assertEqual(body["creator_choices"]["language_version"], "1.26")
        self.assertEqual(body["creator_choices"]["application_framework"], "gin")
        self.assertEqual(body["creator_choices"]["framework_version"], "1.12.0")
        self.assertEqual(body["creator_choices"]["package_manager"], "go")
        self.assertEqual(body["creator_choices"]["primary_database_version"], "18")
        self.assertEqual(body["creator_choices"]["cache_backend_version"], "8")
        self.assertEqual(body["catalog"]["languages"][0]["value"], "go")
        self.assertEqual(body["language_versions"][0]["value"], "1.26")
        self.assertEqual(body["framework_versions"][0]["value"], "1.12.0")
        self.assertIn("Creator approves the final stack contract before generation.", body["notes"])

    def test_create_course_run_from_creator_plan_preserves_creator_choices(self) -> None:
        planned = self.client.post(
            "/v1/course-generation/creator-plan",
            json={
                "goal": "Build a flight booking system that is production ready. Mock external dependent services where required.",
                "learning_outcomes": [
                    "Keep seat inventory correct under load.",
                    "Explain the tradeoffs between different locking strategies.",
                ],
                "creator_choices": {
                    "starter_type": "empty",
                    "implementation_language": "go",
                    "language_version": "1.25",
                    "application_framework": "gin",
                    "framework_version": "1.11",
                    "package_manager": "go",
                    "primary_database": "postgres",
                    "primary_database_version": "17",
                    "cache_backend": "redis",
                    "cache_backend_version": "8",
                },
            },
        )
        self.assertEqual(planned.status_code, 200)

        created = self.client.post(
            "/v1/course-runs/from-creator-plan",
            json={"plan": planned.json()["plan"]},
        )
        self.assertEqual(created.status_code, 200)
        body = created.json()
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["starter_type"], "empty")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["implementation_language"], "go")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["language_version"], "1.25")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["application_framework"], "gin")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["framework_version"], "1.11")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["package_manager"], "go")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["primary_database"], "postgres")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["primary_database_version"], "17")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["cache_backend"], "redis")
        self.assertEqual(body["shared_design_spec"]["runtime_dependencies"]["cache_backend_version"], "8")
        runtime_plan = body["shared_design_spec"]["project_contract"]["runtime_plan"]
        self.assertEqual(runtime_plan["implementation_language"], "go")
        self.assertEqual(runtime_plan["language_version"], "1.25")
        self.assertEqual(runtime_plan["application_framework"], "gin")
        self.assertEqual(runtime_plan["framework_version"], "1.11")
        self.assertEqual(runtime_plan["package_manager"], "go")
        self.assertIsNotNone(body["shared_workflow_run_id"])
        self.assertEqual(
            body["goal"],
            "Build a flight booking system that is production ready. Mock external dependent services where required.",
        )
        self.assertIn(
            "Model the durable state around the invariant that Concurrent or repeated writes do not corrupt critical state.",
            body["requested_learning_outcomes"],
        )
        self.assertNotIn("Use pessimistic locking to protect the hot booking path.", body["requested_learning_outcomes"])
        self.assertEqual(body["generated_plan"]["title"], body["title"])
        creator_view = self.client.get(f"/v1/course-runs/{body['id']}/creator-view")
        self.assertEqual(creator_view.status_code, 200)
        creator_body = creator_view.json()
        creator_deliverable_titles = [deliverable["title"] for deliverable in creator_body["review"]["deliverables"]]
        self.assertEqual(
            creator_deliverable_titles,
            [
                "Service contract and durable model",
                "Read and write path correctness",
                "Runtime integration and failure recovery",
                "Operational hardening",
            ],
        )

    def test_creator_plan_proposes_retrieval_data_source_when_goal_needs_corpus(self) -> None:
        response = self.client.post(
            "/v1/course-generation/creator-plan",
            json={
                "goal": "Build a production-ready internal docs RAG system that answers from a visible corpus with citations.",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["plan"]["creator_choices"]["starter_type"], "partial")
        self.assertEqual(len(body["plan"]["creator_choices"]["data_sources"]), 1)
        source = body["plan"]["creator_choices"]["data_sources"][0]
        self.assertEqual(source["purpose"], "retrieval")
        self.assertEqual(source["workspace_path"], "data/corpus.json")
        self.assertIn("data sources", body["plan"]["creator_summary"].lower())

    def test_creator_asset_upload_list_and_delete(self) -> None:
        created = self.client.post(
            "/v1/creator-assets",
            json={
                "file_name": "refund_policy.md",
                "content": "# Refund policy\n\nRefunds are allowed within 30 days.\n",
                "content_type": "text/markdown",
                "purpose": "retrieval",
            },
        )
        self.assertEqual(created.status_code, 200)
        asset = created.json()
        self.assertEqual(asset["workspace_path"], "data/refund_policy.md")
        self.assertEqual(asset["data_source"]["asset_id"], asset["id"])
        self.assertEqual(asset["data_source"]["purpose"], "retrieval")

        listed = self.client.get("/v1/creator-assets")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["assets"]), 1)

        deleted = self.client.delete(f"/v1/creator-assets/{asset['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["asset_id"], asset["id"])

        missing = self.client.get("/v1/creator-assets")
        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["assets"], [])

    def test_uploaded_creator_asset_materializes_into_bundle_and_publish_snapshot(self) -> None:
        created_asset = self.client.post(
            "/v1/creator-assets",
            json={
                "file_name": "airline_policies.md",
                "content": "# Airline policies\n\nFlights can be rebooked within 24 hours.\n",
                "content_type": "text/markdown",
                "purpose": "retrieval",
            },
        )
        self.assertEqual(created_asset.status_code, 200)
        asset = created_asset.json()

        planned = self.client.post(
            "/v1/course-generation/creator-plan",
            json={
                "goal": "Build a production-ready internal docs RAG system that answers from uploaded airline policies with citations.",
                "creator_choices": {
                    "starter_type": "partial",
                    "data_sources": [asset["data_source"]],
                },
            },
        )
        self.assertEqual(planned.status_code, 200)

        created = self.client.post(
            "/v1/course-runs/from-creator-plan",
            json={"plan": planned.json()["plan"]},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        workflow_run_id = course_run["shared_workflow_run_id"]
        self.assertIsNotNone(workflow_run_id)

        shared_run = app.state.workflow_service.materialize_run(
            workflow_run_id,
            MaterializeBundleRequest(overwrite=True),
        )
        uploaded_bundle_file = app.state.workflow_service.read_bundle_file(
            workflow_run_id,
            "public/starter/deliverable_1/data/airline_policies.md",
        )
        self.assertIn("Flights can be rebooked within 24 hours.", uploaded_bundle_file.content)

        course_run_model = app.state.course_workflow_service.get_run(course_run["id"])
        assert course_run_model is not None
        snapshot = app.state.course_workflow_service.publish_snapshot_service.create_snapshot(
            course_run_model,
            {workflow_run_id: shared_run},
        )
        assert snapshot is not None
        seed_files = {
            file.relative_path: file.content
            for file in snapshot.learner_package.deliverables[0].workspace_seed_files
        }
        self.assertIn("data/airline_policies.md", seed_files)
        self.assertIn("Flights can be rebooked within 24 hours.", seed_files["data/airline_policies.md"])

    def test_queue_course_run_from_creator_plan_returns_placeholder_then_builds(self) -> None:
        queued_jobs: list[object] = []
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
            job_runner=lambda job: queued_jobs.append(job),
        )
        planned = self.client.post(
            "/v1/course-generation/creator-plan",
            json={
                "goal": "Build a flight booking system that is production ready. Mock external dependent services where required.",
                "learning_outcomes": [
                    "Keep seat inventory correct under load.",
                    "Explain the tradeoffs between different locking strategies.",
                ],
                "creator_choices": {
                    "starter_type": "partial",
                    "primary_database": "postgres",
                    "cache_backend": "redis",
                },
            },
        )
        self.assertEqual(planned.status_code, 200)

        queued = self.client.post(
            "/v1/course-runs/from-creator-plan-async",
            json={"plan": planned.json()["plan"]},
        )
        self.assertEqual(queued.status_code, 200)
        body = queued.json()
        self.assertTrue(body["queued"])
        self.assertEqual(body["course_run"]["stage"], "drafting")
        self.assertEqual(body["course_run"]["status"], "active")
        self.assertEqual(body["course_run"]["deliverables"], [])
        self.assertEqual(body["course_run"]["summary"], planned.json()["plan"]["summary"])
        self.assertEqual(body["course_run"]["generated_plan"]["title"], planned.json()["plan"]["title"])
        self.assertEqual(len(queued_jobs), 1)

        course_run_id = body["course_run"]["id"]
        events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(events.status_code, 200)
        event_types = [event["event_type"] for event in events.json()]
        self.assertIn("course_generation_queued", event_types)
        self.assertIn("course_generation_started", event_types)
        self.assertIn("creator_plan_accepted", event_types)

        queued_jobs[0]()

        completed = self.client.get(f"/v1/course-runs/{course_run_id}")
        self.assertEqual(completed.status_code, 200)
        completed_body = completed.json()
        self.assertNotEqual(completed_body["stage"], "drafting")
        self.assertIsNotNone(completed_body["shared_workflow_run_id"])
        self.assertGreaterEqual(len(completed_body["deliverables"]), 1)
        self.assertEqual(completed_body["deliverables"][0]["title"], planned.json()["plan"]["deliverables"][0]["title"])
        creator_view = self.client.get(f"/v1/course-runs/{course_run_id}/creator-view")
        self.assertEqual(creator_view.status_code, 200)
        creator_body = creator_view.json()
        creator_deliverable_titles = [deliverable["title"] for deliverable in creator_body["review"]["deliverables"]]
        self.assertEqual(
            creator_deliverable_titles,
            [
                "Service contract and durable model",
                "Read and write path correctness",
                "Runtime integration and failure recovery",
                "Operational hardening",
            ],
        )

    def test_creator_view_does_not_mutate_shared_workflow_when_course_copy_drifts(self) -> None:
        planned = self.client.post(
            "/v1/course-generation/creator-plan",
            json={
                "goal": "Build a grounded internal docs assistant that answers from a visible corpus with citations and abstains when support is weak.",
                "creator_choices": {
                    "starter_type": "partial",
                },
            },
        )
        self.assertEqual(planned.status_code, 200)

        created = self.client.post(
            "/v1/course-runs/from-creator-plan",
            json={"plan": planned.json()["plan"]},
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]
        shared_workflow_run_id = created.json()["shared_workflow_run_id"]
        self.assertIsNotNone(shared_workflow_run_id)

        stored = app.state.workflow_service.store.get_course_run(course_run_id)
        assert stored is not None
        stored.active_operation = CourseAsyncOperation.generation
        stored.deliverables[0].title = "Locally edited creator title"
        app.state.workflow_service.store.save_course_run(stored)

        before_events = app.state.workflow_service.store.list_events(shared_workflow_run_id)
        before_count = len(before_events)

        creator_view = self.client.get(f"/v1/course-runs/{course_run_id}/creator-view")
        self.assertEqual(creator_view.status_code, 200)
        body = creator_view.json()
        self.assertEqual(body["course_run"]["active_operation"], "generation")

        after_events = app.state.workflow_service.store.list_events(shared_workflow_run_id)
        self.assertEqual(len(after_events), before_count)
        self.assertEqual(
            app.state.workflow_service.store.get_run(shared_workflow_run_id).artifacts.task_agent_spec.deliverables[0].title,
            created.json()["deliverables"][0]["title"],
        )

    def test_creator_plan_with_more_deliverables_than_base_scaffold_requires_new_hidden_coverage(self) -> None:
        planned = self.client.post(
            "/v1/course-generation/creator-plan",
            json={
                "goal": "Build a grounded internal docs assistant that answers from a visible corpus with citations and abstains when support is missing.",
                "learning_outcomes": [
                    "Build grounded retrieval over a learner-visible corpus.",
                    "Return citations for every supported answer.",
                ],
                "creator_choices": {
                    "starter_type": "partial",
                },
            },
        )
        self.assertEqual(planned.status_code, 200)
        plan = copy.deepcopy(planned.json()["plan"])
        while len(plan["deliverables"]) < 5:
            index = len(plan["deliverables"]) + 1
            plan["deliverables"].append(
                {
                    "deliverable_slug": f"custom-deliverable-{index}",
                    "title": f"Custom deliverable {index}",
                    "summary": f"Extend the grounded assistant through custom deliverable {index}.",
                    "learning_outcomes": [f"Practice grounded behavior {index} in the learner-visible service."],
                    "creator_notes": [],
                    "design_spec": plan["shared_design_spec"],
                }
            )

        created = self.client.post(
            "/v1/course-runs/from-creator-plan",
            json={"plan": plan},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]
        self.assertEqual(len(course_run["deliverables"]), len(plan["deliverables"]))
        self.assertTrue(all("checkpoint_deliverable_ids" not in deliverable for deliverable in course_run["deliverables"]))

        decision = self.client.post(
            f"/v1/workflow-runs/{shared_run_id}/decisions",
            json={"gate": "gate_1_spec_review", "decision": "approve"},
        )
        self.assertEqual(decision.status_code, 409)

    def test_normalize_plan_preserves_shared_design_spec_across_progressive_deliverables(self) -> None:
        service = app.state.course_generation_service
        request = GenerateCourseFromBriefRequest(
            title="Operations Training",
            goal="Build a practical engineering training program.",
            learning_outcomes=["operational readiness"],
        )
        shared_design_spec = _design_spec(
            title="Operations Training",
            problem_statement="Build a practical engineering training program.",
            learning_outcomes=["operational readiness"],
        )
        plan = GeneratedCoursePlan(
            title="Operations Training",
            summary="A practical engineering training course.",
            package_type=PackageType.progressive_codebase_course,
            shared_design_spec=shared_design_spec,
            deliverables=[
                CreateCourseDeliverableRequest(
                    title="Bounded agent workflow",
                    summary="Build the run contract and tool flow.",
                    learning_outcomes=["tool selection"],
                    design_spec=shared_design_spec,
                    domain_pack_hint="generic_project_example",
                ),
                CreateCourseDeliverableRequest(
                    title="Production hardening",
                    summary="Add approvals, evals, and observability.",
                    learning_outcomes=["observability"],
                    design_spec=shared_design_spec.model_copy(update={"overlays": ["productionization_overlay"]}),
                    domain_pack_hint="generic_project_example",
                    overlays_hint=["productionization_overlay"],
                ),
            ],
        )

        normalized = service._normalize_plan(plan, request)

        self.assertIsNotNone(normalized.shared_design_spec)
        self.assertEqual(
            [deliverable.design_spec for deliverable in normalized.deliverables],
            [normalized.shared_design_spec, normalized.shared_design_spec],
        )

    def test_reset_local_course_state_clears_runs(self) -> None:
        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)

        before = self.client.get("/v1/course-runs")
        self.assertEqual(before.status_code, 200)
        self.assertGreaterEqual(len(before.json()["runs"]), 1)

        reset = self.client.post("/v1/course-runs/reset-local")
        self.assertEqual(reset.status_code, 200)
        body = reset.json()
        self.assertGreaterEqual(body["deleted_course_runs"], 1)
        self.assertGreaterEqual(body["deleted_workflow_runs"], 1)
        self.assertGreaterEqual(len(body["cleared_directories"]), 1)

        after = self.client.get("/v1/course-runs")
        self.assertEqual(after.status_code, 200)
        self.assertEqual(after.json()["runs"], [])

    def test_course_patterns_include_forward_deployed_engineering(self) -> None:
        response = self.client.get("/v1/course-patterns")
        self.assertEqual(response.status_code, 200)
        titles = {course["course_title"] for course in response.json()}
        self.assertIn("Forward Deployed Engineering", titles)

    def test_course_pattern_lookup_by_slug(self) -> None:
        response = self.client.get("/v1/course-patterns/tusharbisht-cs-demo-agent-to-production")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["course_title"], "Feature Flag Platform — Demo to Production")
        self.assertEqual(body["package_type"], "progressive_codebase_course")

    def test_design_inference_recognizes_support_agent_work(self) -> None:
        response = self.client.post(
            "/v1/designs/infer",
            json={
                "title": "Feature flag service",
                "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                "learning_outcomes": [
                    "tool selection",
                    "fallback handling",
                    "observability",
                    "approval gates",
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "supported")
        self.assertEqual(body["design_spec"]["domain_pack"], "generic_project_example")
        self.assertTrue(body["design_spec"]["capabilities"]["tool_use_required"])
        self.assertIn("productionization_overlay", body["design_spec"]["overlays"])

    def test_design_inference_flags_review_required_clinical_agent(self) -> None:
        response = self.client.post(
            "/v1/designs/infer",
            json={
                "title": "Clinical case triage agent",
                "problem_statement": "Build an agent that reviews patient cases, drafts next steps, and escalates ambiguous diagnoses.",
                "learning_outcomes": ["tool use", "confidence calibration"],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "manual_review")
        self.assertEqual(body["design_spec"]["risk_class"], "review_required")

    def test_workflow_run_creation_persists_task_agent_draft(self) -> None:
        response = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": [
                        "tool selection",
                        "fallback handling",
                        "observability",
                        "approval gates",
                    ],
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["stage"], "awaiting_hil_gate_1")
        self.assertEqual(body["pending_gate"], "gate_1_spec_review")
        self.assertEqual(body["artifacts"]["draft_kind"], "task_agent_spec")
        self.assertIsNone(body["artifacts"]["task_agent_spec"]["domain_pack"])
        self.assertGreaterEqual(len(body["artifacts"]["node_executions"]), 5)
        self.assertEqual(body["artifacts"]["node_executions"][0]["kind"], "authoring_runtime")

        list_response = self.client.get("/v1/workflow-runs")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()["runs"]), 1)

    def test_workflow_run_creation_can_use_openai_authoring_service(self) -> None:
        app.state.task_agent_authoring_service = FakeTaskAgentAuthoringService()
        app.state.workflow_service = WorkflowService(
            app.state.workflow_service.store,
            ArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            app.state.task_agent_blackbox_runner,
            app.state.assignment_node_runtime,
            app.state.task_agent_authoring_service,
            app.state.assignment_workspace_manager,
        )
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        response = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["artifacts"]["origin_template"].startswith("openai_customized:"))
        self.assertIn("Customized with fake OpenAI.", body["artifacts"]["notes"])
        self.assertEqual(body["artifacts"]["task_agent_spec"]["deliverables"][0]["title"], "OpenAI-authored foundation")

    def test_workflow_nodes_endpoint_returns_langgraph_node_results(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        ).json()
        run_id = created["id"]

        nodes = self.client.get(f"/v1/workflow-runs/{run_id}/nodes")
        self.assertEqual(nodes.status_code, 200)
        body = nodes.json()
        self.assertEqual(body[0]["kind"], "authoring_runtime")
        self.assertEqual(body[-1]["kind"], "reviewer_tests")
        self.assertTrue(
            any(
                finding["title"].startswith("Visible learner checks ready")
                and "deeper hidden grader" in finding["detail"]
                for finding in body[-1]["findings"]
            )
        )

    def test_workflow_workspace_endpoints_expose_persistent_generated_files(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        ).json()
        run_id = created["id"]

        workspace = self.client.get(f"/v1/workflow-runs/{run_id}/workspace")
        self.assertEqual(workspace.status_code, 200)
        body = workspace.json()
        self.assertTrue(body["root_dir"].endswith(run_id))

        starter_file = self.client.get(
            f"/v1/workflow-runs/{run_id}/workspace/file",
            params={"path": "public/starter/deliverable_1/app.py"},
        )
        self.assertEqual(starter_file.status_code, 200)
        starter_source = starter_file.json()["content"]
        self.assertIn("create_app_from_manifest", starter_source)

        visible_checks = self.client.get(
            f"/v1/workflow-runs/{run_id}/workspace/file",
            params={"path": "public/starter/deliverable_1/checks/run_visible_checks.py"},
        )
        self.assertEqual(visible_checks.status_code, 200)
        self.assertIn("public_checks_by_case", visible_checks.json()["content"])

        vscode_tasks = self.client.get(
            f"/v1/workflow-runs/{run_id}/workspace/file",
            params={"path": "public/starter/deliverable_1/.vscode/tasks.json"},
        )
        self.assertEqual(vscode_tasks.status_code, 200)
        self.assertIn("Run visible checks", vscode_tasks.json()["content"])
        self.assertNotIn("status_code=501", starter_source)

        runtime_file = self.client.get(
            f"/v1/workflow-runs/{run_id}/workspace/file",
            params={"path": "public/runtime/task_agent_runtime.py"},
        )
        self.assertEqual(runtime_file.status_code, 200)
        self.assertIn("COURSE_GEN_TASK_AGENT_RUNTIME", runtime_file.json()["content"])

    def test_task_agent_spec_update_rematerializes_workspace_and_invalidates_bundle(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        ).json()
        run_id = created["id"]

        original_readme = self.client.get(
            f"/v1/workflow-runs/{run_id}/workspace/file",
            params={"path": "public/starter/deliverable_1/README.md"},
        )
        self.assertEqual(original_readme.status_code, 200)
        self.assertNotIn("deterministic audit trail for booking retries", original_readme.json()["content"])

        materialized = self.client.post(
            f"/v1/workflow-runs/{run_id}/materialize",
            json={"overwrite": True},
        )
        self.assertEqual(materialized.status_code, 200)
        self.assertIsNotNone(materialized.json()["artifacts"]["materialized_bundle"])

        spec = created["artifacts"]["task_agent_spec"]
        spec["deliverables"][0]["learner_brief"]["task_to_build"] = (
            "Edit `app.py` to add a deterministic audit trail for booking retries."
        )

        update = self.client.put(f"/v1/workflow-runs/{run_id}/task-agent-spec", json=spec)
        self.assertEqual(update.status_code, 200)
        updated = update.json()
        self.assertIsNone(updated["artifacts"]["materialized_bundle"])

        updated_readme = self.client.get(
            f"/v1/workflow-runs/{run_id}/workspace/file",
            params={"path": "public/starter/deliverable_1/README.md"},
        )
        self.assertEqual(updated_readme.status_code, 200)
        self.assertIn(
            "deterministic audit trail for booking retries",
            updated_readme.json()["content"],
        )
        self.assertTrue(
            any(
                "Invalidated the workspace snapshot and materialized bundle after task-agent spec update"
                in note
                for note in updated["artifacts"]["notes"]
            )
        )

    def test_workflow_review_endpoint_reports_loop_summary(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        ).json()

        review = self.client.get(f"/v1/workflow-runs/{created['id']}/review")
        self.assertEqual(review.status_code, 200)
        body = review.json()
        self.assertTrue(body["review_ready"])
        self.assertEqual(body["policy"]["max_authoring_attempts"], 3)
        self.assertEqual(body["policy"]["max_reviewer_attempts"], 2)
        self.assertEqual(body["authoring"]["attempts_used"], 1)
        self.assertEqual(body["reviewer"]["attempts_used"], 1)
        self.assertEqual(body["blockers"], [])

    def test_workflow_review_endpoint_marks_authoring_exhaustion_when_sandbox_fails(self) -> None:
        failing_sandbox = FakeSandboxRunner(success=False)
        app.state.docker_sandbox_runner = failing_sandbox
        app.state.assignment_node_runtime = LangGraphAssignmentGraph(
            failing_sandbox,
            workspace_authoring_service=self.workspace_authoring_service,
            test_authoring_service=self.fake_test_authoring_service,
            baseline_verifier=_AlwaysValidBaselineVerifier(),
            max_authoring_attempts=2,
            max_reviewer_attempts=2,
        )
        app.state.workflow_service = WorkflowService(
            app.state.workflow_service.store,
            ArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            app.state.task_agent_blackbox_runner,
            app.state.assignment_node_runtime,
            self.disabled_authoring_service,
            app.state.assignment_workspace_manager,
        )
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection"],
                }
            },
        )
        self.assertEqual(created.status_code, 200)
        body = created.json()
        self.assertEqual(body["stage"], "blocked")
        self.assertEqual(body["status"], "blocked")
        self.assertEqual(body["artifacts"]["review_summary"]["authoring"]["attempts_used"], 1)
        self.assertTrue(body["artifacts"]["review_summary"]["authoring"]["exhausted"])
        self.assertIn(
            "Retry produced no material spec changes",
            "\n".join(body["artifacts"]["review_summary"]["blockers"]),
        )

    def test_authoring_repair_loop_preserves_workspace_and_fixes_broken_deliverable_file(self) -> None:
        compile_sandbox = WorkspaceCompileSandboxRunner()
        broken_workspace_authoring = BrokenFirstWorkspaceAuthoringService(
            self.workspace_manager,
            repo_authoring_service=self.fake_repo_authoring_service,
        )
        app.state.docker_sandbox_runner = compile_sandbox
        app.state.task_agent_workspace_authoring_service = broken_workspace_authoring
        app.state.assignment_node_runtime = LangGraphAssignmentGraph(
            compile_sandbox,
            workspace_authoring_service=broken_workspace_authoring,
            test_authoring_service=self.fake_test_authoring_service,
            baseline_verifier=_AlwaysValidBaselineVerifier(),
            max_authoring_attempts=3,
            max_reviewer_attempts=1,
        )
        app.state.workflow_service = WorkflowService(
            app.state.workflow_service.store,
            ArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            app.state.task_agent_blackbox_runner,
            app.state.assignment_node_runtime,
            self.disabled_authoring_service,
            self.workspace_manager,
        )
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        )
        self.assertEqual(created.status_code, 200)
        body = created.json()
        self.assertEqual(body["stage"], "awaiting_hil_gate_1")
        self.assertEqual(body["artifacts"]["review_summary"]["authoring"]["attempts_used"], 2)
        node_kinds = [node["kind"] for node in body["artifacts"]["node_executions"]]
        self.assertIn("authoring_repair", node_kinds)
        starter_surface = body["artifacts"]["task_agent_spec"]["deliverables"][0]["learner_starter_surface"]
        editable_path = starter_surface["primary_editable_paths"][0]

        starter_path = (
            Path(body["artifacts"]["workspace_snapshot"]["public_dir"])
            / "starter"
            / "deliverable_1"
            / editable_path
        )
        source = starter_path.read_text(encoding="utf-8")
        self.assertNotIn("BROKEN_STARTER_SENTINEL", source)
        self.assertNotIn("def broken(:", source)
        self.assertEqual(
            broken_workspace_authoring.author_calls,
            1,
            "The repaired workspace should go straight back to sandbox verification without re-authoring the whole workspace.",
        )

    def test_workspace_repair_full_repair_rematerializes_learner_artifacts(self) -> None:
        run = app.state.workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Workspace repair rematerialization",
                problem_statement="Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                learning_outcomes=["tool selection"],
            ),
            design_spec=_design_spec(
                title="Workspace repair rematerialization",
                problem_statement="Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                learning_outcomes=["tool selection"],
            ),
        )

        workspace = run.artifacts.workspace_snapshot
        self.assertIsNotNone(workspace)
        readme_path = Path(workspace.public_dir) / "starter" / "deliverable_1" / "README.md"
        original = readme_path.read_text(encoding="utf-8")
        readme_path.write_text("STALE README\n", encoding="utf-8")

        latest_node = WorkflowNodeExecution(
            node_id="reviewer_runtime_1",
            kind=WorkflowNodeKind.reviewer_runtime,
            status=WorkflowNodeStatus.failed,
            attempt=1,
            summary="Sandbox failed during runtime review.",
            created_at=datetime.now(UTC),
            sandbox_result=SandboxExecutionResult(
                status=SandboxExecutionStatus.failed,
                available=True,
                build_succeeded=False,
                run_succeeded=False,
                generated_at=datetime.now(UTC),
                duration_ms=5,
                build_stdout="",
                build_stderr="docker build failed",
                run_stdout="",
                run_stderr="",
                error="sandbox failed",
            ),
            findings=[],
        )
        failure_context = FailureContext(
            source_node_kind=WorkflowNodeKind.reviewer_runtime,
            source_node_attempt=1,
            source_summary="Sandbox failed during runtime review.",
            findings=[
                ReviewerFinding(
                    category="runtime_review",
                    severity=ReviewerFindingSeverity.error,
                    title="Runtime verification failed",
                    detail="The reviewer sandbox run failed.",
                )
            ],
            sandbox=FailureContextSandboxSummary(
                error="sandbox failed",
                build_stderr_excerpt="docker build failed",
            ),
        )

        repaired_run, repaired, message = self.workspace_authoring_service.repair_workspace(
            run,
            latest_node,
            failure_context=failure_context,
        )
        self.assertTrue(repaired)
        self.assertIn("Rematerialized the full learner workspace", message)
        repaired_workspace = repaired_run.artifacts.workspace_snapshot
        self.assertIsNotNone(repaired_workspace)
        repaired_readme = (
            Path(repaired_workspace.public_dir) / "starter" / "deliverable_1" / "README.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(repaired_readme, original)

    def test_survey_course_creation_creates_deliverable_assignment_runs(self) -> None:
        stateful_design = _design_spec(
            title="TinyURL",
            problem_statement="Build a URL shortener with collision resistance, idempotency, and concurrency safety.",
            learning_outcomes=["idempotency", "concurrency"],
            package_type=PackageType.survey_course,
        )
        support_design = _design_spec(
            title="Support triage agent",
            problem_statement="Build a support triage agent with tools, approvals, and observability.",
            learning_outcomes=["tool selection", "observability"],
            package_type=PackageType.survey_course,
        )
        response = self.client.post(
            "/v1/course-runs",
            json={
                "title": "Backend Systems Survey",
                "summary": "A survey course across independent backend system assignments.",
                "package_type": "survey_course",
                "deliverables": [
                    {
                        "deliverable_slug": "tinyurl",
                        "title": "TinyURL",
                        "summary": "Build a URL shortener with collision resistance and concurrency safety.",
                        "learning_outcomes": ["idempotency", "concurrency"],
                        "design_spec": stateful_design.model_dump(mode="json"),
                    },
                    {
                        "deliverable_slug": "support-agent",
                        "title": "Support triage agent",
                        "summary": "Build a support triage agent with tools, approvals, and observability.",
                        "learning_outcomes": ["tool selection", "observability"],
                        "design_spec": support_design.model_dump(mode="json"),
                        "domain_pack_hint": "generic_project_example",
                        "overlays_hint": ["productionization_overlay"],
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["package_type"], "survey_course")
        self.assertEqual(len(body["deliverables"]), 2)
        workflow_ids = {deliverable["workflow_run_id"] for deliverable in body["deliverables"]}
        self.assertEqual(len(workflow_ids), 2)

        workflow_runs = self.client.get("/v1/workflow-runs")
        self.assertEqual(workflow_runs.status_code, 200)
        self.assertEqual(len(workflow_runs.json()["runs"]), 2)

    def test_progressive_course_creation_uses_shared_workflow_run(self) -> None:
        response = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["package_type"], "progressive_codebase_course")
        self.assertIsNotNone(body["shared_workflow_run_id"])
        workflow_ids = {deliverable["workflow_run_id"] for deliverable in body["deliverables"]}
        self.assertEqual(workflow_ids, {body["shared_workflow_run_id"]})
        self.assertIsNotNone(body["shared_design_spec"])
        self.assertTrue(body["shared_design_spec"]["capabilities"]["tool_use_required"])

    def test_workflow_spec_update_repairs_references_even_if_review_blockers_remain(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection"],
                }
            },
        ).json()
        run_id = created["id"]
        spec = created["artifacts"]["task_agent_spec"]
        spec["behaviors"][0]["test"]["case_ids"].append("missing_case")

        update = self.client.put(f"/v1/workflow-runs/{run_id}/task-agent-spec", json=spec)
        self.assertEqual(update.status_code, 200)
        updated = update.json()
        self.assertFalse(updated["artifacts"]["validation_summary"]["valid"])
        self.assertTrue(
            any(error["code"] == "placeholder_domain_scenario" for error in updated["artifacts"]["validation_summary"]["errors"])
        )
        self.assertEqual(
            updated["artifacts"]["task_agent_spec"]["behaviors"][0]["test"]["case_ids"],
            ["happy_path", "escalation_case"],
        )
        self.assertIn("reviewer_repair", [node["kind"] for node in updated["artifacts"]["node_executions"]])

    def test_workflow_gate_decisions_publish_run(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["observability"],
                }
            },
        ).json()
        run_id = created["id"]

        gate_1 = self.client.post(
            f"/v1/workflow-runs/{run_id}/decisions",
            json={"gate": "gate_1_spec_review", "decision": "approve"},
        )
        self.assertEqual(gate_1.status_code, 200)
        self.assertEqual(gate_1.json()["pending_gate"], "gate_2_progression_review")

        gate_2 = self.client.post(
            f"/v1/workflow-runs/{run_id}/decisions",
            json={"gate": "gate_2_progression_review", "decision": "approve"},
        )
        self.assertEqual(gate_2.status_code, 200)
        self.assertEqual(gate_2.json()["pending_gate"], "gate_3_pre_publish")

        gate_3 = self.client.post(
            f"/v1/workflow-runs/{run_id}/decisions",
            json={"gate": "gate_3_pre_publish", "decision": "approve"},
        )
        self.assertEqual(gate_3.status_code, 200)
        self.assertEqual(gate_3.json()["status"], "published")
        self.assertIsNone(gate_3.json()["pending_gate"])

        events = self.client.get(f"/v1/workflow-runs/{run_id}/events")
        self.assertEqual(events.status_code, 200)
        self.assertEqual(len(events.json()), 5)

        spec = gate_3.json()["artifacts"]["task_agent_spec"]
        spec["summary"] = spec["summary"] + " Updated after publish."
        update = self.client.put(f"/v1/workflow-runs/{run_id}/task-agent-spec", json=spec)
        self.assertEqual(update.status_code, 409)
        self.assertIn("immutable", update.json()["detail"])

    def test_workflow_gate_reject_with_comment_reruns_with_feedback(self) -> None:
        fake_authoring = FakeTaskAgentAuthoringService()
        app.state.task_agent_authoring_service = fake_authoring
        app.state.workflow_service.task_agent_authoring_service = app.state.task_agent_authoring_service

        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection"],
                }
            },
        ).json()
        run_id = created["id"]

        decision = self.client.post(
            f"/v1/workflow-runs/{run_id}/decisions",
            json={
                "gate": "gate_1_spec_review",
                "decision": "reject",
                "comment": "Tighten the deliverable 1 contract and make the opening deliverable title clearer.",
            },
        )
        self.assertEqual(decision.status_code, 200)
        body = decision.json()
        self.assertEqual(body["stage"], "awaiting_hil_gate_1")
        self.assertEqual(body["pending_gate"], "gate_1_spec_review")
        self.assertIn("Revised after feedback", body["artifacts"]["task_agent_spec"]["deliverables"][0]["title"])
        self.assertIn("fake OpenAI", "\n".join(body["notes"]))
        self.assertIsNotNone(fake_authoring.last_failure_context)
        self.assertEqual(fake_authoring.last_failure_context.source_node_kind.value, "reviewer_tests")
        self.assertTrue(
            any(
                finding.category == "pedagogy_review"
                for finding in fake_authoring.last_failure_context.findings
            )
        )
        revised_readme = self.client.get(
            f"/v1/workflow-runs/{run_id}/workspace/file",
            params={"path": "public/starter/deliverable_1/README.md"},
        )
        self.assertEqual(revised_readme.status_code, 200)
        self.assertIn("Revised after feedback", revised_readme.json()["content"])

        events = self.client.get(f"/v1/workflow-runs/{run_id}/events")
        self.assertEqual(events.status_code, 200)
        event_types = [event["event_type"] for event in events.json()]
        self.assertIn("gate_rejected", event_types)
        self.assertIn("langgraph_nodes_executed", event_types)

    def test_reviewer_pedagogy_failure_routes_into_repair(self) -> None:
        run = app.state.workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Pedagogy repair flow",
                problem_statement="Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                learning_outcomes=["tool selection"],
            ),
            design_spec=_design_spec(
                title="Pedagogy repair flow",
                problem_statement="Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                learning_outcomes=["tool selection"],
            ),
        )

        stored = app.state.workflow_service.get_run(run.id)
        assert stored is not None
        stored.artifacts.task_agent_spec.deliverables[0].learning_outcomes = []
        stored.artifacts.node_executions = []
        app.state.workflow_service.store.save_run(stored)

        updated = app.state.workflow_service.execute_langgraph_nodes(run.id)
        node_kinds = [node.kind.value for node in updated.artifacts.node_executions]

        self.assertIn("reviewer_pedagogy", node_kinds)
        self.assertIn("reviewer_repair", node_kinds)
        self.assertNotIn("reviewer_tests", node_kinds)
        self.assertTrue(updated.artifacts.task_agent_spec.deliverables[0].learning_outcomes)
        self.assertFalse(updated.artifacts.validation_summary["valid"])
        self.assertTrue(
            any(
                "Rebuilt learner briefs, public checks, and derived deliverable outcomes"
                in note
                for note in updated.notes
            )
        )

    def test_structural_reviewer_code_failure_bounces_back_to_authoring(self) -> None:
        run = app.state.workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Structural authoring reroute",
                problem_statement="Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                learning_outcomes=["tool selection"],
            ),
            design_spec=_design_spec(
                title="Structural authoring reroute",
                problem_statement="Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                learning_outcomes=["tool selection"],
            ),
            execute_nodes=False,
        )

        latest = WorkflowNodeExecution(
            node_id="reviewer_code_1",
            kind=WorkflowNodeKind.reviewer_code,
            status=WorkflowNodeStatus.failed,
            attempt=1,
            summary="Code review found a thin wrapper starter surface.",
            created_at=datetime.now(UTC),
            findings=[
                ReviewerFinding(
                    category="code_review",
                    severity=ReviewerFindingSeverity.error,
                    title="Primary starter surface is still a thin wrapper",
                    detail="The learner-owned files should contain the real application flow, not just import a generated runtime wrapper.",
                )
            ],
        )
        route = app.state.assignment_node_runtime._after_reviewer_code(
            {
                "run": run,
                "node_executions": [latest],
                "authoring_attempt": 1,
                "reviewer_attempt": 1,
                "cached_sandbox_result": None,
            }
        )
        self.assertEqual(route, "authoring_repair")

    def test_out_of_scope_workflow_is_blocked_without_review_gate(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Mobile travel planner",
                    "problem_statement": "Build an iOS and Android app with rich offline UI and device-native navigation flows.",
                    "learning_outcomes": ["mobile UI", "offline sync"],
                }
            },
        )
        self.assertEqual(created.status_code, 200)
        body = created.json()
        self.assertEqual(body["status"], "blocked")
        self.assertEqual(body["stage"], "blocked")
        self.assertEqual(body["artifacts"]["draft_kind"], "scope_blocked")
        self.assertIsNone(body["pending_gate"])
        self.assertIsNone(body["artifacts"]["task_agent_spec"])

    def test_grounded_rag_workflow_is_generated_as_learner_ready_spec(self) -> None:
        run = app.state.workflow_service.create_run_from_explicit_plan(
            intake=GenerationIntake(
                title="Grounded RAG workflow",
                problem_statement="Build a grounded RAG system that answers from a visible corpus with citations and abstains when support is weak.",
                learning_outcomes=["citation correctness", "grounded answers"],
            ),
            design_spec=_design_spec(
                title="Grounded RAG workflow",
                problem_statement="Build a grounded RAG system that answers from a visible corpus with citations and abstains when support is weak.",
                learning_outcomes=["citation correctness", "grounded answers"],
            ),
        )

        materialized = self.client.post(
            f"/v1/workflow-runs/{run.id}/materialize",
            json={"overwrite": True},
        )
        self.assertEqual(materialized.status_code, 200)
        body = materialized.json()
        self.assertEqual(body["artifacts"]["draft_kind"], "task_agent_spec")
        self.assertEqual(
            body["artifacts"]["task_agent_spec"]["capabilities"]["retrieval_mode"],
            "grounded_answers",
        )
        self.assertTrue(
            body["artifacts"]["task_agent_spec"]["capabilities"]["citations_required"]
        )
        self.assertTrue(body["artifacts"]["review_summary"]["review_ready"])

        gate_1 = self.client.post(
            f"/v1/workflow-runs/{run.id}/decisions",
            json={"gate": "gate_1_spec_review", "decision": "approve"},
        )
        self.assertEqual(gate_1.status_code, 200)
        self.assertEqual(gate_1.json()["pending_gate"], "gate_2_progression_review")


    def test_openai_customization_ignores_invalid_public_check_paths(self) -> None:
        service = OpenAITaskAgentAuthoringService(enabled=False)
        spec, _origin = build_task_agent_scaffold(
            title="Grounded RAG contract",
            summary="Return grounded answers with citations.",
            design_spec=_design_spec(
                title="Grounded RAG contract",
                problem_statement="Return grounded answers with citations.",
                learning_outcomes=["grounded answers", "citations"],
            ),
        )
        deliverable = spec.deliverables[0]
        original_checks = [check.model_dump(mode="json") for check in deliverable.public_checks]
        updated = service._apply_customization(
            spec,
            TaskAgentCustomization(
                deliverables=[
                    DeliverableCustomization(
                        id=deliverable.id,
                        public_checks=[
                            PublicCheckCustomization(
                                id="invalid_visible_check",
                                request_method="POST",
                                request_path="not-a-real-route",
                                expected_status=200,
                            )
                        ],
                    ),
                ]
            ),
        )
        updated_deliverable = next(item for item in updated.deliverables if item.id == deliverable.id)
        self.assertEqual(
            [check.model_dump(mode="json") for check in updated_deliverable.public_checks],
            original_checks,
        )

        self.assertIn("## Files to edit", snapshot.learner_package.deliverables[0].content_markdown)
        self.assertNotIn("Hidden checkpoint coverage", snapshot.learner_package.deliverables[0].content_markdown)

        events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(events.status_code, 200)
        event_types = [event["event_type"] for event in events.json()]
        self.assertIn("course_run_created", event_types)
        self.assertIn("course_run_synced", event_types)
        self.assertIn("course_run_published", event_types)

        versions = self.client.get(f"/v1/course-runs/{course_run_id}/published-versions")
        self.assertEqual(versions.status_code, 200)
        version_body = versions.json()
        self.assertEqual(len(version_body["versions"]), 1)
        self.assertEqual(version_body["versions"][0]["version"], 1)
        self.assertTrue(version_body["versions"][0]["default_for_new_enrollments"])
        self.assertIn("Initial published version", "\n".join(version_body["versions"][0]["changes"]))

    def test_out_of_scope_course_stays_out_of_ready_to_publish(self) -> None:
        created = self.client.post(
            "/v1/course-runs",
            json={
                "title": "Mobile App Course",
                "summary": "Teach a mobile product build with native iOS and Android UI flows.",
                "package_type": "progressive_codebase_course",
                "deliverables": [
                    {
                        "deliverable_slug": "foundation",
                        "title": "Foundation",
                        "summary": "Introduce native mobile navigation, gestures, and offline-first UI patterns.",
                        "learning_outcomes": ["Understand mobile UI basics."],
                    }
                ],
            },
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        shared_run_id = course_run["shared_workflow_run_id"]
        self.assertIsNotNone(shared_run_id)

        synced = self.client.post(f"/v1/course-runs/{course_run['id']}/sync")
        self.assertEqual(synced.status_code, 200)
        self.assertEqual(synced.json()["stage"], "blocked")

        review = self.client.get(f"/v1/course-runs/{course_run['id']}/review")
        self.assertEqual(review.status_code, 200)
        review_body = review.json()
        self.assertIn("learner-ready", "\n".join(review_body["blockers"]).lower())
        self.assertIn("learner-ready assignment spec", "\n".join(review_body["next_actions"]).lower())

        published = self.client.post(f"/v1/course-runs/{course_run['id']}/publish")
        self.assertEqual(published.status_code, 409)
        self.assertIn("ready", published.json()["detail"].lower())

    def test_grounded_rag_course_can_publish_for_lms(self) -> None:
        created = self.client.post(
            "/v1/course-runs",
            json={
                "title": "Grounded RAG Course",
                "summary": "Teach a grounded retrieval and answer system over a visible corpus.",
                "package_type": "progressive_codebase_course",
                "shared_design_spec": _design_spec(
                    title="Grounded RAG Course",
                    problem_statement="Teach a grounded retrieval and answer system over a visible corpus.",
                    learning_outcomes=["grounded answers", "citations", "abstention"],
                ).model_dump(mode="json"),
                "deliverables": [
                    {
                        "deliverable_slug": "exercise/01-contract",
                        "title": "Grounded answer contract",
                        "summary": "Return grounded answers with citations through a stable run contract.",
                        "learning_outcomes": ["grounded answers", "citation schema"],
                        "design_spec": _design_spec(
                            title="Grounded answer contract",
                            problem_statement="Return grounded answers with citations through a stable run contract.",
                            learning_outcomes=["grounded answers", "citation schema"],
                        ).model_dump(mode="json"),
                    },
                    {
                        "deliverable_slug": "exercise/02-retrieval",
                        "title": "Retrieval quality",
                        "summary": "Retrieve and rank the strongest supporting evidence before answering.",
                        "learning_outcomes": ["retrieval selection", "evidence ranking"],
                        "design_spec": _design_spec(
                            title="Retrieval quality",
                            problem_statement="Retrieve and rank the strongest supporting evidence before answering.",
                            learning_outcomes=["retrieval selection", "evidence ranking"],
                        ).model_dump(mode="json"),
                    },
                    {
                        "deliverable_slug": "exercise/03-abstention",
                        "title": "Abstention and traceability",
                        "summary": "Abstain when support is weak and expose the retrieval path.",
                        "learning_outcomes": ["abstention", "traceability"],
                        "design_spec": _design_spec(
                            title="Abstention and traceability",
                            problem_statement="Abstain when support is weak and expose the retrieval path.",
                            learning_outcomes=["abstention", "traceability"],
                        ).model_dump(mode="json"),
                    },
                    {
                        "deliverable_slug": "final/integrated",
                        "title": "Production final",
                        "summary": "Meet groundedness, latency, and cost goals together.",
                        "learning_outcomes": ["latency", "operating cost"],
                        "design_spec": _design_spec(
                            title="Production final",
                            problem_statement="Meet groundedness, latency, and cost goals together.",
                            learning_outcomes=["latency", "operating cost"],
                        ).model_dump(mode="json"),
                    },
                ],
            },
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        shared_run_id = course_run["shared_workflow_run_id"]
        self.assertIsNotNone(shared_run_id)

        for gate in [
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ]:
            decision = self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
            self.assertEqual(decision.status_code, 200)

        synced = self.client.post(f"/v1/course-runs/{course_run['id']}/sync")
        self.assertEqual(synced.status_code, 200)
        self.assertEqual(synced.json()["stage"], "ready_to_publish")

        published = self.client.post(f"/v1/course-runs/{course_run['id']}/publish")
        self.assertEqual(published.status_code, 200)
        snapshot = app.state.workflow_service.store.get_publish_snapshot(
            published.json()["latest_publish_snapshot_id"]
        )
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.learner_package)
        self.assertIsNotNone(snapshot.task_agent_spec)
        self.assertEqual(snapshot.task_agent_spec.capabilities.retrieval_mode.value, "grounded_answers")
        self.assertTrue(snapshot.task_agent_spec.capabilities.citations_required)
        self.assertIn(
            "data/corpus.json",
            snapshot.learner_package.deliverables[0].visible_files,
        )

    def test_lms_enrollment_workspace_and_submission_flow(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]
        first_deliverable_id = course_run["deliverables"][0]["deliverable_slug"]
        second_deliverable_id = course_run["deliverables"][1]["deliverable_slug"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            decision = self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
            self.assertEqual(decision.status_code, 200)

        synced = self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        self.assertEqual(synced.status_code, 200)
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)
        snapshot_id = published.json()["latest_publish_snapshot_id"]
        snapshot = app.state.workflow_service.store.get_publish_snapshot(snapshot_id)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        snapshot.learner_package.project_brief_markdown = ""
        app.state.workflow_service.store.save_publish_snapshot(snapshot)

        catalog = self.client.get("/v1/lms/catalog")
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(len(catalog.json()["courses"]), 1)
        self.assertTrue(catalog.json()["courses"][0]["supported_for_lms"])
        self.assertEqual(catalog.json()["courses"][0]["deliverable_count"], len(course_run["deliverables"]))

        enrollment = self.client.post(
            "/v1/lms/enrollments",
            json={"course_run_id": course_run_id},
        )
        self.assertEqual(enrollment.status_code, 200)
        enrollment_body = enrollment.json()
        enrollment_id = enrollment_body["id"]
        self.assertEqual(enrollment_body["current_deliverable_id"], first_deliverable_id)
        self.assertTrue(all(deliverable["status"] == "available" for deliverable in enrollment_body["deliverables"]))

        workspace = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/workspace",
            json={"deliverable_id": first_deliverable_id},
        )
        self.assertEqual(workspace.status_code, 200)
        workspace_body = workspace.json()
        first_deliverable = next(
            deliverable
            for deliverable in workspace_body["deliverables"]
            if deliverable["deliverable_id"] == first_deliverable_id
        )
        self.assertEqual(first_deliverable["workspace_session"]["status"], "running")
        self.assertIn("http://127.0.0.1:18080/", first_deliverable["workspace_session"]["editor_url"])
        workspace_root = Path(first_deliverable["workspace_session"]["workspace_root"])
        self.assertTrue((workspace_root / "app.py").exists())
        self.assertTrue((workspace_root / "checks" / "run_visible_checks.py").exists())
        original_app = (workspace_root / "app.py").read_text(encoding="utf-8")

        (workspace_root / "app.py").unlink()
        self.assertFalse((workspace_root / "app.py").exists())

        healed = self.client.get(f"/v1/lms/enrollments/{enrollment_id}/experience")
        self.assertEqual(healed.status_code, 200)
        self.assertTrue((workspace_root / "app.py").exists())
        self.assertEqual((workspace_root / "app.py").read_text(encoding="utf-8"), original_app)

        experience_page = self.client.get(f"/v1/lms/enrollments/{enrollment_id}/experience")
        self.assertEqual(experience_page.status_code, 200)
        experience_body = experience_page.json()
        self.assertIn(course_run["title"], experience_body["project_brief_markdown"])
        self.assertIn("What review will look at", experience_body["project_brief_markdown"])

        experience = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/submit",
            json={"deliverable_id": first_deliverable_id},
        )
        self.assertEqual(experience.status_code, 200)
        experience_body = experience.json()
        self.assertEqual(experience_body["enrollment"]["id"], enrollment_id)
        self.assertGreaterEqual(len(experience_body["submissions"]), 1)
        latest_submission = experience_body["submissions"][0]
        self.assertEqual(latest_submission["status"], "passed")
        self.assertEqual(latest_submission["passed_tests"], latest_submission["total_tests"])

        refreshed = self.client.get(f"/v1/lms/enrollments/{enrollment_id}")
        self.assertEqual(refreshed.status_code, 200)
        refreshed_body = refreshed.json()
        self.assertIsNone(refreshed_body["current_deliverable_id"])
        self.assertEqual(refreshed_body["status"], "completed")
        deliverable_1 = next(
            deliverable
            for deliverable in refreshed_body["deliverables"]
            if deliverable["deliverable_id"] == first_deliverable_id
        )
        deliverable_2 = next(
            deliverable
            for deliverable in refreshed_body["deliverables"]
            if deliverable["deliverable_id"] == second_deliverable_id
        )
        self.assertEqual(deliverable_1["status"], "passed")
        self.assertEqual(deliverable_2["status"], "passed")

    def test_lms_submission_attaches_synthesized_feedback_to_failed_deliverables(self) -> None:
        feedback_service = FakeLearnerFeedbackService()
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeFailingLearnerStudioService(),
            learner_feedback_service=feedback_service,
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        shared_run_id = course_run["shared_workflow_run_id"]
        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            decision = self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
            self.assertEqual(decision.status_code, 200)
        self.client.post(f"/v1/course-runs/{course_run['id']}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run['id']}/publish")
        self.assertEqual(published.status_code, 200)

        enrollment = self.client.post(
            "/v1/lms/enrollments",
            json={"course_run_id": course_run["id"]},
        )
        self.assertEqual(enrollment.status_code, 200)
        enrollment_id = enrollment.json()["id"]

        submitted = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/submit",
            json={},
        )
        self.assertEqual(submitted.status_code, 200)
        body = submitted.json()
        latest_report = body["latest_assignment_report"]
        failed_review_areas = [
            area for area in latest_report["review_areas"] if area["grade_report"]["status"] == "failed"
        ]
        self.assertTrue(failed_review_areas)
        self.assertEqual(feedback_service.calls, 1)
        self.assertIn("feedback", failed_review_areas[0])
        self.assertEqual(
            failed_review_areas[0]["feedback"]["fundamental_gap"],
            "The unsupported query path is still returning the wrong contract.",
        )
        self.assertIn(
            "Focus on the unsupported-query path next.",
            failed_review_areas[0]["feedback"]["learner_feedback"],
        )

    def test_enrollment_prefers_newest_submission_per_deliverable(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
        )
        course = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(course.status_code, 200)
        course_run_id = course.json()["id"]
        shared_run_id = course.json()["shared_workflow_run_id"]
        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            decision = self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
            self.assertEqual(decision.status_code, 200)
        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)

        enrollment = self.client.post(
            "/v1/lms/enrollments",
            json={"course_run_id": course_run_id},
        )
        self.assertEqual(enrollment.status_code, 200)
        enrollment_id = enrollment.json()["id"]
        first_deliverable_id = enrollment.json()["deliverables"][0]["deliverable_id"]

        submit = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/submit",
            json={"deliverable_id": first_deliverable_id},
        )
        self.assertEqual(submit.status_code, 200)
        latest_pass = submit.json()["submissions"][0]
        self.assertEqual(latest_pass["status"], "passed")

        stored_pass = app.state.workflow_service.store.list_learner_submissions(enrollment_id)[0]
        older_failure = stored_pass.model_copy(deep=True)
        older_failure.id = f"{stored_pass.id}_older_failure"
        older_failure.created_at = stored_pass.created_at - timedelta(minutes=5)
        older_failure.status = GradeStatus.failed.value
        older_failure.passed_tests = 0
        older_failure.pass_rate = 0.0
        older_failure.grade_report = older_failure.grade_report.model_copy(
            update={
                "status": GradeStatus.failed,
                "passed_tests": 0,
                "failed_tests": older_failure.grade_report.total_tests,
                "pass_rate": 0.0,
            }
        )
        app.state.workflow_service.store.save_learner_submission(older_failure)

        refreshed = self.client.get(f"/v1/lms/enrollments/{enrollment_id}")
        self.assertEqual(refreshed.status_code, 200)
        first_deliverable = next(
            deliverable
            for deliverable in refreshed.json()["deliverables"]
            if deliverable["deliverable_id"] == first_deliverable_id
        )
        self.assertEqual(first_deliverable["status"], "passed")
        self.assertEqual(first_deliverable["latest_submission"]["id"], stored_pass.id)
        self.assertEqual(first_deliverable["latest_submission"]["status"], "passed")

    def test_lms_workspace_file_api_reads_and_writes_workspace_files(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]
        first_deliverable_id = course_run["deliverables"][0]["deliverable_slug"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            decision = self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
            self.assertEqual(decision.status_code, 200)

        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)

        enrollment = self.client.post(
            "/v1/lms/enrollments",
            json={"course_run_id": course_run_id},
        )
        self.assertEqual(enrollment.status_code, 200)
        enrollment_id = enrollment.json()["id"]

        workspace = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/workspace",
            json={"deliverable_id": first_deliverable_id},
        )
        self.assertEqual(workspace.status_code, 200)

        files = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/files",
            params={"deliverable_id": first_deliverable_id},
        )
        self.assertEqual(files.status_code, 200)
        file_paths = {item["relative_path"] for item in files.json()["files"]}
        self.assertIn("app.py", file_paths)
        self.assertIn("README.md", file_paths)
        self.assertIn("checks/run_visible_checks.py", file_paths)
        self.assertIn(".vscode/tasks.json", file_paths)

        original_app = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": first_deliverable_id, "path": "app.py"},
        )
        self.assertEqual(original_app.status_code, 200)
        self.assertIn("create_app_from_manifest", original_app.json()["content"])

        starter_readme = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": first_deliverable_id, "path": "README.md"},
        )
        self.assertEqual(starter_readme.status_code, 200)
        self.assertIn("## What we are building", starter_readme.json()["content"])
        self.assertIn("## What review will look at", starter_readme.json()["content"])
        self.assertIn("Submit the whole project for review.", starter_readme.json()["content"])

        deliverables_doc = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": first_deliverable_id, "path": "deliverables.md"},
        )
        self.assertEqual(deliverables_doc.status_code, 200)
        self.assertIn("# Project deliverables", deliverables_doc.json()["content"])
        self.assertIn("Use this as the checklist for what review will look at on submission.", deliverables_doc.json()["content"])

        starter_manifest = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": first_deliverable_id, "path": "starter_manifest.json"},
        )
        self.assertEqual(starter_manifest.status_code, 200)
        starter_manifest_payload = json.loads(starter_manifest.json()["content"])
        self.assertIn("public_checks", starter_manifest_payload)
        self.assertIn("public_check_cases", starter_manifest_payload)
        self.assertNotIn("eval_cases", starter_manifest_payload)
        self.assertIn("course_structure", starter_manifest_payload)
        self.assertIn("runtime_dependencies", starter_manifest_payload)
        self.assertIn("capabilities", starter_manifest_payload)
        self.assertEqual(
            starter_manifest_payload["runtime_dependencies"]["editable_files"],
            starter_manifest_payload["learner_starter_surface"]["primary_editable_paths"],
        )
        self.assertEqual(starter_manifest_payload["visible_check_command"], "sh .coursegen/runtime/check_visible.sh")
        self.assertTrue(starter_manifest_payload["public_checks"][0]["expected_assertions"])

        visible_check_script = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": first_deliverable_id, "path": "checks/run_visible_checks.py"},
        )
        self.assertEqual(visible_check_script.status_code, 200)
        self.assertIn("Visible checks passed", visible_check_script.json()["content"])

        updated_app = "from fastapi import FastAPI\n\napp = FastAPI(title='shim')\n"
        write = self.client.put(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            json={
                "deliverable_id": first_deliverable_id,
                "relative_path": "app.py",
                "content": updated_app,
            },
        )
        self.assertEqual(write.status_code, 200)
        self.assertEqual(write.json()["relative_path"], "app.py")

        reread = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": first_deliverable_id, "path": "app.py"},
        )
        self.assertEqual(reread.status_code, 200)
        self.assertEqual(reread.json()["content"], updated_app)

        review_area_index = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": first_deliverable_id, "path": ".coursegen/review_areas/index.json"},
        )
        self.assertEqual(review_area_index.status_code, 200)
        review_area_payload = json.loads(review_area_index.json()["content"])
        self.assertEqual(len(review_area_payload["review_areas"]), len(course_run["deliverables"]))

    def test_lms_workspace_stays_stable_when_switching_focus_deliverables(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]
        first_deliverable_id = course_run["deliverables"][0]["deliverable_slug"]
        second_deliverable_id = course_run["deliverables"][1]["deliverable_slug"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )

        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)

        enrollment = self.client.post(
            "/v1/lms/enrollments",
            json={"course_run_id": course_run_id},
        )
        self.assertEqual(enrollment.status_code, 200)
        enrollment_id = enrollment.json()["id"]

        first_launch = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/workspace",
            json={"deliverable_id": first_deliverable_id},
        )
        self.assertEqual(first_launch.status_code, 200)
        first_session = next(
            deliverable["workspace_session"]
            for deliverable in first_launch.json()["deliverables"]
            if deliverable["deliverable_id"] == first_deliverable_id
        )

        custom_app = "from fastapi import FastAPI\n\napp = FastAPI(title='shared-project')\n"
        write = self.client.put(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            json={
                "deliverable_id": first_deliverable_id,
                "relative_path": "app.py",
                "content": custom_app,
            },
        )
        self.assertEqual(write.status_code, 200)

        second_launch = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/workspace",
            json={"deliverable_id": second_deliverable_id},
        )
        self.assertEqual(second_launch.status_code, 200)
        second_session = next(
            deliverable["workspace_session"]
            for deliverable in second_launch.json()["deliverables"]
            if deliverable["deliverable_id"] == second_deliverable_id
        )
        self.assertEqual(first_session["id"], second_session["id"])
        self.assertEqual(first_session["editor_url"], second_session["editor_url"])

        reread = self.client.get(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"deliverable_id": second_deliverable_id, "path": "app.py"},
        )
        self.assertEqual(reread.status_code, 200)
        self.assertEqual(reread.json()["content"], custom_app)

    def test_lms_workspace_file_api_blocks_path_escape(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]
        first_deliverable_id = course_run["deliverables"][0]["deliverable_slug"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )

        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        enrollment = self.client.post(
            "/v1/lms/enrollments",
            json={"course_run_id": course_run_id},
        )
        enrollment_id = enrollment.json()["id"]

        escape = self.client.put(
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            json={
                "deliverable_id": first_deliverable_id,
                "relative_path": "../outside.py",
                "content": "print('nope')\n",
            },
        )
        self.assertEqual(escape.status_code, 409)
        self.assertIn("must stay inside the learner workspace", escape.json()["detail"])

    def test_lms_catalog_and_enrollment_are_pinned_to_publish_snapshot(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            decision = self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
            self.assertEqual(decision.status_code, 200)

        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)
        snapshot_id = published.json()["latest_publish_snapshot_id"]
        self.assertIsNotNone(snapshot_id)
        snapshot = app.state.workflow_service.store.get_publish_snapshot(snapshot_id)
        assert snapshot is not None
        expected_deliverable_title = snapshot.learner_package.deliverables[0].title

        original_catalog = self.client.get("/v1/lms/catalog")
        self.assertEqual(original_catalog.status_code, 200)
        original_title = original_catalog.json()["courses"][0]["title"]
        original_summary = original_catalog.json()["courses"][0]["summary"]

        stored_course = app.state.workflow_service.store.get_course_run(course_run_id)
        assert stored_course is not None
        stored_course.title = "Mutated draft title"
        stored_course.summary = "Mutated draft summary"
        app.state.workflow_service.store.save_course_run(stored_course)

        stored_workflow = app.state.workflow_service.store.get_run(shared_run_id)
        assert stored_workflow is not None
        stored_workflow.artifacts.task_agent_spec.deliverables[0].title = "Mutated live deliverable"
        app.state.workflow_service.store.save_run(stored_workflow)

        catalog = self.client.get("/v1/lms/catalog")
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json()["courses"][0]["publish_snapshot_id"], snapshot_id)
        self.assertEqual(catalog.json()["courses"][0]["title"], original_title)
        self.assertEqual(catalog.json()["courses"][0]["summary"], original_summary)

        enrollment = self.client.post("/v1/lms/enrollments", json={"course_run_id": course_run_id})
        self.assertEqual(enrollment.status_code, 200)
        enrollment_body = enrollment.json()
        self.assertEqual(enrollment_body["publish_snapshot_id"], snapshot_id)
        self.assertEqual(enrollment_body["deliverables"][0]["title"], expected_deliverable_title)

        versions = self.client.get(f"/v1/course-runs/{course_run_id}/published-versions")
        self.assertEqual(versions.status_code, 200)
        version_body = versions.json()
        self.assertEqual(version_body["versions"][0]["learner_count"], 1)
        self.assertEqual(version_body["versions"][0]["snapshot_id"], snapshot_id)

    def test_creator_and_learner_testing_views_capture_feedback_and_eval_report(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]

        creator_feedback = self.client.post(
            f"/v1/course-runs/{course_run_id}/feedback",
            json={
                "summary": "Deliverable plan feels close.",
                "details": "The first deliverable is clear, but I want to watch the later deliverables closely.",
                "category": "deliverable-plan",
                "deliverable_slug": course_run["deliverables"][0]["deliverable_slug"],
            },
        )
        self.assertEqual(creator_feedback.status_code, 200)

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)
        snapshot_id = published.json()["latest_publish_snapshot_id"]
        snapshot = app.state.workflow_service.store.get_publish_snapshot(snapshot_id)
        assert snapshot is not None

        enrollment = self.client.post(
            "/v1/lms/enrollments",
            json={"course_run_id": course_run_id, "learner_id": "test-learner"},
        )
        self.assertEqual(enrollment.status_code, 200)
        enrollment_id = enrollment.json()["id"]

        learner_feedback = self.client.post(
            f"/v1/lms/enrollments/{enrollment_id}/feedback",
            json={
                "summary": "The starter is easy to understand.",
                "details": "README and deliverable content were enough to get moving.",
            },
        )
        self.assertEqual(learner_feedback.status_code, 200)

        first_deliverable = snapshot.learner_package.deliverables[0]
        report = self.client.post(
            f"/v1/course-runs/{course_run_id}/learner-eval",
            json={
                "publish_snapshot_id": snapshot_id,
                "learner_id": "test-learner",
                "enrollment_id": enrollment_id,
                "deliverable_results": [
                    {
                        "deliverable_id": first_deliverable.deliverable_id,
                        "title": first_deliverable.title,
                        "deliverable_index": first_deliverable.deliverable_index,
                        "learner_visible_files": first_deliverable.visible_files,
                        "bad_attempt": {
                            "status": "failed",
                            "passed_tests": 0,
                            "total_tests": 1,
                            "pass_rate": 0.0,
                        },
                        "good_attempt": {
                            "status": "passed",
                            "passed_tests": 1,
                            "total_tests": 1,
                            "pass_rate": 1.0,
                        },
                        "next_deliverable_id": snapshot.learner_package.deliverables[1].deliverable_id,
                        "progression_observed": True,
                        "course_completed": False,
                    }
                ],
            },
        )
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["overall_status"], "passed")

        creator_view = self.client.get(f"/v1/course-runs/{course_run_id}/creator-view")
        self.assertEqual(creator_view.status_code, 200)
        creator_body = creator_view.json()
        self.assertEqual(creator_body["creator_feedback"][0]["summary"], "Deliverable plan feels close.")
        self.assertEqual(creator_body["latest_learner_evaluation"]["publish_snapshot_id"], snapshot_id)
        self.assertIsNotNone(creator_body["creator_choices"])
        self.assertGreaterEqual(len(creator_body["diagnostics"]), 1)

        learner_view = self.client.get(f"/v1/lms/enrollments/{enrollment_id}/learner-view")
        self.assertEqual(learner_view.status_code, 200)
        learner_body = learner_view.json()
        self.assertEqual(learner_body["feedback"][0]["summary"], "The starter is easy to understand.")

    def test_creator_view_exposes_machine_readable_diagnostics_for_blocked_draft(self) -> None:
        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]

        stored = app.state.workflow_service.store.get_course_run(course_run_id)
        assert stored is not None
        stored.last_error = "Docker sandbox verification failed for the shared workflow."
        app.state.workflow_service.store.save_course_run(stored)

        creator_view = self.client.get(f"/v1/course-runs/{course_run_id}/creator-view")
        self.assertEqual(creator_view.status_code, 200)
        body = creator_view.json()
        diagnostic_codes = {item["code"] for item in body["diagnostics"]}
        self.assertIn("course_action_failed", diagnostic_codes)
        self.assertIn("review_blocked", diagnostic_codes)

    def test_creator_view_handles_legacy_progressive_spec_without_shared_codebase_flag(self) -> None:
        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_body = created.json()
        course_run_id = course_body["id"]
        shared_run_id = course_body["shared_workflow_run_id"]
        self.assertIsNotNone(shared_run_id)

        with app.state.workflow_service.store._session() as connection:
            row = connection.execute(
                "SELECT payload_json FROM workflow_runs WHERE run_id = ?",
                (shared_run_id,),
            ).fetchone()
            assert row is not None
            payload = json.loads(row["payload_json"])
            course_structure = payload["artifacts"]["task_agent_spec"]["course_structure"]
            course_structure["shared_codebase"] = False
            course_structure["workspace_scope"] = "per_deliverable_workspace"
            connection.execute(
                "UPDATE workflow_runs SET payload_json = ? WHERE run_id = ?",
                (json.dumps(payload), shared_run_id),
            )
            connection.commit()

        creator_view = self.client.get(f"/v1/course-runs/{course_run_id}/creator-view")
        self.assertEqual(creator_view.status_code, 200)
        body = creator_view.json()
        self.assertEqual(body["course_run"]["id"], course_run_id)
        self.assertEqual(body["review"]["shared_workflow_run_id"], shared_run_id)

    def test_creator_view_and_draft_list_include_ai_spend(self) -> None:
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=FakeLivePlanner(),
        )
        response = self.client.post(
            "/v1/course-runs/generate",
            json={
                "goal": "Build a production-ready feature flag service that evaluates rollout requests and uses tools safely.",
                "learning_outcomes": ["tool selection", "observability"],
            },
        )
        self.assertEqual(response.status_code, 200)
        course_run_id = response.json()["course_run"]["id"]

        creator_view = self.client.get(f"/v1/course-runs/{course_run_id}/creator-view")
        self.assertEqual(creator_view.status_code, 200)
        body = creator_view.json()
        self.assertGreater(body["course_run"]["ai_usage"]["estimated_cost_usd"], 0.0)
        self.assertEqual(body["course_run"]["ai_usage"]["request_count"], 1)

        listed = self.client.get("/v1/course-runs")
        self.assertEqual(listed.status_code, 200)
        matching = next(run for run in listed.json()["runs"] if run["id"] == course_run_id)
        self.assertGreater(matching["ai_usage"]["estimated_cost_usd"], 0.0)

    def test_create_revision_produces_new_draft_without_replacing_published_catalog_entry(self) -> None:
        app.state.lms_service = LMSService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            learner_studio_service=FakeLearnerStudioService(),
            base_dir=f"{self.temp_dir.name}/learner-workspaces",
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        original = created.json()
        course_run_id = original["id"]
        shared_run_id = original["shared_workflow_run_id"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)
        published_snapshot = published.json()["latest_publish_snapshot_id"]

        revision = self.client.post(f"/v1/course-runs/{course_run_id}/create-revision")
        self.assertEqual(revision.status_code, 200)
        revision_body = revision.json()
        self.assertNotEqual(revision_body["id"], course_run_id)
        self.assertEqual(revision_body["course_family_id"], original["course_family_id"])
        self.assertEqual(revision_body["status"], "awaiting_human")
        self.assertEqual(revision_body["stage"], "awaiting_course_review")
        self.assertNotEqual(revision_body["shared_workflow_run_id"], shared_run_id)

        versions = self.client.get(f"/v1/course-runs/{revision_body['id']}/published-versions")
        self.assertEqual(versions.status_code, 200)
        version_body = versions.json()
        self.assertEqual(len(version_body["versions"]), 1)
        self.assertEqual(version_body["versions"][0]["snapshot_id"], published_snapshot)

        catalog = self.client.get("/v1/lms/catalog")
        self.assertEqual(catalog.status_code, 200)
        catalog_body = catalog.json()
        self.assertEqual(len(catalog_body["courses"]), 1)
        self.assertEqual(catalog_body["courses"][0]["publish_snapshot_id"], published_snapshot)
        self.assertEqual(catalog_body["courses"][0]["course_run_id"], course_run_id)

    def test_queue_revision_persists_placeholder_before_background_work(self) -> None:
        queued_jobs: list[object] = []
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            job_runner=lambda job: queued_jobs.append(job),
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        original = created.json()
        course_run_id = original["id"]
        shared_run_id = original["shared_workflow_run_id"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
        self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)

        revision = self.client.post(f"/v1/course-runs/{course_run_id}/create-revision-async")
        self.assertEqual(revision.status_code, 200)
        revision_body = revision.json()
        self.assertTrue(revision_body["queued"])
        self.assertEqual(revision_body["course_run"]["stage"], "drafting")
        self.assertEqual(revision_body["course_run"]["status"], "active")
        self.assertEqual(revision_body["course_run"]["deliverables"], [])
        self.assertEqual(revision_body["course_run"]["course_family_id"], original["course_family_id"])
        self.assertEqual(len(queued_jobs), 1)

        revision_id = revision_body["course_run"]["id"]
        events = self.client.get(f"/v1/course-runs/{revision_id}/events")
        self.assertEqual(events.status_code, 200)
        event_types = [event["event_type"] for event in events.json()]
        self.assertIn("course_revision_queued", event_types)
        self.assertIn("course_revision_started", event_types)

        queued_jobs[0]()

        completed = self.client.get(f"/v1/course-runs/{revision_id}")
        self.assertEqual(completed.status_code, 200)
        completed_body = completed.json()
        self.assertEqual(completed_body["status"], "awaiting_human")
        self.assertEqual(completed_body["stage"], "awaiting_course_review")
        self.assertNotEqual(completed_body["shared_workflow_run_id"], shared_run_id)
        self.assertGreaterEqual(len(completed_body["deliverables"]), 1)

        completed_events = self.client.get(f"/v1/course-runs/{revision_id}/events")
        self.assertEqual(completed_events.status_code, 200)
        completed_event_types = [event["event_type"] for event in completed_events.json()]
        self.assertIn("course_revision_completed", completed_event_types)

    def test_queue_course_materialize_persists_operation_before_background_work(self) -> None:
        queued_jobs: list[object] = []
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            job_runner=lambda job: queued_jobs.append(job),
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]

        queued = self.client.post(
            f"/v1/course-runs/{course_run_id}/materialize-async",
            json={"overwrite": True},
        )
        self.assertEqual(queued.status_code, 200)
        body = queued.json()
        self.assertTrue(body["queued"])
        self.assertEqual(body["operation"], "materialize")
        self.assertEqual(body["course_run"]["active_operation"], "materialize")
        self.assertEqual(len(queued_jobs), 1)

        events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(events.status_code, 200)
        event_types = [event["event_type"] for event in events.json()]
        self.assertIn("course_materialize_queued", event_types)
        self.assertIn("course_materialize_started", event_types)

        queued_jobs[0]()

        completed = self.client.get(f"/v1/course-runs/{course_run_id}")
        self.assertEqual(completed.status_code, 200)
        completed_body = completed.json()
        self.assertIsNone(completed_body["active_operation"])
        self.assertIsNotNone(completed_body["materialized_bundle"])

        completed_events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(completed_events.status_code, 200)
        completed_event_types = [event["event_type"] for event in completed_events.json()]
        self.assertIn("course_bundle_materialized", completed_event_types)
        self.assertIn("course_materialize_completed", completed_event_types)

    def test_queue_course_publish_persists_operation_before_background_work(self) -> None:
        queued_jobs: list[object] = []
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            job_runner=lambda job: queued_jobs.append(job),
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]
        shared_run_id = created.json()["shared_workflow_run_id"]

        for gate in (
            "gate_1_spec_review",
            "gate_2_progression_review",
            "gate_3_pre_publish",
        ):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
        self.client.post(f"/v1/course-runs/{course_run_id}/sync")

        queued = self.client.post(f"/v1/course-runs/{course_run_id}/publish-async")
        self.assertEqual(queued.status_code, 200)
        body = queued.json()
        self.assertTrue(body["queued"])
        self.assertEqual(body["operation"], "publish")
        self.assertEqual(body["course_run"]["active_operation"], "publish")
        self.assertEqual(body["course_run"]["stage"], "ready_to_publish")
        self.assertEqual(len(queued_jobs), 1)

        events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(events.status_code, 200)
        event_types = [event["event_type"] for event in events.json()]
        self.assertIn("course_publish_queued", event_types)
        self.assertIn("course_publish_started", event_types)

        queued_jobs[0]()

        completed = self.client.get(f"/v1/course-runs/{course_run_id}")
        self.assertEqual(completed.status_code, 200)
        completed_body = completed.json()
        self.assertEqual(completed_body["status"], "published")
        self.assertEqual(completed_body["stage"], "published")
        self.assertIsNone(completed_body["active_operation"])
        self.assertIsNotNone(completed_body["latest_publish_snapshot_id"])

        completed_events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(completed_events.status_code, 200)
        completed_event_types = [event["event_type"] for event in completed_events.json()]
        self.assertIn("course_run_published", completed_event_types)
        self.assertIn("course_publish_completed", completed_event_types)

    def test_publish_saves_learner_certification_report_when_exact_learner_path_passes(self) -> None:
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            publish_certification_service=PublishLearnerCertificationService(
                learner_studio_service=FakeLearnerStudioService(),
                base_dir=f"{self.temp_dir.name}/publish-cert",
                enabled=True,
            ),
            creator_asset_service=self.creator_asset_service,
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]
        shared_run_id = created.json()["shared_workflow_run_id"]

        for gate in ("gate_1_spec_review", "gate_2_progression_review", "gate_3_pre_publish"):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
        self.client.post(f"/v1/course-runs/{course_run_id}/sync")

        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 200)
        snapshot_id = published.json()["latest_publish_snapshot_id"]
        snapshot = app.state.workflow_service.store.get_publish_snapshot(snapshot_id)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.learner_certification)
        assert snapshot.learner_certification is not None
        self.assertTrue(snapshot.learner_certification.passed)
        check_keys = {check.key for check in snapshot.learner_certification.checks}
        self.assertIn("workspace_seeded", check_keys)
        self.assertIn("editor_launch", check_keys)
        self.assertIn("grading_completed", check_keys)
        self.assertIn("deliverable_mapping", check_keys)

    def test_publish_routes_repairable_learner_certification_failure_back_into_shared_workflow_revision(self) -> None:
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            publish_certification_service=PublishLearnerCertificationService(
                learner_studio_service=FakeBrokenAppLearnerStudioService(),
                base_dir=f"{self.temp_dir.name}/publish-cert",
                enabled=True,
            ),
            creator_asset_service=self.creator_asset_service,
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]
        shared_run_id = created.json()["shared_workflow_run_id"]

        for gate in ("gate_1_spec_review", "gate_2_progression_review", "gate_3_pre_publish"):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
        synced = self.client.post(f"/v1/course-runs/{course_run_id}/sync")
        self.assertEqual(synced.status_code, 200)
        self.assertEqual(synced.json()["stage"], "ready_to_publish")

        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 409)
        self.assertIn("routed the shared assignment workflow back into revision", published.json()["detail"])

        refreshed = self.client.get(f"/v1/course-runs/{course_run_id}")
        self.assertEqual(refreshed.status_code, 200)
        refreshed_body = refreshed.json()
        self.assertEqual(refreshed_body["stage"], "awaiting_course_review")
        self.assertNotEqual(refreshed_body["shared_workflow_run_id"], shared_run_id)
        self.assertIn("Learner-path certification failed before publish", refreshed_body["last_error"])

        revised_shared_run = app.state.workflow_service.get_run(refreshed_body["shared_workflow_run_id"])
        self.assertIsNotNone(revised_shared_run)
        assert revised_shared_run is not None
        node_kinds = [node.kind for node in revised_shared_run.artifacts.node_executions]
        self.assertIn(WorkflowNodeKind.reviewer_learner_runtime, node_kinds)
        learner_runtime_nodes = [
            node
            for node in revised_shared_run.artifacts.node_executions
            if node.kind == WorkflowNodeKind.reviewer_learner_runtime
        ]
        self.assertTrue(any("ImportError" in finding.detail for node in learner_runtime_nodes for finding in node.findings))

        course_events = self.client.get(f"/v1/course-runs/{course_run_id}/events")
        self.assertEqual(course_events.status_code, 200)
        event_types = [event["event_type"] for event in course_events.json()]
        self.assertIn("course_publish_certification_failed", event_types)

    def test_publish_blocks_platform_learner_certification_failure_without_routing_to_revision(self) -> None:
        app.state.course_workflow_service = CourseWorkflowService(
            app.state.workflow_service.store,
            app.state.workflow_service,
            CourseArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            publish_certification_service=PublishLearnerCertificationService(
                learner_studio_service=FakePlatformBrokenLearnerStudioService(),
                base_dir=f"{self.temp_dir.name}/publish-cert",
                enabled=True,
            ),
            creator_asset_service=self.creator_asset_service,
        )
        app.state.course_generation_service = CourseGenerationService(
            app.state.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
        )

        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]
        shared_run_id = created.json()["shared_workflow_run_id"]

        for gate in ("gate_1_spec_review", "gate_2_progression_review", "gate_3_pre_publish"):
            self.client.post(
                f"/v1/workflow-runs/{shared_run_id}/decisions",
                json={"gate": gate, "decision": "approve"},
            )
        self.client.post(f"/v1/course-runs/{course_run_id}/sync")

        published = self.client.post(f"/v1/course-runs/{course_run_id}/publish")
        self.assertEqual(published.status_code, 409)
        self.assertIn("editor could not launch", published.json()["detail"].lower())

        refreshed = self.client.get(f"/v1/course-runs/{course_run_id}")
        self.assertEqual(refreshed.status_code, 200)
        refreshed_body = refreshed.json()
        self.assertEqual(refreshed_body["stage"], "ready_to_publish")
        self.assertEqual(refreshed_body["shared_workflow_run_id"], shared_run_id)
        self.assertIn("editor could not launch", refreshed_body["last_error"].lower())

    def test_survey_course_materialization_creates_author_bundle(self) -> None:
        stateful_design = _design_spec(
            title="TinyURL",
            problem_statement="Build a URL shortener with collision resistance and concurrency safety.",
            learning_outcomes=["idempotency", "concurrency"],
            package_type=PackageType.survey_course,
        )
        support_design = _design_spec(
            title="Support triage agent",
            problem_statement="Build a support triage agent with tools, approvals, and observability.",
            learning_outcomes=["tool selection", "observability"],
            package_type=PackageType.survey_course,
        )
        created = self.client.post(
            "/v1/course-runs",
            json={
                "title": "Backend Systems Survey",
                "summary": "A survey course across independent backend assignments.",
                "package_type": "survey_course",
                "deliverables": [
                    {
                        "deliverable_slug": "tinyurl",
                        "title": "TinyURL",
                        "summary": "Build a URL shortener with collision resistance and concurrency safety.",
                        "design_spec": stateful_design.model_dump(mode="json"),
                    },
                    {
                        "deliverable_slug": "support-agent",
                        "title": "Support triage agent",
                        "summary": "Build a support triage agent with tools, approvals, and observability.",
                        "design_spec": support_design.model_dump(mode="json"),
                        "domain_pack_hint": "generic_project_example",
                    },
                ],
            },
        )
        self.assertEqual(created.status_code, 200)
        course_run_id = created.json()["id"]

        materialize = self.client.post(
            f"/v1/course-runs/{course_run_id}/materialize",
            json={"overwrite": True},
        )
        self.assertEqual(materialize.status_code, 200)
        bundle = materialize.json()["materialized_bundle"]
        self.assertTrue(bundle["root_dir"].endswith(course_run_id))
        self.assertGreater(len(bundle["files"]), 4)

        syllabus = self.client.get(
            f"/v1/course-runs/{course_run_id}/bundle/file",
            params={"path": "public/content/syllabus.md"},
        )
        self.assertEqual(syllabus.status_code, 200)
        self.assertIn("TinyURL", syllabus.json()["content"])
        self.assertIn("Support triage agent", syllabus.json()["content"])

        review = self.client.get(
            f"/v1/course-runs/{course_run_id}/bundle/file",
            params={"path": "public/content/review.md"},
        )
        self.assertEqual(review.status_code, 200)
        self.assertIn("Course Review", review.json()["content"])
        self.assertIn("TinyURL", review.json()["content"])

    def test_progressive_course_materialization_tracks_shared_workflow(self) -> None:
        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]

        child_bundle = self.client.post(
            f"/v1/workflow-runs/{shared_run_id}/materialize",
            json={"overwrite": True},
        )
        self.assertEqual(child_bundle.status_code, 200)

        materialize = self.client.post(
            f"/v1/course-runs/{course_run_id}/materialize",
            json={"overwrite": True},
        )
        self.assertEqual(materialize.status_code, 200)

        deliverable_doc = self.client.get(
            f"/v1/course-runs/{course_run_id}/bundle/file",
            params={"path": "public/content/deliverables/exercise/01-observability.md"},
        )
        self.assertEqual(deliverable_doc.status_code, 200)
        self.assertIn(shared_run_id, deliverable_doc.json()["content"])
        self.assertIn("Bundle available: `True`", deliverable_doc.json()["content"])

        private_snapshot = self.client.get(
            f"/v1/course-runs/{course_run_id}/bundle/file",
            params={"path": "private/linked_workflow_runs.json"},
        )
        self.assertEqual(private_snapshot.status_code, 200)
        self.assertIn(shared_run_id, private_snapshot.json()["content"])

        private_review = self.client.get(
            f"/v1/course-runs/{course_run_id}/bundle/file",
            params={"path": "private/review_report.json"},
        )
        self.assertEqual(private_review.status_code, 200)
        self.assertIn(shared_run_id, private_review.json()["content"])

    def test_materialize_workflow_bundle_and_read_file(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        ).json()
        run_id = created["id"]

        materialize = self.client.post(
            f"/v1/workflow-runs/{run_id}/materialize",
            json={"overwrite": True},
        )
        self.assertEqual(materialize.status_code, 200)
        bundle = materialize.json()["artifacts"]["materialized_bundle"]
        self.assertTrue(bundle["root_dir"].endswith(run_id))
        self.assertGreater(len(bundle["files"]), 5)

        manifest = self.client.get(f"/v1/workflow-runs/{run_id}/bundle")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.json()["bundle_id"], f"{run_id}_bundle")

        readme = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "public/README.md"},
        )
        self.assertEqual(readme.status_code, 200)
        self.assertIn("Feature flag service", readme.json()["content"])

        starter = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "public/starter/deliverable_1/app.py"},
        )
        self.assertEqual(starter.status_code, 200)
        self.assertIn("create_app_from_manifest", starter.json()["content"])

        runtime_helper = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "public/runtime/task_agent_runtime.py"},
        )
        self.assertEqual(runtime_helper.status_code, 200)
        self.assertIn("COURSE_GEN_TASK_AGENT_RUNTIME", runtime_helper.json()["content"])

        grading_guide = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "public/content/deliverable_1_grading.md"},
        )
        self.assertEqual(grading_guide.status_code, 200)
        self.assertIn("Grading Guide", grading_guide.json()["content"])

        private_plan = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "private/grader_plans/deliverable_1.json"},
        )
        self.assertEqual(private_plan.status_code, 200)
        self.assertIn('"deliverable_id": "deliverable_1"', private_plan.json()["content"])

        runtime_dockerfile = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "public/runtime/Dockerfile"},
        )
        self.assertEqual(runtime_dockerfile.status_code, 200)
        self.assertIn("verify_assignment.py", runtime_dockerfile.json()["content"])

        node_report = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "private/node_executions.json"},
        )
        self.assertEqual(node_report.status_code, 200)
        self.assertIn("authoring_runtime", node_report.json()["content"])

        review_summary = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "private/review_summary.json"},
        )
        self.assertEqual(review_summary.status_code, 200)
        self.assertIn("\"review_ready\": true", review_summary.json()["content"])

    def test_invalid_task_agent_spec_repairs_dataset_binding_even_if_review_blockers_remain(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        ).json()
        run_id = created["id"]
        spec = created["artifacts"]["task_agent_spec"]
        spec["qualities"][0]["test"]["dataset_id"] = "missing_dataset"

        update = self.client.put(f"/v1/workflow-runs/{run_id}/task-agent-spec", json=spec)
        self.assertEqual(update.status_code, 200)
        updated = update.json()
        self.assertFalse(updated["artifacts"]["validation_summary"]["valid"])
        self.assertEqual(
            updated["artifacts"]["task_agent_spec"]["qualities"][0]["test"]["dataset_id"],
            updated["artifacts"]["task_agent_spec"]["eval_dataset"]["id"],
        )
        self.assertIn("reviewer_repair", [node["kind"] for node in updated["artifacts"]["node_executions"]])
        self.assertTrue(any("Auto-repair" in note for note in updated["artifacts"]["notes"]))
        materialize = self.client.post(f"/v1/workflow-runs/{run_id}/materialize", json={"overwrite": True})
        self.assertEqual(materialize.status_code, 409)

    def test_bundle_file_endpoint_blocks_path_traversal(self) -> None:
        created = self.client.post(
            "/v1/workflow-runs",
            json={
                "intake": {
                    "title": "Feature flag service",
                    "problem_statement": "Build an agent that evaluates rollout requests, uses tools, drafts replies, escalates edge cases, and is production ready.",
                    "learning_outcomes": ["tool selection", "observability"],
                }
            },
        ).json()
        run_id = created["id"]
        materialize = self.client.post(
            f"/v1/workflow-runs/{run_id}/materialize",
            json={"overwrite": True},
        )
        self.assertEqual(materialize.status_code, 200)

        escaped = self.client.get(
            f"/v1/workflow-runs/{run_id}/bundle/file",
            params={"path": "../outside.txt"},
        )
        self.assertEqual(escaped.status_code, 400)
        self.assertIn("outside the bundle root", escaped.json()["detail"])


if __name__ == "__main__":
    unittest.main()
