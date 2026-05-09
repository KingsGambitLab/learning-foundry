from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.domain.sandbox import (
    DeliverableSandboxReport,
    SandboxAvailability,
    SandboxExecutionResult,
    SandboxExecutionStatus,
)
from app.domain.workflow import (
    ReviewerFinding,
    ReviewerFindingSeverity,
    WorkflowNodeExecution,
    WorkflowNodeKind,
    WorkflowNodeStatus,
)
from app.main import app
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.course_artifact_materializer import CourseArtifactMaterializer
from app.services.course_generation_service import CourseGenerationService
from app.services.course_workflow_service import CourseWorkflowService
from app.services.creator_asset_service import CreatorAssetService
from app.services.langgraph_assignment_graph import LangGraphAssignmentGraph
from app.services.openai_course_planner import OpenAICoursePlanner
from app.services.openai_learner_feedback import OpenAILearnerFeedbackService
from app.services.openai_task_agent_authoring import OpenAITaskAgentAuthoringService
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class FakeSandboxRunner:
    def status(self) -> SandboxAvailability:
        return SandboxAvailability(
            available=True,
            message="Fake Docker sandbox is ready.",
            docker_version="test",
        )

    def execute(self, run) -> SandboxExecutionResult:
        reports = []
        if run.artifacts.task_agent_spec is not None:
            for deliverable in run.artifacts.task_agent_spec.deliverables:
                reports.append(
                    DeliverableSandboxReport(
                        deliverable_id=deliverable.id,
                        compile_succeeded=True,
                        runtime_succeeded=True,
                        health_status_code=200,
                        stdout="sandbox ok",
                        stderr="",
                        error=None,
                    )
                )
        return SandboxExecutionResult(
            status=SandboxExecutionStatus.passed,
            available=True,
            build_succeeded=True,
            run_succeeded=True,
            generated_at=datetime.now(UTC),
            duration_ms=5,
            workspace_root="/tmp/fake-sandbox",
            image_tag="fake-image",
            build_command=["docker", "build"],
            run_command=["docker", "run"],
            build_stdout="build ok",
            build_stderr="",
            run_stdout='{"success": true}',
            run_stderr="",
            deliverable_reports=reports,
            error=None,
        )


class DraftTimelineApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        store = SQLiteWorkflowStore(db_path=f"{self.temp_dir.name}/test.db")
        self.fake_sandbox_runner = FakeSandboxRunner()
        self.workspace_manager = AssignmentWorkspaceManager(base_dir=f"{self.temp_dir.name}/workspaces")
        self.workspace_authoring_service = TaskAgentWorkspaceAuthoringService(self.workspace_manager)
        self.disabled_authoring_service = OpenAITaskAgentAuthoringService(enabled=False)
        self.creator_asset_service = CreatorAssetService(
            store,
            base_dir=f"{self.temp_dir.name}/creator-assets",
        )
        app.state.docker_sandbox_runner = self.fake_sandbox_runner
        app.state.task_agent_workspace_authoring_service = self.workspace_authoring_service
        app.state.assignment_node_runtime = LangGraphAssignmentGraph(
            self.fake_sandbox_runner,
            workspace_authoring_service=self.workspace_authoring_service,
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

    def test_draft_timeline_page_renders_shell(self) -> None:
        response = self.client.get("/draft-timeline?draft=course_demo123")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        body = response.text
        self.assertIn("Inspect a draft's flow", body)
        self.assertIn("Back and forth", body)
        self.assertIn('/static/draft-timeline.css', body)
        self.assertIn('/static/draft-timeline.js', body)
        self.assertIn('id="draft-timeline-state"', body)
        self.assertIn("course_demo123", body)

    def test_course_timeline_merges_course_events_workflow_events_and_node_executions(self) -> None:
        created = self.client.post(
            "/v1/course-runs",
            json={"pattern_slug": "tusharbisht-cs-demo-agent-to-production"},
        )
        self.assertEqual(created.status_code, 200)
        course_run = created.json()
        course_run_id = course_run["id"]
        shared_run_id = course_run["shared_workflow_run_id"]
        self.assertIsNotNone(shared_run_id)

        stored_workflow = app.state.workflow_service.store.get_run(shared_run_id)
        assert stored_workflow is not None
        stored_workflow.artifacts.node_executions.append(
            WorkflowNodeExecution(
                node_id="node_reviewer_tests_demo",
                kind=WorkflowNodeKind.reviewer_tests,
                status=WorkflowNodeStatus.failed,
                attempt=2,
                summary="Tests reviewer found placeholder learner scenarios.",
                created_at=datetime.now(UTC),
                findings=[
                    ReviewerFinding(
                        category="tests",
                        severity=ReviewerFindingSeverity.error,
                        title="Placeholder scenario detected",
                        detail="Replace generic scenarios with domain-specific learner cases.",
                    )
                ],
            )
        )
        app.state.workflow_service.store.save_run(stored_workflow)

        response = self.client.get(f"/v1/course-runs/{course_run_id}/timeline")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["course_run"]["id"], course_run_id)
        self.assertEqual(body["shared_workflow_run_id"], shared_run_id)
        self.assertIn(shared_run_id, body["linked_workflow_run_ids"])
        source_kinds = {item["source_kind"] for item in body["items"]}
        self.assertIn("course_event", source_kinds)
        self.assertIn("workflow_event", source_kinds)
        self.assertIn("workflow_node", source_kinds)
        node_item = next(
            item
            for item in body["items"]
            if item["source_kind"] == "workflow_node" and item["event_type"] == "reviewer_tests"
        )
        self.assertEqual(node_item["event_type"], "reviewer_tests")
        self.assertEqual(node_item["attempt"], 2)
        self.assertIn("Placeholder scenario detected", node_item["detail"])
