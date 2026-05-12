from __future__ import annotations

import tempfile
import unittest

from app.domain.course import CourseAsyncOperation, CreateCourseFromCreatorPlanRequest, GenerateCreatorCoursePlanRequest
from app.domain.workflow import HILGate, WorkflowStage, WorkflowStatus
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.course_generation_service import CourseGenerationService
from app.services.course_workflow_service import CourseWorkflowService
from app.services.openai_course_planner import OpenAICoursePlanner
from app.services.openai_task_agent_authoring import OpenAITaskAgentAuthoringService
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class CourseGenerationAsyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        store = SQLiteWorkflowStore(db_path=f"{self.temp_dir.name}/test.db")
        workspace_manager = AssignmentWorkspaceManager(base_dir=f"{self.temp_dir.name}/workspaces")
        workflow_service = WorkflowService(
            store,
            materializer=ArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            runner=TaskAgentBlackBoxRunner(),
            task_agent_authoring_service=OpenAITaskAgentAuthoringService(enabled=False),
            workspace_manager=workspace_manager,
        )
        self.course_workflow_service = CourseWorkflowService(
            store,
            workflow_service,
            job_runner=lambda job: job(),
        )
        self.course_generation_service = CourseGenerationService(
            self.course_workflow_service,
            live_planner=OpenAICoursePlanner(enabled=False),
            job_runner=lambda job: job(),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_async_creator_plan_generation_stops_at_first_human_gate(self) -> None:
        plan_response = self.course_generation_service.generate_creator_plan(
            GenerateCreatorCoursePlanRequest(
                goal=(
                    "Build a multi-warehouse inventory reservation service that is production ready. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
                creator_choices={
                    "starter_type": "partial",
                    "implementation_language": "python",
                    "application_framework": "fastapi",
                    "primary_database": "postgres",
                    "cache_backend": "redis",
                    "tech_stack": ["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"],
                },
            )
        )

        queued = self.course_generation_service.queue_course_run_from_creator_plan(
            CreateCourseFromCreatorPlanRequest(plan=plan_response.plan)
        )
        course_run = self.course_workflow_service.get_run(queued.course_run.id)
        assert course_run is not None
        self.assertIsNone(course_run.active_operation)
        self.assertEqual(course_run.stage.value, "awaiting_course_review")
        self.assertEqual(course_run.status.value, "awaiting_human")
        self.assertIsNotNone(course_run.shared_design_spec)
        self.assertTrue(course_run.shared_design_spec.project_contract.runtime_plan.services)
        creator_view = self.course_workflow_service.creator_view(course_run.id)
        self.assertEqual(creator_view.course_run.stage.value, "awaiting_course_review")
        self.assertEqual(creator_view.course_run.status.value, "awaiting_human")
        self.assertIsNotNone(course_run.shared_workflow_run_id)
        self.assertEqual(creator_view.review.counts.deliverables_with_blockers, 0)
        self.assertEqual(creator_view.review.blockers, [])
        diagnostic_codes = {item.code for item in creator_view.diagnostics}
        self.assertNotIn("review_blocked", diagnostic_codes)
        pending_review = next(
            (item for item in creator_view.diagnostics if item.code == "linked_workflow_review_pending"),
            None,
        )
        self.assertIsNotNone(pending_review)
        self.assertFalse(pending_review.blocking)

        shared_run = self.course_workflow_service.workflow_service.get_run(course_run.shared_workflow_run_id)
        assert shared_run is not None
        self.assertEqual(shared_run.stage, WorkflowStage.awaiting_hil_gate_1)
        self.assertEqual(shared_run.status, WorkflowStatus.awaiting_human)
        self.assertEqual(shared_run.pending_gate, HILGate.gate_1_spec_review)

        event_types = [
            event.event_type
            for event in self.course_workflow_service.list_events(course_run.id)
        ]
        self.assertIn("course_generation_completed", event_types)
        self.assertNotIn("course_run_published", event_types)

    def test_active_generation_does_not_surface_provisional_hil_gate(self) -> None:
        plan_response = self.course_generation_service.generate_creator_plan(
            GenerateCreatorCoursePlanRequest(
                goal="Build an inventory reservation service.",
                creator_choices={"starter_type": "partial"},
            )
        )
        queued = self.course_generation_service.queue_course_run_from_creator_plan(
            CreateCourseFromCreatorPlanRequest(plan=plan_response.plan)
        )
        course_run = self.course_workflow_service.get_run(queued.course_run.id)
        assert course_run is not None
        course_run.active_operation = CourseAsyncOperation.generation
        self.course_workflow_service.store.save_course_run(course_run)

        creator_view = self.course_workflow_service.creator_view(course_run.id)
        self.assertEqual(creator_view.course_run.stage.value, "drafting")
        self.assertEqual(creator_view.course_run.status.value, "active")
        self.assertTrue(
            all(
                "waiting on `gate_1_spec_review`" not in blocker
                for blocker in creator_view.review.blockers
            )
        )

    def test_creator_plan_preserves_authored_workflow_deliverables(self) -> None:
        plan_response = self.course_generation_service.generate_creator_plan(
            GenerateCreatorCoursePlanRequest(
                goal=(
                    "Build a multi-warehouse inventory reservation service that is production ready. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
                creator_choices={
                    "starter_type": "partial",
                    "implementation_language": "python",
                    "application_framework": "fastapi",
                    "primary_database": "postgres",
                    "cache_backend": "redis",
                },
            )
        )

        queued = self.course_generation_service.queue_course_run_from_creator_plan(
            CreateCourseFromCreatorPlanRequest(plan=plan_response.plan)
        )
        course_run = self.course_workflow_service.get_run(queued.course_run.id)
        assert course_run is not None
        shared_run = self.course_workflow_service.workflow_service.get_run(course_run.shared_workflow_run_id)
        assert shared_run is not None

        authored_titles = [
            deliverable.title
            for deliverable in shared_run.artifacts.task_agent_spec.deliverables
        ]
        self.assertEqual(
            [deliverable.title for deliverable in course_run.deliverables],
            authored_titles,
        )
        self.assertEqual(
            len(course_run.deliverables),
            len(shared_run.artifacts.task_agent_spec.deliverables),
        )
        # Pass 10 Job A: the planner's deliverable titles flow through 1:1.
        # The course-plan fallback emits "Service contract and durable model" for
        # transactional-stateful-service families, so it is now expected to land
        # in the authored workflow deliverables instead of being replaced by an
        # entity-templated family-specific title.
        self.assertIn("Service contract and durable model", authored_titles)
