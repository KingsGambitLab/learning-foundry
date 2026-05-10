from __future__ import annotations

import tempfile
import unittest

from app.domain.ai import AIUsageSummary
from app.domain.course import CreateCourseDeliverableRequest, CreateCourseRunRequest, CreatorCourseSetupChoices
from app.domain.registry import PackageType
from app.domain.task_agent import DataSourceKind, DataSourcePurpose, DataSourceSpec
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import infer_assignment_design
from app.services.course_workflow_service import CourseWorkflowService
from app.services.openai_task_agent_authoring import OpenAITaskAgentAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class CourseWorkflowRuntimeContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteWorkflowStore(db_path=f"{self.temp_dir.name}/test.db")
        self.workflow_service = WorkflowService(
            self.store,
            materializer=ArtifactMaterializer(base_dir=f"{self.temp_dir.name}/generated"),
            task_agent_authoring_service=OpenAITaskAgentAuthoringService(enabled=False),
        )
        self.course_workflow_service = CourseWorkflowService(
            self.store,
            self.workflow_service,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_progressive_workflow_intake_and_creator_view_preserve_runtime_context(self) -> None:
        fixture_source = DataSourceSpec(
            id="seed_inventory",
            kind=DataSourceKind.seed_database,
            title="Inventory seed data",
            purpose=DataSourcePurpose.seed_state,
            learner_visible=True,
            format="json",
            workspace_path="data/inventory_seed.json",
            description="Seed data for inventory, warehouse, and reservation fixtures.",
        )
        inferred = infer_assignment_design(
            title="Build a Multi-warehouse Inventory Reservation Service",
            problem_statement=(
                "Build a multi-warehouse inventory reservation service that is production ready. "
                "Keep reservations correct under concurrency, retries, and stock transfers."
            ),
            implementation_language="python",
            language_version="3.13",
            application_framework="fastapi",
            framework_version="0.116",
            package_manager="uv",
            primary_database="postgres",
            primary_database_version="17",
            cache_backend="redis",
            cache_backend_version="8",
            tech_stack=["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"],
            data_sources=[fixture_source],
        )
        self.assertIsNotNone(inferred.design_spec)
        design_spec = inferred.design_spec
        assert design_spec is not None

        course_run = self.course_workflow_service.create_run(
            CreateCourseRunRequest(
                title="Build a Multi-warehouse Inventory Reservation Service",
                summary=(
                    "Build a multi-warehouse inventory reservation service that is production ready. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
                package_type=PackageType.progressive_codebase_course,
                creator_choices=CreatorCourseSetupChoices(
                    implementation_language="python",
                    language_version="3.13",
                    application_framework="fastapi",
                    framework_version="0.116",
                    package_manager="uv",
                    primary_database="postgres",
                    primary_database_version="17",
                    cache_backend="redis",
                    cache_backend_version="8",
                    tech_stack=["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"],
                    data_sources=[fixture_source],
                ),
                shared_design_spec=design_spec,
                deliverables=[
                    CreateCourseDeliverableRequest(
                        title="Service contract and durable model",
                        summary="Define the API and persistence model for reservations and stock.",
                        learning_outcomes=["Model durable reservation state."],
                    ),
                    CreateCourseDeliverableRequest(
                        title="Read and write path correctness",
                        summary="Implement correct reservation and release behavior under retries.",
                        learning_outcomes=["Keep write paths correct under retries."],
                    ),
                ],
            )
        )

        self.assertIsNotNone(course_run.shared_workflow_run_id)
        self.assertIsNotNone(course_run.creator_choices)
        shared_run = self.workflow_service.get_run(course_run.shared_workflow_run_id)
        assert shared_run is not None
        self.assertEqual(shared_run.intake.implementation_language, "python")
        self.assertEqual(shared_run.intake.language_version, "3.13")
        self.assertEqual(shared_run.intake.application_framework, "fastapi")
        self.assertEqual(shared_run.intake.framework_version, "0.116")
        self.assertEqual(shared_run.intake.package_manager, "uv")
        self.assertEqual(shared_run.intake.primary_database, "postgres")
        self.assertEqual(shared_run.intake.primary_database_version, "17")
        self.assertEqual(shared_run.intake.cache_backend, "redis")
        self.assertEqual(shared_run.intake.cache_backend_version, "8")
        self.assertEqual(shared_run.intake.tech_stack, ["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"])
        self.assertEqual(len(shared_run.intake.data_sources), 1)
        self.assertEqual(shared_run.intake.data_sources[0].workspace_path, "data/inventory_seed.json")

        creator_view = self.course_workflow_service.creator_view(course_run.id)
        self.assertEqual(creator_view.creator_choices.implementation_language, "python")
        self.assertEqual(creator_view.creator_choices.language_version, "3.13")
        self.assertEqual(creator_view.creator_choices.application_framework, "fastapi")
        self.assertEqual(creator_view.creator_choices.framework_version, "0.116")
        self.assertEqual(creator_view.creator_choices.package_manager, "uv")
        self.assertEqual(creator_view.creator_choices.primary_database, "postgres")
        self.assertEqual(creator_view.creator_choices.primary_database_version, "17")
        self.assertEqual(creator_view.creator_choices.cache_backend, "redis")
        self.assertEqual(creator_view.creator_choices.cache_backend_version, "8")
        self.assertEqual(creator_view.creator_choices.tech_stack, ["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"])
        self.assertEqual(len(creator_view.creator_choices.data_sources), 1)
        self.assertEqual(creator_view.creator_choices.data_sources[0].workspace_path, "data/inventory_seed.json")
        self.assertEqual(course_run.creator_choices.language_version, "3.13")
        self.assertEqual(course_run.creator_choices.framework_version, "0.116")
        self.assertEqual(course_run.creator_choices.package_manager, "uv")

    def test_progressive_course_uses_authored_workflow_deliverables_as_source_of_truth(self) -> None:
        inferred = infer_assignment_design(
            title="Build a Multi-warehouse Inventory Reservation Service",
            problem_statement=(
                "Build a multi-warehouse inventory reservation service that is production ready. "
                "Keep reservations correct under concurrency, retries, and stock transfers."
            ),
            implementation_language="python",
            application_framework="fastapi",
            primary_database="postgres",
            cache_backend="redis",
            tech_stack=["Python 3.12", "FastAPI", "Postgres 16", "Redis 7"],
        )
        assert inferred.design_spec is not None

        course_run = self.course_workflow_service.create_run(
            CreateCourseRunRequest(
                title="Build a Multi-warehouse Inventory Reservation Service",
                summary=(
                    "Build a multi-warehouse inventory reservation service that is production ready. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
                package_type=PackageType.progressive_codebase_course,
                shared_design_spec=inferred.design_spec,
                deliverables=[
                    CreateCourseDeliverableRequest(
                        title="Service contract and durable model",
                        summary="Define the API and persistence model for reservations and stock.",
                        learning_outcomes=["Model durable reservation state."],
                    ),
                    CreateCourseDeliverableRequest(
                        title="Read and write path correctness",
                        summary="Implement correct reservation and release behavior under retries.",
                        learning_outcomes=["Keep write paths correct under retries."],
                    ),
                ],
            )
        )

        shared_run = self.workflow_service.get_run(course_run.shared_workflow_run_id)
        assert shared_run is not None
        authored_titles = [
            deliverable.title
            for deliverable in shared_run.artifacts.task_agent_spec.deliverables[:2]
        ]
        self.assertEqual(
            authored_titles,
            [
                "Inventory Reservation contract and state model",
                "Inventory Reservation read and write correctness",
            ],
        )
        self.assertEqual(
            [deliverable.title for deliverable in course_run.deliverables],
            authored_titles,
        )
        self.assertNotEqual(course_run.deliverables[0].title, "Service contract and durable model")

    def test_get_run_refreshes_linked_workflow_ai_usage(self) -> None:
        inferred = infer_assignment_design(
            title="Inventory reservations",
            problem_statement="Build an inventory reservation service with FastAPI and Postgres.",
            implementation_language="python",
            application_framework="fastapi",
            primary_database="postgres",
        )
        assert inferred.design_spec is not None

        course_run = self.course_workflow_service.create_run(
            CreateCourseRunRequest(
                title="Inventory reservations",
                summary="Build an inventory reservation service with FastAPI and Postgres.",
                package_type=PackageType.progressive_codebase_course,
                shared_design_spec=inferred.design_spec,
                deliverables=[
                    CreateCourseDeliverableRequest(
                        title="Service contract",
                        summary="Implement the core reservation surface.",
                        learning_outcomes=["Keep the contract stable."],
                    )
                ],
            )
        )
        shared_run = self.workflow_service.get_run(course_run.shared_workflow_run_id)
        assert shared_run is not None
        shared_run.artifacts.ai_usage = AIUsageSummary(
            request_count=3,
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.123456,
            models=["gpt-5.4"],
        )
        self.store.save_run(shared_run)

        refreshed = self.course_workflow_service.get_run(course_run.id)

        assert refreshed is not None
        self.assertEqual(refreshed.ai_usage.request_count, 3)
        self.assertAlmostEqual(refreshed.ai_usage.estimated_cost_usd, 0.123456)
        self.assertEqual(refreshed.ai_usage.models, ["gpt-5.4"])
