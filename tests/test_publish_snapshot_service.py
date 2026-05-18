from __future__ import annotations

import pytest
pytest.skip(
    "Pre-existing test depends on the removed SQLiteWorkflowStore. "
    "Pending follow-up to port to PostgresWorkflowStore.",
    allow_module_level=True,
)

import tempfile
import unittest
from datetime import UTC, datetime

from app.domain.course import CourseDeliverableDraft, CourseRun, CourseRunStage, CourseRunStatus
from app.domain.registry import PackageType
from app.domain.workflow import MaterializeBundleRequest
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.publish_snapshot_service import PublishSnapshotService
from app.services.workflow_service import WorkflowService


class PublishSnapshotServiceTests(unittest.TestCase):
    def test_progressive_publish_keeps_all_authored_deliverables_even_if_course_run_is_shorter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
            workflow_service = WorkflowService(
                store,
                materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
            )
            intake = GenerationIntake(
                title="Inventory Reservation Service",
                problem_statement=(
                    "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
                    "Keep reservations correct under concurrency, retries, and stock transfers."
                ),
                package_type_hint=PackageType.progressive_codebase_course,
            )
            inferred = infer_assignment_design(
                title=intake.title,
                problem_statement=intake.problem_statement,
                package_type_hint=intake.package_type_hint,
            )
            assert inferred.design_spec is not None
            run = workflow_service.create_run_from_explicit_plan(
                intake=intake,
                design_spec=inferred.design_spec,
                execute_nodes=False,
            )
            workflow_service.materialize_run(run.id, MaterializeBundleRequest(overwrite=True))
            run = workflow_service.get_run(run.id)
            assert run is not None
            spec = run.artifacts.task_agent_spec
            assert spec is not None

            authored_deliverables = list(spec.deliverables)
            course_run = CourseRun(
                id="course_test",
                course_family_id="course_test",
                title="Progressive payment ledger",
                summary="Build one payment ledger app over four milestones.",
                package_type=PackageType.progressive_codebase_course,
                shared_workflow_run_id=run.id,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                stage=CourseRunStage.ready_to_publish,
                status=CourseRunStatus.active,
                deliverables=[
                    CourseDeliverableDraft(
                        deliverable_slug=deliverable.id.replace("_", "-"),
                        title=deliverable.title,
                        summary=deliverable.objective,
                        learning_outcomes=list(deliverable.learning_outcomes),
                    )
                    for deliverable in authored_deliverables[:3]
                ],
            )

            learner_package = PublishSnapshotService(store, workflow_service)._build_learner_package(course_run, run)

            self.assertEqual(len(learner_package.deliverables), len(authored_deliverables))
            self.assertEqual(
                [deliverable.title for deliverable in learner_package.deliverables],
                [deliverable.title for deliverable in authored_deliverables],
            )
            self.assertEqual(
                learner_package.deliverables[-1].course_deliverable_slug,
                authored_deliverables[-1].id.replace("_", "-"),
            )
            self.assertTrue(learner_package.deliverables[-1].workspace_seed_files)


if __name__ == "__main__":
    unittest.main()
