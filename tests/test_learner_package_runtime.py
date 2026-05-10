from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from app.domain.publish import (
    LearnerCoursePackage,
    LearnerDeliverablePackage,
    LearnerPackageFile,
    PublishSnapshot,
    PublishSnapshotProvenance,
)
from app.domain.registry import PackageType
from app.domain.learner import LearnerWorkspaceScope
from app.services.learner_package_runtime import seed_workspace_from_snapshot


class LearnerPackageRuntimeTests(unittest.TestCase):
    def test_progressive_workspace_seed_uses_first_deliverable_snapshot_only(self) -> None:
        learner_package = LearnerCoursePackage(
            course_run_id="course_test",
            title="Progressive app",
            summary="Build one shared app over time.",
            package_type=PackageType.progressive_codebase_course,
            published_at=datetime.now(UTC),
            workspace_scope=LearnerWorkspaceScope.shared_course,
            deliverables=[
                LearnerDeliverablePackage(
                    deliverable_id="deliverable-1",
                    course_deliverable_slug="deliverable-1",
                    title="Stage 1",
                    objective="Base repo",
                    deliverable_index=1,
                    learner_brief={
                        "why_this_deliverable_matters": "Start from the shared repo baseline.",
                        "task_to_build": "Implement the first milestone.",
                    },
                    content_markdown="",
                    starter_readme="Stage 1 README",
                    completion_rule="Do stage 1",
                    workspace_seed_files=[
                        LearnerPackageFile(
                            relative_path="src/main/java/com/example/App.java",
                            media_type="text/x-java-source",
                            content="class App {}\n",
                        )
                    ],
                ),
                LearnerDeliverablePackage(
                    deliverable_id="deliverable-2",
                    course_deliverable_slug="deliverable-2",
                    title="Stage 2",
                    objective="Alternative repo variant",
                    deliverable_index=2,
                    learner_brief={
                        "why_this_deliverable_matters": "Continue the shared repo.",
                        "task_to_build": "Implement the second milestone.",
                    },
                    content_markdown="",
                    starter_readme="Stage 2 README",
                    completion_rule="Do stage 2",
                    workspace_seed_files=[
                        LearnerPackageFile(
                            relative_path="src/main/java/com/example/App.java",
                            media_type="text/x-java-source",
                            content="class App { int drift = 2; }\n",
                        ),
                        LearnerPackageFile(
                            relative_path="src/main/java/com/example/Extra.java",
                            media_type="text/x-java-source",
                            content="class Extra {}\n",
                        ),
                    ],
                ),
            ],
        )
        snapshot = PublishSnapshot(
            id="publish_test",
            course_run_id="course_test",
            course_family_id="course_test",
            created_at=datetime.now(UTC),
            version=1,
            source_hash="hash",
            learner_package=learner_package,
            provenance=PublishSnapshotProvenance(
                generator_version="test",
                course_run_hash="course_hash",
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = seed_workspace_from_snapshot(Path(temp_dir) / "workspace", snapshot)

            app_file = root / "src/main/java/com/example/App.java"
            extra_file = root / "src/main/java/com/example/Extra.java"
            self.assertTrue(app_file.exists())
            self.assertEqual(app_file.read_text(encoding="utf-8"), "class App {}\n")
            self.assertFalse(extra_file.exists())


if __name__ == "__main__":
    unittest.main()
