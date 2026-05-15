"""Regression test: outcome-mode learners can submit for review.

Bug surfaced 2026-05-15 from live enrollment ``enrollment_cb1fb60fd837``
on outcome-mode course ``course_f918e889a33c``
("Production-Quality Finance RAG"). The LMS UI showed:

    Grading didn't complete
    We couldn't submit this project for review right now.

Root cause: ``LMSService.submit_project`` hard-rejects outcome-mode
courses with::

    if snapshot.task_agent_spec is None:
        raise LMSConflictError(
            "The publish snapshot is missing the internal grading spec."
        )

Outcome-mode publish snapshots intentionally carry
``task_agent_spec=None`` — the grader is shipped as scenarios + setup +
reference impl on disk at
``workspaces/outcome/<course_run_id>/private/grader/``, not as a
``TaskAgentServiceSpec``. The catalog (``_lms_support``) and enroll
(``_require_supported_snapshot``) paths were already amended; submit
was left behind.

The fix branches ``submit_project`` on outcome-mode and runs an
``OraclePass`` against the learner's ``public/starter/`` directory,
aggregating per-scenario verdicts into a single-deliverable
``AssignmentGradeReport`` and persisting a ``LearnerSubmissionRecord``.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.domain.course import CourseRun, CourseRunStage, CourseRunStatus, PackageType
from app.domain.learner import (
    CreateEnrollmentRequest,
    LearnerEnrollmentStatus,
    LearnerWorkspaceScope,
    SubmitDeliverableRequest,
)
from app.domain.publish import PublishSnapshot
from app.services.lms_service import LMSConflictError, LMSService
from app.services.oracle_pass import OraclePass
from app.services.outcome_publish_snapshot import build_outcome_publish_snapshot
from app.storage.sqlite_store import SQLiteWorkflowStore


# ---------------- Fakes (mirror tests/test_oracle_pass.py shapes) ----------------


class _FakeSandboxHandle:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class _FakeSandboxRunner:
    """Records every boot dir so the test can assert the learner workspace
    (not the authoring bundle) is what's booted."""

    def __init__(self, base_url: str = "http://127.0.0.1:54321") -> None:
        self.base_url = base_url
        self.boot_calls: list[Path] = []
        self.teardown_calls: list[_FakeSandboxHandle] = []

    def boot(self, reference_impl_dir: Path, *, capabilities: Any = None) -> _FakeSandboxHandle:
        self.boot_calls.append(Path(reference_impl_dir))
        return _FakeSandboxHandle(self.base_url)

    def teardown(self, handle: _FakeSandboxHandle) -> None:
        self.teardown_calls.append(handle)


class _FakeHttpClient:
    def __init__(self, responses: dict[tuple[str, str], tuple[int, dict[str, str], Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(self, *, method, url, headers, body, follow_redirects, timeout):
        self.calls.append({"method": method, "url": url})
        return self.responses[(method.upper(), url)]


# ---------------- Fixture builders ----------------


def _write_minimal_scenarios(scenarios_dir: Path) -> None:
    """One scenario that asserts ``GET /ping`` returns 200 with a literal
    body field. Pure structural rubric — no LLM router needed."""
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    (scenarios_dir / "ping.yaml").write_text(
        """\
id: ping_smoke
description: ping endpoint returns ok
category: happy_path
quality_bar_ids:
  - smoke
trace:
  - id: call
    method: GET
    path: /ping
    expect:
      status_code: 200
rubrics:
  - kind: literal_match
    config:
      target: call.body.status
      expected: ok
"""
    )


def _write_outcome_bundle(workspace_root: Path) -> None:
    """Lay out the minimum on-disk outcome bundle the submit path needs."""
    grader = workspace_root / "private" / "grader"
    _write_minimal_scenarios(grader / "scenarios")
    # Reference impl directory is unused for learner submission (we boot
    # the learner's starter), but the workspace layout includes it.
    (grader / "_reference").mkdir(parents=True, exist_ok=True)
    (grader / "_reference" / "app.py").write_text("# reference\n")
    # Public starter — the artifact the learner edits + we boot at submit.
    starter = workspace_root / "public" / "starter"
    starter.mkdir(parents=True, exist_ok=True)
    (starter / "app.py").write_text("# starter\n")
    (starter / "Dockerfile").write_text("FROM python:3.12-slim\nEXPOSE 8000\n")
    (starter / "requirements.txt").write_text("")
    (workspace_root / "public" / "README.md").write_text("# Test outcome course\n")


def _seed_outcome_course_run(
    store: SQLiteWorkflowStore, course_run_id: str, workspace_root: Path
) -> CourseRun:
    """Persist a course_run row marked outcome-mode + published."""
    now = datetime.now(UTC)
    course_run = CourseRun(
        id=course_run_id,
        course_family_id=f"family_{uuid4().hex[:8]}",
        title="Test Outcome Course",
        summary="Outcome-mode submit smoke",
        package_type=PackageType.progressive_codebase_course,
        pattern_slug="outcome-default",
        creator_choices={},
        created_at=now,
        updated_at=now,
        stage=CourseRunStage.published,
        status=CourseRunStatus.published,
        deliverables=[],
        notes=[],
        goal="Test course",
        requested_learning_outcomes=[],
        payload_json={
            "outcome_state": {
                "workspace_root": str(workspace_root),
            }
        },
    )
    store.save_course_run(course_run)
    return course_run


def _seed_outcome_snapshot(
    store: SQLiteWorkflowStore, course_run: CourseRun, workspace_root: Path
) -> PublishSnapshot:
    """Build + persist an outcome-mode PublishSnapshot via the canonical
    helper, then attach it to the course_run row."""
    from app.services.course_outcome_models import (
        CourseOutcomeSpec,
        EndpointContract,
        HttpMethod,
        JudgeKind,
        LearningHint,
        QualityBar,
        StarterType,
    )

    spec = CourseOutcomeSpec(
        title="Test Outcome Course",
        goal="Test outcome course used by the LMS submit smoke regression",
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.GET,
                path="/ping",
                description="Liveness ping endpoint",
                response_schema={"type": "object", "properties": {"status": {"type": "string"}}},
            )
        ],
        quality_bars=[
            QualityBar(
                id="smoke",
                metric_description="ping smoke check",
                threshold=">= 1",
                judged_by=JudgeKind.literal,
                sample_size=1,
            )
        ],
        learning_path=[
            LearningHint(
                on_metric_fail="smoke",
                hint="Return {'status': 'ok'} from /ping",
            )
        ],
        package_type=PackageType.progressive_codebase_course,
    )

    class _StateLike:
        def __init__(self, ws: Path, spec_obj: Any) -> None:
            self.workspace_root = str(ws)
            self.spec = spec_obj

    snapshot = build_outcome_publish_snapshot(course_run, _StateLike(workspace_root, spec))
    store.save_publish_snapshot(snapshot)
    course_run.latest_publish_snapshot_id = snapshot.id
    store.save_course_run(course_run)
    return snapshot


# ---------------- The actual tests ----------------


class OutcomeSubmitTests(unittest.TestCase):
    def test_outcome_submit_no_longer_409s(self) -> None:
        """Before the fix: ``submit_project`` raised ``LMSConflictError``
        with "publish snapshot is missing the internal grading spec".
        After: it boots, runs scenarios, returns an experience with a
        latest submission attached."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            workspace_root = tmp_root / "outcome" / "course_test"
            learner_root = tmp_root / "learner_workspaces"
            _write_outcome_bundle(workspace_root)

            db_path = tmp_root / "test.db"
            store = SQLiteWorkflowStore(str(db_path))
            course_run = _seed_outcome_course_run(store, "course_test", workspace_root)
            snapshot = _seed_outcome_snapshot(store, course_run, workspace_root)

            # Inject a fake sandbox + canned http response for /ping=ok.
            sandbox = _FakeSandboxRunner()
            http = _FakeHttpClient(
                {
                    ("GET", "http://127.0.0.1:54321/ping"): (
                        200,
                        {"content-type": "application/json"},
                        {"status": "ok"},
                    )
                }
            )
            outcome_grader = OraclePass(sandbox_runner=sandbox, http_client=http)

            # workflow_service is unused by the outcome submit path; pass
            # a placeholder None — LMSService doesn't dereference it for
            # outcome submit.
            service = LMSService(
                store=store,
                workflow_service=None,  # type: ignore[arg-type]
                base_dir=learner_root,
                outcome_grader=outcome_grader,
            )

            enrollment = service.enroll(
                CreateEnrollmentRequest(
                    learner_id="local-learner",
                    course_run_id=course_run.id,
                )
            )

            experience = service.submit_project(
                enrollment.id,
                SubmitDeliverableRequest(deliverable_id="outcome_main"),
            )

            # Submission was actually persisted.
            submissions = store.list_learner_submissions(enrollment.id)
            self.assertEqual(len(submissions), 1, "expected one submission row")
            sub = submissions[0]
            self.assertEqual(sub.deliverable_id, "outcome_main")
            self.assertEqual(sub.total_tests, 1)
            self.assertEqual(sub.passed_tests, 1)
            self.assertEqual(sub.status, "passed")

            # The booted dir is the LEARNER's starter, not the authoring
            # bundle — this is the whole point of the submission flow.
            self.assertEqual(len(sandbox.boot_calls), 1)
            booted = sandbox.boot_calls[0]
            self.assertTrue(
                str(booted).startswith(str(learner_root)),
                f"expected boot dir under learner_root={learner_root}, got {booted}",
            )
            self.assertEqual(booted.name, "starter")

            # The experience surfaces the new submission so the UI can
            # render the scorecard immediately.
            self.assertIsNotNone(experience.latest_assignment_submission)

    def test_outcome_submit_failing_scenario_surfaces_as_failed_status(self) -> None:
        """A scenario whose rubric fails should produce a submission with
        status=failed and zero passed tests — not crash the submit path."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            workspace_root = tmp_root / "outcome" / "course_test2"
            learner_root = tmp_root / "learner_workspaces"
            _write_outcome_bundle(workspace_root)

            db_path = tmp_root / "test.db"
            store = SQLiteWorkflowStore(str(db_path))
            course_run = _seed_outcome_course_run(store, "course_test2", workspace_root)
            _seed_outcome_snapshot(store, course_run, workspace_root)

            sandbox = _FakeSandboxRunner()
            # Wrong field value → rubric fails.
            http = _FakeHttpClient(
                {
                    ("GET", "http://127.0.0.1:54321/ping"): (
                        200,
                        {"content-type": "application/json"},
                        {"status": "WRONG"},
                    )
                }
            )
            outcome_grader = OraclePass(sandbox_runner=sandbox, http_client=http)
            service = LMSService(
                store=store,
                workflow_service=None,  # type: ignore[arg-type]
                base_dir=learner_root,
                outcome_grader=outcome_grader,
            )
            enrollment = service.enroll(
                CreateEnrollmentRequest(
                    learner_id="local-learner",
                    course_run_id=course_run.id,
                )
            )
            service.submit_project(
                enrollment.id,
                SubmitDeliverableRequest(deliverable_id="outcome_main"),
            )

            submissions = store.list_learner_submissions(enrollment.id)
            self.assertEqual(len(submissions), 1)
            sub = submissions[0]
            self.assertEqual(sub.status, "failed")
            self.assertEqual(sub.passed_tests, 0)
            self.assertEqual(sub.total_tests, 1)


    def test_abstain_verdicts_dont_fail_the_scenario(self) -> None:
        """LLM-judge rubrics abstain when no router is configured (design
        contract: judge availability never blocks grading). A scenario
        whose only non-pass verdicts are abstains must be counted as
        passed — otherwise the learner sees 0/N passed for code that's
        structurally correct, just because the LLM judge couldn't run.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            workspace_root = tmp_root / "outcome" / "course_test3"
            learner_root = tmp_root / "learner_workspaces"
            _write_outcome_bundle(workspace_root)

            # Replace the scenario with one that mixes a passing
            # structural rubric and an LLM-judge rubric that will abstain
            # (no router).
            scenarios_dir = workspace_root / "private" / "grader" / "scenarios"
            (scenarios_dir / "ping.yaml").write_text(
                """\
id: ping_smoke
description: ping with abstaining LLM judge
category: happy_path
quality_bar_ids:
  - smoke
trace:
  - id: call
    method: GET
    path: /ping
    expect:
      status_code: 200
rubrics:
  - kind: literal_match
    config:
      target: call.body.status
      expected: ok
  - kind: llm_judge_coverage
    config:
      target: call.body.status
      must_contain_facts: ["ok"]
"""
            )

            db_path = tmp_root / "test.db"
            store = SQLiteWorkflowStore(str(db_path))
            course_run = _seed_outcome_course_run(store, "course_test3", workspace_root)
            _seed_outcome_snapshot(store, course_run, workspace_root)

            sandbox = _FakeSandboxRunner()
            http = _FakeHttpClient(
                {
                    ("GET", "http://127.0.0.1:54321/ping"): (
                        200,
                        {"content-type": "application/json"},
                        {"status": "ok"},
                    )
                }
            )
            # NB: router=None (default) → LLMJudgeCoverage abstains.
            outcome_grader = OraclePass(sandbox_runner=sandbox, http_client=http)
            service = LMSService(
                store=store,
                workflow_service=None,  # type: ignore[arg-type]
                base_dir=learner_root,
                outcome_grader=outcome_grader,
            )
            enrollment = service.enroll(
                CreateEnrollmentRequest(
                    learner_id="local-learner",
                    course_run_id=course_run.id,
                )
            )
            service.submit_project(
                enrollment.id,
                SubmitDeliverableRequest(deliverable_id="outcome_main"),
            )

            sub = store.list_learner_submissions(enrollment.id)[0]
            self.assertEqual(sub.passed_tests, 1, "abstain should count as passed")
            self.assertEqual(sub.total_tests, 1)
            self.assertEqual(sub.status, "passed")
            # Per-test summary still flags the abstain transparently.
            results = sub.grade_report.results
            self.assertEqual(len(results), 1)
            self.assertIn("abstain", results[0].summary.lower())


if __name__ == "__main__":
    unittest.main()
