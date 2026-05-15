from __future__ import annotations

import json
import mimetypes
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.course import CourseRun, CourseRunStatus, CourseRunSummary
from app.domain.grading import (
    AssignmentGradeReport,
    DeliverableGradeReport,
    GradeStatus,
    LearnerReviewGuidance,
    ReviewAreaGradeReport,
    TestGradeResult,
)
from app.services.scenario_rubrics_base import Verdict
from app.domain.learner import (
    CreateEnrollmentRequest,
    LaunchWorkspaceRequest,
    LearnerEnrollment,
    LearnerEnrollmentList,
    LearnerEnrollmentStatus,
    LearnerEnrollmentSummary,
    LearnerDeliverableExperience,
    LearnerDeliverableProgress,
    LearnerDeliverableStatus,
    LearnerSubmissionRecord,
    LearnerWorkspaceFileContent,
    LearnerWorkspaceFileList,
    LearnerWorkspaceFileSummary,
    LearnerWorkspaceFileWriteResult,
    LearnerWorkspaceScope,
    PublishedCourseCatalog,
    PublishedCourseSummary,
    SubmitDeliverableRequest,
    WriteLearnerWorkspaceFileRequest,
)
from app.domain.publish import LearnerDeliverablePackage, PublishSnapshot
from app.domain.testing import (
    CreateLearnerFeedbackRequest,
    LearnerFeedbackList,
    LearnerFeedbackRecord,
    LearnerTestingView,
)
from app.services.learner_package_runtime import (
    project_brief_markdown,
    remap_assignment_report_to_deliverables,
    seed_workspace_from_snapshot,
)
from app.services.learner_studio_service import LearnerStudioService
from app.services.openai_learner_feedback import OpenAILearnerFeedbackService
from app.services.workflow_service import WorkflowService
from app.storage.workflow_store import WorkflowStore


def default_learner_workspace_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "learner_workspaces"


class LMSConflictError(ValueError):
    """Raised when an LMS action is invalid for the current course or enrollment state."""


# ---------------- Rubric diagnostic humanizer ----------------

# Translate scenario-rubric diagnostic strings into plain-English advice the
# learner can act on. The raw diagnostics are mechanically correct but use
# rubric-implementation vocabulary ("not found in captures", "target dict
# failed schema check") that doesn't tell the learner WHAT TO FIX. Rules
# are regex-based, ordered most-specific first.

_RUBRIC_PREFIX_RE = re.compile(r"^\s*[a-z_]+\s*\((fail|abstain)\):\s*", re.IGNORECASE)


def _grader_bundle_digest(grader_root: Path) -> str:
    """SHA-256 over the concatenation of `(relative_path, sha256(bytes))` for
    every file under `grader_root`, sorted by relative_path.

    Used as an audit fingerprint on each submission so post-hoc drift
    detection is possible if a course author modifies the bundle between
    submissions (P0 #4 stopgap).
    """
    import hashlib

    if not grader_root.exists():
        return ""
    h = hashlib.sha256()
    files = sorted(p for p in grader_root.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(grader_root).as_posix().encode("utf-8")
        file_digest = hashlib.sha256(p.read_bytes()).digest()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(file_digest)
    return h.hexdigest()


def _rubric_kinds(scenario) -> list[str]:
    """Return the rubric `kind` strings declared on a Scenario.

    Used by the strict-LLM-judge gate at submit time to detect when a
    bundle requires the LLM router. Tolerant of older scenario shapes
    where rubrics may be dicts or pydantic objects.
    """
    out: list[str] = []
    for rubric in getattr(scenario, "rubrics", None) or []:
        kind = getattr(rubric, "kind", None)
        if kind is None and isinstance(rubric, dict):
            kind = rubric.get("kind")
        if isinstance(kind, str):
            out.append(kind)
    return out


def _strip_rubric_prefix(text: str) -> str:
    """Drop the leading ``rubric_kind (fail): `` prefix the aggregation
    layer adds. Learners care about WHAT broke, not which rubric class
    reported it."""
    return _RUBRIC_PREFIX_RE.sub("", text, count=1)


def humanize_diagnostic(raw: str) -> str:
    """Convert one rubric diagnostic into plain-English advice.

    Unrecognized inputs pass through unchanged — better the learner
    sees raw text than nothing at all when a new rubric kind ships.
    """
    if not raw:
        return ""
    body = _strip_rubric_prefix(raw)

    # ``target path 'X' not found/present in captures`` — by far the most
    # common pattern. Show the path as the missing field.
    m = re.match(
        r"target path ['\"]([^'\"]+)['\"] not (?:found|present) in captures$",
        body,
    )
    if m:
        return f"Response is missing field `{m.group(1)}`"

    # ``expected R at 'X', got G`` — comparison failure.
    m = re.match(r"expected (.+?) at ['\"]([^'\"]+)['\"], got (.+)$", body)
    if m:
        expected, path, got = m.group(1).strip(), m.group(2), m.group(3).strip()
        return f"Expected `{path}` to be `{expected}`, got `{got}`"

    # ``target dict failed schema check`` — structural mismatch with no
    # specific path. Most common when the whole response body doesn't
    # match the expected schema shape.
    if body.strip() == "target dict failed schema check":
        return "Response shape doesn't match the required schema"

    # ``recall A < threshold B (N/M gold items found)`` — retrieval rubric.
    m = re.match(
        r"recall ([0-9.]+) < threshold ([0-9.]+) \((\d+)/(\d+) gold items found\)$",
        body,
    )
    if m:
        a, b, found, total = m.group(1), m.group(2), m.group(3), m.group(4)
        return (
            f"Retrieval recall {a} is below threshold {b} "
            f"(matched {found} of {total} expected items)"
        )

    # ``X not found in captures`` (no ``target path`` prefix) — produced
    # by structural rubrics like schema_match / literal_match / numeric_range
    # when their named target isn't present. Keep the prefix from the raw
    # original ONLY when it tells us about the field type
    # (``numeric_range`` -> "numeric field").
    m = re.match(r"([\w.\[\]]+) not found in captures$", body)
    if m:
        path = m.group(1)
        # Look at the original to recover the rubric kind for typed phrasing.
        kind_match = re.match(r"^\s*([a-z_]+)\s*\(", raw)
        kind = kind_match.group(1) if kind_match else ""
        if kind == "numeric_range":
            return f"Response is missing numeric field `{path}`"
        return f"Response is missing field `{path}`"

    # ``<step_id>.body.<field> does not equal expected literal`` — produced
    # by ``literal_match`` when the resolved value doesn't match. The most
    # frequent case in outcome-mode courses is ``body.abstained`` where the
    # scenario expects the service to refuse (``abstained=true``) but the
    # learner answered confidently from a distractor. Show that as advice,
    # not as a path expression.
    m = re.match(r"[\w_]+\.body\.(\w+) does not equal expected literal$", body)
    if m:
        field = m.group(1)
        if field == "abstained":
            return (
                "Service should have abstained (`abstained=true`) — the "
                "question can't be answered from the supplied passages"
            )
        return f"Response field `{field}` doesn't match the expected value"

    # Unrecognized — pass through. Better than losing signal.
    return raw


# ---------------- Outcome-mode feedback clustering ----------------

# Group humanized diagnostics into root-cause clusters so the scorecard
# can surface "fix this first, it blocks 8 scenarios" instead of a flat
# list of 19 equally-weighted failures.

_MAX_CAUSES = 5  # top-N clusters shown in priority list


def _cluster_key(diagnostic: str) -> tuple[str, str]:
    """Return ``(cluster_key, root_path)`` for a humanized diagnostic.

    The cluster key is what we group by; the root path is what we show
    to the learner. Cascading sub-field misses collapse under their
    parent (``eval.regression_diff.baseline_present`` → root
    ``eval.regression_diff``).
    """
    m = re.match(r"Response is missing (?:numeric )?field `([^`]+)`", diagnostic)
    if m:
        path = m.group(1)
        segments = path.split(".")
        # Use first two segments as the root — captures the
        # ``<top_namespace>.<block>`` shape we keep seeing
        # (``eval.regression_diff``, ``eval.summary``).
        root = ".".join(segments[:2]) if len(segments) >= 2 else segments[0]
        return f"missing_field:{root}", root
    if diagnostic == "Response shape doesn't match the required schema":
        return "schema_mismatch", "response schema"
    m = re.match(r"Expected `([^`]+)` to be", diagnostic)
    if m:
        return f"wrong_value:{m.group(1)}", m.group(1)
    if diagnostic.startswith("Retrieval recall"):
        return "retrieval_recall", "retrieval recall"
    if diagnostic.startswith("Service should have abstained"):
        return "missing_abstention", "abstention on out-of-scope questions"
    if diagnostic.startswith("Response field `"):
        m2 = re.match(r"Response field `([^`]+)` doesn't match", diagnostic)
        if m2:
            return f"wrong_value:{m2.group(1)}", m2.group(1)
    # LLM-judge rationale strings vary per scenario (the judge writes a
    # free-form sentence explaining the mismatch). Cluster them by
    # rubric kind so we surface "judge rejected the answer on N
    # scenarios" instead of N near-duplicate one-off rows.
    m = re.match(r"^(llm_judge_\w+) \(fail\):", diagnostic)
    if m:
        kind = m.group(1)
        readable = {
            "llm_judge_semantic_eq": "answer doesn't semantically match the expected response",
            "llm_judge_coverage": "answer is missing required facts",
            "llm_judge_false_premise": "service didn't refuse the false-premise question",
        }.get(kind, "judge rejected the answer")
        return f"llm_judge:{kind}", readable
    # Fallback: each unique diagnostic is its own cluster.
    return f"other:{diagnostic}", diagnostic


def _describe_cluster(
    cluster_key: str, root: str, scenario_count: int, exemplar: str
) -> str:
    """Render a single cluster as a one-line root-cause description.

    ``exemplar`` is the first humanized diagnostic the cluster saw; we
    parse it for the value-detail (e.g. expected ``422``, got ``400``)
    so the priority line is genuinely actionable rather than a path
    name in isolation.
    """
    plural = "scenarios" if scenario_count != 1 else "scenario"
    if cluster_key.startswith("missing_field:"):
        return (
            f"Add the `{root}` block to your response — "
            f"{scenario_count} {plural} check for it"
        )
    if cluster_key == "schema_mismatch":
        return (
            f"Response shape doesn't match the required schema — "
            f"affects {scenario_count} {plural}"
        )
    if cluster_key.startswith("wrong_value:"):
        # Pull expected/got out of the exemplar so the actionable hint
        # is visible without expanding the per-scenario detail.
        m = re.match(r"Expected `[^`]+` to be `([^`]+)`, got `([^`]+)`", exemplar)
        if m:
            expected, got = m.group(1), m.group(2)
            return (
                f"`{root}` should be `{expected}` but is `{got}` — "
                f"affects {scenario_count} {plural}"
            )
        return (
            f"Wrong value at `{root}` — "
            f"affects {scenario_count} {plural}"
        )
    if cluster_key == "retrieval_recall":
        return (
            f"Retrieval recall below threshold — "
            f"affects {scenario_count} {plural}"
        )
    if cluster_key == "missing_abstention":
        return (
            f"Service should abstain when no passage actually answers "
            f"the question — affects {scenario_count} {plural}"
        )
    if cluster_key.startswith("llm_judge:"):
        return (
            f"The {root} — affects {scenario_count} {plural} "
            f"(expand any failing row to see the judge's specific reasoning)"
        )
    # Fallback to the raw diagnostic.
    return f"{root} — {scenario_count} {plural}"


def build_outcome_feedback(
    results: list[TestGradeResult],
) -> LearnerReviewGuidance | None:
    """Synthesize a tech-lead-style review from per-scenario test results.

    Returns ``None`` when there's nothing to fix. Otherwise populates a
    :class:`LearnerReviewGuidance` whose fields drive the existing
    ``renderLearnerGuidance`` block in the UI:

    - ``learner_feedback`` — one-line headline ("X of N passing, most
      failures cluster around Y").
    - ``fundamental_gap`` — the single top blocker as a sentence.
    - ``likely_root_cause`` — top N clusters, each as a one-line advice
      string with impact count.
    """
    failed = [r for r in results if r.status != GradeStatus.passed]
    if not failed:
        return None

    total = len(results)
    passed = total - len(failed)

    # Build clusters: cluster_key -> {scenarios: set[str], root: str,
    # exemplar: str}. ``exemplar`` is the first humanized diagnostic
    # the cluster saw — used to render value-detail in the descriptor.
    clusters: dict[str, dict[str, Any]] = {}
    for result in failed:
        for diagnostic in result.diagnostics or []:
            key, root = _cluster_key(diagnostic)
            entry = clusters.setdefault(
                key, {"scenarios": set(), "root": root, "exemplar": diagnostic}
            )
            entry["scenarios"].add(result.test_id)

    if not clusters:
        # Scenarios failed but produced no diagnostics — unusual. Surface
        # what we can.
        return LearnerReviewGuidance(
            learner_feedback=(
                f"{passed} of {total} checks passing. "
                f"{len(failed)} scenarios failed without a structured diagnostic — "
                f"see the per-scenario detail for what was checked."
            ),
        )

    # Rank clusters by scenario-impact desc, then by key for stable order.
    ranked = sorted(
        clusters.items(),
        key=lambda kv: (-len(kv[1]["scenarios"]), kv[0]),
    )
    top = ranked[:_MAX_CAUSES]

    likely_root_cause = [
        _describe_cluster(
            key, entry["root"], len(entry["scenarios"]), entry["exemplar"]
        )
        for key, entry in top
    ]

    top_key, top_entry = top[0]
    top_count = len(top_entry["scenarios"])
    top_root = top_entry["root"]
    top_exemplar = top_entry["exemplar"]

    # Headline: one line that names the top cluster.
    if len(top) == 1:
        headline = (
            f"{passed} of {total} checks passing. "
            f"The remaining failures all trace back to `{top_root}`."
        )
    else:
        headline = (
            f"{passed} of {total} checks passing. "
            f"Most failures cluster around `{top_root}` "
            f"({top_count} scenario{'s' if top_count != 1 else ''}) — "
            f"fix that first, then work down the list below."
        )

    fundamental_gap = _describe_cluster(top_key, top_root, top_count, top_exemplar)
    # Avoid the "Fundamental gap" line repeating the first "Likely root
    # cause" verbatim — the UI renders both, so the duplicate is noise.
    # Drop ``fundamental_gap`` when it equals ``likely_root_cause[0]``.
    if likely_root_cause and fundamental_gap == likely_root_cause[0]:
        fundamental_gap = ""

    return LearnerReviewGuidance(
        learner_feedback=headline,
        fundamental_gap=fundamental_gap,
        likely_root_cause=likely_root_cause,
    )


# ---------------- LLM-rewritten learner feedback ----------------

# Optional layer: take the deterministic ``LearnerReviewGuidance`` plus
# the course spec context and ask haiku to rewrite the headline as
# conversational prose tied to the spec's quality bars. Cheap (~1
# haiku call per submit, ~500 tokens) and the fallback is the
# deterministic headline when the router isn't configured or fails.


_FEEDBACK_REWRITE_SYSTEM = (
    "You write 2-3 sentence summary feedback for a learner who just "
    "submitted their implementation of a graded course project. The "
    "user will hand you:\n"
    "- The course goal + the measurable quality bars the project is "
    "judged against\n"
    "- The structured root-cause list our grader already produced\n"
    "- The pass/fail count\n\n"
    "Rewrite the summary so it reads like a senior engineer's PR "
    "comment: name the concrete gap, tie it to which quality bar "
    "it blocks, and give one specific direction to fix first. Do not "
    "list more than the single top priority. Do not repeat the pass "
    "count (the UI shows it elsewhere). Plain prose, no markdown, no "
    "bullet points. Under 60 words total."
)


class _FeedbackRewrite(BaseModel):
    summary: str = Field(min_length=10, max_length=600)


def _rewrite_feedback_with_llm(
    *,
    feedback: LearnerReviewGuidance,
    spec_title: str,
    spec_goal: str,
    quality_bars: list[dict[str, Any]],
    passed: int,
    total: int,
    router: Any,
) -> str | None:
    """Return a rewritten ``learner_feedback`` headline, or ``None`` on
    any failure (caller falls back to the deterministic headline)."""
    try:
        from app.services.llm_router import LLMTier
        user_payload = {
            "course_title": spec_title,
            "course_goal": spec_goal,
            "quality_bars": [
                {"id": bar.get("id"), "metric": bar.get("metric_description")}
                for bar in quality_bars
            ],
            "score": f"{passed} of {total}",
            "current_headline": feedback.learner_feedback,
            "fundamental_gap": feedback.fundamental_gap,
            "top_root_causes": feedback.likely_root_cause[:3],
        }
        user = (
            "Rewrite the summary headline based on this submission "
            "context. Return JSON ``{summary: <text>}``.\n\n"
            + json.dumps(user_payload, indent=2)
        )
        result = router.parse_structured(
            tier=LLMTier.haiku,
            system=_FEEDBACK_REWRITE_SYSTEM,
            user=user,
            text_format=_FeedbackRewrite,
            request_timeout_s=30,
        )
        if result and getattr(result, "parsed", None) is not None:
            return result.parsed.summary.strip()
    except Exception:
        return None
    return None


class LMSService:
    MAX_WORKSPACE_FILE_BYTES = 1_000_000

    def __init__(
        self,
        store: WorkflowStore,
        workflow_service: WorkflowService,
        learner_studio_service: LearnerStudioService | None = None,
        learner_feedback_service: OpenAILearnerFeedbackService | None = None,
        base_dir: str | Path | None = None,
        outcome_grader: Any | None = None,
    ) -> None:
        self.store = store
        self.workflow_service = workflow_service
        self.learner_studio_service = learner_studio_service or LearnerStudioService()
        self.learner_feedback_service = learner_feedback_service or OpenAILearnerFeedbackService(enabled=False)
        self.base_dir = Path(base_dir or default_learner_workspace_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Outcome-mode submit grader. Lazy-built on first use so we don't
        # pull in the Docker sandbox adapter at construction time for
        # legacy-only test fixtures. Tests inject a duck-typed
        # ``OraclePass`` wired to a fake sandbox + canned HTTP responses.
        self._outcome_grader: Any | None = outcome_grader

    def list_catalog(self) -> PublishedCourseCatalog:
        latest_run_by_family: dict[str, tuple[CourseRun, PublishSnapshot | None]] = {}
        for summary in self.store.list_course_runs(limit=200):
            run = self.store.get_course_run(summary.id)
            if run is None or run.status != CourseRunStatus.published:
                continue
            snapshot = self._latest_snapshot(run)
            family_id = run.course_family_id
            current = latest_run_by_family.get(family_id)
            current_snapshot = current[1] if current is not None else None
            current_timestamp = current_snapshot.created_at if current_snapshot is not None else (current[0].updated_at if current is not None else None)
            candidate_timestamp = snapshot.created_at if snapshot is not None else run.updated_at
            if current is None or candidate_timestamp >= current_timestamp:
                latest_run_by_family[family_id] = (run, snapshot)

        courses: list[PublishedCourseSummary] = []
        for run, snapshot in latest_run_by_family.values():
            summary = CourseRunSummary.from_run(run)
            supported, reason = self._lms_support(run, snapshot)
            snapshot_package = snapshot.learner_package if snapshot is not None else None
            courses.append(
                PublishedCourseSummary.from_run(
                    summary,
                    title=snapshot_package.title if snapshot_package is not None else run.title,
                    summary=snapshot_package.summary if snapshot_package is not None else run.summary,
                    deliverable_count=len(snapshot_package.deliverables) if snapshot_package is not None else summary.deliverable_count,
                    shared_workflow_run_id=run.shared_workflow_run_id,
                    supported_for_lms=supported,
                    support_reason=reason,
                    publish_snapshot_id=snapshot.id if snapshot is not None else run.latest_publish_snapshot_id,
                    published_at=snapshot.created_at if snapshot is not None else run.updated_at,
                )
            )
        courses.sort(key=lambda item: item.published_at, reverse=True)
        return PublishedCourseCatalog(courses=courses)

    def list_enrollments(self, learner_id: str) -> LearnerEnrollmentList:
        return LearnerEnrollmentList(enrollments=self.store.list_learner_enrollments(learner_id=learner_id))

    def enroll(self, request: CreateEnrollmentRequest, *, learner_id: str) -> LearnerEnrollment:
        existing = self.store.find_learner_enrollment(learner_id, request.course_run_id)
        if existing is not None:
            return self.get_enrollment(existing.id)

        course_run = self._require_published_course(request.course_run_id)
        snapshot = self._require_supported_snapshot(course_run)
        learner_package = snapshot.learner_package
        assert learner_package is not None
        now = datetime.now(UTC)

        deliverables = [self._deliverable_progress(item) for item in learner_package.deliverables]

        enrollment = LearnerEnrollment(
            id=f"enrollment_{uuid4().hex[:12]}",
            learner_id=learner_id,
            course_run_id=course_run.id,
            publish_snapshot_id=snapshot.id,
            course_title=learner_package.title,
            course_summary=learner_package.summary,
            package_type=learner_package.package_type,
            # Outcome-mode courses have no workflow run; falling back to the
            # literal "shared_workflow" string would collide all outcome-mode
            # enrollments for the same learner into one workspace
            # (`learner_workspaces/<user>/shared_workflow/workspace`), so the
            # second course would silently see the first course's starter.
            # Fall back to the course_run.id when no real workflow run id
            # exists — guaranteed unique per course.
            shared_workflow_run_id=snapshot.shared_workflow_run_id or course_run.shared_workflow_run_id or course_run.id,
            created_at=now,
            updated_at=now,
            status=LearnerEnrollmentStatus.active,
            workspace_scope=learner_package.workspace_scope,
            current_deliverable_id=deliverables[0].deliverable_id if deliverables else None,
            deliverables=deliverables,
            notes=[
                "Enrollment created for the published course.",
                f"Progress is pinned to publish snapshot `{snapshot.id}`.",
            ],
        )
        self.store.save_learner_enrollment(enrollment)
        self._ensure_workspace_seeded(enrollment, snapshot)
        return enrollment

    def get_enrollment(self, enrollment_id: str) -> LearnerEnrollment:
        enrollment = self._require_enrollment(enrollment_id)
        submissions = self.store.list_learner_submissions(enrollment.id)
        sessions = self.store.list_learner_workspace_sessions(enrollment.id)
        latest_session = sessions[0] if sessions else None
        latest_submissions: dict[str, LearnerSubmissionRecord] = {}
        for submission in submissions:
            current = latest_submissions.get(submission.deliverable_id)
            if current is None or submission.created_at > current.created_at:
                latest_submissions[submission.deliverable_id] = submission

        refreshed = enrollment.model_copy(deep=True)
        for deliverable in refreshed.deliverables:
            latest_submission = latest_submissions.get(deliverable.deliverable_id)
            deliverable.latest_submission = latest_submission
            if latest_submission is not None:
                deliverable.status = (
                    LearnerDeliverableStatus.passed
                    if latest_submission.grade_report is not None
                    and latest_submission.grade_report.status == GradeStatus.passed
                    else LearnerDeliverableStatus.available
                )
            deliverable.workspace_session = latest_session
        if all(deliverable.status == LearnerDeliverableStatus.passed for deliverable in refreshed.deliverables):
            refreshed.status = LearnerEnrollmentStatus.completed
            refreshed.current_deliverable_id = None
        if refreshed.model_dump(mode="json") != enrollment.model_dump(mode="json"):
            refreshed.updated_at = datetime.now(UTC)
            self.store.save_learner_enrollment(refreshed)
        return refreshed

    def get_deliverable_experience(self, enrollment_id: str, deliverable_id: str | None = None) -> LearnerDeliverableExperience:
        enrollment = self.get_enrollment(enrollment_id)
        active_deliverable = self._resolve_target_deliverable(enrollment, deliverable_id)
        all_submissions = self.store.list_learner_submissions(enrollment.id)
        latest_assignment_submission = self._latest_assignment_submission(all_submissions)
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)
        self._ensure_workspace_seeded(enrollment, snapshot)
        latest_session = self.store.list_learner_workspace_sessions(enrollment.id)
        workspace_session = latest_session[0] if latest_session else None
        project_brief_markdown = self._project_brief_markdown(snapshot)
        return LearnerDeliverableExperience(
            enrollment=LearnerEnrollmentSummary.from_enrollment(enrollment),
            project_brief_markdown=project_brief_markdown,
            workspace_session=workspace_session,
            latest_assignment_report=(
                latest_assignment_submission.assignment_report
                if latest_assignment_submission is not None
                else None
            ),
            latest_assignment_submission=latest_assignment_submission,
            active_deliverable=active_deliverable,
            deliverables=enrollment.deliverables,
            submissions=all_submissions,
        )

    def get_learner_view(self, enrollment_id: str, deliverable_id: str | None = None) -> LearnerTestingView:
        return LearnerTestingView(
            experience=self.get_deliverable_experience(enrollment_id, deliverable_id),
            feedback=self.store.list_learner_feedback(enrollment_id),
        )

    def record_feedback(
        self,
        enrollment_id: str,
        request: CreateLearnerFeedbackRequest,
    ) -> LearnerFeedbackRecord:
        enrollment = self.get_enrollment(enrollment_id)
        deliverable = self._resolve_target_deliverable(enrollment, request.deliverable_id)
        feedback = LearnerFeedbackRecord(
            id=f"learner_feedback_{uuid4().hex[:12]}",
            enrollment_id=enrollment.id,
            course_run_id=enrollment.course_run_id,
            publish_snapshot_id=enrollment.publish_snapshot_id,
            learner_id=enrollment.learner_id,
            created_at=datetime.now(UTC),
            summary=request.summary.strip(),
            details=request.details.strip() if request.details else None,
            rating=request.rating,
            deliverable_id=deliverable.deliverable_id,
            context=request.context,
        )
        self.store.save_learner_feedback(feedback)
        return feedback

    def list_feedback(self, enrollment_id: str) -> LearnerFeedbackList:
        self._require_enrollment(enrollment_id)
        return LearnerFeedbackList(items=self.store.list_learner_feedback(enrollment_id))

    def launch_workspace(self, enrollment_id: str, request: LaunchWorkspaceRequest) -> LearnerEnrollment:
        enrollment, deliverable, _, workspace_root = self._workspace_context(
            enrollment_id,
            request.deliverable_id,
        )

        existing_sessions = self.store.list_learner_workspace_sessions(enrollment.id)
        latest_session = existing_sessions[0] if existing_sessions else None
        session = self.learner_studio_service.launch_editor(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=workspace_root,
            scope=enrollment.workspace_scope,
            existing_session=latest_session,
        )
        self.store.save_learner_workspace_session(session)
        if enrollment.current_deliverable_id != deliverable.deliverable_id:
            refreshed = enrollment.model_copy(deep=True)
            refreshed.current_deliverable_id = deliverable.deliverable_id
            refreshed.updated_at = datetime.now(UTC)
            self.store.save_learner_enrollment(refreshed)
        return self.get_enrollment(enrollment.id)

    def list_workspace_files(self, enrollment_id: str, deliverable_id: str | None = None) -> LearnerWorkspaceFileList:
        enrollment, deliverable, _, workspace_root = self._workspace_context(enrollment_id, deliverable_id)
        root = workspace_root.resolve()
        files = [
            LearnerWorkspaceFileSummary(
                relative_path=path.resolve().relative_to(root).as_posix(),
                media_type=self._guess_media_type(path),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        ]
        return LearnerWorkspaceFileList(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=str(root),
            files=files,
        )

    def read_workspace_file(
        self,
        enrollment_id: str,
        relative_path: str,
        deliverable_id: str | None = None,
    ) -> LearnerWorkspaceFileContent:
        enrollment, deliverable, _, workspace_root = self._workspace_context(enrollment_id, deliverable_id)
        root = workspace_root.resolve()
        target = self._resolve_workspace_file(workspace_root, relative_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(relative_path)
        return LearnerWorkspaceFileContent(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=str(root),
            relative_path=target.relative_to(root).as_posix(),
            media_type=self._guess_media_type(target),
            content=target.read_text(encoding="utf-8"),
        )

    def write_workspace_file(
        self,
        enrollment_id: str,
        payload: WriteLearnerWorkspaceFileRequest,
    ) -> LearnerWorkspaceFileWriteResult:
        if len(payload.content.encode("utf-8")) > self.MAX_WORKSPACE_FILE_BYTES:
            raise LMSConflictError("Workspace file payload is too large for the LMS prototype.")
        enrollment, deliverable, _, workspace_root = self._workspace_context(enrollment_id, payload.deliverable_id)
        root = workspace_root.resolve()
        target = self._resolve_workspace_file(workspace_root, payload.relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload.content, encoding="utf-8")
        return LearnerWorkspaceFileWriteResult(
            enrollment_id=enrollment.id,
            deliverable_id=deliverable.deliverable_id,
            workspace_root=str(root),
            relative_path=target.relative_to(root).as_posix(),
            media_type=self._guess_media_type(target),
            size_bytes=target.stat().st_size,
        )

    def submit_project(self, enrollment_id: str, request: SubmitDeliverableRequest) -> LearnerDeliverableExperience:
        enrollment, deliverable, deliverable_package, workspace_root = self._workspace_context(
            enrollment_id,
            request.deliverable_id,
        )
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)

        # Outcome-mode courses don't carry a TaskAgentServiceSpec — their
        # grader ships as scenarios + setup + reference impl on disk at
        # ``workspaces/outcome/<course_run_id>/private/grader/`` and runs
        # via an OraclePass against the learner's ``public/starter/``.
        # Route them to ``_submit_outcome_project`` instead of the legacy
        # ``learner_studio_service.grade_assignment`` path which expects
        # the spec.
        course_run = self.store.get_course_run(enrollment.course_run_id)
        if course_run is not None and (course_run.payload_json or {}).get("outcome_state"):
            return self._submit_outcome_project(
                enrollment=enrollment,
                deliverable=deliverable,
                course_run=course_run,
                snapshot=snapshot,
                workspace_root=workspace_root,
            )

        if snapshot.task_agent_spec is None:
            raise LMSConflictError("The publish snapshot is missing the internal grading spec.")

        report = self.learner_studio_service.grade_assignment(
            workspace_root=workspace_root,
            spec=snapshot.task_agent_spec,
        )
        submission_group_id = f"submission_{uuid4().hex[:12]}"
        created_at = datetime.now(UTC)
        assignment_report = self._learner_assignment_report(snapshot, report.assignment_report)
        learner_package = snapshot.learner_package
        if learner_package is not None:
            assignment_report = self.learner_feedback_service.annotate_assignment_report(
                project_brief_markdown=self._project_brief_markdown(snapshot),
                learner_package=learner_package,
                assignment_report=assignment_report,
                workspace_root=workspace_root,
                spec=snapshot.task_agent_spec,
            )
        submissions_by_deliverable: dict[str, LearnerSubmissionRecord] = {}
        for review_area in assignment_report.review_areas:
            submission = LearnerSubmissionRecord(
                id=f"{submission_group_id}_{review_area.deliverable_id.replace('/', '_')}",
                submission_group_id=submission_group_id,
                enrollment_id=enrollment.id,
                deliverable_id=review_area.deliverable_id,
                created_at=created_at,
                status=review_area.grade_report.status.value,
                passed_tests=review_area.grade_report.passed_tests,
                total_tests=review_area.grade_report.total_tests,
                pass_rate=review_area.grade_report.pass_rate,
                grade_report=review_area.grade_report,
                assignment_report=assignment_report,
            )
            self.store.save_learner_submission(submission)
            submissions_by_deliverable[review_area.deliverable_id] = submission

        refreshed = enrollment.model_copy(deep=True)
        for item in refreshed.deliverables:
            latest_submission = submissions_by_deliverable.get(item.deliverable_id)
            if latest_submission is not None:
                item.latest_submission = latest_submission
                item.status = (
                    LearnerDeliverableStatus.passed
                    if latest_submission.grade_report.status == GradeStatus.passed
                    else LearnerDeliverableStatus.available
                )
        refreshed.current_deliverable_id = deliverable.deliverable_id
        if assignment_report.status == GradeStatus.passed:
            refreshed.current_deliverable_id = None
            refreshed.status = LearnerEnrollmentStatus.completed
        else:
            refreshed.status = LearnerEnrollmentStatus.active
        refreshed.updated_at = datetime.now(UTC)
        self.store.save_learner_enrollment(refreshed)
        return self.get_deliverable_experience(refreshed.id, deliverable.deliverable_id)

    # ---------------- Outcome-mode submit ----------------

    def _submit_outcome_project(
        self,
        *,
        enrollment: LearnerEnrollment,
        deliverable: LearnerDeliverableProgress,
        course_run: CourseRun,
        snapshot: PublishSnapshot,
        workspace_root: Path,
    ) -> LearnerDeliverableExperience:
        """Outcome-mode grading: boot the learner's ``public/starter/`` and
        run the bundled scenarios via :class:`OraclePass`.

        The on-disk grader bundle lives at ``<authoring_workspace>/private/grader/``
        where ``<authoring_workspace>`` is recorded on the course run's
        ``payload_json["outcome_state"]["workspace_root"]``. We re-read
        scenarios + setup_data from there at submit time so any bundle
        amendments (re-published course) take effect on the next
        submission without re-seeding learner workspaces.
        """
        # Lazy imports — keep legacy LMS init free of OraclePass / scenario
        # loader dependencies.
        from app.services.oracle_pass import OraclePass, _load_setup_data
        from app.services.scenario_loader import load_scenarios_from_dir
        from app.services.workspace_boot import WorkspaceBootSandboxAdapter

        outcome_state = (course_run.payload_json or {}).get("outcome_state") or {}
        authoring_root = outcome_state.get("workspace_root")
        if not authoring_root:
            raise LMSConflictError(
                "Outcome-mode course is missing its authoring workspace path; "
                "the grader bundle cannot be located."
            )
        grader_root = Path(authoring_root) / "private" / "grader"
        scenarios_dir = grader_root / "scenarios"
        setup_dir = grader_root / "_setup"
        if not scenarios_dir.exists():
            raise LMSConflictError(
                f"Outcome grader bundle missing scenarios directory at {scenarios_dir}."
            )

        # P0 #4 audit: the grader bundle today lives at the mutable
        # authoring workspace, NOT inside the immutable publish snapshot.
        # If the author amends and re-publishes the course after a learner
        # has enrolled, the learner's submission is graded against the new
        # bundle. Long-term fix is to ship grader content inside the
        # snapshot. As a stopgap we hash the bundle contents at submit
        # time, stamp the digest onto the submission row, and log it so
        # any post-hoc drift detection (e.g. comparing two submissions
        # against the same snapshot) is possible.
        grader_bundle_digest = _grader_bundle_digest(grader_root)

        learner_starter = workspace_root / "public" / "starter"
        if not learner_starter.exists():
            raise LMSConflictError(
                "Learner workspace has no public/starter directory to grade."
            )

        scenarios = load_scenarios_from_dir(scenarios_dir)
        setup_data_dir = setup_dir if setup_dir.exists() else None

        # Pull capabilities off the spec on disk so the sandbox boot
        # matches the reference-impl boot (durable_state_required etc.).
        capabilities = None
        try:
            from app.services.course_outcome_models import CourseOutcomeSpec
            spec_path = Path(authoring_root) / "private" / "course_spec.json"
            if spec_path.exists():
                spec_obj = CourseOutcomeSpec.model_validate_json(spec_path.read_text())
                capabilities = spec_obj.capabilities
        except Exception:
            # Capability provisioning is best-effort at submit time; a
            # missing or unparseable spec falls back to default boot.
            capabilities = None

        grader = self._outcome_grader or OraclePass(
            sandbox_runner=WorkspaceBootSandboxAdapter()
        )

        # Live LLM judge (haiku tier) when the env is configured. The
        # judge rubrics call ``router.parse_structured(tier=haiku, ...)``;
        # a missing router can either (a) silently abstain on judged
        # rubrics, or (b) cause submit to refuse to grade. Staging/prod
        # must use (b) so a misconfigured judge can never silently pass
        # learners. Local dev / CI can opt into (a) via the env var
        # below.
        router: Any | None = None
        try:
            from app.services.llm_router import get_default_router
            router = get_default_router()
        except Exception:
            router = None

        strict_judge = os.environ.get("COURSE_GEN_REQUIRE_LLM_JUDGE", "true").lower() == "true"
        if strict_judge and router is None:
            scenarios_needing_judge = sorted({
                rubric_kind
                for scenario in scenarios
                for rubric_kind in _rubric_kinds(scenario)
                if rubric_kind.startswith("llm_judge_")
            })
            if scenarios_needing_judge:
                raise LMSConflictError(
                    "Grader refused: this course uses LLM-judge rubrics "
                    f"({', '.join(scenarios_needing_judge)}) but no LLM "
                    "router is available on this host. Configure "
                    "ANTHROPIC_API_KEY (or set "
                    "COURSE_GEN_REQUIRE_LLM_JUDGE=false for offline dev) "
                    "and resubmit."
                )

        try:
            pass_result = grader.run(
                scenarios=scenarios,
                reference_impl_dir=learner_starter,
                setup_data_dir=setup_data_dir,
                router=router,
                capabilities=capabilities,
            )
        except TypeError:
            # Fake graders without ``capabilities`` / ``router`` kwargs.
            try:
                pass_result = grader.run(
                    scenarios=scenarios,
                    reference_impl_dir=learner_starter,
                    setup_data_dir=setup_data_dir,
                    router=router,
                )
            except TypeError:
                pass_result = grader.run(
                    scenarios=scenarios,
                    reference_impl_dir=learner_starter,
                    setup_data_dir=setup_data_dir,
                )

        # Aggregate per-scenario verdicts into the legacy GradeReport
        # shape so the existing experience / scorecard UI rendering
        # works unchanged.
        #
        # Status mapping mirrors ``ScenarioVerdictReport.overall_status``:
        #   any verdict status="fail"   -> scenario failed
        #   all verdicts status="pass"  -> scenario passed
        #   mix of pass + abstain       -> scenario passed (with diagnostic)
        #
        # ``abstain`` is treated as not-a-failure on purpose: LLM-judge
        # rubrics explicitly abstain when no router is configured
        # (see ``scenario_rubrics_llm.py``) and the design contract is
        # "judge availability never blocks grading". Failing the learner
        # for an infra concern would be wrong; we surface the abstain
        # rationale in diagnostics so they can still see what wasn't
        # judged.
        # Look up scenarios by id so we can surface the scenario's prose
        # ``description`` as each test's summary — that's the "WHAT was
        # being tested" line learners need. The bundle is the source of
        # truth, not the oracle output (which only carries id/category).
        scenario_by_id = {s.id: s for s in scenarios}

        results: list[TestGradeResult] = []
        for output in pass_result.scenario_outputs:
            verdict_statuses: list[str] = []
            raw_diagnostics: list[str] = []
            if output.aborted and output.abort_reason:
                raw_diagnostics.append(output.abort_reason)
            for kind, verdict_dump in output.verdicts:
                verdict = Verdict.model_validate(verdict_dump)
                verdict_statuses.append(verdict.status)
                # ``abstain`` rationales are infrastructure noise ("no LLM
                # router configured") — we treat abstain as pass anyway,
                # so drop them from the surfaced diagnostics. Only ``fail``
                # verdicts get translated and surfaced to the learner.
                if verdict.status == "fail" and verdict.rationale:
                    raw_diagnostics.append(f"{kind} ({verdict.status}): {verdict.rationale}")

            if output.aborted:
                scenario_passed = False
            elif not verdict_statuses:
                scenario_passed = False
            elif any(s == "fail" for s in verdict_statuses):
                scenario_passed = False
            else:
                # all pass, or mix of pass + abstain
                scenario_passed = True

            # Summary = the scenario's own prose description (tells the
            # learner WHAT behavior is being tested), with a status
            # prefix for skimmability.
            scenario = scenario_by_id.get(output.scenario_id)
            description = (scenario.description if scenario else "").strip()
            if scenario_passed:
                summary = description or f"Scenario `{output.scenario_id}` passed"
            else:
                summary = description or f"Scenario `{output.scenario_id}` failed"

            # Translate diagnostics into plain-English advice. Drop
            # duplicates (two rubrics emitting identical text).
            seen: set[str] = set()
            diagnostics: list[str] = []
            for raw in raw_diagnostics:
                advice = humanize_diagnostic(raw)
                if advice and advice not in seen:
                    seen.add(advice)
                    diagnostics.append(advice)

            results.append(
                TestGradeResult(
                    test_id=output.scenario_id,
                    test_type="scenario",
                    kind=output.category,
                    status=GradeStatus.passed if scenario_passed else GradeStatus.failed,
                    score=1.0 if scenario_passed else 0.0,
                    summary=summary,
                    diagnostics=diagnostics,
                )
            )

        total = len(results)
        passed = sum(1 for r in results if r.status == GradeStatus.passed)
        failed = total - passed
        pass_rate = (passed / total) if total else 0.0
        overall = GradeStatus.passed if total > 0 and passed == total else GradeStatus.failed

        deliverable_id = deliverable.deliverable_id
        grade_report = DeliverableGradeReport(
            deliverable_id=deliverable_id,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            pass_rate=pass_rate,
            status=overall,
            results=results,
        )
        feedback = build_outcome_feedback(results)
        # Optional LLM rewrite of the headline — pass the spec context so
        # haiku can tie the failure mode to the course's quality bars.
        # Falls back to the deterministic headline on any error.
        if feedback and router is not None:
            spec_quality_bars: list[dict[str, Any]] = []
            spec_title = deliverable.title
            spec_goal = deliverable.objective
            try:
                from app.services.course_outcome_models import CourseOutcomeSpec
                spec_path = Path(authoring_root) / "private" / "course_spec.json"
                if spec_path.exists():
                    spec_obj = CourseOutcomeSpec.model_validate_json(spec_path.read_text())
                    spec_title = spec_obj.title
                    spec_goal = spec_obj.goal
                    spec_quality_bars = [
                        {"id": bar.id, "metric_description": bar.metric_description}
                        for bar in spec_obj.quality_bars
                    ]
            except Exception:
                pass
            rewritten = _rewrite_feedback_with_llm(
                feedback=feedback,
                spec_title=spec_title,
                spec_goal=spec_goal,
                quality_bars=spec_quality_bars,
                passed=passed,
                total=total,
                router=router,
            )
            if rewritten:
                feedback.learner_feedback = rewritten

        review_area = ReviewAreaGradeReport(
            deliverable_id=deliverable_id,
            title=deliverable.title,
            objective=deliverable.objective,
            deliverable_index=deliverable.deliverable_index,
            grade_report=grade_report,
            feedback=feedback,
        )
        assignment_report = AssignmentGradeReport(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            pass_rate=pass_rate,
            status=overall,
            review_areas=[review_area],
        )

        submission_group_id = f"submission_{uuid4().hex[:12]}"
        created_at = datetime.now(UTC)
        # P0 #4 audit: persist the grader-bundle fingerprint on the
        # submission row so any post-hoc drift between two submissions
        # against the same publish snapshot is recoverable. The bundle
        # lives at a mutable authoring path today; until that bundle
        # ships inside the immutable publish snapshot, this is the
        # durable audit trail.
        submission = LearnerSubmissionRecord(
            id=f"{submission_group_id}_{deliverable_id.replace('/', '_')}",
            submission_group_id=submission_group_id,
            enrollment_id=enrollment.id,
            deliverable_id=deliverable_id,
            created_at=created_at,
            status=overall.value,
            passed_tests=passed,
            total_tests=total,
            pass_rate=pass_rate,
            grade_report=grade_report,
            assignment_report=assignment_report,
            grader_bundle_digest=grader_bundle_digest,
        )
        self.store.save_learner_submission(submission)

        refreshed = enrollment.model_copy(deep=True)
        for item in refreshed.deliverables:
            if item.deliverable_id == deliverable_id:
                item.latest_submission = submission
                item.status = (
                    LearnerDeliverableStatus.passed
                    if overall == GradeStatus.passed
                    else LearnerDeliverableStatus.available
                )
        if overall == GradeStatus.passed:
            refreshed.current_deliverable_id = None
            refreshed.status = LearnerEnrollmentStatus.completed
        else:
            refreshed.current_deliverable_id = deliverable_id
            refreshed.status = LearnerEnrollmentStatus.active
        refreshed.updated_at = datetime.now(UTC)
        self.store.save_learner_enrollment(refreshed)
        return self.get_deliverable_experience(refreshed.id, deliverable_id)

    def _workspace_root(self, enrollment: LearnerEnrollment) -> Path:
        return (
            self.base_dir
            / enrollment.learner_id
            / enrollment.shared_workflow_run_id
            / "workspace"
        )

    def _workspace_context(
        self,
        enrollment_id: str,
        deliverable_id: str | None = None,
    ) -> tuple[LearnerEnrollment, LearnerDeliverableProgress, LearnerDeliverablePackage, Path]:
        enrollment = self.get_enrollment(enrollment_id)
        deliverable = self._resolve_target_deliverable(enrollment, deliverable_id)
        snapshot = self._require_snapshot(enrollment.publish_snapshot_id)
        deliverable_package = self._resolve_deliverable_package(snapshot, deliverable.deliverable_id)
        workspace_root = self._workspace_root(enrollment)
        self._ensure_workspace_seeded(enrollment, snapshot)
        return enrollment, deliverable, deliverable_package, workspace_root

    def _require_published_course(self, course_run_id: str) -> CourseRun:
        course_run = self.store.get_course_run(course_run_id)
        if course_run is None:
            raise KeyError(course_run_id)
        if course_run.status != CourseRunStatus.published:
            raise LMSConflictError("Only published courses can be enrolled.")
        return course_run

    def _require_supported_snapshot(self, course_run: CourseRun) -> PublishSnapshot:
        snapshot = self._latest_snapshot(course_run)
        if snapshot is None:
            raise LMSConflictError("This published course does not have a learner-ready publish snapshot yet.")
        if snapshot.learner_package is None:
            raise LMSConflictError("This published course is not yet packaged for the LMS learner flow.")
        # Outcome-mode courses don't carry a TaskAgentServiceSpec — the
        # legacy spec shape is replaced by the outcome bundle's
        # course_spec.json + scenarios/ + _reference/. Only require
        # ``task_agent_spec`` for legacy multi-deliverable courses.
        is_outcome = bool(
            (course_run.payload_json or {}).get("outcome_state")
        )
        if not is_outcome and snapshot.task_agent_spec is None:
            raise LMSConflictError("This published course is not yet packaged for the LMS learner flow.")
        return snapshot

    def _require_snapshot(self, snapshot_id: str) -> PublishSnapshot:
        snapshot = self.store.get_publish_snapshot(snapshot_id)
        if snapshot is None:
            raise LMSConflictError(f"Publish snapshot '{snapshot_id}' is missing.")
        return snapshot

    def _latest_snapshot(self, course_run: CourseRun) -> PublishSnapshot | None:
        if course_run.latest_publish_snapshot_id:
            snapshot = self.store.get_publish_snapshot(course_run.latest_publish_snapshot_id)
            if snapshot is not None:
                return snapshot
        return self.store.get_latest_publish_snapshot(course_run.id)

    def _lms_support(self, course_run: CourseRun, snapshot: PublishSnapshot | None) -> tuple[bool, str | None]:
        if course_run.status != CourseRunStatus.published:
            return False, "This course is still being prepared and is not available to learners yet."

        # Outcome-mode courses (those whose payload_json carries an
        # ``outcome_state`` blob) don't carry a ``shared_workflow_run_id``
        # or a ``TaskAgentServiceSpec`` — those are legacy fields. Apply
        # a relaxed gate for them: just require the synthesized
        # publish_snapshot exists and has a non-empty learner_package.
        is_outcome = bool(
            (course_run.payload_json or {}).get("outcome_state")
        )
        if is_outcome:
            if snapshot is None:
                return False, "This course is being prepared and is not ready for learners yet."
            if snapshot.learner_package is None:
                return False, "This course is being prepared and is not ready for learners yet."
            if not snapshot.learner_package.deliverables:
                return False, "This course is being prepared and is not ready for learners yet."
            return True, None

        # Legacy multi-deliverable course path (unchanged).
        if not course_run.shared_workflow_run_id:
            return False, "This course is still being prepared and is not available to learners yet."
        if snapshot is None:
            return False, "This course is being prepared and is not ready for learners yet."
        if snapshot.learner_package is None or snapshot.task_agent_spec is None:
            return False, "This course is being prepared and is not ready for learners yet."
        if not snapshot.learner_package.deliverables:
            return False, "This course is being prepared and is not ready for learners yet."
        return True, None

    def _deliverable_progress(self, deliverable_package: LearnerDeliverablePackage) -> LearnerDeliverableProgress:
        return LearnerDeliverableProgress(
            deliverable_id=deliverable_package.deliverable_id,
            title=deliverable_package.title,
            objective=deliverable_package.objective,
            status=LearnerDeliverableStatus.available,
            deliverable_index=deliverable_package.deliverable_index,
            content_markdown=deliverable_package.content_markdown,
            starter_readme=deliverable_package.starter_readme,
            visible_files=list(deliverable_package.visible_files),
        )

    def _ensure_workspace_seeded(self, enrollment: LearnerEnrollment, snapshot: PublishSnapshot) -> Path:
        workspace_root = self._workspace_root(enrollment)
        workspace_root.mkdir(parents=True, exist_ok=True)
        try:
            return seed_workspace_from_snapshot(workspace_root, snapshot)
        except ValueError as exc:
            raise LMSConflictError(str(exc)) from exc

    def _project_brief_markdown(self, snapshot: PublishSnapshot) -> str:
        return project_brief_markdown(snapshot)

    def _learner_assignment_report(
        self,
        snapshot: PublishSnapshot,
        assignment_report: AssignmentGradeReport,
    ) -> AssignmentGradeReport:
        return remap_assignment_report_to_deliverables(snapshot, assignment_report)

    def _resolve_review_area_deliverable(
        self,
        deliverables: list[LearnerDeliverablePackage],
        spec_deliverable_id: str,
        spec_deliverable_order: dict[str, int] | None = None,
    ) -> LearnerDeliverablePackage | None:
        for deliverable in deliverables:
            if deliverable.deliverable_id == spec_deliverable_id:
                return deliverable
        if spec_deliverable_order is not None:
            deliverable_position = spec_deliverable_order.get(spec_deliverable_id)
            if deliverable_position is not None and 0 <= deliverable_position < len(deliverables):
                return deliverables[deliverable_position]
        return None

    def _resolve_workspace_file(self, workspace_root: Path, relative_path: str) -> Path:
        normalized = PurePosixPath(relative_path.strip())
        if not normalized.parts:
            raise LMSConflictError("Workspace file path is required.")
        if normalized.is_absolute():
            raise LMSConflictError("Workspace file path must be relative.")
        if ".." in normalized.parts:
            raise LMSConflictError("Workspace file path must stay inside the learner workspace.")
        target = (workspace_root / Path(*normalized.parts)).resolve()
        root = workspace_root.resolve()
        if root != target and root not in target.parents:
            raise LMSConflictError("Workspace file path must stay inside the learner workspace.")
        return target

    def _guess_media_type(self, path: Path) -> str:
        media_type, _ = mimetypes.guess_type(str(path))
        return media_type or "text/plain"

    def _resolve_target_deliverable(
        self,
        enrollment: LearnerEnrollment,
        requested_deliverable_id: str | None,
    ) -> LearnerDeliverableProgress:
        target_deliverable_id = requested_deliverable_id or enrollment.current_deliverable_id
        if target_deliverable_id is None and enrollment.deliverables:
            target_deliverable_id = enrollment.deliverables[-1].deliverable_id
        for deliverable in enrollment.deliverables:
            if deliverable.deliverable_id == target_deliverable_id:
                return deliverable
        raise LMSConflictError("Could not resolve the active learner deliverable.")

    def _latest_assignment_submission(
        self,
        submissions: list[LearnerSubmissionRecord],
    ) -> LearnerSubmissionRecord | None:
        grouped: dict[str, LearnerSubmissionRecord] = {}
        for submission in submissions:
            group_id = submission.submission_group_id or submission.id
            current = grouped.get(group_id)
            if current is None or submission.created_at > current.created_at:
                grouped[group_id] = submission
        if not grouped:
            return None
        return max(grouped.values(), key=lambda item: item.created_at)

    def _resolve_deliverable_package(self, snapshot: PublishSnapshot, deliverable_id: str) -> LearnerDeliverablePackage:
        learner_package = snapshot.learner_package
        if learner_package is None:
            raise LMSConflictError("This publish snapshot is missing its learner package.")
        for deliverable in learner_package.deliverables:
            if deliverable.deliverable_id == deliverable_id:
                return deliverable
        raise LMSConflictError(f"Deliverable '{deliverable_id}' is not present in publish snapshot '{snapshot.id}'.")

    def _require_enrollment(self, enrollment_id: str) -> LearnerEnrollment:
        enrollment = self.store.get_learner_enrollment(enrollment_id)
        if enrollment is None:
            raise KeyError(enrollment_id)
        return enrollment
