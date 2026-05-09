from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from app.domain.grading import (
    EvalRunEvidence,
    LiveAssignmentGradeReport,
    LiveGradeTaskAgentRequest,
    LiveTaskAgentGradeReport,
    TaskAgentSubmission,
)
from app.domain.task_agent import PublicCheckSpec, TaskAgentServiceSpec
from app.services.task_agent_grader import grade_assignment_submission, grade_task_agent_submission


class TaskAgentRunnerError(RuntimeError):
    """Raised when a learner app cannot be probed successfully."""


ClientFactory = Callable[[str, float], httpx.Client]


def _default_client_factory(base_url: str, timeout_s: float) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s, follow_redirects=False)


class TaskAgentBlackBoxRunner:
    def __init__(self, client_factory: ClientFactory | None = None) -> None:
        self.client_factory = client_factory or _default_client_factory

    def collect_submission(
        self,
        spec: TaskAgentServiceSpec,
        deliverable_id: str,
        request: LiveGradeTaskAgentRequest,
    ) -> TaskAgentSubmission:
        deliverable = next((item for item in spec.deliverables if item.id == deliverable_id), None)
        if deliverable is None:
            raise TaskAgentRunnerError(f"Unknown deliverable '{deliverable_id}'.")
        if not deliverable.public_checks:
            raise TaskAgentRunnerError(f"No visible checks were defined for deliverable '{deliverable_id}'.")

        client = self.client_factory(request.base_url, request.timeout_ms / 1000)
        try:
            runs = [self._execute_public_check(client, check) for check in deliverable.public_checks]
        finally:
            client.close()

        return TaskAgentSubmission(
            submission_id=f"blackbox::{deliverable_id}::{request.base_url}",
            runs=runs,
            metadata={
                "collected_from": request.base_url,
                "deliverable_id": deliverable_id,
                "public_check_ids": [check.id for check in deliverable.public_checks],
            },
        )

    def collect_assignment_submission(
        self,
        spec: TaskAgentServiceSpec,
        request: LiveGradeTaskAgentRequest,
    ) -> TaskAgentSubmission:
        all_checks = [
            (deliverable.id, check)
            for deliverable in spec.deliverables
            for check in deliverable.public_checks
        ]
        if not all_checks:
            raise TaskAgentRunnerError("No visible checks were defined for this assignment.")

        client = self.client_factory(request.base_url, request.timeout_ms / 1000)
        try:
            runs = [self._execute_public_check(client, check) for _deliverable_id, check in all_checks]
        finally:
            client.close()

        return TaskAgentSubmission(
            submission_id=f"blackbox::assignment::{request.base_url}",
            runs=runs,
            metadata={
                "collected_from": request.base_url,
                "scope": "assignment",
                "public_check_ids": [check.id for _deliverable_id, check in all_checks],
            },
        )

    def grade_live(
        self,
        spec: TaskAgentServiceSpec,
        deliverable_id: str,
        request: LiveGradeTaskAgentRequest,
    ) -> LiveTaskAgentGradeReport:
        submission = self.collect_submission(spec, deliverable_id, request)
        grade_report = grade_task_agent_submission(spec, deliverable_id, submission)
        return LiveTaskAgentGradeReport(
            base_url=request.base_url,
            submission=submission,
            grade_report=grade_report,
        )

    def grade_assignment_live(
        self,
        spec: TaskAgentServiceSpec,
        request: LiveGradeTaskAgentRequest,
    ) -> LiveAssignmentGradeReport:
        submission = self.collect_assignment_submission(spec, request)
        assignment_report = grade_assignment_submission(spec, submission)
        return LiveAssignmentGradeReport(
            base_url=request.base_url,
            submission=submission,
            assignment_report=assignment_report,
        )

    def _execute_public_check(
        self,
        client: httpx.Client,
        check: PublicCheckSpec,
    ) -> EvalRunEvidence:
        method = check.request_method.upper()
        request_kwargs: dict[str, Any] = {}
        if method == "GET":
            if check.request_body:
                request_kwargs["params"] = check.request_body
        elif check.request_body:
            request_kwargs["json"] = check.request_body

        started_at = time.perf_counter()
        try:
            response = client.request(method, check.request_path, **request_kwargs)
        except httpx.HTTPError as exc:
            raise TaskAgentRunnerError(f"Request to '{check.request_path}' failed: {exc}") from exc
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        body_text = response.text
        body_payload: dict[str, Any]
        try:
            parsed = response.json()
            body_payload = parsed if isinstance(parsed, dict) else {"value": parsed}
        except ValueError:
            body_payload = {"value": body_text}

        haystack = body_text or json.dumps(body_payload, sort_keys=True, ensure_ascii=True)
        notes: list[str] = []
        if response.status_code != check.expected_status:
            notes.append(f"Expected HTTP {check.expected_status} but observed HTTP {response.status_code}.")
        missing = [
            snippet
            for snippet in check.expected_response_contains
            if str(snippet).strip() and str(snippet).lower() not in haystack.lower()
        ]
        if missing:
            notes.append("Response is missing expected content: " + ", ".join(sorted(missing)) + ".")

        output = {
            **body_payload,
            "_coursegen_http_status": response.status_code,
            "_coursegen_body_text": haystack,
        }
        return EvalRunEvidence(
            run_id=f"public-check::{check.id}",
            case_id=check.id,
            dry_run=False,
            output=output,
            trace_events=[],
            step_count=1,
            latency_ms=latency_ms,
            cost_usd=0.0,
            tool_calls=[],
            approvals=[],
            escalations=[],
            failure_injections=[],
            fallback_actions=[],
            resumed_after_pause=False,
            success=not notes,
            quality_score=None,
            notes=notes,
        )
