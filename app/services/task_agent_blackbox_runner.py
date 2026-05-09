from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable

import httpx

from app.domain.grader import ControlFlag
from app.domain.grading import (
    LiveAssignmentGradeReport,
    ApprovalRecord,
    EvalRunEvidence,
    EscalationRecord,
    FailureInjectionRecord,
    FallbackActionRecord,
    LiveGradeTaskAgentRequest,
    LiveTaskAgentGradeReport,
    TaskAgentSubmission,
    ToolCallRecord,
)
from app.domain.task_agent import TaskAgentServiceSpec
from app.services.grader_planner import build_all_task_agent_review_area_plans, build_task_agent_grader_plan
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
        plan = build_task_agent_grader_plan(spec, deliverable_id)
        ordered_cases = OrderedDict((case.id, case) for case in spec.eval_dataset.cases)
        case_ids: set[str] = set()
        dry_run_case_ids: set[str] = set()

        for entry in plan.entries:
            if entry.dependencies.dataset_id == spec.eval_dataset.id:
                case_ids.update(ordered_cases.keys())
            case_ids.update(entry.dependencies.eval_case_ids)
            if request.include_dry_runs and ControlFlag.dry_run in entry.controls:
                dry_run_case_ids.update(entry.dependencies.eval_case_ids)

        if not case_ids:
            raise TaskAgentRunnerError(f"No eval cases were activated for deliverable '{deliverable_id}'.")

        client = self.client_factory(request.base_url, request.timeout_ms / 1000)
        try:
            runs: list[EvalRunEvidence] = []
            for case_id in ordered_cases.keys():
                if case_id not in case_ids:
                    continue
                case = ordered_cases[case_id]
                runs.append(self._execute_case(client, case.id, case.input, request, dry_run=False))
                if case_id in dry_run_case_ids:
                    dry_input = dict(case.input)
                    dry_input["dry_run"] = True
                    runs.append(self._execute_case(client, case.id, dry_input, request, dry_run=True))
        finally:
            client.close()

        return TaskAgentSubmission(
            submission_id=f"blackbox::{deliverable_id}::{request.base_url}",
            runs=runs,
            metadata={"collected_from": request.base_url, "deliverable_id": deliverable_id},
        )

    def collect_assignment_submission(
        self,
        spec: TaskAgentServiceSpec,
        request: LiveGradeTaskAgentRequest,
    ) -> TaskAgentSubmission:
        plans = build_all_task_agent_review_area_plans(spec)
        ordered_cases = OrderedDict((case.id, case) for case in spec.eval_dataset.cases)
        case_ids: set[str] = set()
        dry_run_case_ids: set[str] = set()

        for plan in plans.deliverable_plans:
            for entry in plan.entries:
                if entry.dependencies.dataset_id == spec.eval_dataset.id:
                    case_ids.update(ordered_cases.keys())
                case_ids.update(entry.dependencies.eval_case_ids)
                if request.include_dry_runs and ControlFlag.dry_run in entry.controls:
                    dry_run_case_ids.update(entry.dependencies.eval_case_ids)

        if not case_ids:
            raise TaskAgentRunnerError("No eval cases were activated for the assignment review.")

        client = self.client_factory(request.base_url, request.timeout_ms / 1000)
        try:
            runs: list[EvalRunEvidence] = []
            for case_id in ordered_cases.keys():
                if case_id not in case_ids:
                    continue
                case = ordered_cases[case_id]
                runs.append(self._execute_case(client, case.id, case.input, request, dry_run=False))
                if case_id in dry_run_case_ids:
                    dry_input = dict(case.input)
                    dry_input["dry_run"] = True
                    runs.append(self._execute_case(client, case.id, dry_input, request, dry_run=True))
        finally:
            client.close()

        return TaskAgentSubmission(
            submission_id=f"blackbox::assignment::{request.base_url}",
            runs=runs,
            metadata={"collected_from": request.base_url, "scope": "assignment"},
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

    def _execute_case(
        self,
        client: httpx.Client,
        case_id: str,
        payload: dict[str, Any],
        request: LiveGradeTaskAgentRequest,
        *,
        dry_run: bool,
    ) -> EvalRunEvidence:
        run_response = self._request_json(client, "POST", "/run", json=payload)
        run_id = str(run_response.get("run_id") or run_response.get("id") or f"{case_id}-inline")
        state = dict(run_response)
        resumed_after_pause = False

        for _ in range(request.max_poll_attempts):
            status = str(state.get("status", "completed")).lower()
            if status in {"completed", "failed", "done", "finished"}:
                break
            if status in {"awaiting_approval", "pending_approval"}:
                if not request.auto_approve:
                    break
                self._request_json(client, "POST", f"/approve/{run_id}", json={"approved": True})
                resumed_after_pause = True
            if request.poll_interval_ms:
                time.sleep(request.poll_interval_ms / 1000)
            state = self._request_json(client, "GET", f"/runs/{run_id}")
        else:
            raise TaskAgentRunnerError(f"Run '{run_id}' for case '{case_id}' did not settle in time.")

        try:
            trace_payload = self._request_json(client, "GET", f"/trace/{run_id}")
        except TaskAgentRunnerError:
            trace_payload = {}

        return self._merge_evidence(case_id, run_id, dry_run, state, trace_payload, resumed_after_pause)

    def _merge_evidence(
        self,
        case_id: str,
        run_id: str,
        dry_run: bool,
        state: dict[str, Any],
        trace_payload: dict[str, Any],
        resumed_after_pause: bool,
    ) -> EvalRunEvidence:
        trace_events = self._parse_trace_events(state, trace_payload)
        return EvalRunEvidence(
            run_id=run_id,
            case_id=case_id,
            dry_run=dry_run,
            output=self._parse_output(state),
            trace_events=trace_events,
            step_count=int(state.get("step_count", len(trace_events) or len(state.get("tool_calls", [])))),
            latency_ms=int(state.get("latency_ms", 0)),
            cost_usd=float(state.get("cost_usd", 0.0)),
            tool_calls=self._parse_records(state.get("tool_calls", []), ToolCallRecord, include_order=True),
            approvals=self._parse_records(state.get("approvals", []), ApprovalRecord, include_order=True),
            escalations=self._parse_records(state.get("escalations", []), EscalationRecord, include_order=True),
            failure_injections=self._parse_records(state.get("failure_injections", []), FailureInjectionRecord),
            fallback_actions=self._parse_records(state.get("fallback_actions", []), FallbackActionRecord),
            resumed_after_pause=bool(state.get("resumed_after_pause", resumed_after_pause)),
            success=bool(state.get("success", str(state.get("status", "")).lower() in {"completed", "done", "finished"})),
            quality_score=state.get("quality_score"),
            notes=[str(item) for item in state.get("notes", [])],
        )

    def _parse_output(self, state: dict[str, Any]) -> dict[str, Any]:
        output = state.get("output")
        if isinstance(output, dict):
            return output
        return {
            key: value
            for key, value in state.items()
            if key
            not in {
                "run_id",
                "id",
                "status",
                "trace_events",
                "tool_calls",
                "approvals",
                "escalations",
                "failure_injections",
                "fallback_actions",
                "step_count",
                "latency_ms",
                "cost_usd",
                "success",
                "quality_score",
                "notes",
                "resumed_after_pause",
            }
        }

    def _parse_trace_events(self, state: dict[str, Any], trace_payload: dict[str, Any]) -> list[str]:
        events: list[str] = []
        for source in (trace_payload.get("events"), state.get("trace_events")):
            if not source:
                continue
            for item in source:
                if isinstance(item, str):
                    if item not in events:
                        events.append(item)
                elif isinstance(item, dict):
                    event_name = item.get("type") or item.get("event_type") or item.get("name")
                    if event_name and event_name not in events:
                        events.append(str(event_name))
        return events

    def _parse_records(self, payload: Any, model, *, include_order: bool = False):
        records: list[Any] = []
        if not isinstance(payload, list):
            return records
        for index, item in enumerate(payload):
            if isinstance(item, model):
                records.append(item)
            elif isinstance(item, dict):
                normalized = dict(item)
                if include_order and "order" not in normalized:
                    normalized["order"] = index
                normalized = self._normalize_record_payload(normalized, model, index=index)
                records.append(model.model_validate(normalized))
        return records

    def _normalize_record_payload(self, payload: dict[str, Any], model, *, index: int) -> dict[str, Any]:
        normalized = dict(payload)
        if model is ApprovalRecord:
            if "approved" not in normalized and "status" in normalized:
                normalized["approved"] = str(normalized["status"]).lower() in {"approved", "accepted", "ok"}
            if not normalized.get("approval_id"):
                tool_id = str(normalized.get("tool_id") or "approval")
                order = normalized.get("order", index)
                normalized["approval_id"] = f"approval::{tool_id}::{order}"
        return normalized

    def _request_json(self, client: httpx.Client, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TaskAgentRunnerError(f"Request to '{path}' failed: {exc}") from exc
        if response.status_code >= 400:
            raise TaskAgentRunnerError(f"Request to '{path}' failed with HTTP {response.status_code}.")
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise TaskAgentRunnerError(f"Request to '{path}' did not return JSON.") from exc
        if not isinstance(payload, dict):
            raise TaskAgentRunnerError(f"Request to '{path}' returned unexpected JSON payload.")
        return payload
