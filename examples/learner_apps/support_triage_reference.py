from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.services.examples import get_support_triage_passing_submission

app = FastAPI(title="Support Triage Reference Learner App")

REFERENCE_SUBMISSION = get_support_triage_passing_submission()
REFERENCE_RUNS = {run.run_id: run.model_dump(mode="json") for run in REFERENCE_SUBMISSION.runs}
RUNS: dict[str, dict] = {}


def _clone_reference(run_id: str) -> dict:
    return deepcopy(REFERENCE_RUNS[run_id])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_agent(payload: dict):
    ticket_id = payload.get("ticket_id")
    dry_run = bool(payload.get("dry_run", False))

    if ticket_id == "T-100" and dry_run:
        run = _clone_reference("run-billing-dry-001")
        run["run_id"] = uuid4().hex
        run["status"] = "completed"
        RUNS[run["run_id"]] = run
        return {"run_id": run["run_id"], "status": "completed", **_response_shape(run)}

    if ticket_id == "T-100":
        final_run = _clone_reference("run-billing-001")
        run_id = uuid4().hex
        final_run["run_id"] = run_id
        RUNS[run_id] = {
            "status": "awaiting_approval",
            "pending": True,
            "final": final_run,
        }
        return {"run_id": run_id, "status": "awaiting_approval"}

    if ticket_id == "T-101":
        run = _clone_reference("run-outage-001")
    elif ticket_id == "T-102":
        run = _clone_reference("run-policy-001")
    else:
        raise HTTPException(status_code=404, detail=f"Unknown ticket '{ticket_id}'.")

    run["run_id"] = uuid4().hex
    run["status"] = "completed"
    RUNS[run["run_id"]] = run
    return {"run_id": run["run_id"], "status": "completed", **_response_shape(run)}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    run = RUNS[run_id]
    if run.get("pending"):
        return {"run_id": run_id, "status": "awaiting_approval"}
    return {"run_id": run_id, "status": run.get("status", "completed"), **_response_shape(run)}


@app.get("/trace/{run_id}")
def get_trace(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    run = RUNS[run_id]
    if run.get("pending"):
        events = ["run_started", "model_called", "tool_selected", "tool_called", "tool_result", "approval_requested"]
        return {"run_id": run_id, "events": events}
    return {"run_id": run_id, "events": run.get("trace_events", [])}


@app.post("/approve/{run_id}")
def approve(run_id: str, payload: dict | None = None):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.")
    run = RUNS[run_id]
    if run.get("pending"):
        final_run = run["final"]
        final_run["status"] = "completed"
        RUNS[run_id] = final_run
        return {"run_id": run_id, "status": "completed", **_response_shape(final_run)}
    return {"run_id": run_id, "status": run.get("status", "completed"), **_response_shape(run)}


def _response_shape(run: dict) -> dict:
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
