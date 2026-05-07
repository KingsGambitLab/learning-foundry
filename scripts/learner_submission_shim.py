#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import uuid4

import httpx


BAD_APP = """from fastapi import FastAPI

app = FastAPI(title="Bad learner app")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_agent(payload: dict):
    ticket_id = payload.get("ticket_id", "unknown")
    return {
        "run_id": f"bad-{ticket_id}",
        "status": "completed",
        "output": {
            "disposition": "resolve",
            "priority": "low",
            "reply_draft": "This is a placeholder response.",
            "confidence": 0.2,
            "needs_human": False,
        },
    }
"""


GOOD_APP = """from fastapi import FastAPI

app = FastAPI(title="Good learner app")

RESPONSES = {
    "T-100": {
        "disposition": "resolve",
        "priority": "medium",
        "reply_draft": "We reviewed your account and will reverse the duplicate charge.",
        "confidence": 0.94,
        "needs_human": False,
    },
    "T-101": {
        "disposition": "escalate",
        "priority": "urgent",
        "reply_draft": "We are escalating this outage to the on-call support team right now.",
        "confidence": 0.97,
        "needs_human": True,
    },
    "T-102": {
        "disposition": "needs_info",
        "priority": "high",
        "reply_draft": "I need a little more contract context before I can answer that safely.",
        "confidence": 0.81,
        "needs_human": True,
    },
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_agent(payload: dict):
    ticket_id = payload.get("ticket_id", "unknown")
    output = RESPONSES.get(
        ticket_id,
        {
            "disposition": "needs_info",
            "priority": "medium",
            "reply_draft": "I need more information before I can help.",
            "confidence": 0.5,
            "needs_human": True,
        },
    )
    return {
        "run_id": f"good-{ticket_id}",
        "status": "completed",
        "output": output,
    }
"""


class ShimError(RuntimeError):
    pass


def _request_json(client: httpx.Client, method: str, path: str, **kwargs) -> dict[str, Any]:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        detail = response.text.strip()
        raise ShimError(f"{method} {path} failed with HTTP {response.status_code}: {detail}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ShimError(f"{method} {path} returned non-object JSON.")
    return payload


def _current_module(enrollment: dict[str, Any], module_id: str) -> dict[str, Any]:
    for module in enrollment.get("modules", []):
        if module.get("module_id") == module_id:
            return module
    raise ShimError(f"Module '{module_id}' is missing from the learner enrollment.")


def _submission_summary(experience: dict[str, Any]) -> dict[str, Any]:
    submissions = experience.get("submissions", [])
    latest = max(submissions, key=lambda item: item.get("created_at", "")) if submissions else None
    active_module = experience.get("active_module", {})
    return {
        "module_id": active_module.get("module_id"),
        "status": latest.get("status") if latest else None,
        "passed_tests": latest.get("passed_tests") if latest else None,
        "total_tests": latest.get("total_tests") if latest else None,
        "pass_rate": latest.get("pass_rate") if latest else None,
    }


def run_shim(base_url: str, course_run_id: str | None, learner_id: str | None) -> dict[str, Any]:
    learner_id = learner_id or f"shim-{uuid4().hex[:10]}"
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0) as client:
        catalog = _request_json(client, "GET", "/v1/lms/catalog")
        courses = catalog.get("courses", [])
        if course_run_id is None:
            supported = [course for course in courses if course.get("supported_for_lms")]
            if not supported:
                raise ShimError("No learner-ready published course is available in the LMS catalog.")
            target_course = supported[0]
            course_run_id = str(target_course["course_run_id"])
        else:
            target_course = next((course for course in courses if course.get("course_run_id") == course_run_id), None)
            if target_course is None:
                raise ShimError(f"Course '{course_run_id}' is not present in the LMS catalog.")
            if not target_course.get("supported_for_lms"):
                raise ShimError(f"Course '{course_run_id}' is not learner-ready: {target_course.get('support_reason')}")

        enrollment = _request_json(
            client,
            "POST",
            "/v1/lms/enrollments",
            json={"course_run_id": course_run_id, "learner_id": learner_id},
        )
        enrollment_id = str(enrollment["id"])
        current_module_id = str(enrollment["current_module_id"])

        workspace_enrollment = _request_json(
            client,
            "POST",
            f"/v1/lms/enrollments/{enrollment_id}/workspace",
            json={"module_id": current_module_id},
        )
        module_state = _current_module(workspace_enrollment, current_module_id)
        workspace_session = module_state.get("workspace_session") or {}
        workspace_root = workspace_session.get("workspace_root")
        if not workspace_root:
            raise ShimError("Learner workspace launch did not return a workspace root.")
        files = _request_json(
            client,
            "GET",
            f"/v1/lms/enrollments/{enrollment_id}/workspace/files",
            params={"module_id": current_module_id},
        )
        if "app.py" not in {item["relative_path"] for item in files.get("files", [])}:
            raise ShimError("Learner workspace did not include app.py.")

        _request_json(
            client,
            "PUT",
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            json={
                "module_id": current_module_id,
                "relative_path": "app.py",
                "content": BAD_APP,
            },
        )
        bad_file = _request_json(
            client,
            "GET",
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"module_id": current_module_id, "path": "app.py"},
        )
        if "Bad learner app" not in bad_file.get("content", ""):
            raise ShimError("Bad app content was not persisted through the workspace file API.")
        bad_experience = _request_json(
            client,
            "POST",
            f"/v1/lms/enrollments/{enrollment_id}/submit",
            json={"module_id": current_module_id},
        )
        bad_summary = _submission_summary(bad_experience)
        if bad_summary["status"] != "failed":
            raise ShimError(f"Expected bad submission to fail, got: {json.dumps(bad_summary)}")

        _request_json(
            client,
            "PUT",
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            json={
                "module_id": current_module_id,
                "relative_path": "app.py",
                "content": GOOD_APP,
            },
        )
        good_file = _request_json(
            client,
            "GET",
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"module_id": current_module_id, "path": "app.py"},
        )
        if "Good learner app" not in good_file.get("content", ""):
            raise ShimError("Good app content was not persisted through the workspace file API.")
        good_experience = _request_json(
            client,
            "POST",
            f"/v1/lms/enrollments/{enrollment_id}/submit",
            json={"module_id": current_module_id},
        )
        good_summary = _submission_summary(good_experience)
        if good_summary["status"] != "passed":
            raise ShimError(f"Expected good submission to pass, got: {json.dumps(good_summary)}")

        refreshed_enrollment = _request_json(client, "GET", f"/v1/lms/enrollments/{enrollment_id}")
        current_after_pass = refreshed_enrollment.get("current_module_id")

        return {
            "course_run_id": course_run_id,
            "learner_id": learner_id,
            "enrollment_id": enrollment_id,
            "module_id": current_module_id,
            "workspace_root": workspace_root,
            "bad_submission": bad_summary,
            "good_submission": good_summary,
            "next_module_id": current_after_pass,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive the learner LMS flow end to end with a bad answer and a good answer.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="Base URL for the local app.")
    parser.add_argument("--course-run-id", default=None, help="Optional published course to target.")
    parser.add_argument("--learner-id", default=None, help="Optional learner id to use for the test enrollment.")
    args = parser.parse_args()

    result = run_shim(args.base_url, args.course_run_id, args.learner_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
