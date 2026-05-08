#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx


SCENARIOS = [
    {
        "slug": "flight-booking",
        "goal": "Build a flight booking system that is production ready. Mock external dependent services where required.",
        "learning_outcomes": [
            "Keep seat inventory correct under load.",
            "Explain the tradeoffs between different locking strategies.",
            "Use caching carefully for availability reads.",
        ],
        "creator_choices": {
            "starter_type": "partial_implementation",
            "primary_database": "postgres",
            "cache_backend": "redis",
            "tech_stack": [],
        },
    },
    {
        "slug": "support-agent",
        "goal": "Build a production-ready customer support agent that triages requests, uses tools safely, and can escalate when confidence is low.",
        "learning_outcomes": [
            "Define a stable support run contract.",
            "Use tools with bounded safety and approval rules.",
            "Add traces and evals that make failures understandable.",
        ],
        "creator_choices": {
            "starter_type": "working_buggy",
            "primary_database": None,
            "cache_backend": None,
            "tech_stack": [],
        },
    },
    {
        "slug": "internal-docs-rag",
        "goal": "Build a grounded internal docs assistant that answers from a visible corpus with citations and abstains when support is missing.",
        "learning_outcomes": [
            "Implement retrieval over a visible knowledge source.",
            "Return grounded answers with citations.",
            "Abstain safely when the corpus does not support an answer.",
        ],
        "creator_choices": {
            "starter_type": "partial_implementation",
            "primary_database": None,
            "cache_backend": None,
            "tech_stack": [],
        },
    },
]


class SmokeError(RuntimeError):
    pass


def _request_json(client: httpx.Client, method: str, path: str, **kwargs) -> dict[str, Any]:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise SmokeError(f"{method} {path} failed with HTTP {response.status_code}: {response.text.strip()}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise SmokeError(f"{method} {path} returned non-object JSON.")
    return payload


def _wait_for_condition(
    client: httpx.Client,
    path: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    predicate,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        payload = _request_json(client, "GET", path)
        last_payload = payload
        if predicate(payload):
            return payload
        time.sleep(poll_interval_s)
    raise SmokeError(
        f"Timed out waiting for {path}. Last payload:\n{json.dumps(last_payload, indent=2)}"
    )


def _run_scenario(client: httpx.Client, scenario: dict[str, Any]) -> dict[str, Any]:
    planned = _request_json(
        client,
        "POST",
        "/v1/course-generation/creator-plan",
        json={
            "goal": scenario["goal"],
            "learning_outcomes": scenario["learning_outcomes"],
            "creator_choices": scenario["creator_choices"],
        },
    )
    plan = planned["plan"]
    if plan.get("goal") != scenario["goal"]:
        raise SmokeError(f"Creator plan lost the original goal for scenario '{scenario['slug']}'.")
    if not planned.get("learning_outcomes"):
        raise SmokeError(f"Creator plan returned no derived outcomes for scenario '{scenario['slug']}'.")
    if not plan.get("deliverables"):
        raise SmokeError(f"Creator plan returned no deliverables for scenario '{scenario['slug']}'.")

    created = _request_json(
        client,
        "POST",
        "/v1/course-runs/from-creator-plan-async",
        json={"plan": plan},
    )
    course_run_id = created["course_run"]["id"]

    deadline = time.time() + 30
    creator_view: dict[str, Any] | None = None
    while time.time() < deadline:
        creator_view = _request_json(client, "GET", f"/v1/course-runs/{course_run_id}/creator-view")
        course_run = creator_view["course_run"]
        if course_run.get("shared_workflow_run_id") or course_run.get("deliverables"):
            break
        time.sleep(0.25)
    if creator_view is None:
        raise SmokeError(f"Creator view never loaded for scenario '{scenario['slug']}'.")
    diagnostics = creator_view.get("diagnostics", [])
    if not diagnostics:
        raise SmokeError(f"Creator view returned no diagnostics for scenario '{scenario['slug']}'.")

    return {
        "slug": scenario["slug"],
        "course_run_id": course_run_id,
        "title": creator_view["course_run"]["title"],
        "stage": creator_view["course_run"]["stage"],
        "status": creator_view["course_run"]["status"],
        "deliverable_titles": [deliverable["title"] for deliverable in creator_view["review"]["deliverables"]],
        "diagnostic_codes": [item["code"] for item in diagnostics],
    }


def _run_full_flow(
    client: httpx.Client,
    scenario: dict[str, Any],
    *,
    creator_timeout_s: float,
) -> dict[str, Any]:
    created = _run_scenario(client, scenario)
    course_run_id = created["course_run_id"]

    creator_view = _wait_for_condition(
        client,
        f"/v1/course-runs/{course_run_id}/creator-view",
        timeout_s=creator_timeout_s,
        poll_interval_s=0.5,
        predicate=lambda payload: payload["course_run"]["stage"] in {"ready_to_publish", "published"},
    )
    course_run = creator_view["course_run"]
    if course_run["stage"] == "ready_to_publish":
        published = _request_json(client, "POST", f"/v1/course-runs/{course_run_id}/publish")
        course_run = published

    catalog = _request_json(client, "GET", "/v1/lms/catalog")
    catalog_course = next(
        (course for course in catalog["courses"] if course["course_run_id"] == course_run_id),
        None,
    )
    if catalog_course is None:
        raise SmokeError(f"Published course '{course_run_id}' is missing from the LMS catalog.")
    if not catalog_course.get("supported_for_lms"):
        raise SmokeError(
            f"Published course '{course_run_id}' is not learner-ready: {catalog_course.get('support_reason')}"
        )

    enrollment = _request_json(
        client,
        "POST",
        "/v1/lms/enrollments",
        json={"course_run_id": course_run_id},
    )
    enrollment_id = enrollment["id"]

    experience = _request_json(client, "GET", f"/v1/lms/enrollments/{enrollment_id}/experience")
    if not experience.get("project_brief_markdown", "").strip():
        raise SmokeError("Learner experience is missing the project brief.")

    launched = _request_json(
        client,
        "POST",
        f"/v1/lms/enrollments/{enrollment_id}/workspace",
        json={},
    )
    workspace_session = launched.get("workspace_session") or launched["deliverables"][0].get("workspace_session")
    if not workspace_session or not workspace_session.get("editor_url"):
        raise SmokeError(f"Workspace launch did not return an editor URL: {json.dumps(launched, indent=2)}")

    deliverables = launched.get("deliverables", [])
    first_deliverable = deliverables[0] if deliverables else None
    if first_deliverable is None:
        raise SmokeError("Enrollment is missing deliverables after launch.")

    reviewed = _request_json(
        client,
        "POST",
        f"/v1/lms/enrollments/{enrollment_id}/submit",
        json={"deliverable_id": first_deliverable["deliverable_id"]},
    )
    latest_submission = reviewed.get("latest_assignment_submission")
    latest_report = reviewed.get("latest_assignment_report")
    if latest_submission is None or latest_report is None:
        raise SmokeError(f"Submit did not return the assignment review report: {json.dumps(reviewed, indent=2)}")

    return {
        **created,
        "final_stage": course_run["stage"],
        "publish_snapshot_id": course_run.get("latest_publish_snapshot_id"),
        "enrollment_id": enrollment_id,
        "workspace_url": workspace_session["editor_url"],
        "review_status": latest_submission["status"],
        "passed_tests": latest_report["passed_tests"],
        "total_tests": latest_report["total_tests"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the creator-plan -> draft -> creator-view flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="Base URL for the local app.")
    parser.add_argument(
        "--scenario",
        choices=[scenario["slug"] for scenario in SCENARIOS],
        action="append",
        help="Only run specific scenario(s). Defaults to all.",
    )
    parser.add_argument(
        "--full-flow",
        action="store_true",
        help="Also publish, enroll, launch the shared workspace, and submit the starter project for review.",
    )
    parser.add_argument(
        "--creator-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for async creator generation to reach ready_to_publish.",
    )
    args = parser.parse_args()

    selected = [scenario for scenario in SCENARIOS if not args.scenario or scenario["slug"] in args.scenario]
    if not selected:
        raise SmokeError("No creator-flow scenarios selected.")

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        if args.full_flow:
            results = [_run_full_flow(client, scenario, creator_timeout_s=args.creator_timeout) for scenario in selected]
        else:
            results = [_run_scenario(client, scenario) for scenario in selected]

    print(json.dumps({"base_url": args.base_url, "results": results}, indent=2))


if __name__ == "__main__":
    main()
