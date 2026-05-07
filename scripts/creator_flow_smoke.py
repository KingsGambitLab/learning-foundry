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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the creator-plan -> draft -> creator-view flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="Base URL for the local app.")
    parser.add_argument(
        "--scenario",
        choices=[scenario["slug"] for scenario in SCENARIOS],
        action="append",
        help="Only run specific scenario(s). Defaults to all.",
    )
    args = parser.parse_args()

    selected = [scenario for scenario in SCENARIOS if not args.scenario or scenario["slug"] in args.scenario]
    if not selected:
        raise SmokeError("No creator-flow scenarios selected.")

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        results = [_run_scenario(client, scenario) for scenario in selected]

    print(json.dumps({"base_url": args.base_url, "results": results}, indent=2))


if __name__ == "__main__":
    main()
