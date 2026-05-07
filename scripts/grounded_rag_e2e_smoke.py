#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import uuid4

import httpx

from app.domain.registry import PackageType
from app.services.assignment_design_inference import infer_assignment_design


BAD_APP = """from fastapi import FastAPI

app = FastAPI(title="Bad grounded RAG app")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(payload: dict):
    return {
        "run_id": "bad-run",
        "status": "completed",
        "output": {
            "answer": "Here is a guessed answer without grounded support.",
            "citations": ["doc:grounding_policy"],
            "confidence": 0.95,
            "abstained": False,
        },
    }
"""


GOOD_APP = """from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Good grounded RAG app")

CORPUS = {
    item["doc_id"]: item
    for item in json.loads(Path("data/corpus.json").read_text(encoding="utf-8"))
}
RUNS = {}

QUERY_TO_DOC = {
    "where was ada lovelace born?": "doc:ada_lovelace",
    "what was alan turing known for?": "doc:alan_turing",
}

SUPPORTED_ANSWERS = {
    "doc:ada_lovelace": "Ada Lovelace was born in London, England.",
    "doc:alan_turing": "Alan Turing was an English mathematician and computer scientist.",
}


@app.get("/health")
def health():
    return {"status": "ok"}


def _tool_calls(query: str, doc_id: str | None) -> list[dict]:
    calls = [
        {
            "order": 1,
            "tool_id": "search_corpus",
            "status": "ok",
            "args": {"query": query},
        }
    ]
    if doc_id is None:
        return calls
    if doc_id == "doc:alan_turing":
        calls.append(
            {
                "order": 2,
                "tool_id": "rerank_passages",
                "status": "ok",
                "args": {"query": query},
            }
        )
    calls.append(
        {
            "order": len(calls) + 1,
            "tool_id": "fetch_document",
            "status": "ok",
            "args": {"doc_id": doc_id},
        }
    )
    return calls


@app.post("/run")
def run(payload: dict):
    query = str(payload.get("query", "")).strip().lower()
    doc_id = QUERY_TO_DOC.get(query)
    run_id = f"good-{uuid4().hex[:8]}"
    tool_calls = _tool_calls(query, doc_id)
    trace_events = [
        "run_started",
        "tool_selected",
        "tool_called",
        "tool_result",
        "run_completed",
    ]
    if doc_id is None:
        record = {
            "run_id": run_id,
            "status": "completed",
            "output": {
                "answer": "I do not have enough grounded support in the corpus to answer that question.",
                "citations": [],
                "confidence": 0.24,
                "abstained": True,
            },
            "trace_events": trace_events,
            "step_count": max(len(tool_calls), 1),
            "latency_ms": 140,
            "cost_usd": 0.0036,
            "tool_calls": tool_calls,
            "success": True,
        }
        RUNS[run_id] = record
        return record

    document = CORPUS[doc_id]
    record = {
        "run_id": run_id,
        "status": "completed",
        "output": {
            "answer": SUPPORTED_ANSWERS[doc_id],
            "citations": [doc_id],
            "confidence": 0.96,
            "abstained": False,
        },
        "trace_events": trace_events,
        "step_count": max(len(tool_calls), 1),
        "latency_ms": 220 if doc_id == "doc:alan_turing" else 180,
        "cost_usd": 0.0075 if doc_id == "doc:alan_turing" else 0.0055,
        "tool_calls": tool_calls,
        "success": True,
    }
    if document.get("content"):
        record["notes"] = [f"Used learner-visible corpus document {doc_id}."]
    RUNS[run_id] = record
    return record


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    record = RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return record


@app.get("/trace/{run_id}")
def get_trace(run_id: str):
    record = RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"run_id": run_id, "events": record["trace_events"]}
"""


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


def _module_record(enrollment: dict[str, Any], module_id: str) -> dict[str, Any]:
    for module in enrollment.get("modules", []):
        if module.get("module_id") == module_id:
            return module
    raise SmokeError(f"Could not find learner module '{module_id}'.")


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


def _published_course_from_catalog(
    client: httpx.Client,
    *,
    course_run_id: str | None = None,
) -> dict[str, Any]:
    catalog = _request_json(client, "GET", "/v1/lms/catalog")
    courses = catalog.get("courses", [])
    if course_run_id is None:
        supported_courses = [course for course in courses if course.get("supported_for_lms")]
        if not supported_courses:
            raise SmokeError("No learner-ready published course is available in the LMS catalog.")
        return supported_courses[0]

    target_course = next((course for course in courses if course.get("course_run_id") == course_run_id), None)
    if target_course is None:
        raise SmokeError(f"Published course '{course_run_id}' is not present in the LMS catalog.")
    if not target_course.get("supported_for_lms"):
        raise SmokeError(
            f"Published course '{course_run_id}' is not learner-ready: {target_course.get('support_reason')}"
        )
    return target_course


def _grounded_design_spec(title: str, summary: str, learning_outcomes: list[str]) -> dict[str, Any]:
    inferred = infer_assignment_design(
        title=title,
        problem_statement=summary,
        learning_outcomes=learning_outcomes,
        package_type_hint=PackageType.progressive_codebase_course,
    )
    if inferred.design_spec is None:
        raise SmokeError("Grounded RAG smoke course is outside the learner-ready generation scope.")
    return inferred.design_spec.model_dump(mode="json")


def _create_and_publish_grounded_rag_course(client: httpx.Client) -> dict[str, Any]:
    shared_design_spec = _grounded_design_spec(
        "Grounded RAG Live Smoke",
        "Build a grounded retrieval and answer system over a visible corpus, with citations and abstention.",
        ["grounded answers", "retrieval quality", "abstention", "traceability"],
    )
    created = _request_json(
        client,
        "POST",
        "/v1/course-runs",
        json={
            "title": "Grounded RAG Live Smoke",
            "summary": "Build a grounded retrieval and answer system over a visible corpus, with citations and abstention.",
            "package_type": "progressive_codebase_course",
            "shared_design_spec": shared_design_spec,
            "modules": [
                {
                    "module_slug": "exercise/01-contract",
                    "title": "Grounded answer contract",
                    "summary": "Return grounded answers with citations through a stable run contract.",
                    "learning_outcomes": ["grounded answers", "citation schema"],
                    "design_spec": _grounded_design_spec(
                        "Grounded answer contract",
                        "Return grounded answers with citations through a stable run contract.",
                        ["grounded answers", "citation schema"],
                    ),
                },
                {
                    "module_slug": "exercise/02-retrieval",
                    "title": "Retrieval quality",
                    "summary": "Retrieve and rank the strongest supporting evidence before answering.",
                    "learning_outcomes": ["retrieval selection", "evidence ranking"],
                    "design_spec": _grounded_design_spec(
                        "Retrieval quality",
                        "Retrieve and rank the strongest supporting evidence before answering.",
                        ["retrieval selection", "evidence ranking"],
                    ),
                },
                {
                    "module_slug": "exercise/03-abstention",
                    "title": "Abstention and traceability",
                    "summary": "Abstain when support is weak and expose the retrieval path.",
                    "learning_outcomes": ["abstention", "traceability"],
                    "design_spec": _grounded_design_spec(
                        "Abstention and traceability",
                        "Abstain when support is weak and expose the retrieval path.",
                        ["abstention", "traceability"],
                    ),
                },
                {
                    "module_slug": "final/integrated",
                    "title": "Production final",
                    "summary": "Meet groundedness, latency, and cost goals together.",
                    "learning_outcomes": ["latency", "operating cost"],
                    "design_spec": _grounded_design_spec(
                        "Production final",
                        "Meet groundedness, latency, and cost goals together.",
                        ["latency", "operating cost"],
                    ),
                },
            ],
        },
    )
    course_run_id = str(created["id"])
    shared_run_id = str(created["shared_workflow_run_id"])

    for gate in [
        "gate_1_spec_review",
        "gate_2_progression_review",
        "gate_3_pre_publish",
    ]:
        _request_json(
            client,
            "POST",
            f"/v1/workflow-runs/{shared_run_id}/decisions",
            json={"gate": gate, "decision": "approve"},
        )

    synced = _request_json(client, "POST", f"/v1/course-runs/{course_run_id}/sync")
    if synced.get("stage") != "ready_to_publish":
        raise SmokeError(f"Expected course to be ready to publish, got: {json.dumps(synced, indent=2)}")

    published = _request_json(client, "POST", f"/v1/course-runs/{course_run_id}/publish")
    target_course = _published_course_from_catalog(client, course_run_id=course_run_id)
    return {
        "course_run_id": course_run_id,
        "shared_workflow_run_id": shared_run_id,
        "publish_snapshot_id": str(published["latest_publish_snapshot_id"]),
        "catalog_course": target_course,
        "setup_mode": "created_and_published",
    }


def _require_learner_content(
    client: httpx.Client,
    enrollment_id: str,
    module_id: str,
    *,
    require_corpus: bool,
) -> dict[str, Any]:
    readme = _request_json(
        client,
        "GET",
        f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
        params={"module_id": module_id, "path": "README.md"},
    )
    module_content = _request_json(
        client,
        "GET",
        f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
        params={"module_id": module_id, "path": "module_content.md"},
    )

    corpus_content: str | None = None
    if require_corpus:
        corpus = _request_json(
            client,
            "GET",
            f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
            params={"module_id": module_id, "path": "data/corpus.json"},
        )
        corpus_content = corpus.get("content", "")
        if "doc:ada_lovelace" not in corpus_content:
            raise SmokeError("Visible corpus fixture does not contain the expected learner-facing documents.")

    readme_content = readme.get("content", "")
    if (
        "grounded response" not in readme_content.lower()
        and "answer questions from the learner-visible corpus" not in readme_content.lower()
    ):
        raise SmokeError("Learner README does not describe the grounded-RAG task clearly enough for the smoke.")
    if "## Files to edit" not in module_content.get("content", ""):
        raise SmokeError("Module content is missing the learner brief structure.")

    return {
        "readme_excerpt": readme_content.splitlines()[:6],
        "module_content_excerpt": module_content.get("content", "").splitlines()[:10],
        "corpus_present": bool(corpus_content),
    }


def _run_module_attempt(
    client: httpx.Client,
    enrollment_id: str,
    module_id: str,
    *,
    require_corpus: bool,
) -> dict[str, Any]:
    workspace = _request_json(
        client,
        "POST",
        f"/v1/lms/enrollments/{enrollment_id}/workspace",
        json={"module_id": module_id},
    )
    current_module = _module_record(workspace, module_id)
    visible_files = set(current_module.get("visible_files", []))
    if "README.md" not in visible_files or "module_content.md" not in visible_files:
        raise SmokeError(f"Learner workspace for '{module_id}' is missing the expected visible brief files.")
    if require_corpus and "data/corpus.json" not in visible_files:
        raise SmokeError("Learner workspace is missing the visible RAG corpus fixture.")

    learner_content = _require_learner_content(
        client,
        enrollment_id,
        module_id,
        require_corpus=require_corpus,
    )

    _request_json(
        client,
        "PUT",
        f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
        json={"module_id": module_id, "relative_path": "app.py", "content": BAD_APP},
    )
    bad_experience = _request_json(
        client,
        "POST",
        f"/v1/lms/enrollments/{enrollment_id}/submit",
        json={"module_id": module_id},
    )
    bad_summary = _submission_summary(bad_experience)
    if bad_summary["status"] != "failed":
        raise SmokeError(
            f"Expected bad submission for '{module_id}' to fail, got: {json.dumps(bad_summary, indent=2)}"
        )

    _request_json(
        client,
        "PUT",
        f"/v1/lms/enrollments/{enrollment_id}/workspace/file",
        json={"module_id": module_id, "relative_path": "app.py", "content": GOOD_APP},
    )
    good_experience = _request_json(
        client,
        "POST",
        f"/v1/lms/enrollments/{enrollment_id}/submit",
        json={"module_id": module_id},
    )
    good_summary = _submission_summary(good_experience)
    if good_summary["status"] != "passed":
        raise SmokeError(
            f"Expected good submission for '{module_id}' to pass, got: {json.dumps(good_summary, indent=2)}"
        )

    refreshed = _request_json(client, "GET", f"/v1/lms/enrollments/{enrollment_id}")
    progression_observed = (
        refreshed.get("status") == "completed"
        if good_summary["module_id"] == module_id and refreshed.get("current_module_id") is None
        else refreshed.get("current_module_id") != module_id
    )
    return {
        "module_id": module_id,
        "title": current_module.get("title"),
        "module_index": int(current_module.get("module_index", 0) or 0),
        "visible_files": sorted(visible_files),
        "learner_content": learner_content,
        "bad_submission": bad_summary,
        "good_submission": good_summary,
        "next_module_id": refreshed.get("current_module_id"),
        "progression_observed": progression_observed,
        "course_completed": refreshed.get("status") == "completed",
    }


def run_smoke(
    base_url: str,
    *,
    course_run_id: str | None = None,
    learner_id: str | None = None,
    store_report: bool = True,
) -> dict[str, Any]:
    learner_id = learner_id or f"rag-smoke-{uuid4().hex[:10]}"
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=240.0) as client:
        if course_run_id is None:
            setup = _create_and_publish_grounded_rag_course(client)
        else:
            target_course = _published_course_from_catalog(client, course_run_id=course_run_id)
            setup = {
                "course_run_id": str(target_course["course_run_id"]),
                "shared_workflow_run_id": (
                    str(target_course["shared_workflow_run_id"])
                    if target_course.get("shared_workflow_run_id") is not None
                    else None
                ),
                "publish_snapshot_id": (
                    str(target_course["publish_snapshot_id"])
                    if target_course.get("publish_snapshot_id") is not None
                    else None
                ),
                "catalog_course": target_course,
                "setup_mode": "existing_published_course",
            }

        target_course_run_id = str(setup["course_run_id"])

        enrollment = _request_json(
            client,
            "POST",
            "/v1/lms/enrollments",
            json={"course_run_id": target_course_run_id, "learner_id": learner_id},
        )
        enrollment_id = str(enrollment["id"])
        current_module_id = enrollment.get("current_module_id")
        if not current_module_id:
            raise SmokeError("Learner enrollment did not activate a starting module.")

        module_results: list[dict[str, Any]] = []
        while current_module_id:
            module_results.append(
                _run_module_attempt(
                    client,
                    enrollment_id,
                    str(current_module_id),
                    require_corpus=True,
                )
            )
            current_module_id = module_results[-1]["next_module_id"]

        refreshed = _request_json(client, "GET", f"/v1/lms/enrollments/{enrollment_id}")
        if refreshed.get("status") != "completed":
            raise SmokeError(f"Expected learner enrollment to complete, got: {json.dumps(refreshed, indent=2)}")

        stored_report: dict[str, Any] | None = None
        if store_report:
            stored_report = _request_json(
                client,
                "POST",
                f"/v1/course-runs/{target_course_run_id}/learner-eval",
                json={
                    "publish_snapshot_id": setup["publish_snapshot_id"],
                    "learner_id": learner_id,
                    "enrollment_id": enrollment_id,
                    "notes": [
                        "Generated by the grounded RAG learner smoke.",
                        "Uses learner-visible workspace files and the real grader path.",
                    ],
                    "module_results": [
                        {
                            "module_id": result["module_id"],
                            "title": result["title"],
                            "module_index": result["module_index"],
                            "learner_visible_files": result["visible_files"],
                            "bad_attempt": result["bad_submission"],
                            "good_attempt": result["good_submission"],
                            "next_module_id": result["next_module_id"],
                            "progression_observed": result["progression_observed"],
                            "course_completed": result["course_completed"],
                            "notes": [
                                "Read only learner-visible brief and workspace files before writing code."
                            ],
                        }
                        for result in module_results
                    ],
                },
            )

        return {
            "course_run_id": target_course_run_id,
            "shared_workflow_run_id": setup["shared_workflow_run_id"],
            "publish_snapshot_id": setup["publish_snapshot_id"],
            "setup_mode": setup["setup_mode"],
            "learner_id": learner_id,
            "enrollment_id": enrollment_id,
            "submitted_modules": module_results,
            "final_enrollment_status": refreshed.get("status"),
            "next_module_id": refreshed.get("current_module_id"),
            "stored_report": stored_report,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a learner-only grounded-RAG LMS smoke across every module.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="Base URL for the local app.")
    parser.add_argument(
        "--course-run-id",
        default=None,
        help="Optional already-published course run to target. When omitted, the script creates and publishes a fresh course first.",
    )
    parser.add_argument("--learner-id", default=None, help="Optional learner id to use for the test enrollment.")
    parser.add_argument(
        "--skip-store-report",
        action="store_true",
        help="Skip posting the structured learner evaluation report back to the app.",
    )
    args = parser.parse_args()

    result = run_smoke(
        args.base_url,
        course_run_id=args.course_run_id,
        learner_id=args.learner_id,
        store_report=not args.skip_store_report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
