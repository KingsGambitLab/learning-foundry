"""Tests for ``OutcomeRepoAuthorAdapter``.

The adapter wraps the existing ``OpenAIStarterRepoAuthoringService``'s
``_generate_bundle`` path with synthesized inputs derived from a
``CourseOutcomeSpec``. The legacy ``WorkflowRun`` + on-disk manifest
+ starter_root machinery is bypassed — the adapter is a single-call,
single-deliverable surface that returns ``list[tuple[str, str]]``
suitable for ``materialize_starter`` to write directly.

These tests use a fake authoring service to verify wiring; no real
LLM calls are made.
"""
from __future__ import annotations

import unittest
from typing import Any

from app.domain.registry import PackageType
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    QualityBar,
    StarterType,
)
from app.services.outcome_repo_author_adapter import (
    OutcomeRepoAuthorAdapter,
    build_outcome_repo_author_payload,
)


# ---------------- Fixtures ----------------


def _spec() -> CourseOutcomeSpec:
    return CourseOutcomeSpec(
        title="Build a Grounded RAG Service",
        goal=(
            "Build an HTTP service that ingests documents, retrieves "
            "passages, and answers grounded questions."
        ),
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str", "citations": "list"},
                description="Answer the question with grounded citations.",
            ),
            EndpointContract(
                method=HttpMethod.GET,
                path="/health",
                request_schema={},
                response_schema={"status": "str"},
                description="Healthcheck.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answers cite supporting passages.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=20,
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
    )


class _FakeBundle:
    """Stand-in for ``_GeneratedRepoBundle`` returned by the LLM service."""

    def __init__(self, files: list[dict[str, str]], notes: list[str] | None = None) -> None:
        self.files = [_FakeFile(p["path"], p["content"]) for p in files]
        self.dependency_contract = _FakeDependencyContract()
        self.notes = notes or []


class _FakeFile:
    def __init__(self, path: str, content: str) -> None:
        self.path = path
        self.content = content


class _FakeDependencyContract:
    manifest_paths: list[str] = []
    lockfile_paths: list[str] = []
    toolchain_paths: list[str] = []
    build_support_paths: list[str] = []
    reproducibility_mode: str | None = None


class _FakeAuthoringService:
    """Records the call signature and returns a canned bundle."""

    def __init__(self, bundle: _FakeBundle | None = None) -> None:
        self.bundle = bundle or _FakeBundle(
            files=[
                {"path": "Dockerfile", "content": "FROM python:3.12-slim\n"},
                {"path": "app/main.py", "content": "# learner code here\n"},
            ],
        )
        self.calls: list[dict[str, Any]] = []

    # The adapter is expected to call this method on the wrapped service.
    def _generate_bundle(
        self,
        client,
        *,
        model_id: str,
        api_key: str,
        base_url: str | None,
        payload: dict[str, Any],
        workflow_run_id: str,
        deliverable_id: str,
    ):
        self.calls.append(
            {
                "client": client,
                "model_id": model_id,
                "api_key": api_key,
                "base_url": base_url,
                "payload": payload,
                "workflow_run_id": workflow_run_id,
                "deliverable_id": deliverable_id,
            }
        )
        return self.bundle, None


class OutcomeRepoAuthorAdapterTests(unittest.TestCase):
    """Pin the adapter's wiring against a fake authoring service."""

    def test_generate_bundle_invokes_service_with_synthesized_payload(self) -> None:
        """The adapter calls the wrapped service's ``_generate_bundle``."""
        service = _FakeAuthoringService()
        adapter = OutcomeRepoAuthorAdapter(service=service, model_id="gpt-5.4")
        spec = _spec()

        files = adapter.generate_bundle(spec=spec)

        self.assertEqual(len(service.calls), 1)
        call = service.calls[0]
        self.assertEqual(call["model_id"], "gpt-5.4")
        # The single synthesized deliverable id is the outcome stand-in.
        self.assertEqual(call["deliverable_id"], "outcome")
        # The workflow_run_id is also a synthetic identifier.
        self.assertTrue(call["workflow_run_id"])
        # And the result is the bundle's files as (path, content) tuples.
        self.assertEqual(
            files,
            [
                ("Dockerfile", "FROM python:3.12-slim\n"),
                ("app/main.py", "# learner code here\n"),
            ],
        )

    def test_synthesized_payload_carries_spec_fields(self) -> None:
        """The payload encodes the spec's goal, title, and endpoints."""
        spec = _spec()
        payload = build_outcome_repo_author_payload(spec=spec)
        self.assertEqual(payload["workflow_title"], spec.title)
        self.assertEqual(payload["problem_statement"], spec.goal)
        self.assertEqual(payload["deliverable_id"], "outcome")
        # The public_endpoints list mirrors the spec's endpoints.
        endpoints = payload["public_endpoints"]
        self.assertEqual(len(endpoints), len(spec.endpoints))
        # Method + path are surfaced in a format the service prompt expects.
        methods = {e["method"] for e in endpoints}
        paths = {e["path"] for e in endpoints}
        self.assertIn("POST", methods)
        self.assertIn("/answer", paths)
        # Starter scaffolding lists are empty on the first pass (no prior workspace state).
        self.assertEqual(payload["current_files"], {})
        self.assertEqual(payload["runtime_protocol_files"], {})
        self.assertEqual(payload["dependency_contract_files"], {})

    def test_failure_context_is_passed_through_to_payload(self) -> None:
        """A non-None ``failure_context`` survives intact into the payload."""
        service = _FakeAuthoringService()
        adapter = OutcomeRepoAuthorAdapter(service=service)
        failure_ctx = {
            "findings": [
                {"category": "boot", "title": "container exited", "detail": "uvicorn crashed"},
            ],
            "boot_result": {"ok": False, "stage": "boot"},
        }
        adapter.generate_bundle(spec=_spec(), failure_context=failure_ctx)
        payload = service.calls[0]["payload"]
        self.assertEqual(payload["failure_context"], failure_ctx)

    def test_no_failure_context_serializes_as_none(self) -> None:
        """When ``failure_context`` is omitted, the payload carries ``None``."""
        service = _FakeAuthoringService()
        adapter = OutcomeRepoAuthorAdapter(service=service)
        adapter.generate_bundle(spec=_spec())
        payload = service.calls[0]["payload"]
        self.assertIsNone(payload["failure_context"])

    def test_returned_files_preserve_service_emit_order(self) -> None:
        """The adapter returns tuples in the same order the service emitted them."""
        bundle = _FakeBundle(
            files=[
                {"path": "requirements.txt", "content": "fastapi==0.111.0\n"},
                {"path": "Dockerfile", "content": "FROM python:3.12-slim\n"},
                {"path": "app/main.py", "content": "# learner code\n"},
            ]
        )
        service = _FakeAuthoringService(bundle=bundle)
        adapter = OutcomeRepoAuthorAdapter(service=service)
        files = adapter.generate_bundle(spec=_spec())
        self.assertEqual(
            [path for path, _ in files],
            ["requirements.txt", "Dockerfile", "app/main.py"],
        )

    def test_service_unavailable_falls_back_to_deterministic_shell(self) -> None:
        """When the wrapped service can't produce a bundle, the adapter degrades.

        The graph still needs SOMETHING to materialize so downstream stages can
        observe a failure rather than crash. The fallback is a tiny FastAPI
        starter shell — deterministic and predictable across retries.
        """
        from app.services.outcome_repo_author_adapter import (
            DeterministicStarterShellFallback,
        )

        class _UnavailableService:
            def _generate_bundle(self, *args, **kwargs):
                raise RuntimeError("OpenAI repo authoring unavailable in this env")

        adapter = OutcomeRepoAuthorAdapter(
            service=_UnavailableService(),
            fallback=DeterministicStarterShellFallback(),
        )
        files = adapter.generate_bundle(spec=_spec())
        # The fallback returns at least a Dockerfile + requirements.txt + an app entry.
        paths = {p for p, _ in files}
        self.assertIn("Dockerfile", paths)
        self.assertIn("app/main.py", paths)


if __name__ == "__main__":
    unittest.main()
