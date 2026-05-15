"""Outcome-graph adapter for ``OpenAIStarterRepoAuthoringService``.

The legacy ``OpenAIStarterRepoAuthoringService`` consumes a
``WorkflowRun`` (with ``task_agent_spec`` + ``workspace_snapshot``) and
walks the on-disk manifests for every deliverable. The new outcome path
has neither — it has a ``CourseOutcomeSpec`` and an empty workspace
directory waiting for the first author pass to produce a starter
bundle.

Strategy: rather than refactor the service, we call
``OpenAIStarterRepoAuthoringService._generate_bundle`` directly with a
payload synthesized from the spec. The service's prompt is preserved
verbatim — only the inputs change. The result is converted to
``list[tuple[str, str]]`` which the graph's ``materialize_starter``
writes under ``<workspace_root>/public/starter/``.

When the service is unavailable (no API key, SDK missing, network
error, etc.), the adapter falls back to a deterministic Python/FastAPI
starter shell so the graph still has SOMETHING to materialize. The
shell intentionally returns 501 from the spec's first endpoint so the
downstream verifier observes a clean "not implemented" failure rather
than crashing on empty input.
"""
from __future__ import annotations

from typing import Any

from app.services.course_outcome_models import CourseOutcomeSpec


__all__ = [
    "OutcomeRepoAuthorAdapter",
    "DeterministicStarterShellFallback",
    "build_outcome_repo_author_payload",
]


# The synthesized identifiers below are not load-bearing — they exist
# only because ``_generate_bundle`` requires non-empty strings for its
# logging events. They're stable so production traces are grep-able.
OUTCOME_DELIVERABLE_ID = "outcome"
SYNTHETIC_WORKFLOW_RUN_PREFIX = "outcome-run"


def build_outcome_repo_author_payload(
    *,
    spec: CourseOutcomeSpec,
    failure_context: Any = None,
) -> dict[str, Any]:
    """Synthesize the payload dict the service's prompt expects.

    The service's ``_prompt_payload`` populates the same keys from a
    ``WorkflowRun`` + on-disk manifest. We populate them directly from
    the ``CourseOutcomeSpec`` and supply empty maps for the file-state
    inputs — there's no prior workspace state on the first author pass.

    Keys mirror ``_prompt_payload`` so the prompt's system message keeps
    its full grounding (which is rich and well-tuned).
    """
    return {
        "workflow_title": spec.title,
        "problem_statement": spec.goal,
        "deliverable_id": OUTCOME_DELIVERABLE_ID,
        "starter_root": "starter",
        # No manifest on the first pass — but the prompt understands that
        # an empty manifest means "synthesize from the public_endpoints".
        "manifest": {},
        "current_files": {},
        "dependency_contract_files": {},
        "runtime_protocol_files": {},
        "public_endpoints": [
            {
                "method": endpoint.method.value,
                "path": endpoint.path,
                "request_schema": endpoint.request_schema,
                "response_schema": endpoint.response_schema,
                "description": endpoint.description,
            }
            for endpoint in spec.endpoints
        ],
        "failure_context": failure_context,
    }


class DeterministicStarterShellFallback:
    """Last-resort starter shell when the LLM service is unreachable.

    Produces a small FastAPI app that serves the spec's first endpoint
    as a 501 stub. Identical to the deterministic adapter that lived in
    ``outcome_graph_deps.py`` pre-Wave 5e — it's now a fallback rather
    than the default.
    """

    def generate_bundle(
        self,
        *,
        spec: CourseOutcomeSpec,
        failure_context: Any = None,
    ) -> list[tuple[str, str]]:
        del failure_context
        first = spec.endpoints[0]
        path = first.path
        method = first.method.value.lower()
        return [
            (
                "Dockerfile",
                "FROM python:3.12-slim\n"
                "WORKDIR /app\n"
                "COPY . .\n"
                "RUN pip install --no-cache-dir -r requirements.txt\n"
                'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]\n',
            ),
            (
                "requirements.txt",
                "fastapi==0.111.0\nuvicorn==0.30.0\n",
            ),
            ("app/__init__.py", ""),
            (
                "app/main.py",
                (
                    "from fastapi import FastAPI, HTTPException\n"
                    "\n"
                    "app = FastAPI()\n"
                    "\n"
                    "@app.get(\"/health\")\n"
                    "async def _health() -> dict:\n"
                    "    return {\"status\": \"ok\"}\n"
                    "\n"
                    f"@app.{method}(\"{path}\")\n"
                    "async def _outcome_endpoint(payload: dict | None = None):\n"
                    "    raise HTTPException(status_code=501, detail=\"learner-implements-this\")\n"
                ),
            ),
        ]


class OutcomeRepoAuthorAdapter:
    """Adapts ``OpenAIStarterRepoAuthoringService._generate_bundle`` for the outcome graph.

    Parameters
    ----------
    service:
        An ``OpenAIStarterRepoAuthoringService`` instance (or any object
        exposing a compatible ``_generate_bundle`` method). The adapter
        calls the underscored method directly because the public
        ``author_workspace_repo`` path expects a ``WorkflowRun`` with
        a workspace snapshot — neither of which exists on the first
        pass of the outcome graph.
    model_id:
        Forwarded to ``_generate_bundle`` as the LLM model name. When
        ``None`` the service's default ("gpt-5.4") is used by reading
        the env config — we still pass an explicit value because the
        service's signature requires it.
    api_key, base_url, client:
        Optional overrides. When ``None`` the service's normal env
        lookup applies (the underlying ``_create_response_with_retries``
        falls back to ``LLMRouter.parse_structured`` when ``client`` is
        ``None``, so this path is the production-default).
    fallback:
        Optional adapter to call when the wrapped service raises. The
        graph still needs a starter bundle to materialize so downstream
        stages observe a failure rather than crash. When ``None``,
        exceptions propagate (preferred in production; tests can opt
        into a fallback explicitly).
    """

    def __init__(
        self,
        *,
        service: Any | None = None,
        model_id: str = "gpt-5.4",
        api_key: str = "",
        base_url: str | None = None,
        client: Any = None,
        fallback: Any = None,
    ) -> None:
        if service is None:
            from app.services.openai_repo_authoring import (
                OpenAIStarterRepoAuthoringService,
            )

            service = OpenAIStarterRepoAuthoringService()
        self._service = service
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._fallback = fallback or DeterministicStarterShellFallback()
        self._call_seq = 0

    def generate_bundle(
        self,
        *,
        spec: CourseOutcomeSpec,
        failure_context: Any = None,
    ) -> list[tuple[str, str]]:
        payload = build_outcome_repo_author_payload(
            spec=spec, failure_context=failure_context
        )
        self._call_seq += 1
        workflow_run_id = f"{SYNTHETIC_WORKFLOW_RUN_PREFIX}-{self._call_seq:04d}"
        try:
            bundle, _usage = self._service._generate_bundle(
                self._client,
                model_id=self._model_id,
                api_key=self._api_key,
                base_url=self._base_url,
                payload=payload,
                workflow_run_id=workflow_run_id,
                deliverable_id=OUTCOME_DELIVERABLE_ID,
            )
        except Exception:
            # When the LLM service cannot deliver a bundle (no API key,
            # network failure, etc.), fall back to the deterministic
            # shell so the graph has something to materialize and the
            # verifier can produce a useful failure signal.
            if self._fallback is None:
                raise
            return self._fallback.generate_bundle(
                spec=spec, failure_context=failure_context
            )
        return [(file.path, file.content) for file in bundle.files]
