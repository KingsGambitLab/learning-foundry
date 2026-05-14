"""Wave 5b: end-to-end smoke of the single-outcome course pipeline.

This is the FIRST end-to-end integration of all the pieces built by
the previous waves (Wave 5d persistence, Wave 5e gate route, Wave 5e.5
Docker boot, Wave 6.6 visible samples, Wave 6.7a templater, Wave 6.7b
RAG scaffold). It exercises real production code paths and mocks only
at the network / Docker boundaries:

* ``LLMRouter.parse_structured`` is faked via ``_MockRouter`` that
  dispatches on the requested ``text_format`` and returns deterministic
  structured outputs.
* ``datasets.load_dataset`` is patched to return a tiny CRAG-shaped
  fixture so the benchmark loader never touches Hugging Face.
* ``boot_and_verify`` is patched so starter verification returns
  success without any Docker subprocess.
* The oracle pass uses a fake sandbox runner + a fake HTTP client so
  scenarios "execute against the reference impl" without any real
  network call.

What's REAL:
* The full state machine in ``langgraph_outcome_graph``.
* Persistence via the SQLite store + outcome-state round-trip.
* Materialization of every file (starter, oracle bundle, runner.py,
  visible-check script, README) on disk under ``tmp_path``.
* The oracle authoring flow (system prompt, payload validation,
  benchmark-loader integration on the CRAG branch).
* ``oracle_validation.validate_oracle`` consuming real on-disk
  scenarios.

The test asserts that the run reaches ``published`` and that every
file the design doc names lands at the correct path.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock as _mock
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.domain.ai import AIUsageSummary
from app.domain.course import (
    CourseGenerationSource,
    CourseGenerationStatus,
    CourseRunStatus,
    CreatorCourseSetupInput,
    GenerateCourseFromBriefRequest,
)
from app.domain.registry import PackageType
from app.domain.workflow import DecisionOutcome, HILGate
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_workspace_manager import AssignmentWorkspaceManager
from app.services.course_generation_service import CourseGenerationService
from app.services.course_outcome_models import (
    CRAGBenchmarkSource,
    CapabilityFlags,
    CourseOutcomeSpec,
    EndpointContract,
    HttpMethod,
    JudgeKind,
    OracleSource,
    QualityBar,
    StarterType,
)
from app.services.course_workflow_service import CourseWorkflowService
from app.services.langgraph_outcome_graph import (
    OutcomeGraphDeps,
    OutcomeWorkflowState,
)
from app.services.oracle_authoring import OracleAuthor
from app.services.oracle_pass import OraclePass, OraclePassResult, OracleScenarioOutput
from app.services.outcome_graph_deps import RealStarterVerifier
from app.services.task_agent_blackbox_runner import TaskAgentBlackBoxRunner
from app.services.workflow_service import WorkflowService
from app.services.workspace_boot import WorkspaceBootHandle
from app.storage.sqlite_store import SQLiteWorkflowStore


# =====================================================================
# Fixtures: CRAG dataset rows
# =====================================================================


def _crag_fixture_rows() -> list[dict]:
    """8-row CRAG-shaped fixture covering finance + non-finance + types.

    Rows 0-4 are finance/valid (matched by the brief's domain_filter +
    default answer_type_filter); rows 5-7 are non-finance noise that the
    filters must drop. Two false-premise rows are included so the
    visibility split exercises stratification.
    """
    return [
        {
            "interaction_id": "fin_simple_valid_001",
            "query": "What was AAPL's revenue in Q1 2023?",
            "answer": "AAPL Q1 2023 revenue was $117.2B.",
            "alt_ans": ["$117.2 billion", "117.2B"],
            "search_results": [
                {
                    "page_url": "https://example.com/aapl-q1-2023",
                    "page_snippet": "Apple Q1 2023 revenue 117.2B...",
                    "page_result": (
                        "<html><body><p>Apple Q1 2023 revenue was 117.2B</p>"
                        "</body></html>"
                    ),
                }
            ],
            "domain": "finance",
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "fin_simple_valid_002",
            "query": "What was Tesla's Q2 2023 vehicle delivery count?",
            "answer": "Tesla delivered 466,140 vehicles in Q2 2023.",
            "alt_ans": ["466,140 vehicles", "~466k"],
            "search_results": [
                {
                    "page_url": "https://example.com/tsla-q2-2023",
                    "page_snippet": "Tesla delivered 466,140 in Q2...",
                    "page_result": (
                        "<html><body><p>Tesla delivered 466,140 vehicles in"
                        " Q2 2023.</p></body></html>"
                    ),
                }
            ],
            "domain": "finance",
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "fin_false_premise_001",
            "query": "Why did MSFT acquire OpenAI in 2024?",
            "answer": "Microsoft did not acquire OpenAI in 2024.",
            "alt_ans": ["MSFT did not acquire OpenAI", "false premise"],
            "search_results": [
                {
                    "page_url": "https://example.com/msft",
                    "page_snippet": "Microsoft invested in OpenAI...",
                    "page_result": "<html><body><p>MSFT invested in OAI</p></body></html>",
                }
            ],
            "domain": "finance",
            "question_type": "false_premise",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "fin_multi_hop_001",
            "query": "Did the company with the highest 2022 revenue also pay a dividend?",
            "answer": "Yes — Apple (the highest-revenue tech company in 2022) paid a dividend.",
            "alt_ans": ["Apple paid a dividend"],
            "search_results": [
                {
                    "page_url": "https://example.com/aapl-divs",
                    "page_snippet": "Apple dividends history...",
                    "page_result": "<html><body><p>Apple paid quarterly dividends.</p></body></html>",
                }
            ],
            "domain": "finance",
            "question_type": "multi-hop",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "fin_simple_valid_003",
            "query": "What is Berkshire Hathaway's largest equity holding?",
            "answer": "Apple has been Berkshire Hathaway's largest equity holding.",
            "alt_ans": ["AAPL"],
            "search_results": [
                {
                    "page_url": "https://example.com/brk-holdings",
                    "page_snippet": "BRK's largest position is AAPL",
                    "page_result": "<html><body><p>BRK largest holding: AAPL</p></body></html>",
                }
            ],
            "domain": "finance",
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        # ---- Noise rows: must be filtered out ----
        {
            "interaction_id": "movie_filtered_out",
            "query": "Who won best picture at the Oscars in 2020?",
            "answer": "Parasite.",
            "alt_ans": [],
            "search_results": [],
            "domain": "movie",  # not 'finance' → filtered
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 0,
        },
        {
            "interaction_id": "fin_no_answer_filtered",
            "query": "What will AAPL's stock price be tomorrow?",
            "answer": "I don't know.",
            "alt_ans": [],
            "search_results": [],
            "domain": "finance",
            "question_type": "simple",
            "static_or_dynamic": "dynamic",
            "answer_type": "no_answer",  # default filter drops this
            "split": 0,
        },
        {
            "interaction_id": "fin_test_split_filtered",
            "query": "What was GOOG's 2022 revenue?",
            "answer": "GOOG had $282.8B in 2022.",
            "alt_ans": [],
            "search_results": [],
            "domain": "finance",
            "question_type": "simple",
            "static_or_dynamic": "static",
            "answer_type": "valid",
            "split": 1,  # test split — filtered when use_split='validation'
        },
    ]


class _FakeHFIterable:
    """Tiny stand-in for a ``datasets.Dataset`` row iterator."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _patched_load_dataset(name: str, split: str = "train"):
    """Replace ``datasets.load_dataset`` so the loader sees the CRAG fixture."""
    if "Quivr/CRAG" in name or name.endswith("CRAG"):
        return _FakeHFIterable(_crag_fixture_rows())
    raise ValueError(f"unexpected dataset name: {name}")


# =====================================================================
# Mock LLM router
# =====================================================================


@dataclass
class _FakeUsageSummary:
    """Mimic ``app.domain.ai.AIUsageSummary`` for the cost-accounting path."""

    estimated_cost_usd: float = 0.01


@dataclass
class _FakeParsedResult:
    """Mimic ``app.services.llm_router.ParsedResult``."""

    parsed: Any
    usage: Any = None
    usage_summary: Any = None

    @property
    def output_parsed(self) -> Any:
        return self.parsed


class _MockRouter:
    """Routes ``parse_structured`` calls to deterministic fixture builders.

    The dispatcher is keyed by the ``text_format`` model class name, so
    every pydantic schema the pipeline asks for can be answered with a
    canned response. Each call records its (tier, text_format) tuple so
    the test can assert which judges fired.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    def parse_structured(
        self,
        *,
        tier: Any,
        system: str,
        user: str,
        text_format: type,
        request_timeout_s: float = 240.0,
        max_tokens: int = 16_000,
        extra_request_kwargs: dict[str, Any] | None = None,
    ) -> _FakeParsedResult:
        name = text_format.__name__
        self.calls.append((tier, name))
        builder = _ROUTER_DISPATCH.get(name)
        if builder is None:
            raise AssertionError(
                f"_MockRouter: no fixture for text_format={name!r}; "
                f"add one or check the call site."
            )
        parsed = builder(text_format=text_format, user=user)
        return _FakeParsedResult(
            parsed=parsed,
            usage=None,
            usage_summary=AIUsageSummary(estimated_cost_usd=0.01),
        )

    def model_id_for(self, tier: Any) -> str:
        return "mock-model"


def _spec_coherence_verdict(*, text_format: type, user: str) -> Any:
    return text_format(
        is_coherent=True,
        rationale="Spec endpoints and quality bars line up with the goal.",
        concerns=[],
    )


def _oracle_authoring_payload(*, text_format: type, user: str) -> Any:
    """Return a fully formed ``_OracleAuthoringPayload`` for the CRAG spec.

    Includes scenarios covering all required categories (happy_path,
    boundary, malformed_input), references the spec's quality bars, and
    pairs with a minimal reference impl (Dockerfile + requirements.txt
    + app/main.py). Setup files are intentionally empty — the
    benchmark loader is authoritative for the CRAG branch and any
    LLM-emitted setup files are dropped by ``_build_result``.
    """
    happy = (
        "id: happy_extractive_answer\n"
        "description: 'Happy path: extractive answer from the corpus.'\n"
        "category: happy_path\n"
        "quality_bar_ids: [faithfulness, retrieval_recall]\n"
        "trace:\n"
        "  - id: ask\n"
        "    method: POST\n"
        "    path: /answer\n"
        "    body: {question: 'What is the AAPL revenue?'}\n"
        "    expect: {status: 200}\n"
        "rubrics:\n"
        "  - kind: schema_match\n"
        "    target: ask.body\n"
        "    must_have_fields: [answer]\n"
        "  - kind: llm_judge_coverage\n"
        "    target: ask.body.answer\n"
        "    must_contain_facts: ['the answer references AAPL revenue']\n"
    )
    boundary = (
        "id: boundary_long_question\n"
        "description: 'Boundary: very long question still produces a shaped answer.'\n"
        "category: boundary\n"
        "quality_bar_ids: [faithfulness]\n"
        "trace:\n"
        "  - id: ask\n"
        "    method: POST\n"
        "    path: /answer\n"
        "    body: {question: 'x' }\n"
        "    expect: {status: 200}\n"
        "rubrics:\n"
        "  - kind: schema_match\n"
        "    target: ask.body\n"
        "    must_have_fields: [answer]\n"
        "  - kind: llm_judge_coverage\n"
        "    target: ask.body.answer\n"
        "    must_contain_facts: ['contains some answer text']\n"
    )
    malformed = (
        "id: malformed_missing_question\n"
        "description: 'Malformed input: missing question key still returns 400/422.'\n"
        "category: malformed_input\n"
        "quality_bar_ids: [faithfulness]\n"
        "trace:\n"
        "  - id: ask\n"
        "    method: POST\n"
        "    path: /answer\n"
        "    body: {}\n"
        "    expect: {status: [400, 422]}\n"
        "rubrics:\n"
        "  - kind: regex_match\n"
        "    target: ask.body\n"
        "    pattern: '.*'\n"
        "  - kind: llm_judge_coverage\n"
        "    target: ask.body\n"
        "    must_contain_facts: ['error mentions missing question']\n"
    )
    idempotency = (
        "id: idempotency_same_question_twice\n"
        "description: 'Idempotency: same question fired twice yields shaped answer.'\n"
        "category: idempotency\n"
        "quality_bar_ids: [retrieval_recall]\n"
        "trace:\n"
        "  - id: ask\n"
        "    method: POST\n"
        "    path: /answer\n"
        "    body: {question: 'What is the AAPL revenue?'}\n"
        "    expect: {status: 200}\n"
        "  - id: ask_again\n"
        "    method: POST\n"
        "    path: /answer\n"
        "    body: {question: 'What is the AAPL revenue?'}\n"
        "    expect: {status: 200}\n"
        "rubrics:\n"
        "  - kind: schema_match\n"
        "    target: ask_again.body\n"
        "    must_have_fields: [answer]\n"
        "  - kind: oracle_set_overlap\n"
        "    target: ask.body.cited_chunks\n"
        "    gold_set_path: gold_answers.fin_simple_valid_001.alt_ans\n"
        "    minimum_overlap: 0\n"
    )
    scenarios = [
        {"filename": "happy.yaml", "yaml_content": happy,
         "quality_bar_ids": ["faithfulness", "retrieval_recall"]},
        {"filename": "boundary.yaml", "yaml_content": boundary,
         "quality_bar_ids": ["faithfulness"]},
        {"filename": "malformed.yaml", "yaml_content": malformed,
         "quality_bar_ids": ["faithfulness"]},
        {"filename": "idempotency.yaml", "yaml_content": idempotency,
         "quality_bar_ids": ["retrieval_recall"]},
    ]
    reference_files = [
        {
            "relative_path": "Dockerfile",
            "content": (
                "FROM python:3.11-slim\nWORKDIR /app\n"
                "COPY requirements.txt ./\nRUN pip install --no-cache-dir -r requirements.txt\n"
                "COPY app ./app\nEXPOSE 8000\n"
                "CMD [\"python\", \"-m\", \"app.main\"]\n"
            ),
        },
        {
            "relative_path": "requirements.txt",
            "content": "fastapi==0.112.0\nuvicorn==0.30.0\n",
        },
        {
            "relative_path": "app/main.py",
            "content": (
                "from fastapi import FastAPI\n"
                "app = FastAPI()\n"
                "@app.get('/health')\n"
                "def health():\n"
                "    return {'status': 'ok'}\n"
                "@app.post('/answer')\n"
                "def answer(payload: dict):\n"
                "    return {'answer': 'extractive answer'}\n"
            ),
        },
    ]
    return text_format(
        scenarios=scenarios,
        reference_files=reference_files,
        setup_files=[],
        notes=["Smoke-test oracle bundle."],
    )


def _llm_judge_coverage_verdict(*, text_format: type, user: str) -> Any:
    return text_format(
        covered_facts=["contains some answer text"],
        missing_facts=[],
        verdict="pass",
        rationale="Answer covers the required facts.",
    )


def _semantic_eq_verdict(*, text_format: type, user: str) -> Any:
    return text_format(
        is_semantically_equivalent=True,
        factual_drift=False,
        verdict="pass",
        rationale="Answer is semantically equivalent.",
    )


def _false_premise_verdict(*, text_format: type, user: str) -> Any:
    return text_format(
        identifies_false_premise=True,
        refuses_to_answer=True,
        verdict="pass",
        rationale="Answer refuses on the false premise.",
    )


# Dispatcher: text_format class name → builder. Populated lazily so we
# don't have to import every verdict class at module load.
_ROUTER_DISPATCH: dict[str, Any] = {
    "SpecCoherenceVerdict": _spec_coherence_verdict,
    "_OracleAuthoringPayload": _oracle_authoring_payload,
    "LLMJudgeCoverageVerdict": _llm_judge_coverage_verdict,
    "SemanticEqVerdict": _semantic_eq_verdict,
    "LLMJudgeFalsePremiseVerdict": _false_premise_verdict,
}


# =====================================================================
# CRAG-backed spec the fake planner returns
# =====================================================================


def _smoke_spec() -> CourseOutcomeSpec:
    """Build the CRAG-backed spec the smoke planner returns."""
    return CourseOutcomeSpec(
        title="Build a Production-Quality RAG Service",
        goal=(
            "Build an HTTP service that answers user questions over the "
            "configured retrieval corpus with measurable faithfulness and "
            "recall. The reference uses extractive synthesis only — HTML "
            "parsing + snippet ranking — so the learner's service does "
            "not need to call an LLM at runtime."
        ),
        starter_type=StarterType.partial,
        endpoints=[
            EndpointContract(
                method=HttpMethod.POST,
                path="/answer",
                request_schema={"question": "str"},
                response_schema={"answer": "str", "cited_chunks": "list[str]"},
                description="Answer the question by extracting from the CRAG per-query retrieval pool.",
            ),
        ],
        quality_bars=[
            QualityBar(
                id="faithfulness",
                metric_description="Answer is grounded in the cited chunks.",
                threshold=">= 0.8",
                judged_by=JudgeKind.llm_haiku,
                sample_size=5,
            ),
            QualityBar(
                id="retrieval_recall",
                metric_description="The cited chunks overlap the gold per-query retrieval pool.",
                threshold=">= 0.7",
                judged_by=JudgeKind.oracle_set_overlap,
                sample_size=5,
            ),
        ],
        package_type=PackageType.progressive_codebase_course,
        oracle_source=OracleSource.reference_run,
        benchmark=CRAGBenchmarkSource(
            dataset="Quivr/CRAG",
            use_split="validation",
            domain_filter=["finance"],
            max_queries=5,
        ),
        capabilities=CapabilityFlags(runtime_llm_required=False),
    )


# =====================================================================
# Fake collaborators (planner, oracle pass, http client)
# =====================================================================


@dataclass
class _FakePlanner:
    spec: CourseOutcomeSpec
    call_count: int = 0

    def plan_course(self, request: Any) -> CourseOutcomeSpec:
        self.call_count += 1
        return self.spec


class _FakeLegacyPlanner:
    """Stands in for the legacy OpenAI planner. ``status()`` only."""

    def status(self) -> CourseGenerationStatus:
        return CourseGenerationStatus(
            provider="anthropic",
            available=False,
            source=CourseGenerationSource.deterministic_fallback,
            message="Live planning disabled for smoke.",
            sdk_installed=False,
            api_key_present=False,
            model_id=None,
            env_file=None,
        )


class _FakeOraclePass:
    """Bypasses booting a real reference impl.

    Records the call (so we can assert capabilities propagation) and
    returns a fully-passing :class:`OraclePassResult` whose
    ``scenario_outputs`` list is keyed off the names the scenarios on
    disk advertise. The scenarios MUST agree with the names the oracle
    author emits so ``validate_oracle`` doesn't claim missing coverage.
    """

    def __init__(self, *, scenario_ids: list[str]) -> None:
        self.scenario_ids = scenario_ids
        self.run_calls: list[dict[str, Any]] = []

    def run(
        self,
        *,
        scenarios: list[Any],
        reference_impl_dir: Path,
        setup_data_dir: Path | None = None,
        course_meta: dict[str, Any] | None = None,
        router: Any = None,
        capabilities: Any = None,
    ) -> OraclePassResult:
        self.run_calls.append(
            {
                "scenario_count": len(scenarios),
                "reference_impl_dir": Path(reference_impl_dir),
                "setup_data_dir": Path(setup_data_dir) if setup_data_dir else None,
                "router": router,
                "capabilities": capabilities,
            }
        )
        # One passing output per scenario. Verdict "pass" everywhere.
        outputs: list[OracleScenarioOutput] = []
        for s in scenarios:
            outputs.append(
                OracleScenarioOutput(
                    scenario_id=s.id,
                    category=s.category,
                    captures={"ask": {"body": {"answer": "ok"}}},
                    verdicts=[
                        (
                            r.kind,
                            {
                                "status": "pass",
                                "rationale": "fake-pass",
                                "diagnostic": {},
                            },
                        )
                        for r in s.rubrics
                    ],
                    aborted=False,
                    abort_reason=None,
                )
            )
        return OraclePassResult(
            reference_impl_hash="fake-hash",
            scenario_set_hash="fake-set-hash",
            generated_at=datetime.now(UTC).isoformat(),
            scenario_outputs=outputs,
            total_scenarios=len(scenarios),
            passed_scenarios=len(scenarios),
            failed_scenarios=0,
            abstained_scenarios=0,
        )


# =====================================================================
# Service construction helper
# =====================================================================


def _build_service(tmp_dir: Path, *, planner: _FakePlanner, router: _MockRouter,
                   oracle_pass: _FakeOraclePass) -> tuple[CourseGenerationService, _MockRouter]:
    """Construct the production service with mocked boundaries.

    Real wiring: planner=fake, router=mock, oracle_author=real (uses
    router), repo_author=real (real CRAG path), starter_verifier=real
    (boot_and_verify is patched at module level), oracle_pass=fake.
    """
    store = SQLiteWorkflowStore(db_path=f"{tmp_dir}/smoke.db")
    workspace_manager = AssignmentWorkspaceManager(base_dir=f"{tmp_dir}/workspaces")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{tmp_dir}/generated"),
        runner=TaskAgentBlackBoxRunner(),
        workspace_manager=workspace_manager,
    )
    course_workflow_service = CourseWorkflowService(
        store,
        workflow_service,
        job_runner=lambda job: job(),
    )

    # Real oracle author wrapping the mock router.
    oracle_author = OracleAuthor(router=router)

    # The repo_author for partial starters is the real OutcomeRepoAuthorAdapter
    # which calls the LLM-driven OpenAIStarterRepoAuthoringService. Since we
    # don't want OpenAI traffic, we substitute a deterministic fallback that
    # just emits a tiny working starter bundle.
    repo_author = _DeterministicRepoAuthor()

    overrides: dict[str, Any] = {
        "router": router,
        "repo_author": repo_author,
        "oracle_author": oracle_author,
        "oracle_pass": oracle_pass,
        # starter_verifier uses RealStarterVerifier — boot_and_verify is patched.
    }

    service = CourseGenerationService(
        course_workflow_service,
        live_planner=_FakeLegacyPlanner(),  # type: ignore[arg-type]
        outcome_planner=planner,
        job_runner=lambda job: job(),
        outcome_workspace_root=Path(tmp_dir) / "outcome_workspaces",
        outcome_deps_overrides=overrides,
    )
    return service, router


# =====================================================================
# A tiny deterministic starter that emits a Dockerfile + main.py
# =====================================================================


class _DeterministicRepoAuthor:
    """Emits a minimal starter bundle without calling any LLM.

    The starter is intentionally aware that the test's
    ``boot_and_verify`` is patched and won't actually exercise the
    Dockerfile contents.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_bundle(self, *, spec: Any, failure_context: dict | None = None) -> list[tuple[str, str]]:
        self.calls.append(
            {"spec_title": spec.title, "failure_context": failure_context}
        )
        return [
            (
                "Dockerfile",
                "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt ./\n"
                "RUN pip install --no-cache-dir -r requirements.txt\n"
                "COPY app ./app\nEXPOSE 8000\n"
                "CMD [\"python\", \"-m\", \"app.main\"]\n",
            ),
            ("requirements.txt", "fastapi==0.112.0\nuvicorn==0.30.0\n"),
            (
                "app/main.py",
                "# starter\n"
                "from fastapi import FastAPI\napp = FastAPI()\n"
                "@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n",
            ),
            (
                ".coursegen/runtime/run.sh",
                "#!/bin/sh\nexec uvicorn app.main:app --host 0.0.0.0 --port 8000\n",
            ),
        ]


# =====================================================================
# Fake boot_and_verify
# =====================================================================


@contextmanager
def _fake_boot_and_verify(workspace_dir: Path, *, capabilities: Any = None, **kwargs):
    """Yield a fake handle without spinning Docker.

    Records every call so the test can assert capabilities propagation.
    """
    _BOOT_CALLS.append({"workspace_dir": Path(workspace_dir), "capabilities": capabilities})
    yield WorkspaceBootHandle(
        base_url="http://127.0.0.1:65535",
        container_id="fake-cid",
        image_tag="fake-tag",
    )


# Module-level so the test can read it across patches.
_BOOT_CALLS: list[dict[str, Any]] = []


# =====================================================================
# THE SMOKE TEST
# =====================================================================


class Wave5bEndToEndSmokeTest(unittest.TestCase):
    """Full single-outcome pipeline end-to-end with all boundaries mocked."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        # Reset shared module-level state.
        _BOOT_CALLS.clear()

    def test_full_flow_reaches_published(self) -> None:
        """Walk the full pipeline; assert published + every artifact lands."""
        spec = _smoke_spec()
        planner = _FakePlanner(spec=spec)
        router = _MockRouter()
        # Scenario IDs we know the fake authoring payload emits. The
        # fake oracle pass uses these to construct passing outputs.
        oracle_pass = _FakeOraclePass(
            scenario_ids=[
                "happy_extractive_answer",
                "boundary_long_question",
                "malformed_missing_question",
                "idempotency_same_question_twice",
            ]
        )

        service, _ = _build_service(
            self.tmp_path, planner=planner, router=router, oracle_pass=oracle_pass
        )

        request = GenerateCourseFromBriefRequest(
            goal=(
                "Build a production-quality RAG over a real document "
                "corpus with measurable quality."
            ),
            creator_setup=CreatorCourseSetupInput(),
        )

        # ------------------------------------------------------------
        # Step 1: Initial generation. Pauses at gate 1.
        # ------------------------------------------------------------
        with _mock.patch(
            "app.services.benchmark_loader.load_dataset",
            _patched_load_dataset,
            create=True,
        ), _mock.patch(
            "app.services.workspace_boot.boot_and_verify",
            _fake_boot_and_verify,
        ):
            response = service.generate_course_run(request)
            course_run_id = response.course_run.id
            self.assertEqual(response.course_run.status, CourseRunStatus.awaiting_human)

            # Step 2: persisted state survives a reload.
            loaded = service._load_outcome_state(course_run_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertIsInstance(loaded, OutcomeWorkflowState)
            self.assertEqual(loaded.stage, "awaiting_gate_1")
            self.assertIsNotNone(loaded.spec)
            assert loaded.spec is not None
            self.assertFalse(loaded.spec.capabilities.runtime_llm_required)
            self.assertIsInstance(loaded.spec.benchmark, CRAGBenchmarkSource)

            # Step 3: refresh check — outcome run must NOT be misclassified
            # as blocked even though it has zero deliverables.
            cws: CourseWorkflowService = service.course_workflow_service
            refreshed = cws.get_run(course_run_id)
            self.assertIsNotNone(refreshed)
            assert refreshed is not None
            self.assertEqual(refreshed.status, CourseRunStatus.awaiting_human)

            # Step 4: approve gate 1 → advance through starter authoring.
            response2 = service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_1_spec_review,
                decision=DecisionOutcome.approve,
            )
            after_gate1 = service._load_outcome_state(course_run_id)
            assert after_gate1 is not None
            # Should be paused at gate 2 (starter verified successfully).
            self.assertEqual(after_gate1.stage, "awaiting_gate_2")
            self.assertEqual(after_gate1.status, "awaiting_human")

            # Verify starter materialized on disk.
            workspace_root = service._outcome_workspace_for(course_run_id)
            starter_dir = workspace_root / "public" / "starter"
            self.assertTrue((starter_dir / "Dockerfile").is_file())
            self.assertTrue((starter_dir / "app" / "main.py").is_file())
            self.assertTrue((starter_dir / "requirements.txt").is_file())
            self.assertTrue(
                (starter_dir / ".coursegen" / "runtime" / "run.sh").is_file()
            )
            # Starter verifier was called with the spec's capabilities
            # (Wave 5e.5 + P0 #3 propagation).
            self.assertGreaterEqual(len(_BOOT_CALLS), 1)
            first_boot = _BOOT_CALLS[0]
            self.assertEqual(first_boot["workspace_dir"], starter_dir)
            self.assertIsNotNone(first_boot["capabilities"])
            self.assertEqual(
                first_boot["capabilities"].runtime_llm_required, False
            )

            # Step 5: approve gate 2 → oracle authoring + oracle pass + validation.
            response3 = service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_2_progression_review,
                decision=DecisionOutcome.approve,
            )
            after_gate2 = service._load_outcome_state(course_run_id)
            assert after_gate2 is not None
            # Diagnostic info on any unexpected stop.
            if after_gate2.stage != "awaiting_gate_3":
                report = after_gate2.oracle_validation_report
                self.fail(
                    f"stage={after_gate2.stage!r} status={after_gate2.status!r} "
                    f"blocking_reasons={after_gate2.blocking_reasons!r} "
                    f"validation_report.publishable={getattr(report, 'publishable', None)} "
                    f"validation_report.blocking_reasons={getattr(report, 'blocking_reasons', None)}"
                )
            self.assertEqual(after_gate2.stage, "awaiting_gate_3")
            self.assertEqual(after_gate2.status, "awaiting_human")

            # Verify oracle authoring + benchmark loader fired.
            # The setup_files end up under private/grader/_setup/.
            setup_dir = workspace_root / "private" / "grader" / "_setup"
            self.assertTrue((setup_dir / "queries.jsonl").is_file())
            self.assertTrue((setup_dir / "gold_answers.json").is_file())
            self.assertTrue((setup_dir / "search_results_index.json").is_file())

            # Visible samples (Wave 6.6) landed under public/examples + public/checks.
            self.assertTrue(
                (workspace_root / "public" / "examples" / "sample_queries.json").is_file()
            )
            self.assertTrue(
                (workspace_root / "public" / "checks" / "run_visible_checks.py").is_file()
            )

            # Scenarios + reference impl files.
            self.assertTrue(
                (workspace_root / "private" / "grader" / "scenarios").is_dir()
            )
            scenario_yamls = list(
                (workspace_root / "private" / "grader" / "scenarios").glob("*.yaml")
            )
            self.assertGreater(len(scenario_yamls), 0)

            self.assertTrue(
                (workspace_root / "private" / "grader" / "_reference" / "Dockerfile").is_file()
            )

            # Oracle pass produced outputs.json + ran with capabilities.
            self.assertTrue(
                (workspace_root / "private" / "grader" / "_oracle" / "outputs.json").is_file()
            )
            self.assertEqual(len(oracle_pass.run_calls), 1)
            self.assertIsNotNone(oracle_pass.run_calls[0]["capabilities"])
            self.assertEqual(
                oracle_pass.run_calls[0]["capabilities"].runtime_llm_required, False
            )

            # Oracle validation passed (publishable).
            self.assertIsNotNone(after_gate2.oracle_validation_report)
            assert after_gate2.oracle_validation_report is not None
            self.assertTrue(
                after_gate2.oracle_validation_report.publishable,
                msg=(
                    "Oracle validation should have produced publishable=True. "
                    f"blocking_reasons={after_gate2.oracle_validation_report.blocking_reasons}"
                ),
            )

            # Step 6: approve gate 3 → publish.
            response4 = service.resume_outcome_workflow_after_gate(
                course_run_id,
                gate=HILGate.gate_3_pre_publish,
                decision=DecisionOutcome.approve,
            )
            final = service._load_outcome_state(course_run_id)
            assert final is not None
            self.assertEqual(final.status, "published")
            self.assertEqual(final.stage, "published")
            self.assertEqual(response4.course_run.status, CourseRunStatus.published)

            # Step 7: published bundle file checks.
            self.assertTrue(
                (workspace_root / "private" / "course_spec.json").is_file()
            )
            runner_path = workspace_root / "private" / "grader" / "runner.py"
            self.assertTrue(runner_path.is_file())

            # Runner.py must be syntactically valid Python.
            runner_src = runner_path.read_text()
            compile(runner_src, str(runner_path), "exec")
            # It imports the scenario loader + trace runner.
            self.assertIn("from app.services.scenario_loader", runner_src)
            self.assertIn("from app.services.scenario_trace_runner", runner_src)
            self.assertIn("def main", runner_src)

            # Visible script must be syntactically valid AND have no app imports.
            visible_src = (
                workspace_root / "public" / "checks" / "run_visible_checks.py"
            ).read_text()
            compile(
                visible_src,
                str(workspace_root / "public" / "checks" / "run_visible_checks.py"),
                "exec",
            )
            self.assertNotIn("from app.", visible_src)
            self.assertNotIn("import anthropic", visible_src)
            self.assertNotIn("import openai", visible_src)

            # README materialized + content checks.
            readme_path = workspace_root / "public" / "README.md"
            self.assertTrue(readme_path.is_file(), "README must be materialized at publish")
            readme = readme_path.read_text()
            # Endpoint table is present.
            self.assertIn("## Endpoint contract", readme)
            self.assertIn("POST", readme)
            self.assertIn("/answer", readme)
            # Quality bars section is present and lists every spec bar.
            self.assertIn("## Quality bars", readme)
            self.assertIn("faithfulness", readme)
            self.assertIn("retrieval_recall", readme)
            # CRAG scaffold block from rag_scaffold (Wave 6.7b). It uses
            # the per-query "search_results" terminology and identifies
            # itself as the CRAG section.
            self.assertIn("CRAG", readme)
            self.assertIn("search_results", readme)
            # NO LLM proxy section since runtime_llm_required=False.
            self.assertNotIn(
                "## LLM access inside the sandbox", readme,
                msg=(
                    "Spec has runtime_llm_required=False, so the LLM "
                    "proxy section MUST NOT appear in the README."
                ),
            )

            # State persistence round-trip: serialize → deserialize → fields agree.
            serialized = final.model_dump(mode="json")
            roundtripped = OutcomeWorkflowState.model_validate(serialized)
            self.assertEqual(roundtripped.stage, final.stage)
            self.assertEqual(roundtripped.status, final.status)
            self.assertEqual(roundtripped.run_id, final.run_id)
            assert roundtripped.spec is not None
            self.assertEqual(roundtripped.spec.title, final.spec.title)
            self.assertIsInstance(
                roundtripped.spec.benchmark, CRAGBenchmarkSource
            )

            # The router exercised every text_format the pipeline needs.
            call_schema_names = {name for _tier, name in router.calls}
            self.assertIn("SpecCoherenceVerdict", call_schema_names)
            self.assertIn("_OracleAuthoringPayload", call_schema_names)
            # planner is faked at the OutcomeCoursePlanner level so we
            # never invoke the router with ``_OutcomePlanPayload``;
            # that's by design — the brief specifies the planner is
            # a fake.

            # ----------------------------------------------------------
            # Final bundle layout report. The test harness captures the
            # tree so the smoke report has a verifiable on-disk artifact.
            # ----------------------------------------------------------
            tree = sorted(
                str(p.relative_to(workspace_root))
                for p in workspace_root.rglob("*")
                if p.is_file()
            )
            # Store for the report (also forces the eager walk).
            self._materialized_tree = tree

            # Sanity check: every artifact named in the design doc
            # appears in the tree.
            required_paths = [
                "public/README.md",
                "public/starter/Dockerfile",
                "public/starter/app/main.py",
                "public/starter/requirements.txt",
                "public/examples/sample_queries.json",
                "public/checks/run_visible_checks.py",
                "private/course_spec.json",
                "private/grader/runner.py",
                "private/grader/_reference/Dockerfile",
                "private/grader/_setup/queries.jsonl",
                "private/grader/_setup/gold_answers.json",
                "private/grader/_setup/search_results_index.json",
                "private/grader/_oracle/outputs.json",
            ]
            for relpath in required_paths:
                self.assertIn(
                    relpath, tree,
                    msg=f"Missing required artifact: {relpath}\nTree:\n  " + "\n  ".join(tree),
                )

            # Sample_queries.json shape — visible payload format.
            sample_text = (
                workspace_root / "public" / "examples" / "sample_queries.json"
            ).read_text()
            samples = json.loads(sample_text)
            self.assertIsInstance(samples, list)
            self.assertGreater(len(samples), 0, "visible sample must have >= 1 query")

            # Gold answers carries the finance-filtered IDs only.
            gold = json.loads(
                (workspace_root / "private" / "grader" / "_setup" / "gold_answers.json").read_text()
            )
            self.assertIsInstance(gold, dict)
            for qid in gold:
                # CRAG fixture only finance/valid IDs survive the filters.
                self.assertNotEqual(qid, "movie_filtered_out")
                self.assertNotEqual(qid, "fin_no_answer_filtered")
                self.assertNotEqual(qid, "fin_test_split_filtered")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
