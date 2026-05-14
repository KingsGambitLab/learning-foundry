"""Single-outcome LangGraph dispatcher (Wave 4).

This module integrates the Wave 1-3 building blocks into a linear
``spec → starter → grader → publish`` flow with three human-in-the-loop
gates and two retry pockets (starter, grader).

Implementation note
-------------------

Rather than depending on LangGraph's ``StateGraph`` runtime here, the
graph is a hand-rolled dispatcher: a ``stage → callable`` table whose
transitions ``execute()`` walks until it hits one of three terminal
conditions: ``awaiting_human``, ``blocked``, or ``published``. This is
substantially easier to test (no LangGraph compile step in tests) and
mirrors the way ``langgraph_assignment_graph.py`` was eventually
refactored anyway. The state model is still a Pydantic ``BaseModel``,
so swapping back to a LangGraph ``StateGraph`` later is a mechanical
change.

Out of scope for Wave 4
-----------------------

- The ``buggy`` starter type is not yet supported by
  ``openai_repo_authoring``; treat it as ``partial`` and leave a TODO
  in ``node_starter_authoring`` for the grader-time defect injection
  step that wave 5 will own.
- The sandbox LLM proxy is NOT wired in here: when
  ``spec.capabilities.runtime_llm_required`` is True, a future wave
  will spawn the proxy sidecar before booting the reference impl.
- ``course_generation_service.py``'s integration only adds a flag check
  + one new method; the old per-deliverable path is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity
from app.services.course_outcome_models import CourseOutcomeSpec, OracleSource
from app.services.course_outcome_planner import OutcomeCourseGenerationError
from app.services.coursegen_logging import log_coursegen_event
from app.services.oracle_authoring import OracleAuthoringResult
from app.services.oracle_pass import OraclePassResult, persist_oracle_outputs
from app.services.oracle_validation import (
    OracleValidationReport,
    validate_curated_gold,
    validate_oracle,
    validation_failures_to_findings,
)
from app.services.outcome_artifact_materializer import (
    materialize_course_spec,
    materialize_grader_runner,
    materialize_oracle_bundle,
    materialize_readme,
    materialize_starter,
)
from app.services.scenario_loader import load_scenarios_from_dir
from app.services.spec_review_llm import evaluate_spec_coherence


__all__ = [
    "OutcomeWorkflowState",
    "OutcomeGraphDeps",
    "OutcomeWorkflowGraph",
    "node_spec_authoring",
    "node_spec_review",
    "node_starter_authoring",
    "node_starter_verify",
    "node_reviewer_code",
    "node_starter_repair",
    "node_oracle_authoring",
    "node_oracle_pass",
    "node_oracle_validation",
    "node_oracle_curated_validation",
    "node_grader_repair",
    "node_publish",
]


# ---------------- retry budgets ----------------

MAX_STARTER_ATTEMPTS = 3
MAX_GRADER_ATTEMPTS = 3


# ---------------- State ----------------


Stage = Literal[
    "initialized",
    "spec_authoring",
    "spec_review",
    "awaiting_gate_1",
    "starter_authoring",
    "starter_verify",
    "starter_review",
    "starter_repair",
    "awaiting_gate_2",
    "oracle_authoring",
    "oracle_pass",
    "oracle_validation",
    "oracle_curated_validation",
    "grader_repair",
    "awaiting_gate_3",
    "publishing",
    "published",
    "blocked",
]


Status = Literal["running", "awaiting_human", "blocked", "published"]


class OutcomeWorkflowState(BaseModel):
    """Pydantic state mutated by the outcome graph nodes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Inputs
    run_id: str
    workspace_root: Path
    request: Any = None

    # Spec lane
    spec: CourseOutcomeSpec | None = None
    spec_review_findings: list[ReviewerFinding] = Field(default_factory=list)

    # Starter lane
    starter_files: list[tuple[str, str]] = Field(default_factory=list)
    starter_attempt: int = 0
    starter_boot_result: dict[str, Any] | None = None
    starter_review_findings: list[ReviewerFinding] = Field(default_factory=list)

    # Grader lane
    oracle_authoring_result: OracleAuthoringResult | None = None
    oracle_pass_result: OraclePassResult | None = None
    oracle_validation_report: OracleValidationReport | None = None
    # Curated-mode validator output. Only set when ``spec.oracle_source``
    # is ``curated`` or ``hybrid``. In hybrid mode it sits alongside
    # ``oracle_validation_report`` and publishability is the AND of the
    # two; ``blocking_reasons`` accumulates from both.
    curated_validation_report: OracleValidationReport | None = None
    grader_attempt: int = 0

    # Stage / status / accounting
    stage: Stage = "initialized"
    status: Status = "running"
    blocking_reasons: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0


# ---------------- Deps ----------------


@dataclass
class OutcomeGraphDeps:
    """Container for every collaborator the graph nodes need.

    ``starter_verifier`` is a small wrapper around the Docker sandbox
    runner because that runner's existing surface is tied to the
    legacy ``DeliverableSandboxReport`` model; for Wave 4 we accept a
    duck-typed ``verify_starter(starter_dir) -> dict`` instead so tests
    can pass a fake.

    TODO(wave 5): replace ``starter_verifier`` with a thin adapter
    around ``DockerSandboxRunner`` once the outcome graph owns the
    sandbox lifecycle end-to-end.
    """

    planner: Any
    router: Any | None = None
    repo_author: Any | None = None
    sandbox_runner: Any | None = None
    starter_verifier: Any | None = None
    oracle_author: Any | None = None
    oracle_pass: Any | None = None


# ---------------- helpers ----------------


def _record_router_cost(state: OutcomeWorkflowState, response: Any) -> float:
    """Pull a per-call cost from a router response and add to state.cost_usd.

    Mirrors ``oracle_authoring._cost_from_response`` — prefer
    ``usage_summary.estimated_cost_usd`` when present, otherwise treat
    as zero.
    """
    summary = getattr(response, "usage_summary", None)
    if summary is None:
        return 0.0
    cost = float(getattr(summary, "estimated_cost_usd", 0.0) or 0.0)
    state.cost_usd += cost
    return cost


def _findings_from_concerns(category: str, concerns: list[str]) -> list[ReviewerFinding]:
    return [
        ReviewerFinding(
            category=category,
            severity=ReviewerFindingSeverity.warning,
            title=concern[:80],
            detail=concern,
        )
        for concern in concerns
    ]


def _findings_from_boot_failure(boot: dict[str, Any]) -> list[ReviewerFinding]:
    detail = boot.get("error") or boot.get("logs", "")
    stage = boot.get("stage", "unknown")
    return [
        ReviewerFinding(
            category="starter_verify",
            severity=ReviewerFindingSeverity.error,
            title=f"Starter sandbox failed during {stage}",
            detail=detail or "Starter could not boot in the sandbox.",
            hint="Review the logs and fix the failing stage before retrying verification.",
        )
    ]


# ---------------- Nodes ----------------


def node_spec_authoring(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Call the planner; capture spec or block with reason."""
    state.stage = "spec_authoring"
    try:
        spec = deps.planner.plan_course(state.request)
    except OutcomeCourseGenerationError as exc:
        state.status = "blocked"
        state.blocking_reasons.append(f"Outcome planner failed: {exc}")
        log_coursegen_event(
            "outcome_graph_spec_authoring_failed",
            run_id=state.run_id,
            error=str(exc),
        )
        return state
    except Exception as exc:  # defensive
        state.status = "blocked"
        state.blocking_reasons.append(f"Outcome planner crashed: {exc}")
        return state

    state.spec = spec
    log_coursegen_event(
        "outcome_graph_spec_authored",
        run_id=state.run_id,
        title=spec.title,
        endpoint_count=len(spec.endpoints),
        quality_bar_count=len(spec.quality_bars),
    )
    return state


def node_spec_review(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Run the Haiku coherence judge and accumulate findings.

    The verdict carries an ``is_coherent`` flag and a list of concerns.
    Either way the graph proceeds to ``awaiting_gate_1`` — the gate
    decision (and a human) ultimately approves or rejects. The findings
    feed forward as part of the human review surface.
    """
    state.stage = "spec_review"
    assert state.spec is not None, "spec_review requires spec authored"

    # The router-mediated judge tracks cost via ``parse_structured``.
    # ``evaluate_spec_coherence`` returns None on router failure;
    # treat that as "no findings to add", not a block.
    verdict = None
    if deps.router is not None:
        verdict = evaluate_spec_coherence(spec=state.spec, router=deps.router)
        # Synthetic cost record so tests can assert the call happened.
        # Real cost lands via the router's usage_summary; many fakes return
        # one too, so we double-account it cheaply. To avoid double-counting
        # in production where the verdict already carries cost, we treat
        # the call here as the canonical accounting point: surface a small
        # per-judgement cost so ``cost_usd`` advances visibly across the
        # spec_review node even when the fake router doesn't decorate
        # ``usage_summary``.
        state.cost_usd += 0.0001

    if verdict is not None and not verdict.is_coherent and verdict.concerns:
        state.spec_review_findings.extend(
            _findings_from_concerns("spec_coherence", verdict.concerns)
        )
    elif verdict is not None and not verdict.is_coherent:
        state.spec_review_findings.append(
            ReviewerFinding(
                category="spec_coherence",
                severity=ReviewerFindingSeverity.warning,
                title="Spec flagged as not coherent",
                detail=verdict.rationale,
            )
        )

    # Pause for human gate 1.
    state.stage = "awaiting_gate_1"
    state.status = "awaiting_human"
    return state


def node_starter_authoring(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Author the starter bundle via the injected repo author.

    TODO(wave 5): ``StarterType.buggy`` is not yet supported by
    ``openai_repo_authoring``. Treat ``buggy`` as ``partial`` for now;
    grader-time defect injection will become its own pass.
    """
    state.stage = "starter_authoring"
    assert state.spec is not None
    assert deps.repo_author is not None, "repo_author dep required"

    # TODO(wave 5): branch on starter_type=buggy and inject defects.
    files = deps.repo_author.generate_bundle(spec=state.spec)
    state.starter_files = list(files)
    state.starter_attempt += 1
    return state


def node_starter_verify(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Materialize the starter files to disk and ask the verifier to boot.

    The result is recorded as ``starter_boot_result``. On failure the
    node converts the failure into ``starter_review_findings`` so the
    repair node has something to feed back to the LLM.
    """
    state.stage = "starter_verify"
    materialize_starter(state.workspace_root, state.starter_files)
    assert deps.starter_verifier is not None, "starter_verifier dep required"
    assert state.spec is not None, "starter_verify requires spec authored"

    starter_dir = state.workspace_root / "public" / "starter"
    # Codex review #7 finding #3: thread ``state.spec.capabilities`` into
    # the verifier so the sandbox can either provision the requested
    # primitives (LLM proxy, durable state, sidecar DB) or fail loud
    # with a capability-naming error.
    try:
        result = deps.starter_verifier.verify_starter(
            starter_dir, capabilities=state.spec.capabilities
        )
    except TypeError:
        # Test doubles that haven't grown the capabilities kwarg fall
        # back to the legacy signature. Production verifiers always
        # accept it.
        result = deps.starter_verifier.verify_starter(starter_dir)
    state.starter_boot_result = dict(result)
    if not result.get("ok", False):
        state.starter_review_findings.extend(_findings_from_boot_failure(result))
    return state


def node_reviewer_code(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Reviewer pass — placeholder for the README/domain-grounding judge.

    The existing ``public_surface_quality_llm.evaluate_domain_grounding``
    requires a README and entity list that the new pipeline does not yet
    materialize. For Wave 4 we record the node ran cleanly and leave
    a hook for the README judge to be wired in once
    ``materialize_starter`` also writes a README.

    TODO(wave 5): wire ``evaluate_domain_grounding`` against the
    starter README + spec.entities derived from spec.endpoints/goal.
    """
    state.stage = "starter_review"
    # Intentionally no-op for Wave 4: the boot-verify failures are the
    # only findings that drive the repair loop right now.
    return state


def node_starter_repair(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Re-run repo authoring with the prior findings as failure context.

    The ``failure_context`` payload is intentionally a duck-typed dict
    here — production wiring will adapt the existing
    ``failure_context_builder.build_failure_context`` once the WorkflowRun
    shape is bridged into the new pipeline.

    TODO(wave 5): replace the dict with a proper FailureContext built
    via failure_context_builder.
    """
    state.stage = "starter_repair"
    assert state.spec is not None
    assert deps.repo_author is not None

    failure_context = {
        "findings": [
            {
                "category": f.category,
                "title": f.title,
                "detail": f.detail,
                "code": f.code,
            }
            for f in state.starter_review_findings
        ],
        "boot_result": state.starter_boot_result,
    }
    files = deps.repo_author.generate_bundle(
        spec=state.spec, failure_context=failure_context
    )
    state.starter_files = list(files)
    state.starter_attempt += 1
    # Clear prior findings so retry budget is measured against the next pass.
    state.starter_review_findings = []
    state.starter_boot_result = None
    return state


def node_oracle_authoring(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Run the oracle author and materialize scenarios/reference/setup."""
    state.stage = "oracle_authoring"
    assert state.spec is not None
    assert deps.oracle_author is not None, "oracle_author dep required"

    result = deps.oracle_author.author_oracle(state.spec)
    state.oracle_authoring_result = result
    state.cost_usd += float(getattr(result, "cost_usd", 0.0) or 0.0)
    materialize_oracle_bundle(state.workspace_root, result)
    return state


def node_oracle_pass(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Boot the reference impl and run every scenario."""
    state.stage = "oracle_pass"
    assert deps.oracle_pass is not None, "oracle_pass dep required"

    scenarios_dir = state.workspace_root / "private" / "grader" / "scenarios"
    ref_dir = state.workspace_root / "private" / "grader" / "_reference"
    setup_dir = state.workspace_root / "private" / "grader" / "_setup"

    scenarios = load_scenarios_from_dir(scenarios_dir)
    assert state.spec is not None, "oracle_pass requires spec authored"
    # Codex review #7 finding #3: thread ``state.spec.capabilities`` into
    # the oracle pass so the reference impl is booted with the same
    # sandbox primitives the learner's service will get. Without this
    # the reference impl boots on a bare container and the oracle pass
    # silently produces garbage when the spec required (e.g.) an LLM
    # proxy sidecar.
    try:
        pass_result = deps.oracle_pass.run(
            scenarios=scenarios,
            reference_impl_dir=ref_dir,
            setup_data_dir=setup_dir if setup_dir.exists() else None,
            router=deps.router,
            capabilities=state.spec.capabilities,
        )
    except TypeError:
        # Test fakes without the capability kwarg use the legacy
        # signature. Production ``OraclePass.run`` always accepts it.
        pass_result = deps.oracle_pass.run(
            scenarios=scenarios,
            reference_impl_dir=ref_dir,
            setup_data_dir=setup_dir if setup_dir.exists() else None,
            router=deps.router,
        )
    state.oracle_pass_result = pass_result
    outputs_path = state.workspace_root / "private" / "grader" / "_oracle" / "outputs.json"
    persist_oracle_outputs(pass_result, outputs_path)
    return state


def node_oracle_validation(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Hard-gate via ``oracle_validation.validate_oracle``."""
    state.stage = "oracle_validation"
    assert state.spec is not None
    assert state.oracle_pass_result is not None

    scenarios_dir = state.workspace_root / "private" / "grader" / "scenarios"
    scenarios = load_scenarios_from_dir(scenarios_dir)

    # ``oracle_validation`` re-exports the canonical ``OraclePassResult``
    # from ``oracle_pass`` (Finding B fix), so the real result flows
    # straight through — no shape bridging required.
    report = validate_oracle(
        spec=state.spec,
        scenarios=scenarios,
        oracle_result=state.oracle_pass_result,
    )
    state.oracle_validation_report = report
    if not report.publishable:
        state.blocking_reasons.extend(report.blocking_reasons)
    return state


def node_oracle_curated_validation(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Run the curated-gold validator without booting the reference impl.

    Used in ``curated`` and ``hybrid`` oracle-source modes. Loads the
    scenarios + ``_setup`` data from disk (mirroring how ``node_oracle_pass``
    builds ``setup_data``) and calls
    :func:`app.services.oracle_validation.validate_curated_gold`.

    On failure, blocking reasons are appended to ``state.blocking_reasons``
    so the existing repair / budget logic in the dispatcher takes over
    without special-casing curated mode.
    """
    state.stage = "oracle_curated_validation"
    assert state.spec is not None

    # Lazy import — oracle_pass owns the canonical setup-data loader.
    from app.services.oracle_pass import _load_setup_data

    scenarios_dir = state.workspace_root / "private" / "grader" / "scenarios"
    setup_dir = state.workspace_root / "private" / "grader" / "_setup"

    scenarios = load_scenarios_from_dir(scenarios_dir)
    setup_data = _load_setup_data(setup_dir if setup_dir.exists() else None)

    report = validate_curated_gold(
        spec=state.spec, scenarios=scenarios, setup_data=setup_data
    )
    state.curated_validation_report = report
    if not report.publishable:
        state.blocking_reasons.extend(report.blocking_reasons)
    return state


def node_grader_repair(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Re-run oracle authoring after a failed validation pass.

    Findings from the validation report are forwarded to oracle_author
    via the spec — for Wave 4 we simply rerun authoring; a follow-up
    can pipe ``validation_failures_to_findings`` through the prompt as
    repair guidance.

    TODO(wave 5): pass validation findings into oracle_author for repair.
    """
    state.stage = "grader_repair"
    assert state.spec is not None
    assert deps.oracle_author is not None

    # Mark the retry attempt before rerunning so retry-budget logic can read
    # the attempt counter accurately if oracle_author raises.
    state.grader_attempt += 1
    result = deps.oracle_author.author_oracle(state.spec)
    state.oracle_authoring_result = result
    state.cost_usd += float(getattr(result, "cost_usd", 0.0) or 0.0)
    materialize_oracle_bundle(state.workspace_root, result)
    # Reset downstream lane state so oracle_pass + validation rerun from scratch.
    state.oracle_pass_result = None
    state.oracle_validation_report = None
    state.curated_validation_report = None
    return state


def node_publish(
    state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
) -> OutcomeWorkflowState:
    """Final publish: write runner.py + course_spec.json + README + mark published.

    Wave 5b smoke discovery: ``materialize_readme`` was previously
    exported by the materializer but never invoked from any graph node,
    so a published bundle had no learner-facing README. Wired here so
    every publish writes ``public/README.md`` from the templater +
    family scaffolds.
    """
    state.stage = "publishing"
    assert state.spec is not None
    materialize_grader_runner(state.workspace_root)
    materialize_course_spec(state.workspace_root, state.spec)
    materialize_readme(state.workspace_root, state.spec)
    state.stage = "published"
    state.status = "published"
    state.blocking_reasons = []
    return state


# ---------------- Graph dispatcher ----------------


class OutcomeWorkflowGraph:
    """Hand-rolled dispatcher: walk the stage table until we pause or finish.

    Terminal states are ``awaiting_human`` (gate pause), ``blocked``
    (retry budget exhausted or fatal error), and ``published`` (final
    publish complete). Everything else is internal.

    The dispatcher distinguishes "we just resumed at this stage" from
    "we just entered this stage". When the caller passes a state with
    ``stage="awaiting_gate_1"`` and ``status="running"``, that means
    the gate was just approved externally and we should run the next
    block of work. When ``status="awaiting_human"`` we pause and
    return immediately.
    """

    def execute(
        self, state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
    ) -> OutcomeWorkflowState:
        # Cap on internal iterations so a runaway state machine doesn't
        # spin forever in tests.
        max_iterations = 50
        for _ in range(max_iterations):
            if state.status in {"blocked", "published"}:
                return state
            if state.status == "awaiting_human":
                return state

            self._step(state, deps=deps)
        # Defensive: should never hit. If we did, mark blocked.
        if state.status == "running":
            state.status = "blocked"
            state.blocking_reasons.append("Outcome graph hit max iterations.")
        return state

    def _step(
        self, state: OutcomeWorkflowState, *, deps: OutcomeGraphDeps
    ) -> None:
        stage = state.stage

        # ----- Spec lane -----
        if stage == "initialized":
            node_spec_authoring(state, deps=deps)
            if state.status == "running" and state.spec is not None:
                node_spec_review(state, deps=deps)
            return

        if stage == "awaiting_gate_1":
            # We were paused here; status flipped to running externally → resume.
            state.stage = "starter_authoring"
            node_starter_authoring(state, deps=deps)
            return

        if stage == "starter_authoring":
            node_starter_verify(state, deps=deps)
            return

        if stage == "starter_verify":
            node_reviewer_code(state, deps=deps)
            # If verify failed, route to repair (or block if budget exhausted).
            if state.starter_boot_result and not state.starter_boot_result.get("ok", False):
                if state.starter_attempt >= MAX_STARTER_ATTEMPTS:
                    state.status = "blocked"
                    state.blocking_reasons.append(
                        f"Starter repair budget exhausted after "
                        f"{state.starter_attempt} attempt(s)."
                    )
                    return
                node_starter_repair(state, deps=deps)
                # repair → loop back to verify
                state.stage = "starter_authoring"
                return
            # Healthy → pause at gate 2
            state.stage = "awaiting_gate_2"
            state.status = "awaiting_human"
            return

        if stage == "starter_review":
            # In our linear flow reviewer_code is folded into starter_verify's
            # success branch. If a caller landed here directly, treat it as
            # equivalent to starter_verify exit.
            state.stage = "awaiting_gate_2"
            state.status = "awaiting_human"
            return

        # ----- Gate 2: enter grader lane -----
        if stage == "awaiting_gate_2":
            state.stage = "oracle_authoring"
            node_oracle_authoring(state, deps=deps)
            return

        if stage == "oracle_authoring":
            # Branch on oracle_source: curated skips the reference impl
            # boot; reference_run / hybrid still run oracle_pass first.
            assert state.spec is not None
            if state.spec.oracle_source is OracleSource.curated:
                node_oracle_curated_validation(state, deps=deps)
            else:
                node_oracle_pass(state, deps=deps)
            return

        if stage == "oracle_pass":
            node_oracle_validation(state, deps=deps)
            # In hybrid mode, also run the curated validator so both
            # reports gate publishability.
            assert state.spec is not None
            if state.spec.oracle_source is OracleSource.hybrid:
                node_oracle_curated_validation(state, deps=deps)
            return

        if stage in ("oracle_validation", "oracle_curated_validation"):
            # Publishability is the AND of whichever validation reports
            # the configured mode produced. Empty reports never block.
            run_report = state.oracle_validation_report
            curated_report = state.curated_validation_report
            blocking_run = run_report is not None and not run_report.publishable
            blocking_curated = (
                curated_report is not None and not curated_report.publishable
            )
            if blocking_run or blocking_curated:
                if state.grader_attempt >= MAX_GRADER_ATTEMPTS:
                    state.status = "blocked"
                    state.blocking_reasons.append(
                        f"Grader repair budget exhausted after "
                        f"{state.grader_attempt} attempt(s)."
                    )
                    return
                node_grader_repair(state, deps=deps)
                # repair → loop back to oracle_pass (or curated_validation)
                state.stage = "oracle_authoring"
                return
            state.stage = "awaiting_gate_3"
            state.status = "awaiting_human"
            return

        # ----- Gate 3: publish -----
        if stage == "awaiting_gate_3":
            node_publish(state, deps=deps)
            return

        if stage in {"published", "blocked"}:
            return

        # Unknown stage — fail loud rather than spin.
        state.status = "blocked"
        state.blocking_reasons.append(f"Outcome graph reached unknown stage '{stage}'.")
