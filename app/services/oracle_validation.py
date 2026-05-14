"""Hard publish-gate that validates the oracle pass outcome.

The ``oracle_validation`` node sits between ``oracle_pass`` and
``gate_3_pre_publish`` in the simplified single-outcome course pipeline
(see ``docs/superpowers/specs/2026-05-14-scenario-rubrics-rag-mvp-design.md``).

It consumes:

- a :class:`CourseOutcomeSpec` (the publishability target),
- the curated ``list[Scenario]`` (one YAML file each),
- an :class:`OraclePassResult` produced by the reference implementation
  running every scenario,

and answers a single question: **is this course publishable?**

A course is publishable IFF:

1. Every scenario passed in oracle_pass (no fails, no aborts).
2. The scenario set covers every category required by the spec
   (see :func:`_required_categories` for the heuristic).
3. No scenario relies only on structural / trivial rubrics — every
   scenario must have at least one rubric beyond ``schema_match`` /
   ``literal_match`` / ``regex_match``. This catches the "schema-only
   grader" anti-pattern where a stub implementation could pass.

When the gate blocks publication, the same module's
:func:`validation_failures_to_findings` converts every blocking reason
into a :class:`ReviewerFinding` so the existing reviewer-repair channel
forwards actionable hints to the repair LLM.

The :class:`OraclePassResult` / :class:`OracleScenarioOutput` types are
re-exported from :mod:`app.services.oracle_pass` — that module owns the
canonical contract. ``verdicts`` are tuple-shaped
``list[tuple[str, dict]]`` (rubric_kind + serialized Verdict payload),
matching what ``oracle_pass.OraclePass.run`` produces and what the
on-disk JSON form preserves.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.workflow import ReviewerFinding, ReviewerFindingSeverity
from app.services.course_outcome_models import (
    CourseOutcomeSpec,
    HttpMethod,
)
# Re-export the canonical oracle_pass contract. ``oracle_validation`` used
# to declare parallel ``OracleScenarioOutput`` / ``OraclePassResult`` models
# with a ``list[dict]`` ``verdicts`` field, which silently disagreed with
# what ``oracle_pass`` actually produced (``list[tuple[str, dict]]``).
# Importing here keeps a single source of truth (Finding B).
from app.services.oracle_pass import OraclePassResult, OracleScenarioOutput
from app.services.scenario_loader import Scenario

__all__ = [
    "OracleScenarioOutput",
    "OraclePassResult",
    "CategoryCoverageStatus",
    "OracleValidationReport",
    "validate_oracle",
    "validate_curated_gold",
    "validation_failures_to_findings",
]


# ---------------- validator report types ----------------


class CategoryCoverageStatus(BaseModel):
    """Coverage status for one scenario category in the validation taxonomy.

    ``is_required`` reflects the spec-derived requirement (see
    :func:`_required_categories`). When ``is_required=False`` the
    category appears for transparency only and missing coverage is not
    blocking.

    ``not_applicable_reason`` is set when the category is not required
    *and* not present, with a short string explaining why the spec did
    not pull it in (e.g. "no abstention quality bar").
    """

    category: str
    present: bool
    scenario_count: int
    is_required: bool
    not_applicable_reason: str | None = None


class OracleValidationReport(BaseModel):
    """Structured outcome of the publish-gate validation.

    ``publishable`` is the single load-bearing field downstream nodes
    consume. The other fields exist so the repair lane (and audit log)
    can surface specific failures back to the LLM / human.
    """

    publishable: bool
    reference_impl_hash: str
    scenario_set_hash: str

    failed_scenarios: list[str] = Field(default_factory=list)
    aborted_scenarios: list[str] = Field(default_factory=list)

    coverage: list[CategoryCoverageStatus] = Field(default_factory=list)
    missing_required_categories: list[str] = Field(default_factory=list)

    trivial_rubric_warnings: list[str] = Field(default_factory=list)

    summary: str = ""
    blocking_reasons: list[str] = Field(default_factory=list)


# ---------------- heuristic helpers ----------------


# Rubrics that are structural / pattern-only. A scenario whose ONLY
# rubrics fall in this set is too easy to satisfy with a stub.
_TRIVIAL_RUBRIC_KINDS: frozenset[str] = frozenset(
    {"schema_match", "literal_match", "regex_match"}
)


# Quality-bar IDs containing any of these substrings flag the course
# as requiring an ``out_of_scope`` scenario category.
_ABSTENTION_BAR_TOKENS: tuple[str, ...] = ("abstention", "refusal", "out_of_scope")


# Endpoint description tokens that flag a create/update verb regardless
# of path shape. ``register`` is included for course-pattern bars like
# "register corpus".
_CREATE_VERB_TOKENS: tuple[str, ...] = ("create", "update", "register")


def _endpoint_is_create_shaped(method: HttpMethod, path: str, description: str) -> bool:
    """Heuristic: does this endpoint plausibly create or update a resource?

    Triggers when:

    - method is ``POST`` or ``PUT``, AND
    - path contains ``{id}`` (path-templated resource creation), OR
    - description (lower-cased) mentions ``create`` / ``update`` /
      ``register``.

    This is intentionally a heuristic — false positives merely add
    coverage requirements; false negatives let create-shaped courses
    sneak through without an idempotency scenario. We prefer the
    over-eager direction so the bar is higher than the floor.
    """
    if method not in (HttpMethod.POST, HttpMethod.PUT):
        return False
    if "{id}" in path:
        return True
    desc = description.lower()
    return any(tok in desc for tok in _CREATE_VERB_TOKENS)


def _required_categories(spec: CourseOutcomeSpec) -> set[str]:
    """Compute the set of scenario categories the spec requires coverage for.

    Always required:
      ``happy_path``, ``boundary``, ``malformed_input``.

    Conditionally required:
      - ``out_of_scope`` — when any ``QualityBar.id`` mentions
        abstention / refusal / out_of_scope (case-insensitive).
      - ``idempotency`` — when at least one endpoint is "create-shaped"
        (see :func:`_endpoint_is_create_shaped`).
      - ``composition`` — when the spec has 3+ endpoints.

    ``adversarial`` is intentionally NOT in the required set; it surfaces
    as a soft warning elsewhere when missing for security-sensitive
    endpoints, not as a publish blocker.
    """
    required: set[str] = {"happy_path", "boundary", "malformed_input"}

    if _spec_has_abstention_bar(spec):
        required.add("out_of_scope")

    if any(
        _endpoint_is_create_shaped(ep.method, ep.path, ep.description)
        for ep in spec.endpoints
    ):
        required.add("idempotency")

    if len(spec.endpoints) >= 3:
        required.add("composition")

    return required


def _spec_has_abstention_bar(spec: CourseOutcomeSpec) -> bool:
    for bar in spec.quality_bars:
        lower = bar.id.lower()
        if any(tok in lower for tok in _ABSTENTION_BAR_TOKENS):
            return True
    return False


def _scenario_passed(output: OracleScenarioOutput) -> bool:
    """A scenario passes iff it did not abort and every verdict is ``pass``.

    ``output.verdicts`` is the canonical
    ``list[tuple[str, dict]]`` shape from ``oracle_pass`` — each entry
    is ``(rubric_kind, Verdict.model_dump())``. We read ``status`` from
    the second element.

    An empty ``verdicts`` list is NOT a pass — it means the scenario
    yielded no judgments, which is treated as a failure to be on the
    safe side (the upstream runner is required to emit a verdict per
    rubric).
    """
    if output.aborted:
        return False
    if not output.verdicts:
        return False
    return all(
        verdict_payload.get("status") == "pass"
        for _rubric_kind, verdict_payload in output.verdicts
    )


def _scenario_only_trivial_rubrics(scenario: Scenario) -> bool:
    """``True`` iff every rubric on this scenario is in the trivial set."""
    if not scenario.rubrics:
        return False
    return all(r.kind in _TRIVIAL_RUBRIC_KINDS for r in scenario.rubrics)


def _uncovered_quality_bars(
    spec: CourseOutcomeSpec, scenarios: list[Scenario]
) -> list[str]:
    """Return the IDs of ``QualityBar``s no scenario references.

    A spec-declared bar that no scenario contributes evidence toward is
    a course-authoring configuration error — the grader would silently
    "abstain" on that contract and the synthesizer used to mark the
    overall report ``pass``. The publish gate blocks this at validation
    time so a broken grader never reaches learners (Codex review #4
    finding #2). Companion defense-in-depth check lives in
    ``grader_feedback_synthesizer.synthesize_grader_feedback``.
    """
    targeted: set[str] = {
        bar_id for s in scenarios for bar_id in s.quality_bar_ids
    }
    return [bar.id for bar in spec.quality_bars if bar.id not in targeted]


def _coverage_for_category(
    category: str,
    *,
    scenarios: list[Scenario],
    required: set[str],
    spec: CourseOutcomeSpec,
) -> CategoryCoverageStatus:
    matched = [s for s in scenarios if s.category == category]
    is_required = category in required
    present = bool(matched)
    not_applicable_reason: str | None = None
    if not is_required and not present:
        not_applicable_reason = _why_not_required(category, spec)
    return CategoryCoverageStatus(
        category=category,
        present=present,
        scenario_count=len(matched),
        is_required=is_required,
        not_applicable_reason=not_applicable_reason,
    )


def _why_not_required(category: str, spec: CourseOutcomeSpec) -> str | None:
    """One-line explanation of why an optional category is unrequired here."""
    if category == "out_of_scope":
        return (
            "no quality_bar with an abstention/refusal/out_of_scope id "
            "is declared"
        )
    if category == "idempotency":
        return "no POST/PUT endpoint looks create-shaped"
    if category == "composition":
        return f"spec has only {len(spec.endpoints)} endpoint(s)"
    if category == "adversarial":
        return "adversarial coverage is advisory, not required"
    return None


# ---------------- public API ----------------


def validate_oracle(
    *,
    spec: CourseOutcomeSpec,
    scenarios: list[Scenario],
    oracle_result: OraclePassResult,
) -> OracleValidationReport:
    """Hard-gate publishability based on oracle-pass outcomes + coverage.

    Returns a structured :class:`OracleValidationReport`. The caller
    (typically the LangGraph node wrapper) inspects ``publishable`` and
    ``blocking_reasons``, then either advances to ``gate_3_pre_publish``
    or routes through the repair lane via
    :func:`validation_failures_to_findings`.
    """
    required = _required_categories(spec)

    # --- per-scenario pass / abort lists ---
    failed_scenarios: list[str] = []
    aborted_scenarios: list[str] = []
    for output in oracle_result.scenario_outputs:
        if output.aborted:
            aborted_scenarios.append(output.scenario_id)
            continue
        if not _scenario_passed(output):
            failed_scenarios.append(output.scenario_id)

    # --- per-category coverage ---
    # We report on the union of (required categories, categories
    # actually present in the scenarios) so the report is informative
    # even when a category is over-supplied for an optional reason.
    present_categories = {s.category for s in scenarios}
    coverage_categories = sorted(required | present_categories)
    coverage = [
        _coverage_for_category(
            cat, scenarios=scenarios, required=required, spec=spec
        )
        for cat in coverage_categories
    ]
    missing_required_categories = sorted(
        required - {s.category for s in scenarios}
    )

    # --- anti-trivial-rubric check ---
    trivial_rubric_warnings = [
        s.id for s in scenarios if _scenario_only_trivial_rubrics(s)
    ]

    # --- compose blocking reasons ---
    blocking_reasons: list[str] = []
    for sid in failed_scenarios:
        blocking_reasons.append(
            f"Scenario '{sid}' did not pass under the reference implementation."
        )
    for sid in aborted_scenarios:
        blocking_reasons.append(
            f"Scenario '{sid}' aborted mid-trace; oracle pass could not complete."
        )
    for cat in missing_required_categories:
        blocking_reasons.append(
            f"Required scenario category '{cat}' has no scenarios."
        )
    for sid in trivial_rubric_warnings:
        blocking_reasons.append(
            f"Scenario '{sid}' uses only structural rubrics "
            f"(schema_match/literal_match/regex_match); add at least one "
            f"semantic rubric (e.g. oracle_set_overlap, llm_judge_coverage, "
            f"behavioral_equivalence, subset_match, numeric_range)."
        )
    for bar_id in _uncovered_quality_bars(spec, scenarios):
        blocking_reasons.append(
            f"Quality bar '{bar_id}' is declared in the spec but no "
            f"scenario contributes to it (no scenario lists this id in "
            f"its quality_bar_ids). Either author at least one scenario "
            f"that references this bar or remove the bar from the spec."
        )

    publishable = not blocking_reasons

    if publishable:
        summary = (
            f"Publishable: {oracle_result.passed_scenarios}/"
            f"{oracle_result.total_scenarios} scenarios passed; "
            f"all required categories covered."
        )
    else:
        summary = (
            f"Not publishable: {len(blocking_reasons)} blocking issue(s) "
            f"({len(failed_scenarios)} failed, {len(aborted_scenarios)} aborted, "
            f"{len(missing_required_categories)} missing categor(ies), "
            f"{len(trivial_rubric_warnings)} trivial-rubric scenario(s))."
        )

    return OracleValidationReport(
        publishable=publishable,
        reference_impl_hash=oracle_result.reference_impl_hash,
        scenario_set_hash=oracle_result.scenario_set_hash,
        failed_scenarios=failed_scenarios,
        aborted_scenarios=aborted_scenarios,
        coverage=coverage,
        missing_required_categories=missing_required_categories,
        trivial_rubric_warnings=trivial_rubric_warnings,
        summary=summary,
        blocking_reasons=blocking_reasons,
    )


# ---------------- curated-gold validator ----------------


def _resolve_gold_path(setup_data: dict[str, Any], dotted_path: str) -> Any:
    """Walk ``dotted_path`` through ``setup_data``.

    Mirrors :func:`app.services.scenario_rubrics_base.resolve_path` at
    the subset used by ``oracle_set_overlap``: dotted keys only (no
    list-index suffix). Local copy so this module has no dependency
    cycle with the rubric library.

    Raises ``KeyError`` when any segment is missing.
    """
    if dotted_path == "":
        return setup_data
    current: Any = setup_data
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            raise KeyError(part)
        if part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _curated_corpus_universe(setup_data: dict[str, Any]) -> set[str] | None:
    """Heuristically derive the corpus's doc_id universe from setup_data.

    Treats any top-level setup_data value that is a non-empty ``list``
    of dicts each carrying a ``doc_id`` key as the corpus. Returns the
    union of those IDs as a ``set``. When no such file is found, returns
    ``None`` so the caller can treat the universe check as "skipped
    with a warning" rather than blocking.
    """
    for value in setup_data.values():
        if not isinstance(value, list) or not value:
            continue
        if all(isinstance(item, dict) and "doc_id" in item for item in value):
            return {str(item["doc_id"]) for item in value}
    return None


def validate_curated_gold(
    *,
    spec: CourseOutcomeSpec,
    scenarios: list[Scenario],
    setup_data: dict[str, Any],
) -> OracleValidationReport:
    """Hard-gate publishability of a curated-gold oracle bundle.

    Used by the LangGraph dispatcher when
    ``spec.oracle_source == OracleSource.curated`` (and as half of the
    ``hybrid`` mode). The reference implementation is NOT booted in
    this mode — gold sets and required-fact lists live on disk under
    ``private/grader/_setup/`` and inline in each scenario rubric.

    Checks (each failure becomes one ``blocking_reasons`` entry):

      1. Every scenario with an ``oracle_set_overlap`` rubric must
         resolve its ``gold_set_path`` to a non-empty value inside
         ``setup_data``.
      2. When the setup_data heuristically exposes a corpus universe
         (a top-level list of ``{"doc_id": ...}`` dicts), every gold
         doc_id referenced by an ``oracle_set_overlap`` rubric must
         appear in the universe. When the universe is unloadable, the
         check is skipped (no warning surfaced today — kept as a future
         hook).
      3. Every scenario with an ``llm_judge_coverage`` rubric must
         carry a non-empty ``must_contain_facts`` config inline.
      4. Category coverage matches :func:`_required_categories`.
      5. Anti-trivial-rubric check, same as :func:`validate_oracle`.

    Returns ``OracleValidationReport`` with ``reference_impl_hash`` and
    ``scenario_set_hash`` set to empty strings — there is no hashable
    reference impl in curated mode.
    """
    required = _required_categories(spec)

    present_categories = {s.category for s in scenarios}
    coverage_categories = sorted(required | present_categories)
    coverage = [
        _coverage_for_category(
            cat, scenarios=scenarios, required=required, spec=spec
        )
        for cat in coverage_categories
    ]
    missing_required_categories = sorted(required - present_categories)

    trivial_rubric_warnings = [
        s.id for s in scenarios if _scenario_only_trivial_rubrics(s)
    ]

    blocking_reasons: list[str] = []

    # --- (1) and (2): gold-set rubrics resolve and IDs are in-universe ---
    corpus_universe = _curated_corpus_universe(setup_data)
    for scenario in scenarios:
        for rubric in scenario.rubrics:
            if rubric.kind != "oracle_set_overlap":
                continue
            gold_path = rubric.config.get("gold_set_path")
            if not isinstance(gold_path, str) or not gold_path:
                blocking_reasons.append(
                    f"Scenario '{scenario.id}' has an oracle_set_overlap "
                    f"rubric with a missing or empty 'gold_set_path' config."
                )
                continue
            try:
                gold_value = _resolve_gold_path(setup_data, gold_path)
            except KeyError:
                blocking_reasons.append(
                    f"Scenario '{scenario.id}' references gold path "
                    f"'{gold_path}' that is not present in the curated "
                    f"setup_data."
                )
                continue
            if not isinstance(gold_value, list) or not gold_value:
                blocking_reasons.append(
                    f"Scenario '{scenario.id}' resolved gold path "
                    f"'{gold_path}' to an empty or non-list value in "
                    f"setup_data."
                )
                continue
            if corpus_universe is not None:
                offenders = [
                    str(item) for item in gold_value if str(item) not in corpus_universe
                ]
                if offenders:
                    blocking_reasons.append(
                        f"Scenario '{scenario.id}' gold path '{gold_path}' "
                        f"references doc_id(s) {offenders} not present in "
                        f"the curated corpus."
                    )

    # --- (3): llm_judge_coverage must carry inline must_contain_facts ---
    for scenario in scenarios:
        for rubric in scenario.rubrics:
            if rubric.kind != "llm_judge_coverage":
                continue
            facts = rubric.config.get("must_contain_facts")
            if not isinstance(facts, list) or not facts:
                blocking_reasons.append(
                    f"Scenario '{scenario.id}' has an llm_judge_coverage "
                    f"rubric with empty or missing 'must_contain_facts' "
                    f"config; curated mode requires the facts inline."
                )

    # --- (4): required-category coverage ---
    for cat in missing_required_categories:
        blocking_reasons.append(
            f"Required scenario category '{cat}' has no scenarios."
        )

    # --- (5): trivial-rubric warnings ---
    for sid in trivial_rubric_warnings:
        blocking_reasons.append(
            f"Scenario '{sid}' uses only structural rubrics "
            f"(schema_match/literal_match/regex_match); add at least one "
            f"semantic rubric (e.g. oracle_set_overlap, llm_judge_coverage, "
            f"behavioral_equivalence, subset_match, numeric_range)."
        )

    # --- (6): quality-bar coverage (Codex review #4 finding #2) ---
    for bar_id in _uncovered_quality_bars(spec, scenarios):
        blocking_reasons.append(
            f"Quality bar '{bar_id}' is declared in the spec but no "
            f"scenario contributes to it (no scenario lists this id in "
            f"its quality_bar_ids). Either author at least one scenario "
            f"that references this bar or remove the bar from the spec."
        )

    publishable = not blocking_reasons

    if publishable:
        summary = (
            f"Publishable (curated): {len(scenarios)} scenarios cover "
            f"all required categories; gold sets and required facts are "
            f"consistent with setup_data."
        )
    else:
        summary = (
            f"Not publishable (curated): {len(blocking_reasons)} blocking "
            f"issue(s) "
            f"({len(missing_required_categories)} missing categor(ies), "
            f"{len(trivial_rubric_warnings)} trivial-rubric scenario(s))."
        )

    return OracleValidationReport(
        publishable=publishable,
        reference_impl_hash="",
        scenario_set_hash="",
        failed_scenarios=[],
        aborted_scenarios=[],
        coverage=coverage,
        missing_required_categories=missing_required_categories,
        trivial_rubric_warnings=trivial_rubric_warnings,
        summary=summary,
        blocking_reasons=blocking_reasons,
    )


def validation_failures_to_findings(
    report: OracleValidationReport,
) -> list[ReviewerFinding]:
    """Convert each blocking reason into a :class:`ReviewerFinding`.

    Returns an empty list when ``report.publishable`` is True. Every
    finding carries an actionable ``hint`` the repair LLM can paste
    directly into a follow-up scenario authoring or rubric tightening
    pass.
    """
    if report.publishable:
        return []

    findings: list[ReviewerFinding] = []

    for sid in report.failed_scenarios:
        findings.append(
            ReviewerFinding(
                category="oracle_validation",
                severity=ReviewerFindingSeverity.error,
                title=f"Scenario '{sid}' failed under reference implementation",
                detail=(
                    f"Scenario '{sid}' did not pass when run against the "
                    f"reference implementation. Either the reference impl "
                    f"is broken or the scenario's rubrics are mis-tuned."
                ),
                code=f"oracle_scenario_failed_{sid}",
                hint=(
                    f"Inspect the reference implementation's response for "
                    f"scenario '{sid}' and either tighten the rubric or fix "
                    f"the reference impl until 100% of scenarios pass."
                ),
            )
        )

    for sid in report.aborted_scenarios:
        findings.append(
            ReviewerFinding(
                category="oracle_validation",
                severity=ReviewerFindingSeverity.error,
                title=f"Scenario '{sid}' aborted mid-trace",
                detail=(
                    f"Scenario '{sid}' could not be completed under the "
                    f"reference implementation (connection error, timeout, "
                    f"or runtime crash)."
                ),
                code=f"oracle_scenario_aborted_{sid}",
                hint=(
                    f"Re-run scenario '{sid}' locally against the reference "
                    f"implementation; surface the underlying error and fix "
                    f"the upstream cause before re-running oracle_pass."
                ),
            )
        )

    for cat in report.missing_required_categories:
        findings.append(
            ReviewerFinding(
                category="oracle_validation",
                severity=ReviewerFindingSeverity.error,
                title=f"Missing required scenario category '{cat}'",
                detail=(
                    f"The scenario set has no entries with "
                    f"category='{cat}', but the spec requires this category "
                    f"for publishability."
                ),
                code=f"oracle_missing_category_{cat}",
                hint=(
                    f"Add at least one scenario with category='{cat}' that "
                    f"exercises the relevant endpoint(s) and includes a "
                    f"non-trivial rubric."
                ),
            )
        )

    for sid in report.trivial_rubric_warnings:
        findings.append(
            ReviewerFinding(
                category="oracle_validation",
                severity=ReviewerFindingSeverity.error,
                title=f"Scenario '{sid}' uses only structural rubrics",
                detail=(
                    f"Scenario '{sid}' relies exclusively on structural "
                    f"rubrics (schema_match/literal_match/regex_match). A "
                    f"stub implementation could pass these without "
                    f"implementing the real behavior."
                ),
                code=f"oracle_trivial_rubric_{sid}",
                hint=(
                    f"Add at least one semantic rubric to scenario '{sid}' "
                    f"(oracle_set_overlap, llm_judge_coverage, "
                    f"behavioral_equivalence, subset_match, or numeric_range)."
                ),
            )
        )

    return findings
