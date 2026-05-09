from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.domain.task_agent import (
    EndpointSpec,
    LearnerDeliverableBrief,
    LearnerStarterSurfaceSpec,
    DeliverableSpec,
    PublicCheckSpec,
    StarterScenarioSpec,
    TaskAgentServiceSpec,
    TaskEvalCase,
)
from app.services.review_area_coverage import apply_inferred_review_area_case_tags


def _editable_files_for_spec(spec: TaskAgentServiceSpec) -> list[str]:
    return list(spec.runtime_dependencies.editable_files or ["app.py"])


def _primary_editable_file(spec: TaskAgentServiceSpec) -> str:
    files = _editable_files_for_spec(spec)
    return files[0] if files else "app.py"


def _visible_check_command(spec: TaskAgentServiceSpec) -> str:
    return spec.runtime_dependencies.visible_check_command or "python checks/run_visible_checks.py"


def _preview_command(spec: TaskAgentServiceSpec) -> str:
    return spec.runtime_dependencies.preview_command or "python -m uvicorn app:app --host 127.0.0.1 --port ${PORT:-8000}"


def _support_paths_for_spec(spec: TaskAgentServiceSpec) -> list[str]:
    paths = ["starter_manifest.json", "checks/run_visible_checks.py"]
    paths.extend(
        source.workspace_path
        for source in spec.runtime_dependencies.data_sources
        if source.learner_visible and source.workspace_path
    )
    return _dedupe(paths, limit=5)


def ensure_task_agent_deliverable_briefs(
    spec: TaskAgentServiceSpec,
    *,
    overwrite: bool = False,
) -> TaskAgentServiceSpec:
    for deliverable in spec.deliverables:
        derived_starter_surface = build_task_agent_deliverable_starter_surface(spec, deliverable)
        deliverable.learner_starter_surface = _merge_learner_starter_surface(
            derived=derived_starter_surface,
            authored=deliverable.learner_starter_surface,
        )
        if overwrite or deliverable.learner_brief is None:
            deliverable.learner_brief = build_task_agent_deliverable_brief(spec, deliverable)
        if overwrite or not deliverable.public_checks:
            deliverable.public_checks = build_task_agent_public_checks(spec, deliverable)
        if overwrite or not deliverable.learning_outcomes:
            deliverable.learning_outcomes = build_task_agent_deliverable_learning_outcomes(spec, deliverable)
    apply_inferred_review_area_case_tags(spec)
    return spec


def build_task_agent_deliverable_starter_surface(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
) -> LearnerStarterSurfaceSpec:
    editable_paths = _editable_files_for_spec(spec)
    support_paths = _support_paths_for_spec(spec)
    required_endpoints = [
        endpoint.model_copy(deep=True)
        for endpoint in spec.production_contract.canonical_endpoints
        if endpoint.required
    ]
    scenarios = [
        _starter_scenario_from_case(spec, case)
        for case in _select_public_check_cases(spec, deliverable.id, limit=3)
    ]
    endpoint_labels = ", ".join(
        f"`{endpoint.method} {endpoint.path}`"
        for endpoint in required_endpoints[:3]
    )
    starter_summary = (
        f"Extend the learner-owned {spec.project_contract.system_kind.lower()} surface so it can "
        f"{deliverable.objective.rstrip('.').lower()}."
    )
    implementation_checklist = _dedupe(
        [
            (
                f"Keep the published endpoints stable: {endpoint_labels}."
                if endpoint_labels
                else "Keep the published application contract stable while you implement this deliverable."
            ),
            (
                f"Make the learner-owned files `{', '.join(editable_paths)}` the source of truth for the core behavior."
                if editable_paths
                else "Keep the learner-owned application files as the source of truth for the core behavior."
            ),
            (
                "Handle the visible domain scenarios with explicit branching and traceable decisions."
                if scenarios
                else "Handle the visible scenarios with explicit branching and traceable decisions."
            ),
            (
                "Preserve approval, trace, and dry-run behavior where the production contract requires them."
                if spec.production_contract.supports_dry_run
                or spec.production_contract.supports_resume
                or spec.capabilities.approval_flow_required
                else "Keep the public response and health endpoints working while you deepen the internal behavior."
            ),
            (
                f"Use support files like `{support_paths[0]}` to ground the implementation instead of inventing new contracts."
                if support_paths
                else "Use the provided support files and visible checks to stay aligned with the published contract."
            ),
        ],
        limit=5,
    )
    return LearnerStarterSurfaceSpec(
        starter_summary=starter_summary,
        primary_editable_paths=editable_paths,
        support_paths=support_paths,
        required_endpoints=required_endpoints,
        implementation_checklist=implementation_checklist,
        domain_scenarios=scenarios,
    )


def _merge_learner_starter_surface(
    *,
    derived: LearnerStarterSurfaceSpec,
    authored: LearnerStarterSurfaceSpec | None,
) -> LearnerStarterSurfaceSpec:
    if authored is None:
        return derived
    authored_domain_scenarios = [
        scenario for scenario in authored.domain_scenarios if not _starter_scenario_looks_placeholder(scenario)
    ]
    return LearnerStarterSurfaceSpec(
        starter_summary=authored.starter_summary or derived.starter_summary,
        primary_editable_paths=authored.primary_editable_paths or derived.primary_editable_paths,
        support_paths=_dedupe([*authored.support_paths, *derived.support_paths], limit=6),
        required_endpoints=authored.required_endpoints or derived.required_endpoints,
        implementation_checklist=_dedupe(
            [*authored.implementation_checklist, *derived.implementation_checklist],
            limit=6,
        ),
        domain_scenarios=authored_domain_scenarios or derived.domain_scenarios,
    )


def build_task_agent_deliverable_brief(spec: TaskAgentServiceSpec, deliverable: DeliverableSpec) -> LearnerDeliverableBrief:
    if spec.capabilities.is_grounded_answer_system:
        return build_grounded_rag_deliverable_brief(spec, deliverable)

    deliverable_index = spec.deliverable_order[deliverable.id] + 1
    task_inputs = _schema_fields(spec.task_schema)
    task_outputs = _schema_fields(spec.output_schema)
    starter_surface = deliverable.learner_starter_surface or build_task_agent_deliverable_starter_surface(spec, deliverable)
    endpoint_labels = ", ".join(
        f"`{endpoint.method} {endpoint.path}`"
        for endpoint in starter_surface.required_endpoints[:3]
    )
    scenario_lines = [
        f"{scenario.title}: {scenario.request_summary} {scenario.expected_behavior}".strip()
        for scenario in starter_surface.domain_scenarios
    ]
    files_to_edit = starter_surface.primary_editable_paths or _editable_files_for_spec(spec)
    primary_file = files_to_edit[0] if files_to_edit else _primary_editable_file(spec)
    title_lower = deliverable.title.lower()

    if deliverable_index == 1:
        why_this_deliverable_matters = (
            "This is the first learner-visible review area. Get the service returning correct, structured results "
            "before you worry about richer controls."
        )
    else:
        why_this_deliverable_matters = (
            "This deliverable builds on the working service from earlier review areas and adds one production-facing "
            "capability without changing the overall contract."
        )

    starter_summary = starter_surface.starter_summary.strip()
    if starter_summary:
        task_to_build = starter_summary
        if not task_to_build.endswith("."):
            task_to_build += "."
        task_to_build += " "
    else:
        task_to_build = (
            f"Extend the learner-owned service in `{primary_file}` so it can {deliverable.objective.rstrip('.').lower()}. "
        )
    task_to_build += (
        f"Keep the published endpoints {endpoint_labels} stable while you improve the behavior behind them."
        if endpoint_labels
        else "Keep the published request and response contract stable while you improve the behavior behind it."
    )

    definition_of_done = [
        (
            f"`/run` accepts the learner-visible input fields: {', '.join(f'`{field}`' for field in task_inputs)}."
            if task_inputs
            else "`/run` accepts the learner-visible task payload."
        ),
        (
            f"The response includes the required output fields: {', '.join(f'`{field}`' for field in task_outputs)}."
            if task_outputs
            else "The response stays within the published output schema."
        ),
        *starter_surface.implementation_checklist,
        "The service keeps `/health` working while you extend the deliverable behavior.",
    ]

    implementation_hints = [
        f"Start in `{primary_file}` and keep the primary application flow readable in learner-owned code.",
        "Make the smallest change that satisfies this deliverable instead of jumping ahead to later production features.",
        "Prefer explicit, predictable branching over magic heuristics so grader feedback is easier to interpret.",
    ]
    if starter_surface.support_paths:
        implementation_hints.append(
            "Use support files like "
            + ", ".join(f"`{path}`" for path in starter_surface.support_paths[:3])
            + " to stay aligned with the published contract."
        )

    non_goals = [
        "Do not hide core deliverable logic behind generated support code or opaque helper wrappers.",
        "Do not optimize for later deliverables before this deliverable works end to end.",
    ]

    keyword_map: list[tuple[str, list[str], list[str], list[str]]] = [
        (
            "structured output",
            [
                "Return a valid decision object for every request instead of a placeholder string or partial payload.",
                "Handle the happy-path visible cases with real values, not TODO responses.",
            ],
            [
                "Map the incoming request into a single response object and validate that each required field is present.",
                "Keep the handler small and deterministic; deliverable 1 is about contract correctness, not sophistication.",
            ],
            [
                "You do not need tool calls, durable state, or approval gates yet.",
            ],
        ),
        (
            "tool selection",
            [
                "Use the right runtime tools before producing the final result.",
                "Pass the correct arguments into each tool instead of hard-coding outputs.",
            ],
            [
                "Make tool usage explicit in the code so you can trace why each lookup happened.",
                "Treat tool errors as normal control-flow inputs rather than as reasons to bluff a final answer.",
            ],
            [
                "You do not need the final latency or quality bar yet.",
            ],
        ),
        (
            "multi-step",
            [
                "Persist enough state to resume a run after a pause or approval boundary.",
                "Make sure the next step can pick up the prior context instead of starting from scratch.",
            ],
            [
                "Store the minimum state needed to reconstruct the run safely.",
                "Keep the resumed path and first-run path aligned so they return the same final shape.",
            ],
            [
                "You do not need final production polish yet; focus on correct state transitions.",
            ],
        ),
        (
            "escalation",
            [
                "Escalate ambiguous or risky cases instead of bluffing.",
                "Require approval before irreversible reply actions when the workflow calls for it.",
            ],
            [
                "Write the escalation logic in one obvious place so the grader can observe it consistently.",
                "Use confidence and risk signals to decide when a human needs to step in.",
            ],
            [
                "Do not try to hide escalation by returning a fake confident answer.",
            ],
        ),
        (
            "fallback",
            [
                "Recover cleanly from tool failures and preserve safe behavior in `dry_run` mode.",
                "Keep write paths idempotent so repeated submissions do not create duplicate side effects.",
            ],
            [
                "Handle timeouts and missing tool responses explicitly.",
                "Make dry-run behavior obvious in the code rather than relying on comments or intent.",
            ],
            [
                "Do not assume tools always succeed or that retries are invisible to callers.",
            ],
        ),
        (
            "observability",
            [
                "Expose enough trace information to explain what the agent did and why.",
                "Make the trace complete enough to debug a failing run after submission.",
            ],
            [
                "Capture a small number of meaningful events instead of logging everything.",
                "Keep trace payloads readable and aligned with actual execution steps.",
            ],
            [
                "Do not add noisy telemetry that obscures the important decision points.",
            ],
        ),
        (
            "eval-driven",
            [
                "Use the frozen evaluation set to improve behavior instead of guessing what the grader wants.",
                "Tighten the agent until it handles the supported scenarios consistently.",
            ],
            [
                "Treat failing eval scenarios as concrete debugging inputs and adjust behavior deliberately.",
                "Keep changes reversible so you can compare before and after behavior.",
            ],
            [
                "Do not overfit to one example at the cost of the rest of the visible scenarios.",
            ],
        ),
        (
            "production final",
            [
                "Get the whole deliverable working together under the final success, latency, and quality bar.",
                "Ship a solution that feels production-ready, not just barely correct on one path.",
            ],
            [
                "Use the earlier deliverables as building blocks; this is mostly about integration and polish.",
                "Keep the public contract stable while improving quality across the visible scenarios.",
            ],
            [
                "Do not throw away earlier deliverable behavior to chase a single metric.",
            ],
        ),
    ]

    for keyword, extra_done, extra_hints, extra_non_goals in keyword_map:
        if keyword in title_lower:
            definition_of_done.extend(extra_done)
            implementation_hints.extend(extra_hints)
            non_goals.extend(extra_non_goals)

    return LearnerDeliverableBrief(
        why_this_deliverable_matters=why_this_deliverable_matters,
        task_to_build=task_to_build,
        files_to_edit=files_to_edit,
        definition_of_done=_dedupe(definition_of_done, limit=5),
        example_scenarios=_dedupe(scenario_lines or _example_scenarios(spec.eval_dataset.cases), limit=3),
        implementation_hints=_dedupe(implementation_hints, limit=4),
        non_goals=_dedupe(non_goals, limit=3),
    )


def build_grounded_rag_deliverable_brief(spec: TaskAgentServiceSpec, deliverable: DeliverableSpec) -> LearnerDeliverableBrief:
    deliverable_index = spec.deliverable_order[deliverable.id] + 1
    task_inputs = _schema_fields(spec.task_schema)
    task_outputs = _schema_fields(spec.output_schema)
    examples = _example_scenarios(spec.eval_dataset.cases)
    title_lower = deliverable.title.lower()
    primary_file = _primary_editable_file(spec)
    visible_sources = _learner_visible_data_sources(spec)
    primary_source = visible_sources[0] if visible_sources else None
    source_title = primary_source.title if primary_source is not None else "the visible corpus"
    source_path = primary_source.workspace_path if primary_source is not None else "data/corpus.json"

    if deliverable_index == 1:
        why_this_deliverable_matters = (
            "This first review area gets the learner-visible question answering contract into a usable state. "
            "Start by returning grounded answers with clear citation output before you optimize retrieval depth."
        )
    else:
        why_this_deliverable_matters = (
            "This deliverable extends the same grounded QA service and raises the bar on retrieval quality, abstention, "
            "or production readiness without changing the public learner-facing contract."
        )

    task_to_build = (
        f"Edit `{primary_file}` to answer questions from `{source_path}` and return a grounded response through "
        "the public `/run` endpoint. Keep the response schema stable, cite supporting documents, and abstain when "
        f"{source_title} does not support a confident answer."
    )

    definition_of_done = [
        (
            f"`/run` accepts the learner-visible input fields: {', '.join(f'`{field}`' for field in task_inputs)}."
            if task_inputs
            else "`/run` accepts the learner-visible question payload."
        ),
        (
            f"The response includes the required output fields: {', '.join(f'`{field}`' for field in task_outputs)}."
            if task_outputs
            else "The response stays within the published output schema."
        ),
        "Supported questions return an answer plus the document ids that justify it.",
        "Unsupported questions abstain instead of fabricating a citation.",
        "The service keeps `/health` working while you improve the grounded answer behavior.",
    ]

    implementation_hints = [
        f"Start in `{primary_file}` and keep the primary request path readable in learner-owned code.",
        f"Use `{source_path}` as the learner-visible data source instead of calling external services.",
        "Keep citation ids stable and learner-readable so the grader can verify what evidence you used.",
        f"Prefer explicit abstention rules over guessing when {source_title} does not support the question.",
    ]

    non_goals = [
        "Do not add network calls or external vector databases in this learner starter.",
        f"Do not fabricate citations or answer beyond what {source_title} supports.",
        "Do not redesign the project structure before this deliverable works end to end.",
    ]

    keyword_map: list[tuple[str, list[str], list[str], list[str]]] = [
        (
            "citation",
            [
                "Return the supporting document ids in `citations` whenever the answer is grounded.",
                "Keep abstained answers paired with an empty citation list.",
            ],
            [
                "Build the answer from the cited passages so the output stays easy to audit.",
            ],
            [
                "Do not return unsupported citations just to satisfy the schema.",
            ],
        ),
        (
            "retrieval",
            [
                f"Search {source_title} before composing the answer.",
                "Pick the passages that best support the final answer instead of dumping every match.",
            ],
            [
                f"Normalize the incoming query before comparing it with `{source_path}`.",
                "Keep retrieval logic deterministic so grader failures are easier to debug.",
            ],
            [
                "Do not short-circuit retrieval with a generic placeholder answer.",
            ],
        ),
        (
            "abstention",
            [
                f"When {source_title} lacks support, return an abstained answer instead of guessing.",
            ],
            [
                "Treat missing support as a normal outcome and make that branch obvious in the code.",
            ],
            [
                "Do not bluff a confident answer when support is weak or absent.",
            ],
        ),
        (
            "trace",
            [
                "Emit enough trace information to show which retrieval steps led to the answer.",
            ],
            [
                "Capture the important retrieval and answer events rather than every intermediate detail.",
            ],
            [
                "Do not hide the retrieval path behind opaque helper output.",
            ],
        ),
        (
            "eval-driven",
            [
                "Use the visible scenarios to tighten groundedness and abstention behavior across the deliverable.",
            ],
            [
                "Work from failing examples one by one instead of changing the whole system at once.",
            ],
            [
                "Do not overfit one question at the expense of the rest of the visible cases.",
            ],
        ),
        (
            "production final",
            [
                "Get the full grounded QA flow working together under the final latency and cost bar.",
            ],
            [
                "Use the earlier deliverables as building blocks and focus on integration quality.",
            ],
            [
                "Do not regress groundedness to chase speed alone.",
            ],
        ),
    ]

    for keyword, extra_done, extra_hints, extra_non_goals in keyword_map:
        if keyword in title_lower:
            definition_of_done.extend(extra_done)
            implementation_hints.extend(extra_hints)
            non_goals.extend(extra_non_goals)

    return LearnerDeliverableBrief(
        why_this_deliverable_matters=why_this_deliverable_matters,
        task_to_build=task_to_build,
        files_to_edit=_editable_files_for_spec(spec),
        definition_of_done=_dedupe(definition_of_done, limit=6),
        example_scenarios=_dedupe(examples, limit=3),
        implementation_hints=_dedupe(implementation_hints, limit=5),
        non_goals=_dedupe(non_goals, limit=4),
    )


def _learner_visible_data_sources(spec: TaskAgentServiceSpec):
    return [
        source
        for source in spec.runtime_dependencies.data_sources
        if source.learner_visible and source.workspace_path
    ]


def build_task_agent_deliverable_learning_outcomes(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
) -> list[str]:
    brief = deliverable.learner_brief or build_task_agent_deliverable_brief(spec, deliverable)
    public_checks = deliverable.public_checks or build_task_agent_public_checks(spec, deliverable)
    title_lower = deliverable.title.lower()

    outcomes: list[str] = []
    if "structured output" in title_lower or "run contract" in title_lower:
        outcomes.extend(
            [
                "Implement a stable `/run` contract with the required structured response fields.",
                "Return learner-visible results that match the published output schema on each supported case.",
            ]
        )
    elif "tool" in title_lower:
        outcomes.extend(
            [
                "Choose and invoke the right tools for each supported workflow path.",
                "Keep the run contract stable while tool usage becomes observable and debuggable.",
            ]
        )
    elif "retrieval" in title_lower or "citation" in title_lower or "grounded" in title_lower:
        outcomes.extend(
            [
                "Use the learner-visible data source to return relevant evidence for each request.",
                "Keep grounded answers faithful to retrieved evidence instead of guessing.",
            ]
        )
    elif "cache" in title_lower:
        outcomes.extend(
            [
                "Use the configured cache to improve read performance without breaking correctness.",
                "Explain the freshness and invalidation tradeoffs introduced by the cache layer.",
            ]
        )
    elif "lock" in title_lower or "concurrency" in title_lower or "retry" in title_lower:
        outcomes.extend(
            [
                "Protect the critical workflow path under concurrent requests and duplicate submissions.",
                "Use locking, retries, or version-aware writes to keep core invariants intact.",
            ]
        )
    elif "observability" in title_lower or "trace" in title_lower:
        outcomes.extend(
            [
                "Make the deliverable observable enough to explain what happened during a run.",
                "Emit traces or diagnostics that turn failing scenarios into debuggable evidence.",
            ]
        )

    outcomes.extend(_outcomes_from_definition_of_done(brief.definition_of_done))
    outcomes.extend(check.learner_goal for check in public_checks[:2])

    if not outcomes:
        outcomes.extend(
            [
                f"Build the deliverable so it can {deliverable.objective.strip().rstrip('.').lower()}.",
                "Use the learner-visible checks to prove the deliverable behaves correctly before submission.",
            ]
        )

    return _dedupe(outcomes, limit=4)


def build_task_agent_public_checks(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
    *,
    limit: int = 3,
) -> list[PublicCheckSpec]:
    selected_cases = _select_public_check_cases(spec, deliverable.id, limit=limit)
    checks: list[PublicCheckSpec] = []
    for index, case in enumerate(selected_cases, start=1):
        coverage = _public_check_coverage(spec, deliverable.id, case.id)
        assertions = _public_check_assertions(case)
        checks.append(
            PublicCheckSpec(
                id=f"{deliverable.id}_public_check_{index}",
                title=_public_check_title(case),
                learner_goal=_public_check_goal(case),
                case_id=case.id,
                files_to_use=[_primary_editable_file(spec)],
                expected_assertions=assertions,
                covers_behavior_ids=coverage["behaviors"],
                covers_quality_ids=coverage["qualities"],
            )
        )
    return checks


def combine_learner_deliverable_briefs(
    *,
    fallback_task: str,
    fallback_why: str,
    fallback_files_to_edit: list[str] | None = None,
    briefs: Iterable[LearnerDeliverableBrief],
) -> LearnerDeliverableBrief:
    brief_list = list(briefs)
    if not brief_list:
        return LearnerDeliverableBrief(
            why_this_deliverable_matters=fallback_why,
            task_to_build=fallback_task,
            files_to_edit=fallback_files_to_edit or ["app.py"],
            definition_of_done=["Implement the deliverable goal in the learner-visible workspace and submit it for grading."],
            example_scenarios=[],
            implementation_hints=[
                "Start with the learner-visible starter files before making broader refactors.",
            ],
            non_goals=[],
        )

    first = brief_list[0]
    return LearnerDeliverableBrief(
        why_this_deliverable_matters=first.why_this_deliverable_matters,
        task_to_build=first.task_to_build or fallback_task,
        files_to_edit=_dedupe(
            (item for brief in brief_list for item in brief.files_to_edit)
        ) or (fallback_files_to_edit or ["app.py"]),
        definition_of_done=_dedupe(
            (item for brief in brief_list for item in brief.definition_of_done),
            limit=6,
        ),
        example_scenarios=_dedupe(
            (item for brief in brief_list for item in brief.example_scenarios),
            limit=4,
        ),
        implementation_hints=_dedupe(
            (item for brief in brief_list for item in brief.implementation_hints),
            limit=5,
        ),
        non_goals=_dedupe(
            (item for brief in brief_list for item in brief.non_goals),
            limit=4,
        ),
    )


def combine_public_checks(
    checks: Iterable[PublicCheckSpec],
    *,
    limit: int = 4,
) -> list[PublicCheckSpec]:
    combined: list[PublicCheckSpec] = []
    seen: set[str] = set()
    for check in checks:
        normalized_id = check.id.strip()
        if not normalized_id or normalized_id in seen:
            continue
        combined.append(check)
        seen.add(normalized_id)
        if len(combined) >= limit:
            break
    return combined


def render_learner_deliverable_markdown(
    *,
    deliverable_index: int,
    title: str,
    summary: str,
    learning_outcomes: list[str],
    brief: LearnerDeliverableBrief,
    public_checks: list[PublicCheckSpec] | None = None,
) -> str:
    lines = [
        f"# Deliverable {deliverable_index}: {title}",
        "",
        "## Why this deliverable matters",
        "",
        brief.why_this_deliverable_matters,
        "",
        "## What to build",
        "",
        brief.task_to_build,
        "",
    ]
    if learning_outcomes:
        lines.extend(["## What you will practice", ""])
        lines.extend(f"- {outcome}" for outcome in learning_outcomes)
        lines.append("")
    if brief.files_to_edit:
        lines.extend(["## Files to edit", ""])
        lines.extend(f"- `{path}`" for path in brief.files_to_edit)
        lines.append("")
    if brief.example_scenarios:
        lines.extend(["## Example scenarios", ""])
        lines.extend(f"- {item}" for item in brief.example_scenarios)
        lines.append("")
    if brief.definition_of_done:
        lines.extend(["## Definition of done", ""])
        lines.extend(f"- {item}" for item in brief.definition_of_done)
        lines.append("")
    if brief.implementation_hints:
        lines.extend(["## Implementation hints", ""])
        lines.extend(f"- {item}" for item in brief.implementation_hints)
        lines.append("")
    public_check_items = public_checks or []
    if public_check_items:
        lines.extend(["## Visible checks you can run", ""])
        for check in public_check_items:
            lines.append(f"### {check.title}")
            lines.append("")
            lines.append(check.learner_goal)
            lines.append("")
            if check.expected_assertions:
                lines.extend(f"- {item}" for item in check.expected_assertions)
                lines.append("")
    if brief.non_goals:
        lines.extend(["## Not in scope yet", ""])
        lines.extend(f"- {item}" for item in brief.non_goals)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_learner_starter_readme(
    *,
    title: str,
    brief: LearnerDeliverableBrief,
    visible_check_command: str = "python checks/run_visible_checks.py",
    preview_command: str = "python -m uvicorn app:app --host 127.0.0.1 --port ${PORT:-8000}",
    public_checks: list[PublicCheckSpec] | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        brief.task_to_build,
        "",
    ]
    if brief.files_to_edit:
        lines.extend(["## Start here", ""])
        lines.extend(f"- Edit `{path}`" for path in brief.files_to_edit)
        lines.append("- Use `deliverable_content.md` as your deliverable brief.")
        lines.append("")
    if brief.definition_of_done:
        lines.extend(["## Done looks like", ""])
        lines.extend(f"- {item}" for item in brief.definition_of_done[:4])
        lines.append("")
    public_check_items = public_checks or []
    if public_check_items:
        lines.extend(["## Visible checks", ""])
        for check in public_check_items:
            lines.append(f"- `{check.title}`: {check.learner_goal}")
        lines.append("")
    lines.extend(
        [
            "## Helpful commands",
            "",
            f"- Run visible checks: `{visible_check_command}`",
            f"- Start a local preview: `{preview_command}`",
            "- In VS Code, open **Run Task** and choose `Run visible checks` or `Start local preview`.",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _schema_fields(schema: dict) -> list[str]:
    required = schema.get("required")
    if isinstance(required, list) and required:
        return [str(item) for item in required]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return [str(key) for key in properties.keys()]
    return []


def _example_scenarios(cases: Iterable[TaskEvalCase]) -> list[str]:
    examples: list[str] = []
    for case in list(cases)[:3]:
        label = case.title or case.id
        input_preview = _preview_mapping(case.input)
        if case.expected_output:
            output_preview = _preview_mapping(case.expected_output)
            examples.append(
                f"`{label}`: when the request looks like {input_preview}, respond in a way that reflects {output_preview}."
            )
        elif case.should_escalate:
            examples.append(
                f"`{label}`: when the request looks like {input_preview}, escalate instead of pretending the agent can safely finish it."
            )
        else:
            examples.append(f"`{label}`: handle a request shaped like {input_preview}.")
    return examples


def _starter_scenario_from_case(
    spec: TaskAgentServiceSpec,
    case: TaskEvalCase,
) -> StarterScenarioSpec:
    request_summary = _starter_request_summary(spec, case)
    expected_behavior = _starter_expected_behavior(case)
    return StarterScenarioSpec(
        id=case.id,
        title=case.title or _humanize_identifier(case.id),
        request_summary=request_summary,
        expected_behavior=expected_behavior,
    )


def _starter_request_summary(spec: TaskAgentServiceSpec, case: TaskEvalCase) -> str:
    payload = case.input or {}
    if "customer_message" in payload:
        issue_type = payload.get("issue_type")
        tier = payload.get("account_tier")
        message = str(payload.get("customer_message", "")).strip()
        parts = []
        if issue_type:
            parts.append(f"a `{issue_type}` support ticket")
        else:
            parts.append("a customer support ticket")
        if tier:
            parts.append(f"from a `{tier}` account")
        if message:
            parts.append(f"where the customer says: {message!r}")
        return "Handle " + " ".join(parts) + "."
    if "question" in payload:
        return f"Handle a question like {payload.get('question')!r}."
    if "task_input" in payload:
        return f"Handle a workflow request like {payload.get('task_input')!r}."
    return f"Handle a request shaped like {_preview_mapping(payload)}."


def _starter_expected_behavior(case: TaskEvalCase) -> str:
    if case.should_escalate and case.requires_approval:
        return "Route it through the safe approval path instead of auto-completing it."
    if case.should_escalate:
        return "Escalate or hand it off instead of pretending the system can finish it safely."
    if case.requires_approval:
        return "Pause for approval before carrying out the risky or irreversible step."
    if case.expected_output:
        return f"Make the response reflect {_preview_mapping(case.expected_output)}."
    return "Keep the published response contract stable while handling it."


def _humanize_identifier(value: str) -> str:
    cleaned = value.replace("-", " ").replace("_", " ").strip()
    if not cleaned:
        return "Scenario"
    return cleaned[:1].upper() + cleaned[1:]


def _starter_scenario_looks_placeholder(scenario: StarterScenarioSpec) -> bool:
    text = " ".join(
        [
            scenario.title.strip().lower(),
            scenario.request_summary.strip().lower(),
            scenario.expected_behavior.strip().lower(),
        ]
    )
    placeholder_markers = [
        "routine case",
        "ambiguous or risky case",
        "happy path",
        "escalation case",
        "workflow request like",
        "handle the routine case cleanly",
    ]
    return any(marker in text for marker in placeholder_markers)


def _preview_mapping(payload: dict) -> str:
    preview_parts: list[str] = []
    for key, value in list(payload.items())[:3]:
        preview_parts.append(f"{key}={value!r}")
    return "{" + ", ".join(preview_parts) + "}"


def _outcomes_from_definition_of_done(items: Iterable[str]) -> list[str]:
    outcomes: list[str] = []
    for item in items:
        cleaned = item.strip().rstrip(".")
        if not cleaned:
            continue
        if cleaned.lower().startswith("`/run`"):
            outcomes.append("Keep the public contract stable while extending the deliverable behavior.")
            continue
        if cleaned.lower().startswith("the response includes"):
            outcomes.append("Return responses that satisfy the learner-visible contract consistently.")
            continue
        if cleaned.lower().startswith("the service keeps"):
            outcomes.append("Preserve the working service surface while improving the deliverable internals.")
            continue
        outcomes.append(cleaned)
    return outcomes


def _dedupe(items: Iterable[str], *, limit: int | None = None) -> list[str]:
    seen: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.append(normalized)
        if limit is not None and len(seen) >= limit:
            break
    return seen


def _select_public_check_cases(
    spec: TaskAgentServiceSpec,
    deliverable_id: str,
    *,
    limit: int = 3,
) -> list[TaskEvalCase]:
    gate = spec.gate_for(deliverable_id)
    active_ids = set(gate.active_behavior_ids + gate.active_quality_ids)
    referenced_case_ids: list[str] = []

    for behavior in spec.behaviors:
        if behavior.id in active_ids:
            referenced_case_ids.extend(_case_ids_from_test(behavior.test))
    for quality in spec.qualities:
        if quality.id in active_ids:
            referenced_case_ids.extend(_case_ids_from_test(quality.test))

    ordered_cases: list[TaskEvalCase] = []
    seen: set[str] = set()
    matching_case_ids = set(referenced_case_ids)

    for case in spec.eval_dataset.cases:
        if case.id in matching_case_ids and case.id not in seen:
            ordered_cases.append(case.model_copy(deep=True))
            seen.add(case.id)
            if len(ordered_cases) >= limit:
                return ordered_cases

    for case in spec.eval_dataset.cases:
        if case.id in seen:
            continue
        ordered_cases.append(case.model_copy(deep=True))
        seen.add(case.id)
        if len(ordered_cases) >= limit:
            break

    return ordered_cases


def _public_check_coverage(spec: TaskAgentServiceSpec, deliverable_id: str, case_id: str) -> dict[str, list[str]]:
    gate = spec.gate_for(deliverable_id)
    active_behavior_ids = set(gate.active_behavior_ids)
    active_quality_ids = set(gate.active_quality_ids)
    behavior_ids = [
        behavior.id
        for behavior in spec.behaviors
        if behavior.id in active_behavior_ids and case_id in _case_ids_from_test(behavior.test)
    ]
    quality_ids = [
        quality.id
        for quality in spec.qualities
        if quality.id in active_quality_ids and case_id in _case_ids_from_test(quality.test)
    ]
    return {
        "behaviors": sorted(set(behavior_ids)),
        "qualities": sorted(set(quality_ids)),
    }


def _public_check_title(case: TaskEvalCase) -> str:
    if case.title:
        return case.title.strip()
    words = case.id.replace("-", "_").split("_")
    normalized = " ".join(word for word in words if word)
    return normalized.capitalize() or "Visible check"


def _public_check_goal(case: TaskEvalCase) -> str:
    input_preview = _preview_mapping(case.input)
    if case.expected_output:
        output_preview = _preview_mapping(case.expected_output)
        return (
            f"Handle a learner-visible request like {input_preview} and return a response consistent with "
            f"{output_preview}."
        )
    if case.should_escalate:
        return (
            f"Handle a request like {input_preview} by escalating or asking for human review when the situation "
            "is ambiguous or risky."
        )
    return f"Handle a learner-visible request shaped like {input_preview} without breaking the published contract."


def _public_check_assertions(case: TaskEvalCase) -> list[str]:
    assertions: list[str] = []
    for key, value in (case.expected_output or {}).items():
        assertions.append(f"The response includes `{key}` with a value consistent with `{value}`.")
    if case.requires_approval:
        assertions.append("The run requests approval before any irreversible action is taken.")
    if case.should_escalate:
        assertions.append("The service escalates or returns a human-review path instead of bluffing a final answer.")
    if case.must_use_any_of_tools:
        assertions.append(
            "The trace shows the agent consulting at least one relevant tool before it answers."
        )
    return _dedupe(assertions, limit=4)


def _case_ids_from_test(test: Any) -> list[str]:
    if hasattr(test, "case_ids"):
        return [str(case_id) for case_id in getattr(test, "case_ids") or []]
    if hasattr(test, "case_id"):
        case_id = getattr(test, "case_id")
        return [str(case_id)] if case_id else []
    if hasattr(test, "expectations"):
        return [str(expectation.case_id) for expectation in getattr(test, "expectations") or []]
    if hasattr(test, "injections"):
        return [str(injection.case_id) for injection in getattr(test, "injections") or []]
    return []
