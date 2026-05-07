from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.domain.task_agent import (
    LearnerModuleBrief,
    ModuleSpec,
    PublicCheckSpec,
    TaskAgentServiceSpec,
    TaskEvalCase,
)
from app.services.review_area_coverage import apply_inferred_review_area_case_tags


def ensure_task_agent_module_briefs(
    spec: TaskAgentServiceSpec,
    *,
    overwrite: bool = False,
) -> TaskAgentServiceSpec:
    for module in spec.modules:
        if overwrite or module.learner_brief is None:
            module.learner_brief = build_task_agent_module_brief(spec, module)
        if overwrite or not module.public_checks:
            module.public_checks = build_task_agent_public_checks(spec, module)
        if overwrite or not module.learning_outcomes:
            module.learning_outcomes = build_task_agent_module_learning_outcomes(spec, module)
    apply_inferred_review_area_case_tags(spec)
    return spec


def build_task_agent_module_brief(spec: TaskAgentServiceSpec, module: ModuleSpec) -> LearnerModuleBrief:
    if spec.capabilities.is_grounded_answer_system:
        return build_grounded_rag_module_brief(spec, module)

    module_index = spec.module_order[module.id] + 1
    task_inputs = _schema_fields(spec.task_schema)
    task_outputs = _schema_fields(spec.output_schema)
    examples = _example_scenarios(spec.eval_dataset.cases)
    files_to_edit = list(spec.runtime_dependencies.editable_files or ["app.py"])
    title_lower = module.title.lower()

    if module_index == 1:
        why_this_module_matters = (
            "This is the first learner-visible review area. Get the service returning correct, structured results "
            "before you worry about richer controls."
        )
    else:
        why_this_module_matters = (
            "This module builds on the working service from earlier review areas and adds one production-facing "
            "capability without changing the overall contract."
        )

    task_to_build = (
        f"Edit `app.py` to make the service {module.objective.rstrip('.').lower()}. "
        f"Keep the public `/run` endpoint stable and return JSON that matches the published output contract."
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
        "The service keeps `/health` working while you extend the module behavior.",
    ]

    implementation_hints = [
        "Start in `app.py`; treat `runtime/` helpers and `starter_manifest.json` as read-only support files.",
        "Make the smallest change that satisfies this module instead of jumping ahead to later production features.",
        "Prefer explicit, predictable branching over magic heuristics so grader feedback is easier to interpret.",
    ]

    non_goals = [
        "Do not redesign the project structure or edit the runtime helpers unless the brief explicitly asks for it.",
        "Do not optimize for later modules before this module works end to end.",
    ]

    keyword_map: list[tuple[str, list[str], list[str], list[str]]] = [
        (
            "structured output",
            [
                "Return a valid decision object for every request instead of a placeholder string or partial payload.",
                "Handle the happy-path support cases with real values, not TODO responses.",
            ],
            [
                "Map the incoming request into a single response object and validate that each required field is present.",
                "Keep the handler small and deterministic; module 1 is about contract correctness, not sophistication.",
            ],
            [
                "You do not need tool calls, durable state, or approval gates yet.",
            ],
        ),
        (
            "tool selection",
            [
                "Use the right support tools before drafting a response.",
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
                "Get the whole module working together under the final success, latency, and quality bar.",
                "Ship a solution that feels production-ready, not just barely correct on one path.",
            ],
            [
                "Use the earlier modules as building blocks; this is mostly about integration and polish.",
                "Keep the public contract stable while improving quality across the visible scenarios.",
            ],
            [
                "Do not throw away earlier module behavior to chase a single metric.",
            ],
        ),
    ]

    for keyword, extra_done, extra_hints, extra_non_goals in keyword_map:
        if keyword in title_lower:
            definition_of_done.extend(extra_done)
            implementation_hints.extend(extra_hints)
            non_goals.extend(extra_non_goals)

    return LearnerModuleBrief(
        why_this_module_matters=why_this_module_matters,
        task_to_build=task_to_build,
        files_to_edit=files_to_edit,
        definition_of_done=_dedupe(definition_of_done, limit=5),
        example_scenarios=_dedupe(examples, limit=3),
        implementation_hints=_dedupe(implementation_hints, limit=4),
        non_goals=_dedupe(non_goals, limit=3),
    )


def build_grounded_rag_module_brief(spec: TaskAgentServiceSpec, module: ModuleSpec) -> LearnerModuleBrief:
    module_index = spec.module_order[module.id] + 1
    task_inputs = _schema_fields(spec.task_schema)
    task_outputs = _schema_fields(spec.output_schema)
    examples = _example_scenarios(spec.eval_dataset.cases)
    title_lower = module.title.lower()
    visible_sources = _learner_visible_data_sources(spec)
    primary_source = visible_sources[0] if visible_sources else None
    source_title = primary_source.title if primary_source is not None else "the visible corpus"
    source_path = primary_source.workspace_path if primary_source is not None else "data/corpus.json"

    if module_index == 1:
        why_this_module_matters = (
            "This first review area gets the learner-visible question answering contract into a usable state. "
            "Start by returning grounded answers with clear citation output before you optimize retrieval depth."
        )
    else:
        why_this_module_matters = (
            "This module extends the same grounded QA service and raises the bar on retrieval quality, abstention, "
            "or production readiness without changing the public learner-facing contract."
        )

    task_to_build = (
        f"Edit `app.py` to answer questions from `{source_path}` and return a grounded response through "
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
        "Start in `app.py`; treat `runtime/` helpers and `starter_manifest.json` as read-only support files.",
        f"Use `{source_path}` as the learner-visible data source instead of calling external services.",
        "Keep citation ids stable and learner-readable so the grader can verify what evidence you used.",
        f"Prefer explicit abstention rules over guessing when {source_title} does not support the question.",
    ]

    non_goals = [
        "Do not add network calls or external vector databases in this learner starter.",
        f"Do not fabricate citations or answer beyond what {source_title} supports.",
        "Do not redesign the project structure before this module works end to end.",
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
                "Use the visible scenarios to tighten groundedness and abstention behavior across the module.",
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
                "Use the earlier modules as building blocks and focus on integration quality.",
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

    return LearnerModuleBrief(
        why_this_module_matters=why_this_module_matters,
        task_to_build=task_to_build,
        files_to_edit=list(spec.runtime_dependencies.editable_files or ["app.py"]),
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


def build_task_agent_module_learning_outcomes(
    spec: TaskAgentServiceSpec,
    module: ModuleSpec,
) -> list[str]:
    brief = module.learner_brief or build_task_agent_module_brief(spec, module)
    public_checks = module.public_checks or build_task_agent_public_checks(spec, module)
    title_lower = module.title.lower()

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
                "Make the module observable enough to explain what happened during a run.",
                "Emit traces or diagnostics that turn failing scenarios into debuggable evidence.",
            ]
        )

    outcomes.extend(_outcomes_from_definition_of_done(brief.definition_of_done))
    outcomes.extend(check.learner_goal for check in public_checks[:2])

    if not outcomes:
        outcomes.extend(
            [
                f"Build the module so it can {module.objective.strip().rstrip('.').lower()}.",
                "Use the learner-visible checks to prove the module behaves correctly before submission.",
            ]
        )

    return _dedupe(outcomes, limit=4)


def build_task_agent_public_checks(
    spec: TaskAgentServiceSpec,
    module: ModuleSpec,
    *,
    limit: int = 3,
) -> list[PublicCheckSpec]:
    selected_cases = _select_public_check_cases(spec, module.id, limit=limit)
    checks: list[PublicCheckSpec] = []
    for index, case in enumerate(selected_cases, start=1):
        coverage = _public_check_coverage(spec, module.id, case.id)
        assertions = _public_check_assertions(case)
        checks.append(
            PublicCheckSpec(
                id=f"{module.id}_public_check_{index}",
                title=_public_check_title(case),
                learner_goal=_public_check_goal(case),
                case_id=case.id,
                files_to_use=["app.py"],
                expected_assertions=assertions,
                covers_behavior_ids=coverage["behaviors"],
                covers_quality_ids=coverage["qualities"],
            )
        )
    return checks


def combine_learner_module_briefs(
    *,
    fallback_task: str,
    fallback_why: str,
    fallback_files_to_edit: list[str] | None = None,
    briefs: Iterable[LearnerModuleBrief],
) -> LearnerModuleBrief:
    brief_list = list(briefs)
    if not brief_list:
        return LearnerModuleBrief(
            why_this_module_matters=fallback_why,
            task_to_build=fallback_task,
            files_to_edit=fallback_files_to_edit or ["app.py"],
            definition_of_done=["Implement the module goal in the learner-visible workspace and submit it for grading."],
            example_scenarios=[],
            implementation_hints=[
                "Start with the learner-visible starter files before making broader refactors.",
            ],
            non_goals=[],
        )

    first = brief_list[0]
    return LearnerModuleBrief(
        why_this_module_matters=first.why_this_module_matters,
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


def render_learner_module_markdown(
    *,
    module_index: int,
    title: str,
    summary: str,
    learning_outcomes: list[str],
    brief: LearnerModuleBrief,
    public_checks: list[PublicCheckSpec] | None = None,
) -> str:
    lines = [
        f"# Module {module_index}: {title}",
        "",
        "## Why this module matters",
        "",
        brief.why_this_module_matters,
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
    brief: LearnerModuleBrief,
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
        lines.append("- Use `module_content.md` as your module brief.")
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
            "- Run visible checks: `python checks/run_visible_checks.py`",
            "- Start a local preview: `python -m uvicorn app:app --host 127.0.0.1 --port 8000`",
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
        input_preview = _preview_mapping(case.input)
        if case.expected_output:
            output_preview = _preview_mapping(case.expected_output)
            examples.append(
                f"`{case.id}`: when the request looks like {input_preview}, respond in a way that reflects {output_preview}."
            )
        elif case.should_escalate:
            examples.append(
                f"`{case.id}`: when the request looks like {input_preview}, escalate instead of pretending the agent can safely finish it."
            )
        else:
            examples.append(f"`{case.id}`: handle a request shaped like {input_preview}.")
    return examples


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
            outcomes.append("Keep the public contract stable while extending the module behavior.")
            continue
        if cleaned.lower().startswith("the response includes"):
            outcomes.append("Return responses that satisfy the learner-visible contract consistently.")
            continue
        if cleaned.lower().startswith("the service keeps"):
            outcomes.append("Preserve the working service surface while improving the module internals.")
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
    module_id: str,
    *,
    limit: int = 3,
) -> list[TaskEvalCase]:
    gate = spec.gate_for(module_id)
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


def _public_check_coverage(spec: TaskAgentServiceSpec, module_id: str, case_id: str) -> dict[str, list[str]]:
    gate = spec.gate_for(module_id)
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
