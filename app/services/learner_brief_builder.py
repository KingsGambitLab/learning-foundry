from __future__ import annotations

from collections.abc import Iterable

from app.domain.task_agent import (
    EndpointSpec,
    LearnerDeliverableBrief,
    LearnerStarterSurfaceSpec,
    DeliverableSpec,
    PublicCheckSpec,
    StarterScenarioSpec,
    TaskAgentServiceSpec,
)
from app.services.task_agent_contract_surface import (
    is_approval_path,
    is_eval_path,
    is_health_path,
    learner_editable_paths_for_spec,
    primary_submit_endpoint_for_spec,
)
from app.services.public_surface_quality import meaningful_domain_entities, pluralize_phrase, starter_surface_markers
from app.services.task_agent_starter_templates import (
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    default_preview_command,
)


def _editable_files_for_spec(spec: TaskAgentServiceSpec) -> list[str]:
    return learner_editable_paths_for_spec(spec)


def _primary_editable_file(spec: TaskAgentServiceSpec) -> str | None:
    files = _editable_files_for_spec(spec)
    return files[0] if files else None


def _visible_check_command(spec: TaskAgentServiceSpec) -> str:
    return spec.runtime_dependencies.visible_check_command or f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"


def _preview_command(spec: TaskAgentServiceSpec) -> str:
    return spec.runtime_dependencies.preview_command or default_preview_command(spec, host="127.0.0.1")


def _endpoint_identity(endpoint: EndpointSpec) -> tuple[str, str]:
    return endpoint.method.upper(), endpoint.path.strip()


def _required_endpoints_for_spec(spec: TaskAgentServiceSpec) -> list[EndpointSpec]:
    return [endpoint.model_copy(deep=True) for endpoint in spec.public_endpoints if endpoint.required]


def _normalize_required_endpoints(
    *,
    spec: TaskAgentServiceSpec,
    authored: Iterable[EndpointSpec],
    derived: Iterable[EndpointSpec],
) -> list[EndpointSpec]:
    published = {
        _endpoint_identity(endpoint): endpoint
        for endpoint in _required_endpoints_for_spec(spec)
    }
    normalized: list[EndpointSpec] = []
    seen: set[tuple[str, str]] = set()

    for endpoint in authored:
        identity = _endpoint_identity(endpoint)
        published_endpoint = published.get(identity)
        if published_endpoint is None or identity in seen:
            continue
        normalized.append(published_endpoint.model_copy(deep=True))
        seen.add(identity)

    if any(
        not is_health_path(endpoint.path)
        and not is_eval_path(endpoint.path)
        and not is_approval_path(endpoint.path)
        for endpoint in normalized
    ):
        return normalized

    for endpoint in derived:
        identity = _endpoint_identity(endpoint)
        if identity in seen:
            continue
        normalized.append(endpoint.model_copy(deep=True))
        seen.add(identity)

    return normalized


def _primary_request_surface(starter_surface: LearnerStarterSurfaceSpec) -> str:
    request_endpoints = [
        endpoint
        for endpoint in starter_surface.required_endpoints
        if not is_health_path(endpoint.path)
        and not is_eval_path(endpoint.path)
        and not is_approval_path(endpoint.path)
    ]
    selected = request_endpoints[:2] or starter_surface.required_endpoints[:2]
    if not selected:
        return "the published request surface"
    return ", ".join(f"`{endpoint.method} {endpoint.path}`" for endpoint in selected)


def _support_paths_for_spec(spec: TaskAgentServiceSpec) -> list[str]:
    # The visible-check script's location relative to the README depends on
    # the workspace layout. For shared-codebase courses the README sits at
    # `public/checks/<id>/README.md` next to `run_visible_checks.py` (same
    # directory). For legacy non-shared courses the README sits at
    # `public/starter/<id>/README.md` with the script under `checks/`.
    # The reviewer enforces that any path the README references actually
    # exists relative to the README, so this must match the on-disk layout.
    if spec.course_structure.shared_codebase:
        paths = ["run_visible_checks.py"]
    else:
        paths = ["checks/run_visible_checks.py"]
    paths.extend(
        source.workspace_path
        for source in spec.runtime_dependencies.data_sources
        if source.learner_visible and source.workspace_path
    )
    return _dedupe(paths, limit=5)


def _primary_domain_entity(spec: TaskAgentServiceSpec) -> str:
    entities = meaningful_domain_entities(spec.project_contract.core_entities)
    return entities[0] if entities else spec.project_contract.system_kind.lower()


def _resolved_starter_surface(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
) -> LearnerStarterSurfaceSpec:
    derived_starter_surface = build_task_agent_deliverable_starter_surface(spec, deliverable)
    return _merge_learner_starter_surface(
        spec=spec,
        derived=derived_starter_surface,
        authored=deliverable.learner_starter_surface,
    )


def ensure_task_agent_deliverable_briefs(
    spec: TaskAgentServiceSpec,
    *,
    overwrite: bool = False,
) -> TaskAgentServiceSpec:
    for deliverable in spec.deliverables:
        deliverable.learner_starter_surface = _resolved_starter_surface(spec, deliverable)
        if overwrite or deliverable.learner_brief is None:
            deliverable.learner_brief = build_task_agent_deliverable_brief(spec, deliverable)
        if overwrite or not deliverable.public_checks:
            deliverable.public_checks = build_task_agent_public_checks(spec, deliverable)
        if overwrite or not deliverable.learning_outcomes:
            deliverable.learning_outcomes = build_task_agent_deliverable_learning_outcomes(spec, deliverable)
    return spec


def build_task_agent_deliverable_starter_surface(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
) -> LearnerStarterSurfaceSpec:
    editable_paths = _editable_files_for_spec(spec)
    support_paths = _support_paths_for_spec(spec)
    required_endpoints = _required_endpoints_for_spec(spec)
    scenarios = _derive_scenarios(spec, deliverable)
    endpoint_labels = ", ".join(
        f"`{endpoint.method} {endpoint.path}`" for endpoint in required_endpoints[:3]
    )
    primary_entity = _primary_domain_entity(spec)
    starter_summary = (
        f"Extend the learner-owned {primary_entity} service so it can "
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
                else "Make the learner-owned repo files the source of truth for the core behavior."
            ),
            "Handle the visible scenarios with explicit, readable branching.",
            "Keep `/health` working while you deepen the real service behavior.",
            (
                "Use the visible support files and checks instead of inventing a second hidden contract."
                if support_paths
                else "Use the visible checks to stay aligned with the published contract."
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


def _derive_scenarios(spec: TaskAgentServiceSpec, deliverable: DeliverableSpec) -> list[StarterScenarioSpec]:
    contract = spec.project_contract
    primary_entity = _primary_domain_entity(spec)
    primary_entity_plural = pluralize_phrase(primary_entity)
    read_focus = contract.primary_read_paths[0] if contract.primary_read_paths else f"handle the main {contract.system_kind.lower()} request"
    write_focus = contract.primary_write_paths[0] if contract.primary_write_paths else deliverable.objective
    invariant = contract.invariants[0] if contract.invariants else "preserve the published contract"
    concern = contract.operational_concerns[0] if contract.operational_concerns else "make failures easy to debug"
    submit_endpoint = primary_submit_endpoint_for_spec(spec)
    primary_title = (
        f"Create or update {primary_entity_plural}"
        if submit_endpoint is not None and submit_endpoint.method in {"POST", "PUT", "PATCH"}
        else f"Inspect {primary_entity_plural}"
    )
    edge_title = f"Recover {primary_entity_plural} without breaking invariants"
    return [
        StarterScenarioSpec(
            id=f"{deliverable.id}_primary",
            title=primary_title,
            request_summary=read_focus.rstrip(".") + ".",
            expected_behavior=f"Return a coherent response while moving the deliverable toward: {deliverable.objective.rstrip('.')}.",
        ),
        StarterScenarioSpec(
            id=f"{deliverable.id}_edge",
            title=edge_title,
            request_summary=write_focus.rstrip(".") + ".",
            expected_behavior=f"Handle the path without violating the invariant that {invariant.rstrip('.')} and keep the result observable enough to {concern.rstrip('.')}.",
        ),
    ]


def _merge_learner_starter_surface(
    *,
    spec: TaskAgentServiceSpec,
    derived: LearnerStarterSurfaceSpec,
    authored: LearnerStarterSurfaceSpec | None,
) -> LearnerStarterSurfaceSpec:
    if authored is None:
        return derived
    authored_domain_scenarios = [
        scenario
        for scenario in authored.domain_scenarios
        if not _starter_scenario_looks_placeholder(scenario)
    ]
    return LearnerStarterSurfaceSpec(
        starter_summary=authored.starter_summary or derived.starter_summary,
        primary_editable_paths=authored.primary_editable_paths or derived.primary_editable_paths,
        support_paths=_dedupe([*authored.support_paths, *derived.support_paths], limit=6),
        required_endpoints=_normalize_required_endpoints(
            spec=spec,
            authored=authored.required_endpoints,
            derived=derived.required_endpoints,
        ),
        implementation_checklist=_dedupe(
            [*authored.implementation_checklist, *derived.implementation_checklist],
            limit=6,
        ),
        domain_scenarios=authored_domain_scenarios or derived.domain_scenarios,
    )


def build_task_agent_deliverable_brief(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
) -> LearnerDeliverableBrief:
    deliverable_index = spec.deliverable_order[deliverable.id] + 1
    starter_surface = _resolved_starter_surface(spec, deliverable)
    files_to_edit = starter_surface.primary_editable_paths or _editable_files_for_spec(spec)
    primary_file = files_to_edit[0] if files_to_edit else _primary_editable_file(spec)
    scenario_lines = [
        f"{scenario.title}: {scenario.request_summary} {scenario.expected_behavior}".strip()
        for scenario in starter_surface.domain_scenarios
    ]
    if deliverable_index == 1:
        why_this_deliverable_matters = (
            "This is the first learner-visible review area. Get the public service shape working before you worry about deeper production polish."
        )
    else:
        why_this_deliverable_matters = (
            "This deliverable builds on the earlier working surface and adds one production-facing capability without changing the overall contract."
        )
    if primary_file:
        task_to_build = (
            f"Edit `{primary_file}` so the real service can {deliverable.objective.rstrip('.').lower()}. "
            f"Keep {_primary_request_surface(starter_surface)} stable while you implement the deliverable in learner-owned code."
        )
    else:
        task_to_build = (
            f"Edit the learner-owned repo files so the real service can {deliverable.objective.rstrip('.').lower()}. "
            f"Keep {_primary_request_surface(starter_surface)} stable while you implement the deliverable in learner-owned code."
        )
    definition_of_done = _dedupe(
        [
            f"{_primary_request_surface(starter_surface)} stays available while you implement the deliverable.",
            *starter_surface.implementation_checklist,
        ],
        limit=5,
    )
    implementation_hints = [
        (
            f"Start in `{primary_file}` and keep the main request path readable in learner-owned code."
            if primary_file
            else "Start in the learner-owned repo files and keep the main request path readable in learner-owned code."
        ),
        "Make the smallest change that satisfies this deliverable instead of jumping ahead to later production features.",
        "Prefer explicit, predictable branching over hidden helpers so failures are easier to debug.",
    ]
    if starter_surface.support_paths:
        implementation_hints.append(
            "Use support files like "
            + ", ".join(f"`{path}`" for path in starter_surface.support_paths[:3])
            + " to stay aligned with the published contract."
        )
    non_goals = [
        "Do not hide core deliverable logic behind generated support code or opaque wrapper layers.",
        "Do not optimize for later deliverables before this deliverable works end to end.",
    ]
    return LearnerDeliverableBrief(
        why_this_deliverable_matters=why_this_deliverable_matters,
        task_to_build=task_to_build,
        files_to_edit=files_to_edit,
        definition_of_done=definition_of_done,
        example_scenarios=_dedupe(scenario_lines, limit=3),
        implementation_hints=_dedupe(implementation_hints, limit=4),
        non_goals=_dedupe(non_goals, limit=3),
    )


def build_task_agent_deliverable_learning_outcomes(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
) -> list[str]:
    starter_surface = _resolved_starter_surface(spec, deliverable)
    outcomes = [
        f"Keep {_primary_request_surface(starter_surface)} stable while you build this deliverable.",
        "Use the visible checks and examples to improve the service deliberately instead of guessing.",
    ]
    if spec.project_contract.invariants:
        outcomes.append(
            "Preserve the core invariant: "
            + spec.project_contract.invariants[0].rstrip(".")
            + "."
        )
    return _dedupe(outcomes, limit=4)


def build_task_agent_public_checks(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
    *,
    limit: int = 3,
) -> list[PublicCheckSpec]:
    starter_surface = _resolved_starter_surface(spec, deliverable)
    checks: list[PublicCheckSpec] = [
        PublicCheckSpec(
            id=f"{deliverable.id}_health",
            title="Health endpoint stays up",
            learner_goal="Keep the service bootable while you work on the deliverable.",
            request_method="GET",
            request_path="/health",
            expected_status=200,
            files_to_use=starter_surface.primary_editable_paths[:1],
        )
    ]
    submit_endpoint = primary_submit_endpoint_for_spec(spec)
    if submit_endpoint is not None:
        scenarios = starter_surface.domain_scenarios[: max(1, limit - 1)]
        for index, scenario in enumerate(scenarios, start=1):
            checks.append(
                PublicCheckSpec(
                    id=f"{deliverable.id}_contract_{index}",
                    title=scenario.title,
                    learner_goal=scenario.expected_behavior,
                    request_method=submit_endpoint.method,
                    request_path=submit_endpoint.path.replace("{id}", "starter-check"),
                    request_body=_default_request_payload(spec, scenario, index=index),
                    expected_status=200,
                    files_to_use=starter_surface.primary_editable_paths[:1],
                )
            )
    return checks[:limit]


def _default_request_payload(
    spec: TaskAgentServiceSpec,
    scenario: StarterScenarioSpec,
    *,
    index: int,
) -> dict[str, object]:
    payload: dict[str, object] = {"request_id": f"starter-check-{index}"}
    slug = _primary_domain_entity(spec)
    payload[f"{slug.replace(' ', '_').lower()}_id"] = f"{slug[:4].lower() or 'item'}-{index:03d}"
    payload["summary"] = scenario.request_summary
    return payload


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
            files_to_edit=fallback_files_to_edit or [],
            definition_of_done=["Implement the deliverable goal in the learner-visible workspace and submit it for review."],
            example_scenarios=[],
            implementation_hints=["Start with the learner-visible starter files before making broader refactors."],
            non_goals=[],
        )
    first = brief_list[0]
    return LearnerDeliverableBrief(
        why_this_deliverable_matters=first.why_this_deliverable_matters,
        task_to_build=first.task_to_build or fallback_task,
        files_to_edit=_dedupe(
            [item for brief in brief_list for item in brief.files_to_edit]
            or list(fallback_files_to_edit or []),
            limit=6,
        ),
        definition_of_done=_dedupe(
            [item for brief in brief_list for item in brief.definition_of_done],
            limit=6,
        ),
        example_scenarios=_dedupe(
            [item for brief in brief_list for item in brief.example_scenarios],
            limit=4,
        ),
        implementation_hints=_dedupe(
            [item for brief in brief_list for item in brief.implementation_hints],
            limit=5,
        ),
        non_goals=_dedupe(
            [item for brief in brief_list for item in brief.non_goals],
            limit=4,
        ),
    )


def combine_public_checks(checks: Iterable[PublicCheckSpec]) -> list[PublicCheckSpec]:
    combined: list[PublicCheckSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for check in checks:
        key = (check.request_method, check.request_path, check.title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        combined.append(check)
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
        summary,
        "",
        "## Why this matters",
        "",
        brief.why_this_deliverable_matters,
        "",
        "## What to build",
        "",
        brief.task_to_build,
        "",
    ]
    if learning_outcomes:
        lines.extend(["## Learning outcomes", ""])
        lines.extend(f"- {item}" for item in learning_outcomes)
        lines.append("")
    if brief.files_to_edit:
        lines.extend(["## Files to edit", ""])
        lines.extend(f"- `{item}`" for item in brief.files_to_edit)
        lines.append("")
    if brief.definition_of_done:
        lines.extend(["## Definition of done", ""])
        lines.extend(f"- {item}" for item in brief.definition_of_done)
        lines.append("")
    if brief.example_scenarios:
        lines.extend(["## Example scenarios", ""])
        lines.extend(f"- {item}" for item in brief.example_scenarios)
        lines.append("")
    if brief.implementation_hints:
        lines.extend(["## Implementation hints", ""])
        lines.extend(f"- {item}" for item in brief.implementation_hints)
        lines.append("")
    if public_checks:
        lines.extend(["## Visible checks", ""])
        for check in public_checks:
            lines.append(f"- **{check.title}** - {check.learner_goal}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_learner_starter_readme(
    *,
    title: str,
    brief: LearnerDeliverableBrief,
    summary: str,
    learning_outcomes: list[str] | None = None,
    visible_check_command: str | None = None,
    preview_command: str | None = None,
    public_checks: list[PublicCheckSpec] | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        summary,
        "",
        "## What to build",
        "",
        brief.task_to_build,
        "",
        "## Files to edit",
        "",
    ]
    lines.extend(f"- `{item}`" for item in brief.files_to_edit)
    lines.extend(["", "## Definition of done", ""])
    lines.extend(f"- {item}" for item in brief.definition_of_done)
    if brief.example_scenarios:
        lines.extend(["", "## Example scenarios", ""])
        lines.extend(f"- {item}" for item in brief.example_scenarios)
    if learning_outcomes:
        lines.extend(["", "## Learning outcomes", ""])
        lines.extend(f"- {item}" for item in learning_outcomes)
    if brief.implementation_hints:
        lines.extend(["", "## Helpful hints", ""])
        lines.extend(f"- {item}" for item in brief.implementation_hints)
    lines.extend(["", "## Helpful commands", ""])
    lines.append("- Preview: `" + (preview_command or "sh .coursegen/runtime/run.sh") + "`")
    lines.append(f"- Visible checks: `{visible_check_command or f'sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}'}`")
    if public_checks:
        lines.extend(["", "## Visible checks", ""])
        for check in public_checks:
            lines.append(f"- **{check.title}** - {check.learner_goal}")
    return "\n".join(lines).strip() + "\n"


def _starter_scenario_looks_placeholder(scenario: StarterScenarioSpec) -> bool:
    text = " ".join([scenario.title, scenario.request_summary, scenario.expected_behavior]).lower()
    return any(phrase in text for phrase in ["routine case", "ambiguous or risky case", "placeholder", *starter_surface_markers()])


def _dedupe(items: Iterable[str], *, limit: int | None = None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
        if limit is not None and len(normalized) >= limit:
            break
    return normalized
