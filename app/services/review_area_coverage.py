from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from app.domain.task_agent import TaskAgentServiceSpec


RESERVED_REVIEW_AREA_TAGS = {"overall_system"}


class ReviewAreaHiddenCoverageSummary(BaseModel):
    module_id: str
    hidden_test_ids: list[str] = Field(default_factory=list)
    hidden_case_ids: list[str] = Field(default_factory=list)


def case_ids_for_test(spec: TaskAgentServiceSpec, test: Any) -> list[str]:
    if hasattr(test, "case_ids"):
        return [str(case_id) for case_id in getattr(test, "case_ids") or []]
    if hasattr(test, "case_id"):
        case_id = getattr(test, "case_id")
        return [str(case_id)] if case_id else []
    if hasattr(test, "expectations"):
        return [str(expectation.case_id) for expectation in getattr(test, "expectations") or []]
    if hasattr(test, "injections"):
        return [str(injection.case_id) for injection in getattr(test, "injections") or []]
    if getattr(test, "dataset_id", None) == spec.eval_dataset.id:
        return [case.id for case in spec.eval_dataset.cases]
    return []


def infer_review_area_case_tags(spec: TaskAgentServiceSpec) -> dict[str, list[str]]:
    inferred: dict[str, set[str]] = defaultdict(set)
    behaviors_by_id = {behavior.id: behavior for behavior in spec.behaviors}
    qualities_by_id = {quality.id: quality for quality in spec.qualities}

    for module in spec.modules:
        gate = spec.gate_for(module.id)
        case_ids: set[str] = {
            check.case_id
            for check in module.public_checks
            if check.case_id in spec.eval_case_ids
        }
        for behavior_id in gate.active_behavior_ids:
            behavior = behaviors_by_id.get(behavior_id)
            if behavior is None:
                continue
            case_ids.update(
                case_id
                for case_id in case_ids_for_test(spec, behavior.test)
                if case_id in spec.eval_case_ids
            )
        for quality_id in gate.active_quality_ids:
            quality = qualities_by_id.get(quality_id)
            if quality is None:
                continue
            case_ids.update(
                case_id
                for case_id in case_ids_for_test(spec, quality.test)
                if case_id in spec.eval_case_ids
            )
        for case_id in case_ids:
            inferred[case_id].add(module.id)

    return {case_id: sorted(tags) for case_id, tags in inferred.items()}


def apply_inferred_review_area_case_tags(spec: TaskAgentServiceSpec) -> TaskAgentServiceSpec:
    inferred_tags = infer_review_area_case_tags(spec)
    allowed_tags = set(spec.module_order.keys()) | RESERVED_REVIEW_AREA_TAGS
    for case in spec.eval_dataset.cases:
        merged_tags = {tag for tag in case.tags if tag in allowed_tags}
        merged_tags.update(inferred_tags.get(case.id, []))
        case.tags = sorted(merged_tags)
    return spec


def summarize_review_area_hidden_coverage(spec: TaskAgentServiceSpec) -> list[ReviewAreaHiddenCoverageSummary]:
    inferred_tags = infer_review_area_case_tags(spec)
    summaries: list[ReviewAreaHiddenCoverageSummary] = []
    for module in spec.modules:
        gate = spec.gate_for(module.id)
        hidden_case_ids = sorted(
            case_id
            for case_id, tags in inferred_tags.items()
            if module.id in tags
        )
        summaries.append(
            ReviewAreaHiddenCoverageSummary(
                module_id=module.id,
                hidden_test_ids=list(gate.active_test_ids),
                hidden_case_ids=hidden_case_ids,
            )
        )
    return summaries
