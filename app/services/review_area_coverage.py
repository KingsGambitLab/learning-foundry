from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.task_agent import TaskAgentServiceSpec

RESERVED_REVIEW_AREA_TAGS: set[str] = set()


class ReviewAreaHiddenCoverageSummary(BaseModel):
    deliverable_id: str
    hidden_case_ids: list[str] = Field(default_factory=list)


def case_ids_for_test(spec: TaskAgentServiceSpec, test) -> list[str]:
    return []


def infer_review_area_case_tags(spec: TaskAgentServiceSpec) -> dict[str, list[str]]:
    return {
        deliverable.id: [check.id for check in deliverable.public_checks]
        for deliverable in spec.deliverables
    }


def apply_inferred_review_area_case_tags(spec: TaskAgentServiceSpec) -> TaskAgentServiceSpec:
    return spec


def summarize_review_area_hidden_coverage(spec: TaskAgentServiceSpec) -> list[ReviewAreaHiddenCoverageSummary]:
    return [
        ReviewAreaHiddenCoverageSummary(
            deliverable_id=deliverable.id,
            hidden_case_ids=[check.id for check in deliverable.public_checks],
        )
        for deliverable in spec.deliverables
    ]
