from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class GraderEntryKind(str, Enum):
    behavior = "behavior"
    quality = "quality"


class ControlFlag(str, Enum):
    approval = "approval"
    budget = "budget"
    cost = "cost"
    dry_run = "dry_run"
    escalation = "escalation"
    fault_injection = "fault_injection"
    idempotency = "idempotency"
    judge = "judge"
    latency = "latency"
    resume = "resume"
    trace = "trace"


class TestDependencies(BaseModel):
    eval_case_ids: list[str] = Field(default_factory=list)
    dataset_id: str | None = None
    tool_ids: list[str] = Field(default_factory=list)
    endpoint_paths: list[str] = Field(default_factory=list)
    required_events: list[str] = Field(default_factory=list)
    allowed_reasons: list[str] = Field(default_factory=list)
    mutating_tool_ids: list[str] = Field(default_factory=list)
    injected_failures: list[str] = Field(default_factory=list)
    idempotency_key_field: str | None = None


class GraderPlanEntry(BaseModel):
    test_id: str
    kind: GraderEntryKind
    test_type: str
    description: str
    first_required_in: str
    controls: list[ControlFlag] = Field(default_factory=list)
    dependencies: TestDependencies
    config: dict[str, Any] = Field(default_factory=dict)


class ModuleGraderPlan(BaseModel):
    module_id: str
    module_title: str
    module_objective: str
    starter_type: str
    overlay_ids: list[str] = Field(default_factory=list)
    cumulative_modules: list[str] = Field(default_factory=list)
    active_behavior_ids: list[str] = Field(default_factory=list)
    active_quality_ids: list[str] = Field(default_factory=list)
    total_tests: int
    endpoint_paths: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    controls: list[ControlFlag] = Field(default_factory=list)
    entries: list[GraderPlanEntry] = Field(default_factory=list)


class TaskAgentGraderPlanCollection(BaseModel):
    title: str
    eval_dataset_id: str
    system_profile: list[str] = Field(default_factory=list)
    module_plans: list[ModuleGraderPlan] = Field(default_factory=list)
