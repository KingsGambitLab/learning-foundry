from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, BaseModel, Field

from app.domain.learner import LearnerWorkspaceScope
from app.domain.registry import PackageType
from app.domain.task_agent import LearnerModuleBrief, PublicCheckSpec, TaskAgentServiceSpec


class LearnerPackageFile(BaseModel):
    relative_path: str
    media_type: str
    content: str


class LearnerModulePackage(BaseModel):
    deliverable_id: str = Field(validation_alias=AliasChoices("deliverable_id", "module_id"))
    course_deliverable_slug: str | None = Field(
        default=None,
        validation_alias=AliasChoices("course_deliverable_slug", "course_module_slug"),
    )
    title: str
    objective: str
    deliverable_index: int = Field(validation_alias=AliasChoices("deliverable_index", "module_index"))
    learner_brief: LearnerModuleBrief
    public_checks: list[PublicCheckSpec] = Field(default_factory=list)
    content_markdown: str
    starter_readme: str
    learning_outcomes: list[str] = Field(default_factory=list)
    active_test_ids: list[str] = Field(default_factory=list)
    completion_rule: str
    visible_files: list[str] = Field(default_factory=list)
    workspace_seed_files: list[LearnerPackageFile] = Field(default_factory=list)


class LearnerCoursePackage(BaseModel):
    course_run_id: str
    title: str
    summary: str
    package_type: PackageType
    published_at: datetime
    workspace_scope: LearnerWorkspaceScope
    project_brief_markdown: str = ""
    deliverables: list[LearnerModulePackage] = Field(
        default_factory=list,
        validation_alias=AliasChoices("deliverables", "modules"),
    )
    notes: list[str] = Field(default_factory=list)


class PublishSnapshotProvenance(BaseModel):
    generator_version: str
    course_run_hash: str
    workflow_run_hashes: dict[str, str] = Field(default_factory=dict)
    workflow_bundle_ids: dict[str, str] = Field(default_factory=dict)
    course_bundle_id: str | None = None


class PublishSnapshot(BaseModel):
    id: str
    course_run_id: str
    course_family_id: str
    created_at: datetime
    version: int = Field(ge=1)
    source_hash: str
    shared_workflow_run_id: str | None = None
    learner_package: LearnerCoursePackage | None = None
    task_agent_spec: TaskAgentServiceSpec | None = None
    provenance: PublishSnapshotProvenance
    notes: list[str] = Field(default_factory=list)


class PublishSnapshotSummary(BaseModel):
    id: str
    course_run_id: str
    course_family_id: str
    created_at: datetime
    version: int = Field(ge=1)
    source_hash: str

    @classmethod
    def from_snapshot(cls, snapshot: PublishSnapshot) -> "PublishSnapshotSummary":
        return cls(
            id=snapshot.id,
            course_run_id=snapshot.course_run_id,
            course_family_id=snapshot.course_family_id,
            created_at=snapshot.created_at,
            version=snapshot.version,
            source_hash=snapshot.source_hash,
        )


class PublishedVersionSummary(BaseModel):
    snapshot_id: str
    course_run_id: str
    version: int = Field(ge=1)
    created_at: datetime
    is_latest: bool = False
    default_for_new_enrollments: bool = False
    learner_count: int = 0
    deliverable_count: int = Field(default=0, validation_alias=AliasChoices("deliverable_count", "module_count"))
    compatibility: str = "new_enrollments_only"
    changes: list[str] = Field(default_factory=list)


class PublishedVersionList(BaseModel):
    versions: list[PublishedVersionSummary] = Field(default_factory=list)
