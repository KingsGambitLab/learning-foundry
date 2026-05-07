from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.task_agent import DataSourcePurpose, DataSourceSpec


class CreateCreatorAssetRequest(BaseModel):
    file_name: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_type: str | None = None
    title: str | None = None
    purpose: DataSourcePurpose = DataSourcePurpose.reference_data
    learner_visible: bool = True
    workspace_path: str | None = None
    description: str | None = None


class CreatorAssetRecord(BaseModel):
    id: str
    file_name: str
    title: str
    content_type: str | None = None
    format: str | None = None
    size_bytes: int = Field(ge=0)
    purpose: DataSourcePurpose = DataSourcePurpose.reference_data
    learner_visible: bool = True
    workspace_path: str
    description: str | None = None
    created_at: datetime
    data_source: DataSourceSpec


class CreatorAssetList(BaseModel):
    assets: list[CreatorAssetRecord] = Field(default_factory=list)


class DeleteCreatorAssetResult(BaseModel):
    deleted: bool = True
    asset_id: str
