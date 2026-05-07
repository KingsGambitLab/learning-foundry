from __future__ import annotations

import mimetypes
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from app.domain.assets import CreateCreatorAssetRequest, CreatorAssetList, CreatorAssetRecord
from app.domain.task_agent import DataSourceKind, DataSourceSpec
from app.storage.sqlite_store import SQLiteWorkflowStore


def default_creator_assets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "creator_assets"


class CreatorAssetService:
    def __init__(self, store: SQLiteWorkflowStore, base_dir: str | Path | None = None) -> None:
        self.store = store
        self.base_dir = Path(base_dir or default_creator_assets_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_assets(self, limit: int = 100) -> CreatorAssetList:
        return CreatorAssetList(assets=self.store.list_creator_assets(limit=limit))

    def get_asset(self, asset_id: str) -> CreatorAssetRecord | None:
        return self.store.get_creator_asset(asset_id)

    def create_asset(self, request: CreateCreatorAssetRequest) -> CreatorAssetRecord:
        safe_file_name = self._sanitize_file_name(request.file_name)
        if not safe_file_name:
            raise ValueError("Uploaded files need a valid file name.")

        asset_id = f"asset_{uuid4().hex[:12]}"
        suffix = Path(safe_file_name).suffix.lower()
        inferred_format = (suffix[1:] if suffix else None) or self._format_from_content_type(request.content_type)
        workspace_path = self._normalize_workspace_path(
            request.workspace_path,
            safe_file_name=safe_file_name,
            inferred_format=inferred_format,
        )

        asset_dir = self.base_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        asset_path = asset_dir / safe_file_name
        asset_path.write_text(request.content, encoding="utf-8")

        data_source = DataSourceSpec(
            id=asset_id,
            kind=DataSourceKind.uploaded_file,
            title=(request.title or Path(safe_file_name).stem.replace("_", " ").replace("-", " ").strip()).title(),
            purpose=request.purpose,
            learner_visible=request.learner_visible,
            format=inferred_format,
            workspace_path=workspace_path,
            asset_id=asset_id,
            description=request.description or f"Uploaded from `{safe_file_name}`.",
        )
        record = CreatorAssetRecord(
            id=asset_id,
            file_name=safe_file_name,
            title=data_source.title,
            content_type=request.content_type or self._content_type_for_path(safe_file_name),
            format=inferred_format,
            size_bytes=len(request.content.encode("utf-8")),
            purpose=request.purpose,
            learner_visible=request.learner_visible,
            workspace_path=workspace_path,
            description=data_source.description,
            created_at=datetime.now(UTC),
            data_source=data_source,
        )
        self.store.save_creator_asset(record)
        return record

    def delete_asset(self, asset_id: str) -> bool:
        deleted = self.store.delete_creator_asset(asset_id)
        asset_dir = self.base_dir / asset_id
        if asset_dir.exists():
            for child in asset_dir.iterdir():
                child.unlink(missing_ok=True)
            asset_dir.rmdir()
        return deleted

    def read_asset_text(self, asset_id: str) -> tuple[CreatorAssetRecord, str]:
        record = self.get_asset(asset_id)
        if record is None:
            raise KeyError(asset_id)
        path = self.base_dir / asset_id / record.file_name
        if not path.exists():
            raise FileNotFoundError(asset_id)
        return record, path.read_text(encoding="utf-8")

    def _sanitize_file_name(self, file_name: str) -> str:
        name = Path(file_name).name.strip()
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return sanitized.strip("._")

    def _normalize_workspace_path(
        self,
        workspace_path: str | None,
        *,
        safe_file_name: str,
        inferred_format: str | None,
    ) -> str:
        if workspace_path:
            candidate = PurePosixPath(workspace_path.strip())
            if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
                raise ValueError("Workspace path must stay inside the learner workspace.")
            return str(candidate)

        stem = Path(safe_file_name).stem or "uploaded_data"
        suffix = Path(safe_file_name).suffix
        if not suffix and inferred_format:
            suffix = f".{inferred_format}"
        file_name = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "uploaded_data"
        return f"data/{file_name}{suffix}"

    def _format_from_content_type(self, content_type: str | None) -> str | None:
        if not content_type:
            return None
        if "json" in content_type:
            return "json"
        if "csv" in content_type:
            return "csv"
        if "markdown" in content_type:
            return "md"
        if "yaml" in content_type or "yml" in content_type:
            return "yml"
        if "text" in content_type:
            return "txt"
        return None

    def _content_type_for_path(self, path: str) -> str | None:
        guessed, _ = mimetypes.guess_type(path)
        return guessed
