from __future__ import annotations

from pathlib import Path

from app.domain.workflow import BundleFileContent, MaterializedBundle, WorkflowRun
from app.services.artifact_materializer import ArtifactMaterializer


def default_workspace_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "workspaces"


class AssignmentWorkspaceManager:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or default_workspace_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def prepare_run_workspace(self, run: WorkflowRun, overwrite: bool = True) -> MaterializedBundle:
        materializer = ArtifactMaterializer(base_dir=self.base_dir)
        return materializer.materialize_run(run, overwrite=overwrite)

    def read_workspace_file(self, workspace: MaterializedBundle, relative_path: str) -> BundleFileContent:
        materializer = ArtifactMaterializer(base_dir=self.base_dir)
        return materializer.read_bundle_file(workspace, relative_path)
