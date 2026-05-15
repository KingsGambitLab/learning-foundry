from __future__ import annotations

import tempfile
from pathlib import Path

from app.services.learner_studio_service import LearnerStudioService


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _make_workspace(root: Path) -> None:
    """Set up a small but realistic workspace skeleton."""
    _write(root / "Dockerfile", "FROM python:3.11-slim\nRUN pip install -r requirements.txt\n")
    _write(root / "requirements.txt", "fastapi==0.115.0\npydantic==2.9.2\n")
    _write(root / ".coursegen/runtime/install.sh", "#!/usr/bin/env sh\npip install -r requirements.txt\n")
    _write(root / ".coursegen/runtime/verify.sh", "#!/usr/bin/env sh\npython -c 'import fastapi'\n")
    _write(root / ".coursegen/runtime/run.sh", "#!/usr/bin/env sh\nuvicorn app.main:app\n")
    _write(root / "app/main.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    _write(root / "data/corpus.json", '[{"id": "doc-1", "text": "hello"}]\n')


def test_cache_key_unchanged_when_only_learner_source_changes() -> None:
    """Changing app/*.py or data/* MUST NOT invalidate the runtime image
    cache. The workspace bind-mounts at runtime; learner code is not baked
    into the image."""
    service = LearnerStudioService(image_name="x")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_workspace(root)
        before = service._workspace_runtime_cache_key(root)

        # Learner edits a source file -> image must stay identical.
        _write(root / "app/main.py", "from fastapi import FastAPI\napp = FastAPI(title='x')\n")
        # New data file appears.
        _write(root / "data/corpus.json", '[{"id": "doc-1", "text": "world"}]\n')
        # A whole new editable module appears.
        _write(root / "app/retrieval.py", "def search(): return []\n")

        after = service._workspace_runtime_cache_key(root)
        assert before == after


def test_cache_key_unchanged_when_check_scripts_change() -> None:
    """authoring_tests writes per-deliverable check scripts. Those MUST NOT
    invalidate the runtime image — they execute against the bind-mounted
    workspace at runtime."""
    service = LearnerStudioService(image_name="x")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_workspace(root)
        before = service._workspace_runtime_cache_key(root)

        _write(root / "public/checks/d1/run_visible_checks.py", "def main(): pass\n")
        _write(root / "private/grader/d1/run_hidden_checks.py", "def main(): pass\n")
        _write(root / "deliverables/d1/spec.md", "# deliverable 1\n")

        after = service._workspace_runtime_cache_key(root)
        assert before == after


def test_cache_key_changes_when_dockerfile_changes() -> None:
    service = LearnerStudioService(image_name="x")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_workspace(root)
        before = service._workspace_runtime_cache_key(root)

        _write(root / "Dockerfile", "FROM python:3.12-slim\nRUN pip install -r requirements.txt\n")
        after = service._workspace_runtime_cache_key(root)
        assert before != after


def test_cache_key_changes_when_requirements_changes() -> None:
    service = LearnerStudioService(image_name="x")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_workspace(root)
        before = service._workspace_runtime_cache_key(root)

        _write(root / "requirements.txt", "fastapi==0.115.0\npydantic==2.9.2\nsentence-transformers==3.1.1\n")
        after = service._workspace_runtime_cache_key(root)
        assert before != after


def test_cache_key_changes_when_install_sh_changes() -> None:
    service = LearnerStudioService(image_name="x")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_workspace(root)
        before = service._workspace_runtime_cache_key(root)

        _write(
            root / ".coursegen/runtime/install.sh",
            "#!/usr/bin/env sh\npip install --index-url https://download.pytorch.org/whl/cpu torch\npip install -r requirements.txt\n",
        )
        after = service._workspace_runtime_cache_key(root)
        assert before != after


def test_cache_key_changes_when_new_dependency_manifest_appears() -> None:
    """Adding a recognized dependency manifest (e.g. package.json) is a real
    dependency-contract change and must invalidate the image."""
    service = LearnerStudioService(image_name="x")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_workspace(root)
        before = service._workspace_runtime_cache_key(root)

        _write(root / "package.json", '{"name":"x","dependencies":{"react":"^18"}}\n')
        after = service._workspace_runtime_cache_key(root)
        assert before != after


def test_cache_key_unchanged_when_ignored_paths_change() -> None:
    service = LearnerStudioService(image_name="x")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_workspace(root)
        before = service._workspace_runtime_cache_key(root)

        _write(root / ".git/HEAD", "ref: refs/heads/main\n")
        _write(root / "__pycache__/foo.cpython-311.pyc", "binary\n")
        _write(root / "node_modules/react/package.json", "{}\n")

        after = service._workspace_runtime_cache_key(root)
        assert before == after
