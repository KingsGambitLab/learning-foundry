from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.registry import PackageType
from app.services.assignment_design_inference import infer_assignment_design
from app.services.learner_studio_service import LearnerStudioService
from app.services.task_agent_scaffolds import build_task_agent_scaffold
from app.services.task_agent_starter_templates import (
    HIDDEN_MANIFEST_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    RUNTIME_VISIBLE_CHECK_SCRIPT_PATH,
    build_task_agent_starter_files,
)


def _build_spec(
    *,
    title: str,
    summary: str,
    problem_statement: str,
    implementation_language: str | None = None,
    application_framework: str | None = None,
):
    inferred = infer_assignment_design(
        title=title,
        problem_statement=problem_statement,
        learning_outcomes=[
            "Design a production-grade backend surface.",
            "Ship a runtime that can be graded end to end.",
        ],
        package_type_hint=PackageType.progressive_codebase_course,
        implementation_language=implementation_language,
        application_framework=application_framework,
    )
    assert inferred.design_spec is not None
    spec, _origin = build_task_agent_scaffold(
        title=title,
        summary=summary,
        design_spec=inferred.design_spec,
    )
    return spec


def test_typescript_runtime_scripts_are_repo_owned_and_pnpm_safe() -> None:
    spec = _build_spec(
        title="Feature Flag Control Plane",
        summary="Build a feature flag control plane service.",
        problem_statement=(
            "Build a feature flag control plane backend with gradual rollout support, "
            "NestJS, MongoDB, pnpm, audit logs, and safe config updates."
        ),
        implementation_language="typescript",
        application_framework="nestjs",
    )

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    manifest = json.loads(starter_files[HIDDEN_MANIFEST_PATH])

    assert manifest["preview_command"] == f"sh {RUNTIME_RUN_SCRIPT_PATH}"
    assert manifest["visible_check_command"] == f"sh {RUNTIME_VISIBLE_CHECK_SCRIPT_PATH}"
    assert manifest["hidden_check_command"] == "sh .coursegen/runtime/check_hidden.sh"
    assert "pnpm install --no-frozen-lockfile" in starter_files[RUNTIME_INSTALL_SCRIPT_PATH]
    assert "--yes" not in starter_files[RUNTIME_INSTALL_SCRIPT_PATH]
    assert "corepack enable" in starter_files[RUNTIME_INSTALL_SCRIPT_PATH]
    assert "exec pnpm start:dev" in starter_files[RUNTIME_RUN_SCRIPT_PATH]
    assert "apt-get install -y --no-install-recommends python3" in starter_files["Dockerfile"]
    assert "exec python3 checks/run_visible_checks.py" in starter_files[".coursegen/runtime/check_visible.sh"]
    assert "exec python3 .coursegen/grader/run_hidden_checks.py" in starter_files[".coursegen/runtime/check_hidden.sh"]


def test_go_runtime_scripts_export_toolchain_path_and_launch_via_repo_scripts() -> None:
    spec = _build_spec(
        title="Inventory Reservation Service",
        summary="Build a concurrency-safe inventory reservation backend.",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with Go, Gin, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
        implementation_language="go",
        application_framework="gin",
    )

    starter_files = build_task_agent_starter_files(spec, spec.deliverables[0].id)
    manifest = json.loads(starter_files[HIDDEN_MANIFEST_PATH])

    assert manifest["preview_command"] == f"sh {RUNTIME_RUN_SCRIPT_PATH}"
    assert "export PATH=\"/usr/local/go/bin:$PATH\"" in starter_files[RUNTIME_INSTALL_SCRIPT_PATH]
    assert "export PATH=\"/usr/local/go/bin:$PATH\"" in starter_files[RUNTIME_VERIFY_SCRIPT_PATH]
    assert "export PATH=\"/usr/local/go/bin:$PATH\"" in starter_files[RUNTIME_RUN_SCRIPT_PATH]
    assert "go mod tidy" in starter_files[RUNTIME_INSTALL_SCRIPT_PATH]
    assert "go build ./..." in starter_files[RUNTIME_VERIFY_SCRIPT_PATH]
    assert "exec go run ." in starter_files[RUNTIME_RUN_SCRIPT_PATH]
    assert "apt-get install -y --no-install-recommends python3" in starter_files["Dockerfile"]
    assert "go.mod" not in starter_files
    assert "main.go" not in starter_files
    assert manifest["starter_repo_bundle"]["source"] == "starter_default"

    with TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        for relative_path, content in starter_files.items():
            output_path = workspace_path / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

        service = LearnerStudioService()
        launch_script = service._runtime_launch_script(
            workspace_path=workspace_path,
            spec=spec,
            include_setup=True,
        )

    assert f"sh {RUNTIME_INSTALL_SCRIPT_PATH}" in launch_script
    assert f"sh {RUNTIME_VERIFY_SCRIPT_PATH}" in launch_script
    assert f"exec sh {RUNTIME_RUN_SCRIPT_PATH}" in launch_script
