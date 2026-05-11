from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SandboxEngine(str, Enum):
    docker = "docker"


class SandboxExecutionStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    unavailable = "unavailable"


class SandboxFailureStage(str, Enum):
    missing_workspace = "missing_workspace"
    dependency_materialization = "dependency_materialization"
    image_build = "image_build"
    install = "install"
    verify = "verify"
    boot = "boot"
    contract = "contract"
    checks = "checks"
    container_launch = "container_launch"
    runtime = "runtime"


class SandboxAvailability(BaseModel):
    engine: SandboxEngine = SandboxEngine.docker
    available: bool
    message: str
    docker_version: str | None = None


class DeliverableSandboxReport(BaseModel):
    deliverable_id: str
    compile_succeeded: bool
    runtime_succeeded: bool
    failed_stage: SandboxFailureStage | None = None
    stage_command: list[str] = Field(default_factory=list)
    stage_exit_code: int | None = None
    public_checks_passed: bool | None = None
    health_status_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    # Extended diagnostic surface (Pass 8). All optional so legacy callers
    # and tests that construct a minimal report keep working.
    stdout_tail: str | None = None
    exit_state: dict | None = None
    sidecar_diagnostics: dict[str, dict] | None = None
    http_response: dict | None = None


class SandboxExecutionResult(BaseModel):
    engine: SandboxEngine = SandboxEngine.docker
    status: SandboxExecutionStatus
    available: bool
    build_succeeded: bool = False
    build_cached: bool = False
    run_succeeded: bool = False
    generated_at: datetime
    duration_ms: int = 0
    workspace_root: str | None = None
    image_tag: str | None = None
    cache_key: str | None = None
    build_command: list[str] = Field(default_factory=list)
    run_command: list[str] = Field(default_factory=list)
    build_stdout: str = ""
    build_stderr: str = ""
    run_stdout: str = ""
    run_stderr: str = ""
    deliverable_reports: list[DeliverableSandboxReport] = Field(default_factory=list)
    error: str | None = None
