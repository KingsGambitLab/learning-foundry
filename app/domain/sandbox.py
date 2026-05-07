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


class SandboxAvailability(BaseModel):
    engine: SandboxEngine = SandboxEngine.docker
    available: bool
    message: str
    docker_version: str | None = None


class ModuleSandboxReport(BaseModel):
    module_id: str
    compile_succeeded: bool
    runtime_succeeded: bool
    health_status_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


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
    module_reports: list[ModuleSandboxReport] = Field(default_factory=list)
    error: str | None = None
