from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.registry import StarterType
from app.domain.task_agent import TaskAgentServiceSpec
from app.services.learner_studio_service import LearnerStudioError, LearnerStudioService
from app.services.task_agent_starter_templates import HIDDEN_MANIFEST_PATH

DEFAULT_VISIBLE_CHECK_COMMAND = "python checks/run_visible_checks.py"
DEFAULT_HIDDEN_CHECK_COMMAND = "python .coursegen/grader/run_hidden_checks.py"


class GeneratedTestCaseReport(BaseModel):
    id: str
    title: str
    status: str
    summary: str = ""
    diagnostics: list[str] = Field(default_factory=list)


class GeneratedTestSuiteReport(BaseModel):
    suite_type: str
    command: str
    exit_code: int
    valid: bool
    passed: bool
    tests: list[GeneratedTestCaseReport] = Field(default_factory=list)
    summary: str = ""
    stdout: str = ""
    stderr: str = ""


class BaselineSuiteOutcome(BaseModel):
    baseline: str
    suite_type: str
    report: GeneratedTestSuiteReport


class BaselineValidationIssue(BaseModel):
    level: str
    code: str
    message: str
    baseline: str | None = None
    suite_type: str | None = None
    relative_path: str | None = None


class BaselineValidationResult(BaseModel):
    valid: bool
    errors: list[BaselineValidationIssue] = Field(default_factory=list)
    warnings: list[BaselineValidationIssue] = Field(default_factory=list)
    outcomes: list[BaselineSuiteOutcome] = Field(default_factory=list)


class GeneratedTestScriptRunner:
    def __init__(self, *, command_timeout_s: int = 90) -> None:
        self.command_timeout_s = command_timeout_s

    def run_suite(
        self,
        *,
        workspace_root: Path,
        command: str,
        base_url: str,
        suite_type: str,
    ) -> GeneratedTestSuiteReport:
        with tempfile.TemporaryDirectory(prefix="coursegen_test_report_") as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            env = os.environ.copy()
            env["BASE_URL"] = base_url
            env["REPORT_PATH"] = str(report_path)
            env.setdefault("PYTHONUNBUFFERED", "1")
            try:
                result = subprocess.run(
                    ["sh", "-lc", command],
                    cwd=workspace_root,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                return GeneratedTestSuiteReport(
                    suite_type=suite_type,
                    command=command,
                    exit_code=124,
                    valid=False,
                    passed=False,
                    summary=f"{suite_type} tests timed out.",
                    stdout=(exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
                    stderr=(exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")),
                )

            payload = self._load_report(report_path, result.stdout)
            cases = self._cases_from_payload(payload)
            suite_valid = bool(cases)
            suite_passed = suite_valid and result.returncode == 0 and all(case.status == "passed" for case in cases)
            summary = str(payload.get("summary") or ("All generated tests passed." if suite_passed else "Generated tests failed."))
            if not suite_valid:
                summary = payload.get("summary") or "Generated test script did not emit a valid report."

            return GeneratedTestSuiteReport(
                suite_type=suite_type,
                command=command,
                exit_code=int(result.returncode),
                valid=suite_valid,
                passed=suite_passed,
                tests=cases,
                summary=str(summary),
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )

    def _load_report(self, report_path: Path, stdout: str) -> dict[str, object]:
        if report_path.exists():
            try:
                return json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        try:
            return json.loads((stdout or "").strip())
        except json.JSONDecodeError:
            return {}

    def _cases_from_payload(self, payload: dict[str, object]) -> list[GeneratedTestCaseReport]:
        raw_tests = payload.get("tests")
        if not isinstance(raw_tests, list):
            return []
        cases: list[GeneratedTestCaseReport] = []
        for index, item in enumerate(raw_tests, start=1):
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("id") or f"test_{index}")
            title = str(item.get("title") or case_id)
            status = str(item.get("status") or "").strip().lower()
            if status not in {"passed", "failed"}:
                continue
            diagnostics = item.get("diagnostics") or []
            cases.append(
                GeneratedTestCaseReport(
                    id=case_id,
                    title=title,
                    status=status,
                    summary=str(item.get("summary") or ""),
                    diagnostics=[str(entry) for entry in diagnostics if str(entry).strip()],
                )
            )
        return cases


class GeneratedTestBaselineVerifier:
    def __init__(
        self,
        *,
        learner_studio_service: LearnerStudioService | None = None,
        script_runner: GeneratedTestScriptRunner | None = None,
    ) -> None:
        self.learner_studio_service = learner_studio_service or LearnerStudioService()
        self.script_runner = script_runner or GeneratedTestScriptRunner()

    def verify_deliverable(
        self,
        *,
        workspace_root: str | Path,
        spec: TaskAgentServiceSpec,
        starter_type: StarterType,
    ) -> BaselineValidationResult:
        workspace_path = Path(workspace_root).resolve()
        manifest = self.learner_studio_service._runtime_manifest(workspace_path)
        visible_command = str(manifest.get("visible_check_command") or DEFAULT_VISIBLE_CHECK_COMMAND)
        hidden_command = str(manifest.get("hidden_check_command") or DEFAULT_HIDDEN_CHECK_COMMAND)

        errors: list[BaselineValidationIssue] = []
        warnings: list[BaselineValidationIssue] = []
        outcomes: list[BaselineSuiteOutcome] = []

        visible_script_path = self._command_target_path(workspace_path, visible_command)
        hidden_script_path = self._command_target_path(workspace_path, hidden_command)
        if visible_script_path is None or not visible_script_path.exists():
            errors.append(
                BaselineValidationIssue(
                    level="error",
                    code="visible_test_command_missing",
                    message="Visible test command does not point to a real script in the learner workspace.",
                    suite_type="visible",
                    relative_path=str(visible_script_path.relative_to(workspace_path)) if visible_script_path and visible_script_path.exists() else None,
                )
            )
        if hidden_script_path is None or not hidden_script_path.exists():
            errors.append(
                BaselineValidationIssue(
                    level="error",
                    code="hidden_test_command_missing",
                    message="Hidden test command does not point to a real script in the learner workspace.",
                    suite_type="hidden",
                    relative_path=str(hidden_script_path.relative_to(workspace_path)) if hidden_script_path and hidden_script_path.exists() else None,
                )
            )

        if errors:
            return BaselineValidationResult(valid=False, errors=errors, warnings=warnings, outcomes=outcomes)

        assert visible_script_path is not None
        assert hidden_script_path is not None
        if visible_script_path.read_text(encoding="utf-8") == hidden_script_path.read_text(encoding="utf-8"):
            errors.append(
                BaselineValidationIssue(
                    level="error",
                    code="hidden_tests_match_visible_tests",
                    message="Hidden and visible test scripts are identical; the hidden grader is not stronger than the learner-facing checks.",
                    suite_type="hidden",
                    relative_path=str(hidden_script_path.relative_to(workspace_path)),
                )
            )

        if not errors:
            outcomes.extend(
                self._evaluate_workspace(
                    label="empty_repo",
                    workspace_root=self._make_empty_repo_copy(workspace_path, spec),
                    spec=spec,
                    visible_command=visible_command,
                    hidden_command=hidden_command,
                )
            )
            outcomes.extend(
                self._evaluate_workspace(
                    label="starter_repo",
                    workspace_root=workspace_path,
                    spec=spec,
                    visible_command=visible_command,
                    hidden_command=hidden_command,
                )
            )

        errors.extend(self._expectation_issues(outcomes, starter_type))
        warnings.extend(self._depth_warnings(outcomes))

        return BaselineValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            outcomes=outcomes,
        )

    def _evaluate_workspace(
        self,
        *,
        label: str,
        workspace_root: Path,
        spec: TaskAgentServiceSpec,
        visible_command: str,
        hidden_command: str,
    ) -> list[BaselineSuiteOutcome]:
        cleanup_after = label == "empty_repo"
        try:
            with self._running_app(workspace_root=workspace_root, spec=spec) as base_url:
                visible = self.script_runner.run_suite(
                    workspace_root=workspace_root,
                    command=visible_command,
                    base_url=base_url,
                    suite_type="visible",
                )
                hidden = self.script_runner.run_suite(
                    workspace_root=workspace_root,
                    command=hidden_command,
                    base_url=base_url,
                    suite_type="hidden",
                )
        except Exception as exc:  # noqa: BLE001
            boot_failure = str(exc)
            visible = GeneratedTestSuiteReport(
                suite_type="visible",
                command=visible_command,
                exit_code=1,
                valid=True,
                passed=False,
                tests=[
                    GeneratedTestCaseReport(
                        id=f"{label}_visible_boot",
                        title=f"{label} visible boot",
                        status="failed",
                        summary="Application failed before visible tests could run.",
                        diagnostics=[boot_failure],
                    )
                ],
                summary="Application failed before visible tests could run.",
                stderr=boot_failure,
            )
            hidden = GeneratedTestSuiteReport(
                suite_type="hidden",
                command=hidden_command,
                exit_code=1,
                valid=True,
                passed=False,
                tests=[
                    GeneratedTestCaseReport(
                        id=f"{label}_hidden_boot",
                        title=f"{label} hidden boot",
                        status="failed",
                        summary="Application failed before hidden tests could run.",
                        diagnostics=[boot_failure],
                    )
                ],
                summary="Application failed before hidden tests could run.",
                stderr=boot_failure,
            )
        finally:
            if cleanup_after:
                shutil.rmtree(workspace_root.parent, ignore_errors=True)
        return [
            BaselineSuiteOutcome(baseline=label, suite_type="visible", report=visible),
            BaselineSuiteOutcome(baseline=label, suite_type="hidden", report=hidden),
        ]

    def _expectation_issues(
        self,
        outcomes: list[BaselineSuiteOutcome],
        starter_type: StarterType,
    ) -> list[BaselineValidationIssue]:
        errors: list[BaselineValidationIssue] = []
        by_key = {(outcome.baseline, outcome.suite_type): outcome.report for outcome in outcomes}

        for suite_type in ("visible", "hidden"):
            empty_report = by_key.get(("empty_repo", suite_type))
            if empty_report is not None and empty_report.passed:
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code=f"empty_repo_{suite_type}_tests_passed",
                        message=f"The {suite_type} suite passed against an empty learner repo.",
                        baseline="empty_repo",
                        suite_type=suite_type,
                    )
                )

        starter_visible = by_key.get(("starter_repo", "visible"))
        starter_hidden = by_key.get(("starter_repo", "hidden"))
        if starter_type in {StarterType.bare_stub, StarterType.partial_implementation}:
            if starter_visible is not None and starter_visible.passed:
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="starter_visible_tests_passed_partial_repo",
                        message="The visible suite passed against a partial starter that should still require learner work.",
                        baseline="starter_repo",
                        suite_type="visible",
                    )
                )
            if starter_hidden is not None and starter_hidden.passed:
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="starter_hidden_tests_passed_partial_repo",
                        message="The hidden suite passed against a partial starter that should still fail deeper checks.",
                        baseline="starter_repo",
                        suite_type="hidden",
                    )
                )
        elif starter_type in {StarterType.working_buggy, StarterType.working_suboptimal}:
            if starter_hidden is not None and starter_hidden.passed:
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="starter_hidden_tests_passed_buggy_repo",
                        message="The hidden suite passed against a starter that is supposed to be incorrect or incomplete in meaningful ways.",
                        baseline="starter_repo",
                        suite_type="hidden",
                    )
                )
        if starter_visible is not None and not starter_visible.valid:
            errors.append(
                BaselineValidationIssue(
                    level="error",
                    code="visible_tests_invalid",
                    message="Visible tests did not emit a valid structured report.",
                    baseline="starter_repo",
                    suite_type="visible",
                )
            )
        if starter_hidden is not None and not starter_hidden.valid:
            errors.append(
                BaselineValidationIssue(
                    level="error",
                    code="hidden_tests_invalid",
                    message="Hidden tests did not emit a valid structured report.",
                    baseline="starter_repo",
                    suite_type="hidden",
                )
            )
        if starter_visible is not None and starter_hidden is not None and starter_visible.passed is False and starter_hidden.passed:
            errors.append(
                BaselineValidationIssue(
                    level="error",
                    code="hidden_tests_weaker_than_visible",
                    message="Hidden tests passed on the starter even though the visible tests already found a failure.",
                    baseline="starter_repo",
                    suite_type="hidden",
                )
            )
        return errors

    def _depth_warnings(self, outcomes: list[BaselineSuiteOutcome]) -> list[BaselineValidationIssue]:
        warnings: list[BaselineValidationIssue] = []
        by_key = {(outcome.baseline, outcome.suite_type): outcome.report for outcome in outcomes}
        starter_visible = by_key.get(("starter_repo", "visible"))
        starter_hidden = by_key.get(("starter_repo", "hidden"))
        if starter_visible is None or starter_hidden is None:
            return warnings
        if len(starter_hidden.tests) < len(starter_visible.tests):
            warnings.append(
                BaselineValidationIssue(
                    level="warning",
                    code="hidden_tests_not_deeper_than_visible",
                    message="Hidden tests currently cover fewer cases than the visible suite.",
                    baseline="starter_repo",
                    suite_type="hidden",
                )
            )
        return warnings

    def _command_target_path(self, workspace_root: Path, command: str) -> Path | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            candidate = workspace_root / token
            if candidate.exists():
                return candidate
        return None

    def _make_empty_repo_copy(self, workspace_root: Path, spec: TaskAgentServiceSpec) -> Path:
        temp_root = Path(tempfile.mkdtemp(prefix="coursegen_empty_repo_"))
        copy_root = temp_root / "workspace"
        shutil.copytree(workspace_root, copy_root)
        manifest = self.learner_studio_service._runtime_manifest(copy_root)
        starter_surface = manifest.get("learner_starter_surface") or {}
        editable_paths = starter_surface.get("primary_editable_paths") or spec.runtime_dependencies.editable_files or ["app.py"]
        for relative_path in editable_paths:
            target = copy_root / str(relative_path)
            if target.exists():
                target.write_text("", encoding="utf-8")
        return copy_root

    @contextmanager
    def _running_app(self, *, workspace_root: Path, spec: TaskAgentServiceSpec):
        host_port = self.learner_studio_service._allocate_port()
        container_name = f"course-gen-test-{uuid4().hex[:12]}"
        network_name = f"{container_name}-net"
        dependency_services = self.learner_studio_service._dependency_services(workspace_root)
        try:
            image_name = self.learner_studio_service._workspace_runtime_image_name(workspace_root)
            self.learner_studio_service._ensure_runtime_image_available(image_name)
            if dependency_services:
                self.learner_studio_service._start_runtime_support_services(
                    workspace_root,
                    network_name=network_name,
                    container_prefix=container_name,
                )
            command = [
                self.learner_studio_service.docker_binary,
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{host_port}:8000",
                "-v",
                f"{workspace_root}:/workspace",
                "-w",
                "/workspace",
                *(
                    [
                        "--network",
                        network_name,
                        "--network-alias",
                        "app",
                    ]
                    if dependency_services
                    else []
                ),
                *self.learner_studio_service._docker_env_args(
                    self.learner_studio_service._app_runtime_environment(workspace_root)
                ),
                image_name,
                "sh",
                "-lc",
                self.learner_studio_service._runtime_launch_script(
                    workspace_path=workspace_root,
                    spec=spec,
                    include_setup=True,
                ),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.learner_studio_service.build_timeout_s,
            )
            if result.returncode != 0:
                raise LearnerStudioError(
                    (result.stderr or result.stdout).strip() or "Could not start test runtime container."
                )
            base_url = f"http://{self.learner_studio_service.host}:{host_port}"
            self.learner_studio_service._wait_for_http(
                f"{base_url}{self.learner_studio_service._healthcheck_path(workspace_root, spec)}",
                container_name=container_name,
            )
            yield base_url
        finally:
            self.learner_studio_service._remove_runtime_support(
                workspace_root,
                network_name=network_name,
                container_prefix=container_name,
            )
