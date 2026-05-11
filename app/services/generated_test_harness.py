from __future__ import annotations

import json
import os
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
from app.services.task_agent_contract_surface import (
    learner_editable_paths_for_manifest,
    learner_editable_paths_for_spec,
)


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
    deliverable_id: str | None = None


class BaselineValidationIssue(BaseModel):
    level: str
    code: str
    message: str
    baseline: str | None = None
    suite_type: str | None = None
    relative_path: str | None = None
    deliverable_id: str | None = None


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

    def verify_course(
        self,
        *,
        workspace_root: str | Path,
        private_root: str | Path,
        spec: TaskAgentServiceSpec,
    ) -> BaselineValidationResult:
        """Boot the shared starter ONCE and run every deliverable's
        visible/hidden suite against the live starter and an empty-repo copy.

        Args:
            workspace_root: ``public/`` root (contains ``starter/`` and ``checks/``).
            private_root: ``private/`` root (contains ``grader/``).
            spec: authoritative task-agent spec.
        """
        public_root = Path(workspace_root).resolve()
        private_path = Path(private_root).resolve()
        shared_starter_root = public_root / "starter"

        errors: list[BaselineValidationIssue] = []
        warnings: list[BaselineValidationIssue] = []
        outcomes: list[BaselineSuiteOutcome] = []

        # Resolve per-deliverable script paths up-front; missing scripts are a
        # blocker that short-circuits the matrix.
        per_deliverable_scripts: list[tuple[str, Path, Path]] = []
        for deliverable in spec.deliverables:
            visible_path = (
                public_root / "checks" / deliverable.id / "run_visible_checks.py"
            )
            hidden_path = (
                private_path / "grader" / deliverable.id / "run_hidden_checks.py"
            )
            if not visible_path.exists():
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="visible_test_command_missing",
                        message=(
                            f"Visible test script missing for deliverable {deliverable.id} "
                            f"at public/checks/{deliverable.id}/run_visible_checks.py."
                        ),
                        suite_type="visible",
                        relative_path=f"checks/{deliverable.id}/run_visible_checks.py",
                        deliverable_id=deliverable.id,
                    )
                )
            if not hidden_path.exists():
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="hidden_test_command_missing",
                        message=(
                            f"Hidden test script missing for deliverable {deliverable.id} "
                            f"at private/grader/{deliverable.id}/run_hidden_checks.py."
                        ),
                        suite_type="hidden",
                        relative_path=f"grader/{deliverable.id}/run_hidden_checks.py",
                        deliverable_id=deliverable.id,
                    )
                )
            if visible_path.exists() and hidden_path.exists():
                per_deliverable_scripts.append((deliverable.id, visible_path, hidden_path))

        if errors:
            return BaselineValidationResult(valid=False, errors=errors, warnings=warnings, outcomes=outcomes)

        # Identical-script check per deliverable.
        for deliverable_id, visible_path, hidden_path in per_deliverable_scripts:
            try:
                if visible_path.read_text(encoding="utf-8") == hidden_path.read_text(encoding="utf-8"):
                    errors.append(
                        BaselineValidationIssue(
                            level="error",
                            code="hidden_tests_match_visible_tests",
                            message=(
                                f"Hidden and visible test scripts for deliverable {deliverable_id} are identical; "
                                "the hidden grader is not stronger than the learner-facing checks."
                            ),
                            suite_type="hidden",
                            relative_path=f"grader/{deliverable_id}/run_hidden_checks.py",
                            deliverable_id=deliverable_id,
                        )
                    )
            except OSError:
                # Read failures are already surfaced by the existence checks above.
                continue

        # Boot the shared starter and run each deliverable's suite.
        outcomes.extend(
            self._evaluate_shared_workspace(
                label="starter_repo",
                workspace_root=shared_starter_root,
                spec=spec,
                per_deliverable_scripts=per_deliverable_scripts,
                cleanup_after=False,
            )
        )

        # Build the empty-repo copy ONCE for the shared starter.
        empty_repo_root = self._make_empty_repo_copy(shared_starter_root, spec)
        outcomes.extend(
            self._evaluate_shared_workspace(
                label="empty_repo",
                workspace_root=empty_repo_root,
                spec=spec,
                per_deliverable_scripts=per_deliverable_scripts,
                cleanup_after=True,
            )
        )

        errors.extend(self._expectation_issues(outcomes, spec))
        warnings.extend(self._depth_warnings(outcomes))

        return BaselineValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            outcomes=outcomes,
        )

    def _evaluate_shared_workspace(
        self,
        *,
        label: str,
        workspace_root: Path,
        spec: TaskAgentServiceSpec,
        per_deliverable_scripts: list[tuple[str, Path, Path]],
        cleanup_after: bool,
    ) -> list[BaselineSuiteOutcome]:
        outcomes: list[BaselineSuiteOutcome] = []
        try:
            try:
                with self._running_app(workspace_root=workspace_root, spec=spec) as base_url:
                    for deliverable_id, visible_path, hidden_path in per_deliverable_scripts:
                        visible_command = f"python {visible_path}"
                        hidden_command = f"python {hidden_path}"
                        visible_report = self.script_runner.run_suite(
                            workspace_root=workspace_root,
                            command=visible_command,
                            base_url=base_url,
                            suite_type="visible",
                        )
                        hidden_report = self.script_runner.run_suite(
                            workspace_root=workspace_root,
                            command=hidden_command,
                            base_url=base_url,
                            suite_type="hidden",
                        )
                        outcomes.append(
                            BaselineSuiteOutcome(
                                baseline=label,
                                suite_type="visible",
                                report=visible_report,
                                deliverable_id=deliverable_id,
                            )
                        )
                        outcomes.append(
                            BaselineSuiteOutcome(
                                baseline=label,
                                suite_type="hidden",
                                report=hidden_report,
                                deliverable_id=deliverable_id,
                            )
                        )
            except Exception as exc:  # noqa: BLE001
                boot_failure = str(exc)
                for deliverable_id, visible_path, hidden_path in per_deliverable_scripts:
                    outcomes.append(
                        BaselineSuiteOutcome(
                            baseline=label,
                            suite_type="visible",
                            deliverable_id=deliverable_id,
                            report=GeneratedTestSuiteReport(
                                suite_type="visible",
                                command=f"python {visible_path}",
                                exit_code=1,
                                valid=True,
                                passed=False,
                                tests=[
                                    GeneratedTestCaseReport(
                                        id=f"{label}_{deliverable_id}_visible_boot",
                                        title=f"{label} {deliverable_id} visible boot",
                                        status="failed",
                                        summary="Application failed before visible tests could run.",
                                        diagnostics=[boot_failure],
                                    )
                                ],
                                summary="Application failed before visible tests could run.",
                                stderr=boot_failure,
                            ),
                        )
                    )
                    outcomes.append(
                        BaselineSuiteOutcome(
                            baseline=label,
                            suite_type="hidden",
                            deliverable_id=deliverable_id,
                            report=GeneratedTestSuiteReport(
                                suite_type="hidden",
                                command=f"python {hidden_path}",
                                exit_code=1,
                                valid=True,
                                passed=False,
                                tests=[
                                    GeneratedTestCaseReport(
                                        id=f"{label}_{deliverable_id}_hidden_boot",
                                        title=f"{label} {deliverable_id} hidden boot",
                                        status="failed",
                                        summary="Application failed before hidden tests could run.",
                                        diagnostics=[boot_failure],
                                    )
                                ],
                                summary="Application failed before hidden tests could run.",
                                stderr=boot_failure,
                            ),
                        )
                    )
        finally:
            if cleanup_after:
                shutil.rmtree(workspace_root.parent, ignore_errors=True)
        return outcomes

    def _expectation_issues(
        self,
        outcomes: list[BaselineSuiteOutcome],
        spec: TaskAgentServiceSpec,
    ) -> list[BaselineValidationIssue]:
        """Per-deliverable expectations for the shared-starter matrix.

        - Any empty-repo pass is broken (suite is not anchored to the real surface).
        - For ``empty`` or ``partial`` starters, every deliverable's visible AND
          hidden suite must FAIL against the shared starter. A pass means the
          test does not measure the learner work and emits the single error code
          ``starter_suite_passed_pre_implementation``.
        - ``hidden_tests_weaker_than_visible`` still fires when visible failed
          but hidden passed against the starter, attributed per deliverable.
        - Invalid (unparseable) reports are surfaced once per (deliverable, suite).
        """
        errors: list[BaselineValidationIssue] = []
        starter_type = spec.runtime_dependencies.starter_type
        # Group outcomes by (deliverable_id, baseline, suite_type) for attribution.
        by_key = {
            (outcome.deliverable_id, outcome.baseline, outcome.suite_type): outcome.report
            for outcome in outcomes
        }
        deliverable_ids = [d.id for d in spec.deliverables]

        for deliverable_id in deliverable_ids:
            for suite_type in ("visible", "hidden"):
                empty_report = by_key.get((deliverable_id, "empty_repo", suite_type))
                if empty_report is not None and empty_report.passed:
                    errors.append(
                        BaselineValidationIssue(
                            level="error",
                            code=f"empty_repo_{suite_type}_tests_passed",
                            message=(
                                f"The {suite_type} suite for deliverable {deliverable_id} "
                                "passed against an empty learner repo."
                            ),
                            baseline="empty_repo",
                            suite_type=suite_type,
                            deliverable_id=deliverable_id,
                        )
                    )

            starter_visible = by_key.get((deliverable_id, "starter_repo", "visible"))
            starter_hidden = by_key.get((deliverable_id, "starter_repo", "hidden"))

            # The starter is always pre-implementation: empty or partial. A pass
            # there means the suite does not measure the gap between starter and
            # the authored solution.
            if starter_type in {StarterType.empty, StarterType.partial}:
                for suite_type, report in (("visible", starter_visible), ("hidden", starter_hidden)):
                    if report is not None and report.passed:
                        errors.append(
                            BaselineValidationIssue(
                                level="error",
                                code="starter_suite_passed_pre_implementation",
                                message=(
                                    f"The {suite_type} suite for deliverable {deliverable_id} "
                                    "passed against the pre-implementation shared starter; the suite "
                                    "must measure work the learner has not done yet."
                                ),
                                baseline="starter_repo",
                                suite_type=suite_type,
                                deliverable_id=deliverable_id,
                            )
                        )

            if starter_visible is not None and not starter_visible.valid:
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="visible_tests_invalid",
                        message=(
                            f"Visible tests for deliverable {deliverable_id} "
                            "did not emit a valid structured report."
                        ),
                        baseline="starter_repo",
                        suite_type="visible",
                        deliverable_id=deliverable_id,
                    )
                )
            if starter_hidden is not None and not starter_hidden.valid:
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="hidden_tests_invalid",
                        message=(
                            f"Hidden tests for deliverable {deliverable_id} "
                            "did not emit a valid structured report."
                        ),
                        baseline="starter_repo",
                        suite_type="hidden",
                        deliverable_id=deliverable_id,
                    )
                )
            if (
                starter_visible is not None
                and starter_hidden is not None
                and starter_visible.passed is False
                and starter_hidden.passed
            ):
                errors.append(
                    BaselineValidationIssue(
                        level="error",
                        code="hidden_tests_weaker_than_visible",
                        message=(
                            f"Hidden tests for deliverable {deliverable_id} passed on the starter "
                            "even though the visible tests already found a failure."
                        ),
                        baseline="starter_repo",
                        suite_type="hidden",
                        deliverable_id=deliverable_id,
                    )
                )
        return errors

    def _depth_warnings(self, outcomes: list[BaselineSuiteOutcome]) -> list[BaselineValidationIssue]:
        warnings: list[BaselineValidationIssue] = []
        by_key = {
            (outcome.deliverable_id, outcome.baseline, outcome.suite_type): outcome.report
            for outcome in outcomes
        }
        deliverable_ids: set[str] = {
            outcome.deliverable_id for outcome in outcomes if outcome.deliverable_id is not None
        }
        for deliverable_id in deliverable_ids:
            starter_visible = by_key.get((deliverable_id, "starter_repo", "visible"))
            starter_hidden = by_key.get((deliverable_id, "starter_repo", "hidden"))
            if starter_visible is None or starter_hidden is None:
                continue
            if len(starter_hidden.tests) < len(starter_visible.tests):
                warnings.append(
                    BaselineValidationIssue(
                        level="warning",
                        code="hidden_tests_not_deeper_than_visible",
                        message=(
                            f"Hidden tests for deliverable {deliverable_id} currently cover "
                            "fewer cases than the visible suite."
                        ),
                        baseline="starter_repo",
                        suite_type="hidden",
                        deliverable_id=deliverable_id,
                    )
                )
        return warnings

    def _make_empty_repo_copy(self, workspace_root: Path, spec: TaskAgentServiceSpec) -> Path:
        temp_root = Path(tempfile.mkdtemp(prefix="coursegen_empty_repo_"))
        copy_root = temp_root / "workspace"
        shutil.copytree(workspace_root, copy_root)
        manifest = self.learner_studio_service._runtime_manifest(copy_root)
        editable_paths = learner_editable_paths_for_manifest(manifest)
        if not editable_paths:
            # Shared starter root no longer carries a per-deliverable manifest;
            # fall back to the authoritative spec to find editable paths.
            editable_paths = learner_editable_paths_for_spec(spec)
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
