from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.domain.registry import PackageType
from app.domain.workflow import (
    FailureContext,
    FailureContextDependencyContract,
    FailureContextVerifiedRuntime,
    FailureContextVerifiedRuntimeFile,
    MaterializeBundleRequest,
    WorkflowNodeKind,
)
from app.services.artifact_materializer import ArtifactMaterializer
from app.services.assignment_design_inference import GenerationIntake, infer_assignment_design
from app.services.openai_course_planner import OpenAICoursePlanner
from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService
from app.services.openai_test_script_authoring import OpenAITestScriptAuthoringService
from app.services.openai_task_agent_authoring import (
    OpenAITaskAgentAuthoringService,
    TaskAgentCustomization,
)
from app.services.openai_runtime_support import parse_structured_openai_response_with_hard_timeout
from app.services.starter_authoring_payload import build_starter_authoring_payload
from app.services.task_agent_starter_templates import (
    HIDDEN_MANIFEST_PATH,
    RUNTIME_INSTALL_SCRIPT_PATH,
    RUNTIME_RUN_SCRIPT_PATH,
    RUNTIME_VERIFY_SCRIPT_PATH,
    build_task_agent_starter_files,
)
from app.services.task_agent_workspace_authoring import TaskAgentWorkspaceAuthoringService
from app.services.workflow_service import WorkflowService
from app.storage.sqlite_store import SQLiteWorkflowStore


class _FakeUsage:
    input_tokens = 10
    output_tokens = 20
    total_tokens = 30
    input_tokens_details = type("InputDetails", (), {"cached_tokens": 0})()
    output_tokens_details = type("OutputDetails", (), {"reasoning_tokens": 0})()


class _FakeParsedResponse:
    def __init__(self, parsed):
        self.output_parsed = parsed
        self.usage = _FakeUsage()


class _FakeResponsesAPI:
    def __init__(self, parsed_response):
        self.parsed_response = parsed_response
        self.parse_calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return self.parsed_response


class _FakeClient:
    def __init__(self, parsed_response):
        self.responses = _FakeResponsesAPI(parsed_response)


class _QueuedFakeResponsesAPI:
    def __init__(self, parsed_responses):
        self.parsed_responses = list(parsed_responses)
        self.parse_calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if not self.parsed_responses:
            raise AssertionError("No queued parsed response left for this test.")
        return _FakeParsedResponse(parsed=self.parsed_responses.pop(0))


class _QueuedFakeClient:
    def __init__(self, parsed_responses):
        self.responses = _QueuedFakeResponsesAPI(parsed_responses)


def _dependency_contract_payload(**overrides):
    payload = {
        "manifest_paths": [],
        "lockfile_paths": [],
        "toolchain_paths": [],
        "build_support_paths": [],
        "reproducibility_mode": None,
    }
    payload.update(overrides)
    return payload


def _materialized_run(temp_dir: str):
    store = SQLiteWorkflowStore(db_path=f"{temp_dir}/test.db")
    workflow_service = WorkflowService(
        store,
        materializer=ArtifactMaterializer(base_dir=f"{temp_dir}/generated"),
    )
    intake = GenerationIntake(
        title="Inventory Reservation Service",
        problem_statement=(
            "Build a multi-warehouse inventory reservation service with FastAPI, Postgres, and Redis. "
            "Keep reservations correct under concurrency, retries, and stock transfers."
        ),
        package_type_hint=PackageType.progressive_codebase_course,
    )
    inferred = infer_assignment_design(
        title=intake.title,
        problem_statement=intake.problem_statement,
        package_type_hint=intake.package_type_hint,
    )
    assert inferred.design_spec is not None
    # In production, primary_editable_paths is authored by the OpenAI task-agent
    # call (model picks files based on the chosen stack). These tests bypass that
    # call, so seed a FastAPI-shaped editable file at the design-spec level
    # before the run is created — every downstream artifact (manifest, brief,
    # starter surface) reads from this.
    inferred.design_spec.runtime_dependencies.editable_files = ["app.py"]
    from app.domain.task_agent import DeliverableSpec
    planner_deliverables = [
        DeliverableSpec(
            id=f"deliverable_{index}",
            title=f"Inventory reservation deliverable {index}",
            objective=f"Build deliverable {index} of the inventory reservation surface.",
            learning_outcomes=[],
            overlay_ids=[],
        )
        for index in range(1, 5)
    ]
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake,
        design_spec=inferred.design_spec,
        execute_nodes=False,
        planner_deliverables=planner_deliverables,
    )
    workflow_service.materialize_run(run.id, MaterializeBundleRequest(overwrite=True))
    run = workflow_service.get_run(run.id)
    assert run is not None
    run, _ = TaskAgentWorkspaceAuthoringService().author_workspace(run)
    return run


class AuthoringPayloadTests(unittest.TestCase):
    def test_runtime_entrypoint_heuristic_is_not_used_for_editable_files(self) -> None:
        """The initial assignment design must not hardcode a language-specific
        primary editable path. The model is responsible for authoring
        `learner_starter_surface.primary_editable_paths` based on the actual
        stack and the files it produces.
        """
        from app.services import assignment_design_inference
        from app.services.assignment_design_inference import infer_assignment_design

        self.assertFalse(
            hasattr(assignment_design_inference, "runtime_entrypoint_for_stack"),
            "runtime_entrypoint_for_stack is a banned language heuristic; "
            "delete it and let the model author primary_editable_paths instead.",
        )

        inferred = infer_assignment_design(
            title="Payment intents",
            problem_statement="Build a payment intent service.",
            implementation_language="java",
            application_framework="spring-boot",
            primary_database="postgres",
            cache_backend="redis",
        )
        assert inferred.design_spec is not None
        self.assertEqual(
            list(inferred.design_spec.runtime_dependencies.editable_files),
            [],
            "Initial design should not pre-commit a language-specific editable file; "
            "leave it empty for the authoring model to fill in.",
        )

    def test_starter_surface_customization_accepts_primary_editable_paths(self) -> None:
        """The customization patch must let the model author the primary editable
        paths so they reflect the real chosen stack and authored layout.
        """
        from app.services.openai_task_agent_authoring import StarterSurfaceCustomization

        surface = StarterSurfaceCustomization(
            starter_summary="Implement the dispute ledger flow.",
            primary_editable_paths=[
                "src/main/java/com/coursegen/payments/DisputeLedgerApplication.java"
            ],
        )

        self.assertEqual(
            surface.primary_editable_paths,
            ["src/main/java/com/coursegen/payments/DisputeLedgerApplication.java"],
        )

    def test_repo_authoring_writes_shebang_files_as_executable(self) -> None:
        import os
        import stat

        service = OpenAIStarterRepoAuthoringService(enabled=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            target = workspace_root / "mvnw"
            content = "#!/bin/sh\n# Apache Maven Wrapper\nexec java -classpath ...\n"

            updated = service._write_if_changed(target, content, workspace_root)

            self.assertEqual(updated, ["mvnw"])
            mode = os.stat(target).st_mode
            self.assertTrue(
                mode & stat.S_IXUSR,
                f"expected mvnw to be executable for owner; mode={oct(mode)}",
            )
            self.assertTrue(
                mode & stat.S_IXGRP,
                f"expected mvnw to be executable for group; mode={oct(mode)}",
            )
            self.assertTrue(
                mode & stat.S_IXOTH,
                f"expected mvnw to be executable for other; mode={oct(mode)}",
            )

    def test_repo_authoring_writes_non_shebang_files_without_execute_bit(self) -> None:
        import os
        import stat

        service = OpenAIStarterRepoAuthoringService(enabled=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            target = workspace_root / "pom.xml"
            content = "<?xml version=\"1.0\"?>\n<project></project>\n"

            service._write_if_changed(target, content, workspace_root)

            mode = os.stat(target).st_mode
            self.assertFalse(
                mode & stat.S_IXUSR,
                f"expected pom.xml NOT to be executable; mode={oct(mode)}",
            )

    def test_repo_authoring_uses_structured_response_parsing(self) -> None:
        client = _FakeClient(
            _FakeParsedResponse(
                parsed=type(
                    "RepoBundle",
                    (),
                    {
                        "files": [type("RepoFile", (), {"path": "src/main.rs", "content": "fn main() {}\n"})()],
                        "dependency_contract": _dependency_contract_payload(
                            manifest_paths=["Cargo.toml"],
                            lockfile_paths=["Cargo.lock"],
                        ),
                        "notes": ["ok"],
                    },
                )()
            )
        )
        service = OpenAIStarterRepoAuthoringService(enabled=True, client_factory=lambda **_: client)

        bundle, usage = service._generate_bundle(
            client,
            model_id="gpt-5.4",
            api_key="test",
            base_url=None,
            payload={"hello": "world"},
            workflow_run_id="run_test",
            deliverable_id="deliverable_1",
        )

        assert bundle.files[0].path == "src/main.rs"
        assert usage is not None
        assert client.responses.parse_calls
        assert client.responses.parse_calls[0]["text_format"].__name__ == "_GeneratedRepoBundle"
        assert client.responses.parse_calls[0]["timeout"] == service.request_timeout_s
        system_prompt = client.responses.parse_calls[0]["input"][0]["content"]
        assert "Return the complete current snapshot" in system_prompt
        assert "Do not use `verify.sh` for formatter, linter, or style-only gates" in system_prompt
        assert "author the install script so it can generate or refresh that lockfile deterministically" in system_prompt

    def test_test_authoring_uses_structured_response_parsing(self) -> None:
        client = _FakeClient(
            _FakeParsedResponse(
                parsed=type(
                    "GeneratedScripts",
                    (),
                    {
                        "visible_script": "print('visible')\n",
                        "hidden_script": "print('hidden')\n",
                        "notes": ["ok"],
                    },
                )()
            )
        )
        service = OpenAITestScriptAuthoringService(enabled=True, client_factory=lambda **_: client)

        scripts, usage = service._generate_scripts(
            client,
            model_id="gpt-5.4",
            api_key="test",
            base_url=None,
            payload={"hello": "world"},
            workflow_run_id="run_test",
            deliverable_id="deliverable_1",
        )

        assert "visible" in scripts.visible_script
        assert usage is not None
        assert client.responses.parse_calls
        assert client.responses.parse_calls[0]["timeout"] == service.request_timeout_s

    def test_task_agent_authoring_uses_structured_response_parsing(self) -> None:
        client = _FakeClient(
            _FakeParsedResponse(
                parsed=TaskAgentCustomization(
                    summary="Grounded summary",
                    public_endpoints=[],
                    deliverables=[],
                    notes=["ok"],
                )
            )
        )
        service = OpenAITaskAgentAuthoringService(enabled=True, client_factory=lambda **_: client)

        response = service._create_response_with_retries(
            client,
            model="gpt-5.4",
            api_key="test",
            base_url=None,
            input=[{"role": "user", "content": "{}"}],
            temperature=0.2,
            text_format=TaskAgentCustomization,
        )

        assert response.output_parsed is not None
        assert client.responses.parse_calls
        assert client.responses.parse_calls[0]["timeout"] == service.request_timeout_s

    def test_repo_authoring_without_client_factory_uses_hard_timeout_helper(self) -> None:
        service = OpenAIStarterRepoAuthoringService(enabled=True, client_factory=None)
        with patch(
            "app.services.openai_repo_authoring.parse_structured_openai_response_with_hard_timeout",
            return_value=_FakeParsedResponse(
                parsed=type(
                    "RepoBundle",
                    (),
                    {
                        "files": [type("RepoFile", (), {"path": "src/main.rs", "content": "fn main() {}\n"})()],
                        "dependency_contract": _dependency_contract_payload(),
                        "notes": [],
                    },
                )()
            ),
        ) as mocked:
            response = service._create_response_with_retries(
                None,
                model="gpt-5.4",
                api_key="test-key",
                base_url="https://example.invalid",
                input=[{"role": "user", "content": "{}"}],
                temperature=0.1,
                workflow_run_id="run_test",
                deliverable_id="deliverable_1",
                text_format=type(
                    "DummyBundle",
                    (),
                    {"model_validate": staticmethod(lambda payload: payload)},
                ),
            )

        assert response.output_parsed is not None
        mocked.assert_called_once()

    def test_course_planner_without_client_factory_uses_hard_timeout_helper(self) -> None:
        planner = OpenAICoursePlanner(enabled=True, client_factory=None)

        class _DummyTextFormat:
            @staticmethod
            def model_validate(payload):
                return payload

        with patch(
            "app.services.openai_course_planner.parse_structured_openai_response_with_hard_timeout",
            return_value=_FakeParsedResponse(parsed={"title": "x"}),
        ) as mocked:
            response = planner._create_response_with_retries(
                object(),
                api_key="test-key",
                base_url="https://example.invalid",
                model="gpt-5.4",
                input=[{"role": "user", "content": "{}"}],
                temperature=0.2,
                text_format=_DummyTextFormat,
            )

        assert response.output_parsed == {"title": "x"}
        mocked.assert_called_once()

    def test_repo_authoring_prompt_payload_includes_current_repo_and_runtime_protocol_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["learner_starter_surface"] = {
                **(manifest.get("learner_starter_surface") or {}),
                "primary_editable_paths": ["src/main.rs"],
            }
            manifest["runtime_plan"] = {
                **(manifest.get("runtime_plan") or {}),
                "package_manager": "cargo",
            }
            manifest["dependency_contract"] = _dependency_contract_payload(
                manifest_paths=["Cargo.toml"],
                lockfile_paths=["Cargo.lock"],
            )

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
            (starter_root / "Cargo.toml").write_text(
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                encoding="utf-8",
            )
            (starter_root / "Cargo.lock").write_text("# lockfile\n", encoding="utf-8")
            (starter_root / "target" / "debug").mkdir(parents=True, exist_ok=True)
            (starter_root / "target" / "debug" / "demo").write_text("binary\n", encoding="utf-8")
            (starter_root / "Dockerfile").write_text("FROM rust:1.86-bookworm\n", encoding="utf-8")
            (starter_root / RUNTIME_INSTALL_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo fetch\n",
                encoding="utf-8",
            )
            (starter_root / RUNTIME_RUN_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo run\n",
                encoding="utf-8",
            )
            (starter_root / "checks").mkdir(parents=True, exist_ok=True)
            (starter_root / "checks" / "run_visible_checks.py").write_text("# generated\n", encoding="utf-8")
            (starter_root / ".coursegen" / "grader").mkdir(parents=True, exist_ok=True)
            (starter_root / ".coursegen" / "grader" / "run_hidden_checks.py").write_text("# hidden\n", encoding="utf-8")

            service = OpenAIStarterRepoAuthoringService(enabled=False)
            payload = service._prompt_payload(
                run,
                deliverable_id=deliverable.id,
                starter_root=starter_root,
                manifest=manifest,
                failure_context=None,
            )

        learner_files = payload["current_files"]
        dependency_files = payload["dependency_contract_files"]
        runtime_files = payload["runtime_protocol_files"]
        assert learner_files == {"src/main.rs": "// learner repo file\n"}
        assert dependency_files == {
            "Cargo.toml": "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"
        }
        assert "Dockerfile" in runtime_files
        assert RUNTIME_INSTALL_SCRIPT_PATH in runtime_files
        assert RUNTIME_RUN_SCRIPT_PATH in runtime_files
        assert "README.md" not in learner_files
        assert HIDDEN_MANIFEST_PATH not in learner_files
        assert "Cargo.lock" not in dependency_files
        assert "target/debug/demo" not in learner_files
        assert "target/debug/demo" not in dependency_files
        assert "checks/run_visible_checks.py" not in runtime_files

    def test_repo_authoring_prompt_payload_includes_dependency_contract_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            failure_context = FailureContext(
                source_node_kind=WorkflowNodeKind.authoring_runtime,
                source_node_attempt=1,
                source_summary="sandbox failed",
                dependency_contracts=[
                    FailureContextDependencyContract(
                        deliverable_id=deliverable.id,
                        starter_root=str(starter_root),
                        implementation_language="rust",
                        language_version="1.82",
                        application_framework="axum",
                        framework_version="0.8.9",
                        package_manager="cargo",
                        container_image="rust:1.82-bookworm",
                        root_files=["Cargo.toml", "src/main.rs"],
                        expected_manifest_paths=["Cargo.toml"],
                        present_manifest_paths=["Cargo.toml"],
                        expected_lockfile_paths=["Cargo.lock"],
                        present_lockfile_paths=[],
                        expected_toolchain_paths=["rust-toolchain.toml", "rust-toolchain"],
                        present_toolchain_paths=[],
                        expected_build_support_paths=[],
                        present_build_support_paths=[],
                        runtime_protocol_paths_present=["Dockerfile", RUNTIME_INSTALL_SCRIPT_PATH],
                        runtime_bundle_complete=False,
                    )
                ],
                previously_verified_runtime=FailureContextVerifiedRuntime(
                    source_node_kind=WorkflowNodeKind.authoring_runtime,
                    source_node_attempt=3,
                    verified_at=datetime.now(UTC),
                    source_deliverable_id=deliverable.id,
                    passed_deliverables=[deliverable.id],
                    current_failed_deliverables=[deliverable.id],
                    verified_files=[
                        FailureContextVerifiedRuntimeFile(
                            path="Dockerfile",
                            sha256="abc123",
                            role="runtime_protocol",
                            content="FROM rust:1.82-bookworm\n",
                            preserve_verbatim=True,
                        ),
                        FailureContextVerifiedRuntimeFile(
                            path="Cargo.toml",
                            sha256="def456",
                            role="dependency_contract",
                            content="[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
                            preserve_verbatim=True,
                        ),
                    ],
                    dependency_contracts=[],
                ),
            )

            service = OpenAIStarterRepoAuthoringService(enabled=False)
            payload = service._prompt_payload(
                run,
                deliverable_id=deliverable.id,
                starter_root=starter_root,
                manifest=manifest,
                failure_context=failure_context,
            )

        dependency_contracts = payload["failure_context"]["dependency_contracts"]
        assert dependency_contracts[0]["package_manager"] == "cargo"
        assert dependency_contracts[0]["expected_lockfile_paths"] == ["Cargo.lock"]
        assert dependency_contracts[0]["present_lockfile_paths"] == []
        assert dependency_contracts[0]["runtime_bundle_complete"] is False
        verified_runtime = payload["failure_context"]["previously_verified_runtime"]
        assert verified_runtime["passed_deliverables"] == [deliverable.id]
        assert verified_runtime["current_failed_deliverables"] == [deliverable.id]
        assert verified_runtime["verified_files"][0]["path"] == "Dockerfile"
        assert verified_runtime["verified_files"][0]["content"] == "FROM rust:1.82-bookworm\n"

    def test_repo_authoring_prompt_includes_last_attempted_runtime(self) -> None:
        """When repair runs after a partial-success attempt (e.g. booted, only
        contract failed), the prompt payload must carry `last_attempted_runtime`
        with stage outcomes and the runtime protocol files so the model can
        preserve what already worked.
        """
        from datetime import UTC, datetime
        from app.domain.workflow import FailureContextLastAttemptedRuntime
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            assert spec is not None
            deliverable = spec.deliverables[0]
            workspace = run.artifacts.workspace_snapshot
            assert workspace is not None
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            failure_context = FailureContext(
                source_node_kind=WorkflowNodeKind.authoring_runtime,
                source_node_attempt=2,
                source_summary="Public checks failed on attempt 1; image_build failed on attempt 2.",
                last_attempted_runtime=FailureContextLastAttemptedRuntime(
                    source_node_kind=WorkflowNodeKind.authoring_runtime,
                    source_node_attempt=1,
                    attempted_at=datetime.now(UTC),
                    source_deliverable_id=deliverable.id,
                    stage_outcomes={
                        "image_build": "passed",
                        "install": "passed",
                        "verify": "passed",
                        "boot": "passed",
                        "contract": "failed",
                    },
                    verified_files=[
                        FailureContextVerifiedRuntimeFile(
                            path="Dockerfile",
                            sha256="abc123",
                            role="runtime_protocol",
                            content="FROM eclipse-temurin:21\n",
                            preserve_verbatim=True,
                        ),
                    ],
                ),
            )

            service = OpenAIStarterRepoAuthoringService(enabled=False)
            payload = service._prompt_payload(
                run,
                deliverable_id=deliverable.id,
                starter_root=starter_root,
                manifest=manifest,
                failure_context=failure_context,
            )

        last_attempted = payload["failure_context"]["last_attempted_runtime"]
        self.assertIsNotNone(last_attempted)
        self.assertEqual(last_attempted["stage_outcomes"]["boot"], "passed")
        self.assertEqual(last_attempted["stage_outcomes"]["contract"], "failed")
        self.assertEqual(last_attempted["verified_files"][0]["path"], "Dockerfile")
        self.assertEqual(last_attempted["verified_files"][0]["content"], "FROM eclipse-temurin:21\n")
        self.assertTrue(last_attempted["verified_files"][0]["preserve_verbatim"])

    def test_repo_authoring_prompt_explains_harness_provided_sidecars(self) -> None:
        """Dependency services like postgres and redis are provided by the
        harness as separate sidecar containers on a shared Docker network,
        reachable from the app container via the service_id as the hostname
        (e.g. `postgres:5432`, `redis:6379`). Without this directive the model
        keeps authoring install/run scripts that try to install or start those
        services locally (e.g. running `initdb` in run.sh, or `docker run
        postgres` in install.sh) which always fails because the app container
        has neither.
        """
        import inspect
        from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService

        source = inspect.getsource(OpenAIStarterRepoAuthoringService)
        # The prompt must explicitly tell the model the harness provides the
        # dependency services as sidecars.
        self.assertIn(
            "sidecar",
            source.lower(),
            "Repo authoring system prompt must call out that dependency services are harness-provided sidecars.",
        )
        # The directive must name the connectivity contract (service_id as hostname)
        # so the model knows how the app reaches them.
        self.assertTrue(
            "postgres:5432" in source or "service_id" in source,
            "Repo authoring system prompt must explain that dependency services are reachable by service_id hostname (e.g. postgres:5432).",
        )
        # The directive must forbid local installation/startup of dependency services.
        self.assertTrue(
            "initdb" in source.lower() or "do not install" in source.lower() or "do not start" in source.lower(),
            "Repo authoring system prompt must forbid the model from installing or starting dependency services inside the app container.",
        )

    def test_repo_authoring_prompt_points_at_pass8_diagnostic_fields(self) -> None:
        """The repair-side system prompt must explicitly call out the Pass-8
        per-deliverable diagnostic surface so the model knows to read
        ``stdout_excerpt`` (framework logs), ``exit_state.oom_killed``
        (raise memory cap), ``sidecar_diagnostics`` (postgres/redis
        stderr — check first on 'connection refused'), and
        ``http_response`` (contract-probe response bodies). Without
        these pointers the model keeps treating the headline ``error``
        as authoritative and ignores the much richer structured fields.
        """
        import inspect
        from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService

        source = inspect.getsource(OpenAIStarterRepoAuthoringService)
        lowered = source.lower()
        self.assertIn("stdout_excerpt", lowered)
        self.assertIn("exit_state", lowered)
        self.assertIn("oom_killed", lowered)
        self.assertIn("sidecar_diagnostics", lowered)
        self.assertIn("http_response", lowered)
        # The prompt should explicitly steer the model to read the
        # structured fields instead of the headline label.
        self.assertTrue(
            "headline" in lowered or "label" in lowered,
            "Prompt must explain that the headline `error` is just a label "
            "and the structured fields are the canonical diagnostic.",
        )

    def test_repo_authoring_prompt_warns_about_structured_output_binary_constraint(self) -> None:
        """Structured outputs can only carry text. The system prompt must warn
        the model that binary-wrapper files (e.g. `.mvn/wrapper/maven-wrapper.jar`,
        `gradle/wrapper/gradle-wrapper.jar`) cannot be bundled, so the install
        script must either generate them or use the system-installed tool.
        Without this warning the model keeps writing install.sh to invoke
        `./mvnw` and the build fails because the binary jar is missing.
        """
        import inspect
        from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService

        source = inspect.getsource(OpenAIStarterRepoAuthoringService)
        # The prompt should call out that structured outputs cannot carry binaries.
        self.assertIn(
            "binary",
            source.lower(),
            "Repo authoring system prompt must explain that structured outputs cannot carry binary assets.",
        )
        # Concrete examples of common build-wrapper binary jars should be named.
        self.assertTrue(
            "maven-wrapper.jar" in source or "gradle-wrapper.jar" in source,
            "Repo authoring system prompt should name common binary wrappers (maven-wrapper.jar, gradle-wrapper.jar) the model must not assume it can bundle.",
        )

    def test_test_authoring_system_prompt_uses_collapsed_starter_type_contract(self) -> None:
        """The test-script authoring system prompt must reflect the Pass-1
        collapse to `empty | partial`. Both starter variants leave the shared
        starter without business-logic implementation, so visible AND hidden
        suites are expected to FAIL the untouched shared starter. The legacy
        `partial_implementation` / `working_buggy` directives must be gone.
        """
        import inspect
        from app.services.openai_test_script_authoring import (
            OpenAITestScriptAuthoringService,
        )

        source = inspect.getsource(OpenAITestScriptAuthoringService)
        # Legacy four-bucket vocabulary must be gone.
        self.assertNotIn(
            "partial_implementation",
            source,
            "Test-script authoring prompt must not reference the retired `partial_implementation` starter type.",
        )
        self.assertNotIn(
            "working_buggy",
            source,
            "Test-script authoring prompt must not reference the retired `working_buggy` starter type.",
        )
        self.assertNotIn(
            "working_suboptimal",
            source,
            "Test-script authoring prompt must not reference the retired `working_suboptimal` starter type.",
        )
        # The prompt must say both visible and hidden suites are expected to
        # fail the untouched shared starter, since neither `empty` nor
        # `partial` ships business-logic implementations.
        lowered = source.lower()
        self.assertIn(
            "untouched shared starter",
            lowered,
            "Test-script authoring prompt must call out that the untouched shared starter has no business-logic implementation.",
        )
        self.assertTrue(
            "must fail" in lowered or "must FAIL".lower() in lowered,
            "Test-script authoring prompt must require visible AND hidden suites to fail against the untouched shared starter.",
        )
        self.assertIn(
            "visible and hidden",
            lowered,
            "Test-script authoring prompt must apply the fail-against-starter directive to both visible AND hidden scripts.",
        )

    def test_test_authoring_prompt_payload_carries_course_starter_type(self) -> None:
        """The user payload for the test-script authoring LLM call must include
        `course_starter_type` so the model can read the course-level `empty`
        vs `partial` setting explicitly (rather than guessing from manifest
        shape).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            service = OpenAITestScriptAuthoringService(enabled=False)
            payload = service._prompt_payload(
                run,
                starter_root=starter_root,
                manifest=manifest,
                failure_context=None,
            )

        self.assertIn("course_starter_type", payload)
        self.assertEqual(
            payload["course_starter_type"],
            spec.runtime_dependencies.starter_type.value,
        )

    def test_repo_authoring_system_prompt_directs_model_to_preserve_passing_stage_files(self) -> None:
        """The system prompt for repo authoring must include explicit guidance
        about `last_attempted_runtime.stage_outcomes`. Without it, the model
        has no instruction to preserve files implicated only in stages that
        already passed.
        """
        import inspect
        from app.services.openai_repo_authoring import OpenAIStarterRepoAuthoringService

        source = inspect.getsource(OpenAIStarterRepoAuthoringService)
        self.assertIn(
            "last_attempted_runtime",
            source,
            "Repo authoring system prompt must reference last_attempted_runtime so the model knows which files are pinned by previously-passing stages.",
        )
        self.assertIn(
            "stage_outcomes",
            source,
            "Repo authoring system prompt must reference stage_outcomes so the model can scope edits to the actually-failing stage.",
        )

    def test_repo_authoring_system_prompt_uses_collapsed_starter_type_contract(self) -> None:
        """The progressive shared-repo authoring prompt must reflect the
        Pass-1 collapse: course-level `course_starter_type` is `empty` or
        `partial`. The prompt must direct the model to leave every business
        endpoint as an explicit unimplemented stub for `partial`, and to
        author only boot scaffolding for `empty`. Legacy four-bucket terms
        (`working_buggy`, etc.) must be gone.
        """
        import inspect
        from app.services.openai_repo_authoring import (
            OpenAIStarterRepoAuthoringService,
        )

        source = inspect.getsource(OpenAIStarterRepoAuthoringService)

        self.assertNotIn(
            "partial_implementation",
            source,
            "Repo authoring prompt must not reference the retired `partial_implementation` starter type.",
        )
        self.assertNotIn(
            "working_buggy",
            source,
            "Repo authoring prompt must not reference the retired `working_buggy` starter type.",
        )
        self.assertNotIn(
            "working_suboptimal",
            source,
            "Repo authoring prompt must not reference the retired `working_suboptimal` starter type.",
        )
        # Strong `partial` directive: explicit unimplemented stubs.
        self.assertIn(
            "explicit unimplemented stub",
            source,
            "Repo authoring prompt must require `partial` starters to leave business endpoints as explicit unimplemented stubs.",
        )
        self.assertIn(
            "course_starter_type",
            source,
            "Repo authoring prompt must reference the course-level `course_starter_type` payload key.",
        )
        # Empty-starter contract.
        lowered = source.lower()
        self.assertIn(
            "health endpoint",
            lowered,
            "Repo authoring prompt must describe the boot/health contract that both `empty` and `partial` starters must satisfy.",
        )

    def test_repo_authoring_progressive_payload_carries_course_starter_type(self) -> None:
        """The user payload for the progressive shared-repo authoring LLM call
        must include `course_starter_type` so the model knows whether to author
        an `empty` skeleton or a `partial` scaffold with stub endpoints.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            public_root = Path(workspace.public_dir)

            service = OpenAIStarterRepoAuthoringService(enabled=False)
            payload = service._progressive_prompt_payload(
                run=run,
                public_root=public_root,
                deliverable_ids=[spec.deliverables[0].id],
                failure_context=None,
            )

        self.assertIn("course_starter_type", payload)
        self.assertEqual(
            payload["course_starter_type"],
            spec.runtime_dependencies.starter_type.value,
        )

    def test_repo_authoring_shared_codebase_uses_single_shared_repo_bundle_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            public_root = Path(workspace.public_dir)
            workspace_root = Path(workspace.root_dir)
            for deliverable_id in ["deliverable_1", "deliverable_2"]:
                manifest_path = (
                    workspace_root
                    / "private"
                    / "grader"
                    / deliverable_id
                    / "deliverable.json"
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["learner_starter_surface"] = {
                    **(manifest.get("learner_starter_surface") or {}),
                    "primary_editable_paths": ["src/shared_stage.txt"],
                }
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            (public_root / "starter" / "src").mkdir(parents=True, exist_ok=True)
            (public_root / "starter" / "src" / "shared_stage.txt").write_text(
                "stage-one\n",
                encoding="utf-8",
            )

            queued_client = _QueuedFakeClient(
                [
                    type(
                        "SharedRepoBundle",
                        (),
                        {
                            "runtime_protocol_files": [
                                type(
                                    "RepoFile",
                                    (),
                                    {
                                        "path": "Dockerfile",
                                        "content": "FROM rust:1.82-bookworm\n",
                                    },
                                )(),
                                type(
                                    "RepoFile",
                                    (),
                                    {
                                        "path": RUNTIME_INSTALL_SCRIPT_PATH,
                                        "content": "#!/usr/bin/env sh\nset -eu\n",
                                    },
                                )(),
                                type(
                                    "RepoFile",
                                    (),
                                    {
                                        "path": RUNTIME_VERIFY_SCRIPT_PATH,
                                        "content": "#!/usr/bin/env sh\nset -eu\n",
                                    },
                                )(),
                                type(
                                    "RepoFile",
                                    (),
                                    {
                                        "path": RUNTIME_RUN_SCRIPT_PATH,
                                        "content": "#!/usr/bin/env sh\nset -eu\n",
                                    },
                                )(),
                            ],
                            "files": [
                                type(
                                    "RepoFile",
                                    (),
                                    {"path": "src/shared_stage.txt", "content": "stage-two\n"},
                                )(),
                                type("RepoFile", (), {"path": "pom.xml", "content": "<project/>\n"})(),
                            ],
                            "dependency_contract": _dependency_contract_payload(
                                manifest_paths=["pom.xml"],
                                reproducibility_mode="locked",
                            ),
                            "notes": [],
                        },
                    )(),
                ]
            )
            service = OpenAIStarterRepoAuthoringService(
                enabled=True,
                client_factory=lambda **_: queued_client,
            )

            run, result = service.author_workspace_repo(
                run,
                deliverable_ids=["deliverable_2"],
            )

            assert result.available is True
            parse_calls = queued_client.responses.parse_calls
            assert len(parse_calls) == 1
            assert parse_calls[0]["text_format"].__name__ == "_GeneratedSharedRepoBundle"
            payload = json.loads(parse_calls[0]["input"][1]["content"])
            assert payload["repair_scope_deliverable_ids"] == ["deliverable_2"]
            assert payload["shared_repo_root"] == "starter"
            assert payload["current_files"]["src/shared_stage.txt"] == "stage-one\n"
            assert "Dockerfile" in payload["shared_runtime_protocol_files"]
            assert {deliverable["deliverable_id"] for deliverable in payload["deliverables"]} == {
                "deliverable_1",
                "deliverable_2",
                "deliverable_3",
                "deliverable_4",
            }
            shared_starter_root = public_root / "starter"
            assert (shared_starter_root / "src" / "shared_stage.txt").read_text(encoding="utf-8") == "stage-two\n"
            assert (shared_starter_root / "pom.xml").read_text(encoding="utf-8") == "<project/>\n"
            # Per-deliverable starter folders no longer exist.
            assert not (public_root / "starter" / "deliverable_1").exists()
            assert not (public_root / "starter" / "deliverable_2").exists()

    def test_repo_authoring_payload_excludes_logs_and_build_artifacts_from_authored_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["starter_repo_bundle"] = {
                "source": "openai_live",
                "authored_paths": [
                    "src/main.rs",
                    "mvnw",
                    "logs/build.log",
                    "target/debug/demo",
                ],
            }
            manifest["dependency_contract"] = _dependency_contract_payload(
                build_support_paths=["mvnw"],
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
            (starter_root / "mvnw").write_text("#!/usr/bin/env sh\n./mvnw \"$@\"\n", encoding="utf-8")
            (starter_root / "logs").mkdir(parents=True, exist_ok=True)
            (starter_root / "logs" / "build.log").write_text("build log\n", encoding="utf-8")
            (starter_root / "target" / "debug").mkdir(parents=True, exist_ok=True)
            (starter_root / "target" / "debug" / "demo").write_text("binary\n", encoding="utf-8")

            service_payload = build_starter_authoring_payload(
                starter_root=starter_root,
                manifest=manifest,
            )

            assert service_payload["learner_files"] == {
                "src/main.rs": "// learner repo file\n",
                "mvnw": "#!/usr/bin/env sh\n./mvnw \"$@\"\n",
            }

    def test_artifact_materializer_excludes_generated_build_artifacts_from_workspace_starter_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            spec.project_contract.runtime_plan.package_manager = "cargo"
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["learner_starter_surface"] = {
                **(manifest.get("learner_starter_surface") or {}),
                "primary_editable_paths": ["src/main.rs"],
            }
            manifest["runtime_plan"] = {
                **(manifest.get("runtime_plan") or {}),
                "package_manager": "cargo",
            }
            manifest["dependency_contract"] = _dependency_contract_payload(
                manifest_paths=["Cargo.toml"],
                lockfile_paths=["Cargo.lock"],
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
            (starter_root / "Cargo.toml").write_text(
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                encoding="utf-8",
            )
            (starter_root / "Cargo.lock").write_text("# lockfile\n", encoding="utf-8")
            (starter_root / "target" / "debug" / ".fingerprint").mkdir(parents=True, exist_ok=True)
            (starter_root / "target" / "debug" / ".fingerprint" / "dep-lib-demo").write_bytes(
                b"\x01\x00\x00\x00\xff\x01\x00\x00"
            )

            bundle = ArtifactMaterializer(base_dir=f"{temp_dir}/generated").materialize_run(run, overwrite=True)
            shared_dir = Path(bundle.public_dir) / "starter"

            assert (shared_dir / "src" / "main.rs").exists()
            assert (shared_dir / "Cargo.toml").exists()
            assert (shared_dir / "Cargo.lock").exists()
            assert not (shared_dir / "target").exists()

    def test_repo_replace_preserves_binary_wrapper_support_files_while_cleaning_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["starter_repo_bundle"] = {
                "source": "openai_live",
                "authored_paths": ["src/main.rs", "mvnw"],
            }
            manifest["dependency_contract"] = _dependency_contract_payload(
                build_support_paths=["mvnw", ".mvn/wrapper/maven-wrapper.jar"],
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
            (starter_root / "mvnw").write_text("#!/usr/bin/env sh\n./mvnw \"$@\"\n", encoding="utf-8")
            (starter_root / ".mvn" / "wrapper").mkdir(parents=True, exist_ok=True)
            (starter_root / ".mvn" / "wrapper" / "maven-wrapper.jar").write_bytes(b"\x50\x4b\x03\x04")
            (starter_root / "logs").mkdir(parents=True, exist_ok=True)
            (starter_root / "logs" / "build.log").write_text("build log\n", encoding="utf-8")

            service = OpenAIStarterRepoAuthoringService(enabled=False)
            updated = service._replace_repo_files(
                starter_root=starter_root,
                manifest=manifest,
                files={
                    "src/main.rs": "// learner repo file\n",
                    "mvnw": "#!/usr/bin/env sh\n./mvnw \"$@\"\n",
                },
                workspace_root=Path(workspace.root_dir),
                visible_fixture_files=set(spec.runtime_dependencies.visible_fixture_files),
            )

            assert not (starter_root / "logs" / "build.log").exists()
            assert (starter_root / ".mvn" / "wrapper" / "maven-wrapper.jar").exists()
            assert any(path.endswith("logs/build.log") for path in updated)

    def test_test_authoring_prompt_payload_includes_current_repo_and_runtime_protocol_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["learner_starter_surface"] = {
                **(manifest.get("learner_starter_surface") or {}),
                "primary_editable_paths": ["src/main.rs"],
            }
            manifest["runtime_plan"] = {
                **(manifest.get("runtime_plan") or {}),
                "package_manager": "cargo",
            }
            manifest["dependency_contract"] = _dependency_contract_payload(
                manifest_paths=["Cargo.toml"],
                lockfile_paths=["Cargo.lock"],
            )

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
            (starter_root / "Cargo.toml").write_text(
                "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                encoding="utf-8",
            )
            (starter_root / "Cargo.lock").write_text("# lockfile\n", encoding="utf-8")
            (starter_root / "target" / "debug").mkdir(parents=True, exist_ok=True)
            (starter_root / "target" / "debug" / "demo").write_text("binary\n", encoding="utf-8")
            (starter_root / "Dockerfile").write_text("FROM rust:1.86-bookworm\n", encoding="utf-8")
            (starter_root / RUNTIME_INSTALL_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo fetch\n",
                encoding="utf-8",
            )
            (starter_root / RUNTIME_RUN_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo run\n",
                encoding="utf-8",
            )

            service = OpenAITestScriptAuthoringService(enabled=False)
            payload = service._prompt_payload(
                run,
                starter_root=starter_root,
                manifest=manifest,
                failure_context=None,
            )

        learner_files = payload["files"]
        dependency_files = payload["dependency_contract_files"]
        runtime_files = payload["runtime_protocol_files"]
        assert learner_files == {"src/main.rs": "// learner repo file\n"}
        assert dependency_files == {
            "Cargo.toml": "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"
        }
        assert "Dockerfile" in runtime_files
        assert RUNTIME_INSTALL_SCRIPT_PATH in runtime_files
        assert RUNTIME_RUN_SCRIPT_PATH in runtime_files
        assert "README.md" not in learner_files
        assert HIDDEN_MANIFEST_PATH not in learner_files
        assert "Cargo.lock" not in dependency_files
        assert "target/debug/demo" not in learner_files

    def test_repo_bundle_state_uses_final_workspace_completeness_not_changed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter"
            manifest_path = (
                Path(workspace.root_dir)
                / "private"
                / "grader"
                / deliverable.id
                / "deliverable.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["learner_starter_surface"] = {
                **(manifest.get("learner_starter_surface") or {}),
                "primary_editable_paths": ["src/main.rs"],
            }

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
            (starter_root / RUNTIME_INSTALL_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo fetch\n",
                encoding="utf-8",
            )

            service = OpenAIStarterRepoAuthoringService(enabled=False)
            starter_repo_bundle, runtime_protocol_bundle = service._bundle_state(
                starter_root=starter_root,
                manifest=manifest,
                default_starter_files=build_task_agent_starter_files(spec, deliverable.id),
                visible_fixture_files=set(spec.runtime_dependencies.visible_fixture_files),
            )

        assert starter_repo_bundle["source"] == "openai_live"
        assert starter_repo_bundle["authored_paths"] == ["src/main.rs"]
        assert runtime_protocol_bundle["source"] == "starter_default"
        assert runtime_protocol_bundle["authored_paths"] == [RUNTIME_INSTALL_SCRIPT_PATH]


class TestScriptAuthoringSelfContainedPromptTests(unittest.TestCase):
    """Pass 12.

    The visible test script the LLM authors lives in `public/checks/<id>/` and
    ships to learners. The per-deliverable manifest lives in `private/grader/<id>/`
    after Pass 2. A visible script that reads the manifest at runtime crashes
    with FileNotFoundError (manifest not at any reachable public-side path) and
    even if the path were correct, learners would never have `private/` in
    their workspace. The fix is structural: the script must be self-contained.

    These assertions pin the prompt directive so a future edit can't quietly
    re-introduce the legacy "read manifest at runtime" pattern.
    """

    def test_test_script_authoring_prompt_forbids_runtime_file_reads(self) -> None:
        import inspect
        from app.services.openai_test_script_authoring import (
            OpenAITestScriptAuthoringService,
        )

        source = inspect.getsource(OpenAITestScriptAuthoringService)
        lowered = source.lower()
        self.assertIn("self-contained", lowered,
                      "Prompt must declare scripts MUST be self-contained.")
        self.assertIn("do not read any file from disk at runtime", lowered,
                      "Prompt must explicitly forbid runtime file reads.")
        # The legacy path the model kept reaching for must be called out by name
        # so the model doesn't re-emit it on the next attempt.
        self.assertIn(".coursegen/deliverable.json", lowered,
                      "Prompt should name the legacy `.coursegen/deliverable.json` "
                      "path explicitly as forbidden.")

    def test_test_script_authoring_prompt_directs_inlining_test_data(self) -> None:
        import inspect
        from app.services.openai_test_script_authoring import (
            OpenAITestScriptAuthoringService,
        )

        source = inspect.getsource(OpenAITestScriptAuthoringService)
        lowered = source.lower()
        # The prompt must direct the model to inline literals.
        self.assertIn("inline", lowered)
        # The prompt should name the canonical sources the model takes literals
        # from (manifest.public_checks + public_endpoints) so it has somewhere
        # concrete to look at authoring time.
        self.assertIn("public_checks", source)

    def test_test_script_authoring_prompt_limits_runtime_inputs_to_env_vars(self) -> None:
        import inspect
        from app.services.openai_test_script_authoring import (
            OpenAITestScriptAuthoringService,
        )

        source = inspect.getsource(OpenAITestScriptAuthoringService)
        # The only runtime inputs allowed are BASE_URL and REPORT_PATH.
        self.assertIn("BASE_URL", source)
        self.assertIn("REPORT_PATH", source)


class DeterministicVisibleScriptInlinesPublicChecksTests(unittest.TestCase):
    """Pass 12 (real cause).

    The visible script that ran in the failed Python+FastAPI validation
    came from the deterministic fallback template
    ``render_task_agent_visible_checks_script`` — NOT from the OpenAI
    test-script authoring service (which only runs at the authoring_tests
    node, after authoring_runtime passes). The deterministic template was
    written for the pre-Pass-2 layout and reads the per-deliverable
    manifest at runtime from ``.coursegen/deliverable.json``, which lives
    nowhere accessible to ``public/checks/<id>/run_visible_checks.py``
    after Pass 2 (manifest moved to ``private/grader/<id>/``).

    The fix is to inline ``public_checks`` as a Python literal at render
    time, so the script is fully self-contained and never reads the
    filesystem.
    """

    def test_visible_script_inlines_public_checks_as_python_literal(self) -> None:
        from app.services.task_agent_starter_templates import (
            render_task_agent_visible_checks_script,
        )
        sample_checks = [
            {
                "id": "create_thing",
                "title": "create thing returns 200",
                "request_method": "POST",
                "request_path": "/things",
                "request_body": {"name": "alpha"},
                "expected_status": 200,
                "expected_response_contains": ["thing_id"],
            }
        ]
        script = render_task_agent_visible_checks_script(public_checks=sample_checks)
        # The public_checks data must appear as Python literals in the
        # rendered script, not be reached for at runtime.
        self.assertIn("PUBLIC_CHECKS = ", script)
        self.assertIn('"/things"', script)
        self.assertIn('"create_thing"', script)
        self.assertIn('"thing_id"', script)

    def test_visible_script_does_not_read_manifest_at_runtime(self) -> None:
        from app.services.task_agent_starter_templates import (
            render_task_agent_visible_checks_script,
        )
        script = render_task_agent_visible_checks_script(public_checks=[])
        # Specifically: the legacy patterns must not appear anywhere.
        self.assertNotIn("MANIFEST_PATH", script,
                         "Visible script must NOT reference a runtime manifest path.")
        self.assertNotIn(".read_text(", script,
                         "Visible script must NOT call read_text on any file.")
        self.assertNotIn(".coursegen/deliverable.json", script,
                         "Visible script must NOT reach for the legacy manifest path.")
        self.assertNotIn("Path(__file__).resolve().parents", script,
                         "Visible script must NOT compute paths from __file__.")

    def test_visible_script_only_reads_BASE_URL_and_REPORT_PATH_env(self) -> None:
        from app.services.task_agent_starter_templates import (
            render_task_agent_visible_checks_script,
        )
        script = render_task_agent_visible_checks_script(public_checks=[])
        # The only runtime inputs allowed are these two env vars.
        env_reads = [line for line in script.splitlines() if "os.environ" in line]
        for line in env_reads:
            self.assertTrue(
                "BASE_URL" in line or "REPORT_PATH" in line,
                f"Unexpected env var read: {line!r}. Only BASE_URL and REPORT_PATH allowed.",
            )

    def test_build_task_agent_starter_files_threads_public_checks(self) -> None:
        """``build_task_agent_starter_files`` must pass the per-deliverable
        ``public_checks`` into the visible-script renderer so the inlined
        literals reflect the deliverable's actual contract surface.
        """
        from app.domain.task_agent import (
            CourseStructureSpec,
            DeliverableSpec,
            PublicCheckSpec,
            RuntimeDependencySpec,
            TaskAgentServiceSpec,
            CapabilitySpec,
            AssessmentStrategySpec,
            ExecutionSurface,
            WorkspaceScope,
            ProgressionMode,
        )
        from app.domain.registry import PackageType, StarterType
        from app.services.task_agent_starter_templates import (
            build_task_agent_starter_files,
        )

        spec = TaskAgentServiceSpec(
            title="t",
            summary="s",
            package_type=PackageType.progressive_codebase_course,
            course_structure=CourseStructureSpec(
                package_type=PackageType.progressive_codebase_course,
                workspace_scope=WorkspaceScope.shared_course_workspace,
                progression_mode=ProgressionMode.independent_deliverables,
                shared_codebase=True,
            ),
            runtime_dependencies=RuntimeDependencySpec(
                execution_surface=ExecutionSurface.http_service,
                starter_type=StarterType.partial,
            ),
            capabilities=CapabilitySpec(),
            assessment_strategy=AssessmentStrategySpec(),
            deliverables=[
                DeliverableSpec(
                    id="deliverable_1",
                    title="D1",
                    objective="do d1",
                    public_checks=[
                        PublicCheckSpec(
                            id="get_resource",
                            title="GET /things/{id} returns 200",
                            learner_goal="basic read",
                            request_method="GET",
                            request_path="/things/abc-123",
                            request_body={},
                            expected_status=200,
                            expected_response_contains=["thing_id"],
                        )
                    ],
                ),
                DeliverableSpec(id="deliverable_2", title="D2", objective="do d2"),
            ],
        )
        files = build_task_agent_starter_files(spec, "deliverable_1")
        script = files["checks/run_visible_checks.py"]
        # The deliverable-specific check fields must be in the script literal,
        # not read from a manifest at runtime.
        self.assertIn("/things/abc-123", script)
        self.assertIn("thing_id", script)
        self.assertIn("get_resource", script)


if __name__ == "__main__":
    unittest.main()
