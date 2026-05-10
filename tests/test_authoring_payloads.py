from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.domain.registry import PackageType
from app.domain.workflow import (
    FailureContext,
    FailureContextDependencyContract,
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
    run = workflow_service.create_run_from_explicit_plan(
        intake=intake,
        design_spec=inferred.design_spec,
        execute_nodes=False,
    )
    workflow_service.materialize_run(run.id, MaterializeBundleRequest(overwrite=True))
    run = workflow_service.get_run(run.id)
    assert run is not None
    run, _ = TaskAgentWorkspaceAuthoringService().author_workspace(run)
    return run


class AuthoringPayloadTests(unittest.TestCase):
    def test_repo_authoring_uses_structured_response_parsing(self) -> None:
        client = _FakeClient(
            _FakeParsedResponse(
                parsed=type(
                    "RepoBundle",
                    (),
                    {
                        "files": [type("RepoFile", (), {"path": "src/main.rs", "content": "fn main() {}\n"})()],
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
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest = json.loads((starter_root / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
            (starter_root / "Dockerfile").write_text("FROM rust:1.86-bookworm\n", encoding="utf-8")
            (starter_root / RUNTIME_INSTALL_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo fetch\n",
                encoding="utf-8",
            )
            (starter_root / RUNTIME_RUN_SCRIPT_PATH).write_text(
                "#!/usr/bin/env sh\nset -eu\ncargo run\n",
                encoding="utf-8",
            )

            service = OpenAIStarterRepoAuthoringService(enabled=False)
            payload = service._prompt_payload(
                run,
                deliverable_id=deliverable.id,
                starter_root=starter_root,
                manifest=manifest,
                failure_context=None,
            )

        files = payload["current_files"]
        assert "src/main.rs" in files
        assert "Dockerfile" in files
        assert RUNTIME_INSTALL_SCRIPT_PATH in files
        assert RUNTIME_RUN_SCRIPT_PATH in files
        assert "README.md" in files
        assert HIDDEN_MANIFEST_PATH in files

    def test_repo_authoring_prompt_payload_includes_dependency_contract_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest = json.loads((starter_root / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))

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
                        runtime_protocol_paths_present=["Dockerfile", RUNTIME_INSTALL_SCRIPT_PATH],
                        runtime_bundle_complete=False,
                    )
                ],
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

    def test_test_authoring_prompt_payload_includes_current_repo_and_runtime_protocol_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest = json.loads((starter_root / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))

            (starter_root / "src").mkdir(parents=True, exist_ok=True)
            (starter_root / "src" / "main.rs").write_text("// learner repo file\n", encoding="utf-8")
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

        files = payload["files"]
        assert "src/main.rs" in files
        assert "Dockerfile" in files
        assert RUNTIME_INSTALL_SCRIPT_PATH in files
        assert RUNTIME_RUN_SCRIPT_PATH in files
        assert "README.md" in files
        assert HIDDEN_MANIFEST_PATH in files

    def test_repo_bundle_state_uses_final_workspace_completeness_not_changed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest = json.loads((starter_root / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))

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


if __name__ == "__main__":
    unittest.main()
