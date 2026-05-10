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
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest = json.loads((starter_root / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))
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
                        expected_build_support_paths=[],
                        present_build_support_paths=[],
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

    def test_repo_authoring_shared_codebase_uses_single_progressive_bundle_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            public_root = Path(workspace.public_dir)
            for deliverable_id in ["deliverable_1", "deliverable_2"]:
                manifest_path = public_root / "starter" / deliverable_id / HIDDEN_MANIFEST_PATH
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["learner_starter_surface"] = {
                    **(manifest.get("learner_starter_surface") or {}),
                    "primary_editable_paths": ["src/shared_stage.txt"],
                }
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            (public_root / "starter" / "deliverable_1" / "src").mkdir(parents=True, exist_ok=True)
            (public_root / "starter" / "deliverable_1" / "src" / "shared_stage.txt").write_text(
                "stage-one\n",
                encoding="utf-8",
            )

            queued_client = _QueuedFakeClient(
                [
                    type(
                        "ProgressiveRepoBundle",
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
                            "deliverables": [
                                type(
                                    "DeliverableBundle",
                                    (),
                                    {
                                        "deliverable_id": "deliverable_2",
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
                                    },
                                )(),
                            ],
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
            assert parse_calls[0]["text_format"].__name__ == "_GeneratedProgressiveRepoBundle"
            payload = json.loads(parse_calls[0]["input"][1]["content"])
            assert payload["deliverable_ids"] == ["deliverable_2"]
            assert payload["lineage_anchor"]["deliverable_id"] == "deliverable_1"
            assert payload["lineage_anchor"]["current_files"]["src/shared_stage.txt"] == "stage-one\n"
            assert "Dockerfile" in payload["shared_runtime_protocol_files"]

    def test_repo_authoring_payload_excludes_logs_and_build_artifacts_from_authored_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest_path = starter_root / HIDDEN_MANIFEST_PATH
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
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest_path = starter_root / HIDDEN_MANIFEST_PATH
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
            deliverable_dir = Path(bundle.public_dir) / "starter" / deliverable.id

            assert (deliverable_dir / "src" / "main.rs").exists()
            assert (deliverable_dir / "Cargo.toml").exists()
            assert (deliverable_dir / "Cargo.lock").exists()
            assert not (deliverable_dir / "target").exists()

    def test_repo_replace_preserves_binary_wrapper_support_files_while_cleaning_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = _materialized_run(temp_dir)
            spec = run.artifacts.task_agent_spec
            workspace = run.artifacts.workspace_snapshot
            assert spec is not None
            assert workspace is not None
            deliverable = spec.deliverables[0]
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest_path = starter_root / HIDDEN_MANIFEST_PATH
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
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest = json.loads((starter_root / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))
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
            starter_root = Path(workspace.public_dir) / "starter" / deliverable.id
            manifest = json.loads((starter_root / HIDDEN_MANIFEST_PATH).read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    unittest.main()
