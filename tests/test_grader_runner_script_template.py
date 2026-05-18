"""Tests for the generic on-disk grader-runner script template.

The template ships as a single source-of-truth string constant
(``GRADER_RUNNER_SCRIPT_SOURCE``) so the harness can drop it verbatim
into ``private/grader/runner.py`` when materializing a learner bundle.
These tests verify the string is well-formed Python and contains the
contractual hooks the harness depends on; they deliberately do NOT
execute the script (its dependencies are only present inside the
sandbox).
"""
from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from app.services.grader_runner_script_template import (
    GRADER_RUNNER_SCRIPT_SOURCE,
)


# ---------- helpers for exec'ing pieces of the runner script ----------


def _exec_runner_namespace(
    *,
    urllib_request_overrides: dict[str, Any] | None = None,
    urllib_error_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the runner source string and exec it in an isolated namespace.

    To avoid importing ``app.services.scenario_*`` at exec time (which
    would re-import the runner inside this test process), we stub those
    two imports. After exec we patch the resulting module-level
    ``urllib_request`` / ``urllib_error`` aliases inside the namespace
    directly — that lets each test substitute its own ``urlopen`` /
    ``URLError`` without touching the real stdlib modules.
    """
    fake_loader = types.ModuleType("app.services.scenario_loader")
    fake_loader.load_scenarios_from_dir = lambda _p: []  # type: ignore[attr-defined]
    fake_runner = types.ModuleType("app.services.scenario_trace_runner")
    fake_runner.run_scenario = lambda **_kw: None  # type: ignore[attr-defined]

    saved = {}
    for mod_name, mod in [
        ("app.services.scenario_loader", fake_loader),
        ("app.services.scenario_trace_runner", fake_runner),
    ]:
        saved[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mod

    namespace: dict[str, Any] = {"__name__": "<grader-runner-under-test>"}
    try:
        code = compile(GRADER_RUNNER_SCRIPT_SOURCE, "<grader-runner>", "exec")
        exec(code, namespace)
    finally:
        for mod_name, prev in saved.items():
            if prev is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = prev

    # Patch the runner's local urllib aliases with caller-supplied attrs.
    if urllib_request_overrides:
        fake_request = types.SimpleNamespace(
            Request=namespace["urllib_request"].Request,
            urlopen=namespace["urllib_request"].urlopen,
        )
        for key, value in urllib_request_overrides.items():
            setattr(fake_request, key, value)
        namespace["urllib_request"] = fake_request
        # Reach into the adapter class so methods see the override too.
        # The adapter references ``urllib_request`` from module globals,
        # which IS our namespace dict — so reassigning the key suffices.

    if urllib_error_overrides:
        fake_error = types.SimpleNamespace(
            URLError=namespace["urllib_error"].URLError,
            HTTPError=namespace["urllib_error"].HTTPError,
        )
        for key, value in urllib_error_overrides.items():
            setattr(fake_error, key, value)
        namespace["urllib_error"] = fake_error

    return namespace


def test_source_is_non_empty_string() -> None:
    assert isinstance(GRADER_RUNNER_SCRIPT_SOURCE, str)
    assert len(GRADER_RUNNER_SCRIPT_SOURCE.strip()) > 200


def test_source_parses_as_valid_python() -> None:
    ast.parse(GRADER_RUNNER_SCRIPT_SOURCE)


def test_source_compiles_without_syntax_errors() -> None:
    compile(GRADER_RUNNER_SCRIPT_SOURCE, "<grader-runner>", "exec")


def test_source_starts_with_shebang_and_docstring() -> None:
    lines = GRADER_RUNNER_SCRIPT_SOURCE.splitlines()
    assert lines[0].startswith("#!"), "expected shebang on first line"
    # An ``ast.parse`` module-level docstring is the second statement
    # node (after any leading from __future__).
    module = ast.parse(GRADER_RUNNER_SCRIPT_SOURCE)
    docstring = ast.get_docstring(module)
    assert docstring is not None
    assert len(docstring.strip()) > 50


def test_source_references_run_scenario_and_loader() -> None:
    assert "run_scenario" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "load_scenarios_from_dir" in GRADER_RUNNER_SCRIPT_SOURCE


def test_source_reads_base_url_and_report_path_from_env() -> None:
    # The script must look up both env vars.
    assert 'os.environ' in GRADER_RUNNER_SCRIPT_SOURCE or 'os.getenv' in GRADER_RUNNER_SCRIPT_SOURCE
    assert "BASE_URL" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "REPORT_PATH" in GRADER_RUNNER_SCRIPT_SOURCE


def test_source_writes_report_to_report_path() -> None:
    # When REPORT_PATH is set, the script writes the report there.
    # We just look for the structural hooks; we don't execute.
    assert "json.dump" in GRADER_RUNNER_SCRIPT_SOURCE or "json.dumps" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "summary" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "tests" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "diagnostics" in GRADER_RUNNER_SCRIPT_SOURCE


def test_source_exits_nonzero_on_failure() -> None:
    # The script uses sys.exit with a non-zero argument when a scenario fails.
    assert "sys.exit" in GRADER_RUNNER_SCRIPT_SOURCE


def test_source_imports_oracle_and_setup_data() -> None:
    # The header / body references oracle outputs and the _setup directory.
    assert "_setup" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "_oracle" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "oracle" in GRADER_RUNNER_SCRIPT_SOURCE


def test_source_documents_dependency_assumption() -> None:
    """The script header should make the runtime-dep assumption explicit."""
    # The script docstring must mention that the rubric library / app.services
    # package is expected to be installed in the runtime environment.
    lowered = GRADER_RUNNER_SCRIPT_SOURCE.lower()
    assert "app.services" in GRADER_RUNNER_SCRIPT_SOURCE
    assert "installed" in lowered or "dependency" in lowered or "runtime" in lowered


# ---------------------------------------------------------------------------
# Finding A: runner must supply an LLM router to ``run_scenario`` so the
# LLMJudgeCoverage rubric (and any future judge rubric) can actually call
# the harness-managed sandbox LLM proxy.
# ---------------------------------------------------------------------------


def test_source_defines_sandbox_llm_router_adapter() -> None:
    """The runner script must inline (or import) a router adapter class."""
    assert "SandboxLLMRouterAdapter" in GRADER_RUNNER_SCRIPT_SOURCE
    # It must look like a class definition or an import, not just a string ref.
    assert (
        "class SandboxLLMRouterAdapter" in GRADER_RUNNER_SCRIPT_SOURCE
        or "import SandboxLLMRouterAdapter" in GRADER_RUNNER_SCRIPT_SOURCE
        or "from app.services" in GRADER_RUNNER_SCRIPT_SOURCE
        and "SandboxLLMRouterAdapter" in GRADER_RUNNER_SCRIPT_SOURCE
    )


def test_source_instantiates_router_and_passes_to_run_scenario() -> None:
    """run_scenario must be invoked with router=... so judge rubrics work."""
    assert "router=" in GRADER_RUNNER_SCRIPT_SOURCE
    # The instantiation should happen before the per-scenario loop. We
    # check structurally: the adapter is constructed somewhere and the
    # constructed object is passed to run_scenario.
    assert "SandboxLLMRouterAdapter(" in GRADER_RUNNER_SCRIPT_SOURCE
    # Every run_scenario call in main() should receive a router. The
    # simplest assertion: the kwarg appears in the same call block as
    # base_url/setup_data.
    assert "run_scenario(" in GRADER_RUNNER_SCRIPT_SOURCE
    # Crude check that router= appears near a run_scenario( call.
    call_idx = GRADER_RUNNER_SCRIPT_SOURCE.index("run_scenario(")
    window = GRADER_RUNNER_SCRIPT_SOURCE[call_idx : call_idx + 400]
    assert "router=" in window, "router= must be passed to run_scenario(...)"


def test_source_reads_submission_token_env_var() -> None:
    """The runner must forward the per-submission rate-limit token."""
    assert "COURSEGEN_SUBMISSION_TOKEN" in GRADER_RUNNER_SCRIPT_SOURCE


def test_source_reads_proxy_url_env_override() -> None:
    """The proxy URL must be overridable for local testing."""
    assert "COURSEGEN_LLM_PROXY_URL" in GRADER_RUNNER_SCRIPT_SOURCE
    # And the production default must be present so the runner works
    # without explicit configuration inside the sandbox.
    assert "http://coursegen-llm:8080" in GRADER_RUNNER_SCRIPT_SOURCE


def test_adapter_posts_to_proxy_url_with_submission_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When parse_structured is called the adapter POSTs to the proxy URL."""

    class _Demo(BaseModel):
        verdict: str
        rationale: str

    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = (
            req.full_url if hasattr(req, "full_url") else req.get_full_url()
        )
        captured["data"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        body = json.dumps(
            {
                "content": json.dumps({"verdict": "pass", "rationale": "ok"}),
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "model_id": "claude-haiku-4-5",
                "cost_usd": 0.0001,
            }
        ).encode("utf-8")
        return _FakeResponse(body)

    monkeypatch.setenv("COURSEGEN_LLM_PROXY_URL", "http://test-proxy:9999")
    monkeypatch.setenv("COURSEGEN_SUBMISSION_TOKEN", "tok-abc")

    namespace = _exec_runner_namespace(
        urllib_request_overrides={"urlopen": fake_urlopen},
    )

    AdapterCls = namespace["SandboxLLMRouterAdapter"]
    adapter = AdapterCls()

    result = adapter.parse_structured(
        tier="haiku",
        system="You are a judge.",
        user="Judge this.",
        text_format=_Demo,
        max_tokens=500,
        request_timeout_s=30.0,
    )

    assert captured["url"] == "http://test-proxy:9999/v1/messages"
    assert captured["data"]["tier"] == "haiku"
    assert captured["data"]["max_tokens"] == 500
    assert captured["data"]["submission_token"] == "tok-abc"
    assert captured["data"]["system"].startswith("You are a judge.")
    # messages list shape — single user turn carrying the prompt.
    assert isinstance(captured["data"]["messages"], list)
    assert captured["data"]["messages"][0]["role"] == "user"
    assert "Judge this." in captured["data"]["messages"][0]["content"]

    # And the adapter exposes a parsed BaseModel of the requested type.
    parsed = getattr(result, "parsed", None) or getattr(result, "output_parsed", None)
    assert isinstance(parsed, _Demo)
    assert parsed.verdict == "pass"
    assert parsed.rationale == "ok"


def test_adapter_abstains_on_urlerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network failure must NOT crash the runner — adapter returns a
    sentinel that causes LLMJudgeCoverage to abstain (fail open)."""

    class _Demo(BaseModel):
        verdict: str
        rationale: str

    import urllib.error as _real_urllib_error

    def fake_urlopen(req: Any, timeout: float | None = None) -> Any:
        raise _real_urllib_error.URLError("connection refused")

    monkeypatch.delenv("COURSEGEN_LLM_PROXY_URL", raising=False)
    monkeypatch.delenv("COURSEGEN_SUBMISSION_TOKEN", raising=False)

    namespace = _exec_runner_namespace(
        urllib_request_overrides={"urlopen": fake_urlopen},
    )

    AdapterCls = namespace["SandboxLLMRouterAdapter"]
    adapter = AdapterCls()

    # Must NOT raise.
    result = adapter.parse_structured(
        tier="haiku",
        system="sys",
        user="user",
        text_format=_Demo,
        max_tokens=200,
        request_timeout_s=10.0,
    )

    # Result must NOT be a _Demo instance — LLMJudgeCoverage checks
    # ``isinstance(parsed, text_format)`` and abstains on a miss.
    parsed = getattr(result, "parsed", None) or getattr(result, "output_parsed", None)
    assert not isinstance(parsed, _Demo)


# ---------------------------------------------------------------------------
# Finding E: setup loader must mirror oracle_pass._load_setup_data — all
# top-level files (JSON parsed, others raw text), skipping directories and
# dotfiles.
# ---------------------------------------------------------------------------


def test_setup_loader_loads_text_files(tmp_path: Path) -> None:
    namespace = _exec_runner_namespace()
    loader = namespace["_load_setup_data"]
    root = tmp_path / "bundle"
    setup = root / "_setup"
    setup.mkdir(parents=True)
    (setup / "notes.txt").write_text("hello world")

    data = loader(root)
    assert data["notes"] == "hello world"


def test_setup_loader_parses_json_and_keeps_text(tmp_path: Path) -> None:
    namespace = _exec_runner_namespace()
    loader = namespace["_load_setup_data"]
    root = tmp_path / "bundle"
    setup = root / "_setup"
    setup.mkdir(parents=True)
    (setup / "gold.json").write_text(json.dumps({"answer": 42}))
    (setup / "context.md").write_text("# context\nbody")

    data = loader(root)
    assert data["gold"] == {"answer": 42}
    assert data["context"] == "# context\nbody"


def test_setup_loader_skips_subdirs_and_dotfiles(tmp_path: Path) -> None:
    namespace = _exec_runner_namespace()
    loader = namespace["_load_setup_data"]
    root = tmp_path / "bundle"
    setup = root / "_setup"
    setup.mkdir(parents=True)
    (setup / ".hidden").write_text("nope")
    (setup / "real.json").write_text(json.dumps({"k": "v"}))
    nested = setup / "subdir"
    nested.mkdir()
    (nested / "ignored.json").write_text(json.dumps({"bad": True}))

    data = loader(root)
    assert "hidden" not in data
    assert ".hidden" not in data
    assert "subdir" not in data
    assert data["real"] == {"k": "v"}
