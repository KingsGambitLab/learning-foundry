"""Tests for the visible-checks script template.

The visible-checks script is a lightweight self-test learners run before
submission. It must be self-contained (stdlib only — no network provider
SDKs, no rubric-library imports) so it ships cleanly in a learner
sandbox.

This script fires sample queries against the learner's running service,
validates response shape, and writes a JSON report in the same
``{summary, tests:[...]}`` envelope the existing harness consumes.
"""
from __future__ import annotations

import ast

from app.services.visible_checks_script_template import (
    VISIBLE_CHECKS_SCRIPT_SOURCE,
)


def test_script_source_compiles_as_valid_python() -> None:
    """A syntax error in the template would break every benchmark course
    bundle silently — guard against it here."""
    ast.parse(VISIBLE_CHECKS_SCRIPT_SOURCE)


def test_script_source_reads_base_url_and_report_path_from_env() -> None:
    """The script reads ``BASE_URL`` and ``REPORT_PATH`` from os.environ
    so it integrates with the harness's existing report-collection
    contract."""
    assert "BASE_URL" in VISIBLE_CHECKS_SCRIPT_SOURCE
    assert "REPORT_PATH" in VISIBLE_CHECKS_SCRIPT_SOURCE
    # And it must actually access them via ``os.environ`` (not just
    # mention them in a docstring).
    tree = ast.parse(VISIBLE_CHECKS_SCRIPT_SOURCE)
    env_lookups: set[str] = set()
    for node in ast.walk(tree):
        # os.environ.get("BASE_URL", ...) or os.environ["BASE_URL"]
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                env_lookups.add(node.args[0].value)
        if isinstance(node, ast.Subscript):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                env_lookups.add(node.slice.value)
    assert "BASE_URL" in env_lookups
    assert "REPORT_PATH" in env_lookups


def test_script_source_is_pure_stdlib_no_heavy_imports() -> None:
    """The script must NOT import anything from ``app.services`` (rubric
    library / LLM proxy / scenario loader) nor any third-party HTTP /
    LLM SDK — it must run cleanly inside a learner sandbox that lacks
    those packages. Stdlib + urllib only."""
    tree = ast.parse(VISIBLE_CHECKS_SCRIPT_SOURCE)
    banned_prefixes = (
        "app.services",
        "anthropic",
        "openai",
        "requests",
        "httpx",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in banned_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"banned import '{alias.name}' in visible-checks "
                        "script template"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for prefix in banned_prefixes:
                assert not node.module.startswith(prefix), (
                    f"banned import-from '{node.module}' in visible-checks "
                    "script template"
                )


def test_script_source_writes_harness_compatible_report() -> None:
    """The report must use the ``{summary, tests:[...]}`` shape the harness
    consumes — same envelope as ``GRADER_RUNNER_SCRIPT_SOURCE`` produces."""
    assert '"summary"' in VISIBLE_CHECKS_SCRIPT_SOURCE
    assert '"tests"' in VISIBLE_CHECKS_SCRIPT_SOURCE
    assert '"status"' in VISIBLE_CHECKS_SCRIPT_SOURCE


def test_script_source_reads_sample_queries_json() -> None:
    """The script must load ``sample_queries.json`` from the visible
    examples directory (sibling to the script location)."""
    assert "sample_queries.json" in VISIBLE_CHECKS_SCRIPT_SOURCE
