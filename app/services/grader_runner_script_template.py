"""Source-of-truth string template for the on-disk grader runner script.

When the harness materializes a learner bundle, it writes
:data:`GRADER_RUNNER_SCRIPT_SOURCE` to ``private/grader/runner.py`` and
runs it via :class:`GeneratedTestScriptRunner` (see
``generated_test_harness.py``). The script:

1. Loads every YAML scenario from a ``scenarios/`` directory next to
   itself via :func:`load_scenarios_from_dir`.
2. Loads optional setup payloads from ``_setup/`` (JSON parsed,
   anything else kept as raw text) and merges them into a ``setup_data``
   dict keyed by filename stem. This mirrors
   :func:`app.services.oracle_pass._load_setup_data` so a course that
   passes oracle generation also reaches the grader with the same
   ``setup_data`` shape.
3. Loads optional oracle outputs from ``_oracle/outputs.json`` and
   merges them under ``setup_data["oracle"]``.
4. Reads ``BASE_URL`` and ``REPORT_PATH`` from the environment.
5. Constructs a :class:`SandboxLLMRouterAdapter` (defined inline in the
   script) and passes it as ``router=...`` to every :func:`run_scenario`
   call so judge rubrics (``LLMJudgeCoverage`` etc.) can reach the
   harness-managed LLM proxy at ``http://coursegen-llm:8080``. The
   adapter is intentionally inlined — it depends only on the standard
   library ``urllib`` so it cannot fail to import inside the learner
   sandbox even when network provider SDKs aren't installed. On any
   proxy error the adapter returns a sentinel whose ``parsed`` is
   ``None``; :class:`LLMJudgeCoverage` then abstains (fail open) rather
   than crashing the grading run.
6. For each scenario, calls :func:`run_scenario` and turns the verdict
   report into the
   ``{"summary": "...", "tests": [{"id","title","status","summary","diagnostics"}]}``
   shape the existing test harness already understands.
7. Writes the JSON report to ``REPORT_PATH`` when set, otherwise prints
   it to stdout, and exits non-zero if any scenario didn't pass.

Storing the script as a string here means it's version-controlled,
diffable, and unit-testable without ever invoking the file system. The
harness simply does ``Path(...).write_text(GRADER_RUNNER_SCRIPT_SOURCE)``
when assembling the bundle.

Runtime dependency assumption:
    The script imports ``app.services.scenario_loader`` and
    ``app.services.scenario_trace_runner`` directly. The grader sandbox
    is expected to have the ``course-gen-codex`` package (or an
    extracted ``app/`` tree) installed and importable — same assumption
    as the rest of the on-disk graders. The script intentionally does
    NOT vendor the rubric library inline; that would force every
    rubric-library change to be re-baked into every learner bundle.

    The LLM router adapter is the exception: it's inlined so it has
    zero deps beyond stdlib. The adapter's job is to translate
    :meth:`LLMRouter.parse_structured` shaped calls into HTTP POSTs
    against the in-network sandbox LLM proxy. The proxy URL is
    overridable via ``COURSEGEN_LLM_PROXY_URL``; the per-submission
    rate-limit token comes from ``COURSEGEN_SUBMISSION_TOKEN``.
"""
from __future__ import annotations


GRADER_RUNNER_SCRIPT_SOURCE = '''#!/usr/bin/env python3
"""Generic scenario-driven grader runner for learner bundles.

Walks every YAML scenario under ``scenarios/`` (sibling to this file)
and produces a JSON report compatible with the harness expectations.

Runtime dependency: the rubric / runner library at
``app.services.scenario_loader`` and
``app.services.scenario_trace_runner`` must be installed in the
sandbox environment that runs this script. This file does not vendor
the library; it imports it. The LLM router adapter, by contrast, is
inlined below so it has zero dependencies beyond the stdlib — the
runner ships standalone inside the learner container and must not rely
on optional provider SDKs being importable.

Environment:
    BASE_URL                    Base URL of the learner-service under
                                test. Defaults to
                                ``http://localhost:8000`` if unset.
    REPORT_PATH                 When set, the JSON report is written
                                here. Otherwise the report is printed
                                to stdout.
    COURSEGEN_LLM_PROXY_URL     Base URL of the harness-managed sandbox
                                LLM proxy. Defaults to
                                ``http://coursegen-llm:8080``.
    COURSEGEN_SUBMISSION_TOKEN  Per-submission rate-limit token; the
                                proxy uses it to meter calls and bound
                                cost.

Exit code:
    0 when every scenario's overall_status is ``pass``.
    1 when any scenario failed or abstained.
    2 on a runner-internal error (config / loader failure).
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.services.scenario_loader import load_scenarios_from_dir
from app.services.scenario_trace_runner import run_scenario


_DEFAULT_PROXY_URL = "http://coursegen-llm:8080"


class _AbstainResult:
    """Sentinel returned by the adapter when the proxy call cannot be
    served. Carries ``parsed=None`` so :class:`LLMJudgeCoverage` falls
    through to its ``isinstance(parsed, text_format)`` check and
    abstains. ``output_parsed`` is provided as an alias for symmetry
    with the production :class:`ParsedResult`."""

    __slots__ = ("parsed", "output_parsed", "usage_summary", "usage")

    def __init__(self) -> None:
        self.parsed = None
        self.output_parsed = None
        self.usage_summary = None
        self.usage = None


class SandboxLLMRouterAdapter:
    """Stdlib-only adapter that satisfies the ``router.parse_structured``
    contract by POSTing to the harness-managed sandbox LLM proxy.

    The adapter is intentionally minimal:

    - It calls only ``urllib.request`` / ``urllib.error`` so it imports
      cleanly inside any learner sandbox.
    - On any error (URLError, HTTPError, decode failure, schema
      mismatch) it returns an ``_AbstainResult`` instead of raising.
      :class:`LLMJudgeCoverage` then abstains gracefully (fail open).
    - It appends a schema-and-JSON-only instruction to the caller's
      ``system`` prompt so the proxy's plain-text response can be
      validated against the requested Pydantic ``text_format``.
    """

    def __init__(self, *, proxy_url: str | None = None, submission_token: str | None = None) -> None:
        self._proxy_url = (
            proxy_url
            or os.environ.get("COURSEGEN_LLM_PROXY_URL")
            or _DEFAULT_PROXY_URL
        ).rstrip("/")
        self._submission_token = submission_token or os.environ.get(
            "COURSEGEN_SUBMISSION_TOKEN"
        )

    def parse_structured(
        self,
        *,
        tier,
        system: str,
        user: str,
        text_format,
        max_tokens: int = 1000,
        request_timeout_s: float = 60.0,
        extra_request_kwargs: dict | None = None,
    ):
        tier_value = getattr(tier, "value", tier)
        if tier_value not in {"haiku", "sonnet"}:
            tier_value = "haiku"

        schema_hint = self._schema_hint(text_format)
        augmented_system = (
            f"{system}\\n\\n"
            "Reply with a single JSON object that validates against this "
            f"JSON Schema:\\n{schema_hint}\\n"
            "Do not wrap the JSON in markdown fences or prose. Output JSON only."
        )

        payload = {
            "tier": tier_value,
            "system": augmented_system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": int(max_tokens),
            "submission_token": self._submission_token,
        }

        url = f"{self._proxy_url}/v1/messages"
        try:
            req = urllib_request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=request_timeout_s) as resp:
                raw = resp.read()
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError):
            return _AbstainResult()
        except Exception:
            # Any other transport error (socket error, etc.) → abstain.
            return _AbstainResult()

        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return _AbstainResult()

        content = body.get("content") if isinstance(body, dict) else None
        if not isinstance(content, str) or not content.strip():
            return _AbstainResult()

        # The model may surround JSON with prose despite the system hint;
        # try a strict parse first, then a best-effort substring extract.
        parsed_payload = self._parse_json_lenient(content)
        if parsed_payload is None:
            return _AbstainResult()

        try:
            parsed_model = text_format.model_validate(parsed_payload)
        except Exception:
            return _AbstainResult()

        result = _AbstainResult()
        result.parsed = parsed_model
        result.output_parsed = parsed_model
        return result

    @staticmethod
    def _schema_hint(text_format) -> str:
        try:
            schema = text_format.model_json_schema()
            return json.dumps(schema)
        except Exception:
            # Fallback to the class name if the model lacks a JSON schema —
            # the rest of the prompt still asks for JSON.
            return getattr(text_format, "__name__", "object")

    @staticmethod
    def _parse_json_lenient(text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        # Extract the first {...} block. Cheap heuristic; the proxy's
        # system prompt asks for JSON-only, so this rarely fires.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None


def _here() -> Path:
    return Path(__file__).resolve().parent


def _load_setup_data(root: Path) -> dict:
    """Merge every top-level file under ``_setup/`` into a single dict
    keyed by stem. JSON files are parsed; everything else is kept as raw
    text. Mirrors :func:`app.services.oracle_pass._load_setup_data` so a
    course that passed oracle generation produces the same
    ``setup_data`` shape at grade time. Subdirectories and dotfiles are
    skipped.

    Oracle outputs at ``_oracle/outputs.json`` are merged in under the
    reserved key ``oracle`` so rubrics can pull them out via
    ``setup_data.oracle.<...>``.
    """
    setup_data: dict = {}
    setup_dir = root / "_setup"
    if setup_dir.is_dir():
        for path in sorted(setup_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            try:
                text = path.read_text()
            except Exception:
                setup_data[path.stem] = None
                continue
            if path.suffix.lower() == ".json":
                try:
                    setup_data[path.stem] = json.loads(text)
                except Exception:
                    # A malformed JSON setup file is a config issue;
                    # surface the raw text so rubrics that consult it
                    # at least see something.
                    setup_data[path.stem] = text
            else:
                setup_data[path.stem] = text

    oracle_path = root / "_oracle" / "outputs.json"
    if oracle_path.is_file():
        try:
            setup_data["oracle"] = json.loads(oracle_path.read_text())
        except Exception:
            setup_data["oracle"] = None
    return setup_data


def _verdict_to_diagnostic_lines(kind: str, verdict) -> list[str]:
    lines = [f"[{kind}] {verdict.status}: {verdict.rationale}"]
    if verdict.diagnostic:
        try:
            lines.append("  " + json.dumps(verdict.diagnostic, default=str))
        except Exception:
            lines.append("  " + repr(verdict.diagnostic))
    return lines


def _scenario_to_test_case(report) -> dict:
    """Convert a ScenarioVerdictReport into the harness test-case shape."""
    status_label = "passed" if report.overall_status == "pass" else "failed"
    diagnostics: list[str] = []
    if report.run_result.aborted and report.run_result.abort_reason:
        diagnostics.append(f"aborted: {report.run_result.abort_reason}")
    for kind, verdict in report.verdicts:
        diagnostics.extend(_verdict_to_diagnostic_lines(kind, verdict))

    fails = [v for _, v in report.verdicts if v.status == "fail"]
    abstains = [v for _, v in report.verdicts if v.status == "abstain"]
    if fails:
        summary = f"{len(fails)} rubric(s) failed"
    elif abstains:
        summary = f"{len(abstains)} rubric(s) abstained"
    else:
        summary = "all rubrics passed"

    return {
        "id": report.scenario_id,
        "title": report.scenario_id,
        "status": status_label,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def main() -> int:
    root = _here()
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    report_path = os.environ.get("REPORT_PATH")

    scenarios_dir = root / "scenarios"
    try:
        scenarios = load_scenarios_from_dir(scenarios_dir)
    except Exception as exc:
        msg = f"failed to load scenarios from {scenarios_dir}: {exc}"
        traceback.print_exc()
        payload = {
            "summary": msg,
            "tests": [
                {
                    "id": "scenario_loader",
                    "title": "scenario loader",
                    "status": "failed",
                    "summary": msg,
                    "diagnostics": [msg],
                }
            ],
        }
        out = json.dumps(payload, indent=2, default=str)
        if report_path:
            Path(report_path).write_text(out)
        else:
            sys.stdout.write(out + "\\n")
        return 2

    setup_data = _load_setup_data(root)
    router = SandboxLLMRouterAdapter()

    tests: list[dict] = []
    passed_count = 0
    for scenario in scenarios:
        try:
            report = run_scenario(
                scenario=scenario,
                base_url=base_url,
                setup_data=setup_data,
                router=router,
            )
        except Exception as exc:
            tests.append(
                {
                    "id": scenario.id,
                    "title": scenario.description or scenario.id,
                    "status": "failed",
                    "summary": f"runner crashed: {exc}",
                    "diagnostics": [traceback.format_exc()],
                }
            )
            continue
        case = _scenario_to_test_case(report)
        case["title"] = scenario.description or scenario.id
        tests.append(case)
        if case["status"] == "passed":
            passed_count += 1

    payload = {
        "summary": f"{passed_count} of {len(scenarios)} scenarios passed",
        "tests": tests,
    }
    out = json.dumps(payload, indent=2, default=str)
    if report_path:
        Path(report_path).write_text(out)
    else:
        sys.stdout.write(out + "\\n")

    return 0 if passed_count == len(scenarios) and tests else 1


if __name__ == "__main__":
    sys.exit(main())
'''


__all__ = ["GRADER_RUNNER_SCRIPT_SOURCE"]
