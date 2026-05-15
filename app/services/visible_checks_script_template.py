"""Source-of-truth string template for the visible self-test script.

When the harness materializes a benchmark-backed learner bundle, it
writes :data:`VISIBLE_CHECKS_SCRIPT_SOURCE` to
``public/checks/run_visible_checks.py`` (sibling to
``public/examples/sample_queries.json``). The script is a lightweight
shape + plausibility check the learner runs against their booted
service BEFORE submitting — it does NOT run the full hidden grader and
does NOT call any LLM judge.

The script intentionally has zero non-stdlib dependencies. It must run
inside whatever Python environment the learner picks for their own
service, even when network provider SDKs (``anthropic``, ``openai``,
etc.) aren't installed. The rubric library lives in the hidden grader
under ``private/grader/runner.py`` — visible checks don't reach for it.

Storing the script as a string here keeps it version-controlled and
unit-testable; the materializer just calls
``Path(...).write_text(VISIBLE_CHECKS_SCRIPT_SOURCE)``.

Environment:
    BASE_URL     Base URL of the learner-service under test. Defaults
                 to ``http://localhost:8000``.
    REPORT_PATH  When set, the JSON report is written here. Otherwise
                 the report is printed to stdout.

Report shape (``{"summary": str, "tests": [...]}``) matches the
existing harness contract so ``generated_test_harness.py`` can consume
this script the same way it consumes the hidden grader runner.

Exit code:
    0 when every visible check passed.
    1 when any check failed.
    2 on a runner-internal error (config / file missing).
"""
from __future__ import annotations


VISIBLE_CHECKS_SCRIPT_SOURCE = '''#!/usr/bin/env python3
"""Visible self-test harness for benchmark-backed learner bundles.

Reads sample queries from ``public/examples/sample_queries.json``
(sibling tree to this script), fires each against the learner-service
under ``BASE_URL``, runs a lightweight shape + plausibility check, and
emits a ``{"summary","tests":[...]}`` JSON report.

This is the VISIBLE check — pass/fail here means "my service responds
in roughly the right shape," NOT "my service hits the quality bars."
The full LLM-judged grading happens behind ``private/grader/runner.py``.

Environment:
    BASE_URL     Learner-service base URL. Defaults to
                 ``http://localhost:8000``.
    REPORT_PATH  When set, the JSON report is written here. Otherwise
                 the report is printed to stdout.

Exit code:
    0 every visible check passed.
    1 any check failed.
    2 runner-internal error (missing sample_queries.json, etc.).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT_S = 30.0


# ---------------- sample loading ----------------


def _here() -> Path:
    return Path(__file__).resolve().parent


def _locate_sample_queries() -> Path:
    """Find ``sample_queries.json``.

    The script ships at ``public/checks/run_visible_checks.py`` and the
    samples ship at ``public/examples/sample_queries.json``. We look in
    the canonical sibling location first, then fall back to a search
    under the workspace root (handy for tests / oddball layouts).
    """
    canonical = _here().parent / "examples" / "sample_queries.json"
    if canonical.is_file():
        return canonical
    # Fall back: search upward for any sample_queries.json next to an
    # ``examples/`` directory.
    cur = _here()
    for _ in range(6):
        candidate = cur / "public" / "examples" / "sample_queries.json"
        if candidate.is_file():
            return candidate
        cur = cur.parent
    raise FileNotFoundError(
        "could not locate sample_queries.json under public/examples/"
    )


def _load_samples(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"sample_queries.json must be a JSON array, got {type(raw).__name__}"
        )
    return raw


# ---------------- HTTP helpers (stdlib only) ----------------


def _http_post_json(url: str, body: dict, *, timeout: float) -> tuple[int, dict | None, str]:
    """POST a JSON body and return ``(status, parsed_json_or_None, raw_text)``.

    Never raises — all transport errors are squashed into a ``status=0``
    return so the visible check can mark the case failed without
    crashing the whole run.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            text = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        status = exc.code
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        return (0, None, f"transport_error: {exc}")
    parsed: dict | None = None
    try:
        parsed_any = json.loads(text)
        if isinstance(parsed_any, dict):
            parsed = parsed_any
    except (json.JSONDecodeError, ValueError):
        parsed = None
    return (status, parsed, text)


# ---------------- sample shape detection ----------------


def _is_crag_sample(sample: dict) -> bool:
    return "expected_answer" in sample and "search_results" in sample


def _is_beir_sample(sample: dict) -> bool:
    return "acceptable_doc_ids" in sample


# ---------------- per-sample check ----------------


def _check_crag_sample(base_url: str, sample: dict) -> dict:
    """Fire a CRAG-style sample at ``POST /answers`` and validate shape."""
    url = base_url.rstrip("/") + "/answers"
    body = {
        "question": sample.get("question", ""),
        "search_results": list(sample.get("search_results", [])),
        "query_id": sample.get("query_id", ""),
    }
    status, parsed, raw = _http_post_json(url, body, timeout=_DEFAULT_TIMEOUT_S)
    diagnostics: list[str] = []
    if status != 200:
        diagnostics.append(f"expected HTTP 200, got {status}")
        if raw:
            diagnostics.append(f"response: {raw[:300]}")
    if parsed is None and status == 200:
        diagnostics.append("response body was not valid JSON")
    # Required fields for a CRAG answer payload.
    if parsed is not None:
        for field in ("answer", "confidence", "abstained"):
            if field not in parsed:
                diagnostics.append(f"missing required field '{field}'")
        cited = parsed.get("cited_chunks", parsed.get("cited_doc_ids"))
        if cited is None:
            diagnostics.append("missing required field 'cited_chunks' or 'cited_doc_ids'")
        answer = parsed.get("answer", "")
        abstained = bool(parsed.get("abstained", False))
        if not answer and not abstained:
            diagnostics.append("answer is empty and abstained != true")
    return {
        "id": "visible_" + sample.get("query_id", "unknown"),
        "title": f"CRAG sample {sample.get('query_id', '?')}",
        "status": "pass" if not diagnostics else "fail",
        "summary": "answer payload looks plausible"
        if not diagnostics
        else "; ".join(diagnostics[:3]),
        "diagnostics": diagnostics,
    }


def _check_beir_sample(base_url: str, sample: dict) -> dict:
    """Fire a BeIR retrieval sample at ``POST /retrieve`` and validate
    that the response cites at least one acceptable doc id."""
    url = base_url.rstrip("/") + "/retrieve"
    body = {
        "question": sample.get("question", ""),
        "query_id": sample.get("query_id", ""),
    }
    status, parsed, raw = _http_post_json(url, body, timeout=_DEFAULT_TIMEOUT_S)
    diagnostics: list[str] = []
    if status != 200:
        diagnostics.append(f"expected HTTP 200, got {status}")
        if raw:
            diagnostics.append(f"response: {raw[:300]}")
    if parsed is None and status == 200:
        diagnostics.append("response body was not valid JSON")
    accepted = set(sample.get("acceptable_doc_ids") or [])
    if parsed is not None:
        cited = parsed.get("cited_doc_ids") or parsed.get("cited_chunks") or []
        if not isinstance(cited, list):
            diagnostics.append("'cited_doc_ids' is not a list")
            cited = []
        cited_ids: set[str] = set()
        for entry in cited:
            if isinstance(entry, str):
                cited_ids.add(entry)
            elif isinstance(entry, dict) and "doc_id" in entry:
                cited_ids.add(str(entry["doc_id"]))
        overlap = cited_ids & accepted
        if not overlap:
            diagnostics.append(
                f"cited_doc_ids {sorted(cited_ids)[:5]} contain no acceptable "
                f"doc from {sorted(accepted)[:5]}"
            )
    return {
        "id": "visible_" + sample.get("query_id", "unknown"),
        "title": f"BeIR sample {sample.get('query_id', '?')}",
        "status": "pass" if not diagnostics else "fail",
        "summary": "retrieval cites at least one acceptable doc"
        if not diagnostics
        else "; ".join(diagnostics[:3]),
        "diagnostics": diagnostics,
    }


# ---------------- main ----------------


def main() -> int:
    base_url = os.environ.get("BASE_URL", _DEFAULT_BASE_URL)
    report_path = os.environ.get("REPORT_PATH")

    try:
        samples_path = _locate_sample_queries()
        samples = _load_samples(samples_path)
    except Exception as exc:
        envelope = {
            "summary": f"visible checks could not start: {exc}",
            "tests": [
                {
                    "id": "visible_bootstrap",
                    "title": "load sample_queries.json",
                    "status": "fail",
                    "summary": str(exc),
                    "diagnostics": [str(exc)],
                }
            ],
        }
        _emit_report(envelope, report_path)
        return 2

    tests: list[dict] = []
    for sample in samples:
        if _is_crag_sample(sample):
            tests.append(_check_crag_sample(base_url, sample))
        elif _is_beir_sample(sample):
            tests.append(_check_beir_sample(base_url, sample))
        else:
            tests.append(
                {
                    "id": "visible_" + sample.get("query_id", "unknown"),
                    "title": f"sample {sample.get('query_id', '?')}",
                    "status": "fail",
                    "summary": "unknown sample shape (neither CRAG nor BeIR)",
                    "diagnostics": ["sample is missing both 'expected_answer' and 'acceptable_doc_ids'"],
                }
            )

    pass_count = sum(1 for t in tests if t["status"] == "pass")
    envelope = {
        "summary": f"visible checks: {pass_count}/{len(tests)} passed against {base_url}",
        "tests": tests,
    }
    _emit_report(envelope, report_path)
    return 0 if pass_count == len(tests) else 1


def _emit_report(envelope: dict, report_path: str | None) -> None:
    payload = json.dumps(envelope, indent=2, sort_keys=True)
    if report_path:
        Path(report_path).write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    raise SystemExit(main())
'''


__all__ = ["VISIBLE_CHECKS_SCRIPT_SOURCE"]
