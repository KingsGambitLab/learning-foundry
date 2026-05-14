"""Scenario trace runner.

Given a parsed :class:`Scenario`, this module walks the ``setup`` and
``trace`` step lists in order, issues HTTP requests through an injected
:class:`ScenarioHttpClient`, captures each step's response into a keyed
dict, and then evaluates every configured rubric against the resulting
:class:`RubricContext`. The returned :class:`ScenarioVerdictReport`
carries both the raw run trace (status codes, latencies, expectation
results, captures) and the rubric verdicts.

Three pieces are surfaced at the public boundary:

- :func:`interpolate` resolves ``${id.dotted.path}`` placeholder strings
  against captured responses. Two-segment shorthand auto-prefixes
  ``body`` when the second segment isn't ``status`` / ``headers`` /
  ``body`` — so ``${created.short_code}`` resolves
  ``created.body.short_code``. Multi-segment paths (3+ segments) are
  treated as explicit and never auto-prefixed.

- :class:`ScenarioHttpClient` is a ``Protocol`` so tests can swap in a
  ``FakeHttpClient`` with canned responses. A default
  :class:`UrllibHttpClient` ships for the on-disk runner script.

- :func:`run_scenario` is the orchestration entry point.

Design notes:

* Rubric instantiation uses ``inspect.signature`` to detect whether the
  rubric's ``__init__`` accepts a ``router`` parameter. Only LLM-judge
  rubrics declare it; structural / set / oracle rubrics don't. This
  keeps the runner free of any ``isinstance``-style rubric classification.

* When a step's ``expect`` fails, ``aborted`` is set, subsequent trace
  steps are skipped, and rubrics still run against whatever was
  captured. The rubrics themselves decide whether the missing data is
  a learner failure or a config issue — the runner does not second-guess.

* Bodies are parsed once: when the response's ``Content-Type`` looks
  like JSON, we try ``json.loads`` and fall back to the raw text if it
  cannot be parsed. Empty bodies become ``None``.
"""
from __future__ import annotations

import inspect
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Literal, Mapping, Protocol

_LOGGER = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from app.services.scenario_loader import HttpExpectation, Scenario, TraceStep
from app.services.scenario_rubrics_base import (
    RUBRIC_REGISTRY,
    RubricContext,
    Verdict,
    resolve_path,
)

# ---------------- Interpolation ----------------


class InterpolationError(Exception):
    """Raised when a ``${id.dotted.path}`` placeholder cannot be resolved.

    The message names the offending placeholder so caller / test
    diagnostics show which path the runner gave up on.
    """


_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")
_CAPTURE_PARTS = frozenset({"status", "headers", "body"})


def _resolve_one_placeholder(
    expression: str,
    captures: Mapping[str, Any],
    *,
    setup_data: Mapping[str, Any] | None = None,
    course_meta: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve a single ``${id.dotted.path}`` placeholder.

    Conventions for the leading segment:

    - ``${setup_data.X.Y}`` resolves against ``setup_data`` (the grader's
      hidden / curated data). This is the convention the LLM keeps
      reaching for and the convention curated scenarios use to inline
      a question + its retrieval pool from disk: e.g.
      ``${setup_data.queries.q_0001.query}`` and
      ``${setup_data.search_results_index.q_0001}``.
    - ``${course_meta.X.Y}`` resolves against ``course_meta``.
    - Otherwise the leading segment is a capture id (a previous trace
      step's id), with the original body-prefix shorthand applied.

    Raises :class:`InterpolationError` on any failure (unknown id, bad
    key, out-of-range index).
    """
    segments = expression.split(".")
    leading = segments[0]

    # ---- setup_data / course_meta routing (Bug 27) ----
    if leading == "setup_data":
        if setup_data is None:
            raise InterpolationError(
                f"Placeholder '${{{expression}}}' references setup_data but "
                f"this trace step has no setup_data context."
            )
        try:
            return resolve_path(setup_data, ".".join(segments[1:]))
        except (KeyError, IndexError, TypeError) as exc:
            raise InterpolationError(
                f"Could not resolve '${{{expression}}}' against setup_data: {exc}"
            ) from exc
    if leading == "course_meta":
        if course_meta is None:
            raise InterpolationError(
                f"Placeholder '${{{expression}}}' references course_meta but "
                f"this trace step has no course_meta context."
            )
        try:
            return resolve_path(course_meta, ".".join(segments[1:]))
        except (KeyError, IndexError, TypeError) as exc:
            raise InterpolationError(
                f"Could not resolve '${{{expression}}}' against course_meta: {exc}"
            ) from exc

    # ---- captures (legacy path) ----
    capture_id = leading
    if capture_id not in captures:
        raise InterpolationError(
            f"Unknown capture id '{capture_id}' in placeholder '${{{expression}}}'"
        )

    if len(segments) == 1:
        return captures[capture_id]

    second = segments[1]
    remainder = segments[1:]
    if len(remainder) == 1 and second not in _CAPTURE_PARTS:
        # Two-segment shorthand: auto-prefix "body".
        remainder = ["body"] + remainder

    dotted = ".".join(remainder)
    try:
        return resolve_path(captures[capture_id], dotted)
    except (KeyError, IndexError, TypeError) as exc:
        raise InterpolationError(
            f"Could not resolve '${{{expression}}}': {exc}"
        ) from exc


def interpolate(
    template: str,
    captures: Mapping[str, Any],
    *,
    setup_data: Mapping[str, Any] | None = None,
    course_meta: Mapping[str, Any] | None = None,
) -> str:
    """Resolve every ``${...}`` placeholder in ``template``.

    Default context is ``captures`` (legacy behavior). Pass
    ``setup_data`` / ``course_meta`` to enable
    ``${setup_data.X.Y}`` / ``${course_meta.X.Y}`` placeholders. The
    result is always a string — non-string resolved values are
    stringified via ``str()`` so a placeholder like ``${step.status}``
    can be embedded in a path. Multiple placeholders per string are
    supported.

    Raises :class:`InterpolationError` if any placeholder fails to
    resolve.
    """

    def _sub(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        return str(
            _resolve_one_placeholder(
                expression,
                captures,
                setup_data=setup_data,
                course_meta=course_meta,
            )
        )

    return _PLACEHOLDER_PATTERN.sub(_sub, template)


def _interpolate_body(
    body: Any,
    captures: Mapping[str, Any],
    *,
    setup_data: Mapping[str, Any] | None = None,
    course_meta: Mapping[str, Any] | None = None,
) -> Any:
    """Recursively walk a request body and interpolate every string leaf.

    Dicts and lists are recursed into. Non-string, non-container values
    (incl. dicts/lists that come back from a ``setup_data`` interpolation
    like ``${setup_data.search_results_index.q1}``) pass through whole.

    ``None`` passes through.
    """
    if body is None:
        return None
    if isinstance(body, str):
        # Pass-through for whole-template placeholders that resolve to
        # non-string values (e.g. ``"${setup_data.search_results_index.q1}"``
        # must return the resolved LIST, not its string repr — the
        # downstream service receives a JSON array, not a stringified
        # array).
        stripped = body.strip()
        match = _PLACEHOLDER_PATTERN.fullmatch(stripped)
        if match is not None:
            return _resolve_one_placeholder(
                match.group(1).strip(),
                captures,
                setup_data=setup_data,
                course_meta=course_meta,
            )
        return interpolate(
            body, captures, setup_data=setup_data, course_meta=course_meta
        )
    if isinstance(body, dict):
        return {
            k: _interpolate_body(
                v, captures, setup_data=setup_data, course_meta=course_meta
            )
            for k, v in body.items()
        }
    if isinstance(body, list):
        return [
            _interpolate_body(
                v, captures, setup_data=setup_data, course_meta=course_meta
            )
            for v in body
        ]
    return body


# ---------------- Pydantic result models ----------------


class TraceStepResult(BaseModel):
    """One HTTP step's outcome.

    ``expect_passed`` is ``True`` when the step's :class:`HttpExpectation`
    matched OR when no expectation was set. ``expect_diagnostic`` is
    ``None`` on pass; otherwise a human-readable explanation suitable
    for the abort reason.
    """

    step_id: str
    status: int
    headers: dict[str, str]
    body: Any = None
    latency_ms: float
    expect_passed: bool
    expect_diagnostic: str | None = None


class ScenarioRunResult(BaseModel):
    """All trace-level data produced by walking one scenario.

    ``captures`` is keyed by step ``id`` (and additionally by the step's
    ``capture`` field when set, which mirrors the same data under that
    alias). Each capture entry has the shape
    ``{"status": int, "headers": dict, "body": Any}``.
    """

    scenario_id: str
    setup_results: list[TraceStepResult] = Field(default_factory=list)
    trace_results: list[TraceStepResult] = Field(default_factory=list)
    captures: dict[str, Any] = Field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None


class ScenarioVerdictReport(BaseModel):
    """Scenario run + rubric verdicts.

    ``verdicts`` is a list of ``(rubric_kind, Verdict)`` pairs in the
    order the scenario declared its rubrics. ``overall_status`` is
    derived: any ``fail`` makes the overall ``fail``; any ``abstain``
    among otherwise-passing rubrics makes it ``abstain``; only when
    every rubric returned ``pass`` is the overall ``pass``.
    """

    scenario_id: str
    category: str
    run_result: ScenarioRunResult
    verdicts: list[tuple[str, Verdict]] = Field(default_factory=list)

    @property
    def overall_status(self) -> Literal["pass", "fail", "abstain"]:
        statuses = [v.status for _, v in self.verdicts]
        if not statuses:
            return "pass"
        if any(s == "fail" for s in statuses):
            return "fail"
        if all(s == "pass" for s in statuses):
            return "pass"
        return "abstain"


# ---------------- HTTP client protocol ----------------


class ScenarioHttpClient(Protocol):
    """HTTP transport contract for the trace runner.

    Implementations must return ``(status, headers, body)`` where
    ``headers`` is a lowercased-key dict and ``body`` is the parsed
    JSON value when the response is JSON-typed, the raw text when it
    isn't, or ``None`` on empty body.
    """

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any | None,
        follow_redirects: bool,
        timeout: float,
    ) -> tuple[int, dict[str, str], Any]:
        ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """``urllib`` handler that stops following redirects.

    Returning ``None`` from each ``redirect_request`` hook causes the
    library to surface the 3xx response to the caller verbatim instead
    of issuing the redirected request.
    """

    def redirect_request(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        return None


def _parse_response_body(raw_bytes: bytes, content_type: str) -> Any:
    if not raw_bytes:
        return None
    text = raw_bytes.decode("utf-8", errors="replace")
    if "application/json" in content_type.lower() or "+json" in content_type.lower():
        try:
            return json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return text
    return text


class UrllibHttpClient:
    """Default :class:`ScenarioHttpClient` backed by ``urllib.request``.

    Suitable for the on-disk grader runner where pulling in ``httpx``
    or ``requests`` would add a dependency. Body payloads that are
    dicts/lists are JSON-encoded; bytes pass through; strings encode as
    UTF-8.
    """

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any | None,
        follow_redirects: bool,
        timeout: float,
    ) -> tuple[int, dict[str, str], Any]:
        data: bytes | None
        send_headers = dict(headers)
        if body is None:
            data = None
        elif isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        elif isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = json.dumps(body).encode("utf-8")
            send_headers.setdefault("content-type", "application/json")

        req = urllib.request.Request(
            url=url, data=data, method=method.upper(), headers=send_headers
        )
        if follow_redirects:
            opener = urllib.request.build_opener()
        else:
            opener = urllib.request.build_opener(_NoRedirectHandler())

        try:
            with opener.open(req, timeout=timeout) as resp:
                status = resp.status
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            resp_headers = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
            raw = exc.read() if hasattr(exc, "read") else b""

        body_out = _parse_response_body(raw, resp_headers.get("content-type", ""))
        return status, resp_headers, body_out


# ---------------- Scenario execution ----------------


def _expect_diagnostic(
    expectation: HttpExpectation | None, status: int
) -> tuple[bool, str | None]:
    """Apply ``expectation`` to a response status.

    Returns ``(passed, diagnostic)`` where diagnostic is ``None`` on
    pass and a short explanation on fail.
    """
    if expectation is None or expectation.status is None:
        return True, None
    expected = expectation.status
    if isinstance(expected, int):
        ok = status == expected
        if ok:
            return True, None
        return False, f"expected status {expected}, got {status}"
    # list of ints
    ok = status in expected
    if ok:
        return True, None
    return False, f"expected status in {expected}, got {status}"


def _resolve_step_url(
    base_url: str,
    path_template: str,
    captures: Mapping[str, Any],
    *,
    setup_data: Mapping[str, Any] | None = None,
    course_meta: Mapping[str, Any] | None = None,
) -> str:
    resolved_path = interpolate(
        path_template, captures, setup_data=setup_data, course_meta=course_meta
    )
    if resolved_path.startswith(("http://", "https://")):
        return resolved_path
    if not resolved_path.startswith("/"):
        resolved_path = "/" + resolved_path
    return base_url.rstrip("/") + resolved_path


def _execute_step(
    step: TraceStep,
    *,
    base_url: str,
    captures: dict[str, Any],
    http_client: ScenarioHttpClient,
    timeout: float,
    setup_data: Mapping[str, Any] | None = None,
    course_meta: Mapping[str, Any] | None = None,
) -> TraceStepResult:
    url = _resolve_step_url(
        base_url,
        step.path,
        captures,
        setup_data=setup_data,
        course_meta=course_meta,
    )
    headers = {
        k: interpolate(v, captures, setup_data=setup_data, course_meta=course_meta)
        for k, v in step.headers.items()
    }
    body = _interpolate_body(
        step.body, captures, setup_data=setup_data, course_meta=course_meta
    )

    import time

    started = time.perf_counter()
    status, resp_headers, resp_body = http_client.request(
        method=step.method,
        url=url,
        headers=headers,
        body=body,
        follow_redirects=step.follow_redirects,
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    expect_passed, diagnostic = _expect_diagnostic(step.expect, status)

    capture_entry = {
        "status": status,
        "headers": resp_headers,
        "body": resp_body,
    }
    captures[step.id] = capture_entry
    if step.capture and step.capture != step.id:
        captures[step.capture] = capture_entry

    return TraceStepResult(
        step_id=step.id,
        status=status,
        headers=resp_headers,
        body=resp_body,
        latency_ms=latency_ms,
        expect_passed=expect_passed,
        expect_diagnostic=diagnostic,
    )


# Observed-in-the-wild kwarg drift between LLM-emitted rubric configs and
# the rubric class signatures, harvested from /tmp/coursegen-*.log:
#
#   56× SchemaMatch.schema      -> must_have_fields (extracted from JSON Schema)
#   28× LiteralMatch.value      -> expected
#   26× OracleSetOverlap.oracle_set -> inline list, would need a path
#   26× LLMJudgeSemanticEq.gold -> inline, would need a path
#   22× NumericRange.min        -> min_value
#   22× SchemaMatch.value       -> must_have_fields (same as .schema)
#   14× LLMJudgeCoverage.question -> drop (not a kwarg)
#   12× SubsetMatch.value       -> inline list, would need acceptable_source path
#   12× LLMJudgeCoverage.reference -> drop
#   10× OracleSetOverlap.reference_set -> inline, would need a path
#   10× OracleSetOverlap.gold_path -> gold_set_path
#    ...
#
# The simple renames (LiteralMatch.value, NumericRange.min, etc.) we can
# handle without any data shuffling. For inline-data cases (oracle_set,
# gold, schema), the rubric expects a path into ctx.setup_data; we
# either translate when feasible (JSON Schema -> required field list)
# or drop the kwarg so the rubric fails predictably with a missing-arg
# TypeError that surfaces in the diagnostics, instead of a confusing
# "unexpected keyword argument" exception.

# Simple rename map: (rubric_kind, llm_kwarg) -> canonical_kwarg.
_RUBRIC_KWARG_ALIASES: dict[tuple[str, str], str] = {
    ("literal_match", "value"): "expected",
    ("numeric_range", "min"): "min_value",
    ("numeric_range", "max"): "max_value",
    ("oracle_set_overlap", "gold_path"): "gold_set_path",
    ("oracle_set_overlap", "min_overlap"): "min_recall",
    # ``value`` is the most common LLM kwarg drift here. It can be
    # either a literal value or a string-shaped path; the rubric now
    # treats string values that LOOK like dotted paths as path-vs-path
    # via the heuristic in ``_normalize_rubric_kwargs`` below.
    ("behavioral_equivalence", "value"): "expected",
    ("behavioral_equivalence", "reference_target"): "expected_path",
    ("behavioral_equivalence", "target_a"): "target",
    ("behavioral_equivalence", "target_b"): "expected_path",
    ("behavioral_equivalence", "reference_trace"): "expected_path",
    ("subset_match", "value"): "acceptable_source",
    ("subset_match", "subset_of"): "acceptable_source",
    ("llm_judge_semantic_eq", "gold"): "gold_path",
    ("llm_judge_coverage", "answer_target"): "target",
    ("llm_judge_coverage", "facts"): "must_contain_facts",
}

# Kwargs the LLM keeps emitting that have no rubric counterpart and
# should just be dropped (rather than rejected as "unexpected").
_RUBRIC_KWARGS_TO_DROP: dict[str, set[str]] = {
    "llm_judge_coverage": {"question", "reference", "citations_target", "evidence"},
    "llm_judge_false_premise": {"question", "evidence"},
    "oracle_set_overlap": {"reference", "selection_mode"},
    "schema_match": {"value"},
}


def _extract_must_have_fields_from_schema(schema: Any) -> list[str]:
    """Pull the field list out of a JSON Schema object the LLM emitted
    under ``schema_match.schema``. Conservative: only honor an explicit
    top-level ``required`` array; otherwise fall back to the keys of
    ``properties``."""
    if not isinstance(schema, dict):
        return []
    required = schema.get("required")
    if isinstance(required, list) and all(isinstance(x, str) for x in required):
        return list(required)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return list(properties.keys())
    return []


# Rubrics whose path kwargs resolve in the MERGED context (captures +
# setup_data + course_meta) — these require a ``setup_data.`` prefix
# when the path points into ``ctx.setup_data``. If the LLM emits a
# bare path (no prefix) and the path looks like it's trying to address
# setup_data, we auto-prefix it here so the rubric can resolve it.
_MERGED_CONTEXT_PATH_KWARGS: dict[str, tuple[str, ...]] = {
    "llm_judge_semantic_eq": ("gold_path", "alt_path"),
    "llm_judge_false_premise": ("expected_falsity_path",),
    "llm_judge_coverage": ("judge_context_path",),
}


def _looks_like_setup_data_path(path: str) -> bool:
    """Heuristic: does ``path`` address a key under setup_data without
    the ``setup_data.`` prefix?

    The bare convention is ``<file_stem>.<key>...`` where ``<file_stem>``
    matches a setup_data file (e.g. ``gold_answers``, ``gold_supports``,
    ``queries``). We don't have ctx.setup_data at normalize time, so
    use a deny-list: paths that obviously address captures (start with
    ``$.``, contain ``response.body``, start with a trace-step-id
    prefix like ``call_``, ``answer_``) are left alone.
    """
    if path.startswith("setup_data.") or path.startswith("course_meta."):
        return False  # already correctly prefixed
    if path.startswith("$.") or path.startswith("captures."):
        return False  # captures path
    if "." not in path:
        return False  # single segment is ambiguous; leave it
    first = path.split(".", 1)[0]
    capture_like = (
        "response",
        "call",
        "trace",
        "request",
    )
    if any(first.startswith(p) for p in capture_like):
        return False
    if first in {"setup_data", "course_meta", "captures"}:
        return False
    # Bare path with a multi-segment file-stem look: prefix it.
    return True


def _normalize_rubric_kwargs(spec_kind: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate observed-in-the-wild LLM-emitted rubric kwargs to the
    canonical rubric-class kwargs.

    This is intentionally LOSSY: kwargs without a known canonical name
    are dropped (per ``_RUBRIC_KWARGS_TO_DROP``) or left alone so the
    downstream ``inspect.signature`` filter trims them.

    Also normalizes path values for the merged-context rubrics so the
    LLM can emit either ``setup_data.X.Y`` or ``X.Y`` and both resolve
    correctly (Bug 16 in the autonomous-fix loop tracker).
    """
    out: dict[str, Any] = {}
    drop_set = _RUBRIC_KWARGS_TO_DROP.get(spec_kind, set())
    for key, value in kwargs.items():
        if key in drop_set:
            continue
        canonical = _RUBRIC_KWARG_ALIASES.get((spec_kind, key), key)
        out[canonical] = value
    # Special: ``schema_match.schema`` (a JSON Schema dict) → derive
    # ``must_have_fields`` from ``schema.required`` or
    # ``schema.properties.keys()``.
    if spec_kind == "schema_match" and "schema" in out:
        schema_obj = out.pop("schema")
        if "must_have_fields" not in out:
            out["must_have_fields"] = _extract_must_have_fields_from_schema(
                schema_obj
            )
    # Special: ``oracle_set_overlap.value`` (inline list) appears in
    # ``_RUBRIC_KWARGS_TO_DROP`` for schema_match but not for oracle_set
    # — at this layer we leave inline-set handling to the rubric (it
    # will raise a missing-arg TypeError, which our outer try/except in
    # run_scenario converts to a fail Verdict with a clear message).

    # Path-prefix normalization for merged-context rubrics. The LLM
    # tends to emit bare paths into setup_data (``"gold_answers.q1"``)
    # which the merged-context resolver can't reach — that resolver
    # walks ``{**captures, "setup_data": setup_data}`` so paths into
    # setup_data MUST start with ``setup_data.``. Auto-prefix when we
    # can detect the LLM forgot.
    merged_kwargs = _MERGED_CONTEXT_PATH_KWARGS.get(spec_kind, ())
    for kw in merged_kwargs:
        if kw not in out:
            continue
        val = out[kw]
        if isinstance(val, str) and _looks_like_setup_data_path(val):
            out[kw] = f"setup_data.{val}"
    return out


def _build_rubric(spec_kind: str, spec_config: dict[str, Any], router: Any) -> Any:
    """Instantiate one rubric, injecting ``router`` only when the class
    accepts it.

    Applies an LLM-kwarg normalization pass first (rename / drop)
    because the scenario-author LLM persistently emits names like
    ``literal_match.value`` (canonical: ``expected``) and
    ``numeric_range.min`` (canonical: ``min_value``). Without this
    layer every scenario fails to even construct its rubrics and the
    trace runner converts each ``TypeError`` into a fail Verdict —
    which is what happened on the live RAG/CRAG smoke (see
    /tmp/coursegen-resume-*.log for the full kwarg-drift tally).
    """
    cls = RUBRIC_REGISTRY[spec_kind]
    kwargs = _normalize_rubric_kwargs(spec_kind, dict(spec_config))
    try:
        sig = inspect.signature(cls.__init__)
        accepts_router = "router" in sig.parameters
    except (TypeError, ValueError):
        accepts_router = False
    if accepts_router and router is not None and "router" not in kwargs:
        kwargs["router"] = router
    return cls(**kwargs)


def run_scenario(
    *,
    scenario: Scenario,
    base_url: str,
    router: Any = None,
    setup_data: dict[str, Any] | None = None,
    course_meta: dict[str, Any] | None = None,
    http_client: ScenarioHttpClient | None = None,
    timeout: float = 30.0,
) -> ScenarioVerdictReport:
    """Run one scenario end-to-end and return a verdict report.

    The runner walks ``scenario.setup`` then ``scenario.trace`` step by
    step, interpolating ``${...}`` placeholders, dispatching through
    ``http_client``, and applying each step's ``expect``. On the first
    failed expectation the trace aborts: subsequent steps are skipped,
    ``aborted`` is set, and rubrics still run against the partial
    captures.
    """
    client = http_client or UrllibHttpClient()
    captures: dict[str, Any] = {}
    aborted = False
    abort_reason: str | None = None

    setup_results: list[TraceStepResult] = []
    for step in scenario.setup:
        result = _execute_step(
            step,
            base_url=base_url,
            captures=captures,
            http_client=client,
            timeout=timeout,
            setup_data=setup_data,
            course_meta=course_meta,
        )
        setup_results.append(result)
        if not result.expect_passed:
            aborted = True
            abort_reason = (
                f"setup step '{step.id}' failed expectation: {result.expect_diagnostic}"
            )
            break

    trace_results: list[TraceStepResult] = []
    if not aborted:
        for step in scenario.trace:
            result = _execute_step(
                step,
                base_url=base_url,
                captures=captures,
                http_client=client,
                timeout=timeout,
            )
            trace_results.append(result)
            if not result.expect_passed:
                aborted = True
                abort_reason = (
                    f"step '{step.id}' failed expectation: {result.expect_diagnostic}"
                )
                break

    ctx = RubricContext(
        captures=captures,
        setup_data=dict(setup_data) if setup_data else {},
        course_meta=dict(course_meta) if course_meta else {},
    )

    verdicts: list[tuple[str, Verdict]] = []
    for spec in scenario.rubrics:
        # Defence in depth: a rubric library bug or an unexpected
        # learner payload that slipped past per-rubric validation must
        # NOT cascade into a total scenario crash. Convert any exception
        # to a fail Verdict with the exception type/message in the
        # diagnostic, log a WARNING, and keep evaluating remaining
        # rubrics. The rubric contract says pass/fail logic is encoded
        # in the Verdict — anything that raises is by definition a bug
        # we want surfaced but not fatal.
        try:
            rubric = _build_rubric(spec.kind, dict(spec.config), router)
            verdict = rubric.judge(ctx)
        except Exception as exc:  # noqa: BLE001 — defensive boundary
            exc_type = type(exc).__name__
            exc_message = str(exc)
            _LOGGER.warning(
                "Rubric %r raised %s during judge(): %s",
                spec.kind,
                exc_type,
                exc_message,
                exc_info=True,
            )
            verdict = Verdict(
                status="fail",
                rationale=(
                    f"Rubric '{spec.kind}' raised {exc_type}: {exc_message}"
                ),
                diagnostic={
                    "exception_type": exc_type,
                    "exception_message": exc_message,
                },
            )
        verdicts.append((spec.kind, verdict))

    return ScenarioVerdictReport(
        scenario_id=scenario.id,
        category=scenario.category,
        run_result=ScenarioRunResult(
            scenario_id=scenario.id,
            setup_results=setup_results,
            trace_results=trace_results,
            captures=captures,
            aborted=aborted,
            abort_reason=abort_reason,
        ),
        verdicts=verdicts,
    )


__all__ = [
    "InterpolationError",
    "ScenarioHttpClient",
    "ScenarioRunResult",
    "ScenarioVerdictReport",
    "TraceStepResult",
    "UrllibHttpClient",
    "interpolate",
    "run_scenario",
]
