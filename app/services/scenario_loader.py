"""Scenario YAML schema + loader.

A scenario is a single YAML file under ``private/grader/scenarios/``
describing one curated learner-service interaction sequence (the
``trace``) and the rubrics that judge it. The loader is parse-only:

- It validates structure via Pydantic models (``Scenario``,
  ``TraceStep``, ``HttpExpectation``, ``RubricSpec``).
- It validates every rubric ``kind`` against
  ``RUBRIC_REGISTRY`` from ``scenario_rubrics_base``. To make sure the
  registry is populated, this module eagerly imports the four rubric
  implementation modules at first use via
  ``_ensure_rubrics_registered()``.
- It does NOT execute HTTP requests (that's the trace runner).
- It does NOT resolve ``${id.path}`` interpolation (also runner's job).
  Variable-substitution strings pass through verbatim.
- It does NOT deduplicate across multiple directories; duplicate ids
  within a single directory are rejected.

``RubricSpec``'s config-packing pattern:
A rubric in YAML looks like ::

    - kind: schema_match
      target: resp.body
      must_have_fields: [answer, citations]

Authors can also write the equivalent explicit form ::

    - kind: schema_match
      config:
        target: resp.body
        must_have_fields: [answer, citations]

A ``model_validator(mode="before")`` collapses both shapes into a
single internal representation: any non-``kind`` / non-``config``
fields are packed into ``config``. The trace runner later
splat-applies ``config`` into the rubric class's ``__init__``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from app.services import scenario_rubrics_base

# Allowed HTTP verbs. Kept inline because no shared enum exists yet
# in the codebase; if a ``course_outcome_models.HttpMethod`` enum is
# introduced later, fold it in here.
_ALLOWED_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
)

_RUBRICS_REGISTERED = False


def _ensure_rubrics_registered() -> None:
    """Import every rubric implementation module exactly once so
    ``RUBRIC_REGISTRY`` is fully populated before a ``kind`` lookup.

    Without this, the registry would only hold rubrics whose modules
    happen to have been imported elsewhere in the process — making
    loader behaviour depend on import order.
    """
    global _RUBRICS_REGISTERED
    if _RUBRICS_REGISTERED:
        return
    # Each import triggers @register_rubric side-effects.
    import app.services.scenario_rubrics_structural  # noqa: F401
    import app.services.scenario_rubrics_set  # noqa: F401
    import app.services.scenario_rubrics_llm  # noqa: F401
    import app.services.scenario_rubrics_oracle  # noqa: F401

    _RUBRICS_REGISTERED = True


class ScenarioLoadError(Exception):
    """Raised when a scenario YAML file cannot be parsed or validated.

    The message names the offending file path; the underlying
    ``yaml.YAMLError`` or ``pydantic.ValidationError`` is attached via
    ``__cause__`` for callers that want full detail.
    """


class HttpExpectation(BaseModel):
    """What a trace step expects from the HTTP response.

    For v1, status assertion is enough: either a single status code or
    a list of acceptable codes (e.g., ``[301, 302]`` for redirects).
    """

    status: int | list[int] | None = None


class TraceStep(BaseModel):
    """One HTTP call in a scenario's setup or trace.

    ``id`` names the step so later steps and rubrics can refer to its
    captured response via ``${<id>.body.foo}`` / ``${<id>.status}``
    interpolation. The loader preserves these placeholder strings
    verbatim — resolution happens at trace-run time.

    ``capture`` is purely advisory metadata for the runner; when set,
    the runner stores the full response under that key. ``id`` already
    plays that role implicitly, so most steps can leave ``capture``
    unset.
    """

    id: str
    method: str
    path: str
    body: dict[str, Any] | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    follow_redirects: bool = True
    expect: HttpExpectation | None = None
    capture: str | None = None

    @field_validator("method", mode="before")
    @classmethod
    def _normalize_method(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.upper()
        return v

    @field_validator("method")
    @classmethod
    def _method_must_be_known(cls, v: str) -> str:
        if v not in _ALLOWED_METHODS:
            raise ValueError(
                f"method '{v}' is not one of {sorted(_ALLOWED_METHODS)}"
            )
        return v


class RubricSpec(BaseModel):
    """One rubric configuration line inside a scenario.

    YAML shape (free-form, per rubric kind)::

        - kind: schema_match
          target: resp.body
          must_have_fields: [answer, citations]

    The model packs every non-``kind`` field into ``config`` so the
    trace runner can ``RubricClass(**config)`` without knowing
    rubric-specific shapes. An explicit ``config: {...}`` block is
    also accepted for authors who prefer the nested form.
    """

    kind: str
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _pack_extra_into_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # If someone passed config= directly via Python kwargs and no
        # extra fields are present, leave it alone.
        kind = data.get("kind")
        explicit_config = data.get("config")
        extras = {
            k: v for k, v in data.items() if k not in {"kind", "config"}
        }
        if extras:
            merged: dict[str, Any] = {}
            if isinstance(explicit_config, dict):
                merged.update(explicit_config)
            merged.update(extras)
            return {"kind": kind, "config": merged}
        return data

    @field_validator("kind")
    @classmethod
    def _kind_must_be_registered(cls, v: str) -> str:
        _ensure_rubrics_registered()
        registry = scenario_rubrics_base.RUBRIC_REGISTRY
        if v not in registry:
            registered = sorted(registry.keys())
            raise ScenarioLoadError(
                f"Unknown rubric kind '{v}'; registered kinds: {registered}"
            )
        return v


_ScenarioCategory = Literal[
    "happy_path",
    "boundary",
    "malformed_input",
    "adversarial",
    "out_of_scope",
    "idempotency",
    "determinism",
    "concurrency",
    "state_persistence",
    "composition",
]


_ScenarioMode = Literal["independent", "live"]


class Scenario(BaseModel):
    """Top-level YAML object: one curated scenario.

    A scenario is a triple of (setup, trace, rubrics) plus metadata.
    ``setup`` runs once to prime the learner's service (e.g., POST a
    corpus before any RAG question). ``trace`` is the actual sequence
    being graded. ``rubrics`` then judge the captured trace.

    ``mode``:
      - ``independent`` (default): the grader compares against a
        pre-computed oracle cache.
      - ``live``: the reference implementation runs alongside the
        learner at grade time. Slower; reserved for non-deterministic
        scenarios.
    """

    id: str
    description: str
    category: _ScenarioCategory
    setup: list[TraceStep] = Field(default_factory=list)
    trace: list[TraceStep] = Field(..., min_length=1)
    rubrics: list[RubricSpec] = Field(..., min_length=1)
    mode: _ScenarioMode = "independent"
    # Which ``QualityBar.id``s this scenario contributes evidence toward.
    # Defaults to ``[]`` for backward compatibility with scenarios authored
    # before the publish-gate started enforcing coverage; the oracle
    # validator blocks publication when any spec-declared bar has zero
    # scenarios referencing it (Codex review #4 finding #2).
    quality_bar_ids: list[str] = Field(default_factory=list)


def load_scenarios_from_dir(dir_path: str | Path) -> list[Scenario]:
    """Parse every ``*.yaml`` file in ``dir_path`` into ``Scenario`` objects.

    Files are processed in lexical order so error messages and the
    returned list are deterministic. Non-YAML files (``.md``, ``.txt``,
    ``.json``, etc.) are ignored.

    Raises ``ScenarioLoadError`` on:
      - malformed YAML (wraps the underlying ``yaml.YAMLError``),
      - any Pydantic validation failure (wraps the underlying
        ``ValidationError``),
      - duplicate ``scenario.id`` across files in the same directory,
      - unknown rubric ``kind``,
      - a missing directory.

    The exception message always names the offending file path.
    """
    _ensure_rubrics_registered()

    path = Path(dir_path)
    if not path.exists():
        raise ScenarioLoadError(f"Scenario directory not found: {path}")
    if not path.is_dir():
        raise ScenarioLoadError(f"Not a directory: {path}")

    scenarios: list[Scenario] = []
    seen_ids: dict[str, Path] = {}

    yaml_files = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))

    for file_path in yaml_files:
        try:
            raw = file_path.read_text()
        except OSError as exc:
            raise ScenarioLoadError(
                f"Could not read scenario file {file_path}: {exc}"
            ) from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ScenarioLoadError(
                f"Malformed YAML in scenario file {file_path}: {exc}"
            ) from exc

        if data is None:
            raise ScenarioLoadError(
                f"Scenario file {file_path} is empty"
            )
        if not isinstance(data, dict):
            raise ScenarioLoadError(
                f"Scenario file {file_path} must contain a mapping at the "
                f"top level; got {type(data).__name__}"
            )

        try:
            scenario = Scenario.model_validate(data)
        except ScenarioLoadError as exc:
            # An unknown rubric kind surfaces as ScenarioLoadError from
            # the inner field validator; re-raise with file context so
            # callers see which file is at fault.
            scenario_id = data.get("id", "<unknown>")
            raise ScenarioLoadError(
                f"{exc} in scenario '{scenario_id}' (file: {file_path})"
            ) from exc
        except Exception as exc:  # pydantic.ValidationError + anything else
            raise ScenarioLoadError(
                f"Invalid scenario in file {file_path}: {exc}"
            ) from exc

        if scenario.id in seen_ids:
            other = seen_ids[scenario.id]
            raise ScenarioLoadError(
                f"Duplicate scenario id '{scenario.id}' in {file_path} "
                f"(already defined in {other})"
            )
        seen_ids[scenario.id] = file_path
        scenarios.append(scenario)

    return scenarios
