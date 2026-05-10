from __future__ import annotations

from typing import Any, Iterable

from app.domain.task_agent import DeliverableSpec, EndpointSpec, LearnerStarterSurfaceSpec, TaskAgentServiceSpec

INTERNAL_RUN_PATH = "/_coursegen/run"
INTERNAL_RUN_STATE_PATH = "/_coursegen/runs/{id}"
INTERNAL_TRACE_PATH = "/_coursegen/trace/{id}"
INTERNAL_APPROVE_PATH = "/_coursegen/approve/{id}"
INTERNAL_EVAL_PATH = "/_coursegen/eval"


def endpoint_specs_from_manifest(manifest: dict[str, Any]) -> list[EndpointSpec]:
    endpoints: list[EndpointSpec] = []
    for item in manifest.get("public_endpoints") or []:
        if not isinstance(item, dict):
            continue
        method = str(item.get("method") or "").strip().upper()
        path = str(item.get("path") or "").strip()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"} or not path.startswith("/"):
            continue
        endpoints.append(
            EndpointSpec(
                method=method,
                path=path,
                required=bool(item.get("required", True)),
            )
        )
    return endpoints


def required_public_endpoints(endpoints: Iterable[EndpointSpec]) -> list[EndpointSpec]:
    return [
        endpoint
        for endpoint in endpoints
        if endpoint.required and not is_internal_harness_path(endpoint.path)
    ]


def required_public_endpoints_for_spec(spec: TaskAgentServiceSpec) -> list[EndpointSpec]:
    return required_public_endpoints(spec.public_endpoints)


def required_public_endpoints_for_manifest(manifest: dict[str, Any]) -> list[EndpointSpec]:
    return required_public_endpoints(endpoint_specs_from_manifest(manifest))


def primary_submit_endpoint(endpoints: Iterable[EndpointSpec]) -> EndpointSpec | None:
    candidates = [
        endpoint
        for endpoint in required_public_endpoints(endpoints)
        if endpoint.method in {"POST", "PUT", "PATCH"}
        and not is_health_path(endpoint.path)
        and not is_eval_path(endpoint.path)
        and not is_approval_path(endpoint.path)
    ]
    for endpoint in candidates:
        if "{" not in endpoint.path:
            return endpoint
    return candidates[0] if candidates else None


def primary_submit_endpoint_for_spec(spec: TaskAgentServiceSpec) -> EndpointSpec | None:
    return primary_submit_endpoint(spec.public_endpoints)


def primary_submit_endpoint_for_manifest(manifest: dict[str, Any]) -> EndpointSpec | None:
    return primary_submit_endpoint(endpoint_specs_from_manifest(manifest))


def learner_editable_paths_from_surface(
    starter_surface: LearnerStarterSurfaceSpec | dict[str, Any] | None,
) -> list[str]:
    if isinstance(starter_surface, LearnerStarterSurfaceSpec):
        raw_paths = starter_surface.primary_editable_paths
    elif isinstance(starter_surface, dict):
        raw_paths = starter_surface.get("primary_editable_paths") or []
    else:
        raw_paths = []
    seen: set[str] = set()
    resolved: list[str] = []
    for path in raw_paths:
        normalized = str(path or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def learner_editable_paths_for_deliverable(
    spec: TaskAgentServiceSpec,
    deliverable: DeliverableSpec,
) -> list[str]:
    starter_paths = learner_editable_paths_from_surface(deliverable.learner_starter_surface)
    if starter_paths:
        return starter_paths
    return learner_editable_paths_for_spec(spec)


def learner_editable_paths_for_spec(spec: TaskAgentServiceSpec) -> list[str]:
    seen: set[str] = set()
    deliverable_paths: list[str] = []
    for deliverable in spec.deliverables:
        for path in learner_editable_paths_from_surface(deliverable.learner_starter_surface):
            if path in seen:
                continue
            seen.add(path)
            deliverable_paths.append(path)
    if deliverable_paths:
        return deliverable_paths
    runtime_paths: list[str] = []
    for path in spec.runtime_dependencies.editable_files:
        normalized = str(path or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        runtime_paths.append(normalized)
    return runtime_paths


def learner_editable_paths_for_manifest(manifest: dict[str, Any]) -> list[str]:
    starter_paths = learner_editable_paths_from_surface(manifest.get("learner_starter_surface"))
    if starter_paths:
        return starter_paths
    runtime_dependencies = manifest.get("runtime_dependencies") or {}
    seen: set[str] = set()
    resolved: list[str] = []
    for path in runtime_dependencies.get("editable_files") or []:
        normalized = str(path or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def is_internal_harness_path(path: str) -> bool:
    return path.startswith("/_coursegen/")


def is_health_path(path: str) -> bool:
    return path.rstrip("/") == "/health"


def is_eval_path(path: str) -> bool:
    normalized = path.rstrip("/").lower()
    return normalized.endswith("/eval")


def is_trace_path(path: str) -> bool:
    normalized = path.lower()
    return "/trace" in normalized or "/audit" in normalized


def is_approval_path(path: str) -> bool:
    normalized = path.lower()
    return any(marker in normalized for marker in ("/approve", "/approval", "/decision"))


def extract_public_output(payload: dict[str, Any]) -> dict[str, Any]:
    output = payload.get("output")
    if isinstance(output, dict):
        return output
    return payload


def is_placeholder_public_surface(endpoints: Iterable[EndpointSpec]) -> bool:
    public_paths = {
        endpoint.path
        for endpoint in required_public_endpoints(endpoints)
        if not is_health_path(endpoint.path)
    }
    return not public_paths
