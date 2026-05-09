from __future__ import annotations

import re
from collections.abc import Iterable


_ARCHETYPE_TOKENS = {
    "service",
    "services",
    "system",
    "systems",
    "api",
    "apis",
    "backend",
    "backends",
    "app",
    "apps",
    "application",
    "applications",
    "bot",
    "bots",
    "agent",
    "agents",
}
_TITLE_VERB_TOKENS = {
    "build",
    "create",
    "design",
    "implement",
    "develop",
    "ship",
    "deliver",
}
_EDGE_NOISE_TOKENS = {
    "a",
    "an",
    "the",
    "for",
    "of",
    "production",
    "ready",
    "productionready",
    "safe",
    "concurrency",
    "concurrent",
    "robust",
    "reliable",
    "scalable",
    "durable",
    "efficient",
    "driven",
    "based",
    "high",
    "low",
    "latency",
    "throughput",
    "stateful",
    "bounded",
}
_TECH_TOKENS = {
    "python",
    "fastapi",
    "flask",
    "django",
    "typescript",
    "javascript",
    "node",
    "nodejs",
    "express",
    "nestjs",
    "hono",
    "go",
    "golang",
    "gin",
    "fiber",
    "rust",
    "axum",
    "actix",
    "postgres",
    "postgresql",
    "mysql",
    "mariadb",
    "mongodb",
    "mongo",
    "redis",
    "pnpm",
    "npm",
    "yarn",
    "bun",
    "uv",
    "poetry",
    "docker",
    "kubernetes",
}
_GENERIC_ENTITY_PHRASES = {
    "durable records",
    "mutable workflow state",
    "service request",
    "service response",
}
_GENERIC_STARTER_MARKERS = (
    "primary request",
    "edge or failure path",
    "serve the current state safely under load",
    "apply state transitions without violating invariants",
    "behavior behind it",
    "core entities",
    "service surface",
)
_GENERIC_DELIVERABLE_TITLE_MARKERS = (
    "service contract",
    "service contract and durable model",
    "read and write path correctness",
    "runtime integration and failure recovery",
    "operational hardening",
    "production hardening",
)


def normalized_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def extract_project_entities(title: str, problem_statement: str) -> list[str]:
    candidates: list[str] = []
    for text in (title, problem_statement):
        candidates.extend(_extract_surface_candidates(text))
    entities: list[str] = []
    for candidate in candidates:
        normalized = _normalize_entity_phrase(candidate)
        if not normalized or normalized in entities:
            continue
        entities.append(normalized)
    return entities


def meaningful_domain_entities(values: Iterable[str]) -> list[str]:
    entities: list[str] = []
    for value in values:
        normalized = _normalize_entity_phrase(value)
        if (
            not normalized
            or normalized in _GENERIC_ENTITY_PHRASES
            or normalized in entities
        ):
            continue
        entities.append(normalized)
    return entities


def collection_slug_for_entity(entity: str) -> str:
    tokens = normalized_tokens(entity)
    if not tokens:
        return "resources"
    if len(tokens) > 2:
        tokens = tokens[-2:]
    tokens[-1] = _pluralize_token(tokens[-1])
    return "-".join(tokens)


def pluralize_phrase(value: str) -> str:
    tokens = normalized_tokens(value)
    if not tokens:
        return value
    tokens[-1] = _pluralize_token(tokens[-1])
    return " ".join(tokens)


def endpoint_uses_title_slug(path: str, *, title: str) -> bool:
    segments = [segment for segment in path.strip("/").split("/") if segment and "{" not in segment]
    if not segments:
        return False
    first_segment_tokens = normalized_tokens(segments[0].replace("-", " "))
    title_tokens = [token for token in normalized_tokens(title) if token not in _ARCHETYPE_TOKENS]
    if len(first_segment_tokens) < 3 or len(title_tokens) < 3:
        return False
    overlap = sum(1 for token in first_segment_tokens if token in set(title_tokens))
    return overlap / max(len(first_segment_tokens), 1) >= 0.75


def endpoint_uses_archetype_words(path: str) -> bool:
    for segment in path.strip("/").split("/"):
        if "{" in segment:
            continue
        tokens = normalized_tokens(segment.replace("-", " "))
        if any(token in _TITLE_VERB_TOKENS or token in _ARCHETYPE_TOKENS for token in tokens):
            return True
    return False


def content_lacks_domain_grounding(content: str, *, entities: Iterable[str]) -> bool:
    grounded_entities = meaningful_domain_entities(entities)
    if not grounded_entities:
        return False
    lowered = content.lower()
    if any(entity in lowered for entity in grounded_entities):
        return False
    return any(marker in lowered for marker in _GENERIC_STARTER_MARKERS)


def starter_surface_markers() -> tuple[str, ...]:
    return _GENERIC_STARTER_MARKERS


def deliverable_title_lacks_domain_grounding(title: str, *, entities: Iterable[str]) -> bool:
    grounded_entities = meaningful_domain_entities(entities)
    lowered = title.lower()
    if grounded_entities and any(entity in lowered for entity in grounded_entities):
        return False
    return any(marker in lowered for marker in _GENERIC_DELIVERABLE_TITLE_MARKERS)


def _extract_surface_candidates(text: str) -> list[str]:
    lowered = text.lower()
    patterns = (
        r"(?:build|create|design|implement|develop|ship|deliver)\s+(?:a|an|the)?\s*([a-z0-9][a-z0-9 -]{2,80}?)\s+(?:service|system|api|backend|app|application|bot|agent)\b",
        r"([a-z0-9][a-z0-9 -]{2,80}?)\s+(?:service|system|api|backend|app|application|bot|agent)\b",
        r"([a-z0-9][a-z0-9 -]{2,80}?)\s+control\s+plane\b",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(match.group(1) for match in re.finditer(pattern, lowered))
    return matches


def _normalize_entity_phrase(value: str) -> str:
    tokens = normalized_tokens(value)
    while tokens and (tokens[0] in _TITLE_VERB_TOKENS or tokens[0] in _EDGE_NOISE_TOKENS):
        tokens.pop(0)
    while len(tokens) >= 2 and tuple(tokens[-2:]) == ("control", "plane"):
        tokens = tokens[:-2]
    while tokens and (tokens[-1] in _ARCHETYPE_TOKENS or tokens[-1] in _EDGE_NOISE_TOKENS):
        tokens.pop()
    tokens = [
        token
        for token in tokens
        if token not in _TECH_TOKENS and token not in _TITLE_VERB_TOKENS
    ]
    while tokens and tokens[0] in _EDGE_NOISE_TOKENS:
        tokens.pop(0)
    while tokens and tokens[-1] in _EDGE_NOISE_TOKENS:
        tokens.pop()
    if len(tokens) > 2:
        tokens = tokens[-2:]
    return " ".join(tokens).strip()


def _pluralize_token(token: str) -> str:
    if token.endswith("ies") or token.endswith("ses"):
        return token
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    if token.endswith(("s", "x", "z", "ch", "sh")):
        return token + "es"
    return token + "s"
