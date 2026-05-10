from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.domain.course import (
    CreatorCourseSetupChoices,
    CreatorStackCatalog,
    CreatorStackCatalogOption,
    RecommendCreatorStackContractResponse,
)


def _fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "course-gen-codex/0.1"})
    with urlopen(request, timeout=15) as response:  # noqa: S310 - public metadata endpoints only
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "course-gen-codex/0.1"})
    with urlopen(request, timeout=15) as response:  # noqa: S310 - public metadata endpoints only
        return response.read().decode("utf-8")


@dataclass(frozen=True)
class _FrameworkMetadata:
    label: str
    registry_kind: str
    identifier: str
    source_url: str


@dataclass(frozen=True)
class _LanguageMetadata:
    label: str
    default_framework: str | None
    default_package_manager: str
    version_source_kind: str
    version_identifier: str
    version_source_url: str
    version_segments: int
    package_managers: tuple[tuple[str, str], ...]
    frameworks: dict[str, _FrameworkMetadata]


@dataclass(frozen=True)
class _ServiceMetadata:
    label: str
    version_source_kind: str | None
    version_identifier: str | None
    version_source_url: str | None
    version_segments: int = 1


class StackCatalogService:
    def __init__(
        self,
        *,
        json_fetcher=_fetch_json,
        text_fetcher=_fetch_text,
        ttl_seconds: int = 3600,
    ) -> None:
        self._fetch_json = json_fetcher
        self._fetch_text = text_fetcher
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, list[str]]] = {}

    LANGUAGE_CATALOG: dict[str, _LanguageMetadata] = {
        "python": _LanguageMetadata(
            label="Python",
            default_framework="fastapi",
            default_package_manager="uv",
            version_source_kind="docker",
            version_identifier="python",
            version_source_url="https://hub.docker.com/_/python",
            version_segments=2,
            package_managers=(("uv", "uv"), ("pip", "pip")),
            frameworks={
                "fastapi": _FrameworkMetadata(
                    label="FastAPI",
                    registry_kind="pypi",
                    identifier="fastapi",
                    source_url="https://pypi.org/project/fastapi/",
                ),
                "flask": _FrameworkMetadata(
                    label="Flask",
                    registry_kind="pypi",
                    identifier="flask",
                    source_url="https://pypi.org/project/flask/",
                ),
                "django": _FrameworkMetadata(
                    label="Django",
                    registry_kind="pypi",
                    identifier="Django",
                    source_url="https://pypi.org/project/Django/",
                ),
            },
        ),
        "typescript": _LanguageMetadata(
            label="TypeScript",
            default_framework="express",
            default_package_manager="pnpm",
            version_source_kind="node",
            version_identifier="node",
            version_source_url="https://nodejs.org/dist/index.json",
            version_segments=1,
            package_managers=(("pnpm", "pnpm"), ("npm", "npm"), ("yarn", "yarn"), ("bun", "bun")),
            frameworks={
                "express": _FrameworkMetadata(
                    label="Express",
                    registry_kind="npm",
                    identifier="express",
                    source_url="https://www.npmjs.com/package/express",
                ),
                "nestjs": _FrameworkMetadata(
                    label="NestJS",
                    registry_kind="npm",
                    identifier="@nestjs/core",
                    source_url="https://www.npmjs.com/package/@nestjs/core",
                ),
                "hono": _FrameworkMetadata(
                    label="Hono",
                    registry_kind="npm",
                    identifier="hono",
                    source_url="https://www.npmjs.com/package/hono",
                ),
            },
        ),
        "javascript": _LanguageMetadata(
            label="JavaScript",
            default_framework="express",
            default_package_manager="pnpm",
            version_source_kind="node",
            version_identifier="node",
            version_source_url="https://nodejs.org/dist/index.json",
            version_segments=1,
            package_managers=(("pnpm", "pnpm"), ("npm", "npm"), ("yarn", "yarn"), ("bun", "bun")),
            frameworks={
                "express": _FrameworkMetadata(
                    label="Express",
                    registry_kind="npm",
                    identifier="express",
                    source_url="https://www.npmjs.com/package/express",
                ),
                "hono": _FrameworkMetadata(
                    label="Hono",
                    registry_kind="npm",
                    identifier="hono",
                    source_url="https://www.npmjs.com/package/hono",
                ),
            },
        ),
        "go": _LanguageMetadata(
            label="Go",
            default_framework="gin",
            default_package_manager="go",
            version_source_kind="go",
            version_identifier="go",
            version_source_url="https://go.dev/dl/?mode=json",
            version_segments=2,
            package_managers=(("go", "go"),),
            frameworks={
                "gin": _FrameworkMetadata(
                    label="Gin",
                    registry_kind="gomodule",
                    identifier="github.com/gin-gonic/gin",
                    source_url="https://pkg.go.dev/github.com/gin-gonic/gin",
                ),
                "fiber": _FrameworkMetadata(
                    label="Fiber",
                    registry_kind="gomodule",
                    identifier="github.com/gofiber/fiber/v2",
                    source_url="https://pkg.go.dev/github.com/gofiber/fiber/v2",
                ),
            },
        ),
        "rust": _LanguageMetadata(
            label="Rust",
            default_framework="actix-web",
            default_package_manager="cargo",
            version_source_kind="docker",
            version_identifier="rust",
            version_source_url="https://hub.docker.com/_/rust",
            version_segments=2,
            package_managers=(("cargo", "cargo"),),
            frameworks={
                "actix-web": _FrameworkMetadata(
                    label="Actix Web",
                    registry_kind="crates",
                    identifier="actix-web",
                    source_url="https://crates.io/crates/actix-web",
                ),
                "axum": _FrameworkMetadata(
                    label="Axum",
                    registry_kind="crates",
                    identifier="axum",
                    source_url="https://crates.io/crates/axum",
                ),
            },
        ),
    }

    DATABASE_CATALOG: dict[str, _ServiceMetadata] = {
        "postgres": _ServiceMetadata(
            label="PostgreSQL",
            version_source_kind="docker",
            version_identifier="postgres",
            version_source_url="https://hub.docker.com/_/postgres",
            version_segments=1,
        ),
        "mysql": _ServiceMetadata(
            label="MySQL",
            version_source_kind="docker",
            version_identifier="mysql",
            version_source_url="https://hub.docker.com/_/mysql",
            version_segments=1,
        ),
        "mongodb": _ServiceMetadata(
            label="MongoDB",
            version_source_kind="docker",
            version_identifier="mongo",
            version_source_url="https://hub.docker.com/_/mongo",
            version_segments=1,
        ),
        "sqlite": _ServiceMetadata(
            label="SQLite",
            version_source_kind=None,
            version_identifier=None,
            version_source_url="https://www.sqlite.org/index.html",
        ),
    }

    CACHE_CATALOG: dict[str, _ServiceMetadata] = {
        "redis": _ServiceMetadata(
            label="Redis",
            version_source_kind="docker",
            version_identifier="redis",
            version_source_url="https://hub.docker.com/_/redis",
            version_segments=1,
        ),
        "memcached": _ServiceMetadata(
            label="Memcached",
            version_source_kind="docker",
            version_identifier="memcached",
            version_source_url="https://hub.docker.com/_/memcached",
            version_segments=1,
        ),
    }

    def catalog(self) -> CreatorStackCatalog:
        return CreatorStackCatalog(
            languages=[
                CreatorStackCatalogOption(value=key, label=meta.label, source_url=meta.version_source_url)
                for key, meta in self.LANGUAGE_CATALOG.items()
            ],
            frameworks_by_language={
                language: [
                    CreatorStackCatalogOption(value=framework, label=framework_meta.label, source_url=framework_meta.source_url)
                    for framework, framework_meta in meta.frameworks.items()
                ]
                for language, meta in self.LANGUAGE_CATALOG.items()
            },
            package_managers_by_language={
                language: [
                    CreatorStackCatalogOption(value=value, label=label)
                    for value, label in meta.package_managers
                ]
                for language, meta in self.LANGUAGE_CATALOG.items()
            },
            databases=[
                CreatorStackCatalogOption(value=key, label=meta.label, source_url=meta.version_source_url)
                for key, meta in self.DATABASE_CATALOG.items()
            ],
            caches=[
                CreatorStackCatalogOption(value=key, label=meta.label, source_url=meta.version_source_url)
                for key, meta in self.CACHE_CATALOG.items()
            ],
        )

    def describe_choices(self, choices: CreatorCourseSetupChoices) -> RecommendCreatorStackContractResponse:
        normalized = choices.model_copy(deep=True)
        notes: list[str] = []

        language = self._normalize_key(normalized.implementation_language)
        language_meta = self.LANGUAGE_CATALOG.get(language or "")
        if language_meta is not None:
            normalized.implementation_language = language
            normalized.application_framework = self._normalize_key(normalized.application_framework)
            if not normalized.application_framework and language_meta.default_framework:
                notes.append(f"Suggested `{language_meta.default_framework}` as the default framework for `{language}`.")
            if not normalized.package_manager:
                notes.append(f"Suggested `{language_meta.default_package_manager}` as the build tool for `{language}`.")
        normalized.package_manager = self._normalize_key(normalized.package_manager)

        framework_versions = self._framework_version_options(language, normalized.application_framework)
        if framework_versions and not normalized.framework_version:
            notes.append(
                f"Suggested `{framework_versions[0].value}` for `{normalized.application_framework}` from the public package registry."
            )
        framework_versions = self._mark_recommended(
            framework_versions,
            normalized.framework_version or (framework_versions[0].value if framework_versions else None),
        )

        language_versions = self._language_version_options(language)
        if language_versions and not normalized.language_version:
            notes.append(f"Suggested `{language_versions[0].value}` for `{language}` from the public runtime source.")
        language_versions = self._mark_recommended(
            language_versions,
            normalized.language_version or (language_versions[0].value if language_versions else None),
        )

        database = self._normalize_key(normalized.primary_database)
        if database is not None:
            normalized.primary_database = database
        database_versions = self._service_version_options(database, self.DATABASE_CATALOG)
        if database_versions and not normalized.primary_database_version:
            notes.append(f"Suggested `{database_versions[0].value}` for `{database}` from the public container registry.")
        database_versions = self._mark_recommended(
            database_versions,
            normalized.primary_database_version or (database_versions[0].value if database_versions else None),
        )

        cache = self._normalize_key(normalized.cache_backend)
        if cache is not None:
            normalized.cache_backend = cache
        cache_versions = self._service_version_options(cache, self.CACHE_CATALOG)
        if cache_versions and not normalized.cache_backend_version:
            notes.append(f"Suggested `{cache_versions[0].value}` for `{cache}` from the public container registry.")
        cache_versions = self._mark_recommended(
            cache_versions,
            normalized.cache_backend_version or (cache_versions[0].value if cache_versions else None),
        )

        catalog = self.catalog()
        if language_meta is not None:
            catalog.frameworks_by_language[language] = self._mark_recommended(
                catalog.frameworks_by_language.get(language, []),
                normalized.application_framework or language_meta.default_framework,
            )
            catalog.package_managers_by_language[language] = self._mark_recommended(
                catalog.package_managers_by_language.get(language, []),
                normalized.package_manager or language_meta.default_package_manager,
            )
        catalog.databases = self._mark_recommended(catalog.databases, normalized.primary_database)
        catalog.caches = self._mark_recommended(catalog.caches, normalized.cache_backend)

        return RecommendCreatorStackContractResponse(
            creator_choices=normalized,
            catalog=catalog,
            language_versions=language_versions,
            framework_versions=framework_versions,
            database_versions=database_versions,
            cache_versions=cache_versions,
            notes=notes,
        )

    def _language_version_options(self, language: str | None) -> list[CreatorStackCatalogOption]:
        metadata = self.LANGUAGE_CATALOG.get(language or "")
        if metadata is None:
            return []
        versions = self._versions_for_source(
            kind=metadata.version_source_kind,
            identifier=metadata.version_identifier,
            segments=metadata.version_segments,
        )
        return [
            CreatorStackCatalogOption(value=version, label=version, source_url=metadata.version_source_url)
            for version in versions
        ]

    def _framework_version_options(
        self,
        language: str | None,
        framework: str | None,
    ) -> list[CreatorStackCatalogOption]:
        metadata = self.LANGUAGE_CATALOG.get(language or "")
        if metadata is None:
            return []
        framework_meta = metadata.frameworks.get((framework or "").strip().lower())
        if framework_meta is None:
            return []
        versions = self._versions_for_registry(framework_meta.registry_kind, framework_meta.identifier)
        return [
            CreatorStackCatalogOption(value=version, label=version, source_url=framework_meta.source_url)
            for version in versions
        ]

    def _service_version_options(
        self,
        service_name: str | None,
        catalog: dict[str, _ServiceMetadata],
    ) -> list[CreatorStackCatalogOption]:
        metadata = catalog.get(service_name or "")
        if metadata is None or metadata.version_source_kind is None or metadata.version_identifier is None:
            return []
        versions = self._versions_for_source(
            kind=metadata.version_source_kind,
            identifier=metadata.version_identifier,
            segments=metadata.version_segments,
        )
        return [
            CreatorStackCatalogOption(value=version, label=version, source_url=metadata.version_source_url)
            for version in versions
        ]

    def _versions_for_source(self, *, kind: str, identifier: str, segments: int) -> list[str]:
        cache_key = f"{kind}:{identifier}:{segments}"
        cached = self._cache.get(cache_key)
        if cached is not None and (time.time() - cached[0]) < self._ttl_seconds:
            return list(cached[1])
        try:
            if kind == "docker":
                versions = self._docker_versions(identifier, segments=segments)
            elif kind == "node":
                versions = self._node_versions()
            elif kind == "go":
                versions = self._go_language_versions()
            else:
                versions = []
        except Exception:
            versions = []
        self._cache[cache_key] = (time.time(), list(versions))
        return versions

    def _versions_for_registry(self, kind: str, identifier: str) -> list[str]:
        cache_key = f"{kind}:{identifier}"
        cached = self._cache.get(cache_key)
        if cached is not None and (time.time() - cached[0]) < self._ttl_seconds:
            return list(cached[1])
        try:
            if kind == "pypi":
                versions = self._pypi_versions(identifier)
            elif kind == "npm":
                versions = self._npm_versions(identifier)
            elif kind == "gomodule":
                versions = self._go_module_versions(identifier)
            elif kind == "crates":
                versions = self._crate_versions(identifier)
            else:
                versions = []
        except Exception:
            versions = []
        self._cache[cache_key] = (time.time(), list(versions))
        return versions

    def _docker_versions(self, image: str, *, segments: int) -> list[str]:
        payload = self._fetch_json(f"https://registry.hub.docker.com/v2/repositories/library/{image}/tags?page_size=100")
        versions = {
            normalized
            for result in payload.get("results", [])
            for normalized in [self._normalize_tag_version(str(result.get("name") or ""), segments=segments)]
            if normalized is not None
        }
        return self._sort_versions(versions)[:8]

    def _node_versions(self) -> list[str]:
        payload = self._fetch_json("https://nodejs.org/dist/index.json")
        versions = {
            str(entry.get("version") or "").removeprefix("v")
            for entry in payload
            if entry.get("lts")
        }
        normalized = {self._normalize_numeric_segments(version, segments=1) for version in versions}
        return self._sort_versions({value for value in normalized if value})[:8]

    def _go_language_versions(self) -> list[str]:
        payload = self._fetch_json("https://go.dev/dl/?mode=json")
        versions = {
            self._normalize_numeric_segments(str(entry.get("version") or "").removeprefix("go"), segments=2)
            for entry in payload
        }
        return self._sort_versions({value for value in versions if value})[:8]

    def _pypi_versions(self, package: str) -> list[str]:
        payload = self._fetch_json(f"https://pypi.org/pypi/{quote(package)}/json")
        releases = payload.get("releases", {})
        versions = {
            str(version)
            for version, files in releases.items()
            if files and self._is_stable_version(str(version))
        }
        latest = str((payload.get("info") or {}).get("version") or "")
        ordered = self._sort_versions(versions)
        if latest and latest in ordered:
            ordered.remove(latest)
            ordered.insert(0, latest)
        return ordered[:10]

    def _npm_versions(self, package: str) -> list[str]:
        payload = self._fetch_json(f"https://registry.npmjs.org/{quote(package, safe='@')}")
        versions = {str(version) for version in (payload.get("versions") or {}).keys() if self._is_stable_version(str(version))}
        latest = str(((payload.get("dist-tags") or {}).get("latest")) or "")
        ordered = self._sort_versions(versions)
        if latest and latest in ordered:
            ordered.remove(latest)
            ordered.insert(0, latest)
        return ordered[:10]

    def _go_module_versions(self, module: str) -> list[str]:
        raw = self._fetch_text(f"https://proxy.golang.org/{quote(module, safe='')}/@v/list")
        versions = {line.strip().removeprefix("v") for line in raw.splitlines() if self._is_stable_version(line.strip())}
        return self._sort_versions(versions)[:10]

    def _crate_versions(self, crate: str) -> list[str]:
        payload = self._fetch_json(f"https://crates.io/api/v1/crates/{quote(crate)}")
        versions = {
            str(version.get("num") or "")
            for version in payload.get("versions", [])
            if not version.get("yanked") and self._is_stable_version(str(version.get("num") or ""))
        }
        return self._sort_versions(versions)[:10]

    def _sort_versions(self, versions: set[str]) -> list[str]:
        return sorted(
            {version for version in versions if version},
            key=self._version_sort_key,
            reverse=True,
        )

    def _normalize_tag_version(self, tag_name: str, *, segments: int) -> str | None:
        match = re.match(r"^v?(\d+(?:\.\d+){0,2})", tag_name)
        if match is None:
            return None
        return self._normalize_numeric_segments(match.group(1), segments=segments)

    def _normalize_numeric_segments(self, version: str, *, segments: int) -> str | None:
        numeric = [part for part in re.findall(r"\d+", version)]
        if not numeric:
            return None
        return ".".join(numeric[:segments])

    def _is_stable_version(self, version: str) -> bool:
        lowered = version.lower()
        return not any(token in lowered for token in ("alpha", "beta", "rc", "preview", "dev", "nightly", "next", "-"))

    def _version_sort_key(self, version: str) -> tuple[int, ...]:
        parts = tuple(int(part) for part in re.findall(r"\d+", version))
        return parts

    def _mark_recommended(
        self,
        options: list[CreatorStackCatalogOption],
        recommended_value: str | None,
    ) -> list[CreatorStackCatalogOption]:
        if not recommended_value:
            return options
        next_options = [option.model_copy(update={"recommended": option.value == recommended_value}) for option in options]
        if any(option.value == recommended_value for option in next_options):
            return next_options
        return [
            CreatorStackCatalogOption(value=recommended_value, label=recommended_value, recommended=True),
            *next_options,
        ]

    def _normalize_key(self, value: str | None) -> str | None:
        return (value or "").strip().lower() or None
