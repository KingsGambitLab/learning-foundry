"""LLMRouter — provider-agnostic structured-output entry point.

One small in-house adapter that dispatches every structured-output LLM
call to either Anthropic (default) or OpenAI (fallback) based on the
``COURSE_GEN_LLM_PROVIDER`` env var. The router owns:

- env-file loading per provider,
- tier → model id mapping (Sonnet for hard tasks, Haiku for simple ones),
- the hard-kill subprocess timeout wrapper (delegated to the existing
  per-provider runtime-support modules — the guardrail is provider-
  agnostic),
- response shape normalization back into a single ``ParsedResult``.

Callsites swap ``client.responses.parse(...)`` /
``client.messages.parse(...)`` for ``router.parse_structured(tier=...,
system=..., user=..., text_format=..., ...)``. Switching providers is
one env var.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from app.services.anthropic_runtime_support import (
    DEFAULT_HAIKU_MODEL_ID,
    DEFAULT_SONNET_MODEL_ID,
    load_anthropic_env_file,
    parse_structured_anthropic_response_with_hard_timeout,
    resolve_anthropic_env_file,
)
from app.services.openai_runtime_support import (
    load_openai_env_file,
    parse_structured_openai_response_with_hard_timeout,
    resolve_openai_env_file,
)


class LLMProvider(StrEnum):
    anthropic = "anthropic"
    openai = "openai"


class LLMTier(StrEnum):
    sonnet = "sonnet"
    haiku = "haiku"


class LLMRouterConfigError(RuntimeError):
    """Raised when the router cannot be configured (bad env var, unknown
    tier, missing api key for the active provider)."""


@dataclass
class ParsedResult:
    """Provider-agnostic result of a structured-output call."""
    parsed: BaseModel
    usage: Any  # SimpleNamespace with input_tokens / output_tokens / cache_* fields

    @property
    def output_parsed(self) -> BaseModel:
        """Alias for ``parsed`` — matches the OpenAI subprocess wrapper's
        return shape so callsites that read ``response.output_parsed``
        continue to work unchanged after migration."""
        return self.parsed


_ALLOWED_PROVIDERS = {p.value for p in LLMProvider}


def resolve_provider_from_env() -> LLMProvider:
    raw = os.environ.get("COURSE_GEN_LLM_PROVIDER")
    if not raw:
        return LLMProvider.anthropic
    value = raw.strip().lower()
    if value not in _ALLOWED_PROVIDERS:
        raise LLMRouterConfigError(
            f"COURSE_GEN_LLM_PROVIDER='{raw}' is not a known provider. "
            f"Valid values: {sorted(_ALLOWED_PROVIDERS)}"
        )
    return LLMProvider(value)


def _coerce_tier(tier: LLMTier | str) -> LLMTier:
    if isinstance(tier, LLMTier):
        return tier
    raw = (tier or "").strip().lower()
    if raw not in {t.value for t in LLMTier}:
        raise LLMRouterConfigError(
            f"Unknown LLM tier '{tier}'. Valid tiers: {[t.value for t in LLMTier]}"
        )
    return LLMTier(raw)


class LLMRouter:
    def __init__(
        self,
        *,
        provider: LLMProvider | str | None = None,
        anthropic_env_file: str | None = None,
        openai_env_file: str | None = None,
    ) -> None:
        if provider is None:
            self.provider = resolve_provider_from_env()
        elif isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            raw = str(provider).strip().lower()
            if raw not in _ALLOWED_PROVIDERS:
                raise LLMRouterConfigError(
                    f"Unknown provider '{provider}'. Valid: {sorted(_ALLOWED_PROVIDERS)}"
                )
            self.provider = LLMProvider(raw)

        self._anthropic_env_path = resolve_anthropic_env_file(anthropic_env_file)
        self._openai_env_path = resolve_openai_env_file(openai_env_file)
        self._anthropic_env = load_anthropic_env_file(self._anthropic_env_path)
        self._openai_env = load_openai_env_file(self._openai_env_path)

    # ----- provider-aware lookups -----

    def model_id_for(self, tier: LLMTier | str) -> str:
        t = _coerce_tier(tier)
        if self.provider == LLMProvider.anthropic:
            if t == LLMTier.sonnet:
                return self._anthropic_env.get("ANTHROPIC_MODEL_SONNET", DEFAULT_SONNET_MODEL_ID)
            return self._anthropic_env.get("ANTHROPIC_MODEL_HAIKU", DEFAULT_HAIKU_MODEL_ID)
        # OpenAI fallback: one model env var per tier; Haiku falls back to
        # the sonnet/default model id when no fast model is declared.
        sonnet_id = self._openai_env.get("OPENAI_MODEL", "gpt-5.4")
        if t == LLMTier.sonnet:
            return sonnet_id
        return self._openai_env.get("OPENAI_MODEL_FAST", sonnet_id)

    def api_key_for_active_provider(self) -> str | None:
        if self.provider == LLMProvider.anthropic:
            return self._anthropic_env.get("ANTHROPIC_API_KEY") or None
        return self._openai_env.get("OPENAI_API_KEY") or None

    def base_url_for_active_provider(self) -> str | None:
        if self.provider == LLMProvider.anthropic:
            return self._anthropic_env.get("ANTHROPIC_BASE_URL") or None
        return self._openai_env.get("OPENAI_BASE_URL") or None

    # ----- structured-output entry point -----

    def parse_structured(
        self,
        *,
        tier: LLMTier | str,
        system: str,
        user: str,
        text_format: type[BaseModel],
        request_timeout_s: float = 240.0,
        max_tokens: int = 16_000,
        extra_request_kwargs: dict[str, Any] | None = None,
    ) -> ParsedResult:
        """One structured-output call against the active provider. Returns a
        validated Pydantic instance + a provider-agnostic usage namespace.

        ``request_timeout_s`` is the wall-clock deadline enforced by the
        subprocess wrapper inside each provider module — not the SDK's own
        timeout.
        """
        api_key = self.api_key_for_active_provider()
        if not api_key:
            raise LLMRouterConfigError(
                f"No API key available for provider '{self.provider.value}'. "
                f"Set the corresponding env file."
            )
        model = self.model_id_for(tier)
        base_url = self.base_url_for_active_provider()

        if self.provider == LLMProvider.anthropic:
            response = parse_structured_anthropic_response_with_hard_timeout(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system=system,
                user=user,
                text_format=text_format,
                request_timeout_s=request_timeout_s,
                max_tokens=max_tokens,
                extra_request_kwargs=extra_request_kwargs,
            )
            return ParsedResult(parsed=response.output_parsed, usage=response.usage)

        # OpenAI fallback. The OpenAI Responses API takes an ``input`` list of
        # role/content dicts; build one from the system + user strings so the
        # legacy path keeps working unchanged.
        openai_input = []
        if system:
            openai_input.append({"role": "system", "content": system})
        openai_input.append({"role": "user", "content": user})
        response = parse_structured_openai_response_with_hard_timeout(
            api_key=api_key,
            base_url=base_url,
            model=model,
            input=openai_input,
            text_format=text_format,
            request_timeout_s=request_timeout_s,
            extra_request_kwargs=extra_request_kwargs,
        )
        return ParsedResult(parsed=response.output_parsed, usage=response.usage)

    # ----- status (compat with /v1/task-agent-authoring/status) -----

    def status(self) -> dict[str, Any]:
        api_key_present = bool(self.api_key_for_active_provider())
        sdk_installed = self._sdk_installed_for(self.provider)
        available = api_key_present and sdk_installed
        sonnet_id = self.model_id_for(LLMTier.sonnet)
        haiku_id = self.model_id_for(LLMTier.haiku)
        env_file = (
            self._anthropic_env_path
            if self.provider == LLMProvider.anthropic
            else self._openai_env_path
        )
        source = (
            f"{self.provider.value}_live" if available else f"{self.provider.value}_unavailable"
        )
        message = (
            f"{self.provider.value.title()} authoring is ready to customize the learner-facing bundle."
            if available
            else f"{self.provider.value.title()} authoring is not available (api_key_present={api_key_present}, sdk_installed={sdk_installed})."
        )
        return {
            "provider": self.provider.value,
            "available": available,
            "source": source,
            "message": message,
            "sdk_installed": sdk_installed,
            "api_key_present": api_key_present,
            # Preserved for backward compatibility — dashboards read `model_id`.
            "model_id": sonnet_id,
            "model_id_sonnet": sonnet_id,
            "model_id_haiku": haiku_id,
            "env_file": env_file,
            "fallback_provider_available": self._sdk_installed_for(
                LLMProvider.openai if self.provider == LLMProvider.anthropic else LLMProvider.anthropic
            ),
        }

    @staticmethod
    def _sdk_installed_for(provider: LLMProvider) -> bool:
        try:
            if provider == LLMProvider.anthropic:
                import anthropic  # noqa: F401
            else:
                import openai  # noqa: F401
            return True
        except Exception:
            return False


# ---------------- module-level singleton + adapter helpers ----------------


_default_router: LLMRouter | None = None


def get_default_router() -> LLMRouter:
    """Return a process-wide singleton router. Re-instantiating the
    router only re-reads the env files — cheap, but we cache to avoid
    surfacing two distinct status snapshots from concurrent callers."""
    global _default_router
    if _default_router is None:
        _default_router = LLMRouter()
    return _default_router


def reset_default_router() -> None:
    """For tests: clear the cached default router so subsequent calls
    re-resolve provider + env files."""
    global _default_router
    _default_router = None


def messages_to_system_user(input_list: Any) -> tuple[str, str]:
    """Flatten an OpenAI-shaped ``input`` list ``[{role, content}, ...]``
    into ``(system_text, user_text)``. ``content`` may be a string or a
    list of content blocks (multimodal-style). System messages are joined
    with ``\\n\\n``; user messages likewise. Assistant turns are ignored —
    structured-output callers do not include them in this codebase."""
    systems: list[str] = []
    users: list[str] = []
    for entry in input_list or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text") or block.get("content") or "")
                else:
                    parts.append(str(block))
            text = "\n".join(p for p in parts if p)
        else:
            text = "" if content is None else str(content)
        if role == "system":
            systems.append(text)
        elif role == "user":
            users.append(text)
    return ("\n\n".join(s for s in systems if s), "\n\n".join(u for u in users if u))
