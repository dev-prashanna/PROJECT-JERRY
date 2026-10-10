"""LLM provider registry — the single place to add a new chat backend.

AURA picks the chat model from ``LLM_PROVIDER`` (see
``config/provider_presets.py``).  Built-ins:

* ``groq``          – native Groq SDK (default, free tier)
* ``openai``        – OpenAI
* ``anthropic``     – Claude via the Messages API
* ``moonshot``/``kimi`` – Moonshot AI (Kimi)
* ``deepseek``      – DeepSeek
* ``mimo``          – Xiaomi MiMo
* ``openai_compat`` – ANY OpenAI-compatible server (vLLM, Ollama, LM Studio,
  Together, Mistral, …) — set ``LLM_BASE_URL`` + ``LLM_API_KEY`` + ``LLM_MODEL``.

Anything whose name is not built in is treated as ``openai_compat``.

Adding your own provider
------------------------
Either (a) point ``LLM_PROVIDER`` at an OpenAI-compatible URL, or (b)
register a factory:

    from src.intelligence.llm.registry import register_llm
    from src.intelligence.llm.base import BaseLLMClient

    class MyClient(BaseLLMClient):
        provider = "myprov"
        ...

    register_llm("myprov", lambda settings: MyClient(settings))

``create_llm_client()`` will then return your client when
``LLM_PROVIDER=myprov``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from config.provider_presets import (
    LLM_ALIASES,
    LLM_PRESETS,
    known_llm_providers,
    resolve_llm_preset,
)
from config.settings import LLMSettings, get_settings
from src.intelligence.llm.base import BaseLLMClient
from src.utils.logger import get_logger

log = get_logger("llm.registry")

#: name → ``factory(settings: LLMSettings) -> BaseLLMClient``
_LLM_FACTORIES: dict[str, Callable[[LLMSettings], BaseLLMClient]] = {}


def register_llm(name: str, factory: Callable[[LLMSettings], BaseLLMClient]) -> None:
    """Register (or replace) a provider factory under ``name``."""
    _LLM_FACTORIES[name.strip().lower()] = factory


def _factory_for(protocol: str) -> Callable[[LLMSettings], BaseLLMClient]:
    if protocol == "groq":
        from src.intelligence.groq_client import GroqClient

        return GroqClient
    if protocol == "anthropic":
        from src.intelligence.llm.anthropic import AnthropicClient

        return AnthropicClient
    from src.intelligence.llm.openai_compat import OpenAICompatClient

    return OpenAICompatClient


def _builtin_factory(settings: LLMSettings) -> BaseLLMClient:
    preset = resolve_llm_preset(settings.provider)
    protocol = str(preset.get("protocol", "openai"))
    factory = _factory_for(protocol)
    return factory(settings)


def create_llm_client(
    settings: Optional[LLMSettings] = None,
    *,
    provider: Optional[str] = None,
) -> BaseLLMClient:
    """Build the chat client selected by ``LLM_PROVIDER``.

    Falls back to the local deterministic path only at call time (the
    router already degrades); this function raises if the chosen provider
    cannot be constructed at all.
    """
    settings = settings or get_settings().llm
    if provider:
        from dataclasses import replace

        settings = replace(settings, provider=provider.strip().lower())

    name = (settings.provider or "groq").strip().lower()
    factory = _LLM_FACTORIES.get(name)
    if factory is None:
        factory = _builtin_factory
    client = factory(settings)
    # Make sure the client reports the registry name (custom providers may
    # leave `provider` unset on the class).
    if not getattr(client, "provider", None):
        client.provider = name
    log.debug("LLM provider ready: %s (model=%s)", client.provider, settings.model)
    return client


def available_providers() -> list[str]:
    """Names accepted by ``LLM_PROVIDER`` (registered + built-in + aliases)."""
    return sorted(set(known_llm_providers()) | set(_LLM_FACTORIES))


def describe_providers() -> list[dict[str, Any]]:
    """Human-readable table for docs/GUI: name, label, protocol, key env."""
    rows: list[dict[str, Any]] = []
    for name, preset in LLM_PRESETS.items():
        rows.append(
            {
                "name": name,
                "label": preset.get("label", name),
                "protocol": preset.get("protocol", "openai"),
                "base_url": preset.get("base_url", ""),
                "key_env": preset.get("key_env", []),
                "default_model": preset.get("model", ""),
                "docs": preset.get("docs", ""),
            }
        )
    for name in sorted(_LLM_FACTORIES):
        if name not in LLM_PRESETS and name not in LLM_ALIASES:
            rows.append({"name": name, "label": name, "protocol": "custom",
                         "base_url": "", "key_env": [], "default_model": "", "docs": ""})
    return rows
